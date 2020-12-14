import torch
import math
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader

from base.base_model import base_model, Module
from base.blocks import AlexBlock, VGGBlock
from base.layers import FeatureConcat, WeightedFeatureFusion, MaxPool2D, Dense, Flatten, Conv
from base.utils import print_cover

ONNX_EXPORT = False


class Model(base_model):
    def __init__(self):
        super(Model, self).__init__()

    @staticmethod
    def create_modules(modules_def, img_size, cfg):
        """Constructs module list of layer blocks from module configuration in module_def

        Param:
            modules_def: 模型定义
            img_size: 图片尺寸
            cfg: 配置文件

        Returns:
            模型
        """

        img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        modules_def.pop(0)  # cfg training hyperparams (unused)
        channels = [3]  # input channels
        filters = 64
        module_list = nn.ModuleList()
        routs = []  # list of layers which rout to deeper layers
        yolo_index = -1
        idx = -1

        for idx, layer_def in enumerate(modules_def):
            modules = nn.Sequential()

            if layer_def["type"] == "convolutional":
                modules, filters, bn = Model._make_conv(layer_def, channels[-1])
                if not bn:
                    routs.append(idx)
            elif layer_def["type"] == "BatchNorm2d":
                pass
            elif layer_def["type"] == "maxpool":
                modules = MaxPool2D(kernel_size=layer_def["size"], stride=layer_def["stride"], padding='same')
            elif layer_def["type"] == "upsample":
                if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                    g = (yolo_index + 1) * 2 / 32  # gain
                    modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
                else:
                    modules = nn.Upsample(scale_factor=layer_def["stride"])
            elif layer_def["type"] == "route":  # [-2],  [-1,-3,-5,-6]
                layers = layer_def["layers"]
                filters = sum([channels[l + 1 if l > 0 else l] for l in layers])
                routs.extend([idx + l if l < 0 else l for l in layers])
                modules = FeatureConcat(layers=layers)
            elif layer_def["type"] == "shortcut":
                layers = layer_def["from"]
                filters = channels[-1]
                routs.extend([idx + l if l < 0 else l for l in layers])
                modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in layer_def)
            elif layer_def["type"] == "reorg3d":
                pass
            elif layer_def["type"] == "yolo":
                yolo_index += 1
                modules = Model._make_yolo_layer(layer_def, yolo_index, cfg, img_size, module_list)
            elif layer_def["type"] == "dropout":
                modules = nn.Dropout(p=float(layer_def["probability"]))
            else:
                print("Warning: Unrecognized Layer Type: " + layer_def["type"])

            # Register module list and number of output filters
            module_list.append(modules)
            channels.append(filters)

        routs_binary = [False] * (idx + 1)
        for idx in routs:
            routs_binary[idx] = True
        return module_list, routs_binary

    @staticmethod
    def _make_conv(model_cfg, in_ch=None):
        modules = nn.Sequential()

        bn = model_cfg['batch_normalize']  # 1 or 0 / use or not
        filters = model_cfg['filters']  # out_channels
        kernel_size = model_cfg['size']  # kernel size
        stride = model_cfg['stride'] if 'stride' in model_cfg else (model_cfg['stride_y'], model_cfg['stride_x'],)
        padding = kernel_size // 2 if model_cfg['pad'] else 0
        groups = model_cfg['groups'] if 'groups' in model_cfg else 1
        activation = model_cfg['activation'] if 'activation' in model_cfg else 'relu'

        if isinstance(kernel_size, int):
            modules.add_module(
                "Conv2d", nn.Conv2d(
                    in_channels=in_ch, out_channels=filters, kernel_size=kernel_size,
                    stride=stride, padding=padding, groups=groups, bias=not bn))

        # 如果该卷积操作没有bn层，意味着该层为yolo的predictor
        if bn:
            modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters))

        # 激活函数 activation
        if activation == 'leaky':
            modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'relu':
            modules.add_module('activation', nn.ReLU(inplace=True))

        return modules, filters, bn

    @staticmethod
    def _make_yolo_layer(mdef, yolo_index, cfg, img_size, module_list):
        stride = [32, 16, 8]
        if any(x in cfg for x in ["panet", "yolo4", "cd53"]):
            stride = list(reversed(stride))
        layers = mdef["from"] if "from" in mdef else []
        modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]], num_classes=mdef["classes"], img_size=img_size,
                            yolo_index=yolo_index, layers=layers, stride=stride[yolo_index])

        # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
        try:
            j = layers[yolo_index] if 'from' in mdef else -1
            # If previous layer is a dropout layer, get the one before
            if module_list[j].__class__.__name__ == 'Dropout':
                j -= 1
            bias_ = module_list[j][0].bias  # shape(255,)
            bias = bias_[:modules.num_outputs * modules.num_anchors].view(modules.num_anchors, -1)  # shape(3,85)
            bias[:, 4] += -4.5  # obj
            bias[:, 5:] += math.log(0.6 / (modules.num_classes - 0.99))  # cls (sigmoid(p) = 1/num_classes)
            module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
        except Exception:
            print('WARNING: smart bias initialization failure.')
        return modules


class NNet(Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.optimizer = None
        self.loss = None
        self.device = None
        self.metrics = None
        self._params_dict = dict()

    def compile(self, optimizer=None, loss=None, call_params=None, metrics=None, device=torch.device('cpu')):
        """
        Args:
            optimizer:
            loss:
            call_params:
            device:
            metrics:

        Returns:
            None
        """
        self.optimizer = self.get_parameters(optimizer, call_params)
        self.loss = loss
        self.metrics = metrics
        self.device = device

    def fit_generator(self, train_data, epochs, batch_size=1, validate_data=None, print_grad=1):
        """fit_generator

        Args:
            train_data (DataLoader):
            epochs:
            batch_size:
            validate_data (DataLoader):
            print_grad:

        Returns:
            None
        """
        print(f'Train on {len(train_data.dataset)} samples, validate on {len(validate_data.dataset)} samples')
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}/{epochs}:')
            self.train_once(train_data, print_grad)
            self.valid_once(validate_data)

    def train_once(self, train_loader, print_grad):
        """train_once

        Args:
            train_loader:
            print_grad:

        Returns:
            None
        """
        self.train()
        correct = 0
        loss_mean = 0.0
        num_data = 0
        count_data = len(train_loader.dataset)
        for batch_idx, (data, y_true) in enumerate(train_loader):
            num_data += len(data)
            data, y_true = data.to(self.device), y_true.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self(data)
            loss = self.loss(y_pred, y_true)
            loss_mean = (loss_mean * batch_idx + loss) / (batch_idx + 1)
            loss.backward()
            self.optimizer.step()

            correct += self._get_correct(y_pred, y_true) if self.metrics else 0.
            if batch_idx % print_grad == 0 or num_data == count_data:
                schedule = self._get_schedule(1. * num_data / count_data)
                msg = self._make_msg(loss, 1. * correct / num_data, prefix='')
                print_cover(f'{num_data}/{count_data} {schedule}{msg}')

    def valid_once(self, valid_loader):
        self.eval()
        correct = 0
        loss_mean = 0.
        with torch.no_grad():
            for data, y_true in valid_loader:
                data, y_true = data.to(self.device), y_true.to(self.device)
                y_pred = self(data)
                loss_mean += self.loss(y_pred, y_true, reduction='sum').item()  # sum up batch loss
                correct += self._get_correct(y_pred, y_true) if self.metrics else 0.

        loss_mean /= len(valid_loader.dataset)
        count_data = len(valid_loader.dataset)
        print(self._make_msg(loss_mean, 1. * correct / count_data, 'val_'))

    def get_parameters(self, opt_type='adam', call_params=None, **kwargs):
        if call_params is None:
            call_params = {'lr': 0.01, 'weight_decay': 32}
        if 'lr' not in call_params:
            call_params['lr'] = 0.01
        if 'weight_decay' not in call_params:
            call_params['weight_decay'] = 32
        opg_bn, opg_weight, opg_bias = [], [], []  # optimizer parameter groups
        for layer in self.get_modules():
            if hasattr(layer, 'bias') and isinstance(layer.bias, nn.Parameter):
                opg_bias.append(layer.bias)  # biases
            if isinstance(layer, nn.BatchNorm2d):
                opg_bn.append(layer.weight)  # no decay
            elif hasattr(layer, 'weight') and isinstance(layer.weight, nn.Parameter):
                opg_weight.append(layer.weight)  # apply decay

        opt_type = opt_type.lower()
        if 'adam' == opt_type:  # adjust beta1 to momentum
            momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.9
            self.optimizer = optim.Adam(opg_bias, lr=call_params['lr'], betas=(momentum, 0.999))
        else:
            momentum = kwargs['momentum'] if 'momentum' in kwargs else 0.9
            self.optimizer = optim.SGD(opg_bias, lr=call_params['lr'], momentum=momentum, nesterov=True)

        self.optimizer.add_param_group({'params': opg_weight})
        self.optimizer.add_param_group({'params': opg_bn})
        del opg_bn, opg_weight, opg_bias
        return self.optimizer

    def _make_msg(self, loss, accuracy=.0, prefix=''):
        msg = ' - {}loss: {:.6f}'.format(prefix, loss)
        if self.metrics:
            msg += ' - {}{}: {:.6f}'.format(prefix, self.metrics[0], accuracy)
        return msg

    @staticmethod
    def _get_correct(y_pred, y_true):
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        return pred.eq(y_true.view_as(pred)).sum().item()

    @staticmethod
    def _get_schedule(scale=1., count=25):
        schedule = f'[{"-" * count}]'
        schedule = schedule.replace('-', '=', int(scale * count))
        return schedule.replace('-', '>', 1)


class LeNet(NNet):
    def __init__(self, num_cls=10, img_size=(1, 28, 28)):
        """LeNet

        Input:
            1 * N * N

        Args:
            num_cls (int, optional): the number of classes. Defaults to 10.

        Returns:
            None
        """
        super(LeNet, self).__init__()
        self.conv1 = Conv(img_size[0], 6, 5)  # 1x28x28 -> 6x24x24
        self.conv2 = Conv(6, 6, 5)  # 6x24x24 -> 6x20x20
        self.conv3 = Conv(6, 16, 5, stride=2)  # 6x20x20 -> 16x8x8 (n + 1) // 2 - 6
        self.conv4 = Conv(16, 16, 5, stride=2)  # 16x8x8 -> 16x2x2 (n + 3) // 4 - 5
        self.flatten = Flatten()  # 16x2x2 -> 64
        self.fc1 = Dense(16 * ((torch.Tensor(img_size[1:]).int() + 3) // 4 - 5).prod(), 120, 'sigmoid')
        self.fc2 = Dense(120, 84, 'sigmoid')
        self.fc3 = Dense(84, num_cls)
        self.softmax = nn.Softmax(dim=-1)
        self.addLayers([self.conv1, self.conv2, self.conv3, self.conv4,
                        self.flatten, self.fc1, self.fc2, self.fc3, self.softmax])


class AlexNet(NNet):
    def __init__(self, num_cls=1000, img_size=(3, 224, 224)):
        super(AlexNet, self).__init__()
        self.alex1 = AlexBlock((img_size[0], 48), (48, 128), (11, 5), (4, 1), pool=True)
        self.alex2 = AlexBlock((256, 192, 192), (192, 192, 128), 3, 1)
        self.pool = MaxPool2D(3, 1, 'same')
        self.flatten = Flatten()
        self.fc1 = Dense(256 * 13 * 13, 4096, 'relu')
        self.fc2 = Dense(4096, 4096, 'relu')
        self.fc3 = Dense(4096, num_cls)
        self.softmax = nn.Softmax(dim=-1)
        self.addLayers([self.alex1, self.alex2, self.pool, self.flatten,
                        self.fc1, self.fc2, self.fc3, self.softmax])


class VGG16(NNet):
    def __init__(self, num_cls=1000, img_size=(3, 224, 224)):
        super(VGG16, self).__init__()
        self.vggb1 = VGGBlock(img_size[0], 64)
        self.pool1 = MaxPool2D(3, stride=2, padding='same')
        self.vggb2 = VGGBlock(64, 128)
        self.pool2 = MaxPool2D(3, stride=2, padding='same')
        self.vggb3 = VGGBlock(128, 256)
        self.pool3 = MaxPool2D(3, stride=2, padding='same')
        self.vggb4 = VGGBlock(256, 512)
        self.pool4 = MaxPool2D(3, stride=2, padding='same')
        self.vggb5 = VGGBlock(512, 512)
        self.pool5 = MaxPool2D(3, stride=2, padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(512 * ((torch.Tensor(img_size[1:]).int() +  + 31) // 32).prod(), 4096, 'relu')
        self.fc2 = Dense(4096, 4096, 'relu')
        self.fc3 = Dense(4096, num_cls)
        self.softmax = nn.Softmax(dim=-1)
        self.addLayers([
            self.vggb1, self.pool1, self.vggb2, self.pool2, self.vggb3, self.pool3, self.vggb4,
            self.pool4, self.vggb5, self.pool5, self.flatten, self.fc1, self.fc2, self.fc3, self.softmax,
        ])


# YOLOLayer
class YOLOLayer(base_model):
    """对YOLO的输出进行处理
    """

    def __init__(self, anchors, num_classes, img_size, yolo_index, layers, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.num_layers = len(layers)  # number of output layers (3)
        self.num_anchors = len(anchors)  # number of anchors (3)
        self.num_classes = num_classes  # number of classes (80)
        self.num_outputs = num_classes + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2)
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        """生成 grids

        Param:
            ng: 特征图大小
            device:

        Returns:

        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, x, out):
        bool_ASFF = False  # https://arxiv.org/abs/1911.09516
        if bool_ASFF:
            i, n = self.index, self.num_layers  # index in layers, number of layers
            x = out[self.layers[i]]
            bs, _, ny, nx = x.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), x.device)

            # outputs and weights
            w = torch.sigmoid(x[:, -n:]) * (2 / n)  # sigmoid weights (faster)

            # weighted bool_ASFF sum
            x = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    x += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = x.shape  # batch_size, predict_param(255), grid(13), grid(13)
            if (self.nx, self.ny) != (nx, ny) or hasattr(self, "grid") is False:  # fix num_outputs grid bug
                self.create_grids((nx, ny), x.device)

        # p.view(batch_size, 255, 13, 13) -> (batch_size, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        x = x.view(bs, self.num_anchors, self.num_outputs,
                   self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return x
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.num_anchors * self.nx * self.ny  # 3*
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.num_anchors, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            x = x.view(m, self.num_outputs)
            x[:, :2] = (torch.sigmoid(x[:, 0:2]) + grid) * ng  # x, y
            x[:, 2:4] = torch.exp(x[:, 2:4]) * anchor_wh  # width, height
            x[:, 4:] = torch.sigmoid(x[:, 4:])
            x[:, 5:] = x[:, 5:self.num_outputs] * x[:, 4:5]
            return x
        else:  # inference
            io = x.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.num_outputs), x  # view [1, 3, 13, 13, 85] as [1, 507, 85]
