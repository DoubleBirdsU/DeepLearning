import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from base.layers import Concat, MixConv2d, Conv, Bottleneck, SPP, DWConv, Focus, \
    BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, CrossConv, CrossConvCSP
from utils.general import check_anchor_order, make_divisible
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights


class Detect(nn.Module):
    def __init__(self, num_cls=80, anchors=(), out_chs=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.num_cls = num_cls  # number of classes
        self.no = num_cls + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        anchor_list = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', anchor_list)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', anchor_list.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in out_chs)  # output conv
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class SeqModel(nn.Module):
    def __init__(self, cfg='yolov4.yaml', in_ch=3, num_cls=None):
        # model, input channels, number of classes
        super(SeqModel, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if num_cls and num_cls != self.yaml['num_cls']:
            print('Overriding %s num_cls=%g with num_cls=%g' % (cfg, self.yaml['num_cls'], num_cls))
            self.yaml['num_cls'] = num_cls  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), in_channels=[in_ch])  # model, save list, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        module = self.model[-1]  # Detect()
        if isinstance(module, Detect):
            min_stride = 128  # 2x min stride
            xs = self.forward(torch.zeros(1, in_ch, min_stride, min_stride))
            module.stride = torch.tensor([min_stride / x.shape[-2] for x in xs])  # forward
            module.anchors /= module.stride.view(-1, 1, 1)
            check_anchor_order(module)
            self.stride = module.stride
            self._initialize_biases()  # only run once
            # print('Strides: %min_stride' % module.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            scales = [1, 0.83, 0.67]  # scales
            flips = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for scale, flip in zip(scales, flips):
                xi = scale_img(x.flip(flip) if flip else x, scale)
                yi = self.forward_once(xi)[0]  # forward
                yi[..., :4] /= scale  # de-scale
                if flip == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif flip == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, diff_time = [], []  # outputs
        for module in self.model:
            if module.f != -1:  # if not from previous layer, from earlier layers
                x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]

            if profile:
                try:
                    import thop
                    out = thop.profile(module, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except TypeError:
                    out = 0
                time_syn = time_synchronized()
                for _ in range(10):
                    _ = module(x)
                diff_time.append((time_synchronized() - time_syn) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (out, module.np, diff_time[-1], module.type))

            x = module(x)  # run
            y.append(x if module.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(diff_time))
        return x

    def _initialize_biases(self, cls_freq=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=num_cls) + 1.
        module = self.model[-1]  # Detect() module
        for mi, stride in zip(module.m, module.stride):  # from
            bias = mi.bias.view(module.na, -1)  # conv.bias(255) to (3,85)
            bias[:, 4] += math.log(8 / (640 / stride) ** 2)  # obj (8 objects per 640 image)
            bias[:, 5:] += math.log(0.6 / (module.num_cls - 0.99)) if cls_freq is None else\
                torch.log(cls_freq / cls_freq.sum())  # cls
            mi.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

    def _print_biases(self):
        module = self.model[-1]  # Detect() module
        for mi in module.m:  # from
            bias = mi.bias.detach().view(module.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *bias[:5].mean(1).tolist(),
                                                         bias[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ', end='')
        for module in self.model.modules():
            if type(module) is Conv:
                module._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                module.conv = fuse_conv_and_bn(module.conv, module.bn)  # update conv
                module.bn = None  # remove batchnorm
                module.forward = module.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        model_info(self)


def parse_model(dict_model, in_channels):  # model_dict, input_channels(3)
    msg_style = '%3s%18s%5s%10s  %-40s%-30s'
    print('\n' + msg_style % ('', 'from', 'num', 'params', 'module', 'arguments'))
    anchors, num_cls = dict_model['anchors'], dict_model['num_cls']
    gain_d, gain_w = dict_model['depth_multiple'], dict_model['width_multiple']
    num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    num_outputs = num_anchors * (num_cls + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, out_ch = [], [], in_channels[-1]  # layers, save list, ch out
    # from, number, module, args
    for i, (idx_f, num, module, args) in enumerate(dict_model['backbone'] + dict_model['head']):
        module = eval(module) if isinstance(module, str) else module  # eval strings
        for j, arg in enumerate(args):
            try:
                args[j] = eval(arg) if isinstance(arg, str) else arg  # eval strings
            except:
                pass

        num = max(round(num * gain_d), 1) if num > 1 else num  # depth gain
        if module in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus,
                      CrossConv, BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, CrossConvCSP]:
            in_ch, out_ch = in_channels[idx_f], args[0]

            out_ch = make_divisible(out_ch * gain_w, 8) if out_ch != num_outputs else out_ch

            args = [in_ch, out_ch, *args[1:]]
            if module in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, CrossConvCSP]:
                args.insert(2, num)
                num = 1
        elif module is nn.BatchNorm2d:
            args = [in_channels[idx_f]]
        elif module is Concat:
            out_ch = sum([in_channels[-1 if x == -1 else x + 1] for x in idx_f])
        elif module is Detect:
            args.append([in_channels[x + 1] for x in idx_f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(idx_f)
        else:
            out_ch = in_channels[idx_f]

        module_seq = nn.Sequential(*[module(*args) for _ in range(num)]) if num > 1 else module(*args)  # module
        module_type = str(module)[8:-2].replace('__main__.', '')  # module type
        num_params = sum([x.numel() for x in module_seq.parameters()])  # number params
        # attach index, 'from' index, type, number params
        module_seq.i, module_seq.f, module_seq.type, module_seq.np = i, idx_f, module_type, num_params
        print(msg_style % (i, idx_f, num, num_params, module_type, args))  # print
        save.extend(x % i for x in ([idx_f] if isinstance(idx_f, int) else idx_f) if x != -1)  # append to save list
        layers.append(module_seq)
        in_channels.append(out_ch)
    return nn.Sequential(*layers), sorted(save)
