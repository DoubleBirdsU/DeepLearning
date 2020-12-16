import torch
import pickle
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader

from base.base_model import base_model, Module
from base.blocks import AlexBlock, VGGBlock
from base.layers import MaxPool2D, Dense, Flatten, FeatureExtractor
from base.utils import print_cover

ONNX_EXPORT = False
DEFAULT_PROTOCOL = 2


class Model(base_model):
    def __init__(self):
        super(Model, self).__init__()


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

    def fit_generator(self,
                      train_data=None,
                      batch_size=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_split=0.0,
                      validation_data=None,
                      shuffle=True,
                      class_weight=None,
                      sample_weight=None,
                      initial_epoch=0,
                      steps_per_epoch=None,
                      validation_steps=None,
                      validation_batch_size=None,
                      validation_freq=1,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False):
        """fit_generator

        Args:
            train_data (DataLoader):
            batch_size ():
            epochs ():
            verbose ():
            callbacks ():
            validation_split ():
            validation_data (DataLoader):
            shuffle ():
            class_weight ():
            sample_weight ():
            initial_epoch ():
            steps_per_epoch ():
            validation_steps ():
            validation_batch_size ():
            validation_freq ():
            max_queue_size ():
            workers ():
            use_multiprocessing ():

        Returns:
            None
        """
        print(f'Train on {len(train_data.dataset)} samples, validate on {len(validation_data.dataset)} samples')
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}/{epochs}:')
            self.train_once(train_data, batch_size)
            if epoch % validation_freq == 0 or epoch == epochs:
                self.valid_once(validation_data)

    def train_once(self, train_loader, batch_size):
        """train_once

        Args:
            train_loader:
            batch_size:

        Returns:
            None
        """
        self.train()
        correct = 0
        num_data = 0
        loss_mean = 0.0
        count_data = len(train_loader.dataset)
        count_batch = (count_data + batch_size - 1) // batch_size
        for batch_idx, (data, y_true) in enumerate(train_loader):
            data, y_true = data.to(self.device), y_true.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self(data)
            loss = self.loss(y_pred, y_true)
            loss.backward()
            self.optimizer.step()

            # 输出信息
            num_data += len(data)
            loss_mean = (loss_mean * batch_idx + loss) / (batch_idx + 1)
            correct += self._get_correct(y_pred, y_true) if self.metrics else 0.
            msg = self._print_cover(loss_mean, correct, num_data, batch_idx + 1, count_batch)
            print_cover(msg)

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

        # 输出信息
        loss_mean /= len(valid_loader.dataset)
        print(self._make_msg(loss_mean, 1. * correct / len(valid_loader.dataset), 'val_'))

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

    def save(self, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True):
        torch.save(self.state_dict(), f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)

    @staticmethod
    def get_channels(channels, in_shape, down_size=32, channels_bias=0, shape_bias=0):
        """get_channels

        Args:
            channels:
            in_shape:
            down_size (Union[int, tuple]):
            channels_bias (Union[int, tuple]):
            shape_bias (Union[int, tuple]):

        Returns:
            channels
        """
        return channels * ((torch.Tensor(in_shape).int() + down_size - 1 - shape_bias) //
                           down_size - channels_bias).prod()

    def _print_cover(self, loss, correct, count_data, cur_batch, count_batch):
        schedule = self._get_schedule(1. * cur_batch / count_batch)
        msg = self._make_msg(loss, 1. * correct / count_data, prefix='')
        return f'{cur_batch}/{count_batch} {schedule}{msg}'

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
        self.addLayers([
            FeatureExtractor(img_size[0], 6, 5),  # 1x28x28 -> 6x24x24
            FeatureExtractor(6, 6, 5),  # 6x24x24 -> 6x20x20
            FeatureExtractor(6, 16, 5, stride=2),  # 6x20x20 -> 16x8x8 (n + 1) // 2 - 6
            FeatureExtractor(16, 16, 5, stride=2),  # 16x8x8 -> 16x2x2 (n + 3) // 4 - 5
            Flatten(),  # 16x2x2 -> 64
            Dense(self.get_channels(16, img_size[1:], 4, 5), 120, 'sigmoid'),
            Dense(120, 84, 'sigmoid'),
            Dense(84, num_cls),
            nn.Softmax(dim=-1),
        ])


class AlexNet(NNet):
    def __init__(self, num_cls=1000, img_size=(3, 224, 224)):
        super(AlexNet, self).__init__()
        self.addLayers([
            AlexBlock((img_size[0], 48), (48, 128), (11, 5), (4, 1), pool=True),
            AlexBlock((256, 192, 192), (192, 192, 128), 3, 1),
            MaxPool2D(3, 1, 'same'),
            Flatten(),
            Dense(256 * 13 * 13, 4096, 'relu'),
            Dense(4096, 4096, 'relu'),
            Dense(4096, num_cls),
            nn.Softmax(dim=-1),
        ])


class VGG16(NNet):
    def __init__(self, num_cls=1000, img_size=(3, 224, 224)):
        super(VGG16, self).__init__()
        self.addLayers([
            VGGBlock(img_size[0], 64, pool=True, kernel_size=3, stride=2),
            VGGBlock(64, 128, pool=True, kernel_size=3, stride=2),
            VGGBlock(128, 256, num_layer=3, pool=True, kernel_size=3, stride=2),
            VGGBlock(256, 512, num_layer=3, pool=True, kernel_size=3, stride=2),
            VGGBlock(512, 512, num_layer=3, pool=True, kernel_size=3, stride=2),
            Flatten(),
            Dense(512 * ((torch.Tensor(img_size[1:]).int() + 31) // 32).prod(), 512, 'relu'),
            Dense(512, 512, 'relu'),
            Dense(512, num_cls),
            nn.Softmax(dim=-1),
        ])


class VGG16_V2(NNet):
    def __init__(self, num_cls=1000, img_size=(3, 224, 224)):
        super(VGG16_V2, self).__init__()
        self.addLayers([
            FeatureExtractor(img_size[0], 64, 3, padding='same', bn=True, act='relu'),
            FeatureExtractor(64, 64, 3, padding='same', bn=True, act='relu', pool=True, pool_size=3, pool_stride=2),
            FeatureExtractor(64, 128, 3, padding='same', bn=True, act='relu'),
            FeatureExtractor(128, 128, 3, padding='same', bn=True, act='relu', pool=True, pool_size=3, pool_stride=2),
            FeatureExtractor(128, 256, 3, padding='same', bn=True, act='relu'),
            FeatureExtractor(256, 256, 3, padding='same', bn=True, act='relu'),
            FeatureExtractor(256, 256, 3, padding='same', bn=True, act='relu', pool=True, pool_size=3, pool_stride=2),
            FeatureExtractor(256, 512, 3, padding='same', bn=True, act='relu'),
            FeatureExtractor(512, 512, 3, padding='same', bn=True, act='relu'),
            FeatureExtractor(512, 512, 3, padding='same', bn=True, act='relu', pool=True, pool_size=3, pool_stride=2),
            FeatureExtractor(512, 512, 3, padding='same', bn=True, act='relu'),
            FeatureExtractor(512, 512, 3, padding='same', bn=True, act='relu'),
            FeatureExtractor(512, 512, 3, padding='same', bn=True, act='relu', pool=True, pool_size=3, pool_stride=2),
            Flatten(),
            Dense(self.get_channels(512, img_size[1:]), 512, 'relu'),
            Dense(512, 512, 'relu'),
            Dense(512, num_cls),
            nn.Softmax(dim=-1),
        ])
