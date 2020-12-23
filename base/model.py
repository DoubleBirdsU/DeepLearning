import os

import yaml
import torch
import pickle
import dill
import torch.optim as optim

from torch import nn
from base.base_model import base_model, Module
from base.blocks import AlexBlock, VGGPoolBlock, InceptionBlock_v1B, InceptionBlock_v3B, ResConvBlock, ResBlockB
from base.layers import MaxPool2D, Dense, FeatureExtractor, RoIDense
from base.utils import print_cover

ONNX_EXPORT = False
DEFAULT_PROTOCOL = 2


class Model(base_model):
    def __init__(self):
        super(Model, self).__init__()


class ModelCheckpoint(object):
    Weight = 'weights'
    Model = 'model'
    All = 'all'
    Last = 'last'
    Best = 'best'
    Index = 'index'
    ckpt_filename = 'ckpt.yaml'

    def __init__(
            self, filepath,
            save_weights_only=False,
            save_best_only=False,
            pickle_module=dill,
            pickle_protocol=DEFAULT_PROTOCOL,
            _use_new_zipfile_serialization=True
    ):
        self.filepath = filepath
        self.root_path = os.path.dirname(filepath)
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.pickle_module = pickle_module
        self.pickle_protocol = pickle_protocol
        self._use_new_zipfile_serialization = _use_new_zipfile_serialization

        # ckpt
        self.ckpt = self.check_ckpt_complete(self.ckpt_read(self.root_path, self.ckpt_filename))
        self.accuracy_best = self.ckpt['accuracy'] if 'accuracy' in self.ckpt else 0
        self.ckpt['filename'] = os.path.abspath(self.filepath)
        self.ckpt['save_best_only'] = self.save_best_only

    def __call__(self, net, **kwargs):
        """
            accuracy:

        Args:
            net (NNet):
        """
        acc_cur = kwargs.pop('accuracy') if 'accuracy' in kwargs else 0.  # 取出 accuracy 参数
        if self.save_weights_only:  # parameters
            f = net.state_dict()
            self.ckpt['save_mode'] = self.Weight
        else:  # model and parameters
            f = net
            self.ckpt['save_mode'] = self.Model

        index_last = self.Last
        index_best = self.Best
        mode = None
        if not self.save_best_only:
            mode = '{}-of-{}_{:.4f}'.format(net.epoch, net.epochs, acc_cur)
            index_last = self.Index

        # Save last 存在问题: 如果新训练的覆盖了旧的概率模型
        suffix_last = self.get_suffix(index=index_last, mode=mode)
        file_last = self.filepath + suffix_last
        acc_change = (self.accuracy_best < acc_cur or (self.save_best_only and not os.path.exists(file_last)))
        torch.save(f, self.filepath + suffix_last, self.pickle_module, self.pickle_protocol,
                   self._use_new_zipfile_serialization)

        # Accuracy
        suffix_best = self.ckpt['suffix_best'] if 'suffix_best' in self.ckpt else suffix_last
        if acc_change:
            suffix_best = suffix_last
            self.accuracy_best = acc_cur

        # Save best
        if self.save_best_only:
            suffix_best = self.get_suffix(index=index_best)
            file_best = self.filepath + suffix_best
            if acc_change or not os.path.exists(file_best):
                torch.save(f, file_best, self.pickle_module, self.pickle_protocol,
                           self._use_new_zipfile_serialization)

        # Dump ckpt
        self.ckpt['accuracy'] = self.accuracy_best
        self.ckpt['suffix_best'] = suffix_best
        self.ckpt['suffix_last'] = suffix_last
        self.ckpt_dump(self.ckpt, os.path.join(self.root_path, self.ckpt_filename))

    @staticmethod
    def get_suffix(index='index', mode=None):
        """
        Args:
            index: 'last', 'best', 'index'
            mode:

        Returns:
            str
        """
        return (f'.{index}' if index else '') + (f'-{mode}' if mode else '')

    @staticmethod
    def ckpt_read(ckpt_path, ckpt_name='ckpt.yaml'):
        ckpt_file = os.path.join(ckpt_path, ckpt_name)

        ckpt = dict()
        if os.path.exists(ckpt_file):
            with open(ckpt_file, 'r', encoding='utf-8') as f:
                ckpt = yaml.load(f.read(), Loader=yaml.SafeLoader)

        return ckpt

    @staticmethod
    def ckpt_dump(ckpt, ckpt_path):
        with open(ckpt_path, 'w', encoding='utf-8') as f:
            yaml.dump(ckpt, f, Dumper=yaml.SafeDumper)

    @staticmethod
    def check_ckpt_complete(ckpt):
        if 'filename' in ckpt and 'suffix_best' in ckpt and 'suffix_last' in ckpt:
            file_best = ckpt['filename'] + ckpt['suffix_best']
            file_last = ckpt['filename'] + ckpt['suffix_last']
            if os.path.exists(file_best) and os.path.exists(file_last):
                return ckpt
        return dict()


class NNet(Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.optimizer = None
        self.loss = None
        self.device = None
        self.metrics = None
        self.checkpoint_fn = None
        self.epoch = 0
        self.epochs = 0
        self._callbacks = None
        self._save_weights_only = False,
        self._save_best_only = False,
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

    def fit_generator(
            self,
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
            use_multiprocessing=False
    ):
        """fit_generator

        Args:
            train_data (DataLoader):
            batch_size ():
            epochs ():
            verbose ():
            callbacks ():
            validation_split ():
            validation_data (DataLoader):
            shuffle (bool):
            class_weight ():
            sample_weight ():
            initial_epoch ():
            steps_per_epoch ():
            validation_steps ():
            validation_batch_size ():
            validation_freq ():
            max_queue_size ():
            workers ():
            use_multiprocessing (bool):

        Returns:
            None
        """
        self.epochs = epochs
        self._callbacks = callbacks
        self._make_callbacks(callbacks)

        # Train
        print(f'Train on {len(train_data.dataset)} samples, validate on {len(validation_data.dataset)} samples')
        for epoch in range(1, epochs + 1):
            print(f'Epoch {epoch}/{epochs}:')
            self.epoch = epoch
            self.train_once(train_data, batch_size)
            if epoch % validation_freq == 0 or epoch == epochs:
                _, acc_cur = self.valid_once(validation_data)

                # checkpoint
                if self.checkpoint_fn is not None:
                    self.checkpoint_fn(self, accuracy=acc_cur)

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
        val_acc = 1. * correct / len(valid_loader.dataset)
        print(self._make_msg(loss_mean, val_acc, 'val_'))
        return loss_mean, val_acc

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

    def save(
            self, filepath,
            save_weights_only=False,
            save_best_only=False,
            pickle_module=pickle,
            pickle_protocol=DEFAULT_PROTOCOL,
            _use_new_zipfile_serialization=True,
            **kwargs
    ):
        """
            checkpoint.yaml
            model_name_weights.pt.last
            model_name_weights.pt.best
        """
        # Parameters
        self._save_weights_only = save_weights_only
        self._save_best_only = save_best_only

        # Save last, best
        if save_weights_only:  # parameters
            f = self.state_dict()
            mode = ModelCheckpoint.Weight
        else:  # model and parameters
            f = self
            mode = ModelCheckpoint.Model

        if not save_best_only:
            mode += f'_{self.epoch}-of-{self.epochs}'
        else:
            mode += f'_{ModelCheckpoint.Best}'

        best = kwargs['best'] if 'best' in kwargs else False
        if save_best_only and best:
            filename_best = ModelCheckpoint.get_suffix(filepath, mode=mode)
            torch.save(f, filename_best, pickle_module, pickle_protocol, _use_new_zipfile_serialization)

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

    @staticmethod
    def get_roi_size(roi_size, in_shape, down_size=32, channels_bias=0, shape_bias=0):
        output_size: torch.Tensor = (torch.Tensor(in_shape).int() + down_size - 1 - shape_bias) //\
                                    down_size - channels_bias
        if roi_size is None:
            roi_size = output_size
        return output_size.minimum(roi_size)

    def _make_callbacks(self, callbacks):
        if callbacks is None:
            return

        for fn in callbacks:
            if isinstance(fn, ModelCheckpoint):
                self.checkpoint_fn = fn

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
    def __init__(self, num_cls=10, img_size=(1, 28, 28), roi_size=None):
        """LeNet

        Input:
            1 * N * N

        Args:
            num_cls (int, optional): the number of classes. Defaults to 10.

        Returns:
            None
        """
        super(LeNet, self).__init__()
        roi_size = self.get_roi_size(roi_size, img_size[1:], 4, 5)
        self.addLayers([
            FeatureExtractor(img_size[0], 6, 5),  # 1x28x28 -> 6x24x24
            FeatureExtractor(6, 6, 5),  # 6x24x24 -> 6x20x20
            FeatureExtractor(6, 16, 5, stride=2),  # 6x20x20 -> 16x8x8 (n + 1) // 2 - 6
            FeatureExtractor(16, 16, 5, stride=2),  # 16x8x8 -> 16x2x2 (n + 3) // 4 - 5
            RoIDense(16, 120, roi_size, activation='sigmoid'),
            Dense(120, 84, 'sigmoid'),
            Dense(84, num_cls, activation='softmax'),
        ])


class AlexNet(NNet):
    def __init__(self, num_cls=1000, img_size=(3, 224, 224), roi_size=None):
        super(AlexNet, self).__init__()
        roi_size = self.get_roi_size(roi_size, img_size[1:], down_size=16, channels_bias=-1)
        self.addLayers([
            AlexBlock((img_size[0], 48), (48, 128), (11, 5), (4, 1), pool=True),
            AlexBlock((256, 192, 192), (192, 192, 128), 3, 1),
            MaxPool2D(3, 1, 'same'),
            RoIDense(256, 4096, roi_size, 'relu'),
            Dense(4096, 4096, 'relu'),
            Dense(4096, num_cls, activation='softmax'),
        ])


class VGG16(NNet):
    def __init__(self, num_cls=1000, img_size=(3, 224, 224), roi_size=None):
        super(VGG16, self).__init__()
        roi_size = self.get_roi_size(roi_size, img_size[1:])
        self.addLayers([
            VGGPoolBlock(img_size[0], 64, num_layer=2, pool_size=3, pool_stride=2),
            VGGPoolBlock(64, 128, num_layer=2, pool_size=3, pool_stride=2),
            VGGPoolBlock(128, 256, num_layer=3, pool_size=3, pool_stride=2),
            VGGPoolBlock(256, 512, num_layer=3, pool_size=3, pool_stride=2),
            VGGPoolBlock(512, 512, num_layer=3, pool_size=3, pool_stride=2),
            RoIDense(512, 512, roi_size, 'relu'),
            Dense(512, 512, 'relu'),
            Dense(512, num_cls, activation='softmax'),
        ])


class InceptionNet_v1B(NNet):
    def __init__(self, num_cls=1000, img_size=(3, 224, 224), roi_size=None):
        super(InceptionNet_v1B, self).__init__()
        roi_size = self.get_roi_size(roi_size, img_size[1:], 16)
        self.net = nn.Sequential(
            FeatureExtractor(img_size[0], 64, 3, padding='same', bn=True, activation='relu'),
            InceptionBlock_v1B(64, 64),
            FeatureExtractor(64, 128, 3, padding='same', bn=True, activation='relu'),
            InceptionBlock_v1B(128, 128),
            FeatureExtractor(128, 256, 3, padding='same', bn=True, activation='relu'),
            FeatureExtractor(256, 256, 3, padding='same', bn=True, activation='relu'),
            InceptionBlock_v1B(256, 256),
            FeatureExtractor(256, 512, 3, padding='same', bn=True, activation='relu'),
            FeatureExtractor(512, 512, 3, padding='same', bn=True, activation='relu'),
            InceptionBlock_v1B(512, 512),
            RoIDense(512, 512, roi_size, 'relu'),
            Dense(512, 512, 'relu'),
            Dense(512, num_cls, activation='softmax'),
        )
        self.addLayers(self.net)

    def forward(self, x):
        return self.net(x)


class InceptionNet_v3B(NNet):
    def __init__(self, num_cls=1000, img_size=(3, 224, 224), roi_size=None):
        super(InceptionNet_v3B, self).__init__()
        roi_size = self.get_roi_size(roi_size, img_size[1:], 16)
        self.net = nn.Sequential(
            FeatureExtractor(img_size[0], 64, 3, padding='same', bn=True, activation='relu'),
            InceptionBlock_v3B(64, 64),
            FeatureExtractor(64, 128, 3, padding='same', bn=True, activation='relu'),
            InceptionBlock_v3B(128, 128),
            FeatureExtractor(128, 256, 3, padding='same', bn=True, activation='relu'),
            FeatureExtractor(256, 256, 3, padding='same', bn=True, activation='relu'),
            InceptionBlock_v3B(256, 256),
            FeatureExtractor(256, 512, 3, padding='same', bn=True, activation='relu'),
            FeatureExtractor(512, 512, 3, padding='same', bn=True, activation='relu'),
            InceptionBlock_v3B(512, 512),
            RoIDense(512, 512, roi_size, 'relu'),
            Dense(512, num_cls, activation='softmax'),
        )
        self.addLayers(self.net)

    def forward(self, x):
        return self.net(x)


class ResNet(NNet):
    in_chs = [64, 256, 512, 1024]
    out_chs = [256, 512, 1024, 2048]
    mid_chs = [64, 128, 256, 512]

    def __init__(self, num_cls=1000, img_size=(3, 512, 512), num_res_block=5, include_top=True, roi_size=None):
        """ResNet
            ResNet18: [2, 2, 2, 2] ResBlockA

            ResNet34: [3, 4, 6, 3] ResBlockA

            ResNet50: [3, 4, 6, 3] ResBlockB

            ResNet101: [3, 4, 23, 3] ResBlockB

            ResNet152: [3, 8, 36, 3] ResBlockB

        Args:
            num_cls:
            img_size:

        Returns:
            None
        """
        super(ResNet, self).__init__()
        self.net = list([
            FeatureExtractor(img_size[0], 64, 7, stride=2, padding='same', bn=True, activation='relu'),
            MaxPool2D(3, stride=2, padding='same'),
        ])

        num = num_res_block - 1
        for in_ch, out_ch, mid_ch in zip(self.in_chs[:num], self.out_chs[:num], self.mid_chs[:num]):
            self.net.append(ResConvBlock(in_ch, out_ch, mid_ch, 3, ResBlockB))

        self.out_ch_last = self.out_chs[num - 2] << (num - 2)
        if include_top:
            roi_size = self.get_roi_size(roi_size, img_size[1:], 2 >> num)
            self.net.append(RoIDense(2028, num_cls, roi_size, 'softmax'))
        self.addLayers(self.net)

    def forward(self, x):
        for block in self.net:
            x = block(x)
        return x
