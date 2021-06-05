import torch
import torch.nn as nn

from base.base_model import Module
from base.layers import FeatureExtractor, MaxPool2D, Equ, ConvSameBnRelu2D


def trans_tuple(parm, num_param=2):
    if isinstance(parm, int) or len(tuple(parm)) == 1:
        parm = tuple([parm for k in range(num_param)])
    return parm


class ModuleBlock(Module):
    def __init__(self):
        super(ModuleBlock, self).__init__()
        self.blocks = list()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def addLayers(self, layers):
        if not isinstance(layers, nn.ModuleList) and isinstance(layers, list):
            self.blocks += layers
        else:
            self.blocks.append(layers)
        return super().addLayers(layers)


class AlexBlock(Module):
    def __init__(self, in_chs, out_chs, kernel_channels=3,
                 strides=3, padding='valid', pool=False):
        """AlexBlock

        Args:
            in_chs (Any):
            out_chs (Any):
            kernel_channels (Any):
            strides (Any):
            padding:
            pool:

        Returns:
            None
        """
        super(AlexBlock, self).__init__()
        in_chs = trans_tuple(in_chs)
        out_chs = trans_tuple(out_chs)
        kernel_channels = trans_tuple(kernel_channels, num_param=len(out_chs))
        strides = trans_tuple(strides, num_param=len(out_chs))
        self._layers = list([[], []])

        for in_ch, out_ch, kernel_size, stride in zip(in_chs, out_chs, kernel_channels, strides):
            for i in range(2):
                self._layers[i].append(
                    FeatureExtractor(in_ch, out_ch, kernel_size, stride, padding=padding, bn=True, activation='relu'))
                if pool:
                    self._layers[i].append(MaxPool2D(3, 1, 'same'))
            self.addLayers(self._layers[0] + self._layers[1])

    def forward(self, x):
        x_l, x_r = x, x
        for lay_l, lay_r in zip(self._layers[0], self._layers[1]):
            x_l = lay_l(x_l)
            x_r = lay_r(x_r)
        return torch.cat((x_l, x_r), dim=1)


class VGGPoolBlock(Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, num_layer=2, pool_size=1, pool_stride=1):
        """VGGPoolBlock

        Args:
            in_ch:
            out_ch:
            num_layer:
            kernel_size:

        Returns:
            None
        """
        super(VGGPoolBlock, self).__init__()
        self.add_module('fe1', ConvSameBnRelu2D(in_ch, out_ch, 3))
        if 3 <= num_layer:
            self.add_module('fe2', ConvSameBnRelu2D(out_ch, out_ch, 3))
        self.add_module(f'fe{num_layer}', ConvSameBnRelu2D(out_ch, out_ch, kernel_size))
        self.add_module('max_pool', MaxPool2D(pool_size, pool_stride, padding='same'))


class ConcatBlock(Module):
    def __init__(self, out_ch=0):
        super(ConcatBlock, self).__init__()
        self.inc_list = None
        self.out_ch = out_ch // 4

    def forward(self, x):
        return torch.cat([module(x) for module in self.inc_list], dim=1)


class InceptionBlock_v1A(ConcatBlock):
    def __init__(self, in_ch, out_ch):
        """InceptionBlock_v1A
        __init__ 构造函数

        Args:
            in_ch (None): 输入维度
            out_ch (None): 输出维度
        """
        super(InceptionBlock_v1A, self).__init__(out_ch)
        self.inc_list = [
            ConvSameBnRelu2D(in_ch, self.out_ch, 1, stride=2),
            ConvSameBnRelu2D(in_ch, self.out_ch, 3, stride=2),
            ConvSameBnRelu2D(in_ch, self.out_ch, 5, stride=2),
            MaxPool2D(3, stride=2, padding='same')
        ]
        self.addLayers(self.inc_list)


class InceptionBlock_v1B(ConcatBlock):
    def __init__(self, in_ch, out_ch):
        """InceptionBlock_v1B
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), prob-s2(x)], -3)

        Args:
            in_ch:
            out_ch:

        Returns:
            None
        """
        super(InceptionBlock_v1B, self).__init__(out_ch)
        self.inc_list = [
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1, stride=2)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, 3, stride=2)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, 5, stride=2)),
            nn.Sequential(
                MaxPool2D(3, stride=2, padding='same'),
                ConvSameBnRelu2D(in_ch, self.out_ch, 1))
        ]
        self.addLayers(self.inc_list)


class InceptionBlock_v3A(ConcatBlock):
    def __init__(self, in_ch, out_ch):
        """InceptionBlock_v3A
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), prob-s2(x)], -3)

        Args:
            in_ch:
            out_ch:

        Returns:
            None
        """
        super(InceptionBlock_v3A, self).__init__(out_ch)
        self.inc_list = [
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1, stride=2)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, 3, stride=2)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, 3),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, 3, stride=2)),
            nn.Sequential(
                MaxPool2D(3, stride=2, padding='same'),
                ConvSameBnRelu2D(in_ch, self.out_ch, 1))
        ]
        self.addLayers(self.inc_list)


class InceptionBlock_v3B(ConcatBlock):
    def __init__(self, in_ch, out_ch, kernel_n=3):
        """InceptionBlock_v3B
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), prob-s2(x)], -3)

        Args:
            in_ch:
            out_ch:
            kernel_n:

        Returns:
            None
        """
        super(InceptionBlock_v3B, self).__init__(out_ch)
        self.inc_list = [
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, kernel_size=1, stride=2)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(1, kernel_n), stride=(2, 1)),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(kernel_n, 1), stride=(1, 2))),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(1, kernel_n)),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(kernel_n, 1)),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(1, kernel_n), stride=(2, 1)),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(kernel_n, 1), stride=(1, 2))),
            nn.Sequential(
                MaxPool2D(3, stride=2, padding='same'),
                ConvSameBnRelu2D(in_ch, self.out_ch, 1))
        ]
        self.addLayers(self.inc_list)


class ReductionBlock_v4B(ConcatBlock):
    def __init__(self, in_ch, out_ch):
        """ReductionBlock_v4B
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), prob-s2(x)], -3)

        Args:
            in_ch:
            out_ch:

        Returns:
            None
        """
        super(ReductionBlock_v4B, self).__init__(out_ch)
        self.inc_list = [
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, 256, 1),
                ConvSameBnRelu2D(256, 384, 3, stride=2)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, 256, 1),
                ConvSameBnRelu2D(256, 288, 3, stride=2)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, 256, 1),
                ConvSameBnRelu2D(256, 288, 3),
                FeatureExtractor(288, 320, 3, stride=2)),
            nn.Sequential(MaxPool2D(3, stride=2, padding='same')),
        ]
        self.addLayers(self.inc_list)


class IncResBlock_v4A(ConcatBlock):
    def __init__(self, in_ch, out_ch):
        """IncResBlock_v4A
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), prob-s2(x)], -3)

        Args:
            in_ch:
            out_ch:

        Returns:
            None
        """
        super(IncResBlock_v4A, self).__init__(out_ch)
        self.inc_list = [
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, stride=2)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, 3, stride=2)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, 3),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, 3, stride=2)),
            nn.Sequential(
                MaxPool2D(3, stride=2, padding='same'),
                ConvSameBnRelu2D(in_ch, self.out_ch, 1)),
        ]
        self.addLayers(self.inc_list)


class IncResBlock_v4B(ConcatBlock):
    def __init__(self, in_ch, out_ch, kernel_n=3, stride=2):
        """IncResBlock_v4B
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), prob-s2(x)], -3)

        Args:
            in_ch:
            out_ch:
            kernel_n:

        Returns:
            None
        """
        super(IncResBlock_v4B, self).__init__(out_ch)
        self.inc_list = [
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, kernel_size=1, stride=stride)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(1, kernel_n), stride=(stride, 1)),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(kernel_n, 1), stride=(1, stride))),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(1, kernel_n)),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(kernel_n, 1)),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(1, kernel_n), stride=(stride, 1)),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, kernel_size=(kernel_n, 1), stride=(1, stride))),
            nn.Sequential(
                MaxPool2D(3, stride=stride, padding='same'),
                ConvSameBnRelu2D(in_ch, self.out_ch, 1))
        ]
        self.addLayers(self.inc_list)


class ResConvBlock(ModuleBlock):
    def __init__(self, in_ch, out_ch, hid_ch, num_layer, res_block, residual_path_first='conv'):
        """ResConvBlock
            residual_path

        Args:
            in_ch:
            out_ch:
            hid_ch:
            num_layer:
            res_block:
            residual_path_first:

        Returns:
            None
        """
        super(ResConvBlock, self).__init__()
        self.blocks.append(res_block(in_ch, out_ch, hid_ch=hid_ch, residual_path=residual_path_first))
        for i in range(1, num_layer - 1):
            self.blocks.append(res_block(out_ch, out_ch, hid_ch=hid_ch, residual_path='equal'))
        self.blocks.append(res_block(out_ch, out_ch, stride=2, hid_ch=hid_ch, residual_path='pool'))
        self.addLayers(self.blocks)

    @staticmethod
    def double_channels(in_ch, out_ch, hid_ch):
        return in_ch * 2, out_ch * 2, hid_ch * 2


class ResBlock(Module):
    @staticmethod
    def make_shortcut(in_ch, out_ch, pool_size, residual_path='equal'):
        """make_shortcut
            Conv, MaxPool2D, Equ

        Args:
            in_ch:
            out_ch:
            pool_size:
            residual_path: 'conv', 'pool', 'equal', Default 'equal'

        Returns:
            None
        """
        if residual_path == 'conv':
            shortcut = FeatureExtractor(in_ch, out_ch, 1, padding='same', strides=1)
        elif residual_path == 'pool':
            shortcut = MaxPool2D(pool_size, stride=2, padding='same')
        else:
            shortcut = Equ()
        return shortcut


class ResBlockA(ResBlock):
    def __init__(self, in_ch, out_ch, stride=1, hid_ch=None, pool_size=1, residual_path='equal'):
        super(ResBlockA, self).__init__()
        self.res_fun = nn.Sequential(
            ConvSameBnRelu2D(in_ch, hid_ch, 3),
            FeatureExtractor(hid_ch, out_ch, 3, stride=stride, padding='same', bn=True),
        )
        self.shortcut = self.make_shortcut(in_ch, out_ch, pool_size, residual_path)
        self.act = nn.ReLU(inplace=True)
        self.addLayers([self.res_fun, self.shortcut, self.act])

    def forward(self, x):
        return self.act(self.res_fun(x) + self.shortcut(x))


class ResBlockB(ResBlock):
    def __init__(self, in_ch, out_ch, stride=1, hid_ch=None, pool_size=1, residual_path='equal'):
        super(ResBlockB, self).__init__()
        self.res_list = list([
            ConvSameBnRelu2D(in_ch, hid_ch, 1),
            ConvSameBnRelu2D(hid_ch, hid_ch, 3, stride=stride),
            FeatureExtractor(hid_ch, out_ch, 1, padding='same', bn=True),
        ])
        self.shortcut = self.make_shortcut(in_ch, out_ch, pool_size, residual_path)
        self.act = nn.ReLU(inplace=True)
        self.addLayers([self.res_list, self.shortcut, self.act])

    def forward(self, x):
        y = self.shortcut(x)
        for layer in self.res_list:
            x = layer(x)
        return self.act(x + y)


block_name_set = {
    'AlexBlock',
    'VGGPoolBlock',
    'ConcatBlock',
    'InceptionBlock_v1A',
    'InceptionBlock_v1B',
    'InceptionBlock_v3A',
    'InceptionBlock_v3B',
    'ReductionBlock_v4B',
    'IncResBlock_v4A',
    'IncResBlock_v4B',
    'ResConvBlock',
    'ResBlockA',
    'ResBlockB',
    'FeatureExtractor',
}
