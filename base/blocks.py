import torch
import torch.nn as nn
import numpy as np

from base.base_model import Module
from base.layers import FeatureExtractor, MaxPool2D


def trans_tuple(in_, shape=(1, 2)):
    if isinstance(in_, int) or len(tuple(in_)) == 1:
        in_ = np.ones(*shape) * in_
    return in_


class AlexBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_channels=3, strides=3, padding='valid', pool=False):
        """AlexBlock

        Args:
            in_channels (Any):
            out_channels (Any):
            kernel_channels (Any):
            strides (Any):
            padding:
            pool:

        Returns:
            None
        """
        super(AlexBlock, self).__init__()
        in_channels = trans_tuple(in_channels)
        out_channels = trans_tuple(out_channels)
        kernel_channels = trans_tuple(kernel_channels)
        strides = trans_tuple(strides)
        self._layers = list([[], []])

        for in_ch, out_ch, kernel_size, stride in zip(in_channels, out_channels, kernel_channels, strides):
            layer_list = list()
            for i in range(2):
                layer_list.append(FeatureExtractor(in_ch, out_ch, kernel_size, stride, padding, activation='relu'))
                if pool:
                    layer_list.append(MaxPool2D(3, 1, 'same'))
                self.addLayers(layer_list)

    def forward(self, x):
        x_l, x_r = x
        for i in range(len(self._layers[0])):
            x_l = self._layers[0][i](x_l)
            x_r = self._layers[1][i](x_r)
        return torch.cat((x_l, x_r), dim=-1)


class VGGBlock(Module):
    def __init__(self, in_channels, out_channels, num_layer=2, out_kernel_size=3,
                 pool=False, kernel_size=1, stride=1):
        """VGGBlock

        Args:
            in_channels:
            out_channels:
            num_layer:
            out_kernel_size:

        Returns:
            None
        """
        super(VGGBlock, self).__init__()
        self.add_module('fe1',
                        FeatureExtractor(in_channels, out_channels, kernel_size=3, padding='same', bn=True,
                                         activation='relu'))
        if 3 == num_layer:
            self.add_module('fe2', FeatureExtractor(out_channels, out_channels, kernel_size=3, padding='same', bn=True,
                                                    activation='relu'))
        self.add_module(f'fe{num_layer}',
                        FeatureExtractor(out_channels, out_channels, kernel_size=out_kernel_size, padding='same',
                                         bn=True, activation='relu', pool=pool, pool_size=kernel_size,
                                         pool_stride=stride))


class InceptionBlock(Module):
    def __init__(self, out_channels=0):
        super(InceptionBlock, self).__init__()
        self.inc_list = None
        self.out_ch = out_channels // 4

    def forward(self, x):
        return torch.cat([module(x) for module in self.inc_list], dim=-3)


class InceptionBlock_v1A(InceptionBlock):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock_v1A, self).__init__(out_channels)
        self.inc_list = [
            FeatureExtractor(in_channels, self.out_ch, kernel_size=1, stride=2, padding='same', bn=True,
                             activation='relu'),
            FeatureExtractor(in_channels, self.out_ch, kernel_size=3, stride=2, padding='same', bn=True,
                             activation='relu'),
            FeatureExtractor(in_channels, self.out_ch, kernel_size=5, stride=2, padding='same', bn=True,
                             activation='relu'),
            MaxPool2D(3, stride=2, padding='same')
        ]
        self.addLayers(self.inc_list)


class InceptionBlock_v1B(InceptionBlock):
    def __init__(self, in_channels, out_channels):
        """InceptionBlock_v1B
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), p-s2(x)], -3)

        Args:
            in_channels:
            out_channels:

        Returns:
            None
        """
        super(InceptionBlock_v1B, self).__init__(out_channels)
        self.inc_list = [
            nn.Sequential(
                FeatureExtractor(in_channels, self.out_ch, kernel_size=1, stride=2, padding='same', bn=True,
                                 activation='relu')),
            nn.Sequential(
                FeatureExtractor(in_channels, self.out_ch, padding='same', bn=True, activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=3, stride=2, padding='same', bn=True,
                                 activation='relu')),
            nn.Sequential(
                FeatureExtractor(in_channels, self.out_ch, padding='same', bn=True, activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=5, stride=2, padding='same', bn=True,
                                 activation='relu')),
            nn.Sequential(
                MaxPool2D(3, stride=2, padding='same'),
                FeatureExtractor(in_channels, self.out_ch, padding='same', bn=True, activation='relu'))]
        self.addLayers(self.inc_list)


class InceptionBlock_v3A(InceptionBlock):
    def __init__(self, in_channels, out_channels):
        """InceptionBlock_v1B
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), p-s2(x)], -3)

        Args:
            in_channels:
            out_channels:

        Returns:
            None
        """
        super(InceptionBlock_v3A, self).__init__(out_channels)
        self.inc_list = [
            nn.Sequential(
                FeatureExtractor(in_channels, self.out_ch, kernel_size=1, stride=2, padding='same', bn=True,
                                 activation='relu')),
            nn.Sequential(
                FeatureExtractor(in_channels, self.out_ch, padding='same', bn=True, activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=3, stride=2, padding='same', bn=True,
                                 activation='relu')),
            nn.Sequential(
                FeatureExtractor(in_channels, self.out_ch, padding='same', bn=True, activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=3, padding='same', bn=True, activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=3, stride=2, padding='same', bn=True,
                                 activation='relu')),
            nn.Sequential(
                MaxPool2D(3, stride=2, padding='same'),
                FeatureExtractor(in_channels, self.out_ch, padding='same', bn=True, activation='relu'))]
        self.addLayers(self.inc_list)


class InceptionBlock_v3B(InceptionBlock):
    def __init__(self, in_channels, out_channels, kernel_n=3):
        """InceptionBlock_v1B
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), p-s2(x)], -3)

        Args:
            in_channels:
            out_channels:
            kernel_n:

        Returns:
            None
        """
        super(InceptionBlock_v3B, self).__init__(out_channels)
        self.inc_list = [
            nn.Sequential(
                FeatureExtractor(in_channels, self.out_ch, kernel_size=1, stride=2, padding='same', bn=True,
                                 activation='relu')),
            nn.Sequential(
                FeatureExtractor(in_channels, self.out_ch, padding='same', bn=True, activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=(1, kernel_n), stride=(2, 1), padding='same',
                                 bn=True, activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=(kernel_n, 1), stride=(1, 2), padding='same',
                                 bn=True, activation='relu')),
            nn.Sequential(
                FeatureExtractor(in_channels, self.out_ch, padding='same', bn=True, activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=(1, kernel_n), padding='same', bn=True,
                                 activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=(kernel_n, 1), padding='same', bn=True,
                                 activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=(1, kernel_n), stride=(2, 1), padding='same',
                                 bn=True, activation='relu'),
                FeatureExtractor(self.out_ch, self.out_ch, kernel_size=(kernel_n, 1), stride=(1, 2), padding='same',
                                 bn=True, activation='relu')),
            nn.Sequential(
                MaxPool2D(3, stride=2, padding='same'),
                FeatureExtractor(in_channels, self.out_ch, padding='same', bn=True, activation='relu'))]
        self.addLayers(self.inc_list)
