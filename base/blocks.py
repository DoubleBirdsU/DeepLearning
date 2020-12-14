import torch
import torch.nn as nn
import numpy as np

from typing import Union

from base.base_model import Module
from base.layers import Conv, MaxPool2D


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
                layer_list.append(Conv(in_ch, out_ch, kernel_size, stride, padding, act='relu'))
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
    def __init__(self, in_channels, out_channels, num_layer=2, out_kernel_size=3):
        super(VGGBlock, self).__init__()
        self.modules_list.append(Conv(in_channels, out_channels, kernel_size=3))
        for i in range(1, num_layer - 1):
            self.modules_list.append(Conv(out_channels, out_channels, kernel_size=3))
        self.modules_list.append(Conv(out_channels, out_channels, kernel_size=out_kernel_size))
        self.addLayers(self.modules_list)


class InceptionBlock_v1(Module):
    def __init__(self, in_channels, out_channels, type='one_kernel'):
        super(InceptionBlock_v1, self).__init__()
        self.addLayers([
            Conv(in_channels, out_channels, kernel_size=1, stride=2, padding='same', act='relu'),
            Conv(in_channels, out_channels, kernel_size=3, stride=2, padding='same', act='relu'),
            Conv(in_channels, out_channels, kernel_size=5, stride=2, padding='same', act='relu'),
            MaxPool2D(3, stride=2, padding='same')
        ])

    def forward(self, x):
        y = tuple()
        for layer in self._layers:
            y.append(layer(x))
        return torch.cat(tuple(y), dim=-1)
