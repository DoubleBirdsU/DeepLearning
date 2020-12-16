import torch
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
                layer_list.append(FeatureExtractor(in_ch, out_ch, kernel_size, stride, padding, act='relu'))
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
        self.add_module('fe1', FeatureExtractor(
            in_channels, out_channels, kernel_size=3, padding='same', bn=True, act='relu'))
        if 3 == num_layer:
            self.add_module('fe2', FeatureExtractor(
                out_channels, out_channels, kernel_size=3, padding='same', bn=True, act='relu'))
        self.add_module(f'fe{num_layer}', FeatureExtractor(
            out_channels, out_channels, kernel_size=out_kernel_size, padding='same',
            bn=True, act='relu', pool=pool, pool_size=kernel_size, pool_stride=stride))


class InceptionBlock_v1(Module):
    def __init__(self, in_channels, out_channels, type='one_kernel'):
        super(InceptionBlock_v1, self).__init__()
        self.addLayers([
            FeatureExtractor(in_channels, out_channels, kernel_size=1, stride=2, padding='same', act='relu'),
            FeatureExtractor(in_channels, out_channels, kernel_size=3, stride=2, padding='same', act='relu'),
            FeatureExtractor(in_channels, out_channels, kernel_size=5, stride=2, padding='same', act='relu'),
            MaxPool2D(3, stride=2, padding='same')
        ])

    def forward(self, x):
        y = tuple()
        for layer in self.modules_list:
            y.append(layer(x))
        return torch.cat(tuple(y), dim=-1)
