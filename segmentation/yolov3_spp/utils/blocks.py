import torch
import torch.nn as nn
import numpy as np

from utils import layers
from utils.base_model import Module


class FeatureExtractorBlk(Module):
    def __init__(self, in_ch, out_ch, kernel_size=(1, 1),
                 stride=1, padding='valid', groups=1, bias=False, bn=False, activation='valid',
                 pool=False, pool_size=(1, 1), pool_stride=1, pool_padding='valid', dp=0):
        r"""FeatureExtractorBlk is 'CBAPD'
            Conv2D, BatchNormal, Activation, MaxPool2D, Dropout

        Args:
            padding: 'valid', 'same', 'tiny-same'
            activation: 'valid', 'relu', 'leaky'
            pool_padding: 'valid', 'same', 'tiny-same'

        Returns:
            None
        """
        super(FeatureExtractorBlk, self).__init__()
        modules = nn.Sequential()
        modules.add_module(
            'Conv2d',
            nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                      padding=kernel_size // 2 if 'same' in padding else 0,
                      groups=groups, bias=(bias and not bn)))

        if bn:
            modules.add_module('BatchNorm2d', nn.BatchNorm2d(out_ch))

        if activation != 'valid':
            modules.add_module('activation', layers.Activation(activation, inplace=True, negative_slope=0.1))

        if pool:
            modules.add_module('MaxPool2D', layers.MaxPool2D(pool_size, pool_stride, pool_padding))

        if dp:
            modules.add_module('Dropout', nn.Dropout(p=dp))

        self.collect_layers(modules)


class FeatureConvBlk(Module):
    def __init__(self, in_ch, out_ch, kernels_size=(1, 3, 1, 3, 1), strides=(1, 1, 1, 1, 1), in_ch_first=None):
        super(FeatureConvBlk, self).__init__()
        self.filters = out_ch if len(kernels_size) % 2 else in_ch
        self.feb_list = list([
            FeatureExtractorBlk(
                in_ch_first if in_ch_first else in_ch, out_ch, kernels_size[0], strides[0],
                padding='tiny-same', bias=True, bn=True, activation='leaky')])
        for kernel_size, stride in zip(kernels_size[1:], strides[1:]):
            in_ch, out_ch = self.swap(in_ch, out_ch)
            self.feb_list.append(FeatureExtractorBlk(
                in_ch, out_ch, kernel_size, stride,
                padding='tiny-same', bias=True, bn=True, activation='leaky'))

        self.collect_layers(self.feb_list)


class DarknetBlk(Module):
    def __init__(self, in_ch, out_ch):
        super(DarknetBlk, self).__init__()
        self.feb_list = [
            FeatureExtractorBlk(
                in_ch, out_ch, 1, 1, padding='tiny-same', bias=True, bn=True, activation='leaky'),
            FeatureExtractorBlk(
                out_ch, in_ch, 3, 1, padding='tiny-same', bias=True, bn=True, activation='leaky')]

        self.collect_layers(self.feb_list)

    def __call__(self, x):  # 优先级高于 forward
        return x + super().forward(x)


class SPPBlk(Module):
    def __init__(self):
        super(SPPBlk, self).__init__()
        modules = list()
        for kernel_size in [5, 9, 13]:
            modules.append(layers.MaxPool2D(kernel_size=kernel_size, stride=1, padding='tiny-same'))
        self.collect_layers(modules)

    def __call__(self, x):
        return torch.cat([x] + super().forward_list(x), 1)


class UpSampleBlk(Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super(UpSampleBlk, self).__init__()
        self.collect_layers([
            FeatureExtractorBlk(in_ch, out_ch, 1, 1, padding='tiny-same', bn=True, activation='leaky'),
            nn.Upsample(scale_factor=scale_factor)])

    def __call__(self, inputs):
        return torch.cat([super().forward(inputs[0]), inputs[1]], 1)


class AnchorBlk(Module):
    def __init__(self, in_ch, out_ch, upsample=False, scale_factor=2,
                 kernels_size=(1, 3, 1, 3, 1), stride=None,
                 anchors_vec=None, in_ch_first=None):
        r"""AnchorBlk

        Args:
            in_ch:
            out_ch:
            upsample: bool,
            scale_factor:
            kernels_size:
            stride:

        Returns:
            out, anchor
        """
        super(AnchorBlk, self).__init__()
        self.anchor_vec = anchors_vec
        self.us_block = None
        if upsample:
            self.us_block = UpSampleBlk(in_ch, out_ch, scale_factor)

        if not stride:
            stride = np.ones_like(kernels_size)
        self.fcb = FeatureConvBlk(in_ch, out_ch, kernels_size, stride, in_ch_first=in_ch_first)
        self.filters = self.fcb.filters
        if self.filters == out_ch:
            in_ch, out_ch = self.swap(in_ch, out_ch)
        self.anchor_list = [
            FeatureExtractorBlk(in_ch, out_ch, 3, 1, padding='tiny-same', bn=True, activation='leaky'),
            FeatureExtractorBlk(out_ch, 255, 1, 1, padding='tiny-same', bias=True)]

        self.collect_layers([self.us_block, self.fcb, self.anchor_list])

    def __call__(self, inputs):
        out = inputs[0]
        if self.us_block and len(inputs) > 1:
            out = self.us_block(inputs)
        anchor = out = self.fcb(out)
        for layer in self.anchor_list:
            anchor = layer(anchor)
        return out, anchor


class YOLOBlk(Module):
    def __init__(self, in_ch, out_ch, anchors, anchor_strides, in_ch_first=2048):
        """YOLOBlk

        Args:
            in_ch:
            out_ch:

        Returns:
            None
        """
        super(YOLOBlk, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.anchor_strides = anchor_strides
        self.anchors_vec = [self.anchors[i] / self.anchor_strides[i] for i in range(3)]
        self.anchor1 = AnchorBlk(in_ch, out_ch, kernels_size=(1, 3, 1),
                                 anchors_vec=self.anchors_vec[-1], in_ch_first=in_ch_first)
        self.anchor2 = AnchorBlk(self.anchor1.filters, self.anchor1.filters // 2, upsample=True, scale_factor=2,
                                 anchors_vec=self.anchors_vec[-2], in_ch_first=self.anchor1.filters // 2 * 3)
        self.anchor3 = AnchorBlk(self.anchor2.filters, self.anchor2.filters // 2, upsample=True, scale_factor=2,
                                 anchors_vec=self.anchors_vec[-3], in_ch_first=self.anchor2.filters // 2 * 3)
        self.collect_layers([self.anchor1, self.anchor2, self.anchor3])

    def __call__(self, inputs):
        """YOLOBlk
        Args:
            inputs: [in_anchor1, in_anchor2, in_anchor3]

        Returns:
            anchor1, anchor2, anchor3
        """
        x = inputs[0]
        x, anchor1 = self.anchor1([x])
        x, anchor2 = self.anchor2([x, inputs[1]])
        x, anchor3 = self.anchor3([x, inputs[2]])
        return anchor1, anchor2, anchor3
