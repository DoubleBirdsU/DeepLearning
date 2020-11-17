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
        self.in_ch_first = in_ch
        self.out_ch_last = out_ch
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
    def __init__(self, in_ch_first=32):
        super(SPPBlk, self).__init__()
        self.in_ch_first = in_ch_first
        self.out_ch_last = in_ch_first * 4
        modules = list()
        for kernel_size in [5, 9, 13]:
            modules.append(layers.MaxPool2D(kernel_size=kernel_size, stride=1, padding='tiny-same'))
        self.collect_layers(modules, bool_in=True, bool_out=True)

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
                 anchors_vec=None, in_ch_first=None, out_ch_last=None):
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
        self.out_ch_last = self.fcb.out_ch_last
        if self.out_ch_last == out_ch:
            in_ch, out_ch = self.swap(in_ch, out_ch)
        self.anchor_list = [
            FeatureExtractorBlk(in_ch, out_ch, 3, 1, padding='tiny-same', bn=True, activation='leaky'),
            FeatureExtractorBlk(out_ch, out_ch_last if out_ch_last else in_ch, 1, 1, padding='tiny-same', bias=True)]

        self.collect_layers([self.us_block, self.fcb, self.anchor_list], bool_out=True)

    def __call__(self, inputs):  # 禁止异常传递, 反常案例 e.g. x = inputs[1], shortcut = inputs[0].
        out = self.us_block([inputs[1], inputs[0]]) if \
            self.us_block and isinstance(inputs[1], torch.Tensor) else inputs[0]
        out = anchor = self.fcb(out)
        for layer in self.anchor_list:
            anchor = layer(anchor)
        shape_anchor = anchor.shape
        num_anchor = len(self.anchor_vec)
        info_size = shape_anchor[1] // num_anchor
        anchor = anchor.view(shape_anchor[0], num_anchor, info_size,
                             shape_anchor[-2], shape_anchor[-1]).permute(0, 1, 3, 4, 2).contiguous()
        return anchor, out


class YOLOBlk(Module):
    def __init__(self, in_ch, out_ch, anchors_vec, in_ch_first=2048, out_ch_last=255):
        """YOLOBlk

        Args:
            in_ch:
            out_ch:

        Returns:
            None
        """
        super(YOLOBlk, self).__init__()
        self.anchors_vec = anchors_vec if isinstance(anchors_vec[0], torch.Tensor) else torch.Tensor(anchors_vec)
        self.anchor_layers = [
            AnchorBlk(in_ch, out_ch, scale_factor=2, kernels_size=(1, 3, 1), anchors_vec=self.anchors_vec[0],
                      in_ch_first=in_ch_first, out_ch_last=out_ch_last)]
        for i in range(1, 3):
            anchor_layer = self.anchor_layers[-1]
            self.anchor_layers.append(
                AnchorBlk(anchor_layer.out_ch_last, anchor_layer.out_ch_last // 2, upsample=True, scale_factor=2,
                          anchors_vec=self.anchors_vec[i], in_ch_first=anchor_layer.out_ch_last // 2 * 3,
                          out_ch_last=out_ch_last))

        self.collect_layers(self.anchor_layers)

    def __call__(self, inputs):
        """YOLOBlk
        Args:
            inputs: [in_anchor1, in_anchor2, in_anchor3]

        Returns:
            anchor1, anchor2, anchor3
        """
        ret_anchors = []
        x = None
        for i, anchor_layer in enumerate(self.anchor_layers):
            anchor, x = anchor_layer([inputs[i], x])
            ret_anchors.append(anchor)

        return ret_anchors
