import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.base_model import base_module


def make_divisible(v, divisor):
    """Function ensures all layers have a channel number that is divisible by 8
        url https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    Param:
        v:
        divisor:

    Returns:

    """
    return math.ceil(v / divisor) * divisor


class Flatten(base_module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def __call__(x):
        return x.view(x.size(0), -1)


class Concat(base_module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def __call__(self, x):
        return torch.cat(x, self.d)


class FeatureConcat(base_module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def __call__(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(base_module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    """将多个特征矩阵的值进行融合
    """

    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.bool_weight = weight  # apply weights boolean
        self.num_layers = len(layers) + 1  # number of layers 融合的特征矩阵个数
        if weight:
            self.weight = nn.Parameter(torch.zeros(self.num_layers), requires_grad=True)  # layer weights

    def __call__(self, x, outputs):
        # Weights
        if self.bool_weight:
            w = torch.sigmoid(self.weight) * (2 / self.num_layers)  # sigmoid weights (0-1)
            x = x * w[0]
        else:
            # raise KeyError("backward doesn't have the param 'w'.")
            pass

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.num_layers - 1):
            a = outputs[self.layers[i]]
            if self.bool_weight:
                a *= w[i + 1]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            # 根据相加的两个特征矩阵的channel选择相加方式
            if nx == na:  # same shape 如果channel相同，直接相加
                x = x + a
            elif nx > na:  # slice input 如果channel不同，将channel多的特征矩阵砍掉部分channel保证相加的channel一致
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class Conv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='valid', groups=1, bias=False):
        super(Conv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding=kernel_size // 2 if padding == 'valid' else 0,
                                     groups=groups, bias=bias)

    def _forward_unimplemented(self, *input: Any) -> None:
        return super(Conv2D, self)._forward_unimplemented(*input)


class Activation(base_module):
    def __init__(self, activation=None, inplace=True, **kwargs):
        b"""Activation

        Param:
            activation: 'leaky', 'relu', 'sigmoid'

        Returns:
            None
        """
        super(Activation, self).__init__()
        if isinstance(activation, nn.Module):
            self.act = activation
        elif isinstance(activation, str):
            if activation == 'leaky':
                negative_slope = kwargs['negative_slope'] if 'negative_slope' in kwargs else 1e-2
                self.act = nn.LeakyReLU(negative_slope, inplace)
            elif activation == 'relu':
                self.act = nn.ReLU(inplace)
            elif activation == 'sigmoid':
                self.act = nn.Sigmoid()

    def __call__(self, x):
        return self.act(x)


class MaxPool2D(nn.MaxPool2d):
    def __init__(self, kernel_size, stride, padding='valid', padding_value=0.):
        """MaxPool2D
            padding

        :type padding: str
        :type padding_value: float

        :param kernel_size: 内核尺寸
        :param stride: 步长
        :param padding: 'valid', 'same', 'tiny-same'; 'valid': 不进行补充, 'same': 补足. default 'valid'
        :param padding_value: float
        """
        super(MaxPool2D, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if 'same' in padding and padding_value == 0. else 0)
        self.pad = None
        if padding == 'tiny-same' and kernel_size == 2 and stride == 1:
            self.pad = nn.ZeroPad2d((0, 1, 0, 1))  # l, r, t, b; yoloV3-tiny
        elif padding_value != 0.:
            self.pad = nn.ConstantPad2d((kernel_size - 1) // 2, padding_value)

    def __call__(self, x):
        return super().forward(self.pad(x) if self.pad else x)

    def _forward_unimplemented(self, *input: Any) -> None:
        return super(MaxPool2D, self)._forward_unimplemented(*input)


class MixConv2d(base_module):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self, in_ch, out_ch, kernel_size=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()

        groups = len(kernel_size)
        if method == 'equal_ch':  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(kernel_size) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(in_channels=in_ch,
                      out_channels=ch[g],
                      kernel_size=kernel_size[g],
                      stride=stride,
                      padding=kernel_size[g] // 2,  # 'same' pad
                      dilation=dilation,
                      bias=bias) for g in range(groups)])

    def __call__(self, x):
        return torch.cat([m(x) for m in self.m], 1)


# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(x, **kwargs):
        if 'ctx' in kwargs:
            ctx = kwargs.pop('ctx')
        else:
            raise KeyError("forward doesn't have the param 'ctx'.")
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(grad_output, **kwargs):
        if 'ctx' in kwargs:
            ctx = kwargs.pop('ctx')
        else:
            raise KeyError("forward doesn't have the param 'ctx'.")
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)  # sigmoid(ctx)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(x, **kwargs):
        if 'ctx' in kwargs:
            ctx = kwargs.pop('ctx')
        else:
            raise KeyError("forward doesn't have the param 'ctx'.")
        ctx.save_for_backward(x, **kwargs)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(grad_output, **kwargs):
        if 'ctx' in kwargs:
            ctx = kwargs.pop('ctx')
        else:
            raise KeyError("backward doesn't have the param 'ctx'.")
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientSwish(base_module):
    def __call__(self, x):
        return SwishImplementation.apply(x)


class MemoryEfficientMish(base_module):
    def __call__(self, x):
        return MishImplementation.apply(x)


class Swish(base_module):
    def __call__(self, x):
        return x * torch.sigmoid(x)


class HardSwish(base_module):  # https://arxiv.org/pdf/1905.02244.pdf
    def __call__(self, x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class Mish(base_module):  # https://github.com/digantamisra98/Mish
    def __call__(self, x):
        return x * F.softplus(x).tanh()
