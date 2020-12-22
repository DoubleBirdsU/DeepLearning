import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from base.base_model import base_model, Module


def make_divisible(v, divisor):
    """Function ensures all layers have a channel number that is divisible by 8
        url https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    Param:
        v:
        divisor:

    Returns:

    """
    return math.ceil(v / divisor) * divisor


def auto_padding(kernel_size=0, padding='same', padding_value=0.):
    if not isinstance(kernel_size, list) and not isinstance(kernel_size, tuple):
        kernel_size = list([kernel_size])
    return tuple((torch.Tensor(kernel_size).int() - 1).numpy() // 2) if 'same' in padding and padding_value == 0. else 0


def pad(kernel_size=0, padding='same', stride=1, padding_value=0.):
    pad_layer = None
    # Padding
    if 'tiny-same' == padding and kernel_size == 2 and stride == 1:
        pad_layer = nn.ZeroPad2d((0, 1, 0, 1))  # l, r, t, b; yoloV3-tiny
    elif 'same' in padding and padding_value != 0.:
        pad_layer = nn.ConstantPad2d((kernel_size - 1) // 2, padding_value)
    return pad_layer


def Activation(activation='relu', **kwargs):
    if isinstance(activation, nn.Module):
        return activation

    activation = activation.lower()
    if 'leaky' == activation:
        negativate_slope = 1e-2 if 'negativate_slope' not in kwargs else kwargs['negativate_slope']
        inplace = kwargs['inplace'] if 'inplace' in kwargs else False
        act = nn.LeakyReLU(negativate_slope, inplace)
    elif 'selu' == activation:
        inplace = kwargs['inplace'] if 'inplace' in kwargs else False
        act = nn.SELU(inplace)
    elif 'sigmoid' == activation:
        act = nn.Sigmoid()
    elif 'softmax' == activation:
        dim = kwargs['dim'] if 'dim' in kwargs else -1
        act = nn.Softmax(dim)
    else:
        act = Equ()
    return act


class Layer(Module):
    def __init__(self):
        super(Layer, self).__init__()


class Equ(Layer):
    def forward(self, x):
        return x


class Flatten(Layer):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class Concat(Layer):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class FeatureConcat(Layer):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(Layer):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    """将多个特征矩阵的值进行融合
    """

    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers 融合的特征矩阵个数
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        input_shape = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            # 根据相加的两个特征矩阵的channel选择相加方式
            if input_shape == na:  # same shape 如果channel相同，直接相加
                x = x + a
            elif input_shape > na:  # slice input 如果channel不同，将channel多的特征矩阵砍掉部分channel保证相加的channel一致
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :input_shape]

        return x


class MaxPool2D(nn.MaxPool2d):
    def __init__(self, kernel_size, stride, padding='valid', padding_value=0.):
        """MaxPool2D
            padding

        :type padding: str
        :type padding_value: float

        :param kernel_size: 内核尺寸
        :param stride: 步长
        :param padding: 'valid', 'same', 'tiny-same', 'valid': 不进行补充, 'same': 补足. default 'valid'
        :param padding_value: float
        """
        super(MaxPool2D, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=auto_padding(kernel_size, padding, padding_value))
        self.pad = pad(kernel_size, padding, stride, padding_value)

    def forward(self, x):
        return super().forward(self.pad(x) if self.pad else x)


class MixConv2d(Layer):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
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

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)


class FeatureExtractor(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='valid', groups=1, bias=True,
                 bn=False, activation='valid', pool=False, pool_size=1, pool_stride=1, **kwargs):
        """FeatureExtractor
            Conv2d, BatchNormal, Activation, Pooling

        Args:
            activation: default 'valid', 'relu', 'leaky', 'selu'. all activation has parameter 'inplace' default False;
             leaky parameter: 'negativate_slope' default 1e-2.
        """
        super(FeatureExtractor, self).__init__()
        self.add_module('Conv2d', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            auto_padding(kernel_size, padding), groups=groups, bias=not bn and bias))
        if bn:  # BatchNormal
            self.add_module('bn', nn.BatchNorm2d(out_channels))
        if 'valid' != activation:
            self.add_module('act', Activation(activation, **kwargs))  # Activation
        if pool:  # Pooling
            self.add_module('MaxPool2D', MaxPool2D(pool_size, pool_stride, padding))
        pass


class Dense(Layer):
    def __init__(self, in_channels, out_channels, activation=None, bias=True, **kwargs):
        super(Dense, self).__init__()
        self.add_module('linear', nn.Linear(in_channels, out_channels, bias))
        self.add_module('act', Activation(activation, **kwargs))


class GlobalAvgPool2D(Layer):
    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
        """
        return torch.mean(torch.mean(x, dim=-1), dim=-1)


class RoIDense(Layer):
    def __init__(self, in_ch, out_ch, roi_size, activation=None, bias=True, **kwargs):
        super(RoIDense, self).__init__()
        self.addLayers([
            nn.AdaptiveMaxPool2d(roi_size),
            nn.Flatten(),
            Dense(in_ch * roi_size.prod(), out_ch, activation, bias, **kwargs),
        ])


# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    def __init__(self, ctx):
        super(SwishImplementation, self).__init__()
        self.ctx = ctx

    def forward(self, x, **kwargs):
        self.ctx.save_for_backward(x, **kwargs)
        return x * torch.sigmoid(x)

    def backward(self, grad_output):
        x = self.ctx.saved_tensors[0]
        sx = torch.sigmoid(x)  # sigmoid(ctx)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MishImplementation(torch.autograd.Function):
    def __init__(self, ctx):
        super(MishImplementation, self).__init__()
        self.ctx = ctx

    def forward(self, x, **kwargs):
        self.ctx.save_for_backward(x, **kwargs)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    def backward(self, grad_output):
        x = self.ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientSwish(base_model):
    @staticmethod
    def forward(x):
        return SwishImplementation.apply(x)


class MemoryEfficientMish(base_model):
    @staticmethod
    def forward(x):
        return MishImplementation.apply(x)


class Swish(base_model):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class HardSwish(base_model):  # https://arxiv.org/pdf/1905.02244.pdf
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class Mish(base_model):  # https://github.com/digantamisra98/Mish
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()
