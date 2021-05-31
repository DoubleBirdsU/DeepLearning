import torch
import math
import numpy as np
import torch.nn as nn

from base.base_model import base_model, Module
from base.share.activations import Mish


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
        w = None
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
            ch = [(i == groups).sum() for groups in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, kernel_size=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(kernel_size) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(in_channels=in_ch,
                      out_channels=ch[groups],
                      kernel_size=kernel_size[groups],
                      stride=stride,
                      padding=kernel_size[groups] // 2,  # 'same' pad
                      dilation=dilation,
                      bias=bias) for groups in range(groups)])

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


# -------------------------------------------------------------------------------------------------------------------- #
def DWConv(in_ch, out_ch, kernel_size=1, stride=1, act=None):
    # Depthwise convolution
    return Conv(in_ch, out_ch, kernel_size, stride, groups=math.gcd(in_ch, out_ch), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding='same', groups=1, act=True):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                              auto_padding(kernel_size, padding), groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_ch, out_ch, shortcut=True, groups=1, expansion=0.5):
        # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        hid_ch = int(out_ch * expansion)  # hidden channels
        self.conv1 = Conv(in_ch, hid_ch, 1, 1)
        self.conv2 = Conv(hid_ch, out_ch, 3, 1, groups=groups)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_ch, out_ch, n=1, shortcut=True, groups=1, expansion=0.5):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        hid_ch = int(out_ch * expansion)  # hidden channels
        self.conv1 = Conv(in_ch, hid_ch, 1, 1)
        self.conv2 = nn.Conv2d(in_ch, hid_ch, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hid_ch, hid_ch, 1, 1, bias=False)
        self.conv4 = Conv(2 * hid_ch, out_ch, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hid_ch)  # applied to cat(cv2, cv3)
        self.act = Mish()
        self.m = nn.Sequential(*[Bottleneck(hid_ch, hid_ch, shortcut, groups, expansion=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_ch, out_ch, n=1, shortcut=False, groups=1, e=0.5):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        hid_ch = int(out_ch)  # hidden channels
        self.conv1 = Conv(in_ch, hid_ch, 1, 1)
        self.conv2 = nn.Conv2d(hid_ch, hid_ch, 1, 1, bias=False)
        self.conv3 = Conv(2 * hid_ch, out_ch, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hid_ch)
        self.act = Mish()
        self.m = nn.Sequential(*[Bottleneck(hid_ch, hid_ch, shortcut, groups, expansion=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.m(x1)
        y2 = self.conv2(x1)
        return self.conv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class VoVCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_ch, out_ch, n=1, shortcut=True, groups=1, e=0.5):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(VoVCSP, self).__init__()
        hid_ch = int(out_ch)  # hidden channels
        self.conv1 = Conv(in_ch // 2, hid_ch // 2, 3, 1)
        self.conv2 = Conv(hid_ch // 2, hid_ch // 2, 3, 1)
        self.conv3 = Conv(hid_ch, out_ch, 1, 1)

    def forward(self, x):
        _, x1 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        return self.conv3(torch.cat((x1, x2), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, in_ch, out_ch, kernel_size=(5, 9, 13)):
        super(SPP, self).__init__()
        hid_ch = in_ch // 2  # hidden channels
        self.conv1 = Conv(in_ch, hid_ch, 1, 1)
        self.conv2 = Conv(hid_ch * (len(kernel_size) + 1), out_ch, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel_size])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_ch, out_ch, n=1, shortcut=False, groups=1, e=0.5, kernel_size=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        hid_ch = int(2 * out_ch * e)  # hidden channels
        self.conv1 = Conv(in_ch, hid_ch, 1, 1)
        self.conv2 = nn.Conv2d(in_ch, hid_ch, 1, 1, bias=False)
        self.conv3 = Conv(hid_ch, hid_ch, 3, 1)
        self.conv4 = Conv(hid_ch, hid_ch, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel_size])
        self.conv5 = Conv(4 * hid_ch, hid_ch, 1, 1)
        self.conv6 = Conv(hid_ch, hid_ch, 3, 1)
        self.bn = nn.BatchNorm2d(2 * hid_ch)
        self.act = Mish()
        self.conv7 = Conv(2 * hid_ch, out_ch, 1, 1)

    def forward(self, x):
        x1 = self.conv4(self.conv3(self.conv1(x)))
        y1 = self.conv6(self.conv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.conv2(x)
        return self.conv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class MP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, kernel_size=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        return self.m(x)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding='same', groups=1, act=True):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(in_ch * 4, out_ch, kernel_size, stride, padding, groups, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Classify(nn.Module):
    # Classification head, i.e. x(b,in_ch,20,20) to x(b,out_ch)
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding='same', groups=1):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,in_ch,1,1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, auto_padding(kernel_size, padding), groups=groups,
                              bias=False)  # to x(b,out_ch,1,1)
        self.flat = Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,out_ch)


# This file contains experimental modules
class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        hid_ch = int(out_ch * e)  # hidden channels
        self.conv1 = Conv(in_ch, hid_ch, (1, kernel_size), (1, stride))
        self.conv2 = Conv(hid_ch, out_ch, (kernel_size, 1), (stride, 1), groups=groups)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class CrossConvCSP(nn.Module):
    # Cross Convolution CSP
    def __init__(self, in_ch, out_ch, n=1, shortcut=True, groups=1, e=0.5):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(CrossConvCSP, self).__init__()
        hid_ch = int(out_ch * e)  # hidden channels
        self.conv1 = Conv(in_ch, hid_ch, 1, 1)
        self.conv2 = nn.Conv2d(in_ch, hid_ch, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hid_ch, hid_ch, 1, 1, bias=False)
        self.conv4 = Conv(2 * hid_ch, out_ch, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hid_ch)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[CrossConv(hid_ch, hid_ch, 3, 1, groups, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):
        # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, groups=1, act=True):
        # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        hid_ch = out_ch // 2  # hidden channels
        self.conv1 = Conv(in_ch, hid_ch, kernel_size, stride, groups, act)
        self.conv2 = Conv(hid_ch, hid_ch, 5, 1, hid_ch, act)

    def forward(self, x):
        y = self.conv1(x)
        return torch.cat([y, self.conv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(GhostBottleneck, self).__init__()
        hid_ch = out_ch // 2
        self.conv = nn.Sequential(
            GhostConv(in_ch, hid_ch, 1, 1),  # pw
            DWConv(hid_ch, hid_ch, kernel_size, stride, act=False) if stride == 2 else nn.Identity(),  # dw
            GhostConv(hid_ch, out_ch, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(in_ch, in_ch, kernel_size, stride, act=False),
                                      Conv(in_ch, out_ch, 1, 1, act=False)) if stride == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
