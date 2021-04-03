import torch
import torch.nn as nn
import torch.nn.functional as F


def Activation(activation='relu', **kwargs):
    if isinstance(activation, nn.Module):
        return activation

    activation = activation.lower()
    if 'relu' == activation:
        inplace = kwargs['inplace'] if 'inplace' in kwargs else False
        act = nn.ReLU(inplace)
    elif 'leaky' == activation:
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
    elif 'logsoftmax' == activation:
        dim = kwargs['dim'] if 'dim' in kwargs else -1
        act = nn.LogSoftmax(dim)
    else:
        act = None
    return act


# 激活函数基类
class ActivationBase(nn.Module):
    def __init__(self):
        super(ActivationBase, self).__init__()


# Swish https://arxiv.org/pdf/1905.02244.pdf
class Swish(ActivationBase):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class HardSwish(ActivationBase):
    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class MemoryEfficientSwish(ActivationBase):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


# Mish https://github.com/digantamisra98/Mish
class Mish(ActivationBase):
    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(ActivationBase):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


# FReLU https://arxiv.org/abs/2007.11824
class FReLU(ActivationBase):
    def __init__(self, in_ch, kernel_size=3):
        super(FReLU, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size, 1, 1, groups=in_ch)
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))


# Activation functions below
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
