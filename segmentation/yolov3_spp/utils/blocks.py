import torch.nn as nn

from utils.layers import MaxPool2D
from utils.base_model import base_model


class FeatureExtractorBlk(base_model):
    def __init__(self, in_ch, out_ch, kernel_size=(1, 1), padding='valid',
                 stride=1, bias=False, bn=False, activation='relu',
                 pool=False, pool_size=(1, 1), pool_stride=1, pool_padding='valid', dp=0):
        """FeatureExtractorBlk is 'CBAPD'.
            Conv2D, BatchNormal, Activation, MaxPool2D, Dropout

        :rtype: None
        :type in_ch: Any
        :type out_ch: Any
        :type kernel_size: Any
        :type activation: Any
        :type stride: int
        :type padding: str
        :type bias: bool
        :type pool_size: Tuple[int, int]
        :type pool_stride: Any
        :type pool_padding: Any
        :type dp: float

        :param padding: 'valid', 'same', 'tiny-same'
        :param activation: 'relu', 'leaky'
        :param pool_padding: 'valid', 'same', 'tiny-same'
        """
        super(FeatureExtractorBlk, self).__init__()
        self.pad = None
        if padding == 'same':
            pass
        elif padding == 'tiny-same':
            if kernel_size == 2 and stride == 1:
                self.pad = nn.ZeroPad2d((0, 1, 0, 1))

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                              padding=0 if self.pad else (kernel_size - 1) // 2,
                              groups=1, bias=bias and not bn)

        self.bn = nn.BatchNorm2d(out_ch) if bn else None

        if activation == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:  # default ReLU
            self.act = nn.ReLU(inplace=True)

        self.pool = MaxPool2D(pool_size, pool_stride, pool_padding) if pool else None
        self.dp = nn.Dropout(p=dp) if dp else None

    def forward(self, x):
        x = self.conv(x if not self.pad else self.pad(x))  # pad -> conv
        x = self.act(x if not self.bn else self.bn(x))  # bn -> activation
        if self.pool:
            x = self.pool(x)
        if self.dp:
            x = self.dp(x)
        return x


class DarknetBlk(base_model):
    def __init__(self):
        super(DarknetBlk, self).__init__()

    def forward(self, x):
        return x
