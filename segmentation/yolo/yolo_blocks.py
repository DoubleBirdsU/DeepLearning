import torch.nn as nn

from base.blocks import ModuleBlock
from base.layers import ConvSameBnRelu2D, Concat, Layer, Shortcut, UpSample


def create_yolo_blocks(layer_param, in_ch, channels, idx_from=None):
    module_type, args, kwargs = layer_param
    args.insert(0, in_ch)
    out_ch = kwargs['out_ch'] if 'out_ch' in kwargs else args[1]

    layer = None
    if 'DarkBlock' == module_type:
        layer = DarkBlock(*args, **kwargs)
    elif 'DarkCatBlock' == module_type:
        layer = DarkCatBlock(*args, **kwargs)
    elif 'UpSampleBlock' == module_type:
        layer = UpSampleBlock(*args, **kwargs)
    elif 'YoloNeck' == module_type:
        args[0] = args[0][0]  # 原始的 args[0] 是一个 list[int], args[0][0]
        layer = YoloNeck(*args, **kwargs)
    elif 'YoloBlock' == module_type:
        layer = YoloBlock(*args, **kwargs)
    elif 'YoloClassifier' == module_type:
        layer = YoloClassifier(*args, **kwargs)
    out_ch = layer.out_ch_last if layer.out_ch_last is not None else out_ch
    return layer, out_ch


def check_kernel_stride(kernels_size, strides, num=6):
    if isinstance(kernels_size, int):
        kernels_size = [kernels_size for _ in range(num)]
    elif len(kernels_size) < num:
        len_size = len(kernels_size)
        kernels_size = [kernels_size[k % len_size] for k in range(num)]
    if isinstance(strides, int):
        strides = [strides for _ in range(num)]
    elif len(strides) < num:
        len_size = len(strides)
        strides = [strides[k % len_size] for k in range(num)]
    return kernels_size, strides


class DarkBlock(Layer):
    def __init__(self, in_ch, out_ch, kernel_size=1, activation='valid', residual_path='equal', **kwargs):
        super(DarkBlock, self).__init__()
        self.blocks = nn.Sequential(
            ConvSameBnRelu2D(in_ch, out_ch, activation=activation),
            ConvSameBnRelu2D(out_ch, out_ch, kernel_size, activation=activation),
        )
        self.shortcut = Shortcut(residual_path=residual_path)
        self.addLayers([self.blocks, self.shortcut])
        pass

    def forward(self, x):
        return self.shortcut(x, self.blocks(x))


class DarkCatBlock(Layer):
    def __init__(self, in_chs, out_ch, activation='valid', dim=1, **kwargs):
        super(DarkCatBlock, self).__init__()
        self.conv1 = ConvSameBnRelu2D(in_chs[0], out_ch, activation=activation)
        self.conv2 = ConvSameBnRelu2D(in_chs[1], out_ch, activation=activation)
        self.concat = Concat(dim=dim)
        self.out_ch_last = in_chs[0] + in_chs[1]
        self.addLayers([self.conv1, self.conv2, self.concat])
        pass

    def forward(self, *args):
        return self.concat([self.conv1(args[0]), self.conv2(args[1])])


class UpSampleBlock(Layer):
    def __init__(self, in_ch, out_ch, hid_ch, activation='valid', **kwargs):
        super(UpSampleBlock, self).__init__()
        us_size = kwargs['us_size'] if 'us_size' in kwargs else None
        us_stride = kwargs['us_stride'] if 'us_stride' in kwargs else None
        us_mode = kwargs['us_mode'] if 'us_mode' in kwargs else 'nearest'
        self.us_block = nn.Sequential(
            ConvSameBnRelu2D(in_ch, hid_ch, activation=activation),
            ConvSameBnRelu2D(hid_ch, hid_ch * 2, kernel_size=3, activation=activation),
            ConvSameBnRelu2D(hid_ch * 2, hid_ch, activation=activation),
            ConvSameBnRelu2D(hid_ch, out_ch, activation=activation),
            UpSample(size=us_size, stride=us_stride, mode=us_mode),
        )
        self.addLayers(self.us_block)
        pass

    def forward(self, x):
        return self.us_block(x)


class YoloNeck(Layer):
    def __init__(self, in_ch, out_ch, hid_ch=None, kernel_size=1, stride=1, activation='valid', dim=0, **kwargs):
        """YoloNeck

        Args:
            in_ch (int):
            out_ch (int):
            hid_ch (int):
            kernel_size (int): default 1.
            stride (int): default 1.
            activation (Union[str, Module]): default 'valid'.
            dim (int): default 0.

        Returns:
            Module
        """
        super(YoloNeck, self).__init__()
        if hid_ch is None:
            hid_ch = out_ch // 2
        self.conv = ConvSameBnRelu2D(in_ch, hid_ch, kernel_size=kernel_size, stride=stride, activation=activation)
        self.cat = Concat(dim=dim)

        self.out_ch_last = out_ch
        self.addLayers([self.conv, self.cat])

    def forward(self, args):
        return self.cat([self.conv(args[0]), args[1]])


class YoloBlock(ModuleBlock):
    def __init__(self, in_ch, out_ch, hid_ch=None, kernels_size=1, strides=1, activation='valid', **kwargs):
        """YoloBlock

        Args:
            in_ch (int):
            out_ch (int):
            hid_ch (int):
            kernels_size (Union[int, List[int]]):
            strides (Union[int, List[int]]):

        Returns:
            Module
        """
        super(YoloBlock, self).__init__()
        kernels_size, strides = check_kernel_stride(kernels_size, strides)
        if hid_ch is None:
            hid_ch = out_ch
        self.block = nn.Sequential(
            ConvSameBnRelu2D(in_ch, hid_ch, kernels_size[0], strides[0], activation=activation),
            ConvSameBnRelu2D(hid_ch, in_ch, kernels_size[1], strides[1], activation=activation),
            ConvSameBnRelu2D(in_ch, hid_ch, kernels_size[2], strides[2], activation=activation),
            ConvSameBnRelu2D(hid_ch, in_ch, kernels_size[3], strides[3], activation=activation),
            ConvSameBnRelu2D(in_ch, out_ch, kernels_size[4], strides[4], activation=activation),
        )
        self.out_ch_last = out_ch
        self.addLayers(self.block)
        pass


class YoloClassifier(ModuleBlock):
    def __init__(self, in_ch, num_cls, hid_ch, kernel_size=1, stride=1, activation='valid', anchors=None, **kwargs):
        """YoloClassifier

        Args:
            in_ch (int):
            num_cls (int):
            hid_ch (int):
            kernel_size (Union[int, List[int]]):
            stride (Union[int, List[int]]):
            activation (Union[str, Module]):
            anchors (List[int]):

        Returns:
            Module
        """
        super(YoloClassifier, self).__init__()
        self.anchors = anchors
        self.num_cls = num_cls
        self.num_anchors = kwargs['num_anchors'] if 'num_anchors' in kwargs else 3
        self.iou = kwargs['iou'] if 'iou' in kwargs else 'iou'
        self.nms = kwargs['nms'] if 'nms' in kwargs else 'nms'
        self.out_ch_last = self.num_anchors * (num_cls + 4 + 1)
        self.conv = ConvSameBnRelu2D(in_ch, hid_ch, 3, activation=activation)
        self.anchor = ConvSameBnRelu2D(hid_ch, self.out_ch_last, kernel_size, stride, 'valid')
        self.flatten = nn.Flatten(start_dim=-2)
        self.addLayers([self.conv, self.anchor, self.flatten])
        pass

    import torch

    def forward(self, x: torch.Tensor):
        anchors = self.flatten(self.anchor(self.conv(x)))
        return anchors
