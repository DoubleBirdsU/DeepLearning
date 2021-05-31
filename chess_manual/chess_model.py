import pickle
from collections import OrderedDict

import torch
import numpy as np

from torch import nn
from base.model import NNet


def get_shape_id(shape, layer_name='layer'):
    str_id = f'{layer_name}'
    if isinstance(shape, int):
        shape = [shape]
    for i in shape:
        str_id += f'_{i}'
    return str_id


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


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Activation(activation='relu', **kwargs):
    if isinstance(activation, nn.Module):
        return activation

    activation = activation.lower()
    if 'relu' == activation:
        inplace = kwargs['inplace'] if 'inplace' in kwargs else False
        act = nn.ReLU(inplace)
    elif 'tanh' == activation:
        act = nn.Tanh()
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


class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding='valid',
                 activation='valid', bn=False, groups=1, **kwargs):
        super(Conv2D, self).__init__()
        bias = not bn and (kwargs['bias'] if 'bias' in kwargs else True)
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, auto_padding(kernel_size, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None  # BatchNormal
        self.act = Activation(activation, **kwargs) if 'valid' != activation else None  # Activation
        self.out_ch_last = out_ch

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
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


class ConvSameBnRelu2D(Conv2D):
    """ConvPBA
        padding='same'
        activation='relu'
    """

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding='same', activation='relu', bn=True, groups=1):
        super(ConvSameBnRelu2D, self).__init__(in_ch, out_ch, kernel_size, stride, padding=padding,
                                               activation=activation, bn=bn, groups=groups)
        self.out_ch_last = out_ch


class Dense(nn.Module):
    def __init__(self, in_ch, out_ch, activation='valid', bias=True, **kwargs):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias)
        self.act = Activation(activation, **kwargs) if 'valid' != activation else None
        self.out_ch_last = out_ch

    def forward(self, x):
        x = self.linear(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConcatBlock(nn.Module):
    def __init__(self, out_ch=0):
        super(ConcatBlock, self).__init__()
        self.inc_list = None
        self.out_ch = out_ch // 4

    def forward(self, x):
        return torch.cat([module(x) for module in self.inc_list], dim=1)

    def collect_layers(self):
        for i, layer in enumerate(self.inc_list):
            setattr(self, f'conv{i}', layer)


class IncResBlock(ConcatBlock):
    def __init__(self, in_ch, out_ch, kernel_n=3, activation='valid'):
        """IncResBlock
            c1-s2, c3-s2, c5-s2, pool-s2

            c1-s2: Conv1-s2

            c3-s2: Conv1 -> Conv3-s2

            c5-s2: Conv1 -> Conv5-s2

            pool-s2: MaxPooling-s2 -> Conv1

            call: concat([c1-s2(x), c3-s2(x), c5-s2(x), p-s2(x)], -3)

        Args:
            in_ch:
            out_ch:
            kernel_n:

        Returns:
            None
        """
        super(IncResBlock, self).__init__(out_ch)
        self.inc_list = [
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, kernel_size=1, activation=activation)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, (1, kernel_n), activation=activation),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, (kernel_n, 1), activation=activation)),
            nn.Sequential(
                ConvSameBnRelu2D(in_ch, self.out_ch, 1),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, (1, kernel_n), activation=activation),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, (kernel_n, 1), activation=activation),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, (1, kernel_n), activation=activation),
                ConvSameBnRelu2D(self.out_ch, self.out_ch, (kernel_n, 1), activation=activation)),
            nn.Sequential(
                MaxPool2D(3, stride=1, padding='same'),
                ConvSameBnRelu2D(in_ch, self.out_ch, 1, activation=activation))
        ]
        self.collect_layers()


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class BackBoneNetRes(NNet):
    """policy-value network module"""

    def __init__(self, board_width, board_height, in_ch=4):
        super(BackBoneNetRes, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        chess_count = board_width * board_height

        # common layers
        activation = 'relu'
        self.common_conv1 = IncResBlock(in_ch, 32, 3, activation=activation)
        self.common_conv2 = IncResBlock(32, 64, 3, activation=activation)
        self.common_conv3 = IncResBlock(64, 128, 3, activation=activation)
        self.common_conv4 = IncResBlock(128, 64, 3, activation=activation)
        self.common_conv5 = IncResBlock(64, 128, 3, activation=activation)
        self.common_conv6 = IncResBlock(128, 64, 3, activation=activation)
        self.common_conv7 = IncResBlock(64, 128, 3, activation=activation)

        self.flatten = Flatten()

        # action policy layers
        self.action_conv = Conv2D(128, 4, padding='same', bn=True, activation=activation)
        self.action_fc = Dense(chess_count * 4, chess_count, activation='logsoftmax')  # logsoftmax

        # state value layers
        self.state_conv = Conv2D(128, 2, padding='same', bn=True, activation=activation)
        self.state_fc1 = Dense(chess_count * 2, chess_count)
        self.state_fc2 = Dense(chess_count, 1, activation='tanh')  # tanh

    def forward(self, x):
        x = self.common_conv1(x)
        x = self.common_conv2(x)
        x = self.common_conv3(x)
        x = self.common_conv4(x)
        x = self.common_conv5(x)
        x = self.common_conv6(x)
        x = self.common_conv7(x)
        x_action = self.action_fc(self.flatten(self.action_conv(x)))
        x_state = self.state_fc2(self.state_fc1(self.flatten(self.state_conv(x))))
        return x_action, x_state

    def freeze_all_conv(self):
        conv_list = [self.common_conv1, self.common_conv2, self.common_conv3, self.action_conv, self.state_conv, ]
        for conv in conv_list:
            for layer in conv.children():
                for param in layer.parameters():
                    param.requires_grad = False
        pass


class BackBoneNetBig(NNet):
    """policy-value network module"""

    def __init__(self, board_width, board_height, in_ch=4):
        super(BackBoneNetBig, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        chess_count = board_width * board_height

        # common layers
        activation = 'relu'
        self.common_conv1 = ConvSameBnRelu2D(in_ch, 32, 3, activation=activation)
        self.common_conv2 = ConvSameBnRelu2D(32, 64, 3, activation=activation)
        self.common_conv3 = ConvSameBnRelu2D(64, 128, 3, activation=activation)
        self.common_conv4 = ConvSameBnRelu2D(128, 64, 3, activation=activation)
        self.common_conv5 = ConvSameBnRelu2D(64, 128, 3, activation=activation)
        self.common_conv6 = ConvSameBnRelu2D(128, 64, 3, activation=activation)
        self.common_conv7 = ConvSameBnRelu2D(64, 128, 3, activation=activation)

        self.flatten = Flatten()

        # action policy layers
        self.action_conv = Conv2D(128, 4, padding='same', bn=True, activation=activation)
        self.action_fc = Dense(chess_count * 4, chess_count, activation='logsoftmax')  # logsoftmax

        # state value layers
        self.state_conv = Conv2D(128, 2, padding='same', bn=True, activation=activation)
        self.state_fc1 = Dense(chess_count * 2, chess_count)
        self.state_fc2 = Dense(chess_count, 1, activation='tanh')  # tanh

    def forward(self, x):
        x = self.common_conv1(x)
        x = self.common_conv2(x)
        x = self.common_conv3(x)
        x = self.common_conv4(x)
        x = self.common_conv5(x)
        x = self.common_conv6(x)
        x = self.common_conv7(x)
        x_action = self.action_fc(self.flatten(self.action_conv(x)))
        x_state = self.state_fc2(self.state_fc1(self.flatten(self.state_conv(x))))
        return x_action, x_state

    def freeze_all_conv(self):
        conv_list = [self.common_conv1, self.common_conv2, self.common_conv3, self.action_conv, self.state_conv, ]
        for conv in conv_list:
            for layer in conv.children():
                for param in layer.parameters():
                    param.requires_grad = False
        pass


class BackBoneNetBTiny(NNet):
    """policy-value network module"""

    def __init__(self, board_width, board_height, in_ch=4):
        super(BackBoneNetBTiny, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        chess_count = board_width * board_height

        # common layers
        activation = 'relu'
        self.common_conv1 = Conv2D(in_ch, 32, kernel_size=3, padding='same', activation=activation)
        self.common_conv2 = Conv2D(32, 64, kernel_size=3, padding='same', activation=activation)
        self.common_conv3 = Conv2D(64, 128, kernel_size=3, padding='same', activation=activation)
        self.common_conv4 = Conv2D(128, 64, kernel_size=3, padding='same', activation=activation)
        self.common_conv5 = Conv2D(64, 128, kernel_size=3, padding='same', activation=activation)

        self.flatten = Flatten()

        # action policy layers
        self.action_conv = Conv2D(128, 4, activation=activation)
        self.action_fc = Dense(chess_count * 4, chess_count, activation='logsoftmax')  # logsoftmax

        # state value layers
        self.state_conv = Conv2D(128, 2, padding=activation)
        self.state_fc1 = Dense(chess_count * 2, 64)
        self.state_fc2 = Dense(64, 1, activation='tanh')  # tanh

    def forward(self, x):
        x = self.common_conv1(x)
        x = self.common_conv2(x)
        x = self.common_conv3(x)
        x = self.common_conv4(x)
        x = self.common_conv5(x)
        x_action = self.action_fc(self.flatten(self.action_conv(x)))
        x_state = self.state_fc2(self.state_fc1(self.flatten(self.state_conv(x))))
        return x_action, x_state

    def freeze_all_conv(self):
        conv_list = [self.common_conv1, self.common_conv2, self.common_conv3, self.action_conv, self.state_conv, ]
        for conv in conv_list:
            for layer in conv.children():
                for param in layer.parameters():
                    param.requires_grad = False

    def load_weight(self, file, replace_params=None):
        try:
            net_params = pickle.load(open(file, 'rb'), encoding='bytes')
        except:
            net_params = torch.load(file)

        if replace_params:
            net_params = self.replace_weights(net_params, orig_file=replace_params)
        self.load_state_dict(net_params)
        pass


class BackBoneNetTiny(NNet):
    """policy-value network module"""

    def __init__(self, board_width, board_height, in_ch=4):
        super(BackBoneNetTiny, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        chess_count = board_width * board_height

        # common layers
        activation = 'relu'
        self.common_conv1 = Conv2D(in_ch, 32, kernel_size=3, padding='same', activation=activation)
        self.common_conv2 = Conv2D(32, 64, kernel_size=3, padding='same', activation=activation)
        self.common_conv3 = Conv2D(64, 128, kernel_size=3, padding='same', activation=activation)

        self.flatten = Flatten()

        # action policy layers
        self.action_conv = Conv2D(128, 4, activation=activation)
        self.action_fc = Dense(chess_count * 4, chess_count, activation='logsoftmax')  # logsoftmax

        # state value layers
        self.state_conv = Conv2D(128, 2, padding='valid')
        self.state_fc1 = Dense(chess_count * 2, 64)
        self.state_fc2 = Dense(64, 1, activation='tanh')  # tanh

    def forward(self, x):
        x = self.common_conv1(x)
        x = self.common_conv2(x)
        x = self.common_conv3(x)
        x_action = self.action_fc(self.flatten(self.action_conv(x)))
        x_state = self.state_fc2(self.state_fc1(self.flatten(self.state_conv(x))))
        return x_action, x_state

    def freeze_all_conv(self):
        conv_list = [self.common_conv1, self.common_conv2, self.common_conv3, self.action_conv, self.state_conv, ]
        for conv in conv_list:
            for layer in conv.children():
                for param in layer.parameters():
                    param.requires_grad = False

    def load_weight(self, file, replace_params=None):
        try:
            net_params = pickle.load(open(file, 'rb'), encoding='bytes')
        except:
            net_params = torch.load(file)

        if replace_params:
            net_params = self.replace_weights(net_params, orig_file=replace_params)
        self.load_state_dict(net_params)
        pass


class BackBoneNetTiny_v2(NNet):
    """policy-value network module"""

    def __init__(self, board_width, board_height, in_ch=4):
        super(BackBoneNetTiny_v2, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        chess_count = board_width * board_height

        # common layers
        self.common_conv1 = ConvSameBnRelu2D(in_ch, 32, kernel_size=3)
        self.common_conv2 = ConvSameBnRelu2D(32, 64, kernel_size=3)
        self.common_conv3 = ConvSameBnRelu2D(64, 128, kernel_size=3)
        self.common_conv4 = ConvSameBnRelu2D(128, 64, kernel_size=3)
        self.common_conv5 = ConvSameBnRelu2D(64, 128, kernel_size=3)

        self.flatten = Flatten()

        # action policy layers
        self.action_conv = ConvSameBnRelu2D(128, 4)
        self.action_fc = Dense(chess_count * 4, chess_count, activation='logsoftmax')  # logsoftmax

        # state value layers
        self.state_conv = ConvSameBnRelu2D(128, 2)
        self.state_fc1 = Dense(chess_count * 2, chess_count)
        self.state_fc2 = Dense(chess_count, 1, activation='tanh')  # tanh

    def forward(self, x):
        x = self.common_conv1(x)
        x = self.common_conv2(x)
        x = self.common_conv3(x)
        x = self.common_conv4(x)
        x = self.common_conv5(x)
        x_action = self.action_fc(self.flatten(self.action_conv(x)))
        x_state = self.state_fc2(self.state_fc1(self.flatten(self.state_conv(x))))
        return x_action, x_state

    def freeze_all_conv(self):
        conv_list = [self.common_conv1, self.common_conv2, self.common_conv3, self.action_conv, self.state_conv, ]
        for conv in conv_list:
            for layer in conv.children():
                for param in layer.parameters():
                    param.requires_grad = False

    def load_weight(self, file, replace_params=None):
        try:
            net_params = pickle.load(open(file, 'rb'), encoding='bytes')
        except:
            net_params = torch.load(file)

        if replace_params:
            net_params = self.replace_weights(net_params, orig_file=replace_params)
        self.load_state_dict(net_params)
        pass

    def replace_weights(self, net_params, orig_file='best_policy_8_8_5.model', *args, **kwargs):
        try:
            orig_params = pickle.load(open(orig_file, 'rb'), encoding='bytes')
        except:
            orig_params = torch.load(orig_file)

        keys = list()
        for key in net_params:
            keys.append((key, net_params[key].shape))
        net_params = self.get_weights(orig_params, 'pickle', keys)
        return net_params

    @staticmethod
    def get_weights(orig_param, pickle_module='pickle', keys=None):
        weights_dict = dict()
        if pickle_module == 'pickle':
            weights_dict = OrderedDict()
            for i, weight in enumerate(orig_param):
                (key, obj_shape) = keys[i]
                if len(weight.shape) == 2:
                    weight = np.transpose(weight, [1, 0])
                elif len(weight.shape) == 4:
                    weight = np.transpose(weight, [0, 1, 3, 2])

                shape = weight.shape
                if get_shape_id(shape) != get_shape_id(obj_shape):
                    if len(shape) > 0:
                        num_weight = obj_shape[0] // shape[0] + 1
                        weight = np.concatenate([weight for _ in range(num_weight)],
                                                axis=0)[:obj_shape[0], ...]
                    if len(shape) > 1:
                        num_weight = obj_shape[1] // shape[1] + 1
                        weight = np.concatenate([weight for _ in range(num_weight)],
                                                axis=1)[:, :obj_shape[1], ...]
                    if len(shape) > 2:
                        num_weight = obj_shape[2] // shape[2] + 1
                        weight = np.concatenate([weight for _ in range(num_weight)],
                                                axis=2)[:, :, :obj_shape[2], ...]
                    if len(shape) > 3:
                        num_weight = obj_shape[3] // shape[3] + 1
                        weight = np.concatenate([weight for _ in range(num_weight)],
                                                axis=2)[:, :, :, :obj_shape[2], ...]

                weights_dict[key] = torch.Tensor(weight)
        elif pickle_module == 'torch':
            weights_dict = OrderedDict()
            for key in orig_param:
                if '.Conv2d.' in key:
                    str_key = key.replace('Conv2d', 'conv')
                    weights_dict[str_key] = orig_param[key]
                elif '.conv.' in key:
                    weights_dict[key] = orig_param[key]
                elif '.bn.' in key:
                    weights_dict[key] = orig_param[key]
                elif '.linear.' in key:
                    weights_dict[key] = orig_param[key]
                elif '.liner.' in key:
                    str_key = key.replace('liner', 'linear')
                    weights_dict[str_key] = orig_param[key]
        return weights_dict


class BackBoneNetTiny_v3(NNet):
    """policy-value network module"""

    def __init__(self, board_width, board_height, in_ch=4):
        super(BackBoneNetTiny_v3, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        chess_count = board_width * board_height

        # common layers
        self.common_conv1 = ConvSameBnRelu2D(in_ch, 32, kernel_size=3)
        self.common_conv2 = ConvSameBnRelu2D(32, 32, kernel_size=3)
        self.common_conv3 = ConvSameBnRelu2D(32, 64, kernel_size=3)
        self.common_conv4 = ConvSameBnRelu2D(64, 128, kernel_size=3)

        self.flatten = Flatten()

        # action policy layers
        self.action_conv = ConvSameBnRelu2D(128, 4)
        self.action_fc = Dense(chess_count * 4, chess_count, activation='logsoftmax')  # logsoftmax

        # state value layers
        self.state_conv = ConvSameBnRelu2D(128, 2)
        self.state_fc1 = Dense(chess_count * 2, chess_count, activation='relu')
        self.state_fc2 = Dense(chess_count, 1, activation='tanh')  # tanh

    def forward(self, x):
        x = self.common_conv1(x)
        x = self.common_conv2(x)
        x = self.common_conv3(x)
        x = self.common_conv4(x)
        x_action = self.action_fc(self.flatten(self.action_conv(x)))
        x_state = self.state_fc2(self.state_fc1(self.flatten(self.state_conv(x))))
        return x_action, x_state

    def freeze_all_conv(self):
        conv_list = [self.common_conv1, self.common_conv2, self.common_conv3,
                     self.common_conv4, self.action_conv, self.state_conv, ]
        for conv in conv_list:
            for layer in conv.children():
                for param in layer.parameters():
                    param.requires_grad = False
    pass
