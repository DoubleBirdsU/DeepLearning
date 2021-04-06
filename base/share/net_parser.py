from torch import nn

import base.blocks as blocks
import base.layers as layers


class NetParser:
    def __init__(self, net, cfg):
        self.net = net
        self.cfg = cfg
        pass

    def create_net(self, include_top=True):
        self.net.channels = [self.cfg['img_size'][0]]
        for param in self.cfg['backbone']:
            idx_from, block_params = param[0], param[1:]
            block = self.create_sequential(block_params, idx_from)
            self.net.cfg_from_list.append(idx_from)
            self.net.cfg_module_list.append(block)

        if include_top:
            for param in self.cfg['head']:
                idx_from, block_params = param[0], param[1:]
                block = self.create_sequential(block_params, idx_from)
                self.net.cfg_from_list.append(idx_from)
                self.net.cfg_module_list.append(block)
        self.net.addLayers(self.net.cfg_module_list)

    def create_sequential(self, block_param, idx_from=None):
        num_layer, layer_param = block_param[0], block_param[1:]
        if num_layer == 1:
            sequential = self.get_module(layer_param, idx_from)
        else:
            sequential = nn.Sequential()
            for i in range(num_layer):
                layer = self.get_module(layer_param, idx_from)
                sequential.add_module(f'{layer_param[0]}_{i}', layer)
        return sequential

    def get_module(self, layer_param, idx_from=None):
        """
        get_module 获取模块

        生成模块

        Args:
            layer_param (List[str, List[Union[str, int]]]): 模块信息参数
            idx_from (Union[int, List[int]]), optional): 模块输入. Defaults to None.

        Returns:
            Module: 模块
        """
        in_ch, out_ch = self.net.channels[-1], self.net.channels[-1]
        layer_param = layer_param[:1] + self.check_args(layer_param[1:])

        channels = None if idx_from is None else self.net.channels[idx_from]
        layer, out_ch = self.create_layer(in_ch, channels, layer_param)
        if layer is None:
            layer, out_ch = self.create_block(layer_param, self.cfg, in_ch, self.net.layers_fn)
        self.net.channels.append(out_ch)
        return layer

    def check_args(self, layer_params):
        layer_args, layer_kwargs = [], {}
        for param in layer_params:
            if isinstance(param, list):
                layer_args = param
            elif isinstance(param, dict):
                layer_kwargs = param

        for i, arg in enumerate(layer_args):
            if isinstance(arg, str) and arg in self.cfg:
                layer_args[i] = self.cfg[arg]

        for key, value in layer_kwargs.items():
            if isinstance(value, str) and value in self.cfg:
                layer_kwargs[key] = self.cfg[value]
        return [layer_args, layer_kwargs]

    @staticmethod
    def create_layer(in_ch, channels, layer_param):
        layer = None
        out_ch = in_ch
        module_type, args, kwargs = layer_param
        if 'ConvSameBnRelu2D' == module_type:
            args.insert(0, in_ch)
            layer = layers.ConvSameBnRelu2D(*args, **kwargs)
            out_ch = args[1]
        elif 'Conv2D' == module_type:
            args.insert(0, in_ch)
            layer = layers.Conv2D(*args, **kwargs)
            out_ch = args[1]
        elif 'Dense' == module_type:
            args.insert(0, in_ch)
            layer = layers.Dense(*args, **kwargs)
            out_ch = args[1]
        elif 'Flatten' == module_type:
            layer = layers.Flatten()
            out_ch = out_ch.prod()
        elif 'MaxPool2D' == module_type:
            layer = layers.MaxPool2D(*args, **kwargs)
        elif 'RoI' == module_type:
            layer = layers.RoI(*args, **kwargs)
            out_ch = in_ch * layer.out_ch_last.prod()
        elif 'RoIDense' == module_type:
            args.insert(0, in_ch)
            layer = layers.RoIDense(*args, **kwargs)
            out_ch = layer.out_ch_last
        elif 'RoIFlatten' == module_type:
            args.insert(0, in_ch)
            layer = layers.RoIFlatten(*args, **kwargs)
            out_ch = layer.out_ch_last
        elif 'Shortcut' == module_type:
            if 'conv' == args[0]:
                kwargs['in_ch'] = channels
            layer = layers.Shortcut(*args, **kwargs)

        return layer, out_ch

    @staticmethod
    def create_block(layer_param, cfg, in_ch=None, layers_fn=None):
        module_type, args, kwargs = layer_param
        args.insert(0, in_ch)
        out_ch = args[0] if isinstance(args[0], int) else in_ch
        if 'AlexBlock' == module_type:
            layer = blocks.AlexBlock(*args, **kwargs)
        elif 'VGGPoolBlock' == module_type:
            layer = blocks.VGGPoolBlock(*args, **kwargs)
        elif 'ConcatBlock' == module_type:
            args.pop(0)
            layer = blocks.ConcatBlock(*args, **kwargs)
            # TODO out_ch = args[1]
        elif 'InceptionBlock_v1A' == module_type:
            layer = blocks.InceptionBlock_v1A(*args, **kwargs)
        elif 'InceptionBlock_v1B' == module_type:
            layer = blocks.InceptionBlock_v1B(*args, **kwargs)
        elif 'InceptionBlock_v3A' == module_type:
            layer = blocks.InceptionBlock_v3A(*args, **kwargs)
        elif 'InceptionBlock_v3B' == module_type:
            layer = blocks.InceptionBlock_v3B(*args, **kwargs)
        elif 'ReductionBlock_v4B' == module_type:
            layer = blocks.ReductionBlock_v4B(*args, **kwargs)
        elif 'IncResBlock_v4A' == module_type:
            layer = blocks.IncResBlock_v4A(*args, **kwargs)
        elif 'IncResBlock_v4B' == module_type:
            layer = blocks.IncResBlock_v4B(*args, **kwargs)
        elif 'ResConvBlock' == module_type:
            res_dict = {
                'ResBlockA': blocks.ResBlockA,
                'ResBlockB': blocks.ResBlockB,
            }

            if args[4] in res_dict.keys():
                args[4] = res_dict[args[4]]
            layer = blocks.ResConvBlock(*args, **kwargs)
            out_ch = args[1]
        elif 'ResBlockA' == module_type:
            layer = blocks.ResBlockA(*args, **kwargs)
        elif 'ResBlockB' == module_type:
            layer = blocks.ResBlockB(*args, **kwargs)
        elif 'FeatureExtractor' == module_type:
            if len(args) > 7 and not isinstance(args[7], bool):
                args.insert(7, False)
            if len(args) > 8 and not isinstance(args[8], str):
                args.insert(8, 'valid')
            if len(args) > 9 and not isinstance(args[9], bool):
                args.insert(9, True)
            layer = blocks.FeatureExtractor(*args, **kwargs)
        elif layers_fn is not None:
            layer, out_ch = layers_fn(layer_param, cfg, in_ch)
        else:
            raise KeyError(f'{module_type} is not block\'s key.')
        return layer, out_ch
