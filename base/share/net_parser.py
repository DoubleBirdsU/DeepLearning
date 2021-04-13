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
        for params in self.cfg['backbone']:
            idx_from, sequential_params = params[0], params[1:]
            block = self.create_sequential(sequential_params, idx_from)
            self.net.cfg_from_list.append(idx_from)
            self.net.cfg_module_list.append(block)

        if include_top:
            for params in self.cfg['head']:
                idx_from, sequential_params = params[0], params[1:]
                block = self.create_sequential(sequential_params, idx_from)
                self.net.cfg_from_list.append(idx_from)
                self.net.cfg_module_list.append(block)
        self.net.addLayers(self.net.cfg_module_list)

    def create_sequential(self, sequential_params, idx_from=None):
        num_layer, module_params = sequential_params[0], sequential_params[1:]
        if num_layer == 1:
            sequential = self.get_module(module_params, idx_from)
        else:
            sequential = nn.Sequential()
            for i in range(num_layer):
                layer = self.get_module(module_params, idx_from)
                sequential.add_module(f'{module_params[0]}_{i}', layer)
        return sequential

    def get_module(self, module_params, idx_from=None):
        """
        get_module 获取模块

        生成模块

        Args:
            module_params (List[str, List[Union[str, int]]]): 模块信息参数
            idx_from (Union[int, List[int]]), optional): 模块输入. Defaults to None.

        Returns:
            Module: 模块
        """
        in_ch, out_ch = self.net.channels[-1], self.net.channels[-1]
        module_params = module_params[:1] + self.check_params(module_params[1:])

        if module_params[0] in layers.layer_name_set:
            layer, out_ch = self.create_layer(module_params, in_ch, self.net.channels, idx_from)
        elif module_params[0] in blocks.block_name_set:
            layer, out_ch = self.create_block(module_params, in_ch, self.net.channels, idx_from)
        elif self.net.layers_fn is not None:
            layer, out_ch = self.net.layers_fn(module_params, in_ch, self.net.channels, idx_from)
        else:
            raise KeyError(f'{module_params[0]} is not block\'s key.')
        self.net.channels.append(out_ch)
        return layer

    def check_params(self, params):
        arg_list, kwarg_dict = list(), dict()
        for param in params:
            if isinstance(param, list):
                arg_list = param
            elif isinstance(param, dict):
                kwarg_dict = param
        return [self.check_args(arg_list), self.check_kwargs(kwarg_dict)]

    def check_args(self, arg_list):
        for i, arg in enumerate(arg_list):
            if isinstance(arg, str) and arg in self.cfg:
                arg_list[i] = self.cfg[arg]
            elif isinstance(arg, list):
                arg_list[i] = self.check_args(arg)
        return arg_list

    def check_kwargs(self, kwarg_dict):
        for key, value in kwarg_dict.items():
            if isinstance(value, str) and value in self.cfg:
                kwarg_dict[key] = self.cfg[value]
        return kwarg_dict

    def create_layer(self, layer_param, in_ch, channels, idx_from=None):
        module_type, args, kwargs = layer_param
        args.insert(0, in_ch)
        out_ch = kwargs['out_ch'] if 'out_ch' in kwargs else args[1]
        if 'ConvSameBnRelu2D' == module_type:
            layer = layers.ConvSameBnRelu2D(*args, **kwargs)
        elif 'Conv2D' == module_type:
            layer = layers.Conv2D(*args, **kwargs)
        elif 'Dense' == module_type:
            layer = layers.Dense(*args, **kwargs)
        elif 'Flatten' == module_type:
            layer = layers.Flatten()
            out_ch = in_ch.prod()
        elif 'RoIDense' == module_type:
            layer = layers.RoIDense(*args, **kwargs)
            out_ch = layer.out_ch_last
        elif 'RoIFlatten' == module_type:
            layer = layers.RoIFlatten(*args, **kwargs)
            out_ch = layer.out_ch_last
        else:
            layer, out_ch = self.create_no_in_layer(layer_param, in_ch, channels, idx_from)
        return layer, out_ch

    @staticmethod
    def create_no_in_layer(layer_param, in_ch, channels, idx_from=None):
        layer = None
        module_type, args, kwargs = layer_param
        out_ch = kwargs['out_ch'] if 'out_ch' in kwargs else args[1]
        if 'Concat' == module_type:
            layer = layers.Concat(*args, **kwargs)
            out_ch = 0
            for idx in idx_from:
                out_ch += channels[idx]
        elif 'MaxPool2D' == module_type:
            layer = layers.MaxPool2D(*args, **kwargs)
        elif 'UpSample' == module_type:
            layer = layers.UpSample(*args, **kwargs)
        elif 'RoI' == module_type:
            layer = layers.RoI(*args, **kwargs)
            out_ch = in_ch * layer.out_ch_last.prod()
        elif 'Shortcut' == module_type:
            if 'conv' == args[0]:
                kwargs['in_ch'] = channels[idx_from]
            layer = layers.Shortcut(*args, **kwargs)
        return layer, out_ch

    @staticmethod
    def create_block(block_param, in_ch, channels=None, idx_from=None):
        module_type, args, kwargs = block_param
        out_ch = args[0] if isinstance(args[0], int) else in_ch
        args.insert(0, in_ch)

        layer = None
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
            layer = blocks.FeatureExtractor(*args, **kwargs)
        return layer, out_ch
