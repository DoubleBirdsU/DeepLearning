from torch import nn


class base_model(nn.Module):
    def get_index(self, class_name):
        if isinstance(class_name, str):
            return [i for i, module in enumerate(self.modules_list) if module.__class__.__name__ == class_name]
        else:
            return [i for i, module in enumerate(self.modules_list) if isinstance(module, class_name)]


class Module(base_model):
    def __init__(self, cls_name=None):
        super(Module, self).__init__()
        self.set_class_name(cls_name)
        self.in_ch_first = None
        self.out_ch_last = None
        self.modules_list = nn.ModuleList()

    def forward(self, *args, **kwargs):
        x = args[0]
        for layer in self.modules_list:
            x = layer(x)
        return x

    def addLayers(self, layers):
        self.collect_layers(layers, bool_in=True, bool_out=True)

    def getLayers(self):
        return self.modules_list

    def get_modules(self):
        return self.modules_list

    def add_module(self, name, module):
        super(Module, self).add_module(name, module)
        self.addLayers(module)

    def collect_layers(self, layers, bool_in=False, bool_out=False):
        r"""Collect Layers

        Args:
            layers: Union[List[Module], Module]
            bool_in:
            bool_out:

        Returns:
            None
        """
        self._init_in_ch_first(layers, bool_in)
        self._collect_layers(layers, bool_out)

    def set_class_name(self, cls_name):
        if cls_name is not None:
            self.__class__.__name__ = cls_name

    def _collect_layers(self, modules, bool_out=False):
        if modules is None:
            pass
        elif isinstance(modules, Module) and len(modules.modules_list) != 0:
            self._collect_layers(modules.modules_list, bool_out)
            self._init_out_ch_last(modules, bool_out)
        elif isinstance(modules, nn.ModuleList) or isinstance(modules, nn.Sequential) or isinstance(modules, list):
            for module in modules:
                self._collect_layers(module, bool_out)
        elif isinstance(modules, nn.Module):
            self.modules_list.append(modules)

    def _init_in_ch_first(self, modules, bool_in=False):
        if modules is None or (isinstance(self, Module) and (bool_in or self.in_ch_first)):
            pass
        elif isinstance(modules, Module):
            self.in_ch_first = modules.in_ch_first
        elif isinstance(modules, nn.Module) and hasattr(modules, 'weight') and modules.weight.ndim > 1:
            self.in_ch_first = modules.weight.shape[1]
        elif isinstance(modules, list) or isinstance(modules, nn.Sequential):
            for module in modules:
                self._init_in_ch_first(module, bool_in)
                if isinstance(self, Module) and self.in_ch_first:
                    break

    def _init_out_ch_last(self, modules, bool_out=False):
        if not bool_out:
            self.out_ch_last = modules.out_ch_last
