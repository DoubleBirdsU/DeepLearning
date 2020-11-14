from torch import nn


class base_module(nn.Module):  # 简单结构继承该类
    def _forward_unimplemented(self, *input) -> None:
        return super(base_module, self)._forward_unimplemented(*input)


class Module(base_module):  # 复杂结构继承该类
    def __init__(self):
        super(Module, self).__init__()
        self.module_list = nn.ModuleList()

    # forward 优先级低于 __call__, 基类中包含 __call__ 时, 不会调用子类的 forward
    # 在基类中只能使用 forward, 不能使用 __call__
    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_list(self, x):
        return [layer(x) for layer in self.module_list]

    def collect_layers(self, modules):
        """Collect Layers
        :type modules: Union[List[Module], Module]
        """
        if isinstance(modules, nn.Sequential):
            self.module_list.append(modules)
        elif isinstance(modules, nn.ModuleList):
            self.module_list += modules
        elif isinstance(modules, Module):
            self.module_list += modules.module_list
        elif isinstance(modules, nn.Module):
            self.add_module(modules.__class__.__name__, modules)
        elif isinstance(modules, list):
            for module in modules:
                self.collect_layers(module)

    def add_module(self, name: str, module) -> None:
        super(Module, self).add_module(name, module)
        if not isinstance(module, Module) and isinstance(module, nn.Module):
            self.module_list.append(module)

    def get_index(self, class_name):
        if isinstance(class_name, str):
            return [i for i, module in enumerate(self.module_list) if module.__class__.__name__ == class_name]
        else:
            return [i for i, module in enumerate(self.module_list) if isinstance(module, class_name)]

    def get_indexes(self, class_name_tuple: tuple):
        if isinstance(class_name_tuple[0], str):
            for i, module in enumerate(self.module_list):
                print(i, module.__class__.__name__)
            return [i for i, module in enumerate(self.module_list) if module.__class__.__name__ in class_name_tuple]
        else:
            return [i for i, module in enumerate(self.module_list) if isinstance(module, tuple(class_name_tuple))]

    @staticmethod
    def swap(a, b):
        return b, a
