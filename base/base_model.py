from torch import nn


class base_model(nn.Module):
    def _forward_unimplemented(self, *input) -> None:
        return super(base_model, self)._forward_unimplemented(*input)

    def get_index(self, class_name):
        if isinstance(class_name, str):
            return [i for i, module in enumerate(self.module_list) if module.__class__.__name__ == class_name]
        else:
            return [i for i, module in enumerate(self.module_list) if isinstance(module, class_name)]
