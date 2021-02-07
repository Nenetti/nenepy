from torch import nn

from .input import Input
from .output import Output
from .parameter import Parameter


class Module:

    def __init__(self, module, module_in, module_out):
        """

        Args:
            module (nn.Module):
            module_in:
            module_out:
        """
        self.module = module
        self.input = Input(module, module_in)
        self.output = Output(module, module_out)
        self.parameter = Parameter(module)

    @property
    def module_name(self):
        return self.module.__class__.__name__

    @property
    def is_trained(self):
        return self.module.training

    @property
    def is_requires_grad(self):
        if self.parameter.weight_requires_grad or self.parameter.bias_requires_grad:
            return True
        return False
