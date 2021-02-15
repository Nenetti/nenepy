from torch import nn

from .input import Input
from .output import Output
from .parameter import Parameter


class Module:

    def __init__(self, ):
        """

        Args:
            module (nn.Module):
        """
        self.module = None
        self.input = None
        self.output = None
        self.parameter = None

        self.parent_module = None
        self.child_modules = []
        self.processing_time = 0
        self.is_root = False
        self.is_last_module_in_sequential = False

    def initialize(self, module, module_in, module_out):
        self.module = module
        self.input = Input(module, module_in)
        self.output = Output(module, module_out)
        self.parameter = Parameter(module)

    def has_children(self):
        return len(self.child_modules) > 0

    @property
    def module_name(self):
        return self.module.__class__.__name__

    @staticmethod
    def construction(roots):
        """
        Args:
            roots (list[Block]):

        """

        def recursive(block):
            if len(block.child_modules) > 0:
                for b in block.child_modules:
                    recursive(b)
                block.child_modules[-1].is_last_module_in_sequential = True

        for root in roots:
            recursive(root)
