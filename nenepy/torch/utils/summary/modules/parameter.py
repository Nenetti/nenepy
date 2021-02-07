import numpy as np
import torch
from torch import nn


class Parameter:
    def __init__(self, module):
        """

        Args:
            module (nn.Module):
        """
        self.is_train = module.training
        self.n_weight_params, self.weight_requires_grad = self.analyze_weight(module)
        self.n_bias_params, self.bias_requires_grad = self.analyze_bias(module)

    @classmethod
    def analyze_weight(cls, module):
        weight_params = 0
        requires_grad = False
        if cls.has_weight(module):
            weight_params = cls.calc_n_params(module.weight)
            requires_grad = module.weight.requires_grad

        return weight_params, requires_grad

    @classmethod
    def analyze_bias(cls, module):
        bias_params = 0
        requires_grad = False
        if cls.has_bias(module):
            bias_params = cls.calc_n_params(module.bias)
            requires_grad = module.bias.requires_grad

        return bias_params, requires_grad

    @staticmethod
    def has_weight(module):
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            return True

        return False

    @staticmethod
    def has_bias(module):
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            return True

        return False

    @staticmethod
    def calc_n_params(tensor):
        """

        Args:
            tensor (torch.Tensor):

        Returns:

        """
        return np.prod(tensor.size())
