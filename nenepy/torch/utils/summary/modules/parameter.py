import numpy as np
import torch
from torch import nn


class Parameter:
    def __init__(self, module):
        """

        Args:
            module (nn.Module):
        """
        self.module = module
        self.is_train = module.training
        self.has_weight = self._has_weight(module)
        self.has_bias = self._has_bias(module)
        self.n_weight_params, self.weight_requires_grad = self._analyze_weight(module)
        self.n_bias_params, self.bias_requires_grad = self._analyze_bias(module)

    def weight_str(self):
        if self.has_weight:
            return f"{self.n_weight_params:,}"
        else:
            return "-"

    def bias_str(self):
        if self.has_bias:
            return f"{self.n_bias_params:,}"
        else:
            return "-"

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def _analyze_weight(cls, module):
        if cls._has_weight(module):
            n_params = cls._calc_n_params(module.weight)
            requires_grad = module.weight.requires_grad
            return n_params, requires_grad

        return 0, False

    @classmethod
    def _analyze_bias(cls, module):
        if cls._has_bias(module):
            n_params = cls._calc_n_params(module.bias)
            requires_grad = module.bias.requires_grad
            return n_params, requires_grad

        return 0, False

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def _has_weight(module):
        if hasattr(module, "weight") and (module.weight is not None) and isinstance(module.weight, torch.Tensor):
            return True

        return False

    @staticmethod
    def _has_bias(module):
        if hasattr(module, "bias") and (module.weight is not None) and isinstance(module.bias, torch.Tensor):
            return True

        return False

    @staticmethod
    def _calc_n_params(tensor):
        """

        Args:
            tensor (torch.Tensor):

        Returns:

        """
        return np.prod(tensor.size())
