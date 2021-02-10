from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict
from nenepy.torch.utils.summary.modules.parameter import Parameter
import numpy as np


class ParameterPrinter(AbstractPrinter):
    max_weight_length = 0
    max_bias_length = 0

    def __init__(self, parameter):
        """

        Args:
            parameter (Parameter):

        """
        self.parameter = parameter
        self.set_max_weight_length(parameter)
        self.set_max_bias_length(parameter)

    def to_print_format(self):
        weight_format = f"{self.parameter.weight_str():>{self.max_weight_length}}"
        bias_format = f"{self.parameter.bias_str():>{self.max_bias_length}}"
        train_format = self.to_train_format(self.parameter.is_train)
        untrain_format = self.to_train_format(not self.parameter.is_train)

        print_format = f"{weight_format} │ {bias_format} │ {train_format} │ {untrain_format}"
        return print_format

    @staticmethod
    def to_train_format(is_train):
        """

        Args:
            is_train (bool):

        Returns:

        """
        if is_train:
            return "✓"
        else:
            return " "

    @staticmethod
    def calc_max_text_length(texts):
        if len(texts) > 0:
            return max([len(text) for text in texts])
        else:
            return 0

    @staticmethod
    def to_key_format(key, adjustment_length=None):
        if adjustment_length is None:
            adjustment_length = len(key)
        return f"{key:>{adjustment_length}}: "

    @classmethod
    def set_max_weight_length(cls, parameter):
        """

        Args:
            parameter (Parameter):

        Returns:

        """
        n_weight = len(parameter.weight_str())
        if n_weight > cls.max_weight_length:
            cls.max_weight_length = n_weight

    @classmethod
    def set_max_bias_length(cls, parameter):
        """

        Args:
            parameter (Parameter):

        Returns:

        """
        n_bias = len(parameter.bias_str())
        if n_bias > cls.max_bias_length:
            cls.max_bias_length = n_bias
