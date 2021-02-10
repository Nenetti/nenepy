from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict
from nenepy.torch.utils.summary.modules.parameter import Parameter
import numpy as np


class ParameterPrinter(AbstractPrinter):
    parameter_repr = "Parameter"
    weight_repr = "Weight"
    bias_repr = "Bias"
    train_repr = "Train"
    untrain_repr = "Untrain"

    max_weight_length = len(weight_repr)
    max_bias_length = len(bias_repr)
    max_train_length = len(train_repr)
    max_untrain_length = len(untrain_repr)

    def __init__(self, parameter):
        """

        Args:
            parameter (Parameter):

        """
        self.parameter = parameter

    @classmethod
    def to_parameter_repr(cls):
        return f"{cls.parameter_repr:^{cls.n_max_length}}"

    @classmethod
    def to_weight_repr(cls):
        return f"{cls.weight_repr:^{cls.max_weight_length}}"

    @classmethod
    def to_bias_repr(cls):
        return f"{cls.bias_repr:^{cls.max_bias_length}}"

    @classmethod
    def to_train_repr(cls):
        return f"{cls.train_repr:^{cls.max_train_length}}"

    @classmethod
    def to_untrain_repr(cls):
        return f"{cls.untrain_repr:^{cls.max_untrain_length}}"

    @classmethod
    def to_adjust(cls, printers):
        cls.max_weight_length = max([cls.max_weight_length, max([cls.calc_max_weight_length(printer.parameter) for printer in printers])])
        cls.max_bias_length = max([cls.max_bias_length, max([cls.calc_max_bias_length(printer.parameter) for printer in printers])])

    def to_print_format(self):
        weight_format = f"{self.parameter.weight_str():>{self.max_weight_length}}"
        bias_format = f"{self.parameter.bias_str():>{self.max_bias_length}}"
        train_format = f"{self.to_train_format(self.parameter.is_train):^{self.max_train_length}}"
        untrain_format = f"{self.to_train_format(not self.parameter.is_train):^{self.max_untrain_length}}"

        print_format = f"{weight_format} │ {bias_format} │ {train_format} │ {untrain_format} "
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

    @classmethod
    def calc_max_weight_length(cls, parameter):
        """

        Args:
            parameter (Parameter):

        Returns:

        """
        return len(parameter.weight_str())

    @classmethod
    def calc_max_bias_length(cls, parameter):
        """

        Args:
            parameter (Parameter):

        Returns:

        """
        return len(parameter.bias_str())
