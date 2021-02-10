from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict
from nenepy.torch.utils.summary.modules.parameter import Parameter
import numpy as np


class TimePrinter(AbstractPrinter):
    max_length = 0

    def __init__(self, time):
        """

        Args:
            parameter (Parameter):

        """
        self.time = time
        self.set_max_length(time)

    def to_print_format(self):
        time_str = self.to_time_str(self.time)
        time_format = f"{time_str:>{self.max_length}}"
        return time_format

    @staticmethod
    def to_time_str(time):
        # return str(int(time * 1000))
        t = time * 1000
        if int(t) > 0:
            return f"{t:.2f}"
        else:
            return f"{t:.2f}"[1:]

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

    @classmethod
    def set_max_length(cls, time):
        """

        Args:
            time (int)

        Returns:

        """
        length = len(cls.to_time_str(time))
        if length > cls.max_length:
            cls.max_length = length
