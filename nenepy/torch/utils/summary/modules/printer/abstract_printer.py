from abc import ABCMeta, abstractmethod
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict


class AbstractPrinter(metaclass=ABCMeta):
    indent_space = 4
    n_max_length = 0

    @abstractmethod
    def to_print_format(self):
        raise NotImplementedError()


    @classmethod
    def set_n_max_length(cls, print_format):
        if cls.n_max_length < len(print_format):
            cls.n_max_length = len(print_format)