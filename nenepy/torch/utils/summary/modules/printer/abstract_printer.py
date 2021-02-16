from abc import ABCMeta


class AbstractPrinter(metaclass=ABCMeta):
    indent_space = 5
    n_max_length = 0

    @classmethod
    def set_n_max_length(cls, print_format):
        if cls.n_max_length < len(print_format):
            cls.n_max_length = len(print_format)

    @staticmethod
    def to_replace(text, char=" "):
        return char * len(text)

    @staticmethod
    def to_empty(text):
        return " " * len(text)

    @staticmethod
    def generate_empty(length):
        return " " * length

    @staticmethod
    def _is_iterable(value):
        if isinstance(value, (tuple, list, set, dict)):
            return True
        return False
