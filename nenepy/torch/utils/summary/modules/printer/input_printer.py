from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict
import numpy as np


class InputPrinter(AbstractPrinter):
    max_n_dims = 0
    max_each_dim_size = []

    max_key_length = 0

    def __init__(self, module_in):
        """

        Args:
            module_in (Input):

        """
        self.module_in = module_in
        self.text_format = self.to_text_format(module_in)
        self.set_n_max_length(self.text_format)

    def to_print_formats(self):
        return self.to_value_dict_format(self.module_in.values, self.max_key_length, True)

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def to_adjust(cls, printers):
        cls.max_n_dims = max([cls.calc_max_n_dims(printer.module_in) for printer in printers])
        cls.max_each_dim_size = np.max(np.stack([cls.calc_max_each_dim_size(printer.module_in) for printer in printers], axis=0), axis=0)
        cls.max_key_length = max([cls.calc_max_key_length(printer.module_in) for printer in printers])

    @classmethod
    def to_value_dict_format(cls, value_dict, max_key_length=None, is_root=False):
        """

        Args:
            value_dict (ValueDict):
            max_key_length (int):
            is_root (bool):

        Returns:

        """
        if max_key_length is None:
            max_key_length = cls.calc_max_text_length(value_dict.keys)
        formatted_keys = [cls.to_key_format(key, max_key_length) for key in value_dict.keys]
        if len(formatted_keys) == 0:
            return []

        formatted_key_length = len(formatted_keys[0])

        texts = []
        for i, (key, value) in enumerate(value_dict.items()):
            formatted_key = formatted_keys[i]
            if len(value_dict.items()) == 1:
                formatted_key = " " * len(formatted_key)

            if isinstance(value, Value):
                value_format = value.to_adjusted_text(cls.max_each_dim_size)
                text = f"{formatted_key}{value_format}"
                texts.append(text)
            elif isinstance(value, (ValueList, ValueDict)):
                if isinstance(value, ValueList):
                    formatted_values = cls.to_value_list_format(value)
                else:
                    formatted_values = cls.to_value_dict_format(value)
                child_texts = [""] * len(formatted_values)
                for k, v in enumerate(formatted_values):
                    if k == 0:
                        child_texts[k] = f"{formatted_key}{v}"
                    else:
                        child_texts[k] = f"{'':>{formatted_key_length}}{v}"
                texts += child_texts
            else:
                raise TypeError()

        return texts

    @classmethod
    def to_value_list_format(cls, value_list):
        """

        Args:
            value_list (ValueList):
            n_indent:

        Returns:

        """
        type = f"<{value_list.type}>"
        key_length = len(type)
        texts = []
        for i, value in enumerate(value_list.values):
            if isinstance(value, Value):
                value_format = value.to_adjusted_text(cls.max_each_dim_size)
                texts.append(value_format)
            elif isinstance(value, ValueList):
                texts += cls.to_value_list_format(value)
            elif isinstance(value, ValueDict):
                texts += cls.to_value_dict_format(value)
            else:
                raise TypeError()

        size = len(texts)

        for i, text in enumerate(texts):
            list_char = cls.to_list_char_front(i, size)
            if i == 0:
                type_format = f"{type:>{key_length}}"
            else:
                type_format = f"{'':>{key_length}}"

            texts[i] = f"{type_format}{list_char}{text}"

        max_length = cls.calc_max_text_length(texts)

        for i, text in enumerate(texts):
            list_char = cls.to_list_char_back(i, size)
            text = f"{text:<{max_length}}"
            texts[i] = f"{text}{list_char}"

        return texts

    @classmethod
    def to_text_format(cls, module_in):
        return str(module_in.values)

    @classmethod
    def calc_max_key_length(cls, module_in):
        """

        Args:
            module_in (Input):

        Returns:

        """
        return max([len(key) for key in module_in.values.value_dict.keys()])

    @classmethod
    def calc_max_n_dims(cls, module_in):
        """

        Args:
            module_in (Input):

        Returns:

        """
        return cls._calc_max_n_dims_recursive(module_in.values)

    @classmethod
    def _calc_max_n_dims_recursive(cls, value):
        if isinstance(value, Value):
            if value.is_tensor:
                return len(value.shapes)
        elif isinstance(value, (ValueList, ValueDict)):
            if len(value.values) > 0:
                return max([cls._calc_max_n_dims_recursive(v) for v in value.values])
        else:
            raise TypeError()

        return 0

    @classmethod
    def calc_max_each_dim_size(cls, module_in):
        """

        Args:
            module_in (Input):

        Returns:

        """

        each_dim_size = cls._calc_max_each_dim_size_recursive(module_in.values, cls.max_n_dims)
        return each_dim_size

    @classmethod
    def _calc_max_each_dim_size_recursive(cls, value, max_n_dims):
        if isinstance(value, Value):
            if value.is_tensor:
                each_size = np.zeros(shape=max_n_dims, dtype=np.int)
                for i in range(len(value.shapes)):
                    each_size[i] = len(value.shapes[i])
                return each_size
            else:
                return np.zeros(shape=max_n_dims, dtype=np.int)

        elif isinstance(value, (ValueList, ValueDict)):
            if len(value.values) > 0:
                return np.max(np.stack([cls._calc_max_each_dim_size_recursive(v, max_n_dims) for v in value.values], axis=0), axis=0)
        else:
            raise TypeError()

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
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

    @staticmethod
    def to_list_char_front(index, size):
        if size <= 1:
            return "[ "
        if index == 0:
            return "┌ "
        elif index == size - 1:
            return "└ "
        else:
            return "│ "

    @staticmethod
    def to_list_char_back(index, size):
        if size <= 1:
            return " ]"
        if index == 0:
            return " ┐"
        elif index == size - 1:
            return " ┘"
        else:
            return " │"
