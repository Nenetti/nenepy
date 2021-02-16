from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.value import Value
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

    def to_print_formats(self):
        return self.to_value_dict_format(self.module_in.values, self.max_key_length)

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

    # @classmethod
    # def value_to_text_formats(cls, value):
    #     # if cls._is_iterable(value):
    #     if isinstance(value, (ValueList, ValueDict)):
    #
    #     else:
    #

    @classmethod
    def to_value_dict_format(cls, value_dict, max_key_length=None):
        """

        Args:
            value_dict (ValueDict):
            max_key_length (int):
            is_root (bool):

        Returns:

        """
        if max_key_length is None:
            max_key_length = cls.calc_max_text_length(value_dict.keys())
        formatted_keys = [cls.to_key_format(key, max_key_length) for key in value_dict.keys()]
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
            elif cls._is_iterable(value):
                if isinstance(value, list):
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
        type2 = f"<{type(value_list)}>"
        key_length = len(type2)
        texts = []
        for i, value in enumerate(value_list):
            if isinstance(value, Value):
                value_format = value.to_adjusted_text(cls.max_each_dim_size)
                texts.append(value_format)
            elif isinstance(value, list):
                texts += cls.to_value_list_format(value)
            elif isinstance(value, dict):
                texts += cls.to_value_dict_format(value)
            else:
                raise TypeError()

        brackets = cls._get_list_brackets(len(texts))
        max_length = cls.calc_max_text_length(texts)

        for i, text in enumerate(texts):
            bracket_top, bracket_bottom = brackets[i]
            if i == 0:
                type_format = f"{type2:>{key_length}}"
            else:
                type_format = f"{'':>{key_length}}"

            texts[i] = f"{type_format}{bracket_top} {text:<{max_length}} {bracket_bottom}"

        return texts

    @classmethod
    def to_text_format(cls, module_in):
        return str(module_in.values)

    @classmethod
    def calc_max_key_length(cls, module_in):
        return max([len(key) for key in module_in.values.keys()])

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
        elif cls._is_iterable(value):
            if isinstance(value, dict):
                return max([cls._calc_max_n_dims_recursive(v) for v in value.values()], default=0)
            else:
                return max([cls._calc_max_n_dims_recursive(v) for v in value], default=0)
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

        elif cls._is_iterable(value):
            if isinstance(value, dict):
                return np.max(np.stack([cls._calc_max_each_dim_size_recursive(v, max_n_dims) for v in value.values()], axis=0), axis=0)
            else:
                return np.max(np.stack([cls._calc_max_each_dim_size_recursive(v, max_n_dims) for v in value], axis=0), axis=0)
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
    def _get_max_key_length(value_dict):
        return max([len(key) for key in value_dict.keys()], default=0)

    @staticmethod
    def _get_max_value_length(value_dict):
        return max([len(value) for value in value_dict.values()], default=0)

    @staticmethod
    def _get_max_text_length(texts):
        return max([len(text) for text in texts], default=0)

    @staticmethod
    def to_key_format(key, adjustment_length=None):
        if adjustment_length is None:
            adjustment_length = len(key)
        return f"{key:>{adjustment_length}}: "

    @classmethod
    def _get_list_brackets(cls, size):
        return [cls._get_list_bracket(i, size) for i in range(size)]

    @staticmethod
    def _get_list_bracket(index, size):
        if size <= 1:
            return ["[", "]"]
        if index == 0:
            return ["┌", "┐"]
        elif index == size - 1:
            return ["└", "┘"]
        else:
            return ["│", "│"]
