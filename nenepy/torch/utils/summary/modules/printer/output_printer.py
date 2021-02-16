import numpy as np

from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.value import Value


class OutputPrinter(AbstractPrinter):
    max_n_dims = 0
    max_each_dim_size = []

    max_key_length = 0

    def __init__(self, module_out):
        """

        Args:
            module_out (Input):

        """
        self.module_out = module_out

    def to_print_formats(self):
        if isinstance(self.module_out.values, dict):
            return self.to_value_dict_format(self.module_out.values, self.max_key_length)
        elif isinstance(self.module_out.values, list):
            return self.to_value_list_format(self.module_out.values)
        elif isinstance(self.module_out.values, Value):
            return [self.module_out.values.to_adjusted_text(self.max_each_dim_size)]
        else:
            raise TypeError()

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def to_adjust(cls, printers):
        cls.max_n_dims = max([cls.calc_max_n_dims(printer.module_out) for printer in printers])
        cls.max_each_dim_size = np.max(np.stack([cls.calc_max_each_dim_size(printer.module_out) for printer in printers], axis=0), axis=0)
        cls.max_key_length = max([cls.calc_max_key_length(printer.module_out) for printer in printers])

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

        size = len(texts)

        for i, text in enumerate(texts):
            list_char = cls.to_list_char_front(i, size)
            if i == 0:
                type_format = f"{type2:>{key_length}}"
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
    def to_text_format(cls, module_out):
        return str(module_out.values)

    @classmethod
    def calc_max_key_length(cls, module_out):
        """

        Args:
            module_out (Input):

        Returns:

        """
        if isinstance(module_out.values, dict):
            return max([len(key) for key in module_out.values.keys()], default=0)
        return 0

    @classmethod
    def calc_max_n_dims(cls, module_out):
        """

        Args:
            module_out (Input):

        Returns:

        """
        return cls._calc_max_n_dims_recursive(module_out.values)

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
    def calc_max_each_dim_size(cls, module_out):
        """

        Args:
            module_out (Input):

        Returns:

        """
        each_dim_size = cls._calc_max_each_dim_size_recursive(module_out.values, cls.max_n_dims)
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
            if len(value) > 0:
                if isinstance(value, dict):
                    return np.max(np.stack([cls._calc_max_each_dim_size_recursive(v, max_n_dims) for v in value.values()], axis=0), axis=0)
                else:
                    return np.max(np.stack([cls._calc_max_each_dim_size_recursive(v, max_n_dims) for v in value], axis=0), axis=0)
            else:
                return np.zeros(shape=max_n_dims, dtype=np.int)
        else:
            raise TypeError()

    # ==================================================================================================
    #
    #   Class Method
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
