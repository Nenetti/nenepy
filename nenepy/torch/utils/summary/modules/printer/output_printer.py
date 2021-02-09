from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict
import numpy as np


class OutputPrinter(AbstractPrinter):
    max_n_dims = 0
    max_each_dim_size = []

    max_value_length = 0

    def __init__(self, module_in):
        """

        Args:
            module_in (Input):

        """
        self.module_in = module_in
        self.text_format = self.to_text_format(module_in)
        self.set_n_max_length(self.text_format)
        self.set_max_n_dims(module_in)
        self.set_max_each_dim_size(module_in)
        self.n_lines = self.calc_n_lines(module_in)

    def to_print_formats(self):
        if isinstance(self.module_in.values, ValueDict):
            return self.to_value_dict_format(self.module_in.values)
        elif isinstance(self.module_in.values, ValueList):
            return self.to_value_list_format(self.module_in.values)
        elif isinstance(self.module_in.values, Value):
            return [self.module_in.values.text]
        else:
            raise TypeError()

    def to_print_format(self):
        if isinstance(self.module_in.values, ValueDict):
            texts = self.to_value_dict_format(self.module_in.values)
            return "\n".join(texts)
        elif isinstance(self.module_in.values, ValueList):
            texts = self.to_value_list_format(self.module_in.values)
            return "\n".join(texts)
        elif isinstance(self.module_in.values, Value):
            return self.module_in.values.text
        else:
            raise TypeError()

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
    def to_value_dict_format(cls, value_dict, max_key_length=None):
        """

        Args:
            value_dict (ValueDict):
            n_indent:

        Returns:

        """
        if len(value_dict.keys) > 0:
            if max_key_length is None:
                max_key_length = cls.calc_max_text_length(value_dict.keys)
            formatted_keys = [cls.to_key_format(key, max_key_length) for key in value_dict.keys]
            formatted_key_length = len(formatted_keys[0])

            texts = []
            for i, (key, value) in enumerate(value_dict.items()):
                formatted_key = formatted_keys[i]
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
        else:
            return []

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

    @classmethod
    def to_text_format(cls, module_in):
        return str(module_in.values)

    @classmethod
    def set_max_n_dims(cls, module_in):
        """

        Args:
            module_in (Input):

        Returns:

        """

        def recursive(value):
            if isinstance(value, Value):
                if value.is_tensor:
                    return len(value.shapes)
            elif isinstance(value, (ValueList, ValueDict)):
                if len(value.values) > 0:
                    return max([recursive(v) for v in value.values])
            else:
                raise TypeError()
            return 0

        size = recursive(module_in.values)
        if cls.max_n_dims < size:
            cls.max_n_dims = size

    @classmethod
    def set_max_each_dim_size(cls, module_in):
        """

        Args:
            module_in (Input):

        Returns:

        """

        def recursive(value):
            if isinstance(value, Value):
                if value.is_tensor:
                    each_size = np.zeros(shape=cls.max_n_dims, dtype=np.int)
                    for i in range(len(value.shapes)):
                        each_size[i] = len(value.shapes[i])
                    return each_size
                else:
                    return np.zeros(shape=cls.max_n_dims, dtype=np.int)

            elif isinstance(value, (ValueList, ValueDict)):
                if len(value.values) > 0:
                    return np.max(np.stack([recursive(v) for v in value.values], axis=0), axis=0)
                else:
                    return np.zeros(shape=cls.max_n_dims, dtype=np.int)
            else:
                raise TypeError()

        each_dim_size = recursive(module_in.values)

        if len(cls.max_each_dim_size) != len(each_dim_size):
            t = [0] * cls.max_n_dims
            t[:len(cls.max_each_dim_size)] = cls.max_each_dim_size
            cls.max_each_dim_size = t

        for i in range(len(each_dim_size)):
            if each_dim_size[i] > cls.max_each_dim_size[i]:
                cls.max_each_dim_size[i] = each_dim_size[i]

    @staticmethod
    def calc_n_lines(module_in):
        def recursive(value):
            if isinstance(value, Value):
                return 1
            elif isinstance(value, (ValueList, ValueDict)):
                if len(value.values) > 0:
                    return sum([recursive(v) for v in value.values])
                else:
                    return 1
            else:
                raise TypeError()

        return recursive(module_in.values)
