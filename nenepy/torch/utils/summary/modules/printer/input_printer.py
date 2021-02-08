from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict
import numpy as np


class InputPrinter(AbstractPrinter):
    n_max_tensor_elements = 0
    n_each_max_tensor_elements = []

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
        self.n_lines = self.set_n_lines(module_in)

    @staticmethod
    def set_n_lines(module_in):
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

    def to_print_format(self):
        texts = self.to_dict_value_format(self.module_in.values, 0)
        return "\n".join(texts)

    @classmethod
    def to_value_format(cls, value_str, nest):
        return f"{value_str:>{cls.indent_space * nest}}"

    @classmethod
    def to_dict_value_format(cls, value_dict, nest):
        """

        Args:
            value_dict (ValueDict):
            nest:

        Returns:

        """
        texts = []
        for key, value in value_dict.value_dict.items():
            if isinstance(value, Value):
                text = f"{'':>{cls.indent_space * nest}}{key}:{value.text}"
                texts.append(text)
            elif isinstance(value, ValueList):
                texts += cls.to_value_list_format(value, nest + 1)
            elif isinstance(value, ValueDict):
                texts += cls.to_dict_value_format(value, nest + 1)
            else:
                raise TypeError()
        return texts

    @classmethod
    def to_value_list_format(cls, value_list, nest):
        """

        Args:
            value_list (ValueList):
            nest:

        Returns:

        """
        texts = []
        for value in value_list.values:
            if isinstance(value, Value):
                text = f"{'':>{cls.indent_space * nest}}{value.text}"
                texts.append(text)
            elif isinstance(value, ValueList):
                texts += cls.to_value_list_format(value, nest + 1)
            elif isinstance(value, ValueDict):
                texts += cls.to_dict_value_format(value, nest + 1)
            else:
                raise TypeError()
        return texts

    #
    # def to_print_format(self):
    #     def recursive(value):
    #         if isinstance(value, Value):
    #             return value.text
    #         elif isinstance(value, (Values, DictValue)):
    #             if len(value.values) > 1:
    #                 return [recursive(v) for v in value.values]
    #             elif len(value.values) != 0:
    #                 return recursive(value.values[0])
    #             else:
    #                 return 1
    #         else:
    #             raise TypeError()
    #
    #     texts = recursive(self.module_in.values)
    #     out_texts = ""
    #
    #     def recursive2(value):
    #         if isinstance(value, list):
    #             [recursive2(v) for v in value]
    #         else:
    #             out_texts += f"{text}\n"
    #
    #     for text in texts:
    #         return texts

    @classmethod
    def to_text_format(cls, module_in):
        return str(module_in.values)
        parent_directory_format = cls.to_parent_formant(module_in)
        directory_format = cls.to_directory_format(module_in)
        print_format = f"{parent_directory_format}{directory_format}{module_in.module.module_name}"
        return print_format

    @staticmethod
    def tensor_to_print_format():
        pass

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

        cls.n_max_tensor_elements = recursive(module_in.values)

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
                    each_size = np.zeros(shape=cls.n_max_tensor_elements, dtype=np.int)
                    for i in range(len(value.shapes)):
                        each_size[i] = len(value.shapes[i])
                    return each_size
                else:
                    return np.zeros(shape=cls.n_max_tensor_elements)

            elif isinstance(value, (ValueList, ValueDict)):
                if len(value.values) > 0:
                    return np.max(np.stack([recursive(v) for v in value.values], axis=0), axis=0)
            else:
                raise TypeError()

        cls.n_each_max_tensor_elements = np.max(recursive(module_in.values), axis=0)
    #
    # @classmethod
    # def set_n_lines(cls, module_in):
    #     """
    #
    #     Args:
    #         module_in (Input):
    #
    #     Returns:
    #
    #     """
    #
    #     def recursive(value):
    #         if isinstance(value, Value):
    #             if value.is_tensor:
    #                 return len(value.shapes)
    #         elif isinstance(value, (Values, DictValue)):
    #             if len(value.values) > 0:
    #                 value.value_strs
    #                 return max([recursive(v) for v in value.values])
    #         else:
    #             raise TypeError()
    #         return 0
    #
    #     cls.n_max_tensor_elements = recursive(module_in.values)
