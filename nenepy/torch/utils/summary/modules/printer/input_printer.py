from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter


class InputPrinter(AbstractPrinter):
    max_each_dim_size = []

    max_key_length = 0

    def __init__(self, module_in):
        """

        Args:
            module_in (Input):

        """
        self.module_in = module_in

    def to_print_formats(self):
        return self._iterable_to_text_formats(self.module_in.values, self.max_key_length)

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def _iterable_to_text_formats(cls, value, key_length=None):
        if len(value) == 0:
            return []
        if isinstance(value, dict):
            return cls.dict_to_text(value, key_length)
        else:
            return cls._list_to_text(value)

    @classmethod
    def _list_to_text(cls, value_list):
        value_type = f"<{cls._to_type(value_list)}>"
        key_length = len(value_type)
        texts = []
        for value in value_list:
            if cls._is_iterable(value):
                texts += cls._iterable_to_text_formats(value)
            else:
                texts += [value.to_adjusted_text(cls.max_each_dim_size)]

        brackets = cls._get_list_brackets(len(texts))
        max_length = cls._get_max_text_length(texts)

        for i, (text, bracket) in enumerate(zip(texts, brackets)):
            bracket_top, bracket_bottom = bracket
            if i == 0:
                type_format = value_type
            else:
                type_format = f"{'':>{key_length}}"

            texts[i] = f"{type_format}{bracket_top} {text:<{max_length}} {bracket_bottom}"

        return texts

    @classmethod
    def dict_to_text(cls, value_dict, key_length=None):

        if key_length is None:
            key_length = cls._get_max_dict_key_length(value_dict)

            return [f"{cls.generate_empty(key_length)}{value.to_adjusted_text(cls.max_each_dim_size)}" for value in value_dict.values()]

        adjusted_keys = cls._to_adjust_length(list(value_dict.keys()), key_length)
        empty_key = cls.generate_empty(key_length)

        texts = []
        for key, value in zip(adjusted_keys, value_dict.values()):
            if len(value_dict) == 1:
                key = f"{cls.to_empty(key)}"
            if cls._is_iterable(value):
                formatted_values = cls._iterable_to_text_formats(value)
                child_texts = [""] * len(formatted_values)
                for i, formatted_value in enumerate(formatted_values):
                    if i == 0:
                        child_texts[i] = f"{key}:{formatted_value}"
                    else:
                        child_texts[i] = f"{empty_key} {formatted_value}"
                texts += child_texts
            else:
                value_format = value.to_adjusted_text(cls.max_each_dim_size)
                text = f"{key} {value_format}"
                texts.append(text)
        return texts

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def _get_max_dict_key_length(value_dict):
        return max([len(key) for key in value_dict.keys()], default=0)

    @staticmethod
    def _get_max_text_length(texts):
        return max([len(text) for text in texts], default=0)

    @staticmethod
    def _to_adjust_length(texts, adjustment_length=None):
        if adjustment_length is None:
            adjustment_length = max([len(text) for text in texts], default=0)
        return [f"{text:>{adjustment_length}}" for text in texts]

    @classmethod
    def _get_list_brackets(cls, size):
        def get_list_bracket(index, size):
            if size <= 1:
                return ["[", "]"]
            if index == 0:
                return ["┌", "┐"]
            elif index == size - 1:
                return ["└", "┘"]
            else:
                return ["│", "│"]

        return [get_list_bracket(i, size) for i in range(size)]

    @staticmethod
    def _to_type(value):
        return str(value.__class__.__name__)
