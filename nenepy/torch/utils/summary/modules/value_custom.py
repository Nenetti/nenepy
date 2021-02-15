import torch


class Value:

    def __init__(self, value):
        self.value = value
        self.is_tensor = self._is_tensor(value)
        self.shapes = self._to_shape_list(value)
        self.text = self._to_text(value)
        self.type = self._to_type(value)

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================
    def to_adjusted_text(self, each_dim_size):
        """

        Args:
            each_dim_size (list[int]):

        Returns:

        """
        if self.is_tensor:
            texts = [f"{size:>{each_dim_size[i]}}" for i, size in enumerate(self.shapes)]
            text = ", ".join(texts)
            text = f"[{text}]"
            return text
        else:
            return self.text

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def _to_text(cls, value):
        if cls._is_iterable(value):
            return cls._to_iterable_text(value)

        elif isinstance(value, torch.Tensor):
            return str(cls._tensor_to_str(value))

        elif cls._is_built_in_type(value):
            return f"<'{cls._to_type(value)}' {str(value)}>"

        elif value is None:
            return str(None)

        else:
            return cls._to_type(value)

    @classmethod
    def _to_iterable_text(cls, value):
        if isinstance(value, dict):
            texts = []
            for v in value:
                texts += cls._value_dict_to_texts(v) if cls._is_iterable(v) else [v]
        else:
            texts = []
            for v in value:
                texts += cls._list_values_to_texts(v) if cls._is_iterable(v) else [v]

    @classmethod
    def _value_dict_to_texts(cls, values):
        key_length = cls._get_max_key_length(values)
        value_length = cls._get_max_value_length(values)

        texts = []
        for key, value in values.items():
            if cls._is_iterable(value):
                texts += cls._to_iterable_text(value)
            else:
                text = cls._to_key_value_text(key, value, key_length, value_length)
                texts.append(text)

        return texts

    @classmethod
    def _list_values_to_texts(cls, values):
        brackets = cls._get_list_brackets(len(values))
        texts = [""] * len(values)
        for i, value in enumerate(values):
            texts[i] = f"{brackets[i][0]} {value.text} {brackets[i][1]}"
        return texts

    @classmethod
    def _get_list_brackets(cls, size):
        return [cls._get_list_bracket(i, size) for i in range(size)]

    @classmethod
    def _to_shape_list(cls, value):
        if cls._is_tensor(value):
            return cls._tensor_to_str(value)
        return None

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
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
    def _is_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        return False

    @staticmethod
    def _to_key_value_text(key, value, key_length, value_length):
        return f"{key:>{key_length}}: {value:>{value_length}}"

    @staticmethod
    def _tensor_to_str(value):
        size = list(value.shape)
        if len(size) == 0:
            size = [1]

        return list(map(str, size))

    @staticmethod
    def _to_type(value):
        return str(value.__class__.__name__)

    @staticmethod
    def _is_built_in_type(value):
        if isinstance(value, (bool, int, float, complex, str)):
            return True

        return False

    @staticmethod
    def _is_iterable(value):
        if isinstance(value, (tuple, list, set, dict)):
            return True
        return False

    @classmethod
    def _has_iterable(cls, values):
        if True in [cls._is_iterable(value) for value in values]:
            return True
        return False
