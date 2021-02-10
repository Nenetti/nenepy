import torch


class Value:

    def __init__(self, value):
        self.value = value
        self.is_tensor = self._is_tensor(value)
        self.shapes = self.to_shape_list(value)
        self.text = self._to_text(value)
        self.type = self._to_type(value)

    # ==================================================================================================
    #
    #   Class Method
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

    @classmethod
    def _to_text(cls, value):
        if isinstance(value, torch.Tensor):
            return str(cls._tensor_to_str(value))
        elif cls._is_built_in_type(value):
            return f"<'{cls._to_type(value)}' {str(value)}>"
        elif value is None:
            return str(None)
        else:
            return cls._to_type(value)

    @staticmethod
    def _is_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        return False

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @classmethod
    def to_shape_list(cls, value):
        if cls._is_tensor(value):
            return cls._tensor_to_str(value)
        return None

    @staticmethod
    def _tensor_to_str(value):
        size = list(value.shape)
        # if len(size) > 0:
        #     size[0] = -1

        return list(map(str, size))

    @staticmethod
    def _to_type(value):
        return str(value.__class__.__name__)

    @staticmethod
    def _is_built_in_type(value):
        if isinstance(value, (bool, int, float, complex, str)):
            return True

        return False
