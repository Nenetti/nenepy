import torch
import numpy as np


class Value:

    def __init__(self, value):
        self.value = value
        self.is_tensor = self._is_tensor(value)
        self.sizes = self._to_shape_list(value)
        self.text = self._to_text(value)
        self.type = self._to_type(value)
        self.adjusted_text = None

    def to_adjusted_text(self, each_dim_size):
        """

        Args:
            each_dim_size (list[int]):

        Returns:

        """
        if self.is_tensor:
            return self._shape_to_text(self.sizes, each_dim_size)
        else:
            return self.text

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def calc_max_n_dims(cls, tensors):
        return max([len(cls._tensor_to_str(tensor)) for tensor in tensors])

    @classmethod
    def calc_max_each_dim_size(cls, tensors, max_n_dims):
        def func(tensor):
            each_size = np.zeros(shape=max_n_dims, dtype=np.int)
            shapes = cls._tensor_to_str(tensor)
            for i, value_str in enumerate(shapes):
                each_size[i] = len(value_str)

            return each_size

        return np.max(np.stack([func(tensor) for tensor in tensors], axis=0), axis=0)

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

    @classmethod
    def _to_shape_list(cls, value):
        if cls._is_tensor(value):
            return cls._tensor_to_str(value)
        return None

    @staticmethod
    def _shape_to_text(sizes, each_dim_size):
        texts = [f"{size:>{each_dim_size[i]}}" for i, size in enumerate(sizes)]
        text = ", ".join(texts)
        return f"[{text}]"

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def _is_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        return False

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
