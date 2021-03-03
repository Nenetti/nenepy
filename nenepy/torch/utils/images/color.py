import numpy as np
import matplotlib.pyplot as plt


class Color:
    """

    Attributes:
        _COLOR_MAP (np.ndarray):  RGB Color Palette.

    """
    _COLOR_MAP = None

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def init_color_map(cls, n=256):
        cls._COLOR_MAP = cls._generate_color_map(n)

    @classmethod
    def index_to_rgb(cls, index):
        """
        Args:
            index (int):

        Returns:
            np.ndarray:

        Shapes:
            [1] -> [3]

        Examples:
            >>> color = Color.index_to_rgb(1)
            >>> print(color.shape)
            [3, 10]

        """
        return cls._COLOR_MAP[index]

    @classmethod
    def indexes_to_rgbs(cls, indexes):
        """
        Args:
            indexes (list or np.ndarray):

        Returns:
            np.ndarray:

        Shapes:
            [N] -> [N, 3]

        Examples:
            >>> x = [0] * 10
            >>> colors = Color.indexes_to_rgbs(x)
            >>> print(colors.shape)
            [3, 10]

        """
        return np.stack([cls._COLOR_MAP[index] for index in indexes], axis=0)

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================

    @classmethod
    def _generate_color_map(cls, n):
        """
        Args:
            n (int):

        Returns:
            np.ndarray:

        Shapes:
            <- [N, 3]

        """
        color_map = np.zeros(shape=(n, 3), dtype=np.uint8)
        for i in range(n):
            color_map[i] = cls._index_to_rgb(i)

        return color_map

    @staticmethod
    def _index_to_rgb(index):
        """
        Args:
            index (int):

        Returns:
            (int, int, int):

        """

        def bit(v, i):
            return (v & (1 << i)) != 0

        r, g, b = (0, 0, 0)
        for k in range(8):
            r = r | (bit(index, 0) << 7 - k)
            g = g | (bit(index, 1) << 7 - k)
            b = b | (bit(index, 2) << 7 - k)
            index = index >> 3

        return r, g, b


# Initialize color pallet
Color.init_color_map(n=256)
