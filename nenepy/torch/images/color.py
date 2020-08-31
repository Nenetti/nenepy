import numpy as np
import torch

import matplotlib.pyplot as plt


class Color:
    """

    Attributes:
        _color_map (torch.Tensor):  RGB Color Palette.

    """
    _color_map = None

    # ==============================================================================
    #
    #   Public Method
    #
    # ==============================================================================

    @staticmethod
    def to_jet(image):
        """
        Examples:
            >>> x = torch.rand(size=(128,128))
            >>> color = Color.to_jet(x)
            >>> print(color.shape)
            [3, 128, 128]

        Args:
            image (torch.Tensor): [H, W]

        Returns:
            torch.Tensor

        Shape:
            ([H, W] or [1, H, W]) -> [3, H, W]

        """
        if len(image.shape) != 2:
            image = image.squeeze()
        rgb = plt.get_cmap("jet")(image.numpy())[:, :, :3]
        return torch.from_numpy(rgb).type(torch.float32).permute(dims=(2, 0, 1))

    @classmethod
    def to_color(cls, indexes):
        """
        Examples:
            >>> x = [0] * 10
            >>> color = Color.to_color(x)
            >>> print(color.shape)
            [3, 10]

        Args:
            indexes (list or np.ndarray or torch.Tensor):

        Returns:
            torch.Tensor:

        Shape:
            [N] -> [N, 3]

        """
        return torch.stack([cls._color_map[index] for index in indexes], dim=0)

    @classmethod
    def index_to_color(cls, image):
        """
        Examples:
            >>> x = torch.randint(low=0, high=20, size=(128,128))
            >>> color = Color.index_to_color(x)
            >>> print(color.shape)
            [3, 128, 128]

        Args:
            image (torch.Tensor):

        Returns:
            torch.Tensor:

        Shape:
            ([H, W] or [1, H, W]) -> [3, H, W]

        """
        if len(image.shape) != 2:
            image = image.squeeze()

        height, width = image.size()
        color_image = torch.zeros(size=(height, width, 3), dtype=torch.uint8)
        labels = torch.unique(image)

        for label in labels:
            color_image[image == label] = cls._color_map[label]

        color_image = color_image.type(torch.float32).permute(dims=(2, 0, 1))

        return color_image / 255.0

    # ==============================================================================
    #
    #   Private Method
    #
    # ==============================================================================

    @classmethod
    def init_color_map(cls, n=256):
        cls._color_map = cls._create_color_map(n)

    @classmethod
    def _create_color_map(cls, n):

        color_map = np.zeros(shape=(n, 3), dtype=np.uint8)
        for i in range(n):
            color_map[i] = cls._index2rgb(i)

        return torch.from_numpy(color_map)

    @staticmethod
    def _index2rgb(index):
        def bit(v, i):
            return (v & (1 << i)) != 0

        r, g, b = (0, 0, 0)
        for k in range(8):
            r = r | (bit(index, 0) << 7 - k)
            g = g | (bit(index, 1) << 7 - k)
            b = b | (bit(index, 2) << 7 - k)
            index = index >> 3

        return r, g, b


Color.init_color_map(n=256)
