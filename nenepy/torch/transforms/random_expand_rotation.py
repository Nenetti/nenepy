import random

import numpy as np
from PIL import Image
from nenepy.torch.images import Color
from torchvision import transforms

class RandomExpandRotation:

    def __init__(self, size, p=0.5, degrees=(0, 0), tile_size=40):
        self._resized = transforms.Resize(size=size)
        self._p = p
        self._degrees = degrees
        self._tile_size = tile_size
        self._color_palette = Color.to_color(np.arange(100))
        self._color_tile = [None] * len(self._color_palette)

        for i, color in enumerate(self._color_palette):
            tile = np.empty((tile_size, tile_size, 3), dtype=np.uint8)
            tile[:, :] = color
            self._color_tile[i] = tile

    @staticmethod
    def _random_rotate(img, degree, fillcolor):
        """

        Args:
            img (Image.Image):

        Returns:

        """
        return img.rotate(degree, expand=True, fillcolor=fillcolor)

    @staticmethod
    def _random_rotate_noise_fill(img, degree):
        """

        Args:
            img (Image.Image):

        Returns:

        """
        mode = img.mode
        img = img.convert("RGBA").rotate(degree, expand=True)
        fff = Image.fromarray(np.random.randint(0, 255, size=(img.height, img.width, 3), dtype=np.uint8))
        return Image.composite(img, fff, img).convert(mode)

    @staticmethod
    def _random_rotate_random_tile(img, degree, color_tile, tile_size):
        """

        Args:
            img (Image.Image):

        Returns:

        """
        mode = img.mode
        size = img.size
        img = img.convert("RGBA").rotate(degree, expand=True)
        w, h = (int(size[0] // tile_size), int(size[1] // tile_size))
        tile = np.concatenate([np.concatenate(random.choices(color_tile, k=h), axis=0) for i in range(w)], axis=1)
        fff = Image.fromarray(tile).resize(img.size)
        return Image.composite(img, fff, img).convert(mode)

    def __call__(self, *images):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        if random.random() < self._p:
            degree = random.uniform(*self._degrees)
            if random.random() < 0.5:
                fillcolor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                images = [self._random_rotate(img, degree, fillcolor) for img in images]
            else:
                images = [self._random_rotate_random_tile(img, degree, self._color_tile, self._tile_size) for img in images]

            images = [self._resized(img) for img in images]

        return images
