import math
import random

import PIL
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class RandomResizedCrop(transforms.RandomResizedCrop):

    @staticmethod
    def to_resized_crop(img, top, left, height, width, size, interpolation=Image.BILINEAR):
        if isinstance(img, PIL.Image.Image) and img.mode == "P" or img.mode == "I":
            return F.resized_crop(img, top, left, height, width, size, Image.NEAREST)
        else:
            return F.resized_crop(img, top, left, height, width, size, interpolation)

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.size

        s = random.uniform(*scale)
        w = math.ceil(width * s)
        h = math.ceil(height * s)
        w_range = width - w
        h_range = height - h
        i, j = random.randint(0, h_range), random.randint(0, w_range)
        return i, j, h, w

    def __call__(self, *images):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(images[0], self.scale, self.ratio)

        return (self.to_resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in images)
