import math
import random

import PIL
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class Resize(transforms.Resize):

    @staticmethod
    def resize(img, size, interpolation=Image.BILINEAR):
        if isinstance(img, PIL.Image.Image) and img.mode == "P" or img.mode == "I":
            return F.resize(img, size, Image.NEAREST)
        else:
            return F.resize(img, size, interpolation)

    def __call__(self, *images):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """

        return (self.resize(img, self.size, self.interpolation) for img in images)
