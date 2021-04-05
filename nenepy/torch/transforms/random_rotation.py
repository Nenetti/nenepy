import random

import torch
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode


class RandomRotation(transforms.RandomRotation):

    def __init__(self, p, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None):
        super(RandomRotation, self).__init__(degrees, interpolation, expand, center, fill, resample)
        self.p = p

    def to_fill_color(self, img):
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        return fill

    def forward(self, *images):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        if random.random() < self.p:
            angle = self.get_params(self.degrees)
            return (F.rotate(img, angle, self.resample, self.expand, self.center, self.to_fill_color(img)) for img in images)

        return images
