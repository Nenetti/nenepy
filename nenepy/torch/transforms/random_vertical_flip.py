import random

from torchvision import transforms
from torchvision.transforms import functional as F


class RandomVerticalFlip(transforms.RandomVerticalFlip):

    def forward(self, *images):
        if random.random() < self.p:
            return (F.vflip(img) for img in images)

        return images