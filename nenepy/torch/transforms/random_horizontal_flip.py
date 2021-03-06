import random

from torchvision import transforms
from torchvision.transforms import functional as F


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def forward(self, *images):
        if random.random() < self.p:
            return (F.hflip(img) for img in images)

        return images