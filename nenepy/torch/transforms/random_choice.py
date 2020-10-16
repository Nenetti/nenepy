import random

from torchvision.transforms.transforms import RandomTransforms


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """

    def __call__(self, *img):
        t = random.choice(self.transforms)
        return t(*img)
