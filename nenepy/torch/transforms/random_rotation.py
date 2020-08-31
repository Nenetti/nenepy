import random

from torchvision import transforms
from torchvision.transforms import functional as F


class RandomRotation(transforms.RandomRotation):

    def __init__(self, p, degrees, resample=False, expand=False, center=None, fill=None):
        super(RandomRotation, self).__init__(degrees, resample, expand, center, fill)
        self.p = p

    def __call__(self, *images):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        if random.random() < self.p:
            angle = self.get_params(self.degrees)
            return (F.rotate(img, angle, self.resample, self.expand, self.center, self.fill) for img in images)

        return images