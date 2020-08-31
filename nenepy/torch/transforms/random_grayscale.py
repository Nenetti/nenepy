import random

from torchvision import transforms
from torchvision.transforms import functional as F


class RandomGrayscale(transforms.RandomGrayscale):

    def __call__(self, *images):
        if random.random() < self.p:
            return (F.to_grayscale(img, num_output_channels=(1 if img.mode == 'L' else 3)) for img in images)

        return images