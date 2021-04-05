import random

from torchvision import transforms
from torchvision.transforms import functional as F


class RandomGrayscale(transforms.RandomGrayscale):

    def forward(self, *images):
        if random.random() < self.p:
            return (F.rgb_to_grayscale(img, num_output_channels=F._get_image_num_channels(img)) if img.mode != "P" and img.mode != "I" else img for img in images)

        return images
