import random
import numpy as np
from PIL import Image


class RandomColorFlip:

    def __init__(self, p=0.5):
        super(RandomColorFlip, self).__init__()
        self.p = p

    @staticmethod
    def transform(image, channels):
        image = np.array(image)
        for channel in channels:
            image[:, :, channel] = 255 - image[:, :, channel]
        return Image.fromarray(image)

    def __call__(self, *images):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            c = random.sample([0, 1, 2], k)
            return (self.transform(img, c) if img.mode != "L" else img for img in images)

        return images
