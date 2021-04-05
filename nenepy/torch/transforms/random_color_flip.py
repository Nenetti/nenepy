import random
import numpy as np
from PIL import Image


class RandomColorFlip:

    def __init__(self, p=0.5):
        super(RandomColorFlip, self).__init__()
        self.p = p

    @staticmethod
    def transform(image):
        image = np.array(image)
        return Image.fromarray(255 - image)

    def __call__(self, *images):
        if random.random() < self.p:
            return (self.transform(img) if img.mode != "L" else img for img in images)

        return images
