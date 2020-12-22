import random

from torchvision import transforms


class RandomColorJitter(transforms.ColorJitter):

    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        super(RandomColorJitter, self).__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p

    def __call__(self, *images):
        if random.random() < self.p:
            transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            return (transform(img) if (img.mode != "L") and (img.mode != "P") and (img.mode != "I") else img for img in images)

        return images
