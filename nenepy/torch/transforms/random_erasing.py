import random

from torchvision import transforms
from torchvision.transforms import functional as F


class RandomErasing(transforms.RandomErasing):

    def __call__(self, *images):
        if random.random() < self.p:
            x, y, h, w, v = self.get_params(images[0], scale=self.scale, ratio=self.ratio, value=self.value)
            return (F.erase(img, x, y, h, w, v, self.inplace) for img in images)

        return images