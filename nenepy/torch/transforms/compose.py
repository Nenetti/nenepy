import PIL
import numpy as np
import torch

from torchvision import transforms


class Compose(transforms.Compose):

    def __call__(self, *images):

        if len(images) > 1:

            indexes, tensors = self.is_tensors(images)
            for t in self.transforms:
                tensors = t(*tensors)

            output = list(images)
            for index, tensor in zip(indexes, tensors):
                output[index] = tensor

            return tensors

        else:
            for t in self.transforms:
                images = t(*images)

            return images

    @staticmethod
    def is_tensors(images):
        indexes = [None] * len(images)
        tensors = [None] * len(images)
        index = 0
        for i, image in enumerate(images):
            if isinstance(image, PIL.Image.Image) or isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
                indexes[index] = i
                tensors[index] = image

        return indexes[:index + 1], tensors[:index + 1]
