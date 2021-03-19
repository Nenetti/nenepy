import PIL
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F


class ToTensor(transforms.ToTensor):

    @staticmethod
    def to_tensor(pic):
        if isinstance(pic, PIL.Image.Image):
            if pic.mode == "P":
                pic = np.array(pic, np.uint8).astype(np.int16)
            elif pic.mode == "I":
                pic = np.array(pic, np.int32)
            elif pic.mode == "L":
                pic = np.array(pic, np.int32)

        return F.to_tensor(pic)

    def __call__(self, *images):
        if len(images) > 1:
            return (self.to_tensor(img) for img in images)

        return self.to_tensor(*images)
