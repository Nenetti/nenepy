import PIL
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F


class ToTensor(transforms.ToTensor):

    @staticmethod
    def to_tensor(pic):
        """
        F.to_tensor()はuint8はすべてfloat型に変換後255で割るため，mode=="L" の時はint16に変換する必要あり
        """
        if isinstance(pic, PIL.Image.Image):
            if pic.mode == "P":
                pic = np.array(pic, np.uint8, copy=False).astype(np.int16)
            elif pic.mode == "I":
                pic = np.array(pic, np.int32)

        return F.to_tensor(pic)

    def __call__(self, *images):
        return (self.to_tensor(img) for img in images)