import random

from torchvision import transforms
from torchvision.transforms import functional as F


class RandomColorJitter(transforms.ColorJitter):

    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        super(RandomColorJitter, self).__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p

    @staticmethod
    def to_transform(img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img

    def forward(self, *images):
        if random.random() < self.p:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )

            return (
                self.to_transform(img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
                if (img.mode != "L") and (img.mode != "P") and (img.mode != "I") else img
                for img in images
            )

        return images
