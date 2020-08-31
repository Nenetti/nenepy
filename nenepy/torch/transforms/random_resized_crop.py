import PIL
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


class RandomResizedCrop(transforms.RandomResizedCrop):

    @staticmethod
    def to_resized_crop(img, top, left, height, width, size, interpolation=Image.BILINEAR):
        if isinstance(img, PIL.Image.Image) and img.mode == "P" or img.mode == "I":
            return F.resized_crop(img, top, left, height, width, size, Image.NEAREST)
        else:
            return F.resized_crop(img, top, left, height, width, size, interpolation)

    def __call__(self, *images):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(images[0], self.scale, self.ratio)

        return (self.to_resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in images)