from PIL import Image

from nenepy.torch.datasets import AbstractDataset


class AbstractImageDataset(AbstractDataset):

    @staticmethod
    def _load_rgb_image(path):
        return Image.open(path).convert("RGB")

    @staticmethod
    def _load_index_mask_image_with_palette(path):
        return Image.open(path).convert("P")

    @staticmethod
    def _load_rgb_mask_image_with_palette(path):
        return Image.open(path).convert("P").convert("RGB")
