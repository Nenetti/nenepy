import numpy as np
import matplotlib.pyplot as plt

from nenepy.torch.utils.images.color import Color


class Mask:

    @classmethod
    def mask_to_heatmap(cls, mask):
        """
        Create Class Activation Map.

        Args:
            mask (np.ndarray):

        Returns:
            np.ndarray:

        Shapes:
            [C, H, W] -> [3, H, W]

        """
        return cls.to_jet(mask)

    @classmethod
    def to_viridis(cls, image):
        """
        Args:
            image (np.ndarray): [H, W]

        Returns:
            np.ndarray:

        Shapes:
            [H, W] -> [3, H, W]

        """
        image = cls._to_2channels(image)

        rgb = plt.get_cmap("viridis")(image)[:, :, :3]
        return rgb.astype(np.float32).transpose((2, 0, 1))

    @classmethod
    def to_plasma(cls, image):
        """
        Args:
            image (np.ndarray): [H, W]

        Returns:
            np.ndarray:

        Shapes:
            [H, W] -> [3, H, W]

        """
        image = cls._to_2channels(image)

        rgb = plt.get_cmap("plasma")(image)[:, :, :3]
        return rgb.astype(np.float32).transpose((2, 0, 1))

    @classmethod
    def to_jet(cls, image):
        """
        Args:
            image (np.ndarray): [H, W]

        Returns:
            np.ndarray:

        Shapes:
            [H, W] -> [3, H, W]

        """
        image = cls._to_2channels(image)

        rgb = plt.get_cmap("jet")(image)[:, :, :3]
        return rgb.astype(np.float32).transpose((2, 0, 1))

    @classmethod
    def index_image_to_rgb_image(cls, image):
        """
        Args:
            image (np.ndarray):

        Returns:
            np.ndarray:

        Shapes:
            -> [H, W] or [1, H, W]
            <- [3, H, W]

        """
        image = cls._to_2channels(image)

        H, W = image.shape
        color_image = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        labels = np.unique(image)

        rgbs = Color.indexes_to_rgbs(labels)
        for label, rgb in zip(labels, rgbs):
            color_image[image == label] = rgb

        color_image = color_image.astype(np.float32).transpose((2, 0, 1))

        return color_image / 255.0

    @classmethod
    def to_binary_mask(cls, mask, threshold=0.0):
        """
        Colorize prediction mask.

        Args:
            mask (np.ndarray):
            threshold (float):

        Returns:
            np.ndarray:

        Shapes:
            -> [C, H, W]
            <- [3, H, W]

        """
        binary_mask = np.zeros_like(mask)
        binary_mask[mask > threshold] = 1
        return binary_mask

    @classmethod
    def probabilistic_mask_to_rgb_image(cls, mask, threshold=0.0):
        """
        Colorize prediction mask.

        Args:
            mask (np.ndarray):
            threshold (float):

        Returns:
            np.ndarray:

        Shapes:
            -> [C, H, W]
            <- [3, H, W]

        """
        alpha = np.max(mask, axis=0)
        if threshold > 0:
            alpha[alpha > threshold] = 0

        indexes = np.argmax(mask, axis=0)
        rgb_mask = cls.index_image_to_rgb_image(indexes) * alpha
        return rgb_mask

    @classmethod
    def binary_mask_to_rgb_image(cls, binary_mask):
        """
        Colorize prediction mask.

        Args:
            binary_mask (np.ndarray):

        Returns:
            np.ndarray:

        Shapes:
            -> [C, H, W]
            <- [3, H, W]

        """
        indexes = np.argmax(binary_mask, axis=0)
        rgb_mask = cls.index_image_to_rgb_image(indexes)

        return rgb_mask

    @staticmethod
    def blend_images(image1, image2, alpha=0.5):
        """
        Blend image1 with image2.

        Args:
            image1 (np.ndarray):
            image2 (np.ndarray):
            alpha (float):

        Returns:
            np.ndarray

        Shapes:
            -> [3, H, W]
            -> [3, H, W]
            <- [3, H, W]
                or
            -> [H, W, 3]
            -> [H, W, 3]
            <- [H, W, 3]

        """
        if image1.dtype != image2.dtype:
            raise TypeError(f"The type of numpy must be same, but given '{image1.dtype}' and '{image2.dtype}'")
        if image1.dtype != np.float or image2.dtype != np.float:
            dtype = image1.dtype
            image = alpha * image1.astype(np.float) + (1 - alpha) * image2.astype(np.float)
            return image.astype(dtype)
        else:
            return alpha * image1 + (1 - alpha) * image2

    @staticmethod
    def _to_2channels(array):
        """

        Args:
            array (np.ndarray):

        Returns:

        """
        if len(array.shape) == 2:
            return array

        if (len(array.shape) == 3) and (array.shape[0] == 1):
            return array[0]

        raise ValueError(f"The Shape must be '[1, H, W] or [H, W]' but given {array.shape}")
