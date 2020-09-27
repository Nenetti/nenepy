from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from nenepy.torch import transforms as tf
from nenepy.torch.interfaces import Mode


class MNISTDataset(Dataset):
    """

    """

    def __init__(self, mode, size=None, scale=None, root_dir="./data"):
        """

        Args:
            root_dir (str):
            mode (Mode):
        """
        root_dir = Path(root_dir)
        self.mode = mode

        is_train = (mode is Mode.TRAIN)
        self.dataset = torchvision.datasets.MNIST(root=root_dir, train=is_train, download=True)

        if size is None:
            size = (28, 28)
        if scale is None:
            scale = (1, 1)

        self._transform = tf.Compose([
            tf.RandomColorFlip(p=0.5),
            tf.RandomColorJitter(p=1.0, brightness=0.3, contrast=0.5, saturation=0.5, hue=0),
            tf.RandomResizedCrop(size=size, scale=scale),
            tf.ToTensor(),
        ])

    # ==============================================================================
    #
    #   Public Method
    #
    # ==============================================================================

    @property
    def class_names(self):
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # ==============================================================================
    #
    #   Special Attribute
    #
    # ==============================================================================

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:

        """
        img, label_index = self.dataset[index]
        img = img.convert("RGB")
        label = np.zeros(shape=10)
        label[label_index] = 1

        img = self._transform(img)

        label = torch.from_numpy(label).type(torch.float32)
        return img, label, str(index)

        # labels = torch.from_numpy(label).type(torch.float32)
        # # Ignoring background
        # return raw_image, rgb_depth, torch.empty(size=(1, 1)), labels, self._raw_image_paths[index].name

    def __len__(self):
        """

        """
        return len(self.dataset)
