from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from nenepy.torch import transforms as tf
from nenepy.torch.interfaces import Mode


class MultiMNIST(Dataset):
    """

    """

    def __init__(self, mode, crop_size=None, scale=None, root_dir="./data",
                 is_transform=False):
        """

        Args:
            root_dir (str):
            mode (Mode):
        """
        root_dir = Path(root_dir)
        self.mode = mode

        self._is_transform = is_transform

        is_train = (mode is Mode.TRAIN)
        self.dataset = torchvision.datasets.MNIST(root=root_dir, train=is_train, download=True)

        if crop_size is None:
            crop_size = (28, 28)
        if scale is None:
            scale = (1, 1)

        crop_size = (32, 32)

        self.multi_transform = tf.Compose([
            tf.RandomChannelColorFlip(p=1.0),
            tf.RandomColorJitter(p=1.0, brightness=0, contrast=0, saturation=0, hue=(-0.5, 0.5)),
            tf.RandomResizedCrop(size=crop_size, scale=(1, 1)),
            tf.ToTensor(),
        ])

        self._transforms = tf.Compose([
            # tf.RandomColorFlip(p=0.5),
            tf.RandomChannelColorFlip(p=1.0),
            tf.RandomColorJitter(p=1.0, brightness=0, contrast=0, saturation=0, hue=(-0.5, 0.5)),
            tf.RandomResizedCrop(size=crop_size, scale=scale),
            tf.ToTensor(),
        ])

        self._resize_only_transforms = tf.Compose([
            tf.RandomResizedCrop(size=crop_size, scale=(1.0, 1.0)),
            tf.ToTensor(),
        ])

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    @property
    def class_names(self):
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # ==================================================================================================
    #
    #   Special Attribute
    #
    # ==================================================================================================

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

        imgs = [img.copy() for i in range(4)]
        for i in range(4):
            imgs[i] = self.multi_transform(imgs[i])

        a = torch.cat([imgs[0], imgs[1]], dim=1)
        b = torch.cat([imgs[2], imgs[3]], dim=1)
        img = torch.cat([a, b], dim=2)

        label = torch.from_numpy(label).type(torch.float32)
        return img, label, str(index)

    def __len__(self):
        """

        """
        return len(self.dataset)
