from abc import ABCMeta
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from nenepy.torch.interfaces import Mode
from torch.utils.data import Dataset as TorchDataset
from nenepy.torch import transforms as tf


class Dataset(TorchDataset, metaclass=ABCMeta):

    def __init__(self, crop_size, is_memory_stack=False):
        self._is_memory_stack = is_memory_stack

        self._resize_only_transform = tf.Compose([
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
        return list(self._class_names.values())

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================

    @classmethod
    def _load_image_path(cls, root_dir, mode):
        """

        Args:
            root_dir (Path):
            mode (Mode):

        Returns:
            list[Path]:
            list[Path]:
            list[Path]:
            list[Path]:

        """

        raw_image_paths = []
        raw_depth_paths = []
        rgb_depth_paths = []

        if mode == Mode.TRAIN:
            data_list_file = root_dir.joinpath(cls.TRAIN_LIST)
            with open(str(data_list_file), "r") as f:
                for line in f.readlines():
                    paths = line.strip("\n").split(" ")

                    raw_image_paths.append(root_dir.joinpath(paths[0]))
                    raw_depth_paths.append(root_dir.joinpath(paths[1]))
                    rgb_depth_paths.append(root_dir.joinpath(paths[2]))

        elif mode == Mode.VALIDATE:
            data_list_file = root_dir.joinpath(cls.TRAIN_LIST)
            with open(str(data_list_file), "r") as f:
                for line in f.readlines():
                    paths = line.strip("\n").split(" ")

                    raw_image_paths.append(root_dir.joinpath(paths[0]))
                    raw_depth_paths.append(root_dir.joinpath(paths[1]))
                    rgb_depth_paths.append(root_dir.joinpath(paths[2]))

        else:
            raise ValueError(f"Undefined mode {mode}")

        return raw_image_paths, raw_depth_paths, rgb_depth_paths

    @classmethod
    def load_label(cls, root_dir):
        class_ids = np.genfromtxt(root_dir.joinpath(cls.CLASS_LABEL), delimiter=',')
        class_ids[np.isnan(class_ids)] = 0
        class_ids = class_ids.astype(np.int)
        return class_ids

    @classmethod
    def _load_class_list(cls, root_dir):
        """

        Args:
            root_dir (Path):

        Returns:
            dict[int, str]:
            dict[str, int]:

        """
        class_list_file = root_dir.joinpath(cls.CLASS_LIST)

        class_names = {}
        class_indexes = {}
        with open(str(class_list_file), "r") as f:
            for line in f.readlines():
                index, class_name = line.strip("\n").split(" ")
                index = int(index)
                class_names[index] = class_name
                class_indexes[class_name] = index
        return class_names, class_indexes

    @classmethod
    def _load_images(cls, raw_image_paths, raw_depth_paths, rgb_depth_paths):
        size = len(raw_image_paths)
        raw_images = [None] * size
        raw_depths = [None] * size
        rgb_depths = [None] * size
        for i in range(size):
            raw_images[i] = Image.open(raw_image_paths[i]).convert("RGB")

        return raw_images, raw_depths, rgb_depths

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
        if self._is_memory_stack:
            raw_image = self._raw_images[index]

        else:
            raw_image = Image.open(self._raw_image_paths[index]).convert("RGB")
            # rgb_depth = Image.open(self._rgb_depth_paths[index]).convert("RGB")
            # rgb_depth = Image.open(self._raw_depth_paths[index]).convert("L")
            # rgb_depth = rgb_depth.convert("RGB")
            # # rgb_depth = Image.fromarray(np.array(raw_depth)[:, :, None])
            # label = self._label[index]

        if self._mode == Mode.TRAIN:
            if cfg.DATASET.TRAIN.APPLY_TRANSFORM:
                raw_image, rgb_depth = self._train_transform(raw_image)
            else:
                raw_image, rgb_depth = self._resize_only_transform(raw_image)

        elif self._mode == Mode.VALIDATE:
            if cfg.DATASET.VALIDATE.APPLY_TRANSFORM:
                raw_image, rgb_depth = self._validate_transform(raw_image)
            else:
                raw_image, rgb_depth = self._resize_only_transform(raw_image)

        # if self._mode == Mode.TRAIN:
        #     if cfg.DATASET.TRAIN.APPLY_TRANSFORM:
        #         raw_image, rgb_depth = self._train_transform(raw_image, rgb_depth)
        #     else:
        #         raw_image, rgb_depth = self._resize_only_transform(raw_image, rgb_depth)
        #
        # elif self._mode == Mode.VALIDATE:
        #     if cfg.DATASET.VALIDATE.APPLY_TRANSFORM:
        #         raw_image, rgb_depth = self._validate_transform(raw_image, rgb_depth)
        #     else:
        #         raw_image, rgb_depth = self._resize_only_transform(raw_image, rgb_depth)

        # labels = torch.from_numpy(label).type(torch.float32)
        # Ignoring background
        # return raw_image, rgb_depth, torch.empty(size=(1, 1)), labels, self._raw_image_paths[index].name
        return raw_image, None, None, None, self._raw_image_paths[index].name

    def __len__(self):
        """

        """
        return len(self._raw_image_paths)
