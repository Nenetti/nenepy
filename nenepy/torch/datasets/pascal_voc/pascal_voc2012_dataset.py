from collections import OrderedDict
from multiprocessing.pool import Pool
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from nenepy.torch import transforms as tf
from nenepy.torch.datasets import AbstractImageDataset

from nenepy.torch.interfaces import Mode


class PascalVoc2012Dataset(AbstractImageDataset):
    _TRAIN_LIST = "train_augvoc.txt"
    _VALIDATE_LIST = "val_voc.txt"

    _NUM_CLASSES = 20

    _CLASS_DICT = OrderedDict({
        "background": 0,
        "aeroplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "person": 15,
        "potted-plant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tv/monitor": 20,
        "ambiguous": 255
    })

    _CLASS_INDEXES = list(_CLASS_DICT.values())
    _CLASS_NAMES = list(_CLASS_DICT.keys())

    def __init__(self, mode, root_dir, crop_size=(256, 256), scale=(1.0, 1.0), is_transform=False, is_on_memory=True, n_processes=4):
        """

        Args:
            mode (Mode):
            root_dir (str):
            crop_size ((int, int)):
            scale ((float, float)):
            is_transform (bool):
            is_on_memory (bool):

        """
        super(PascalVoc2012Dataset, self).__init__()

        root_dir = Path(root_dir.replace("~", str(Path.home())))
        self._mode = mode
        self._is_train = (self._mode == Mode.TRAIN)
        self._is_transform = is_transform if self._is_train else False
        self._is_on_memory = is_on_memory
        self._n_processes = n_processes

        self._rgb_image_paths, self._index_mask_image_paths = self._load_image_path(self._mode, root_dir)
        self._n_data = len(self._rgb_image_paths)

        self._full_transform = tf.Compose([
            tf.RandomResizedCrop(size=crop_size, scale=scale),
            tf.RandomColorJitter(p=1.0, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            tf.RandomRotation(p=0.2, degrees=180),
            tf.RandomHorizontalFlip(p=0.5),
            tf.ToTensor(),
        ])

        self._resize_transform = tf.Compose([
            tf.RandomResizedCrop(size=crop_size, scale=(1.0, 1.0)),
            tf.ToTensor(),
        ])

        self._transform = self._full_transform if self._is_transform else self._resize_transform

        self._rgb_images, self._index_mask_images, self._rgb_mask_images = (None, None, None)
        if self._is_on_memory:
            self._rgb_images, self._index_mask_images, self._rgb_mask_images = self._load_images_on_memory()

    # ==================================================================================================
    #
    #   Property
    #
    # ==================================================================================================
    @property
    def class_names(self):
        return self._CLASS_NAMES[1:-1]

    @property
    def n_classes(self):
        return self._NUM_CLASSES

    # ==================================================================================================
    #
    #   Special Method
    #
    # ==================================================================================================
    def __len__(self):
        return len(self._rgb_image_paths)

    def __getitem__(self, index):
        """

        Args:
            index (int):

        Returns:
            torch.Tensor:
            torch.Tensor:
            torch.Tensor:
            torch.Tensor:
            str:

        """
        file_name = self._rgb_image_paths[index].name
        rgb_image, index_mask_image, rgb_mask_image = self._get_image_set(index)

        if not (self._is_on_memory and self._is_transform):
            rgb_image, index_mask_image, rgb_mask_image = self._transform(rgb_image, index_mask_image, rgb_mask_image)

        indexes = torch.unique(index_mask_image)
        # Ignoring background and ambiguous
        if indexes[-1] == self._CLASS_INDEXES[-1]:
            indexes = indexes[:-1]
        if indexes[0] == self._CLASS_INDEXES[0]:
            indexes = indexes[1:]
        indexes -= 1

        labels = torch.zeros(size=(self._NUM_CLASSES,))
        for i in indexes:
            labels[i] = 1

        if self._is_train:
            return rgb_image, labels, file_name
        else:
            mask = self._index_to_channel(index_mask_image.squeeze(0), self._NUM_CLASSES)
            return rgb_image, mask, rgb_mask_image, labels, file_name

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    def _load_images_on_memory(self):
        if self._is_transform:
            rgb_images = [None] * self._n_data
            index_mask_images = [None] * self._n_data
            rgb_mask_images = [None] * self._n_data

            for i in tqdm(range(self._n_data), ascii=True):
                rgb_image = self._load_rgb_image(self._rgb_image_paths[i])
                index_mask_image = self._load_index_mask_image_with_palette(self._index_mask_image_paths[i])
                rgb_mask_image = self._load_rgb_mask_image_with_palette(self._index_mask_image_paths[i])

                image_set = (rgb_image, index_mask_image, rgb_mask_image)
                resized_images = self._resize_transform(*image_set)
                rgb_images[i], index_mask_images[i], rgb_mask_images[i] = resized_images
        else:
            with Pool(processes=self._n_processes) as p:
                rgb_images = p.map(self._load_rgb_image, tqdm(self._rgb_image_paths, ascii=True))
                index_mask_images = p.map(self._load_index_mask_image_with_palette, tqdm(self._index_mask_image_paths, ascii=True))
                rgb_mask_images = p.map(self._load_rgb_mask_image_with_palette, tqdm(self._index_mask_image_paths, ascii=True))

        return rgb_images, index_mask_images, rgb_mask_images

    def _get_image_set(self, index):
        if self._is_on_memory:
            rgb_image = self._rgb_images[index]
            index_mask_image = self._index_mask_images[index]
            rgb_mask_image = self._rgb_mask_images[index]
        else:
            rgb_image = self._load_rgb_image(self._rgb_image_paths[index])
            index_mask_image = self._load_index_mask_image_with_palette(self._index_mask_image_paths[index])
            rgb_mask_image = self._load_rgb_mask_image_with_palette(self._index_mask_image_paths[index])

        return rgb_image, index_mask_image, rgb_mask_image

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================
    @classmethod
    def _load_image_path(cls, mode, root_dir):
        data_list_file = cls._get_load_file(mode, root_dir)
        with open(str(data_list_file), "r") as f:
            lines = f.readlines()

            size = len(lines)
            rgb_image_paths = [None] * size
            mask_image_paths = [None] * size

            for i, line in enumerate(lines):
                paths = line.strip("\n").split(" ")
                rgb_image_paths[i] = root_dir.joinpath(paths[0])
                mask_image_paths[i] = root_dir.joinpath(paths[1])

        return rgb_image_paths, mask_image_paths

    @classmethod
    def _get_load_file(cls, mode, root):
        if mode == Mode.TRAIN:
            return root.joinpath(cls._TRAIN_LIST)

        elif mode == Mode.VALIDATE:
            return root.joinpath(cls._VALIDATE_LIST)

        else:
            raise TypeError(f"mode must be '{Mode.TRAIN}' or '{Mode.VALIDATE}', but given '{mode}'")

    @staticmethod
    def _index_to_channel(index_mask, n_indexes):
        """

        Args:
            index_mask (torch.Tensor):
            n_indexes (int):

        Returns:
            torch.Tensor:

        Shapes:
            -> [H, W]
            <- [N, H, W]

        """
        H, W = index_mask.shape
        gt_mask = torch.zeros(size=(H, W, n_indexes), dtype=torch.uint8)
        indexes = torch.unique(index_mask)
        if len(indexes) > 0:
            if indexes[-1] == 255:
                indexes = indexes[:-1]
            if indexes[0] == 0:
                indexes -= 1
        for index in indexes:
            gt_mask[:, :, index][index_mask == index] = 1
        return gt_mask.permute(2, 0, 1)
