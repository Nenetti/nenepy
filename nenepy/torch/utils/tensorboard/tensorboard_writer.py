import shutil
from enum import Enum, auto
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from nenepy.torch.utils.tensorboard.tensorboard import TensorBoard


class TensorBoardWriter(TensorBoard):

    def __init__(self, log_dir):
        """

        Args:
            log_dir:

        """
        self._log_dir = Path(log_dir)
        self._is_already_exist = True if self._log_dir.exists() else False
        self._writer = SummaryWriter(log_dir=log_dir)

    @property
    def is_already_exist(self):
        return self._is_already_exist

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def reset_directory(self):
        if self._log_dir.exists():
            shutil.rmtree(self._log_dir)
            self._log_dir.mkdir()

    def add_scalar(self, namespace, graph_name, scalar_value, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_value (float):
            step (int):

        """
        self._writer.add_scalar(tag=self._to_scalar_tag(namespace, graph_name), scalar_value=scalar_value, global_step=step)

    def add_scalars(self, namespace, graph_name, scalar_dict, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_dict (dict[str, torch.Tensor or float]):
            step (int):

        """
        self._writer.add_scalars(main_tag=self._to_scalar_tag(namespace, graph_name), tag_scalar_dict=scalar_dict, global_step=step)

    def add_image(self, namespace, name, image, step):
        """

        Args:
            namespace (str):
            name (str):
            image (torch.Tensor):
            step (int):

        """
        self._writer.add_image(tag=self._to_image_tag(namespace, name), img_tensor=image, global_step=step)

    def add_images(self, namespace, image_dict, step):
        """

        Args:
            namespace (str):
            image_dict (dict[str, torch.Tensor]):
            step (int):

        """
        for name, image in image_dict.items():
            self.add_image(namespace=namespace, name=name, image=image, step=step)

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
