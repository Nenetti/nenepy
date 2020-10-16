from enum import Enum, auto
from multiprocessing.connection import Pipe
from multiprocessing.context import Process
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import numpy as np


def launch(log_dir, child_conn):
    writer = _MultiProcessWriter(log_dir, child_conn)
    writer.loop_callback()


class Type(Enum):
    SCALAR = auto()
    IMAGE = auto()


class _Writer:

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalars(self, namespace, graph_name, scalar_dict, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_dict (dict[str, torch.Tensor]):
            step (int):

        """
        self.writer.add_scalars(main_tag=f"{namespace}/{graph_name}", tag_scalar_dict=scalar_dict, global_step=step)
        self.flush()

    def add_image(self, namespace, name, image, step):
        """

        Args:
            namespace (str):
            name (str):
            image (torch.Tensor):
            step (int):

        """
        self.writer.add_image(tag=f"{namespace}/{name}", img_tensor=image, global_step=step)
        self.flush()

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
        """

        """
        self.writer.flush()


class _MultiProcessWriter(_Writer):

    def __init__(self, log_dir, connection):
        super(_MultiProcessWriter, self).__init__(log_dir)
        self.connection = connection

    def loop_callback(self):
        while True:
            data = self.connection.recv()

            if data[0] is Type.SCALAR:
                self.add_scalars(*data[1:])

            elif data[0] is Type.IMAGE:
                self.add_images(*data[1:])

            self.flush()


class TensorBoardWriter:

    def __init__(self, log_dir, multi_process=False):
        """

        Args:
            log_dir:

        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.multi_process = multi_process

        if self.multi_process:
            self.publisher, subscriber = Pipe()
            process = Process(target=launch, args=(log_dir, subscriber), daemon=True)
            process.start()

            self.image_publisher, image_subscriber = Pipe()
            image_only_process = Process(target=launch, args=(log_dir, image_subscriber), daemon=True)
            image_only_process.start()
        else:
            self._writer = _Writer(log_dir)

    def add_scalars(self, namespace, graph_name, scalar_dict, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_dict (dict[str, (torch.Tensor or np.ndarray or int or float)]):
            step (int):

        """
        if self.multi_process:
            self.publisher.send((Type.SCALAR, namespace, graph_name, scalar_dict, step))
        else:
            self._writer.add_scalars(namespace, graph_name, scalar_dict, step)

    def add_images(self, namespace, image_dict, step):
        """

        Args:
            namespace (str):
            image_dict (dict[str, torch.Tensor]):
            step (int):

        """
        if self.multi_process:
            self.image_publisher.send((Type.IMAGE, namespace, image_dict, step))
        else:
            self._writer.add_images(namespace, image_dict, step)
