from enum import Enum, auto
from multiprocessing.connection import Pipe
from multiprocessing.context import Process
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import numpy as np


def launch(log_dir, child_conn):
    writer = _MultiProcessWriter(log_dir, child_conn)
    writer.loop_callback()


# ==================================================================================================
#
#   Type
#
# ==================================================================================================
class Type(Enum):
    SCALAR = auto()
    SCALARS = auto()
    IMAGE = auto()
    IMAGES = auto()
    COMPLETE = auto()


# ==================================================================================================
#
#   _SingleProcessWriter
#
# ==================================================================================================
class _SingleProcessWriter:

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, namespace, graph_name, scalar_value, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_value (float):
            step (int):

        """
        self.writer.add_scalar(tag=f"{namespace}/{graph_name}", scalar_value=scalar_value, global_step=step)
        self.flush()

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

    def add_images(self, tag, image_dict, step):
        """

        Args:
            tag (str):
            image_dict (dict[str, torch.Tensor]):
            step (int):

        """
        for name, image in image_dict.items():
            self.add_image(namespace=tag, name=name, image=image, step=step)

    def flush(self):
        """

        """
        self.writer.flush()


# ==================================================================================================
#
#   _MultiProcessWriter
#
# ==================================================================================================
class _MultiProcessWriter(_SingleProcessWriter):

    def __init__(self, log_dir, subscriber):
        """

        Args:
            log_dir:
            subscriber (PipeConnection):

        """
        super(_MultiProcessWriter, self).__init__(log_dir)
        self.subscriber = subscriber

    def add_images_with_process(self, function, tag, names, args, step):
        """

        Args:
            function (function):
            tag (str):
            names (tuple[str])
            args (dict[str, tuple]):
            step (int):

        """
        image_dict = dict(zip(names, function(*args)))
        self.add_images(tag, image_dict, step)

    def loop_callback(self):
        while True:
            data = self.subscriber.recv()
            if data[0] is Type.SCALAR:
                self.add_scalar(*data[1:])

            elif data[0] is Type.SCALARS:
                self.add_scalars(*data[1:])

            elif data[0] is Type.IMAGE:
                self.add_image(*data[1:])

            elif data[0] is Type.IMAGES:
                self.add_images(*data[1:])

            elif data[0] is Type.COMPLETE:
                self.subscriber.send("")
                break

            self.flush()


# ==================================================================================================
#
#   TensorBoardWriter
#
# ==================================================================================================
class TensorBoardWriter:

    def __init__(self, log_dir, multi_process=False, n_processes=1):
        """

        Args:
            log_dir:

        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.multi_process = multi_process
        self.process_id = 0
        self.n_processes = n_processes

        if self.multi_process:
            self.publishers = [None] * n_processes
            for i in range(n_processes):
                self.publishers[i], subscriber = Pipe()
                Process(target=launch, args=(log_dir, subscriber), daemon=True).start()

        else:
            self._writer = _SingleProcessWriter(log_dir)

    def send(self, data_type, args):
        self.publishers[self.process_id].send((data_type, *args))
        self.process_id += 1
        if self.process_id >= self.n_processes:
            self.process_id = 0

    def add_scalar(self, namespace, graph_name, scalar_value, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_value (float):
            step (int):

        """
        if self.multi_process:
            self.send(Type.SCALAR, (namespace, graph_name, scalar_value, step))
        else:
            self._writer.add_scalar(namespace, graph_name, scalar_value, step)

    def add_scalars(self, namespace, graph_name, scalar_dict, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_dict (dict[str, (torch.Tensor or np.ndarray or int or float)]):
            step (int):

        """
        if self.multi_process:
            self.send(Type.SCALARS, (namespace, graph_name, scalar_dict, step))
        else:
            self._writer.add_scalars(namespace, graph_name, scalar_dict, step)

    def add_image(self, namespace, name, image, step):
        """

        Args:
            namespace (str):
            image_dict (dict[str, torch.Tensor]):
            step (int):

        """
        if self.multi_process:
            self.send(Type.IMAGE, (namespace, name, image, step))
        else:
            self._writer.add_image(namespace, name, image, step)

    def add_images(self, namespace, image_dict, step):
        """

        Args:
            namespace (str):
            image_dict (dict[str, torch.Tensor]):
            step (int):

        """
        if self.multi_process:
            self.send(Type.IMAGES, (namespace, image_dict, step))
        else:
            self._writer.add_images(namespace, image_dict, step)

    def wait_process_completed(self):
        if self.multi_process:
            for i in range(self.n_processes):
                self.publishers[i].send((Type.COMPLETE,))
                self.publishers[i].recv()

        return
