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

    SCALARS = auto()
    IMAGE = auto()

    COMPLETE = auto()


class _Writer:

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag, graph_name, scalar_value, step):
        """

        Args:
            tag (str):
            graph_name (str):
            scalar_value (float):
            step (int):

        """
        self.writer.add_scalar(tag=f"{tag}/{graph_name}", scalar_value=scalar_value, global_step=step)
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
        """

        Args:
            log_dir:
            connection (PipeConnection):

        """
        super(_MultiProcessWriter, self).__init__(log_dir)
        self.connection = connection

    def loop_callback(self):
        while True:
            data = self.connection.recv()

            if data[0] is Type.SCALAR:
                self.add_scalar(*data[1:])

            if data[0] is Type.SCALARS:
                self.add_scalars(*data[1:])

            elif data[0] is Type.IMAGE:
                self.add_images(*data[1:])

            elif data[0] is Type.COMPLETE:
                self.connection.send("")

            self.flush()


class TensorBoardWriter:

    def __init__(self, log_dir, multi_process=False, n_processes=1):
        """

        Args:
            log_dir:

        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.multi_process = multi_process
        self.target_process_no = 0
        self.n_processes = n_processes

        if self.multi_process:
            self.scalar_main_connection, scalar_sub_connection = Pipe()
            Process(target=launch, args=(log_dir, scalar_sub_connection), daemon=True).start()

            self.image_main_connections = [None] * n_processes
            for i in range(n_processes):
                self.image_main_connections[i], image_sub_connection = Pipe()
                Process(target=launch, args=(log_dir, image_sub_connection), daemon=True).start()

        else:
            self._writer = _Writer(log_dir)

    def add_scalar(self, tag, graph_name, scalar_value, step):
        """

        Args:
            tag (str):
            graph_name (str):
            scalar_value (float):
            step (int):

        """
        if self.multi_process:
            self.scalar_main_connection.send((Type.SCALAR, tag, graph_name, scalar_value, step))
        else:
            self._writer.add_scalar(tag, graph_name, scalar_value, step)

    def add_scalars(self, namespace, graph_name, scalar_dict, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_dict (dict[str, (torch.Tensor or np.ndarray or int or float)]):
            step (int):

        """
        if self.multi_process:
            self.scalar_main_connection.send((Type.SCALARS, namespace, graph_name, scalar_dict, step))
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
            self.image_main_connections[self.target_process_no].send((Type.IMAGE, namespace, image_dict, step))
            self.next_process()
        else:
            self._writer.add_images(namespace, image_dict, step)

    def wait_process_completed(self):
        if self.multi_process:
            self.scalar_main_connection.send((Type.COMPLETE,))
            self.scalar_main_connection.recv()
            for i in range(self.n_processes):
                self.image_main_connections[i].send((Type.COMPLETE,))
                self.image_main_connections[i].recv()

        return

    def next_process(self):
        self.target_process_no += 1
        if self.target_process_no >= self.n_processes:
            self.target_process_no = 0
