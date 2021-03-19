from enum import Enum, auto
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from nenepy.utils.multi.multi_task_process import MultiTaskProcess


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
    IMAGES_WITH_FUNCTION = auto()
    COMPLETE = auto()


# ==================================================================================================
#
#   TensorBoardWriter
#
# ==================================================================================================
class TensorBoardWriter(MultiTaskProcess):

    def __init__(self, log_dir):
        """

        Args:
            log_dir:

        """
        super(TensorBoardWriter, self).__init__()
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._log_dir = log_dir
        self._writer = None

    # ==================================================================================================
    #
    #   Main Process function
    #
    # ==================================================================================================
    def add_scalar(self, namespace, graph_name, scalar_value, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_value (float):
            step (int):

        """
        self.add_task(Type.SCALAR, (namespace, graph_name, scalar_value, step))

    def add_scalars(self, namespace, graph_name, scalar_dict, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_dict (dict[str, torch.Tensor or float]):
            step (int):

        """
        self.add_task(Type.SCALARS, (namespace, graph_name, scalar_dict, step))

    def add_image(self, namespace, name, image, step):
        """

        Args:
            namespace (str):
            name (str):
            image (torch.Tensor):
            step (int):

        """
        self.add_task(Type.IMAGE, (namespace, name, image, step))

    def add_images(self, tag, image_dict, step):
        """

        Args:
            tag (str):
            image_dict (dict[str, torch.Tensor]):
            step (int):

        """
        for name, image in image_dict.items():
            self.add_task(Type.IMAGES, (tag, name, image, step))

    def add_images_with_process(self, func, args, tag, step):
        self.add_task(Type.IMAGES_WITH_FUNCTION, ((func, args), (tag, step)))

    # ==================================================================================================
    #
    #   Other Process function
    #
    # ==================================================================================================
    def on_start(self):
        self._writer = SummaryWriter(log_dir=self._log_dir)

    def on_exit(self):
        self._writer.close()

    def process(self, *task):
        self._write(*task)

    def _write(self, data_type, args):
        if data_type is Type.SCALAR:
            self._add_scalar(*args)

        elif data_type is Type.SCALARS:
            self._add_scalars(*args)

        elif data_type is Type.IMAGE:
            self._add_image(*args)

        elif data_type is Type.IMAGES:
            self._add_images(*args)

        elif data_type is Type.IMAGES_WITH_FUNCTION:
            (func, func_args), (tag, step) = args
            output = func(*func_args)
            self._add_images(tag, output, step)

        else:
            raise ValueError()

        self._flush()

    def _add_scalar(self, namespace, graph_name, scalar_value, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_value (float):
            step (int):

        """
        self._writer.add_scalar(tag=f"{namespace}/{graph_name}", scalar_value=scalar_value, global_step=step)
        self._flush()

    def _add_scalars(self, namespace, graph_name, scalar_dict, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_dict (dict[str, torch.Tensor]):
            step (int):

        """
        self._writer.add_scalars(main_tag=f"{namespace}/{graph_name}", tag_scalar_dict=scalar_dict, global_step=step)
        self._flush()

    def _add_image(self, namespace, name, image, step):
        """

        Args:
            namespace (str):
            name (str):
            image (torch.Tensor):
            step (int):

        """
        self._writer.add_image(tag=f"{namespace}/{name}", img_tensor=image, global_step=step)
        self._flush()

    def _add_images(self, tag, image_dict, step):
        """

        Args:
            tag (str):
            image_dict (dict[str, torch.Tensor]):
            step (int):

        """
        for name, image in image_dict.items():
            self._add_image(namespace=tag, name=name, image=image, step=step)

    def _flush(self):
        """

        """
        self._writer.flush()
