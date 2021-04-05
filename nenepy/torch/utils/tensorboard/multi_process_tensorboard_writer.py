from enum import Enum, auto
from pathlib import Path
from nenepy.torch.utils.tensorboard import TensorBoardWriter
from nenepy.utils.multi import MultiProcess


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
    IMAGE_WITH_FUNCTION = auto()
    IMAGES_WITH_FUNCTION = auto()
    COMPLETE = auto()


# ==================================================================================================
#
#   TensorBoardWriter
#
# ==================================================================================================
class MultiProcessTensorBoardWriter(MultiProcess):

    def __init__(self, log_dir):
        """

        Args:
            log_dir:

        """
        super(MultiProcessTensorBoardWriter, self).__init__()
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        self._log_dir = log_dir
        self._writer = None

    # ==================================================================================================
    #
    #
    #   Main Process
    #
    #
    # ==================================================================================================
    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def on_start(self):
        self._writer = TensorBoardWriter(log_dir=self._log_dir)

    def on_exit(self):
        self._writer.close()
    #
    # def add_scalar(self, namespace, graph_name, scalar_value, step):
    #     """
    #
    #     Args:
    #         namespace (str):
    #         graph_name (str):
    #         scalar_value (float):
    #         step (int):
    #
    #     """
    #     self.add_task(Type.SCALAR, (namespace, graph_name, scalar_value, step))
    #
    # def add_scalars(self, namespace, graph_name, scalar_dict, step):
    #     """
    #
    #     Args:
    #         namespace (str):
    #         graph_name (str):
    #         scalar_dict (dict[str, torch.Tensor]):
    #         step (int):
    #
    #     """
    #     self.add_task(Type.SCALARS, (namespace, graph_name, scalar_dict, step))
    #
    # def add_image(self, namespace, name, image, step):
    #     """
    #
    #     Args:
    #         namespace (str):
    #         name (str):
    #         image (torch.Tensor):
    #         step (int):
    #
    #     """
    #     self.add_task(Type.IMAGE, (namespace, name, image, step))
    #
    # def add_images(self, namespace, image_dict, step):
    #     """
    #
    #     Args:
    #         namespace (str):
    #         image_dict (dict[str, torch.Tensor]):
    #         step (int):
    #
    #     """
    #     for name, image in image_dict.items():
    #         self.add_task(Type.IMAGES, (namespace, name, image, step))

    #
    # def add_image_with_process(self, func, args, namespace, step):
    #     self.add_task(Type.IMAGE_WITH_FUNCTION, ((func, args), (namespace, step)))
    #
    # def add_images_with_process(self, func, args, namespace, step):
    #     self.add_task(Type.IMAGES_WITH_FUNCTION, ((func, args), (namespace, step)))

    # ==================================================================================================
    #
    #
    #   Sub Process
    #
    #
    # ==================================================================================================
    # ==================================================================================================
    #
    #   Override
    #
    # ==================================================================================================
    def process(self, *task):
        self._write(*task)

    # ==================================================================================================
    #
    #   Instance Method (private)
    #
    # ==================================================================================================
    def _write(self, data_type, args):
        if data_type is Type.SCALAR:
            self._writer.add_scalar(*args)

        elif data_type is Type.SCALARS:
            self._writer.add_scalars(*args)

        elif data_type is Type.IMAGE:
            self._writer.add_image(*args)

        elif data_type is Type.IMAGES:
            self._writer.add_images(*args)

        elif data_type is Type.IMAGE_WITH_FUNCTION:
            (func, func_args), (namespace, name, step) = args
            image = func(*func_args)
            self._writer.add_image(namespace, name, image, step)

        elif data_type is Type.IMAGES_WITH_FUNCTION:
            (func, func_args), (namespace, step) = args
            images = func(*func_args)
            self._writer.add_images(namespace, images, step)

        else:
            raise TypeError(data_type)

        self._writer.flush()
