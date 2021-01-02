import time

from nenepy.torch.utils.tensorboard.tensorboard_writer import Type
from nenepy.utils.multi.multi_task_process import MultiTaskProcess
from nenepy.utils.multi.multi_task_process_manager import MultiTaskProcessManager


class MultiTaskTensorBoardWriter(MultiTaskProcessManager):

    def add_images_with_process(self, func, args):
        self._target_process.add_images_with_process(func, args)
        self._change_next_process()

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
            scalar_dict (dict[str, torch.Tensor]):
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