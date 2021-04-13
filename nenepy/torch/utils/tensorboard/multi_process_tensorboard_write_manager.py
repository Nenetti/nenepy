from nenepy.torch.utils.tensorboard import Type
from nenepy.utils.multi.multi_process_manager import MultiProcessManager


class MultiProcessTensorBoardWriteManager(MultiProcessManager):

    def add_scalar(self, namespace, graph_name, scalar_value, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_value (float):
            step (int):

        """
        self.set_task(process_id=0, args=(Type.SCALAR, (namespace, graph_name, scalar_value, step)))

    def add_scalars(self, namespace, graph_name, scalar_dict, step):
        """

        Args:
            namespace (str):
            graph_name (str):
            scalar_dict (dict[str, torch.Tensor]):
            step (int):

        """
        self.set_task(process_id=0, args=(Type.SCALARS, (namespace, graph_name, scalar_dict, step)))

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

    def add_image_with_process(self, func, args, namespace, name, step):
        self.add_task(Type.IMAGE_WITH_FUNCTION, ((func, args), (namespace, name, step)))

    def add_images_with_process(self, func, args, namespace, step):
        self.add_task(Type.IMAGES_WITH_FUNCTION, ((func, args), (namespace, step)))
