import collections
from pathlib import Path

from natsort import natsorted
from tensorboard.backend.event_processing import tag_types
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nenepy.torch.utils.tensorboard.modules.image import Image
from nenepy.torch.utils.tensorboard.modules.scalar import Scalar
from nenepy.torch.utils.tensorboard.tensorboard import TensorBoard

DEFAULT_SIZE_GUIDANCE = {
    tag_types.AUDIO: 0,
    tag_types.COMPRESSED_HISTOGRAMS: 0,
    tag_types.HISTOGRAMS: 0,
    tag_types.IMAGES: 0,
    tag_types.SCALARS: 0,
    tag_types.TENSORS: 0,
}


class TensorBoardLoader(TensorBoard):

    def __init__(self, log_dir, file_size_limit=10 ** 10):
        """

        Args:
            log_dir (str):

        """
        super(TensorBoardLoader, self).__init__()
        self._log_dir = Path(log_dir)
        self._file_size_limit = file_size_limit
        self._value_dict = {}
        self._scalar = {}
        self._scalars = {}
        self._images = {}

        self._load_files()

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def load_scalar(self, namespace, graph_name):
        key = (namespace, graph_name)
        if key not in self._scalar:
            raise KeyError(key)

        values = self._scalar[key]
        return [Scalar(value) for value in values]

    def load_scalars(self, namespace, graph_name, scalar_tag):
        key = (namespace, graph_name, scalar_tag)
        if key not in self._scalars:
            raise KeyError(key)

        values = self._scalars[key]
        return [Scalar(value) for value in values]

    def load_images(self, namespace, image_name, step=None):
        key = (namespace, image_name)
        if key not in self._images:
            raise KeyError(key)

        values = self._images[key]
        if step is not None:
            return [Image(value) for value in values if value.step == step]
        else:
            return [Image(value) for value in values]

    def get_scalar_namespaces(self):
        namespaces = [key[0] for key in self._scalar.keys()]
        return natsorted(set(namespaces))

    def get_scalar_graph_names(self, namespace):
        graph_names = [key[1] for key in self._scalar.keys() if key[0] == namespace]
        return natsorted(graph_names)

    def get_scalars_namespaces(self):
        namespaces = [key[0] for key in self._scalars.keys()]
        return natsorted(set(namespaces))

    def get_scalars_graph_names(self, namespace):
        graph_names = [key[1] for key in self._scalars.keys() if key[0] == namespace]
        return natsorted(graph_names)

    def get_scalars_tags(self, namespace, graph_names):
        graph_names = [key[2] for key in self._scalars.keys() if ((key[0] == namespace) and (key[1] == graph_names))]
        return natsorted(graph_names)

    def get_images_namespaces(self):
        namespaces = [key[0] for key in self._images.keys()]
        return natsorted(set(namespaces))

    def get_images_names(self, namespace):
        image_names = [key[1] for key in self._images.keys() if key[0] == namespace]
        return natsorted(image_names)

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    def _load_files(self):
        for path in self._log_dir.iterdir():
            if path.stat().st_size > self._file_size_limit:
                continue

            if path.is_dir():
                for file in path.iterdir():
                    self._load_file(path.name, file)
            elif path.is_file():
                self._load_file(None, path)
            else:
                raise TypeError(path)

    def _load_file(self, tag, log_file):
        event_acc = EventAccumulator(str(log_file), size_guidance=DEFAULT_SIZE_GUIDANCE, purge_orphaned_data=False)
        event_acc.Reload()
        if len(event_acc.scalars.Keys()) > 0:
            if tag is None:
                scalar = self._load_scalar(event_acc)
                self._scalar = self._merge_dict(self._scalar, scalar)

            else:
                scalars = self._load_scalars(event_acc, tag)
                self._scalars = self._merge_dict(self._scalars, scalars)
        elif len(event_acc.images.Keys()) > 0:
            images = self._load_image(event_acc)
            self._images = self._merge_dict(self._images, images)

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================
    @classmethod
    def _load_scalar(cls, event_acc):
        d = {}
        for key in event_acc.scalars.Keys():
            values = cls._step_sort(event_acc.Scalars(key))
            namespace, graph_name = cls._decompose_scalar_tag(key)
            d[(namespace, graph_name)] = values
        return d

    @classmethod
    def _load_scalars(cls, event_acc, dir_name):
        d = {}
        for key in event_acc.scalars.Keys():
            values = cls._step_sort(event_acc.Scalars(key))
            namespace, graph_name, scalar_tag = cls._decompose_scalars_dir_name(dir_name)
            d[(namespace, graph_name, scalar_tag)] = values
        return d

    @classmethod
    def _load_image(cls, event_acc):
        d = {}
        for key in event_acc.images.Keys():
            values = cls._step_sort(event_acc.Images(key))
            namespace, image_name = cls._decompose_image_tag(key)
            d[(namespace, image_name)] = values
            # d[(namespace, image_name)] = [Image(v) for v in values]
        return d

    @staticmethod
    def _step_sort(values):
        return sorted(values, key=lambda x: x.step)

    @classmethod
    def _merge_dict(cls, a, b):
        """
        Args:
            a (dict):
            b (dict):

        Returns:
            dict

        """
        keys = list(a.keys()) + list(b.keys())
        duplication_keys = [k for k, v in collections.Counter(keys).items() if v > 1]

        d = {}
        d.update(a)
        d.update(b)

        if len(duplication_keys) > 0:
            Warning(f"Duplicate Keys: {duplication_keys}")
            for key in duplication_keys:
                d[key] = cls._step_sort(a[key] + b[key])

        return d
