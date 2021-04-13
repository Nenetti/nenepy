import numpy as np

from nenepy.utils.dictionary import AttrDict


class ListDict(AttrDict):
    EXCLUDES = ("_init_size", "_is_init_size", "_size_dict")

    def __init__(self, *args, **kwargs):
        super(ListDict, self).__init__(*args, **kwargs)

    @classmethod
    def init_size(cls, init_size=-1, keys=[]):
        instance = cls()
        instance._init_size = init_size
        instance._is_init_size = (init_size != -1)
        instance._size_dict = {}
        if len(keys) > 0:
            for key in keys:
                instance[key] = [None] * instance._init_size
                instance._size_dict[key] = 0

        return instance

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================
    def add_dict(self, d):
        """

        Args:
            d (dict[str, Any]):

        """
        if self._is_init_size:
            for key, value in d.items():
                index = self._size_dict[key]
                self[key][index] = value
                self._size_dict[key] = index + 1
        else:
            for key, value in d.items():
                self[key].append(value)

    def add_value(self, key, value):
        """

        Args:
            key:
            value:

        Returns:

        """
        if self._is_init_size:
            index = self._size_dict[key]
            self[key][index] = value
            self._size_dict[key] = index + 1
        else:
            self[key].append(value)

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================
    def _add_dict(self, d):
        """

        Args:
            d (dict[str, Any]):

        """
        for key, value in d.items():
            if key in self:
                self[key].append(value)
            else:
                self[key] = [value]

    def _set_dict(self, d):
        """

        Args:
            d (dict[str, Any]):

        """
        for key, value in d.items():
            if key in self:
                index = self._size_dict[key]
            else:
                self[key] = [None] * self._init_size
                index = 0

            self[key][index] = value
            self._size_dict[key] = index + 1

    def keys(self):
        return tuple(key for key in super(ListDict, self).keys() if key not in self.EXCLUDES)

    def values(self):
        return tuple(value for key, value in super(ListDict, self).items() if key not in self.EXCLUDES)

    def items(self):
        return dict(((key, value) for key, value in super(ListDict, self).items() if key not in self.EXCLUDES)).items()

    def mean(self, axis=0):
        return ListDict([(key, np.array(value).mean(axis)) for key, value in self.items()])

    def concatenate(self, axis=0):
        return ListDict([(key, np.concatenate(value, axis)) for key, value in self.items()])

    def transpose(self, axes=()):
        return ListDict([(key, np.transpose(value, axes)) for key, value in self.items()])
