from nenepy.utils.dictionary import AttrDict


class ListDict(AttrDict):
    EXCLUDES = ("_init_size", "_size_dict")

    def __init__(self, init_size=-1, keys=[]):
        super(ListDict, self).__init__()
        self._init_size = init_size
        self._size_dict = {}
        self.items()
        if len(keys) > 0:
            for key in keys:
                self[key] = [None] * self._init_size
                self._size_dict[key] = 0

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
        if self._init_size == -1:
            self._add_dict(d)
        else:
            self._set_dict(d)

    def add_value(self, key, value):
        """

        Args:
            key:
            value:

        Returns:

        """
        if self._init_size == -1:
            self._add_value(key, value)
        else:
            self._set_value(key, value)

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================
    def _add_value(self, key, value):
        self[key].append(value)

    def _set_value(self, key, value):
        index = self._size_dict[key]
        self[key][index] = value
        self._size_dict[key] = index + 1

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
