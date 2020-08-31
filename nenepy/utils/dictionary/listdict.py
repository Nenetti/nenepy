
class ListDict(dict):

    def __init__(self, init_size=-1):
        super(ListDict, self).__init__()
        self._init_size = init_size
        self._size_dict = {}

    # ==============================================================================
    #
    #   Public Method
    #
    # ==============================================================================

    def add_dict(self, d, excludes=[]):
        """

        Args:
            d (dict[str, Any]):
            excludes (str or list[str]):

        """
        if not isinstance(excludes, list):
            excludes = list(excludes)

        if self._init_size == -1:
            self._append_dict(d, excludes)
        else:
            self._set_dict(d, excludes)

    def _append_dict(self, d, excludes):
        """

        Args:
            d (dict[str, Any]):
            excludes (list[str]):

        """
        for key, value in d.items():
            if key in excludes:
                continue

            if key in self:
                self[key].append(value)
            else:
                self[key] = [value]

    def _set_dict(self, d, excludes):
        """

        Args:
            d (dict[str, Any]):
            excludes (list[str]):

        """
        for key, value in d.items():
            if key in excludes:
                continue

            if key in self:
                index = self._size_dict[key]
            else:
                self[key] = [None] * self._init_size
                index = 0

            self[key][index] = value
            self._size_dict[key] = index + 1
