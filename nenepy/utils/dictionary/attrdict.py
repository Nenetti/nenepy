import numbers


class AttrDict(dict):
    """
    Attributes:
        IMMUTABLE (str):

    Examples:
        >>> d = AttrDict()
        >>> d.a = 1
        >>> d["b"] = 2

        >>> print(d)
        {'a': 1, 'b': 2}

    """

    IMMUTABLE = "__immutable__"

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False
        self._recursive_conversion(self)

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    @property
    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]

    def immutable(self):
        self.__dict__[AttrDict.IMMUTABLE] = True

        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable()
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable()

    def merge(self, other):
        """
        Args:
            other (AttrDict):

        """
        self._recursive_merge(self, other)

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================

    @classmethod
    def _recursive_conversion(cls, d):
        """
        AttrDict内のdictをすべてAttrDictに変換

        Args:
            d (AttrDict):

        """
        for k, v in d.items():
            if isinstance(v, dict):
                v = AttrDict(v)
                cls._recursive_conversion(v)

            d.__setattr__(k, v)

    @classmethod
    def _recursive_merge(cls, main, other):
        """
        Args:
            main (AttrDict):
            other (AttrDict):

        """

        for key, value in other.items():
            if isinstance(value, AttrDict):
                if key not in main:
                    main[key] = AttrDict()
                cls._recursive_merge(main[key], value)
            else:
                main[key] = value

    @staticmethod
    def _to_output_format(d):
        """

        Args:
            d (AttrDict):

        """

        def recursive(sub_d):
            for key, value in sub_d.items():
                if isinstance(value, dict):
                    sub_d[key] = recursive(value)

                elif isinstance(value, list):
                    sub_d[key] = str(tuple(value))

                elif not isinstance(value, numbers.Number):
                    sub_d[key] = str(value)

            return dict(sub_d)

        d = recursive(d)

        return d

    # ==================================================================================================
    #
    #   Special Attribute
    #
    # ==================================================================================================

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                self[key] = value
        else:
            raise AttributeError(f"AttrDict is immutable")

    def __setitem__(self, key, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            super(AttrDict, self).__setitem__(key, value)
        else:
            raise AttributeError(f"AttrDict is immutable")
