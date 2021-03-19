class AbstractConfig:
    __IMMUTABLE__ = "__immutable__"

    def __init__(self, **kwagrs):
        self.__setattr__(self.__IMMUTABLE__, False)
        for key, value in kwagrs.items():
            if key in self.__dict__:
                self.__setattr__(key, value)
            else:
                raise KeyError(f"'{key}' does not contain {self.__dict__}")

    # ==================================================================================================
    #
    #   Property
    #
    # ==================================================================================================
    @property
    def is_immutable(self):
        return self.__getattribute__(self.__IMMUTABLE__)

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def to_immutable(self):
        self.__setattr__(self.__IMMUTABLE__, True)

    # ==================================================================================================
    #
    #   Special Method
    #
    # ==================================================================================================
    def __setattr__(self, key, value):
        if self.__IMMUTABLE__ not in self.__dict__:
            super(AbstractConfig, self).__setattr__(key, value)
            return

        if not self.is_immutable:
            super(AbstractConfig, self).__setattr__(key, value)
        else:
            raise AttributeError(f"{self.__class__.__name__} is immutable")
