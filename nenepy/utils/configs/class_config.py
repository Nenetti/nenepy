import inspect
import pprint
from importlib import import_module
from pathlib import Path

from nenepy.utils.configs import Config
from nenepy.utils.dictionary import AttrDict


class ClassConfig(Config):

    def __init__(self, cls, **kwargs):
        super(ClassConfig, self).__init__(**kwargs)
        self._cls = cls
        self._kwargs = AttrDict(kwargs)
        args = set(inspect.signature(self._cls.__init__).parameters.keys())
        for key in kwargs.keys():
            if key not in args:
                raise KeyError(f"'{key}' is not needed, {cls} has {args}")

        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def init(self, *args, **kwargs):
        args = list(args)
        unite_kwargs = {**kwargs, **self._kwargs}
        if len(unite_kwargs) != len(kwargs) + len(self._kwargs):
            raise KeyError(f"Some or one key is duplicated, {kwargs} vs {self._kwargs}")

        # for i, arg in enumerate(args):
        #     if isinstance(arg, ClassConfig):
        #         args[i] = arg.init()

        # for i, (key, value) in enumerate(unite_kwargs.items()):
        #     if isinstance(value, ClassConfig):
        #         unite_kwargs[key] = value.init()

        return self._cls(*args, **unite_kwargs)

    # ==================================================================================================
    #
    #   Property
    #
    # ==================================================================================================
    @property
    def parameters(self):
        return self._kwargs

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

    def load(self, py_file):
        if Path(py_file).suffix == ".py":
            module = import_module(py_file[:-3].replace("/", "."))
            self.__dict__.update(module.__dict__)
        else:
            raise ValueError(f"{py_file} is not '.py'")

    # ==================================================================================================
    #
    #   Special Method
    #
    # ==================================================================================================
    def __setattr__(self, key, value):
        if self.__IMMUTABLE__ not in self.__dict__:
            super(ClassConfig, self).__setattr__(key, value)
            return

        if not self.is_immutable:
            super(ClassConfig, self).__setattr__(key, value)
        else:
            raise AttributeError(f"{self.__class__.__name__} is immutable")
