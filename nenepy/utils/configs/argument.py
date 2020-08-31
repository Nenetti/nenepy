import argparse
from abc import ABCMeta, abstractmethod


class AbstractArgument(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def add_argument(parser):
        """
        Examples:
            parser.add_argument('--cfg', dest='cfg_file', required=True, help='Config file for training (and optionally testing)')

        Args:
            parser:

        Returns:

        """
        raise NotImplementedError()

    @classmethod
    def get_args(cls):
        parser = argparse.ArgumentParser()
        parser = cls.add_argument(parser)

        return parser.parse_args()
