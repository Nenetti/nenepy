from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset


class AbstractDataset(Dataset, metaclass=ABCMeta):

    # ==================================================================================================
    #
    #   Special Attribute
    #
    # ==================================================================================================
    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()
