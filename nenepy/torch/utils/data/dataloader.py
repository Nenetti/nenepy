from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm

from nenepy.torch.interfaces.mode import Mode
from nenepy.torch.utils.data import DesignatedIterativeBatchSampler


class DataLoader(TorchDataLoader):

    def __init__(self,
                 dataset,
                 mode,
                 batch_size,
                 num_workers=0,
                 pin_memory=True,
                 break_iteration=-1,
                 **kwargs
                 ):
        """

        Args:
            dataset (Dataset):
            mode (Mode):
            batch_size (int):
            n_workers (int):
            pin_memory (bool):
            break_iteration (int):

        """
        shuffle, drop_last = (True, True)

        if mode is Mode.VALIDATE:
            shuffle, drop_last = (False, False)

        if break_iteration == -1:
            kwargs.update({"batch_size": batch_size, "shuffle": shuffle, "drop_last": drop_last})

        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

            kwargs.update({"batch_sampler": DesignatedIterativeBatchSampler(sampler, batch_size, drop_last, break_iteration)})

        super(DataLoader, self).__init__(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )

    def tqdm(self):
        return tqdm(self, leave=False, ascii=True)
