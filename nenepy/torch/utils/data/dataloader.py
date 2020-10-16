from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from torch.utils.data import SequentialSampler
from tqdm import tqdm

from nenepy.torch.interfaces import Mode
from nenepy.torch.utils.data import DesignatedIterativeBatchSampler
from nenepy.torch.utils.data.IterativeRandomSampler import IterativeRandomSampler


class DataLoader(TorchDataLoader):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_workers=0,
                 pin_memory=False,
                 break_iteration=-1,
                 shuffle=False,
                 shuffle_iteration=0,
                 drop_last=False,
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
        n_data = len(dataset)
        # if (n_data % batch_size == 1) and (not drop_last):
        #     raise ValueError()

        if break_iteration == -1:
            kwargs.update({"batch_size": batch_size, "shuffle": shuffle, "drop_last": drop_last})

        else:
            if shuffle:
                sampler = IterativeRandomSampler(dataset, shuffle_iteration=shuffle_iteration)
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
