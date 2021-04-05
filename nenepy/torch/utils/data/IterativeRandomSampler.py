from torch.utils.data import RandomSampler


class IterativeRandomSampler(RandomSampler):

    def __init__(self, data_source, replacement=False, num_samples=None, generator=None, shuffle_iteration=0):
        super(IterativeRandomSampler, self).__init__(data_source, replacement, num_samples, generator)

        if shuffle_iteration < 0:
            raise ValueError()

        self._shuffle_iteration = shuffle_iteration
        self.counter = 0

    def __iter__(self):
        if self.counter == self._shuffle_iteration:
            self.counter = 0
            return super(IterativeRandomSampler, self).__iter__()
        else:
            self.counter += 1
            return iter(range(len(self.data_source)))
