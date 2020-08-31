from torch.utils.data import BatchSampler


class DesignatedIterativeBatchSampler(BatchSampler):

    def __init__(self, sampler, batch_size, drop_last, break_iteration):
        super(DesignatedIterativeBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self._break_iteration = break_iteration

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                i += 1
                yield batch
                batch = []
                if i == self._break_iteration:
                    break
        if len(batch) > 0 and not self.drop_last:
            print(batch)
            yield batch
