import math

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class ExponentialCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):

    def __init__(self, optimizer,
                 gamma=0.9, exponential_min=1e-4,
                 annealing_cycle=100, annealing_cycle_multi=1, annealing_min=1e-5,
                 last_epoch=-1):
        """

        Args:
            optimizer (torch.optim.Optimizer):
            gamma (float):
            exponential_min (float):
            annealing_cycle (int):
            annealing_cycle_multi (int):
            annealing_min (float):
            last_epoch (int):

        """
        super(ExponentialCosineAnnealingWarmRestarts, self).__init__(optimizer, annealing_cycle, annealing_cycle_multi, annealing_min, last_epoch)
        self.gamma = gamma
        self.exponential_min_limit = exponential_min

    # ==================================================================================================
    #
    #   Override Method
    #
    # ==================================================================================================

    def get_lr(self):
        return [self._cosine_annealing_warm_restart(self._exponential(lr)) for lr in self.base_lrs]

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================

    def _exponential(self, base_lr):
        c = self.gamma ** self.last_epoch
        limit = self.exponential_min_limit * base_lr
        if base_lr * c > limit:
            return base_lr * c

        return limit

    def _cosine_annealing_warm_restart(self, lr):
        eta_min = self.eta_min * lr
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
