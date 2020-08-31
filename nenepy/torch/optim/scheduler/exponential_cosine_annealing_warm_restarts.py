import math

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class ExponentialCosineAnnealingWarmRestarts(CosineAnnealingWarmRestarts):

    def __init__(self, optimizer, gamma, exponential_min_limit, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        super(ExponentialCosineAnnealingWarmRestarts, self).__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        self.gamma = gamma
        self.exponential_min_limit = exponential_min_limit

    def exponential(self, base_lr):
        c = self.gamma ** self.last_epoch
        if base_lr * c > self.exponential_min_limit:
            return base_lr * c

        return self.exponential_min_limit

    def cosine_annealing_warm_restart(self, lr):
        return self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2

    def get_lr(self):
        return [self.cosine_annealing_warm_restart(self.exponential(lr)) for lr in self.base_lrs]
