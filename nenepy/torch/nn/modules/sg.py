import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticGate(nn.Module):
    """
    深層特徴量と浅層特徴量を一定比率で混ぜる
    """

    def __init__(self, p):
        super(StochasticGate, self).__init__()
        self.p = p

    def forward(self, x1, x2):
        """

        Args:
            x1 (torch.Tensor):   Shallow Feature
            x2 (torch.Tensor):      Deep Feature
            p (float):                  混合率 (大きいほど浅層を考慮，小さいほど深層を考慮)

        Returns:
            torch.Tensor:

        """
        self.p = 0
        if self.training:
            r = (1 - self.p) * F.dropout(torch.ones_like(x1), p=self.p)
            delta = 1 / (1 - self.p)
            x = (r * (x1 - (self.p * x2)) * delta) + ((1 - r) * x2)

        else:
            x = ((1 - self.p) * x1) + (self.p * x2)

        return x

        # if self.training:
        #     r = 1 - (1 - p) * F.dropout(torch.ones_like(x1), p=p)
        #     delta = 1 / (1 - p)
        #     x = (1 - r) * delta * (x2 - p * x1) + r * x1
        #
        # else:
        #     x = ((1 - p) * x2) + (p * x1)
        #
        # return x
