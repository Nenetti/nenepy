import torch
from nenepy.torch.losses.distributions import Loss


class Log(Loss):

    def __init__(self, p):
        """
        Args:
            p (torch.distributions.Distribution):

        """
        super().__init__(p)

    def forward(self, x):
        return torch.log(self.p(x))
