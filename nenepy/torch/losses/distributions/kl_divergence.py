import torch
from nenepy.torch.losses.distributions import Loss
from torch.distributions import kl_divergence


class KullbackLeibler(Loss):

    def __init__(self, p, q):
        """

        Args:
            prior (torch.distributions.Distribution):

        """
        super(KullbackLeibler, self).__init__()
        self.p = p
        self.q = q

    def forward(self):
        return kl_divergence(self.p, self.q)
