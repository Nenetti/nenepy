import torch
from torch import nn
from torch.distributions import kl_divergence, Distribution


class KullbackLeibler(nn.Module):

    def __init__(self, prior):
        """

        Args:
            prior (Distribution):

        """
        super(KullbackLeibler, self).__init__()
        self.prior = prior

    def forward(self, mu, sigma):
        """

        Args:
            x (Distribution):

        Returns:

        """
        return -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        # return kl_divergence(self.prior, Normal(mu, sigma))
