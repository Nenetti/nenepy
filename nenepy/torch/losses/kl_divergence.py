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

    def forward(self, x):
        """

        Args:
            x (Distribution):

        Returns:

        """
        return kl_divergence(self.prior, x)
