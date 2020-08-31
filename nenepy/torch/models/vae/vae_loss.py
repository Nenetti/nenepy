from torch import nn
from torch.distributions import Distribution

from nenepy.torch.losses import KullbackLeibler, ReconstructionError


class VAELoss(nn.Module):

    def __init__(self, prior_distribution):
        """

        Args:
            prior_distribution (Distribution):

        """
        super(VAELoss, self).__init__()

        self.kl_divergence = KullbackLeibler(prior_distribution)
        self.reconst_loss_func = ReconstructionError()

    def forward(self, distribution, x, reconst_x):
        """

        Args:
            distribution (Distribution):
            x (torch.Tensor):
            reconst_x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        kl_loss = self.kl_divergence(distribution)
        reconstruction = self.reconst_loss_func(x, reconst_x)

        return kl_loss + reconstruction
