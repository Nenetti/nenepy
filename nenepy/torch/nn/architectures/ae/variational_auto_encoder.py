import torch
from torch import nn
from torch.distributions import Normal, Bernoulli
from nenepy.torch.losses.distributions import Log, E, KullbackLeibler


class VariationalAutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, z_dim):
        """

        Args:
            encoder (nn.Module):
            decoder (nn.Module):
            z_dim (int):

        """
        super(VariationalAutoEncoder, self).__init__()

        self._encoder = encoder
        self._decoder = decoder
        self._z_dim = z_dim

    @property
    def z_dim(self):
        return self._z_dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
            torch.Tensor:
            torch.Tensor:
            torch.Tensor:

        Shapes:
            [-1, C, H, W]
            ->
            [-1, N]
            [-1, N]
            [-1, N]
            [-1, C, H, W]

        """
        mu, sigma = self._encoder(x)
        z = self.sampling(mu, sigma)
        decoder_out = self._decoder(z)

        return mu, sigma, z, decoder_out

    def sampling(self, mu, sigma):
        if self.training:
            return Normal(mu, sigma).rsample()
        else:
            return mu


class Loss(nn.Module):

    def __init__(self, prior):
        """

        Args:
            prior (torch.distributions.Distribution):

        """
        super(Loss, self).__init__()
        self.prior = prior

    def forward(self, x, reconst_x, mu, sigma):
        """

        Args:
            x (torch.Tensor):
            reconst_x (torch.Tensor):
            mu (torch.Tensor):
            sigma (torch.Tensor):

        Returns:
             torch.Tensor:

        Shapes:
            [-1, C, H, W]
            [-1, C, H, W]
            [-1, N]
            [-1, N]
            ->
            [-1]

        """
        return self._kl_divergence(self.prior, Normal(mu, sigma)) + self._reconstruction(x, reconst_x)

    @staticmethod
    def _reconstruction(x, reconst_x):
        """

        Args:
            x (torch.Tensor):
            reconst_x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        p = Bernoulli(reconst_x)
        q = Bernoulli(x)
        loss = -E(p=Log(p), q=q)()

        dims = list(range(1, loss.dim()))
        return loss.sum(dim=dims)

    @staticmethod
    def _kl_divergence(prior, posterior):
        """

        Args:
            prior (torch.distributions.Distribution):
            posterior (torch.distributions.Distribution):

        Returns:
            torch.Tensor:

        """
        loss = KullbackLeibler(prior, posterior)()

        dims = list(range(1, loss.dim()))
        return loss.sum(dim=dims)
