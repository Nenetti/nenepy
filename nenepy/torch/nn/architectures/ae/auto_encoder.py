import torch
from torch import nn
from torch.distributions import Normal


class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, z_dim):
        """

        Args:
            encoder (nn.Module):
            decoder (nn.Module):
            z_dim (int):

        """
        super(AutoEncoder, self).__init__()

        self._encoder = encoder
        self._decoder = decoder
        self._z_dim = z_dim

    @property
    def z_dim(self):
        return self._z_dim

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:
            torch.Tensor:

        Shapes:
            [-1, C, H, W]
            ->
            [-1, z_dim]
            [-1, C, H, W]

        """
        z = self._encoder(x)
        decoder_out = self._decoder(z)

        return z, decoder_out
