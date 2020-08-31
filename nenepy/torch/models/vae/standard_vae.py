import torch
from torch import nn


class StandardVAE(nn.Module):

    def __init__(self, encoder, decoder):
        super(StandardVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.sampling(mu, sigma)
        decoder_out = self.decoder(z)

        return mu, sigma, z, decoder_out

    @staticmethod
    def sampling(mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + (eps * std)
