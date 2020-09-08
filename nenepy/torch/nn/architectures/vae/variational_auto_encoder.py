import torch
from torch import nn
from torch.distributions import Normal


class VariationalAutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.sampling(mu, sigma)
        decoder_out = self.decoder(z)

        return mu, sigma, z, decoder_out

    def sampling(self, mu, sigma):
        if self.training:
            return Normal(mu, sigma).rsample()
        else:
            return mu
