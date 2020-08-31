import torch
from torch import nn


class Concat(nn.Module):

    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *tensors):
        return torch.cat(tensors, dim=self.dim)
