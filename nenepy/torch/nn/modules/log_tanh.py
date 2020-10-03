import torch
from torch import nn
from torch.distributions import RelaxedOneHotCategorical
from torch.functional import F


class LogTanh(nn.Module):

    def __init__(self):
        super(LogTanh, self).__init__()
        self.eps = torch.finfo(torch.float).eps

    def forward(self, x):
        return torch.log(((torch.tanh(x) + 1) / 2) + self.eps)
