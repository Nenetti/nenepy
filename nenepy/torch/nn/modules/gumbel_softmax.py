import torch
from torch import nn
from torch.distributions import RelaxedOneHotCategorical
from torch.functional import F


class GumbelSoftmax(nn.Module):

    def __init__(self, tau=1, hard=False, noise=True, eps=1e-10, dim=-1):
        super(GumbelSoftmax, self).__init__()
        self._tau = tau
        self._hard = hard
        self._noise = noise
        self._eps = eps
        self._dim = dim

    def forward(self, logits):
        """

        Args:
            logits (torch.Tensor):

        """
        if self._noise:
            return F.gumbel_softmax(logits, tau=self._tau, hard=self._hard, dim=self._dim)
        else:
            x = torch.softmax((logits / self._tau), dim=self._dim)
            if self._hard:
                index = x.max(self._dim, keepdim=True)[1]
                hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(self._dim, index, 1.0)
                x = hard - x.detach() + x
            return x
