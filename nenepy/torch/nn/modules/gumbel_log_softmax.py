import torch
from torch import nn
from torch.distributions import RelaxedOneHotCategorical
from torch.functional import F


class GumbelLogSoftmax(nn.Module):

    def __init__(self, tau=1, hard=False, noise=True, eps=1e-10, dim=-1):
        super(GumbelLogSoftmax, self).__init__()
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
            # noise = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            noise = torch.rand_like(logits)
            x = torch.log_softmax(((logits + noise) / self._tau), dim=self._dim)
        else:
            x = torch.log_softmax((logits / self._tau), dim=self._dim)

        if self._hard:
            index = x.max(self._dim, keepdim=True)[1]
            hard = torch.log(torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format) + 1e-20).scatter_(self._dim, index, 0.0)
            x = hard - x.detach() + x

        return x
