from torch import nn
from torch.functional import F


class GumbelSoftmax(nn.Module):

    def __init__(self, tau=1, hard=False, eps=1e-10, dim=-1):
        super(GumbelSoftmax, self).__init__()
        self._tau = tau
        self._hard = hard
        self._eps = eps
        self._dim = dim

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self._tau, hard=self._hard, dim=self._dim)
