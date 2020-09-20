import torch
from torch import nn
from torch.distributions import RelaxedOneHotCategorical
from torch.functional import F


class GumbelSoftmax(nn.Module):

    def __init__(self, tau=1, hard=False, eps=1e-10, dim=-1):
        super(GumbelSoftmax, self).__init__()
        self._tau = tau
        self._hard = hard
        self._eps = eps
        self._dim = dim

    def forward(self, logits):
        """

        Args:
            logits (torch.Tensor):

        Returns:

        """
        if self._hard:
            dim = self._dim
            n_dims = logits.dim()

            dims = [*range(0, dim), *range(dim + 1, n_dims), dim]
            reconst_dims = [*range(0, dim), (n_dims - 1), *range(dim, (n_dims - 1))]

            probs = logits.contiguous().permute(dims=dims)
            sample = RelaxedOneHotCategorical(temperature=self._tau, probs=probs).rsample()
            return sample.contiguous().permute(reconst_dims).contiguous()
        else:
            return F.gumbel_softmax(torch.log(logits + self._eps), tau=self._tau, hard=self._hard, dim=self._dim)
