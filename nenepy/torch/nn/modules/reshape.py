from torch import nn


class Reshape(nn.Module):

    def __init__(self, size=None):
        super(Reshape, self).__init__()
        self._size = size

    def forward(self, x, size=None):
        """

        Args:
            x (torch.Tensor):
            size:

        Returns:

        """
        if size is None:
            size = self._size

        return x.contiguous().view(size)
