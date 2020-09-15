from torch import nn


class Expand(nn.Module):

    def __init__(self):
        super(Expand, self).__init__()

    def forward(self, x, size):
        """

        Args:
            x (torch.Tensor):
            size:

        Returns:

        """
        return x.expand(size)
