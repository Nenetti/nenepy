from torch import nn
from torch.functional import F


class ReconstructionError(nn.BCELoss):

    def forward(self, input, target):
        """

        Args:
            input (torch.Tensor):
            target (torch.Tensor):

        Returns:

        """
        if target.grad is not None:
            raise ValueError("Target tensor is not allowed to have a gradient.")

        return super(ReconstructionError, self).forward(input, target)
