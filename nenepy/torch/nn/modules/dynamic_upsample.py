from typing import Iterable

from torch import nn
from torch.functional import F


class Upsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

        if (size is not None) and (scale_factor is not None):
            raise ValueError()

        if scale_factor is not None:
            if isinstance(scale_factor, Iterable):
                self.scale_factor = tuple(map(float, scale_factor))
            else:
                self.scale_factor = float(scale_factor)

    def forward(self, x, size=None, scale_factor=None):
        """

        Args:
            x (torch.Tensor):
            size (tuple[int, int] or None):
            scale_factor (tuple[int, int] or None):

        Returns:

        """

        if (size is None) and (scale_factor is None):
            size = self.size
            scale_factor = self.scale_factor
            return F.interpolate(x, size, scale_factor, self.mode, self.align_corners)

        else:
            if ((size is None) and (scale_factor is not None)) or (size is not None) and (scale_factor is None):
                return F.interpolate(x, size, scale_factor, self.mode, self.align_corners)
            else:
                raise ValueError
