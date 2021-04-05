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

        if size is not None:
            if size == tuple(x.shape[-2:]):
                return x
            else:
                return F.interpolate(x, size, None, self.mode, self.align_corners)

        elif scale_factor is not None:
            if scale_factor == 1:
                return x
            else:
                return F.interpolate(x, None, scale_factor, self.mode, self.align_corners)

        else:
            return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)
