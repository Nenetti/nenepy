from torch import nn
from torch.functional import F


class DynamicUpsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(DynamicUpsample, self).__init__()
        self.size = size

        if (size is not None) and (scale_factor is not None):
            raise ValueError()

        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None

        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, size=None, scale_factor=None):
        """

        Args:
            x (torch.Tensor):
            size (tuple[int, int] or None):
            scale_factor (tuple[int, int] or None):

        Returns:

        """
        height, width = x.shape[-2:]

        if (size is None) and (scale_factor is None):
            size = self.size
            scale_factor = self.scale_factor

        if scale_factor is not None:
            h, w = (float(height * scale_factor[0]), float(width * scale_factor[1]))
            if float.is_integer(h) and float.is_integer(w):
                size = (int(h), int(w))
            else:
                raise ValueError()

        if (height == size[0]) and (width == size[1]):
            return x

        return F.interpolate(x, size, None, self.mode, self.align_corners)
