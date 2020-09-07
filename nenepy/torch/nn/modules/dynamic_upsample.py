from torch import nn
from torch.functional import F


class DynamicUpsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(DynamicUpsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, size=None, scale_factor=None):
        height, width = x.shape[-2:]

        if size is not None:
            scale_factor = None

        elif scale_factor is not None:
            size = None

        else:
            size = self.size
            scale_factor = self.scale_factor
        if (size is not None) and ((height == size[0]) and (width == size[1])):
            return x

        return F.interpolate(x, size, scale_factor, self.mode, self.align_corners)
