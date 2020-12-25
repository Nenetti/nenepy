import torch
import torch.nn as nn
from nenepy.torch.nn.modules.concat import Concat
from torch.functional import F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, output_stride=8):
        """

        Args:
            in_channels (int):
            output_stride (int):

        """
        super(ASPP, self).__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPConv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPConv2d(in_channels, out_channels, kernel_size=3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPConv2d(in_channels, out_channels, kernel_size=3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPConv2d(in_channels, out_channels, kernel_size=3, padding=dilations[3], dilation=dilations[3])
        self.aspp5 = ASPPPool2d(in_channels, out_channels)
        self.concat_out = ASPPConcatConv2d(in_channels=out_channels * 5, out_channels=out_channels)

        self.concat = Concat(dim=1)

        self._init_weight()

    def _init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): [B, C, H, W]

        Returns:
            torch.Tensor: [B, C, H, W]

        """
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp5(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode="bilinear", align_corners=True)

        x = self.concat(x1, x2, x3, x4, x5)
        return self.concat_out(x)


class ASPPConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        """

        Args:
            in_channels (int):
            out_channels (int):
            kernel_size (int):
            padding (int):
            dilation (int):

        """
        super(ASPPConv2d, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        return self.layers(x)


class ASPPPool2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        """

        Args:
            in_channels (int):
            out_channels (int):

        """
        super(ASPPPool2d, self).__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        return self.layers(x)


class ASPPConcatConv2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        """

        Args:
            in_channels (int):
            out_channels (int):

        """
        super(ASPPConcatConv2d, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        """

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        return self.layers(x)
