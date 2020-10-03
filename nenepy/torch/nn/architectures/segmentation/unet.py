import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# import torch
# from torch import nn
#
# from nenepy.torch.nn.modules.concat import Concat
#
#
# class UNet(nn.Module):
#
#     def __init__(self, in_channels, out_channels=None, base_channels=64):
#         super().__init__()
#
#         if out_channels is None:
#             out_channels = in_channels
#
#         self.encode_block1 = self.DownBlock(in_channels=in_channels, out_channels=base_channels, pooling=False)
#         self.encode_block2 = self.DownBlock(in_channels=base_channels, out_channels=base_channels * 2, pooling=True)
#         self.encode_block3 = self.DownBlock(in_channels=base_channels * 2, out_channels=base_channels * 4, pooling=True)
#         self.encode_block4 = self.DownBlock(in_channels=base_channels * 4, out_channels=base_channels * 8, pooling=True)
#         self.encode_block5 = self.DownBlock(in_channels=base_channels * 8, out_channels=base_channels * 16, pooling=True)
#
#         self.decode_block4 = self.UpBlock(in_channels=base_channels * 16, out_channels=base_channels * 8)
#         self.decode_block3 = self.UpBlock(in_channels=base_channels * 8, out_channels=base_channels * 4)
#         self.decode_block2 = self.UpBlock(in_channels=base_channels * 4, out_channels=base_channels * 2)
#         self.decode_block1 = self.UpBlock(in_channels=base_channels * 2, out_channels=base_channels)
#         self.out_block = self.OutBlock(in_channels=base_channels, out_channels=out_channels, up_sampling=False)
#
#     def forward(self, x):
#         x1 = self.encode_block1(x)
#         x2 = self.encode_block2(x1)
#         x3 = self.encode_block3(x2)
#         x4 = self.encode_block4(x3)
#         x5 = self.encode_block5(x4)
#
#         y4 = self.decode_block4(x5, x4)
#         y3 = self.decode_block3(y4, x3)
#         y2 = self.decode_block2(y3, x2)
#         y1 = self.decode_block1(y2, x1)
#         y = self.out_block(y1)
#
#         return y
#
#     class ConvBlock(nn.Module):
#
#         def __init__(self, in_channels, out_channels):
#             """
#             Args:
#                 in_channels (int):
#                 out_channels (int):
#             """
#             super(UNet.ConvBlock, self).__init__()
#
#             self.block = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
#                 nn.InstanceNorm2d(num_features=out_channels),
#                 nn.ReLU(inplace=True),
#
#                 nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
#                 nn.InstanceNorm2d(num_features=out_channels),
#                 nn.ReLU(inplace=True)
#             )
#
#         def forward(self, x):
#             x = self.block(x)
#             return x
#
#     class DownBlock(nn.Module):
#
#         def __init__(self, in_channels, out_channels, pooling):
#             """
#             Args:
#                 in_channels (int):
#                 out_channels (int):
#             """
#             super(UNet.DownBlock, self).__init__()
#
#             if pooling:
#                 self.block = nn.Sequential(
#                     nn.MaxPool2d(kernel_size=2),
#                     UNet.ConvBlock(in_channels, out_channels)
#                 )
#             else:
#                 self.block = nn.Sequential(
#                     UNet.ConvBlock(in_channels, out_channels)
#                 )
#
#         def forward(self, x):
#             x = self.block(x)
#             return x
#
#     class UpBlock(nn.Module):
#
#         def __init__(self, in_channels, out_channels):
#             """
#             Args:
#                 in_channels (int):
#                 out_channels (int):
#             """
#             super(UNet.UpBlock, self).__init__()
#             self.up_block = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
#             self.block = UNet.ConvBlock(in_channels, out_channels)
#             self.concat = Concat(dim=1)
#
#         def forward(self, x, y):
#             x = self.up_block(x)
#             x = self.concat(x, y)
#             x = self.block(x)
#             return x
#
#     class OutBlock(nn.Module):
#
#         def __init__(self, in_channels, out_channels, up_sampling):
#             """
#             Args:
#                 in_channels (int):
#                 out_channels (int):
#             """
#             super(UNet.OutBlock, self).__init__()
#             self.block = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
#
#         def forward(self, x):
#             x = self.block(x)
#             return x
