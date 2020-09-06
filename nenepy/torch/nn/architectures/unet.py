import torch
from torch import nn


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels=None, base_channels=64):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.encode_block1 = self.DownBlock(in_channels=in_channels, out_channels=base_channels, pooling=False)
        self.encode_block2 = self.DownBlock(in_channels=base_channels, out_channels=base_channels * 2, pooling=True)
        self.encode_block3 = self.DownBlock(in_channels=base_channels * 2, out_channels=base_channels * 4, pooling=True)
        self.encode_block4 = self.DownBlock(in_channels=base_channels * 4, out_channels=base_channels * 8, pooling=True)
        self.encode_block5 = self.DownBlock(in_channels=base_channels * 8, out_channels=base_channels * 16, pooling=True)

        self.decode_block4 = self.UpBlock(in_channels=base_channels * 16, out_channels=base_channels * 8)
        self.decode_block3 = self.UpBlock(in_channels=base_channels * 8, out_channels=base_channels * 4)
        self.decode_block2 = self.UpBlock(in_channels=base_channels * 4, out_channels=base_channels * 2)
        self.decode_block1 = self.UpBlock(in_channels=base_channels * 2, out_channels=base_channels)
        self.out_block = self.OutBlock(in_channels=base_channels, out_channels=out_channels, up_sampling=False)

    def forward(self, x):
        x1 = self.encode_block1(x)
        x2 = self.encode_block2(x1)
        x3 = self.encode_block3(x2)
        x4 = self.encode_block4(x3)
        x5 = self.encode_block5(x4)

        y4 = self.decode_block4(x5, x4)
        y3 = self.decode_block3(y4, x3)
        y2 = self.decode_block2(y3, x2)
        y1 = self.decode_block1(y2, x1)
        y = self.out_block(y1)

        return y

    class ConvBlock(nn.Module):

        def __init__(self, in_channels, out_channels):
            """
            Args:
                in_channels (int):
                out_channels (int):
            """
            super(UNet.ConvBlock, self).__init__()

            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.block(x)
            return x

    class DownBlock(nn.Module):

        def __init__(self, in_channels, out_channels, pooling):
            """
            Args:
                in_channels (int):
                out_channels (int):
            """
            super(UNet.DownBlock, self).__init__()

            if pooling:
                self.block = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2),
                    UNet.ConvBlock(in_channels, out_channels)
                )
            else:
                self.block = nn.Sequential(
                    UNet.ConvBlock(in_channels, out_channels)
                )

        def forward(self, x):
            x = self.block(x)
            return x

    class UpBlock(nn.Module):

        def __init__(self, in_channels, out_channels):
            """
            Args:
                in_channels (int):
                out_channels (int):
            """
            super(UNet.UpBlock, self).__init__()
            self.up_block = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
            self.block = UNet.ConvBlock(in_channels, out_channels)

        def forward(self, x, y):
            x = self.up_block(x)
            x = torch.cat((x, y), dim=1)
            x = self.block(x)
            return x

    class OutBlock(nn.Module):

        def __init__(self, in_channels, out_channels, up_sampling):
            """
            Args:
                in_channels (int):
                out_channels (int):
            """
            super(UNet.OutBlock, self).__init__()
            self.block = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        def forward(self, x):
            x = self.block(x)
            return x
