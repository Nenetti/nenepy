import itertools

import torch
from torch import nn

from nenepy.torch.nn.architectures.backbones import ResNeXt50_32x4d, ResNeXt101_32x8d, WideResNet50_2, WideResNet101_2
from nenepy.torch.nn.architectures.backbones import ResNet50, ResNet18, ResNet34, ResNet101, ResNet152
from nenepy.torch.nn.modules import Upsample, AtrousSpatialPyramidPooling

backbone_list = [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt50_32x4d, ResNeXt101_32x8d, WideResNet50_2, WideResNet101_2]


class DeepLabV3Plus(nn.Module):
    # ==================================================================================================
    #
    #   Initialization
    #
    # ==================================================================================================
    def __init__(self, in_channels, out_channels, backbone_cls=ResNet50, backbone_pretrained=True, backbone_kwargs={}):
        super(DeepLabV3Plus, self).__init__()
        self._init_encoder(in_channels, backbone_cls, backbone_pretrained, backbone_kwargs)
        self._init_decoder(out_channels)

    def _init_encoder(self, in_channels, backbone_cls, backbone_pretrained, backbone_kwargs):
        self._backbone = backbone_cls(in_channels=in_channels, reduction_rate=16, pretrained=backbone_pretrained, is_feature_extraction=True)
        self._aspp = AtrousSpatialPyramidPooling(in_channels=2048, out_channels=256, output_stride=8)

    def _init_decoder(self, out_channels):
        self._upsampling_x4 = Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self._skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self._last_conv = nn.Sequential(
            nn.Conv2d(in_channels=256 + 48, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1),
        )

    # ==================================================================================================
    #
    #   Property
    #
    # ==================================================================================================
    @property
    def backbone_modules(self):
        return self._backbone.modules()

    @property
    def not_backbone_modules(self):
        return itertools.chain(self._aspp.modules(), self._skip_conv.modules(), self._last_conv.modules())

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================
    def forward(self, x):
        # 1/2, 1/4, 1/8, 1/16, 1/16
        _, x2, _, _, x5 = self._backbone(x)
        x = self._aspp(x5)

        x = self._last_conv(torch.cat([self._upsampling_x4(x), self._skip_conv(x2)], dim=1))
        x = self._upsampling_x4(x)

        return x
