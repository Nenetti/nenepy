from torch import nn

from nenepy.torch.nn.architectures import AbstractNetworkArchitecture
from nenepy.torch.nn.architectures.backbones import ResNeXt50_32x4d, ResNeXt101_32x8d, WideResNet50_2, WideResNet101_2
from nenepy.torch.nn.architectures.backbones import ResNet50, ResNet18, ResNet34, ResNet101, ResNet152
from nenepy.torch.nn.modules import GlobalCueInjection, StochasticGate, AtrousSpatialPyramidPooling, Upsample
from nenepy.torch.nn.modules.concat import Concat

backbone_list = [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt50_32x4d, ResNeXt101_32x8d, WideResNet50_2, WideResNet101_2]


class DeepLabV3Plus(AbstractNetworkArchitecture):

    def __init__(self, in_channels, out_channels, backbone=ResNet50, sg_psi=0.3, backbone_pretrained=True, backbone_kwargs={}):
        super(DeepLabV3Plus, self).__init__()

        self._sg_psi = sg_psi

        self._init_encoder(in_channels, backbone, backbone_pretrained, backbone_kwargs)
        self._init_decoder(out_channels)

    def _init_encoder(self, in_channels, backbone_cls, backbone_pretrained, backbone_kwargs):
        """

        Args:
            in_channels (int):
            backbone_cls (nn.Module):
            backbone_pretrained (bool):
            backbone_kwargs (dict):

        Returns:

        """
        if backbone_cls not in backbone_list:
            raise ValueError()

        self._backbone_encoder = backbone_cls(
            in_channels=in_channels, reduction_rate=16, pretrained=backbone_pretrained, is_feature_extraction=True,
            **backbone_kwargs
        )

        self._gci = GlobalCueInjection()

        self._aspp = AtrousSpatialPyramidPooling(in_channels=2048, output_stride=8)

        self._stochastic_gate = StochasticGate()

        self._low_level_skip = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self._conv_3x3 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self._concat = Concat(dim=1)

        self._add_training_modules(self._gci)
        self._add_training_modules(self._aspp)
        self._add_training_modules(self._low_level_skip)
        self._add_training_modules(self._conv_3x3)
        for m in self._backbone_encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                self._add_untraining_modules(m)
        self._add_untraining_modules(self._backbone_encoder.conv1)

    def _init_decoder(self, out_channels):
        """

        Args:
            out_channels (int):

        """
        self._deconv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1)
        )

        self._post_process = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
        self._upsampling = Upsample(scale_factor=(4, 4), mode="bilinear", align_corners=True)

        self._add_training_modules(self._deconv)

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    def forward(self, x, return_features=False, output_size=None):
        """

        Args:
            x (torch.Tensor):
            return_features (bool):

        Returns:

        """

        low_feature, deep_feature = self._forward_encoder(x)
        feature_maps = self._forward_decoder(low_feature, deep_feature)

        if return_features:
            return feature_maps

        else:
            masks = self._post_process(feature_maps)
            masks = self._upsampling(masks, size=output_size)

            return masks

    def train(self, mode=True):
        super(DeepLabV3Plus, self).train()
        if mode:
            self.requires_grad_(requires_grad=True)
            for m in self.untraining_layers:
                m.requires_grad_(requires_grad=False)
        else:
            self.requires_grad_(requires_grad=False)

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================

    def _forward_encoder(self, x):
        """
        Args:
            x (torch.Tensor): raw image

        Returns:
            torch.Tensor:

        """
        _, low_feature, _, _, deep_feature = self._backbone_encoder(x)

        deep_feature = self._aspp(deep_feature)
        deep_feature = self._upsampling(deep_feature, size=tuple(low_feature.shape[-2:]))

        return low_feature, deep_feature

    def _forward_decoder(self, low_feature, deep_feature):
        """
        Decoder部分の計算

        Args:
            low_feature (torch.Tensor):
            deep_feature (torch.Tensor):

        Returns:
            torch.Tensor:

        """
        skip_x = self._low_level_skip(low_feature)
        concat_x = self._concat(deep_feature, skip_x)
        concat_x = self._conv_3x3(concat_x)

        # 4. Deep feature context for shallow features
        gci_x = self._gci(low_feature, concat_x)
        sg_x = self._stochastic_gate(concat_x, gci_x, p=self._sg_psi)

        x_out = self._deconv(sg_x)

        return x_out
