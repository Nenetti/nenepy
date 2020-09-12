from torch import nn

from nenepy.torch.nn.architectures import AbstractNetworkArchitecture
from nenepy.torch.nn.architectures.backbones import ResNeXt50_32x4d, ResNeXt101_32x8d, WideResNet50_2, WideResNet101_2
from nenepy.torch.nn.architectures.backbones import ResNet50, ResNet18, ResNet34, ResNet101, ResNet152
from nenepy.torch.nn.modules import GlobalCueInjection, StochasticGate, ASPP, DynamicUpsample
from nenepy.torch.nn.modules.concat import Concat

backbone_list = [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNeXt50_32x4d, ResNeXt101_32x8d, WideResNet50_2, WideResNet101_2]


class DeepLabV3Plus(AbstractNetworkArchitecture):

    # ==============================================================================
    #
    #   Initialization
    #
    # ==============================================================================

    def __init__(self, in_channels, out_channels, backbone=ResNet50, backbone_pretrained=True, sg_psi=0.3, backbone_kwargs={}):
        super(DeepLabV3Plus, self).__init__()

        self._sg_psi = sg_psi

        self._init_encoder(in_channels, backbone, backbone_pretrained, backbone_kwargs)
        self._init_decoder(out_channels)
        self.train()

    def _init_encoder(self, in_channels, backbone, backbone_pretrained, backbone_kwargs):
        """

        Returns:

        """
        if backbone not in backbone_list:
            raise ValueError()

        self._backbone_encoder = backbone(in_channels=in_channels, reduction_rate=16, pretrained=backbone_pretrained, is_feature_extraction=True, **backbone_kwargs)

        self._gci = GlobalCueInjection()

        self._aspp = ASPP(in_channels=2048, output_stride=8)

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

    def _init_decoder(self, out_channels):
        """

        Args:
            out_channels (int):

        """
        self._deconv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(256, out_channels, kernel_size=1, stride=1)
        )

        self._post_processing = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
        self._upsampling = DynamicUpsample(scale_factor=(4, 4), mode="bilinear", align_corners=True)

        self._add_training_modules(self._deconv)

    # ==============================================================================
    #
    #   Public Method
    #
    # ==============================================================================

    def forward(self, x, return_features=False, output_size=None):
        """

        Args:
            x (torch.Tensor):
            return_features (bool):

        Returns:

        """

        # ----- Forward Encoder (Feature Extraction) ----- #
        low_feature, encoder_x = self._forward_encoder(x)

        # ----- Forward Decoder (Convert feature to mask) ----- #
        feature_maps = self._forward_decoder(low_feature, encoder_x)

        masks = self._post_processing(feature_maps)
        masks = self._upsampling(masks, size=output_size)

        if return_features:
            feature_maps = self._upsampling(feature_maps, size=output_size)
            return feature_maps, masks

        return masks

    def train(self, mode=True):
        super(DeepLabV3Plus, self).train()
        self.requires_grad_(requires_grad=True)
        self._backbone_encoder.train()

    def eval(self):
        super(DeepLabV3Plus, self).eval()
        self.requires_grad_(requires_grad=False)

    #
    # def parameters_dict(self, base_lr, wd):
    #
    #     # 1., 2., 10., 20.
    #     w_old, b_old, w_new, b_new = 1., 1., 10., 10.
    #
    #     groups = [
    #         {"params": [], "weight_decay": wd, "lr": w_old * base_lr},  # weight learning
    #         {"params": [], "weight_decay": 0.0, "lr": b_old * base_lr},  # bias finetuning
    #         {"params": [], "weight_decay": wd, "lr": w_new * base_lr},  # weight finetuning
    #         {"params": [], "weight_decay": 0.0, "lr": b_new * base_lr}
    #     ]
    #
    #     for m in self.modules():
    #
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
    #
    #             if m.weight is not None:
    #                 if m in self.training_modules:
    #                     groups[2]["params"].append(m.weight)
    #                 else:
    #                     groups[0]["params"].append(m.weight)
    #
    #             if m.bias is not None:
    #                 if m in self.training_modules:
    #                     groups[3]["params"].append(m.bias)
    #                 else:
    #                     groups[1]["params"].append(m.bias)
    #
    #         elif hasattr(m, "weight"):
    #             print("! Skipping learnable: ", m)
    #
    #     for i, g in enumerate(groups):
    #         print("Group {}: #{}, LR={:4.3e}".format(i, len(g["params"]), g["lr"]))
    #
    #     return groups

    # ==============================================================================
    #
    #   Private Method
    #
    # ==============================================================================

    def _forward_encoder(self, x):
        """
        Encoder部分の計算

        Args:
            x (torch.Tensor): raw image

        Returns:
            torch.Tensor:

        """

        # 1. Feature Extraction (Backbone (ResNet))
        _, low_feature, _, _, high_feature = self._backbone_encoder(x)

        # 2. Atrous Spatial Pyramid Pooling (ASPP) (from deeplabv3)
        aspp_x = self._aspp(high_feature)
        aspp_x = self._upsampling(aspp_x, size=tuple(low_feature.shape[-2:]))

        return low_feature, aspp_x

    def _forward_decoder(self, low_feature, x):
        """
        Decoder部分の計算

        Args:
            x (torch.Tensor):

        Returns:
            torch.Tensor:

        """

        # 3. Connect low-level feature to high-level feature
        #    Concatenate low-level feature with high-level feature

        skip_x = self._low_level_skip(low_feature)
        concat_x = self._concat(x, skip_x)
        concat_x = self._conv_3x3(concat_x)

        # 4. Deep feature context for shallow features
        gci_x = self._gci(low_feature, concat_x)
        sg_x = self._stochastic_gate(gci_x, concat_x, p=self._sg_psi)

        x_out = self._deconv(sg_x)

        return x_out
