import torch

from nenepy.torch.nn.architectures import AbstractNetworkArchitecture
from nenepy.torch.nn.architectures.backbones import resnet50
from nenepy.torch.nn.modules import GlobalCueInjection, StochasticGate, ASPP, DynamicUpsample
from nenepy.torch.nn.modules.concat import Concat
from torch import nn
from torch.functional import F


class DeepLabV3Plus(AbstractNetworkArchitecture):
    """

    """

    # ==============================================================================
    #
    #   Initialization
    #
    # ==============================================================================

    def __init__(self, out_channels, backbone_pretrained=True, sg_psi=0.3):
        super(DeepLabV3Plus, self).__init__()

        self.backbone_pretrained = backbone_pretrained
        self._sg_psi = sg_psi

        self._init_encoder(backbone_pretrained)
        self._init_decoder(out_channels)
        self.train()

    def _init_encoder(self, backbone_pretrained):
        """

        Returns:

        """
        self._backbone_encoder = resnet50(pretrained=backbone_pretrained)

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
        self.deconv = nn.Sequential(
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

        self.upsampling = DynamicUpsample(scale_factor=(4, 4), mode="bilinear", align_corners=True)

        self._add_training_modules(self.deconv)

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
        masks = F.softmax(feature_maps, dim=1)

        masks = self.upsampling(masks, size=output_size)

        if return_features:
            feature_maps = self.upsampling(feature_maps, size=output_size)
            return feature_maps, masks

        return masks

    def train(self, mode=True):
        super(DeepLabV3Plus, self).train()
        self.requires_grad_(requires_grad=True)
        if self.backbone_pretrained:
            self._backbone_encoder.requires_grad_(requires_grad=False)

    def eval(self):
        super(DeepLabV3Plus, self).eval()
        self.requires_grad_(requires_grad=False)

    def parameters_dict(self, base_lr, wd):

        # 1., 2., 10., 20.
        w_old, b_old, w_new, b_new = 1., 1., 10., 10.

        groups = [
            {"params": [], "weight_decay": wd, "lr": w_old * base_lr},  # weight learning
            {"params": [], "weight_decay": 0.0, "lr": b_old * base_lr},  # bias finetuning
            {"params": [], "weight_decay": wd, "lr": w_new * base_lr},  # weight finetuning
            {"params": [], "weight_decay": 0.0, "lr": b_new * base_lr}
        ]

        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):

                if m.weight is not None:
                    if m in self.training_modules:
                        groups[2]["params"].append(m.weight)
                    else:
                        groups[0]["params"].append(m.weight)

                if m.bias is not None:
                    if m in self.training_modules:
                        groups[3]["params"].append(m.bias)
                    else:
                        groups[1]["params"].append(m.bias)

            elif hasattr(m, "weight"):
                print("! Skipping learnable: ", m)

        for i, g in enumerate(groups):
            print("Group {}: #{}, LR={:4.3e}".format(i, len(g["params"]), g["lr"]))

        return groups

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
        low_feature, high_feature = self._backbone_encoder.feature_extraction(x)

        # 2. Atrous Spatial Pyramid Pooling (ASPP) (from deeplabv3)
        aspp_x = self._aspp(high_feature)
        aspp_x = self.upsampling(aspp_x)

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

        x_out = self.deconv(sg_x)

        return x_out
