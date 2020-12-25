import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalCueInjection(nn.Module):
    """
    Global Cue Injection (GCI) is almost like Adaptive Instance Normalization (AdaIN)

    """

    def __init__(self):
        super(GlobalCueInjection, self).__init__()

        self.conv_shallow = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, affine=False)
        )
        self.conv_deep = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv_adain = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, shallow_x, deep_x):
        """

        Args:
            shallow_x (torch.Tensor):    Shallow Features    [B, C, H, W]
            deep_x (torch.Tensor):       Deep Features       [B, C, H, W]

        Returns:
            torch.Tensor:

        """
        shallow_x = self.conv_shallow(shallow_x)
        deep_x = self.conv_deep(deep_x)

        adain_features = self.adain(shallow_x, deep_x)
        return self.conv_adain(adain_features)

    @staticmethod
    def adain(shallow_x, deep_x):
        """

        Args:
            shallow_x (torch.Tensor):    Shallow Features    [B, C, H, W]
            deep_x (torch.Tensor):       Deep Features       [B, C, H, W]

        Returns:
            torch.Tensor:

        """
        B, C, _, _ = shallow_x.size()

        deep_x, _ = deep_x.contiguous().view(B, C * 2, -1).max(dim=2)
        deep_x = deep_x.contiguous().view(B, C, 2)
        gamma, beta = deep_x[:, :, 0], deep_x[:, :, 1]

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return F.relu(shallow_x * (gamma + 1) + beta)
    #
    # @staticmethod
    # def adain(shallow_x, deep_x):
    #     """
    #
    #     Args:
    #         shallow_x (torch.Tensor):    Shallow Features    [B, C, H, W]
    #         deep_x (torch.Tensor):       Deep Features       [B, C, H, W]
    #
    #     Returns:
    #         torch.Tensor:
    #
    #     """
    #     B, C, _, _ = shallow_x.size()
    #
    #     deep_x, _ = deep_x.contiguous().view(B, C * 2, -1).max(dim=2)
    #     deep_x = deep_x.contiguous().view(B, C, 2)
    #
    #     gamma, beta = deep_x[:, :, 0], deep_x[:, :, 1]
    #
    #     reshape_x = shallow_x.contiguous().view(B, C, -1)
    #     mean = reshape_x.mean(dim=2)
    #     variance = reshape_x.var(dim=2) + 1e-6
    #     standard = variance.sqrt()
    #
    #     mean = mean[:, :, None, None]
    #     standard = standard[:, :, None, None]
    #     gamma = gamma[:, :, None, None]
    #     beta = beta[:, :, None, None]
    #
    #     return F.relu(gamma * ((shallow_x - mean) / standard) + beta)
