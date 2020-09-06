import torch
from torch import nn
from torch.functional import F

from nenepy.torch.nn.architectures import AbstractNetworkArchitecture
from nenepy.torch.nn.architectures.segmentation import DeepLabV3Plus
from nenepy.torch.nn.modules import PAMR


class WeaklySupervisedSegmentation(AbstractNetworkArchitecture):
    """

    """

    def __init__(self, out_channels,
                 backbone_pretrained=True, sg_psi=0.3, focal_p=3, focal_lambda=0.01,
                 pamr_iter=10, pamr_kernel=[1, 2, 4, 8, 12, 24],
                 pseudo_lower=0.2
                 ):
        super(WeaklySupervisedSegmentation, self).__init__()

        self.semantic_segmentation = DeepLabV3Plus(
            out_channels=out_channels,
            backbone_pretrained=backbone_pretrained,
            sg_psi=sg_psi
        )

        self.pamr = PAMR(
            num_iter=pamr_iter,
            dilations=pamr_kernel
        )

        self.upsampling_x4 = nn.Upsample(
            scale_factor=(4, 4),
            mode="bilinear",
            align_corners=True
        )

        self.focal_p = focal_p
        self.focal_lambda = focal_lambda
        self.pseudo_lower = pseudo_lower

    # ==============================================================================
    #
    #   Public Method
    #
    # ==============================================================================

    def forward(self, x, x_for_pamr=None, label=None, is_classification_only=False, use_gt_label=False):
        """

        Args:
            x (torch.Tensor):       RGB Image (場合によっては前処理されている) [B, C, H, W]
            x_for_pamr (torch.Tensor):   RGB Image [B, C, H, W]

            is_classification_only (bool):     評価モードで計算するかどうか (評価時はラベル情報がないため)

        Returns:

        """
        if x_for_pamr is None:
            x_for_pamr = x

        # ----- Forward Encoder (Feature Extraction) ----- #
        # ----- Forward Decoder (Convert feature to mask) ----- #
        feature_maps, masks = self.semantic_segmentation(x, return_features=True)

        # ----- Reshaping ([H, W] -> [H * W]) ----- #
        batch_size, n_channels, _, _ = feature_maps.size()
        reshape_features = feature_maps.contiguous().view(batch_size, n_channels, -1)
        reshape_masks = masks.contiguous().view(batch_size, n_channels, -1)

        # ----- Classification ----- #
        loss_ngwp = self._normalized_global_weighted_pooling(features=reshape_features, masks=reshape_masks)
        loss_size_focal = self._size_focal_penalty(masks=reshape_masks, p=self.focal_p, lambda_=self.focal_lambda)
        classification = torch.sigmoid(loss_ngwp[:, 1:] + loss_size_focal[:, 1:])

        if is_classification_only:
            return classification, feature_maps, self.upsampling_x4(masks), None, None

        else:
            # ----- Mask Refinement (PAMR) ----- #
            pamr_masks = self._forward_pamr(x_for_pamr, masks)

            # ----- Up-Scaling ----- #
            if use_gt_label:
                masks = self._upsampling_x4_and_clean(masks, label)
                pamr_masks = self._upsampling_x4_and_clean(pamr_masks, label)
            else:
                masks = self.upsampling_x4(masks)
                pamr_masks = self.upsampling_x4(pamr_masks)

            # ----- Pseudo GT-Mask ----- #
            pseudo_gt = self._pseudo_gtmask(pamr_masks, lowest_limit=self.pseudo_lower).detach()

            return classification, feature_maps, masks, pamr_masks, pseudo_gt

    def parameters_dict(self, base_lr, wd):
        return self.semantic_segmentation.parameters_dict(base_lr, wd)

    # ==============================================================================
    #
    #   Private Method
    #
    # ==============================================================================

    def _forward_pamr(self, image_raws, masks):
        """

        Args:
            image_raws (torch.Tensor):  RGB Image [B, C, H, W]
            masks (torch.Tensor):       RGB Image [B, C, H, W]

        Returns:

        """
        masks = masks.detach()
        image_raws = F.interpolate(image_raws, masks.size()[-2:], mode="bilinear", align_corners=True)
        pamr_masks = self.pamr(image_raws, masks)
        return pamr_masks

    def _upsampling_x4_and_clean(self, masks, labels):
        masks = self.upsampling_x4(masks)
        masks[:, 1:] *= labels[:, :, None, None].type_as(masks)
        return masks

    @staticmethod
    def _pseudo_gtmask(masks, lowest_limit):
        """

        Args:
            masks (torch.Tensor):
            lowest_limit (float): Maskの下限値 (これ以下の値はすべて切り捨て)

        Returns:
            torch.Tensor: Channel方向がone-hot (各ピクセルにおいて1つのChannelだけ1，他Channelは0)

        """
        batch_size, n_channels, height, width = masks.size()
        reshape_masks = masks.contiguous().view(batch_size, n_channels, -1)

        # 各Channelの最大値
        max_confidences, _ = reshape_masks.max(dim=2, keepdim=True)
        max_confidences[:, :1] *= 0.7
        max_confidences[:, 1:] *= 0.6

        # 各Channelの最大確率値に対して一定以上の確率であれば擬似ラベルとする
        # ただし最大値の値に関わらず20%以上の確率でなければ切り捨て
        lowest_confidences = torch.full(size=[1], fill_value=lowest_limit).type_as(max_confidences)
        lowest_limit = torch.max(max_confidences, lowest_confidences)

        pseudo_gt_binary = (reshape_masks > lowest_limit).type(dtype=torch.float32)

        # Channel方向で合計値が1を超える(複数クラスの確率が高い)曖昧なピクセルは無視
        ambiguous_binary = (pseudo_gt_binary.sum(dim=1, keepdim=True) > 1).type(dtype=torch.float32)
        pseudo_gt_binary = (1 - ambiguous_binary) * pseudo_gt_binary

        return pseudo_gt_binary.contiguous().view(batch_size, n_channels, height, width)

    @staticmethod
    def _normalized_global_weighted_pooling(features, masks, epsilon=1.0):
        return (features * masks).sum(2) / (epsilon + masks.sum(2))

    @staticmethod
    def _size_focal_penalty(masks, p=3, lambda_=0.01):
        m_c = masks.mean(dim=2)
        return torch.pow(1 - m_c, p) * torch.log(lambda_ + m_c)
