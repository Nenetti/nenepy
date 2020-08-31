import torch
from torch import nn
from torch.functional import F


class BalancedMaskLoss(nn.Module):

    def __init__(self, ignore_index=0):
        super(BalancedMaskLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, feature_maps, pseudo_gt_masks, gt_labels):
        """

        Args:
            feature_maps (torch.Tensor):
            pseudo_gt_masks (torch.Tensor):
            gt_labels (torch.Tensor):

        Returns:
            torch.Tensor;

        Shape:
            [B, C, H, W], [B, C, H, W], [B, C] -> [B]

        """
        batch_size, n_channels, height, width = pseudo_gt_masks.shape

        # ----- Multi GPU ----- #
        feature_maps = F.interpolate(feature_maps, size=(height, width), mode="bilinear", align_corners=True)
        gt_mask_indices = pseudo_gt_masks.argmax(dim=1)

        # ----- Cross Entropy loss (Ignore Background Index) ----- #
        loss = F.cross_entropy(feature_maps, gt_mask_indices, ignore_index=self.ignore_index, reduction="none")
        loss = loss.contiguous().view(batch_size, -1)

        # ----- Class Weight Balances ----- #
        n_pixels_per_class = pseudo_gt_masks.contiguous().view(batch_size, n_channels, -1).sum(dim=2)
        n_pixels_total = n_pixels_per_class.sum(1, keepdim=True)

        class_weight = (n_pixels_total - n_pixels_per_class) / (1 + n_pixels_total)
        class_weight = (pseudo_gt_masks * class_weight[:, :, None, None]).sum(dim=1).contiguous().view(batch_size, -1)

        # ----- Apply GT-Label ----- #
        predict_labels = (n_pixels_per_class[:, 1:] > 0).type(torch.float32)
        n_match_labels = (gt_labels == predict_labels).type(torch.float32).sum(dim=1)
        available_element = (n_match_labels == gt_labels.shape[1]).type(torch.float32)

        loss = available_element * (class_weight * loss).mean(dim=1)

        return loss
