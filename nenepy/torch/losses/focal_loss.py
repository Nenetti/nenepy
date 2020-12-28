import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=0.4, gamma=2, class_weight=None):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._class_weight = class_weight

    def forward(self, inputs, targets):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        class_weight = self.to_class_weight(targets) if self._class_weight is None else self._class_weight
        # print(targets[0])
        # print(torch.sigmoid(inputs)[0])
        # return class_weight * loss

        # return 10 * class_weight * ((targets - torch.sigmoid(inputs).detach()) ** self._gamma) * loss
        return class_weight * ((targets - torch.sigmoid(inputs).detach()) ** self._gamma) * loss
        # return class_weight * ((1 - torch.sigmoid(inputs).detach()) ** self._gamma) * loss

    def to_class_weight(self, targets):
        class_weight = torch.ones_like(targets) * self._alpha
        class_weight[targets == 1] = 1 - self._alpha
        return class_weight
