import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        at = torch.ones_like(targets) * self.alpha
        at[targets == 1] = 1 - self.alpha

        loss = at * ((targets - torch.sigmoid(inputs)) ** self.gamma) * loss

        return loss
