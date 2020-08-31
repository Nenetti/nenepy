import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticGate(nn.Module):
    """
    深層特徴量と浅層特徴量を一定比率で混ぜる
    """

    def __init__(self):
        super(StochasticGate, self).__init__()

    def forward(self, shallow_x, deep_x, p=0.3):
        """

        Args:
            shallow_x (torch.Tensor):   Shallow Feature
            deep_x (torch.Tensor):      Deep Feature
            p (float):                  混合率 (大きいほど浅層を考慮，小さいほど深層を考慮)

        Returns:
            torch.Tensor:

        """

        if self.training:
            # {0, 1}のバイナリマスクを生成
            r = 1 - (1 - p) * F.dropout(torch.ones_like(shallow_x), p=p)

            # xの期待値が深層部分の特徴量と同じになるように (E[x] = deep_x)
            delta = 1 / (1 - p)
            x = (1 - r) * delta * (deep_x - p * shallow_x) + r * shallow_x

        else:
            # 評価時は決定論的に足し合わせ
            x = ((1 - p) * deep_x) + (p * shallow_x)

        return x
