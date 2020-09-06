from abc import ABCMeta

import torch
import torch.nn as nn


class AbstractNetworkArchitecture(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        self.training_layers = []  # new layers -> higher LR

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _add_training_modules(self, module):
        """

        Args:
            module (nn.Module):

        Returns:

        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                self.training_layers.append(m)
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                self.training_layers.append(m)
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def parameters_dict(self, base_lr, wd):

        modules = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    modules.append(m.weight)

                if m.bias is not None:
                    modules.append(m.bias)

        groups = [{"params": modules, "weight_decay": wd, "lr": base_lr}]

        for i, g in enumerate(groups):
            print("Group {}: #{}, LR={:4.3e}".format(i, len(g["params"]), g["lr"]))

        return groups
