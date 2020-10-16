from abc import ABCMeta

import torch
import torch.nn as nn


class AbstractNetworkArchitecture(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.training_layers = []
        self.untraining_layers = []

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _add_training_modules(self, module):
        self.training_layers.extend(module.modules())
        self._initialize_weights(module)

    def _add_untraining_modules(self, module):
        self.untraining_layers.extend(module.modules())

    def _initialize_weights(self, module):
        for m in module.modules():
            if m not in self.training_layers:
                continue

            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def parameters_dict(self, base_lr, wd):

        modules = []
        for m in self.modules():
            if m in self.untraining_layers:
                continue

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    modules.append(m.weight)

                if m.bias is not None:
                    modules.append(m.bias)

        groups = [{"params": modules, "weight_decay": wd, "lr": base_lr}]

        for i, g in enumerate(groups):
            print("Group {}: #{}, LR={:4.3e}".format(i, len(g["params"]), g["lr"]))

        return groups
