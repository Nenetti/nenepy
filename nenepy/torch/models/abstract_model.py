from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim


class AbstractModel(metaclass=ABCMeta):

    def __init__(self, network_module=None, optimizer=None, scheduler=None, loss=None):
        """

        Args:
            network_module (nn.Module):
            loss (nn.Module):
            optimizer (optim.Optimizer):
            scheduler (optim.lr_scheduler.LambdaLR):

        """
        # ----- Network ----- #
        self._network_module = network_module

        # ----- Optimizer and Scheduler ----- #
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._loss_func = loss
        self._device = "cpu"

    @property
    def lr_dict(self):
        if self._scheduler is not None:
            return dict([(f"Param{i}", lr) for i, lr in enumerate(self._scheduler.get_last_lr())])

        elif self._optimizer is not None:
            return dict([(f"Param{i}", group["lr"]) for i, group in enumerate(self._optimizer.param_groups)])

        else:
            return None

    def to(self, device):
        """
        Args:
            device (str): 'cuda' or 'cpu'

        """
        self._network_module.to(device)
        self._loss_func.to(device)
        self._device = device

    def to_multi_gpu(self, n_gpus):
        if n_gpus > 1:
            self._network_module = nn.DataParallel(self._network_module).to(self.device)

    @staticmethod
    def _init_weights(net, init_type="normal", init_gain=0.02):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use "normal" in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """

        def initialize(m):
            if hasattr(m, "weight") and (isinstance(m, nn.modules.conv._ConvNd) or isinstance(m, nn.Linear)):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, init_gain)

                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)

                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")

                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)

                else:
                    raise NotImplementedError("initialization method [%s] is not implemented" % init_type)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, nn.modules.batchnorm._NormBase):
                if hasattr(m, "weight"):
                    nn.init.normal_(m.weight.data, 1.0, init_gain)
                if hasattr(m, "bias"):
                    nn.init.constant_(m.bias.data, 0.0)

        print("initialize network with %s" % init_type)
        net.apply(initialize)  # apply the initialization function <init_func>

    # ==================================================================================================
    #
    #   Abstract Method
    #
    # ==================================================================================================
    @abstractmethod
    def training_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def validation_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def test_step(self, *args, **kwargs):
        raise NotImplementedError()

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    def scheduler_step(self):
        if self._scheduler is not None:
            self._scheduler.step()

    def train_mode(self):
        self._network_module.train()

    def validate_mode(self):
        self._network_module.eval()

    def load_weight(self, path=None):
        """

        Args:
            path (Path or str):

        """
        if path is None:
            path = self.weight_path

        self._network_module.load_state_dict(torch.load(path))

    def save_weight(self, path=None):
        """

        Args:
            path (Path or str):

        """
        if path is None:
            path = self.weight_path

        torch.save(self._network_module.state_dict(), path)

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================

    def _backward(self, loss, optimizer=None):
        if optimizer is None:
            optimizer = self._optimizer

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None
        loss.backward()
        optimizer.step()

    def _to_device(self, *args):
        return [arg.to(self._device) for arg in args]

    @staticmethod
    def _to_detach_cpu_numpy(*args):

        def recursive(arg):
            if isinstance(arg, torch.Tensor):
                return arg.detach().cpu().numpy()

            if isinstance(arg, tuple):
                arg = (recursive(a) for a in arg)
                return arg

            if isinstance(arg, list):
                arg = [recursive(a) for a in arg]
                return arg

            if isinstance(arg, dict):
                arg = dict([(key, recursive(value)) for key, value in arg.items()])
                return arg

            return arg

        if len(args) == 1:
            return recursive(args[0])
        else:
            return (recursive(arg) for arg in args)
