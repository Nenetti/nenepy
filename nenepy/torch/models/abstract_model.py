from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim


class AbstractModel(metaclass=ABCMeta):

    def __init__(self, network_modules=[], optimizers=None, schedulers=None, loss_funcs=None):
        """

        Args:
            network_modules (list[nn.Module] or dict[str, nn.Module]):
            loss_funcs (list[nn.Module] or dict[str, nn.Module]):
            optimizers (list[optim.Optimizer] or dict[str, optim.Optimizer]):
            schedulers (list, dict):

        """
        # ----- Network ----- #
        self._network_modules = network_modules if isinstance(network_modules, (tuple, list, dict)) else [network_modules]

        # ----- Optimizer and Scheduler ----- #
        self._optimizers = optimizers if isinstance(optimizers, (tuple, list, dict)) else [optimizers]
        self._schedulers = schedulers if isinstance(schedulers, (tuple, list, dict)) else [schedulers]
        self._loss_funcs = loss_funcs if isinstance(loss_funcs, (tuple, list, dict)) else [loss_funcs]
        self._device = "cpu"

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
    #   Instance Method (Public)
    #
    # ==================================================================================================
    @property
    def lr_dict(self):
        # if self._schedulers is not None:
        #     return dict([(f"Param{i}", lr) for i, lr in enumerate(self._schedulers.get_last_lr())])
        #
        # elif self._optimizers is not None:
        #     return dict([(f"Param{i}", group["lr"]) for i, group in enumerate(self._optimizers.param_groups)])
        if self._optimizers is not None:
            param_dict = {}
            if isinstance(self._optimizers, dict):
                for key, optimizer in self._optimizers.items():
                    d = dict([(f"{optimizer.__class__.__name__}_{key}_{k + 1}", group["lr"]) for k, group in enumerate(optimizer.param_groups)])
                    param_dict.update(d)
            else:
                for i, optimizer in enumerate(self._optimizers):
                    d = dict([(f"{optimizer.__class__.__name__}_{i + 1}_{k + 1}", group["lr"]) for k, group in enumerate(optimizer.param_groups)])
                    param_dict.update(d)
                return param_dict
        else:
            return None

    def to(self, device):
        """
        Args:
            device (str): 'cuda' or 'cpu'

        """

        def func(modules):
            if isinstance(modules, dict):
                return dict([(k, v.to(device)) for k, v in modules.items()])
            else:
                return [v.to(device) for v in modules]

        self._network_modules = func(self._network_modules)
        self._loss_funcs = func(self._loss_funcs)
        self._device = device

    def to_multi_gpu(self, n_gpus):
        def func(modules):
            if isinstance(modules, dict):
                return dict([(k, nn.DataParallel(v).to(self._device)) for k, v in modules.items()])
            else:
                return [nn.DataParallel(v).to(self._device) for v in modules]

        if n_gpus > 1:
            self._network_modules = func(self._network_modules)

    def scheduler_step(self):
        if self._schedulers is not None:
            if isinstance(self._schedulers, dict):
                for scheduler in self._schedulers.values():
                    scheduler.step()
            else:
                for scheduler in self._schedulers:
                    scheduler.step()

    def train_mode(self, mode=True):
        if isinstance(self._schedulers, dict):
            for module in self._network_modules.values():
                module.train(mode)
                module.requires_grad_(mode)
        else:
            for module in self._network_modules:
                module.train(mode)
                module.requires_grad_(mode)

    def validate_mode(self):
        self.train_mode(False)

    def test_mode(self):
        self.train_mode(False)

    def load_weight(self, path=None):
        """

        Args:
            path (Path or str):

        """
        if path is None:
            path = self.weight_path

        self._network_modules.load_state_dict(torch.load(path))

    def save_weight(self, path=None):
        """

        Args:
            path (Path or str):

        """
        if path is None:
            path = self.weight_path

        torch.save(self._network_modules.state_dict(), path)

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    def _backward(self, loss, optimizer=None):
        if optimizer is None:
            optimizer = self._optimizers

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None

        loss.backward()
        optimizer.step()

    def _to_device(self, *args):
        if len(args) == 1:
            return args[0].to(self._device)
        return [arg.to(self._device) for arg in args]

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================

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
