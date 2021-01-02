from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from inspect import signature
from pathlib import Path

import torch
import torch.nn as nn

from nenepy.torch.interfaces import Mode


class AbstractModel(metaclass=ABCMeta):

    def __init__(self, device, weights_path, optimizer_kwargs={}):
        """

        Args:

        """
        self.mode = Mode.TRAIN
        # ----- Network ----- #
        self.network_model = None

        # ----- Optimizer and Scheduler ----- #
        self.optimizer = None
        self.scheduler = None

        self.weight_path = Path(weights_path)
        self.device = device

    def _init_multi_gpu(self, n_gpus):
        if n_gpus > 1:
            self.network_model = nn.DataParallel(self.network_model).to(self.device)
        else:
            self.network_model = self.network_model.to(self.device)

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
    def train_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def validate_step(self, *args, **kwargs):
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
        if self.scheduler is not None:
            self.scheduler.step()

    def train_mode(self):
        self.mode = Mode.TRAIN
        self.network_model.train()

    def validate_mode(self):
        self.mode = Mode.VALIDATE
        self.network_model.eval()

    def test_mode(self):
        self.mode = Mode.TEST
        self.network_model.eval()

    def load_weight(self, path=None):
        """

        Args:
            path (Path or str):

        """
        if path is None:
            path = self.weight_path

        self.network_model.load_state_dict(torch.load(path))

    def save_weight(self, path=None):
        """

        Args:
            path (Path or str):

        """
        if path is None:
            path = self.weight_path

        torch.save(self.network_model.state_dict(), path)

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================

    def _backward(self, loss, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer

        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None
        loss.backward()
        optimizer.step()

    def _to_device(self, *args, **kwargs):

        def recursive(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(self.device)

            if isinstance(arg, tuple):
                arg = (recursive(a) for a in arg)
                return arg

            if isinstance(arg, list):
                arg = [recursive(a) for a in arg]
                return arg

            if isinstance(arg, dict):
                arg = dict(((key, recursive(value)) for key, value in arg.items()))
                return arg

            return arg

        if len(args) > 0:
            args = recursive(args)
        if len(kwargs) > 0:
            kwargs = recursive(kwargs)

        return args, kwargs

    @staticmethod
    def _to_detach_cpu(args):

        def recursive(arg):
            if isinstance(arg, torch.Tensor):
                return arg.detach().cpu()

            if isinstance(arg, tuple):
                arg = tuple([recursive(a) for a in arg])
                return arg

            if isinstance(arg, list):
                arg = [recursive(a) for a in arg]
                return arg

            if isinstance(arg, dict):
                arg = dict([(key, recursive(value)) for key, value in arg.items()])
                return arg

            return arg

        if len(args) > 0:
            args = recursive(args)

        return args

    # ==================================================================================================
    #
    #   Special Attribute
    #
    # ==================================================================================================

    def __call__(self, *args, **kwargs):

        args, kwargs = self._to_device(*args, **kwargs)

        if self.mode is Mode.TRAIN:
            output = self.train_step(*args, **kwargs)
        elif self.mode is Mode.VALIDATE:
            output = self.validate_step(*args, **kwargs)
        else:
            output = self.test_step(*args, **kwargs)

        output = self._to_detach_cpu(output)

        return output
