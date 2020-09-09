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

        self._train_pre_hooks = OrderedDict()
        self._train_hooks = OrderedDict()
        self._validate_pre_hooks = OrderedDict()
        self._validate_hooks = OrderedDict()
        self._test_pre_hooks = OrderedDict()
        self._test_hooks = OrderedDict()

        self.register_train_pre_hook(self._to_device)
        self.register_train_hook(self._to_detach_cpu)
        self.register_validate_pre_hook(self._to_device)
        self.register_validate_hook(self._to_detach_cpu)
        self.register_test_pre_hook(self._to_device)
        self.register_test_hook(self._to_detach_cpu)

    def _init_multi_gpu(self, n_gpus):
        if n_gpus > 1:
            self.network_model = nn.DataParallel(self.network_model).to(self.device)
        else:
            self.network_model = self.network_model.to(self.device)

    # ==============================================================================
    #
    #   Abstract Method
    #
    # ==============================================================================

    @abstractmethod
    def train_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def validate_step(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def test_step(self, *args, **kwargs):
        raise NotImplementedError()

    # ==============================================================================
    #
    #   Public Method
    #
    # ==============================================================================

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def register_train_pre_hook(self, hook):
        n_args = len(list(signature(hook).parameters))
        self._train_pre_hooks[id(hook)] = (n_args, hook)

    def register_train_hook(self, hook):
        n_args = len(list(signature(hook).parameters))
        self._train_hooks[id(hook)] = (n_args, hook)

    def register_validate_pre_hook(self, hook):
        n_args = len(list(signature(hook).parameters))
        self._validate_pre_hooks[id(hook)] = (n_args, hook)

    def register_validate_hook(self, hook):
        n_args = len(list(signature(hook).parameters))
        self._validate_hooks[id(hook)] = (n_args, hook)

    def register_test_pre_hook(self, hook):
        n_args = len(list(signature(hook).parameters))
        self._test_pre_hooks[id(hook)] = (n_args, hook)

    def register_test_hook(self, hook):
        n_args = len(list(signature(hook).parameters))
        self._test_hooks[id(hook)] = (n_args, hook)

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

    # ==============================================================================
    #
    #   Private Method
    #
    # ==============================================================================

    def _backward(self, loss):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None
        loss.backward()
        self.optimizer.step()

    def _to_device(self, model, model_in):

        def recursive(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(self.device)
            else:
                if isinstance(arg, tuple):
                    arg = tuple([recursive(a) for a in arg])
                    return arg

                elif isinstance(arg, list):
                    arg = [recursive(a) for a in arg]
                    return arg

                elif isinstance(arg, dict):
                    arg = dict([(key, recursive(value)) for key, value in arg.items()])
                    return arg

            return arg

        args, kwargs = recursive(model_in)

        return args, kwargs

    @staticmethod
    def _to_detach_cpu(model, model_in, model_out):

        def recursive(arg):
            if isinstance(arg, torch.Tensor):
                return arg.detach().cpu()
            else:
                if isinstance(arg, tuple):
                    arg = tuple([recursive(a) for a in arg])
                    return arg

                elif isinstance(arg, list):
                    arg = [recursive(a) for a in arg]
                    return arg

                elif isinstance(arg, dict):
                    arg = dict([(key, recursive(value)) for key, value in arg.items()])
                    return arg

            return arg

        return tuple(recursive(model_out))

    # ==============================================================================
    #
    #   Special Attribute
    #
    # ==============================================================================

    def __call__(self, *args, **kwargs):
        if self.mode is Mode.TRAIN:
            forward_step = self.train_step
            forward_pre_hooks = self._train_pre_hooks
            forward_hooks = self._train_hooks

        elif self.mode is Mode.VALIDATE:
            forward_step = self.validate_step
            forward_pre_hooks = self._validate_pre_hooks
            forward_hooks = self._validate_hooks

        else:
            forward_step = self.test_step
            forward_pre_hooks = self._test_pre_hooks
            forward_hooks = self._test_hooks

        for n_args, hook in forward_pre_hooks.values():
            if n_args == 0:
                hook()
            else:
                hook_out = hook(self, (args, kwargs))

                if hook_out is None:
                    continue
                elif (isinstance(hook_out, tuple)) and (len(hook_out) == 2) and isinstance(hook_out[0], tuple) and isinstance(hook_out[1], dict):
                    args, kwargs = hook_out
                else:
                    raise ValueError()

        output = forward_step(*args, **kwargs)

        if not isinstance(output, tuple):
            output = (output,)

        for n_args, hook in forward_hooks.values():
            if n_args == 0:
                hook()
            else:
                hook_out = hook(self, (args, kwargs), output)
                if hook_out is not None:
                    output = hook_out

        if len(output) == 1:
            output = output[0]

        return output
