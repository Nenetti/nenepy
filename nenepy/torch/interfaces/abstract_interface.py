from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from inspect import signature
from pathlib import Path

import numpy as np

from nenepy.torch.interfaces import Mode
from nenepy.torch.models import AbstractModel
from nenepy.torch.utils.tensorboard import TensorBoardWriter
from nenepy.utils import Timer
from nenepy.utils.dictionary import ListDict


class AbstractInterface(metaclass=ABCMeta):

    def __init__(self, mode, model, logger, save_interval=10):
        """

        Args:
            mode (Mode):
            model (AbstractModel):
            log_dir (Path):
            logger (Log):

        """
        self.mode = mode

        # ----- Data ----- #
        self.dataset = None
        self.dataloader = None

        # ----- Network Model ----- #
        self.model = model

        # ----- Log ----- #
        self.logger = logger
        self.board_writer = TensorBoardWriter(log_dir=Path(logger.log_dir).joinpath(mode.name))

        # ----- etc ----- #
        self.timer = Timer()
        self.save_interval = save_interval

        self.epoch = 0
        self.log_time_key = f"{mode.name}_ELAPSED_TIME"
        self.log_average_time_key = f"{mode.name}_AVERAGE_ELAPSED_TIME"
        self.log_epoch_key = f"{mode.name}_NUM_EPOCH"
        self._init_log()

        self._forward_pre_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self.register_forward_pre_hook(self._default_pre_hook)
        self.register_forward_hook(self._default_hook)

    def _init_log(self):
        if self.logger.get(self.log_time_key) is None:
            self.logger[self.log_time_key] = 0

    # ==============================================================================
    #
    #   Abstract Method
    #
    # ==============================================================================

    @abstractmethod
    def forward_epoch(self, *args, **kwargs):
        raise NotImplementedError()

    # ==============================================================================
    #
    #   Public Method
    #
    # ==============================================================================

    def register_forward_pre_hook(self, hook):
        n_args = len(list(signature(hook).parameters))
        self._forward_pre_hooks[id(hook)] = (n_args, hook)

    def register_forward_hook(self, hook):
        n_args = len(list(signature(hook).parameters))
        self._forward_hooks[id(hook)] = (n_args, hook)

    # ==============================================================================
    #
    #   Private Method
    #
    # ==============================================================================

    def _default_pre_hook(self):
        self.epoch += 1
        if self.mode is Mode.TRAIN:
            self.model.train_mode()
        elif self.mode is Mode.VALIDATE:
            self.model.validate_mode()
        else:
            self.model.test_mode()

        self.timer.start()

    def _default_hook(self):
        self.timer.stop()
        self.model.scheduler_step()

        self.logger[self.log_epoch_key] = self.epoch
        self.logger[self.log_time_key] += self.timer.elapsed_time
        self.logger[self.log_average_time_key] = self.logger[self.log_time_key] / self.epoch

        if (self.mode is Mode.TRAIN) and (self.epoch % self.save_interval == 0):
            self.logger.save()
            self.model.save_weight()
            print(f"Model and Log Saved. Next Save Epoch: {(self.epoch + self.save_interval)}")

    def _output_loss(self, epoch, output_loss):
        """

        Args:
            epoch (int):
            output_loss (ListDict):

        """
        if len(output_loss) == 0:
            return

        for name, value in output_loss.items():
            value = np.array(value)
            scalar_values = {
                "max": float(np.max(value)),
                "mean": float(np.mean(value)),
                "min": float(np.min(value))
            }

            self.board_writer.add_scalars(namespace=name, graph_name=self.mode.name, scalar_dict=scalar_values, step=epoch)

    def _output_each_class_loss(self, epoch, each_class_losses, class_names):
        each_loss_dict = {}
        for i, loss in enumerate(each_class_losses.tolist()):
            each_loss_dict[class_names[i]] = loss / len(self.dataloader)
        self.board_writer.add_scalars(namespace="Class_Loss", graph_name="Validation", scalar_dict=each_loss_dict, step=epoch)

    def _output_learning_rate(self, epoch):
        """

        Args:
            epoch (int):

        """
        if self.model.scheduler is None:
            return

        lr_dict = {}
        for i, lr in enumerate(self.model.scheduler.get_last_lr()):
            lr_dict[f"Param{i}"] = lr

        self.board_writer.add_scalars(namespace="Learning_Rate", graph_name=self.mode.name, scalar_dict=lr_dict, step=epoch)

    # ==============================================================================
    #
    #   Special Attribute
    #
    # ==============================================================================

    def __call__(self, *args, **kwargs):
        for n_args, hook in self._forward_pre_hooks.values():
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

        output = self.forward_epoch(*args, **kwargs)

        if not isinstance(output, tuple):
            output = (output,)

        for n_args, hook in self._forward_hooks.values():
            if n_args == 0:
                hook_out = hook()
            else:
                hook_out = hook(self, (args, kwargs), output)
            if hook_out is not None:
                output = hook_out

        if len(output) == 1:
            output = output[0]

        return output
