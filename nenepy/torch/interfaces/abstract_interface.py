from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from inspect import signature
from pathlib import Path

import numpy as np

from nenepy.torch.interfaces import Mode
from nenepy.torch.utils.tensorboard import TensorBoardWriter
from nenepy.utils import Timer, Logger
from nenepy.utils.dictionary import ListDict


class AbstractInterface(metaclass=ABCMeta):

    def __init__(self, mode, model, logger, save_interval, save_multi_process=False, n_processes=1):
        """

        Args:
            mode (Mode):
            model (nenepy.torch.models.AbstractModel):
            logger (Logger):
            save_interval (int):

        """
        # ----- Data ----- #
        self.dataset = None
        self.dataloader = None

        # ----- Network Model ----- #
        self.model = model

        # ----- Log ----- #
        self.logger = logger
        self.board_writer = TensorBoardWriter(log_dir=Path(logger.log_dir).joinpath(mode.name), multi_process=save_multi_process, n_processes=n_processes)

        # ----- etc ----- #
        self.mode = mode
        self.timer = Timer()
        self.save_interval = save_interval

        self.epoch = 0
        self.log_time_key = f"{mode.name}_ELAPSED_TIME"
        self.log_average_time_key = f"{mode.name}_AVERAGE_ELAPSED_TIME"
        self.log_epoch_key = f"{mode.name}_NUM_EPOCH"
        self._init_log()

        # self._forward_pre_hooks = OrderedDict()
        # self._forward_hooks = OrderedDict()
        # self.register_forward_pre_hook(self._default_pre_hook)
        # self.register_forward_hook(self._default_hook)

    def _init_log(self):
        if self.log_time_key not in self.logger:
            self.logger[self.log_time_key] = 0

    # ==================================================================================================
    #
    #   Abstract Method
    #
    # ==================================================================================================

    @abstractmethod
    def forward_epoch(self, *args, **kwargs):
        raise NotImplementedError()

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    def load_log(self):
        self.epoch = self.logger[self.log_epoch_key]

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================

    def _pre_process(self):
        self.epoch += 1
        if self.mode is Mode.TRAIN:
            self.model.train_mode()
        elif self.mode is Mode.VALIDATE:
            self.model.validate_mode()
        else:
            self.model.test_mode()

        self.timer.start()

    def _post_process(self):
        self.timer.stop()
        self.model.scheduler_step()

        self.logger[self.log_epoch_key] = self.epoch
        self.logger[self.log_time_key] += self.timer.elapsed_time
        self.logger[self.log_average_time_key] = self.logger[self.log_time_key] / self.epoch

        if (self.mode is Mode.TRAIN) and (self.epoch % self.save_interval == 0):
            self.logger.save()
            self.model.save_weight()
            print(f"Model and Log Saved. Next Save Epoch: {(self.epoch + self.save_interval)}")

    def _output_time(self, epoch):
        """

        Args:
            epoch (int):

        """
        scalar_dict = {self.mode.name: self.timer.elapsed_time}
        self.board_writer.add_scalars(namespace="Summary", graph_name="Each Epoch Time", scalar_dict=scalar_dict, step=epoch)

        scalar_dict = {self.mode.name: self.logger[self.log_time_key]}
        self.board_writer.add_scalars(namespace="Summary", graph_name="Elapsed Time", scalar_dict=scalar_dict, step=epoch)

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

            self.board_writer.add_scalars(namespace=f"Loss_{self.mode.name}", graph_name=name, scalar_dict=scalar_values, step=epoch)

    def _output_learning_rate(self, epoch):
        """

        Args:
            epoch (int):

        """
        if self.model.scheduler is None and self.model.optimizer is None:
            return

        lr_dict = {}
        if self.model.scheduler is not None:
            for i, lr in enumerate(self.model.scheduler.get_last_lr()):
                lr_dict[f"Param{i}"] = lr
        else:
            for i, group in enumerate(self.model.optimizer.param_groups):
                lr_dict[f"Param{i}"] = group["lr"]

        self.board_writer.add_scalars(namespace="Summary", graph_name="Learning_Rate", scalar_dict=lr_dict, step=epoch)

    def _output_scalar(self, epoch, namespace, metric_name, scalar):
        self.board_writer.add_scalar(namespace=namespace, graph_name=f"{self.mode.name}/{metric_name}", scalar_value=scalar, step=epoch)

    def _output_scalar_dict(self, epoch, namespace, metric_name, scalar_dict):
        self.board_writer.add_scalars(namespace=namespace, graph_name=f"{self.mode.name}/{metric_name}", scalar_dict=scalar_dict, step=epoch)

    # ==================================================================================================
    #
    #   Special Attribute
    #
    # ==================================================================================================

    def __call__(self, *args, **kwargs):
        self._pre_process()

        output = self.forward_epoch(*args, **kwargs)

        self._post_process()

        return output

    def wait_process_completed(self):
        self.board_writer.wait_process_completed()
