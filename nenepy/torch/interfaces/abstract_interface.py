from abc import ABCMeta, abstractmethod

from nenepy.torch.interfaces import Mode
from nenepy.torch.models import AbstractModel
from nenepy.torch.utils.data import DataLoader
from nenepy.torch.utils.tensorboard.writer import TimeBoard
from nenepy.utils import Timer


class AbstractInterface(metaclass=ABCMeta):

    def __init__(self, log_dir, model, dataloader, mode, tensorboard):
        """

        Args:
            log_dir (str or pathlib.Path):
            model (AbstractModel):
            dataloader (DataLoader):
            mode (Mode):
            tensorboard (MultiProcessTensorBoardWriteManager):

        """

        self._mode = mode

        # ----- Data ----- #
        self._dataloader = dataloader

        # ----- Network Model ----- #
        self._model = model

        # ----- Log ----- #
        self._time_board = TimeBoard(log_dir, self._mode.name)
        self._tensorboard = tensorboard

        # ----- etc ----- #
        self._timer = Timer()

    @property
    def total_elapsed_time(self):
        return self._time_board.total_elapsed_time

    # ==================================================================================================
    #
    #   Abstract Method
    #
    # ==================================================================================================
    @abstractmethod
    def forward_epoch(self, epoch):
        raise NotImplementedError()

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    # ==================================================================================================
    #
    #   Private Method
    #
    # ==================================================================================================
    # def _output_loss(self, epoch, output_loss):
    #     """
    #
    #     Args:
    #         epoch (int):
    #         output_loss (ListDict):
    #
    #     """
    #     if len(output_loss) == 0:
    #         return
    #     for name, value in output_loss.items():
    #         value = np.array(value)
    #         scalar_values = {
    #             "max": float(np.max(value)),
    #             "mean": float(np.mean(value)),
    #             "min": float(np.min(value))
    #         }
    #
    #         self.board_writer.add_scalars(namespace=f"{self.mode.name} Loss", graph_name=name, scalar_dict=scalar_values, step=epoch)
    # #
    # def _output_learning_rate(self, epoch):
    #     """
    #
    #     Args:
    #         epoch (int):
    #
    #     """
    #     if self.model.scheduler is None and self.model.optimizer is None:
    #         return
    #
    #     lr_dict = {}
    #     if self.model.scheduler is not None:
    #         for i, lr in enumerate(self.model.scheduler.get_last_lr()):
    #             lr_dict[f"Param{i}"] = lr
    #     else:
    #         for i, group in enumerate(self.model.optimizer.param_groups):
    #             lr_dict[f"Param{i}"] = group["lr"]
    #
    #     self.board_writer.add_scalars(namespace="Summary", graph_name="Learning_Rate", scalar_dict=lr_dict, step=epoch)

    # ==================================================================================================
    #
    #   Special Attribute
    #
    # ==================================================================================================
    def __call__(self, epoch):
        self._timer.start()
        output = self.forward_epoch(epoch)
        self._timer.stop()

        self._time_board.add_elapsed_time(self._timer.elapsed_time, epoch)

        return output
    #
    # def _output_scalar(self, epoch, namespace, metric_name, scalar):
    #     self.board_writer.add_scalar(namespace=f"{namespace} {self.mode}", graph_name=metric_name, scalar_value=scalar, step=epoch)
    #
    # def _output_scalar_dict(self, epoch, namespace, metric_name, scalar_dict):
    #     self.board_writer.add_scalars(namespace=f"{namespace} {self.mode}", graph_name=metric_name, scalar_dict=scalar_dict, step=epoch)
