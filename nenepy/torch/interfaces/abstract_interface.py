import numpy as np

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
        self._time_board = TimeBoard(tensorboard, self._mode.name)
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
    @staticmethod
    def _to_numpy(*args):
        return (arg.numpy() for arg in args)

    @staticmethod
    def _to_statistical_dict(values):
        """

        Args:
            values (np.ndarray):

        """
        scalar_values = {
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "min": float(np.min(values))
        }
        return scalar_values

    def _output_each_class_evaluation(self, class_names, evaluation_dict, epoch):
        """
        Args:
            class_names (list[str])
            evaluation_dict (dict[str, np.ndarray]):
            epoch (int):

        Returns:

        """
        for evaluation_name, values in evaluation_dict.items():
            scalar_dict = dict([(class_names[i], each_class_value) for i, each_class_value in enumerate(values)])
            self._tensorboard.add_scalars(namespace="Evaluation", graph_name=evaluation_name, scalar_dict=scalar_dict, step=epoch)

        # for evaluation_name, values in evaluation_dict.items():
        #     for i, each_class_value in enumerate(values):
        #         statistical_dict = self._to_statistical_dict(each_class_value[i])
        #         self._tensorboard.add_scalars(namespace=f"Evaluation-{evaluation_name}", graph_name=class_names[i], scalar_dict=statistical_dict, step=epoch)

    def _output_loss(self, loss_dict, epoch):
        """
        Args:
            loss_dict (ListDict):
            epoch (int):

        Returns:

        """
        for name, values in loss_dict.items():
            statistical_dict = self._to_statistical_dict(values)
            self._tensorboard.add_scalars(namespace="Loss", graph_name=f"{self._mode.name}-{name}", scalar_dict=statistical_dict, step=epoch)

    def _output_learning_rate(self, epoch):
        """

        Args:
            epoch (int):

        """
        lr_dict = self._model.lr_dict
        if lr_dict is not None:
            self._tensorboard.add_scalars(namespace="Summary", graph_name="Learning-Rate", scalar_dict=lr_dict, step=epoch)

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
