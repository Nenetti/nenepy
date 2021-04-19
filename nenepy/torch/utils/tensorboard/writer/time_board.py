from nenepy.torch.utils.tensorboard import TensorBoardWriter


class TimeBoard:
    _NAMESPACE = "Time"

    def __init__(self, tensorboard, scalar_name):
        """

        Args:
            tensorboard (TensorBoardWriter):
            scalar_name (str):

        """
        self._tensorboard = tensorboard
        self._scalar_name = scalar_name
        self._total_elapsed_time = 0

    @property
    def total_elapsed_time(self):
        return self._total_elapsed_time

    def load_total_elapsed_time(self, total_elapsed_time):
        """

        Args:
            total_elapsed_time (float):

        """
        self._total_elapsed_time = total_elapsed_time

    def add_elapsed_time(self, elapsed_time, epoch):
        """

        Args:
            elapsed_time (float):
            epoch:

        """
        self._total_elapsed_time += elapsed_time
        time_chart_scalar_dict = {self._scalar_name: elapsed_time}
        elapsed_time_scalar_dict = {self._scalar_name: self._total_elapsed_time}
        self._tensorboard.add_scalars(namespace=self._NAMESPACE, graph_name="Time-Chart", scalar_dict=time_chart_scalar_dict, step=epoch)
        self._tensorboard.add_scalars(namespace=self._NAMESPACE, graph_name="Elapsed-Time", scalar_dict=elapsed_time_scalar_dict, step=epoch)
