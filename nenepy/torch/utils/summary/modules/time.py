from nenepy.torch.utils.summary.modules import AbstractModule


class Time(AbstractModule):
    time_repr = "Time (ms)"

    max_length = len(time_repr)

    def __init__(self, time):
        """

        Args:

        """
        super(Time, self).__init__()
        self.time = time

    @property
    def print_formats(self):
        time_str = self._to_time_str(self.time)
        time_format = f"{time_str:>{self.max_length}}"
        return time_format

    def to_print_format(self):
        time_str = self._to_time_str(self.time)
        time_format = f"{time_str:>{self.max_length}}"
        return time_format

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def to_adjust(cls, modules):
        cls.max_length = max([cls.max_length, max([cls._calc_max_length(module.processing_time) for module in modules])])

    @classmethod
    def to_time_repr(cls):
        return f"{cls.time_repr:^{cls.max_length}}"

    @classmethod
    def _calc_max_length(cls, time):
        """

        Args:
            time (int)

        Returns:

        """
        return len(cls._to_time_str(time))

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    @staticmethod
    def _to_time_str(time):
        t = time * 1000
        if int(t) > 0:
            return f"{t:.2f}"
        else:
            return f"{t:.2f}"[1:]
