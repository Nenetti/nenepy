from nenepy.torch.utils.summary.modules.parameter import Parameter
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter


class TimePrinter(AbstractPrinter):
    time_repr = "Time (ms)"

    max_length = len(time_repr)

    def __init__(self, time):
        """

        Args:
            parameter (Parameter):

        """
        self.time = time

    def to_print_format(self):
        time_str = self.to_time_str(self.time)
        time_format = f"{time_str:>{self.max_length}}"
        return time_format

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def to_adjust(cls, printers):
        cls.max_length = max([cls.max_length, max([cls.calc_max_length(printer.time) for printer in printers])])

    @classmethod
    def to_time_repr(cls):
        return f"{cls.time_repr:^{cls.max_length}}"

    @classmethod
    def calc_max_length(cls, time):
        """

        Args:
            time (int)

        Returns:

        """
        return len(cls.to_time_str(time))

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    @staticmethod
    def to_time_str(time):
        t = time * 1000
        if int(t) > 0:
            return f"{t:.2f}"
        else:
            return f"{t:.2f}"[1:]
