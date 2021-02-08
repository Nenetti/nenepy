from nenepy.torch.utils.summary.modules.printer.architecture_printer import ArchitecturePrinter
from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.printer.input_printer import InputPrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict


class BlockPrinter(AbstractPrinter):

    def __init__(self, block):
        self.block = block
        self.architecture_print = ArchitecturePrinter(block)
        # self.architecture_text = self.block.architecture.print_format
        self.input_printer = InputPrinter(block.module.input)

    def to_print_format(self):
        architecture_format = self.architecture_print.to_print_format()
        input_format = self.input_printer.to_print_format()
        # return f"{architecture_format} : {input_format}"
        return f"{input_format}"

    @classmethod
    def calc(cls, blocks):
        InputPrinter.calc(blocks)

    @staticmethod
    def input_to_print_format(input):
        """

        Args:
            input (Input):

        Returns:

        """
        input_arguments = input.values.values
        for key, value in input_arguments.items():
            pass

    @classmethod
    def get_input_max_tensor_length(cls, blocks):
        """

        Args:
            blocks (list[Block]):

        Returns:

        """
        return max([cls.get_max_tensor_length_recursive(block.module.input.values) for block in blocks])

    @classmethod
    def get_output_max_tensor_length(cls, blocks):
        """

        Args:
            blocks (list[Block]):

        Returns:

        """
        return max([cls.get_max_tensor_length_recursive(block.module.output.values) for block in blocks])

    @classmethod
    def get_max_tensor_length_recursive(cls, value):
        if isinstance(value, Value):
            if value.is_tensor:
                return len(value.shapes)

        elif isinstance(value, ValueList):
            if len(value.values) > 0:
                return max([cls.get_max_tensor_length_recursive(v) for v in value.values])

        elif isinstance(value, ValueDict):
            if len(value.values) > 0:
                return max([cls.get_max_tensor_length_recursive(v) for v in value.values])
        else:
            raise TypeError()

        return 0

    @staticmethod
    def calc(printers):
        """

        Args:
            printers (list[BlockPrinter]):

        Returns:

        """
