from nenepy.torch.utils.summary.modules.printer.architecture_printer import ArchitecturePrinter
from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.printer.input_printer import InputPrinter
from nenepy.torch.utils.summary.modules.printer.output_printer import OutputPrinter
from nenepy.torch.utils.summary.modules.printer.parameter_printer import ParameterPrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict


class BlockPrinter(AbstractPrinter):
    max_architecture_length = 0
    max_input_length = 0
    max_output_length = 0
    max_parameter_length = 0

    def __init__(self, block):
        self.block = block
        self.architecture_printer = ArchitecturePrinter(block)
        self.input_printer = InputPrinter(block.module.input)
        self.output_printer = OutputPrinter(block.module.output)
        self.parameter_printer = ParameterPrinter(block.module.parameter)

    def to_adjust(self):
        self.set_max_architecture_length(self.architecture_printer)
        self.set_max_input_length(self.input_printer)
        self.set_max_output_length(self.output_printer)
        self.set_max_parameter_length(self.parameter_printer)

    @classmethod
    def to_print_header(cls):
        architecture_repr = "Network Architecture"
        input_repr = "Input"
        output_repr = "Output"
        parameter_repr = "Parameters"
        weight_repr = "Weight"
        bias_repr = "Bias"
        train_repr = "Train"
        untrain_repr = "Untrain"

        parameter_sub_format = f"{weight_repr} │ {bias_repr} │ {train_repr} │ {untrain_repr}"

        architecture_format = f"{architecture_repr:^{cls.max_architecture_length}}"
        input_format = f"{input_repr:^{cls.max_input_length}}"
        output_format = f"{output_repr:^{cls.max_output_length}}"
        parameter_format = f"{parameter_repr:^{cls.max_parameter_length}}"

        empty_architecture_format = " " * cls.max_architecture_length
        empty_input_format = " " * cls.max_input_length
        empty_output_format = " " * cls.max_output_length
        empty_parameter_format = " " * cls.max_parameter_length

        bar_architecture_format = "=" * cls.max_architecture_length
        bar_input_format = "=" * cls.max_input_length
        bar_output_format = "=" * cls.max_output_length
        bar_parameter_format = "=" * cls.max_parameter_length

        bar = f"{bar_architecture_format}=│={bar_input_format}=│={bar_output_format}=│={bar_parameter_format}=│"
        title = f"{architecture_format} │ {input_format} │ {output_format} │ {parameter_format} │ "
        sub_title = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │  {parameter_sub_format} │"
        empty = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {empty_parameter_format} │"

        return "\n".join([bar, empty, title, sub_title, empty, bar])

    # def to_repeat_text(self, char, n):
    #     return char * n

    def to_print_format(self):
        print_formats = []

        architecture_format = self.architecture_printer.to_print_format()
        input_formats = self.input_printer.to_print_formats()
        output_formats = self.output_printer.to_print_formats()
        parameter_format = self.parameter_printer.to_print_format()

        max_n_elements = max([len(input_formats), len(output_formats)])

        architectures = [self.architecture_printer.to_child_formant(self.block)] * max_n_elements
        inputs = [""] * max_n_elements
        outputs = [""] * max_n_elements
        parameters = [""] * max_n_elements

        architectures[0] = architecture_format
        inputs[:len(input_formats)] = input_formats
        outputs[:len(output_formats)] = output_formats
        parameters[0] = parameter_format

        for i in range(max_n_elements):
            architecture_format = f"{architectures[i]:<{self.max_architecture_length}}"
            input_format = f"{inputs[i]:<{self.max_input_length}}"
            output_format = f"{outputs[i]:<{self.max_output_length}}"
            parameter_format = f"{parameters[i]}"

            print_format = f"{architecture_format} │ {input_format} │ {output_format} │ {parameter_format} │"
            print_formats.append(print_format)

        return print_formats

    @classmethod
    def set_max_parameter_length(cls, parameter_printer):
        length = len(parameter_printer.to_print_format())
        if length > cls.max_parameter_length:
            cls.max_parameter_length = length

    @classmethod
    def set_max_output_length(cls, output_printer):
        formatted_texts = output_printer.to_print_formats()
        for text in formatted_texts:
            length = len(text)
            if length > cls.max_output_length:
                cls.max_output_length = length

    @classmethod
    def set_max_input_length(cls, input_printer):
        formatted_texts = input_printer.to_print_formats()
        for text in formatted_texts:
            length = len(text)
            if length > cls.max_input_length:
                cls.max_input_length = length

    @classmethod
    def set_max_architecture_length(cls, architecture_printer):
        length = len(architecture_printer.to_print_format())
        if length > cls.max_architecture_length:
            cls.max_architecture_length = length

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
