from nenepy.torch.utils.summary.modules.printer.architecture_printer import ArchitecturePrinter
from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.printer.input_printer import InputPrinter
from nenepy.torch.utils.summary.modules.printer.output_printer import OutputPrinter
from nenepy.torch.utils.summary.modules.printer.parameter_printer import ParameterPrinter
from nenepy.torch.utils.summary.modules.printer.time_printer import TimePrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict


class BlockPrinter(AbstractPrinter):
    max_architecture_length = 0
    max_input_length = 0
    max_output_length = 0
    max_parameter_length = 0
    max_time_length = 0

    def __init__(self, block):
        self.block = block
        self.architecture_printer = ArchitecturePrinter(block)
        self.input_printer = InputPrinter(block.module.input)
        self.output_printer = OutputPrinter(block.module.output)
        self.parameter_printer = ParameterPrinter(block.module.parameter)
        self.time_printer = TimePrinter(block.processing_time)

    @classmethod
    def to_adjust(cls, printers):
        InputPrinter.to_adjust([printer.input_printer for printer in printers])
        OutputPrinter.to_adjust([printer.output_printer for printer in printers])
        ParameterPrinter.to_adjust([printer.parameter_printer for printer in printers])
        TimePrinter.to_adjust([printer.time_printer for printer in printers])

        cls.max_architecture_length = max([cls.calc_max_architecture_length(printer) for printer in printers])
        cls.max_input_length = max([cls.calc_max_input_length(printer) for printer in printers])
        cls.max_output_length = max([cls.calc_max_output_length(printer) for printer in printers])
        cls.max_parameter_length = max([cls.calc_max_parameter_length(printer) for printer in printers])
        cls.max_time_length = max([cls.calc_max_time_length(printer) for printer in printers])

    @classmethod
    def to_print_header(cls):
        architecture_repr = "Network Architecture"
        input_repr = "Input"
        output_repr = "Output"
        parameter_repr = ParameterPrinter.to_parameter_repr()
        weight_repr = ParameterPrinter.to_weight_repr()
        bias_repr = ParameterPrinter.to_bias_repr()
        train_repr = ParameterPrinter.to_train_repr()
        untrain_repr = ParameterPrinter.to_untrain_repr()
        time_repr = TimePrinter.to_time_repr()

        parameter_sub_format = f"{weight_repr}   {bias_repr}   {train_repr}   {untrain_repr}"

        architecture_format = f"{architecture_repr:^{cls.max_architecture_length}}"
        input_format = f"{input_repr:^{cls.max_input_length}}"
        output_format = f"{output_repr:^{cls.max_output_length}}"
        parameter_format = f"{parameter_repr:^{cls.max_parameter_length}}"
        time_format = f"{time_repr:^{cls.max_time_length}}"

        empty_architecture_format = " " * cls.max_architecture_length
        empty_input_format = " " * cls.max_input_length
        empty_output_format = " " * cls.max_output_length
        empty_parameter_format = " " * cls.max_parameter_length
        empty_time_format = " " * cls.max_time_length

        bar_architecture_format = "=" * cls.max_architecture_length
        bar_input_format = "=" * cls.max_input_length
        bar_output_format = "=" * cls.max_output_length
        bar_parameter_format = "=" * cls.max_parameter_length
        bar_time_format = "=" * cls.max_time_length

        bar = f"{bar_architecture_format}=│={bar_input_format}=│={bar_output_format}=│={bar_parameter_format}=│={bar_time_format}=│"
        line1 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {parameter_format} │ {empty_time_format} │"
        line2 = f"{architecture_format} │ {input_format} │ {output_format} │ {empty_parameter_format} │ {time_format} │"
        line3 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │  {parameter_sub_format} │ {empty_time_format} │"

        return "\n".join([bar, line1, line2, line3, bar])

    def to_print_format(self):
        print_formats = []

        architecture_format = self.architecture_printer.to_print_format()
        input_formats = self.input_printer.to_print_formats()
        output_formats = self.output_printer.to_print_formats()
        parameter_format = self.parameter_printer.to_print_format()
        time_format = self.time_printer.to_print_format()

        max_n_elements = max([len(input_formats), len(output_formats)])
        s = 0
        if max_n_elements > 1 or self.block.has_children():
            s = 1
            max_n_elements += 2

        architectures = [self.architecture_printer.to_child_formant(self.block)] * max_n_elements
        inputs = [""] * max_n_elements
        outputs = [""] * max_n_elements
        parameters = [self.parameter_printer.to_empty_format()] * max_n_elements
        times = [""] * max_n_elements
        if s == 1:
            architectures[0] = self.architecture_printer.to_parent_formant2(self.block)
        architectures[s] = architecture_format
        inputs[s:s + len(input_formats)] = input_formats
        outputs[s:s + len(output_formats)] = output_formats
        parameters[s] = parameter_format
        times[s] = time_format

        for i in range(max_n_elements):
            architecture_format = f"{architectures[i]:<{self.max_architecture_length}}"
            input_format = f"{inputs[i]:<{self.max_input_length}}"
            output_format = f"{outputs[i]:<{self.max_output_length}}"
            parameter_format = f"{parameters[i]:<{self.max_parameter_length}}"
            time_format = f"{times[i]:<{self.max_time_length}}"

            print_format = f"{architecture_format} │ {input_format} │ {output_format} │ {parameter_format} │ {time_format} │"
            print_formats.append(print_format)

        return print_formats

    def calc_max_time_length(self):
        return len(self.time_printer.to_print_format())

    def calc_max_parameter_length(self):
        return len(self.parameter_printer.to_print_format())

    def calc_max_output_length(self):
        texts = self.output_printer.to_print_formats()
        if len(texts) > 0:
            return max([len(text) for text in texts])
        else:
            return 0

    def calc_max_input_length(self):
        texts = self.input_printer.to_print_formats()
        if len(texts) > 0:
            return max([len(text) for text in texts])
        else:
            return 0

    def calc_max_architecture_length(self):
        return len(self.architecture_printer.to_print_format())

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
