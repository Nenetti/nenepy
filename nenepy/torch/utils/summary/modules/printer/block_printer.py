from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.printer.architecture_printer import ArchitecturePrinter
from nenepy.torch.utils.summary.modules.printer.input_printer import InputPrinter
from nenepy.torch.utils.summary.modules.printer.memory_printer import MemoryPrinter
from nenepy.torch.utils.summary.modules.printer.output_printer import OutputPrinter
from nenepy.torch.utils.summary.modules.printer.parameter_printer import ParameterPrinter
from nenepy.torch.utils.summary.modules.printer.time_printer import TimePrinter


class BlockPrinter(AbstractPrinter):
    max_architecture_length = 0
    max_input_length = 0
    max_output_length = 0
    max_parameter_length = 0
    max_time_length = 0

    def __init__(self, module):
        self.block = module
        self.architecture_printer = ArchitecturePrinter(module)
        self.input_printer = InputPrinter(module.input)
        self.output_printer = OutputPrinter(module.output)
        self.parameter_printer = ParameterPrinter(module.parameter)
        self.time_printer = TimePrinter(module.processing_time)
        self.memory_printer = MemoryPrinter(module)

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================
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

        architectures = [self.architecture_printer.to_top_formant()] * max_n_elements
        inputs = [""] * max_n_elements
        outputs = [""] * max_n_elements
        parameters = [self.parameter_printer.to_empty_format()] * max_n_elements
        times = [""] * max_n_elements
        if s == 1:
            architectures[0] = self.architecture_printer.to_bottom_format()
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
        return max([len(text) for text in texts], default=0)

    def calc_max_input_length(self):
        texts = self.input_printer.to_print_formats()
        return max([len(text) for text in texts], default=0)

    def calc_max_architecture_length(self):
        return len(self.architecture_printer.to_print_format())

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def to_adjust(cls, printers):
        InputPrinter.to_adjust([printer.input_printer for printer in printers])
        OutputPrinter.to_adjust([printer.output_printer for printer in printers])
        ParameterPrinter.to_adjust([printer.parameter_printer for printer in printers])
        TimePrinter.to_adjust([printer.time_printer for printer in printers])
        MemoryPrinter.to_adjust([printer.memory_printer for printer in printers])

        cls.max_architecture_length = max([cls.calc_max_architecture_length(printer) for printer in printers])
        cls.max_input_length = max([cls.calc_max_input_length(printer) for printer in printers])
        cls.max_output_length = max([cls.calc_max_output_length(printer) for printer in printers])
        cls.max_parameter_length = max([cls.calc_max_parameter_length(printer) for printer in printers])
        cls.max_time_length = max([cls.calc_max_time_length(printer) for printer in printers])

    @classmethod
    def to_parameter_headers(cls):
        weight_repr = ParameterPrinter.to_weight_repr()
        bias_repr = ParameterPrinter.to_bias_repr()
        train_repr = ParameterPrinter.to_train_repr()
        requires_grad_repr = ParameterPrinter.to_requires_grad_repr()

        line1 = f"{ParameterPrinter.parameter_repr:^{cls.max_parameter_length}}"
        line2 = cls.to_replace(line1, "-")
        line3 = f"{cls.to_empty(weight_repr)} │ {cls.to_empty(bias_repr)} │ {cls.to_empty(train_repr)} │ {requires_grad_repr}"
        line4 = f"{weight_repr} │ {bias_repr} │ {train_repr} │ {cls.to_replace(requires_grad_repr, '-')}"
        line5 = f"{cls.to_empty(weight_repr)} │ {cls.to_empty(bias_repr)} │ {cls.to_empty(train_repr)} │ {ParameterPrinter.to_requires_grad_bool_repr()}"

        return [line1, line2, line3, line4, line5]

    @classmethod
    def to_print_header(cls, reverse=False):
        architecture_repr = "Network Architecture"
        input_repr = "Input"
        output_repr = "Output"
        parameter_repr = ParameterPrinter.to_parameter_repr()
        time_repr = TimePrinter.to_time_repr()

        architecture_format = f"{architecture_repr:^{cls.max_architecture_length}}"
        input_format = f"{input_repr:^{cls.max_input_length}}"
        output_format = f"{output_repr:^{cls.max_output_length}}"
        parameter_format = f"{parameter_repr:^{cls.max_parameter_length}}"
        time_format = f"{time_repr:^{cls.max_time_length}}"

        empty_architecture_format = " " * cls.max_architecture_length
        empty_input_format = " " * cls.max_input_length
        empty_output_format = " " * cls.max_output_length
        empty_time_format = " " * cls.max_time_length

        bar_architecture_format = "=" * cls.max_architecture_length
        bar_input_format = "=" * cls.max_input_length
        bar_output_format = "=" * cls.max_output_length
        bar_parameter_format = "=" * cls.max_parameter_length
        bar_time_format = "=" * cls.max_time_length

        parameter_headers = cls.to_parameter_headers()

        bar = f"{bar_architecture_format}=│={bar_input_format}=│={bar_output_format}=│={bar_parameter_format}=│={bar_time_format}=│"

        line1 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {parameter_headers[0]} │ {empty_time_format} │"
        line2 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {parameter_headers[1]} │ {time_format} │"
        line3 = f"{architecture_format} │ {input_format} │ {output_format} │ {parameter_headers[2]} │ {empty_time_format} │"
        line4 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {parameter_headers[3]} │ {empty_time_format} │"
        line5 = f"{empty_architecture_format} │ {empty_input_format} │ {empty_output_format} │ {parameter_headers[4]} │ {empty_time_format} │"

        lines = [bar, line1, line2, line3, line4, line5, bar]
        # lines = [bar, line1, line2, line3, bar]

        if reverse:
            return "\n".join(reversed(lines))
        else:
            return "\n".join(lines)

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def to_print_footer():
        return MemoryPrinter.to_print_format()
