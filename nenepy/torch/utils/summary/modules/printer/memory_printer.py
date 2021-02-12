import torch

from nenepy.torch.utils.summary.modules.input import Input
from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter
from nenepy.torch.utils.summary.modules.value import Value
from nenepy.torch.utils.summary.modules.value_list import ValueList
from nenepy.torch.utils.summary.modules.value_dict import ValueDict
from nenepy.torch.utils.summary.modules.parameter import Parameter
import numpy as np


class MemoryPrinter(AbstractPrinter):
    _total_param_repr = "Total params"
    _trainable_params_repr = "Trainable params"
    _non_trainable_params_repr = "Non-Trainable params"
    _trainable_weight_params_repr = "Trainable weight params"
    _non_trainable_weight_params_repr = "Non-Trainable weight params"
    _trainable_bias_params_repr = "Trainable bias params"
    _non_trainable_bias_params_repr = "Non-Trainable bias params"

    _input_size_per_batch_repr = "Input size (/Batch) (MB)"
    _forward_size_per_batch_repr = "Forward size (/Batch) (MB)"
    _total_input_size_repr = "Total Input size (MB)"
    _total_forward_size_repr = "Total Forward size (MB)"
    _total_params_size_repr = "Total Params size (MB)"
    _total_size_repr = "Total size (MB)"

    _max_name_length = 0
    _max_value_length = 0

    _total_input_size = 0
    _total_output_size = 0
    _total_size = 0

    _trainable_weight = 0
    _non_trainable_weight = 0
    _trainable_bias = 0
    _non_trainable_bias = 0

    _total_param = 0
    _total_trainable_param = 0
    _total_non_trainable_param = 0
    _total_weight = 0
    _total_bias = 0

    def __init__(self, module):
        self.module = module
        pass

    @classmethod
    def get_all_tensors(cls, printers):
        input_tensors = []
        output_tensors = []
        for printer in printers:
            input_tensors += cls._get_all_tensors(printer.module.input.values)
            output_tensors += cls._get_all_tensors(printer.module.output.values)

        return input_tensors, output_tensors

    @classmethod
    def init(cls, input_tensors, output_tensors):
        total_input_size = cls.get_param_size(input_tensors)
        total_output_size = cls.get_param_size(output_tensors)
        cls._total_input_size += total_input_size
        cls._total_output_size += total_output_size
        cls._total_size += total_input_size + total_output_size

    @classmethod
    def init2(cls, module):

        trainable_weight, non_trainable_weight, trainable_bias, non_trainable_bias = cls.get_parameters(module.parameter)
        cls._trainable_weight += trainable_weight
        cls._non_trainable_weight += non_trainable_weight
        cls._trainable_bias += trainable_bias
        cls._non_trainable_bias += non_trainable_bias

        cls._total_param += trainable_weight + non_trainable_weight + trainable_bias + non_trainable_bias
        cls._total_trainable_param += trainable_weight + trainable_bias
        cls._total_non_trainable_param += non_trainable_weight + non_trainable_bias
        cls._total_weight += trainable_weight + non_trainable_weight
        cls._total_bias += trainable_bias + non_trainable_bias

    @classmethod
    def to_adjust(cls, printers):
        input_tensors, output_tensors = cls.get_all_tensors(printers)
        cls.init(input_tensors, output_tensors)
        [cls.init2(printer.module) for printer in printers]
        cls._max_name_length = cls.get_max_text_length([
            cls._total_param_repr, cls._trainable_params_repr, cls._non_trainable_params_repr, cls._input_size_per_batch_repr,
            cls._trainable_weight_params_repr, cls._non_trainable_weight_params_repr, cls._trainable_bias_params_repr, cls._non_trainable_bias_params_repr,
            cls._forward_size_per_batch_repr, cls._total_input_size_repr, cls._total_forward_size_repr, cls._total_params_size_repr, cls._total_size_repr
        ])
        cls._max_value_length = max([
            cls.get_max_text_length([str(printer._total_size), str(printer._total_output_size), str(printer._total_param)]) for printer in
            printers
        ])

    @classmethod
    def to_print_format(cls):
        texts = [
            cls.to_value_format(cls._total_param_repr, cls._total_param),
            "",
            cls.to_value_format(cls._trainable_params_repr, cls._total_trainable_param),
            cls.to_value_format(cls._non_trainable_params_repr, cls._total_non_trainable_param),
            cls.to_value_format(cls._trainable_weight_params_repr, cls._trainable_weight),
            cls.to_value_format(cls._non_trainable_weight_params_repr, cls._non_trainable_weight),
            cls.to_value_format(cls._trainable_bias_params_repr, cls._trainable_bias),
            cls.to_value_format(cls._non_trainable_bias_params_repr, cls._non_trainable_bias),
            "",
            cls.to_value_format(cls._input_size_per_batch_repr, cls._total_input_size),
            cls.to_value_format(cls._forward_size_per_batch_repr, cls._total_output_size),
            "",
            cls.to_value_format(cls._total_input_size_repr, cls._total_input_size),
            cls.to_value_format(cls._total_forward_size_repr, cls._total_output_size),
            cls.to_value_format(cls._total_size_repr, cls._total_param),
        ]

        return "\n".join(texts)

    @classmethod
    def to_value_format(cls, text, value):
        return f"{text:>{cls._max_name_length}}: {(value * 4) / (10 ** 6):>{cls._max_value_length},.2f}"

    @staticmethod
    def get_param_size(tensors):
        if len(tensors) > 0:
            print(len(tensors), len(set(tensors)))
            return sum([np.prod(list(tensor.shape)) for tensor in set(tensors)])
        return 0

    @staticmethod
    def get_parameters(parameter):
        trainable_weight = 0
        non_trainable_weight = 0
        trainable_bias = 0
        non_trainable_bias = 0

        if parameter.has_weight:
            if parameter.weight_requires_grad:
                trainable_weight = parameter.n_weight_params
            else:
                non_trainable_weight = parameter.n_weight_params

        if parameter.has_bias:
            if parameter.bias_requires_grad:
                trainable_bias = parameter.n_bias_params
            else:
                non_trainable_bias = parameter.n_bias_params

        return trainable_weight, non_trainable_weight, trainable_bias, non_trainable_bias

    @staticmethod
    def get_max_text_length(texts):
        if len(texts) > 0:
            return max([len(text) for text in texts])
        return 0

    @classmethod
    def _get_all_tensors(cls, values):
        def recursive(value):
            if isinstance(value, Value):
                if value.is_tensor:
                    return value.value
                return []
            elif isinstance(value, (ValueList, ValueDict)):
                x = []
                for v in value.values:
                    y = recursive(v)
                    if isinstance(y, list):
                        x += y
                    else:
                        x.append(y)
                return x
            else:
                raise TypeError()

        return [*recursive(values)]
