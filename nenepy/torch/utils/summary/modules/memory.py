import numpy as np

from .abstract_module import AbstractModule
from .output import Output


class Memory(AbstractModule):
    _total_n_param_repr = "Total Params"
    _total_weight_param_repr = "Total Weight Params"
    _total_bias_param_repr = "Total Bias Params"
    _trainable_params_repr = "Total Trainable Params"
    _non_trainable_params_repr = "Total Non-Trainable Params"
    _trainable_weight_params_repr = "Trainable Weight Params"
    _non_trainable_weight_params_repr = "Non-Trainable Weight Params"
    _trainable_bias_params_repr = "Trainable Bias Params"
    _non_trainable_bias_params_repr = "Non-Trainable Bias Params"

    _total_param_mb_repr = "Total params (MB)"
    _forward_size_per_batch_repr = "Forward size (Per Batch) (MB)"
    _total_forward_size_repr = "Forward size (All Batch) (MB)"
    _total_params_size_repr = "Total Params size (MB)"
    _total_size_repr = "Total size (MB)"

    _all_repr = [
        _total_n_param_repr, _total_param_mb_repr, _trainable_params_repr, _non_trainable_params_repr,
        _trainable_weight_params_repr, _non_trainable_weight_params_repr, _trainable_bias_params_repr, _non_trainable_bias_params_repr,
        _forward_size_per_batch_repr, _total_forward_size_repr, _total_params_size_repr, _total_size_repr
    ]

    _max_name_length = 0
    _max_value_length = 0

    _total_input_size = 0
    _total_output_size = 0
    _total_size = 0

    _trainable_weights = 0
    _non_trainable_weights = 0
    _trainable_biases = 0
    _non_trainable_biases = 0

    _total_params = 0
    _total_trainable_params = 0
    _total_non_trainable_params = 0
    _total_weights = 0
    _total_biases = 0

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def adjust(cls, modules):
        cls._total_output_size = cls._get_param_size(Output.get_all_tensors([module.output for module in modules]))

        params = np.sum([cls._get_parameters(module.parameter) for module in modules], axis=0)
        trainable_weights, non_trainable_weights, trainable_biases, non_trainable_biases = params
        cls._trainable_weights = trainable_weights
        cls._non_trainable_weights = non_trainable_weights
        cls._trainable_biases = trainable_biases
        cls._non_trainable_biases = non_trainable_biases

        cls._total_params = trainable_weights + non_trainable_weights + trainable_biases + non_trainable_biases
        cls._total_trainable_params = trainable_weights + trainable_biases
        cls._total_non_trainable_params = non_trainable_weights + non_trainable_biases
        cls._total_weights = trainable_weights + non_trainable_weights
        cls._total_biases = trainable_biases + non_trainable_biases
        cls._total_size = cls._total_output_size + cls._total_params

        cls._max_name_length = cls._get_max_text_length(cls._all_repr)
        cls._max_value_length = cls._get_max_text_length([str(cls._total_size), str(cls._total_output_size), str(cls._total_params)])

    @classmethod
    def to_print_format(cls):
        texts = [
            "",
            cls._value_to_text(cls._total_n_param_repr, cls._total_params),
            cls._value_to_text(cls._total_weight_param_repr, cls._total_weights),
            cls._value_to_text(cls._total_bias_param_repr, cls._total_biases),

            "",
            cls._value_to_text(cls._trainable_params_repr, cls._total_trainable_params),
            cls._value_to_text(cls._trainable_weight_params_repr, cls._trainable_weights),
            cls._value_to_text(cls._trainable_bias_params_repr, cls._trainable_biases),
            "",
            cls._value_to_text(cls._non_trainable_params_repr, cls._total_non_trainable_params),
            cls._value_to_text(cls._non_trainable_weight_params_repr, cls._non_trainable_weights),
            cls._value_to_text(cls._non_trainable_bias_params_repr, cls._non_trainable_biases),
            "",
            "",
            cls._value_to_mb_formats(cls._total_param_mb_repr, cls._total_params),
            cls._value_to_mb_formats(cls._forward_size_per_batch_repr, cls._total_output_size),
            cls._value_to_mb_formats(cls._total_forward_size_repr, cls._total_output_size),
            "",
            cls._value_to_mb_formats(cls._total_size_repr, cls._total_size),
        ]

        return "\n".join(texts)

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================

    @classmethod
    def _value_to_text(cls, text, value):
        return f"{text:>{cls._max_name_length}}: {value:>{cls._max_value_length},}"

    @classmethod
    def _value_to_mb_formats(cls, text, value):
        return f"{text:>{cls._max_name_length}}: {(value * 4) / (10 ** 6):>{cls._max_value_length},.2f}"

    @staticmethod
    def _get_param_size(tensors):
        return sum([np.prod(list(tensor.shape)) for tensor in set(tensors)])

    @staticmethod
    def _get_n_params(tensors):
        return sum([np.prod(list(tensor.shape)) for tensor in set(tensors)])

    @staticmethod
    def _get_parameters(parameter):
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
