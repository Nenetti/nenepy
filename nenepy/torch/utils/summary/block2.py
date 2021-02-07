import time
from collections import Counter
from numbers import Number
from typing import List, Any

import numpy as np
import torch


class Block:
    total_params = 0

    input_max_tensor_dims = 0
    input_each_max_dims = 0
    input_coefficient_length = 0

    output_max_tensor_dims = 0
    output_each_max_dims = 0
    output_coefficient_length = 0

    architecture_length = 0
    input_shape_length = 0
    input_arg_length = 0
    output_shape_length = 0
    param_length = 0
    param_per_length = 0
    weight_param_length = 0
    bias_param_length = 0
    train_length = 0
    untrain_length = 0
    time_length = 0

    all_blocks = []
    ids = dict()

    def __init__(self, module):
        self.module = module
        self.module_in = None
        self.module_out = None
        self.blocks = []
        self.input_kwargs = None
        self.output_kwargs = None
        self.depth = 0
        self.bottom = False

        self.is_trained_weight = False
        self.trained_weight_params = 0
        self.untrained_weight_params = 0

        self.is_trained_bias = False
        self.trained_bias_params = 0
        self.untrained_bias_params = 0

        self.processing_time = 0
        self.start_time = time.time()

        self.duplication = False

        self.is_training = False

        module_id = id(self.module)
        if module_id in self.ids:
            self.duplication = True
        else:
            self.ids[module_id] = module
        self.all_blocks.append(self)

    def add_block(self, block):
        self.blocks.append(block)

    def add_weight_params(self, n_params, requires_grad):
        self.is_trained_weight = requires_grad

        if requires_grad:
            self.trained_weight_params += n_params
        else:
            self.untrained_weight_params += n_params

    def add_bias_params(self, n_params, requires_grad):
        self.is_trained_bias = requires_grad

        if requires_grad:
            self.trained_bias_params += n_params
        else:
            self.untrained_bias_params += n_params

    def add_is_training(self, is_training):
        self.is_training = is_training

    # Property
    @property
    def input_size(self):
        return sum([size_str.size() for size_str in self.input_kwargs.values()])

    @property
    def output_size(self):
        size = 0
        if (self.n_params > 0) and (len(self.blocks) == 0):
            for size_str in self.output_kwargs.values():
                size += size_str.size()

        return size

    @property
    def weight_n_params(self):
        return self.trained_weight_params + self.untrained_weight_params

    @property
    def bias_n_params(self):
        return self.trained_bias_params + self.untrained_bias_params

    @property
    def n_params(self):
        return self.trained_weight_params + self.untrained_weight_params + self.trained_bias_params + self.untrained_bias_params

    @property
    def is_trained(self):
        is_train = (self.is_trained_weight or self.is_trained_bias) and self.is_training
        has_param = self.n_params > 0
        if is_train and has_param:
            return "✓"
        else:
            return ""

    @property
    def is_untrained(self):
        is_train = (self.is_trained_weight or self.is_trained_bias) and self.is_training
        has_param = self.n_params > 0
        if not is_train and has_param:
            return "✓"
        else:
            return ""

    @property
    def architecture_str(self):
        return str(self.module.__class__).split(".")[-1].split("'")[0]

    @property
    def param_str(self):
        if self.n_params == 0:
            return ""
        else:
            return f"{self.n_params:,}"

    @property
    def weight_param_str(self):
        n_params = self.trained_weight_params + self.untrained_weight_params
        if n_params == 0:
            return ""
        else:
            return f"{n_params:,}"

    @property
    def bias_param_str(self):
        n_params = self.trained_bias_params + self.untrained_bias_params
        if n_params == 0:
            return ""
        else:
            return f"{n_params:,}"

    @property
    def param_per_str(self):
        if self.n_params == 0:
            return ""
        else:
            per = (self.n_params / self.total_params) * 100
            text = f"{per:.1f}"
            if text == "0.0":
                text = "0"
            return text

    @property
    def input_str_args(self):
        if len(self.input_kwargs) > 1:
            return [f"{arg}" for arg in self.input_kwargs.keys()]
        else:
            return [""]

    @property
    def input_str_shapes(self):
        return [f"{size_str.tensors_to_str(self.input_each_max_dims)}" for size_str in self.input_kwargs.values()]

    @property
    def input_str_coefficients(self):
        return [f"{size_str.coefficient}" for size_str in self.input_kwargs.values()]

    @property
    def output_str_shapes(self):
        return [f"{size_str.tensors_to_str(self.output_each_max_dims)}" for size_str in self.output_kwargs.values()]

    @property
    def output_str_coefficients(self):
        return [f"{size_str.coefficient}" for size_str in self.output_kwargs.values()]

    # Text

    @property
    def architecture_text(self):
        return f"{self.architecture_str:<{self.architecture_length}}"

    @property
    def param_per_text(self):
        return f"{self.param_per_str:>{self.param_per_length}}"

    @property
    def param_text(self):
        return f"{self.param_str:>{self.param_length}} {self.param_per_str:>{self.param_per_length}}"

    @property
    def weight_param_text(self):
        return f"{self.weight_param_str:>{self.weight_param_length}}"

    @property
    def bias_param_text(self):
        return f"{self.bias_param_str:>{self.bias_param_length}}"

    @property
    def is_train_text(self):
        return f"{self.is_trained:^{self.train_length}}"

    @property
    def is_untrain_text(self):
        return f"{self.is_untrained:^{self.untrain_length}}"

    @property
    def time_text(self):
        t = f"{self.processing_time * 1000:.0f}"
        return f"{t:^{self.time_length}}"

    @property
    def input_texts(self):
        args = self.input_str_args
        shapes = self.input_str_shapes
        coefficient = self.input_str_coefficients
        texts = [None] * len(shapes)
        for i in range(len(shapes)):
            if args[i] != "":
                text = f"{args[i]:>{self.input_arg_length}}: {shapes[i]:<{self.input_shape_length}}"
            else:
                text = f"{args[i]:>{self.input_arg_length}}  {shapes[i]:<{self.input_shape_length}}"

            if self.input_coefficient_length == 0:
                texts[i] = f"{text}"
            elif coefficient[i] != "":
                texts[i] = f"{text} * {coefficient[i]:<{self.input_coefficient_length}}"
            else:
                texts[i] = f"{text}   {coefficient[i]:<{self.input_coefficient_length}}"
        return texts

    @property
    def output_texts(self):
        shapes = self.output_str_shapes
        coefficient = self.output_str_coefficients
        texts = [None] * len(shapes)
        for i in range(len(shapes)):
            if self.output_coefficient_length == 0:
                texts[i] = f"{shapes[i]:<{self.output_shape_length}}"
            elif coefficient[i] != "":
                texts[i] = f"{shapes[i]:<{self.output_shape_length}} * {coefficient[i]:<{self.output_coefficient_length}}"
            else:
                texts[i] = f"{shapes[i]:<{self.output_shape_length}}   {coefficient[i]:<{self.output_coefficient_length}}"
        return texts

    # classmethod
    @classmethod
    def get_total_params(cls):
        return cls.get_total_trainable_params() + cls.get_total_untrainable_params()

    @classmethod
    def get_total_trainable_params(cls):
        params = 0
        for block in cls.all_blocks:
            if not block.duplication:
                params += block.trained_weight_params + block.trained_bias_params
        return params

    @classmethod
    def get_total_untrainable_params(cls):
        params = 0
        for block in cls.all_blocks:
            if not block.duplication:
                params += block.untrained_weight_params + block.untrained_bias_params
        return params

    @classmethod
    def get_input_length(cls):
        length = cls.input_arg_length + cls.input_shape_length + cls.input_coefficient_length
        if cls.input_arg_length > 0:
            length += len(": ")
        if cls.input_coefficient_length > 0:
            length += len(" * ")

        return length

    @classmethod
    def get_output_length(cls):
        length = cls.output_shape_length + cls.output_coefficient_length
        if cls.output_coefficient_length > 0:
            length += len(" * ")
        return length

    @classmethod
    def calc_length(cls):

        cls.total_params = cls.get_total_params()

        cls.input_max_tensor_dims = cls._get_input_max_tensor_dims()
        cls.input_each_max_dims = cls._get_input_each_dim_max_size()
        cls.input_coefficient_length = cls._get_input_max_coefficient()

        cls.output_max_tensor_dims = cls._get_output_max_tensor_dims()
        cls.output_each_max_dims = cls._get_output_each_dim_max_size()
        cls.output_coefficient_length = cls._get_output_max_coefficient()

        cls.input_shape_length = cls._get_max_input_shape_length()
        cls.input_arg_length = cls._get_max_input_args_length()
        cls.output_shape_length = cls._get_max_output_length()
        cls.param_length = cls._get_max_param_length()
        cls.param_per_length = cls._get_max_param_per_length()
        cls.weight_param_length = cls._get_max_weight_param_per_length()
        cls.bias_param_length = cls._get_max_bias_param_per_length()
        cls.train_length = cls._get_max_train_length()
        cls.untrain_length = cls._get_max_untrain_length()
        cls.time_length = cls._get_max_time_length()

    @classmethod
    def _get_input_max_tensor_dims(cls):
        max_dims = 0
        for block in cls.all_blocks:
            for size_str in block.input_kwargs.values():
                if size_str.is_tensor:
                    max_dims = max(max_dims, size_str.n_dims)

        return max_dims

    @classmethod
    def _get_output_max_tensor_dims(cls):
        max_dims = 0
        for block in cls.all_blocks:
            for size_str in block.output_kwargs.values():
                if size_str.is_tensor:
                    max_dims = max(max_dims, size_str.n_dims)

        return max_dims

    @classmethod
    def _get_input_each_dim_max_size(cls):
        dims = [0 for _ in range(cls.input_max_tensor_dims)]
        for block in cls.all_blocks:
            for size_str in block.input_kwargs.values():
                if size_str.is_tensor:
                    tensors = size_str.tensors
                    for i in range(len(tensors)):
                        dims[i] = max(dims[i], len(tensors[i]))

        return dims

    @classmethod
    def _get_input_max_coefficient(cls):
        max_dims = 0
        for block in cls.all_blocks:
            for size_str in block.input_kwargs.values():
                length = len(size_str.coefficient)
                max_dims = max(max_dims, length)

        return max_dims

    @classmethod
    def _get_output_each_dim_max_size(cls):
        dims = [0 for _ in range(cls.output_max_tensor_dims)]
        for block in cls.all_blocks:
            for size_str in block.output_kwargs.values():
                if size_str.is_tensor:
                    tensors = size_str.tensors
                    for i in range(len(tensors)):
                        dims[i] = max(dims[i], len(tensors[i]))

        return dims

    @classmethod
    def _get_max_input_shape_length(cls):
        max_length = 0
        for block in cls.all_blocks:
            if len(block.input_str_shapes) > 0:
                length = max([len(s) for s in block.input_str_shapes])
                max_length = max(max_length, length)

        return max_length

    @classmethod
    def _get_output_max_coefficient(cls):
        max_dims = 0
        for block in cls.all_blocks:
            for size_str in block.output_kwargs.values():
                length = len(size_str.coefficient)
                max_dims = max(max_dims, length)

        return max_dims

    @classmethod
    def _get_max_input_args_length(cls):
        max_length = 0
        for block in cls.all_blocks:
            if len(block.input_kwargs) > 0:
                length = max([len(f"{arg}: ") for arg in block.input_kwargs.keys()])
                max_length = max(max_length, length)

        return max_length

    @classmethod
    def _get_max_output_length(cls):
        max_length = 0
        for block in cls.all_blocks:
            length = max([len(s) for s in block.output_str_shapes])
            max_length = max(max_length, length)

        return max_length

    @classmethod
    def _get_max_param_length(cls):
        max_length = 0
        for block in cls.all_blocks:
            length = len(f"{block.n_params:,}")
            max_length = max(max_length, length)

        return max_length

    @classmethod
    def _get_max_weight_param_per_length(cls):
        max_length = len("Weight")
        for block in cls.all_blocks:
            length = len(f"{block.weight_n_params :,}")
            max_length = max(max_length, length)

        return max_length

    @classmethod
    def _get_max_bias_param_per_length(cls):
        max_length = len("Bias")
        for block in cls.all_blocks:
            length = len(f"{block.bias_n_params :,}")
            max_length = max(max_length, length)

        return max_length

    @classmethod
    def _get_max_param_per_length(cls):
        max_length = len("Total(%)")
        for block in cls.all_blocks:
            length = len(block.param_per_str)
            max_length = max(max_length, length)

        return max_length

    @classmethod
    def _get_max_train_length(cls):
        max_length = len("Train")
        for block in cls.all_blocks:
            length = len(f"{block.is_trained}")
            max_length = max(max_length, length)

        return max_length

    @classmethod
    def _get_max_untrain_length(cls):
        max_length = len("Untrain")
        for block in cls.all_blocks:
            length = len(f"{block.is_untrained}")
            max_length = max(max_length, length)

        return max_length

    @classmethod
    def _get_max_time_length(cls):
        max_length = len("Time (ms)")
        for block in cls.all_blocks:
            length = len(f"{block.processing_time * 1000:.0f}")
            max_length = max(max_length, length)

        return max_length


class Input:

    def __init__(self, values):
        self.raw_values = values
        self.values = self.recursive(values)

    @classmethod
    def recursive(cls, value):
        if cls.is_iterable(value):
            if isinstance(value, dict):
                return dict((key, cls.recursive(v)) for key, v in value.items())

            return Values([cls.recursive(v) for v in value])

        return Value(value)

    @staticmethod
    def is_iterable(value):
        if isinstance(value, (tuple, list, set, dict)):
            return True
        return False


class Output(Input):
    pass


class Values:

    def __init__(self, values):
        self.values = values
        self.value_strs, self.value_coefficients = self._summarize_texts(values)
        self.texts = self.to_texts(self.value_strs, self.value_coefficients)
        self.text = self._to_text(self.texts)

    @staticmethod
    def _summarize_texts(values):
        texts = []
        for value in values:
            if isinstance(value, Value):
                texts.append(value.text)
            elif isinstance(value, Values):
                texts.append(value.text)
            else:
                raise ValueError

        counter_sorted = Counter(texts).most_common()
        summary_texts = []
        summary_coefficients = []
        for value, n in counter_sorted:
            summary_texts.append(value)
            summary_coefficients.append(n)

        return summary_texts, summary_coefficients

    @staticmethod
    def to_texts(value_strs, value_coefficients):
        texts = []
        for value_str, value_coefficient in zip(value_strs, value_coefficients):
            text = f"{value_str} * {value_coefficient}"
            texts.append(text)

        return texts

    @staticmethod
    def _to_text(texts):
        return str(texts)

    #
    # class Values:
    #
    #     @staticmethod
    #     def init(values):
    #         values.recursive()
    #
    #     def __init__(self, values):
    #         self.values = self.recursive(values)
    #         self.text, self.cefficient = self._to_text(self.values, self.iterable)
    #
    #     @classmethod
    #     def to_values(cls, value):
    #         values = cls.recursive(value)
    #
    #     @classmethod
    #     def recursive(cls, value):
    #         if cls.is_iterable(value):
    #             if isinstance(value, dict):
    #                 return dict((key, cls.recursive(v)) for key, v in value.items())
    #
    #             return [cls.recursive(v) for v in value]
    #
    #         return Value(value)
    #
    #     @classmethod
    #     def _to_text(cls, values):
    #         if isinstance(values, Value):
    #             return values.text, ""
    #         else:
    #             texts = [value for value in values]
    #
    #             return

    @classmethod
    def init(cls, tensor):
        string = ""
        is_tensor = False
        coefficient = ""

        if isinstance(tensor, torch.Tensor):
            is_tensor = True
            string = cls.tensor_to_string_size(tensor)

        elif isinstance(tensor, (list, tuple)):
            v = tensor[0]
            if isinstance(v, torch.Tensor):
                shapes = []
                shapes_dict = dict()
                for t in tensor:
                    s = str(t.shape)
                    shapes.append(s)
                    shapes_dict[s] = t
                counter = Counter(shapes)
                sort = counter.most_common()
                if len(sort) == 1:
                    is_tensor = True
                    string = cls.tensor_to_string_size(v)
                    coefficient = f"{len(tensor)}"

                else:
                    out = []
                    for i, (value, n) in enumerate(sort):
                        is_tensor = True
                        string = cls.tensor_to_string_size(shapes_dict[value])
                        coefficient = f"{n}"
                        size_str = Value(tensor, string, is_tensor, coefficient)
                        out.append(size_str)
                    return out

            else:
                is_tensor = False
                if isinstance(v, Number):
                    string = str(tensor)
                else:
                    string = str(v.__class__.__name__)
                    string = f"<class: {string}>"
                    coefficient = f"{len(tensor)}"

        elif isinstance(tensor, dict):
            is_tensor = False
            string = ",".join(tensor.keys())

            if isinstance(tensor, (Number, bool)):
                string = str(tensor)

        else:
            is_tensor = False
            string = "Unknown"

            if isinstance(tensor, (Number, bool)):
                string = str(tensor)

        return Value(tensor, string, is_tensor, coefficient)

    @staticmethod
    def is_iterable(value):
        if isinstance(value, (tuple, list, set, dict)):
            return True
        return False

    # @staticmethod
    # def is_iterable(value):
    #     if hasattr(value, "__iter__"):
    #         return True
    #
    #     return False
    #
    # @staticmethod
    # def contains_iterable(values):
    #     for value in values:
    #         if isinstance(value, (tuple, list, set, dict)):
    #             return True
    #
    #     return False


class Value:

    def __init__(self, value):
        self._value = value
        self._text = self._to_text(value)
        self._type = self._to_type(value)

    @property
    def value(self):
        return self._value

    @property
    def text(self):
        return self._text

    @property
    def type(self):
        return self._type

    @classmethod
    def _to_text(cls, value):
        if isinstance(value, torch.Tensor):
            return cls._tensor_to_str(value)
        else:
            return str(value)

    @staticmethod
    def _tensor_to_str(value):
        size = list(value.shape)
        if len(size) > 0:
            size[0] = -1

        return list(map(str, size))

    @staticmethod
    def _to_type(value):
        return str(value.__class__.__name__)


class Value2:

    def __init__(self, value):
        self.value = value
        self.type = None
        self.str = list

    @staticmethod
    def _to_string(value):
        if isinstance(value, (tuple, list, set)):
            return type(value),
        elif isinstance(value, dict):
            pass

        elif isinstance(value, torch.Tensor):
            is_tensor = True
            string = cls.tensor_to_string_size(value)

        else:
            pass

    @classmethod
    def _iterable_to_shape(cls, values):
        """

        Args:
            values (tuple or list or set):

        Returns:

        """
        texts = []
        for value in values:
            if isinstance(value, torch.Tensor):
                texts.append(cls.shape_to_str(value.shape))
            elif hasattr(value, "__iter__"):
                texts.append(f"{value.__class__.__name__} * {len(value)}")
            else:
                texts.append(str(value))

        counter = Counter(texts)
        if len(counter.keys()) == 1:
            text = texts[0]
            n = len(values)
            return Value(type=type(values[0]), text=text, length=1, is_tensor=False)
        else:
            for text, n in counter.most_common():
                Value(text, n)

    # sort = counter.most_common()
    @staticmethod
    def recursive_str(value):
        pass

    @classmethod
    def init(cls, tensor):
        string = ""
        is_tensor = False
        coefficient = ""

        if isinstance(tensor, torch.Tensor):
            is_tensor = True
            string = cls.tensor_to_string_size(tensor)

        elif isinstance(tensor, (list, tuple)):
            v = tensor[0]
            if isinstance(v, torch.Tensor):
                shapes = []
                shapes_dict = dict()
                for t in tensor:
                    s = str(t.shape)
                    shapes.append(s)
                    shapes_dict[s] = t
                counter = Counter(shapes)
                sort = counter.most_common()
                if len(sort) == 1:
                    is_tensor = True
                    string = cls.tensor_to_string_size(v)
                    coefficient = f"{len(tensor)}"

                else:
                    out = []
                    for i, (value, n) in enumerate(sort):
                        is_tensor = True
                        string = cls.tensor_to_string_size(shapes_dict[value])
                        coefficient = f"{n}"
                        size_str = Value(tensor, string, is_tensor, coefficient)
                        out.append(size_str)
                    return out

            else:
                is_tensor = False
                if isinstance(v, Number):
                    string = str(tensor)
                else:
                    string = str(v.__class__.__name__)
                    string = f"<class: {string}>"
                    coefficient = f"{len(tensor)}"

        elif isinstance(tensor, dict):
            is_tensor = False
            string = ",".join(tensor.keys())

            if isinstance(tensor, (Number, bool)):
                string = str(tensor)

        else:
            is_tensor = False
            string = "Unknown"

            if isinstance(tensor, (Number, bool)):
                string = str(tensor)

        return Value(tensor, string, is_tensor, coefficient)

    @staticmethod
    def shape_to_str(shape):
        size = list(shape)
        if len(size) > 0:
            size[0] = -1
        size = list(map(str, size))
        return size

    @staticmethod
    def tensor_to_string_size(tensor):
        size = list(tensor.size())
        if len(size) > 0:
            size[0] = -1
        size = list(map(str, size))
        return size

    def tensors_to_str(self, each_max_dims):
        if self.is_tensor:
            shape = [f"{self.string[i]:>{each_max_dims[i]}}" for i in range(len(self.string))]
            return f"[{', '.join(shape)}]"
        else:
            return self.string

    def size(self):
        def recursive(tensor):
            if isinstance(tensor, torch.Tensor):
                return np.prod(list(tensor.size()))
            elif isinstance(tensor, (list, tuple)):
                return sum([recursive(tensor) for tensor in tensor])
            elif isinstance(tensor, dict):
                return sum([recursive(tensor) for tensor in tensor.values()])
            return 0

        return recursive(self.tensor)

    def __str__(self):
        return str(self.string)

    @property
    def tensors(self):
        return self.string

    @property
    def n_dims(self):
        return len(self.string)

    def __len__(self):
        return len(str(self.string))

# class Value:
#
#     def __init__(self, tensor, string, is_tensor, coefficient):
#         self.tensor = tensor
#         self.string = string
#         self.is_tensor = is_tensor
#         self.coefficient = coefficient
#
#     @classmethod
#     def init(cls, tensor):
#         string = ""
#         is_tensor = False
#         coefficient = ""
#
#         if isinstance(tensor, torch.Tensor):
#             is_tensor = True
#             string = cls.tensor_to_string_size(tensor)
#
#         elif isinstance(tensor, (list, tuple)):
#             v = tensor[0]
#             if isinstance(v, torch.Tensor):
#                 shapes = []
#                 shapes_dict = dict()
#                 for t in tensor:
#                     s = str(t.shape)
#                     shapes.append(s)
#                     shapes_dict[s] = t
#                 counter = Counter(shapes)
#                 sort = counter.most_common()
#                 if len(sort) == 1:
#                     is_tensor = True
#                     string = cls.tensor_to_string_size(v)
#                     coefficient = f"{len(tensor)}"
#
#                 else:
#                     out = []
#                     for i, (value, n) in enumerate(sort):
#                         is_tensor = True
#                         string = cls.tensor_to_string_size(shapes_dict[value])
#                         coefficient = f"{n}"
#                         size_str = Value(tensor, string, is_tensor, coefficient)
#                         out.append(size_str)
#                     return out
#
#             else:
#                 is_tensor = False
#                 if isinstance(v, Number):
#                     string = str(tensor)
#                 else:
#                     string = str(v.__class__.__name__)
#                     string = f"<class: {string}>"
#                     coefficient = f"{len(tensor)}"
#
#         elif isinstance(tensor, dict):
#             is_tensor = False
#             string = ",".join(tensor.keys())
#
#             if isinstance(tensor, (Number, bool)):
#                 string = str(tensor)
#
#         else:
#             is_tensor = False
#             string = "Unknown"
#
#             if isinstance(tensor, (Number, bool)):
#                 string = str(tensor)
#
#         return Value(tensor, string, is_tensor, coefficient)
#
#     @staticmethod
#     def tensor_to_string_size(tensor):
#         size = list(tensor.size())
#         if len(size) > 0:
#             size[0] = -1
#         size = list(map(str, size))
#         return size
#
#     def tensors_to_str(self, each_max_dims):
#         if self.is_tensor:
#             shape = [f"{self.string[i]:>{each_max_dims[i]}}" for i in range(len(self.string))]
#             return f"[{', '.join(shape)}]"
#         else:
#             return self.string
#
#     def size(self):
#         def recursive(tensor):
#             if isinstance(tensor, torch.Tensor):
#                 return np.prod(list(tensor.size()))
#             elif isinstance(tensor, (list, tuple)):
#                 return sum([recursive(tensor) for tensor in tensor])
#             elif isinstance(tensor, dict):
#                 return sum([recursive(tensor) for tensor in tensor.values()])
#             return 0
#
#         return recursive(self.tensor)
#
#     def __str__(self):
#         return str(self.string)
#
#     @property
#     def tensors(self):
#         return self.string
#
#     @property
#     def n_dims(self):
#         return len(self.string)
#
#     def __len__(self):
#         return len(str(self.string))
