import inspect
import sys
from collections import OrderedDict, Counter
from time import sleep

import numpy as np
import torch
import torch.nn as nn


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
        self.duplication = False

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

    # Property
    @property
    def input_size(self):
        return sum([size_str.size() for size_str in self.input_kwargs.values()])

    @property
    def output_size(self):
        size = 0
        if (self.n_params > 0) or (len(self.blocks) == 0):
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
        is_train = self.is_trained_weight or self.is_trained_bias
        if is_train:
            return "✓"
        else:
            return ""

    # String

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


class InputSizeStr:

    def __init__(self, tensor, string, is_tensor, coefficient):
        self.tensor = tensor
        self.string = string
        self.is_tensor = is_tensor
        self.coefficient = coefficient

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
                        size_str = InputSizeStr(tensor, string, is_tensor, coefficient)
                        out.append(size_str)
                    return out

            else:
                is_tensor = False
                string = str(v.__class__).split(".")[-1].split("'")[0]
                string = f"<class: {string}>"
                coefficient = f"{len(tensor)}"

        else:
            is_tensor = False
            string = "Unknown"

        return InputSizeStr(tensor, string, is_tensor, coefficient)

    @staticmethod
    def tensor_to_string_size(tensor):
        size = list(tensor.size())
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


class Summary:

    def __init__(self, model, batch_size, device="cuda"):
        """

        Args:
            model (nn.Module):
            batch_size:
            device:
        """
        self.model = model.to(device)
        self.model.train()
        self.batch_size = batch_size
        self.device = device
        self.summary = OrderedDict()
        self.hooks = []
        self.call_forwards = {}
        self.called_modules = []
        self.calling_indexes = []
        self.blocks = []
        self.roots = []
        self.now_block = None
        self.n_blocks = 0
        self.ordered_blocks = []

    def __call__(self, input_size, *args, **kwargs):
        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(self.batch_size, *in_size).to(self.device) for in_size in input_size]
        # print(type(x[0]))

        # create properties
        # register hook
        self.model.apply(self.register_hook)

        # make a forward pass
        self.model(*x, **kwargs)

        Block.calc_length()

        self.calc_depth()

        # print(self.roots[0].output_texts)
        # sys.exit()

        self.name_texts()
        # sleep(1000)
        # remove these hooks
        for h in self.hooks:
            h.remove()
        sys.exit()

    def calc_depth(self):

        def recursive(block, d):
            block.depth = d
            if len(block.blocks) > 0:
                block.bottom = False
                for b in block.blocks:
                    recursive(b, d + 1)
            else:
                block.bottom = True

        for root in self.roots:
            recursive(root, 0)

    def _get_max_directory_structure_length(self):
        max_length = 0
        for block in self.ordered_blocks:
            index_space = " " * 4 * block.depth
            text = f"{index_space}    {block.architecture_str}    "
            max_length = max(max_length, len(text))

        return max_length

    @staticmethod
    def to_directories(root, directory, directory_length, max_length, append, is_last):
        if root.depth == 0:
            directories = [f"  {directory}"]
        else:
            if is_last:
                directories = [f"{append}└ {directory}"]
            else:
                directories = [f"{append}├ {directory}"]

        if is_last:
            append = f"{append}     "
        else:
            append = f"{append}│    "

        if len(root.blocks) > 0:
            directories += [f"{append}│    "] * (max_length - len(directories))
        else:
            directories += [f"{append}    "] * (max_length - len(directories))

        for i in range(len(directories)):
            directories[i] = f"{directories[i]:<{directory_length}}"

        return directories, append

    def print_string(self, directory, input_text, output_text, weight_param_text, bias_param_text, param_per_text, is_train_text, is_boundary):

        partition = "  │  "
        if is_boundary:
            partition = " -│- "

        print(
            f"{directory}{partition}{input_text}{partition}{output_text}{partition}{weight_param_text}{partition}{bias_param_text}{partition}{param_per_text}{partition}{is_train_text}  │ "
        )

    @staticmethod
    def to_line(partition, *args):
        return partition.join(args)

    def name_texts(self):

        architecture_length = self._get_max_directory_structure_length()

        input_length = Block.get_input_length()
        output_length = Block.get_output_length()
        weight_length = Block.weight_param_length
        bias_length = Block.bias_param_length
        param_per_length = Block.param_per_length
        train_length = Block.train_length
        param_length = weight_length + bias_length + param_per_length + train_length + 15

        directory_empty = f"{' ' * architecture_length}"
        input_empty = f"{' ' * input_length}"
        output_empty = f"{' ' * output_length}"
        weight_empty = f"{' ' * weight_length}"
        bias_empty = f"{' ' * bias_length}"
        param_per_empty = f"{' ' * param_per_length}"
        train_empty = f"{' ' * train_length}"
        param_empty = f"{'-' * param_length}"

        architecture_title = f"{'Network Architecture':^{architecture_length}}"
        input_title = f"{'Input':^{input_length}}"
        output_title = f"{'Output':^{output_length}}"
        param_title = f"{'Parameters':^{param_length}}"
        weight_title = f"{'Weight':^{weight_length}}"
        bias_title = f"{'Bias':^{bias_length}}"
        param_per_title = f"{'Total(%)':^{param_per_length}}"
        train_title = f"{'Train':^{train_length}}"

        param_line = self.to_line("     ", weight_title, bias_title, param_per_title, train_title)
        border_line = self.to_line("==│==", '=' * architecture_length, '=' * input_length, '=' * output_length, '=' * param_length) + "==│"
        param_detail_line = self.to_line("  │  ", directory_empty, input_empty, output_empty, param_line) + "  │"
        param_line = self.to_line("  │  ", directory_empty, input_empty, output_empty, param_title) + "  │"
        title_line = self.to_line("  │  ", architecture_title, input_title, output_title, param_empty) + "  │"

        print()
        print(border_line)
        print(param_line)
        print(title_line)
        print(param_detail_line)
        print(border_line)

        def recursive(root, append="", is_last=False, before_is_boundary=False, before_is_space=True, child_first=False):

            n_child_blocks = len(root.blocks)
            directory = root.architecture_str
            inputs = root.input_texts

            outputs = root.output_texts
            weight_params = [root.weight_param_text]
            bias_params = [root.bias_param_text]
            per_params = [root.param_per_text]
            is_trains = [root.is_train_text]

            is_boundaries = [False]

            max_length = max(len(inputs), len(outputs))
            directories, new_append = self.to_directories(root, directory, architecture_length, max_length, append, is_last)
            inputs += [" " * len(inputs[0])] * (max_length - len(inputs))
            outputs += [" " * len(outputs[0])] * (max_length - len(outputs))
            weight_params += [" " * len(weight_params[0])] * (max_length - len(weight_params))
            bias_params += [" " * len(bias_params[0])] * (max_length - len(bias_params))
            per_params += [" " * len(per_params[0])] * (max_length - len(per_params))
            is_trains += [" " * len(is_trains[0])] * (max_length - len(is_trains))

            is_boundaries += [False] * (max_length - len(is_boundaries))

            need_before_boundary = ((max_length > 1) and (not before_is_boundary)) or root.depth == 0
            need_before_space = (n_child_blocks > 0 or child_first) and (not before_is_space and not before_is_boundary)
            need_boundary = max_length > 1 or root.depth == 0
            need_space = is_last and len(root.blocks) == 0

            if need_before_space:
                d = f"{f'{append}│ ':<{architecture_length}}"
                self.print_string(d, input_empty, output_empty, weight_empty, bias_empty, param_per_empty, train_empty, False)

            if need_before_boundary:
                if root.depth != 0:
                    d = f"{f'{append}│    ':<{architecture_length}}"
                else:
                    d = f"{f'{append}     ':<{architecture_length}}"
                self.print_string(d, input_empty, output_empty, weight_empty, bias_empty, param_per_empty, train_empty, True)

            for i in range(len(directories)):
                self.print_string(directories[i], inputs[i], outputs[i], weight_params[i], bias_params[i], per_params[i], is_trains[i], is_boundaries[i])

            before_is_space = False
            before_is_boundary = False
            if need_boundary:
                d = f"{new_append + '│    '}" if len(root.blocks) > 0 else f"{new_append + '     '}"
                d = f"{d:<{architecture_length}}"
                self.print_string(d, input_empty, output_empty, weight_empty, bias_empty, param_per_empty, train_empty, True)
                before_is_boundary = True

            if need_space:
                d = f"{new_append:<{architecture_length}}"
                self.print_string(d, input_empty, output_empty, weight_empty, bias_empty, param_per_empty, train_empty, False)
                before_is_space = True

            for i, block in enumerate(root.blocks):
                if i == 0:
                    if is_last:
                        before_is_boundary, before_is_space = recursive(block, new_append, before_is_boundary=before_is_boundary,
                                                                        before_is_space=before_is_space, child_first=True)
                    else:
                        before_is_boundary, before_is_space = recursive(block, new_append, before_is_boundary=before_is_boundary,
                                                                        before_is_space=before_is_space, child_first=True)
                elif i + 1 == len(root.blocks):
                    if is_last:
                        before_is_boundary, before_is_space = recursive(block, new_append, is_last=True, before_is_boundary=before_is_boundary,
                                                                        before_is_space=before_is_space)
                    else:
                        before_is_boundary, before_is_space = recursive(block, new_append, is_last=True, before_is_boundary=before_is_boundary,
                                                                        before_is_space=before_is_space)
                else:
                    if is_last:
                        before_is_boundary, before_is_space = recursive(block, new_append, before_is_boundary=before_is_boundary,
                                                                        before_is_space=before_is_space)
                    else:
                        before_is_boundary, before_is_space = recursive(block, new_append, before_is_boundary=before_is_boundary,
                                                                        before_is_space=before_is_space)

            return before_is_boundary, before_is_space

        for r in self.roots:
            recursive(r, is_last=True)

        print(border_line)
        print(param_detail_line)
        print(title_line)
        print(param_line)
        print(border_line)

        total_params = Block.total_params
        trainable_param_size = Block.get_total_trainable_params()
        untrainable_param_size = Block.get_total_untrainable_params()
        total_input_size = (sum([r.input_size for r in self.roots]) * 4) / (1024 ** 2)
        total_output_size = (sum([r.output_size for r in Block.all_blocks]) * 4) / (1024 ** 2)
        total_params_size = (Block.total_params * 4) / (1024 ** 2)
        total_size = total_params_size + total_output_size + total_input_size

        total_params_title = "Total params"
        trainable_params_title = "Trainable params"
        untrainable_params_title = "Non-trainable params"

        input_size_title = "Input size (/Batch) (MB)"
        forward_backward_size_title = "Forward/backward pass size (/Batch) (MB)"

        total_input_size_title = "Total Input size (MB)"
        total_forward_backward_size_title = "Total Forward/backward pass size (MB)"
        total_params_size_title = "Total Params size (MB)"
        total_estimated_total_size_title = "Total Estimated Size (MB)"

        total_params_text = f"{total_params:,}"
        trainable_params_text = f"{trainable_param_size:,}"
        untrainable_params_text = f"{untrainable_param_size:,}"

        input_size_text = f"{total_input_size / self.batch_size:.2f}"
        forward_backward_size_text = f"{total_output_size / self.batch_size:.2f}"

        total_input_size_text = f"{total_input_size:.2f}"
        total_forward_backward_size_text = f"{total_output_size:.2f}"
        total_params_size_text = f"{total_params_size:.2f}"
        total_estimated_total_size_text = f"{total_size:.2f}"

        titles = [
            total_params_title,
            trainable_params_title,
            untrainable_params_title,
            "",
            input_size_title,
            forward_backward_size_title,
            "",
            total_input_size_title,
            total_forward_backward_size_title,
            total_params_size_title,
            total_estimated_total_size_title,
        ]

        values = [
            total_params_text,
            trainable_params_text,
            untrainable_params_text,
            "",
            input_size_text,
            forward_backward_size_text,
            "",
            total_input_size_text,
            total_forward_backward_size_text,
            total_params_size_text,
            total_estimated_total_size_text,
        ]

        title_length = max([len(t) for t in titles])
        value_length = max([len(t) for t in values])

        for title, values in zip(titles, values):
            print(f"{title:>{title_length}}:   {values:>{value_length}}")

        print()
        counter = Counter(map(lambda x: x.__class__, Block.ids.values()))
        sort = counter.most_common()
        for module, count, in sort:
            print(count, str(module).split("'")[1])

    # ==============================================================================
    #
    #   Hook
    #
    # ==============================================================================

    def register_hook(self, module):

        if (isinstance(module, nn.Sequential)) or (isinstance(module, nn.ModuleList)):
            return

        self.hooks.append(module.register_forward_pre_hook(self.pre_hook))
        self.hooks.append(module.register_forward_hook(self.hook))

    def pre_hook(self, module, module_in):
        """

        Args:
            module (nn.Module):
            module_in:

        Returns:

        """
        block = Block(module)
        if len(self.blocks) == 0:
            self.roots.append(block)

        self.blocks.append(block)
        self.ordered_blocks.append(block)

    def hook(self, module, module_in, module_out):
        if isinstance(module_out, torch.Tensor):
            module_out = [module_out]

        block = self.blocks.pop(-1)
        input_kwargs = OrderedDict()
        parameter_dict = OrderedDict(inspect.signature(module.forward).parameters.items())
        for i, (key, value) in enumerate(parameter_dict.items()):
            if i < len(module_in):
                if value.kind == inspect.Parameter.VAR_POSITIONAL:
                    input_kwargs[key] = module_in[i:]
                else:
                    input_kwargs[key] = module_in[i]
            else:
                break

        output_kwargs = OrderedDict()
        for i, out in enumerate(module_out):
            output_kwargs[f"{i}"] = out

        block.input_kwargs = self.tensors_to_size_str(input_kwargs)
        block.output_kwargs = self.tensors_to_size_str(output_kwargs)

        if len(self.blocks) > 0:
            self.blocks[-1].add_block(block)

        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            n_params = torch.prod(torch.LongTensor(list(module.weight.size()))).item()
            block.add_weight_params(n_params, module.weight.requires_grad)

        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            n_params = torch.prod(torch.LongTensor(list(module.bias.size()))).item()
            block.add_bias_params(n_params, module.bias.requires_grad)

    @staticmethod
    def tensors_to_size_str(tensors):
        out = OrderedDict()
        for key, value in tensors.items():
            size_str = InputSizeStr.init(value)
            if isinstance(size_str, list):
                for i, s in enumerate(size_str):
                    key = f"*args({i})"
                    out[key] = s

            else:
                out[key] = size_str
        return out

    # @staticmethod
    # def get_memory_size(tensors):
    #     def recursive(tensor):
    #         total = 0
    #         if isinstance(tensor, torch.Tensor):
    #             total += np.prod(list(tensor.size()))
    #
    #         elif isinstance(tensor, (list, tuple)):
    #             for t in tensor:
    #                 total += recursive(t)
    #         elif isinstance(tensor, dict):
    #             for t in tensor.values():
    #                 total += recursive(t)
    #
    #         return total
    #
    #     return recursive(tensors)
