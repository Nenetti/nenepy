import inspect
import sys
from collections import OrderedDict, Counter

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

    directory_length = 0
    input_shape_length = 0
    input_arg_length = 0
    output_shape_length = 0
    param_n_length = 0
    param_per_length = 0

    all_blocks = []

    def __init__(self):
        self.module = None
        self.module_in = None
        self.module_out = None
        self.blocks = []
        self.kwargs = []
        self.input_kwargs = None
        self.output_kwargs = None
        self.depth = 0
        self.bottom = False
        self.n_params = 0
        self.weight_grad = None
        self.all_blocks.append(self)

    @classmethod
    def input_length(cls):
        length = cls.input_arg_length + cls.input_shape_length + cls.input_coefficient_length
        if cls.input_arg_length > 0:
            length += len(": ")
        if cls.input_coefficient_length > 0:
            length += len(" * ")

        return length

    @classmethod
    def output_length(cls):
        length = cls.output_shape_length + cls.output_coefficient_length
        if cls.output_coefficient_length > 0:
            length += len(" * ")
        return length

    @classmethod
    def param_length(cls):
        length = cls.param_n_length + cls.param_per_length
        if cls.param_n_length > 0:
            length += len(" ")
        return length

    @property
    def name(self):
        return str(self.module.__class__).split(".")[-1].split("'")[0]

    @property
    def directory(self):
        return f"{self.name:<{self.directory_length}}"

    @property
    def param(self):
        if self.n_params == 0:
            return ""
        else:
            return f"{self.n_params:,}"

    @property
    def param_per(self):
        if self.n_params == 0:
            return ""
        else:
            per = (self.n_params / self.total_params) * 100
            return f"({per:.1f}%)"

    @property
    def param_text(self):
        return f"{self.param:>{self.param_n_length}} {self.param_per:>{self.param_per_length}}"

    @property
    def input_args(self):
        if len(self.input_kwargs) > 1:
            return [f"{arg}" for arg in self.input_kwargs.keys()]
        else:
            return [""]

    @property
    def input_shapes(self):
        return [f"{size_str.tensors_to_str(self.output_each_max_dims)}" for size_str in self.input_kwargs.values()]

    @property
    def input_coefficients(self):
        return [f"{size_str.coefficient}" for size_str in self.input_kwargs.values()]

    @property
    def input_texts(self):
        args = self.input_args
        shapes = self.input_shapes
        coefficient = self.input_coefficients
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
    def output_shapes(self):
        return [f"{size_str.tensors_to_str(self.output_each_max_dims)}" for size_str in self.output_kwargs.values()]

    @property
    def output_coefficients(self):
        return [f"{size_str.coefficient}" for size_str in self.output_kwargs.values()]

    @property
    def output_texts(self):
        shapes = self.output_shapes
        coefficient = self.output_coefficients
        texts = [None] * len(shapes)
        for i in range(len(shapes)):
            if self.output_coefficient_length == 0:
                texts[i] = f"{shapes[i]:<{self.output_shape_length}}"
            elif coefficient[i] != "":
                texts[i] = f"{shapes[i]:<{self.output_shape_length}} * {coefficient[i]:<{self.output_coefficient_length}}"
            else:
                texts[i] = f"{shapes[i]:<{self.output_shape_length}}   {coefficient[i]:<{self.output_coefficient_length}}"
        return texts

    def add(self, module):
        self.blocks.append(module)

    @classmethod
    def calc_length(cls):
        cls.input_max_tensor_dims = cls._get_input_max_tensor_dims()
        cls.input_each_max_dims = cls._get_input_each_dim_max_size()
        cls.input_coefficient_length = cls._get_input_max_coefficient()

        cls.output_max_tensor_dims = cls._get_output_max_tensor_dims()
        cls.output_each_max_dims = cls._get_output_each_dim_max_size()
        cls.output_coefficient_length = cls._get_output_max_coefficient()

        cls.input_shape_length = cls._get_max_input_shape_length()
        cls.input_arg_length = cls._get_max_input_args_length()
        cls.output_shape_length = cls._get_max_output_length()
        cls.param_n_length = cls._get_max_param_length()
        cls.param_per_length = cls._get_max_param_per_length()

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
            length = max([len(size_str.tensors_to_str(cls.input_each_max_dims)) for size_str in block.input_kwargs.values()])
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
            length = max([len(size_str.tensors_to_str(cls.output_each_max_dims)) for size_str in block.output_kwargs.values()])
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
    def _get_max_param_per_length(cls):
        max_length = 0
        for block in cls.all_blocks:
            per = (block.n_params / cls.total_params) * 100
            length = len(f"({per:.1f}%)")
            max_length = max(max_length, length)

        return max_length


class InputSizeStr:

    def __init__(self, string, is_tensor, coefficient):

        self.string = string
        self.is_tensor = is_tensor
        self.coefficient = coefficient

    @staticmethod
    def init(tensor):
        string = ""
        is_tensor = False
        coefficient = ""

        if isinstance(tensor, torch.Tensor):
            is_tensor = True
            string = list(map(str, list(tensor.size())))

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
                    string = list(map(str, list(v.size())))
                    coefficient = f"{len(tensor)}"

                else:
                    out = []
                    for i, (value, n) in enumerate(sort):
                        is_tensor = True
                        string = list(map(str, list(shapes_dict[value].size())))
                        coefficient = f"{n}"
                        size_str = InputSizeStr(string, is_tensor, coefficient)
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

        return InputSizeStr(string, is_tensor, coefficient)

    def tensors_to_str(self, each_max_dims):
        if self.is_tensor:
            shape = [f"{self.string[i]:>{each_max_dims[i]}}" for i in range(len(self.string))]
            return f"[{', '.join(shape)}]"
        else:
            return self.string

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
        self.model = model
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
        # print(x.shape)
        self.model(*x)
        # print(len(self.ordered_blocks))

        self.total_params = 0
        for block in self.ordered_blocks:
            self.total_params += block.n_params
        Block.total_params = self.total_params
        Block.calc_length()

        self.calc_depth()
        self.name_texts()
        # sys.exit()

        # remove these hooks
        for h in self.hooks:
            h.remove()

        sys.exit()
        #
        # print("----------------------------------------------------------------")
        # line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        # print(line_new)
        # print("================================================================")
        # total_params = self.total_params
        # total_input = sum([self.get_memory_size(block.module_in) for block in self.ordered_blocks])
        # total_output = sum([self.get_memory_size(block.module_out) for block in self.ordered_blocks])
        # trainable_params = 0
        # input_param_size = total_input
        #
        # total_input_size = abs(input_param_size * self.batch_size * 4. / (1024 ** 2.))
        # total_output_size = abs(2 * self.batch_size * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        # total_params_size = abs(total_params * 4. / (1024 ** 2.))
        # total_size = total_params_size + total_output_size + total_input_size
        #
        # print("================================================================")
        # print(f"Total params:           {total_params:,}")
        # print(f"Trainable params:       {trainable_params:,}")
        # print(f"Non-trainable params:   {total_params - trainable_params:,}")
        # print("----------------------------------------------------------------")
        # print(f"Input size (MB):                    {total_input_size / self.batch_size:.2f}")
        # print(f"Forward/backward pass size (MB):    {total_output_size / self.batch_size:.2f}")
        # print(f"Params size (MB):                   {total_params_size / self.batch_size:.2f}")
        # print(f"Estimated Total Size (MB):          {total_size / self.batch_size:.2f}")
        # print("----------------------------------------------------------------")
        # print(f"Total Input size (MB):                  {total_input_size:.2f}")
        # print(f"Total Input size (MB):                  {total_input_size:.2f}")
        # print(f"Total Forward/backward pass size (MB):  {total_output_size:.2f}")
        # print(f"Total Params size (MB):                 {total_params_size:.2f}")
        # print(f"Total Estimated Total Size (MB):        {total_size:.2f}")
        # print("----------------------------------------------------------------")
        # return summary

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
            text = f"{index_space}    {block.name}    "
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

    def print_string(self, directory, input_text, output_text, param_text, is_boundary):

        partition = "  │  "
        if is_boundary:
            partition = " -│- "

        print(f"{directory}{partition}{input_text}{partition}{output_text}{partition}{param_text}  │")

    def name_texts(self):

        directory_length = self._get_max_directory_structure_length()

        input_length = Block.input_length()
        output_length = Block.output_length()
        param_length = Block.param_length()

        input_empty = f"{' ' * input_length}"
        output_empty = f"{' ' * output_length}"
        param_empty = f"{' ' * param_length}"

        line = "--│--".join([
            f"{'-' * directory_length}",
            f"{'-' * input_length}",
            f"{'-' * output_length}",
            f"{'-' * param_length}",
        ]) + "--│ "

        line2 = "  │  ".join([
            f"{' ' * directory_length}",
            f"{' ' * input_length}",
            f"{' ' * output_length}",
            f"{' ' * param_length}",
        ]) + "  │ "

        indexes_str = "  │  ".join([
            f"{'Network Architecture':^{directory_length}}",
            f"{'Input':^{input_length}}",
            f"{'Output':^{output_length}}",
            f"{'Parameters':^{param_length}}",
        ]) + "  │  "

        print()
        print(line)
        print(indexes_str)
        print(line)
        print(line2)

        def recursive(root, append="", is_last=False, before_is_boundary=False, before_is_space=True):

            n_child_blocks = len(root.blocks)
            directory, inputs, outputs, params, boundaries = root.name, root.input_texts, root.output_texts, [root.param_text], [False]
            max_length = max(len(inputs), len(outputs))
            directories, new_append = self.to_directories(root, directory, directory_length, max_length, append, is_last)
            inputs += [" " * len(inputs[0])] * (max_length - len(inputs))
            outputs += [" " * len(outputs[0])] * (max_length - len(outputs))
            params += [" " * len(params[0])] * (max_length - len(params))
            boundaries += [False] * (max_length - len(boundaries))

            need_before_boundary = (max_length > 1) and (not before_is_boundary)
            need_before_space = (n_child_blocks > 0) and (not before_is_space and not before_is_boundary)
            need_boundary = max_length > 1
            need_space = is_last and len(root.blocks) == 0

            if need_before_space:
                d = f"{f'{append}│ ':<{directory_length}}"
                self.print_string(d, input_empty, output_empty, param_empty, False)

            if need_before_boundary:
                d = f"{f'{append}│    ':<{directory_length}}"
                self.print_string(d, input_empty, output_empty, param_empty, True)

            for i in range(len(directories)):
                self.print_string(directories[i], inputs[i], outputs[i], params[i], boundaries[i])

            before_is_space = False
            before_is_boundary = False
            if need_boundary:
                d = f"{new_append + '│    '}" if len(root.blocks) > 0 else f"{new_append + '     '}"
                d = f"{d:<{directory_length}}"
                self.print_string(d, input_empty, output_empty, param_empty, True)
                before_is_boundary = True

            if need_space:
                d = f"{new_append:<{directory_length}}"
                self.print_string(d, input_empty, output_empty, param_empty, False)
                before_is_space = True

            for i, block in enumerate(root.blocks):
                if i + 1 == len(root.blocks):
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

        for root in self.roots:
            recursive(root, is_last=True)

        print(line2)
        print(line)
        print(indexes_str)
        print(line)
        print()

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
        block = Block()
        if len(self.blocks) == 0:
            self.roots.append(block)

        self.blocks.append(block)
        self.ordered_blocks.append(block)

    def hook(self, module, module_in, module_out):
        if isinstance(module_out, torch.Tensor):
            module_out = [module_out]

        block = self.blocks.pop(-1)
        block.module = module
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
            self.blocks[-1].add(block)

        n_params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            n_params += torch.prod(torch.LongTensor(list(module.weight.size()))).item()
            block.weight_grad = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            n_params += torch.prod(torch.LongTensor(list(module.bias.size()))).item()
        block.n_params = n_params

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

    @staticmethod
    def get_memory_size(tensors):
        def recursive(tensor):
            total = 0
            if isinstance(tensor, torch.Tensor):
                total += np.prod(list(tensor.size()))

            elif isinstance(tensor, (list, tuple)):
                for t in tensor:
                    total += recursive(t)
            elif isinstance(tensor, dict):
                for t in tensor.values():
                    total += recursive(t)

            return total

        return recursive(tensors)
