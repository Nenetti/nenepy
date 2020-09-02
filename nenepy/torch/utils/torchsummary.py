import inspect
import sys
from collections import OrderedDict, Counter

import numpy as np
import torch
import torch.nn as nn


class Block:

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

    @property
    def name(self):
        return str(self.module.__class__).split(".")[-1].split("'")[0]

    def add(self, module):
        self.blocks.append(module)

    def input_kwargs_str(self):
        out = OrderedDict()
        if len(self.input_kwargs) == 1:
            out[""] = f"{list(self.input_kwargs.values())[0]}"
        else:
            for arg, value in self.input_kwargs.items():
                out[f"{arg}: "] = f"{value}"

        return out

    def output_kwargs_str(self):
        out = OrderedDict()
        if len(self.input_kwargs) == 1:
            out[""] = list(self.input_kwargs.values())[0]
        else:
            for arg, value in self.input_kwargs.items():
                out[f"{arg}"] = f"{value}"

        return out

    @staticmethod
    def get_input_max_tensor_dims(blocks):
        max_dims = 0
        for block in blocks:
            for size_str in block.input_kwargs.values():
                if size_str.is_tensor:
                    max_dims = max(max_dims, size_str.n_dims)

        return max_dims

    @staticmethod
    def get_output_max_tensor_dims(blocks):
        max_dims = 0
        for block in blocks:
            for size_str in block.output_kwargs.values():
                if size_str.is_tensor:
                    max_dims = max(max_dims, size_str.n_dims)

        return max_dims

    @staticmethod
    def get_input_each_dim_max_size(blocks, n_dims):
        dims = [0 for _ in range(n_dims)]
        for block in blocks:
            for size_str in block.input_kwargs.values():
                if size_str.is_tensor:
                    tensors = size_str.tensors
                    for i in range(len(tensors)):
                        dims[i] = max(dims[i], len(tensors[i]))

        return dims

    @staticmethod
    def get_input_max_coefficient(blocks):
        max_dims = 0
        for block in blocks:
            for size_str in block.input_kwargs.values():
                length = len(size_str.coefficient)
                max_dims = max(max_dims, length)

        return max_dims

    @staticmethod
    def get_output_each_dim_max_size(blocks, n_dims):
        dims = [0 for _ in range(n_dims)]
        for block in blocks:
            for size_str in block.output_kwargs.values():
                if size_str.is_tensor:
                    tensors = size_str.tensors
                    for i in range(len(tensors)):
                        dims[i] = max(dims[i], len(tensors[i]))

        return dims

    @staticmethod
    def get_max_input_length(blocks, each_max_dims):
        max_length = 0
        for block in blocks:
            length = max([len(size_str.tensors_to_str(each_max_dims)) for size_str in block.input_kwargs.values()])
            max_length = max(max_length, length)

        return max_length

    @staticmethod
    def get_output_max_coefficient(blocks):
        max_dims = 0
        for block in blocks:
            for size_str in block.output_kwargs.values():
                length = len(size_str.coefficient)
                max_dims = max(max_dims, length)

        return max_dims

    @staticmethod
    def get_max_input_args_length(blocks):
        max_length = 0
        for block in blocks:
            length = max([len(f"{arg}: ") for arg in block.input_kwargs.keys()])
            max_length = max(max_length, length)

        return max_length

    @staticmethod
    def get_max_output_length(blocks, each_max_dims):
        max_length = 0
        for block in blocks:
            length = max([len(size_str.tensors_to_str(each_max_dims)) for size_str in block.output_kwargs.values()])
            max_length = max(max_length, length)

        return max_length

    @staticmethod
    def get_max_param_length(blocks):
        max_length = 0
        for block in blocks:
            length = len(f"{block.n_params:,}")
            max_length = max(max_length, length)

        return max_length

    @staticmethod
    def get_max_param_per_length(blocks, total_params):
        max_length = 0
        for block in blocks:
            per = (block.n_params / total_params) * 100
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

        self.calc_depth()
        self.name_texts()
        # sys.exit()

        # remove these hooks
        for h in self.hooks:
            h.remove()

        # sys.exit()

        print("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
        total_params = self.total_params
        total_input = sum([self.get_memory_size(block.module_in) for block in self.ordered_blocks])
        total_output = sum([self.get_memory_size(block.module_out) for block in self.ordered_blocks])
        trainable_params = 0
        input_param_size = total_input

        total_input_size = abs(input_param_size * self.batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2 * self.batch_size * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        print("================================================================")
        print(f"Total params:           {total_params:,}")
        print(f"Trainable params:       {trainable_params:,}")
        print(f"Non-trainable params:   {total_params - trainable_params:,}")
        print("----------------------------------------------------------------")
        print(f"Input size (MB):                    {total_input_size / self.batch_size:.2f}")
        print(f"Forward/backward pass size (MB):    {total_output_size / self.batch_size:.2f}")
        print(f"Params size (MB):                   {total_params_size / self.batch_size:.2f}")
        print(f"Estimated Total Size (MB):          {total_size / self.batch_size:.2f}")
        print("----------------------------------------------------------------")
        print(f"Total Input size (MB):                  {total_input_size:.2f}")
        print(f"Total Input size (MB):                  {total_input_size:.2f}")
        print(f"Total Forward/backward pass size (MB):  {total_output_size:.2f}")
        print(f"Total Params size (MB):                 {total_params_size:.2f}")
        print(f"Total Estimated Total Size (MB):        {total_size:.2f}")
        print("----------------------------------------------------------------")
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

    def get_max_directory_structure_length(self):
        max_length = 0
        for block in self.ordered_blocks:
            index_space = " " * 4 * block.depth
            text = f"{index_space}    {block.name}    "
            max_length = max(max_length, len(text))

        return max_length

    def to_string(self, block, input_each_max_dims, output_each_max_dims):
        """

        Args:
            block (Block):

        """
        name = f"{block.name}"

        input_args = []
        inputs = []
        input_coefficients = []
        outputs = []
        output_coefficients = []

        for arg, size_str in block.input_kwargs.items():
            input_args_text = arg if len(block.input_kwargs) > 1 else ""
            input_shape = f"{size_str.tensors_to_str(input_each_max_dims)}"
            input_coefficient = size_str.coefficient
            inputs.append(input_shape)
            input_coefficients.append(input_coefficient)
            input_args.append(input_args_text)

        for size_str in block.output_kwargs.values():
            text = f"{size_str.tensors_to_str(output_each_max_dims)}"
            outputs_coefficient = size_str.coefficient
            output_coefficients.append(outputs_coefficient)
            outputs.append(text)

        if block.n_params != 0:
            param_text = f"{block.n_params:,}"
        else:
            param_text = ""
        return name, inputs, input_args, input_coefficients, outputs, output_coefficients, [param_text]

    def print_string(self, directory, input_arg, input_shape, input_coefficient, output_shape, output_coefficient, param,
                     directory_length, input_arg_length, input_shape_length, input_coefficient_length, output_shape_length, output_coefficient_length,
                     param_length, param_per_length, is_boundary=False):

        directory_text = f"{directory:<{directory_length}}"
        input_arg_text = f"{input_arg:>{input_arg_length}}"
        input_shape_text = f"{input_shape:<{input_shape_length}}"
        input_coefficient_text = f"{input_coefficient:>{input_coefficient_length}}"
        output_shape_text = f"{output_shape:<{output_shape_length}}"
        output_coefficient_text = f"{output_coefficient:>{output_coefficient_length}}"
        param_text = f"{param:>{param_length}}"
        if param != "":
            per = (int(param.replace(",", "")) / self.total_params) * 100
            per = f"({per:.1f}%)"
            param_text = f"{param_text} {per:>{param_per_length}}"
        else:
            param_text = f"{param_text} {' ':>{param_per_length}}"

        input_text = f"{input_arg_text}"
        if input_arg != "":
            input_text = f"{input_text}: {input_shape_text}"
        else:
            input_text = f"{input_text}  {input_shape_text}"

        if input_coefficient_length > 0:
            if input_coefficient != "":
                input_text = f"{input_text} * {input_coefficient_text}"
            else:
                input_text = f"{input_text}   {input_coefficient_text}"

        output_text = f"{output_shape_text}"
        if output_coefficient_length > 0:
            if output_coefficient != "":
                output_text = f"{output_text} * {output_coefficient_text}"
            else:
                output_text = f"{output_text}   {output_coefficient_text}"

        partition = "  │  "
        if is_boundary:
            partition = " -│- "

        print(f"{directory_text}{partition}{input_text}{partition}{output_text}{partition}{param_text}  │")

    def name_texts(self):
        input_max_tensor_dims = Block.get_input_max_tensor_dims(self.ordered_blocks)
        input_each_max_dims = Block.get_input_each_dim_max_size(self.ordered_blocks, input_max_tensor_dims)
        input_coefficient_length = Block.get_input_max_coefficient(self.ordered_blocks)

        output_max_tensor_dims = Block.get_output_max_tensor_dims(self.ordered_blocks)
        output_each_max_dims = Block.get_output_each_dim_max_size(self.ordered_blocks, output_max_tensor_dims)
        output_coefficient_length = Block.get_output_max_coefficient(self.ordered_blocks)

        directory_length = self.get_max_directory_structure_length()
        input_shape_length = Block.get_max_input_length(self.ordered_blocks, input_each_max_dims)
        input_arg_length = Block.get_max_input_args_length(self.ordered_blocks)
        output_shape_length = Block.get_max_output_length(self.ordered_blocks, output_each_max_dims)
        param_n_length = Block.get_max_param_length(self.ordered_blocks)
        param_per_length = Block.get_max_param_per_length(self.ordered_blocks, self.total_params)

        input_length = input_arg_length + input_shape_length + input_coefficient_length
        if input_arg_length > 0:
            input_length += len(": ")
        if input_coefficient_length > 0:
            input_length += len(" * ")

        output_length = output_shape_length + output_coefficient_length
        if output_coefficient_length > 0:
            output_length += len(" * ")

        param_length = param_n_length + param_per_length + 1

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

        def recursive(root, append="", is_last=False, before_is_boundary=True, before_is_space=True):
            length = len(root.blocks)
            empty = ""
            directories, inputs, input_args, input_coefficients, outputs, output_coefficients, params = self.to_string(root, input_each_max_dims,
                                                                                                                       output_each_max_dims)
            if root.depth == 0:
                directories = [f"  {directories}"]
            else:
                if is_last:
                    directories = [f"{append}└ {directories}"]
                else:
                    directories = [f"{append}├ {directories}"]

            if (not before_is_space and not before_is_boundary) and len(root.blocks) > 0:
                directory = f"{append}│ "
                self.print_string(directory, empty, empty, empty, empty, empty, empty,
                                  directory_length, input_arg_length, input_shape_length, input_coefficient_length, output_shape_length,
                                  output_coefficient_length, param_n_length, param_per_length, is_boundary=False)

            if (len(inputs) > 1 or len(outputs) > 1) and not (before_is_boundary):
                directory = f"{append}│    "
                self.print_string(directory, empty, empty, empty, empty, empty, empty,
                                  directory_length, input_arg_length, input_shape_length, input_coefficient_length, output_shape_length,
                                  output_coefficient_length, param_n_length, param_per_length, is_boundary=True)

            if is_last:
                append = f"{append}     "
            else:
                append = f"{append}│    "

            max_length = max(len(inputs), len(outputs))
            if len(root.blocks) > 0:
                directories += [f"{append}│    "] * (max_length - len(directories))
            else:
                directories += [f"{append}    "] * (max_length - len(directories))
            input_args += [""] * (max_length - len(input_args))
            inputs += [""] * (max_length - len(inputs))
            input_coefficients += [""] * (max_length - len(input_coefficients))
            outputs += [""] * (max_length - len(outputs))
            output_coefficients += [""] * (max_length - len(output_coefficients))
            params += [""] * (max_length - len(params))

            for i in range(max(len(directories), len(inputs), len(outputs))):
                self.print_string(directories[i], input_args[i], inputs[i], input_coefficients[i], outputs[i], output_coefficients[i],
                                  params[i],
                                  directory_length, input_arg_length, input_shape_length, input_coefficient_length, output_shape_length,
                                  output_coefficient_length, param_n_length, param_per_length, is_boundary=False)

            if len(inputs) > 1 or len(outputs) > 1:
                if len(root.blocks) > 0:
                    directory = f"{append + '│    ':<{directory_length}}"
                else:
                    directory = f"{append + '     ':<{directory_length}}"

                self.print_string(directory, empty, empty, empty, empty, empty, empty,
                                  directory_length, input_arg_length, input_shape_length, input_coefficient_length, output_shape_length,
                                  output_coefficient_length, param_n_length, param_per_length, is_boundary=True)
                before_is_boundary = True
            else:
                before_is_boundary = False

            if is_last and len(root.blocks) == 0:
                self.print_string(append, empty, empty, empty, empty, empty, empty,
                                  directory_length, input_arg_length, input_shape_length, input_coefficient_length, output_shape_length,
                                  output_coefficient_length, param_n_length, param_per_length, is_boundary=False)
                before_is_space = True
            else:
                before_is_space = False

            for i, block in enumerate(root.blocks):
                if i == length - 1:
                    if is_last:
                        before_is_boundary, before_is_space = recursive(block, append, is_last=True, before_is_boundary=before_is_boundary,
                                                                        before_is_space=before_is_space)
                    else:
                        before_is_boundary, before_is_space = recursive(block, append, is_last=True, before_is_boundary=before_is_boundary,
                                                                        before_is_space=before_is_space)
                else:
                    if is_last:
                        before_is_boundary, before_is_space = recursive(block, append, before_is_boundary=before_is_boundary, before_is_space=before_is_space)
                    else:
                        before_is_boundary, before_is_space = recursive(block, append, before_is_boundary=before_is_boundary, before_is_space=before_is_space)

            return before_is_boundary, before_is_space

        for root in self.roots:
            recursive(root, is_last=True)

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
