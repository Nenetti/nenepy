import inspect
import sys
from collections import OrderedDict

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


class InputSizeStr:

    def __init__(self, tensor):

        self.string = ""
        self.is_tensor = False

        if isinstance(tensor, torch.Tensor):
            self.is_tensor = True
            self.string = list(map(str, list(tensor.size())))

        elif isinstance(tensor, (list, tuple)):
            self.is_tensor = False
            v = tensor[0]
            if isinstance(v, torch.Tensor):
                self.string = f"{str(list(v.size()))} * {len(tensor)}"
            else:
                self.string = f"{v}".replace("torch.Size", "")

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
            index_space = self.space(block.depth)
            text = f"{index_space}    {block.name}    "
            max_length = max(max_length, len(text))

        return max_length

    def get_input_max_tensor_dims(self):
        max_dims = 0
        for block in self.ordered_blocks:
            for size_str in block.input_kwargs.values():
                if size_str.is_tensor:
                    max_dims = max(max_dims, size_str.n_dims)

        return max_dims

    def get_output_max_tensor_dims(self):
        max_dims = 0
        for block in self.ordered_blocks:
            for size_str in block.output_kwargs.values():
                if size_str.is_tensor:
                    max_dims = max(max_dims, size_str.n_dims)

        return max_dims

    def get_input_each_dim_max_size(self, max_dims):
        dims = [0 for _ in range(max_dims)]
        for block in self.ordered_blocks:
            for size_str in block.input_kwargs.values():
                if size_str.is_tensor:
                    tensors = size_str.tensors
                    for i in range(len(tensors)):
                        dims[i] = max(dims[i], len(tensors[i]))

        return dims

    def get_output_each_dim_max_size(self, max_dims):
        dims = [0 for _ in range(max_dims)]
        for block in self.ordered_blocks:
            for size_str in block.output_kwargs.values():
                if size_str.is_tensor:
                    tensors = size_str.tensors
                    for i in range(len(tensors)):
                        dims[i] = max(dims[i], len(tensors[i]))

        return dims

    def get_max_input_length(self, each_max_dims):
        max_length = 0
        for block in self.ordered_blocks:
            length = max([len(size_str.tensors_to_str(each_max_dims)) for size_str in block.input_kwargs.values()])
            max_length = max(max_length, length)

        return max_length

    def get_max_input_args_length(self):

        max_length = 0
        for block in self.ordered_blocks:
            length = max([len(f"{arg}: ") for arg in block.input_kwargs.keys()])
            max_length = max(max_length, length)

        return max_length

    def get_max_output_length(self, each_max_dims):
        max_length = 0
        for block in self.ordered_blocks:
            length = max([len(size_str.tensors_to_str(each_max_dims)) for size_str in block.output_kwargs.values()])
            max_length = max(max_length, length)

        return max_length

    def get_max_param_length(self):
        def recursive(block):
            a = [len(str(b.n_params)) for b in block.blocks]
            if len(a) > 0:
                length = max(a)
            else:
                length = 0

            for b in block.blocks:
                length = max(length, recursive(b))

            return length

        return max([recursive(b) for b in self.roots])

    def to_string(self, block, input_each_max_dims, output_each_max_dims):
        """

        Args:
            block (Block):

        """
        name = f"{block.name}"

        input_texts = []
        input_args_texts = []

        for arg, size_str in block.input_kwargs.items():
            input_texts.append(f"{size_str.tensors_to_str(input_each_max_dims)}")
            if len(block.input_kwargs) > 1:
                input_args_text = f"{arg}: "
            else:
                input_args_text = ""

            input_args_texts.append(input_args_text)

        outputs = [f"{size_str.tensors_to_str(output_each_max_dims)}" for size_str in block.output_kwargs.values()]
        param_text = str(block.n_params)
        return (name, (input_texts, input_args_texts), outputs, [param_text])

    def name_texts(self, space_size=4):
        input_max_tensor_dims = self.get_input_max_tensor_dims()
        input_each_max_dims = self.get_input_each_dim_max_size(input_max_tensor_dims)

        output_max_tensor_dims = self.get_output_max_tensor_dims()
        output_each_max_dims = self.get_output_each_dim_max_size(output_max_tensor_dims)

        name_max_length = self.get_max_directory_structure_length()
        input_max_length = self.get_max_input_length(input_each_max_dims)
        input_max_args_length = self.get_max_input_args_length()
        output_max_length = self.get_max_output_length(output_each_max_dims)
        param_max_length = self.get_max_param_length()

        total_length = name_max_length + input_max_length + input_max_args_length + output_max_length + param_max_length + 3 * 6
        line = "-│-".join([f"{'-' * name_max_length}", f"{'-' * (input_max_length + input_max_args_length)}", f"{'-' * (output_max_length)}"]) + "-│ "

        indexes_str = " │ ".join([f"{'Network Architecture':^{name_max_length}}", f"{'Input':^{input_max_length + input_max_args_length}}",
                                  f"{'Output':^{output_max_length}}"]) + " │ "
        print(line)
        print(indexes_str)
        print(line)

        def recursive(root, append="", is_last=False):
            length = len(root.blocks)

            lines, inputs, outputs, params = self.to_string(root, input_each_max_dims, output_each_max_dims)

            if is_last:
                if root.depth == 0:
                    lines = [f"  {lines}"]
                else:
                    lines = [f"{append}└ {lines}"]
            else:
                lines = [f"{append}├ {lines}"]

            input_texts, input_args_texts = inputs
            output_texts = outputs

            if is_last:
                append = append + "     "
            else:
                append = append + "│    "

            max_length = max(len(input_texts), len(output_texts))
            lines += [append + "│    "] * (max_length - len(lines))
            input_texts += [""] * (max_length - len(input_texts))
            input_args_texts += [""] * (max_length - len(input_args_texts))
            output_texts += [""] * (max_length - len(output_texts))
            params += [""] * (max_length - len(params))

            for i in range(max(len(lines), len(input_texts), len(output_texts))):
                name_text = lines[i]
                input_text = input_texts[i]
                input_args_text = input_args_texts[i]
                output_text = output_texts[i]
                param_text = params[i]
                n_params = root.n_params
                per = (n_params / self.total_params) * 100

                name_text = f"{name_text:<{name_max_length}}"
                input_text = f"{input_text:<{input_max_length}}"
                input_args_text = f"{input_args_text:>{input_max_args_length}}"
                output_text = f"{output_text:<{output_max_length}}"
                if n_params == 0:
                    param_text = f"{param_text:>{param_max_length}}"
                else:
                    param_text = f"{param_text:>{param_max_length}} ({per:.1f}%)"

                print(f"{name_text} │ {input_args_text}{input_text} │ {output_text} │ {param_text}")

            if len(input_texts) > 1 or len(output_texts) > 1:
                empty = ""
                if len(root.blocks) > 0:
                    name_text = f"{append + '│    ':<{name_max_length}}"
                else:
                    name_text = f"{append + '     ':<{name_max_length}}"
                input_text = f"{empty:>{input_max_length}}"
                input_args_text = f"{empty:>{input_max_args_length}}"
                output_text = f"{empty:>{output_max_length}}"
                param_text = f"{empty:>{param_max_length}}"
                print(f"{name_text} │-{input_args_text}{input_text}-│-{output_text}-│ {param_text}")

            if is_last and len(root.blocks) == 0:
                empty = ""
                name_text = f"{append:<{name_max_length}}"
                input_text = f"{empty:>{input_max_length}}"
                input_args_text = f"{empty:>{input_max_args_length}}"
                output_text = f"{empty:>{output_max_length}}"
                param_text = f"{empty:>{param_max_length}}"
                print(f"{name_text} │ {input_args_text}{input_text} │ {output_text} │ {param_text}")

            for i, block in enumerate(root.blocks):
                if i == length - 1:
                    if is_last:
                        recursive(block, append, is_last=True)
                    else:
                        recursive(block, append, is_last=True)
                else:
                    if is_last:
                        recursive(block, append)
                    else:
                        recursive(block, append)

        for root in self.roots:
            recursive(root, is_last=True)

    def print_line(self, block, directory_text, input_args_text, input_size_text, output_size_text, param_text):
        print(f"{directory_text} │ {input_args_text}{input_size_text} │ {output_size_text} │ {param_text}")

    def input_texts(self, block):
        lines = []
        shape = self.get_shape(block.module_in)
        line = [str(list(s)) for s in shape]
        lines.append(line)
        for b in block.blocks:
            lines += self.input_texts(b)

        return lines

    def output_texts(self, block):
        lines = []
        shape = self.get_shape(block.module_out)
        line = []
        for s in shape:
            if isinstance(s, list):
                line.append(f"{str(list(s[0]))} * {len(s)}")
            else:
                line.append(str(list(s)))

        lines.append(line)
        for b in block.blocks:
            lines += self.output_texts(b)

        return lines

    def space(self, depth):
        s = ""
        for _ in range(depth):
            s += ' ' * 4
        return s

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
                    input_kwargs[key] = module_in[i:-1]
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
            out[key] = InputSizeStr(value)
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
