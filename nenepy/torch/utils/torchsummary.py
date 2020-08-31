import inspect
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class Block:

    def __init__(self, index):
        self.module = None
        self.module_in = None
        self.module_out = None
        self.blocks = []
        self.kwargs = []
        self.index = index
        self.depth = 0
        self.bottom = False
        self.n_params = 0
        self.weight_grad = None

    @property
    def name(self):
        return str(self.module.__class__).split(".")[-1].split("'")[0]

    def add(self, module):
        self.blocks.append(module)

    def has_bottom_children(self):
        if len(self.blocks) > 0:
            for block in self.blocks:
                if block.bottom:
                    return True

        return False

    def get_depth(self):
        def recursive(b, d):
            if len(b.blocks) > 0:
                d = max(d, *[recursive(bb, d + 1) for bb in b.blocks])
            return d

        return recursive(self, 0)


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
        device = self.device.lower()

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(self.batch_size, *in_size).to(device) for in_size in input_size]
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

    def get_length(self, texts):
        lengths = [0] * len(texts)
        for i, text in enumerate(texts):
            if isinstance(texts, str):
                lengths[i] = len(text)
            else:
                lengths[i] = max([len(t) for t in text])

        return max(lengths)

    def get_max_directory_structure_length(self):
        def recursive(block):
            index_space = self.space(block.depth)
            length = len(f"{index_space}    {block.name}    ")

            for b in block.blocks:
                length = max(length, recursive(b))

            return length

        return max([recursive(b) for b in self.roots])

    def get_max_input_length(self):
        def recursive(block):
            shapes = self.get_dict_shape_str(block.kwargs)
            if len(shapes) > 1:
                length = max([len(f"{arg}: {str(shape)}") for arg, shape in shapes.items()])

            else:
                length = max([len(str(shape)) for shape in shapes.values()])

            for b in block.blocks:
                length = max(length, recursive(b))

            return length

        return max([recursive(b) for b in self.roots])

    def get_max_output_length(self):
        def recursive(block):
            shapes = self.get_list_shape_str(block.module_out)
            length = max([len(str(shape)) for shape in shapes])

            for b in block.blocks:
                length = max(length, recursive(b))

            return length

        return max([recursive(b) for b in self.roots])

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

    def name_texts(self, space_size=4):
        name_max_length = self.get_max_directory_structure_length()
        input_max_length = self.get_max_input_length()
        output_max_length = self.get_max_output_length()
        param_max_length = self.get_max_param_length()

        def recursive(root, append="", is_last=False):
            length = len(root.blocks)
            lines = []
            if is_last:
                name = f"{append}└ {root.name}"
                lines.append(name)
            else:
                name = f"{append}├ {root.name}"
                lines.append(name)

            input_shapes = self.get_dict_shape_str(root.kwargs)
            input_texts = []
            for arg, value in input_shapes.items():
                if len(input_shapes) > 1:
                    input_text = f"{arg}: {value}"
                    input_texts.append(input_text)
                else:
                    input_text = f"{value}"
                    input_texts.append(input_text)

            output_shapes = self.get_list_shape_str(root.module_out)
            output_texts = []
            for value in output_shapes:
                input_text = f"{value}"
                output_texts.append(input_text)

            if is_last:
                append = append + "     "
            else:
                append = append + "│    "

            max_length = max(len(input_texts), len(output_texts))
            lines += [append + "│    "] * (max_length - len(lines))
            input_texts += [""] * (max_length - len(input_texts))
            output_texts += [""] * (max_length - len(output_texts))

            for i in range(max(len(lines), len(input_texts), len(output_texts))):
                name_text = lines[i]
                input_text = input_texts[i]
                output_text = output_texts[i]
                param_text = str(root.n_params)
                n_params = root.n_params
                per = (n_params / self.total_params) * 100

                name_text = f"{name_text:<{name_max_length}}"
                input_text = f"{input_text:<{input_max_length}}"
                output_text = f"{output_text:<{output_max_length}}"
                if n_params == 0:
                    param_text = f"{param_text:>{param_max_length}}"
                else:
                    param_text = f"{param_text:>{param_max_length}} ({per:.1f}%)"

                print(name_text + " │ " + input_text + " │ " + output_text + " │ " + param_text)

            if len(input_texts) > 1 or len(output_texts) > 1:
                empty = ""
                name_text = f"{append + '│    ':<{name_max_length}}"
                input_text = f"{empty:>{input_max_length}}"
                output_text = f"{empty:>{output_max_length}}"
                print(name_text + " │-" + input_text + "-│-" + output_text + "-│")

            if is_last and len(root.blocks) == 0:
                empty = ""
                name_text = f"{append:<{name_max_length}}"
                input_text = f"{empty:>{input_max_length}}"
                output_text = f"{empty:>{output_max_length}}"
                print(name_text + " │ " + input_text + " │ " + output_text + " │")

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
        self.n_blocks += 1
        block = Block(index=self.n_blocks)
        if len(self.blocks) == 0:
            self.roots.append(block)

        self.blocks.append(block)
        self.ordered_blocks.append(block)

    def hook(self, module, module_in, module_out):
        if isinstance(module_out, torch.Tensor):
            module_out = [module_out]

        block = self.blocks.pop(-1)
        block.module = module
        block.module_in = module_in
        block.module_out = module_out
        kwargs = OrderedDict()
        keys = list(inspect.signature(module.forward).parameters.keys())
        for i in range(len(module_in)):
            kwargs[keys[i]] = module_in[i]

        block.kwargs = kwargs

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
    def get_dict_shape_str(kwargs):
        out = OrderedDict()
        for arg, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                out[arg] = str(list(value.size()))

            elif isinstance(value, (list, tuple)):
                if len(value) > 0:
                    out[arg] = f"{str(list(value[0].size()))} * {len(value)}"

            elif isinstance(value, dict):
                if len(value) > 0:
                    for k, v in value.items():
                        out[f"{arg}-{k}"] = f"{str(list(v[0].size()))} * {len(value)}"

        return out

    @staticmethod
    def get_list_shape_str(tensors):
        out = []
        for value in tensors:
            if isinstance(value, torch.Tensor):
                out.append(str(list(value.size())))

            elif isinstance(value, (list, tuple)):
                t = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        t.append(str(list(v.size())))
                    else:
                        t.append(str(v).replace("torch.Size", ""))

                if len(set(t)) == 1:
                    out.append(f"{t[0]} * {len(value)}")
                else:
                    out.append(f"{t} * {len(value)}")

            elif isinstance(value, dict):
                if len(value) > 0:
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            out.append(f"{k}: {str(list(v.size()))}")
                        else:
                            out.append(f"{k}: {str(v.__class__)}")

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
