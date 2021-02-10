import inspect
import sys
from collections import OrderedDict, Counter
from numbers import Number
from time import sleep
import time

import numpy as np
import torch
import torch.nn as nn
from .modules import *


class TorchSummary:

    def __init__(self, model, batch_size=2, is_validate=False, device="cuda", sleep_time=0, is_exit=False):
        """

        Args:
            model (nn.Module):
            batch_size:
            device:
        """
        self.model = model.to(device)
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
        self.sleep_time = sleep_time
        self.is_exit = is_exit
        if is_validate:
            self.model.eval()
        else:
            self.model.train()

    def __call__(self, input_size, *args, **kwargs):
        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        # x = [torch.rand(self.batch_size, *in_size).to(self.device) for in_size in input_size]
        x = [torch.ones(self.batch_size, *in_size).to(self.device) for in_size in input_size]
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
        sleep(self.sleep_time)
        # remove these hooks
        for h in self.hooks:
            h.remove()

    def forward_tensor(self, input_tensor, **kwargs):
        # batch_size of 2 for batchnorm
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = [input_tensor]

        x = input_tensor

        # create properties
        # register hook
        self.model.apply(self.register_hook)

        # make a forward pass
        self.model(*x, **kwargs)

        Block.calc_depth(self.roots)
        # print(BlockPrinter.get_input_max_tensor_length(self.ordered_blocks))
        # print(BlockPrinter.get_output_max_tensor_length(self.ordered_blocks))

        # self.get_input_max_tensor_length(self.block.module.input)
        # Architecture.init_constructions(self.roots)

        self.print_network()

        sleep(self.sleep_time)
        # remove these hooks
        for h in self.hooks:
            h.remove()

    def print_network(self):
        printers = []
        for block in self.ordered_blocks:
            printer = BlockPrinter(block)
            printers.append(printer)

        BlockPrinter.to_adjust(printers)

        print(BlockPrinter.to_print_header())

        for printer in printers:
            print_formats = printer.to_print_format()
            for print_format in print_formats:
                print(print_format)

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

    def print_string(self, directory, input_text, output_text, weight_param_text, bias_param_text, param_per_text, is_train_text, is_untrain_text, time_text,
                     is_boundary):

        partition = "  │  "
        if is_boundary:
            partition = " -│- "

        print(
            f"{directory}{partition}{input_text}{partition}{output_text}{partition}{weight_param_text}{partition}{bias_param_text}{partition}{param_per_text}{partition}{is_train_text}{partition}{is_untrain_text}{partition}{time_text}  │ "
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
        untrain_length = Block.untrain_length
        time_length = Block.time_length

        param_length = weight_length + bias_length + param_per_length + train_length + untrain_length + 20

        directory_empty = f"{' ' * architecture_length}"
        input_empty = f"{' ' * input_length}"
        output_empty = f"{' ' * output_length}"
        weight_empty = f"{' ' * weight_length}"
        bias_empty = f"{' ' * bias_length}"
        param_per_empty = f"{' ' * param_per_length}"
        train_empty = f"{' ' * train_length}"
        untrain_empty = f"{' ' * untrain_length}"
        param_empty = f"{'-' * param_length}"
        time_empty = f"{' ' * time_length}"

        architecture_title = f"{'Network Architecture':^{architecture_length}}"
        input_title = f"{'Input':^{input_length}}"
        output_title = f"{'Output':^{output_length}}"
        param_title = f"{'Parameters':^{param_length}}"
        weight_title = f"{'Weight':^{weight_length}}"
        bias_title = f"{'Bias':^{bias_length}}"
        param_per_title = f"{'Total(%)':^{param_per_length}}"
        train_title = f"{'Train':^{train_length}}"
        untrain_title = f"{'Untrain':^{untrain_length}}"
        time_title = f"{'Time (s)':^{time_length}}"

        param_line = self.to_line("     ", weight_title, bias_title, param_per_title, train_title, untrain_title)
        border_line = self.to_line("==│==", '=' * architecture_length, '=' * input_length, '=' * output_length, '=' * param_length, '=' * time_length) + "==│"
        param_detail_line = self.to_line("  │  ", directory_empty, input_empty, output_empty, param_line, time_empty) + "  │"
        param_line = self.to_line("  │  ", directory_empty, input_empty, output_empty, param_title, time_empty) + "  │"
        title_line = self.to_line("  │  ", architecture_title, input_title, output_title, param_empty, time_title) + "  │"

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
            is_untrains = [root.is_untrain_text]
            times = [root.time_text]

            is_boundaries = [False]

            max_length = max(len(inputs), len(outputs))
            directories, new_append = self.to_directories(root, directory, architecture_length, max_length, append, is_last)
            inputs += [" " * len(inputs[0])] * (max_length - len(inputs))
            outputs += [" " * len(outputs[0])] * (max_length - len(outputs))
            weight_params += [" " * len(weight_params[0])] * (max_length - len(weight_params))
            bias_params += [" " * len(bias_params[0])] * (max_length - len(bias_params))
            per_params += [" " * len(per_params[0])] * (max_length - len(per_params))
            is_trains += [" " * len(is_trains[0])] * (max_length - len(is_trains))
            is_untrains += [" " * len(is_untrains[0])] * (max_length - len(is_untrains))
            times += [" " * len(times[0])] * (max_length - len(times))

            is_boundaries += [False] * (max_length - len(is_boundaries))

            need_before_boundary = ((max_length > 1) and (not before_is_boundary)) or root.depth == 0
            need_before_space = (n_child_blocks > 0 or child_first) and (not before_is_space and not before_is_boundary)
            need_boundary = max_length > 1 or root.depth == 0
            need_space = is_last and len(root.blocks) == 0

            if need_before_space:
                d = f"{f'{append}│ ':<{architecture_length}}"
                self.print_string(d, input_empty, output_empty, weight_empty, bias_empty, param_per_empty, train_empty, untrain_empty, time_empty, False)

            if need_before_boundary:
                if root.depth != 0:
                    d = f"{f'{append}│    ':<{architecture_length}}"
                else:
                    d = f"{f'{append}     ':<{architecture_length}}"
                self.print_string(d, input_empty, output_empty, weight_empty, bias_empty, param_per_empty, train_empty, untrain_empty, time_empty, True)

            for i in range(len(directories)):
                self.print_string(directories[i], inputs[i], outputs[i], weight_params[i], bias_params[i], per_params[i], is_trains[i], is_untrains[i],
                                  times[i], is_boundaries[i])

            before_is_space = False
            before_is_boundary = False
            if need_boundary:
                d = f"{new_append + '│    '}" if len(root.blocks) > 0 else f"{new_append + '     '}"
                d = f"{d:<{architecture_length}}"
                self.print_string(d, input_empty, output_empty, weight_empty, bias_empty, param_per_empty, train_empty, untrain_empty, time_empty, True)
                before_is_boundary = True

            if need_space:
                d = f"{new_append:<{architecture_length}}"
                self.print_string(d, input_empty, output_empty, weight_empty, bias_empty, param_per_empty, train_empty, untrain_empty, time_empty, False)
                before_is_space = True

            for i, block in enumerate(root.blocks):
                if i == 0 and i + 1 != len(root.blocks):
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
        total_output_size = (sum([r.output_size for r in Block.all_blocks]) * 4 * 2) / (1024 ** 2)
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

    # ==================================================================================================
    #
    #   Hook
    #
    # ==================================================================================================

    def register_hook(self, module):
        """

        Args:
            module (nn.Module):

        """
        if (isinstance(module, nn.Sequential)) or (isinstance(module, nn.ModuleList)):
            return

        self.hooks.append(module.register_forward_pre_hook(self.pre_hook))
        self.hooks.append(module.register_forward_hook(self.hook))

    def pre_hook(self, module, module_in):
        """

        Args:
            module (nn.Module):
            module_in:

        """
        block = Block()
        if len(self.blocks) == 0:
            self.roots.append(block)

        self.ordered_blocks.append(block)
        self.blocks.append((block, time.time()))

    def hook(self, module, module_in, module_out):
        """

        Args:
            module (nn.Module):
            module_in:
            module_out:

        """
        block, start_time = self.blocks.pop(-1)
        block.processing_time = time.time() - start_time
        block.module = Module(module, module_in, module_out)

        if len(self.blocks) > 0:
            parent_block = self.blocks[-1][0]
            block.parent = parent_block
            parent_block.child_blocks.append(block)
