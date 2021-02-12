import inspect
import sys
from collections import OrderedDict, Counter
from numbers import Number
from time import sleep
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

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
            self.model.requires_grad_(False)
        else:
            self.model.train()
            self.model.requires_grad_(True)

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
        y = self.model(*x, **kwargs)
        # print(y)
        # (y["loss_classifier"] + y["loss_box_reg"] + y["loss_box_reg"] + y["loss_rpn_box_reg"]).backward()
        # (y["loss_classifier"] - 1).sum().backward()
        # print(input_tensor[0].grad)
        # print(y[0]["boxes"].grad)
        #
        # print("A")
        # time.sleep(1000)

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

        print(BlockPrinter.to_print_header(reverse=True))
        print(BlockPrinter.to_print_footer())

    def _get_max_directory_structure_length(self):
        max_length = 0
        for block in self.ordered_blocks:
            index_space = " " * 4 * block.depth
            text = f"{index_space}    {block.architecture_str}    "
            max_length = max(max_length, len(text))

        return max_length

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
