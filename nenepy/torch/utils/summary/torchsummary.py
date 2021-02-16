import sys
from collections import OrderedDict
from time import sleep

import torch
import torch.nn as nn

from .modules import *
from .modules.printer.block_printer import BlockPrinter
import time


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
        self.modules = []
        self.roots = []
        self.now_block = None
        self.n_blocks = 0
        self.ordered_modules = []
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

        # print(self.roots[0].output_texts)
        # sys.exit()

        self.name_texts()
        sleep(self.sleep_time)
        # remove these hooks
        for h in self.hooks:
            h.remove()

    def forward_tensor(self, input_tensor, **kwargs):
        # batch_size of 2 for batchnorm
        if not isinstance(input_tensor, (tuple, list, dict, set)):
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

        # print(BlockPrinter.get_input_max_tensor_length(self.ordered_blocks))
        # print(BlockPrinter.get_output_max_tensor_length(self.ordered_blocks))

        # self.get_input_max_tensor_length(self.block.module.input)
        # Architecture.init_constructions(self.roots)
        self.adjust()
        self.print_network()

        sleep(self.sleep_time)
        # remove these hooks
        for h in self.hooks:
            h.remove()

    def adjust(self):
        # outputs = [module.output for module in self.ordered_modules]
        # tensors = Output.get_all_tensors(outputs)
        # max_n_dims = Value.calc_max_n_dims(tensors)
        # max_each_dim_size = Value.calc_max_each_dim_size(tensors, max_n_dims)
        #
        # OutputPrinter.max_n_dims = max_n_dims
        # OutputPrinter.max_each_dim_size = max_each_dim_size
        #
        # # print(max_n_dims)
        # # print(max_each_dim_size)
        # # sys.exit()
        # inputs = [module.input for module in self.ordered_modules]
        # tensors = Input.get_all_tensors(inputs)
        # max_n_dims = Value.calc_max_n_dims(tensors)
        # max_each_dim_size = Value.calc_max_each_dim_size(tensors, max_n_dims)
        #
        # InputPrinter.max_each_dim_size = max_each_dim_size
        # InputPrinter.max_key_length = Input.get_max_dict_key_length(outputs)
        # print(InputPrinter.max_key_length)

        # for i, tensor in enumerate(tensors):
        #     print(i, tensor.shape)
        pass

    def print_network(self):
        printers = []
        for module in self.ordered_modules:
            module.adjust()
            printer = BlockPrinter(module)
            printers.append(printer)

        BlockPrinter.to_adjust(self.ordered_modules, printers)

        print(BlockPrinter.to_print_header())

        for printer in printers:
            print_formats = printer.to_print_format()
            for print_format in print_formats:
                print(print_format)

        print(BlockPrinter.to_print_header(reverse=True))
        print(BlockPrinter.to_print_footer())

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
        summary_module = Module()
        if len(self.modules) == 0:
            summary_module.is_root = True
            self.roots.append(summary_module)

        self.ordered_modules.append(summary_module)
        self.modules.append((summary_module, time.time()))

    def hook(self, module, module_in, module_out):
        """

        Args:
            module (nn.Module):
            module_in:
            module_out:

        """
        summary_module, start_time = self.modules.pop(-1)
        summary_module.processing_time = time.time() - start_time
        summary_module.initialize(module, module_in, module_out)

        if len(summary_module.child_modules) > 0:
            summary_module.child_modules[-1].is_last_module_in_sequential = True

        if len(self.modules) > 0:
            parent_block = self.modules[-1][0]
            parent_block.child_modules.append(summary_module)
            summary_module.parent_module = parent_block
