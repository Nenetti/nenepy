import sys
import time
from time import sleep

import torch
import torch.nn as nn
from nenepy.torch.interfaces import Mode
from nenepy.torch.utils.summary.modules.block_printer import BlockPrinter
from nenepy.torch.utils.summary.modules.module import Module


class TorchSummary:

    def __init__(self, model, forward_func=None, batch_size=2, mode=Mode.TRAIN, is_print=True, display_delay_time=0, device="cuda", is_exit=True):
        """

        Args:
            model (nn.Module):
            batch_size:
            device:
        """
        self.model = model.to(device)
        self.forward_func = forward_func
        self.batch_size = batch_size
        self.display_delay_time = display_delay_time
        self.device = device
        self.is_print = is_print
        self.hooks = []
        self.modules = []
        self.modules_dict = dict()
        self.roots = []
        self.ordered_modules = []
        self.is_exit = is_exit
        if mode == Mode.TRAIN:
            self.model.train()
        elif mode == Mode.PRETRAIN:
            self.model.pretrain()
        elif mode == Mode.VALIDATE:
            self.model.test()
        elif mode == Mode.VALIDATE:
            self.model.eval()

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def forward_size(self, *input_size, **kwargs):
        if not isinstance(input_size[0], (tuple, list, dict, set)):
            input_size = [input_size]

        x = [torch.randn(self.batch_size, *in_size).to(self.device) for in_size in input_size]

        return self._forward(x, **kwargs)

    def forward_tensor(self, *input_tensor, **kwargs):
        def recur(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            if isinstance(x, (tuple, list, set)):
                return [recur(v) for v in x]
            if isinstance(x, dict):
                return dict([(k, recur(v)) for k, v in x.items()])
            return x

        return self._forward(recur(input_tensor), **recur(kwargs))

    def __call__(self, input_tensor, **kwargs):
        return self.forward_tensor(input_tensor, **kwargs)

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    def _forward(self, x, **kwargs):
        self.model.apply(self._register_hook)
        if self.forward_func is not None:
            func = getattr(self.model, self.forward_func.__name__)
            out = func(*x, **kwargs)
        else:
            out = self.model(*x, **kwargs)

        sleep(self.display_delay_time)
        if self.is_print:
            self._print_network()

        self._remove()

        if self.is_exit:
            sys.exit()

        return out

    def _print_network(self):
        printers = [BlockPrinter(module) for module in self.ordered_modules]
        BlockPrinter.adjust(self.ordered_modules)

        print(BlockPrinter.to_header_text())

        for printer in printers:
            print_formats = printer.to_formatted_texts()
            for print_format in print_formats:
                print(print_format)

        print(BlockPrinter.to_header_text(reverse=True))
        print(BlockPrinter.to_footer_text())

        print()
        print(Module.to_summary_text(self.ordered_modules))

    def _remove(self):
        for h in self.hooks:
            h.remove()
        del self.hooks
        del self.modules
        del self.roots
        del self.ordered_modules

    # ==================================================================================================
    #
    #   Hook
    #
    # ==================================================================================================
    def _register_hook(self, module):
        """

        Args:
            module (nn.Module):

        """
        # if (isinstance(module, nn.Sequential)) or (isinstance(module, nn.ModuleList)):
        #     return

        self.hooks.append(module.register_forward_pre_hook(self._pre_hook))
        self.hooks.append(module.register_forward_hook(self._hook))

    def _pre_hook(self, module, module_in):
        """

        Args:
            module (nn.Module):
            module_in:

        """

        module_id = len(self.modules_dict) + 1
        is_duplicated = False
        if module in self.modules_dict:
            module_id = self.modules_dict[module]
            is_duplicated = True
        else:
            self.modules_dict[module] = module_id

        summary_module = Module(module, module_id, is_duplicated)
        if len(self.modules) == 0:
            summary_module.is_root = True
            self.roots.append(summary_module)

        self.ordered_modules.append(summary_module)
        self.modules.append((summary_module, time.time()))

    def _hook(self, module, module_in, module_out):
        """

        Args:
            module (nn.Module):
            module_in:
            module_out:

        """
        summary_module, start_time = self.modules.pop(-1)
        summary_module.processing_time = time.time() - start_time

        summary_module.init_in_out(module_in, module_out)

        if len(summary_module.child_modules) > 0:
            summary_module.child_modules[-1].is_last_module_in_sequential = True

        if len(self.modules) > 0:
            parent_block = self.modules[-1][0]
            parent_block.child_modules.append(summary_module)
            summary_module.parent_module = parent_block
