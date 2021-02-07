from .architecture import Architecture


class BlockPrinter:

    def __init__(self, block):
        self.block = block

    def print_info(self):
        print(self.block.architecture.print_format)
