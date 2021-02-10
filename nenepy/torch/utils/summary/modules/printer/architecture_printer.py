from nenepy.torch.utils.summary.modules.block import Block
from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter


class ArchitecturePrinter(AbstractPrinter):

    def __init__(self, block):
        """

        Args:
            block (Block):

        """
        self.text_format = self.to_text_format(block)
        self.set_n_max_length(self.text_format)

    def to_print_format(self):
        return self.text_format

    @classmethod
    def to_child_formant(cls, block):
        """

        Args:
            block (Block):

        Returns:

        """

        def recursive(b, child_format):
            if b.parent is not None:
                self_format = cls.to_child_directory_format(b)
                return recursive(b.parent, f"{self_format}{child_format}")
            else:
                return child_format

        if len(block.child_blocks) > 0:
            return recursive(block.child_blocks[0], "")
        else:
            return recursive(block, "")

    @classmethod
    def to_parent_formant(cls, block):
        """

        Args:
            block (Block):

        Returns:

        """

        def recursive(b, child_format):
            if b.parent is not None:
                self_format = cls.to_child_directory_format(b)
                return recursive(b.parent, f"{self_format}{child_format}")
            else:
                return child_format

        if block.parent is not None:
            return recursive(block.parent, "")
        else:
            return ""

    @classmethod
    def to_text_format(cls, block):
        parent_directory_format = cls.to_parent_formant(block)
        directory_format = cls.to_directory_format(block)
        print_format = f"{parent_directory_format}{directory_format}{block.module.module_name}"
        return print_format

    @classmethod
    def to_child_directory_format(cls, block):
        if block.is_last_module_in_sequential:
            return f"{'':>{cls.indent_space}}"
        else:
            return f"{'│ ':>{cls.indent_space}}"

    @classmethod
    def to_directory_format(cls, block):
        if block.is_root:
            return ""
        else:
            directory_type = cls.to_directory_type(block)
            return f"{directory_type:>{cls.indent_space}}"

    @staticmethod
    def to_directory_type(block):
        if block.is_root:
            return ""
        if block.is_last_module_in_sequential:
            return "└ "
        else:
            return "├ "

    def _get_max_directory_structure_length(self):
        max_length = 0
        for block in self.ordered_blocks:
            space = " " * self.indent_space * block.depth
            text = f"{space}{block.architecture_str}"
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
