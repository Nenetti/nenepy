from .block import Block


class Architecture:
    indent_space = 4

    def __init__(self, block, parent_directory_format):
        """

        Args:
            block (Block):
        """
        self.print_format = self.construct(block, parent_directory_format)
        self.directory_format = self.to_directory_format(block)
        self.child_directory_format = f"{parent_directory_format}{self.to_child_directory_format(block)}"

    @staticmethod
    def init_constructions(roots):
        """
        Args:
            roots (list[Block]):

        """

        def recursive(block, directory_format=""):
            block.architecture = Architecture(block, directory_format)
            for b in block.child_blocks:
                recursive(b, block.architecture.child_directory_format)

        for root in roots:
            recursive(root)

    @classmethod
    def construct(cls, block, parent_directory_format):
        n_space = cls.indent_space
        # if block.has_children():
        #     pass
        directory_type = cls.to_directory_type(block)
        directory_format = f"{directory_type:>{n_space}}"
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

    @classmethod
    def to_print_format(cls, block):
        """

        Args:
            block (Block):

        Returns:

        """
        n_space = cls.indent_space * block.depth
        directory_type = ""
        if block.is_last_module_in_sequential:
            directory_type = "└"
        else:
            directory_type = "├"

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
