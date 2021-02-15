from nenepy.torch.utils.summary.modules.printer.abstract_printer import AbstractPrinter


class ArchitecturePrinter(AbstractPrinter):

    def __init__(self, module):
        """

        Args:
            module (Module):

        """
        self.text_format = self.to_text_format(module)
        self.set_n_max_length(self.text_format)

    def to_print_format(self):
        return self.text_format

    @classmethod
    def to_child_formant(cls, module):
        """

        Args:
            module (Module):

        Returns:

        """

        def recursive(b, child_format):
            if b.parent_module is not None:
                self_format = cls.to_child_directory_format(b)
                return recursive(b.parent_module, f"{self_format}{child_format}")
            else:
                return child_format

        if len(module.child_modules) > 0:
            return recursive(module.child_modules[0], "")
        else:
            return recursive(module, "")

    @classmethod
    def to_parent_formant2(cls, module):
        """

        Args:
            module (Module):

        Returns:

        """

        def recursive(b, child_format):
            if b.parent_module is not None:
                self_format = cls.to_child_directory_format(b)
                return recursive(b.parent_module, f"{self_format}{child_format}")
            else:
                return child_format

        if module.parent_module is not None:
            return recursive(module.parent_module, "") + f"{'│ ':>{cls.indent_space}}"
        else:
            return "" + f"{' ':>{cls.indent_space}}"

    @classmethod
    def to_parent_formant(cls, module):
        """

        Args:
            module (Module):

        Returns:

        """

        def recursive(b, child_format):
            if b.parent_module is not None:
                self_format = cls.to_child_directory_format(b)
                return recursive(b.parent_module, f"{self_format}{child_format}")
            else:
                return child_format

        if module.parent_module is not None:
            return recursive(module.parent_module, "")
        else:
            return ""

    @classmethod
    def to_text_format(cls, module):
        parent_directory_format = cls.to_parent_formant(module)
        directory_format = cls.to_directory_format(module)
        print_format = f"{parent_directory_format}{directory_format}{module.module_name}"
        return print_format

    @classmethod
    def to_child_directory_format(cls, module):
        if module.is_last_module_in_sequential:
            return f"{'':>{cls.indent_space}}"
        else:
            return f"{'│ ':>{cls.indent_space}}"

    @classmethod
    def to_directory_format(cls, module):
        if module.is_root:
            return ""
        else:
            directory_type = cls.to_directory_type(module)
            return f"{directory_type:>{cls.indent_space}}"

    @staticmethod
    def to_directory_type(module):
        if module.is_root:
            return ""
        if module.is_last_module_in_sequential:
            return "└ "
        else:
            return "├ "
