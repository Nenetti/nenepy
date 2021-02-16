from nenepy.torch.utils.summary.modules import AbstractModule


class NetworkArchitecture(AbstractModule):

    def __init__(self, module):
        """

        Args:
            module (Module):

        """
        super(NetworkArchitecture, self).__init__()
        self.module = module
        self.text = self._to_text_format(module)
        self.set_n_max_length(self.text)

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================
    @property
    def print_formats(self):
        return self.text

    def to_top_formant(self):
        def recursive(m):
            if m is not None and m.parent_module is not None:
                parent_format = recursive(m.parent_module)
                self_format = self._to_connection_format(m)
                return f"{parent_format}{self_format}"
            else:
                return ""

        if len(self.module.child_modules) > 0:
            return recursive(self.module.child_modules[0])
        else:
            return recursive(self.module)

    def to_bottom_format(self):
        def recursive(m):
            if m is not None and m.parent_module is not None:
                parent_format = recursive(m.parent_module)
                self_format = self._to_connection_format(m)
                return f"{parent_format}{self_format}"
            else:
                return ""

        if self.module.parent_module is not None:
            return recursive(self.module.parent_module) + f"{'│ ':>{self.indent_space}}"
        else:
            return "" + f"{' ':>{self.indent_space}}"

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def _to_text_format(cls, module):
        parent_directory_format = cls._to_parent_formant(module)
        self_directory_format = cls._to_directory_format(module)
        return f"{parent_directory_format}{self_directory_format}{module.module_name}"

    @classmethod
    def _to_parent_formant(cls, module):
        """

        Args:
            module (Module):

        Returns:

        """

        def recursive(m):
            if m is not None and m.parent_module is not None:
                parent_format = recursive(m.parent_module)
                self_format = cls._to_connection_format(m)
                return f"{parent_format}{self_format}"
            else:
                return ""

        return recursive(module.parent_module)

    @classmethod
    def _to_connection_format(cls, module):
        if module.is_last_module_in_sequential:
            text = ""
        else:
            text = "│ "
        return f"{text:>{cls.indent_space}}"

    @classmethod
    def _to_directory_format(cls, module):
        if module.is_root:
            return ""
        else:
            directory_type = cls._to_directory_type(module)
            return f"{directory_type:>{cls.indent_space}}"

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def _to_directory_type(module):
        if module.is_root:
            return ""
        if module.is_last_module_in_sequential:
            return "└ "
        else:
            return "├ "
