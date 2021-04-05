class TensorBoard:

    @classmethod
    def _to_scalar_tag(cls, namespace, graph_name):
        """

        Args:
            namespace (str):
            graph_name (str):

        """
        cls.check_to_contain_forbidden_char(namespace, graph_name)
        return f"{namespace}/{graph_name}"

    @classmethod
    def _decompose_scalar_tag(cls, tag):
        """

        Args:
            tag (str):

        """
        namespace, graph_name = tag.split("/")
        return namespace, graph_name

    @classmethod
    def _decompose_scalars_dir_name(cls, dir_name):
        """

        Args:
            dir_name (str):

        """
        dir_name = dir_name.replace("_", "/")
        namespace, graph_name, scalar_key = dir_name.split("/")
        return namespace, graph_name, scalar_key

    @classmethod
    def _to_image_tag(cls, namespace, name):
        """

        Args:
            namespace (str):
            name (str):

        """
        # cls.check_to_contain_forbidden_char(namespace, name)
        return f"{namespace}/{name}"

    @classmethod
    def _decompose_image_tag(cls, tag):
        """

        Args:
            tag (str):

        """
        namespace, name = tag.split("/")
        return namespace, name

    @staticmethod
    def check_to_contain_forbidden_char(*texts):
        for text in texts:
            if "/" in text or "_" in text:
                raise ValueError(f"It cannot contain '/' and '_' -> '{text}' contains ")
