import warnings


class TensorBoard:

    def __init__(self):
        self._tags = set()
        self._conversions = dict()

    def _to_scalar_tag(self, namespace, graph_name):
        """

        Args:
            namespace (str):
            graph_name (str):

        """
        namespace, graph_name = self.check_to_contain_forbidden_char(namespace, graph_name)
        return f"{namespace}/{graph_name}"

    @staticmethod
    def _decompose_scalar_tag(tag):
        """

        Args:
            tag (str):

        """
        namespace, graph_name = tag.split("/")
        return namespace, graph_name

    @staticmethod
    def _decompose_scalars_dir_name(dir_name):
        """

        Args:
            dir_name (str):

        """
        dir_name = dir_name.replace("_", "/")
        namespace, graph_name, scalar_key = dir_name.split("/")
        return namespace, graph_name, scalar_key

    @staticmethod
    def _to_image_tag(namespace, name):
        """

        Args:
            namespace (str):
            name (str):

        """
        return f"{namespace}/{name}"

    @staticmethod
    def _decompose_image_tag(tag):
        """

        Args:
            tag (str):

        """
        namespace, name = tag.split("/")
        return namespace, name

    def check_to_contain_forbidden_char(self, *texts):
        """

        Args:
            *texts (str):

        Returns:

        """
        texts = list(texts)
        for i, text in enumerate(texts):
            if text in self._tags:
                continue
            else:
                if text in self._conversions:
                    t = self._conversions[text]
                    texts[i] = t
                    self._tags.add(t)
                elif "/" in text or "_" in text or " " in text:
                    t = text.replace("/", "-").replace("_", "-").replace(" ", "-")
                    warnings.warn(f"\nIt cannot contain '/' , '_' and ' ' -> '{text}' contains\nChange {text} -> {t}")
                    texts[i] = t
                    self._tags.add(t)
                    self._conversions[text] = t

        return texts
