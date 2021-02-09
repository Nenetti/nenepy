from collections import Counter
from .value import Value


class ValueDict:

    def __init__(self, value_dict):
        """

        Args:
            value_dict (dict):
        """
        self.value_dict = value_dict
        self.text = self._to_text(value_dict)
        self.n_print_elements = len(value_dict)

    def items(self):
        return self.value_dict.items()

    @property
    def values(self):
        return list(self.value_dict.values())

    @property
    def keys(self):
        return list(self.value_dict.keys())

    @staticmethod
    def _to_text(values):
        return str([(key, value.text) for key, value in values.items()])

    @staticmethod
    def is_dict(values):
        return isinstance(values, dict)

    @staticmethod
    def _to_type(value):
        return str(value.__class__.__name__)
