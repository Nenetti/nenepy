from collections import Counter
from .value import Value


class DictValue:

    def __init__(self, values):
        self.values = values
        self.text = self._to_text(values)
        self.n_print_elements = len(values)

    @staticmethod
    def _to_text(values):
        return str([(key, value.text) for key, value in values.items()])

    @staticmethod
    def is_dict(values):
        return isinstance(values, dict)

    @staticmethod
    def _to_type(value):
        return str(value.__class__.__name__)
