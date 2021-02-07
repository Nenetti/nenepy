from .value import Value
from .values import Values
from .dict_value import DictValue


class Output:

    def __init__(self, module, values):
        self.raw_values = values
        self.values = self.analyze_values(values)

    @classmethod
    def analyze_values(cls, values):
        def recursive(v):
            if cls.is_iterable(v):
                if isinstance(v, dict):
                    return DictValue(dict((key, recursive(v)) for key, v in v.items()))

                return Values([recursive(v) for v in v])

            return Value(v)

        return recursive(values)

    @staticmethod
    def is_iterable(value):
        if isinstance(value, (tuple, list, set, dict)):
            return True
        return False
