from .value import Value
from .value_list import ValueList
from .value_dict import ValueDict


class Output:

    def __init__(self, module, values):
        self.raw_values = values
        self.values = self.analyze_values(values)
        self.n_nest = self.calc_nest(self.values)

    @staticmethod
    def calc_nest(values):
        def recursive(value):
            if isinstance(value, (ValueList, ValueDict)):
                if len(value.values) > 0:
                    return max([recursive(v) for v in value.values]) + 1

            return 0

        return recursive(values)

    @classmethod
    def analyze_values(cls, values):
        def recursive(v):
            if cls.is_iterable(v):
                if isinstance(v, dict):
                    return ValueDict(dict((key, recursive(v)) for key, v in v.items()))

                return ValueList([recursive(v) for v in v])

            return Value(v)

        return recursive(values)

    @staticmethod
    def is_iterable(value):
        if isinstance(value, (tuple, list, set, dict)):
            return True
        return False
