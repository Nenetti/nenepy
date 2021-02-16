from .value import Value


class Output:

    def __init__(self, module, values):
        self.values = self.analyze_values(values)

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    @classmethod
    def analyze_values(cls, values):
        def recursive(v):
            if cls.is_iterable(v):
                if isinstance(v, dict):
                    return dict((key, recursive(v)) for key, v in v.items())
                else:
                    return [recursive(v) for v in v]
            else:
                return Value(v)

        if not isinstance(values, dict):
            values = {"": values}
        return recursive(values)

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def is_iterable(value):
        if isinstance(value, (tuple, list, set, dict)):
            return True
        return False
