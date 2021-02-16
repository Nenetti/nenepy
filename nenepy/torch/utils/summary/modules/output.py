import itertools

from .value import Value


class Output:

    def __init__(self, module, values):
        self.values = self._analyze_values(values)
        self.adjusted_texts = None

    # ==================================================================================================
    #
    #   Class Method
    #
    # ==================================================================================================
    # def to_adjust(self, max_n_dims, max_each_dim_size, max_key_length):
    @classmethod
    def _analyze_values(cls, values):
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

    @classmethod
    def get_all_tensors(cls, outputs):
        def recursive(value):
            if cls.is_iterable(value):
                if isinstance(value, dict):
                    return itertools.chain.from_iterable([recursive(v) for v in value.values()])
                else:
                    return itertools.chain.from_iterable([recursive(v) for v in value])
            elif isinstance(value, Value):
                return [value.value] if value.is_tensor else []
            else:
                raise TypeError()

        output_tensors = []
        for output in outputs:
            output_tensors += recursive(output.values)

        return output_tensors

    @classmethod
    def get_max_dict_key_length(cls, outputs):
        def func(dict):
            return max([len(key) for key in dict.keys()], default=0)

        return max([func(output.values) if isinstance(output.values, dict) else 0 for output in outputs], default=0)
