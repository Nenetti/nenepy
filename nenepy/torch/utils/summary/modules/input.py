import inspect
from collections import OrderedDict

from . import Value
from .output import Output


class Input(Output):

    def __init__(self, module, values):
        super(Input, self).__init__(self.argument_to_dict(module, values))

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def argument_to_dict(module, values):
        input_dict = OrderedDict()
        argument_dict = inspect.signature(module.forward).parameters
        n_module_in = len(values)
        for i, (key, value) in enumerate(argument_dict.items()):
            if i < n_module_in:
                if value.kind == inspect.Parameter.VAR_POSITIONAL:
                    # 引数名なし(**kwargsでない) *argsの場合は すべて*argsとしてまとめる．
                    input_dict[key] = values[i:]
                else:
                    input_dict[key] = values[i]
            else:
                # 入力で設定されなかったデフォルト値を持つ引数
                input_dict[key] = value.default
        return input_dict

    @classmethod
    def adjust(cls, modules):
        outputs = [module.input for module in modules]
        tensors = cls.get_all_tensors(outputs)
        max_n_dims = Value.calc_max_n_dims(tensors)
        cls._max_each_dim_size = Value.calc_max_each_dim_size(tensors, max_n_dims)
        cls._max_key_length = cls.get_max_dict_key_length(outputs)
