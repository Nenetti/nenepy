import inspect
from collections import OrderedDict

from .output import Output


class Input(Output):

    def __init__(self, module, values):
        super(Input, self).__init__(module, self.to_argument_dict(module, values))

    @staticmethod
    def to_argument_dict(module, values):
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
