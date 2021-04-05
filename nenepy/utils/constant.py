class ConstantMeta(type):

    def __new__(mcs, name, bases, namespace):
        base_attrs = set(*[base.__dict__.keys() for base in bases])
        base_const_variables = set([attr for attr in base_attrs if not mcs.is_builtin(attr)])
        const_variables = set([attr for attr in namespace if not mcs.is_builtin(attr)])

        duplicated_variables = [v for v in const_variables if v in base_const_variables]
        if len(duplicated_variables) > 0:
            raise ValueError(f"Duplicate Variables'{duplicated_variables}'")

        return type.__new__(mcs, name, bases, namespace)

    def __setattr__(self, key, value):
        if self.is_builtin(key):
            super(ConstantMeta, self).__setattr__(key, value)
        else:
            raise Exception(f"Can't set attribute to Constant {key}")

    @staticmethod
    def is_builtin(x):
        """
        Args:
            x (str):

        Returns:
            bool:

        """
        if x.startswith("__") and x.endswith("__"):
            return True
        return False


class Constant(metaclass=ConstantMeta):

    def __setattr__(self, key, value):
        super(Constant, self).__setattr__(key, value)
