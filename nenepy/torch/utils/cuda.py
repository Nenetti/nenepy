import os
from collections import Iterable


class Cuda:
    CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
    CUDA_LAUNCH_BLOCKING = "CUDA_LAUNCH_BLOCKING"

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def set_device_ids(cls, ids):
        if isinstance(ids, str):
            pass
        elif isinstance(ids, int):
            ids = str(ids)
        elif isinstance(ids, Iterable):
            ids = ",".join(map(str, ids))
        else:
            raise TypeError(f"{ids} must be 'int', 'str' or 'Iterable', but got {type(ids)}")

        os.environ[cls.CUDA_VISIBLE_DEVICES] = ids

    @classmethod
    def to_synchronizing_cpu(cls):
        print("Warning: CUDA_LAUNCH_BLOCKING was applied")
        os.environ[cls.CUDA_LAUNCH_BLOCKING] = "1"
