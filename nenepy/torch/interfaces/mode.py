from enum import Enum


class Mode(Enum):
    PRETRAIN = "Pretrain"
    TRAIN = "Train"
    VALIDATE = "Validate"
    TEST = "Test"
