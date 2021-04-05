from enum import Enum, auto
from pathlib import Path

import torch

from nenepy.utils.dictionary import AttrDict


class CheckPointType(Enum):
    LATEST = auto()
    BEST_SCORE = auto()


class CheckPoint(dict):
    _EPOCH = "epoch"
    _SCORE = "score"
    _MODEL_STATE_DICT = "model_state_dict"
    _OPTIMIZER_STATE_DICT = "optimizer_state_dict"

    def __init__(self, path, state_dict):
        """

        Args:
            path (Path):
            state_dict (dict):
        """
        super(CheckPoint, self).__init__()
        self._path = path
        for key, value in state_dict.items():
            self[key] = value

    @property
    def epoch(self):
        return self[self._EPOCH]

    @property
    def score(self):
        return self[self._SCORE]

    @property
    def model_state_dict(self):
        return self[self._MODEL_STATE_DICT]

    @property
    def optimizer_state_dict(self):
        return self[self._OPTIMIZER_STATE_DICT]

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def save(self):
        torch.save(self, self._path)

    def delete(self):
        self._path.unlink()

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def generate(cls, path, model, optimizer, score, epoch, **kwargs):
        state_dicts = {
            cls._MODEL_STATE_DICT: model.state_dict(),
            cls._OPTIMIZER_STATE_DICT: optimizer.state_dict(),
            cls._EPOCH: epoch,
            cls._SCORE: score,
            **kwargs
        }
        return cls(path, state_dicts)

    @classmethod
    def load(cls, path):
        state_dicts = torch.load(path)
        return cls(path, state_dicts)
