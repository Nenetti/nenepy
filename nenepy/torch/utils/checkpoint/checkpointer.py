import shutil
from pathlib import Path

import torch

from nenepy.torch.utils.checkpoint import CheckPoint, CheckPointType
from nenepy.utils.dictionary import AttrDict


class CheckPointer:
    _SUFFIX = ".pkl"

    def __init__(self, save_dir, model, optimizer, needs_load_checkpoints, n_saves=5):
        self._save_dir = Path(save_dir)
        self._model = model
        self._optimizer = optimizer
        self._latest_checkpoint = None
        self._score_checkpoints = []
        self._n_saves = n_saves
        self._needs_load_checkpoints = needs_load_checkpoints

        if self._needs_load_checkpoints:
            self._latest_checkpoint, self._score_checkpoints = self._load_checkpoints(self._save_dir, self._n_saves)
        else:
            if self._save_dir.exists():
                shutil.rmtree(self._save_dir)

            self._save_dir.mkdir()

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def add_checkpoint(self, score, epoch, **kwargs):
        if self._is_higher_score_checkpoint(score):
            score_checkpoint = self._create_score_checkpoint(score, epoch, **kwargs)
            score_checkpoint.save()
            if self._need_replace():
                self._score_checkpoints[-1].delete()
                self._score_checkpoints[-1] = score_checkpoint
            else:
                self._score_checkpoints.append(score_checkpoint)

            self._score_checkpoints = sorted(self._score_checkpoints, reverse=True, key=lambda x: x.score)

        latest_checkpoint = self._create_latest_checkpoint(score, epoch, **kwargs)
        latest_checkpoint.save()
        self._latest_checkpoint = latest_checkpoint

    def load_checkpoint(self, check_point_type=CheckPointType.LATEST):
        """

        Args:
            check_point_type (CheckPointType):

        Returns:
            CheckPoint:

        """
        if check_point_type == CheckPointType.LATEST:
            checkpoint = self._load_latest_checkpoint()
        elif check_point_type == CheckPointType.BEST_SCORE:
            checkpoint = self._load_best_score_checkpoint()

        self._model.load_state_dict(checkpoint.model_state_dict)
        self._optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        return checkpoint

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    def _load_best_score_checkpoint(self):
        checkpoint = self._score_checkpoints[0]
        return checkpoint

    def _load_latest_checkpoint(self):
        checkpoint = sorted(self._score_checkpoints, reverse=True, key=lambda x: x.epoch)[0]
        return checkpoint

    def _create_latest_checkpoint(self, score, epoch, **kwargs):
        file_name = self._to_latest_file_name()
        path = self._save_dir.joinpath(file_name)
        checkpoint = CheckPoint.generate(path, self._model, self._optimizer, score, epoch, **kwargs)
        return checkpoint

    def _create_score_checkpoint(self, score, epoch, **kwargs):
        file_name = self._to_score_file_name(score, epoch)
        path = self._save_dir.joinpath(file_name)
        checkpoint = CheckPoint.generate(path, self._model, self._optimizer, score, epoch, **kwargs)
        return checkpoint

    def _is_higher_score_checkpoint(self, score):
        if (len(self._score_checkpoints) < self._n_saves) or (self._score_checkpoints[-1].score < score):
            return True
        return False

    def _need_replace(self):
        return len(self._score_checkpoints) >= self._n_saves

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def _to_score_file_name(cls, score, epoch):
        return f"checkpoint_epoch={epoch}_score={score:.4f}{cls._SUFFIX}"

    @classmethod
    def _to_latest_file_name(cls):
        return f"checkpoint_latest{cls._SUFFIX}"

    @classmethod
    def _load_checkpoints(cls, save_dir, n_load_checkpoint):
        """

        Args:
            save_dir (Path):

        Returns:

        """
        if not save_dir.exists():
            raise FileNotFoundError(save_dir)

        latest_score = None
        score_checkpoints = []
        for file in save_dir.iterdir():
            if file.is_file() and file.suffix == cls._SUFFIX:
                if "latest" in file.stem:
                    latest_score = CheckPoint.load(file)
                else:
                    score_checkpoints.append(CheckPoint.load(file))

        if len(score_checkpoints) > 0:
            score_checkpoints = sorted(score_checkpoints, reverse=True, key=lambda x: x.score)

        if len(score_checkpoints) > n_load_checkpoint:
            score_checkpoints = score_checkpoints[:n_load_checkpoint]

        return latest_score, score_checkpoints
