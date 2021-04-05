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
        self._checkpoints = []
        self._n_saves = n_saves
        self._needs_load_checkpoints = needs_load_checkpoints

        if self._needs_load_checkpoints:
            self._checkpoints = self._load_checkpoints(self._save_dir, self._n_saves)
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
        if self._needs_generate_checkpoint(score):
            checkpoint = self._create_checkpoint(score, epoch, **kwargs)
            checkpoint.save()
            if self._need_replace():
                self._checkpoints[-1].delete()
                self._checkpoints[-1] = checkpoint
            else:
                self._checkpoints.append(checkpoint)

            self._checkpoints = sorted(self._checkpoints, reverse=True, key=lambda x: x.score)

    def load_checkpoint(self, mode):
        """

        Args:
            mode (CheckPointType):

        Returns:
            CheckPoint:

        """
        if mode == CheckPointType.LATEST:
            return self.load_latest_checkpoint()
        elif mode == CheckPointType.BEST_SCORE:
            return self.load_best_score_checkpoint()

    def load_best_score_checkpoint(self):
        checkpoint = self._checkpoints[0]
        return checkpoint

    def load_latest_checkpoint(self):
        checkpoint = sorted(self._checkpoints, reverse=True, key=lambda x: x.epoch)[0]
        return checkpoint

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    def _create_checkpoint(self, score, epoch, **kwargs):
        file_name = self._to_file_name(score, epoch)
        path = self._save_dir.joinpath(file_name)
        checkpoint = CheckPoint.generate(path, self._model, self._optimizer, score, epoch, **kwargs)
        return checkpoint

    def _needs_generate_checkpoint(self, score):
        if (len(self._checkpoints) < self._n_saves) or (score > self._checkpoints[-1].score):
            return True
        return False

    def _need_replace(self):
        return len(self._checkpoints) >= self._n_saves

    # ==================================================================================================
    #
    #   Class Method (Public)
    #
    # ==================================================================================================
    @classmethod
    def _to_file_name(cls, score, epoch):
        return f"checkpoint_epoch={epoch}_score={score:.4f}{cls._SUFFIX}"

    @classmethod
    def _load_checkpoints(cls, save_dir, n_load_checkpoint):
        """

        Args:
            save_dir (Path):

        Returns:

        """
        if not save_dir.exists():
            raise FileNotFoundError(save_dir)

        checkpoints = []
        for file in save_dir.iterdir():
            if file.is_file() and file.suffix == cls._SUFFIX:
                checkpoint = CheckPoint.load(file)
                checkpoints.append(checkpoint)

        if len(checkpoints) > 0:
            checkpoints = sorted(checkpoints, reverse=True, key=lambda x: x.score)

        if len(checkpoints) > n_load_checkpoint:
            checkpoints = checkpoints[:n_load_checkpoint]

        return checkpoints
