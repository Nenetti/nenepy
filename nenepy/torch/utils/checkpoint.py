import shutil
from pathlib import Path

import torch

from nenepy.utils.dictionary import AttrDict


class CheckPoint:

    def __init__(self, root_dir, model, optimizer, n_saves=5):
        self._root_dir = Path(root_dir)
        self._model = model
        self._optimizer = optimizer
        self._checkpoints = []
        self._n_saves = n_saves

    def clean(self):
        if self._root_dir.exists():
            shutil.rmtree(self._root_dir)

        self._root_dir.mkdir()

    def add_checkpoint(self, epoch, score):
        if self._is_high_score(score):
            checkpoint = self._create_checkpoint(epoch, score)
            if self._need_replace():
                self._checkpoints[-1] = checkpoint
            else:
                self._checkpoints.append(checkpoint)

            self._save_checkpoint(checkpoint)
            self._checkpoints = sorted(self._checkpoints, reverse=True, key=lambda x: x.score)

    def _create_checkpoint(self, epoch, score):
        checkpoint = AttrDict({
            "score": score,
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._model.state_dict(),
        })
        return checkpoint

    def _save_checkpoint(self, checkpoint):
        file = self._root_dir.joinpath(f"checkpoint_{checkpoint.epoch}epoch_{checkpoint.score}score.pt")
        torch.save(checkpoint, file)

    def _load_checkpoint(self, checkpoint):
        file = self._root_dir.joinpath(f"checkpoint_{checkpoint.epoch}epoch_{checkpoint.score}score.pt")
        torch.save(checkpoint, file)

    def _is_high_score(self, score):
        if len(self._checkpoints) > 0:
            s = self._checkpoints[-1].score
            if s > score:
                return False

        return True

    def _need_replace(self):
        return len(self._checkpoints) == self._n_saves
