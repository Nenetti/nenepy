import numbers
import shutil
from pathlib import Path

import yaml

from nenepy.utils.dictionary import AttrDict


class Log(AttrDict):
    """

    Examples:
        >>> log = Log(log_dir="./logs", is_load=False)
        >>> log.A = 1
        >>> log["B"] = 2
        >>> log.save()

    """

    def __init__(self, log_dir, log_file="log.yaml", is_load=True):
        """

        Args:
            log_dir (Path or str):
            is_load (bool):

        """
        super(Log, self).__init__()

        # if log_dir.exists() and not is_load:
        #     shutil.rmtree(log_dir)

        self._log_dir = Path(log_dir)
        self._file_name = log_file

        if is_load:
            if self._log_dir.exists():
                self.load()
            else:
                raise FileNotFoundError(self._log_dir)

        else:
            if not self._log_dir.exists():
                self._log_dir.mkdir(parents=True)

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def file_name(self):
        return self._file_name

    # ==============================================================================
    #
    #   Public Method
    #
    # ==============================================================================

    def load(self, file_dir=None, file_name=None):
        """

        Args:
            file_dir (Path or str):
            file_name (Path or str):

        """
        # ----- Initialize ----- #
        if file_dir is None:
            file_dir = self._log_dir

        if file_name is None:
            file_name = self._file_name

        # ----- Load ----- #
        with open(Path(file_dir).joinpath(file_name), "r") as f:
            self.merge(AttrDict(yaml.load(f)))

    def save(self, file_dir=None, file_name=None):
        """

        Args:
            file_dir (Path or str):
            file_name (Path or str):

        """
        # ----- Initialize ----- #
        if file_dir is None:
            file_dir = self._log_dir

        if file_name is None:
            file_name = self._file_name

        # ----- Write and Output ----- #
        with open(Path(file_dir).joinpath(file_name), "w") as f:
            out_dict = self._to_output_format(self)
            yaml.dump(out_dict, f, default_flow_style=False)
