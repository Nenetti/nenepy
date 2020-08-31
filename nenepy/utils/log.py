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

    def __init__(self, log_dir, is_load):
        """

        Args:
            log_dir (Path or str):
            is_load (bool):

        """
        super(Log, self).__init__()

        log_dir = Path(log_dir)

        if log_dir.exists() and not is_load:
            shutil.rmtree(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)

    # ==============================================================================
    #
    #   Public Method
    #
    # ==============================================================================

    def load(self, file_name, file_dir=None):
        """

        Args:
            file_dir (Path or str):
            file_name (Path or str):

        """
        # ----- Initialize ----- #
        if file_dir is None:
            file_dir = self.log_dir

        # ----- Load ----- #
        with open(Path(file_dir).joinpath(file_name), "r") as f:
            self.merge(AttrDict(yaml.load(f)))

    def save(self, file_name, file_dir=None):
        """

        Args:
            file_dir (Path or str):
            file_name (Path or str):

        """
        # ----- Initialize ----- #
        if file_dir is None:
            file_dir = self.log_dir

        # ----- Write and Output ----- #
        with open(Path(file_dir).joinpath(file_name), "w") as f:
            out_dict = self._to_output_format(self)
            yaml.dump(out_dict, f, default_flow_style=False)
