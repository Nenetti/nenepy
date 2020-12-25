import shutil
from pathlib import Path

import yaml

from nenepy.utils.dictionary import AttrDict


class Logger(AttrDict):
    """

    Examples:
        >>> log = Logger(log_dir="./logs", is_load=False)
        >>> log.A = 1
        >>> log["B"] = 2
        >>> log.save()

    """

    def __init__(self, log_dir, log_file="log.yaml", is_load=True):
        """

        Args:
            log_dir (Path or str):
            log_file (Path or str):
            is_load (bool):

        """
        super(Logger, self).__init__()

        self._log_dir = log_dir
        self._path = Path(log_dir).joinpath(log_file)

        if is_load:
            if self._path.exists():
                self.load()
            else:
                raise FileNotFoundError(self._path)
        else:
            if log_dir.exists():
                shutil.rmtree(log_dir)

            log_dir.mkdir(parents=True)

    @property
    def log_dir(self):
        return self._log_dir

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================
    def load(self):
        if not self._path.exists():
            raise FileNotFoundError()

        with open(self._path, "r") as f:
            self.merge(AttrDict(yaml.load(f)))

    def save(self):
        with open(self._path, "w") as f:
            out_dict = self._to_output_format(self)
            yaml.dump(out_dict, f, default_flow_style=False)
