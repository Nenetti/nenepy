import shutil
from importlib import import_module
from pathlib import Path

from nenepy.utils.dictionary import AttrDict


class Config(AttrDict):

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def output_config(self, file_dir):
        """

        Args:
            file_dir (Path or str):
            file_name (Path or str):

        """
        # ----- Write and Output ----- #
        file_dir = Path(file_dir)
        if not file_dir.exists():
            file_dir.mkdir(parents=True)

        shutil.copy(self._PY_FILE, file_dir)

    @staticmethod
    def load_py(py_file):
        if Path(py_file).suffix != ".py":
            raise ValueError(f"{py_file} is not '.py'")

        module = import_module(py_file[:-3].replace("/", "."))
        configs = []
        for key, value in module.__dict__.items():
            if isinstance(value, Config):
                configs.append((key, value))

        if len(configs) != 1:
            raise ValueError(f"{py_file} contains some Config class. {[c[0] for c in configs]}")

        cfg = configs[0][1]

        cfg._PY_FILE = py_file

        cfg.to_immutable()
        return cfg

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================
    @classmethod
    def _recursive_merge(cls, main_cfg, other_cfg, hierarchy=""):
        """

        Args:
            main_cfg (AttrDict):
            other_cfg (AttrDict):
            hierarchy (str or None):

        """
        for key, value in other_cfg.items():
            if key not in main_cfg:
                full_key = f"{hierarchy}{key}"
                raise KeyError("Non-existent config key: {}".format(full_key))

            if isinstance(value, AttrDict):
                cls._recursive_merge(main_cfg[key], value, hierarchy=".".join([hierarchy, key]))
            else:
                main_cfg[key] = value

    # ==================================================================================================
    #
    #   Special Method
    #
    # ==================================================================================================
    def __repr__(self):
        def recur(key, value, depth=0):
            if key.startswith("_"):
                return None

            if isinstance(value, dict):
                ls = [f"{'':>{depth * 4}}{key}:"]
                for k, v in value.items():
                    l = recur(k, v, depth + 1)
                    if l is None:
                        continue
                    if isinstance(l, list):
                        ls.extend(l)
                    else:
                        ls.append(l)
                return ls
            else:
                return f"{'':>{depth * 4}}{key}: {value}"

        lines = []
        for key, value in self.items():
            line = recur(key, value)
            if line is None:
                continue

            if isinstance(line, list):
                lines.extend(line)
            elif line is not None:
                lines.append(line)

        return "\n".join(lines)
