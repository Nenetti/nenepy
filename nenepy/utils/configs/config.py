import pprint
from collections import OrderedDict
from pathlib import Path

from nenepy.utils import yaml

from natsort import natsorted

from nenepy.utils.dictionary import AttrDict


class Config(AttrDict):

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def import_yaml(self, yaml_file_path):
        """

        Args:
            yaml_file_path (Path or str): yamlファイルのパス

        """
        with open(yaml_file_path, "r") as f:
            yaml_cfg = AttrDict(yaml.safe_load(f))
        self._recursive_merge(self, yaml_cfg)

    def merge_cfg_from_cfg(self, cfg_other):
        self._recursive_merge(self, cfg_other)

    def output_config(self, file_dir, file_name):
        """

        Args:
            file_dir (Path or str):
            file_name (Path or str):

        """
        # ----- Write and Output ----- #
        with open(Path(file_dir).joinpath(file_name), "w") as f:
            cfg_dict = AttrDict()
            cfg_dict.merge(self)
            out_dict = self._to_output_format(cfg_dict)

            yaml.dump(out_dict, f, default_flow_style=False)

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
