from pathlib import Path

import yaml

from nenepy.utils.dictionary import AttrDict


class Config(AttrDict):

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    def make_immutable(self):
        """
        Set Immutable attribute.

        """
        self.immutable()

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
    #   Private Method
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

    def __repr__(self):
        def recur(key, value, depth=0):
            if isinstance(value, dict):
                lines = [f"{'':>{depth * 4}}{key}:"]
                for k, v in value.items():
                    l = recur(k, v, depth + 1)
                    if isinstance(l, list):
                        lines.extend(l)
                    else:
                        lines.append(l)
                return lines
            else:
                return f"{'':>{depth * 4}}{key}: {value}"

        lines = []
        for key, value in self.items():
            l = recur(key, value)
            if isinstance(l, list):
                lines.extend(l)
            else:
                lines.append(l)

        return "\n".join(lines)
