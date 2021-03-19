from collections import OrderedDict

import yaml
from yaml import *


def map_representer(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(dict, map_representer)
yaml.add_representer(OrderedDict, map_representer)

del map_representer
