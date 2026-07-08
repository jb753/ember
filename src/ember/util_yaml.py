"""Functions for reading and writing YAML files.

Ported from turbigen's ``yaml_utils`` module. Provides ``read_yaml`` and
``write_yaml`` with numpy/``Path`` scalar support and correct parsing of
scientific-notation floats.
"""

import re
import yaml
import numpy as np
from pathlib import Path, PosixPath


# Allow dumping of numpy float to yaml
def represent_float(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:float", str(data))


# Allow dumping numpy int to yaml
def represent_int(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:int", str(data))


# Allow dumping np.ndarray as a list to yaml
def represent_ndarray(dumper, data):
    return dumper.represent_list(data.tolist())


# Dump path objects as strings
def represent_path(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data.expanduser()))


yaml.representer.SafeRepresenter.add_representer(np.float64, represent_float)
yaml.representer.SafeRepresenter.add_representer(np.float32, represent_float)
yaml.representer.SafeRepresenter.add_representer(np.int64, represent_int)
yaml.representer.SafeRepresenter.add_representer(np.int32, represent_int)
yaml.representer.SafeRepresenter.add_representer(np.ndarray, represent_ndarray)
yaml.representer.SafeRepresenter.add_representer(Path, represent_path)
yaml.representer.SafeRepresenter.add_representer(PosixPath, represent_path)


# Match scientific-notation floats that the default SafeLoader resolver misses
FLOAT_PATTERN = """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$"""


def _float_loader():
    """SafeLoader patched to parse scientific notation as floats."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(FLOAT_PATTERN, re.X),
        list("-+0123456789."),
    )
    return loader


def read_yaml(fname):
    """Read a dictionary from a YAML file."""
    with open(fname, "r") as f:
        return yaml.load(f, Loader=_float_loader())


def write_yaml(d, fname, mode="w"):
    """Write a dictionary to a YAML file."""
    with open(fname, mode) as f:
        yaml.safe_dump(d, f, explicit_start=True, explicit_end=True)
