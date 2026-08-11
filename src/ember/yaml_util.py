"""Read and write YAML files with numpy and :class:`~pathlib.Path` support.

This module is a convenience for **plugins**, not core code -- nothing
in :mod:`ember` itself depends on it. A plugin that wants to load or save
its own configuration as YAML can use :func:`read_yaml` and
:func:`write_yaml` in place of ``yaml.safe_load`` and ``yaml.safe_dump`` to
avoid two footguns in plain `PyYAML <https://pyyaml.org/>`_:

- ``yaml.SafeDumper`` raises ``yaml.representer.RepresenterError`` on
  ``numpy.float64``, ``numpy.int64``, :class:`numpy.ndarray` and
  :class:`~pathlib.Path`, all of which turn up routinely in dicts built
  from computed quantities. This module registers representers so they dump
  as plain YAML scalars and lists instead.
- ``yaml.SafeLoader`` fails to recognise scientific-notation floats
  that have no decimal point, such as ``1e-5``, loading them back as
  strings instead of floats. This module patches the loader's implicit
  float resolver to catch that case too.

Ported from turbigen's ``yaml_utils`` module.
"""

import re
import yaml
import numpy as np
from pathlib import Path, PosixPath


def _represent_float(dumper, data):
    """Represent a numpy float as a YAML float scalar.

    Registered as a representer for ``numpy.float64`` and ``numpy.float32``
    so that ``yaml.safe_dump`` does not raise on them.

    Parameters
    ----------
    dumper : Dumper
        Dumper instance requesting the representation.
    data : numpy scalar (float64 or float32)
        Value to represent.

    Returns
    -------
    ScalarNode
        YAML float scalar node.
    """
    return dumper.represent_scalar("tag:yaml.org,2002:float", str(data))


def _represent_int(dumper, data):
    """Represent a numpy int as a YAML int scalar.

    Registered as a representer for ``numpy.int64`` and ``numpy.int32`` so
    that ``yaml.safe_dump`` does not raise on them.

    Parameters
    ----------
    dumper : Dumper
        Dumper instance requesting the representation.
    data : numpy scalar (int64 or int32)
        Value to represent.

    Returns
    -------
    ScalarNode
        YAML int scalar node.
    """
    return dumper.represent_scalar("tag:yaml.org,2002:int", str(data))


def _represent_ndarray(dumper, data):
    """Represent a numpy array as a YAML list.

    Registered as a representer for :class:`numpy.ndarray` so that
    ``yaml.safe_dump`` does not raise on it. The array is converted with
    :meth:`numpy.ndarray.tolist` before representing, so it round-trips
    through YAML as a plain (possibly nested) list, not an array.

    Parameters
    ----------
    dumper : Dumper
        Dumper instance requesting the representation.
    data : numpy.ndarray
        Array to represent.

    Returns
    -------
    SequenceNode
        YAML sequence node.
    """
    return dumper.represent_list(data.tolist())


def _represent_path(dumper, data):
    """Represent a path object as a YAML string scalar.

    Registered as a representer for :class:`pathlib.Path` and
    :class:`pathlib.PosixPath`. The path is expanded with
    :meth:`~pathlib.Path.expanduser` before representing, so a leading
    ``~`` is resolved to the user's home directory in the dumped string.

    Parameters
    ----------
    dumper : Dumper
        Dumper instance requesting the representation.
    data : Path or PosixPath
        Path to represent.

    Returns
    -------
    ScalarNode
        YAML string scalar node.
    """
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data.expanduser()))


yaml.representer.SafeRepresenter.add_representer(np.float64, _represent_float)
yaml.representer.SafeRepresenter.add_representer(np.float32, _represent_float)
yaml.representer.SafeRepresenter.add_representer(np.int64, _represent_int)
yaml.representer.SafeRepresenter.add_representer(np.int32, _represent_int)
yaml.representer.SafeRepresenter.add_representer(np.ndarray, _represent_ndarray)
yaml.representer.SafeRepresenter.add_representer(Path, _represent_path)
yaml.representer.SafeRepresenter.add_representer(PosixPath, _represent_path)


#: Regex matching floats that ``yaml.SafeLoader``'s default implicit
#: resolver misses, in particular scientific notation with no decimal point
#: (``1e-5``). Passed to ``yaml.BaseResolver.add_implicit_resolver`` by
#: :func:`_float_loader`.
_FLOAT_PATTERN = """^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$"""


def _float_loader():
    """Build a ``yaml.SafeLoader`` that parses scientific-notation floats.

    Patches ``yaml.SafeLoader``'s implicit resolver for the
    ``tag:yaml.org,2002:float`` tag with :data:`_FLOAT_PATTERN`, so values
    such as ``1e-5`` load as :class:`float` rather than :class:`str`. The
    patch is applied to the ``yaml.SafeLoader`` class itself, so it
    persists for the lifetime of the process once called.

    Returns
    -------
    type
        ``yaml.SafeLoader``, with the corrected float resolver installed,
        suitable for passing as the ``Loader`` argument to ``yaml.load``.
    """
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(_FLOAT_PATTERN, re.X),
        list("-+0123456789."),
    )
    return loader


def read_yaml(fname):
    """Read a dictionary from a YAML file.

    Uses :func:`_float_loader` in place of the stock ``yaml.SafeLoader``,
    so that scientific-notation floats without a decimal point parse
    correctly; see the module docstring.

    Parameters
    ----------
    fname : str or Path
        Path to the YAML file to read.

    Returns
    -------
    dict
        Parsed contents of the file.
    """
    with open(fname, "r") as f:
        return yaml.load(f, Loader=_float_loader())


def write_yaml(d, fname, mode="w"):
    """Write a dictionary to a YAML file.

    Uses ``yaml.safe_dump`` with the representers registered by this
    module for numpy scalars, numpy arrays and :class:`~pathlib.Path`
    objects; see the module docstring. The output is bracketed with
    ``---``/``...`` document markers.

    Parameters
    ----------
    d : dict
        Dictionary to write.
    fname : str or Path
        Path to the YAML file to write.
    mode : str, optional
        Mode to open `fname` with. Defaults to ``"w"``; pass ``"a"`` to
        append a further document to an existing file.
    """
    with open(fname, mode) as f:
        yaml.safe_dump(d, f, explicit_start=True, explicit_end=True)
