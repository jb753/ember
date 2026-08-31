"""Read and write YAML files with numpy and :class:`~pathlib.Path` support.

This module is a convenience for plugins, not core code -- nothing
in :mod:`ember` itself depends on it. A plugin that wants to load or save
its own configuration as YAML can use :func:`read_yaml` and
:func:`write_yaml` in place of ``yaml.safe_load`` and ``yaml.safe_dump``.

The following fixes are applied:

- Dump numpy arrays and scalars as plain YAML lists and scalars
- :class:`~pathlib.Path` as an expanded string
- Parse scientific notation without decimal points e.g. ``1e-5``
- Use PyYAML's ``CSafeLoader``/``CSafeDumper`` for speed where
  available, falling back to the pure-Python versions otherwise
- Always read and write files as UTF-8 encoded

"""

import re
import yaml
import numpy as np
from pathlib import Path, PurePath


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

    Registered as a *multi* representer for :class:`pathlib.PurePath`, which
    matches along the mro and so covers every concrete path class at once.
    Registering :class:`~pathlib.Path` and :class:`~pathlib.PosixPath` by
    name did not: PyYAML dispatches on the exact runtime type, ``Path(...)``
    instantiates ``PosixPath`` on this platform and ``WindowsPath`` on
    Windows, and the bare ``Path`` registration therefore never fires at all.

    The path is expanded with :meth:`~pathlib.Path.expanduser` before
    representing, so a leading ``~`` is resolved to the user's home directory
    in the dumped string. Pure paths have no ``expanduser``, and none is
    wanted: they name a path without a filesystem to resolve it against.

    Parameters
    ----------
    dumper : Dumper
        Dumper instance requesting the representation.
    data : PurePath
        Path to represent.

    Returns
    -------
    ScalarNode
        YAML string scalar node.
    """
    if isinstance(data, Path):
        data = data.expanduser()
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.representer.SafeRepresenter.add_representer(np.float64, _represent_float)
yaml.representer.SafeRepresenter.add_representer(np.float32, _represent_float)
yaml.representer.SafeRepresenter.add_representer(np.int64, _represent_int)
yaml.representer.SafeRepresenter.add_representer(np.int32, _represent_int)
yaml.representer.SafeRepresenter.add_representer(np.ndarray, _represent_ndarray)
yaml.representer.SafeRepresenter.add_multi_representer(PurePath, _represent_path)


#: Loader/dumper classes used by :func:`read_yaml`/:func:`write_yaml`. The
#: libyaml-backed ``CSafeLoader``/``CSafeDumper`` are used when available,
#: since their C scanner/parser/emitter is faster than the pure-Python
#: ``SafeLoader``/``SafeDumper``. ``CSafeDumper`` already inherits
#: ``SafeRepresenter``, so the representers registered above apply to it
#: unchanged; ``CSafeLoader`` does *not* inherit ``SafeLoader``, so its
#: implicit float resolver is patched separately by :func:`_float_loader`.
_Loader = yaml.CSafeLoader if yaml.__with_libyaml__ else yaml.SafeLoader
_Dumper = yaml.CSafeDumper if yaml.__with_libyaml__ else yaml.SafeDumper


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
    """Build a loader that parses scientific-notation floats.

    Patches :data:`_Loader`'s implicit resolver for the
    ``tag:yaml.org,2002:float`` tag with :data:`_FLOAT_PATTERN`, so values
    such as ``1e-5`` load as :class:`float` rather than :class:`str`. The
    patch is applied to the :data:`_Loader` class itself, so it persists
    for the lifetime of the process once called.

    Note that :data:`_Loader` may be ``yaml.CSafeLoader``, which inherits
    ``yaml.resolver.Resolver`` directly rather than ``yaml.SafeLoader`` --
    patching ``yaml.SafeLoader`` would not affect it, so this must patch
    :data:`_Loader` itself.

    Returns
    -------
    type
        :data:`_Loader`, with the corrected float resolver installed,
        suitable for passing as the ``Loader`` argument to ``yaml.load``.
    """
    _Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(_FLOAT_PATTERN, re.X),
        list("-+0123456789."),
    )
    return _Loader


def read_yaml(fname):
    """Read a dictionary from a YAML file.

    Parameters
    ----------
    fname : str or Path
        Path to the YAML file to read.

    Returns
    -------
    dict
        Parsed contents of the file.
    """
    # Explicit encoding, because Python's default is the locale's: a file with
    # a degree sign in it reads differently on a machine that is not UTF-8, and
    # YAML is UTF-8 by specification anyway.
    with open(fname, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_float_loader())


def write_yaml(d, fname, mode="w"):
    """Write a dictionary to a YAML file.

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
    with open(fname, mode, encoding="utf-8") as f:
        yaml.dump(d, f, Dumper=_Dumper, explicit_start=True, explicit_end=True)
