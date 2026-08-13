"""Tests for ember.yaml_util.

Covers the numpy/Path representers, the scientific-notation float resolver
patch, and that the module actually selects the libyaml-backed
``CSafeLoader``/``CSafeDumper`` when available.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

import ember.yaml_util as util_yaml


def test_roundtrip_numpy_types(tmp_path):
    data = {
        0: {0: {"shape": [1, 2, 3], "Omega": np.float32(376.99), "Nb": np.int64(24)}}
    }
    fname = tmp_path / "meta.yaml"
    util_yaml.write_yaml(data, str(fname))
    loaded = util_yaml.read_yaml(str(fname))
    assert loaded[0][0]["shape"] == [1, 2, 3]
    assert loaded[0][0]["Nb"] == 24
    assert loaded[0][0]["Omega"] == pytest.approx(376.99, rel=1e-4)


def test_roundtrip_path(tmp_path):
    data = {"workdir": Path("/some/directory")}
    fname = tmp_path / "path.yaml"
    util_yaml.write_yaml(data, str(fname))
    loaded = util_yaml.read_yaml(str(fname))
    assert loaded["workdir"] == str(Path("/some/directory").expanduser())


def test_read_yaml_parses_bare_scientific_notation(tmp_path):
    """1e-5 has no decimal point, which the stock resolver mishandles."""
    fname = tmp_path / "sci.yaml"
    fname.write_text("a: 1e-5\n")
    loaded = util_yaml.read_yaml(str(fname))
    assert isinstance(loaded["a"], float)
    assert loaded["a"] == pytest.approx(1e-5)


def test_write_then_read_preserves_bare_scientific_notation(tmp_path):
    fname = tmp_path / "sci_roundtrip.yaml"
    util_yaml.write_yaml({"a": 1e-5}, str(fname))
    loaded = util_yaml.read_yaml(str(fname))
    assert isinstance(loaded["a"], float)
    assert loaded["a"] == pytest.approx(1e-5)


@pytest.mark.skipif(
    not yaml.__with_libyaml__, reason="libyaml not available in this environment"
)
def test_uses_libyaml_backed_classes():
    assert util_yaml._Loader is yaml.CSafeLoader
    assert util_yaml._Dumper is yaml.CSafeDumper
