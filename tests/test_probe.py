"""Tests for ProbePatch passive-overlay behaviour in core ember.

Covers the passive-overlay semantics of ``ember.patch.ProbePatch`` (a probe may
sit on an interior plane, may coincide with another patch, and does not change
the wall treatment of the face it samples). The TS3 file I/O for probes lives in
the ember-cfd-ts plugin and is tested there.
"""

import numpy as np
import pytest

from ember.block import Block
from ember.fluid import PerfectFluid
from ember.patch import InletPatch, ProbePatch
import ember.util_yaml as util_yaml


def _make_block(shape=(6, 5, 4), rpm=3600.0, Nb=24):
    """Single block with coordinates, fluid, and flow data set."""
    block = Block(shape=shape)
    xv, rv, tv = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0]),
        np.linspace(0.8, 1.2, shape[1]),
        np.linspace(0.0, np.pi / 8, shape[2]),
        indexing="ij",
    )
    block.set_x(xv)
    block.set_r(rv)
    block.set_t(tv)
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7))
    block.set_rpm(rpm)
    block.set_Nb(Nb)

    rho = np.ones(shape, dtype=np.float32) * 1.2
    conserved = np.stack(
        [rho, 150.0 * rho, 0.0 * rho, 100.0 * rho, np.ones(shape) * 2.5e5], axis=-1
    )
    block.set_conserved(conserved)
    block.set_wdist(np.ones(shape, dtype=np.float32) * 1e-3)
    return block


# --- ProbePatch behaviour (passive overlay) -------------------------------


def test_point_probe_attaches():
    block = _make_block()
    block.patches.append(ProbePatch(i=2, j=3, k=1, label="pt"))
    assert len(block.patches.probe) == 1
    assert block.patches.probe[0].size == 1


def test_interior_plane_probe_attaches():
    # An interior constant plane (i=2 on an ni=6 block) is rejected for an
    # ordinary patch but allowed for a ProbePatch.
    block = _make_block()
    block.patches.append(ProbePatch(i=2, label="plane"))
    assert len(block.patches.probe) == 1

    with pytest.raises(ValueError):
        block.patches.append(InletPatch(i=2, label="bad_inlet"))


def test_probe_may_coincide_with_other_patches():
    # A region probe may overlap another patch; two of the same real type
    # still raise.
    block = _make_block()
    block.patches.append(InletPatch(i=0, label="inlet"))
    block.patches.append(ProbePatch(i=0, label="probe_on_inlet"))
    assert len(block.patches.probe) == 1

    # Two coincident probes are also allowed (overlap-exempt).
    block.patches.append(ProbePatch(i=0, label="probe2"))
    assert len(block.patches.probe) == 2

    with pytest.raises(ValueError):
        block.patches.append(InletPatch(i=0, label="inlet2"))


def test_probe_does_not_change_wall_arrays():
    # A probe is not permeable, so it must not alter the face wall indicators.
    base = _make_block()
    iwall0, jwall0, kwall0 = base._get_face_wall_arrays()

    probed = _make_block()
    probed.patches.append(ProbePatch(i=0, label="probe"))
    iwall1, jwall1, kwall1 = probed._get_face_wall_arrays()

    assert np.array_equal(iwall0, iwall1)
    assert np.array_equal(jwall0, jwall1)
    assert np.array_equal(kwall0, kwall1)


# --- util_yaml round-trip -------------------------------------------------


def test_util_yaml_roundtrip(tmp_path):
    data = {
        0: {0: {"shape": [1, 2, 3], "Omega": np.float32(376.99), "Nb": np.int64(24)}}
    }
    fname = tmp_path / "meta.yaml"
    util_yaml.write_yaml(data, str(fname))
    loaded = util_yaml.read_yaml(str(fname))
    assert loaded[0][0]["shape"] == [1, 2, 3]
    assert loaded[0][0]["Nb"] == 24
    assert loaded[0][0]["Omega"] == pytest.approx(376.99, rel=1e-4)
