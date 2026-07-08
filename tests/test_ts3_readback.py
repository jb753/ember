"""Tests for ``ember.ts3.read_conserved``: reading a TS3 flow field into an
existing grid, with correct thermodynamic-datum handling.

The key correctness property is the datum round-trip: a grid on a general datum
(e.g. ``T_dtm=300``) written out and read back must recover the *dimensional*
pressure, temperature and velocities, not merely the raw bytes. ``write_ts3``
always writes total energy on TS3's zero datum (u = cv * T), and
``read_conserved`` converts it back onto each block's own datum.
"""

import h5py
import numpy as np
import pytest

from ember.block import Block
from ember.fluid import PerfectFluid
from ember.grid import Grid
import ember.ts3 as ts3
import ember.util as util


def _make_grid(T_dtm=300.0, set_mu_turb=False):
    """Two-block grid on the given datum with a non-trivial flow field set.

    If ``set_mu_turb`` is True, a varying turbulent viscosity is also set, so
    ``write_ts3`` emits ``trans_dyn_vis_bp`` (omitted otherwise).
    """
    shapes = [(3, 4, 5), (4, 3, 6)]
    spans = [
        ([0.0, 1.0], [0.5, 1.5], [0.0, 0.2]),
        ([1.0, 2.0], [1.5, 2.5], [0.1, 0.3]),
    ]
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7, T_dtm=T_dtm)

    blocks = []
    for shape, (xs, rs, ts_) in zip(shapes, spans):
        block = Block(shape=shape)
        xrt = util.linmesh3(xs, rs, ts_, shape)
        block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_rpm(0.0).set_Nb(1)

        # A genuinely non-trivial state: varying P/T and all three velocities.
        ramp = np.linspace(0.0, 1.0, shape[0]).reshape(-1, 1, 1)
        P = 1.0e5 + 2.0e4 * ramp * np.ones(shape)
        T = 300.0 + 50.0 * ramp * np.ones(shape)
        block.set_P_T(P, T)
        block.set_Vxrt(
            np.stack(
                [
                    120.0 * np.ones(shape),
                    10.0 * np.ones(shape),
                    -30.0 * np.ones(shape),
                ],
                axis=-1,
            )
        )
        block.set_wdist(np.ones(shape, dtype=np.float32) * 0.01)
        if set_mu_turb:
            block.set_mu_turb(1e-3 + 5e-4 * ramp * np.ones(shape))
        blocks.append(block)

    return Grid(blocks)


def test_read_conserved_datum_roundtrip(tmp_path):
    """P, T and velocities survive write_ts3 -> read_conserved."""
    grid = _make_grid(T_dtm=300.0)

    # Record the dimensional state we expect to recover.
    expected = [
        (b.P.copy(), b.T.copy(), b.Vx.copy(), b.Vr.copy(), b.Vt.copy()) for b in grid
    ]

    fname = str(tmp_path / "output_avg.hdf5")
    grid.write_ts3(fname)

    # A fresh grid with identical geometry on the SAME (300 K) datum, blank flow.
    target = _make_grid(T_dtm=300.0)
    for b in target:
        b.set_P_T(np.full(b.shape, 1e5), np.full(b.shape, 300.0))
        b.set_Vxrt(np.zeros((*b.shape, 3)))

    ts3.read_conserved(target, fname)

    for i, (b, (P, T, Vx, Vr, Vt)) in enumerate(zip(target, expected)):
        np.testing.assert_allclose(b.P, P, rtol=1e-4, err_msg=f"block {i} P")
        np.testing.assert_allclose(b.T, T, rtol=1e-4, err_msg=f"block {i} T")
        np.testing.assert_allclose(b.Vx, Vx, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(b.Vr, Vr, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(b.Vt, Vt, rtol=1e-4, atol=1e-3)


def test_write_ts3_does_not_mutate_grid(tmp_path):
    """write_ts3 must leave the grid's datum and state intact."""
    grid = _make_grid(T_dtm=300.0)
    P_before = [b.P.copy() for b in grid]
    T_before = [b.T.copy() for b in grid]
    dtm_before = [b.fluid.T_dtm for b in grid]

    grid.write_ts3(str(tmp_path / "out.hdf5"))

    for i, b in enumerate(grid):
        assert b.fluid.T_dtm == dtm_before[i], f"block {i} datum mutated"
        np.testing.assert_array_equal(b.P, P_before[i])
        np.testing.assert_array_equal(b.T, T_before[i])


def test_read_conserved_returns_grid(tmp_path):
    """read_conserved returns the same grid object for chaining."""
    grid = _make_grid()
    fname = str(tmp_path / "out.hdf5")
    grid.write_ts3(fname)
    assert ts3.read_conserved(grid, fname) is grid


def test_read_conserved_block_count_mismatch(tmp_path):
    """A grid with the wrong number of blocks is rejected."""
    grid = _make_grid()
    fname = str(tmp_path / "out.hdf5")
    grid.write_ts3(fname)

    one_block = Grid([grid[0]])
    with pytest.raises(ValueError, match="blocks but file"):
        ts3.read_conserved(one_block, fname)


def test_read_conserved_diverged_negative_density(tmp_path):
    """A file with negative density raises (diverged solution)."""
    grid = _make_grid()
    fname = str(tmp_path / "out.hdf5")
    grid.write_ts3(fname)

    # Corrupt block0 density to negative in place.
    with h5py.File(fname, "r+") as f:
        ro = f["block0"]["ro_bp"]
        ro[...] = -np.abs(ro[...]) - 1.0

    with pytest.raises(ValueError):
        ts3.read_conserved(_make_grid(), fname)


def test_read_mu_turb_roundtrip(tmp_path):
    """Turbulent viscosity survives write -> read_mu_turb into an existing grid."""
    grid = _make_grid(set_mu_turb=True)
    expected = [b.mu_turb.copy() for b in grid]

    fname = str(tmp_path / "output.hdf5")
    grid.write_ts3(fname)

    target = _make_grid()  # geometry only, mu_turb left at its NaN default
    ret = ts3.read_mu_turb(target, fname)
    assert ret is target
    for i, (b, mu) in enumerate(zip(target, expected)):
        np.testing.assert_allclose(b.mu_turb, mu, rtol=1e-5, err_msg=f"block {i}")


def test_read_mu_turb_missing_dataset(tmp_path):
    """A file without trans_dyn_vis_bp raises KeyError."""
    grid = _make_grid()  # no mu_turb set -> writer omits trans_dyn_vis_bp
    fname = str(tmp_path / "no_visc.hdf5")
    grid.write_ts3(fname)

    with pytest.raises(KeyError):
        ts3.read_mu_turb(_make_grid(), fname)


def test_read_mu_turb_block_count_mismatch(tmp_path):
    """A grid with the wrong number of blocks is rejected."""
    grid = _make_grid(set_mu_turb=True)
    fname = str(tmp_path / "out.hdf5")
    grid.write_ts3(fname)

    with pytest.raises(ValueError, match="blocks but file"):
        ts3.read_mu_turb(Grid([grid[0]]), fname)
