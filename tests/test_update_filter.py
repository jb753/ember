"""Unit tests for the SFD low-pass filter update in :meth:`ember.grid.Grid.update_timestep`.

The filter update rides in the timestep kernel (``set_timestep_sources``),
gated on a nonzero ``gain_filt``, because that walk already holds the cell
averages it needs and is where the step's ``dt_vol`` is computed. These tests
pin the exponential-moving-average arithmetic against a numpy reference, and
that sharing the walk costs the timestep and the body force nothing: the gates
may not change ``dt_vol``, and a filter-only step may not touch ``F_body``.
"""

import numpy as np

import ember.block
from conftest import cell_conserved
import ember.grid
from ember import util
from ember.fluid import PerfectFluid

SHAPE = (7, 9, 9)

# Any nonzero gain switches the filter on; its value only scales the force,
# which none of these calls add (add_sources is left off).
GAIN = 1.0


def _build_block():
    """Small non-rotating block, populated dt_vol, filter seeded off the flow."""
    block = ember.block.Block(shape=SHAPE)
    block.set_Nb(36)
    xrt = util.linmesh3((0.0, 0.15), (0.5, 0.9), (0.0, 0.1), SHAPE)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
    block.set_P_T(101325.0, 300.0)
    x, r, t = block.x, block.r, block.t
    block.set_Vx((100.0 + 10.0 * (r - r.min())).astype(np.float32))
    block.set_Vr((5.0 * np.cos(t)).astype(np.float32))
    block.set_Vt((40.0 + 15.0 * np.sin(x)).astype(np.float32))
    ember.grid.Grid([block]).update_timestep(rf=1.0)

    # Seed the filter 2% off the flow, so one EMA step has somewhere to go: on
    # first access it seeds itself to the current state, where the update is
    # a no-op a broken kernel would pass.
    cons_filt = block.conserved_filt_nd
    cons_filt.flags.writeable = True
    cons_filt *= np.float32(1.02)
    cons_filt.flags.writeable = False
    return block


def _expected_ema(cons_filt, cons_cell, cfl, dt_vol, vol, delta):
    """Reference EMA: cons_filt += cfl*dt_vol*vol*(cons_cell - cons_filt)/delta."""
    dt = cfl * dt_vol[..., None] * vol[..., None]
    return cons_filt + dt * (cons_cell - cons_filt) / delta


def test_update_filter_matches_reference():
    """One step of the EMA, on the dt_vol the same call has just computed."""
    block = _build_block()
    delta = 2.0
    cfl = 0.4

    before = block.conserved_filt_nd.copy()
    ember.grid.Grid([block]).update_timestep(
        rf=1.0, gain_filt=GAIN, cfl=cfl, delta_filt=delta
    )
    expected = _expected_ema(
        before, cell_conserved(block), cfl, block.dt_vol_nd, block.vol_nd, delta
    )
    np.testing.assert_allclose(block.conserved_filt_nd, expected, rtol=1e-5)


def test_update_filter_relocks_buffer():
    """conserved_filt_nd is read-only to consumers after the update."""
    block = _build_block()
    ember.grid.Grid([block]).update_timestep(
        rf=1.0, gain_filt=GAIN, cfl=0.4, delta_filt=1.0
    )
    assert not block.conserved_filt_nd.flags.writeable


def test_update_filter_spans_all_blocks():
    """Every block in the grid is advanced, not just the first."""
    blocks = [_build_block(), _build_block()]
    before = [b.conserved_filt_nd.copy() for b in blocks]
    ember.grid.Grid(blocks).update_timestep(
        rf=1.0, gain_filt=GAIN, cfl=0.4, delta_filt=1.0
    )
    for block, filt in zip(blocks, before):
        want = _expected_ema(
            filt, cell_conserved(block), 0.4, block.dt_vol_nd, block.vol_nd, 1.0
        )
        np.testing.assert_allclose(block.conserved_filt_nd, want, rtol=1e-5)


def test_gates_do_not_change_the_timestep():
    """dt_vol is bitwise the same whichever of the source halves run."""
    dt_vols = []
    for add_sources, gain in [(False, 0.0), (True, 0.0), (False, GAIN), (True, GAIN)]:
        block = _build_block()
        grid = ember.grid.Grid([block])
        grid.update_sources(inviscid=True)
        grid.update_timestep(
            rf=0.2, add_sources=add_sources, gain_filt=gain, cfl=0.4, delta_filt=1.0
        )
        dt_vols.append(np.array(block.dt_vol_nd))
    for dt_vol in dt_vols[1:]:
        np.testing.assert_array_equal(dt_vol, dt_vols[0])


def test_filter_alone_leaves_the_body_force():
    """Off a source-refresh step the filter advances and F_body is held."""
    block = _build_block()
    grid = ember.grid.Grid([block])
    grid.update_sources(inviscid=True)
    grid.update_timestep(rf=1.0, add_sources=True, gain_filt=GAIN)
    held = np.array(block.F_body_nd)
    filt = np.array(block.conserved_filt_nd)

    grid.update_timestep(rf=1.0, gain_filt=GAIN, cfl=0.4, delta_filt=1.0)
    np.testing.assert_array_equal(block.F_body_nd, held)
    assert not np.array_equal(block.conserved_filt_nd, filt)
