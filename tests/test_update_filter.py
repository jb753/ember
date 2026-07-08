"""Unit tests for :meth:`ember.block.Block.update_filter`.

The SFD low-pass filter update was split out of the ``adapt_cfl`` kernel into a
dedicated ``Block.update_filter`` with two Fortran variants (array / scalar CFL).
These tests pin the exponential-moving-average arithmetic and confirm the rank
dispatch: a scalar CFL must match an array CFL filled with that scalar.
"""

import numpy as np

import ember.block
import ember.fortran
from ember import util
from ember.fluid import PerfectFluid

SHAPE = (7, 9, 9)


def _build_block():
    """Small non-rotating block with a smooth flow and populated dt_vol."""
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
    # Populate dt_vol_nd (mirrors Grid.update_timestep's per-block body).
    block.dt_vol_nd.flags.writeable = True
    ember.fortran.set_timestep_spectral(
        dt_vol=block.dt_vol_nd,
        a=block.a_nd,
        cons_cell=block.conserved_cell_nd,
        r=block.r_nd,
        omega=block.Omega_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        mu_turb=block._get_data_by_keys(("mu_turb",), raise_uninit=False),
        vol=block.vol_nd,
        rf=1.0,
        fac_visc=1.0,
    )
    block.dt_vol_nd.flags.writeable = False
    return block


def _expected_ema(cons_filt, cons_cell, cfl, dt_vol, vol, delta):
    """Reference EMA: cons_filt += cfl*dt_vol*vol*(cons_cell - cons_filt)/delta."""
    dt = cfl * dt_vol[..., None] * vol[..., None]
    return cons_filt + dt * (cons_cell - cons_filt) / delta


def test_update_filter_array_matches_reference():
    block = _build_block()
    delta = 1.5
    cfl = np.full(block.shape_cell + (5,), 0.4, dtype=np.float32)
    # Vary CFL per equation so a per-component bug would show.
    cfl *= np.array([1.0, 0.8, 0.9, 1.1, 1.2], dtype=np.float32)

    before = block.conserved_filt_nd.copy()
    expected = _expected_ema(
        before, block.conserved_cell_nd, cfl, block.dt_vol_nd, block.vol_nd, delta
    )
    block.update_filter(cfl, delta)
    np.testing.assert_allclose(block.conserved_filt_nd, expected, rtol=1e-5)


def test_update_filter_scalar_matches_reference():
    block = _build_block()
    delta = 2.0
    cfl = 0.4

    before = block.conserved_filt_nd.copy()
    expected = _expected_ema(
        before, block.conserved_cell_nd, cfl, block.dt_vol_nd, block.vol_nd, delta
    )
    block.update_filter(cfl, delta)
    np.testing.assert_allclose(block.conserved_filt_nd, expected, rtol=1e-5)


def test_scalar_equals_array_filled_with_scalar():
    cfl_scalar = 0.4
    delta = 1.0

    b_scalar = _build_block()
    b_scalar.update_filter(cfl_scalar, delta)

    b_array = _build_block()
    cfl_arr = np.full(b_array.shape_cell + (5,), cfl_scalar, dtype=np.float32)
    b_array.update_filter(cfl_arr, delta)

    np.testing.assert_allclose(
        b_scalar.conserved_filt_nd, b_array.conserved_filt_nd, rtol=1e-6
    )


def test_update_filter_relocks_buffer():
    """conserved_filt_nd is read-only to consumers after the update."""
    block = _build_block()
    block.update_filter(0.4, 1.0)
    assert not block.conserved_filt_nd.flags.writeable
