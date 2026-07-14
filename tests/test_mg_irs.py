"""Tests for coarse-level implicit residual smoothing (IRS) in the RK multigrid
kernel ``advance_rk_stage_mg_fused`` (shared engine ``mg_hier2_core``).

Correctness contract:

- ``sf_irs == 0`` makes the coarse smoothing an exact no-op
  (``smooth_residual_tri``'s own guard).
- ``sf_irs > 0`` must actually change the result relative to ``sf_irs == 0``
  (proves the branch is wired in) and damp a high-frequency (checkerboard)
  component of the coarse correction, mirroring
  ``tests/test_residual_smoothing.py``.
"""

import numpy as np
import pytest

import ember.grid  # noqa: F401  binds ember.fortran
import ember
import ember.solver
from ember.block import Block
from ember.fluid import PerfectFluid

NP = 5


def _make_inputs(ni, nj, nk, seed):
    """Build Fortran-ordered inputs for one advance_rk_stage_mg_fused call."""
    rng = np.random.default_rng(seed)
    residual = np.asfortranarray(
        rng.standard_normal((ni - 1, nj - 1, nk - 1, NP)).astype(np.float32)
    )
    dt_vol = np.asfortranarray(
        (0.5 + rng.random((ni - 1, nj - 1, nk - 1))).astype(np.float32)
    )
    vol = np.asfortranarray(
        (0.5 + rng.random((ni - 1, nj - 1, nk - 1))).astype(np.float32)
    )
    snapshot = np.asfortranarray(
        rng.standard_normal((ni, nj, nk, NP)).astype(np.float32)
    )
    return residual, dt_vol, vol, snapshot


def _run(residual, dt_vol, vol, snapshot, ni, nj, nk, n_levels, sf_irs=0.0,
         kernel=None):
    """Call the RK MG kernel with freshly zeroed scratch (the size args are
    inferred by f2py from the array shapes, exactly as ember.solver does).

    ``kernel`` selects the entry point (default ``advance_rk_stage_mg_fused``,
    the IRS variant); pass ``advance_rk_stage_mg_fused_noirs`` for the plain
    variant. Both take the identical signature."""
    if kernel is None:
        kernel = ember.fortran.advance_rk_stage_mg_fused
    cons = np.asfortranarray(snapshot.copy())
    nc1i, nc1j, nc1k = (ni - 1) // 2, (nj - 1) // 2, (nk - 1) // 2
    n_corr, n_res, n_tri = ember.solver._mg_hier2_scratch_sizes(ni, nj, nk, n_levels)
    acc_sz = nc1i * nc1j * nc1k * NP

    def Z(*shape):
        return np.asfortranarray(np.zeros(shape, dtype=np.float32))

    kernel(
        cons=cons,
        snapshot=snapshot,
        residual=residual,
        dt_vol=dt_vol,
        vol=vol,
        alpha=1.0,
        cfl=0.4,
        fmgrid=0.2,
        sf_irs=sf_irs,
        n_levels=n_levels,
        tmp=Z(ni - 1, nj - 1, nk - 1, NP),
        dtblk=Z(nc1i, nc1j, nc1k),
        aplane=Z(ni - 1, nc1j),
        bb=Z(ni - 1, nj - 1, nc1k, NP),
        rawbuf=Z(nc1i, nc1j, nc1k, NP),
        sdt=Z(nc1i, nc1j, nc1k),
        sv=Z(nc1i, nc1j, nc1k),
        corr_all=Z(n_corr),
        acc0=Z(acc_sz),
        acc1=Z(acc_sz),
        cres=Z(n_res),
        triw=Z(n_tri),
    )
    return cons


# Shape divisible by 2**3, small enough to be fast.
NI, NJ, NK = 33, 33, 33
N_LEVELS = 3


def test_sf_irs_positive_changes_result():
    """A positive sf_irs must actually alter the coarse correction vs sf_irs=0."""
    residual, dt_vol, vol, snapshot = _make_inputs(NI, NJ, NK, seed=1)

    cons_plain = _run(residual, dt_vol, vol, snapshot, NI, NJ, NK, N_LEVELS)
    cons_irs = _run(
        residual, dt_vol, vol, snapshot, NI, NJ, NK, N_LEVELS, sf_irs=0.5
    )
    assert not np.array_equal(cons_plain, cons_irs)


def test_noirs_kernel_matches_fused_at_sf_zero():
    """advance_rk_stage_mg_fused_noirs must reproduce advance_rk_stage_mg_fused
    with sf_irs=0 byte-for-byte. The two share mg_hier2_core and differ only in
    the smoother passed, so the plain kernel is the exact non-IRS engine that
    solver.advance_rk_stage_mg dispatches to when sf_irs==0."""
    residual, dt_vol, vol, snapshot = _make_inputs(NI, NJ, NK, seed=5)

    cons_irs = _run(residual, dt_vol, vol, snapshot, NI, NJ, NK, N_LEVELS)
    cons_no = _run(
        residual, dt_vol, vol, snapshot, NI, NJ, NK, N_LEVELS,
        kernel=ember.fortran.advance_rk_stage_mg_fused_noirs,
    )
    np.testing.assert_array_equal(cons_irs, cons_no)


def test_sf_irs_damps_checkerboard_coarse_correction():
    """A checkerboard residual's coarse contribution shrinks under IRS.

    The checkerboard must alternate at the scale of a *coarse* block (b=2, so
    period 2 in block index), not per fine cell: a per-cell checkerboard sums
    to exactly zero inside every aligned 2x2x2 restriction block, leaving
    nothing for IRS to damp. Isolate the coarse correction (n_levels=1, no
    fine term) by differencing cons from the snapshot, and compare its
    magnitude with and without IRS.
    """
    ni, nj, nk = 17, 17, 17  # divisible by 2**1
    n_levels = 1
    b = 2
    rng = np.random.default_rng(2)
    ii, jj, kk = np.indices((ni - 1, nj - 1, nk - 1))
    sign = (((ii // b) + (jj // b) + (kk // b)) % 2) * 2 - 1
    residual = np.asfortranarray(
        (sign[..., None] * np.ones((ni - 1, nj - 1, nk - 1, NP))).astype(np.float32)
    )
    dt_vol = np.asfortranarray(
        (0.5 + rng.random((ni - 1, nj - 1, nk - 1))).astype(np.float32)
    )
    vol = np.asfortranarray(
        (0.5 + rng.random((ni - 1, nj - 1, nk - 1))).astype(np.float32)
    )
    snapshot = np.asfortranarray(
        rng.standard_normal((ni, nj, nk, NP)).astype(np.float32)
    )

    cons_plain = _run(residual, dt_vol, vol, snapshot, ni, nj, nk, n_levels)
    cons_irs = _run(
        residual, dt_vol, vol, snapshot, ni, nj, nk, n_levels, sf_irs=0.8
    )

    corr_plain = np.abs(cons_plain - snapshot).mean()
    corr_irs = np.abs(cons_irs - snapshot).mean()
    assert corr_irs < corr_plain


def _make_block(shape):
    """Minimal block for exercising ember.solver.advance_rk_stage_mg directly.

    Only geometry/fluid/thermodynamic state are set up (mirrors
    conftest._make_block); residual_nd/dt_vol_nd/store are poked directly
    with synthetic data below rather than derived from a real flow solve, so
    this only exercises the multigrid RK-stage wiring, not the physics.
    """
    block = Block(shape=shape)
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    block.set_fluid(fluid)
    ni, nj, nk = shape
    x = np.linspace(0.0, 1.0, nj).reshape(1, -1, 1) * np.ones(shape)
    r = np.ones(shape) * 0.5
    t = np.linspace(0.0, 0.2, nk).reshape(1, 1, -1) * np.ones(shape)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)
    block.set_P_T(101325.0, 300.0)
    return block


def test_advance_rk_stage_mg_sf_irs_wiring():
    """ember.solver.advance_rk_stage_mg threads sf_irs through to the kernel.

    sf_irs=0 (the default) must reproduce omitting it exactly (the coarse
    smoothing is an exact no-op); sf_irs>0 must actually change the result.
    """
    ni, nj, nk = 17, 17, 17  # divisible by 2**1
    n_levels = 1
    rng = np.random.default_rng(3)
    # Drawn once and reused across the three builds below -- each _build() call
    # must see identical residual/dt_vol data, not fresh draws from a shared rng.
    residual_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))
    dt_vol_data = 0.5 + rng.random((ni - 1, nj - 1, nk - 1))

    def _build():
        block = _make_block((ni, nj, nk))
        block.residual_nd.flags.writeable = True
        block.residual_nd[...] = residual_data
        block.residual_nd.flags.writeable = False
        block.dt_vol_nd.flags.writeable = True
        block.dt_vol_nd[...] = dt_vol_data
        block.dt_vol_nd.flags.writeable = False
        block.store[...] = block.conserved_nd
        return ember.grid.Grid([block])

    grid_default = _build()
    ember.solver.advance_rk_stage_mg(
        grid_default, alpha=1.0, cfl=0.4, fac_mgrid=0.2, n_levels=n_levels
    )
    cons_default = grid_default[0].conserved_nd.copy()

    grid_zero = _build()
    ember.solver.advance_rk_stage_mg(
        grid_zero, alpha=1.0, cfl=0.4, fac_mgrid=0.2, n_levels=n_levels, sf_irs=0.0
    )
    cons_zero = grid_zero[0].conserved_nd.copy()
    np.testing.assert_array_equal(cons_default, cons_zero)

    grid_irs = _build()
    ember.solver.advance_rk_stage_mg(
        grid_irs, alpha=1.0, cfl=0.4, fac_mgrid=0.2, n_levels=n_levels, sf_irs=0.6
    )
    cons_irs = grid_irs[0].conserved_nd.copy()
    assert not np.array_equal(cons_default, cons_irs)


def test_advance_rk_stage_mg_fac_mgrid_zero_skips_coarse():
    """fac_mgrid=0 with n_levels>0 must collapse to the plain-RK path.

    The coarse correction scales by fac_mgrid, so fac_mgrid=0 contributes
    nothing; the dispatch should route to the n_levels=0 fast path, giving a
    result byte-identical to passing n_levels=0 -- even with sf_irs>0, which is
    inert then.
    """
    ni, nj, nk = 17, 17, 17
    n_levels = 2
    rng = np.random.default_rng(31)
    residual_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))
    dt_vol_data = 0.5 + rng.random((ni - 1, nj - 1, nk - 1))

    def _build():
        block = _make_block((ni, nj, nk))
        block.residual_nd.flags.writeable = True
        block.residual_nd[...] = residual_data
        block.residual_nd.flags.writeable = False
        block.dt_vol_nd.flags.writeable = True
        block.dt_vol_nd[...] = dt_vol_data
        block.dt_vol_nd.flags.writeable = False
        block.store[...] = block.conserved_nd
        return ember.grid.Grid([block])

    grid_off = _build()
    ember.solver.advance_rk_stage_mg(
        grid_off, alpha=1.0, cfl=0.4, fac_mgrid=0.2, n_levels=0
    )
    cons_off = grid_off[0].conserved_nd.copy()

    grid_zero_strength = _build()
    ember.solver.advance_rk_stage_mg(
        grid_zero_strength, alpha=1.0, cfl=0.4, fac_mgrid=0.0, n_levels=n_levels
    )
    np.testing.assert_array_equal(cons_off, grid_zero_strength[0].conserved_nd)

    grid_zero_irs = _build()
    ember.solver.advance_rk_stage_mg(
        grid_zero_irs, alpha=1.0, cfl=0.4, fac_mgrid=0.0, n_levels=n_levels, sf_irs=0.6
    )
    np.testing.assert_array_equal(cons_off, grid_zero_irs[0].conserved_nd)


if __name__ == "__main__":
    pytest.main([__file__])
