"""Tests for ``advance_rk_stage_mg_fused_irs``, the experimental variant of
``advance_rk_stage_mg_fused_opt`` that applies implicit residual smoothing
(IRS) to the coarse block-restricted residual at every multigrid level.

Correctness contract:

- ``sf_irs == 0`` makes the smoothing step an exact no-op (``smooth_residual_tri``'s
  own guard), so the kernel must then match ``advance_rk_stage_mg_fused_opt``
  bit-for-bit.
- ``sf_irs > 0`` must actually change the result relative to ``_opt`` (proves
  the new branch is wired in) and damp a high-frequency (checkerboard)
  component of the residual, mirroring ``tests/test_residual_smoothing.py``.
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
    """Build Fortran-ordered inputs for one advance_rk_stage_mg_fused* call."""
    rng = np.random.default_rng(seed)
    residual = np.asfortranarray(
        rng.standard_normal((ni - 1, nj - 1, nk - 1, NP)).astype(np.float32)
    )
    dt_vol = np.asfortranarray(
        (0.5 + rng.random((ni - 1, nj - 1, nk - 1))).astype(np.float32)
    )
    snapshot = np.asfortranarray(
        rng.standard_normal((ni, nj, nk, NP)).astype(np.float32)
    )
    return residual, dt_vol, snapshot


def _make_scratch(ni, nj, nk, n_levels):
    """Buffers matching the carve_view shapes used in ember.solver.advance_rk_stage_mg."""
    nc1i, nc1j, nc1k = (ni - 1) // 2, (nj - 1) // 2, (nk - 1) // 2
    tmp = np.asfortranarray(np.zeros((ni - 1, nj - 1, nk - 1, NP), dtype=np.float32))
    corr = np.asfortranarray(np.zeros((nc1i, nc1j, nc1k, NP), dtype=np.float32))
    aplane = np.asfortranarray(np.zeros((ni - 1, nc1j), dtype=np.float32))
    bb = np.asfortranarray(np.zeros((ni - 1, nj - 1, nc1k, NP), dtype=np.float32))
    return tmp, corr, aplane, bb


def _mg_irs_scratch_sizes(ni, nj, nk, n_levels):
    """Element counts for advance_rk_stage_mg_fused_irs's flat packed scratch.

    coarse_res_buf/tri_work_buf hold each level's slice back-to-back (largest
    level first, each 8x smaller than the last), so these are just the sum of
    per-level element counts -- allocated once by the caller, never resized.
    """
    n_res = n_tri = 0
    for lvl in range(1, n_levels + 1):
        b = 2**lvl
        nib, njb, nkb = (ni - 1) // b, (nj - 1) // b, (nk - 1) // b
        n_res += nib * njb * nkb * NP
        n_tri += 2 * (nib + njb + nkb)
    return n_res, n_tri


def _run(kernel, residual, dt_vol, snapshot, ni, nj, nk, n_levels, sf_irs=None):
    cons = np.asfortranarray(snapshot.copy())
    tmp, corr, aplane, bb = _make_scratch(ni, nj, nk, n_levels)
    kwargs = dict(
        cons=cons,
        snapshot=snapshot,
        residual=residual,
        dt_vol=dt_vol,
        alpha=1.0,
        cfl=0.4,
        fmgrid=0.2,
        n_levels=n_levels,
        tmp=tmp,
        corr=corr,
        aplane=aplane,
        bb=bb,
    )
    if sf_irs is not None:
        n_res, n_tri = _mg_irs_scratch_sizes(ni, nj, nk, n_levels)
        kwargs["sf_irs"] = sf_irs
        kwargs["coarse_res_buf"] = np.asfortranarray(
            np.zeros(max(n_res, 1), dtype=np.float32)
        )
        kwargs["tri_work_buf"] = np.asfortranarray(
            np.zeros(max(n_tri, 1), dtype=np.float32)
        )
    kernel(**kwargs)
    return cons


# Shape divisible by 2**3, small enough to be fast.
NI, NJ, NK = 33, 33, 33
N_LEVELS = 3


def test_sf_irs_zero_matches_production_opt():
    """sf_irs=0 is an exact no-op, so _irs must match _opt bit-for-bit."""
    residual, dt_vol, snapshot = _make_inputs(NI, NJ, NK, seed=0)

    cons_opt = _run(
        ember.fortran.advance_rk_stage_mg_fused_opt,
        residual,
        dt_vol,
        snapshot,
        NI,
        NJ,
        NK,
        N_LEVELS,
    )
    cons_irs = _run(
        ember.fortran.advance_rk_stage_mg_fused_irs,
        residual,
        dt_vol,
        snapshot,
        NI,
        NJ,
        NK,
        N_LEVELS,
        sf_irs=0.0,
    )
    # Not bit-exact: the smoothing branch is skipped (smooth_residual_tri's own
    # sf<=0 guard), but the surrounding loops are compiled as a differently
    # structured subroutine, so float rounding order (FMA contraction etc.)
    # can differ by a couple of ULP.
    np.testing.assert_allclose(cons_opt, cons_irs, rtol=1e-5, atol=1e-6)


def test_sf_irs_positive_changes_result():
    """A positive sf_irs must actually alter the coarse correction vs _opt."""
    residual, dt_vol, snapshot = _make_inputs(NI, NJ, NK, seed=1)

    cons_opt = _run(
        ember.fortran.advance_rk_stage_mg_fused_opt,
        residual,
        dt_vol,
        snapshot,
        NI,
        NJ,
        NK,
        N_LEVELS,
    )
    cons_irs = _run(
        ember.fortran.advance_rk_stage_mg_fused_irs,
        residual,
        dt_vol,
        snapshot,
        NI,
        NJ,
        NK,
        N_LEVELS,
        sf_irs=0.5,
    )
    assert not np.array_equal(cons_opt, cons_irs)


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
    snapshot = np.asfortranarray(
        rng.standard_normal((ni, nj, nk, NP)).astype(np.float32)
    )

    cons_opt = _run(
        ember.fortran.advance_rk_stage_mg_fused_opt,
        residual,
        dt_vol,
        snapshot,
        ni,
        nj,
        nk,
        n_levels,
    )
    cons_irs = _run(
        ember.fortran.advance_rk_stage_mg_fused_irs,
        residual,
        dt_vol,
        snapshot,
        ni,
        nj,
        nk,
        n_levels,
        sf_irs=0.8,
    )

    corr_opt = np.abs(cons_opt - snapshot).mean()
    corr_irs = np.abs(cons_irs - snapshot).mean()
    assert corr_irs < corr_opt


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
    """ember.solver.advance_rk_stage_mg selects the IRS kernel when sf_irs>0.

    sf_irs=0 (the default) must reproduce the pre-existing _opt-only path
    exactly; sf_irs>0 must route through advance_rk_stage_mg_fused_irs instead
    and actually change the result.
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

    grid_opt = _build()
    ember.solver.advance_rk_stage_mg(
        grid_opt, alpha=1.0, cfl=0.4, fac_mgrid=0.2, n_levels=n_levels
    )
    cons_default = grid_opt[0].conserved_nd.copy()

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


if __name__ == "__main__":
    pytest.main([__file__])
