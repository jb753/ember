"""Tests for the scree multigrid kernels ``scree_mg_irs`` / ``scree_mg_noirs``
(the Denton block-sum multigrid variants of ``scree_plain``) for the scree
("n_stage == 0") explicit time-march.

Correctness contract:

- ``fac_mgrid == 0`` must reproduce ``scree_plain`` (coarse coefficient is
  exactly zero, so every coarse level's correction and prolong contribute
  nothing) to float32 rounding.
- The history roll (``store <- residual``) happens exactly once per cell.
- Every level -- fine and coarse -- restricts/uses the same Denton-extrapolated
  fine quantity ``q = 2*residual - store`` (formed once by ``scree_form_q``),
  verified against multall's ``TSTEP`` reference
  (``~/multall-open-20.9.f:6457-6532``), which sums its own lagged ``STORE``
  into the multigrid block accumulators, not the raw flux imbalance.
- ``sf_irs > 0`` applies coarse-level IRS and changes the result; ``scree_mg_noirs``
  reproduces ``scree_mg_irs`` at ``sf_irs == 0`` byte-for-byte.
"""

import numpy as np
import pytest

import ember.block
import ember.fluid
import ember.grid  # noqa: F401  binds ember.fortran
import ember.patch
import ember.set_iter
import ember.solver
from ember import util
from ember.block import Block
from ember.fluid import PerfectFluid

NP = 5
CFL = 0.4
FAC_MGRID = 0.2

# Shape divisible by 2**3, small enough to be fast.
NI, NJ, NK = 33, 33, 33
N_LEVELS = 3


def _make_inputs(ni, nj, nk, seed):
    """Build Fortran-ordered inputs for one scree_plain / scree_mg_* call."""
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
    store = np.asfortranarray(
        rng.standard_normal((ni - 1, nj - 1, nk - 1, NP)).astype(np.float32)
    )
    cons = np.asfortranarray(rng.standard_normal((ni, nj, nk, NP)).astype(np.float32))
    return residual, dt_vol, vol, store, cons


def _run_plain(residual, dt_vol, store, cons, ni, nj, nk):
    cons_out = np.asfortranarray(cons.copy())
    store_out = np.asfortranarray(store.copy())
    tmp = np.asfortranarray(np.zeros((ni - 1, nj - 1, nk - 1, NP), dtype=np.float32))
    ember.fortran.scree_plain(
        cons=cons_out,
        residual=residual,
        store=store_out,
        dt_vol=dt_vol,
        cfl=CFL,
        tmp=tmp,
    )
    return cons_out, store_out


def _run_mg(
    residual, dt_vol, vol, store, cons, ni, nj, nk, n_levels,
    fmgrid=FAC_MGRID, sf_irs=0.0, kernel=None,
):
    """Call a scree multigrid kernel with freshly zeroed scratch (the size args
    are inferred by f2py from the array shapes, exactly as ember.solver does).
    Requires ``n_levels >= 1`` (the engine has no fine-only path; that is
    ``scree_plain``, exercised by :func:`_run_plain`).

    ``kernel`` selects the entry point (default ``scree_mg_irs``, the IRS
    variant); pass ``scree_mg_noirs`` for the plain variant. Both take the
    identical signature.
    """
    if kernel is None:
        kernel = ember.fortran.scree_mg_irs
    cons_out = np.asfortranarray(cons.copy())
    store_out = np.asfortranarray(store.copy())
    nc1i, nc1j, nc1k = (ni - 1) // 2, (nj - 1) // 2, (nk - 1) // 2
    n_corr, n_res, n_tri = ember.solver._mg_coarse_scratch_sizes(ni, nj, nk, n_levels)
    acc_sz = nc1i * nc1j * nc1k * NP

    def Z(*shape):
        return np.asfortranarray(np.zeros(shape, dtype=np.float32))

    kernel(
        cons=cons_out,
        residual=residual,
        store=store_out,
        dt_vol=dt_vol,
        vol=vol,
        cfl=CFL,
        fmgrid=fmgrid,
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
    return cons_out, store_out


def test_fac_mgrid_zero_matches_plain():
    """fac_mgrid=0 with n_levels>0 must reproduce scree_plain (coarse coef=0, so
    every coarse correction is exactly zero and only the fine term survives).

    Not bit-exact: the mg path's fused fine term goes through mg_prolong2x_fine
    (ft + zero-correction prolong), whose FMA contraction of the +0 differs from
    fine_term's bare multiply by <=1 ULP. The store roll is exact."""
    residual, dt_vol, vol, store, cons = _make_inputs(NI, NJ, NK, seed=1)

    cons_plain, store_plain = _run_plain(residual, dt_vol, store, cons, NI, NJ, NK)
    cons_mg, store_mg = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS, fmgrid=0.0
    )

    np.testing.assert_allclose(cons_plain, cons_mg, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(store_plain, store_mg)


def test_fac_mgrid_positive_changes_result():
    """A positive fac_mgrid must actually alter the result vs plain scree_plain."""
    residual, dt_vol, vol, store, cons = _make_inputs(NI, NJ, NK, seed=2)

    cons_plain, _ = _run_plain(residual, dt_vol, store, cons, NI, NJ, NK)
    cons_mg, _ = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS
    )

    assert not np.array_equal(cons_plain, cons_mg)


def test_store_roll_happens_once():
    """store must equal residual exactly after the call, for any n_levels."""
    residual, dt_vol, vol, store, cons = _make_inputs(NI, NJ, NK, seed=3)

    _, store_mg = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS
    )

    np.testing.assert_array_equal(store_mg, residual)


def test_uniform_dt_vol_is_independent_of_vol():
    """The coarse dt is a weighted mean of dt_vol, so a uniform dt_vol must
    give back that constant whatever the volume distribution.

    This is the invariant that makes the block-average a no-op on a uniform
    mesh, where multall's PERPMIN/VOLB reduces to dt_vol/b**2 exactly.

    Not bit-exact: sum(c*vol)/sum(vol) recovers c only up to float32
    accumulation error, which grows with block size (~2 ULP over the 512-cell
    sum of the coarsest level, b=8).
    """
    residual, _, vol, store, cons = _make_inputs(NI, NJ, NK, seed=6)
    dt_vol = np.asfortranarray(np.full((NI - 1, NJ - 1, NK - 1), 0.7, dtype=np.float32))
    vol_other = np.asfortranarray(vol[::-1, ::-1, ::-1].copy())

    cons_a, _ = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS
    )
    cons_b, _ = _run_mg(
        residual, dt_vol, vol_other, store, cons, NI, NJ, NK, n_levels=N_LEVELS
    )

    np.testing.assert_allclose(cons_a, cons_b, rtol=1e-5, atol=1e-6)


def test_coarse_dt_is_scale_invariant_in_vol():
    """sum(dt_vol*vol)/sum(vol) normalises, so scaling vol must not move the result.

    Tolerance as for test_uniform_dt_vol_is_independent_of_vol: the scaled sums
    round differently in float32.
    """
    residual, dt_vol, vol, store, cons = _make_inputs(NI, NJ, NK, seed=7)
    vol_scaled = np.asfortranarray((vol * 3.0).astype(np.float32))

    cons_a, _ = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS
    )
    cons_b, _ = _run_mg(
        residual, dt_vol, vol_scaled, store, cons, NI, NJ, NK, n_levels=N_LEVELS
    )

    np.testing.assert_allclose(cons_a, cons_b, rtol=1e-5, atol=1e-6)


def test_vol_weighting_affects_nonuniform_dt_vol():
    """With dt_vol varying, the volume weights must actually enter the coarse dt.

    Guards against the kernel ignoring vol and falling back to an unweighted
    mean (or the old centre-cell sample).
    """
    residual, dt_vol, vol, store, cons = _make_inputs(NI, NJ, NK, seed=8)
    vol_uniform = np.asfortranarray(np.ones_like(vol))

    cons_weighted, _ = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS
    )
    cons_unweighted, _ = _run_mg(
        residual, dt_vol, vol_uniform, store, cons, NI, NJ, NK, n_levels=N_LEVELS
    )

    assert not np.array_equal(cons_weighted, cons_unweighted)


def test_irs_positive_changes_result():
    """A positive sf_irs must alter the coarse correction vs sf_irs=0."""
    residual, dt_vol, vol, store, cons = _make_inputs(NI, NJ, NK, seed=11)

    cons_mg, _ = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS
    )
    cons_irs, _ = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS, sf_irs=0.5
    )

    assert not np.array_equal(cons_mg, cons_irs)


def test_noirs_kernel_matches_fused_at_sf_zero():
    """scree_mg_noirs must reproduce scree_mg_irs with sf_irs=0 to float32
    rounding. The two share mg_coarse_correction and differ only in the
    smoother passed (mg_smooth_noop vs smooth_residual_tri, whose sf_irs<=0
    guard is itself a no-op), so the plain kernel is the non-IRS engine that
    solver.scree_step dispatches to when sf_irs==0; the branch-free q-engine
    reorders some floating-point sums between the two paths, so equality is
    only to a tolerance rather than byte-for-byte."""
    residual, dt_vol, vol, store, cons = _make_inputs(NI, NJ, NK, seed=21)

    cons_irs, store_irs = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS, sf_irs=0.0
    )
    cons_no, store_no = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS, sf_irs=0.0,
        kernel=ember.fortran.scree_mg_noirs,
    )

    np.testing.assert_allclose(cons_irs, cons_no, atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(store_irs, store_no, atol=1e-6, rtol=1e-3)


def test_irs_store_roll_happens_once():
    """The IRS path must still roll store <- residual exactly once per cell."""
    residual, dt_vol, vol, store, cons = _make_inputs(NI, NJ, NK, seed=12)

    _, store_irs = _run_mg(
        residual, dt_vol, vol, store, cons, NI, NJ, NK, n_levels=N_LEVELS, sf_irs=0.5
    )

    np.testing.assert_array_equal(store_irs, residual)


def _make_block(shape):
    """Minimal block for exercising ember.solver.scree_step directly.

    Only geometry/fluid/thermodynamic state are set up (mirrors
    tests/conftest.py's _make_block); residual_nd/dt_vol_nd/store are poked
    directly with synthetic data below rather than derived from a real flow
    solve, so this only exercises the multigrid scree-step wiring, not the
    physics.
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


def test_scree_step_mg_changes_result():
    """ember.solver.scree_step selects the mg kernel when n_levels>0."""
    ni, nj, nk = 17, 17, 17  # 16 cells, divisible by 2**2
    rng = np.random.default_rng(4)
    residual_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))
    dt_vol_data = 0.5 + rng.random((ni - 1, nj - 1, nk - 1))
    store_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))

    def _build():
        block = _make_block((ni, nj, nk))
        block.residual_nd.flags.writeable = True
        block.residual_nd[...] = residual_data
        block.residual_nd.flags.writeable = False
        block.dt_vol_nd.flags.writeable = True
        block.dt_vol_nd[...] = dt_vol_data
        block.dt_vol_nd.flags.writeable = False
        store_cell = util.carve_view(block.store, (ni - 1, nj - 1, nk - 1, NP))
        store_cell[...] = store_data
        return ember.grid.Grid([block])

    grid_plain = _build()
    ember.solver.scree_step(grid_plain, cfl=CFL, fac_mgrid=0.0, n_levels=0)
    cons_plain = grid_plain[0].conserved_nd.copy()

    grid_mg = _build()
    ember.solver.scree_step(grid_mg, cfl=CFL, fac_mgrid=0.2, n_levels=2)
    cons_mg = grid_mg[0].conserved_nd.copy()

    assert not np.array_equal(cons_plain, cons_mg)


def test_scree_step_default_unchanged():
    """scree_step with no new kwargs must be byte-identical to explicit zeros."""
    ni, nj, nk = 17, 17, 17
    rng = np.random.default_rng(5)
    residual_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))
    dt_vol_data = 0.5 + rng.random((ni - 1, nj - 1, nk - 1))
    store_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))

    def _build():
        block = _make_block((ni, nj, nk))
        block.residual_nd.flags.writeable = True
        block.residual_nd[...] = residual_data
        block.residual_nd.flags.writeable = False
        block.dt_vol_nd.flags.writeable = True
        block.dt_vol_nd[...] = dt_vol_data
        block.dt_vol_nd.flags.writeable = False
        store_cell = util.carve_view(block.store, (ni - 1, nj - 1, nk - 1, NP))
        store_cell[...] = store_data
        return ember.grid.Grid([block])

    grid_default = _build()
    ember.solver.scree_step(grid_default, cfl=CFL)
    cons_default = grid_default[0].conserved_nd.copy()

    grid_explicit = _build()
    ember.solver.scree_step(grid_explicit, cfl=CFL, fac_mgrid=0.0, n_levels=0)
    cons_explicit = grid_explicit[0].conserved_nd.copy()

    np.testing.assert_array_equal(cons_default, cons_explicit)


def test_scree_step_sf_irs_wiring():
    """ember.solver.scree_step threads sf_irs through to the fused kernel.

    sf_irs=0 (the default) must reproduce omitting it byte-for-byte (the coarse
    smoothing is an exact no-op); sf_irs>0 must change the coarse correction.
    """
    ni, nj, nk = 17, 17, 17  # 16 cells, divisible by 2**2
    n_levels = 2
    rng = np.random.default_rng(9)
    residual_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))
    dt_vol_data = 0.5 + rng.random((ni - 1, nj - 1, nk - 1))
    store_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))

    def _build():
        block = _make_block((ni, nj, nk))
        block.residual_nd.flags.writeable = True
        block.residual_nd[...] = residual_data
        block.residual_nd.flags.writeable = False
        block.dt_vol_nd.flags.writeable = True
        block.dt_vol_nd[...] = dt_vol_data
        block.dt_vol_nd.flags.writeable = False
        store_cell = util.carve_view(block.store, (ni - 1, nj - 1, nk - 1, NP))
        store_cell[...] = store_data
        return ember.grid.Grid([block])

    grid_no_irs = _build()
    ember.solver.scree_step(grid_no_irs, cfl=CFL, fac_mgrid=0.2, n_levels=n_levels)
    cons_no_irs = grid_no_irs[0].conserved_nd.copy()

    grid_zero = _build()
    ember.solver.scree_step(
        grid_zero, cfl=CFL, fac_mgrid=0.2, n_levels=n_levels, sf_irs=0.0
    )
    cons_zero = grid_zero[0].conserved_nd.copy()

    # sf_irs=0 is an exact no-op, so byte-identical to omitting it.
    np.testing.assert_array_equal(cons_no_irs, cons_zero)

    grid_irs = _build()
    ember.solver.scree_step(
        grid_irs, cfl=CFL, fac_mgrid=0.2, n_levels=n_levels, sf_irs=0.6
    )
    cons_irs = grid_irs[0].conserved_nd.copy()

    assert not np.array_equal(cons_no_irs, cons_irs)


def test_scree_step_fac_mgrid_zero_skips_coarse():
    """fac_mgrid=0 with n_levels>0 must collapse to the plain scree_plain path.

    The coarse correction scales by fac_mgrid, so fac_mgrid=0 contributes
    nothing; the dispatch sets n_levels_eff=0 and routes to scree_plain, giving a
    result byte-identical to passing n_levels=0 -- even with sf_irs>0, inert then.
    """
    ni, nj, nk = 17, 17, 17
    n_levels = 2
    rng = np.random.default_rng(14)
    residual_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))
    dt_vol_data = 0.5 + rng.random((ni - 1, nj - 1, nk - 1))
    store_data = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))

    def _build():
        block = _make_block((ni, nj, nk))
        block.residual_nd.flags.writeable = True
        block.residual_nd[...] = residual_data
        block.residual_nd.flags.writeable = False
        block.dt_vol_nd.flags.writeable = True
        block.dt_vol_nd[...] = dt_vol_data
        block.dt_vol_nd.flags.writeable = False
        store_cell = util.carve_view(block.store, (ni - 1, nj - 1, nk - 1, NP))
        store_cell[...] = store_data
        return ember.grid.Grid([block])

    grid_off = _build()
    ember.solver.scree_step(grid_off, cfl=CFL, fac_mgrid=0.0, n_levels=0)
    cons_off = grid_off[0].conserved_nd.copy()

    grid_zero_strength = _build()
    ember.solver.scree_step(grid_zero_strength, cfl=CFL, fac_mgrid=0.0, n_levels=n_levels)
    np.testing.assert_array_equal(cons_off, grid_zero_strength[0].conserved_nd)

    grid_zero_irs = _build()
    ember.solver.scree_step(
        grid_zero_irs, cfl=CFL, fac_mgrid=0.0, n_levels=n_levels, sf_irs=0.5
    )
    np.testing.assert_array_equal(cons_off, grid_zero_irs[0].conserved_nd)


def test_validate_mg_still_triggers_for_scree():
    """_validate_mg gates n_levels for scree the same way it does for RK."""
    ni, nj, nk = 10, 10, 10  # 9 cells, not divisible by 2**2
    block = _make_block((ni, nj, nk))
    grid = ember.grid.Grid([block])

    with pytest.raises(ValueError):
        ember.solver._validate_mg(grid, n_levels=2)


@pytest.fixture(scope="module")
def duct_grid_builder():
    """Factory for a small single-block duct with inlet/outlet patches."""

    def _build():
        shape = (17, 17, 17)  # 16 cells, divisible by 2**2
        L = 0.05
        fluid = ember.fluid.PerfectFluid(
            cp=1005.0, gamma=1.4, mu=1.0e-3, Pr=0.72, T_dtm=400.0
        )
        xrt = util.linmesh3([0.0, L], [1.0, 1.0 + L], [-0.02, 0.02], shape)

        block = ember.block.Block(shape=shape)
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)

        rho_o, e_o = fluid.set_P_T(1.0e5, 300.0)
        ho = fluid.get_h(rho_o, e_o)
        so = fluid.get_s(rho_o, e_o)
        a_o = fluid.get_a(rho_o, e_o)
        Ma = 0.3
        ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho, so, Ma, 0.0, 0.0)

        # Perturb Vx with a streamwise ramp so the march has a real transient
        # to relax, rather than starting at (near) its own steady state.
        Vx = np.asarray(block.Vx)
        ramp = np.linspace(0.95, 1.05, shape[0], dtype=Vx.dtype)
        block.set_Vx(Vx * ramp[:, None, None])

        block.patches.extend(
            [ember.patch.InletPatch(i=0), ember.patch.OutletPatch(i=-1)]
        )

        Po_in = block.Po[0].mean()
        To_in = block.To[0].mean()
        Alpha_in = block.Alpha[0].mean()
        P_out = block.P[-1].mean()
        T_out = block.T[-1].mean()

        grid = ember.grid.Grid([block])
        grid.set_L_ref(L)
        grid.patches.inlet[0].set_Po_To_Alpha_Beta(Po_in, To_in, Alpha_in, 0.0)
        grid.patches.outlet[0].set_P(P_out)
        grid.set_fluid(
            fluid.change_datum(P_out, T_out).change_ref(rho_o, Ma * a_o, block.Rgas.mean())
        )
        grid.calculate_wdist()
        return grid

    return _build


def test_scree_mg_converges_faster_than_plain_scree(duct_grid_builder):
    """A small synthetic duct converges further with multigrid than without."""
    n_step = 100
    common = dict(
        n_step=n_step,
        n_step_log=10,
        n_step_avg=1,
        # cfl in true-Courant units after the max-of-directional-radii timestep
        # normalisation (~0.17 here reproduces the physical step the old
        # sum-normalised cfl=0.4 gave, comfortably below the ~0.577 scree limit).
        cfl=0.17,
        n_stage=0,
        inviscid=True,
    )

    grid_plain = duct_grid_builder()
    hist_plain = ember.solver.run(
        grid_plain, ember.solver.SolverConfig(n_levels=0, fac_mgrid=0.0, **common)
    )

    grid_mg = duct_grid_builder()
    hist_mg = ember.solver.run(
        grid_mg, ember.solver.SolverConfig(n_levels=2, fac_mgrid=0.2, **common)
    )

    assert not hist_plain.diverged
    assert not hist_mg.diverged

    # The fixed-CFL march oscillates step to step (visible in both traces), so
    # compare the mean over the logged history rather than a single endpoint
    # -- a single-point comparison is noise-dominated and can land on either
    # side depending on exactly where the oscillation is sampled.
    n_plain = hist_plain.i_log + 1
    n_mg = hist_mg.i_log + 1
    resid_plain = np.abs(np.asarray(hist_plain.residual, dtype=float)[:n_plain, 4]).mean()
    resid_mg = np.abs(np.asarray(hist_mg.residual, dtype=float)[:n_mg, 4]).mean()
    assert resid_mg < resid_plain


def test_run_returns_trimmed_history_on_divergence(duct_grid_builder):
    """A blown-up march hands back only the rows it logged, all of them finite."""
    grid = duct_grid_builder()

    # Reversed flow is expected once the march blows up, so give the outlet a
    # backflow state; without it the patch raises before check_nan can fire.
    block = grid[0]
    fluid = block.fluid
    rho_o, e_o = fluid.set_P_T(1.0e5, 300.0)
    grid.patches.outlet[0].set_backflow(
        fluid.get_h(rho_o, e_o), fluid.get_s(rho_o, e_o), 0.0, 0.0
    )

    n_step, n_step_log = 20, 2
    n_alloc = -(-n_step // n_step_log)  # what from_grid would have allocated
    conf = ember.solver.SolverConfig(
        n_step=n_step,
        n_step_log=n_step_log,
        n_step_avg=1,
        cfl=50.0,  # far past the stability limit, so this must blow up
        n_stage=0,
        n_levels=2,
        fac_mgrid=0.4,
        sf_resid=0.0,
        inviscid=False,
    )

    # A field on its way to NaN drives temperature negative, so the entropy
    # log() legitimately goes invalid before check_nan trips. The suite turns
    # warnings into errors, so scope that to the march itself.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        hist = ember.solver.run(grid, conf)

    assert hist.diverged
    assert hist.shape[0] < n_alloc  # the unwritten tail is gone
    assert hist.i_log + 1 == hist.shape[0]  # and i_log still agrees with it

    # Nothing for a caller to mask: every row that survived holds real data.
    for name in ("i_step", "err_mdot", "zeta"):
        assert np.isfinite(np.asarray(getattr(hist, name), dtype=float)).all(), name
    assert np.isfinite(np.asarray(hist.residual, dtype=float)).all()


if __name__ == "__main__":
    pytest.main([__file__])
