"""Gate Grid.update_residual's post-processing order: smooth, then damp.

Historically (commits 0384c83..495b415) the change limiter was fused into
``set_residual``/the IRS i-solve for a performance win, which as a side
effect reordered the post-processing to damp-before-IRS -- a genuine
numerics change (see those commits' messages: the two orderings differ by
~19% of the field scale at production defaults). ``update_residual`` has
since been rewired back to the original order -- IRS then the limiter --
by calling the same kernels unfused: ``set_residual`` with ``dampin=0``,
then ``smooth_residual_scale_tri`` with ``dampin=0`` (IRS only) when
``sf > 0``, then ``damp_residual`` as a separate pass when ``dampin`` is
set.

This test gates that wiring: the reference calls those three kernels
directly, independent of ``Grid.update_residual``, so a future edit to the
method's call order (e.g. reintroducing a fusion that changes the
composition) is caught rather than silently accepted.
"""

import numpy as np
import pytest

import ember
import ember.grid  # noqa: F401  binds ember.fortran
from ember import util
from ember.cases import build_duct_grid

SF = 0.5
DAMPIN = 2.0


def _case(ncell=100_000):
    """A duct driven to the state update_residual consumes, with cross-stream
    momentum seeded so the j- and k-face mass fluxes are not identically zero.

    build_duct_grid is axially straight (Vr = Vt = 0), so without the swirl the
    j/k face flows vanish and the j- and k-direction Thomas solves -- half of
    what this test is gating -- would act on an all-zero field.
    """
    grid = build_duct_grid(ncell)
    grid.update_cached_conserved()
    grid.apply_bconds()
    grid.update_sources(False, 0.0)
    grid.update_timestep(rf=1.0)

    b = grid[0]
    cons = b.conserved_nd
    cons.flags.writeable = True
    rng = np.random.default_rng(0)
    scale = 0.05 * float(np.abs(cons[..., 1]).max())
    for m in (2, 3):
        cons[..., m] += scale * rng.standard_normal(cons.shape[:3]).astype(cons.dtype)
    # Essential: arms reading cached nodal primitives would otherwise solve a
    # different state from the one written here.
    grid.update_cached_conserved()
    return grid, b


def _reference(grid, b, dampin, sf):
    """set_residual, then IRS (if sf > 0), then damp_residual (if dampin is
    set) -- called kernel-by-kernel, independent of update_residual's own
    wiring."""
    b.update_primitive()
    ni, nj, nk = b.shape
    i_cusp_start, i_cusp_end = b.i_cusp
    kb = min(ember.grid._KB_SLAB, nk - 1)
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
    planes, rows = util.carve_view(b.scratch, (ni, njp, 5, 2), (ni, 5, 3))
    du = b.residual_nd
    du.flags.writeable = True
    ember.fortran.set_residual(
        cons=b.conserved_nd,
        p=b.P_nd,
        p_offset=b.P_offset_nd,
        r=b.r_nd,
        omega=b.Omega_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        du=du,
        f_body=b.F_body_nd,
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_nd,
        ho=b.ho_nd,
        planes=planes,
        rows=rows,
        **b.ijk_wall_conv,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
        kb=kb,
        njp=njp,
        ni=ni,
        nj=nj,
        nk=nk,
        dt_vol=b.dt_vol_nd,
        dampin=0.0,
    )
    if sf > 0.0:
        nwork = 2 * ((ni - 1) + (nj - 1) + (nk - 1))
        ember.fortran.smooth_residual_scale_tri(
            du=du,
            dt_vol=b.dt_vol_nd,
            ravg=np.zeros(5, dtype=du.dtype),
            dampin=0.0,
            sf=sf,
            work=util.carve_view(b.scratch, (nwork,)),
            ni=ni,
            nj=nj,
            nk=nk,
        )
    if dampin is not None:
        ember.fortran.damp_residual(
            du=du,
            dt_vol=b.dt_vol_nd,
            dampin=dampin,
            ni=ni,
            nj=nj,
            nk=nk,
        )
    du.flags.writeable = False
    return np.array(du, copy=True)


@pytest.mark.parametrize("dampin", [None, DAMPIN])
@pytest.mark.parametrize("sf", [0.0, SF])
def test_update_residual_matches_unfused_kernels(dampin, sf):
    """update_residual(dampin, sf) == set_residual, IRS, damp -- byte for byte."""
    grid, b = _case()
    want = _reference(grid, b, dampin, sf)

    # Rebuild the same state so update_residual starts from identical inputs.
    grid2, b2 = _case()
    grid2.update_residual(dampin=dampin, sf=sf)
    got = np.array(b2.residual_nd, copy=True)

    assert np.array_equal(got, want), (
        f"update_residual differs from the unfused kernel sequence at "
        f"dampin={dampin}, sf={sf}: max |diff| = {np.abs(got - want).max():.3e}, "
        f"{np.count_nonzero(got != want)} of {got.size} elements"
    )


def test_limiter_and_irs_both_actually_act():
    """Guard against the gate above passing because nothing happened.

    If either flag were silently inert, every parametrisation would agree
    trivially and the bitwise test would prove nothing.
    """
    grid, b = _case()
    grid.update_residual(dampin=None, sf=0.0)
    plain = np.array(b.residual_nd, copy=True)

    grid.update_residual(dampin=DAMPIN, sf=0.0)
    damped = np.array(b.residual_nd, copy=True)

    grid.update_residual(dampin=None, sf=SF)
    smoothed = np.array(b.residual_nd, copy=True)

    assert not np.array_equal(plain, damped), "change limiter had no effect"
    assert not np.array_equal(plain, smoothed), "IRS had no effect"


def test_damp_runs_after_irs_not_before():
    """The order is smooth-then-damp: damping the already-smoothed residual
    must differ from smoothing an already-damped one, confirming
    update_residual takes the former path (the pre-0384c83 order)."""
    grid, b = _case()
    grid.update_residual(dampin=DAMPIN, sf=SF)
    smooth_then_damp = np.array(b.residual_nd, copy=True)

    ni, nj, nk = b.shape
    grid2, b2 = _case()
    du = b2.residual_nd
    du.flags.writeable = True
    ember.fortran.damp_residual(
        du=du, dt_vol=b2.dt_vol_nd, dampin=DAMPIN, ni=ni, nj=nj, nk=nk
    )
    nwork = 2 * ((ni - 1) + (nj - 1) + (nk - 1))
    ember.fortran.smooth_residual_scale_tri(
        du=du,
        dt_vol=b2.dt_vol_nd,
        ravg=np.zeros(5, dtype=du.dtype),
        dampin=0.0,
        sf=SF,
        work=util.carve_view(b2.scratch, (nwork,)),
        ni=ni,
        nj=nj,
        nk=nk,
    )
    du.flags.writeable = False
    damp_then_smooth = np.array(du, copy=True)

    assert not np.array_equal(smooth_then_damp, damp_then_smooth), (
        "update_residual's output matches damp-before-IRS: the order "
        "reverted to the fused (0384c83) sequence"
    )
