"""Gate Grid.update_residual's post-processing order: damp, then smooth.

The change limiter runs BEFORE the IRS smoother, fused into whichever kernel
is already traversing the volume -- ``set_residual``'s own trailing scaling
pass when IRS is off, and ``smooth_residual_scale_tri``'s i-solve gather when
it is on. Both fusions must be BITWISE equal to the unfused sequence: the
arithmetic is unchanged and only the traversal differs, so any deviation is a
bug rather than a tolerance to widen.

The reference here is that unfused sequence, called kernel-by-kernel and
independent of ``update_residual``'s own wiring: full ``set_residual``
including its ``scale_du``, then the standalone ``smooth_residual_tri_tiled``
that ``scree.f90``'s coarse-MG path still uses. All four (``dampin``, ``sf``)
combinations are checked, since the two flags select different paths through
the kernels, plus a guard that the composition really is damp-before-IRS --
the opposite order (in place between commits 9ff054e and this one) is a
genuine numerics difference, ~19% of the field scale at production defaults.
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


def _set_residual(b, dampin):
    """set_residual on block ``b``, with its own scaling pass when dampin > 0.

    Leaves ``residual_nd`` writeable for the smoother that follows.
    """
    b.update_primitive()
    ni, nj, nk = b.shape
    i_cusp_start, i_cusp_end = b.i_cusp
    kb = min(ember.grid._KB_SLAB, nk - 1)
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
    planes, rows = util.carve_view(b.scratch, (ni, njp, 5, 2), (ni, 5, 3))
    b.residual_nd.flags.writeable = True
    return ember.fortran.set_residual(
        cons=b.conserved_nd,
        p=b.P_nd,
        p_offset=b.P_offset_nd,
        r=b.r_nd,
        omega=b.Omega_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        du=b.residual_nd,
        f_body=b.F_body_nd,
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
        dampin=0.0 if dampin is None else dampin,
    )


def _reference(grid, b, dampin, sf):
    """The unfused sequence: set_residual with its own scaling, then the
    standalone three-direction smoother -- neither of which knows about the
    other, so this composition is independent of update_residual's wiring."""
    ni, nj, nk = b.shape
    _set_residual(b, dampin)
    du = b.residual_nd
    if sf > 0.0:
        nwork = 2 * ((ni - 1) + (nj - 1) + (nk - 1))
        ember.fortran.smooth_residual_tri_tiled(
            du=du,
            sf=sf,
            work=util.carve_view(b.scratch, (nwork,)),
            ni=ni,
            nj=nj,
            nk=nk,
        )
    du.flags.writeable = False
    return np.array(du, copy=True)


@pytest.mark.parametrize("dampin", [None, DAMPIN])
@pytest.mark.parametrize("sf", [0.0, SF])
def test_update_residual_matches_unfused_kernels(dampin, sf):
    """update_residual(dampin, sf) == scale-then-smooth, byte for byte."""
    grid, b = _case()
    want = _reference(grid, b, dampin, sf)

    # Rebuild the same state so the fused path starts from identical inputs.
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


def test_damp_runs_before_irs_not_after():
    """The order is damp-then-smooth: smoothing an already-damped residual
    must differ from damping an already-smoothed one, confirming
    update_residual takes the former path."""
    grid, b = _case()
    grid.update_residual(dampin=DAMPIN, sf=SF)
    damp_then_smooth = np.array(b.residual_nd, copy=True)

    grid2, b2 = _case()
    ni, nj, nk = b2.shape
    _set_residual(b2, None)
    du = b2.residual_nd
    nwork = 2 * ((ni - 1) + (nj - 1) + (nk - 1))
    ember.fortran.smooth_residual_tri_tiled(
        du=du,
        sf=SF,
        work=util.carve_view(b2.scratch, (nwork,)),
        ni=ni,
        nj=nj,
        nk=nk,
    )
    ember.fortran.damp_residual(
        du=du, dt_vol=b2.dt_vol_nd, dampin=DAMPIN, ni=ni, nj=nj, nk=nk
    )
    du.flags.writeable = False
    smooth_then_damp = np.array(du, copy=True)

    assert not np.array_equal(damp_then_smooth, smooth_then_damp), (
        "update_residual's output matches smooth-before-damp: the order "
        "reverted to the unfused (9ff054e) sequence"
    )
