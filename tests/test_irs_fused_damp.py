"""Gate the fine-grid IRS path that fuses the change limiter into the i-solve.

``Grid.update_residual`` used to run three full-volume read/write pairs after
the residual sweep: the limiter's scaling (``set_residual``'s trailing
``scale_du``), then the IRS i-solve, then the j+k solves. The scaling is
pointwise and the i-solve for a row depends only on that row's scaled values,
so the first two now share one traversal -- ``set_residual`` hands out the
block means it already accumulates and skips its own scaling, and
``smooth_residual_scale_tri`` applies it inside the tile gather.

That fusion must be BITWISE, not merely close: the arithmetic is unchanged and
only the traversal differs, so any deviation is a bug rather than a tolerance
to widen. It is gated here because nothing else covers it -- the header of
``set_residual`` records that no test drove ``update_residual`` with ``dampin``
set and ``sf > 0`` together, which is precisely the combination the fusion
changes. All four (``dampin``, ``sf``) combinations are checked, since the two
flags select different paths through the kernel.

The reference is the unfused sequence, called kernel-by-kernel: full
``set_residual`` including its own ``scale_du``, then the standalone
``smooth_residual_tri_tiled`` that ``scree.f90``'s coarse-MG path still uses.
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
    """The unfused sequence: set_residual with its own scaling, then the
    standalone three-direction smoother."""
    ni, nj, nk = b.shape
    # dampin=None disables the limiter in update_residual's own convention.
    grid.update_residual(dampin=dampin, sf=0.0)
    du = b.residual_nd
    if sf > 0.0:
        du.flags.writeable = True
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
def test_fused_limiter_is_bitwise(dampin, sf):
    """update_residual(dampin, sf) == scale-then-smooth, byte for byte."""
    grid, b = _case()
    want = _reference(grid, b, dampin, sf)

    # Rebuild the same state so the fused path starts from identical inputs.
    grid2, b2 = _case()
    grid2.update_residual(dampin=dampin, sf=sf)
    got = np.array(b2.residual_nd, copy=True)

    assert np.array_equal(got, want), (
        f"fused limiter+IRS differs at dampin={dampin}, sf={sf}: "
        f"max |diff| = {np.abs(got - want).max():.3e}, "
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
