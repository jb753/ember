r"""The interior viscous body force has the magnitude of the physical stress.

The other viscous tests pin the kernel against goldens or against itself, so a
scale error common to every interior face passes all of them. This one checks
``F_body`` against an analytic answer instead. It imposes a field whose flux
divergence is known in closed form, and compares cell by cell.

The block is a thin annular sector. It is frictionless on its i and j faces,
so no wall function is involved, and periodic in k. The wall distance is zero,
so there is no mixing-length viscosity. Two fields are imposed:

* shear, :math:`V_x = c\,(r - r_0)^2` at uniform state, for which
  :math:`\tau_{xr} = \mu\,\mathrm{d}V_x/\mathrm{d}r` and the axial force per
  unit volume is :math:`\frac{1}{r}\frac{\mathrm{d}}{\mathrm{d}r}(r\,\tau_{xr})
  = 2\mu c\,(2r - r_0)/r`;
* conduction, :math:`T = T_0 + b\,(r - r_0)^2` at rest, for which the energy
  source per unit volume is :math:`2 k b\,(2r - r_0)/r`.

The discrete operator is exact for a quadratic up to the curvature correction,
so interior cells should match to well under 1%.
"""

import numpy as np

import ember.block
import ember.grid
from ember import util
from ember.fluid import PerfectFluid
from ember.inviscid import InviscidPatch
from ember.periodic import PeriodicPatch

SHAPE = (9, 17, 5)
NB = 2000
R0, R1 = 1.0, 1.05
MU, PR, CP = 1.8e-5, 0.72, 1005.0


def _build_grid(Vx, T):
    """Frictionless-walled thin annular sector carrying the given fields."""
    block = ember.block.Block(shape=SHAPE)
    block.set_Nb(NB)
    xrt = util.linmesh3((0.0, 0.05), (R0, R1), (0.0, 2.0 * np.pi / NB), SHAPE)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=CP, gamma=1.4, mu=MU, Pr=PR))
    r = np.asarray(block.r)
    block.set_P_T(1e5, T(r).astype(np.float32))
    block.set_Vx(Vx(r).astype(np.float32))
    block.set_Vr(0.0)
    block.set_Vt(0.0)
    block.set_wdist(np.zeros(SHAPE, dtype=np.float32))
    block.patches.append(PeriodicPatch(k=0))
    block.patches.append(PeriodicPatch(k=-1))
    block.patches.append(InviscidPatch(i=0))
    block.patches.append(InviscidPatch(i=-1))
    block.patches.append(InviscidPatch(j=0))
    block.patches.append(InviscidPatch(j=-1))
    grid = ember.grid.Grid([block])
    grid.connectivity.periodic.pair()
    grid.update_sources(inviscid=False)
    return block


def _cell_r_vol(block):
    """Cell-centre radius, as the eight-corner mean, and cell volume."""
    r = np.asarray(block.r, dtype=np.float64)
    ni, nj, nk = SHAPE
    rc = 0.125 * sum(
        r[a : a + ni - 1, b : b + nj - 1, c : c + nk - 1]
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    )
    return rc, np.asarray(block.vol, dtype=np.float64)


# Interior cells only: the j-boundary cells take their outer face stress from
# the one-sided halo, which is a separate question from the scale.
INTERIOR = (slice(1, -1), slice(2, -2), slice(None))


def test_shear_stress_scale():
    c = 1e4
    block = _build_grid(lambda r: c * (r - R0) ** 2, lambda r: 300.0 + 0.0 * r)
    rc, vol = _cell_r_vol(block)
    expect = 2.0 * MU * c * (2.0 * rc - R0) / rc * vol
    got = np.asarray(block.F_body_nd, dtype=np.float64)[..., 1]
    ratio = got[INTERIOR] / expect[INTERIOR]
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-2)


def test_heat_flux_scale():
    b = 4e4
    block = _build_grid(lambda r: 0.0 * r, lambda r: 300.0 + b * (r - R0) ** 2)
    rc, vol = _cell_r_vol(block)
    k = MU * CP / PR
    expect = 2.0 * k * b * (2.0 * rc - R0) / rc * vol
    got = np.asarray(block.F_body_nd, dtype=np.float64)[..., 4]
    ratio = got[INTERIOR] / expect[INTERIOR]
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-2)
