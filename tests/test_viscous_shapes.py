"""The viscous pair on blocks too small for its usual assumptions.

``set_visc_force`` walks k producing a rolling tau/q cell-plane pair, and
panels that walk in j. Both structures have edges -- the first and last k
plane, the first and last panel -- and a block can be small enough that an
edge is all there is: one cell plane in k leaves the walk with no interior
step at all, and a j extent at or under the minimum panel width leaves a
single panel that is also both boundaries.

Those are the shapes where an off-by-one in the panel bounds or the rolling
slot swap stops being masked by an interior that hides it, and they are also
the shapes the rest of the suite does not cover: the fixtures elsewhere are
comfortably larger than a panel. What is asserted is deliberately weak --
finite, non-trivial, and independent of the panel width -- because there is
no independent reference for these shapes; the point is that the kernel
neither reads off the end nor silently skips the block.

``nk = 2`` (one cell plane in k) is included because the cusp seam pass
explicitly does not support it: the two seam cells coincide there. The kernel
must skip that pass and still produce the rest, rather than corrupt it.
"""

import numpy as np
import pytest

import ember.block
from ember import util
from ember.fluid import PerfectFluid

import viscous_util

PR_TURB = 1.0

# ni, nj, nk in NODES. nj = 5 is four cell rows, the narrowest panel the
# kernel will use, so the panel and the block boundary coincide.
#
# A SINGLE cell row in j (nj = 2) is not here, and not because of the panel:
# the wall masks are (1, nj-1, nk-1) slabs, and f2py cannot map one with two
# singleton dimensions onto the kernel's (nj-1, nk-1) dummy. That predates the
# fused kernel -- the wall arguments are unchanged -- so it is a limitation to
# know about, not a regression to gate.
SHAPES = [
    (5, 5, 3),    # one interior k plane
    (5, 5, 2),    # ONE cell plane in k: no cusp pass, no interior k step
    (4, 5, 5),    # two cell rows in i, shorter than a SIMD vector
    (9, 13, 4),   # three panels of four rows, the last one short
]


def _build(shape):
    """Smallest sensible viscous block: sheared, swirling, walls all round."""
    Nb = 36
    pitch = 2.0 * np.pi / Nb
    block = ember.block.Block(shape=shape)
    block.set_Nb(Nb)
    xrt = util.linmesh3((0.0, 0.1), (0.5, 1.0), (0.0, pitch), shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
    block.set_P_T(101325.0, 300.0)
    block.set_wdist(np.full(block.r.shape, 0.004, dtype=np.float32))

    x, r, t = block.x, block.r, block.t
    span = max(float(r.max() - r.min()), 1e-9)
    axial = max(float(x.max()), 1e-9)
    block.set_Vx((100.0 + 20.0 * (r - r.min()) / span
                  + 10.0 * np.sin(2.0 * np.pi * x / axial)).astype(np.float32))
    block.set_Vr((5.0 * np.cos(2.0 * np.pi * t / pitch)).astype(np.float32))
    block.set_Vt((30.0 + 12.0 * (t - t.min()) / pitch).astype(np.float32))
    block.set_Omega(50.0)
    return block


@pytest.mark.parametrize("shape", SHAPES)
def test_viscous_pair_survives_degenerate_shapes(shape):
    block = _build(shape)
    fvisc = viscous_util.run_pair(block, PR_TURB)

    assert np.all(np.isfinite(fvisc)), f"non-finite viscous force at {shape}"
    assert np.abs(fvisc).max() > 0.0, (
        f"the viscous force is identically zero at {shape} -- the kernel "
        "returned without doing anything"
    )
    mu_turb = np.asarray(block.mu_turb)[:-1, :-1, :-1]
    assert np.all(np.isfinite(mu_turb)) and np.all(mu_turb >= 0.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_panel_width_is_irrelevant_at_degenerate_shapes(shape):
    """Whatever the panel width, a block this small is one or two panels.

    The widths below bracket every j extent here, so between them they cover
    a panel narrower than the block, exactly the block, and wider -- and the
    answer may not depend on which. This is the same invariance
    test_viscous_phases_golden makes on a normal block, at the sizes where
    the panel bookkeeping has no interior to hide in.
    """
    block = _build(shape)
    viscous_util.fill_faces(block, PR_TURB)
    ref = viscous_util.run_visc_force(block, PR_TURB, jbw=4096)
    for jbw in (4, 8):
        got = viscous_util.run_visc_force(block, PR_TURB, jbw=jbw)
        np.testing.assert_array_equal(got, ref, err_msg=f"jbw={jbw} at {shape}")
