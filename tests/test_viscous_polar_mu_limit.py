"""Regression test: the composed viscous+polar force reduces to polar alone as mu -> 0.

The body force the march hands the residual is built in two passes: the
viscous pair (``set_tau_q_faces`` then ``set_visc_force``, via
:meth:`ember.grid.Grid.update_sources`) and then the cell sources, which the
timestep pass adds on top (``set_timestep_sources`` with ``add_sources``).
Taking the viscosity to (numerically) zero drives the viscous part to zero, so
the composed force must match the polar source on its own. That isolates the
polar term inside the composition, where a sign or scale error in it -- or a
second, stale copy of it left in the viscous kernel, which is where it used to
live -- would be easy to miss against O(1) viscous terms. The polar source was
once fused into ``set_visc_force`` and a turbine divergence was investigated
through that fusion; this test outlived the move out, because the property it
locks is about the composition, not where the term is computed.

The fixture is the rotating, swirling, sheared single block used by
``test_viscous_phases_golden.py``, with ``mu`` parametrized down to a
negligible value -- except the i/j faces are frictionless
(:class:`~ember.inviscid.InviscidPatch`) rather than that golden's no-slip
walls. The wall function's ``Re = rho*V*d/mu`` term is a log-law formulation
that approaches a finite, mu-*independent* value as ``mu -> 0`` (physically
correct -- high-Re wall shear does not vanish), so a walled fixture would
leave a genuine, non-shrinking wall-shear residual. Frictionless faces
sidestep the wall function entirely, leaving only the direct
laminar/mixing-length stress terms, both of which do vanish as mu -> 0 (the
mixing-length clamp ``visc_lim = 3000*mu`` collapses ``mu_turb`` right along
with the molecular term).
"""

import numpy as np

import ember.block
import ember.grid
from ember import util
from ember.fluid import PerfectFluid

import viscous_util
from ember.inviscid import InviscidPatch
from ember.periodic import PeriodicPatch

SHAPE = (7, 9, 9)  # k (theta) has 8 cells = two wavelengths of the Vx pattern
NB = 36
PR_TURB = 1.0

# ~7 orders of magnitude smaller than production-scale mu (1.8e-5, matches
# test_viscous_phases_golden.py) -- small enough that both the molecular
# stress and the mu_turb mixing-length clamp (visc_lim = 3000*mu, see
# set_tau_q_soa) collapse the composed force's viscous residual to ~1e-5 of
# the polar term's own scale (confirmed empirically: the residual scales
# linearly with mu down to at least 1e-18 once the wall function is
# bypassed -- see module docstring), but still strictly positive since
# PerfectFluid rejects mu <= 0.
MU_NEGLIGIBLE = 1e-12


def _build_block(mu):
    """Single-block rotating, swirling, sheared flow -- see module docstring.

    Modelled on test_viscous_phases_golden.py's fixture (same geometry, flow
    field, wall distance, rotation) so this exercises the same rotating-frame
    code paths as that golden, just parametrized by mu and with frictionless
    (not no-slip) i/j faces -- see module docstring for why.
    """
    pitch = 2.0 * np.pi / NB

    block = ember.block.Block(shape=SHAPE)
    block.set_Nb(NB)
    xrt = util.linmesh3((0.0, 0.15), (0.5, 0.9), (0.0, pitch), SHAPE)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=mu, Pr=0.72))

    x, r, t = block.x, block.r, block.t
    r_span = float(r.max() - r.min())

    Temp = (
        300.0
        + 20.0 * (r - r.min()) / r_span
        + 8.0 * np.sin(2.0 * np.pi * x / float(x.max()))
    ).astype(np.float32)
    block.set_P_T(101325.0, Temp)

    Vx = (
        100.0
        + 20.0 * np.sin(4.0 * np.pi * t / pitch + np.pi / 4.0)
        + 10.0 * (r - r.min()) / r_span
    ).astype(np.float32)
    Vr = (5.0 * np.cos(2.0 * np.pi * t / pitch)).astype(np.float32)
    Vt = (40.0 + 15.0 * np.sin(2.0 * np.pi * x / float(x.max()))).astype(np.float32)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Nonzero rotation -- the LISA turbine's rotor is exactly this: the one
    # block with large Omega, and hence a large polar source relative to the
    # (here, negligible) viscous term it is composed with.
    block.set_Omega(50.0)

    wdist = 0.008 * (1.0 + np.sin(np.pi * (r - r.min()) / r_span))
    block.set_wdist(wdist.astype(np.float32))

    block.patches.append(PeriodicPatch(k=0))
    block.patches.append(PeriodicPatch(k=-1))
    block.patches.append(InviscidPatch(i=0))
    block.patches.append(InviscidPatch(i=-1))
    block.patches.append(InviscidPatch(j=0))
    block.patches.append(InviscidPatch(j=-1))
    return block


def _add_cell_sources(block):
    """The timestep pass's source half, as the march runs it after the viscous
    pair: polar onto whatever F_body_nd holds (SFD off)."""
    ember.grid.Grid([block]).update_timestep(rf=1.0, add_sources=True)
    return np.array(block.F_body_nd)


def _run_polar_only(block):
    """The polar source alone, onto a freshly zeroed F_body_nd."""
    fbody = block.F_body_nd
    fbody.flags.writeable = True
    fbody.fill(0.0)
    fbody.flags.writeable = False
    return _add_cell_sources(block)


def _run_composed(block):
    """The real viscous pair, exactly as Grid.update_sources runs it
    (``set_tau_q_faces`` for the boundary shell, then ``set_visc_force``,
    which zeroes F_body_nd and derives every interior tau/q itself), then the
    cell sources on top."""
    viscous_util.fill_faces(block, PR_TURB)
    viscous_util.run_visc_force(block, PR_TURB)
    return _add_cell_sources(block)


def test_polar_source_is_nonzero():
    """Sanity check on the fixture itself: if this fires, the comparison
    below would pass vacuously (both sides ~0) without exercising anything."""
    block = _build_block(MU_NEGLIGIBLE)
    polar_only = _run_polar_only(block)
    assert np.abs(polar_only).max() > 1e-3


def test_composed_matches_polar_only_as_mu_vanishes():
    """The regression this guards against: a sign/scale error in how the
    polar source enters the composed force -- or a stale second copy of it --
    invisible next to O(1) viscous terms but decisive once the viscous term
    is driven to negligible size relative to the polar term."""
    block_polar = _build_block(MU_NEGLIGIBLE)
    polar_only = _run_polar_only(block_polar)

    block_composed = _build_block(MU_NEGLIGIBLE)
    composed = _run_composed(block_composed)

    polar_scale = float(np.abs(polar_only).max())
    assert polar_scale > 0.0  # see test_polar_source_is_nonzero

    # atol scaled to the polar term's own magnitude: at MU_NEGLIGIBLE the
    # residual viscous contribution measures ~1e-5 of polar_scale (see module
    # docstring) -- 1e-3 leaves two orders of magnitude of headroom over that
    # residual while still catching an O(1) polar error (a dropped or
    # double-counted or sign-flipped polar term) many orders of magnitude
    # larger than any leftover viscous noise.
    np.testing.assert_allclose(composed, polar_only, rtol=0, atol=1e-3 * polar_scale)


def test_composed_viscous_contribution_shrinks_with_mu():
    """Companion check: the viscous-minus-polar residual should shrink
    roughly linearly as mu shrinks (tau_cell/q_cell are linear in mu -- see
    set_tau_q_soa), not just happen to be small at one particular mu. Compares
    the residual at MU_NEGLIGIBLE against a ~100x-larger-but-still-small mu;
    a polar bug that injects a mu-independent error (e.g. a fixed sign flip
    losing a mu-independent-magnitude term) would fail to shrink here even if
    it happened to pass the single-mu comparison above."""
    block_small = _build_block(MU_NEGLIGIBLE)
    polar_small = _run_polar_only(_build_block(MU_NEGLIGIBLE))
    composed_small = _run_composed(block_small)
    residual_small = float(np.abs(composed_small - polar_small).max())

    mu_larger = 100.0 * MU_NEGLIGIBLE
    block_larger = _build_block(mu_larger)
    polar_larger = _run_polar_only(_build_block(mu_larger))
    composed_larger = _run_composed(block_larger)
    residual_larger = float(np.abs(composed_larger - polar_larger).max())

    # Loose factor (not exactly 100x): only checks the residual shrinks with
    # mu, not that it shrinks exactly proportionally (empirically it does,
    # to several digits -- see module docstring -- but the loose bound is
    # what actually matters for catching a mu-independent polar error).
    assert residual_small < 0.5 * residual_larger
