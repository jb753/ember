"""Regression test: fused viscous+polar force reduces to polar-only as mu -> 0.

Motivated by a real turbine-case regression traced to
``2381658745`` "Fuse polar source into set_visc_force and drop its separate
negation pass" (see ember-paper's
``turbine/run_final-divergence-findings.md``). That commit folded the polar
(radial-momentum) source into ``set_visc_force``'s final pass over
``fvisc``, replacing what used to be a separate ``set_polar_source`` call for
every viscous run (previously that call ran only on the inviscid branch --
see :meth:`ember.grid.Grid.update_sources`). The commit's own correctness
check compared the fused kernel's *combined* output against the old
viscous-alone-plus-separate-polar combination on a synthetic fixture; it did
not isolate the polar contribution on its own, so a sign or scale error
introduced specifically in how the fusion folds the two terms together could
pass that check yet still corrupt the physical case (most visible on the
rotor block, the one place in the LISA turbine with a large `Omega` and
hence a large polar source relative to the fused viscous term).

This test isolates exactly that: with viscosity taken to (numerically) zero,
the fused kernel's viscous contribution should vanish, leaving only the
polar term -- so the fused ``set_visc_force`` output must match a standalone
``set_polar_source`` call on the same flow field.

Note this test currently *passes* against the commit that introduced the
turbine regression: the polar-fusion arithmetic itself checks out correct in
this limit, which narrows rather than confirms the hypothesis -- the actual
divergence must trace to the *other* change bundled into the same commit,
the viscous face-difference accumulates' sign-convention rewrite
(low-minus-high -> high-minus-low), not the fusion. Worth keeping this test
regardless: it locks down a real correctness property of the fused kernel
that a future change to either ``set_visc_force`` or ``set_polar_source``
could still break.

The fixture is the same rotating, swirling, sheared single block used by
``test_viscous_phases_golden.py``, with `mu` parametrized down to a
negligible value instead of its production magnitude -- except the i/j
faces are frictionless (:class:`~ember.inviscid.InviscidPatch`) here rather
than that golden's no-slip walls. The wall function's ``Re = rho*V*d/mu``
term is a log-law formulation that approaches a finite, mu-*independent*
value as ``mu -> 0`` (physically correct -- high-Re wall shear does not
vanish), not zero, so a walled fixture cannot isolate the polar-fusion
question this test is after: driving mu to zero would leave a genuine,
non-shrinking wall-shear residual that has nothing to do with the fusion
bug. Frictionless faces sidestep the wall function entirely, leaving only
the direct laminar/mixing-length stress terms, both of which do vanish
as mu -> 0 (the mixing-length clamp ``visc_lim = 3000*mu`` in
``set_tau_q_soa`` collapses ``mu_turb`` right along with the molecular
term).
"""

import numpy as np

import ember.block
import ember.fortran
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
# set_tau_q_soa) collapse the fused kernel's viscous residual to ~1e-5 of
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
    # (here, negligible) viscous term it's fused with.
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


def _run_polar_only(block):
    """Standalone set_polar_source on a freshly zeroed F_body_nd."""
    fbody = block.F_body_nd
    fbody.flags.writeable = True
    fbody.fill(0.0)
    ember.fortran.set_polar_source(
        cons_cell=block.conserved_cell_nd,
        r=block.r_nd,
        p=block.P_nd,
        p_offset=block.P_offset_nd,
        vol=block.vol_nd,
        net_flow=fbody,
    )
    fbody.flags.writeable = False
    return np.array(fbody)


def _run_fused_viscous_and_polar(block):
    """The real viscous pair, exactly as Grid.update_sources runs it:
    ``set_tau_q_faces`` for the boundary shell from this block's own flow
    field, then ``set_visc_force``, which derives every interior tau/q itself
    and folds the polar source into its own final pass over fvisc."""
    viscous_util.fill_faces(block, PR_TURB)
    viscous_util.run_visc_force(block, PR_TURB)
    return np.array(block.F_body_nd)


def test_polar_source_is_nonzero():
    """Sanity check on the fixture itself: if this fires, the comparison
    below would pass vacuously (both sides ~0) without exercising anything."""
    block = _build_block(MU_NEGLIGIBLE)
    polar_only = _run_polar_only(block)
    assert np.abs(polar_only).max() > 1e-3


def test_fused_matches_polar_only_as_mu_vanishes():
    """The regression this guards against: a sign/scale error specific to how
    set_visc_force folds the polar source into its own pass, invisible to a
    check that only compares the *combined* viscous+polar output against the
    pre-fusion combination (both terms O(1), so a small relative error in
    the fold is easy to miss) but decisive once the viscous term is driven
    to negligible size relative to the polar term."""
    block_polar = _build_block(MU_NEGLIGIBLE)
    polar_only = _run_polar_only(block_polar)

    block_fused = _build_block(MU_NEGLIGIBLE)
    fused = _run_fused_viscous_and_polar(block_fused)

    polar_scale = float(np.abs(polar_only).max())
    assert polar_scale > 0.0  # see test_polar_source_is_nonzero

    # atol scaled to the polar term's own magnitude: at MU_NEGLIGIBLE the
    # residual viscous contribution measures ~1e-5 of polar_scale (see module
    # docstring) -- 1e-3 leaves two orders of magnitude of headroom over that
    # residual while still catching an O(1) fold error (a dropped or
    # double-counted or sign-flipped polar term) many orders of magnitude
    # larger than any leftover viscous noise.
    np.testing.assert_allclose(fused, polar_only, rtol=0, atol=1e-3 * polar_scale)


def test_fused_viscous_contribution_shrinks_with_mu():
    """Companion check: the viscous-minus-polar residual should shrink
    roughly linearly as mu shrinks (tau_cell/q_cell are linear in mu -- see
    set_tau_q_soa), not just happen to be small at one particular mu. Compares
    the residual at MU_NEGLIGIBLE against a ~100x-larger-but-still-small mu;
    a fold bug that injects a mu-independent error (e.g. a fixed sign flip
    losing a mu-independent-magnitude term) would fail to shrink here even if
    it happened to pass the single-mu comparison above."""
    block_small = _build_block(MU_NEGLIGIBLE)
    polar_small = _run_polar_only(_build_block(MU_NEGLIGIBLE))
    fused_small = _run_fused_viscous_and_polar(block_small)
    residual_small = float(np.abs(fused_small - polar_small).max())

    mu_larger = 100.0 * MU_NEGLIGIBLE
    block_larger = _build_block(mu_larger)
    polar_larger = _run_polar_only(_build_block(mu_larger))
    fused_larger = _run_fused_viscous_and_polar(block_larger)
    residual_larger = float(np.abs(fused_larger - polar_larger).max())

    # Loose factor (not exactly 100x): only checks the residual shrinks with
    # mu, not that it shrinks exactly proportionally (empirically it does,
    # to several digits -- see module docstring -- but the loose bound is
    # what actually matters for catching a mu-independent fold error).
    assert residual_small < 0.5 * residual_larger
