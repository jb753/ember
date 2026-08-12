"""The non-reflecting conditions on a boundary of any orientation.

The condition is written against a face-normal velocity, and used to insist its
face was a plane of constant x so that the velocity was ``Vx``. It no longer
does: the meridional velocity pair is rotated into the interface frame on the
way in and back out on the way out, which leaves the characteristic algebra
untouched and generalises the condition to any surface of revolution.

These tests drive that from the outside. The duct the fixtures build is laid
out in its own meridional frame and turned through ``chi`` -- axial at 0,
radially outward at 90, running backwards at 180, radially inward at 270,
conical in between -- with ``bow`` curving the end faces so the frame angle
varies from hub to tip rather than being one number for the face. Every case
below is the same physical duct in its own frame, so the condition should do
the same thing to all of them.

The sharpest statement of that, and the one most of these tests are built on,
is that the whole condition is *equivariant*: run in the interface frame it is
not merely similar between orientations but identical, to float32 round-off.
"""

import numpy as np
import pytest

from ember.patch import InletPatch
from nonreflecting_util import (
    FLUID,
    PATCH_KINDS,
    PITCH,
    VX_MEAN,
    attached,
    face_chic,
    face_prim,
    harmonic,
    make_block,
    seed_chic,
    turn,
)

# Axial, conical, radially outward, reversed, radially inward, and conical the
# other way. The first is the case every other test in the suite runs, and is
# here so a failure that is not about orientation shows up as one.
CHI = [0.0, 30.0, 90.0, 180.0, 270.0, -45.0]

# Enough curvature that the frame angle swings about +/-8.5 degrees from hub to
# tip, so a face-constant treatment of it would show.
BOW = 0.02

# The mismatch each kind is driven with, in duct coordinates.
TARGET_OFF = {
    "Vx": 110.0,
    "Vr": 0.0,
    "Vt": 70.0,
    "P": 1.02e5,
    "T": 305.0,
}

# The coefficients of the harmonic relation that is live on each kind. An inlet
# runs Giles Eq. 5.17 on its entering stations, an outlet Eq. 5.32 on its
# leaving ones; the other branch of each is the zeroed-harmonic fallback that a
# reversed station takes.
LIVE_COEFS = {
    "inlet": ("coef_local", "coef_hilbert"),
    "outlet": ("coef_t", "coef_t_hilbert", "coef_down", "coef_down_hilbert"),
}


@pytest.fixture(params=list(PATCH_KINDS))
def kind(request):
    return request.param


def _pitch_harmonic(patch, eps=2.0e-2):
    """A pitchwise-varying wave of unit shape on the patch's own pitch."""
    return eps * np.cos(2.0 * np.pi * patch.block_view.t / PITCH)


def _outgoing(kind):
    """The characteristic the interior owns, which is fixed by the geometry."""
    return 0 if kind == "inlet" else 1


def _driven(kind, chi, bow=0.0, n_cycle=3, seed=True, **kwargs):
    """A patch driven off its target by a few cycles, and its characteristic state.

    Seeded with an outgoing pitchwise harmonic so the non-reflecting relations
    have something to act on, not just the mean-mode solve. Pass ``seed=False``
    for a caller that wants to deposit its own wave afterwards.
    """
    _, patch = attached(kind, chi=chi, bow=bow, sigma=1.0, target=TARGET_OFF, **kwargs)
    patch.update_soln()
    if seed:
        wave = np.zeros(patch.shape + (5,), dtype=np.float32)
        wave[..., _outgoing(kind)] = _pitch_harmonic(patch)
        seed_chic(patch, wave)
    for _ in range(n_cycle):
        patch.advance()
        patch.apply()
    return patch, face_chic(patch)


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chi", CHI)
@pytest.mark.parametrize("bow", [0.0, BOW])
def test_attaches_at_any_orientation(kind, chi, bow):
    """Every orientation attaches, and the frame it builds is the face's own."""
    _, patch = attached(kind, chi=chi, bow=bow)

    # The class says which side the interior is on; the geometry is not
    # consulted to second-guess it, so this holds at every orientation.
    assert patch._sign_interior == PATCH_KINDS[kind][0]._sign_interior

    # The frame angle is the duct's, to within the curvature bow puts on it.
    # Taken as a signed difference wrapped onto (-180, 180], so the branch cut
    # at chi = 180 does not decide the answer.
    got = np.asarray(patch.chi_node).ravel()
    delta = got - np.radians(chi)
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    tol = np.radians(10.0) if bow else 1e-5
    assert np.abs(delta).max() <= tol

    # Curvature is the whole point of bow, so it had better have produced some.
    if bow:
        assert np.ptp(delta) > np.radians(5.0)


@pytest.mark.parametrize("chi", CHI)
def test_the_rotation_is_skipped_only_when_it_is_the_identity(kind, chi):
    """An axial face pays nothing for a generalisation it does not use."""
    _, patch = attached(kind, chi=chi)
    assert patch._rot_identity == (chi == 0.0)


@pytest.mark.parametrize("chi", CHI)
def test_frame_axis_runs_along_the_duct(kind, chi):
    """In the frame, the through-flow is the normal velocity and nothing else.

    Both kinds agree: the axis points downstream at an inlet because the
    interior is downstream of it, and at an outlet because the interior is
    upstream, so a uniform duct flow reads as a positive normal velocity on
    either.
    """
    _, patch = attached(kind, chi=chi)
    with patch._resolved():
        patch.set_block_avg()
        avg = patch.block_avg
        Vn = np.asarray(avg.Vx_nd).ravel()
        Vs = np.asarray(avg.Vr_nd).ravel()

    np.testing.assert_allclose(Vn, VX_MEAN / FLUID.V_ref, rtol=1e-5)
    np.testing.assert_allclose(Vs, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# The rotation window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chi", CHI)
@pytest.mark.parametrize("bow", [0.0, BOW])
def test_a_patch_at_its_own_target_leaves_the_face_alone(kind, chi, bow):
    """The fixed point is a fixed point at every orientation.

    Which is the sharpest test that the window closes properly: a face left
    rotated, rotated twice, or rotated back through the wrong angle would come
    out of this at a different velocity, and by a fraction of the through-flow
    rather than by a round-off.
    """
    block, patch = attached(kind, chi=chi, bow=bow)
    before = block.conserved.copy()

    patch.update_soln()
    patch.advance()
    patch.apply()

    assert patch._rot_depth == 0
    np.testing.assert_allclose(
        block.conserved, before, rtol=0.0, atol=1e-5 * np.abs(before).max()
    )


@pytest.mark.parametrize("chi", [0.0, 90.0])
def test_the_window_closes_when_the_body_raises(kind, chi):
    """A face left in interface coordinates would corrupt whatever read it next.

    Everything else on the block -- the interior march, the periodic seam, the
    residual -- works in machine coordinates, so an exception escaping the
    window must not take the rotation with it.
    """
    block, patch = attached(kind, chi=chi)
    patch.update_soln()
    before = block.conserved.copy()

    def boom():
        raise RuntimeError("kaboom")

    patch._recombine = boom
    with pytest.raises(RuntimeError, match="kaboom"):
        patch.apply()

    assert patch._rot_depth == 0
    np.testing.assert_array_equal(block.conserved, before)


def test_the_window_rotates_once_however_deeply_it_is_nested():
    """set_block_avg enters the window the condition is already inside."""
    _, patch = attached("inlet", chi=90.0)
    outer = face_prim(patch)

    with patch._resolved():
        inner = np.stack(
            (
                patch.block_view.rho_nd,
                patch.block_view.Vx_nd,
                patch.block_view.Vr_nd,
                patch.block_view.Vt_nd,
                patch.block_view.P_nd,
            ),
            axis=-1,
        )
        # Nested entry must not rotate a second time.
        with patch._resolved():
            patch.set_block_avg()
            again = np.stack(
                (
                    patch.block_view.rho_nd,
                    patch.block_view.Vx_nd,
                    patch.block_view.Vr_nd,
                    patch.block_view.Vt_nd,
                    patch.block_view.P_nd,
                ),
                axis=-1,
            )
        np.testing.assert_allclose(again, inner, rtol=1e-6)

    np.testing.assert_allclose(outer, inner, rtol=1e-6)


# ---------------------------------------------------------------------------
# The condition itself, which is the same condition at every orientation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chi", CHI[1:])
def test_the_harmonic_relation_is_the_same_relation_at_any_orientation(kind, chi):
    """Its coefficients are frame quantities, so they cannot depend on the angle.

    Every one is built from the normal and tangential Mach numbers of the
    pitchwise mean and the wave parameter, all of them read in the interface
    frame. Turning the duct turns the frame with it and leaves them alone --
    which is what it means for the algebra to be untouched by the
    generalisation.
    """
    patch0, _ = _driven(kind, 0.0)
    patch, _ = _driven(kind, chi)

    for key in LIVE_COEFS[kind]:
        np.testing.assert_allclose(
            np.asarray(patch._ref[key]),
            np.asarray(patch0._ref[key]),
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.parametrize("chi", CHI[1:])
@pytest.mark.parametrize("bow", [0.0, BOW])
def test_the_whole_condition_is_equivariant(kind, chi, bow):
    """Run in the frame, a turned duct is not merely similar to the axial one.

    The same wave seeded in the frame, advanced the same number of times,
    leaves the same characteristic state -- to float32 round-off through the
    two rotations, and no further. Nothing about the boundary condition sees
    the orientation.
    """
    _, chic0 = _driven(kind, 0.0, bow=bow)
    _, chic = _driven(kind, chi, bow=bow)

    np.testing.assert_allclose(chic, chic0, rtol=0.0, atol=1e-4 * np.abs(chic0).max())


@pytest.mark.parametrize("chi", CHI)
@pytest.mark.parametrize("bow", [0.0, BOW])
def test_the_relation_stays_live_at_any_orientation(kind, chi, bow):
    """The harmonics are absorbed, not zeroed, whichever way the face points.

    The condition zeroes the harmonics where the mean flow opposes the frame
    axis, which is the honest thing to do at a reversed station and would be a
    silent loss of the non-reflecting property everywhere else. So the branch
    that is live has to be the one matching the flow, at every orientation.
    """
    patch, _ = _driven(kind, chi, bow=bow)

    # The live branch was the one built, and the fallback branch's coefficients
    # were not even computed.
    assert set(LIVE_COEFS[kind]) <= set(patch._ref)
    other = "outlet" if kind == "inlet" else "inlet"
    assert not set(LIVE_COEFS[other]) & set(patch._ref)

    # And the stations are on the split that branch belongs to.
    assert patch._entering.all() == (kind == "inlet")


@pytest.mark.parametrize("chi", CHI)
def test_an_outgoing_harmonic_still_passes_through(kind, chi):
    """A wave reaching a turned boundary leaves through it rather than bouncing."""
    patch, _ = _driven(kind, chi, n_cycle=1, seed=False)
    out = _outgoing(kind)

    before = face_chic(patch)
    wave = np.zeros(patch.shape + (5,), dtype=np.float32)
    wave[..., out] = _pitch_harmonic(patch)
    seed_chic(patch, wave)
    patch.apply()
    after = face_chic(patch)

    # The outgoing component is read from the marched face every stage, so it
    # comes through; the incoming ones are this patch's own and do not move,
    # because nothing advanced the condition between the two reads.
    moved = np.abs(after[..., out] - before[..., out]).max()
    assert moved > 1e-2 * np.abs(wave[..., out]).max()

    incoming = [c for c in range(5) if c != out]
    np.testing.assert_allclose(
        after[..., incoming], before[..., incoming], rtol=0.0, atol=1e-3 * moved
    )


@pytest.mark.parametrize("chi", CHI)
def test_reversal_is_read_in_the_duct_frame(kind, chi):
    """Which way the flow runs is judged against the face, not against +x.

    The test reads the first interior layer, which is not part of the rotated
    face and stays in machine coordinates throughout, so it is the one place
    the two frames meet. Getting its projection wrong would leave a radial duct
    reporting every station reversed, or a reversed one reporting none.
    """
    forward = attached(kind, chi=chi)[1]
    forward.update_soln()
    assert forward._entering.all() == (kind == "inlet")

    # Reverse the duct-frame through-flow, which at chi = 90 is a purely radial
    # velocity and at chi = 180 an axial one of the opposite sign to the axial
    # case: nothing about the machine frame distinguishes them. The target is
    # left running forward, since it is the state a reversed station would be
    # driven toward rather than the one it is in.
    reversed_ = attached(kind, chi=chi, Vx=-10.0, target=TARGET_OFF)[1]
    reversed_.update_soln()
    assert reversed_._entering.all() == (kind == "outlet")


# ---------------------------------------------------------------------------
# Angles, which stay in the machine frame the user gives them in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chi", CHI)
def test_Beta_is_measured_from_the_machine_axis(chi):
    r"""tan(Beta) = Vr/Vx, whatever the face it is prescribed on.

    The condition works in the interface frame, where the same flow makes the
    angle ``Beta - chi`` with the frame axis, so the setter carries the offset.
    Checked at a genuine cross-flow, so an implementation that dropped the
    offset entirely would still have to get the duct angle right to pass.
    """
    Vn = 20.0
    block, patch = attached("inlet", chi=chi, Vr=Vn)

    # The angle the block's own flow makes with the axis, which is what the
    # fixture prescribed, and which is the duct angle plus the cross-flow's.
    Vx, Vr = turn(VX_MEAN, Vn, chi)
    Beta_machine = np.degrees(np.arctan2(Vr, Vx))
    assert np.abs(np.sin(np.radians(Beta_machine - chi))) == pytest.approx(
        np.sin(np.arctan2(Vn, VX_MEAN)), rel=1e-5
    )

    # Prescribed in machine coordinates, stored in face coordinates, and read
    # back off the face by the condition's own reader: the round trip has to
    # close, or the patch would not be at the fixed point the fixture built.
    face = patch._target_from_prim(face_prim(patch))
    np.testing.assert_allclose(
        np.asarray(patch._target[..., 3]).ravel(),
        np.asarray(patch._pitch_mean(face[3])).ravel(),
        rtol=1e-5,
        atol=1e-6,
    )

    # And it holds up as a fixed point: the machine-frame angle on the face is
    # still the one asked for after the condition has run.
    patch.update_soln()
    patch.advance()
    patch.apply()
    b = patch.block_view
    got = np.degrees(np.arctan2(np.asarray(b.Vr_nd), np.asarray(b.Vx_nd)))
    np.testing.assert_allclose(got, Beta_machine, rtol=0.0, atol=0.05)


@pytest.mark.parametrize("chi", CHI)
def test_Beta_refuses_an_angle_that_would_not_enter_the_face(chi):
    """The stored sine cannot tell an inflow from an outflow, so it is refused.

    The guard is that the flow must have a component along the frame axis. On
    an axial face that is the familiar |Beta| <= 90; this is the same condition
    stated against the face, which is what makes it mean anything at all on a
    radial or reversed one.
    """
    block = make_block(chi=chi)
    patch = InletPatch(i=0, label="inlet_nrbc")
    block.patches.append(patch)

    # Along the face rather than through it: no flow enters.
    with pytest.raises(ValueError, match="face normal"):
        patch.set_Beta(chi + 90.0 + 5.0)

    # Through it: accepted, and accepted however the angle is spelled.
    patch.set_Beta(chi)
    stored = np.copy(patch._target[..., 3])
    patch.set_Beta(chi + 360.0)
    np.testing.assert_allclose(patch._target[..., 3], stored, atol=1e-6)


@pytest.mark.parametrize("chi", CHI)
def test_Alpha_needs_no_offset(chi):
    """Yaw is measured against the meridional speed, which the rotation preserves.

    Vm = sqrt(Vn^2 + Vs^2) = sqrt(Vx^2 + Vr^2), so the yaw angle is the same
    number in either frame and the setter has nothing to correct.
    """
    patch0 = attached("inlet", chi=0.0)[1]
    patch = attached("inlet", chi=chi)[1]
    np.testing.assert_allclose(
        np.asarray(patch._target[..., 2]),
        np.asarray(patch0._target[..., 2]),
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Harmonic content, to show the pitch machinery is untouched by the rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chi", CHI[1:])
def test_the_pitch_transform_is_untouched_by_the_rotation(kind, chi):
    """The rotation is in the meridional plane, so it cannot reach the pitch.

    A surface of revolution has a pitch direction of pure theta by
    construction, which is why the Hilbert transform survives the
    generalisation unchanged rather than needing a canted counterpart.
    """
    patch0, chic0 = _driven(kind, 0.0)
    patch, chic = _driven(kind, chi)

    np.testing.assert_array_equal(patch._hilbert, patch0._hilbert)
    # Component by component, since the pitch mean is taken on patch-shaped
    # fields rather than on a stack of them.
    for c in range(5):
        want = harmonic(patch0, chic0[..., c])
        np.testing.assert_allclose(
            harmonic(patch, chic[..., c]),
            want,
            rtol=0.0,
            atol=1e-4 * max(np.abs(want).max(), np.abs(chic0).max()),
        )
