"""Tests for NonReflectingOutletPatch.

Module tested: ember.outlet_nonreflecting.NonReflectingOutletPatch

The machinery this patch shares with the inflow condition -- attachment, the
Hilbert transform, the mean-state guards, the relaxation, copy semantics -- is
tested in test_nonreflecting.py. What is left here is the outflow condition
itself.

Test cases:
- Target: what set_P accepts and refuses
- Physics: fixed point at the target, the mean-mode Newton step, the harmonic
  relation against an independent complex evaluation of Saxer Eq. 57, outgoing
  characteristics left untouched, zero pressure harmonic for pure acoustics, an
  analytic potential disturbance passing through, harmonics not disturbing the
  mean, convergence to the prescribed pressure
- Backflow: what set_backflow accepts and what is seeded when it is not called,
  the split switched at a reversed span station and the state that station is
  driven to, the exit pressure it stops imposing there, the node-level override
  and what it leaves alone, the hysteresis on release
"""

import numpy as np
import pytest

from ember.convergence_history import ConvergenceHistory
from ember.grid import Grid
from ember.outlet import calc_radial_equilibrium
from ember.patch import (
    InviscidPatch,
    NonReflectingInletPatch,
    NonReflectingOutletPatch,
    PeriodicPatch,
)
from nonreflecting_util import (
    FLUID,
    PITCH,
    P_MEAN,
    VT_MEAN,
    attached,
    face_chic,
    face_prim,
    harmonic,
    make_block,
    reference_state,
    seed_chic,
)


def _attached(**kwargs):
    """Block with a non-reflecting outlet patch on its i=-1 face."""
    return attached("outlet", **kwargs)


def _phase(patch, mode):
    """Pitchwise phase of a harmonic, over the patch face."""
    return 2.0 * np.pi * mode * patch.block_view.t / PITCH


def _seed_harmonic(patch, mode, amp_t=0.0, amp_down=0.0):
    """Seed outgoing vorticity and acoustic harmonics of complex amplitude amp.

    Each field is written as ``Re(amp * exp(i * phase))``, the convention the
    theory value in :func:`test_harmonic_relation_matches_theory` is evaluated
    with.
    """
    wave = np.zeros(patch.shape + (5,), dtype=np.float32)
    carrier = np.exp(1j * _phase(patch, mode))
    wave[..., 3] = (amp_t * carrier).real
    wave[..., 1] = (amp_down * carrier).real
    seed_chic(patch, wave)
    return wave


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


def test_set_P_accepts_any_broadcastable_shape():
    """A scalar, a spanwise profile and a full patch array are all accepted.

    Whatever the shape, what is stored is one value per span station: the
    pitchwise mean is all this condition ever imposes.
    """
    block = make_block()
    patch = NonReflectingOutletPatch(i=-1)
    block.patches.append(patch)
    nspan = patch.shape[patch.span_dim]

    patch.set_P(P_MEAN)
    scalar = np.copy(patch.P_nd)
    assert scalar.shape[patch.span_dim] == nspan
    assert scalar.shape[patch.pitch_dim] == 1
    np.testing.assert_allclose(scalar, P_MEAN / FLUID.P_ref, rtol=1e-6)

    patch.set_P(np.full(patch.shape, P_MEAN))
    np.testing.assert_allclose(patch.P_nd, scalar, rtol=1e-6)

    profile = P_MEAN * (1.0 + 0.05 * np.linspace(0.0, 1.0, nspan))
    patch.set_P(profile.reshape(1, nspan, 1))
    np.testing.assert_allclose(patch.P_nd[0, :, 0], profile / FLUID.P_ref, rtol=1e-6)
    # Uniform along the pitch, varying along the span.
    assert np.ptp(patch.P_nd, axis=patch.pitch_dim).max() < 1e-6


@pytest.mark.parametrize(
    "value, match",
    [
        (np.nan, "P must be finite"),
        (np.inf, "P must be finite"),
        (-1.0e5, "P must be positive"),
        (0.0, "P must be positive"),
    ],
)
def test_set_P_rejects_invalid(value, match):
    """Non-finite or non-positive pressures are refused."""
    block = make_block()
    patch = NonReflectingOutletPatch(i=-1)
    block.patches.append(patch)
    with pytest.raises(ValueError, match=match):
        patch.set_P(value)


def test_set_P_rejects_bad_shape():
    """A value that does not broadcast to the patch shape is refused."""
    block = make_block()
    patch = NonReflectingOutletPatch(i=-1)
    block.patches.append(patch)
    with pytest.raises(ValueError, match="broadcast"):
        patch.set_P(np.full(patch.shape + (3,), P_MEAN))


def test_set_P_before_attach_raises():
    """The target is stored nondimensionally, so a block is needed to set it."""
    patch = NonReflectingOutletPatch(i=-1)
    with pytest.raises(ValueError, match="not attached"):
        patch.set_P(P_MEAN)


# ---------------------------------------------------------------------------
# Spanwise adjustment
# ---------------------------------------------------------------------------


def _span_profile(patch, field):
    """Span profile of a patch-shaped field, pitch-averaged and squeezed."""
    return np.asarray(patch._pitch_mean(field)).squeeze()


def test_set_adjustment_stores_values():
    """set_adjustment records the flag and the relaxation factor."""
    _, patch = _attached()
    assert patch._adjustment == {}
    patch.set_adjustment(radial_equilibrium=True, rf=0.2)
    assert patch._adjustment == {"radial_equilibrium": True, "rf": 0.2}


def test_adjustment_and_nonscalar_P_are_incompatible():
    """A prescribed spanwise profile plus a derived one would double count."""
    block = make_block()
    patch = NonReflectingOutletPatch(i=-1)
    block.patches.append(patch)
    nspan = patch.shape[patch.span_dim]
    profile = np.full((1, nspan, 1), P_MEAN)

    patch.set_P(profile)
    with pytest.raises(ValueError, match="incompatible"):
        patch.set_adjustment()

    # ... and refused in the other order too.
    patch.set_P(P_MEAN)
    patch.set_adjustment()
    with pytest.raises(ValueError, match="incompatible"):
        patch.set_P(profile)


def test_update_target_without_adjustment_is_the_prescribed_level():
    """With no adjustment configured the target is exactly what set_P stored."""
    _, patch = _attached()
    patch.update_target()
    np.testing.assert_array_equal(patch.P_nd, patch._P_level_nd)


def test_target_carries_the_radial_equilibrium_profile():
    """At rf=1 the target is the prescribed level plus the integrated profile."""
    block, patch = _attached(Vt=VT_MEAN)
    patch.set_adjustment(radial_equilibrium=True, rf=1.0)
    patch.update_target()

    got = _span_profile(patch, patch.P_nd - patch._P_level_nd)
    expect = calc_radial_equilibrium(patch)
    assert np.abs(got - expect).max() < 1e-5 * np.abs(expect).max()
    # Anchored at the hub, rising toward the casing, and worth having: a few
    # percent of the prescribed level on this much swirl.
    assert got[0] == 0.0
    assert got[-1] > 0.01 * P_MEAN / FLUID.P_ref


def test_relaxation_approaches_the_profile_geometrically():
    """rf sets how fast the target moves onto the profile, not where it lands."""
    rf = 0.25
    _, patch = _attached(Vt=VT_MEAN)
    patch.set_adjustment(radial_equilibrium=True, rf=rf)

    # The first call seeds the history with the profile itself, so the target
    # starts where the flow is rather than crawling out from zero.
    patch.update_target()
    profile = calc_radial_equilibrium(patch)
    first = _span_profile(patch, patch.P_nd - patch._P_level_nd)
    np.testing.assert_allclose(first, profile, rtol=1e-5)

    # Halve the swirl behind the face; the target must walk toward the new
    # profile at the relaxation rate rather than jumping to it.
    Vt = patch.block.Vt.copy()
    Vt[-2] *= 0.5
    patch.block.set_Vt(Vt)
    target_profile = calc_radial_equilibrium(patch)

    errs = []
    for _ in range(10):
        patch.update_target()
        got = _span_profile(patch, patch.P_nd - patch._P_level_nd)
        errs.append(np.abs(got - target_profile).max())
    assert errs[0] / errs[1] == pytest.approx(1.0 / (1.0 - rf), rel=1e-3)
    assert errs[-1] < errs[0] * (1.0 - rf) ** 8


def test_apply_lands_on_the_adjusted_target():
    """The mean mode drives each span station to its adjusted pressure.

    The plumbing test for the whole feature: with the adjustment on, one
    sigma=1 visit must land the pitch-mean pressure on the sloped target, not
    on the uniform level that set_P prescribed.
    """
    _, patch = _attached(Vt=VT_MEAN)
    patch.set_adjustment(radial_equilibrium=True, rf=1.0)
    patch.update_target()
    patch.update_soln()
    patch.apply()

    got = _span_profile(patch, patch.block_view.P_nd)
    expect = _span_profile(patch, patch.P_nd)
    assert np.abs(got - expect).max() < 1e-5 * np.abs(expect).max()
    # Not vacuous: the target really does slope, so an implementation that
    # ignored the adjustment would fail here.
    assert np.ptp(expect) > 0.01 * expect.mean()


def test_reference_scale_change_drops_the_derived_profile():
    """Rescaling the block invalidates a target integrated over its old geometry."""
    block, patch = _attached(Vt=VT_MEAN)
    patch.set_adjustment(radial_equilibrium=True, rf=1.0)
    patch.update_target()
    assert np.ptp(_span_profile(patch, patch.P_nd)) > 0.0

    # Block.set_L_ref rescales x and r, so the radial integral behind the
    # profile no longer belongs to this geometry.
    block.set_L_ref(2.0 * block.L_ref)
    np.testing.assert_array_equal(patch.P_nd, patch._P_level_nd)
    assert patch._P_last_nd is None


def test_copy_carries_the_adjustment_and_drops_its_caches():
    """copy() keeps the configuration but not the solution-derived state."""
    _, patch = _attached(Vt=VT_MEAN)
    patch.set_adjustment(radial_equilibrium=True, rf=0.3)
    patch.update_target()
    clone = patch.copy()
    assert clone._adjustment == patch._adjustment
    np.testing.assert_array_equal(clone._P_raw, patch._P_raw)
    np.testing.assert_array_equal(clone._P_level_nd, patch._P_level_nd)
    assert clone._P_last_nd is None


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Vt, Vr", [(0.0, 0.0), (VT_MEAN, 0.0), (VT_MEAN, 25.0)])
def test_fixed_point(Vt, Vr):
    """A face already at the prescribed pressure is left untouched."""
    block, patch = _attached(Vt=Vt, Vr=Vr)
    before = block.conserved_nd.copy()
    scale = np.abs(before).max()
    patch.update_soln()
    patch.apply()
    assert np.abs(block.conserved_nd - before).max() < 1e-5 * scale
    # And still a fixed point on a second visit.
    patch.apply()
    assert np.abs(block.conserved_nd - before).max() < 1e-5 * scale


@pytest.mark.parametrize("ratio", [0.97, 1.03])
def test_mean_mode_lands_on_the_prescribed_pressure(ratio):
    """One mean-mode step sets the pitch-mean pressure exactly.

    Static pressure is an exactly linear function of the characteristics,
    dp/dc_up = 1/2 (Giles Eq. 5.29), so unlike the inflow condition the Newton
    step here is not an approximation: with sigma = 1 the mean must land on the
    target in one visit. A wrong factor in the step shows up immediately.
    """
    _, patch = _attached(target={"P": P_MEAN * ratio})
    patch.update_soln()
    patch.apply()
    P_target_nd = P_MEAN * ratio / FLUID.P_ref
    got = patch._pitch_mean(patch.block_view.P_nd)
    assert np.abs(got - P_target_nd).max() < 1e-5 * P_target_nd


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("Vt", [0.0, VT_MEAN, -VT_MEAN])
def test_harmonic_relation_matches_theory(mode, Vt):
    """Each harmonic satisfies the non-reflecting relation of Saxer Eq. 57.

    The reference is a direct complex evaluation of

        c_up = (2 M_n / (beta - M_t)) c_t - ((beta + M_t)/(beta - M_t)) c_down,
        beta = i sign(m) sqrt(1 - M^2),

    written per Fourier mode, against the patch's rationalised real-coefficient
    form with its Hilbert transform. Two independent implementations of the same
    equation, and the only place the four reference coefficients are checked.
    """
    eps = 2.0e-2
    amp_t = eps * (1.0 + 0.0j)
    amp_down = eps * (0.5 - 0.7j)

    _, patch = _attached(Vt=Vt)
    patch.update_soln()
    _seed_harmonic(patch, mode, amp_t=amp_t, amp_down=amp_down)
    patch.apply()

    avg = patch.block_avg
    Mn = patch._span_bcast(avg.Vx_nd / avg.a_nd)
    Mt = patch._span_bcast(avg.Vt_nd / avg.a_nd)
    # sign(m) = +1 for the positive mode written by _seed_harmonic; the
    # negative mode is its complex conjugate and is carried by taking the real
    # part below.
    beta = 1j * np.sqrt(1.0 - Mn**2 - Mt**2)
    amp_up = (2.0 * Mn * amp_t - (beta + Mt) * amp_down) / (beta - Mt)
    expect = (amp_up * np.exp(1j * _phase(patch, mode))).real

    got = harmonic(patch, face_chic(patch)[..., 0])
    assert np.abs(got - expect).max() < 5e-3 * np.abs(expect).max()


@pytest.mark.parametrize("idx", [1, 2, 3, 4])
def test_outgoing_characteristics_are_untouched(idx):
    """Only the upstream-running pressure characteristic is written.

    Entropy, both vorticity waves and the downstream-running pressure wave are
    outgoing at an outflow plane, so whatever the interior march deposited in
    them must survive the boundary condition untouched.
    """
    eps = 2.0e-2
    _, patch = _attached()
    patch.update_soln()

    wave = np.zeros(patch.shape + (5,), dtype=np.float32)
    wave[..., idx] = eps * np.cos(_phase(patch, 1))
    seed_chic(patch, wave)

    before = face_chic(patch)
    patch.apply()
    after = face_chic(patch)
    assert np.abs(after[..., 1:] - before[..., 1:]).max() < 1e-3 * eps
    # ... and the incoming one did respond, so this is not a vacuous test.
    if idx in (1, 3):
        assert np.abs(after[..., 0] - before[..., 0]).max() > 1e-2 * eps


def test_acoustic_harmonic_leaves_no_pressure_without_swirl():
    """Without swirl a pure outgoing acoustic harmonic exits at zero pressure.

    With M_t = 0 and no vorticity the relation collapses to c_up = -c_down, so
    the harmonic pressure perturbation p' = (c_up + c_down)/2 must vanish while
    the axial velocity perturbation survives.
    """
    eps = 2.0e-2
    _, patch = _attached(Vt=0.0)
    patch.update_soln()
    _seed_harmonic(patch, 1, amp_down=eps)
    b = patch.block_view
    dP_before = np.abs(harmonic(patch, b.P_nd)).max()
    dVx_before = np.abs(harmonic(patch, b.Vx_nd)).max()

    patch.apply()

    assert np.abs(harmonic(patch, b.P_nd)).max() < 1e-2 * dP_before
    assert np.abs(harmonic(patch, b.Vx_nd)).max() > 1.5 * dVx_before


@pytest.mark.parametrize("mode", [1, 2])
def test_analytic_potential_disturbance_is_a_fixed_point(mode):
    r"""An admissible downstream potential field passes through unchanged.

    Steady linearised potential flow downstream of a blade row satisfies
    :math:`(1-M^2)\phi_{xx} + \phi_{yy} = 0`, so
    :math:`\phi = A e^{-\mu x}\cos(ly)` with :math:`\mu = l/\sqrt{1-M^2}` decays
    downstream and is exactly what a non-reflecting outlet must admit without
    distortion. It carries no entropy and no pitchwise-mean pressure change, so
    every condition the patch imposes is already satisfied and one apply must be
    a no-op.

    The reference here is an independent solution of the governing equations
    rather than the patch's own formula, so this pins down the *physics* -- in
    particular the sign of the wave parameter, which is ambiguous in Giles' text
    between the continuous wavenumber and the discrete mode index. With the sign
    reversed the patch inverts the tangential velocity of this field instead of
    passing it.
    """
    _, patch = _attached(Vt=0.0)
    patch.update_soln()
    b = patch.block_view
    avg = patch.block_avg

    rho = patch._span_bcast(np.asarray(avg.rho))
    a = patch._span_bcast(np.asarray(avg.a))
    Vx = patch._span_bcast(np.asarray(avg.Vx))
    P = patch._span_bcast(np.asarray(avg.P))

    wavenumber = 2.0 * np.pi * mode / (b.r * PITCH)
    decay = wavenumber / np.sqrt(1.0 - (Vx / a) ** 2)
    phase = _phase(patch, mode)

    # Same field as the inlet's admissible mode with the decay direction
    # reversed, which flips the sign of the axial velocity and pressure
    # perturbations but not the tangential one. Scaled so the pressure
    # perturbation is 1% of the mean.
    amp = 0.01 * P / (rho * Vx * decay)
    dP = rho * Vx * decay * amp * np.cos(phase)
    b.set_P_rho(P + dP, rho + dP / a**2)
    b.set_Vx(Vx - decay * amp * np.cos(phase))
    b.set_Vr(np.zeros(patch.shape))
    b.set_Vt(-amp * wavenumber * np.sin(phase))

    before = face_prim(patch)
    disturbance = np.abs(before - patch._ref["prim"]).max()
    patch.apply()
    change = np.abs(face_prim(patch) - before).max()

    assert change < 0.05 * disturbance, (
        f"admissible potential field was altered by {change / disturbance:.1%}"
    )


def test_harmonics_do_not_disturb_the_mean():
    """The harmonic correction is pitchwise mean-free.

    The mean and harmonic parts of the incoming characteristic are computed
    separately and added, so a harmonic term leaking into the pitch mean would
    double count against the prescribed pressure. Several harmonics at once,
    with swirl, so no single mode can cancel the error by symmetry.
    """
    _, patch = _attached()
    patch.update_soln()
    b = patch.block_view
    rng = np.random.default_rng(2)
    wave = np.zeros(patch.shape + (5,), dtype=np.float32)
    for mode in (1, 2, 3):
        wave[..., 1] += 2.0e-2 * rng.uniform(0.5, 1.5) * np.cos(_phase(patch, mode))
        wave[..., 3] += 2.0e-2 * rng.uniform(0.5, 1.5) * np.sin(_phase(patch, mode))
    seed_chic(patch, wave)

    P_target_nd = P_MEAN / FLUID.P_ref
    patch.apply()
    assert np.abs(harmonic(patch, b.P_nd)).max() > 1e-4, "no harmonic content left"
    got = patch._pitch_mean(b.P_nd)
    assert np.abs(got - P_target_nd).max() < 1e-4 * P_target_nd


def test_converges_to_prescribed_pressure():
    """Iterating the boundary condition drives the mean pressure to the target.

    The interior is frozen, so this isolates the fixed-point structure of the
    condition itself: the Newton step plus the sigma relaxation.
    """
    sigma = 0.25
    _, patch = _attached(sigma=sigma, P=P_MEAN * 1.1, target={})
    P_target_nd = P_MEAN / FLUID.P_ref

    b = patch.block_view
    errs = []
    for _ in range(100):
        patch.update_soln()
        patch.apply()
        errs.append(float(np.abs(patch._pitch_mean(b.P_nd) - P_target_nd).max()))

    assert errs[-1] < 1e-6 * P_target_nd, f"did not converge: {errs[-1]}"
    # Geometric decay at the relaxation rate over the early iterations, where
    # the residual is still well above float32 noise.
    assert errs[10] < errs[0] * (1.0 - sigma) ** 8


def test_nonuniform_pitch_end_to_end():
    """A stretched pitchwise mesh still absorbs, if less exactly.

    The Hilbert transform is a quadrature on a non-uniform pitch, so the
    harmonic relation is only approximate there; the check is that the boundary
    still lands close to the theory rather than reflecting.
    """
    eps = 2.0e-2
    amp_t = eps * (1.0 + 0.0j)
    amp_down = eps * (0.5 - 0.7j)

    _, patch = _attached(npitch=33, stretch=0.4)
    patch.update_soln()
    _seed_harmonic(patch, 1, amp_t=amp_t, amp_down=amp_down)
    patch.apply()

    avg = patch.block_avg
    Mn = patch._span_bcast(avg.Vx_nd / avg.a_nd)
    Mt = patch._span_bcast(avg.Vt_nd / avg.a_nd)
    beta = 1j * np.sqrt(1.0 - Mn**2 - Mt**2)
    amp_up = (2.0 * Mn * amp_t - (beta + Mt) * amp_down) / (beta - Mt)
    expect = (amp_up * np.exp(1j * _phase(patch, 1))).real

    got = harmonic(patch, face_chic(patch)[..., 0])
    assert np.abs(got - expect).max() < 0.05 * np.abs(expect).max()


# ---------------------------------------------------------------------------
# Solver wiring
# ---------------------------------------------------------------------------


def test_grid_closed_by_this_patch_alone():
    """The grid-level bookkeeping works with no OutletPatch anywhere.

    Several consumers reach for ``grid.patches.outlet[0]`` or test for an
    ``OutletPatch`` by type to find the exit plane; each one has to know about
    this patch too, or a duct closed by it cannot be marched at all.
    """
    block = make_block(ni=9, nspan=5, npitch=9)
    inlet = NonReflectingInletPatch(i=0)
    outlet = NonReflectingOutletPatch(i=-1)
    block.patches.extend(
        [
            inlet,
            outlet,
            PeriodicPatch(k=0),
            PeriodicPatch(k=-1),
            InviscidPatch(j=0),
            InviscidPatch(j=-1),
        ]
    )
    state = reference_state()
    inlet.set_Po_To(float(state.Po), float(state.To))
    inlet.set_Alpha(float(state.Alpha))
    inlet.set_Beta(float(state.Beta))
    outlet.set_P(P_MEAN)

    grid = Grid([block])
    grid.set_L_ref(0.4)
    grid.calculate_wdist()
    grid.connectivity.periodic.pair()

    # The exit measurement station is found by type.
    up_idx, dn_idx = grid.row_station_bid_pid[0]
    assert up_idx and dn_idx

    # Convergence monitors work with no throttle to report on.
    step = grid.get_convergence()
    assert len(step.mdot) == 2
    assert step.mdot_target == 0.0 and step.P_throttle == 0.0
    assert ConvergenceHistory.from_grid(1, grid) is not None

    # And a full boundary-condition cycle runs: the patch is visited by both
    # update_bconds and apply_bconds, and asked only for what it implements.
    outlet.set_adjustment(radial_equilibrium=True, rf=1.0)
    grid.update_bconds()
    grid.apply_bconds()
    assert outlet._ref is not None and outlet._prim_prev is not None
    # update_bconds reached update_target, so the spanwise profile is in place.
    assert not np.array_equal(outlet.P_nd, outlet._P_level_nd)
    assert np.ptp(_span_profile(outlet, outlet.P_nd)) > 0.0

    # A frozen update leaves the target alone, so an averaging window sees a
    # fixed boundary; an unfrozen one picks the new profile up.
    before = np.copy(outlet.P_nd)
    Vt = block.Vt.copy()
    Vt[-2] *= 0.5
    block.set_Vt(Vt)
    grid.update_bconds(freeze=True)
    np.testing.assert_array_equal(outlet.P_nd, before)
    grid.update_bconds()
    assert np.abs(outlet.P_nd - before).max() > 0.0


# ---------------------------------------------------------------------------
# Backflow
# ---------------------------------------------------------------------------

# Reversed flow returns with some swirl and a little more entropy than the
# through-flow, so every prescribed quantity differs from the block state and
# an imposed node is told apart from an untouched one by any of them.
BACK_VR = 5.0
BACK_VT = 30.0


def _snapshot():
    """Dimensional ``(ho, s, Vr, Vt)`` to hand to set_backflow."""
    state = reference_state(Vx=0.0, Vr=BACK_VR, Vt=BACK_VT, T=320.0)
    return float(state.ho), float(state.s), BACK_VR, BACK_VT


def _backflow_block(
    rev_span=(), rev_node=(), Vx_rev=-20.0, i_rev=slice(None), **kwargs
):
    """Block with the axial velocity reversed over span stations or single nodes.

    ``rev_span`` reverses every node of those span stations, so the pitchwise
    mean reverses with them and the station is one the characteristic split has
    to be switched at, over the axial extent ``i_rev`` -- which wants narrowing
    to the downstream end if the block carries an inflow condition too, since
    that one has no handling to reach for. ``rev_node`` reverses single
    ``(span, pitch)`` nodes of the first interior layer alone, which leaves the
    mean of their station forward and is the node-level case instead.

    Indices are given as ``(span, pitch)`` whichever block axis each is on, so a
    caller reads the same for either ``span_dim``.
    """
    block = make_block(**kwargs)
    span_dim = kwargs.get("span_dim", 1)
    pitch_dim = 3 - span_dim
    Vx = block.Vx.copy()
    for j in rev_span:
        idx = [i_rev, slice(None), slice(None)]
        idx[span_dim] = j
        Vx[tuple(idx)] = Vx_rev
    for j, k in rev_node:
        idx = [-2, None, None]
        idx[span_dim] = j
        idx[pitch_dim] = k
        Vx[tuple(idx)] = Vx_rev
    block.set_Vx(Vx)
    return block


def _backflow_patch(block, P=P_MEAN, sigma=1.0, backflow=True):
    """Attach an outlet to the i=-1 face, with reversed-flow handling by default."""
    patch = NonReflectingOutletPatch(i=-1, label="outlet_nrbc")
    block.patches.append(patch)
    patch.set_P(P)
    patch.sigma = sigma
    if backflow:
        patch.set_backflow(*_snapshot())
    return patch


def _backflow_state(patch, j=0):
    """The four prescribed backflow quantities at one span station, as floats."""
    return [float(np.asarray(row).ravel()[j]) for row in patch._backflow()]


def test_set_backflow_stores_the_state_nondimensionally():
    """The four quantities are stored in the units the residuals are taken in."""
    _, patch = _attached()
    ho, s, Vr, Vt = _snapshot()
    patch.set_backflow(ho, s, Vr, Vt)

    expect = (ho / FLUID.u_ref, s / FLUID.Rgas_ref, Vr / FLUID.V_ref, Vt / FLUID.V_ref)
    np.testing.assert_allclose(_backflow_state(patch), expect, rtol=1e-6)
    assert patch._target_set[0:4].all()


def test_set_backflow_accepts_a_spanwise_profile():
    """A profile is allowed, since the seed it replaces is one; the pitch mean is kept."""
    _, patch = _attached()
    nspan = patch.shape[patch.span_dim]
    ho, s, Vr, _ = _snapshot()
    Vt = BACK_VT * np.linspace(0.5, 1.5, nspan).reshape(
        [nspan if d == patch.span_dim else 1 for d in range(3)]
    )
    patch.set_backflow(ho, s, Vr, Vt)

    got = np.asarray(patch._backflow()[3]).ravel()
    np.testing.assert_allclose(got, Vt.ravel() / FLUID.V_ref, rtol=1e-6)


def test_backflow_seeds_from_the_exit_plane_when_not_set():
    """Left alone, the four rows are taken from the face once and then frozen."""
    block = _backflow_block(rev_span=(2,))
    patch = _backflow_patch(block, backflow=False)
    assert not patch._target_set[0:4].any()
    patch.update_soln()

    # Station 0 still runs forward, so its seed is the block state it was built
    # with rather than anything the reversed station carries.
    assert patch._target_set[0:4].all()
    b = patch.block_view
    expect = [_span_profile(patch, f)[0] for f in (b.ho_nd, b.s_nd, b.Vr_nd, b.Vt_nd)]
    np.testing.assert_allclose(_backflow_state(patch, 0), expect, rtol=1e-5)

    # Frozen: a later step does not re-derive it from a face this patch has
    # since been writing, which would leave the residual identically zero.
    seed = np.copy(patch._target[..., 0:4])
    Vt = block.Vt.copy()
    Vt *= 2.0
    block.set_Vt(Vt)
    patch.update_soln()
    np.testing.assert_array_equal(patch._target[..., 0:4], seed)


def test_copy_carries_the_backflow_state():
    """The prescribed state survives a copy; what is derived from the solution does not."""
    block = _backflow_block(rev_span=(2,))
    patch = _backflow_patch(block)
    patch.update_soln()
    clone = patch.copy()

    np.testing.assert_array_equal(clone._target, patch._target)
    np.testing.assert_array_equal(clone._target_set, patch._target_set)
    assert clone._entering is None
    assert clone._rho_nd_soln is None


@pytest.mark.parametrize("span_dim", [1, 2])
def test_reversed_mean_is_carried_without_configuring_anything(span_dim):
    """Reversal never raises here: the state a reversed station needs is always there."""
    block = _backflow_block(rev_span=(2,), span_dim=span_dim)
    patch = _backflow_patch(block, backflow=False)
    patch.update_soln()
    assert patch._entering[2]
    assert not patch._entering[[0, 1, 3, 4, 5, 6]].any()


@pytest.mark.parametrize("span_dim", [1, 2])
def test_reversed_station_switches_the_characteristic_split(span_dim):
    """Only the downstream-running pressure wave still leaves a reversed station."""
    block = _backflow_block(rev_span=(2,), span_dim=span_dim)
    patch = _backflow_patch(block)
    patch.update_soln()

    mask = np.broadcast_to(patch._mask_out, patch.shape + (5,))
    at = [0, 0, 0]
    at[patch.span_dim] = 2
    np.testing.assert_array_equal(mask[tuple(at)], [False, True, False, False, False])
    at[patch.span_dim] = 0
    np.testing.assert_array_equal(mask[tuple(at)], [False, True, True, True, True])


@pytest.mark.parametrize("span_dim", [1, 2])
def test_reversed_station_converges_to_the_prescribed_state(span_dim):
    """The four quantities set_backflow prescribes are what the station is driven to."""
    block = _backflow_block(rev_span=(2,), span_dim=span_dim)
    patch = _backflow_patch(block, sigma=0.5)
    for _ in range(80):
        patch.update_soln()
        patch.apply()

    b = patch.block_view
    got = [_span_profile(patch, f)[2] for f in (b.ho_nd, b.s_nd, b.Vr_nd, b.Vt_nd)]
    np.testing.assert_allclose(got, _backflow_state(patch, 2), rtol=2e-3)
    # And it really is flow coming back in through the face.
    assert _span_profile(patch, b.Vx_nd)[2] < 0.0


def test_reversed_station_does_not_impose_the_exit_pressure():
    """Pressure is carried by the one wave still leaving, so it is not prescribed."""
    block = _backflow_block(rev_span=(2,))
    patch = _backflow_patch(block, P=1.2 * P_MEAN, sigma=0.5)
    for _ in range(80):
        patch.update_soln()
        patch.apply()

    P_face = _span_profile(patch, patch.block_view.P_nd)
    P_target = float(patch.P_nd.ravel()[0])
    # A forward station sits on the prescribed level.
    assert abs(P_face[0] - P_target) < 1e-3 * P_target
    # The reversed one is left where its own flow puts it, near the interior
    # level the calculation started from rather than the prescribed one.
    assert abs(P_face[2] - P_target) > 0.1 * P_target


def test_nodal_backflow_falls_back_to_the_seeded_state():
    """With nothing prescribed the node is imposed with the seed, not left alone."""
    block = _backflow_block(rev_node=((3, 5),))
    patch = _backflow_patch(block, backflow=False)
    patch.update_soln()
    patch.apply()

    assert not patch._entering.any()
    b = patch.block_view
    # The seed is the exit plane's own mean, so what lands on the node is that
    # and emphatically not the state set_backflow would have prescribed.
    assert float(b.Vt_nd[0, 3, 5]) == pytest.approx(
        _backflow_state(patch, 3)[3], rel=1e-5
    )
    assert float(b.Vt_nd[0, 3, 5]) != pytest.approx(BACK_VT / FLUID.V_ref, rel=1e-6)


def test_nodal_backflow_imposes_the_prescribed_state_on_reversed_nodes():
    """A node the interior is pushing flow in through is treated as an inlet."""
    block = _backflow_block(rev_node=((3, 5),))
    patch = _backflow_patch(block)
    patch.update_soln()
    patch.apply()

    assert not patch._entering.any()
    b = patch.block_view
    ho_snap, s_snap, Vr_snap, Vt_snap = _backflow_state(patch, 3)
    assert float(b.ho_nd[0, 3, 5]) == pytest.approx(float(ho_snap), rel=1e-5)
    assert float(b.s_nd[0, 3, 5]) == pytest.approx(float(s_snap), rel=1e-5)
    assert float(b.Vr_nd[0, 3, 5]) == pytest.approx(float(Vr_snap), rel=1e-5)
    assert float(b.Vt_nd[0, 3, 5]) == pytest.approx(float(Vt_snap), rel=1e-5)
    # Reversed, which the primitive write can express directly: unlike the
    # reflecting outlet there is no sign to flip back afterwards.
    assert float(b.Vx_nd[0, 3, 5]) < 0.0


def test_nodal_backflow_leaves_the_rest_of_the_face_untouched():
    """Only the nodes the override flags depend on what the backflow state is.

    The control is the same case with that state left to seed itself, which
    imposes something different at the same node and nothing anywhere else.
    """
    node = (3, 5)
    plain = _backflow_patch(_backflow_block(rev_node=(node,)), backflow=False)
    imposed = _backflow_patch(_backflow_block(rev_node=(node,)))
    for patch in (plain, imposed):
        patch.update_soln()
        patch.apply()

    got = imposed.block_view.conserved_nd.copy()
    expect = plain.block_view.conserved_nd.copy()
    j, k = node
    assert not np.array_equal(got[0, j, k], expect[0, j, k])
    got[0, j, k] = expect[0, j, k]
    np.testing.assert_array_equal(got, expect)


def test_nodal_backflow_stays_out_of_the_carried_state():
    """The override changes what reaches the block, not what the solve carries."""
    node = (3, 5)
    plain = _backflow_patch(_backflow_block(rev_node=(node,)), backflow=False)
    imposed = _backflow_patch(_backflow_block(rev_node=(node,)))
    for patch in (plain, imposed):
        patch.update_soln()
        patch.apply()

    np.testing.assert_array_equal(imposed._prim_prev, plain._prim_prev)
    # Which is only meaningful because the block did diverge between the two.
    j, k = node
    assert imposed._prim_prev[0, j, k, 1] > 0.0
    assert float(imposed.block_view.Vx_nd[0, j, k]) < 0.0


def test_nodal_backflow_defers_to_a_reversed_station():
    """A station the characteristic solve owns is not also treated node by node."""
    block = _backflow_block(rev_span=(2,), rev_node=((3, 5),))
    patch = _backflow_patch(block, sigma=0.5)
    patch.update_soln()
    patch.apply()

    b = patch.block_view
    Vr_snap = _backflow_state(patch, 3)[2]
    # The node-level case is imposed outright, so it lands on the snapshot.
    assert float(b.Vr_nd[0, 3, 5]) == pytest.approx(Vr_snap, rel=1e-5)
    # The reversed station is stepped toward it under-relaxed instead, so after
    # one stage it is on the way rather than there.
    Vr_station = float(b.Vr_nd[0, 2, 5])
    assert 0.0 < Vr_station < 0.9 * Vr_snap


def test_reversed_station_release_is_hysteretic():
    """A station hovering about zero settles into one split rather than alternating."""
    block = _backflow_block(rev_span=(2,))
    patch = _backflow_patch(block)
    patch.update_soln()
    assert patch._entering[2]

    # Forward again, but only just: held, so the split cannot chatter.
    Vx = block.Vx.copy()
    Vx[:, 2, :] = 3.0
    block.set_Vx(Vx)
    patch.update_soln()
    assert patch._entering[2]

    # Clear of the threshold: released.
    Vx[:, 2, :] = 50.0
    block.set_Vx(Vx)
    patch.update_soln()
    assert not patch._entering[2]


def test_backflow_runs_through_the_solver_loop():
    """Both reversed paths are driven by the grid, with no special casing anywhere.

    The station reversal is confined to the last two axial planes so the
    inflow condition at the other end of the block never sees it; that one has
    no handling to reach for and stops, which is the point of the guard.
    """
    block = _backflow_block(
        rev_span=(3,),
        rev_node=((1, 4),),
        i_rev=slice(-2, None),
        ni=9,
        nspan=7,
        npitch=9,
    )
    inlet = NonReflectingInletPatch(i=0)
    outlet = NonReflectingOutletPatch(i=-1)
    block.patches.extend(
        [
            inlet,
            outlet,
            PeriodicPatch(k=0),
            PeriodicPatch(k=-1),
            InviscidPatch(j=0),
            InviscidPatch(j=-1),
        ]
    )
    state = reference_state()
    inlet.set_Po_To(float(state.Po), float(state.To))
    inlet.set_Alpha(float(state.Alpha))
    inlet.set_Beta(float(state.Beta))
    outlet.set_P(P_MEAN)
    outlet.set_backflow(*_snapshot())

    grid = Grid([block])
    grid.set_L_ref(0.4)
    grid.calculate_wdist()
    grid.connectivity.periodic.pair()

    grid.update_bconds()
    grid.apply_bconds()

    # update_bconds reached the new update_soln, so the relaxation is anchored.
    assert outlet._rho_nd_soln is not None
    b = outlet.block_view

    # The reversed station is carried by the characteristic solve.
    assert outlet._entering[3]
    assert _span_profile(outlet, b.Vx_nd)[3] < 0.0

    # The lone reversed node, at a station whose mean still runs forward, is
    # carried by the override instead.
    assert not outlet._entering[1]
    assert float(b.Vt_nd[0, 1, 4]) == pytest.approx(
        _backflow_state(outlet, 1)[3], rel=1e-5
    )
