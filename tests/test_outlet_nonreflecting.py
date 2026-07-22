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
"""

import numpy as np
import pytest

from ember.convergence_history import ConvergenceHistory
from ember.grid import Grid
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
    """A scalar, a spanwise profile and a full patch array are all accepted."""
    block = make_block()
    patch = NonReflectingOutletPatch(i=-1)
    block.patches.append(patch)
    nspan = patch.shape[patch.span_dim]

    patch.set_P(P_MEAN)
    scalar = np.copy(patch.P_nd)
    assert scalar.shape == patch.shape
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
    grid.update_bconds()
    grid.apply_bconds()
    assert outlet._ref is not None and outlet._prim_prev is not None
