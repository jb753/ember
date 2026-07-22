"""Tests for NonReflectingInletPatch.

Module tested: ember.inlet_nonreflecting.NonReflectingInletPatch

The machinery this patch shares with the outflow condition -- attachment, the
Hilbert transform, the mean-state guards, the relaxation, copy semantics -- is
tested in test_nonreflecting.py. What is left here is the inflow condition
itself.

Test cases:
- Targets: validation, independence, the Po/To and ho/s routes agreeing
- Reversed stations: what set_backflow_P accepts and what is seeded without it,
  the split switched at a station the flow leaves, the pressure it is driven to
  and the inflow state it stops imposing there
- Physics: fixed point at the target, absorption of a mean acoustic wave, the
  mean Newton step, harmonic reflection coefficient against the non-reflecting
  theory, an analytic potential disturbance passing through, pitch uniformity
  of entropy and stagnation enthalpy, convergence to the prescribed state
"""

import numpy as np
import pytest

from ember.patch import NonReflectingInletPatch
from nonreflecting_util import (
    FLUID,
    PITCH,
    P_MEAN,
    T_MEAN,
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
    """Block with a non-reflecting inlet patch on its i=0 face."""
    return attached("inlet", **kwargs)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setter, args, match",
    [
        ("set_ho_s", (np.nan, 10.0), "ho must be finite"),
        ("set_ho_s", (3.0e5, np.inf), "s must be finite"),
        ("set_Alpha", (90.0,), "Alpha"),
        ("set_Beta", (91.0,), "Beta"),
        ("set_Po_To", (np.nan, 300.0), "Po must be finite"),
        ("set_Po_To", (1.0e5, -300.0), "To must be positive"),
    ],
)
def test_set_targets_rejects_invalid(setter, args, match):
    """Non-finite or out-of-range boundary condition values are refused."""
    block = make_block()
    patch = NonReflectingInletPatch(i=0)
    block.patches.append(patch)
    with pytest.raises(ValueError, match=match):
        getattr(patch, setter)(*args)


def test_set_targets_rejects_bad_shape():
    """Values that do not broadcast to the patch shape are refused."""
    block = make_block()
    patch = NonReflectingInletPatch(i=0)
    block.patches.append(patch)
    with pytest.raises(ValueError, match="broadcast"):
        patch.set_ho_s(np.ones(patch.shape + (3,)), 10.0)


def test_set_targets_are_independent():
    """Each setter leaves the targets owned by the others alone."""
    block = make_block()
    patch = NonReflectingInletPatch(i=0)
    block.patches.append(patch)
    patch.set_ho_s(3.0e5, 10.0)
    patch.set_Alpha(20.0)
    patch.set_Beta(0.0)
    ho_nd = np.copy(patch.ho_nd)
    patch.set_Alpha(30.0)
    np.testing.assert_allclose(patch.tanAlpha, np.tan(np.radians(30.0)), rtol=1e-6)
    np.testing.assert_array_equal(patch.ho_nd, ho_nd)


def test_set_Po_To_matches_set_ho_s():
    """Prescribing a stagnation state agrees with prescribing ho and s."""
    block_a, block_b = make_block(), make_block()
    patch_a = NonReflectingInletPatch(i=0)
    patch_b = NonReflectingInletPatch(i=0)
    block_a.patches.append(patch_a)
    block_b.patches.append(patch_b)

    state = reference_state()
    patch_a.set_Po_To(float(state.Po), float(state.To))
    patch_b.set_ho_s(float(state.ho), float(state.s))

    np.testing.assert_allclose(patch_a.ho_nd, patch_b.ho_nd, rtol=1e-6)
    np.testing.assert_allclose(patch_a.s_nd, patch_b.s_nd, rtol=1e-6, atol=1e-6)


def test_set_targets_before_attach_raises():
    """The targets are stored nondimensionally, so a block is needed to set them."""
    patch = NonReflectingInletPatch(i=0)
    with pytest.raises(ValueError, match="not attached"):
        patch.set_ho_s(3.0e5, 10.0)


def test_apply_names_the_missing_setters():
    """A partly prescribed inflow state reports exactly what is still missing."""
    block = make_block()
    patch = NonReflectingInletPatch(i=0)
    block.patches.append(patch)
    patch.set_Alpha(0.0)
    with pytest.raises(ValueError, match="set_ho_s or set_Po_To, set_Beta"):
        patch.apply()


# ---------------------------------------------------------------------------
# Reversed stations
# ---------------------------------------------------------------------------


def _reversed_block(rev_span=(2,), Vx_rev=-100.0, **kwargs):
    """Block whose inflow face has one span station running backwards out of it.

    Reversed over the whole axial extent, so the face and the interior layer
    behind it agree; a reversal only one plane deep would be released as soon as
    the condition raised the pressure enough to push the face velocity back
    through zero, which is the hysteresis working rather than a station to test
    against. Reversed hard for the same reason: the acoustic relation ties a
    rise in static pressure to a rise in axial velocity, so the station has to
    start far enough from zero to stay the other side of it.
    """
    block = make_block(**kwargs)
    span_dim = kwargs.get("span_dim", 1)
    Vx = block.Vx.copy()
    for j in rev_span:
        idx = [slice(None), slice(None), slice(None)]
        idx[span_dim] = j
        Vx[tuple(idx)] = Vx_rev
    block.set_Vx(Vx)
    return block


def _reversed_patch(block, P_back=None, sigma=1.0):
    """Attach an inlet to the i=0 face of such a block."""
    patch = NonReflectingInletPatch(i=0, label="inlet_nrbc")
    block.patches.append(patch)
    state = reference_state()
    patch.set_ho_s(float(state.ho), float(state.s))
    patch.set_Alpha(float(state.Alpha))
    patch.set_Beta(float(state.Beta))
    if P_back is not None:
        patch.set_backflow_P(P_back)
    patch.sigma = sigma
    return patch


def _span_profile(patch, field):
    """Span profile of a patch-shaped field, pitch-averaged and squeezed."""
    return np.asarray(patch._pitch_mean(field)).squeeze()


def test_set_backflow_P_stores_the_pressure_nondimensionally():
    """The pressure is stored in the units the residual is taken in."""
    _, patch = _attached()
    patch.set_backflow_P(1.1 * P_MEAN)
    np.testing.assert_allclose(
        _span_profile(patch, patch.P_nd), 1.1 * P_MEAN / FLUID.P_ref, rtol=1e-6
    )
    assert patch._target_set[4]


def test_set_backflow_P_accepts_a_spanwise_profile():
    """A profile is allowed, as for every other target; the pitch mean is kept."""
    _, patch = _attached()
    nspan = patch.shape[patch.span_dim]
    profile = P_MEAN * np.linspace(0.9, 1.1, nspan)
    patch.set_backflow_P(
        profile.reshape([nspan if d == patch.span_dim else 1 for d in range(3)])
    )
    np.testing.assert_allclose(
        _span_profile(patch, patch.P_nd), profile / FLUID.P_ref, rtol=1e-6
    )


@pytest.mark.parametrize("value, match", [(np.nan, "finite"), (-1.0e5, "positive")])
def test_set_backflow_P_rejects_invalid(value, match):
    """Non-finite or non-positive pressures are refused, as for set_P."""
    _, patch = _attached()
    with pytest.raises(ValueError, match=match):
        patch.set_backflow_P(value)


def test_backflow_P_seeds_from_the_inflow_plane_when_not_set():
    """Left alone, row 4 is taken from the face once and then frozen."""
    patch = _reversed_patch(_reversed_block())
    assert not patch._target_set[4]
    patch.update_soln()

    assert patch._target_set[4]
    expect = _span_profile(patch, patch.block_view.P_nd)
    np.testing.assert_allclose(_span_profile(patch, patch.P_nd), expect, rtol=1e-5)

    # Frozen: a later step does not re-derive it from a face this patch has
    # since been writing, which would leave the residual identically zero.
    seed = np.copy(patch.P_nd)
    patch.apply()
    patch.update_soln()
    np.testing.assert_array_equal(patch.P_nd, seed)


@pytest.mark.parametrize("span_dim", [1, 2])
def test_reversed_station_switches_the_characteristic_split(span_dim):
    """Only the downstream-running pressure wave still enters a reversed station."""
    patch = _reversed_patch(_reversed_block(span_dim=span_dim))
    patch.update_soln()
    assert not patch._entering[2]
    assert patch._entering[[0, 1, 3, 4, 5, 6]].all()

    mask = np.broadcast_to(patch._mask_out, patch.shape + (5,))
    at = [0, 0, 0]
    at[patch.span_dim] = 2
    np.testing.assert_array_equal(mask[tuple(at)], [True, False, True, True, True])
    at[patch.span_dim] = 0
    np.testing.assert_array_equal(mask[tuple(at)], [True, False, False, False, False])


@pytest.mark.parametrize("span_dim", [1, 2])
def test_reversed_station_converges_to_the_prescribed_pressure(span_dim):
    """The one quantity a reversed station prescribes is set_backflow_P's value."""
    P_back = 1.05 * P_MEAN
    patch = _reversed_patch(
        _reversed_block(span_dim=span_dim), P_back=P_back, sigma=0.5
    )
    for _ in range(80):
        patch.update_soln()
        patch.apply()

    P_face = _span_profile(patch, patch.block_view.P_nd)
    assert P_face[2] == pytest.approx(P_back / FLUID.P_ref, rel=2e-3)
    # And it really is flow leaving through the inflow face.
    assert _span_profile(patch, patch.block_view.Vx_nd)[2] < 0.0
    # A forward station is still driven to the prescribed inflow state, not to
    # a pressure: nothing about the reversed one leaks along the span.
    assert P_face[0] != pytest.approx(P_back / FLUID.P_ref, rel=1e-2)


def test_reversed_station_does_not_impose_the_inflow_state():
    """Stagnation enthalpy is carried by the outgoing waves there, not prescribed."""
    ho_target = 1.2 * float(reference_state().ho)
    patch = _reversed_patch(_reversed_block(), sigma=0.5)
    patch.set_ho_s(ho_target, float(reference_state().s))
    for _ in range(80):
        patch.update_soln()
        patch.apply()

    ho_face = _span_profile(patch, patch.block_view.ho_nd)
    target = ho_target / FLUID.u_ref
    # A forward station sits on the prescribed level.
    assert ho_face[0] == pytest.approx(target, rel=2e-3)
    # The reversed one is left where its own flow puts it.
    assert abs(ho_face[2] - target) > 0.01 * abs(target)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Vt, Vr", [(0.0, 0.0), (VT_MEAN, 0.0), (VT_MEAN, 25.0)])
def test_fixed_point(Vt, Vr):
    """A face already at the prescribed state is left untouched."""
    block, patch = _attached(Vt=Vt, Vr=Vr)
    before = block.conserved_nd.copy()
    scale = np.abs(before).max()
    patch.update_soln()
    patch.apply()
    assert np.abs(block.conserved_nd - before).max() < 1e-5 * scale
    # And still a fixed point on a second visit.
    patch.apply()
    assert np.abs(block.conserved_nd - before).max() < 1e-5 * scale


def test_mean_acoustic_wave_passes_through():
    """A uniform outgoing acoustic wave leaves the domain untouched.

    The mean mode is where the four prescribed quantities are imposed, so the
    incoming characteristics do respond to the wave; what must not happen is
    the outgoing characteristic itself being altered.
    """
    eps = 2.0e-2
    _, patch = _attached()
    patch.update_soln()
    wave = np.zeros(patch.shape + (5,), dtype=np.float32)
    wave[..., 0] = eps
    seed_chic(patch, wave)
    patch.apply()
    assert np.abs(face_chic(patch)[..., 0] - eps).max() < 1e-3 * eps


def test_mean_newton_step_is_second_order():
    """One mean-mode step lands the prescribed quantities to O(eps^2).

    Perturbing the face by a uniform acoustic wave of amplitude ``eps`` moves
    all four prescribed quantities at first order. A single modified-Newton
    step with an exact Jacobian must remove that error up to second order, so
    halving ``eps`` must cut the leftover residual fourfold. A wrong entry in
    chic_to_bcond shows up here as a factor of two.
    """
    state = reference_state()
    targets = (
        float(state.ho) / FLUID.u_ref,
        float(state.s) / FLUID.Rgas_ref,
        float(np.tan(np.radians(float(state.Alpha)))),
        float(np.sin(np.radians(float(state.Beta)))),
    )

    resid = []
    for eps in (4.0e-2, 2.0e-2):
        _, patch = _attached()
        patch.update_soln()
        wave = np.zeros(patch.shape + (5,), dtype=np.float32)
        wave[..., 0] = eps
        seed_chic(patch, wave)
        patch.apply()
        b = patch.block_view
        resid.append(
            max(
                abs(float(b.ho_nd.mean()) - targets[0]),
                abs(float(b.s_nd.mean()) - targets[1]),
                abs(float(b.tanAlpha.mean()) - targets[2]),
                abs(float(b.sinBeta.mean()) - targets[3]),
            )
        )
    assert resid[0] / resid[1] > 3.0, f"not second order: {resid}"


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("Vt", [0.0, VT_MEAN])
def test_harmonic_reflection_coefficient(mode, Vt):
    """Each harmonic satisfies the non-reflecting relation of Giles Eq. 5.17.

    Seeding a pure outgoing acoustic harmonic must leave the tangential
    vorticity characteristic at exactly the amplitude the theory demands,
    c_t = -((beta + M_t)/(1 + M_n)) c_up.
    """
    eps = 2.0e-2
    _, patch = _attached(Vt=Vt)
    patch.update_soln()

    b = patch.block_view
    wave = np.zeros(patch.shape + (5,), dtype=np.float32)
    wave[..., 0] = eps * np.cos(2.0 * np.pi * mode * b.t / PITCH)
    seed_chic(patch, wave)
    patch.apply()

    dchic = face_chic(patch)
    avg = patch.block_avg
    Mn = patch._span_bcast(avg.Vx_nd / avg.a_nd)
    Mt = patch._span_bcast(avg.Vt_nd / avg.a_nd)
    Msq = Mn**2 + Mt**2

    c_up = harmonic(patch, dchic[..., 0])
    expect = -Mt / (1.0 + Mn) * c_up + np.sqrt(1.0 - Msq) / (1.0 + Mn) * (
        patch._transform_pitch(c_up)
    )
    got = harmonic(patch, dchic[..., 3])
    assert np.abs(got - expect).max() < 1e-3 * np.abs(expect).max()


@pytest.mark.parametrize("mode", [1, 2])
def test_analytic_potential_disturbance_is_a_fixed_point(mode):
    r"""An admissible upstream potential field passes through unchanged.

    Steady linearised potential flow upstream of a blade row satisfies
    :math:`(1-M^2)\phi_{xx} + \phi_{yy} = 0`, so
    :math:`\phi = A e^{\mu x}\cos(ly)` with :math:`\mu = l/\sqrt{1-M^2}` decays
    upstream and is exactly what a non-reflecting inlet must admit without
    distortion. It carries no entropy or stagnation enthalpy perturbation
    (:math:`h_0' = p'/\bar\rho + \bar{u}u' = 0`), so every condition the patch
    imposes is already satisfied and one apply must be a no-op.

    Unlike the reflection-coefficient test, the reference here is an
    independent solution of the governing equations rather than the patch's own
    formula, so this pins down the *physics* -- in particular the sign of the
    wave parameter, which is ambiguous in Giles' text between the continuous
    wavenumber and the discrete mode index. With the sign reversed the patch
    inverts the tangential velocity of this field instead of passing it.
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
    phase = 2.0 * np.pi * mode * b.t / PITCH

    # Scale so the pressure perturbation is 1% of the mean.
    amp = 0.01 * P / (rho * Vx * decay)
    dP = -rho * Vx * decay * amp * np.cos(phase)
    b.set_P_rho(P + dP, rho + dP / a**2)
    b.set_Vx(Vx + decay * amp * np.cos(phase))
    b.set_Vr(np.zeros(patch.shape))
    b.set_Vt(-amp * wavenumber * np.sin(phase))

    before = face_prim(patch)
    perturbation = np.abs(before - patch._ref["prim"]).max()
    patch.apply()
    change = np.abs(face_prim(patch) - before).max()

    assert change < 0.05 * perturbation, (
        f"admissible potential field was altered by {change / perturbation:.1%}"
    )


def test_entropy_and_ho_stay_pitch_uniform():
    """Entropy and stagnation enthalpy are uniform along the pitch.

    This is what distinguishes Giles' formulation (his Section 5.4) from the
    pure Saxer harmonics, and nothing else guards it.
    """
    _, patch = _attached()
    patch.update_soln()
    b = patch.block_view
    rng = np.random.default_rng(1)
    wave = np.zeros(patch.shape + (5,), dtype=np.float32)
    # An arbitrary outgoing signal: several harmonics at once.
    for mode in (1, 2, 3):
        amp = rng.uniform(0.5, 1.5)
        wave[..., 0] += 2.0e-2 * amp * np.cos(2.0 * np.pi * mode * b.t / PITCH)
    seed_chic(patch, wave)
    patch.apply()

    spread_ho = np.ptp(b.ho_nd, axis=patch.pitch_dim).max()
    spread_s = np.ptp(b.s_nd, axis=patch.pitch_dim).max()
    assert spread_ho < 1e-4 * abs(float(b.ho_nd.mean()))
    assert spread_s < 1e-4


def test_converges_to_prescribed_state():
    """Iterating the boundary condition drives the face to the target.

    The interior is frozen, so this isolates the fixed-point structure of the
    condition itself: the modified Newton step plus the sigma relaxation.
    """
    sigma = 0.25
    _, patch = _attached(sigma=sigma, Vt=VT_MEAN + 40.0, T=T_MEAN + 25.0, target={})
    state = reference_state()
    target = (
        float(state.ho) / FLUID.u_ref,
        float(state.s) / FLUID.Rgas_ref,
        float(np.tan(np.radians(float(state.Alpha)))),
    )

    b = patch.block_view
    errs = []
    for _ in range(200):
        patch.update_soln()
        patch.apply()
        errs.append(
            max(
                abs(float(b.ho_nd.mean()) - target[0]) / abs(target[0]),
                abs(float(b.s_nd.mean()) - target[1]),
                abs(float(b.tanAlpha.mean()) - target[2]),
            )
        )

    assert errs[-1] < 1e-4, f"did not converge: {errs[-1]}"
    assert errs[-1] < errs[0]
    # Geometric decay at roughly the relaxation rate over the early iterations,
    # where the residual is still well above float32 noise.
    assert errs[10] < errs[0] * (1.0 - sigma) ** 8


def test_nonuniform_pitch_end_to_end():
    """A stretched pitchwise mesh still absorbs and keeps ho and s uniform."""
    _, patch = _attached(npitch=33, stretch=0.4)
    patch.update_soln()
    b = patch.block_view
    wave = np.zeros(patch.shape + (5,), dtype=np.float32)
    wave[..., 0] = 2.0e-2 * np.cos(2.0 * np.pi * b.t / PITCH)
    seed_chic(patch, wave)
    patch.apply()

    dchic = face_chic(patch)
    assert np.abs(harmonic(patch, dchic[..., 3])).max() > 1e-4  # did something
    assert np.ptp(b.s_nd, axis=patch.pitch_dim).max() < 1e-4
