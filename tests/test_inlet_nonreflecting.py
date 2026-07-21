"""Tests for NonReflectingInletPatch.

Module tested: ember.inlet_nonreflecting.NonReflectingInletPatch

Test cases:
- Setup and validation: axis detection, plane restrictions, target validation,
  copy semantics, class member order, collection membership
- Hilbert transform: annihilation of the mean, modal response, structural
  properties, agreement with an independent DFT, non-uniform pitch spacing
- Physics: fixed point at the target, absorption of a mean acoustic wave,
  harmonic reflection coefficient against the non-reflecting theory, pitch
  uniformity of entropy and stagnation enthalpy, relaxation linearity and
  convergence to the prescribed state
- Guards: backflow, axially supersonic, absolutely supersonic
"""

import numpy as np
import pytest

from ember import perturbation, util
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.patch import PERMEABLE_TYPES, NonReflectingInletPatch


# Reference scales chosen so nondimensional quantities are O(1); with the
# default unit references P_nd would be O(1e5) and a small perturbation would
# fall below float32 resolution.
_FLUID = PerfectFluid(
    cp=1005.0,
    gamma=1.4,
    mu=1.8e-4,
    Pr=1.0,
    T_dtm=500.0,
    Rgas_ref=287.0,
    rho_ref=1.1,
    V_ref=100.0,
)
_Nb = 20
_PITCH = 2.0 * np.pi / _Nb
_P = 1.0e5
_T = 300.0
_VX = 100.0
_VT = 60.0


def _pitch_coords(npitch, stretch):
    """Pitchwise node angles, optionally stretched away from uniform spacing."""
    u = np.linspace(0.0, 1.0, npitch)
    # Monotonic for |stretch| < 1 and fixes both end points, so the patch still
    # spans exactly one pitch with the periodic node repeated.
    return _PITCH * (u + stretch * np.sin(2.0 * np.pi * u) / (2.0 * np.pi))


def _make_block(
    ni=5,
    nspan=7,
    npitch=17,
    *,
    span_dim=1,
    Vx=_VX,
    Vr=0.0,
    Vt=_VT,
    P=_P,
    T=_T,
    stretch=0.0,
):
    """Build a constant-x inlet block; span_dim selects the spanwise axis."""
    assert span_dim in (1, 2)
    nj, nk = (nspan, npitch) if span_dim == 1 else (npitch, nspan)
    shape = (ni, nj, nk)

    r_vec = np.linspace(0.3, 0.7, nspan)
    t_vec = _pitch_coords(npitch, stretch)

    x = np.linspace(0.0, 0.1, ni)[:, None, None] * np.ones(shape)
    if span_dim == 1:
        r = r_vec[None, :, None] * np.ones(shape)
        t = t_vec[None, None, :] * np.ones(shape)
    else:
        r = r_vec[None, None, :] * np.ones(shape)
        t = t_vec[None, :, None] * np.ones(shape)

    block = Block(shape=shape)
    block.set_fluid(_FLUID)
    block.set_Nb(_Nb)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)
    block.set_P_T(P, T)
    block.set_Vx(Vx * np.ones(shape))
    block.set_Vr(Vr * np.ones(shape))
    block.set_Vt(Vt * np.ones(shape))
    return block


def _reference_state(Vx=_VX, Vr=0.0, Vt=_VT, P=_P, T=_T):
    """Scalar block holding the intended inflow state, for reading off targets."""
    block = Block(shape=())
    block.set_fluid(_FLUID)
    block.set_x(np.array([0.0]))
    block.set_r(np.array([0.5]))
    block.set_t(np.array([0.0]))
    block.set_P_T(P, T)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)
    return block


def _attached(sigma=1.0, target=None, **kwargs):
    """Block with a non-reflecting inlet patch on its i=0 face.

    The prescribed inflow state defaults to whatever flow the block was built
    with, so the patch starts at its own fixed point; pass ``target`` as a dict
    of :func:`_reference_state` arguments to prescribe something else and
    create a deliberate mismatch.
    """
    block = _make_block(**kwargs)
    patch = NonReflectingInletPatch(i=0, label="inflow")
    block.patches.append(patch)
    if target is None:
        target = {
            "Vx": kwargs.get("Vx", _VX),
            "Vr": kwargs.get("Vr", 0.0),
            "Vt": kwargs.get("Vt", _VT),
            "P": kwargs.get("P", _P),
            "T": kwargs.get("T", _T),
        }
    state = _reference_state(**target)
    patch.set_ho_s_Alpha_Beta(
        ho=float(state.ho),
        s=float(state.s),
        Alpha=float(state.Alpha),
        Beta=float(state.Beta),
    )
    patch.sigma = sigma
    return block, patch


def _face_prim(patch):
    """Primitive vector [rho, Vx, Vr, Vt, P] on the patch face, nondimensional."""
    b = patch.block_view
    return np.stack((b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1)


def _face_chic(patch):
    """Characteristic deviation of the face from the frozen reference state."""
    p2c = patch._span_bcast(perturbation.primitive_to_chic(patch.block_avg))
    return util.matvec(p2c, _face_prim(patch) - patch._ref["prim"])


def _seed_chic(patch, wave):
    """Write a face state whose characteristic deviation from the mean is wave."""
    c2p = patch._span_bcast(perturbation.chic_to_primitive(patch.block_avg))
    prim = patch._ref["prim"] + util.matvec(c2p, wave)
    b = patch.block_view
    rho_nd, u_nd = b.fluid.set_P_rho(prim[..., 4], prim[..., 0])
    b.set_rho_u_Vxrt_nd(rho_nd, u_nd, prim[..., 1], prim[..., 2], prim[..., 3])


def _harmonic(patch, field):
    """Pitchwise-varying part of a patch-shaped field."""
    return field - patch._pitch_mean(field)


@pytest.fixture(params=["span_j", "span_k"])
def attached(request):
    """Target-matched patch, parametrised over the spanwise axis."""
    span_dim = 1 if request.param == "span_j" else 2
    return _attached(span_dim=span_dim)


# ---------------------------------------------------------------------------
# Setup and validation
# ---------------------------------------------------------------------------


def test_attach_detects_axes(attached):
    """Span and pitch axes are detected and the Hilbert matrix is square."""
    _, patch = attached
    assert patch.span_dim != patch.pitch_dim
    assert patch.const_dim == 0
    npitch = patch.shape[patch.pitch_dim]
    assert patch._hilbert.shape == (npitch, npitch)


def test_attach_rejects_canted_plane():
    """An inflow plane that is not constant-x is refused."""
    block = _make_block()
    # Skew the i=0 face in x so it is no longer a plane of constant x.
    x = block.x.copy()
    x[0] += 0.01 * np.linspace(0.0, 1.0, x.shape[1])[:, None]
    block.set_x(x)
    with pytest.raises(ValueError, match="constant x"):
        block.patches.append(NonReflectingInletPatch(i=0))


def test_attach_rejects_interior_on_minus_x():
    """The interior must lie on the +x side of the inflow plane."""
    block = _make_block()
    with pytest.raises(NotImplementedError, match=r"\+x"):
        block.patches.append(NonReflectingInletPatch(i=-1))


def test_attach_rejects_partial_pitch():
    """A patch covering only part of the pitch cannot define the transform."""
    block = _make_block(npitch=17)
    with pytest.raises(ValueError, match="whole pitch"):
        block.patches.append(NonReflectingInletPatch(i=0, k=(0, 8)))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"ho": np.nan}, "must be finite"),
        ({"s": np.inf}, "must be finite"),
        ({"Alpha": 90.0}, "Alpha"),
        ({"Beta": 91.0}, "Beta"),
    ],
)
def test_set_targets_rejects_invalid(kwargs, match):
    """Non-finite or out-of-range boundary condition values are refused."""
    block = _make_block()
    patch = NonReflectingInletPatch(i=0)
    block.patches.append(patch)
    with pytest.raises(ValueError, match=match):
        patch.set_ho_s_Alpha_Beta(**kwargs)


def test_set_targets_rejects_bad_shape():
    """Values that do not broadcast to the patch shape are refused."""
    block = _make_block()
    patch = NonReflectingInletPatch(i=0)
    block.patches.append(patch)
    with pytest.raises(ValueError, match="broadcast"):
        patch.set_ho_s_Alpha_Beta(ho=np.ones(patch.shape + (3,)))


def test_set_targets_are_independent():
    """Omitted arguments retain their previous value."""
    block = _make_block()
    patch = NonReflectingInletPatch(i=0)
    block.patches.append(patch)
    patch.set_ho_s_Alpha_Beta(ho=3.0e5, s=10.0, Alpha=20.0, Beta=0.0)
    patch.set_ho_s_Alpha_Beta(Alpha=30.0)
    assert float(patch.ho) == pytest.approx(3.0e5)
    assert float(patch.Alpha) == pytest.approx(30.0)


def test_apply_without_targets_raises():
    """Applying before the inflow state is set is an error, not a silent NaN."""
    block = _make_block()
    patch = NonReflectingInletPatch(i=0)
    block.patches.append(patch)
    with pytest.raises(ValueError, match="set_ho_s_Alpha_Beta"):
        patch.apply()


def test_copy_preserves_targets_and_drops_caches():
    """copy() carries the boundary condition but not geometry-derived caches."""
    _, patch = _attached(sigma=0.25)
    patch.update_soln()
    clone = patch.copy()
    assert float(clone.ho) == pytest.approx(float(patch.ho))
    assert float(clone.s) == pytest.approx(float(patch.s))
    assert float(clone.Alpha) == pytest.approx(float(patch.Alpha))
    assert float(clone.Beta) == pytest.approx(float(patch.Beta))
    assert clone.sigma == patch.sigma
    assert clone._hilbert is None
    assert clone._ref is None
    assert clone._target_nd is None
    assert clone._prim_prev is None


def test_class_member_order():
    """Class members follow the repository ordering convention."""
    import inspect

    from conftest import assert_class_member_order

    import ember.inlet_nonreflecting as module

    assert_class_member_order(inspect.getsource(module), "NonReflectingInletPatch")


def test_collection_and_permeable(attached):
    """The patch is discoverable and counts as a permeable, non-wall face."""
    block, patch = attached
    assert block.patches.inlet_nonreflecting == [patch]
    # Its own collection, not the reflecting inlet one, whose consumers poke
    # InletPatch-private attributes.
    assert block.patches.inlet == []
    assert isinstance(patch, PERMEABLE_TYPES)


# ---------------------------------------------------------------------------
# Hilbert transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stretch", [0.0, 0.4])
def test_hilbert_annihilates_constant(stretch):
    """The transform kills the pitch mean, so it cannot corrupt the mean mode."""
    _, patch = _attached(stretch=stretch)
    npitch = patch.shape[patch.pitch_dim]
    assert np.abs(patch._hilbert @ np.ones(npitch)).max() < 1e-5


def test_hilbert_modal_response():
    """H[cos] = +sin and H[sin] = -cos for every resolved mode.

    This pins the sign convention against Giles Eq. 5.16/5.19; with the sign
    reversed the boundary condition would amplify rather than absorb, and no
    other test would catch it.
    """
    _, patch = _attached(npitch=17)
    theta = _pitch_coords(17, 0.0)
    m_max = (17 - 1 - 1) // 2
    for m in range(1, m_max + 1):
        phase = 2.0 * np.pi * m * theta / _PITCH
        assert np.abs(patch._hilbert @ np.cos(phase) - np.sin(phase)).max() < 1e-5
        assert np.abs(patch._hilbert @ np.sin(phase) + np.cos(phase)).max() < 1e-5


def test_hilbert_structure():
    """Uniform spacing gives a real antisymmetric matrix with wrapped end rows."""
    _, patch = _attached(npitch=17)
    hilbert = patch._hilbert
    assert hilbert.dtype == np.float32
    # Antisymmetric up to the node weights, which are uniform except for the
    # half weights on the repeated periodic node.
    interior = hilbert[1:-1, 1:-1]
    assert np.abs(interior + interior.T).max() < 1e-5
    # The repeated periodic node must transform identically at both ends.
    assert np.abs(hilbert[0] - hilbert[-1]).max() < 1e-5


def test_hilbert_matches_dft():
    """The matrix reproduces an independent DFT implementation of Giles Eq. 5.17.

    Two independent implementations of the same equation: the collapse to a
    Hilbert transform is only valid if it agrees with the transform pair
    written out directly.
    """
    _, patch = _attached(npitch=17)
    npitch = 17
    n_dist = npitch - 1
    m_max = (n_dist - 1) // 2

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(npitch)
    signal[-1] = signal[0]  # periodic node repeated, as on the patch
    distinct = signal[:-1]

    # Giles Eq. 5.16 and 5.19-5.20 with sign(m) weighting, written directly.
    # His analysis sum carries exp(+2i pi jk/N), which is numpy's ifft, not fft.
    coeff = np.fft.ifft(distinct)
    expect = np.zeros(n_dist)
    for m in range(1, m_max + 1):
        phase = np.exp(-2j * np.pi * m * np.arange(n_dist) / n_dist)
        expect += 2.0 * np.real(1j * coeff[m] * phase)

    got = (patch._hilbert @ signal)[:-1]
    assert np.abs(got - expect).max() < 1e-4 * np.abs(expect).max()


@pytest.mark.parametrize("stretch", [0.2, 0.4])
def test_hilbert_nonuniform_resolves_low_modes(stretch):
    """Stretched pitch spacing stays accurate for the resolved harmonics.

    "Resolved" means at least four nodes per wavelength at the *coarsest*
    local spacing; above that the non-uniform quadrature degrades, which is a
    documented limitation rather than a bug.
    """
    npitch = 33
    _, patch = _attached(npitch=npitch, stretch=stretch)
    theta = _pitch_coords(npitch, stretch)
    m_resolved = int(_PITCH / (4.0 * np.diff(theta).max()))
    assert m_resolved >= 3, "test mesh is too coarse to be meaningful"
    for m in range(1, m_resolved + 1):
        phase = 2.0 * np.pi * m * theta / _PITCH
        err = np.abs(patch._hilbert @ np.cos(phase) - np.sin(phase)).max()
        assert err < 0.05, f"mode {m} error {err}"


@pytest.mark.parametrize("stretch", [0.0, 0.4])
@pytest.mark.parametrize("npitch", [17, 33])
def test_hilbert_norm_is_bounded(stretch, npitch):
    """The transform cannot amplify, whatever the pitchwise spacing.

    The discrete Hilbert transform grows only logarithmically with the node
    count, so an unresolved harmonic can leave the boundary slightly
    reflecting but never unstable.
    """
    _, patch = _attached(npitch=npitch, stretch=stretch)
    assert np.abs(patch._hilbert).sum(axis=1).max() < 4.0


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


def test_chic_round_trip(attached):
    """The frozen characteristic transform pair is an exact inverse."""
    _, patch = attached
    patch.update_soln()
    avg = patch.block_avg
    p2c = perturbation.primitive_to_chic(avg)
    c2p = perturbation.chic_to_primitive(avg)
    identity = util.matmat(c2p, p2c)
    expect = np.broadcast_to(np.eye(5, dtype=np.float32), identity.shape)
    assert np.abs(identity - expect).max() < 1e-5


@pytest.mark.parametrize("Vt, Vr", [(0.0, 0.0), (_VT, 0.0), (_VT, 25.0)])
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
    _seed_chic(patch, wave)
    patch.apply()
    assert np.abs(_face_chic(patch)[..., 0] - eps).max() < 1e-3 * eps


def test_mean_newton_step_is_second_order():
    """One mean-mode step lands the prescribed quantities to O(eps^2).

    Perturbing the face by a uniform acoustic wave of amplitude ``eps`` moves
    all four prescribed quantities at first order. A single modified-Newton
    step with an exact Jacobian must remove that error up to second order, so
    halving ``eps`` must cut the leftover residual fourfold. A wrong entry in
    chic_to_bcond shows up here as a factor of two.
    """
    state = _reference_state()
    targets = (
        float(state.ho) / _FLUID.u_ref,
        float(state.s) / _FLUID.Rgas_ref,
        float(np.tan(np.radians(float(state.Alpha)))),
        float(np.sin(np.radians(float(state.Beta)))),
    )

    resid = []
    for eps in (4.0e-2, 2.0e-2):
        _, patch = _attached()
        patch.update_soln()
        wave = np.zeros(patch.shape + (5,), dtype=np.float32)
        wave[..., 0] = eps
        _seed_chic(patch, wave)
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
@pytest.mark.parametrize("Vt", [0.0, _VT])
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
    wave[..., 0] = eps * np.cos(2.0 * np.pi * mode * b.t / _PITCH)
    _seed_chic(patch, wave)
    patch.apply()

    dchic = _face_chic(patch)
    avg = patch.block_avg
    Mn = patch._span_bcast(avg.Vx_nd / avg.a_nd)
    Mt = patch._span_bcast(avg.Vt_nd / avg.a_nd)
    Msq = Mn**2 + Mt**2

    c_up = _harmonic(patch, dchic[..., 0])
    expect = -Mt / (1.0 + Mn) * c_up + np.sqrt(1.0 - Msq) / (1.0 + Mn) * (
        patch._transform_pitch(c_up)
    )
    got = _harmonic(patch, dchic[..., 3])
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

    wavenumber = 2.0 * np.pi * mode / (b.r * _PITCH)
    decay = wavenumber / np.sqrt(1.0 - (Vx / a) ** 2)
    phase = 2.0 * np.pi * mode * b.t / _PITCH

    # Scale so the pressure perturbation is 1% of the mean.
    amp = 0.01 * P / (rho * Vx * decay)
    dP = -rho * Vx * decay * amp * np.cos(phase)
    b.set_P_rho(P + dP, rho + dP / a**2)
    b.set_Vx(Vx + decay * amp * np.cos(phase))
    b.set_Vr(np.zeros(patch.shape))
    b.set_Vt(-amp * wavenumber * np.sin(phase))

    before = _face_prim(patch)
    perturbation = np.abs(before - patch._ref["prim"]).max()
    patch.apply()
    change = np.abs(_face_prim(patch) - before).max()

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
        wave[..., 0] += 2.0e-2 * amp * np.cos(2.0 * np.pi * mode * b.t / _PITCH)
    _seed_chic(patch, wave)
    patch.apply()

    spread_ho = np.ptp(b.ho_nd, axis=patch.pitch_dim).max()
    spread_s = np.ptp(b.s_nd, axis=patch.pitch_dim).max()
    assert spread_ho < 1e-4 * abs(float(b.ho_nd.mean()))
    assert spread_s < 1e-4


def test_sigma_scales_the_correction_linearly():
    """Halving sigma halves the applied change.

    Measured in primitives, which is where the relaxation is applied; the
    conserved variables are a nonlinear function of these so their change
    carries a second-order term.
    """
    changes = {}
    for sigma in (1.0, 0.5):
        _, patch = _attached(sigma=sigma, target={"Vt": _VT + 30.0})
        patch.update_soln()
        before = _face_prim(patch)
        patch.apply()
        changes[sigma] = _face_prim(patch) - before
    ratio = np.abs(changes[0.5]).max() / np.abs(changes[1.0]).max()
    assert ratio == pytest.approx(0.5, abs=1e-3)


def test_converges_to_prescribed_state():
    """Iterating the boundary condition drives the face to the target.

    The interior is frozen, so this isolates the fixed-point structure of the
    condition itself: the modified Newton step plus the sigma relaxation.
    """
    sigma = 0.25
    _, patch = _attached(sigma=sigma, Vt=_VT + 40.0, T=_T + 25.0, target={})
    state = _reference_state()
    target = (
        float(state.ho) / _FLUID.u_ref,
        float(state.s) / _FLUID.Rgas_ref,
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
    wave[..., 0] = 2.0e-2 * np.cos(2.0 * np.pi * b.t / _PITCH)
    _seed_chic(patch, wave)
    patch.apply()

    dchic = _face_chic(patch)
    assert np.abs(_harmonic(patch, dchic[..., 3])).max() > 1e-4  # did something
    assert np.ptp(b.s_nd, axis=patch.pitch_dim).max() < 1e-4


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_backflow_raises():
    """Reversed mean flow invalidates the characteristic split."""
    _, patch = _attached(Vx=-10.0, target={})
    with pytest.raises(ValueError, match="Backflow"):
        patch.update_soln()


def test_axially_supersonic_raises():
    """Axially supersonic inflow is not implemented."""
    _, patch = _attached(Vx=500.0, Vt=0.0, target={})
    with pytest.raises(NotImplementedError, match="axially"):
        patch.update_soln()


def test_absolutely_supersonic_raises():
    """A supersonic but axially subsonic mean state is not implemented."""
    _, patch = _attached(Vx=100.0, Vt=400.0, target={})
    with pytest.raises(NotImplementedError, match="supersonic mean state"):
        patch.update_soln()
