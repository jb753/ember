"""Tests for InletPatch.

Module tested: ember.inlet.InletPatch

The characteristic solve in ``apply()`` exists to keep the inlet well
conditioned at low Mach number, so most of these tests sweep Mach number and
assert that behaviour does *not* degrade as it falls.
"""

import numpy as np
import pytest
from ember import set_iter, util
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.inlet import InletPatch


PO = 1e5
TO = 300.0
ALPHA = 15.0
BETA = 5.0
GAMMA = 1.4
_FLUID = PerfectFluid(cp=1004.5, gamma=GAMMA, mu=1.8e-5, Pr=0.7)

# Reference scales are unity on _FLUID, so dimensional and nondimensional
# values coincide and set_iter can be fed Po/To directly (as test_patch.py does).
_SHAPE = (8, 6, 5)

# Spans the regime of interest. The pressure-inversion scheme this replaced
# diverged towards the low end, where its gain gets to ~1/(gamma*Ma^2).
_MACH_SWEEP = [0.5, 0.2, 0.05, 0.01]


def _make_block(Ma, *, Alpha=ALPHA, Beta=BETA):
    """Uniform block sitting exactly on the inlet target state."""
    block = Block(shape=_SHAPE)
    block.set_fluid(_FLUID)
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], _SHAPE)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    rhoo, uo = _FLUID.set_P_T(PO, TO)
    set_iter.set_ho_s_Ma_Alpha_Beta(
        block, _FLUID.get_h(rhoo, uo), _FLUID.get_s(rhoo, uo), Ma, Alpha, Beta
    )
    return block


def _attach(block, *, Alpha=ALPHA, Beta=BETA):
    """InletPatch on the i=0 face, attached and fully specified."""
    patch = InletPatch(i=0)
    patch.attach_to_block(block)
    patch.set_Po_To_Alpha_Beta(Po=PO, To=TO, Alpha=Alpha, Beta=Beta)
    return patch


def _face_V(patch):
    """Mean face velocity along the prescribed flow direction."""
    return float(np.mean(patch._face_V_nd()))


# ---------------------------------------------------------------------------
# The target state must be a fixed point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Ma", _MACH_SWEEP)
def test_apply_at_target_is_a_fixed_point(Ma):
    """A block already at the inlet target must survive apply() untouched.

    At the target, the extrapolated interior pressure and velocity reproduce the
    face values, so the characteristic invariant R is satisfied by the existing
    face state and g(V_face) = 0 identically. Swept over Mach number because the
    scheme this replaced held this property at Ma ~ 0.5 but not below.
    """
    block = _make_block(Ma)
    patch = _attach(block)
    conserved_before = block.conserved.copy()

    patch.update_soln()
    patch.apply()

    np.testing.assert_allclose(block.conserved, conserved_before, rtol=1e-3)


def test_apply_is_idempotent_across_repeated_calls():
    """Repeated apply() on an undisturbed block must not drift.

    Each Runge-Kutta stage calls apply() again from the same anchor, so a
    per-call bias would accumulate into a slow inlet drift.
    """
    block = _make_block(0.05)
    patch = _attach(block)
    patch.update_soln()
    patch.apply()
    after_first = block.conserved.copy()

    for _ in range(20):
        patch.apply()

    np.testing.assert_allclose(block.conserved, after_first, rtol=1e-4)


# ---------------------------------------------------------------------------
# Bounded sensitivity to interior pressure -- the reason for the change
# ---------------------------------------------------------------------------


def _velocity_response(Ma, eps=1e-4):
    """Face velocity response to a relative interior pressure perturbation.

    Returns ``(dV/a) / (dp/p)``, normalising by acoustic speed rather than by V
    itself: as Ma -> 0 the velocity vanishes, so dV/V carries a 1/Ma from the
    normalisation alone and would not isolate the conditioning of the boundary
    condition. Characteristic theory puts this at 1/(gamma*(1+Ma)), independent
    of Mach number. Inverting pressure through the steady isentropic relation
    puts it at 1/(gamma*Ma) instead.
    """
    block = _make_block(Ma)
    patch = _attach(block)
    patch.update_soln()
    patch.apply()
    V0 = _face_V(patch)
    a_nd = float(np.mean(patch.block_view.a_nd))

    block = _make_block(Ma)
    patch = _attach(block)
    patch.update_soln()
    interior = block[1:]
    interior.set_P_rho_nd((1.0 + eps) * interior.P_nd, interior.rho_nd.copy())
    patch.apply()
    V1 = _face_V(patch)

    return abs(V1 - V0) / a_nd / eps


def test_pressure_sensitivity_is_mach_independent():
    """Sensitivity to interior pressure must not grow as Mach number falls.

    This is the regression test for the instability that motivated the
    characteristic solve. Under the previous pressure-inversion scheme the
    response scaled as 1/(gamma*Ma), a factor of 50 across this sweep.
    """
    responses = np.array([_velocity_response(Ma) for Ma in _MACH_SWEEP])

    # 1/(gamma*(1+Ma)) lies in [0.48, 0.72] over the sweep.
    assert np.all(responses > 0.3), responses
    assert np.all(responses < 1.0), responses

    # The decisive assertion: the low-Ma end is not amplified relative to the
    # high-Ma end. 1/(gamma*Ma) would put this ratio near 50.
    assert responses[-1] / responses[0] < 2.0, responses


# ---------------------------------------------------------------------------
# Newton solve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Ma", _MACH_SWEEP)
def test_characteristic_residual_is_satisfied(Ma):
    """After apply(), the face state must satisfy g(V) = 0 to round-off.

    Recomputes the invariant independently of the patch internals: if the Newton
    loop exited early or converged to the wrong root, the residual would show it.
    """
    block = _make_block(Ma)
    patch = _attach(block)
    patch.update_soln()

    # Perturb so the solve has real work to do rather than starting at the root.
    interior = block[1:]
    interior.set_P_rho_nd(1.002 * interior.P_nd, interior.rho_nd.copy())
    patch.apply()

    b, b1, b2 = patch.block_view, patch.block_view_offset_1, patch.block_view_offset_2
    _, _, nx, nr, nt = patch._target_nd
    V_interior = (
        (2.0 * b1.Vx_nd - b2.Vx_nd) * nx
        + (2.0 * b1.Vr_nd - b2.Vr_nd) * nr
        + (2.0 * b1.Vt_nd - b2.Vt_nd) * nt
    )
    Z = b1.rho_nd * b1.a_nd
    R = (2.0 * b1.P_nd - b2.P_nd) - Z * V_interior

    residual = b.P_nd - Z * patch._face_V_nd() - R
    assert np.max(np.abs(residual)) < 1e-4 * float(np.mean(b.P_nd))


def test_solve_converges_from_a_far_start():
    """The solve must recover from an anchor far from the root.

    g is monotone over the subsonic range, so Newton is globally convergent
    there; this guards that property against a future change to the iteration.
    """
    block = _make_block(0.05)
    patch = _attach(block)
    patch.update_soln()
    reference = block.conserved.copy()

    block = _make_block(0.05)
    patch = _attach(block)
    patch.update_soln()
    patch._V_nd_soln = np.full_like(patch._V_nd_soln, 200.0)
    patch.apply()

    # rf=1 takes the converged root in full, so the far start must not persist.
    np.testing.assert_allclose(block.conserved, reference, rtol=1e-3)


# ---------------------------------------------------------------------------
# Robustness at the low-Mach and reversed-flow limits
# ---------------------------------------------------------------------------


def test_very_low_mach_stays_finite():
    """Ma = 0.005 must produce a finite face state.

    Velocity is solved for directly rather than taken as sqrt(2*(ho - h)), a
    difference of two nearly equal O(1/Ma^2) numbers reached by two different
    code paths. In float32 that difference had a round-off floor of unknown
    sign, so the old form could take the square root of a negative number.
    """
    block = _make_block(0.005)
    patch = _attach(block)
    patch.update_soln()
    patch.apply()

    assert np.all(np.isfinite(block.conserved))
    assert np.all(block.rho > 0.0)
    assert _face_V(patch) > 0.0


def test_backflow_resolves_without_raising():
    """A reversed interior must yield a finite reversed face, not an error.

    With velocity as the primary variable, V < 0 is representable and the solve
    stays monotone for -a < V < 0, so backflow needs no special case.
    """
    block = _make_block(0.05)
    patch = _attach(block)
    patch.update_soln()

    interior = block[1:]
    interior.set_Vx(-2.0 * interior.Vx)
    patch.apply()

    assert np.all(np.isfinite(block.conserved))
    assert np.all(block.rho > 0.0)
    assert _face_V(patch) < 0.0


def test_velocity_is_capped_below_the_vacuum_limit():
    """The iterate must stay where set_h_s sees a positive temperature.

    The cap should never bind in practice; this drives it deliberately with an
    absurd interior state to confirm the result is still physical.
    """
    block = _make_block(0.05)
    patch = _attach(block)
    patch.update_soln()

    interior = block[1:]
    interior.set_P_rho_nd(1e-3 * interior.P_nd, interior.rho_nd.copy())
    patch.apply()

    assert np.all(np.isfinite(block.conserved))
    assert np.all(block.T > 0.0)
    assert np.all(np.abs(patch._face_V_nd()) <= patch._V_nd_max + 1e-6)


# ---------------------------------------------------------------------------
# Relaxation factor
# ---------------------------------------------------------------------------


def test_rf_zero_freezes_the_face_velocity():
    """rf = 0 must hold the face at the anchor regardless of the interior."""
    block = _make_block(0.05)
    patch = _attach(block)
    patch.rf = 0.0
    patch.update_soln()
    V_anchor = _face_V(patch)

    interior = block[1:]
    interior.set_P_rho_nd(1.01 * interior.P_nd, interior.rho_nd.copy())
    patch.apply()

    assert _face_V(patch) == pytest.approx(V_anchor, rel=1e-5)


def test_rf_interpolates_between_anchor_and_solution():
    """rf = 0.5 must land midway between the anchor and the rf = 1 result."""

    def _apply_with(rf):
        block = _make_block(0.05)
        patch = _attach(block)
        patch.rf = rf
        patch.update_soln()
        anchor = _face_V(patch)
        interior = block[1:]
        interior.set_P_rho_nd(1.01 * interior.P_nd, interior.rho_nd.copy())
        patch.apply()
        return anchor, _face_V(patch)

    anchor, V_full = _apply_with(1.0)
    _, V_half = _apply_with(0.5)

    assert V_half == pytest.approx(0.5 * (anchor + V_full), rel=1e-4)
