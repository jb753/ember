"""Tests for the shared radial-equilibrium pressure profile.

Function tested: ember.outlet.calc_radial_equilibrium

Both outlet conditions build their spanwise pressure adjustment from this one
integration, so it is tested directly against closed-form solutions of
dp/dr = rho Vt^2 / r rather than only through the patches that call it.

Test cases:
- Closed forms: constant swirl, free vortex, no swirl
- Properties: hub anchor, second-order convergence, patch-type independence
- Inputs: reads the first interior layer, weights the pitch mean by area
"""

import numpy as np
import pytest

from ember.outlet import OutletPatch, calc_radial_equilibrium
from ember.patch import NonReflectingOutletPatch
from nonreflecting_util import PITCH, P_MEAN, make_block, pitch_coords

# Swirl scale for the analytic cases; large enough that the profile is well
# above float32 noise on the pressure it is added to.
VT_SWIRL = 120.0


def _attached(patch_type=NonReflectingOutletPatch, **kwargs):
    """Outlet patch of the given type on the i=-1 face of a fresh block."""
    block = make_block(**kwargs)
    patch = patch_type(i=-1)
    block.patches.append(patch)
    return block, patch


def _interior(patch):
    """Nondimensional (rho, r) of the layer the integration reads, per span."""
    b1 = patch.block_view_offset_1
    w = patch.weight_pitch
    pd = patch.pitch_dim
    rho = np.sum(b1.rho_nd * w, axis=pd).squeeze()
    r = np.sum(b1.r_nd * w, axis=pd).squeeze()
    return rho, r


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------


def test_constant_swirl_matches_log_profile():
    """Uniform Vt integrates to rho Vt^2 ln(r / r_hub)."""
    block, patch = _attached(nspan=41, Vt=VT_SWIRL)
    got = calc_radial_equilibrium(patch)

    rho, r = _interior(patch)
    Vt_nd = VT_SWIRL / block.fluid.V_ref
    expect = rho * Vt_nd**2 * np.log(r / r[0])
    assert np.abs(got - expect).max() < 1e-4 * np.abs(expect).max()


def test_free_vortex_matches_inverse_square_profile():
    """Vt = C/r integrates to (rho C^2 / 2) (1/r_hub^2 - 1/r^2).

    Needs a finer span mesh than the constant-swirl case for the same accuracy:
    the integrand goes as 1/r^3 rather than 1/r, so the trapezoidal error is
    around six times larger at a given spacing.
    """
    block, patch = _attached(nspan=81, Vt=0.0)
    # Free vortex about the mid radius, so the swirl magnitude stays comparable
    # with the constant-swirl case.
    r_mid = 0.5 * (block.r.min() + block.r.max())
    block.set_Vt(VT_SWIRL * r_mid / block.r)
    got = calc_radial_equilibrium(patch)

    rho, r = _interior(patch)
    C = VT_SWIRL * r_mid / block.fluid.V_ref / block.L_ref
    expect = 0.5 * rho * C**2 * (1.0 / r[0] ** 2 - 1.0 / r**2)
    assert np.abs(got - expect).max() < 5e-4 * np.abs(expect).max()


def test_no_swirl_gives_no_profile():
    """Without swirl there is no centrifugal gradient to balance."""
    _, patch = _attached(Vt=0.0)
    assert np.abs(calc_radial_equilibrium(patch)).max() == 0.0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_anchored_at_the_hub():
    """The profile is exactly zero at the first span station."""
    _, patch = _attached(Vt=VT_SWIRL)
    assert calc_radial_equilibrium(patch)[0] == 0.0


def test_second_order_in_span_spacing():
    """The trapezoidal integration converges at second order.

    A first-order slip -- a midpoint rule, or a rectangle rule from one end --
    would show up here as a halving rather than a quartering.
    """
    errs = []
    for nspan in (21, 41):
        block, patch = _attached(nspan=nspan, Vt=VT_SWIRL)
        rho, r = _interior(patch)
        Vt_nd = VT_SWIRL / block.fluid.V_ref
        expect = rho * Vt_nd**2 * np.log(r / r[0])
        errs.append(np.abs(calc_radial_equilibrium(patch) - expect).max())
    assert errs[0] / errs[1] > 3.0, f"not second order: {errs}"


def test_same_profile_for_both_outlet_types():
    """The integration depends on the geometry and flow, not the patch class."""
    profiles = []
    for patch_type in (OutletPatch, NonReflectingOutletPatch):
        _, patch = _attached(patch_type=patch_type, nspan=21, Vt=VT_SWIRL)
        profiles.append(calc_radial_equilibrium(patch))
    np.testing.assert_allclose(profiles[0], profiles[1], rtol=1e-12)


def test_outlet_patch_target_carries_the_profile():
    """OutletPatch's spanwise target is this profile, on top of its P.

    Guards the extraction: the patch must go on using the shared integration
    rather than a copy of it that could drift.
    """
    block, patch = _attached(patch_type=OutletPatch, nspan=21, Vt=VT_SWIRL)
    patch.set_P(P_MEAN)
    patch.set_adjustment(radial_equilibrium=True, rf=1.0)
    patch.update_target()

    got = patch._P_target_nd - P_MEAN / block.fluid.P_ref
    expect = patch._span_bcast(calc_radial_equilibrium(patch))
    assert np.abs(got - expect).max() < 1e-5 * np.abs(expect).max()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def test_reads_the_interior_layer_not_the_face():
    """The integrand comes from the offset-1 slice, not the boundary face.

    Which layer feeds it decides whether the target is driven by the interior
    or by whatever the boundary condition last wrote, so it is worth pinning.
    """
    block, patch = _attached(nspan=21, Vt=VT_SWIRL)
    expect = calc_radial_equilibrium(patch)

    # Wipe the swirl on the face alone; the profile must not notice.
    Vt = block.Vt.copy()
    Vt[-1] = 0.0
    block.set_Vt(Vt)
    np.testing.assert_allclose(calc_radial_equilibrium(patch), expect, rtol=1e-6)

    # Wipe it on the layer behind the face instead; now it must.
    Vt[-2] = 0.0
    block.set_Vt(Vt)
    assert np.abs(calc_radial_equilibrium(patch)).max() < 1e-3 * np.abs(expect).max()


def test_pitch_mean_is_area_weighted():
    """A pitchwise-varying swirl is averaged with the node quadrature weights.

    On a stretched pitch the weighted and unweighted means differ, and using
    the wrong one would bias every profile computed on a clustered mesh.
    """
    npitch = 33
    block, patch = _attached(nspan=21, npitch=npitch, stretch=0.4, Vt=0.0)
    theta = pitch_coords(npitch, 0.4)
    Vt = VT_SWIRL * (1.0 + 0.5 * np.cos(2.0 * np.pi * theta / PITCH))
    block.set_Vt(np.broadcast_to(Vt, block.shape))

    got = calc_radial_equilibrium(patch)
    rho, r = _interior(patch)
    w = patch.weight_pitch.ravel()
    Vt_nd = Vt / block.fluid.V_ref

    weighted = rho * np.sum(w * Vt_nd) ** 2 * np.log(r / r[0])
    plain = rho * Vt_nd.mean() ** 2 * np.log(r / r[0])
    assert np.abs(got - weighted).max() < 1e-3 * np.abs(weighted).max()
    # ... and the distinction is not academic on this mesh.
    assert np.abs(plain - weighted).max() > 1e-2 * np.abs(weighted).max()


@pytest.mark.parametrize("patch_type", [OutletPatch, NonReflectingOutletPatch])
def test_profile_scales_with_swirl_squared(patch_type):
    """Doubling the swirl quadruples the profile, as rho Vt^2 / r demands."""
    _, patch_1 = _attached(patch_type=patch_type, nspan=21, Vt=VT_SWIRL)
    _, patch_2 = _attached(patch_type=patch_type, nspan=21, Vt=2.0 * VT_SWIRL)
    ratio = calc_radial_equilibrium(patch_2)[1:] / calc_radial_equilibrium(patch_1)[1:]
    np.testing.assert_allclose(ratio, 4.0, rtol=1e-4)
