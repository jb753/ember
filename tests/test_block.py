"""Test module for ember.block.Block.

Tests Block class functionality for structured grid operations including coordinate setting, fluid properties, and thermodynamic state management.

Test cases:
- test_block_dA: Block face area calculations
- test_block_vol_positive_left_handed: Cell volumes are positive for the normal (left-handed) i,j,k grid
- test_vol_sign_matches_left_handed_jacobian: get_vol's sign convention matches the left-handed real-space Jacobian
- test_props_read_only: Read-only property validation
- test_default_Omega_Nb: Default rotation rate and blade count
- test_set_rho_u_conserves_velocity: Velocity conservation during density updates
- test_set_P_T: Pressure and temperature setting
- test_set_h_s: Enthalpy and entropy setting
- test_set_P_s: Pressure and entropy setting
- test_set_P_h: Pressure and enthalpy setting
- test_set_P_rho: Pressure and density setting
- test_flow_angles: Flow angle calculations and validation
- test_velocity_setters_preserve_thermodynamics: Thermodynamic preservation during velocity updates
- test_set_Omega: Rotation rate setting
- test_set_Nb: Blade count setting
- test_set_V_Alpha_Beta: Velocity magnitude and angle setting
- test_stagnation_enthalpy: Stagnation enthalpy calculations
- test_stagnation_properties: Stagnation property calculations
- test_derivative_properties: Derivative property calculations
- test_coordinate_access_before_initialization: Error handling for uninitialized coordinates
- test_thermodynamic_access_before_initialization: Error handling for uninitialized thermodynamics
- test_velocity_access_before_initialization: Error handling for uninitialized velocities
- test_setter_independence: Independence of setter methods
- test_V_magnitudes: Velocity magnitude calculations
- test_velocity_directions: Velocity direction calculations
- test_basic_round_trip: Round-trip property setting and getting
- test_individual_coordinate_setters: Individual coordinate component setters
- test_set_r_validates_zero_radius: Validation of zero radius values
- test_individual_velocity_setters: Individual velocity component setters
- test_individual_velocity_setters_preserve_thermodynamics: Thermodynamic preservation during velocity updates
- test_individual_setters_equivalence_with_combined: Equivalence of individual vs combined setters
- test_individual_coordinate_access_before_initialization: Error handling for individual coordinate access
- test_individual_velocity_access_before_initialization: Error handling for individual velocity access
- test_coordinate_access_works_after_setters: Coordinate access after initialization
- test_velocity_access_works_after_setters: Velocity access after initialization
- test_set_conserved_args_variations: Argument variations for conserved variable setting
- test_set_primitive_args_variations: Argument variations for primitive variable setting
- test_set_datum: Thermodynamic datum setting
- test_set_datum_effect: Effect of datum changes on u and s
- test_set_datum_preserves_fluid_properties: Fluid property preservation during datum changes
- test_conserved_writable_in_place_modification_with_slice: In-place modification with array slicing
- test_conserved_writable_full_array: Full array modification of conserved variables
- test_conserved_writable_partial_variables: Partial variable modification
- test_conserved_writable_performance_vs_copy: Performance comparison of in-place vs copy operations
- test_block_conserved_avg_property: Average conserved variable property
- test_block_conserved_avg_different_shapes: Average calculations for different grid shapes
- test_rothalpy_property: Rothalpy property calculations
- test_P_rot_no_rotation: P_rot equals P when Omega = 0
- test_P_rot_reduces_pressure: P_rot < P for both positive and negative rotation
- test_P_rot_analytical_perfect_gas: P_rot analytical formula validation
- test_P_rot_varying_radii: P_rot variation with radius
- test_P_rot_immutability: P_rot doesn't modify original block
- test_set_I_s_Ma_rel_Alpha_rel_Beta: Relative flow property setting
- test_no_patches_all_walls: Block with no patches has all-zero wall indicators
- test_inlet_patch_at_i0_is_free_surface: InletPatch at i=0 sets walli1 to 1.0
- test_outlet_patch_at_ini_is_free_surface: OutletPatch at i=-1 sets wallni to 1.0
- test_periodic_patch_at_j0_is_free_surface: PeriodicPatch at j=0 sets wallj1 to 1.0
- test_mixing_patch_at_jnj_is_free_surface: MixingPatch at j=-1 sets wallnj to 1.0
- test_Vxrt_rel_no_rotation: Vxrt_rel equals Vxrt when Omega is zero.
- test_Vxrt_rel_with_rotation: Vxrt_rel tangential component equals Vt minus blade speed.
"""

import ember.block
import ember.set_iter
import pytest
import ember.geometry
import ember.fluid
import numpy as np
from ember import util
from ember.patch import InletPatch, OutletPatch, PeriodicPatch
from ember.mixing import MixingPatch


@pytest.fixture
def block():
    """Create a configured Block with hard-coded shape, coordinates and fluid."""
    # Hard-coded shape
    shape = (3, 4, 5)

    # Create block
    b = ember.block.Block(shape=shape)

    # Set up simple coordinates
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Set up fluid
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)

    return b


def test_block_dA(block):
    # Test with the block fixture using its coordinate system
    xrt = block.xrt

    dAi = ember.geometry.get_dAi(xrt)
    dAj = ember.geometry.get_dAj(xrt)
    dAk = ember.geometry.get_dAk(xrt)
    vol = ember.geometry.get_vol(xrt, dAi, dAj, dAk)

    assert np.allclose(block.dAi, np.moveaxis(dAi, -1, 0))
    assert np.allclose(block.dAj, np.moveaxis(dAj, -1, 0))
    assert np.allclose(block.dAk, np.moveaxis(dAk, -1, 0))
    assert np.allclose(block.vol, vol)

    # Verify properties are consistent across multiple accesses
    dAi_second = block.dAi
    dAj_second = block.dAj
    dAk_second = block.dAk
    vol_second = block.vol

    # Properties should return the same values
    assert np.allclose(block.dAi, dAi_second)
    assert np.allclose(block.dAj, dAj_second)
    assert np.allclose(block.dAk, dAk_second)
    assert np.allclose(block.vol, vol_second)


def test_block_vol_positive_left_handed(block):
    """Cell volumes must be positive when i, j, k increase in the x, r,
    theta directions respectively (x increasing with i, r increasing with
    j, theta increasing with k). Per docs/coordinate_system.rst, theta is
    measured clockwise looking upstream, so this index triple is a
    left-handed set in real (x, y, z) space -- see
    test_vol_sign_matches_left_handed_jacobian for the underlying
    orientation check.
    """
    assert (block.vol > 0.0).all()


def test_vol_sign_matches_left_handed_jacobian(block):
    """get_vol's sign convention is the divergence-theorem volume with a
    deliberately flipped face-normal winding, so that a left-handed (i, j,
    k) triad in real Cartesian space -- the normal grid, since theta runs
    clockwise -- gives a positive volume. Confirm this against an
    independent Jacobian triple product computed directly from x, y, z
    (which is right-handed by construction), so any accidental change to
    the get_dAi/dAj/dAk winding order that flips this convention is caught.
    """
    xyz = np.stack([block.x, block.y, block.z], axis=-1)

    P000 = xyz[0, 0, 0]
    di = xyz[1, 0, 0] - P000
    dj = xyz[0, 1, 0] - P000
    dk = xyz[0, 0, 1] - P000
    jac_det = np.dot(np.cross(di, dj), dk)

    # (i, j, k) increasing along (x, r, theta) is left-handed in real xyz
    # space, i.e. a negative Jacobian triple product...
    assert jac_det < 0.0
    # ...yet get_vol reports positive cell volumes for this same grid.
    assert (block.vol > 0.0).all()


def test_props_read_only():
    shape = (2, 3, 4)
    b = ember.block.Block(shape=shape)
    b.set_x(np.ones(shape))
    b.set_r(np.ones(shape))
    b.set_t(np.ones(shape))

    prop_names = [
        "x",
        "r",
        "t",
        "rho",
        "rhoVx",
        "rhoVr",
        "rhorVt",
        "rhoe",
        "wdist",
        "xrt",
    ]
    for prop in prop_names:
        with pytest.raises(AttributeError):
            setattr(b, prop, "new_value")


def test_default_Omega_Nb():
    shape = (2, 3, 4)
    b = ember.block.Block(shape=shape)
    b.set_x(np.ones(shape))
    b.set_r(np.ones(shape))
    b.set_t(np.ones(shape))

    assert b.Omega == 0.0
    assert b.Nb == 1
    assert b.pitch == 2.0 * np.pi
    assert b.rpm == 0.0


def test_set_rho_u_conserves_velocity(block):
    """Test that set_rho_u changes rho and u but keeps velocities constant."""
    # Define tolerances
    rtol = 1e-6

    # Set initial conservative variables with some non-zero velocities
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    Vx_initial = np.full(block.shape, 10.0, dtype=np.float32)
    Vr_initial = np.full(block.shape, 2.0, dtype=np.float32)
    Vt_initial = np.full(block.shape, 5.0, dtype=np.float32)

    # Calculate conserved variables
    rhoVx = rho_initial * Vx_initial
    rhoVr = rho_initial * Vr_initial
    rhorVt = rho_initial * block.r * Vt_initial

    # Set internal energy for initial temperature of 300K
    u_initial = block.fluid._cv * (300.0 - block.fluid.T_dtm)
    halfVsq = 0.5 * (Vx_initial**2 + Vr_initial**2 + Vt_initial**2)
    e_initial = u_initial + halfVsq
    rhoe = rho_initial * e_initial

    # Set initial conserved variables
    conserved = np.stack([rho_initial, rhoVx, rhoVr, rhorVt, rhoe], axis=-1)
    block.set_conserved(conserved)

    # Store initial velocities for comparison
    Vx_before = block.Vx.copy()
    Vr_before = block.Vr.copy()
    Vt_before = block.Vt.copy()

    # New density and internal energy (different temperature)
    rho_new = np.full(block.shape, 0.8, dtype=np.float32)
    u_new = block.fluid._cv * (400.0 - block.fluid.T_dtm)  # 400K temperature

    # Apply the method we're testing
    block.set_rho_u(rho_new, u_new)

    # Check that velocities are preserved
    np.testing.assert_allclose(block.Vx, Vx_before, rtol=rtol)
    np.testing.assert_allclose(block.Vr, Vr_before, rtol=rtol)
    np.testing.assert_allclose(block.Vt, Vt_before, rtol=rtol)

    # Check that density and internal energy changed as expected
    np.testing.assert_allclose(block.rho, rho_new, rtol=rtol)
    np.testing.assert_allclose(block.u, u_new, rtol=rtol)

    # Verify conserved variables are consistent
    np.testing.assert_allclose(block.rhoVx, rho_new * Vx_before, rtol=rtol)
    np.testing.assert_allclose(block.rhoVr, rho_new * Vr_before, rtol=rtol)
    np.testing.assert_allclose(block.rhorVt, rho_new * block.r * Vt_before, rtol=rtol)

    # Check total energy consistency
    expected_e = u_new + 0.5 * (Vx_before**2 + Vr_before**2 + Vt_before**2)
    np.testing.assert_allclose(block.rhoe, rho_new * expected_e, rtol=rtol)


def test_set_P_T(block):
    """Test that set_P_T correctly sets pressure and temperature."""
    # Define tolerance
    rtol = 1e-6

    # Set initial state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 250000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # New pressure and temperature
    P_new = np.full(block.shape, 150000.0, dtype=np.float32)
    T_new = np.full(block.shape, 350.0, dtype=np.float32)

    # Apply the method we're testing
    block.set_P_T(P_new, T_new)

    # Check that P and T match what we get from fluid.get_P and fluid.get_T
    expected_P = block.fluid.get_P(block.rho, block.u)
    expected_T = block.fluid.get_T(block.rho, block.u)

    np.testing.assert_allclose(block.P, expected_P, rtol=rtol)
    np.testing.assert_allclose(block.T, expected_T, rtol=rtol)


def test_set_h_s(block):
    """Test that set_h_s correctly sets enthalpy and entropy."""
    rtol = 1e-6

    # Set initial state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # New enthalpy and entropy
    h_new = np.full(block.shape, 300000.0, dtype=np.float32)
    s_new = np.full(block.shape, 1000.0, dtype=np.float32)

    # Apply the method
    block.set_h_s(h_new, s_new)

    # Check that h and s match what we get from fluid calculations
    expected_h = block.fluid.get_h(block.rho, block.u)
    expected_s = block.fluid.get_s(block.rho, block.u)

    np.testing.assert_allclose(block.h, expected_h, rtol=rtol)
    np.testing.assert_allclose(block.s, expected_s, rtol=rtol)


def test_set_P_s(block):
    """Test that set_P_s correctly sets pressure and entropy."""
    rtol = 1e-6

    # Set initial state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # New pressure and entropy
    P_new = np.full(block.shape, 120000.0, dtype=np.float32)
    s_new = np.full(block.shape, 1200.0, dtype=np.float32)

    # Apply the method
    block.set_P_s(P_new, s_new)

    # Check that P and s match what we get from fluid calculations
    expected_P = block.fluid.get_P(block.rho, block.u)
    expected_s = block.fluid.get_s(block.rho, block.u)

    np.testing.assert_allclose(block.P, expected_P, rtol=rtol)
    np.testing.assert_allclose(block.s, expected_s, rtol=rtol)


def test_set_P_h(block):
    """Test that set_P_h correctly sets pressure and enthalpy."""
    rtol = 1e-6

    # Set initial state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # New pressure and enthalpy
    P_new = np.full(block.shape, 130000.0, dtype=np.float32)
    h_new = np.full(block.shape, 320000.0, dtype=np.float32)

    # Apply the method
    block.set_P_h(P_new, h_new)

    # Check that P and h match what we get from fluid calculations
    expected_P = block.fluid.get_P(block.rho, block.u)
    expected_h = block.fluid.get_h(block.rho, block.u)

    np.testing.assert_allclose(block.P, expected_P, rtol=rtol)
    np.testing.assert_allclose(block.h, expected_h, rtol=rtol)


def test_set_rho_u_Vxrt_nd(block):
    """Test that set_rho_u_Vxrt_nd correctly writes conserved variables from nondimensional state and velocity."""
    rtol = 1e-5

    # Set initial state to seed conserved variables (must be overwritten)
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)
    block.set_Vx(np.full(block.shape, 50.0, dtype=np.float32))
    block.set_Vr(np.full(block.shape, 5.0, dtype=np.float32))
    block.set_Vt(np.full(block.shape, 2.0, dtype=np.float32))

    # New target state, expressed nondimensionally
    fluid = block.fluid
    rho_target = np.full(block.shape, 0.9, dtype=np.float32)
    u_target = np.full(block.shape, 230000.0, dtype=np.float32)
    Vx_target = np.full(block.shape, 80.0, dtype=np.float32)
    Vr_target = np.full(block.shape, 3.0, dtype=np.float32)
    Vt_target = np.full(block.shape, 4.0, dtype=np.float32)

    rho_nd = rho_target / fluid.rho_ref
    u_nd = u_target / (fluid.Rgas_ref * fluid.T_ref)
    Vx_nd = Vx_target / fluid.V_ref
    Vr_nd = Vr_target / fluid.V_ref
    Vt_nd = Vt_target / fluid.V_ref

    # Apply the method
    block.set_rho_u_Vxrt_nd(rho_nd, u_nd, Vx_nd, Vr_nd, Vt_nd)

    # Verify the imposed quantities round-trip through the conserved state
    np.testing.assert_allclose(block.rho, rho_target, rtol=rtol)
    np.testing.assert_allclose(block.u, u_target, rtol=rtol)
    np.testing.assert_allclose(block.Vx, Vx_target, rtol=rtol)
    np.testing.assert_allclose(block.Vr, Vr_target, rtol=rtol)
    np.testing.assert_allclose(block.Vt, Vt_target, rtol=rtol)


def test_set_P_rho(block):
    """Test that set_P_rho correctly sets pressure and density."""
    rtol = 1e-6

    # Set initial state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # New pressure and density
    P_new = np.full(block.shape, 140000.0, dtype=np.float32)
    rho_new = np.full(block.shape, 1.5, dtype=np.float32)

    # Apply the method
    block.set_P_rho(P_new, rho_new)

    # Check that P and rho match what we get from fluid calculations
    expected_P = block.fluid.get_P(block.rho, block.u)

    np.testing.assert_allclose(block.P, expected_P, rtol=rtol)
    np.testing.assert_allclose(block.rho, rho_new, rtol=rtol)


def test_flow_angles(block):
    """Test flow angle properties."""
    rtol = 1e-6

    # Set initial state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # Set velocities: Vx=10, Vr=5, Vt=8 with Omega=2 rad/s
    block._set_metadata_by_key("Omega", 2.0)
    Vxrt = np.zeros(block.shape + (3,), dtype=np.float32)
    Vxrt[..., 0] = 10.0  # Vx
    Vxrt[..., 1] = 5.0  # Vr
    Vxrt[..., 2] = 8.0  # Vt
    block.set_Vx(Vxrt[..., 0])
    block.set_Vr(Vxrt[..., 1])
    block.set_Vt(Vxrt[..., 2])

    # Test derived velocity properties
    np.testing.assert_allclose(block.U, 2.0 * block.r, rtol=rtol)  # U = Omega * r
    np.testing.assert_allclose(
        block.Vm, np.sqrt(10**2 + 5**2), rtol=rtol
    )  # Vm = sqrt(Vx^2 + Vr^2)
    np.testing.assert_allclose(
        block.Vt_rel, 8.0 - block.U, rtol=rtol
    )  # Vt_rel = Vt - U
    np.testing.assert_allclose(
        block.V_rel, np.sqrt(block.Vm**2 + block.Vt_rel**2), rtol=rtol
    )  # V_rel = sqrt(Vm^2 + Vt_rel^2)

    # Test flow angles
    expected_Alpha = np.degrees(np.arctan2(8.0, block.Vm))  # Alpha = atan2(Vt, Vm)
    expected_Alpha_rel = np.degrees(
        np.arctan2(block.Vt_rel, block.Vm)
    )  # Alpha_rel = atan2(Vt_rel, Vm)
    expected_Beta = np.degrees(np.arctan2(5.0, 10.0))  # Beta = atan2(Vr, Vx)

    np.testing.assert_allclose(block.Alpha, expected_Alpha, rtol=rtol)
    np.testing.assert_allclose(block.Alpha_rel, expected_Alpha_rel, rtol=rtol)
    np.testing.assert_allclose(block.Beta, expected_Beta, rtol=rtol)

    # Test tangent properties
    np.testing.assert_allclose(block.tanAlpha, 8.0 / block.Vm, rtol=rtol)
    np.testing.assert_allclose(block.tanAlpha_rel, block.Vt_rel / block.Vm, rtol=rtol)
    np.testing.assert_allclose(
        block.tanBeta, 0.5, rtol=rtol
    )  # tan(Beta) = Vr/Vx = 5/10 = 0.5


def test_velocity_setters_preserve_thermodynamics(block):
    """Test that set_Vx/set_Vr/set_Vt changes velocities but preserves thermodynamic properties."""
    rtol = 1e-6

    # Set initial thermodynamic state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # Set initial velocities
    Vxrt_initial = np.zeros(block.shape + (3,), dtype=np.float32)
    Vxrt_initial[..., 0] = 50.0  # Vx
    Vxrt_initial[..., 1] = -10.0  # Vr
    Vxrt_initial[..., 2] = 30.0  # Vt
    block.set_Vx(Vxrt_initial[..., 0])
    block.set_Vr(Vxrt_initial[..., 1])
    block.set_Vt(Vxrt_initial[..., 2])

    # Store initial thermodynamic properties
    rho_before = block.rho.copy()
    P_before = block.P.copy()
    T_before = block.T.copy()
    u_before = block.u.copy()

    # Change velocities
    Vxrt_new = np.zeros(block.shape + (3,), dtype=np.float32)
    Vxrt_new[..., 0] = 80.0  # Vx
    Vxrt_new[..., 1] = 20.0  # Vr
    Vxrt_new[..., 2] = -15.0  # Vt

    # Apply the method we're testing
    block.set_Vx(Vxrt_new[..., 0])
    block.set_Vr(Vxrt_new[..., 1])
    block.set_Vt(Vxrt_new[..., 2])

    # Check that velocities changed as expected
    np.testing.assert_allclose(block.Vx, Vxrt_new[..., 0], rtol=rtol)
    np.testing.assert_allclose(block.Vr, Vxrt_new[..., 1], rtol=rtol)
    np.testing.assert_allclose(block.Vt, Vxrt_new[..., 2], rtol=rtol)

    # Check that thermodynamic properties are preserved
    np.testing.assert_allclose(block.rho, rho_before, rtol=rtol)
    np.testing.assert_allclose(block.P, P_before, rtol=rtol)
    np.testing.assert_allclose(block.T, T_before, rtol=rtol)
    np.testing.assert_allclose(block.u, u_before, rtol=rtol)
    # Note: e (total energy) should change as kinetic energy changes

    # Check that conserved variables are consistent
    np.testing.assert_allclose(block.rhoVx, rho_before * Vxrt_new[..., 0], rtol=rtol)
    np.testing.assert_allclose(block.rhoVr, rho_before * Vxrt_new[..., 1], rtol=rtol)
    np.testing.assert_allclose(
        block.rhorVt, rho_before * block.r * Vxrt_new[..., 2], rtol=rtol
    )
    np.testing.assert_allclose(
        block.rhoe, rho_before * (block.u + 0.5 * block.V**2), rtol=rtol
    )


def test_set_Omega():
    """Test set_Omega method."""
    b = ember.block.Block(shape=(2, 2))

    # Test setting Omega
    b.set_Omega(1000.0)
    assert b.Omega == 1000.0

    # Test type casting
    b.set_Omega(np.float64(500.5))
    assert b.Omega == np.float32(500.5)


def test_set_Nb():
    """Test set_Nb method."""
    b = ember.block.Block(shape=(2, 2))

    # Test setting Nb
    b.set_Nb(24)
    assert b.Nb == 24

    # Test type casting
    b.set_Nb(np.int64(12))
    assert b.Nb == 12


def test_set_V_Alpha_Beta(block):
    """Test set_V_Alpha_Beta method."""
    rtol = 1e-6

    # Set initial thermodynamic state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # Set velocities using V, Alpha, Beta
    V = np.full(block.shape, 100.0, dtype=np.float32)  # 100 m/s magnitude
    Alpha = np.full(block.shape, 30.0, dtype=np.float32)  # 30 deg yaw angle
    Beta = np.full(block.shape, 15.0, dtype=np.float32)  # 15 deg pitch angle

    block.set_V_Alpha_Beta(V, Alpha, Beta)

    # Check that velocity magnitude matches
    np.testing.assert_allclose(
        np.sqrt(block.Vx**2 + block.Vr**2 + block.Vt**2), V, rtol=rtol
    )

    # Check that angles match
    expected_Alpha = np.degrees(np.arctan2(block.Vt, block.Vm))
    expected_Beta = np.degrees(np.arctan2(block.Vr, block.Vx))

    np.testing.assert_allclose(block.Alpha, expected_Alpha, rtol=rtol)
    np.testing.assert_allclose(block.Beta, expected_Beta, rtol=rtol)

    # Check specific angle values
    np.testing.assert_allclose(block.Alpha, Alpha, rtol=rtol)
    np.testing.assert_allclose(block.Beta, Beta, rtol=rtol)


def test_stagnation_enthalpy(block):
    """Test stagnation enthalpy properties ho and ho_rel."""
    rtol = 1e-6

    # Set initial thermodynamic state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # Set reference frame angular velocity
    block.set_Omega(100.0)  # 100 rad/s

    # Set velocities
    Vxrt = np.zeros(block.shape + (3,), dtype=np.float32)
    Vxrt[..., 0] = 50.0  # Vx = 50 m/s
    Vxrt[..., 1] = 10.0  # Vr = 10 m/s
    Vxrt[..., 2] = 80.0  # Vt = 80 m/s
    block.set_Vx(Vxrt[..., 0])
    block.set_Vr(Vxrt[..., 1])
    block.set_Vt(Vxrt[..., 2])

    # Test ho (absolute stagnation enthalpy)
    expected_ho = block.h + 0.5 * block.V**2
    np.testing.assert_allclose(block.ho, expected_ho, rtol=rtol)

    # Test ho_rel (relative stagnation enthalpy)
    expected_ho_rel = block.h + 0.5 * block.V_rel**2
    np.testing.assert_allclose(block.ho_rel, expected_ho_rel, rtol=rtol)

    # Verify stagnation enthalpy relationships
    # ho should be higher than h (kinetic energy added)
    assert np.all(block.ho > block.h)

    # ho_rel should generally be different from ho (due to frame rotation)
    # Vt_rel = Vt - U, so relative kinetic energy is different
    assert not np.allclose(block.ho, block.ho_rel, rtol=rtol)


def test_stagnation_properties(block):
    """Test stagnation density, internal energy, pressure and temperature properties."""
    rtol = 1e-6

    # Set initial thermodynamic state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # Set reference frame angular velocity
    block.set_Omega(50.0)  # 50 rad/s

    # Set velocities
    Vxrt = np.zeros(block.shape + (3,), dtype=np.float32)
    Vxrt[..., 0] = 100.0  # Vx = 100 m/s
    Vxrt[..., 1] = 20.0  # Vr = 20 m/s
    Vxrt[..., 2] = 150.0  # Vt = 150 m/s
    block.set_Vx(Vxrt[..., 0])
    block.set_Vr(Vxrt[..., 1])
    block.set_Vt(Vxrt[..., 2])

    # Test that stagnation properties are computed correctly using fluid.set_h_s
    # rhoo and uo should be consistent with ho and s
    expected_rho_stag, expected_u_stag = block.fluid.set_h_s(block.ho, block.s)
    np.testing.assert_allclose(block.rhoo, expected_rho_stag, rtol=rtol)
    np.testing.assert_allclose(block.uo, expected_u_stag, rtol=rtol)

    # rhoo_rel and uo_rel should be consistent with ho_rel and s
    expected_rho_stag_rel, expected_u_stag_rel = block.fluid.set_h_s(
        block.ho_rel, block.s
    )
    np.testing.assert_allclose(block.rhoo_rel, expected_rho_stag_rel, rtol=rtol)
    np.testing.assert_allclose(block.uo_rel, expected_u_stag_rel, rtol=rtol)

    # Test stagnation pressure and temperature
    expected_Po = block.fluid.get_P(block.rhoo, block.uo)
    expected_To = block.fluid.get_T(block.rhoo, block.uo)
    np.testing.assert_allclose(block.Po, expected_Po, rtol=rtol)
    np.testing.assert_allclose(block.To, expected_To, rtol=rtol)

    # Test relative stagnation pressure and temperature
    expected_Po_rel = block.fluid.get_P(block.rhoo_rel, block.uo_rel)
    expected_To_rel = block.fluid.get_T(block.rhoo_rel, block.uo_rel)
    np.testing.assert_allclose(block.Po_rel, expected_Po_rel, rtol=rtol)
    np.testing.assert_allclose(block.To_rel, expected_To_rel, rtol=rtol)

    # Test that stagnation properties are higher than static properties
    # (since we have non-zero velocity, kinetic energy is added)
    assert np.all(block.Po > block.P)
    assert np.all(block.To > block.T)
    assert np.all(
        block.rhoo > block.rho
    )  # For isentropic process with increasing enthalpy

    # Relative stagnation properties should generally be different from absolute
    # (due to frame rotation effects)
    assert not np.allclose(block.Po, block.Po_rel, rtol=rtol)
    assert not np.allclose(block.To, block.To_rel, rtol=rtol)
    assert not np.allclose(block.rhoo, block.rhoo_rel, rtol=rtol)
    assert not np.allclose(block.uo, block.uo_rel, rtol=rtol)

    # Test entropy conservation - stagnation state should have same entropy as static
    expected_s_stag = block.fluid.get_s(block.rhoo, block.uo)
    expected_s_stag_rel = block.fluid.get_s(block.rhoo_rel, block.uo_rel)
    np.testing.assert_allclose(block.s, expected_s_stag, rtol=rtol)
    np.testing.assert_allclose(block.s, expected_s_stag_rel, rtol=rtol)


def test_derivative_properties(block):
    """Test derivative properties for volumetric total energy."""
    rtol = 1e-6

    # Set initial thermodynamic state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # Set some velocities to have non-zero kinetic energy
    Vxrt = np.zeros(block.shape + (3,), dtype=np.float32)
    Vxrt[..., 0] = 50.0  # Vx = 50 m/s
    Vxrt[..., 1] = 10.0  # Vr = 10 m/s
    Vxrt[..., 2] = 30.0  # Vt = 30 m/s
    block.set_Vx(Vxrt[..., 0])
    block.set_Vr(Vxrt[..., 1])
    block.set_Vt(Vxrt[..., 2])

    # Test that derivatives are consistent with underlying fluid properties
    # The derivatives should match what we get from the fluid's derivative methods
    np.testing.assert_allclose(
        block.dudrho_P_nd,
        block.fluid.get_dudrho_P(block.rho_nd, block.u_nd),
        rtol=rtol,
    )
    np.testing.assert_allclose(
        block.dudP_rho_nd,
        block.fluid.get_dudP_rho(block.rho_nd, block.u_nd),
        rtol=rtol,
    )

    # Test that the derivatives make physical sense
    # For a perfect gas, dudrho_P should be negative (internal energy decreases with increasing density at constant pressure)
    # and dudP_rho should be positive (internal energy increases with increasing pressure at constant density)
    assert np.all(block.dudrho_P_nd < 0)
    assert np.all(block.dudP_rho_nd > 0)


def test_coordinate_access_before_initialization():
    """Test that accessing coordinate properties before coordinate initialization raises errors."""
    shape = (2, 3, 2)
    b = ember.block.Block(shape=shape)

    # Test: Accessing coordinate properties before coordinate initialization should raise errors
    with pytest.raises(
        ValueError, match="Data for variable x has not been initialised"
    ):
        _ = b.x

    with pytest.raises(
        ValueError, match="Data for variable r has not been initialised"
    ):
        _ = b.r

    with pytest.raises(
        ValueError, match="Data for variable t has not been initialised"
    ):
        _ = b.t

    with pytest.raises(ValueError):
        _ = b.xrt

    # After setting coordinates, these should work without errors
    xrt = np.zeros(shape + (3,), dtype=np.float32)
    xrt[..., 0] = np.linspace(0, 1, shape[0])[:, None, None]  # x coordinates
    xrt[..., 1] = np.linspace(0.5, 1.5, shape[1])[None, :, None]  # r coordinates
    xrt[..., 2] = np.linspace(0, 0.1, shape[2])[None, None, :]  # t coordinates
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Now these should work without errors
    assert b.x.shape == shape
    assert b.r.shape == shape
    assert b.t.shape == shape
    assert b.xrt.shape == shape + (3,)


def test_thermodynamic_access_before_initialization():
    """Test that accessing thermodynamic properties before set_rho_u raises errors."""
    shape = (2, 3, 2)
    b = ember.block.Block(shape=shape)

    # Set up coordinates (needed for some thermodynamic calculations)
    xrt = np.zeros(shape + (3,), dtype=np.float32)
    xrt[..., 0] = np.linspace(0, 1, shape[0])[:, None, None]  # x coordinates
    xrt[..., 1] = np.linspace(0.5, 1.5, shape[1])[None, :, None]  # r coordinates
    xrt[..., 2] = np.linspace(0, 0.1, shape[2])[None, None, :]  # t coordinates
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Test: Accessing fluid properties before set_fluid should raise errors
    with pytest.raises(ValueError, match="Working fluid must be set using set_fluid"):
        _ = b.fluid

    # Set up fluid (needed for thermo property calculations)
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)

    # Test: Accessing thermodynamic properties before set_rho_u should raise errors
    with pytest.raises(
        ValueError, match="Data for variable rho has not been initialised"
    ):
        _ = b.P

    with pytest.raises(
        ValueError, match="Data for variable rho has not been initialised"
    ):
        _ = b.T

    with pytest.raises(
        ValueError, match="Data for variable rho has not been initialised"
    ):
        _ = b.a

    with pytest.raises(
        ValueError, match="Data for variable rho has not been initialised"
    ):
        _ = b.cp

    with pytest.raises(
        ValueError, match="Data for variable rho has not been initialised"
    ):
        _ = b.s

    with pytest.raises(
        ValueError, match="Data for variable rho has not been initialised"
    ):
        _ = b.h

    # After set_rho_u, these should work without errors
    rho = np.full(shape, 1.2, dtype=np.float32)
    u = np.full(shape, 200000.0, dtype=np.float32)
    b.set_rho_u(rho, u)

    # Now these should work without errors
    assert b.P.shape == shape
    assert b.T.shape == shape
    assert b.a.shape == shape
    assert b.cp.shape == shape
    assert b.s.shape == shape
    assert b.h.shape == shape


def test_velocity_access_before_initialization():
    """Test that accessing velocity properties before velocity initialization raises errors."""
    shape = (2, 3, 2)
    b = ember.block.Block(shape=shape)

    # Set up coordinates
    xrt = np.zeros(shape + (3,), dtype=np.float32)
    xrt[..., 0] = np.linspace(0, 1, shape[0])[:, None, None]  # x coordinates
    xrt[..., 1] = np.linspace(0.5, 1.5, shape[1])[None, :, None]  # r coordinates
    xrt[..., 2] = np.linspace(0, 0.1, shape[2])[None, None, :]  # t coordinates
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Set up fluid and thermodynamic state
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)
    rho = np.full(shape, 1.2, dtype=np.float32)
    u = np.full(shape, 200000.0, dtype=np.float32)
    b.set_rho_u(rho, u)

    # Test: Accessing velocity momentum properties before velocity initialization should raise errors
    # (These are the conserved momentum variables that aren't initialized)
    with pytest.raises(
        ValueError, match="Data for variable rhoVx has not been initialised"
    ):
        _ = b.rhoVx

    with pytest.raises(
        ValueError, match="Data for variable rhoVr has not been initialised"
    ):
        _ = b.rhoVr

    with pytest.raises(
        ValueError, match="Data for variable rhorVt has not been initialised"
    ):
        _ = b.rhorVt

    # Derived velocity properties should also fail since they depend on the momentum variables
    with pytest.raises(
        ValueError, match="Data for variable rhoVx has not been initialised"
    ):
        _ = b.Vx

    with pytest.raises(
        ValueError, match="Data for variable rhoVr has not been initialised"
    ):
        _ = b.Vr

    with pytest.raises(
        ValueError, match="Data for variable rhorVt has not been initialised"
    ):
        _ = b.Vt

    # After setting velocities, these should work without errors
    Vxrt = np.zeros(shape + (3,), dtype=np.float32)
    Vxrt[..., 0] = 50.0  # Vx = 50 m/s
    Vxrt[..., 1] = 10.0  # Vr = 10 m/s
    Vxrt[..., 2] = 30.0  # Vt = 30 m/s
    b.set_Vx(Vxrt[..., 0])
    b.set_Vr(Vxrt[..., 1])
    b.set_Vt(Vxrt[..., 2])

    # Now these should work without errors
    assert b.rhoVx.shape == shape
    assert b.rhoVr.shape == shape
    assert b.rhorVt.shape == shape
    assert b.Vx.shape == shape
    assert b.Vr.shape == shape
    assert b.Vt.shape == shape


def test_setter_independence():
    """Test that set_rho_u, coordinates, and velocities can be called in any order."""
    shape = (1, 1, 1)

    # Test data
    xrt = 2.0 * np.ones(shape + (3,), dtype=np.float32)
    rho = np.full(shape, 1.2, dtype=np.float32)
    u = np.full(shape, 20000.0, dtype=np.float32)

    Vxrt = np.zeros(shape + (3,), dtype=np.float32)
    Vxrt[..., 0] = 50.0  # Vx = 50 m/s
    Vxrt[..., 1] = 10.0  # Vr = 10 m/s
    Vxrt[..., 2] = 30.0  # Vt = 30 m/s

    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)

    # Test all 6 possible orderings of the three setter methods
    orderings = [
        ("set_coords", "set_rho_u", "set_vels"),
        ("set_coords", "set_vels", "set_rho_u"),
        ("set_rho_u", "set_coords", "set_vels"),
        ("set_rho_u", "set_vels", "set_coords"),
        ("set_vels", "set_coords", "set_rho_u"),
        ("set_vels", "set_rho_u", "set_coords"),
    ]

    for i, order in enumerate(orderings):
        # Create a fresh block for each test
        b = ember.block.Block(shape=shape)
        b.set_fluid(fluid)

        # Apply setters in the specified order
        print(f"Ordering {i + 1}: ", order)
        for method_name in order:
            if method_name == "set_coords":
                b.set_x(xrt[..., 0])
                b.set_r(xrt[..., 1])
                b.set_t(xrt[..., 2])
                assert np.allclose(b.xrt, xrt)
            elif method_name == "set_rho_u":
                b.set_rho_u(rho, u)
                assert np.allclose(b.rho, rho)
                assert np.allclose(b.u, u)
            elif method_name == "set_vels":
                b.set_Vx(Vxrt[..., 0])
                b.set_Vr(Vxrt[..., 1])
                b.set_Vt(Vxrt[..., 2])
                assert np.allclose(b.Vxrt, Vxrt)

        # After all setters are called, all properties should work
        # Coordinate properties
        assert b.x.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.r.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.t.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.xrt.shape == shape + (3,), f"Failed for ordering {i + 1}: {order}"

        # Thermodynamic properties
        assert b.rho.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.P.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.T.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.u.shape == shape, f"Failed for ordering {i + 1}: {order}"

        # Velocity properties
        assert b.rhoVx.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.rhoVr.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.rhorVt.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.Vx.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.Vr.shape == shape, f"Failed for ordering {i + 1}: {order}"
        assert b.Vt.shape == shape, f"Failed for ordering {i + 1}: {order}"

        # Verify the values are consistent regardless of order
        np.testing.assert_allclose(
            b.Vx,
            Vxrt[..., 0],
            rtol=1e-6,
            err_msg=f"Vx inconsistent for ordering {i + 1}: {order}",
        )
        np.testing.assert_allclose(
            b.Vr,
            Vxrt[..., 1],
            rtol=1e-6,
            err_msg=f"Vr inconsistent for ordering {i + 1}: {order}",
        )
        np.testing.assert_allclose(
            b.Vt,
            Vxrt[..., 2],
            rtol=1e-6,
            err_msg=f"Vt inconsistent for ordering {i + 1}: {order}",
        )
        np.testing.assert_allclose(
            b.rho,
            rho,
            rtol=1e-6,
            err_msg=f"rho inconsistent for ordering {i + 1}: {order}",
        )
        np.testing.assert_allclose(
            b.u, u, rtol=1e-6, err_msg=f"u inconsistent for ordering {i + 1}: {order}"
        )


def test_V_magnitudes(block):
    block.set_conserved(np.random.rand(*block.shape, 5) + 1.0)

    Vx, Vr, Vt = block.Vx, block.Vr, block.Vt
    Vy, Vz = block.Vy, block.Vz
    Vm, V = block.Vm, block.V
    V_rel, Vt_rel = block.V_rel, block.Vt_rel
    U = block.U
    r, t, y, z = block.r, block.t, block.y, block.z
    cost, sint = np.cos(t), np.sin(t)

    assert np.allclose(Vx**2 + Vr**2 + Vt**2, V**2, rtol=1e-6)
    assert np.allclose(Vx**2 + Vy**2 + Vz**2, V**2, rtol=1e-6)
    assert np.allclose(Vx**2 + Vr**2, Vm**2, rtol=1e-6)
    assert np.allclose(Vr**2 + Vt**2, Vy**2 + Vz**2, rtol=1e-6)
    assert np.allclose(Vt_rel, Vt - U, rtol=1e-6)
    assert np.allclose(Vx**2 + Vr**2 + Vt_rel**2, V_rel**2, rtol=1e-6)

    assert np.allclose(t, np.arctan2(-z, y), rtol=1e-6)
    assert np.allclose(r, np.sqrt(y**2 + z**2), rtol=1e-6)
    assert np.allclose(y, r * cost, rtol=1e-6)
    assert np.allclose(z, -r * sint, rtol=1e-6)

    assert np.allclose(Vy, Vr * cost - Vt * sint, rtol=1e-6)
    assert np.allclose(Vz, -Vr * sint - Vt * cost, rtol=1e-6)
    assert np.allclose(Vr, Vy * cost - Vz * sint, rtol=1e-6)
    assert np.allclose(Vt, -Vy * sint - Vz * cost, rtol=1e-6)


def test_velocity_directions():
    """Test that Vy and Vz velocities point in correct coordinate directions across theta quadrants."""

    def create_quadrant_block(theta_range, shape=(2, 2, 3)):
        """Create a block with points in specified theta range."""
        b = ember.block.Block(shape=shape)

        # Set up coordinates
        xrt = util.linmesh3(
            [0.0, 1.0], [0.5, 1.5], [theta_range[0], theta_range[1]], shape
        )
        b.set_x(xrt[..., 0])
        b.set_r(xrt[..., 1])
        b.set_t(xrt[..., 2])

        # Set up fluid
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        b.set_fluid(fluid)

        return b

    # Define theta quadrants
    # (name, range, Vr_sign, Vt_sign)
    dt = np.pi / 16.0
    pi_2 = np.pi / 2.0
    quadrants = [
        ("Q1", (0, pi_2), (1, -1), (-1, -1)),  # First quadrant: 0 to π/2
        ("Q2", (pi_2, np.pi), (-1, -1), (-1, 1)),  # Second quadrant: π/2 to π
        ("Q3", (np.pi, 3 * pi_2), (-1, 1), (1, 1)),  # Third quadrant: π to 3π/2
        ("Q4", (3 * pi_2, 2 * np.pi), (1, 1), (1, -1)),  # Fourth quadrant: 3π/2 to 2π
    ]

    # Test Case 1: Vr = constant, Vt = 0 (pure radial motion)
    Vr_test = 10.0
    for quad_name, (t_min, t_max), exp_signs, _ in quadrants:
        b = create_quadrant_block((t_min + dt, t_max - dt))

        # Set pure radial velocity
        Vxrt = np.zeros(b.shape + (3,), dtype=np.float32)
        Vxrt[..., 0] = 0.0  # Vx = 0
        Vxrt[..., 1] = Vr_test  # Vr = constant (outward)
        Vxrt[..., 2] = 0.0  # Vt = 0
        b.set_Vx(Vxrt[..., 0])
        b.set_Vr(Vxrt[..., 1])
        b.set_Vt(Vxrt[..., 2])

        np.testing.assert_allclose(
            np.sign(b.Vy),
            exp_signs[0],
            err_msg=f"Vy sign mismatch in {quad_name} for pure radial motion",
        )

        np.testing.assert_allclose(
            np.sign(b.Vz),
            exp_signs[1],
            err_msg=f"Vz sign mismatch in {quad_name} for pure radial motion",
        )

    # Test Case 2: Vt = constant, Vr = 0 (pure tangential motion)
    Vt_test = 15.0
    for quad_name, (t_min, t_max), _, exp_signs in quadrants:
        b = create_quadrant_block((t_min + dt, t_max - dt))

        # Set pure tangential velocity
        Vxrt = np.zeros(b.shape + (3,), dtype=np.float32)
        Vxrt[..., 0] = 0.0  # Vx = 0
        Vxrt[..., 1] = 0.0  # Vr = 0
        Vxrt[..., 2] = Vt_test  # Vt = constant (counter-clockwise)
        b.set_Vx(Vxrt[..., 0])
        b.set_Vr(Vxrt[..., 1])
        b.set_Vt(Vxrt[..., 2])

        np.testing.assert_allclose(
            np.sign(b.Vy),
            exp_signs[0],
            err_msg=f"Vy sign mismatch in {quad_name} for pure tangential motion",
        )

        np.testing.assert_allclose(
            np.sign(b.Vz),
            exp_signs[1],
            err_msg=f"Vz sign mismatch in {quad_name} for pure tangential motion",
        )


def test_basic_round_trip(block):
    """Test round trip: initialize with primitive, convert to conserved, re-set primitive, verify conserved recovery."""

    # Step 1: Generate physically consistent primitive variables
    np.random.seed(42)  # For reproducible tests

    shape = block.shape

    # Generate random primitive data
    rho = 1.0 + np.random.rand(*shape) * 1.0  # Density: 1.0 to 2.0
    Vx = 1.0 + np.random.rand(*shape) * 10.0  # x-velocity: 1 to 10 m/s
    Vr = 1.0 + np.random.rand(*shape) * 5.0  # r-velocity: 1 to 5 m/s
    Vt = 1.0 + np.random.rand(*shape) * 8.0  # t-velocity: 1 to 8 m/s
    P = 1e5 + np.random.rand(*shape) * 1e5  # Pressure: 50k to 100k Pa

    # Set primitive variables to get physically consistent conserved variables
    block.set_P_rho(P, rho)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Store the resulting conserved variables
    conserved_true = block.conserved.copy()

    # Now do the round trip: extract primitives and set them again
    block.set_P_rho(block.P.copy(), block.rho.copy())
    block.set_Vx(block.Vx.copy())
    block.set_Vr(block.Vr.copy())
    block.set_Vt(block.Vt.copy())

    np.testing.assert_allclose(block.rho, rho, rtol=1e-6)
    np.testing.assert_allclose(block.Vx, Vx, rtol=1e-6)
    np.testing.assert_allclose(block.Vr, Vr, rtol=1e-6)
    np.testing.assert_allclose(block.Vt, Vt, rtol=1e-6)
    np.testing.assert_allclose(block.P, P, rtol=1e-6)
    np.testing.assert_allclose(
        block.conserved[..., :4], conserved_true[..., :4], rtol=1e-6
    )
    np.testing.assert_allclose(
        block.conserved[..., 4], conserved_true[..., 4], rtol=1e-4
    )


def test_individual_coordinate_setters(block):
    """Test individual coordinate setters (set_x, set_r, set_t) work correctly."""
    rtol = 1e-6

    # Test individual coordinate setters using the fixture block shape
    shape = block.shape
    x_vals = np.zeros(shape)
    r_vals = np.ones(shape)
    t_vals = np.full(shape, 0.5)

    # Set x values to vary along first axis
    for i in range(shape[0]):
        x_vals[i, :, :] = float(i)

    # Set r values to vary along second axis
    for j in range(shape[1]):
        r_vals[:, j, :] = 0.5 + 0.5 * j

    # Set t values to vary along third axis
    for k in range(shape[2]):
        t_vals[:, :, k] = 0.1 * k

    # Set coordinates individually
    block.set_x(x_vals)
    block.set_r(r_vals)
    block.set_t(t_vals)

    # Verify coordinates are set correctly
    np.testing.assert_allclose(block.x, x_vals, rtol=rtol)
    np.testing.assert_allclose(block.r, r_vals, rtol=rtol)
    np.testing.assert_allclose(block.t, t_vals, rtol=rtol)

    # Verify xrt property combines them correctly
    expected_xrt = np.stack([x_vals, r_vals, t_vals], axis=-1)
    np.testing.assert_allclose(block.xrt, expected_xrt, rtol=rtol)


def test_set_r_validates_zero_radius(block):
    """Test that set_r raises ValueError for zero radius."""
    # Test zero radius validation
    r_with_zero = np.ones(block.shape)
    r_with_zero.flat[0] = 0.0

    with pytest.raises(ValueError, match="Radial coordinate cannot be zero"):
        block.set_r(r_with_zero)


def test_individual_velocity_setters(block):
    """Test individual velocity setters (set_Vx, set_Vr, set_Vt) work correctly."""
    rtol = 1e-6

    # Test individual velocity setters with scalar values
    block.set_Vx(15.0)
    block.set_Vr(8.0)
    block.set_Vt(12.0)

    # Verify velocities are set correctly (scalars should broadcast to full arrays)
    np.testing.assert_allclose(block.Vx, 15.0, rtol=rtol)
    np.testing.assert_allclose(block.Vr, 8.0, rtol=rtol)
    np.testing.assert_allclose(block.Vt, 12.0, rtol=rtol)

    # Verify Vxrt property combines them correctly
    shape = block.shape
    expected_Vxrt = np.stack(
        [np.full(shape, 15.0), np.full(shape, 8.0), np.full(shape, 12.0)], axis=-1
    )
    np.testing.assert_allclose(block.Vxrt, expected_Vxrt, rtol=rtol)


def test_individual_velocity_setters_preserve_thermodynamics(block):
    """Test that individual velocity setters preserve internal energy."""
    rtol = 1e-6

    # Set up thermodynamic state first using scalar values
    block.set_P_T(101325.0, 300.0)

    # Set initial velocities to non-zero values to get baseline internal energy
    shape = block.shape
    Vxrt_initial = np.zeros(shape + (3,))
    Vxrt_initial[..., 0] = 5.0  # Initial Vx
    Vxrt_initial[..., 1] = 3.0  # Initial Vr
    Vxrt_initial[..., 2] = 4.0  # Initial Vt
    block.set_Vx(Vxrt_initial[..., 0])
    block.set_Vr(Vxrt_initial[..., 1])
    block.set_Vt(Vxrt_initial[..., 2])

    u_baseline = block.u.copy()
    rho_initial = block.rho.copy()

    # Change individual velocity components one at a time using scalar values
    # Each change should preserve internal energy from the current state
    block.set_Vx(20.0)
    u_after_Vx = block.u.copy()

    # Test that internal energy is preserved (accounting for the kinetic energy change)
    # When using individual setters, internal energy is preserved from previous state
    np.testing.assert_allclose(
        u_after_Vx,
        u_baseline,
        rtol=rtol,
        err_msg="Internal energy should be preserved after set_Vx",
    )

    # Update baseline for next test
    u_baseline = u_after_Vx.copy()

    block.set_Vr(10.0)
    u_after_Vr = block.u.copy()

    np.testing.assert_allclose(
        u_after_Vr,
        u_baseline,
        rtol=rtol,
        err_msg="Internal energy should be preserved after set_Vr",
    )

    # Update baseline for next test
    u_baseline = u_after_Vr.copy()

    block.set_Vt(15.0)
    u_after_Vt = block.u.copy()

    np.testing.assert_allclose(
        u_after_Vt,
        u_baseline,
        rtol=rtol,
        err_msg="Internal energy should be preserved after set_Vt",
    )

    # Density should remain unchanged throughout
    np.testing.assert_allclose(
        block.rho,
        rho_initial,
        rtol=rtol,
        err_msg="Density should be unchanged by velocity setters",
    )


def test_individual_setters_equivalence_with_combined(block):
    """Test that individual setters produce same result as combined setters."""
    rtol = 1e-6

    # Create second block to compare with fixture block
    shape = block.shape
    block2 = ember.block.Block(shape=shape)
    block2.set_fluid(block.fluid)

    # Test coordinates with fixture block's coordinates
    xrt_original = block.xrt.copy()
    x_vals = xrt_original[..., 0]
    r_vals = xrt_original[..., 1]
    t_vals = xrt_original[..., 2]

    # Set coordinates using individual methods on block2
    block2.set_x(x_vals)
    block2.set_r(r_vals)
    block2.set_t(t_vals)

    # Coordinates should be identical
    np.testing.assert_allclose(
        block.xrt,
        block2.xrt,
        rtol=rtol,
        err_msg="Individual coordinate setters should match combined setter",
    )

    # Set up identical thermodynamic state on both blocks using scalar values
    block.set_P_T(101325.0, 300.0)
    block2.set_P_T(101325.0, 300.0)

    # Test velocities - NOTE: Individual velocity setters may have different
    # thermodynamic behavior than individual setters, so we test them separately
    # Use scalar values for testing
    Vx_val, Vr_val, Vt_val = 25.0, 12.0, 18.0

    # Set initial zero velocities on both using scalar values
    block.set_Vx(np.zeros(shape))
    block.set_Vr(np.zeros(shape))
    block.set_Vt(np.zeros(shape))
    block2.set_Vx(0.0)
    block2.set_Vr(0.0)
    block2.set_Vt(0.0)

    # Now set the target velocities
    Vxrt_combined = np.stack(
        [np.full(shape, Vx_val), np.full(shape, Vr_val), np.full(shape, Vt_val)],
        axis=-1,
    )
    block.set_Vx(Vxrt_combined[..., 0])
    block.set_Vr(Vxrt_combined[..., 1])
    block.set_Vt(Vxrt_combined[..., 2])

    # For individual setters, we expect different thermodynamic behavior
    # but the velocities themselves should match
    block2.set_Vx(Vx_val)
    block2.set_Vr(Vr_val)
    block2.set_Vt(Vt_val)

    # Velocities should be identical
    np.testing.assert_allclose(block.Vx, block2.Vx, rtol=rtol)
    np.testing.assert_allclose(block.Vr, block2.Vr, rtol=rtol)
    np.testing.assert_allclose(block.Vt, block2.Vt, rtol=rtol)

    # Note: We don't compare thermodynamic properties because individual
    # setters preserve state at each step, while combined setter preserves
    # state based on initial velocities


def test_individual_coordinate_access_before_initialization():
    """Test that accessing individual coordinate properties raises errors when no coordinate setter has been called."""
    shape = (2, 3, 2)
    b = ember.block.Block(shape=shape)

    # Test: Accessing coordinate properties before any coordinate setter should raise errors
    with pytest.raises(
        ValueError, match="Data for variable x has not been initialised"
    ):
        _ = b.x

    with pytest.raises(
        ValueError, match="Data for variable r has not been initialised"
    ):
        _ = b.r

    with pytest.raises(
        ValueError, match="Data for variable t has not been initialised"
    ):
        _ = b.t

    # Test: After calling individual setters, properties should work
    b.set_x(1.0)
    assert b.x.shape == shape
    np.testing.assert_allclose(b.x, 1.0)

    # r and t should still raise errors
    with pytest.raises(
        ValueError, match="Data for variable r has not been initialised"
    ):
        _ = b.r

    with pytest.raises(
        ValueError, match="Data for variable t has not been initialised"
    ):
        _ = b.t

    # Set remaining coordinates individually
    b.set_r(1.5)
    b.set_t(0.3)

    # Now all should work
    assert b.r.shape == shape
    assert b.t.shape == shape
    np.testing.assert_allclose(b.r, 1.5)
    np.testing.assert_allclose(b.t, 0.3)


def test_individual_velocity_access_before_initialization():
    """Test that accessing individual velocity properties raises errors when no velocity setter has been called."""
    shape = (2, 3, 2)
    b = ember.block.Block(shape=shape)

    # Set up coordinates and thermodynamic state (required for velocity calculations)
    b.set_x(np.ones(shape))
    b.set_r(np.ones(shape))
    b.set_t(np.ones(shape))
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)
    b.set_P_T(101325.0, 300.0)

    # Test: Accessing velocity properties before any velocity setter should raise errors
    with pytest.raises(
        ValueError, match="Data for variable rhoVx has not been initialised"
    ):
        _ = b.Vx

    with pytest.raises(
        ValueError, match="Data for variable rhoVr has not been initialised"
    ):
        _ = b.Vr

    with pytest.raises(
        ValueError, match="Data for variable rhorVt has not been initialised"
    ):
        _ = b.Vt

    # Test: After calling individual setters, properties should work
    b.set_Vx(10.0)
    assert b.Vx.shape == shape
    np.testing.assert_allclose(b.Vx, 10.0)

    # Vr and Vt should still raise errors
    with pytest.raises(
        ValueError, match="Data for variable rhoVr has not been initialised"
    ):
        _ = b.Vr

    with pytest.raises(
        ValueError, match="Data for variable rhorVt has not been initialised"
    ):
        _ = b.Vt

    # Set remaining velocities individually
    b.set_Vr(5.0)
    b.set_Vt(8.0)

    # Now all should work
    assert b.Vr.shape == shape
    assert b.Vt.shape == shape
    np.testing.assert_allclose(b.Vr, 5.0)
    np.testing.assert_allclose(b.Vt, 8.0)


def test_coordinate_access_works_after_setters():
    """Test that individual coordinate access works after individual coordinate setters."""
    shape = (2, 3, 2)
    b = ember.block.Block(shape=shape)

    # Initialize all coordinates using individual setters
    xrt = np.zeros(shape + (3,))
    xrt[..., 0] = 1.5  # x
    xrt[..., 1] = 2.0  # r
    xrt[..., 2] = 0.5  # t
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Individual coordinate access should work
    np.testing.assert_allclose(b.x, 1.5)
    np.testing.assert_allclose(b.r, 2.0)
    np.testing.assert_allclose(b.t, 0.5)


def test_velocity_access_works_after_setters():
    """Test that individual velocity access works after individual velocity setters."""
    shape = (2, 3, 2)
    b = ember.block.Block(shape=shape)

    # Set up required state
    b.set_x(np.ones(shape))
    b.set_r(np.ones(shape))
    b.set_t(np.ones(shape))
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)
    b.set_P_T(101325.0, 300.0)

    # Initialize all velocities using individual setters
    Vxrt = np.zeros(shape + (3,))
    Vxrt[..., 0] = 15.0  # Vx
    Vxrt[..., 1] = 8.0  # Vr
    Vxrt[..., 2] = 12.0  # Vt
    b.set_Vx(Vxrt[..., 0])
    b.set_Vr(Vxrt[..., 1])
    b.set_Vt(Vxrt[..., 2])

    # Individual velocity access should work
    np.testing.assert_allclose(b.Vx, 15.0)
    np.testing.assert_allclose(b.Vr, 8.0)
    np.testing.assert_allclose(b.Vt, 12.0)


def test_set_conserved_args_variations(block):
    """Test that set_conserved works with single array only."""
    rtol = 1e-6

    # Define test conserved variables
    rho_vals = np.full(block.shape, 1.2, dtype=np.float32)
    rhoVx_vals = np.full(block.shape, 60.0, dtype=np.float32)
    rhoVr_vals = np.full(block.shape, 12.0, dtype=np.float32)
    rhorVt_vals = np.full(block.shape, 36.0, dtype=np.float32)  # rho * r * Vt
    rhoe_vals = np.full(block.shape, 300000.0, dtype=np.float32)

    conserved_array = np.stack(
        [rho_vals, rhoVx_vals, rhoVr_vals, rhorVt_vals, rhoe_vals], axis=-1
    )
    block.set_conserved(conserved_array)

    np.testing.assert_allclose(block.rho, rho_vals, rtol=rtol)
    np.testing.assert_allclose(block.rhoVx, rhoVx_vals, rtol=rtol)
    np.testing.assert_allclose(block.rhoVr, rhoVr_vals, rtol=rtol)
    np.testing.assert_allclose(block.rhorVt, rhorVt_vals, rtol=rtol)
    np.testing.assert_allclose(block.rhoe, rhoe_vals, rtol=rtol)
    np.testing.assert_allclose(block.conserved, conserved_array, rtol=rtol)


def test_conserved_nd_writable_in_place_modification_with_slice(block):
    """Test that block.conserved_nd allows in-place modification without copying."""
    rtol = 1e-6

    # Initialize block state first
    rho = np.full(block.shape, 1.2, dtype=np.float32)
    Vx = np.full(block.shape, 50.0, dtype=np.float32)
    Vr = np.full(block.shape, 10.0, dtype=np.float32)
    Vt = np.full(block.shape, 30.0, dtype=np.float32)
    P = np.full(block.shape, 101325.0, dtype=np.float32)
    block.set_P_rho(P, rho)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Store reference to original data array
    data_ref = block._data
    data_id = id(block._data)

    # Store original nondimensional conserved variables
    conserved_nd_orig = block.conserved_nd.copy()

    # Create test delta for a specific slice
    test_slice = (slice(1, 3), slice(2, 4), slice(None))
    delta = np.zeros(block.conserved_nd[test_slice].shape, dtype=np.float32)
    delta[..., 0] = 0.1  # Density increment
    delta[..., 1] = 5.0  # rhoVx increment
    delta[..., 4] = 1000.0  # Energy increment

    # Apply increment with slice
    block.conserved_nd[test_slice] += delta

    # Check that the same data array object is still being used (no copy)
    assert id(block._data) == data_id
    assert block._data is data_ref

    # Check that only the specified slice was modified
    np.testing.assert_allclose(
        block.conserved_nd[test_slice],
        conserved_nd_orig[test_slice] + delta,
        rtol=rtol,
    )

    # Check that other regions are unchanged
    mask = np.ones(block.shape, dtype=bool)
    mask[test_slice] = False
    unchanged = block.conserved_nd[mask]
    unchanged_orig = conserved_nd_orig[mask]

    np.testing.assert_allclose(unchanged, unchanged_orig, rtol=rtol)


def test_conserved_nd_writable_full_array(block):
    """Test that block.conserved_nd works for full array assignment (no slice)."""
    rtol = 1e-6

    # Initialize block state first
    rho = np.full(block.shape, 1.2, dtype=np.float32)
    Vx = np.full(block.shape, 50.0, dtype=np.float32)
    Vr = np.full(block.shape, 10.0, dtype=np.float32)
    Vt = np.full(block.shape, 30.0, dtype=np.float32)
    P = np.full(block.shape, 101325.0, dtype=np.float32)
    block.set_P_rho(P, rho)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Store original nondimensional conserved variables
    conserved_nd_orig = block.conserved_nd.copy()

    # Create test delta for full array
    delta = np.zeros(block.conserved_nd.shape, dtype=np.float32)
    delta[..., 0] = 0.05  # Density increment
    delta[..., 4] = 200.0  # Energy increment

    # Apply increment without slice (full array) - use [...] to modify in-place
    block.conserved_nd[...] += delta

    # Check that entire array was modified
    np.testing.assert_allclose(block.conserved_nd, conserved_nd_orig + delta, rtol=rtol)


def test_conserved_nd_writable_partial_variables(block):
    """Test block.conserved_nd writable behavior with conserved variable structure."""
    rtol = 1e-6

    # Initialize block state first
    rho = np.full(block.shape, 1.2, dtype=np.float32)
    Vx = np.full(block.shape, 50.0, dtype=np.float32)
    Vr = np.full(block.shape, 10.0, dtype=np.float32)
    Vt = np.full(block.shape, 30.0, dtype=np.float32)
    P = np.full(block.shape, 101325.0, dtype=np.float32)
    block.set_P_rho(P, rho)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Store original values (dimensional, with unity refs these equal nondim)
    rho_orig = block.rho.copy()
    rhoVx_orig = block.rhoVx.copy()
    rhoVr_orig = block.rhoVr.copy()
    rhorVt_orig = block.rhorVt.copy()
    rhoe_orig = block.rhoe.copy()

    # Create delta that modifies only some conserved variables
    test_slice = (slice(None), slice(None), 0)  # First theta plane
    delta = np.zeros(block.conserved_nd[test_slice].shape, dtype=np.float32)
    delta[..., 0] = 0.05  # Small density increment
    delta[..., 4] = 500.0  # Energy increment
    # Leave momentum components (indices 1,2,3) unchanged

    # Apply increment via conserved_nd
    block.conserved_nd[test_slice] += delta

    # Check that density and energy were modified (unity refs)
    np.testing.assert_allclose(
        block.rho[test_slice], rho_orig[test_slice] + 0.05, rtol=rtol
    )
    np.testing.assert_allclose(
        block.rhoe[test_slice], rhoe_orig[test_slice] + 500.0, rtol=rtol
    )

    # Check that momentum components were unchanged in the slice
    np.testing.assert_allclose(
        block.rhoVx[test_slice], rhoVx_orig[test_slice], rtol=rtol
    )
    np.testing.assert_allclose(
        block.rhoVr[test_slice], rhoVr_orig[test_slice], rtol=rtol
    )
    np.testing.assert_allclose(
        block.rhorVt[test_slice], rhorVt_orig[test_slice], rtol=rtol
    )

    # Check that other slices are completely unchanged
    other_slice = (slice(None), slice(None), slice(1, None))
    np.testing.assert_allclose(block.rho[other_slice], rho_orig[other_slice], rtol=rtol)
    np.testing.assert_allclose(
        block.rhoVx[other_slice], rhoVx_orig[other_slice], rtol=rtol
    )
    np.testing.assert_allclose(
        block.rhoVr[other_slice], rhoVr_orig[other_slice], rtol=rtol
    )
    np.testing.assert_allclose(
        block.rhorVt[other_slice], rhorVt_orig[other_slice], rtol=rtol
    )
    np.testing.assert_allclose(
        block.rhoe[other_slice], rhoe_orig[other_slice], rtol=rtol
    )


def test_conserved_nd_writable_performance_vs_copy():
    """Test that writable conserved_nd slice is more efficient than copy-based approach."""
    import time

    # Create larger block for performance testing
    shape = (50, 50, 20)
    block = ember.block.Block(shape=shape)

    # Set up coordinates
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set up fluid
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)

    # Set up primitive state
    rho = np.full(shape, 1.2, dtype=np.float32)
    Vx = np.full(shape, 50.0, dtype=np.float32)
    Vr = np.full(shape, 10.0, dtype=np.float32)
    Vt = np.full(shape, 30.0, dtype=np.float32)
    P = np.full(shape, 101325.0, dtype=np.float32)
    block.set_P_rho(P, rho)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Prepare test data
    test_slice = (slice(10, 40), slice(10, 40), slice(None))
    delta = np.ones(block.conserved_nd[test_slice].shape, dtype=np.float32) * 0.01

    # Time the in-place method (current implementation)
    start_time = time.time()
    for _ in range(10):
        block.conserved_nd[test_slice] += delta
    inplace_time = time.time() - start_time

    # Reset block state
    block.set_P_rho(P, rho)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Time the copy-based approach (old implementation)
    start_time = time.time()
    for _ in range(10):
        current = block.conserved_nd.copy()
        current[test_slice] += delta
        block.set_conserved(current)
    copy_time = time.time() - start_time

    # In-place method should be faster (though this is a rough test)
    # At minimum, it shouldn't be significantly slower
    assert inplace_time <= copy_time * 1.5, (
        f"In-place method took {inplace_time:.4f}s vs copy method {copy_time:.4f}s"
    )


def test_rothalpy_property(block):
    """Test rothalpy (I) property calculation."""
    rtol = 1e-6

    # Set up a rotating block
    Omega = 50.0  # rad/s
    block.set_Omega(Omega)

    # Set thermodynamic state
    shape = block.shape
    P = np.full(shape, 101325.0, dtype=np.float32)
    T = np.full(shape, 300.0, dtype=np.float32)
    block.set_P_T(P, T)

    # Set velocities with varying tangential component
    Vx = np.full(shape, 100.0, dtype=np.float32)
    Vr = np.full(shape, 50.0, dtype=np.float32)
    Vt = block.U * 0.5  # Half the blade speed
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Test rothalpy calculation: I = ho - U*Vt
    I_expected = block.ho - block.U * block.Vt
    I_actual = block.I

    np.testing.assert_allclose(
        I_actual, I_expected, rtol=rtol, err_msg="Rothalpy calculation incorrect"
    )

    # Test case with zero tangential velocity (no swirl)
    block.set_Vt(np.zeros(shape, dtype=np.float32))
    np.testing.assert_allclose(
        block.I, block.ho, rtol=rtol, err_msg="Rothalpy should equal ho when Vt=0"
    )

    # Test case with solid body rotation (Vt = U)
    block.set_Vt(block.U.copy())
    I_solid_body = block.ho - block.U**2
    np.testing.assert_allclose(
        block.I,
        I_solid_body,
        rtol=rtol,
        err_msg="Rothalpy incorrect for solid body rotation",
    )

    # Test alternative formulae for rothalpy
    # Reset to test case with varying velocities
    Vt_test = block.U * 0.3  # 30% of blade speed
    block.set_Vt(Vt_test)

    # Formula 1: I = ho - U*Vt (current implementation)
    I1 = block.I

    # Formula 2: I = ho_rel - U²/2
    I2 = block.ho_rel - 0.5 * block.U**2
    np.testing.assert_allclose(
        I1, I2, rtol=rtol, err_msg="Alternative formula I = ho_rel - U²/2 incorrect"
    )

    # Formula 3: I = h + V²/2 - U*Vt
    I3 = block.h + 0.5 * block.V**2 - block.U * block.Vt
    np.testing.assert_allclose(
        I1, I3, rtol=rtol, err_msg="Alternative formula I = h + V²/2 - U*Vt incorrect"
    )


def test_P_rot_no_rotation(block):
    """Test P_rot equals P when Omega = 0 (no rotation)."""
    rtol = 1e-6

    # Set up non-rotating block
    block.set_Omega(0.0)

    # Set thermodynamic state
    shape = block.shape
    P = np.full(shape, 101325.0, dtype=np.float32)
    T = np.full(shape, 300.0, dtype=np.float32)
    block.set_P_T(P, T)

    # Set velocities
    Vx = np.full(shape, 100.0, dtype=np.float32)
    Vr = np.full(shape, 50.0, dtype=np.float32)
    Vt = np.full(shape, 25.0, dtype=np.float32)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # When Omega = 0, U = 0, so P_rot should equal P
    P_rot = block.P_rot
    P_actual = block.P

    np.testing.assert_allclose(
        P_rot, P_actual, rtol=rtol, err_msg="P_rot should equal P when Omega = 0"
    )


def test_P_rot_reduces_pressure(block):
    """Test P_rot < P when Omega > 0 or Omega < 0 (rotation reduces pressure)."""

    # Set thermodynamic state
    shape = block.shape
    P = np.full(shape, 101325.0, dtype=np.float32)
    T = np.full(shape, 300.0, dtype=np.float32)

    # Set velocities
    Vx = np.full(shape, 100.0, dtype=np.float32)
    Vr = np.full(shape, 50.0, dtype=np.float32)
    Vt = np.full(shape, 25.0, dtype=np.float32)

    # Test with positive rotation
    block.set_Omega(100.0)
    block.set_P_T(P, T)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # With positive rotation, P_rot should be less than P
    P_rot_positive = block.P_rot
    P_actual_positive = block.P

    assert np.all(P_rot_positive < P_actual_positive), (
        "P_rot should be less than P when Omega > 0"
    )

    # Test with negative rotation
    block.set_Omega(-100.0)
    block.set_P_T(P, T)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # With negative rotation, P_rot should still be less than P (correction uses U^2)
    P_rot_negative = block.P_rot
    P_actual_negative = block.P

    assert np.all(P_rot_negative < P_actual_negative), (
        "P_rot should be less than P when Omega < 0"
    )

    # The magnitude of correction should be the same for +/- Omega
    rtol = 1e-6
    np.testing.assert_allclose(
        P_rot_positive,
        P_rot_negative,
        rtol=rtol,
        err_msg="P_rot should be the same for +/- Omega (correction uses U^2)",
    )


def test_P_rot_analytical_perfect_gas(block):
    """Test P_rot analytical formula for perfect gas."""
    rtol = 1e-5

    # Set up rotating block
    Omega = 50.0  # rad/s
    block.set_Omega(Omega)

    # Set thermodynamic state
    shape = block.shape
    P = np.full(shape, 101325.0, dtype=np.float32)
    T = np.full(shape, 300.0, dtype=np.float32)
    block.set_P_T(P, T)

    # Set velocities
    Vx = np.full(shape, 100.0, dtype=np.float32)
    Vr = np.full(shape, 50.0, dtype=np.float32)
    Vt = np.full(shape, 25.0, dtype=np.float32)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # For perfect gas, compute P_rot using isentropic relations
    # At constant entropy: T2/T1 = (P2/P1)^((gamma-1)/gamma)
    # Or equivalently: P2/P1 = (T2/T1)^(gamma/(gamma-1))
    # Where state 2 has h2 = h1 - 0.5*U^2 at constant s

    U = block.U
    cp = block.cp
    gamma = block.gamma

    # Get current state
    P_actual = block.P
    T_actual = block.T

    # Compute rotated temperature
    # For perfect gas: h = cp*T (+ constant datum)
    # Delta_h = cp * Delta_T
    # Delta_T = -0.5 * U^2 / cp
    T_rot = T_actual - 0.5 * U**2 / cp

    # Apply isentropic relation
    P_rot_expected = P_actual * (T_rot / T_actual) ** (gamma / (gamma - 1.0))
    P_rot_actual = block.P_rot

    np.testing.assert_allclose(
        P_rot_actual,
        P_rot_expected,
        rtol=rtol,
        err_msg="P_rot analytical formula incorrect for perfect gas",
    )


def test_P_rot_varying_radii(block):
    """Test P_rot with varying radii."""

    # Set up rotating block
    Omega = 100.0  # rad/s
    block.set_Omega(Omega)

    # Set up coordinates with varying radii
    shape = block.shape
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set uniform thermodynamic state
    P = np.full(shape, 101325.0, dtype=np.float32)
    T = np.full(shape, 300.0, dtype=np.float32)
    block.set_P_T(P, T)

    # Set velocities
    Vx = np.full(shape, 100.0, dtype=np.float32)
    Vr = np.full(shape, 50.0, dtype=np.float32)
    Vt = np.full(shape, 25.0, dtype=np.float32)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Get P_rot at different radii
    P_rot = block.P_rot
    U = block.U  # U = r * Omega

    # At larger radii, U is larger, so P_rot should be lower
    # Check this for the j-direction which varies in radius
    for i in range(shape[0]):
        for k in range(shape[2]):
            # Get P_rot along j-direction (varying radius)
            P_rot_slice = P_rot[i, :, k]
            U_slice = U[i, :, k]

            # Verify U increases with r
            assert np.all(np.diff(U_slice) > 0), "U should increase with r"

            # Verify P_rot decreases with r (since larger U means more correction)
            assert np.all(np.diff(P_rot_slice) < 0), (
                "P_rot should decrease with increasing radius"
            )


def test_P_rot_immutability(block):
    """Test P_rot doesn't modify original block."""
    rtol = 1e-6

    # Set up rotating block
    Omega = 50.0  # rad/s
    block.set_Omega(Omega)

    # Set thermodynamic state
    shape = block.shape
    P = np.full(shape, 101325.0, dtype=np.float32)
    T = np.full(shape, 300.0, dtype=np.float32)
    block.set_P_T(P, T)

    # Set velocities
    Vx = np.full(shape, 100.0, dtype=np.float32)
    Vr = np.full(shape, 50.0, dtype=np.float32)
    Vt = np.full(shape, 25.0, dtype=np.float32)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Store original values
    P_original = block.P.copy()
    h_original = block.h.copy()
    s_original = block.s.copy()
    T_original = block.T.copy()
    rho_original = block.rho.copy()

    # Call P_rot (triggers computation but we don't need the value)
    _ = block.P_rot

    # Verify original block is unchanged
    np.testing.assert_allclose(
        block.P, P_original, rtol=rtol, err_msg="P was modified by P_rot access"
    )
    np.testing.assert_allclose(
        block.h, h_original, rtol=rtol, err_msg="h was modified by P_rot access"
    )
    np.testing.assert_allclose(
        block.s, s_original, rtol=rtol, err_msg="s was modified by P_rot access"
    )
    np.testing.assert_allclose(
        block.T, T_original, rtol=rtol, err_msg="T was modified by P_rot access"
    )
    np.testing.assert_allclose(
        block.rho, rho_original, rtol=rtol, err_msg="rho was modified by P_rot access"
    )


def test_set_I_s_Ma_rel_Alpha_rel_Beta(block):
    """Test setting flow field using rothalpy, entropy, Mach number, and relative flow angles."""
    rtol = 1e-4

    # Set up a rotating block
    Omega = 50.0  # rad/s
    block.set_Omega(Omega)

    # Define flow field parameters
    shape = block.shape
    I = np.full(shape, 90000.0, dtype=np.float32)  # Constant rothalpy
    s = np.full(shape, 287.0, dtype=np.float32)  # Constant entropy
    Ma = np.full(shape, 0.5, dtype=np.float32)  # Constant Mach number
    Alpha_rel = np.zeros(shape, dtype=np.float32)  # No relative swirl
    Beta = np.full(shape, 90.0, dtype=np.float32)  # Pure radial flow

    # Set flow field
    ember.set_iter.set_I_s_Ma_rel_Alpha_rel_Beta(block, I, s, Ma, Alpha_rel, Beta)

    # Test rothalpy conservation
    np.testing.assert_allclose(block.I, I, rtol=rtol, err_msg="Rothalpy not conserved")

    # Test pure radial flow (Vx ≈ 0)
    np.testing.assert_allclose(
        block.Vx, 0, atol=1e-3, err_msg="Should be pure radial flow (Vx ≈ 0)"
    )

    # Test no relative swirl (Vt_rel ≈ 0)
    np.testing.assert_allclose(
        block.Vt_rel, 0, atol=1e-3, err_msg="Should have no relative swirl (Vt_rel ≈ 0)"
    )

    # Test axial flow case
    Beta_axial = np.zeros(shape, dtype=np.float32)  # Pure axial flow
    ember.set_iter.set_I_s_Ma_rel_Alpha_rel_Beta(block, I, s, Ma, Alpha_rel, Beta_axial)

    np.testing.assert_allclose(
        block.Vr, 0, atol=1e-3, err_msg="Should be pure axial flow (Vr ≈ 0)"
    )
    np.testing.assert_allclose(
        block.I, I, rtol=rtol, err_msg="Rothalpy not conserved in axial flow"
    )

    # Test flow with relative swirl
    Alpha_rel_swirl = np.full(shape, 30.0, dtype=np.float32)  # 30° relative swirl
    Beta_mixed = np.full(shape, 45.0, dtype=np.float32)  # 45° pitch angle

    ember.set_iter.set_I_s_Ma_rel_Alpha_rel_Beta(
        block, I, s, Ma, Alpha_rel_swirl, Beta_mixed
    )

    # Verify flow angles
    Alpha_rel_check = np.degrees(
        np.arctan2(block.Vt_rel, np.sqrt(block.Vx**2 + block.Vr**2))
    )
    Beta_check = np.degrees(np.arctan2(block.Vr, block.Vx))

    np.testing.assert_allclose(
        Alpha_rel_check,
        Alpha_rel_swirl,
        atol=1e-1,
        err_msg="Relative yaw angle not preserved",
    )
    np.testing.assert_allclose(
        Beta_check, Beta_mixed, atol=1e-1, err_msg="Pitch angle not preserved"
    )
    np.testing.assert_allclose(
        block.I, I, rtol=rtol, err_msg="Rothalpy not conserved with swirl"
    )


# ---------------------------------------------------------------------------
# Wall indicator property tests
# ---------------------------------------------------------------------------


def test_no_patches_all_walls(block):
    """Block with no patches has all-zero wall indicators (all walls)."""
    w = block.ijk_wall_conv
    assert np.all(w["walli1"] == 0.0)
    assert np.all(w["wallni"] == 0.0)
    assert np.all(w["wallj1"] == 0.0)
    assert np.all(w["wallnj"] == 0.0)
    assert np.all(w["wallk1"] == 0.0)
    assert np.all(w["wallnk"] == 0.0)


def test_inlet_patch_at_i0_is_free_surface(block):
    """InletPatch at i=0 marks walli1 as free surface (1.0), wallni stays wall (0.0)."""
    block.patches.append(InletPatch(i=0, j=(0, -1), k=(0, -1)))
    w = block.ijk_wall_conv
    assert np.all(w["walli1"] == 1.0)
    assert np.all(w["wallni"] == 0.0)


def test_outlet_patch_at_ini_is_free_surface(block):
    """OutletPatch at i=-1 marks wallni as free surface (1.0), walli1 stays wall (0.0)."""
    block.patches.append(OutletPatch(i=-1, j=(0, -1), k=(0, -1)))
    w = block.ijk_wall_conv
    assert np.all(w["wallni"] == 1.0)
    assert np.all(w["walli1"] == 0.0)


def test_periodic_patch_at_j0_is_free_surface(block):
    """PeriodicPatch at j=0 marks wallj1 as free surface (1.0), wallnj stays wall (0.0)."""
    block.patches.append(PeriodicPatch(i=(0, -1), j=0, k=(0, -1)))
    w = block.ijk_wall_conv
    assert np.all(w["wallj1"] == 1.0)
    assert np.all(w["wallnj"] == 0.0)


def test_mixing_patch_at_jnj_is_free_surface(block):
    """MixingPatch at j=-1 marks wallnj as free surface (1.0), wallj1 stays wall (0.0)."""
    block.patches.append(MixingPatch(i=(0, -1), j=-1, k=(0, -1)))
    w = block.ijk_wall_conv
    assert np.all(w["wallnj"] == 1.0)
    assert np.all(w["wallj1"] == 0.0)


def test_setter_roundtrips_nonunity_refs():
    """Roundtrip set-then-get with non-unity reference values."""
    rho_ref = 1.2
    V_ref = 300.0
    Rgas_ref = 287.0
    rtol = 1e-4

    shape = (2, 3, 4)
    b = ember.block.Block(shape=shape)
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=1e-5,
        Pr=0.72,
        rho_ref=rho_ref,
        V_ref=V_ref,
        Rgas_ref=Rgas_ref,
    )
    b.set_fluid(fluid)

    # Initial velocity state
    Vx0 = np.full(shape, 100.0, dtype=np.float32)
    b.set_rho_u(
        np.full(shape, 1.0, dtype=np.float32),
        np.full(shape, 200000.0, dtype=np.float32),
    )
    b.set_Vx(Vx0)

    # Target dimensional values
    P_want = np.full(shape, 101325.0, dtype=np.float32)
    T_want = np.full(shape, 300.0, dtype=np.float32)

    # --- set_rho_u ---
    rho_want = np.full(shape, 1.1, dtype=np.float32)
    u_want = np.full(shape, 215000.0, dtype=np.float32)
    b.set_rho_u(rho_want, u_want)
    np.testing.assert_allclose(b.rho, rho_want, rtol=rtol)
    np.testing.assert_allclose(b.u, u_want, rtol=rtol)

    # --- set_P_T ---
    b.set_P_T(P_want, T_want)
    np.testing.assert_allclose(b.P, P_want, rtol=rtol)
    np.testing.assert_allclose(b.T, T_want, rtol=rtol)

    # --- set_h_s ---
    h_want = b.h.copy()
    s_want = b.s.copy()
    b.set_rho_u(rho_want, u_want)  # perturb
    b.set_h_s(h_want, s_want)
    np.testing.assert_allclose(b.h, h_want, rtol=rtol)
    np.testing.assert_allclose(b.s, s_want, rtol=rtol)

    # --- set_T_s ---
    b.set_P_T(P_want, T_want)
    T_got = b.T.copy()
    s_got = b.s.copy()
    b.set_rho_u(rho_want, u_want)  # perturb
    b.set_T_s(T_got, s_got)
    np.testing.assert_allclose(b.T, T_got, rtol=rtol)
    np.testing.assert_allclose(b.s, s_got, rtol=rtol)

    # --- set_P_s ---
    b.set_P_T(P_want, T_want)
    P_got = b.P.copy()
    s_got = b.s.copy()
    b.set_rho_u(rho_want, u_want)  # perturb
    b.set_P_s(P_got, s_got)
    np.testing.assert_allclose(b.P, P_got, rtol=rtol)
    np.testing.assert_allclose(b.s, s_got, rtol=rtol)

    # --- set_P_h ---
    b.set_P_T(P_want, T_want)
    P_got = b.P.copy()
    h_got = b.h.copy()
    b.set_rho_u(rho_want, u_want)  # perturb
    b.set_P_h(P_got, h_got)
    np.testing.assert_allclose(b.P, P_got, rtol=rtol)
    np.testing.assert_allclose(b.h, h_got, rtol=rtol)

    # --- set_P_rho ---
    b.set_P_T(P_want, T_want)
    P_got = b.P.copy()
    rho_got = b.rho.copy()
    b.set_rho_u(
        np.full(shape, 0.9, dtype=np.float32),
        np.full(shape, 180000.0, dtype=np.float32),
    )  # perturb
    b.set_P_rho(P_got, rho_got)
    np.testing.assert_allclose(b.P, P_got, rtol=rtol)
    np.testing.assert_allclose(b.rho, rho_got, rtol=rtol)

    # --- set_rho_s ---
    b.set_P_T(P_want, T_want)
    rho_got = b.rho.copy()
    s_got = b.s.copy()
    b.set_rho_u(rho_want, u_want)  # perturb
    b.set_rho_s(rho_got, s_got)
    np.testing.assert_allclose(b.rho, rho_got, rtol=rtol)
    np.testing.assert_allclose(b.s, s_got, rtol=rtol)


# ---------------------------------------------------------------------------
# ao and Mam
# ---------------------------------------------------------------------------


def _block_with_velocity(Vx=100.0, Vr=20.0, Vt=50.0):
    """Return a small block with known velocity and thermodynamic state."""
    b = ember.block.Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)
    b.set_P_T(101325.0, 300.0)
    b.set_Vx(np.full(b.shape, Vx))
    b.set_Vr(np.full(b.shape, Vr))
    b.set_Vt(np.full(b.shape, Vt))
    return b


def test_ao_exceeds_static_a():
    """Stagnation speed of sound is greater than static when V > 0."""
    b = _block_with_velocity()
    assert np.all(b.ao > b.a)


def test_ao_equals_a_at_zero_velocity():
    """ao equals static a when velocity is zero."""
    b = _block_with_velocity(Vx=0.0, Vr=0.0, Vt=0.0)
    np.testing.assert_allclose(b.ao, b.a, rtol=1e-6)


def test_ao_consistent_with_To():
    """ao^2 / a^2 == To / T for a calorically perfect gas (gamma*R*T == a^2)."""
    b = _block_with_velocity()
    np.testing.assert_allclose(b.ao**2 / b.a**2, b.To / b.T, rtol=1e-5)


def test_Mam_definition():
    """Mam equals Vm / a."""
    b = _block_with_velocity()
    np.testing.assert_allclose(b.Mam, b.Vm / b.a, rtol=1e-6)


def test_Mam_equals_Max_for_axial_only():
    """With Vr=0 and Vt=0, meridional and axial Mach numbers are equal."""
    b = _block_with_velocity(Vx=100.0, Vr=0.0, Vt=0.0)
    np.testing.assert_allclose(b.Mam, b.Max, rtol=1e-6)


def test_Mam_zero_for_tangential_only():
    """Mam is zero when only tangential velocity is non-zero."""
    b = _block_with_velocity(Vx=0.0, Vr=0.0, Vt=100.0)
    np.testing.assert_allclose(b.Mam, np.zeros(b.shape), atol=1e-10)


def test_Vxrt_rel_no_rotation():
    """Vxrt_rel equals Vxrt when Omega is zero."""
    b = _block_with_velocity(Vx=100.0, Vr=20.0, Vt=50.0)
    # Default Omega is 0, so Vt_rel == Vt and Vxrt_rel must equal Vxrt
    np.testing.assert_allclose(b.Vxrt_rel, b.Vxrt, rtol=1e-6)


def test_Vxrt_rel_with_rotation():
    """Vxrt_rel tangential component equals Vt minus blade speed Omega*r."""
    Vx, Vr, Vt = 100.0, 20.0, 50.0
    Omega = 200.0  # rad/s
    b = _block_with_velocity(Vx=Vx, Vr=Vr, Vt=Vt)
    b.set_Omega(Omega)

    result = b.Vxrt_rel
    assert result.shape == b.shape + (3,)
    np.testing.assert_allclose(result[..., 0], b.Vx, rtol=1e-6)
    np.testing.assert_allclose(result[..., 1], b.Vr, rtol=1e-6)
    np.testing.assert_allclose(result[..., 2], b.Vt - Omega * b.r, rtol=1e-5)


def test_block_member_order():
    """Block members follow the standard ordering convention."""
    import pathlib
    from conftest import assert_class_member_order

    src = pathlib.Path(ember.block.__file__).read_text()
    assert_class_member_order(src, "Block")


def _exec_docstring_example(name):
    """Find the '::' code block tagged '# example: <name>' in the ember.block module
    docstring, exec it, assert all '# value' annotations against captured print output,
    and return (ns, raises) where raises is a list of (expr, exc_type) from
    '# raises ExcType' annotations."""
    import ember.block
    import textwrap
    import re
    import io
    import contextlib

    blocks = re.findall(r"::\n\n((?:    .+\n|\n)+)", ember.block.__doc__)
    for raw in blocks:
        code = textwrap.dedent(raw)
        first_line = code.lstrip().splitlines()[0]
        if first_line.strip() != f"# example: {name}":
            continue

        raises = []
        expected_outputs = []  # expected string per printed line
        clean_lines = []
        for line in code.splitlines():
            m_raises = re.match(r"\s*print\((.+?)\)\s*#\s*raises\s+(\w+)", line)
            m_value = re.match(r"\s*print\(.+?\)\s*#\s*(\S.*)", line)
            if m_raises:
                raises.append((m_raises.group(1), m_raises.group(2)))
            elif m_value:
                expected_outputs.append(m_value.group(1))
                clean_lines.append(line)
            else:
                clean_lines.append(line)

        ns = {"np": np}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec("\n".join(clean_lines), ns)  # noqa: S102
        actual_outputs = [line for line in buf.getvalue().splitlines() if line.strip()]

        assert len(actual_outputs) == len(expected_outputs), (
            f"example '{name}': {len(actual_outputs)} printed lines but "
            f"{len(expected_outputs)} annotated"
        )

        def _parse(s):
            # numpy prints arrays without commas; restore them so eval works
            import re as _re

            s = _re.sub(r"(?<=[\d.])\s+(?=[-\d.])", ", ", s)
            return np.array(eval(s, {"np": np}))  # noqa: S307

        for actual_str, expected_str in zip(actual_outputs, expected_outputs):
            actual = _parse(actual_str)
            expected = _parse(expected_str)
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"printed {actual_str!r}, expected {expected_str!r}",
            )

        exc_map = {
            "ValueError": ValueError,
            "TypeError": TypeError,
            "AttributeError": AttributeError,
        }
        for expr_str, exc_name in raises:
            with pytest.raises(exc_map[exc_name]):
                eval(expr_str, ns)  # noqa: S307

        return ns
    raise KeyError(f"No docstring example named {name!r}")


def test_block_example_construct():
    """Module docstring 'construct': set fluid/coords/state/velocity."""
    _exec_docstring_example("construct")


def test_block_example_indexing():
    """Module docstring 'indexing': indexing and slicing."""
    _exec_docstring_example("indexing")


def test_block_example_copy():
    """Module docstring 'copy': copy() decouples backing array."""
    _exec_docstring_example("copy")


def _masked_block_1d():
    """1D block with uniform 300 K state and Vx=5, for masked() tests."""
    b = ember.block.Block((4,))
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    b.set_fluid(fluid)
    b.set_x(np.linspace(0.0, 0.5, 4))
    b.set_r(1.0)
    b.set_t(0.0)
    b.set_P_T(1e5, 300.0)
    b.set_Vx(5.0)
    b.set_Vr(0.0)
    b.set_Vt(0.0)
    return b


def test_masked_setter_confined_to_mask():
    """masked().set_P_T only changes nodes where the mask is True."""
    b = _masked_block_1d()
    mask = np.array([True, False, True, False])

    b.masked(mask).set_P_T(1e5, 600.0)

    assert np.allclose(b.T, [600.0, 300.0, 600.0, 300.0])


def test_masked_setter_preserves_other_state():
    """A thermodynamic setter under masked() leaves velocity untouched everywhere."""
    b = _masked_block_1d()
    mask = np.array([True, False, True, False])

    b.masked(mask).set_P_T(1e5, 600.0)

    assert np.allclose(b.Vx, 5.0)
    assert np.allclose(b.P, 1e5)


def test_masked_reused_proxy_applies_same_mask():
    """Calling multiple setters on the same masked() proxy applies the same mask each time."""
    b = _masked_block_1d()
    mask = np.array([True, False, True, False])

    out = b.masked(mask)
    out.set_P_T(1e5, 400.0)
    out.set_Vx(50.0)

    assert np.allclose(b.T, [400.0, 300.0, 400.0, 300.0])
    assert np.allclose(b.Vx, [50.0, 5.0, 50.0, 5.0])
    assert out._block is b


def test_masked_2d():
    """masked() works with a multidimensional boolean mask."""
    b = ember.block.Block((3, 2))
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    b.set_fluid(fluid)
    b.set_x(np.zeros((3, 2)))
    b.set_r(1.0)
    b.set_t(0.0)
    b.set_P_T(1e5, 300.0)
    b.set_Vx(0.0)
    b.set_Vr(0.0)
    b.set_Vt(0.0)

    mask = np.array([[True, False], [False, True], [True, False]])
    b.masked(mask).set_P_T(1e5, 400.0)

    expected = np.array([[400.0, 300.0], [300.0, 400.0], [400.0, 300.0]])
    assert np.allclose(b.T, expected)


def test_masked_on_slice_writes_through_to_parent():
    """block[i].masked(mask) updates only masked nodes of the parent slice."""
    b = ember.block.Block((4, 3, 2))
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    b.set_fluid(fluid)
    b.set_x(0.0)
    b.set_r(1.0)
    b.set_t(0.0)
    b.set_P_T(1e5, 300.0)
    b.set_Vx(5.0)
    b.set_Vr(0.0)
    b.set_Vt(0.0)

    mask = np.array([[True, False], [False, True], [True, False]])
    b[0].masked(mask).set_P_T(1e5, 900.0)

    # Masked nodes of slice 0 updated, others in slice 0 unchanged.
    expected0 = np.array([[900.0, 300.0], [300.0, 900.0], [900.0, 300.0]])
    assert np.allclose(b.T[0], expected0)
    # Other slices are completely untouched.
    assert np.allclose(b.T[1:], 300.0)
    # Velocity preserved by the thermodynamic setter.
    assert np.allclose(b.Vx, 5.0)


def test_masked_shape_mismatch_raises():
    """A mask whose shape differs from the block shape is rejected."""
    b = _masked_block_1d()
    with pytest.raises(ValueError, match="does not match block shape"):
        b.masked(np.array([True, False]))


def test_masked_forwards_non_setter_attributes():
    """Non-setter attribute access on the proxy is forwarded to the block."""
    b = _masked_block_1d()
    proxy = b.masked(np.array([True, False, True, False]))

    assert proxy.shape == b.shape
    assert np.allclose(proxy.T, b.T)


def test_masked_docstring_example():
    """The masked() method docstring example runs and matches its annotations."""
    import ember.block
    import inspect
    import textwrap
    import re
    import io
    import contextlib

    # inspect.cleandoc (not raw __doc__) so this is consistent regardless of
    # Python version: 3.13+ dedents docstrings at compile time, 3.11/3.12
    # don't, so __doc__'s indentation otherwise depends on interpreter version.
    doc = inspect.cleandoc(ember.block.Block.masked.__doc__)
    blocks = re.findall(r"::\n\n((?:    .+\n|\n)+)", doc)
    for raw in blocks:
        code = textwrap.dedent(raw)
        if code.lstrip().splitlines()[0].strip() != "# example: masked":
            continue

        expected_outputs = []
        clean_lines = []
        for line in code.splitlines():
            m_value = re.match(r"\s*print\(.+?\)\s*#\s*(\S.*)", line)
            if m_value:
                expected_outputs.append(m_value.group(1))
            clean_lines.append(line)

        ns = {"np": np}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec("\n".join(clean_lines), ns)  # noqa: S102
        actual = [ln for ln in buf.getvalue().splitlines() if ln.strip()]

        assert len(actual) == len(expected_outputs)
        for got, want in zip(actual, expected_outputs):
            assert got.strip() == want.strip(), f"got {got!r}, want {want!r}"
        return

    raise AssertionError("masked docstring example not found")


def test_stagnation_props_require_velocity():
    """Every public stagnation property (o / o_rel) must raise if velocity is unset.

    Stagnation quantities include kinetic energy by definition, so they are
    undefined until a velocity has been set. The set of properties is
    discovered dynamically: any public property whose name ends in the
    stagnation marker ``o`` (optionally followed by ``_rel`` and/or ``_nd``)
    is covered. The test is a tripwire -- adding a new such property forces a
    conscious choice to either guard it against uninitialised velocity or list
    it in NON_STAGNATION below.
    """
    import re

    stagnation = re.compile(r"o(?:_rel)?(?:_nd)?$")
    # Density and derivatives held at constant density also end in 'o' (the 'o'
    # of "rho"), but are not stagnation quantities and do not depend on velocity.
    # tau_q_halo ends in 'o' (the 'o' of "halo") but is pure transient scratch.
    NON_STAGNATION = {
        "rho",
        "rho_nd",
        "dhdP_rho_nd",
        "dsdP_rho_nd",
        "dudP_rho_nd",
        "tau_q_halo",
    }
    names = sorted(
        n
        for n in dir(ember.block.Block)
        if not n.startswith("_")
        and stagnation.search(n)
        and n not in NON_STAGNATION
        and isinstance(getattr(ember.block.Block, n), property)
    )
    assert names, "introspection found no stagnation properties"

    # Build a block with coordinates and thermodynamic state but NO velocity,
    # so the only uninitialised data is the momentum (velocity) field.
    shape = (2, 2, 2)
    b = ember.block.Block(shape=shape)
    b.set_fluid(ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72))
    b.set_x(np.zeros(shape))
    b.set_r(np.ones(shape))
    b.set_t(np.zeros(shape))
    b.set_P_T(1e5, 300.0)

    offenders = []
    for name in names:
        try:
            getattr(b, name)
        except ValueError as exc:
            if not re.search(r"rhoV[xr]|rhorVt", str(exc)):
                offenders.append(f"{name}: raised unexpected error {exc!r}")
        else:
            offenders.append(f"{name}: did not raise on uninitialised velocity")

    assert not offenders, "stagnation properties not guarding velocity:\n" + "\n".join(
        offenders
    )


def test_conserved_does_not_mutate(block):
    """Reading .conserved is pure: repeated reads agree and raw data is intact.

    .conserved rescales the writeable conserved_nd view into a fresh array; a
    regression that wrote back into the view would corrupt the flow field.
    """
    block.set_P_T(1e5, 300.0)
    block.set_Vx(50.0)
    block.set_Vr(10.0)
    block.set_Vt(30.0)

    nd_before = block.conserved_nd.copy()

    first = block.conserved.copy()
    second = block.conserved

    np.testing.assert_array_equal(
        first, second, err_msg="repeated .conserved reads disagree"
    )
    np.testing.assert_array_equal(
        block.conserved_nd,
        nd_before,
        err_msg=".conserved mutated the underlying conserved_nd data",
    )
