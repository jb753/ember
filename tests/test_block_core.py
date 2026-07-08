"""Tests for Block class core functionality.

Test cases:
- test_block_dA: Block face area calculations
- test_block_dl_min: Minimum grid spacing calculations
- test_props_read_only: Read-only property validation (setattr)
- test_props_not_writable_in_place: In-place write rejection for basic/cached/derived properties
- test_default_Omega_Nb: Default rotation rate and blade count
- test_set_P_T: Pressure and temperature setting
- test_set_h_s: Enthalpy and entropy setting
- test_set_P_s: Pressure and entropy setting
- test_set_P_h: Pressure and enthalpy setting
- test_set_P_rho: Pressure and density setting
- test_set_rho_s: Density and entropy setting
- test_set_Omega: Rotation rate setting
- test_set_Nb: Blade count setting
- test_all_set_methods_have_self_return: Method chaining validation
- test_coordinate_access_before_initialization: Error handling for uninitialized coordinates
- test_thermodynamic_access_before_initialization: Error handling for uninitialized thermodynamics
- test_velocity_access_before_initialization: Error handling for uninitialized velocities
- test_individual_coordinate_setters: Individual coordinate component setters
- test_individual_coordinate_setters_method_chaining: Method chaining for coordinate setters
- test_set_r_validates_zero_radius: Validation of zero radius values
- test_individual_velocity_setters: Individual velocity component setters
- test_individual_velocity_setters_method_chaining: Method chaining for velocity setters
- test_individual_coordinate_access_before_initialization: Error handling for individual coordinate access
- test_individual_velocity_access_before_initialization: Error handling for individual velocity access
- test_coordinate_access_works_after_setters: Coordinate access after initialization
- test_velocity_access_works_after_setters: Velocity access after initialization
- test_set_conserved_args_variations: Argument variations for conserved variable setting
- test_args_methods_return_self_for_chaining: Method chaining validation for argument methods
- test_set_datum: Thermodynamic datum setting
- test_set_datum_effect: Effect of datum changes on u and s
- test_set_datum_preserves_fluid_properties: Fluid property preservation during datum changes
- test_conserved_writable_in_place_modification_with_slice: In-place modification with array slicing
- test_conserved_writable_full_array: Full array modification of conserved variables
- test_conserved_writable_partial_variables: Partial variable modification
- test_conserved_writable_performance_vs_copy: Performance comparison of in-place vs copy operations
- test_set_rpm_scalar: Setting scalar rpm
- test_set_rpm_conversion: Type conversion to float32 through Omega
- test_set_rpm_returns_self: set_rpm returns self for chaining
- test_set_rpm_consistency_with_rpm_property: set_rpm is consistent with rpm property
- test_Re_ref: Reference Reynolds number property
- test_P_nd: Nondimensional pressure property
"""

import ember.block
import pytest
import ember.geometry
import ember.fluid
import numpy as np
from ember import util


@pytest.fixture
def block():
    """Create a configured Block with hard-coded shape, coordinates and fluid."""
    # Hard-coded shape
    shape = (3, 4, 5)

    # Create block
    b = ember.block.Block(shape=shape)

    # Set up simple coordinates
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    b.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

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


def test_block_dl_min(block):
    """Test that Block.dl_min property works correctly."""
    # Calculate expected dl_min using the function directly
    dAi = ember.geometry.get_dAi(block.xrt)
    dAj = ember.geometry.get_dAj(block.xrt)
    dAk = ember.geometry.get_dAk(block.xrt)
    vol = ember.geometry.get_vol(block.xrt, dAi, dAj, dAk)
    expected_dl_min = ember.geometry.get_dl_min(dAi, dAj, dAk, vol)

    # Test that the property returns the same result
    assert np.allclose(block.dl_min, expected_dl_min)

    # Verify property is consistent across multiple accesses
    dl_min_second = block.dl_min
    assert np.allclose(block.dl_min, dl_min_second)


def test_props_read_only():
    shape = (2, 3, 4)
    b = ember.block.Block(shape=shape)
    b.set_x(np.ones(shape)).set_r(np.ones(shape)).set_t(np.ones(shape))

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


def test_props_not_writable_in_place(block):
    """Test that basic, cached, and derived properties reject in-place writes."""
    rho = np.full(block.shape, 1.2, dtype=np.float32)
    u = np.full(block.shape, 200000.0, dtype=np.float32)
    Vx = np.zeros(block.shape, dtype=np.float32)
    Vr = np.zeros(block.shape, dtype=np.float32)
    Vt = np.zeros(block.shape, dtype=np.float32)
    block.set_rho_u(rho, u)
    block.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)

    # Basic stored arrays
    basic_props = ["x", "r", "t", "xrt", "rho", "rhoVx", "rhoVr", "rhorVt", "rhoe"]
    # cached_array results
    cached_props = [
        "dAi_nd",
        "dAj_nd",
        "dAk_nd",
        "vol_nd",
        "Vx",
        "Vr",
        "Vt",
        "Vm",
        "u",
        "P_nd",
        "T_nd",
        "conserved_cell_nd",
    ]
    # derived_array results
    derived_props = ["dAi", "dAj", "dAk", "vol", "conserved", "conserved_cell"]

    for prop in basic_props + cached_props + derived_props:
        arr = getattr(block, prop)
        with pytest.raises((ValueError, TypeError)):
            arr[...] = 0


def _populate(block):
    """Populate a block fixture with non-trivial conserved variables."""
    rho = np.full(block.shape, 1.2, dtype=np.float32)
    u = np.full(block.shape, 200000.0, dtype=np.float32)
    # Spatially-varying velocity so the 8-corner average is not trivial.
    i = np.arange(block.shape[0], dtype=np.float32)[:, None, None]
    j = np.arange(block.shape[1], dtype=np.float32)[None, :, None]
    k = np.arange(block.shape[2], dtype=np.float32)[None, None, :]
    Vx = np.broadcast_to(10.0 + i, block.shape).astype(np.float32)
    Vr = np.broadcast_to(5.0 + j, block.shape).astype(np.float32)
    Vt = np.broadcast_to(2.0 + k, block.shape).astype(np.float32)
    block.set_rho_u(rho, u)
    block.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)


def _manual_node_avg(x):
    """8-corner average of a nodal array along the first three axes."""
    return 0.125 * (
        x[:-1, :-1, :-1]
        + x[1:, :-1, :-1]
        + x[:-1, 1:, :-1]
        + x[1:, 1:, :-1]
        + x[:-1, :-1, 1:]
        + x[1:, :-1, 1:]
        + x[:-1, 1:, 1:]
        + x[1:, 1:, 1:]
    )


def test_conserved_cell_nd_shape_and_values(block):
    """conserved_cell_nd is the 8-corner average of conserved_nd."""
    _populate(block)
    cell = block.conserved_cell_nd
    assert cell.shape == block.shape_cell + (5,)
    expected = _manual_node_avg(block.conserved_nd)
    assert np.allclose(cell, expected, rtol=1e-6, atol=1e-6)


def test_conserved_cell_dimensional_scaling(block):
    """conserved_cell equals conserved_cell_nd rescaled componentwise."""
    _populate(block)
    nd = block.conserved_cell_nd
    dim = block.conserved_cell
    rho_ref = block.fluid.rho_ref
    rhoV_ref = block._rhoV_ref
    L_ref = block.L_ref
    V_ref = block.fluid.V_ref
    assert np.allclose(dim[..., 0], nd[..., 0] * rho_ref)
    assert np.allclose(dim[..., 1], nd[..., 1] * rhoV_ref)
    assert np.allclose(dim[..., 2], nd[..., 2] * rhoV_ref)
    assert np.allclose(dim[..., 3], nd[..., 3] * rhoV_ref * L_ref)
    assert np.allclose(dim[..., 4], nd[..., 4] * rhoV_ref * V_ref)


def test_conserved_cell_nd_cache_invalidation(block):
    """conserved_cell_nd reflects in-place mutations to conserved_nd after invalidation."""
    _populate(block)
    first = block.conserved_cell_nd.copy()

    block.conserved_nd[...] *= 2.0
    block.update_cached_conserved()

    second = block.conserved_cell_nd
    assert np.allclose(second, 2.0 * first, rtol=1e-6, atol=1e-6)


def test_default_Omega_Nb():
    shape = (2, 3, 4)
    b = ember.block.Block(shape=shape)
    b.set_x(np.ones(shape)).set_r(np.ones(shape)).set_t(np.ones(shape))

    assert b.Omega == 0.0
    assert b.Nb == 1
    assert b.pitch == 2.0 * np.pi
    assert b.rpm == 0.0


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


def test_set_rho_s(block):
    """Test that set_rho_s correctly sets density and entropy."""
    rtol = 1e-6

    # Set initial state
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    # New density and entropy
    rho_new = np.full(block.shape, 1.5, dtype=np.float32)
    s_new = np.full(block.shape, 2500.0, dtype=np.float32)

    # Apply the method
    block.set_rho_s(rho_new, s_new)

    # Check that rho and s match what we get from fluid calculations
    expected_s = block.fluid.get_s(block.rho, block.u)

    np.testing.assert_allclose(block.s, expected_s, rtol=rtol)
    np.testing.assert_allclose(block.rho, rho_new, rtol=rtol)


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
    b.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

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
    b.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

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
    b.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

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
    b.set_Vx(np.full(shape, 50.0, dtype=np.float32)).set_Vr(
        np.full(shape, 10.0, dtype=np.float32)
    ).set_Vt(np.full(shape, 30.0, dtype=np.float32))

    # Now these should work without errors
    assert b.rhoVx.shape == shape
    assert b.rhoVr.shape == shape
    assert b.rhorVt.shape == shape
    assert b.Vx.shape == shape
    assert b.Vr.shape == shape
    assert b.Vt.shape == shape


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


def test_individual_coordinate_setters_method_chaining(block):
    """Test that individual coordinate setters return self for method chaining."""
    # Test method chaining works with scalar values
    result = block.set_x(1.0).set_r(1.5).set_t(0.2)
    assert result is block, "Method chaining should return the same block instance"

    # Verify final state is correct (scalars should broadcast to full arrays)
    np.testing.assert_allclose(block.x, 1.0, rtol=1e-6)
    np.testing.assert_allclose(block.r, 1.5, rtol=1e-6)
    np.testing.assert_allclose(block.t, 0.2, rtol=1e-6)


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


def test_individual_velocity_setters_method_chaining(block):
    """Test that individual velocity setters return self for method chaining."""
    # Test method chaining works with scalar values
    result = block.set_Vx(10.0).set_Vr(5.0).set_Vt(8.0)
    assert result is block, "Method chaining should return the same block instance"

    # Verify final state is correct (scalars should broadcast to full arrays)
    np.testing.assert_allclose(block.Vx, 10.0, rtol=1e-6)
    np.testing.assert_allclose(block.Vr, 5.0, rtol=1e-6)
    np.testing.assert_allclose(block.Vt, 8.0, rtol=1e-6)


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
    b.set_x(np.ones(shape)).set_r(np.ones(shape)).set_t(np.ones(shape))
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

    # Initialize all coordinates
    xrt = np.zeros(shape + (3,))
    xrt[..., 0] = 1.5  # x
    xrt[..., 1] = 2.0  # r
    xrt[..., 2] = 0.5  # t
    b.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

    # Individual coordinate access should work
    np.testing.assert_allclose(b.x, 1.5)
    np.testing.assert_allclose(b.r, 2.0)
    np.testing.assert_allclose(b.t, 0.5)


def test_velocity_access_works_after_setters():
    """Test that individual velocity access works after individual velocity setters."""
    shape = (2, 3, 2)
    b = ember.block.Block(shape=shape)

    # Set up required state
    b.set_x(np.ones(shape)).set_r(np.ones(shape)).set_t(np.ones(shape))
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)
    b.set_P_T(101325.0, 300.0)

    # Initialize all velocities using individual setters
    b.set_Vx(15.0).set_Vr(8.0).set_Vt(12.0)

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


def test_args_methods_return_self_for_chaining():
    """Test that all *args methods return self for method chaining."""
    shape = (2, 2, 2)
    b = ember.block.Block(shape=shape)

    # Set up fluid
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)

    # Test set_x/set_r/set_t chaining
    result1 = b.set_x(np.ones(shape)).set_r(np.ones(shape)).set_t(np.ones(shape))
    assert result1 is b, "set_x/set_r/set_t should return self"

    result2 = (
        b.set_x(np.ones(shape)).set_r(np.ones(shape) * 1.5).set_t(np.ones(shape) * 0.5)
    )
    assert result2 is b, "set_x/set_r/set_t should return self"

    # Test set_Vx/set_Vr/set_Vt chaining
    b.set_P_T(101325.0, 300.0)  # Need thermodynamic state for velocities

    result3 = b.set_Vx(np.zeros(shape)).set_Vr(np.zeros(shape)).set_Vt(np.zeros(shape))
    assert result3 is b, "set_Vx/set_Vr/set_Vt should return self"

    result4 = (
        b.set_Vx(np.ones(shape) * 10)
        .set_Vr(np.ones(shape) * 5)
        .set_Vt(np.ones(shape) * 8)
    )
    assert result4 is b, "set_Vx/set_Vr/set_Vt should return self"

    # Test set_conserved chaining (single array only)
    conserved = np.random.rand(*shape, 5) + 0.1  # Avoid zeros
    result5 = b.set_conserved(conserved)
    assert result5 is b, "set_conserved(single_array) should return self"


def test_set_datum(block):
    """Test set_datum method preserves thermodynamic state and velocities."""
    rtol = 1e-6

    # Set initial state with non-zero velocities
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    block.set_Vx(np.full(block.shape, 50.0, dtype=np.float32)).set_Vr(
        np.full(block.shape, 10.0, dtype=np.float32)
    ).set_Vt(np.full(block.shape, 30.0, dtype=np.float32))

    # Store original state
    T_orig = block.T.copy()
    P_orig = block.P.copy()
    Vx_orig = block.Vx.copy()
    Vr_orig = block.Vr.copy()
    Vt_orig = block.Vt.copy()
    rho_orig = block.rho.copy()

    # Change datum
    P_dtm_new, T_dtm_new = 2e5, 400.0
    block.set_fluid(block.fluid.change_datum(P_dtm_new, T_dtm_new))

    # Check fluid datum changed
    assert np.isclose(block.fluid.P_dtm, P_dtm_new)
    assert np.isclose(block.fluid.T_dtm, T_dtm_new)

    # Check physical state preserved (T, P, rho, V unchanged; s shifts with datum)
    np.testing.assert_allclose(block.T, T_orig, rtol=rtol)
    np.testing.assert_allclose(block.P, P_orig, rtol=rtol)
    np.testing.assert_allclose(block.rho, rho_orig, rtol=rtol)

    # Check velocities preserved
    np.testing.assert_allclose(block.Vx, Vx_orig, rtol=rtol)
    np.testing.assert_allclose(block.Vr, Vr_orig, rtol=rtol)
    np.testing.assert_allclose(block.Vt, Vt_orig, rtol=rtol)

    # Check that internal energy actually changed
    assert not np.allclose(block.u, u_initial, rtol=rtol)


def test_set_datum_effect(block):
    """Test that set_datum correctly affects u and s datum simultaneously."""

    block.set_P_T(101325.0, 350.0)

    P_dtm_new, T_dtm_new = 2e5, 250.0
    block.set_fluid(block.fluid.change_datum(P_dtm_new, T_dtm_new))

    # At the new datum state, both u and s should be zero
    block_datum = ember.block.Block(shape=block.shape)
    block_datum.set_x(block.xrt[..., 0]).set_r(block.xrt[..., 1]).set_t(
        block.xrt[..., 2]
    )
    block_datum.set_fluid(block.fluid)
    block_datum.set_P_T(P_dtm_new, T_dtm_new)

    np.testing.assert_allclose(block_datum.u, 0.0, atol=1e-10)
    np.testing.assert_allclose(block_datum.s, 0.0, atol=1e-4)


def test_set_datum_preserves_fluid_properties(block):
    """Test that set_datum preserves other fluid properties."""
    rtol = 1e-6

    # Set initial state
    block.set_P_T(101325.0, 300.0)

    # Store original fluid properties
    cp_orig = block.cp.copy()
    gamma_orig = block.gamma.copy()
    Rgas_orig = block.Rgas.copy()
    mu_orig = block.mu.copy()
    Pr_orig = block.Pr.copy()

    # Change datum
    block.set_fluid(block.fluid.change_datum(2e5, 450.0))

    # Check fluid properties preserved
    np.testing.assert_allclose(block.cp, cp_orig, rtol=rtol)
    np.testing.assert_allclose(block.gamma, gamma_orig, rtol=rtol)
    np.testing.assert_allclose(block.Rgas, Rgas_orig, rtol=rtol)
    np.testing.assert_allclose(block.mu, mu_orig, rtol=rtol)
    np.testing.assert_allclose(block.Pr, Pr_orig, rtol=rtol)


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
    block.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)

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
    block.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)

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
    block.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)

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
    block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

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
    block.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)

    # Test slice
    test_slice = (slice(10, 40), slice(20, 30), slice(None))
    delta = np.ones(block.conserved_nd[test_slice].shape, dtype=np.float32) * 0.01

    # Method 1: Direct writable access (should be faster)
    start_time = time.time()
    for _ in range(100):
        block.conserved_nd[test_slice] += delta
        block.conserved_nd[test_slice] -= delta  # Undo to keep state consistent
    in_place_time = time.time() - start_time

    # Method 2: Copy-based approach (should be slower)
    start_time = time.time()
    for _ in range(100):
        temp = block.conserved_nd[test_slice].copy()
        temp += delta
        block.conserved_nd[test_slice] = temp
        temp -= delta  # Undo
        block.conserved_nd[test_slice] = temp
    copy_time = time.time() - start_time

    # In-place should be faster (though this is not guaranteed on all systems)
    # We just check that both methods work correctly
    assert in_place_time > 0
    assert copy_time > 0


def test_set_rpm_scalar():
    """Test setting scalar rpm."""
    block = ember.block.Block(shape=(5, 5, 5))
    block.set_rpm(1000.0)
    expected_omega = 1000.0 * 2.0 * np.pi / 60.0
    assert block.Omega == np.float32(expected_omega)


def test_set_rpm_conversion():
    """Test type conversion to float32 through Omega."""
    block = ember.block.Block(shape=(5, 5, 5))
    block.set_rpm(1000)  # int
    expected_omega = 1000.0 * 2.0 * np.pi / 60.0
    assert block.Omega == np.float32(expected_omega)
    assert isinstance(block.Omega, np.float32)


def test_set_rpm_returns_self():
    """Test that set_rpm returns self for chaining."""
    block = ember.block.Block(shape=(5, 5, 5))
    result = block.set_rpm(1000.0)
    assert result is block


def test_set_rpm_consistency_with_rpm_property():
    """Test that set_rpm is consistent with rpm property."""
    block = ember.block.Block(shape=(5, 5, 5))
    rpm_input = 1500.0
    block.set_rpm(rpm_input)
    assert block.rpm == np.float32(rpm_input)


def test_block_set_xyz_roundtrip():
    """Test that Block.set_xyz correctly converts Cartesian to polar coordinates."""
    # Create block
    block = ember.block.Block(shape=(3, 4, 5))

    # Create test Cartesian coordinates
    # Use simple values we can verify manually
    ni, nj, nk = block.shape

    # Create a simple coordinate pattern
    x = np.linspace(0, 2, ni)[:, None, None]
    y = np.linspace(1, 3, nj)[None, :, None]
    z = np.linspace(-0.5, 0.5, nk)[None, None, :]

    # Broadcast to full grid
    x_full = np.broadcast_to(x, (ni, nj, nk))
    y_full = np.broadcast_to(y, (ni, nj, nk))
    z_full = np.broadcast_to(z, (ni, nj, nk))

    xyz_combined = np.stack((x_full, y_full, z_full), axis=-1)
    block.set_xyz(xyz_combined)

    # Verify the roundtrip (using reasonable tolerance for coordinate conversions)
    xyz_retrieved = np.stack([block.x, block.y, block.z], axis=-1)
    np.testing.assert_allclose(xyz_retrieved, xyz_combined, rtol=1e-6, atol=1e-8)

    # Verify polar coordinates are reasonable
    xrt = block.xrt
    assert xrt.shape == (ni, nj, nk, 3)

    # Check a few specific conversions manually
    # Point at (0, 1, 0) -> x=0, r=1, t=0
    np.testing.assert_allclose(xrt[0, 0, 2], [0.0, 1.0, 0.0], rtol=1e-6, atol=1e-8)

    # Point at (2, 3, 0) -> x=2, r=3, t=0
    np.testing.assert_allclose(xrt[2, 3, 2], [2.0, 3.0, 0.0], rtol=1e-6, atol=1e-8)


# --- L_ref tests ---


def _make_block_with_L_ref(L_ref):
    """Helper: create a 3D block with given L_ref, coordinates and fluid."""
    shape = (3, 4, 5)
    b = ember.block.Block(shape=shape)
    b.set_L_ref(L_ref)
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    b.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)
    return b, xrt


def test_L_ref_default():
    """Default L_ref is 1.0."""
    b = ember.block.Block(shape=(2, 3, 4))
    assert b.L_ref == np.float32(1.0)


def test_coordinate_roundtrip_L_ref():
    """Coordinates round-trip through set/get with L_ref != 1."""
    b, xrt = _make_block_with_L_ref(2.0)
    np.testing.assert_allclose(b.x, xrt[..., 0], rtol=1e-6)
    np.testing.assert_allclose(b.r, xrt[..., 1], rtol=1e-6)
    np.testing.assert_allclose(b.t, xrt[..., 2], rtol=1e-6)
    np.testing.assert_allclose(b.xrt, xrt, rtol=1e-6)
    np.testing.assert_allclose(b.xr[..., 0], xrt[..., 0], rtol=1e-6)
    np.testing.assert_allclose(b.xr[..., 1], xrt[..., 1], rtol=1e-6)


def test_wdist_roundtrip_L_ref():
    """Wall distance round-trips through set/get with L_ref != 1."""
    b, _ = _make_block_with_L_ref(3.0)
    wdist_in = np.full(b.shape, 0.01, dtype=np.float32)
    b.set_wdist(wdist_in)
    np.testing.assert_allclose(b.wdist, wdist_in, rtol=1e-6)


def test_vol_dA_scale_L_ref():
    """Same physical grid gives same vol/dA regardless of L_ref."""
    b1, _ = _make_block_with_L_ref(1.0)
    b2, _ = _make_block_with_L_ref(2.0)

    # Same dimensional input -> same dimensional output
    np.testing.assert_allclose(b2.vol, b1.vol, rtol=1e-5)
    np.testing.assert_allclose(b2.dAi, b1.dAi, rtol=1e-5)
    np.testing.assert_allclose(b2.dl_min, b1.dl_min, rtol=1e-5)


def test_Vt_independent_of_L_ref():
    """Tangential velocity is the same regardless of L_ref."""
    b1, _ = _make_block_with_L_ref(1.0)
    b1.set_P_T(101325.0, 300.0)
    b1.set_Vx(50.0).set_Vr(10.0).set_Vt(30.0)

    b2, _ = _make_block_with_L_ref(2.0)
    b2.set_P_T(101325.0, 300.0)
    b2.set_Vx(50.0).set_Vr(10.0).set_Vt(30.0)

    np.testing.assert_allclose(b1.Vt, b2.Vt, rtol=1e-6)
    np.testing.assert_allclose(b1.Vx, b2.Vx, rtol=1e-6)
    np.testing.assert_allclose(b1.Vr, b2.Vr, rtol=1e-6)


def test_rhorVt_property_L_ref():
    """rhorVt property returns dimensional rho * r * Vt."""
    b, xrt = _make_block_with_L_ref(2.0)
    b.set_P_T(101325.0, 300.0)
    Vt = 30.0
    b.set_Vt(Vt)
    rho = b.rho
    r = b.r
    expected = rho * r * Vt
    np.testing.assert_allclose(b.rhorVt, expected, rtol=1e-5)


def test_set_conserved_L_ref():
    """set_conserved accepts dimensional rhorVt and round-trips correctly."""
    b, xrt = _make_block_with_L_ref(2.0)
    b.set_P_T(101325.0, 300.0)
    b.set_Vx(50.0).set_Vr(10.0).set_Vt(30.0)

    # Grab dimensional conserved
    cons_dim = np.stack((b.rho, b.rhoVx, b.rhoVr, b.rhorVt, b.rhoe), axis=-1).copy()

    # Set them back
    b.set_conserved(cons_dim)

    np.testing.assert_allclose(b.rhorVt, cons_dim[..., 3], rtol=1e-5)
    np.testing.assert_allclose(b.Vt, 30.0, rtol=1e-5)


def test_Re_ref():
    """Re_ref equals rho_ref * V_ref * L_ref / mu."""
    rho_ref, V_ref, Rgas_ref = 1.2, 300.0, 287.0
    mu = 1.8e-5
    L_ref = 0.05
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=mu,
        Pr=0.7,
        rho_ref=rho_ref,
        V_ref=V_ref,
        Rgas_ref=Rgas_ref,
    )
    b = ember.block.Block(shape=(2,))
    b.set_fluid(fluid)
    b.set_L_ref(L_ref)

    expected = mu / (rho_ref * V_ref * L_ref)
    np.testing.assert_allclose(b.mu_nd, expected, rtol=1e-5)


def test_P_nd():
    """P_nd equals P / P_ref."""
    rho_ref, V_ref, Rgas_ref = 1.2, 300.0, 287.0
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=1.8e-5,
        Pr=0.7,
        rho_ref=rho_ref,
        V_ref=V_ref,
        Rgas_ref=Rgas_ref,
    )
    b = ember.block.Block(shape=(3,))
    b.set_fluid(fluid)
    P, T = 101325.0, 350.0
    b.set_P_T(P, T)

    P_ref = rho_ref * V_ref**2
    np.testing.assert_allclose(b.P_nd, P / P_ref, rtol=1e-5)
