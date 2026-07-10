"""Tests for Block class core functionality.

Note: tests shared with test_block.py (e.g. property/setter round-trips)
live there; this module covers the remaining core behaviour not
duplicated elsewhere (datum handling, conserved-cell caching, rpm/L_ref
scaling, etc).

Test cases:
- test_props_not_writable_in_place: In-place write rejection for basic/cached/derived properties
- test_conserved_cell_nd_shape_and_values: Shape/value checks for conserved_cell_nd
- test_conserved_cell_dimensional_scaling: Dimensional scaling of conserved_cell_nd
- test_conserved_cell_nd_cache_invalidation: Cache invalidation for conserved_cell_nd
- test_set_rho_s: Density and entropy setting
- test_set_datum: Thermodynamic datum setting
- test_set_datum_effect: Effect of datum changes on u and s
- test_set_datum_preserves_fluid_properties: Fluid property preservation during datum changes
- test_set_rpm_scalar: Setting scalar rpm
- test_set_rpm_conversion: Type conversion to float32 through Omega
- test_set_rpm_consistency_with_rpm_property: set_rpm is consistent with rpm property
- test_block_set_xyz_roundtrip: Round-trip of combined xyz setter
- test_L_ref_default: Default reference length
- test_coordinate_roundtrip_L_ref: Coordinate round-trip under L_ref scaling
- test_wdist_roundtrip_L_ref: wdist round-trip under L_ref scaling
- test_vol_dA_scale_L_ref: Volume/area scaling under L_ref
- test_Vt_independent_of_L_ref: Vt invariance under L_ref scaling
- test_rhorVt_property_L_ref: rhorVt property under L_ref scaling
- test_set_conserved_L_ref: set_conserved under L_ref scaling
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
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Set up fluid
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)

    return b


def test_props_not_writable_in_place(block):
    """Test that basic, cached, and derived properties reject in-place writes."""
    rho = np.full(block.shape, 1.2, dtype=np.float32)
    u = np.full(block.shape, 200000.0, dtype=np.float32)
    Vx = np.zeros(block.shape, dtype=np.float32)
    Vr = np.zeros(block.shape, dtype=np.float32)
    Vt = np.zeros(block.shape, dtype=np.float32)
    block.set_rho_u(rho, u)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

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
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)


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


def test_set_datum(block):
    """Test set_datum method preserves thermodynamic state and velocities."""
    rtol = 1e-6

    # Set initial state with non-zero velocities
    rho_initial = np.full(block.shape, 1.2, dtype=np.float32)
    u_initial = np.full(block.shape, 200000.0, dtype=np.float32)
    block.set_rho_u(rho_initial, u_initial)

    block.set_Vx(np.full(block.shape, 50.0, dtype=np.float32))
    block.set_Vr(np.full(block.shape, 10.0, dtype=np.float32))
    block.set_Vt(np.full(block.shape, 30.0, dtype=np.float32))

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
    block_datum.set_x(block.xrt[..., 0])
    block_datum.set_r(block.xrt[..., 1])
    block_datum.set_t(block.xrt[..., 2])
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
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])
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


def test_Vt_independent_of_L_ref():
    """Tangential velocity is the same regardless of L_ref."""
    b1, _ = _make_block_with_L_ref(1.0)
    b1.set_P_T(101325.0, 300.0)
    b1.set_Vx(50.0)
    b1.set_Vr(10.0)
    b1.set_Vt(30.0)

    b2, _ = _make_block_with_L_ref(2.0)
    b2.set_P_T(101325.0, 300.0)
    b2.set_Vx(50.0)
    b2.set_Vr(10.0)
    b2.set_Vt(30.0)

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
    b.set_Vx(50.0)
    b.set_Vr(10.0)
    b.set_Vt(30.0)

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
