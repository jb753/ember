"""Test module for pure Python flux calculation functions.

Tests pure Python implementations for 0D, 1D, and 2D flux calculations.

Test cases:
- test_2d_fluxes_methods: 2D flux calculation methods
- test_flux_3d_error: Error handling for 3D flux calculations
- test_flow_2d: 2D flow calculations
- test_flow_3d_error: Error handling for 3D flow calculations
- test_flux_basic_0d: Basic 0D flux calculations
- test_flux_3d_error_message: Error message validation for 3D flux operations
- test_flux_1d: 1D flux calculations
- test_flow_0d_1d_errors: Error handling for 0D and 1D flow operations
- test_flux_flow_dimensionality_coverage: Dimensionality coverage testing
- test_flux_triangulated_shape: Triangulated mesh flux calculations

Note: 3D Fortran-based flux calculations are tested in test_solver_block.py
"""

import numpy as np
import pytest
import ember.block
import ember.fluid
import ember.patch
import ember.fluxes
import ember.grid
from ember import util


@pytest.fixture
def uniform_flow_block():
    """Create a block with uniform flow for flux testing (without patches)."""
    shape = tuple(41 + i for i in range(3))
    block = ember.block.Block(shape=shape)

    L = 0.1
    rmid = 2.0
    Nb = int(2.0 * np.pi * rmid / L)  # Number of blocks around circumference
    pitch = 2 * np.pi / Nb
    rlim = np.array([-L, L]) / 2 + rmid
    xrt = util.linmesh3((0, L), rlim, (-pitch / 2, pitch / 2), shape)

    # Set coordinates with Fortran order
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set up perfect gas fluid
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,  # Specific heat at constant pressure [J/kg/K]
        gamma=1.4,  # Heat capacity ratio
        mu=1e-5,  # Dynamic viscosity [Pa⋅s]
        Pr=0.72,  # Prandtl number
    )
    block.set_fluid(fluid)

    # Set uniform thermodynamic state
    block.set_P_T(2e5, 350.0)
    V_ref = 100.0
    block.set_Vx(V_ref)
    block.set_Vr(0.8 * V_ref)
    block.set_Vt(-0.3 * V_ref)
    block.set_Omega(100.0)  # rad/s

    return block


@pytest.fixture
def uniform_flow_block_1d():
    """Create a 1D block for flux testing."""
    shape = (5,)  # 1D block
    block = ember.block.Block(shape=shape)

    # Set up 1D coordinates
    x = np.linspace(0, 0.1, shape[0])
    r = np.full_like(x, 1.0)  # Constant radius
    t = np.full_like(x, 0.0)  # Constant theta
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Set up fluid
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)

    # Set up simple uniform flow
    Vx = 100.0
    Vr = 0.0
    Vt = 0.0
    P = 101325.0
    block.set_P_T(P * np.ones(shape), 288.15 * np.ones(shape))
    block.set_Vx(Vx * np.ones(shape))
    block.set_Vr(Vr * np.ones(shape))
    block.set_Vt(Vt * np.ones(shape))

    return block


@pytest.fixture
def scalar_block():
    """Create a scalar block for flux testing."""
    block = ember.block.Block(shape=())

    # Set up perfect gas fluid
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,  # Specific heat at constant pressure [J/kg/K]
        gamma=1.4,  # Heat capacity ratio
        mu=1e-5,  # Dynamic viscosity [Pa⋅s]
        Pr=0.72,  # Prandtl number
    )
    block.set_fluid(fluid)

    # Set coordinates and thermodynamic state
    block.set_x(np.array([0.05]))
    block.set_r(np.array([2.0]))
    block.set_t(np.array([0.1]))
    block.set_P_T(2e5, 350.0)
    V_ref = 100.0
    block.set_Vx(V_ref)
    block.set_Vr(0.8 * V_ref)
    block.set_Vt(-0.3 * V_ref)
    block.set_Omega(100.0)  # rad/s

    return block


@pytest.fixture
def uniform_flow_block_2d():
    """Create a 2D block with uniform flow for flux testing."""
    shape = (41, 42)  # 2D shape
    block = ember.block.Block(shape=shape)

    L = 0.1
    rmid = 2.0
    rlim = np.array([-L, L]) / 2 + rmid

    # Create 2D mesh manually
    x_vec = np.linspace(0, L, shape[0])
    r_vec = np.linspace(rlim[0], rlim[1], shape[1])
    x_mesh, r_mesh = np.meshgrid(x_vec, r_vec, indexing="ij")
    t_mesh = np.zeros_like(x_mesh)  # Zero theta for 2D cut
    xrt = np.stack([x_mesh, r_mesh, t_mesh], axis=-1)

    # Set coordinates with Fortran order
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set up perfect gas fluid
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,  # Specific heat at constant pressure [J/kg/K]
        gamma=1.4,  # Heat capacity ratio
        mu=1e-5,  # Dynamic viscosity [Pa⋅s]
        Pr=0.72,  # Prandtl number
    )
    block.set_fluid(fluid)

    # Set uniform thermodynamic state
    block.set_P_T(2e5, 350.0)
    V_ref = 100.0
    block.set_Vx(V_ref)
    block.set_Vr(0.8 * V_ref)
    block.set_Vt(-0.3 * V_ref)
    block.set_Omega(100.0)  # rad/s

    return block


def test_2d_fluxes_methods(uniform_flow_block_2d):
    """Test that all Fluxes methods can be called on 2D blocks.

    For 2D blocks, only fluxk should be calculated while fluxi and fluxj
    should remain empty after calling the methods.
    """
    block = uniform_flow_block_2d

    # Test the flux property for 2D blocks
    flux = ember.fluxes.get_flux(block)
    assert flux.shape == (40, 41, 3, 5), f"flux should have 2D shape, got {flux.shape}"
    assert np.any(flux != 0), "flux should contain non-zero calculated values"


def test_flux_3d_error(uniform_flow_block):
    """Test that flux raises an error for 3D blocks."""
    block = uniform_flow_block

    with pytest.raises(ValueError):
        _ = ember.fluxes.get_flux(block)


def test_flux_basic_0d(scalar_block):
    """Test the flux property for scalar (0D) blocks."""
    block = scalar_block

    # Test flux property for 0D blocks
    flux_0d = ember.fluxes.get_flux(block)

    # Should have shape (3, 5) for scalar block
    expected_shape = (3, 5)
    assert flux_0d.shape == expected_shape, (
        f"Expected {expected_shape}, got {flux_0d.shape}"
    )

    # Should contain non-zero values
    assert np.any(flux_0d != 0), "flux should contain non-zero calculated values"


def test_flux_3d_error_message(uniform_flow_block_2d, uniform_flow_block):
    """Test that flux raises an error for 3D blocks."""
    # Test with 2D block
    # 2D blocks should work fine with flux
    _ = ember.fluxes.get_flux(uniform_flow_block_2d)

    # Test with 3D block
    with pytest.raises(ValueError):
        _ = ember.fluxes.get_flux(uniform_flow_block)


def test_flux_1d(uniform_flow_block_1d):
    """Test that flux works correctly for 1D blocks."""
    block = uniform_flow_block_1d

    # Test flux property for 1D
    flux_1d = ember.fluxes.get_flux(block)
    expected_shape = (block.shape[0] - 1, 3, 5)  # (ni-1, 3, 5)
    assert flux_1d.shape == expected_shape, (
        f"1D flux shape: {flux_1d.shape} vs {expected_shape}"
    )

    # Should contain finite values
    assert np.all(np.isfinite(flux_1d)), "1D flux should contain finite values"


def test_flux_triangulated_shape(uniform_flow_block_2d):
    """Test that block.flux has correct shape (ntri, 3, 5) for triangulated cuts."""
    from ember.cut import triangulate_to_unstructured

    # Create triangulated block from 2D structured block
    tri_block = triangulate_to_unstructured(uniform_flow_block_2d)

    # Verify block is triangulated and has expected shape
    assert tri_block.triangulated, "Block should be triangulated"
    ntri = tri_block.shape[0]
    assert tri_block.shape == (ntri, 3), (
        f"Triangulated block shape should be (ntri, 3), got {tri_block.shape}"
    )

    # Test flux property has correct shape (ntri, 3, 5)
    flux = ember.fluxes.get_flux(tri_block)
    expected_shape = (ntri, 3, 5)
    assert flux.shape == expected_shape, (
        f"Triangulated flux shape: expected {expected_shape}, got {flux.shape}"
    )

    # Verify flux contains finite values
    assert np.all(np.isfinite(flux)), "Triangulated flux should contain finite values"
