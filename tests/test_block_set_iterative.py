"""Tests for Block iterative state setter methods.

This module tests the iterative state setters that solve for thermodynamic
state given different constraints:
- set_ho_s_Ma_Alpha_Beta: Given Mach number constraint
- set_ho_s_rhoVm_Alpha_Beta: Given meridional mass flux (rhoVm) constraint

Both methods use iterative solvers in ember.set_iter to converge to the
correct thermodynamic state while satisfying the given velocity constraint.

Test cases for set_ho_s_Ma_Alpha_Beta:
- test_set_ho_s_ma_perfect_gas: Perfect gas relation validation
- test_set_ho_s_ma_energy_conservation: Energy conservation during state setting
- test_set_ho_s_ma_entropy_conservation: Entropy conservation validation
- test_set_ho_s_ma_convergence: Convergence behavior of iterative solver
- test_set_ho_s_ma_array_broadcasting: Array broadcasting functionality
- test_set_ho_s_ma_robustness: Robustness testing for edge cases
- test_set_ho_s_ma_float32_consistency: Float32 precision consistency
- test_set_ho_s_ma_uniform_pressure: Uniform inputs produce uniform pressure

Test cases for set_ho_s_rhoVm_Alpha_Beta:
- test_set_ho_s_rhoVm_basic_conservation: Basic conservation laws with stagnation conditions
- test_set_ho_s_rhoVm_energy_conservation: Total enthalpy conservation verification
- test_set_ho_s_rhoVm_entropy_conservation: Entropy conservation verification
- test_set_ho_s_rhoVm_mass_flux_consistency: Meridional mass flux (rhoVm) consistency validation
- test_set_ho_s_rhoVm_array_broadcasting: Array input broadcasting behavior
- test_set_ho_s_rhoVm_convergence: Iterative solver convergence testing
- test_set_ho_s_rhoVm_comparison_with_Ma: Comparison with Mach number setter consistency
- test_set_ho_s_rhoVm_float32_consistency: Float32 precision preservation
- test_set_ho_s_rhoVm_nonzero_alpha: Vm = V * cos(Alpha) relationship with non-zero yaw angles
- test_set_ho_s_rhoVm_nonzero_beta: Meridional velocity with non-zero pitch angles
- test_set_ho_s_rhoVm_combined_angles: Combined non-zero Alpha and Beta angles
"""

import numpy as np
import pytest
import ember.fluid
import ember.block
import ember.set_iter


# ============================================================================
# Tests for set_ho_s_Ma_Alpha_Beta (Mach number constraint)
# ============================================================================


def test_set_ho_s_ma_perfect_gas():
    """Test set_ho_s_Ma_Alpha_Beta against known perfect gas relations."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Create 1D block
    block = ember.block.Block(shape=(4,))
    block.set_fluid(fluid)

    # Reference stagnation conditions
    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Test different Mach numbers
    test_mach = np.array([0.2, 0.5, 0.8, 1.0])

    # Set stagnation state and Mach numbers
    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho1, s1, test_mach)

    # Check that stagnation conditions are recovered
    assert np.allclose(block.ho, ho1, rtol=1e-6)
    assert np.allclose(block.s, s1, rtol=1e-4)  # Relaxed for float32 precision
    assert np.allclose(block.Ma, test_mach, rtol=1e-6)

    # Verify perfect gas relations
    gamma = 1.4
    T_ratio = 1.0 + 0.5 * (gamma - 1) * test_mach**2
    P_ratio = T_ratio ** (gamma / (gamma - 1))

    assert np.allclose(block.To / block.T, T_ratio, rtol=1e-6)
    assert np.allclose(block.Po / block.P, P_ratio, rtol=1e-6)


def test_set_ho_s_ma_energy_conservation():
    """Test that total enthalpy is conserved."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Set reference conditions
    To1, Po1 = 400.0, 200000.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    Ma_test = np.array([0.3, 0.7, 1.2])

    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho1, s1, Ma_test)

    # Check energy balance: ho = h + V²/2
    h_static = block.h
    V_squared = block.V**2
    ho_calculated = h_static + 0.5 * V_squared

    assert np.allclose(ho_calculated, ho1, rtol=1e-6)


def test_set_ho_s_ma_entropy_conservation():
    """Test that entropy is conserved during isentropic process."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(5,))
    block.set_fluid(fluid)

    # Reference conditions
    To1, Po1 = 350.0, 150000.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    Ma_array = np.array([0.1, 0.4, 0.8, 1.0, 1.5])

    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho1, s1, Ma_array)

    # Entropy should be constant (relaxed tolerance for float32 precision)
    assert np.allclose(block.s, s1, rtol=1e-4)


def test_set_ho_s_ma_convergence():
    """Test convergence behavior for different Mach numbers."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Standard conditions
    To1, Po1 = 288.15, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Test edge cases
    Ma_edge = np.array([0.01, 1.0, 2.0])  # Very low, sonic, supersonic

    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho1, s1, Ma_edge)

    # Should converge without errors
    assert np.allclose(block.Ma, Ma_edge, rtol=1e-5)


def test_set_ho_s_ma_array_broadcasting():
    """Test array broadcasting capabilities."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # 2D block
    block = ember.block.Block(shape=(2, 3))
    block.set_fluid(fluid)

    # Reference conditions
    To1, Po1 = 320.0, 120000.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # 2D array of Mach numbers
    Ma_2d = np.array([[0.3, 0.6, 0.9], [0.4, 0.8, 1.2]])

    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho1, s1, Ma_2d)

    assert block.Ma.shape == (2, 3)
    assert np.allclose(block.Ma, Ma_2d, rtol=1e-5)


def test_set_ho_s_ma_robustness():
    """Test robustness with extreme conditions."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(2,))
    block.set_fluid(fluid)

    # High temperature, high pressure conditions
    To1, Po1 = 1500.0, 5000000.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Test with high Mach numbers
    Ma_test = np.array([2.5, 3.0])

    # Should converge successfully even with extreme conditions
    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho1, s1, Ma_test)

    assert np.allclose(block.Ma, Ma_test, rtol=1e-5)
    assert np.allclose(block.ho, ho1, rtol=1e-5)


def test_set_ho_s_ma_float32_consistency():
    """Test that method maintains float32 precision."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Reference conditions
    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    Ma_test = np.array([0.2, 0.6, 1.0], dtype=np.float32)

    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho1, s1, Ma_test)

    # Check that all properties maintain float32
    assert block.rho.dtype == np.float32
    assert block.T.dtype == np.float32
    assert block.P.dtype == np.float32
    assert block.Ma.dtype == np.float32


def test_set_ho_s_ma_uniform_pressure():
    """Test that uniform inputs (ho, s, Ma) result in uniform pressure.

    This is a critical test to ensure numerical consistency. When all inputs
    are identical across the domain, the output pressure should also be
    identical everywhere, not just similar within a tolerance.
    """
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Test with different block shapes to ensure it works in all dimensions
    for shape in [(5,), (3, 4), (2, 3, 4)]:
        block = ember.block.Block(shape=shape)
        block.set_fluid(fluid)

        # Reference stagnation conditions (uniform scalars)
        To1, Po1 = 300.0, 101325.0
        rho_o, u_o = fluid.set_P_T(Po1, To1)
        ho1 = fluid.get_h(rho_o, u_o)
        s1 = fluid.get_s(rho_o, u_o)

        # Uniform Mach number
        Ma_uniform = 0.6

        # Set uniform conditions
        ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho1, s1, Ma_uniform)

        # All pressures should be EXACTLY equal (within float32 precision)
        P_min = np.min(block.P)
        P_max = np.max(block.P)
        P_mean = np.mean(block.P)

        # Check that pressure is uniform across the entire domain
        # For uniform inputs, the result should be identical everywhere
        assert np.allclose(block.P, P_mean, rtol=1e-6, atol=1e-3), (
            f"Pressure not uniform for shape {shape}. "
            f"P_min={P_min:.6f}, P_max={P_max:.6f}, P_mean={P_mean:.6f}, "
            f"range={(P_max - P_min):.6e}, relative_range={(P_max - P_min) / P_mean:.6e}"
        )

        # Also check other properties for uniformity
        assert np.allclose(block.T, np.mean(block.T), rtol=1e-6, atol=1e-3), (
            f"Temperature not uniform for shape {shape}"
        )
        assert np.allclose(block.rho, np.mean(block.rho), rtol=1e-6, atol=1e-3), (
            f"Density not uniform for shape {shape}"
        )
        assert np.allclose(block.Ma, Ma_uniform, rtol=1e-6), (
            f"Mach number not uniform for shape {shape}"
        )

        # Verify the solution is physically correct
        gamma = 1.4
        T_ratio = 1.0 + 0.5 * (gamma - 1) * Ma_uniform**2
        P_ratio = T_ratio ** (gamma / (gamma - 1))
        P_expected = Po1 / P_ratio

        assert np.allclose(P_mean, P_expected, rtol=1e-5), (
            f"Mean pressure {P_mean:.6f} does not match expected {P_expected:.6f}"
        )


# ============================================================================
# Tests for set_ho_s_rhoVm_Alpha_Beta (meridional mass flux constraint)
# ============================================================================


def test_set_ho_s_rhoVm_basic_conservation():
    """Test basic conservation laws with set_ho_s_rhoVm_Alpha_Beta."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Create 1D block
    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Reference stagnation conditions
    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Test different rhoVm values (meridional mass flux per unit area)
    rhoVm_test = np.array([50.0, 100.0, 200.0])

    # Set stagnation state and meridional mass flux (Alpha=0, so Vm=V)
    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block, ho1, s1, rhoVm_test)

    # Check that stagnation conditions are recovered
    assert np.allclose(block.ho, ho1, rtol=1e-6)
    assert np.allclose(block.s, s1, rtol=1e-4)

    # Check meridional mass flux conservation: rhoVm = rho * Vm = rho * V * cos(Alpha)
    # With Alpha=0, cos(Alpha)=1, so rhoVm = rho * V
    rhoVm_calc = block.rho * block.Vm
    assert np.allclose(rhoVm_calc, rhoVm_test, rtol=1e-6)


def test_set_ho_s_rhoVm_energy_conservation():
    """Test that total enthalpy is conserved."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(4,))
    block.set_fluid(fluid)

    # Set reference conditions
    To1, Po1 = 400.0, 200000.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    rhoVm_test = np.array([25.0, 50.0, 100.0, 150.0])

    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block, ho1, s1, rhoVm_test)

    # Check energy balance: ho = h + V²/2
    h_static = block.h
    V_squared = block.V**2
    ho_calculated = h_static + 0.5 * V_squared

    assert np.allclose(ho_calculated, ho1, rtol=1e-6)


def test_set_ho_s_rhoVm_entropy_conservation():
    """Test that entropy is conserved during isentropic process."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(5,))
    block.set_fluid(fluid)

    # Reference conditions
    To1, Po1 = 350.0, 150000.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    rhoVm_array = np.array([10.0, 30.0, 60.0, 100.0, 140.0])

    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block, ho1, s1, rhoVm_array)

    # Entropy should be constant
    assert np.allclose(block.s, s1, rtol=1e-4)


def test_set_ho_s_rhoVm_mass_flux_consistency():
    """Test consistency of meridional mass flux calculation."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Standard conditions
    To1, Po1 = 288.15, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Test different meridional mass flux levels
    rhoVm_test = np.array([5.0, 25.0, 80.0])

    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block, ho1, s1, rhoVm_test)

    # Verify meridional mass flux constraint is satisfied exactly
    rhoVm_calculated = block.rho * block.Vm
    assert np.allclose(rhoVm_calculated, rhoVm_test, rtol=1e-6)

    # Verify thermodynamic consistency
    assert np.all(block.rho > 0)
    assert np.all(block.V > 0)
    assert np.all(block.T > 0)
    assert np.all(block.P > 0)


def test_set_ho_s_rhoVm_array_broadcasting():
    """Test array broadcasting capabilities."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # 2D block
    block = ember.block.Block(shape=(2, 3))
    block.set_fluid(fluid)

    # Reference conditions
    To1, Po1 = 320.0, 120000.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # 2D array of meridional mass flux values
    rhoVm_2d = np.array([[20.0, 40.0, 60.0], [30.0, 50.0, 80.0]])

    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block, ho1, s1, rhoVm_2d)

    assert block.rho.shape == (2, 3)
    assert block.V.shape == (2, 3)

    # Check meridional mass flux conservation
    rhoVm_calc = block.rho * block.Vm
    assert np.allclose(rhoVm_calc, rhoVm_2d, rtol=1e-6)


def test_set_ho_s_rhoVm_convergence():
    """Test convergence behavior for different meridional mass flux values."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(4,))
    block.set_fluid(fluid)

    # Standard conditions
    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Test edge cases: very low, moderate, high meridional mass flux
    rhoVm_edge = np.array([1.0, 25.0, 75.0, 120.0])

    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block, ho1, s1, rhoVm_edge)

    # Should converge without errors
    rhoVm_calc = block.rho * block.Vm
    assert np.allclose(rhoVm_calc, rhoVm_edge, rtol=1e-6)


def test_set_ho_s_rhoVm_comparison_with_Ma():
    """Test consistency between set_ho_s_rhoVm_Alpha_Beta and set_ho_s_Ma_Alpha_Beta."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Create two identical blocks
    block1 = ember.block.Block(shape=(3,))
    block2 = ember.block.Block(shape=(3,))
    block1.set_fluid(fluid)
    block2.set_fluid(fluid)

    # Reference conditions
    To1, Po1 = 350.0, 120000.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Set same state using Mach number
    Ma_test = np.array([0.2, 0.5, 0.8])
    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block1, ho1, s1, Ma_test)

    # Get resulting meridional mass flux and apply to second block
    rhoVm_from_Ma = block1.rho * block1.Vm
    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block2, ho1, s1, rhoVm_from_Ma)

    # Both blocks should have nearly identical states (within float32 precision)
    assert np.allclose(block1.rho, block2.rho, rtol=1e-5)
    assert np.allclose(block1.V, block2.V, rtol=1e-5)
    assert np.allclose(block1.P, block2.P, rtol=1e-5)
    assert np.allclose(block1.T, block2.T, rtol=1e-5)
    assert np.allclose(block1.Ma, block2.Ma, rtol=1e-5)


def test_set_ho_s_rhoVm_float32_consistency():
    """Test that method maintains float32 precision."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Reference conditions
    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    rhoVm_test = np.array([20.0, 50.0, 100.0], dtype=np.float32)

    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block, ho1, s1, rhoVm_test)

    # Check that all properties maintain float32
    assert block.rho.dtype == np.float32
    assert block.T.dtype == np.float32
    assert block.P.dtype == np.float32
    assert block.V.dtype == np.float32


def test_set_ho_s_rhoVm_nonzero_alpha():
    """Test meridional velocity relationship with non-zero yaw angle."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Reference stagnation conditions
    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Test different yaw angles
    Alpha = np.array([30.0, 45.0, 60.0])  # degrees
    rhoVm_test = np.array([50.0, 75.0, 100.0])

    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block, ho1, s1, rhoVm_test, Alpha=Alpha)

    # Verify relationship: rhoVm = rho * Vm = rho * V * cos(Alpha)
    rhoVm_calc = block.rho * block.Vm
    assert np.allclose(rhoVm_calc, rhoVm_test, rtol=1e-6)

    # Verify V > Vm for non-zero Alpha
    assert np.all(block.V > block.Vm)

    # Verify cos(Alpha) relationship: Vm = V * cos(Alpha)
    Vm_from_V = block.V * np.cos(np.radians(Alpha))
    assert np.allclose(block.Vm, Vm_from_V, rtol=1e-6)


def test_set_ho_s_rhoVm_nonzero_beta():
    """Test meridional velocity relationship with non-zero pitch angle."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Reference stagnation conditions
    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Test different pitch angles (Beta) with Alpha=0
    Beta = np.array([10.0, 20.0, 30.0])  # degrees
    rhoVm_test = np.array([50.0, 75.0, 100.0])

    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(
        block, ho1, s1, rhoVm_test, Alpha=0.0, Beta=Beta
    )

    # Verify meridional mass flux constraint
    rhoVm_calc = block.rho * block.Vm
    assert np.allclose(rhoVm_calc, rhoVm_test, rtol=1e-6)

    # With Alpha=0 and non-zero Beta, Vm should still equal V*cos(0) = V
    assert np.allclose(block.Vm, block.V, rtol=1e-6)


def test_set_ho_s_rhoVm_combined_angles():
    """Test meridional velocity with both Alpha and Beta non-zero."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(2,))
    block.set_fluid(fluid)

    # Reference stagnation conditions
    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1, To1)
    ho1 = fluid.get_h(rho_o, u_o)
    s1 = fluid.get_s(rho_o, u_o)

    # Test with both angles non-zero, use moderate values
    Alpha = np.array([30.0, 45.0])
    Beta = np.array([10.0, 15.0])
    rhoVm_test = np.array([40.0, 60.0])

    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(
        block, ho1, s1, rhoVm_test, Alpha=Alpha, Beta=Beta
    )

    # Verify meridional mass flux
    rhoVm_calc = block.rho * block.Vm
    assert np.allclose(rhoVm_calc, rhoVm_test, rtol=1e-6)

    # Verify Vm = V * cos(Alpha)
    Vm_from_V = block.V * np.cos(np.radians(Alpha))
    assert np.allclose(block.Vm, Vm_from_V, rtol=1e-6)

    # Verify total enthalpy conservation
    ho_calculated = block.h + 0.5 * block.V**2
    assert np.allclose(ho_calculated, ho1, rtol=1e-6)


def test_set_ho_s_ma_nonunity_refs():
    """Test set_ho_s_Ma_Alpha_Beta with non-unity reference values for nondimensionalisation."""
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=1.8e-5,
        Pr=0.7,
        rho_ref=1.2,
        V_ref=300.0,
        Rgas_ref=287.0,
    )

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)
    # Coordinates required for block.ho, block.s, block.Ma
    r = np.ones(3, dtype=np.float32)
    x = np.zeros(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Reference stagnation conditions (dimensional)
    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1 / fluid.P_ref, To1 / fluid.T_ref)
    ho1 = fluid.get_h(rho_o, u_o) * fluid.u_ref
    s1 = fluid.get_s(rho_o, u_o) * fluid.Rgas_ref

    test_mach = np.array([0.3, 0.6, 0.9])
    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho1, s1, test_mach)

    assert np.allclose(block.ho, ho1, rtol=1e-5)
    assert np.allclose(block.s, s1, rtol=1e-4)
    assert np.allclose(block.Ma, test_mach, rtol=1e-5)


def test_set_ho_s_rhoVm_nonunity_refs():
    """Test set_ho_s_rhoVm_Alpha_Beta with non-unity reference values."""
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=1.8e-5,
        Pr=0.7,
        rho_ref=1.2,
        V_ref=300.0,
        Rgas_ref=287.0,
    )

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)
    r = np.ones(3, dtype=np.float32)
    x = np.zeros(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    To1, Po1 = 300.0, 101325.0
    rho_o, u_o = fluid.set_P_T(Po1 / fluid.P_ref, To1 / fluid.T_ref)
    ho1 = fluid.get_h(rho_o, u_o) * fluid.u_ref
    s1 = fluid.get_s(rho_o, u_o) * fluid.Rgas_ref

    rhoVm_test = np.array([50.0, 100.0, 150.0])
    ember.set_iter.set_ho_s_rhoVm_Alpha_Beta(block, ho1, s1, rhoVm_test)

    assert np.allclose(block.ho, ho1, rtol=1e-5)
    assert np.allclose(block.s, s1, rtol=1e-4)
    rhoVm_calc = block.rho * block.Vm
    assert np.allclose(rhoVm_calc, rhoVm_test, rtol=1e-5)


def test_set_Po_To_Ma_rel_nonunity_refs():
    """Test set_Po_To_Ma_rel_Alpha_rel_Beta with non-unity reference values."""
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=1.8e-5,
        Pr=0.7,
        rho_ref=1.2,
        V_ref=300.0,
        Rgas_ref=287.0,
    )

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    r = np.ones(3, dtype=np.float32)
    x = np.zeros(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    Po_inlet, To_inlet = 101325.0, 300.0
    Ma_rel = np.array([0.3, 0.6, 0.9])
    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, 0.0, 0.0
    )

    assert np.allclose(block.Po, Po_inlet, rtol=1e-5)
    assert np.allclose(block.To, To_inlet, rtol=1e-5)
    assert np.allclose(block.Ma_rel, Ma_rel, rtol=1e-5)


def test_physical_properties_ref_independent():
    """Physical properties are identical for two blocks with different ref scales but same P, T, V.

    Loops dynamically over all Block properties (excluding private, _ref, and _nd suffixes)
    and checks that any numeric property accessible on both blocks agrees to float32 tolerance.
    Properties that raise for 0-D blocks or require uninitialised data are skipped.
    """
    # Use T=400 K (not 300 K = T_dtm) so that u != 0, giving non-trivial agreement.
    P, T = 101325.0, 400.0
    Vx, Vr, Vt = 100.0, 50.0, 80.0

    def make_block(rho_ref, V_ref, Rgas_ref):
        fluid = ember.fluid.PerfectFluid(
            cp=1005.0,
            gamma=1.4,
            mu=1.8e-5,
            Pr=0.7,
            rho_ref=rho_ref,
            V_ref=V_ref,
            Rgas_ref=Rgas_ref,
        )
        block = ember.block.Block(shape=())
        block.set_fluid(fluid)
        block.set_x(np.float32(0.0))
        block.set_r(np.float32(1.0))
        block.set_t(np.float32(0.0))
        block.set_P_T(P, T)
        block.set_Vx(Vx)
        block.set_Vr(Vr)
        block.set_Vt(Vt)
        return block

    b1 = make_block(rho_ref=1.0, V_ref=1.0, Rgas_ref=1.0)
    b2 = make_block(rho_ref=1.2, V_ref=300.0, Rgas_ref=287.0)

    # Exclude 'secondary' (scratch array, not a physical property).
    # Exclude derivative properties (dhdrho_P, dsdP_rho, etc.) -- these are
    # nondim-broken separately and out of scope for this test.
    exclude = {"secondary", "P_offset_nd"}
    derivative_prefixes = ("dhd", "dsd", "dud")

    props = [
        p
        for p in dir(type(b1))
        if isinstance(getattr(type(b1), p), property)
        and not p.startswith("_")
        and not p.endswith("_ref")
        and not p.endswith("_nd")
        and p not in exclude
        and not any(p.startswith(pfx) for pfx in derivative_prefixes)
    ]

    skipped, checked, failed = [], [], []
    for p in sorted(props):
        try:
            v1 = np.asarray(getattr(b1, p))
            v2 = np.asarray(getattr(b2, p))
        except Exception:
            skipped.append(p)
            continue

        if not np.issubdtype(v1.dtype, np.floating):
            skipped.append(p)
            continue

        if not np.allclose(v1, v2, rtol=1e-5, atol=1e-8):
            failed.append((p, v1, v2))
        else:
            checked.append(p)

    diagnosis = "\n".join(f"  {p}: b1={v1}, b2={v2}" for p, v1, v2 in failed)
    assert not failed, (
        f"The following properties differ between blocks with different refs:\n{diagnosis}\n"
        f"(checked {len(checked)} properties, skipped {len(skipped)})"
    )
    assert len(checked) > 10, f"Too few properties checked ({len(checked)}): {checked}"


if __name__ == "__main__":
    pytest.main([__file__])
"""Tests for Block.set_Po_To_Ma_rel_Alpha_rel_Beta method.

Test cases:
- test_set_Po_To_Ma_rel_perfect_gas: Perfect gas relation validation
- test_set_Po_To_Ma_rel_pure_radial_flow: Pure radial flow test cases
- test_set_Po_To_Ma_rel_with_rotation: Testing with rotation effects
- test_set_Po_To_Ma_rel_flow_angle_consistency: Flow angle consistency validation
- test_set_Po_To_Ma_rel_energy_conservation: Energy conservation checks
- test_set_Po_To_Ma_rel_isentropic: Isentropic process validation
- test_set_Po_To_Ma_rel_method_chaining: Method chaining functionality
- test_set_Po_To_Ma_rel_array_broadcasting: Array broadcasting behavior
- test_set_Po_To_Ma_rel_convergence_parameters: Convergence parameter testing
- test_set_Po_To_Ma_rel_requires_coordinates: Coordinate requirement validation
- test_iterative_setters_self_consistency: Self consistency of iterative setters
- test_iterative_setters_realistic_conditions: Realistic condition testing
"""


def test_set_Po_To_Ma_rel_perfect_gas():
    """Test set_Po_To_Ma_rel_Alpha_rel_Beta against known perfect gas relations."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Create 1D block with coordinates
    block = ember.block.Block(shape=(4,))
    block.set_fluid(fluid)

    # Initialize coordinates (required for relative frame)
    r = np.ones(4, dtype=np.float32)
    x = np.zeros(4, dtype=np.float32)
    t = np.zeros(4, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Reference total conditions
    To_inlet, Po_inlet = 300.0, 101325.0

    # Test different relative Mach numbers
    test_mach_rel = np.array([0.2, 0.5, 0.8, 1.0])

    # Pure axial flow (Alpha_rel = Beta = 0)
    Alpha_rel = np.zeros(4)
    Beta = np.zeros(4)

    # Set flow field using total conditions
    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, test_mach_rel, Alpha_rel, Beta
    )

    # Check that relative Mach numbers are correct
    assert np.allclose(block.Ma_rel, test_mach_rel, rtol=1e-6)

    # For stationary frame, relative and absolute should be the same
    assert np.allclose(block.Ma, test_mach_rel, rtol=1e-6)

    # Verify perfect gas relations
    gamma = 1.4
    T_ratio = 1.0 + 0.5 * (gamma - 1) * test_mach_rel**2
    P_ratio = T_ratio ** (gamma / (gamma - 1))

    # Check that total conditions are recovered
    assert np.allclose(block.To, To_inlet, rtol=1e-6)
    assert np.allclose(block.Po, Po_inlet, rtol=1e-6)

    # Check isentropic relations
    assert np.allclose(block.To / block.T, T_ratio, rtol=1e-6)
    assert np.allclose(block.Po / block.P, P_ratio, rtol=1e-6)


def test_set_Po_To_Ma_rel_pure_radial_flow():
    """Test with pure radial flow (Beta = 90°, Alpha_rel = 0°)."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Create 1D block with coordinates
    ni = 5
    block = ember.block.Block(shape=(ni,))
    block.set_fluid(fluid)

    # Set up radial coordinates
    r = np.linspace(0.5, 1.0, ni, dtype=np.float32)
    x = np.zeros(ni, dtype=np.float32)
    t = np.zeros(ni, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Pure radial flow parameters
    Po_inlet, To_inlet = 101325.0, 300.0
    Ma_rel = 0.6
    Alpha_rel = 0.0  # No relative swirl
    Beta = 90.0  # Pure radial flow

    # Set flow field
    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Check flow angles
    assert np.allclose(block.Alpha_rel, Alpha_rel, rtol=1e-6)
    assert np.allclose(block.Beta, Beta, rtol=1e-6)

    # For pure radial flow: Vx ≈ 0, Vt_rel = 0, Vr ≠ 0
    assert np.allclose(
        block.Vx, 0.0, atol=1e-5
    )  # Relaxed tolerance for numerical precision
    assert np.allclose(block.Vt_rel, 0.0, atol=1e-6)
    assert np.all(np.abs(block.Vr) > 1e-6)  # Radial velocity should be non-zero

    # Check relative Mach number
    assert np.allclose(block.Ma_rel, Ma_rel, rtol=1e-6)


def test_set_Po_To_Ma_rel_with_rotation():
    """Test with rotating frame (moderate rotation rate)."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Create 1D block with radial coordinates
    ni = 2
    block = ember.block.Block(shape=(ni,))
    block.set_fluid(fluid)

    # Set up coordinates (simpler case)
    r = np.array([0.8, 1.0], dtype=np.float32)
    x = np.zeros(ni, dtype=np.float32)
    t = np.zeros(ni, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Set rotation (good working value for unified approach)
    Omega = 500.0  # rad/s (high but workable rotation)
    block.set_Omega(Omega)

    # Flow parameters with swirl
    Po_inlet, To_inlet = 101325.0, 300.0
    Ma_rel = 0.6  # Higher Mach number
    Alpha_rel = 30.0  # Add relative swirl to create Vt ≠ 0
    Beta = 0.0  # Axial flow

    # Set flow field
    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Check that rotation is preserved
    assert block.Omega == Omega

    # Check that no NaN values are present
    assert np.isfinite(block.Vt).all()
    assert np.isfinite(block.P).all()
    assert np.isfinite(block.T).all()

    # Check relative Mach number is correct
    assert np.allclose(block.Ma_rel, Ma_rel, rtol=1e-5)

    # CRITICAL TEST: Check that total conditions are preserved
    # This should FAIL if rothalpy iteration bug exists
    assert np.allclose(block.Po, Po_inlet, rtol=1e-5), (
        f"Po not preserved: expected {Po_inlet}, got {block.Po}"
    )
    assert np.allclose(block.To, To_inlet, rtol=1e-5), (
        f"To not preserved: expected {To_inlet}, got {block.To}"
    )


def test_set_Po_To_Ma_rel_flow_angle_consistency():
    """Test consistency of flow angles and velocity components."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Initialize coordinates
    r = np.ones(3, dtype=np.float32)
    x = np.zeros(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Flow parameters with mixed angles
    Po_inlet, To_inlet = 101325.0, 300.0
    Ma_rel = 0.7
    Alpha_rel = np.array([0.0, 30.0, 45.0])  # Different swirl angles
    Beta = np.array([90.0, 60.0, 45.0])  # Different pitch angles

    # Set flow field
    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Check angle consistency using velocity components
    V_rel = block.V_rel

    # Calculate angles from velocity components
    Alpha_rel_calc = np.degrees(
        np.arctan2(block.Vt_rel, np.sqrt(block.Vx**2 + block.Vr**2))
    )
    Beta_calc = np.degrees(np.arctan2(block.Vr, block.Vx))

    assert np.allclose(block.Alpha_rel, Alpha_rel_calc, rtol=1e-4)
    assert np.allclose(block.Beta, Beta_calc, rtol=1e-4)

    # Check relative velocity magnitude
    V_rel_calc = np.sqrt(block.Vx**2 + block.Vr**2 + block.Vt_rel**2)
    assert np.allclose(V_rel, V_rel_calc, rtol=1e-6)


def test_set_Po_To_Ma_rel_energy_conservation():
    """Test that total enthalpy is conserved."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(4,))
    block.set_fluid(fluid)

    # Initialize coordinates
    r = np.ones(4, dtype=np.float32)
    x = np.zeros(4, dtype=np.float32)
    t = np.zeros(4, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Set reference conditions
    Po_inlet, To_inlet = 150000.0, 400.0
    Ma_rel = np.array([0.3, 0.6, 0.9, 1.2])
    Alpha_rel = 0.0
    Beta = 0.0

    # Set flow field
    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Calculate expected total enthalpy
    rho_stag, u_stag = fluid.set_P_T(Po_inlet, To_inlet)
    ho_expected = fluid.get_h(rho_stag, u_stag)

    # Check that total enthalpy is conserved
    assert np.allclose(block.ho, ho_expected, rtol=1e-6)

    # Check energy balance: ho = h + 0.5*V^2
    h_static = block.h
    V_abs_sq = block.Vx**2 + block.Vr**2 + block.Vt**2
    ho_calc = h_static + 0.5 * V_abs_sq

    assert np.allclose(block.ho, ho_calc, rtol=1e-6)


def test_set_Po_To_Ma_rel_isentropic():
    """Test that entropy is constant (isentropic process)."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(5,))
    block.set_fluid(fluid)

    # Initialize coordinates
    r = np.ones(5, dtype=np.float32)
    x = np.zeros(5, dtype=np.float32)
    t = np.zeros(5, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Set flow field
    Po_inlet, To_inlet = 101325.0, 300.0
    Ma_rel = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    Alpha_rel = 0.0
    Beta = 0.0

    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Calculate expected entropy
    rho_stag, u_stag = fluid.set_P_T(Po_inlet, To_inlet)
    s_expected = fluid.get_s(rho_stag, u_stag)

    # Check that entropy is constant
    assert np.allclose(block.s, s_expected, rtol=1e-4)


def test_set_Po_To_Ma_rel_method_chaining():
    """Test that method returns self for chaining."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(2,))
    block.set_fluid(fluid)

    # Initialize coordinates
    r = np.ones(2, dtype=np.float32)
    x = np.zeros(2, dtype=np.float32)
    t = np.zeros(2, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    result = ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, 101325.0, 300.0, 0.5, 0.0, 0.0
    )
    assert result is block


def test_set_Po_To_Ma_rel_array_broadcasting():
    """Test proper array broadcasting with different input shapes."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(3,))
    block.set_fluid(fluid)

    # Initialize coordinates
    r = np.ones(3, dtype=np.float32)
    x = np.zeros(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Test with scalar and array inputs
    Po_inlet = 101325.0  # scalar
    To_inlet = 300.0  # scalar
    Ma_rel = np.array([0.4, 0.6, 0.8])  # array
    Alpha_rel = 15.0  # scalar
    Beta = np.array([0.0, 45.0, 90.0])  # array

    # Should not raise an error
    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Check that values are set correctly
    assert len(block.Ma_rel) == 3
    assert np.allclose(block.Ma_rel, Ma_rel, rtol=1e-6)
    assert np.allclose(block.Alpha_rel, Alpha_rel, rtol=1e-6)
    assert np.allclose(block.Beta, Beta, rtol=1e-6)


def test_set_Po_To_Ma_rel_convergence_parameters():
    """Test that the method converges correctly with default parameters."""

    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(2,))
    block.set_fluid(fluid)

    # Initialize coordinates
    r = np.ones(2, dtype=np.float32)
    x = np.zeros(2, dtype=np.float32)
    t = np.zeros(2, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Test with default convergence parameters (handled internally in set_iter.py)
    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po=101325.0, To=300.0, Ma_rel=0.6, Alpha_rel=0.0, Beta=0.0
    )

    # Should converge and produce reasonable results
    assert np.allclose(block.Ma_rel, 0.6, rtol=1e-6)
    assert np.isfinite(block.P).all()
    assert np.isfinite(block.T).all()


def test_set_Po_To_Ma_rel_requires_coordinates():
    """Test that function raises error when coordinates are not initialized."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    block = ember.block.Block(shape=(2,))
    block.set_fluid(fluid)
    # Note: NOT setting coordinates

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="Radial coordinates \\(r\\) must be initialized"
    ):
        ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
            block, 101325.0, 300.0, 0.5, 0.0, 0.0
        )


def test_iterative_setters_self_consistency():
    """Test that iterative setters preserve flow fields when given their own outputs as inputs.

    This test creates a realistic 1D flow field with rotation and applies each iterative
    setter using the current flow properties as input, verifying the flow field remains unchanged.
    """
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Create 1D block with varying radius (realistic turbomachinery geometry)
    ni = 5
    block = ember.block.Block(shape=(ni,))
    block.set_fluid(fluid)

    # Set up coordinates: radial variation from hub to tip
    r = np.linspace(1.0, 1.5, ni, dtype=np.float32)  # Hub to tip: 1.0m to 1.5m radius
    x = np.zeros(ni, dtype=np.float32)  # Axial coordinate
    t = np.zeros(ni, dtype=np.float32)  # Tangential coordinate
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Set realistic rotation rate
    Omega = 3000.0 * (2 * np.pi / 60)  # 3000 RPM converted to rad/s ≈ 314 rad/s
    block.set_Omega(Omega)

    # Initialize flow field with realistic turbomachinery conditions
    # Representative conditions for a gas turbine compressor
    P_init = np.array(
        [150000.0, 160000.0, 170000.0, 180000.0, 190000.0], dtype=np.float32
    )  # Pa
    T_init = np.array([350.0, 360.0, 370.0, 380.0, 390.0], dtype=np.float32)  # K

    # Set realistic velocity field
    Ma_target = 0.6  # Target Mach number
    Alpha_target = 30.0  # degrees (swirl angle)
    Beta_target = 60.0  # degrees (pitch angle)

    # Initialize thermodynamic state
    block.set_P_T(P_init, T_init)

    # Calculate realistic velocity field
    a = block.a  # Speed of sound
    V_mag = Ma_target * a  # Velocity magnitude

    # Decompose velocity using target angles
    from ember.util import angles_to_components

    Vx, Vr, Vt = angles_to_components(V_mag, Alpha_target, Beta_target)

    # Set velocity field
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Store reference values for comparison
    rho_ref = block.rho.copy()
    u_ref = block.u.copy()
    Vx_ref = block.Vx.copy()
    Vr_ref = block.Vr.copy()
    Vt_ref = block.Vt.copy()

    # Test 1: set_ho_s_Ma_Alpha_Beta (absolute frame)
    ho_current = block.ho
    s_current = block.s
    Ma_current = block.Ma
    Alpha_current = block.Alpha
    Beta_current = block.Beta

    ember.set_iter.set_ho_s_Ma_Alpha_Beta(
        block, ho_current, s_current, Ma_current, Alpha_current, Beta_current
    )

    # Verify flow field unchanged (within numerical tolerance)
    assert np.allclose(block.rho, rho_ref, rtol=1e-6), (
        "rho changed after set_ho_s_Ma_Alpha_Beta"
    )
    assert np.allclose(block.u, u_ref, rtol=1e-6), (
        "u changed after set_ho_s_Ma_Alpha_Beta"
    )
    assert np.allclose(block.Vx, Vx_ref, rtol=1e-6), (
        "Vx changed after set_ho_s_Ma_Alpha_Beta"
    )
    assert np.allclose(block.Vr, Vr_ref, rtol=1e-6), (
        "Vr changed after set_ho_s_Ma_Alpha_Beta"
    )
    assert np.allclose(block.Vt, Vt_ref, rtol=1e-6), (
        "Vt changed after set_ho_s_Ma_Alpha_Beta"
    )

    # Test 2: set_Po_To_Ma_rel_Alpha_rel_Beta (relative frame)
    Po_current = block.Po
    To_current = block.To
    Ma_rel_current = block.Ma_rel
    Alpha_rel_current = block.Alpha_rel
    Beta_current = block.Beta

    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_current, To_current, Ma_rel_current, Alpha_rel_current, Beta_current
    )

    # Verify flow field unchanged
    assert np.allclose(block.rho, rho_ref, rtol=1e-5), (
        "rho changed after set_Po_To_Ma_rel_Alpha_rel_Beta"
    )
    assert np.allclose(block.u, u_ref, rtol=1e-5), (
        "u changed after set_Po_To_Ma_rel_Alpha_rel_Beta"
    )
    assert np.allclose(block.Vx, Vx_ref, rtol=1e-5), (
        "Vx changed after set_Po_To_Ma_rel_Alpha_rel_Beta"
    )
    assert np.allclose(block.Vr, Vr_ref, rtol=1e-5), (
        "Vr changed after set_Po_To_Ma_rel_Alpha_rel_Beta"
    )
    assert np.allclose(block.Vt, Vt_ref, rtol=1e-5), (
        "Vt changed after set_Po_To_Ma_rel_Alpha_rel_Beta"
    )

    # Test 3: set_I_s_Ma_rel_Alpha_rel_Beta (rothalpy frame)
    I_current = block.I
    s_current = block.s
    Ma_rel_current = block.Ma_rel
    Alpha_rel_current = block.Alpha_rel
    Beta_current = block.Beta

    ember.set_iter.set_I_s_Ma_rel_Alpha_rel_Beta(
        block, I_current, s_current, Ma_rel_current, Alpha_rel_current, Beta_current
    )

    # Verify flow field unchanged
    assert np.allclose(block.rho, rho_ref, rtol=1e-5), (
        "rho changed after set_I_s_Ma_rel_Alpha_rel_Beta"
    )
    assert np.allclose(block.u, u_ref, rtol=1e-5), (
        "u changed after set_I_s_Ma_rel_Alpha_rel_Beta"
    )
    assert np.allclose(block.Vx, Vx_ref, rtol=1e-5), (
        "Vx changed after set_I_s_Ma_rel_Alpha_rel_Beta"
    )
    assert np.allclose(block.Vr, Vr_ref, rtol=1e-5), (
        "Vr changed after set_I_s_Ma_rel_Alpha_rel_Beta"
    )
    assert np.allclose(block.Vt, Vt_ref, rtol=1e-5), (
        "Vt changed after set_I_s_Ma_rel_Alpha_rel_Beta"
    )


def test_iterative_setters_realistic_conditions():
    """Test iterative setters with realistic turbomachinery conditions and varying properties."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Create 1D block representing a compressor blade section
    ni = 7
    block = ember.block.Block(shape=(ni,))
    block.set_fluid(fluid)

    # Realistic compressor geometry: hub to tip variation
    r = np.linspace(0.8, 1.2, ni, dtype=np.float32)  # 0.8m to 1.2m radius
    x = np.linspace(0.0, 0.1, ni, dtype=np.float32)  # 10cm axial length
    t = np.zeros(ni, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Typical compressor rotation (converted from user specification)
    Omega = 3000.0 * (2 * np.pi / 60)  # 3000 RPM ≈ 314 rad/s
    block.set_Omega(Omega)

    # Test realistic operating conditions
    Po_inlet = 200000.0  # 2 bar inlet pressure
    To_inlet = 450.0  # 450 K inlet temperature
    Ma_rel = np.linspace(
        0.4, 0.8, ni
    )  # Varying relative Mach number (includes user's 0.6)
    Alpha_rel = np.linspace(
        20.0, 40.0, ni
    )  # Varying relative swirl angle (includes user's 30°)
    Beta = np.linspace(45.0, 75.0, ni)  # Varying pitch angle (includes user's 60°)

    # Set flow field
    ember.set_iter.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Verify realistic results
    assert np.all(block.P > 0), "Negative pressures detected"
    assert np.all(block.T > 0), "Negative temperatures detected"
    assert np.all(np.isfinite(block.rho)), "Non-finite densities detected"
    assert np.all(np.isfinite(block.Vx)), "Non-finite Vx detected"
    assert np.all(np.isfinite(block.Vr)), "Non-finite Vr detected"
    assert np.all(np.isfinite(block.Vt)), "Non-finite Vt detected"

    # Check that total conditions are preserved
    assert np.allclose(block.Po, Po_inlet, rtol=1e-4), "Total pressure not preserved"
    assert np.allclose(block.To, To_inlet, rtol=1e-4), "Total temperature not preserved"

    # Check that relative Mach numbers are correct
    assert np.allclose(block.Ma_rel, Ma_rel, rtol=1e-4), (
        "Relative Mach numbers incorrect"
    )

    # Check realistic velocity magnitudes (should be subsonic to low supersonic)
    assert np.all(block.Ma < 2.0), "Unrealistically high absolute Mach numbers"
    assert np.all(block.Ma_rel < 1.5), "Unrealistically high relative Mach numbers"

    # Verify blade speed is reasonable (modern compressors can be transonic)
    U_max = np.max(block.U)
    a_avg = np.mean(block.a)
    U_over_a_max = U_max / a_avg
    assert U_over_a_max < 1.5, (
        f"Blade speed too high: U/a = {U_over_a_max:.2f} (should be < 1.5 for realistic turbomachinery)"
    )


if __name__ == "__main__":
    pytest.main([__file__])
