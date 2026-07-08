"""Tests for interface velocity resolution functions (ember.util).

Module tested: ember.util

Test cases:
- test_chi_zero: Chi=0 case where Vm equals Vx and Vn equals Vr
- test_chi_45deg: Chi=45° case with rotation transformation
- test_chi_90deg: Chi=90° case where Vm equals Vr and Vn equals -Vx
- test_chi_array_input: Chi as spatially varying array
- test_chi_zero: Chi=0 case should recover original velocities
- test_chi_90deg: Chi=90° case transforms interface velocities back
- test_roundtrip_scalar_chi: Roundtrip consistency for scalar chi values (0°, 30°, 45°, 60°, 90°)
- test_roundtrip_array_chi: Roundtrip consistency with spatially varying chi
- test_zero_velocities: Zero meridional velocities
- test_large_velocities: Large velocity magnitudes
- test_negative_velocities: Negative velocity components
- test_full_circle_chi: Chi values around full circle
- test_orthogonal_transformation: Transformation is orthogonal
- test_chi_180deg: Chi=180° case
- test_chi_270deg: Chi=270° case
- test_functions_return_block: Functions return the block they act on
- test_preserves_block_properties: Transformation preserves non-velocity properties
- test_derived_properties_update: Derived velocity properties update correctly
- test_extremely_small_velocities: Extremely small velocity magnitudes
- test_mixed_sign_velocities: Mixed positive/negative velocities across domain
"""

import pytest
import numpy as np
import ember.block
import ember.block_util
import ember.fluid
import ember.set_iter
import ember.patch
import ember.util as util


@pytest.fixture
def block():
    """Create a configured Block for testing interface velocity functions."""
    shape = (3, 4, 5)

    # Create block with coordinates
    b = ember.block.Block(shape=shape)
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Set up fluid and thermodynamic state
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)
    b.set_rho_u(np.ones(shape, dtype=np.float32), np.ones(shape, dtype=np.float32))

    return b


class TestResolveToInterface:
    """Test resolve_to_interface function."""

    def test_chi_zero(self, block):
        """Test chi=0 case: Vm should equal Vx, Vn should equal Vr."""
        Vx_orig, Vr_orig, Vt_orig = 100.0, 50.0, 25.0
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        util.resolve_to_interface(block, 0.0)

        # When chi=0: Vm=Vx, Vn=Vr, Vt unchanged
        np.testing.assert_allclose(block.Vx, Vx_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vr, Vr_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)

    def test_chi_45deg(self, block):
        """Test chi=45° case: Vm=(Vx+Vr)/√2, Vn=(Vr-Vx)/√2."""
        Vx_orig, Vr_orig, Vt_orig = 100.0, 50.0, 25.0
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        chi = 45.0  # 45 degrees
        util.resolve_to_interface(block, chi)

        # Expected values for 45° rotation
        sqrt2 = np.sqrt(2.0)
        expected_Vm = (Vx_orig + Vr_orig) / sqrt2  # ~106.07
        expected_Vn = (Vr_orig - Vx_orig) / sqrt2  # ~-35.36

        np.testing.assert_allclose(block.Vx, expected_Vm, rtol=1e-6)
        np.testing.assert_allclose(block.Vr, expected_Vn, rtol=1e-6)
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)

    def test_chi_90deg(self, block):
        """Test chi=90° case: Vm should equal Vr, Vn should equal -Vx."""
        Vx_orig, Vr_orig, Vt_orig = 100.0, 50.0, 25.0
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        util.resolve_to_interface(block, 90.0)

        # When chi=90°: Vm=Vr, Vn=-Vx, Vt unchanged
        np.testing.assert_allclose(block.Vx, Vr_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vr, -Vx_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)

    def test_chi_array_input(self, block):
        """Test with chi as an array (spatially varying interface angle)."""
        Vx_orig, Vr_orig, Vt_orig = 100.0, 50.0, 25.0
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        # Create spatially varying chi
        chi = np.linspace(0, 90.0, block.shape[0])  # Varies along i direction
        chi_full = np.broadcast_to(chi[:, np.newaxis, np.newaxis], block.shape)

        util.resolve_to_interface(block, chi_full)

        # Check first plane (chi≈0): should be close to original
        np.testing.assert_allclose(block.Vx[0], Vx_orig, rtol=1e-2)
        np.testing.assert_allclose(block.Vr[0], Vr_orig, rtol=1e-2)

        # Check last plane (chi≈90°): should be swapped
        np.testing.assert_allclose(block.Vx[-1], Vr_orig, rtol=1e-2)
        np.testing.assert_allclose(block.Vr[-1], -Vx_orig, rtol=1e-2)

        # Vt should be unchanged everywhere
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)


class TestResolveFromInterface:
    """Test resolve_from_interface function."""

    def test_chi_zero(self, block):
        """Test chi=0 case: should recover original velocities."""
        # Set interface-aligned velocities (Vm=Vx=100, Vn=Vr=50)
        block.set_Vx(100.0)
        block.set_Vr(50.0)
        block.set_Vt(25.0)

        util.resolve_from_interface(block, 0.0)

        # When chi=0: should be unchanged (Vx=Vm, Vr=Vn)
        np.testing.assert_allclose(block.Vx, 100.0, rtol=1e-6)
        np.testing.assert_allclose(block.Vr, 50.0, rtol=1e-6)
        np.testing.assert_allclose(block.Vt, 25.0, rtol=1e-6)

    def test_chi_90deg(self, block):
        """Test chi=90° case: should transform interface velocities back."""
        # Set interface-aligned velocities (Vm=Vx=50, Vn=Vr=-100)
        # These would have come from original Vx=100, Vr=50 with chi=90°
        block.set_Vx(50.0)
        block.set_Vr(-100.0)
        block.set_Vt(25.0)

        util.resolve_from_interface(block, 90.0)

        # When chi=90°: Vx=-Vn=100, Vr=Vm=50
        np.testing.assert_allclose(block.Vx, 100.0, rtol=1e-6)
        np.testing.assert_allclose(block.Vr, 50.0, rtol=1e-6)
        np.testing.assert_allclose(block.Vt, 25.0, rtol=1e-6)


class TestRoundtripConsistency:
    """Test that forward and inverse transformations are consistent."""

    @pytest.mark.parametrize("chi", [0.0, 30.0, 45.0, 60.0, 90.0, 180.0])
    def test_roundtrip_scalar_chi(self, block, chi):
        """Test roundtrip consistency for various scalar chi values."""
        Vx_orig, Vr_orig, Vt_orig = 100.0, 50.0, 25.0
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        # Forward transformation
        util.resolve_to_interface(block, chi)

        # Inverse transformation
        util.resolve_from_interface(block, chi)

        # Should recover original values
        np.testing.assert_allclose(block.Vx, Vx_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vr, Vr_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)

    def test_roundtrip_array_chi(self, block):
        """Test roundtrip consistency with spatially varying chi."""
        # Set up random initial velocities
        np.random.seed(42)
        Vx_orig = np.random.uniform(50, 150, block.shape).astype(np.float32)
        Vr_orig = np.random.uniform(-50, 50, block.shape).astype(np.float32)
        Vt_orig = np.random.uniform(10, 40, block.shape).astype(np.float32)

        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        # Create spatially varying chi
        chi = np.random.uniform(0, 360.0, block.shape).astype(np.float32)

        # Forward transformation
        util.resolve_to_interface(block, chi)

        # Inverse transformation
        util.resolve_from_interface(block, chi)

        # Should recover original values
        np.testing.assert_allclose(block.Vx, Vx_orig, rtol=1e-5)
        np.testing.assert_allclose(block.Vr, Vr_orig, rtol=1e-5)
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)


class TestSpecialCases:
    """Test special cases and edge conditions."""

    def test_zero_velocities(self, block):
        """Test with zero meridional velocities."""
        block.set_Vx(0.0)
        block.set_Vr(0.0)
        block.set_Vt(25.0)

        util.resolve_to_interface(block, 45.0)

        # Should remain zero for meridional components
        np.testing.assert_allclose(block.Vx, 0.0, rtol=1e-6)
        np.testing.assert_allclose(block.Vr, 0.0, rtol=1e-6)
        np.testing.assert_allclose(block.Vt, 25.0, rtol=1e-6)

    def test_large_velocities(self, block):
        """Test with large velocity magnitudes."""
        Vx_orig, Vr_orig, Vt_orig = 1e6, -5e5, 1e4
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        # Test roundtrip with large values
        util.resolve_to_interface(block, 180.0 / 3)

        util.resolve_from_interface(block, 180.0 / 3)

        np.testing.assert_allclose(block.Vx, Vx_orig, rtol=1e-5)
        np.testing.assert_allclose(block.Vr, Vr_orig, rtol=1e-5)
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)

    def test_negative_velocities(self, block):
        """Test with negative velocity components."""
        Vx_orig, Vr_orig, Vt_orig = -100.0, -50.0, -25.0
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        util.resolve_to_interface(block, 180.0 / 6)  # 30 degrees

        # Verify transformation preserves magnitude
        Vm_squared = block.Vx**2 + block.Vr**2
        original_squared = Vx_orig**2 + Vr_orig**2
        np.testing.assert_allclose(Vm_squared, original_squared, rtol=1e-6)

        # Vt should be unchanged
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)

    def test_full_circle_chi(self, block):
        """Test chi values around full circle (0 to 2π)."""
        Vx_orig, Vr_orig, Vt_orig = 100.0, 50.0, 25.0

        chi_values = np.linspace(0, 360.0, 8, endpoint=False)

        for chi in chi_values:
            block.set_Vx(Vx_orig)
            block.set_Vr(Vr_orig)
            block.set_Vt(Vt_orig)
            util.resolve_to_interface(block, chi)

            # Magnitude should be preserved for meridional components
            Vm_squared = block.Vx**2 + block.Vr**2
            original_squared = Vx_orig**2 + Vr_orig**2
            np.testing.assert_allclose(Vm_squared, original_squared, rtol=1e-6)

            # Vt should be unchanged
            np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)

    def test_orthogonal_transformation(self, block):
        """Test that the transformation is orthogonal (preserves magnitudes)."""
        # Random initial velocities
        np.random.seed(123)
        Vx_orig = np.random.uniform(-200, 200, block.shape).astype(np.float32)
        Vr_orig = np.random.uniform(-100, 100, block.shape).astype(np.float32)
        Vt_orig = np.random.uniform(-50, 50, block.shape).astype(np.float32)

        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        # Store original meridional magnitude
        original_mag = np.sqrt(Vx_orig**2 + Vr_orig**2)

        # Apply transformation
        chi = 25.7  # Random angle
        util.resolve_to_interface(block, chi)

        # Check magnitude preservation
        new_mag = np.sqrt(block.Vx**2 + block.Vr**2)
        np.testing.assert_allclose(new_mag, original_mag, rtol=1e-5)

        # Vt should be unchanged
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)


class TestSpecificAngles:
    """Test mathematically important angles."""

    def test_chi_180deg(self, block):
        """Test chi=180° case: Vm should equal -Vx, Vn should equal -Vr."""
        Vx_orig, Vr_orig, Vt_orig = 100.0, 50.0, 25.0
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        util.resolve_to_interface(block, 180.0)  # 180 degrees

        # When chi=180°: Vm=-Vx, Vn=-Vr, Vt unchanged
        np.testing.assert_allclose(block.Vx, -Vx_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vr, -Vr_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)

    def test_chi_270deg(self, block):
        """Test chi=270° case: Vm should equal -Vr, Vn should equal Vx."""
        Vx_orig, Vr_orig, Vt_orig = 100.0, 50.0, 25.0
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        util.resolve_to_interface(block, 270.0)  # 270 degrees

        # When chi=270°: Vm=-Vr, Vn=Vx, Vt unchanged
        np.testing.assert_allclose(block.Vx, -Vr_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vr, Vx_orig, rtol=1e-6)
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)


class TestIntegration:
    """Test integration with existing block functionality."""

    def test_functions_return_block(self, block):
        """Test that resolve_to_interface/resolve_from_interface return the block."""
        block.set_Vx(100)
        block.set_Vr(50)
        block.set_Vt(25)
        result = util.resolve_to_interface(block, 0.0)
        assert result is block, "Function should return the block"

        result2 = util.resolve_from_interface(block, 0.0)
        assert result2 is block, "Function should return the block"

    def test_preserves_block_properties(self, block):
        """Test that transformation preserves non-velocity block properties."""
        # Set up block with some properties
        block.set_Vx(100.0)
        block.set_Vr(50.0)
        block.set_Vt(25.0)

        # Store non-velocity properties
        orig_rho = block.rho.copy()
        orig_rhoe = block.rhoe.copy()
        orig_shape = block.shape
        orig_coords = block.xrt.copy()

        # Apply transformation
        util.resolve_to_interface(block, 45.0)

        # Verify non-velocity properties unchanged
        np.testing.assert_array_equal(block.rho, orig_rho)
        np.testing.assert_array_equal(block.rhoe, orig_rhoe)
        assert block.shape == orig_shape
        np.testing.assert_array_equal(block.xrt, orig_coords)

    def test_derived_properties_update(self, block):
        """Test that derived velocity properties update correctly."""
        Vx_orig, Vr_orig, Vt_orig = 100.0, 50.0, 25.0
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        # Store original derived properties
        orig_Vm = block.Vm.copy()
        orig_V = block.V.copy()

        # Apply transformation
        util.resolve_to_interface(block, 45.0)

        # New meridional magnitude should equal old one (orthogonal transformation)
        np.testing.assert_allclose(block.Vm, orig_Vm, rtol=1e-5)

        # Total velocity magnitude should be preserved
        np.testing.assert_allclose(block.V, orig_V, rtol=1e-5)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_extremely_small_velocities(self, block):
        """Test with extremely small velocity magnitudes."""
        Vx_orig, Vr_orig, Vt_orig = 1e-10, 1e-11, 1e-12
        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        util.resolve_to_interface(block, 45.0)

        # Should handle small values without numerical issues
        assert np.all(np.isfinite(block.Vx))
        assert np.all(np.isfinite(block.Vr))
        assert np.all(np.isfinite(block.Vt))

    def test_mixed_sign_velocities(self, block):
        """Test with mixed positive/negative velocities across the domain."""
        # Create checkerboard pattern of velocities
        i, j, k = np.meshgrid(
            range(block.shape[0]),
            range(block.shape[1]),
            range(block.shape[2]),
            indexing="ij",
        )

        Vx_orig = 100.0 * (-1) ** (i + j + k)
        Vr_orig = 50.0 * (-1) ** (i + j)
        Vt_orig = 25.0 * (-1) ** k

        block.set_Vx(Vx_orig)
        block.set_Vr(Vr_orig)
        block.set_Vt(Vt_orig)

        # Test roundtrip
        util.resolve_to_interface(block, 180.0 / 3)

        util.resolve_from_interface(block, 180.0 / 3)

        np.testing.assert_allclose(block.Vx, Vx_orig, rtol=1e-5)
        np.testing.assert_allclose(block.Vr, Vr_orig, rtol=1e-5)
        np.testing.assert_allclose(block.Vt, Vt_orig, rtol=1e-6)


"""Tests for block concatenation functions (ember.block).

Module tested: ember.block

Test cases:
- test_no_blocks_raises_error: Concatenating no blocks raises ValueError
- test_single_block_returns_copy: Concatenating one block returns a copy
- test_two_blocks_axis0: Concatenating two blocks along axis 0
- test_two_blocks_axis1: Concatenating two blocks along axis 1
- test_three_blocks: Concatenating three blocks
- test_incompatible_shapes_axis1: Error when shapes incompatible for concatenation along axis 1
- test_incompatible_shapes_axis2: Error when shapes incompatible for concatenation along axis 2
- test_invalid_axis: Error for invalid concatenation axis
- test_incompatible_fluids: Different fluid objects cause errors
- test_metadata_mismatch_error: Error when metadata doesn't match
- test_missing_metadata_error: Error when metadata exists in only one block
- test_concatenate_with_safe_patches: Concatenating blocks with patches not on interface
- test_interface_patches_raise_error: Error when blocks have patches on concatenation interface
- test_non_interface_patches_preserved: Non-interface patches are correctly preserved and adjusted
- test_concatenate_with_negative_indices: Concatenating blocks with patches using negative indices
- test_data_concatenation_axis0: Data arrays are correctly concatenated along axis 0
- test_data_concatenation_axis1: Data concatenation along axis 1
- test_initialization_flags: Initialization flags are handled correctly
- test_three_blocks_sequential: Concatenating three blocks sequentially
- test_multiple_blocks_with_kwargs: Axis can be specified as keyword argument
- test_different_fluid_objects_error: Different fluid objects cause errors
- test_single_dimension_blocks: Concatenating minimal 1D blocks
- test_geometry_properties_available: Geometry properties are available in result
"""


@pytest.fixture
def perfect_fluid():
    """Create a PerfectFluid for testing."""
    return ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)


@pytest.fixture
def block1(perfect_fluid):
    """Create first test block."""
    shape = (3, 4, 5)
    b = ember.block.Block(shape=shape)

    # Set coordinates
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Set fluid and flow state
    b.set_fluid(perfect_fluid)
    b.set_P_T(101325.0, 300.0)
    b.set_Vx(100.0)
    b.set_Vr(10.0)
    b.set_Vt(5.0)
    b.set_Omega(1000.0)
    b.set_Nb(24)

    return b


@pytest.fixture
def block2(perfect_fluid):
    """Create second test block with compatible shape."""
    shape = (5, 4, 5)  # Different i-dimension for concatenation
    b = ember.block.Block(shape=shape)

    # Set coordinates
    xrt = util.linmesh3([1.0, 2.0], [0.5, 1.5], [0.0, 0.2], shape)
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Set fluid and flow state
    b.set_fluid(perfect_fluid)
    b.set_P_T(101325.0, 300.0)
    b.set_Vx(120.0)
    b.set_Vr(15.0)
    b.set_Vt(8.0)
    b.set_Omega(1000.0)
    b.set_Nb(24)

    return b


@pytest.fixture
def block3(perfect_fluid):
    """Create third test block."""
    shape = (2, 4, 5)
    b = ember.block.Block(shape=shape)

    # Set coordinates
    xrt = util.linmesh3([2.0, 3.0], [0.5, 1.5], [0.0, 0.2], shape)
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])

    # Set fluid and flow state
    b.set_fluid(perfect_fluid)
    b.set_P_T(101325.0, 300.0)
    b.set_Vx(80.0)
    b.set_Vr(5.0)
    b.set_Vt(2.0)
    b.set_Omega(1000.0)
    b.set_Nb(24)

    return b


class TestConcatenateBasic:
    """Test basic concatenate functionality."""

    def test_no_blocks_raises_error(self):
        """Test that concatenating no blocks raises ValueError."""
        with pytest.raises(ValueError, match="At least 1 block required"):
            ember.block_util.concatenate()

    def test_single_block_returns_copy(self, block1):
        """Test that concatenating one block returns a copy."""
        # Initialize all data in the block before concatenation
        block1.set_x(block1.xrt[..., 0])
        block1.set_r(block1.xrt[..., 1])
        block1.set_t(block1.xrt[..., 2])
        conserved = np.stack(
            [block1.rho, block1.rhoVx, block1.rhoVr, block1.rhorVt, block1.rhoe],
            axis=-1,
        )
        block1.set_conserved(conserved)  # Ensure all flow data is set

        # Set wall distance to initialize all data
        block1.set_wdist(np.ones(block1.shape, dtype=np.float32))

        # Set turbulent viscosity to initialize all data
        block1.set_mu_turb(np.ones(block1.shape, dtype=np.float32))

        result = ember.block_util.concatenate(block1)

        # Should be a copy, not the same object
        assert result is not block1
        assert np.array_equal(result._data, block1._data)
        assert result.shape == block1.shape
        assert result.Omega == block1.Omega
        assert result.Nb == block1.Nb

    def test_two_blocks_axis0(self, block1, block2):
        """Test concatenating two blocks along axis 0 (i-direction)."""
        result = ember.block_util.concatenate(block1, block2, axis=0)

        # Check shape is correct
        expected_shape = (
            block1.shape[0] + block2.shape[0],
            block1.shape[1],
            block1.shape[2],
        )
        assert result.shape == expected_shape

        # Check data concatenation for initialized arrays
        for key in block1._data_keys:
            if block1._versions[key] and block2._versions[key]:
                data1 = getattr(block1, key)
                data2 = getattr(block2, key)
                result_data = getattr(result, key)
                assert np.array_equal(result_data[: block1.shape[0]], data1)
                assert np.array_equal(result_data[block1.shape[0] :], data2)

        # Check metadata from first block is preserved
        assert result.Omega == block1.Omega
        assert result.Nb == block1.Nb

    def test_two_blocks_axis1(self, block1, perfect_fluid):
        """Test concatenating two blocks along axis 1 (j-direction)."""
        # Create block2 with compatible shape for j-concatenation
        shape2 = (3, 6, 5)  # Same i,k but different j
        block2 = ember.block.Block(shape=shape2)
        xrt2 = util.linmesh3([0.0, 1.0], [1.5, 2.5], [0.0, 0.2], shape2)
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])
        block2.set_fluid(perfect_fluid)
        block2.set_P_T(101325.0, 300.0)
        block2.set_Vx(120.0)
        block2.set_Vr(15.0)
        block2.set_Vt(8.0)
        # Set same metadata as block1
        block2.set_Omega(1000.0)
        block2.set_Nb(24)

        result = ember.block_util.concatenate(block1, block2, axis=1)

        expected_shape = (
            block1.shape[0],
            block1.shape[1] + block2.shape[1],
            block1.shape[2],
        )
        assert result.shape == expected_shape

    def test_three_blocks(self, block1, block2, block3):
        """Test concatenating three blocks."""
        result = ember.block_util.concatenate(block1, block2, block3, axis=0)

        expected_shape = (
            block1.shape[0] + block2.shape[0] + block3.shape[0],
            block1.shape[1],
            block1.shape[2],
        )
        assert result.shape == expected_shape


class TestConcatenateValidation:
    """Test concatenate validation and error handling."""

    def test_incompatible_shapes_axis1(self, block1, block2):
        """Test error when shapes incompatible for concatenation along axis 1."""
        # block1: (3,4,5), block2: (5,4,5) - incompatible for axis=1 concatenation
        with pytest.raises(ValueError, match="Incompatible shapes for concatenation"):
            ember.block_util.concatenate(block1, block2, axis=1)

    def test_incompatible_shapes_axis2(self, block1, block2):
        """Test error when shapes incompatible for concatenation along axis 2."""
        with pytest.raises(ValueError, match="Incompatible shapes for concatenation"):
            ember.block_util.concatenate(block1, block2, axis=2)

    def test_invalid_axis(self, block1, block2):
        """Test error for invalid concatenation axis."""
        with pytest.raises(ValueError, match="Invalid axis"):
            ember.block_util.concatenate(block1, block2, axis=3)

        with pytest.raises(ValueError, match="Invalid axis"):
            ember.block_util.concatenate(block1, block2, axis=-1)

    def test_incompatible_fluids(self, block1):
        """Test that different fluid objects cause errors."""
        # Create block with different fluid object (even with same properties)
        block2 = ember.block.Block(shape=(5, 4, 5))
        xrt2 = util.linmesh3([1.0, 2.0], [0.5, 1.5], [0.0, 0.2], (5, 4, 5))
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])

        # Different fluid object (even with same properties) should cause error
        different_fluid = ember.fluid.PerfectFluid(
            cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72
        )
        block2.set_fluid(different_fluid)
        block2.set_P_T(101325.0, 300.0)
        # Set same metadata as block1
        block2.set_Omega(1000.0)
        block2.set_Nb(24)

        # Should error due to different fluid object identity
        with pytest.raises(ValueError, match="Metadata mismatch for 'fluid'"):
            ember.block_util.concatenate(block1, block2, axis=0)


class TestConcatenateMetadata:
    """Test metadata handling during concatenation."""

    def test_missing_metadata_error(self, block1, perfect_fluid):
        """Test error when metadata exists in only one block."""
        # Create block2 without setting Omega (will use default)
        block2 = ember.block.Block(shape=(5, 4, 5))
        xrt2 = util.linmesh3([1.0, 2.0], [0.5, 1.5], [0.0, 0.2], (5, 4, 5))
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])
        block2.set_fluid(perfect_fluid)

        # block1 has Omega set, block2 doesn't - should error
        with pytest.raises(ValueError, match="Metadata keys .* exist only in block1"):
            ember.block_util.concatenate(block1, block2, axis=0)


class TestConcatenatePatches:
    """Test patch handling during concatenation."""

    def test_concatenate_with_safe_patches(self, block1, block2):
        """Test concatenating blocks with patches not on interface."""
        # Add patches that don't interfere with concatenation
        inlet_patch = ember.patch.InletPatch(
            i=0, j=(0, 3), k=(0, 4)
        )  # i=0 face (not interface)
        block1.patches.append(inlet_patch)

        outlet_patch = ember.patch.OutletPatch(
            j=0, i=(0, 4), k=(0, 4)
        )  # j=0 face (not on concatenation interface)
        block2.patches.append(outlet_patch)

        result = ember.block_util.concatenate(block1, block2, axis=0)

        # Should have 2 patches
        assert len(result.patches) == 2

        # Check that concatenation succeeded
        expected_shape = (
            block1.shape[0] + block2.shape[0],
            block1.shape[1],
            block1.shape[2],
        )
        assert result.shape == expected_shape

    def test_interface_patches_raise_error(self, block1, block2):
        """Test error when blocks have patches on concatenation interface."""
        # Add patch on interface of block1 (i=-1 face, which is the interface for axis=0)
        interface_patch1 = ember.patch.OutletPatch(i=-1, j=(0, -1), k=(0, -1))
        block1.patches.append(interface_patch1)

        with pytest.raises(
            ValueError,
            match="Cannot concatenate blocks with patches on concatenation interface",
        ):
            ember.block_util.concatenate(block1, block2, axis=0)

        # Test with patch on block2 interface (i=0 face)
        block1.patches.clear()
        interface_patch2 = ember.patch.InletPatch(i=0, j=(0, -1), k=(0, -1))
        block2.patches.append(interface_patch2)

        with pytest.raises(
            ValueError,
            match="Cannot concatenate blocks with patches on concatenation interface",
        ):
            ember.block_util.concatenate(block1, block2, axis=0)

    def test_non_interface_patches_preserved(self, block1, block2):
        """Test that non-interface patches are correctly preserved and adjusted."""
        # Add simple patches not on interface (using available patch types)
        inlet_patch1 = ember.patch.InletPatch(j=0, i=(0, 2), k=(0, 4))  # j=0 face
        block1.patches.append(inlet_patch1)

        outlet_patch2 = ember.patch.OutletPatch(j=0, i=(0, 4), k=(0, 4))  # j=0 face
        block2.patches.append(outlet_patch2)

        result = ember.block_util.concatenate(block1, block2, axis=0)

        # Should have 2 patches
        assert len(result.patches) == 2

        # Check that concatenation succeeded
        expected_shape = (
            block1.shape[0] + block2.shape[0],
            block1.shape[1],
            block1.shape[2],
        )
        assert result.shape == expected_shape

    def test_concatenate_with_negative_indices(self, block1, block2):
        """Test concatenating blocks with patches using negative indices."""
        # Add patches with negative indices that don't interfere with concatenation
        inlet_patch = ember.patch.InletPatch(
            i=0, j=(0, -1), k=(0, -1)
        )  # Uses -1 for "end"
        block1.patches.append(inlet_patch)

        outlet_patch = ember.patch.OutletPatch(
            j=-1, i=(0, -1), k=(0, -1)
        )  # j=-1 face (last j index)
        block2.patches.append(outlet_patch)

        result = ember.block_util.concatenate(block1, block2, axis=0)

        # Should have 2 patches
        assert len(result.patches) == 2

        # Check that concatenation succeeded
        expected_shape = (
            block1.shape[0] + block2.shape[0],
            block1.shape[1],
            block1.shape[2],
        )
        assert result.shape == expected_shape

        # Verify that negative indices were properly handled
        # First patch (from block1) should be unchanged
        assert result.patches[0]._ijk_lim[0, 0] == 0  # i start
        assert result.patches[0]._ijk_lim[0, 1] == 0  # i end
        assert result.patches[0]._ijk_lim[1, 0] == 0  # j start
        assert result.patches[0]._ijk_lim[1, 1] == -1  # j end (negative preserved)

        # Second patch (from block2) should have i-indices shifted
        # block1.shape[0] = 3, block2.shape = (5, 4, 5)
        # Original patch has i=(0, -1) and j=-1
        # For i dimension: (0, -1) -> (0, 4) in block2 -> (3, 7) after shift
        # For j dimension: (-1, -1) stays as (-1, -1) since it's the constant dimension
        axis_offset = block1.shape[0]  # 3
        block2_i_size = block2.shape[0]  # 5

        assert result.patches[1]._ijk_lim[0, 0] == 0 + axis_offset  # i start: 0 + 3 = 3
        assert (
            result.patches[1]._ijk_lim[0, 1] == (block2_i_size - 1) + axis_offset
        )  # i end: 4 + 3 = 7
        assert result.patches[1]._ijk_lim[1, 0] == -1  # j unchanged (constant dim)
        assert result.patches[1]._ijk_lim[1, 1] == -1  # j unchanged (constant dim)


class TestConcatenateData:
    """Test data integrity during concatenation."""

    def test_data_concatenation_axis0(self, block1, block2):
        """Test that data arrays are correctly concatenated along axis 0."""
        result = ember.block_util.concatenate(block1, block2, axis=0)

        # Check each data component
        for key in block1._data_keys:
            if block1._versions[key] and block2._versions[key]:
                # Get data from original blocks
                data1 = getattr(block1, key)
                data2 = getattr(block2, key)
                result_data = getattr(result, key)

                # Check concatenation is correct
                assert np.array_equal(result_data[: block1.shape[0]], data1)
                assert np.array_equal(result_data[block1.shape[0] :], data2)

    def test_data_concatenation_axis1(self, block1, perfect_fluid):
        """Test data concatenation along axis 1."""
        # Create compatible block for j-concatenation
        shape2 = (3, 6, 5)
        block2 = ember.block.Block(shape=shape2)
        xrt2 = util.linmesh3([0.0, 1.0], [1.5, 2.5], [0.0, 0.2], shape2)
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])
        block2.set_fluid(perfect_fluid)
        block2.set_P_T(101325.0, 300.0)
        block2.set_Vx(120.0)
        block2.set_Vr(15.0)
        block2.set_Vt(8.0)
        # Set same metadata as block1
        block2.set_Omega(1000.0)
        block2.set_Nb(24)

        result = ember.block_util.concatenate(block1, block2, axis=1)

        # Check shape
        expected_shape = (3, block1.shape[1] + block2.shape[1], 5)
        assert result.shape == expected_shape

        # Check data concatenation along j-axis
        assert np.array_equal(result.x[:, : block1.shape[1]], block1.x)
        assert np.array_equal(result.x[:, block1.shape[1] :], block2.x)

    def test_initialization_flags(self, block1, perfect_fluid):
        """Test that initialization flags are handled correctly."""
        # Create block2 with some uninitialized data
        block2 = ember.block.Block(shape=(5, 4, 5))
        xrt2 = util.linmesh3([1.0, 2.0], [0.5, 1.5], [0.0, 0.2], (5, 4, 5))
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])
        block2.set_fluid(perfect_fluid)
        # Set same metadata as block1
        block2.set_Omega(1000.0)
        block2.set_Nb(24)
        # Don't set flow state, so some data remains uninitialized

        result = ember.block_util.concatenate(block1, block2, axis=0)

        # Only data initialized in both blocks should be marked as initialized
        for key in block1._data_keys:
            if block1._versions[key] and block2._versions[key]:
                assert result._versions[key]
            else:
                assert not result._versions.get(key, 0)  # Should be 0 or not present


class TestConcatenateMultiple:
    """Test concatenating multiple blocks."""

    def test_three_blocks_sequential(self, block1, block2, block3):
        """Test concatenating three blocks sequentially."""
        result = ember.block_util.concatenate(block1, block2, block3, axis=0)

        expected_i_size = block1.shape[0] + block2.shape[0] + block3.shape[0]
        expected_shape = (expected_i_size, block1.shape[1], block1.shape[2])
        assert result.shape == expected_shape

        # Check data from each block is in correct position
        assert np.array_equal(result.x[: block1.shape[0]], block1.x)
        assert np.array_equal(
            result.x[block1.shape[0] : block1.shape[0] + block2.shape[0]], block2.x
        )
        assert np.array_equal(result.x[block1.shape[0] + block2.shape[0] :], block3.x)

    def test_multiple_blocks_with_kwargs(self, block1, block2, block3):
        """Test that axis can be specified as keyword argument."""
        result = ember.block_util.concatenate(block1, block2, block3, axis=0)

        # Should work the same as positional argument
        expected_shape = (
            block1.shape[0] + block2.shape[0] + block3.shape[0],
            block1.shape[1],
            block1.shape[2],
        )
        assert result.shape == expected_shape


class TestConcatenateEdgeCases:
    """Test edge cases and error conditions."""

    def test_different_fluid_objects_error(self, block1):
        """Test that different fluid objects cause errors."""
        # Create block with different fluid object
        block2 = ember.block.Block(shape=(5, 4, 5))
        xrt2 = util.linmesh3([1.0, 2.0], [0.5, 1.5], [0.0, 0.2], (5, 4, 5))
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])

        # Different fluid object should cause error
        different_fluid = ember.fluid.PerfectFluid(
            cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72
        )
        block2.set_fluid(different_fluid)
        block2.set_P_T(101325.0, 300.0)
        # Set same metadata as block1
        block2.set_Omega(1000.0)
        block2.set_Nb(24)

        # Should error due to different fluid object identity
        with pytest.raises(ValueError, match="Metadata mismatch for 'fluid'"):
            ember.block_util.concatenate(block1, block2, axis=0)

    def test_single_dimension_blocks(self, perfect_fluid):
        """Test concatenating minimal 1D blocks."""
        block1 = ember.block.Block(shape=(2, 1, 1))
        block2 = ember.block.Block(shape=(3, 1, 1))

        # Set minimal required data with non-zero r coordinates
        xrt1 = np.zeros(block1.shape + (3,))
        xrt1[..., 1] = 1.0  # Set r=1.0 to avoid zero radius error
        block1.set_x(xrt1[..., 0])
        block1.set_r(xrt1[..., 1])
        block1.set_t(xrt1[..., 2])
        block1.set_fluid(perfect_fluid)

        xrt2 = np.zeros(block2.shape + (3,))
        xrt2[..., 1] = 1.0  # Set r=1.0 to avoid zero radius error
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])
        block2.set_fluid(perfect_fluid)

        result = ember.block_util.concatenate(block1, block2, axis=0)
        assert result.shape == (5, 1, 1)

    def test_geometry_properties_available(self, block1, block2):
        """Test that geometry properties are available in result."""
        result = ember.block_util.concatenate(block1, block2, axis=0)

        # Should be able to access geometry properties directly
        assert hasattr(result, "dAi")
        assert hasattr(result, "dAj")
        assert hasattr(result, "dAk")


class TestMemoryUsage:
    """Test memory_usage function."""

    @pytest.fixture
    def mixing_grid(self):
        """Small initialized block for memory_usage tests."""
        import ember.grid

        shape = (25, 25, 25)
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)

        xrt = util.linmesh3([0.0, 0.1], [1.0, 1.1], [-0.05, 0.05], shape)
        b = ember.block.Block(shape=shape)
        b.set_x(xrt[..., 0])
        b.set_r(xrt[..., 1])
        b.set_t(xrt[..., 2])
        b.set_fluid(fluid)

        Po1, To1 = 1e5, 300.0
        s1 = fluid.get_s(*fluid.set_P_T(Po1, To1))
        ho1 = fluid.get_h(*fluid.set_P_T(Po1, To1))
        ember.set_iter.set_ho_s_Ma_Alpha_Beta(b, ho1, s1, 0.3, 0.0, 0.0)
        b.set_wdist(0.0)

        return ember.grid.Grid([b])

    def test_returns_three_nonempty_dicts(self, mixing_grid):
        """All 3 returned dicts are non-empty."""
        from ember.block_util import memory_usage

        block = mixing_grid[0]
        # Access some cached properties to populate cache
        _ = block.P
        _ = block.Ma

        data_usage, metadata_usage, cache_usage = memory_usage(block)

        assert isinstance(data_usage, dict) and len(data_usage) > 0
        assert isinstance(metadata_usage, dict) and len(metadata_usage) > 0
        assert isinstance(cache_usage, dict) and len(cache_usage) > 0

    def test_data_has_all_keys(self, mixing_grid):
        """Data dict has all 10 data keys."""
        from ember.block_util import memory_usage

        block = mixing_grid[0]
        data_usage, _, _ = memory_usage(block)

        expected_keys = {
            "x",
            "r",
            "t",
            "rho",
            "rhoVx",
            "rhoVr",
            "rhorVt",
            "rhoe",
            "wdist",
            "mu_turb",
        }
        assert set(data_usage.keys()) == expected_keys

    def test_all_values_positive_int(self, mixing_grid):
        """All values are positive integers."""
        from ember.block_util import memory_usage

        block = mixing_grid[0]
        _ = block.P
        _ = block.Ma

        data_usage, metadata_usage, cache_usage = memory_usage(block)

        for d in (data_usage, metadata_usage, cache_usage):
            for key, val in d.items():
                assert isinstance(val, int), f"{key}: expected int, got {type(val)}"
                assert val > 0, f"{key}: expected positive, got {val}"

    def test_cache_nonempty_after_access(self, mixing_grid):
        """After accessing cached properties, cache dict is non-empty."""
        from ember.block_util import memory_usage

        block = mixing_grid[0]
        _ = block.P
        _ = block.Ma

        _, _, cache_usage = memory_usage(block)
        assert len(cache_usage) > 0

    def test_metadata_contains_expected_keys(self, mixing_grid):
        """Metadata dict contains at least fluid, patches, triangulated."""
        from ember.block_util import memory_usage

        block = mixing_grid[0]
        _, metadata_usage, _ = memory_usage(block)

        for key in ("fluid", "patches", "triangulated"):
            assert key in metadata_usage, f"Expected '{key}' in metadata_usage"


if __name__ == "__main__":
    pytest.main([__file__])
