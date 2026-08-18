"""Tests for permutation/flip and clustering utility functions (ember.util).

Module tested: ember.util
"""

import itertools
import numpy as np
import pytest
import ember.block
import ember.fluid
import ember.set_iterative
from ember import util
from ember.util import angles_to_components, components_to_angles


class TestApplyPermFlip:
    """Test apply_perm_flip function."""

    def test_identity_transformation(self):
        """Test identity transformation (no permutation, no flip)."""
        coords = np.random.randn(3, 4, 5, 3).astype(np.float32)
        result = util.apply_perm_flip(coords, perm=(0, 1, 2), flip=())
        np.testing.assert_array_equal(result, coords)

    def test_permutation_only(self):
        """Test permutation without flipping."""
        coords = np.random.randn(3, 4, 5, 3).astype(np.float32)
        result = util.apply_perm_flip(coords, perm=(1, 0, 2))

        # After perm=(1,0,2), dimensions should be swapped: (3,4,5,3) -> (4,3,5,3)
        assert result.shape == (4, 3, 5, 3)
        # Coordinate dimension should be preserved
        assert result.shape[-1] == 3

    def test_flip_only(self):
        """Test flipping without permutation."""
        coords = np.random.randn(3, 4, 5, 3).astype(np.float32)
        result = util.apply_perm_flip(coords, perm=(0, 1, 2), flip=(0,))

        # Shape should be unchanged
        assert result.shape == coords.shape
        # But data should be flipped along axis 0
        expected = np.flip(coords, axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_permutation_and_flip(self):
        """Test combined permutation and flipping."""
        coords = np.random.randn(3, 4, 5, 3).astype(np.float32)
        result = util.apply_perm_flip(coords, perm=(1, 0, 2), flip=(0, 1))

        # After perm=(1,0,2), shape: (3,4,5,3) -> (4,3,5,3)
        # After flip=(0,1), flip along axes 0 and 1
        assert result.shape == (4, 3, 5, 3)

        # Manual calculation for verification
        manual = coords.transpose((1, 0, 2, 3))
        manual = np.flip(manual, axis=(0, 1))
        np.testing.assert_array_equal(result, manual)

    def test_coordinate_dimension_preserved(self):
        """Test that coordinate dimension is always preserved regardless of transformations."""
        coords = np.random.randn(2, 3, 4, 3).astype(np.float32)

        # Test various transformations
        transformations = [
            ((0, 1, 2), ()),
            ((1, 0, 2), ()),
            ((2, 1, 0), ()),
            ((0, 1, 2), (0,)),
            ((1, 0, 2), (0, 1)),
            ((2, 0, 1), (1, 2)),
        ]

        for perm, flip in transformations:
            result = util.apply_perm_flip(coords, perm, flip)
            assert result.shape[-1] == 3, (
                f"Coordinate dimension lost for perm={perm}, flip={flip}"
            )


class TestFunctionProperties:
    """Test mathematical properties of the functions."""

    def test_double_application_flips(self):
        """Test that applying the same flip twice returns to original."""
        coords = np.random.randn(3, 4, 5, 3).astype(np.float32)

        flip_cases = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

        for flip in flip_cases:
            # Apply flip twice with identity permutation
            once = util.apply_perm_flip(coords, (0, 1, 2), flip)
            twice = util.apply_perm_flip(once, (0, 1, 2), flip)
            np.testing.assert_array_equal(twice, coords)

    def test_coordinate_dimension_invariant(self):
        """Test that the coordinate dimension is never affected by transformations."""
        coords = np.random.randn(2, 3, 4, 3).astype(np.float32)

        # Test that the last dimension (coordinates) remains unchanged
        all_perms = list(itertools.permutations([0, 1, 2]))[:6]  # Subset for speed
        all_flips = [(), (0,), (1,), (0, 1)]  # Subset for speed

        for perm in all_perms:
            for flip in all_flips:
                result = util.apply_perm_flip(coords, perm, flip)
                # Last dimension should always be 3
                assert result.shape[-1] == 3

                # Verify that coordinate values are preserved (just reordered/flipped)
                original_coords_flat = coords.reshape(-1, 3)
                result_coords_flat = result.reshape(-1, 3)

                # The set of coordinate vectors should be the same
                # (though order may be different due to spatial transformations)
                assert len(original_coords_flat) == len(result_coords_flat)


class TestCosineCluster:
    """Tests for cosine_cluster function."""

    def test_cosine_cluster_basic_properties(self):
        """Endpoints, length, dtype, and monotonicity."""
        n = 9
        z = util.cosine_cluster(n)

        assert z.shape == (n,)
        assert z.dtype == np.float32
        assert z[0] == 0.0
        assert z[-1] == 1.0
        assert np.all(np.diff(z) > 0)

    def test_cosine_cluster_symmetric(self):
        """Distribution is symmetric about 0.5."""
        z = util.cosine_cluster(11)
        assert np.allclose(z + z[::-1], 1.0, atol=1e-6)

    def test_cosine_cluster_dense_at_ends(self):
        """Spacing is finer at both ends than in the middle."""
        z = util.cosine_cluster(21)
        d = np.diff(z)
        # end gaps are smaller than the central gap
        assert d[0] < d[len(d) // 2]
        assert d[-1] < d[len(d) // 2]
        # symmetric spacing
        assert np.allclose(d, d[::-1], atol=1e-6)

    def test_cosine_cluster_matches_formula(self):
        """Matches the half-cosine definition."""
        n = 7
        k = np.arange(n)
        expected = 0.5 * (1.0 - np.cos(np.pi * k / (n - 1)))
        assert np.allclose(util.cosine_cluster(n), expected, atol=1e-6)

    def test_cosine_cluster_requires_two_points(self):
        """n < 2 raises ValueError."""
        with pytest.raises(ValueError):
            util.cosine_cluster(1)


class TestCluster:
    """Tests for cluster function."""

    def test_cluster_basic_properties(self):
        """Test basic properties like endpoints and length."""
        ni = 10
        ER = 1.2
        dmax = 0.5

        x = util.cluster(ni, ER, dmax)

        # Check shape and dtype
        assert x.shape == (ni,)
        assert x.dtype == np.float32

        # Check endpoints
        assert x[0] == 0.0
        assert x[-1] == 1.0

        # Check monotonic increasing
        assert np.all(np.diff(x) > 0)

    def test_cluster_uniform_spacing(self):
        """Test ER=1.0 gives uniform spacing."""
        ni = 5
        ER = 1.0
        dmax = 1.0

        x = util.cluster(ni, ER, dmax)
        x_expected = np.linspace(0, 1, ni, dtype=np.float32)

        np.testing.assert_allclose(x, x_expected, rtol=1e-6)

    def test_cluster_expansion_ratio(self):
        """Test spacing follows expansion ratio when not capped."""
        ni = 5
        ER = 2.0
        dmax = 1.0  # Large enough to not cap

        x = util.cluster(ni, ER, dmax)
        dx = np.diff(x)

        # Check that ratios are approximately ER (within tolerance for rescaling)
        ratios = dx[1:] / dx[:-1]
        expected_ratios = np.full_like(ratios, ER)

        # Allow some tolerance due to rescaling to unit length
        np.testing.assert_allclose(ratios, expected_ratios, rtol=0.1)

    def test_cluster_max_spacing_cap(self):
        """Test dmax cap is respected when feasible."""
        ni = 8
        ER = 1.5
        dmax = 0.3  # Moderately restrictive but feasible

        x = util.cluster(ni, ER, dmax)
        dx = np.diff(x)

        # Check that no spacing exceeds dmax (within tolerance)
        assert np.all(dx <= dmax + 1e-6)

    def test_cluster_edge_cases(self):
        """Test edge cases like ni=2."""
        # Test ni=2
        x = util.cluster(2, 1.5, 0.5)
        assert x.shape == (2,)
        assert x[0] == 0.0
        assert x[1] == 1.0

        # Test very small ni
        x = util.cluster(3, 1.1, 0.8)
        assert x.shape == (3,)
        assert x[0] == 0.0
        assert x[-1] == 1.0
        assert np.all(np.diff(x) > 0)

    def test_cluster_input_validation(self):
        """Test input validation and error handling."""
        # Test ni < 2
        with pytest.raises(ValueError, match="ni must be >= 2"):
            util.cluster(1, 1.2, 0.5)

        # Test ER <= 0
        with pytest.raises(ValueError, match="ER must be > 0"):
            util.cluster(5, 0.0, 0.5)

        with pytest.raises(ValueError, match="ER must be > 0"):
            util.cluster(5, -1.0, 0.5)

        # Test dmax <= 0
        with pytest.raises(ValueError, match="dmax must be > 0"):
            util.cluster(5, 1.2, 0.0)

        with pytest.raises(ValueError, match="dmax must be > 0"):
            util.cluster(5, 1.2, -0.1)

    def test_cluster_convergence(self):
        """Test algorithm converges to unit length."""
        ni = 20
        ER = 1.5
        dmax = 0.2

        x = util.cluster(ni, ER, dmax)
        total_length = x[-1] - x[0]

        # Should be exactly 1.0 due to final enforcement
        assert abs(total_length - 1.0) < 1e-10

    def test_cluster_monotonic(self):
        """Test points are monotonically increasing."""
        ni = 15
        ER = 1.8
        dmax = 0.3

        x = util.cluster(ni, ER, dmax)
        dx = np.diff(x)

        # All spacings should be positive
        assert np.all(dx > 0)

    def test_cluster_dtype_and_shape(self):
        """Test output dtype and shape for various inputs."""
        test_cases = [
            (5, 1.0, 1.0),
            (10, 1.5, 0.1),
            (50, 2.0, 0.05),
        ]

        for ni, ER, dmax in test_cases:
            x = util.cluster(ni, ER, dmax)

            # Check shape
            assert x.shape == (ni,)

            # Check dtype
            assert x.dtype == np.float32

            # Check basic properties
            assert x[0] == 0.0
            assert x[-1] == 1.0

    def test_cluster_large_expansion_ratio(self):
        """Test behavior with large expansion ratios."""
        ni = 10
        ER = 10.0  # Very large expansion ratio
        dmax = 0.2

        x = util.cluster(ni, ER, dmax)
        dx = np.diff(x)

        # Should still respect dmax
        assert np.all(dx <= dmax + 1e-6)

        # Should still span 0 to 1
        assert x[0] == 0.0
        assert x[-1] == 1.0

    def test_cluster_very_small_dmax(self):
        """Test behavior with very small dmax falls back to uniform."""
        ni = 5
        ER = 2.0
        dmax = 0.01  # Very small - too restrictive

        x = util.cluster(ni, ER, dmax)
        dx = np.diff(x)

        # Should fall back to uniform spacing when dmax is too restrictive
        expected_spacing = 1.0 / (ni - 1)  # Uniform spacing
        np.testing.assert_allclose(dx, expected_spacing, rtol=1e-6)

        # All spacings should be equal (uniform)
        assert np.allclose(dx, dx[0], rtol=1e-6)

    def test_cluster_consistency(self):
        """Test that function gives consistent results for same inputs."""
        ni = 8
        ER = 1.3
        dmax = 0.4

        x1 = util.cluster(ni, ER, dmax)
        x2 = util.cluster(ni, ER, dmax)

        np.testing.assert_array_equal(x1, x2)

    def test_cluster_increasing_ni(self):
        """Test behavior as ni increases."""
        ER = 1.4
        dmax = 0.3

        for ni in [5, 10, 20]:
            x = util.cluster(ni, ER, dmax)

            # Basic properties should hold
            assert x.shape == (ni,)
            assert x[0] == 0.0
            assert x[-1] == 1.0
            assert np.all(np.diff(x) > 0)

            # Spacing should respect dmax
            dx = np.diff(x)
            assert np.all(dx <= dmax + 1e-6)

    def test_cluster_strict_monotonicity(self):
        """Test that cluster function produces strictly monotonic coordinates."""
        test_cases = [
            (10, 1.5, 0.2),
            (25, 2.0, 0.1),
            (50, 1.2, 0.05),
            (65, 1.5, 0.05),  # More reasonable parameters
        ]

        for ni, ER, dmax in test_cases:
            x = util.cluster(ni, ER, dmax)

            # Check strict monotonicity (no equal adjacent values)
            dx = np.diff(x)
            assert np.all(dx > 0), (
                f"Non-monotonic points found for ni={ni}, ER={ER}, dmax={dmax}"
            )

            # Check no repeated points (minimum spacing)
            min_spacing = np.min(dx)
            assert min_spacing > 1e-10, (
                f"Repeated points detected for ni={ni}, ER={ER}, dmax={dmax}"
            )

    def test_cluster_no_duplicate_points(self):
        """Test that cluster function produces no duplicate coordinate values."""
        test_cases = [
            (15, 1.8, 0.15),
            (30, 1.1, 0.08),
            (57, 1.5, 0.05),  # More reasonable parameters
        ]

        for ni, ER, dmax in test_cases:
            x = util.cluster(ni, ER, dmax)

            # Check for unique values
            unique_x = np.unique(x)
            assert len(unique_x) == len(x), (
                f"Duplicate points found for ni={ni}, ER={ER}, dmax={dmax}"
            )

            # Verify sorted order matches original (monotonic)
            np.testing.assert_array_equal(
                x,
                unique_x,
                err_msg=f"Points not in sorted order for ni={ni}, ER={ER}, dmax={dmax}",
            )

    def test_cluster_minimum_spacing(self):
        """Test that cluster function maintains minimum spacing between points."""
        test_cases = [
            (20, 1.3, 0.1),
            (40, 1.5, 0.05),
            (65, 1.5, 0.05),  # More reasonable parameters
        ]

        for ni, ER, dmax in test_cases:
            x = util.cluster(ni, ER, dmax)
            dx = np.diff(x)

            # Calculate expected minimum spacing for numerical precision
            # Should be much larger than machine epsilon for float32
            expected_min_spacing = 1e-12

            assert np.all(dx > expected_min_spacing), (
                f"Spacing too small (potential duplicates) for ni={ni}, ER={ER}, dmax={dmax}. "
                f"Min spacing: {np.min(dx)}"
            )

            # Check that all spacings are positive
            assert np.all(dx >= 0), (
                f"Negative spacing found for ni={ni}, ER={ER}, dmax={dmax}"
            )


class TestClusterSymmetric:
    """Tests for cluster_symmetric function."""

    def test_cluster_symmetric_basic_properties(self):
        """Endpoints, length, dtype, and monotonicity."""
        n = 9
        z = util.cluster_symmetric(n, 1.2, 1.0)

        assert z.shape == (n,)
        assert z.dtype == np.float32
        assert z[0] == 0.0
        assert z[-1] == 1.0
        assert np.all(np.diff(z) > 0)

    def test_cluster_symmetric_symmetric(self):
        """Distribution is symmetric about 0.5, with a node on the centreline."""
        n = 11
        z = util.cluster_symmetric(n, 1.2, 1.0)
        assert np.allclose(z + z[::-1], 1.0, atol=1e-6)
        assert z[n // 2] == 0.5

    def test_cluster_symmetric_dense_at_ends(self):
        """Spacing is finer at both ends than in the middle."""
        z = util.cluster_symmetric(21, 1.2, 1.0)
        d = np.diff(z)
        assert d[0] < d[len(d) // 2]
        assert d[-1] < d[len(d) // 2]
        assert np.allclose(d, d[::-1], atol=1e-6)

    def test_cluster_symmetric_expansion_ratio(self):
        """Spacing grows at ER away from each wall, unlike cosine_cluster."""
        ER = 1.1
        d = np.diff(util.cluster_symmetric(21, ER, 1.0))
        # growth rate over the first half, walking in from the wall
        ratio = d[1 : len(d) // 2] / d[: len(d) // 2 - 1]
        assert np.allclose(ratio, ER, atol=1e-4)

    def test_cluster_symmetric_uniform_when_er_one(self):
        """ER=1.0 degenerates to uniform spacing."""
        n = 5
        assert np.allclose(util.cluster_symmetric(n, 1.0, 1.0), np.linspace(0, 1, n))

    def test_cluster_symmetric_respects_dmax(self):
        """No spacing in the returned vector exceeds dmax."""
        dmax = 0.06
        d = np.diff(util.cluster_symmetric(21, 1.3, dmax))
        assert np.all(d <= dmax + 1e-6)

    def test_cluster_symmetric_requires_odd_n(self):
        """Even n, or n < 3, raises ValueError."""
        for bad in (0, 1, 2, 4, 10):
            with pytest.raises(ValueError):
                util.cluster_symmetric(bad, 1.2, 1.0)

    def test_cluster_symmetric_matches_mirrored_cluster(self):
        """Equivalent to mirroring a half-width cluster about the centreline."""
        n, ER = 65, 1.05
        half = util.cluster((n + 1) // 2, ER, 2.0)
        expected = np.concatenate([0.5 * half, 1.0 - 0.5 * half[-2::-1]])
        assert np.allclose(util.cluster_symmetric(n, ER, 1.0), expected, atol=1e-7)


if __name__ == "__main__":
    """Run all tests when executed directly."""
    pytest.main([__file__, "-v"])
"""Tests for angle/component velocity utility functions.

Test cases:
- test_angles_to_components_basic: Pure axial flow angle-to-component conversion
- test_angles_to_components_pure_radial_outward: Outward radial flow conversion
- test_angles_to_components_pure_radial_inward: Inward radial flow conversion
- test_angles_to_components_pure_tangential: Pure tangential flow conversion
- test_angles_to_components_45_degree_cases: 45-degree pitch and yaw angle cases
- test_angles_to_components_array_inputs: Array input handling and broadcasting
- test_angles_to_components_broadcasting: Scalar/array broadcasting behavior
- test_angles_to_components_edge_angles: Extreme angle cases and magnitude conservation
- test_angles_to_components_numerical_stability: Near-90-degree numerical stability
- test_angles_to_components_float32_consistency: Float32 precision preservation
- test_angles_to_components_comparison_with_math: Mathematical relationship verification
- test_components_to_angles_basic: Pure axial flow component-to-angle conversion
- test_components_to_angles_pure_radial_outward: Outward radial component conversion
- test_components_to_angles_pure_radial_inward: Inward radial component conversion
- test_components_to_angles_pure_tangential: Pure tangential component conversion
- test_components_to_angles_45_degree_cases: 45-degree angle reconstruction
- test_components_to_angles_array_inputs: Array input component conversion
- test_components_to_angles_zero_velocity: Zero velocity edge case handling
- test_components_to_angles_small_velocities: Very small velocity handling
- test_components_to_angles_float32_consistency: Float32 precision preservation
- test_roundtrip_consistency: Forward-inverse function consistency verification
- test_roundtrip_consistency_arrays: Array roundtrip consistency testing
- test_negative_angle_handling: Negative angle and component handling
- test_consistency_with_block_properties: Block flow angle property consistency
- test_consistency_with_rotating_block: Rotating block property consistency
- test_tangential_velocity_consistency: Absolute/relative tangential velocity verification
"""


def test_angles_to_components_basic():
    """Test basic functionality of angles_to_components."""
    V_rel = 100.0
    Alpha_rel_deg = 0.0
    Beta_deg = 0.0

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    # Pure axial flow: Alpha=0, Beta=0
    assert np.allclose(Vx, 100.0, rtol=1e-6)
    assert np.allclose(Vr, 0.0, atol=1e-10)
    assert np.allclose(Vt_rel, 0.0, atol=1e-10)


def test_angles_to_components_pure_radial_outward():
    """Test pure radial outward flow (Beta = +90°)."""
    V_rel = 50.0
    Alpha_rel_deg = 0.0
    Beta_deg = 90.0

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    # Pure radial outward: Alpha=0, Beta=+90
    assert np.allclose(Vx, 0.0, atol=1e-5)
    assert np.allclose(Vr, 50.0, rtol=1e-6)
    assert np.allclose(Vt_rel, 0.0, atol=1e-10)


def test_angles_to_components_pure_radial_inward():
    """Test pure radial inward flow (Beta = -90°)."""
    V_rel = 75.0
    Alpha_rel_deg = 0.0
    Beta_deg = -90.0

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    # Pure radial inward: Alpha=0, Beta=-90
    assert np.allclose(Vx, 0.0, atol=1e-5)
    assert np.allclose(Vr, -75.0, rtol=1e-6)
    assert np.allclose(Vt_rel, 0.0, atol=1e-10)


def test_angles_to_components_pure_tangential():
    """Test pure tangential flow (Alpha = 90°)."""
    V_rel = 80.0
    Alpha_rel_deg = 90.0
    Beta_deg = 0.0

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    # Pure tangential: Alpha=90, Beta=0
    assert np.allclose(Vx, 0.0, atol=1e-5)
    assert np.allclose(Vr, 0.0, atol=1e-10)
    assert np.allclose(Vt_rel, 80.0, rtol=1e-6)


def test_angles_to_components_45_degree_cases():
    """Test 45-degree angle cases."""
    V_rel = 100.0

    # 45-degree pitch angle
    Alpha_rel_deg = 0.0
    Beta_deg = 45.0

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    expected_component = 100.0 / np.sqrt(2)  # cos(45°) = sin(45°) = 1/√2

    assert np.allclose(Vx, expected_component, rtol=1e-6)
    assert np.allclose(Vr, expected_component, rtol=1e-6)
    assert np.allclose(Vt_rel, 0.0, atol=1e-10)

    # 45-degree yaw angle
    Alpha_rel_deg = 45.0
    Beta_deg = 0.0

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    assert np.allclose(Vx, expected_component, rtol=1e-6)
    assert np.allclose(Vr, 0.0, atol=1e-10)
    assert np.allclose(Vt_rel, expected_component, rtol=1e-6)


def test_angles_to_components_array_inputs():
    """Test with array inputs."""
    V_rel = np.array([50.0, 100.0, 150.0])
    Alpha_rel_deg = np.array([0.0, 30.0, 45.0])
    Beta_deg = np.array([0.0, 60.0, 90.0])

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    # Check shapes
    assert Vx.shape == (3,)
    assert Vr.shape == (3,)
    assert Vt_rel.shape == (3,)

    # Check first case: pure axial
    assert np.allclose(Vx[0], 50.0, rtol=1e-6)
    assert np.allclose(Vr[0], 0.0, atol=1e-10)
    assert np.allclose(Vt_rel[0], 0.0, atol=1e-10)

    # Check third case: pure radial
    assert np.allclose(Vx[2], 0.0, atol=1e-5)  # cos(45°) * cos(90°) ≈ 0
    assert np.allclose(Vr[2], 150.0 * np.cos(np.radians(45)), rtol=1e-6)


def test_angles_to_components_broadcasting():
    """Test broadcasting behavior."""
    V_rel = 100.0  # Scalar
    Alpha_rel_deg = np.array([0.0, 30.0])  # Array
    Beta_deg = 45.0  # Scalar

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    # Should broadcast to shape (2,)
    assert Vx.shape == (2,)
    assert Vr.shape == (2,)
    assert Vt_rel.shape == (2,)


def test_angles_to_components_edge_angles():
    """Test edge cases and extreme angles."""
    V_rel = 60.0

    # Test various edge angles
    test_cases = [
        (0.0, 0.0),  # Pure axial
        (0.0, 90.0),  # Pure radial outward
        (0.0, -90.0),  # Pure radial inward
        (90.0, 0.0),  # Pure tangential
        (0.0, 180.0),  # Reverse axial
        (180.0, 0.0),  # Reverse tangential
        (0.0, -180.0),  # Same as +180°
    ]

    for alpha, beta in test_cases:
        Vx, Vr, Vt_rel = angles_to_components(V_rel, alpha, beta)

        # Verify magnitude conservation (total velocity magnitude)
        V_total = np.sqrt(Vx**2 + Vr**2 + Vt_rel**2)
        assert np.allclose(V_total, V_rel, rtol=1e-6), (
            f"Failed for Alpha={alpha}, Beta={beta}"
        )


def test_angles_to_components_numerical_stability():
    """Test numerical stability at critical angles."""
    V_rel = 100.0

    # Test very close to ±90°
    near_90_angles = [89.9, 89.99, 89.999, 90.0, -89.9, -89.99, -89.999, -90.0]

    for beta in near_90_angles:
        Vx, Vr, Vt_rel = angles_to_components(V_rel, 0.0, beta)

        # For Beta near ±90°, Vx should be near 0 and Vr should dominate
        if abs(beta) > 89.0:
            assert abs(Vx) < 1.0, f"Vx too large for Beta={beta}: {Vx}"
            assert abs(Vr) > 90.0, f"Vr too small for Beta={beta}: {Vr}"


def test_angles_to_components_float32_consistency():
    """Test that output maintains float32 precision."""
    V_rel = np.array([50.0, 100.0], dtype=np.float32)
    Alpha_rel_deg = np.array([30.0, 45.0], dtype=np.float32)
    Beta_deg = np.array([60.0, 90.0], dtype=np.float32)

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    # Check that all outputs are float32
    assert Vx.dtype == np.float32
    assert Vr.dtype == np.float32
    assert Vt_rel.dtype == np.float32


def test_angles_to_components_comparison_with_math():
    """Test against known mathematical relationships."""
    V_rel = 100.0
    Alpha_rel_deg = 30.0
    Beta_deg = 60.0

    Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel_deg, Beta_deg)

    # Manual calculation for verification
    alpha_rad = np.radians(Alpha_rel_deg)
    beta_rad = np.radians(Beta_deg)

    V_rel_m_expected = V_rel * np.cos(alpha_rad)
    Vt_rel_expected = V_rel * np.sin(alpha_rad)
    Vx_expected = V_rel_m_expected * np.cos(beta_rad)
    Vr_expected = V_rel_m_expected * np.sin(beta_rad)

    assert np.allclose(Vx, Vx_expected, rtol=1e-6)
    assert np.allclose(Vr, Vr_expected, rtol=1e-6)
    assert np.allclose(Vt_rel, Vt_rel_expected, rtol=1e-6)


# Tests for inverse function


def test_components_to_angles_basic():
    """Test basic functionality of components_to_angles."""
    Vx = 100.0
    Vr = 0.0
    Vt_rel = 0.0

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    # Pure axial flow
    assert np.allclose(V_rel, 100.0, rtol=1e-6)
    assert np.allclose(Alpha_rel_deg, 0.0, atol=1e-10)
    assert np.allclose(Beta_deg, 0.0, atol=1e-10)


def test_components_to_angles_pure_radial_outward():
    """Test pure radial outward flow conversion."""
    Vx = 0.0
    Vr = 50.0
    Vt_rel = 0.0

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    # Pure radial outward
    assert np.allclose(V_rel, 50.0, rtol=1e-6)
    assert np.allclose(Alpha_rel_deg, 0.0, atol=1e-10)
    assert np.allclose(Beta_deg, 90.0, rtol=1e-6)


def test_components_to_angles_pure_radial_inward():
    """Test pure radial inward flow conversion."""
    Vx = 0.0
    Vr = -75.0
    Vt_rel = 0.0

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    # Pure radial inward
    assert np.allclose(V_rel, 75.0, rtol=1e-6)
    assert np.allclose(Alpha_rel_deg, 0.0, atol=1e-10)
    assert np.allclose(Beta_deg, -90.0, rtol=1e-6)


def test_components_to_angles_pure_tangential():
    """Test pure tangential flow conversion."""
    Vx = 0.0
    Vr = 0.0
    Vt_rel = 80.0

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    # Pure tangential
    assert np.allclose(V_rel, 80.0, rtol=1e-6)
    assert np.allclose(Alpha_rel_deg, 90.0, rtol=1e-6)
    assert np.allclose(Beta_deg, 0.0, atol=1e-10)


def test_components_to_angles_45_degree_cases():
    """Test 45-degree angle reconstructions."""
    # Test 45-degree pitch
    expected_component = 100.0 / np.sqrt(2)
    Vx = expected_component
    Vr = expected_component
    Vt_rel = 0.0

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    assert np.allclose(V_rel, 100.0, rtol=1e-6)
    assert np.allclose(Alpha_rel_deg, 0.0, atol=1e-10)
    assert np.allclose(Beta_deg, 45.0, rtol=1e-6)

    # Test 45-degree yaw
    Vx = expected_component
    Vr = 0.0
    Vt_rel = expected_component

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    assert np.allclose(V_rel, 100.0, rtol=1e-6)
    assert np.allclose(Alpha_rel_deg, 45.0, rtol=1e-6)
    assert np.allclose(Beta_deg, 0.0, atol=1e-10)


def test_components_to_angles_array_inputs():
    """Test with array inputs."""
    Vx = np.array([50.0, 0.0, 0.0])
    Vr = np.array([0.0, 100.0, 0.0])
    Vt_rel = np.array([0.0, 0.0, 150.0])

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    # Check shapes
    assert V_rel.shape == (3,)
    assert Alpha_rel_deg.shape == (3,)
    assert Beta_deg.shape == (3,)

    # Check values
    assert np.allclose(V_rel, [50.0, 100.0, 150.0], rtol=1e-6)
    assert np.allclose(Alpha_rel_deg, [0.0, 0.0, 90.0], rtol=1e-6)
    assert np.allclose(Beta_deg, [0.0, 90.0, 0.0], rtol=1e-6)


def test_components_to_angles_zero_velocity():
    """Test zero velocity handling."""
    Vx = 0.0
    Vr = 0.0
    Vt_rel = 0.0

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    # Zero velocity should return all zeros
    assert np.allclose(V_rel, 0.0, atol=1e-12)
    assert np.allclose(Alpha_rel_deg, 0.0, atol=1e-12)
    assert np.allclose(Beta_deg, 0.0, atol=1e-12)


def test_components_to_angles_small_velocities():
    """Test handling of very small velocities."""
    # Very small but non-zero velocities
    Vx = 1e-10
    Vr = 1e-11
    Vt_rel = 1e-12

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    # Should not crash and should give reasonable results
    assert np.isfinite(V_rel)
    assert np.isfinite(Alpha_rel_deg)
    assert np.isfinite(Beta_deg)


def test_components_to_angles_float32_consistency():
    """Test that output maintains float32 precision."""
    Vx = np.array([50.0, 0.0], dtype=np.float32)
    Vr = np.array([0.0, 100.0], dtype=np.float32)
    Vt_rel = np.array([30.0, 40.0], dtype=np.float32)

    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(Vx, Vr, Vt_rel)

    # Check that all outputs are float32
    assert V_rel.dtype == np.float32
    assert Alpha_rel_deg.dtype == np.float32
    assert Beta_deg.dtype == np.float32


def test_roundtrip_consistency():
    """Test that angles_to_components and components_to_angles are inverses."""
    # Test various combinations (avoid pure axial case that gives ±180°)
    test_cases = [
        (100.0, 0.0, 15.0),  # Near axial
        (100.0, 0.0, 90.0),  # Pure radial outward
        (100.0, 0.0, -90.0),  # Pure radial inward
        (100.0, 90.0, 0.0),  # Pure tangential
        (100.0, 30.0, 45.0),  # Mixed flow
        (50.0, -45.0, -60.0),  # Negative angles
    ]

    for V_rel_orig, Alpha_rel_orig, Beta_orig in test_cases:
        # Forward: angles to components
        Vx, Vr, Vt_rel = angles_to_components(V_rel_orig, Alpha_rel_orig, Beta_orig)

        # Inverse: components back to angles
        V_rel_recovered, Alpha_rel_recovered, Beta_recovered = components_to_angles(
            Vx, Vr, Vt_rel
        )

        # Check that we get back the original values
        assert np.allclose(V_rel_recovered, V_rel_orig, rtol=1e-6), (
            f"V_rel mismatch for case {V_rel_orig, Alpha_rel_orig, Beta_orig}"
        )
        assert np.allclose(Alpha_rel_recovered, Alpha_rel_orig, rtol=1e-5), (
            f"Alpha mismatch for case {V_rel_orig, Alpha_rel_orig, Beta_orig}"
        )

        # Handle angle wrapping: Beta can differ by 360° for equivalent angles.
        # 1e-3 degrees, not 1e-5: atan2/degrees float32 rounding through this
        # branch cut differs by a few ULP across platforms (observed 1.5e-5
        # degrees off on arm64 vs exact on x86_64 for the ±180 case).
        Beta_diff = np.abs(Beta_recovered - Beta_orig)
        angle_equivalent = (
            Beta_diff < 1e-3
            or np.abs(Beta_diff - 360) < 1e-3
            or np.abs(Beta_diff - 180) < 1e-3
        )
        if not angle_equivalent:
            # For pure tangential case (Alpha=90°), Vx≈0 can give ambiguous
            # Beta. 1e-4 not 1e-6: Vx's noise floor here is V_rel * float32
            # eps (~4e-6 for V_rel=100), not machine epsilon itself.
            pure_tangential = np.abs(Alpha_rel_orig) == 90.0 and np.abs(Vx) < 1e-4
            if not pure_tangential:
                assert False, (
                    f"Beta mismatch for case {V_rel_orig, Alpha_rel_orig, Beta_orig}: got {Beta_recovered}, expected {Beta_orig}"
                )


def test_roundtrip_consistency_arrays():
    """Test roundtrip consistency with arrays."""
    V_rel_orig = np.array([50.0, 100.0, 150.0])
    Alpha_rel_orig = np.array([0.0, 30.0, -45.0])
    Beta_orig = np.array([45.0, -60.0, 90.0])

    # Forward: angles to components
    Vx, Vr, Vt_rel = angles_to_components(V_rel_orig, Alpha_rel_orig, Beta_orig)

    # Inverse: components back to angles
    V_rel_recovered, Alpha_rel_recovered, Beta_recovered = components_to_angles(
        Vx, Vr, Vt_rel
    )

    # Check roundtrip accuracy
    assert np.allclose(V_rel_recovered, V_rel_orig, rtol=1e-6)
    assert np.allclose(Alpha_rel_recovered, Alpha_rel_orig, rtol=1e-5)
    assert np.allclose(Beta_recovered, Beta_orig, rtol=1e-5)


def test_negative_angle_handling():
    """Test proper handling of negative angles."""
    # Test negative Alpha (reverse swirl)
    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(50.0, 0.0, -30.0)
    assert Alpha_rel_deg < 0, "Negative Vt_rel should give negative Alpha"

    # Test negative Beta (inward radial)
    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(0.0, -50.0, 0.0)
    assert Beta_deg < 0, "Negative Vr should give negative Beta"

    # Test negative Vx (reverse axial)
    V_rel, Alpha_rel_deg, Beta_deg = components_to_angles(-50.0, 0.0, 0.0)
    # atan2(0, -50) is exactly pi, but degrees(atan2(...)) in float32 lands
    # a few ULP off 180.0 depending on the platform's libm (observed
    # 179.99998 on arm64 vs exactly 180.0 on x86_64) -- not a formula bug,
    # this branch cut is inherently platform-sensitive at the float32 level.
    assert abs(abs(Beta_deg) - 180.0) < 1e-3 or abs(Beta_deg) < 1e-10, (
        "Negative Vx should give ±180° Beta"
    )


def test_consistency_with_block_properties():
    """Test that utility functions are consistent with block flow angle properties."""
    # Create a test block with known flow field from scratch
    fluid = ember.fluid.PerfectFluid(gamma=1.4, cp=1005.0, mu=1.8e-5, Pr=0.72)

    # Create block and set coordinates
    ni = 10
    block = ember.block.Block(shape=(ni,))
    block.set_fluid(fluid)

    # Set up coordinates
    r = np.linspace(0.5, 1.0, ni, dtype=np.float32)
    x = np.zeros(ni, dtype=np.float32)
    t = np.zeros(ni, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Set up rotation
    block.set_Omega(500.0)

    # Create mixed flow field manually using our new setter
    Po_inlet, To_inlet = 101325.0, 300.0
    Ma_rel = 0.5
    Alpha_rel = np.zeros(ni)  # No relative swirl (pure radial in relative frame)
    Beta = np.full(ni, 90.0)  # Pure radial flow

    ember.set_iterative.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Test 1: Forward conversion (components → angles)
    V_rel_calc, Alpha_rel_calc, Beta_calc = components_to_angles(
        block.Vx, block.Vr, block.Vt_rel
    )

    # Check that calculated angles match block properties
    assert np.allclose(Alpha_rel_calc, block.Alpha_rel, rtol=1e-5), "Alpha_rel mismatch"
    assert np.allclose(Beta_calc, block.Beta, rtol=1e-5), "Beta mismatch"

    # Check that calculated relative velocity magnitude is correct
    V_rel_expected = np.sqrt(block.Vx**2 + block.Vr**2 + block.Vt_rel**2)
    assert np.allclose(V_rel_calc, V_rel_expected, rtol=1e-6), (
        "V_rel magnitude mismatch"
    )

    # Test 2: Backward conversion (angles → components)
    Vx_calc, Vr_calc, Vt_rel_calc = angles_to_components(
        V_rel_calc, Alpha_rel_calc, Beta_calc
    )

    # Check roundtrip consistency
    assert np.allclose(Vx_calc, block.Vx, rtol=1e-5), "Vx roundtrip mismatch"
    assert np.allclose(Vr_calc, block.Vr, rtol=1e-5), "Vr roundtrip mismatch"
    assert np.allclose(Vt_rel_calc, block.Vt_rel, rtol=1e-5), (
        "Vt_rel roundtrip mismatch"
    )


def test_consistency_with_rotating_block():
    """Test consistency with rotating block (rotor) properties."""

    # Create a rotating block from scratch
    fluid = ember.fluid.PerfectFluid(gamma=1.4, cp=1005.0, mu=1.8e-5, Pr=0.72)

    # Create block and coordinates
    ni = 8
    block = ember.block.Block(shape=(ni,))
    block.set_fluid(fluid)

    # Set up coordinates
    r = np.linspace(0.5, 1.0, ni, dtype=np.float32)  # radius_ratio = 2.0
    x = np.zeros(ni, dtype=np.float32)
    t = np.zeros(ni, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Set rotation (lower to avoid convergence issues)
    block.set_Omega(300.0)

    # Create pure radial flow field
    Po_inlet, To_inlet = 101325.0, 300.0
    Ma_rel = 0.3  # Lower Mach number for better convergence
    Alpha_rel = np.zeros(ni)  # No relative swirl (pure radial in relative frame)
    Beta = np.full(ni, 90.0)  # Pure radial outward flow

    ember.set_iterative.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Test the utility functions with rotating block
    V_rel_calc, Alpha_rel_calc, Beta_calc = components_to_angles(
        block.Vx, block.Vr, block.Vt_rel
    )

    # Check consistency with block properties
    assert np.allclose(Alpha_rel_calc, block.Alpha_rel, rtol=1e-5), (
        "Rotor Alpha_rel mismatch"
    )
    assert np.allclose(Beta_calc, block.Beta, rtol=1e-5), "Rotor Beta mismatch"

    # For pure radial rotor, Alpha_rel should be ~0 and Beta should be ±90°
    assert np.allclose(Alpha_rel_calc, 0.0, atol=1e-4), (
        "Pure radial rotor should have Alpha_rel ≈ 0"
    )
    assert np.allclose(np.abs(Beta_calc), 90.0, atol=1e-3), (
        "Pure radial rotor should have |Beta| ≈ 90°"
    )

    # Test roundtrip
    Vx_calc, Vr_calc, Vt_rel_calc = angles_to_components(
        V_rel_calc, Alpha_rel_calc, Beta_calc
    )

    assert np.allclose(Vx_calc, block.Vx, rtol=1e-5), "Rotor Vx roundtrip mismatch"
    assert np.allclose(Vr_calc, block.Vr, rtol=1e-5), "Rotor Vr roundtrip mismatch"
    assert np.allclose(Vt_rel_calc, block.Vt_rel, rtol=1e-5), (
        "Rotor Vt_rel roundtrip mismatch"
    )


def test_tangential_velocity_consistency():
    """Test that V,Alpha gives Vt and V_rel,Alpha_rel gives Vt_rel as calculated by block."""

    # Create a test block with known flow field from scratch
    fluid = ember.fluid.PerfectFluid(gamma=1.4, cp=1005.0, mu=1.8e-5, Pr=0.72)

    # Create block and coordinates
    ni = 10
    block = ember.block.Block(shape=(ni,))
    block.set_fluid(fluid)

    # Set up coordinates
    r = np.linspace(0.5, 1.0, ni, dtype=np.float32)  # radius_ratio = 2.0
    x = np.zeros(ni, dtype=np.float32)
    t = np.zeros(ni, dtype=np.float32)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    # Set moderate rotation to create tangential velocities
    block.set_Omega(400.0)

    # Create pure radial flow field
    Po_inlet, To_inlet = 101325.0, 300.0
    Ma_rel = 0.3  # Lower Mach number for better convergence
    Alpha_rel = np.zeros(ni)  # No relative swirl (pure radial in relative frame)
    Beta = np.full(ni, 90.0)  # Pure radial outward flow

    ember.set_iterative.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po_inlet, To_inlet, Ma_rel, Alpha_rel, Beta
    )

    # Test 1: V_rel, Alpha_rel → Vt_rel
    # Calculate relative velocity magnitude and yaw angle
    V_rel_block = np.sqrt(block.Vx**2 + block.Vr**2 + block.Vt_rel**2)
    Alpha_rel_block = block.Alpha_rel

    # Use utility function to get components
    Vx_calc, Vr_calc, Vt_rel_calc = angles_to_components(
        V_rel_block,
        Alpha_rel_block,
        90.0,  # Beta=90° for pure radial rotor
    )

    # Check that calculated Vt_rel matches block's Vt_rel
    assert np.allclose(Vt_rel_calc, block.Vt_rel, rtol=1e-5), (
        f"V_rel,Alpha_rel → Vt_rel mismatch: calc={Vt_rel_calc}, block={block.Vt_rel}"
    )

    # Test 2: V, Alpha → Vt (absolute frame)
    # Calculate absolute velocity magnitude and yaw angle
    V_abs_block = np.sqrt(block.Vx**2 + block.Vr**2 + block.Vt**2)
    Alpha_abs_block = block.Alpha

    # Use utility function to get absolute tangential velocity
    _, _, Vt_abs_calc = angles_to_components(
        V_abs_block,
        Alpha_abs_block,
        90.0,  # Beta=90° for pure radial rotor
    )

    # Check that calculated Vt matches block's Vt
    assert np.allclose(Vt_abs_calc, block.Vt, rtol=1e-5), (
        f"V,Alpha → Vt mismatch: calc={Vt_abs_calc}, block={block.Vt}"
    )

    # Test 3: Verify relationship Vt = Vt_rel + U
    U_block = block.U
    assert np.allclose(block.Vt, block.Vt_rel + U_block, rtol=1e-10), (
        "Block relationship Vt = Vt_rel + U not satisfied"
    )

    # Test 4: Cross-check with different flow angles
    # Create a block with mixed flow (not pure radial) from scratch
    ni_mixed = 5
    mixed_block = ember.block.Block(shape=(ni_mixed,))
    mixed_block.set_fluid(fluid)

    # Set up coordinates
    r_mixed = np.linspace(0.5, 0.75, ni_mixed, dtype=np.float32)  # radius_ratio = 1.5
    x_mixed = np.zeros(ni_mixed, dtype=np.float32)
    t_mixed = np.zeros(ni_mixed, dtype=np.float32)
    mixed_block.set_x(x_mixed)
    mixed_block.set_r(r_mixed)
    mixed_block.set_t(t_mixed)

    # Set rotation
    mixed_block.set_Omega(300.0)

    # Create mixed flow field (not pure radial)
    Po_mixed, To_mixed = 101325.0, 300.0
    Ma_rel_mixed = 0.3  # Lower Mach number for better convergence
    Alpha_rel_mixed = np.zeros(ni_mixed)  # No relative swirl
    Beta_mixed = np.full(ni_mixed, -45.0)  # Mixed axial/radial inward flow

    ember.set_iterative.set_Po_To_Ma_rel_Alpha_rel_Beta(
        mixed_block, Po_mixed, To_mixed, Ma_rel_mixed, Alpha_rel_mixed, Beta_mixed
    )

    # For the mixed flow, use actual flow angles (not assuming Beta=90°)
    V_rel_mixed = np.sqrt(mixed_block.Vx**2 + mixed_block.Vr**2 + mixed_block.Vt_rel**2)
    Alpha_rel_mixed = mixed_block.Alpha_rel
    Beta_mixed = mixed_block.Beta

    # Use utility function with actual angles
    Vx_mixed_calc, Vr_mixed_calc, Vt_rel_mixed_calc = angles_to_components(
        V_rel_mixed, Alpha_rel_mixed, Beta_mixed
    )

    # Check that all components match
    assert np.allclose(Vx_mixed_calc, mixed_block.Vx, rtol=1e-5), (
        "Mixed flow Vx mismatch"
    )
    assert np.allclose(Vr_mixed_calc, mixed_block.Vr, rtol=1e-5), (
        "Mixed flow Vr mismatch"
    )
    assert np.allclose(Vt_rel_mixed_calc, mixed_block.Vt_rel, rtol=1e-5), (
        "Mixed flow Vt_rel mismatch"
    )

    # Test absolute frame for mixed flow
    V_abs_mixed = np.sqrt(mixed_block.Vx**2 + mixed_block.Vr**2 + mixed_block.Vt**2)
    Alpha_abs_mixed = mixed_block.Alpha

    _, _, Vt_abs_mixed_calc = angles_to_components(
        V_abs_mixed, Alpha_abs_mixed, Beta_mixed
    )

    assert np.allclose(Vt_abs_mixed_calc, mixed_block.Vt, rtol=1e-5), (
        "Mixed flow absolute Vt mismatch"
    )


class TestCarveView:
    """Test zero-copy scratch-buffer carving (util.carve_view)."""

    def _buf(self):
        # Oversized Fortran-contiguous buffer, distinct values per element.
        return np.asfortranarray(
            np.arange(5 * 4 * 3 * 10, dtype=np.float32).reshape(5, 4, 3, 10)
        )

    def test_shares_memory(self):
        """Carved view aliases the source buffer (no copy)."""
        buf = self._buf()
        view = util.carve_view(buf, (5, 4, 3, 5))
        assert np.shares_memory(view, buf)

    def test_writes_visible_in_source(self):
        """Writing through the view mutates the underlying buffer."""
        buf = self._buf()
        view = util.carve_view(buf, (5, 4, 3, 5))
        view[...] = -1.0
        # The carved prefix in flat F-order must now be -1 in the source.
        n = 5 * 4 * 3 * 5
        assert np.all(buf.reshape(-1, order="F")[:n] == -1.0)

    def test_shape_and_contiguity(self):
        """View has the requested shape and stays Fortran-contiguous."""
        buf = self._buf()
        view = util.carve_view(buf, (5, 4, 3, 5))
        assert view.shape == (5, 4, 3, 5)
        assert view.flags.f_contiguous

    def test_single_shape_returns_bare_view(self):
        """One shape returns the array itself, not a length-1 list."""
        buf = self._buf()
        view = util.carve_view(buf, (5, 4, 3, 5))
        assert isinstance(view, np.ndarray)

    def test_multi_shape_packs_at_matching_offsets(self):
        """Successive views land on their end-to-end flat slices, zero-copy."""
        buf = self._buf()
        flat = buf.reshape(-1, order="F")
        a, b = util.carve_view(buf, (5, 4, 3, 3), (5, 4, 3))
        n_a = 5 * 4 * 3 * 3
        n_b = 5 * 4 * 3
        assert np.shares_memory(a, buf) and np.shares_memory(b, buf)
        assert np.array_equal(a.reshape(-1, order="F"), flat[:n_a])
        assert np.array_equal(b.reshape(-1, order="F"), flat[n_a : n_a + n_b])

    def test_disjoint_slots_do_not_overlap(self):
        """Packed carved slots address non-overlapping memory."""
        buf = self._buf()
        a, b = util.carve_view(buf, (5, 4, 3, 3), (5, 4, 3, 3))
        a[...] = 1.0
        b[...] = 2.0
        assert np.all(a == 1.0)
        assert np.all(b == 2.0)

    def test_overflow_raises(self):
        """Requesting more elements than the buffer holds raises ValueError."""
        buf = self._buf()
        cap = buf.size
        with pytest.raises(ValueError):
            util.carve_view(buf, (cap,), (1,))


def test_bcast_if_needed_already_right_shape_returns_same_object():
    """No broadcast_to call when the shape already matches -- same object back."""
    a = np.ones((3, 4, 5), dtype=np.float32)
    assert util.bcast_if_needed(a, (3, 4, 5)) is a


def test_bcast_if_needed_broadcasts_when_shape_differs():
    """A smaller-shaped array is broadcast to the target shape."""
    a = np.arange(5, dtype=np.float32)
    out = util.bcast_if_needed(a, (3, 5))
    assert out.shape == (3, 5)
    np.testing.assert_array_equal(out, np.broadcast_to(a, (3, 5)))


class TestUnwrapMeridional:
    """Test unwrap_meridional function.

    Test cases:
    - test_cylindrical_curve_is_distance_over_radius: the analytic case
    - test_conical_curve_matches_the_closed_form: r varying along a segment
    - test_a_refined_polyline_agrees_with_a_coarse_one: no quadrature error
    - test_curve_start_is_the_datum: zero where the integration begins
    - test_many_segments_accumulate: a polyline is its segments summed
    - test_a_point_off_the_curve_projects_onto_it: perpendicular offset dropped
    - test_a_point_beyond_the_end_clamps: no extrapolation off the curve
    - test_query_shape_is_preserved: grid-shaped input, grid-shaped output
    - test_two_cuts_of_one_curve_share_a_datum: the reason it takes a curve
    - test_a_curve_on_the_axis_is_refused: m' diverges there
    - test_a_curve_needs_two_points: nothing to integrate along
    """

    def test_cylindrical_curve_is_distance_over_radius(self):
        """At constant radius the integral collapses to m' = m / r."""
        radius = 2.0
        curve = np.array([[0.0, radius], [1.0, radius]])

        mp = util.unwrap_meridional(curve, np.array([[0.5, radius], [1.0, radius]]))

        np.testing.assert_allclose(mp, [0.5 / radius, 1.0 / radius], rtol=1e-6)

    def test_conical_curve_matches_the_closed_form(self):
        """With r linear along the segment, m' = (ds / dr) ln(r1 / r0)."""
        curve = np.array([[0.0, 1.0], [3.0, 5.0]])  # ds = 5, r from 1 to 5

        mp = util.unwrap_meridional(curve, curve[-1])

        np.testing.assert_allclose(mp, 5.0 / 4.0 * np.log(5.0), rtol=1e-6)

    def test_a_refined_polyline_agrees_with_a_coarse_one(self):
        """The segment integral is exact, so refining the curve changes nothing.

        A quadrature would converge on this answer from below; the closed form
        is already there.
        """
        ends = np.array([[0.0, 1.0], [3.0, 5.0]])
        fine = np.stack(
            [np.linspace(0.0, 3.0, 257), np.linspace(1.0, 5.0, 257)], axis=1
        )

        coarse_mp = util.unwrap_meridional(ends, ends[-1])
        fine_mp = util.unwrap_meridional(fine, fine[-1])

        np.testing.assert_allclose(fine_mp, coarse_mp, rtol=1e-6)

    def test_curve_start_is_the_datum(self):
        curve = np.array([[1.0, 2.0], [2.0, 2.0]])

        assert util.unwrap_meridional(curve, curve[0]) == pytest.approx(0.0)

    def test_many_segments_accumulate(self):
        """A point at the end of a polyline sums every segment before it."""
        curve = np.stack([np.linspace(0.0, 1.0, 11), np.full(11, 2.0)], axis=1)

        mp = util.unwrap_meridional(curve, np.array([[0.25, 2.0], [1.0, 2.0]]))

        np.testing.assert_allclose(mp, [0.125, 0.5], rtol=1e-6)

    def test_a_point_off_the_curve_projects_onto_it(self):
        """The perpendicular offset is discarded, not added to the distance.

        This is what lets a triangulated cut, or a blade surface, share the
        coordinate of the curve it sits near.
        """
        curve = np.array([[0.0, 2.0], [1.0, 2.0]])

        on = util.unwrap_meridional(curve, np.array([0.5, 2.0]))
        off = util.unwrap_meridional(curve, np.array([0.5, 2.3]))

        assert off == pytest.approx(on)

    def test_a_point_beyond_the_end_clamps(self):
        curve = np.array([[0.0, 2.0], [1.0, 2.0]])

        mp = util.unwrap_meridional(curve, np.array([[-5.0, 2.0], [7.0, 2.0]]))

        np.testing.assert_allclose(mp, [0.0, 0.5], rtol=1e-6)

    def test_query_shape_is_preserved(self):
        curve = np.array([[0.0, 2.0], [1.0, 2.0]])
        query = np.zeros((4, 3, 2))
        query[..., 1] = 2.0

        assert util.unwrap_meridional(curve, query).shape == (4, 3)

    def test_two_cuts_of_one_curve_share_a_datum(self):
        """Points evaluated separately land on one scale, with nothing to match.

        This is the whole reason the coordinate belongs to the curve: two
        blocks of a cut, or two cuts of a machine, are directly comparable
        because they were integrated from the same origin.
        """
        curve = np.stack([np.linspace(0.0, 2.0, 21), np.linspace(1.0, 3.0, 21)], axis=1)
        upstream = curve[:8]
        downstream = curve[12:]

        separately = np.concatenate(
            [
                util.unwrap_meridional(curve, upstream),
                util.unwrap_meridional(curve, downstream),
            ]
        )
        together = util.unwrap_meridional(curve, np.concatenate([upstream, downstream]))

        np.testing.assert_allclose(separately, together, rtol=1e-6)

    def test_a_curve_on_the_axis_is_refused(self):
        with pytest.raises(ValueError, match="diverges on the axis"):
            util.unwrap_meridional(np.array([[0.0, 0.0], [1.0, 1.0]]), np.zeros(2))

    def test_a_curve_needs_two_points(self):
        with pytest.raises(ValueError, match="at least two points"):
            util.unwrap_meridional(np.array([[0.0, 1.0]]), np.zeros(2))


if __name__ == "__main__":
    pytest.main([__file__])
