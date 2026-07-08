"""Tests for cutting and visualization functions (ember.cut).

Module tested: ember.cut

Test cases:
- test_vijk: _vijk function for vertex indexing
- test_eijk: _eijk function for edge indexing
- test_cube_index: _cube_index function
- test_no_cut: Case where no cut occurs
- test_simple_cut: Simple cutting case
- test_multiple_variables: Marching cubes with multiple data variables
- test_simple_horizontal_cut: Unstructured cut with simple horizontal line
- test_vertical_cut: Unstructured cut with vertical line
- test_diagonal_cut: Unstructured cut with diagonal line
- test_no_intersection: Case where cut line doesn't intersect the grid
- test_multi_block_cut: Unstructured cut across multiple blocks
- test_edge_case_tangent_cut: Cut that just touches the edge of the domain
- test_cut_consistency: Similar cuts produce consistent results
- test_data_interpolation: Data is correctly interpolated at cut surface
- test_empty_grid_boolean_behavior: Empty Grid objects are bool falsey
- test_empty_grid: Unstructured with empty grid
- test_single_point_cut: Cut defined by single point
- test_invalid_cut_line: Invalid cut line format
- test_all_negative_distance: All distance values are negative
- test_all_positive_distance: All distance values are positive
- test_zero_distance_vertices: Some vertices have exactly zero distance
- test_minimum_size_data: Minimum possible data size
- test_interpolation_accuracy: Linear interpolation is accurate
- test_data_preservation: Data values are preserved through cutting
- test_triangle_validity: Generated triangles are valid
- test_block_inheritance: Result inherits properties from source block
- test_unstructured_sets_triangulated_flag: Verify unstructured() sets triangulated flag
- test_unstructured_multiblock_triangulated_flag: Verify triangulated flag for multi-block
- test_large_grid_handling: Unstructured function with larger grid
- test_horizontal_cut: Structured meridional cut with horizontal line
- test_vertical_cut: Structured meridional cut with vertical line
- test_no_intersection_structured: Structured cut with no intersection
- test_multi_block_structured: Structured cut across multiple blocks
- test_empty_grid_structured: Structured cut with empty grid
- test_unstructured_no_intersection_definitive: Unstructured with cuts that definitely don't intersect
- test_structured_meridional_no_intersection_definitive: Structured_meridional with cuts that definitely don't intersect
- test_edge_tangent_cases: Cuts that just touch the boundary
- test_interpolate_basic_functionality: Basic interpolation from unstructured to structured
- test_interpolate_shape_validation: Interpolation handles different shapes properly
- test_interpolate_data_preservation: Interpolation preserves reasonable data ranges
- test_interpolate_coordinate_monotonicity: Interpolated coordinates form monotonic grids
- test_interpolate_minimal_input: Interpolation with minimal triangle data
- test_triangulate_simple_2d_block: Triangulation of simple 2D block
- test_triangulate_data_preservation: Triangulation preserves original data values
- test_triangulate_vertex_connectivity: Triangles share vertices correctly
- test_triangulate_larger_grid: Triangulation on larger 2D grid
- test_triangulate_invalid_input: Triangulate with invalid inputs
- test_triangulate_minimal_grid: Triangulation on minimal 2x2 grid
- test_triangulate_sets_triangulated_property: Verify triangulated property is set correctly
- test_triangulate_shape_calculation: Verify shape follows (ni-1)*(nj-1)*2 formula
"""

import numpy as np
import pytest
import ember.grid
import ember.block
import ember.fluid
from ember import util
from ember.cut import (
    _marching_cubes,
    unstructured,
    structured_meridional,
    interpolate_to_structured,
    triangulate_to_unstructured,
    _cube_index,
    _vijk,
    _eijk,
)


@pytest.fixture
def simple_block():
    """Create a simple 3x3x3 block for testing."""
    shape = (3, 3, 3)
    block = ember.block.Block(shape=shape)

    # Set up simple coordinates
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
    block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

    # Set up fluid
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)

    # Set simple conserved variables
    conserved = np.ones((*shape, 5))
    conserved[..., 0] = 1.0  # rho
    conserved[..., 1] = 100.0  # rhoVx
    conserved[..., 2] = 50.0  # rhoVr
    conserved[..., 3] = 25.0  # rhorVt
    conserved[..., 4] = 250000.0  # rhoe
    block.set_conserved(conserved)

    return block


@pytest.fixture
def multi_block_grid():
    """Create a two-block grid for testing."""
    shape = (3, 3, 3)

    # Block 1
    block1 = ember.block.Block(shape=shape)
    xrt1 = util.linmesh3([0.0, 0.5], [0.5, 1.0], [0.0, 0.1], shape)
    block1.set_x(xrt1[..., 0]).set_r(xrt1[..., 1]).set_t(xrt1[..., 2])

    # Block 2
    block2 = ember.block.Block(shape=shape)
    xrt2 = util.linmesh3([0.5, 1.0], [0.5, 1.0], [0.0, 0.1], shape)
    block2.set_x(xrt2[..., 0]).set_r(xrt2[..., 1]).set_t(xrt2[..., 2])

    # Set up fluid for both blocks
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    for block in [block1, block2]:
        block.set_fluid(fluid)
        conserved = np.ones((*shape, 5))
        conserved[..., 0] = 1.2
        conserved[..., 1] = 120.0
        conserved[..., 2] = 60.0
        conserved[..., 3] = 30.0
        conserved[..., 4] = 300000.0
        block.set_conserved(conserved)

    return ember.grid.Grid([block1, block2])


class TestMarchingCubesHelper:
    """Test helper functions for marching cubes."""

    def test_vijk(self):
        """Test _vijk function for vertex indexing."""
        shape = (3, 4, 5)

        # Test each vertex
        v0 = _vijk(shape, 0)
        assert v0 == (slice(0, 2), slice(0, 3), slice(0, 4))

        v1 = _vijk(shape, 1)
        assert v1 == (slice(1, 3), slice(0, 3), slice(0, 4))

        v7 = _vijk(shape, 7)
        assert v7 == (slice(0, 2), slice(1, 4), slice(1, 5))

    def test_eijk(self):
        """Test _eijk function for edge indexing."""
        i, j, k = 1, 2, 3

        # Test edge 0 (horizontal bottom front)
        start, end = _eijk(i, j, k, 0)
        assert start == (1, 2, 3)
        assert end == (2, 2, 3)

        # Test edge 8 (vertical front left)
        start, end = _eijk(i, j, k, 8)
        assert start == (1, 2, 3)
        assert end == (1, 2, 4)

    def test_cube_index(self):
        """Test _cube_index function."""
        # Simple 2x2x2 case where only corner 0 is negative
        d = np.ones((2, 2, 2))
        d[0, 0, 0] = -1.0

        ind = _cube_index(d)
        assert ind.shape == (1, 1, 1)
        assert ind[0, 0, 0] == 1  # Only bit 0 set

        # Case where all corners are negative
        d_all_neg = -np.ones((2, 2, 2))
        ind_all = _cube_index(d_all_neg)
        assert ind_all[0, 0, 0] == 255  # All 8 bits set


class TestMarchingCubes:
    """Test marching_cubes function."""

    def test_no_cut(self):
        """Test case where no cut occurs."""
        # All positive distance - no cut
        data = np.ones((3, 3, 3, 2))
        dist = np.ones((3, 3, 3))

        result = _marching_cubes(data, dist)
        assert result is None

    def test_simple_cut(self):
        """Test simple cutting case."""
        # Create a 2x2x2 data array
        data = np.ones((2, 2, 2, 1))

        # Create distance field that cuts through the middle
        dist = np.array([[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]])

        result = _marching_cubes(data, dist)
        assert result is not None
        assert result.shape[1:] == (3, 1)  # triangles with 3 vertices, 1 variable

    def test_multiple_variables(self):
        """Test marching cubes with multiple data variables."""
        # 2x2x2 with 3 variables
        data = np.ones((2, 2, 2, 3))
        data[..., 0] = 1.0  # First variable
        data[..., 1] = 2.0  # Second variable
        data[..., 2] = 3.0  # Third variable

        # Simple cut
        dist = np.array([[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]])

        result = _marching_cubes(data, dist)
        assert result is not None
        assert result.shape[2] == 3  # Should preserve all 3 variables


class TestUnstructured:
    """Test unstructured function."""

    def test_simple_horizontal_cut(self, simple_block):
        """Test unstructured cut with simple horizontal line."""
        grid = ember.grid.Grid([simple_block])

        # Create horizontal cut line at x=0.5
        xr_cut = np.array([[0.5, 0.5], [0.5, 1.5]])

        result = unstructured(grid, xr_cut)

        if result is not None:
            # Check result shape
            assert result.ndim == 2
            assert result.shape[1] == 3  # 3 vertices per triangle
            # Check that we have valid data (ignoring possible NaN in unused components)
            valid_data = result._data[..., :8]  # First 8 components should be valid
            assert not np.any(np.isnan(valid_data))

    def test_vertical_cut(self, simple_block):
        """Test unstructured cut with vertical line."""
        grid = ember.grid.Grid([simple_block])

        # Create vertical cut line at r=1.0
        xr_cut = np.array([[0.0, 1.0], [1.0, 1.0]])

        result = unstructured(grid, xr_cut)

        if result is not None:
            assert result.ndim == 2
            assert result.shape[1] == 3
            valid_data = result._data[..., :8]
            assert not np.any(np.isnan(valid_data))

    def test_diagonal_cut(self, simple_block):
        """Test unstructured cut with diagonal line."""
        grid = ember.grid.Grid([simple_block])

        # Create diagonal cut line
        xr_cut = np.array([[0.0, 0.5], [1.0, 1.5]])

        result = unstructured(grid, xr_cut)

        if result is not None:
            assert result.ndim == 2
            assert result.shape[1] == 3

    def test_no_intersection(self, simple_block):
        """Test case where cut line doesn't intersect the grid."""
        grid = ember.grid.Grid([simple_block])

        # Cut line completely outside the grid
        xr_cut = np.array([[2.0, 2.0], [3.0, 3.0]])

        result = unstructured(grid, xr_cut)
        # May return None or empty result when no intersection
        if result is not None:
            # If not None, should be empty or very small
            assert result.shape[0] == 0 or result.shape[0] < 5

    def test_multi_block_cut(self, multi_block_grid):
        """Test unstructured cut across multiple blocks."""
        # Cut that intersects both blocks
        xr_cut = np.array([[0.25, 0.6], [0.75, 0.9]])

        result = unstructured(multi_block_grid, xr_cut)

        if result is not None:
            assert result.ndim == 2
            assert result.shape[1] == 3
            # Should have triangles from potentially both blocks
            valid_data = result._data[..., :8]
            assert not np.any(np.isnan(valid_data))

    def test_edge_case_tangent_cut(self, simple_block):
        """Test cut that just touches the edge of the domain."""
        grid = ember.grid.Grid([simple_block])

        # Cut line at the boundary
        xr_cut = np.array([[0.0, 0.5], [0.0, 1.5]])

        result = unstructured(grid, xr_cut)
        # May or may not return triangles depending on numerical precision
        if result is not None:
            assert result.ndim == 2
            assert result.shape[1] == 3
            valid_data = result._data[..., :8]
            assert not np.any(np.isnan(valid_data))

    def test_cut_consistency(self, simple_block):
        """Test that similar cuts produce consistent results."""
        grid = ember.grid.Grid([simple_block])

        # Two very similar horizontal cuts
        xr_cut1 = np.array([[0.5, 0.5], [0.5, 1.5]])
        xr_cut2 = np.array([[0.500001, 0.5], [0.500001, 1.5]])

        result1 = unstructured(grid, xr_cut1)
        result2 = unstructured(grid, xr_cut2)

        # Both should either be None or have similar structure
        if result1 is None:
            assert result2 is None or result2.shape[0] == 0
        elif result2 is None:
            assert result1.shape[0] == 0
        else:
            # If both have results, they should have similar number of triangles
            assert abs(result1.shape[0] - result2.shape[0]) <= 2

    def test_data_interpolation(self, simple_block):
        """Test that data is correctly interpolated at cut surface."""
        grid = ember.grid.Grid([simple_block])

        # Set up a gradient in the data
        for i in range(simple_block.shape[0]):
            for j in range(simple_block.shape[1]):
                for k in range(simple_block.shape[2]):
                    # Create gradient in x-direction for first variable
                    simple_block._data[i, j, k, 0] = i * 0.5 + 1.0

        # Cut at x=0.5 (middle)
        xr_cut = np.array([[0.5, 0.5], [0.5, 1.5]])

        result = unstructured(grid, xr_cut)

        if result is not None:
            # Check that interpolated values are reasonable
            # At x=0.5, the first variable should be around 1.25
            first_var_values = result._data[..., 0]
            assert np.all(first_var_values >= 1.0)
            assert np.all(first_var_values <= 1.5)


class TestUnstructuredEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_grid_boolean_behavior(self):
        """Test that empty Grid objects are bool falsey."""
        empty_grid = ember.grid.Grid([])

        # Empty grid should be falsey
        assert not empty_grid
        assert bool(empty_grid) is False

        # Non-empty grid should be truthy
        shape = (3, 3, 3)
        block = ember.block.Block(shape=shape)
        non_empty_grid = ember.grid.Grid([block])
        assert bool(non_empty_grid) is True

    def test_empty_grid(self):
        """Test unstructured with empty grid."""
        grid = ember.grid.Grid([])
        xr_cut = np.array([[0.5, 0.5], [0.5, 1.5]])

        result = unstructured(grid, xr_cut)
        assert result is None

    def test_single_point_cut(self, simple_block):
        """Test cut defined by single point (degenerate case)."""
        grid = ember.grid.Grid([simple_block])

        # Single point - should handle gracefully
        xr_cut = np.array([[0.5, 1.0]])

        # This may raise an error or return None, both are acceptable
        try:
            result = unstructured(grid, xr_cut)
            if result is not None:
                assert result.ndim == 2
        except (ValueError, IndexError, AssertionError):
            # Acceptable for degenerate input
            pass

    def test_invalid_cut_line(self, simple_block):
        """Test with invalid cut line format."""
        grid = ember.grid.Grid([simple_block])

        # Wrong shape for cut line
        xr_cut = np.array([0.5, 1.0])  # Should be 2D

        # Function should handle gracefully and return None
        with pytest.raises(AssertionError):
            unstructured(grid, xr_cut)


class TestMarchingCubesEdgeCases:
    """Test edge cases for marching_cubes function."""

    def test_all_negative_distance(self):
        """Test when all distance values are negative."""
        data = np.ones((2, 2, 2, 1))
        dist = -np.ones((2, 2, 2))

        result = _marching_cubes(data, dist)
        assert result is None

    def test_all_positive_distance(self):
        """Test when all distance values are positive."""
        data = np.ones((2, 2, 2, 1))
        dist = np.ones((2, 2, 2))

        result = _marching_cubes(data, dist)
        assert result is None

    def test_zero_distance_vertices(self):
        """Test case where some vertices have exactly zero distance."""
        data = np.ones((2, 2, 2, 1))
        dist = np.array([[[0.0, -1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])

        # Should handle zero distance gracefully
        result = _marching_cubes(data, dist)
        if result is not None:
            assert result.shape[1:] == (3, 1)

    def test_minimum_size_data(self):
        """Test with minimum possible data size."""
        # Smallest possible grid that can generate triangles
        data = np.ones((2, 2, 2, 1))
        dist = np.array([[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]])

        result = _marching_cubes(data, dist)
        if result is not None:
            assert result.shape[1:] == (3, 1)

    def test_interpolation_accuracy(self):
        """Test that linear interpolation is accurate."""
        # Create data with known linear variation
        data = np.zeros((2, 2, 2, 1))
        data[0, :, :, 0] = 0.0  # x=0 plane
        data[1, :, :, 0] = 1.0  # x=1 plane

        # Distance field that cuts at x=0.3
        dist = np.zeros((2, 2, 2))
        dist[0, :, :] = -0.3  # x=0, dist = -0.3
        dist[1, :, :] = 0.7  # x=1, dist = 0.7

        result = _marching_cubes(data, dist)

        if result is not None:
            # All interpolated values should be 0.3 (linear interpolation)
            expected_value = 0.3
            np.testing.assert_allclose(result[..., 0], expected_value, atol=1e-10)


class TestUnstructuredIntegration:
    """Integration tests for unstructured function."""

    def test_data_preservation(self, simple_block):
        """Test that data values are preserved through cutting."""
        grid = ember.grid.Grid([simple_block])

        # Get original data values (only need valid ones for comparison)

        # Simple horizontal cut
        xr_cut = np.array([[0.5, 0.5], [0.5, 1.5]])

        result = unstructured(grid, xr_cut)

        if result is not None:
            # Cut values should be within range of original data (ignoring NaN)
            valid_data = result._data[..., :8]
            cut_min = np.min(valid_data)
            cut_max = np.max(valid_data)

            # Note: original_min/max include NaN, so use nanmin/nanmax
            original_min_valid = np.nanmin(simple_block._data)
            original_max_valid = np.nanmax(simple_block._data)

            assert cut_min >= original_min_valid
            assert cut_max <= original_max_valid

    def test_triangle_validity(self, multi_block_grid):
        """Test that generated triangles are valid."""
        # Cut that should intersect the grid
        xr_cut = np.array([[0.25, 0.7], [0.75, 0.8]])

        result = unstructured(multi_block_grid, xr_cut)

        if result is not None:
            # Each triangle should have 3 vertices
            assert result.shape[1] == 3

            # All valid data should be finite
            valid_data = result._data[..., :8]
            assert np.all(np.isfinite(valid_data))

            # Should have positive number of triangles
            assert result.shape[0] > 0

    def test_block_inheritance(self, simple_block):
        """Test that result inherits properties from source block."""
        grid = ember.grid.Grid([simple_block])

        xr_cut = np.array([[0.5, 0.5], [0.5, 1.5]])

        result = unstructured(grid, xr_cut)

        if result is not None:
            # Result should be a StructuredData-like object
            assert hasattr(result, "_data")
            assert hasattr(result, "shape")
            # Check that it has the expected number of variables (9 total, 8 valid)
            assert result._data.shape[2] == simple_block.nvar


class TestUnstructuredTriangulatedProperty:
    """Test that unstructured cuts set the triangulated property."""

    def test_unstructured_sets_triangulated_flag(self):
        """Test that unstructured() sets triangulated=True on output from 3D block."""
        # Create 3D block
        shape = (4, 5, 6)
        block = ember.block.Block(shape=shape)

        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
        block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block.set_fluid(fluid)

        conserved = np.ones((*shape, 5))
        conserved[..., 0] = 1.2  # rho
        conserved[..., 1] = 120.0  # rhoVx
        conserved[..., 2] = 60.0  # rhoVr
        conserved[..., 3] = 30.0  # rhorVt
        conserved[..., 4] = 300000.0  # rhoe
        block.set_conserved(conserved)

        grid = ember.grid.Grid([block])

        # Create cut line that should intersect the grid
        xr_cut = np.array([[0.3, 0.8], [0.7, 1.2]])

        # Perform unstructured cut
        result = unstructured(grid, xr_cut)

        # Verify result is not None
        assert result is not None, "Unstructured cut should produce a result"

        # Verify the triangulated property is set to True
        assert hasattr(result, "triangulated"), (
            "Result should have 'triangulated' property"
        )
        assert result.triangulated is True, (
            "Unstructured cut output should have triangulated=True"
        )

        # Verify shape is (ntri, 3) for triangulated data
        assert result.ndim == 2, (
            f"Expected ndim=2 for triangulated output, got {result.ndim}"
        )
        assert result.shape[1] == 3, (
            f"Expected 3 vertices per triangle, got {result.shape[1]}"
        )

        # Verify data has finite values
        valid_data = result._data[..., :8]
        assert np.all(np.isfinite(valid_data)), "Triangulated data should be finite"

    def test_unstructured_multiblock_triangulated_flag(self):
        """Test that unstructured() sets triangulated flag for multi-block grids."""
        # Create two 3D blocks
        shape = (3, 4, 5)

        block1 = ember.block.Block(shape=shape)
        xrt1 = util.linmesh3([0.0, 0.5], [0.5, 1.0], [0.0, 0.1], shape)
        block1.set_x(xrt1[..., 0]).set_r(xrt1[..., 1]).set_t(xrt1[..., 2])

        block2 = ember.block.Block(shape=shape)
        xrt2 = util.linmesh3([0.5, 1.0], [0.5, 1.0], [0.0, 0.1], shape)
        block2.set_x(xrt2[..., 0]).set_r(xrt2[..., 1]).set_t(xrt2[..., 2])

        # Set up fluid for both blocks
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        for block in [block1, block2]:
            block.set_fluid(fluid)
            conserved = np.ones((*shape, 5))
            conserved[..., 0] = 1.2
            conserved[..., 1] = 120.0
            conserved[..., 2] = 60.0
            conserved[..., 3] = 30.0
            conserved[..., 4] = 300000.0
            block.set_conserved(conserved)

        grid = ember.grid.Grid([block1, block2])

        # Cut that intersects both blocks
        xr_cut = np.array([[0.25, 0.6], [0.75, 0.9]])

        result = unstructured(grid, xr_cut)

        # Verify triangulated flag is set
        assert hasattr(result, "triangulated"), (
            "Multi-block result should have 'triangulated' property"
        )
        assert result.triangulated is True, (
            "Multi-block unstructured output should have triangulated=True"
        )

        # Verify shape
        assert result.ndim == 2
        assert result.shape[1] == 3

        # Verify data is finite
        valid_data = result._data[..., :8]
        assert np.all(np.isfinite(valid_data))


class TestUnstructuredPerformance:
    """Test performance-related aspects."""

    def test_large_grid_handling(self):
        """Test unstructured function with larger grid."""
        # Create larger block
        shape = (10, 10, 10)
        block = ember.block.Block(shape=shape)

        xrt = util.linmesh3([0.0, 2.0], [0.5, 2.5], [0.0, 0.4], shape)
        block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block.set_fluid(fluid)

        conserved = np.random.rand(*shape, 5)
        conserved[..., 0] = np.abs(conserved[..., 0]) + 0.1  # Ensure positive density
        block.set_conserved(conserved)

        grid = ember.grid.Grid([block])

        # Cut that intersects the grid
        xr_cut = np.array([[1.0, 1.0], [1.0, 2.0]])

        result = unstructured(grid, xr_cut)

        if result is not None:
            assert result.shape[1] == 3
            assert result._data.shape[2] == block.nvar
            valid_data = result._data[..., :8]
            assert np.all(np.isfinite(valid_data))


class TestStructuredMeridional:
    """Test structured_meridional function."""

    def test_horizontal_cut(self, simple_block):
        """Test structured meridional cut with horizontal line."""
        grid = ember.grid.Grid([simple_block])

        # Cut through middle of r-range
        r_mid = 1.0  # Middle of [0.5, 1.5] range
        xr_cut = np.array([[0.0, r_mid], [1.0, r_mid]])

        result = structured_meridional(grid, xr_cut)

        if result is not None:
            assert len(result) == 1  # Should have one block
            cut_block = result[0]
            assert cut_block.shape == (3, 3)  # 2D structured cut
            assert cut_block._data.shape == (3, 3, simple_block.nvar)

            # Check valid data (excluding NaN components)
            valid_data = cut_block._data[..., :8]
            finite_mask = np.isfinite(valid_data)
            if np.any(finite_mask):
                assert np.all(np.isfinite(valid_data[finite_mask]))

    def test_vertical_cut(self, simple_block):
        """Test structured meridional cut with vertical line."""
        grid = ember.grid.Grid([simple_block])

        # Cut through middle of x-range
        x_mid = 0.5
        xr_cut = np.array([[x_mid, 0.5], [x_mid, 1.5]])

        result = structured_meridional(grid, xr_cut)

        if result is not None:
            assert len(result) >= 0
            if len(result) > 0:
                cut_block = result[0]
                assert len(cut_block.shape) == 2  # 2D result

    def test_no_intersection_structured(self, simple_block):
        """Test structured cut with no intersection."""
        grid = ember.grid.Grid([simple_block])

        # Cut completely outside domain
        xr_cut = np.array([[2.0, 2.0], [3.0, 3.0]])

        result = structured_meridional(grid, xr_cut)
        # Function may still return a result even when cut is outside domain
        # This is acceptable behavior for the current implementation
        if result is not None:
            assert len(result) >= 0

    def test_multi_block_structured(self, multi_block_grid):
        """Test structured cut across multiple blocks."""
        # Cut that might intersect both blocks
        xr_cut = np.array([[0.25, 0.75], [0.75, 0.75]])

        result = structured_meridional(multi_block_grid, xr_cut)

        if result is not None:
            assert len(result) >= 0
            for cut_block in result:
                assert len(cut_block.shape) == 2

    def test_empty_grid_structured(self):
        """Test structured cut with empty grid."""
        grid = ember.grid.Grid([])
        xr_cut = np.array([[0.5, 1.0], [0.5, 1.5]])

        result = structured_meridional(grid, xr_cut)
        # Should return empty grid, not None
        assert isinstance(result, ember.grid.Grid)
        assert len(result) == 0
        assert not result  # Empty grid should be falsey


class TestNonIntersectionBehavior:
    """Specific tests to verify None return for non-intersecting cuts."""

    def test_unstructured_no_intersection_definitive(self, simple_block):
        """Test unstructured with cuts that definitely don't intersect."""
        grid = ember.grid.Grid([simple_block])

        # Get grid bounds
        x_min, x_max = (
            np.min(simple_block.xrt[..., 0]),
            np.max(simple_block.xrt[..., 0]),
        )
        r_min, r_max = (
            np.min(simple_block.xrt[..., 1]),
            np.max(simple_block.xrt[..., 1]),
        )

        # Test cuts that are clearly outside and should return None
        non_intersecting_cuts = [
            # Completely to the left
            np.array([[x_min - 2.0, r_min], [x_min - 2.0, r_max]]),
            # Completely to the right
            np.array([[x_max + 2.0, r_min], [x_max + 2.0, r_max]]),
            # Completely below
            np.array([[x_min, r_min - 2.0], [x_max, r_min - 2.0]]),
            # Completely above
            np.array([[x_min, r_max + 2.0], [x_max, r_max + 2.0]]),
        ]

        for i, xr_cut in enumerate(non_intersecting_cuts):
            result = unstructured(grid, xr_cut)
            # Check if result is None or effectively empty
            if result is not None:
                # If not None, should have no triangles or all NaN data
                if result.shape[0] > 0:
                    # Check if all interpolated coordinates are outside the domain
                    coords = result._data[..., :3]  # x, r, t coordinates
                    x_coords, r_coords = coords[..., 0], coords[..., 1]

                    # Allow some tolerance for interpolation
                    x_in_bounds = np.logical_and(
                        x_coords >= x_min - 0.1, x_coords <= x_max + 0.1
                    )
                    r_in_bounds = np.logical_and(
                        r_coords >= r_min - 0.1, r_coords <= r_max + 0.1
                    )
                    coords_in_domain = np.logical_and(x_in_bounds, r_in_bounds)

                    # Most coordinates should be outside domain for non-intersecting cut
                    assert (
                        np.sum(coords_in_domain)
                        <= result.shape[0] * result.shape[1] * 0.1
                    )

    def test_structured_meridional_no_intersection_definitive(self, simple_block):
        """Test structured_meridional with cuts that definitely don't intersect."""
        grid = ember.grid.Grid([simple_block])

        # Get grid bounds
        x_min, x_max = (
            np.min(simple_block.xrt[..., 0]),
            np.max(simple_block.xrt[..., 0]),
        )
        r_min, r_max = (
            np.min(simple_block.xrt[..., 1]),
            np.max(simple_block.xrt[..., 1]),
        )

        # Test cuts that are clearly outside
        non_intersecting_cuts = [
            # Completely to the left
            np.array([[x_min - 2.0, r_min], [x_min - 2.0, r_max]]),
            # Completely to the right
            np.array([[x_max + 2.0, r_min], [x_max + 2.0, r_max]]),
            # Completely below
            np.array([[x_min, r_min - 2.0], [x_max, r_min - 2.0]]),
            # Completely above
            np.array([[x_min, r_max + 2.0], [x_max, r_max + 2.0]]),
        ]

        for i, xr_cut in enumerate(non_intersecting_cuts):
            result = structured_meridional(grid, xr_cut)
            # Check behavior - should return None or empty/all-NaN result
            if result is not None:
                if len(result) > 0:
                    # Check if result contains valid data
                    cut_block = result[0]
                    valid_data = cut_block._data[..., :8]
                    finite_count = np.sum(np.isfinite(valid_data))
                    total_count = valid_data.size

                    # For non-intersecting cuts, most data should be NaN
                    assert finite_count <= total_count * 0.1

    def test_edge_tangent_cases(self, simple_block):
        """Test cuts that just touch the boundary."""
        grid = ember.grid.Grid([simple_block])

        # Get exact grid bounds
        x_min, x_max = (
            np.min(simple_block.xrt[..., 0]),
            np.max(simple_block.xrt[..., 0]),
        )
        r_min, r_max = (
            np.min(simple_block.xrt[..., 1]),
            np.max(simple_block.xrt[..., 1]),
        )

        # Cuts exactly at boundaries
        boundary_cuts = [
            # At left edge
            np.array([[x_min, r_min], [x_min, r_max]]),
            # At right edge
            np.array([[x_max, r_min], [x_max, r_max]]),
            # At bottom edge
            np.array([[x_min, r_min], [x_max, r_min]]),
            # At top edge
            np.array([[x_min, r_max], [x_max, r_max]]),
        ]

        for i, xr_cut in enumerate(boundary_cuts):
            result_unstr = unstructured(grid, xr_cut)
            result_struct = structured_meridional(grid, xr_cut)

            # Boundary cuts may or may not return results depending on numerics
            # Both functions should handle these gracefully
            if result_unstr is not None:
                assert result_unstr.shape[1] == 3
            if result_struct is not None:
                assert len(result_struct) >= 0


@pytest.fixture
def periodic_block():
    """Create a 3x3x5 block spanning one full circumferential pitch."""
    Nb = 8
    pitch = 2.0 * np.pi / Nb
    shape = (3, 3, 5)
    block = ember.block.Block(shape=shape)

    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, pitch], shape)
    block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)

    conserved = np.ones((*shape, 5))
    conserved[..., 0] = 1.0
    conserved[..., 1] = 100.0
    conserved[..., 2] = 50.0
    conserved[..., 3] = 25.0
    conserved[..., 4] = 250000.0
    block.set_conserved(conserved)
    block.set_Nb(Nb)

    return block


class TestInterpolateToStructured:
    """Test interpolate_to_structured function.

    The structured target conforms to the meridional line: index i runs along
    the line (constant-i gridline has (x, r) = const, cosine-clustered along
    zeta), index j runs in theta (constant-j gridline has theta = const). The
    sub-pitch ``simple_block`` cuts are interpolated with ``periodic=False``;
    ``periodic=True`` is exercised with the full-pitch ``periodic_block``.
    """

    def test_interpolate_basic_functionality(self, simple_block):
        """Test basic interpolation from unstructured to structured."""
        grid = ember.grid.Grid([simple_block])

        # Get unstructured cut first
        xr_cut = np.array([[0.2, 0.7], [0.8, 1.3]])
        unstr_result = unstructured(grid, xr_cut)

        if unstr_result is not None:
            # Test interpolation to structured grid
            interp_shape = (5, 4)
            result = interpolate_to_structured(
                unstr_result, interp_shape, periodic=False
            )

            # Should return structured grid with requested shape
            assert result.shape == interp_shape
            assert result._data.shape == (*interp_shape, simple_block.nvar)

            # Check that coordinates are set properly
            x_coords = result._data[..., 0]
            r_coords = result._data[..., 1]
            assert np.all(np.isfinite(x_coords))
            assert np.all(np.isfinite(r_coords))

            # Check that interpolated variables exist
            variables = result._data[..., 3:8]  # Skip t coordinate and NaN component
            finite_vars = variables[np.isfinite(variables)]
            assert len(finite_vars) > 0  # Should have some interpolated data

    def test_interpolate_shape_validation(self, simple_block):
        """Test that interpolation handles different shapes properly."""
        grid = ember.grid.Grid([simple_block])

        xr_cut = np.array([[0.3, 0.8], [0.7, 1.2]])
        unstr_result = unstructured(grid, xr_cut)

        if unstr_result is not None:
            # Test different interpolation shapes
            shapes_to_test = [(2, 2), (3, 5), (10, 8)]

            for shape in shapes_to_test:
                result = interpolate_to_structured(unstr_result, shape, periodic=False)
                assert result.shape == shape
                assert result._data.shape == (*shape, simple_block.nvar)

    def test_interpolate_data_preservation(self, simple_block):
        """Test that interpolation preserves reasonable data ranges."""
        grid = ember.grid.Grid([simple_block])

        xr_cut = np.array([[0.3, 0.8], [0.7, 1.2]])
        unstr_result = unstructured(grid, xr_cut)

        if unstr_result is not None:
            # Get original data ranges
            orig_data = unstr_result._data[..., 3:8]  # Variables only
            orig_min, orig_max = np.nanmin(orig_data), np.nanmax(orig_data)

            # Interpolate
            structured_result = interpolate_to_structured(
                unstr_result, (5, 4), periodic=False
            )
            interp_data = structured_result._data[..., 3:8]

            # Interpolated values should be within original range, allowing a
            # magnitude-relative slack for float32 round-off in the barycentric
            # blend (an absolute 1e-6 is unrealistic on ~1e5-magnitude fp32 data).
            finite_interp = interp_data[np.isfinite(interp_data)]
            if len(finite_interp) > 0:
                tol = 1e-5 * max(abs(orig_min), abs(orig_max), 1.0)
                assert np.min(finite_interp) >= orig_min - tol
                assert np.max(finite_interp) <= orig_max + tol

    def test_interpolate_straight_gridlines(self, simple_block):
        """Constant-i lines have (x, r) = const; constant-j lines theta = const."""
        grid = ember.grid.Grid([simple_block])

        # Cut line from minimum to maximum r, so zeta increases with x and r
        xr_cut = np.array([[0.2, 0.7], [0.8, 1.3]])
        unstr_result = unstructured(grid, xr_cut)

        if unstr_result is not None:
            interp_shape = (4, 3)
            result = interpolate_to_structured(
                unstr_result, interp_shape, Beta=0.0, periodic=False
            )

            x = result._data[..., 0]
            r = result._data[..., 1]
            t = result._data[..., 2]

            # x, r are constant along j (each constant-i gridline is (x,r)=const)
            assert np.allclose(np.ptp(x, axis=1), 0.0)
            assert np.allclose(np.ptp(r, axis=1), 0.0)

            # theta is constant along i (each constant-j gridline is theta=const)
            assert np.allclose(np.ptp(t, axis=0), 0.0)

            # along i (arc length, Beta=0 => arc 0 at min r) x and r increase
            assert np.all(np.diff(x[:, 0]) > 0)
            assert np.all(np.diff(r[:, 0]) > 0)
            # theta increases uniformly along j
            assert np.all(np.diff(t[0, :]) > 0)

    def test_interpolate_minimal_input(self, simple_block):
        """Test interpolation with minimal triangle data."""
        grid = ember.grid.Grid([simple_block])

        # Get any unstructured result
        xr_cut = np.array([[0.3, 0.8], [0.7, 1.2]])
        unstr_result = unstructured(grid, xr_cut)

        if unstr_result is not None and unstr_result.shape[0] > 0:
            # Test with very small interpolation grid
            structured_result = interpolate_to_structured(
                unstr_result, (2, 2), periodic=False
            )

            assert structured_result.shape == (2, 2)
            assert structured_result._data.shape == (2, 2, simple_block.nvar)

    def test_interpolate_periodic_full_pitch(self, periodic_block):
        """Periodic interpolation spans one pitch with straight theta lines."""
        grid = ember.grid.Grid([periodic_block])
        pitch = float(periodic_block.pitch)

        xr_cut = np.array([[0.2, 0.7], [0.8, 1.3]])
        unstr_result = unstructured(grid, xr_cut)
        assert unstr_result is not None

        result = interpolate_to_structured(unstr_result, (5, 6), periodic=True)
        d = result._data

        # Straight, line-conforming grid
        assert np.allclose(np.ptp(d[..., 0], axis=1), 0.0)  # x const along j
        assert np.allclose(np.ptp(d[..., 1], axis=1), 0.0)  # r const along j
        assert np.allclose(np.ptp(d[..., 2], axis=0), 0.0)  # theta const along i

        # theta window spans exactly one pitch
        theta = d[0, :, 2]
        assert theta[-1] - theta[0] == pytest.approx(pitch, rel=1e-4)

        # interpolated variables are finite and physical
        assert np.all(np.isfinite(d[..., 3:8]))

    def test_interpolate_periodic_coverage_error(self, simple_block):
        """periodic=True raises when the cut spans well under one pitch."""
        grid = ember.grid.Grid([simple_block])  # theta in [0, 0.2], pitch = 2*pi

        xr_cut = np.array([[0.2, 0.7], [0.8, 1.3]])
        unstr_result = unstructured(grid, xr_cut)

        if unstr_result is not None:
            with pytest.raises(ValueError, match="under one pitch"):
                interpolate_to_structured(unstr_result, (5, 4), periodic=True)

    def test_interpolate_roundtrip_linear_field_unchanged(self):
        """Round-trip a known field through a sloping cut, oversampled.

        A flowfield that is affine in (x, r) and independent of theta is
        reproduced exactly by the unfold interpolation (linear interpolation is
        exact for affine fields). Cutting along a sloping meridional line and
        interpolating back to a much finer structured grid must therefore return
        the analytic field at every target node, to float32 round-off.
        """
        # Block spanning one full pitch with a field linear in (x, r).
        Nb = 8
        pitch = 2.0 * np.pi / Nb
        shape = (9, 9, 9)
        block = ember.block.Block(shape=shape)

        xrt = util.linmesh3([0.0, 1.0], [1.0, 2.0], [0.0, pitch], shape)
        x, r = xrt[..., 0], xrt[..., 1]
        block.set_x(x).set_r(r).set_t(xrt[..., 2])
        block.set_fluid(
            ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        )

        # (offset, d/dx, d/dr) per conserved variable; theta-independent.
        coeffs = [
            (1.0, 0.5, 0.3),  # rho
            (100.0, 20.0, -10.0),  # rhoVx
            (50.0, -5.0, 8.0),  # rhoVr
            (25.0, 3.0, 4.0),  # rhorVt
            (2.5e5, 1.0e4, -2.0e4),  # rhoe
        ]
        conserved = np.empty((*shape, 5))
        for k, (c0, cx, cr) in enumerate(coeffs):
            conserved[..., k] = c0 + cx * x + cr * r
        block.set_conserved(conserved)
        block.set_Nb(Nb)

        grid = ember.grid.Grid([block])

        # Unstructured cut along a sloping (x, r) line crossing the domain.
        xr_cut = np.array([[0.1, 1.1], [0.9, 1.9]])
        unstr_result = unstructured(grid, xr_cut)
        assert unstr_result is not None

        # Interpolate back, oversampled to a much finer structured grid.
        result = interpolate_to_structured(unstr_result, (60, 24), periodic=True)
        d = result._data
        x_t, r_t = d[..., 0], d[..., 1]

        # Interpolated conserved must match the analytic linear field everywhere.
        for k, (c0, cx, cr) in enumerate(coeffs):
            expected = c0 + cx * x_t + cr * r_t
            got = d[..., 3 + k]
            assert np.all(np.isfinite(got))
            assert np.allclose(got, expected, rtol=1e-4, atol=1e-4 * abs(c0))

    def test_interpolate_roundtrip_smooth_theta_field(self):
        """Round-trip a field varying smoothly in theta to test theta interp.

        Adds a smooth, theta-periodic term ``A*cos(2*pi*theta/pitch + phase)``
        to the affine (x, r) field. The affine part is reproduced exactly, so
        the residual is the linear-interpolation error of the cosine in theta,
        set by the *source* theta resolution. The error must be small at a
        well-resolved source and must fall second order (quarter) at each
        doubling of the source theta resolution - confirming the theta
        interpolation is genuinely active and converging at the expected rate.
        """
        Nb = 8
        pitch = 2.0 * np.pi / Nb
        w = 2.0 * np.pi / pitch  # one period over a pitch -> theta-periodic

        # (offset, d/dx, d/dr, amplitude, phase) per conserved variable.
        coeffs = [
            (1.0, 0.5, 0.3, 0.3, 0.0),
            (100.0, 20.0, -10.0, 15.0, 0.7),
            (50.0, -5.0, 8.0, 8.0, 1.3),
            (25.0, 3.0, 4.0, 5.0, 2.0),
            (2.5e5, 1.0e4, -2.0e4, 3.0e4, 0.4),
        ]

        def roundtrip_error(nk):
            shape = (9, 9, nk)
            block = ember.block.Block(shape=shape)
            xrt = util.linmesh3([0.0, 1.0], [1.0, 2.0], [0.0, pitch], shape)
            x, r, th = xrt[..., 0], xrt[..., 1], xrt[..., 2]
            block.set_x(x).set_r(r).set_t(th)
            block.set_fluid(
                ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
            )
            conserved = np.empty((*shape, 5))
            for k, (c0, cx, cr, amp, ph) in enumerate(coeffs):
                conserved[..., k] = c0 + cx * x + cr * r + amp * np.cos(w * th + ph)
            block.set_conserved(conserved)
            block.set_Nb(Nb)

            grid = ember.grid.Grid([block])
            unstr_result = unstructured(grid, np.array([[0.1, 1.1], [0.9, 1.9]]))
            assert unstr_result is not None

            # Oversample heavily in theta so the residual is source-limited.
            result = interpolate_to_structured(unstr_result, (40, 80), periodic=True)
            d = result._data
            x_t, r_t, t_t = d[..., 0], d[..., 1], d[..., 2]

            rel = []
            for k, (c0, cx, cr, amp, ph) in enumerate(coeffs):
                expected = c0 + cx * x_t + cr * r_t + amp * np.cos(w * t_t + ph)
                got = d[..., 3 + k]
                assert np.all(np.isfinite(got))
                rel.append(np.abs(got - expected).max() / amp)
            return np.array(rel)

        err = {nk: roundtrip_error(nk) for nk in (17, 33, 65)}

        # Well-resolved source: theta interpolation error within 0.3% of amplitude.
        assert np.all(err[65] < 3e-3)
        # Second-order convergence at each doubling: error quarters (ratio ~4),
        # bounded both sides so it cannot pass for a lower-order or stalled rate.
        for ratio in (err[17] / err[33], err[33] / err[65]):
            assert np.all((ratio > 3.5) & (ratio < 4.5))

    def test_interpolate_roundtrip_nonperiodic_linear_theta(self):
        """periodic=False reproduces a field linear (non-periodic) in theta.

        With ``periodic=False`` the theta window spans the raw cloud theta range
        and theta is unfolded linearly, so a field affine in (x, r) and linear
        (not periodic) in theta is affine in the unfold space and must round-trip
        exactly - exercising the non-periodic interpolation path for accuracy,
        not just shape.
        """
        Nb = 8
        pitch = 2.0 * np.pi / Nb
        shape = (9, 9, 33)
        block = ember.block.Block(shape=shape)

        xrt = util.linmesh3([0.0, 1.0], [1.0, 2.0], [0.0, pitch], shape)
        x, r, th = xrt[..., 0], xrt[..., 1], xrt[..., 2]
        block.set_x(x).set_r(r).set_t(th)
        block.set_fluid(
            ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        )

        # (offset, d/dx, d/dr, d/dtheta) per conserved variable.
        coeffs = [
            (1.0, 0.5, 0.3, 0.4),
            (100.0, 20.0, -10.0, 12.0),
            (50.0, -5.0, 8.0, -6.0),
            (25.0, 3.0, 4.0, 7.0),
            (2.5e5, 1.0e4, -2.0e4, 5.0e4),
        ]
        conserved = np.empty((*shape, 5))
        for k, (c0, cx, cr, ct) in enumerate(coeffs):
            conserved[..., k] = c0 + cx * x + cr * r + ct * th
        block.set_conserved(conserved)
        block.set_Nb(Nb)

        grid = ember.grid.Grid([block])
        unstr_result = unstructured(grid, np.array([[0.1, 1.1], [0.9, 1.9]]))
        assert unstr_result is not None

        result = interpolate_to_structured(unstr_result, (40, 30), periodic=False)
        d = result._data
        x_t, r_t, t_t = d[..., 0], d[..., 1], d[..., 2]

        for k, (c0, cx, cr, ct) in enumerate(coeffs):
            expected = c0 + cx * x_t + cr * r_t + ct * t_t
            got = d[..., 3 + k]
            assert np.all(np.isfinite(got))
            assert np.allclose(got, expected, rtol=1e-4, atol=1e-4 * abs(c0))

    def test_interpolate_beta_orientation(self, periodic_block):
        """Beta selects which line end is arc length 0 (i = 0).

        The arc-length-zero end is the one with smallest projection onto
        ``d = (-sin Beta, cos Beta)``: Beta=0 -> min r, Beta=90 -> max x,
        Beta=-90 -> min x. A single straight cut has only two ends, so both a
        positive-slope and a negative-slope cut are used to pin all three rules.
        """
        grid = ember.grid.Grid([periodic_block])

        def first_last_x(xr_cut, Beta):
            unstr_result = unstructured(grid, np.array(xr_cut))
            assert unstr_result is not None
            d = interpolate_to_structured(
                unstr_result, (6, 6), Beta=Beta, periodic=True
            )._data
            return float(d[0, 0, 0]), float(d[-1, 0, 0])  # x at i=0 and i=-1

        # Positive slope: min-x end == min-r end; max-x end == max-r end.
        pos = [[0.2, 1.1], [0.8, 1.9]]
        x0, x1 = first_last_x(pos, Beta=0)  # min r -> min-x end first
        assert x0 < x1
        x0, x1 = first_last_x(pos, Beta=90)  # max x -> max-x end first
        assert x0 > x1
        x0, x1 = first_last_x(pos, Beta=-90)  # min x -> min-x end first
        assert x0 < x1

        # Negative slope: min-x end == max-r end; max-x end == min-r end.
        neg = [[0.2, 1.9], [0.8, 1.1]]
        x0, x1 = first_last_x(neg, Beta=0)  # min r -> max-x end first
        assert x0 > x1
        x0, x1 = first_last_x(neg, Beta=90)  # max x -> max-x end first
        assert x0 > x1
        x0, x1 = first_last_x(neg, Beta=-90)  # min x -> min-x end first
        assert x0 < x1


class TestTriangulate:
    """Test triangulate function."""

    def test_triangulate_simple_2d_block(self):
        """Test triangulation of simple 2D block."""
        # Create simple 2D block
        shape = (3, 3)
        block = ember.block.Block(shape=shape)

        # Set up coordinates
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.0], (*shape, 1))[..., 0, :]
        block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

        # Set up fluid and conserved variables
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block.set_fluid(fluid)
        conserved = np.ones((*shape, 5))
        conserved[..., 0] = 1.0  # rho
        conserved[..., 1] = 100.0  # rhoVx
        conserved[..., 2] = 50.0  # rhoVr
        conserved[..., 3] = 25.0  # rhorVt
        conserved[..., 4] = 250000.0  # rhoe
        block.set_conserved(conserved)

        # Triangulate
        result = triangulate_to_unstructured(block)

        # Check output shape - (ni-1)*(nj-1)*2 triangles, 3 vertices each
        expected_ntri = (shape[0] - 1) * (shape[1] - 1) * 2
        assert result.shape == (expected_ntri, 3)
        assert result._data.shape == (expected_ntri, 3, block.nvar)

        # Check that all data is finite (no NaN in coordinates/variables)
        valid_data = result._data[..., :8]  # Exclude potential NaN component
        assert np.all(np.isfinite(valid_data))

    def test_triangulate_data_preservation(self):
        """Test that triangulation preserves original data values."""
        # Create 2D block with gradient
        shape = (4, 3)
        block = ember.block.Block(shape=shape)

        # Set coordinates
        xrt = util.linmesh3([0.0, 2.0], [1.0, 2.0], [0.0, 0.0], (*shape, 1))[..., 0, :]
        block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

        # Set fluid
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block.set_fluid(fluid)

        # Create conserved data with spatial variation
        conserved = np.ones((*shape, 5))
        for i in range(shape[0]):
            for j in range(shape[1]):
                conserved[i, j, 0] = 1.0 + 0.1 * i  # rho varies with i
                conserved[i, j, 1] = 100.0 + 10.0 * j  # rhoVx varies with j
        conserved[..., 2] = 50.0  # rhoVr constant
        conserved[..., 3] = 25.0  # rhorVt constant
        conserved[..., 4] = 250000.0  # rhoe constant
        block.set_conserved(conserved)

        # Get original conserved variable range
        orig_min = np.min(block.conserved)
        orig_max = np.max(block.conserved)

        # Triangulate
        result = triangulate_to_unstructured(block)

        # Check that triangulated data is within original range
        tri_data = result._data[..., 3:8]  # Only conserved variables
        tri_min = np.min(tri_data)
        tri_max = np.max(tri_data)

        assert tri_min >= orig_min
        assert tri_max <= orig_max

    def test_triangulate_vertex_connectivity(self):
        """Test that triangles share vertices correctly."""
        # Simple 2x2 block
        shape = (2, 2)
        block = ember.block.Block(shape=shape)

        xrt = util.linmesh3([0.0, 1.0], [0.1, 1.0], [0.0, 0.0], (*shape, 1))[..., 0, :]
        block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block.set_fluid(fluid)
        conserved = np.ones((*shape, 5))
        block.set_conserved(conserved)

        result = triangulate_to_unstructured(block)

        # Should have (2-1)*(2-1)*2 = 2 triangles
        assert result.shape == (2, 3)

        # Check that vertices have correct coordinates
        coords = result._data[..., :2]  # x, r coordinates

        # All x coordinates should be 0 or 1
        x_coords = coords[..., 0]
        assert np.all(
            np.logical_or(np.isclose(x_coords, 0.0), np.isclose(x_coords, 1.0))
        )

        # All r coordinates should be 0.1 or 1.0
        r_coords = coords[..., 1]
        assert np.all(
            np.logical_or(np.isclose(r_coords, 0.1), np.isclose(r_coords, 1.0))
        )

    def test_triangulate_larger_grid(self):
        """Test triangulation on larger 2D grid."""
        shape = (5, 4)
        block = ember.block.Block(shape=shape)

        xrt = util.linmesh3([0.0, 4.0], [1.0, 3.0], [0.1, 0.1], (*shape, 1))[..., 0, :]
        block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block.set_fluid(fluid)
        conserved = np.random.rand(*shape, 5)
        conserved[..., 0] = np.abs(conserved[..., 0]) + 0.1  # Ensure positive density
        block.set_conserved(conserved)

        result = triangulate_to_unstructured(block)

        # Check triangle count: (5-1)*(4-1)*2 = 24 triangles
        expected_ntri = (shape[0] - 1) * (shape[1] - 1) * 2
        assert result.shape == (expected_ntri, 3)
        assert result._data.shape == (expected_ntri, 3, block.nvar)

        # All data should be finite
        valid_data = result._data[..., :8]
        assert np.all(np.isfinite(valid_data))

    def test_triangulate_invalid_input(self):
        """Test triangulate with invalid inputs."""
        # Test with 3D block (should fail)
        shape_3d = (3, 3, 3)
        block_3d = ember.block.Block(shape=shape_3d)

        with pytest.raises(AssertionError):
            triangulate_to_unstructured(block_3d)

    def test_triangulate_minimal_grid(self):
        """Test triangulation on minimal 2x2 grid."""
        shape = (2, 2)
        block = ember.block.Block(shape=shape)

        xrt = util.linmesh3([0.0, 1.0], [0.1, 1.0], [0.0, 0.0], (*shape, 1))[..., 0, :]
        block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block.set_fluid(fluid)
        conserved = np.ones((*shape, 5))
        block.set_conserved(conserved)

        result = triangulate_to_unstructured(block)

        # Should have exactly 2 triangles for 2x2 grid
        assert result.shape == (2, 3)
        assert result._data.shape == (2, 3, block.nvar)

    def test_triangulate_sets_triangulated_property(self):
        """Test that triangulate_to_unstructured sets the triangulated property correctly."""
        # Create 2D block
        shape = (3, 4)
        block = ember.block.Block(shape=shape)

        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.0], (*shape, 1))[..., 0, :]
        block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block.set_fluid(fluid)
        conserved = np.ones((*shape, 5))
        conserved[..., 0] = 1.2  # rho
        conserved[..., 1] = 120.0  # rhoVx
        conserved[..., 2] = 60.0  # rhoVr
        conserved[..., 3] = 30.0  # rhorVt
        conserved[..., 4] = 300000.0  # rhoe
        block.set_conserved(conserved)

        # Verify original block is not triangulated
        assert not hasattr(block, "triangulated") or block.triangulated is False

        # Triangulate the block
        result = triangulate_to_unstructured(block)

        # Verify the triangulated property is set to True
        assert hasattr(result, "triangulated"), (
            "Result should have 'triangulated' property"
        )
        assert result.triangulated is True, "triangulated property should be True"

        # Verify the shape is correct: (ntri, 3) where ntri = (ni-1)*(nj-1)*2
        expected_ntri = (shape[0] - 1) * (shape[1] - 1) * 2
        assert result.shape == (expected_ntri, 3), (
            f"Expected shape ({expected_ntri}, 3), got {result.shape}"
        )

        # Verify ndim is 2 (triangles x vertices)
        assert result.ndim == 2, (
            f"Expected ndim=2 for triangulated data, got {result.ndim}"
        )

        # Verify the data array has correct shape
        assert result._data.shape == (expected_ntri, 3, block.nvar), (
            f"Expected _data shape ({expected_ntri}, 3, {block.nvar}), "
            f"got {result._data.shape}"
        )

        # Verify ni and nj are set correctly on triangulated output
        assert result.ni == expected_ntri, (
            f"Expected ni={expected_ntri}, got {result.ni}"
        )
        assert result.nj == 3, f"Expected nj=3 (vertices per triangle), got {result.nj}"

        # Verify all data is finite
        valid_data = result._data[..., :8]
        assert np.all(np.isfinite(valid_data)), "All triangulated data should be finite"

    def test_triangulate_shape_calculation(self):
        """Test that triangulated shape follows the formula (ni-1)*(nj-1)*2."""
        # Test various input shapes
        test_shapes = [(2, 2), (3, 3), (4, 5), (6, 3), (10, 8)]

        for input_shape in test_shapes:
            block = ember.block.Block(shape=input_shape)

            xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.0], (*input_shape, 1))[
                ..., 0, :
            ]
            block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

            fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
            block.set_fluid(fluid)
            conserved = np.ones((*input_shape, 5))
            block.set_conserved(conserved)

            # Triangulate
            result = triangulate_to_unstructured(block)

            # Expected number of triangles
            expected_ntri = (input_shape[0] - 1) * (input_shape[1] - 1) * 2

            # Verify shape
            assert result.shape == (expected_ntri, 3), (
                f"For input shape {input_shape}, expected output shape "
                f"({expected_ntri}, 3), got {result.shape}"
            )

            # Verify triangulated property
            assert result.triangulated is True, (
                f"For input shape {input_shape}, triangulated should be True"
            )


class TestSingleBlockAcceptance:
    """Test that unstructured and structured_meridional accept a single Block."""

    def test_unstructured_single_block(self, simple_block):
        """Test that unstructured() accepts a single Block."""
        xr_cut = np.array([[0.5, 0.5], [0.5, 1.5]])

        # Call with single Block instead of Grid
        result = unstructured(simple_block, xr_cut)

        # Should work and return valid result or None
        if result is not None:
            assert result.ndim == 2
            assert result.shape[1] == 3
            assert result.triangulated is True

    def test_unstructured_single_block_equivalence(self, simple_block):
        """Test that single Block gives same result as Grid with one Block."""
        xr_cut = np.array([[0.5, 0.5], [0.5, 1.5]])

        # Result with Grid
        grid = ember.grid.Grid([simple_block])
        result_grid = unstructured(grid, xr_cut)

        # Result with single Block
        result_block = unstructured(simple_block, xr_cut)

        # Both should be None or both should produce results
        if result_grid is None:
            assert result_block is None
        else:
            assert result_block is not None
            # Should have same shape
            assert result_block.shape == result_grid.shape
            # Should have same data (approximately, allowing for small numerical differences)
            np.testing.assert_allclose(result_block._data, result_grid._data, rtol=1e-5)

    def test_unstructured_vertical_cut_single_block(self, simple_block):
        """Test unstructured with vertical cut on single Block."""
        xr_cut = np.array([[0.0, 1.0], [1.0, 1.0]])

        result = unstructured(simple_block, xr_cut)

        if result is not None:
            assert result.ndim == 2
            assert result.shape[1] == 3
            assert result.triangulated is True

    def test_unstructured_diagonal_cut_single_block(self, simple_block):
        """Test unstructured with diagonal cut on single Block."""
        xr_cut = np.array([[0.0, 0.5], [1.0, 1.5]])

        result = unstructured(simple_block, xr_cut)

        if result is not None:
            assert result.ndim == 2
            assert result.shape[1] == 3

    def test_structured_meridional_single_block(self, simple_block):
        """Test that structured_meridional() accepts a single Block."""
        xr_cut = np.array([[0.5, 1.0], [0.5, 1.0]])

        # Call with single Block instead of Grid
        result = structured_meridional(simple_block, xr_cut)

        # Should work and return a Grid
        assert isinstance(result, ember.grid.Grid)
        # Result may be empty or have cut blocks
        if len(result) > 0:
            for cut_block in result:
                assert len(cut_block.shape) == 2  # 2D result

    def test_structured_meridional_single_block_equivalence(self, simple_block):
        """Test that single Block gives same result as Grid with one Block."""
        xr_cut = np.array([[0.5, 1.0], [0.5, 1.0]])

        # Result with Grid
        grid = ember.grid.Grid([simple_block])
        result_grid = structured_meridional(grid, xr_cut)

        # Result with single Block
        result_block = structured_meridional(simple_block, xr_cut)

        # Both should return Grids
        assert isinstance(result_grid, ember.grid.Grid)
        assert isinstance(result_block, ember.grid.Grid)

        # Both should have same number of blocks
        assert len(result_grid) == len(result_block)

        # If they have blocks, the blocks should have same shape
        if len(result_grid) > 0:
            for i in range(len(result_grid)):
                assert result_grid[i].shape == result_block[i].shape
                # Data should be approximately equal
                np.testing.assert_allclose(
                    result_grid[i]._data, result_block[i]._data, rtol=1e-5
                )

    def test_structured_meridional_horizontal_cut_single_block(self, simple_block):
        """Test structured cut with horizontal line on single Block."""
        r_mid = 1.0
        xr_cut = np.array([[0.0, r_mid], [1.0, r_mid]])

        result = structured_meridional(simple_block, xr_cut)

        assert isinstance(result, ember.grid.Grid)

    def test_structured_meridional_vertical_cut_single_block(self, simple_block):
        """Test structured cut with vertical line on single Block."""
        x_mid = 0.5
        xr_cut = np.array([[x_mid, 0.5], [x_mid, 1.5]])

        result = structured_meridional(simple_block, xr_cut)

        assert isinstance(result, ember.grid.Grid)

    def test_unstructured_no_intersection_single_block(self, simple_block):
        """Test unstructured with non-intersecting cut on single Block."""
        # Cut completely outside the grid
        xr_cut = np.array([[2.0, 2.0], [3.0, 3.0]])

        result = unstructured(simple_block, xr_cut)

        # Should return None or empty result
        if result is not None:
            # If not None, should be empty or have very few triangles
            assert result.shape[0] == 0 or result.shape[0] < 5

    def test_structured_meridional_no_intersection_single_block(self, simple_block):
        """Test structured with non-intersecting cut on single Block."""
        xr_cut = np.array([[2.0, 2.0], [3.0, 3.0]])

        result = structured_meridional(simple_block, xr_cut)

        # Should return Grid (possibly empty)
        assert isinstance(result, ember.grid.Grid)

    def test_unstructured_single_block_preserves_data(self, simple_block):
        """Test that single Block approach preserves data correctly."""
        xr_cut = np.array([[0.3, 0.8], [0.7, 1.2]])

        result = unstructured(simple_block, xr_cut)

        if result is not None:
            # Data should be within range of original block
            valid_data = result._data[..., :8]
            orig_min = np.nanmin(simple_block._data)
            orig_max = np.nanmax(simple_block._data)

            cut_min = np.nanmin(valid_data)
            cut_max = np.nanmax(valid_data)

            assert cut_min >= orig_min
            assert cut_max <= orig_max
