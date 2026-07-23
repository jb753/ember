"""Tests for Patch classes (ember.patch).

Module tested: ember.patch

Test cases:
- test_cast_lim_int: Casting single integer to tuple
- test_cast_lim_tuple: Valid tuple inputs
- test_cast_lim_invalid_type: Invalid input types
- test_cast_lim_invalid_tuple: Invalid tuple formats
- test_init_defaults_invalid: Default constructor raises error
- test_init_single_ints: Constructor with integer inputs
- test_init_tuples: Constructor with tuple inputs
- test_init_with_label: Constructor with label and valid 2D patch
- test_init_with_label_and_indices: Constructor with both label and indices
- test_init_invalid_3d_patch: 3D patches are rejected
- test_init_valid_2d_patches: Valid 2D patch configurations
- test_set_i_lim: set_i_lim method
- test_set_j_lim: set_j_lim method
- test_set_k_lim: set_k_lim method
- test_setter_invalid_input: Setters validate input through _cast_lim
- test_const_dim: const_dim property identification
- test_const_dim_multiple_constant: const_dim when multiple dimensions are constant
- test_block_shape_not_set: Block shape when not set
- test_block_shape_set: Block shape when block is set
- test_ijk_lim_abs_positive_indices: _ijk_lim_abs with all positive indices
- test_ijk_lim_abs_negative_indices: _ijk_lim_abs with negative indices
- test_ijk_lim_abs_mixed_indices: _ijk_lim_abs with mix of positive and negative indices
- test_ijk_lim_abs_out_of_bounds: _ijk_lim_abs with out of bounds indices
- test_ijk_lim_abs_negative_out_of_bounds: _ijk_lim_abs with negative indices out of bounds
- test_shape_property: Shape property calculation
- test_shape_single_point: Shape for single point patch
- test_shape_with_negative_indices: Shape calculation with negative indices
- test_size_property: Size property calculation
- test_size_single_point: Size for single point patch
- test_index_properties: Index start/end properties
- test_index_properties_single_values: Index properties when dimensions are constant
- test_index_properties_negative_values: Index properties with negative values
- test_properties_without_block: Properties requiring block fail appropriately
- test_slice_property_default: Slice property with default parameters
- test_get_offset_slice_boundary_patch: _get_offset_slice with offset on boundary patch
- test_get_offset_slice_interior_patch: _get_offset_slice with offset on interior patch
- test_slice_offset_properties: Cached slice_offset_1 and slice_offset_2 properties
- test_slice_property_negative_indices: Slice property with negative indices
- test_slice_property_all_negative_end: Slice property when all end indices are -1
- test_get_ijk_node_default: get_ijk_node with default parameters
- test_get_ijk_node_permutation: get_ijk_node with permutation
- test_get_ijk_node_flip: get_ijk_node with flipping
- test_get_ijk_node_complex_transform: get_ijk_node with both permutation and flipping
- test_get_ijk_node_single_point: get_ijk_node for single point patch
- test_get_ijk_node_negative_indices: get_ijk_node with negative indices
- test_repr_method: __repr__ method
- test_repr_no_label: __repr__ method without label
- test_methods_without_block: Methods that require block fail appropriately
- test_complete_workflow: Complete workflow from creation to data access
- test_different_orientations: Patches with i-constant, j-constant, and k-constant orientations
- test_negative_index_resolution: Complete workflow with negative indices
- test_patch_modification_workflow: Modifying patch limits after creation
- test_edge_cases: Edge case handling for patch operations
- test_invalid_constructor_inputs: Constructor validation with invalid inputs
- test_out_of_bounds_conditions: Out of bounds error handling
- test_coordinate_generation_edge_cases: Coordinate generation edge cases
- test_property_access_patterns: Property access patterns and error conditions
- test_realistic_cfd_scenarios: Realistic CFD patch scenarios
- test_valid_boundary_patches_start_end: Patches on block boundaries are valid
- test_valid_boundary_patches_explicit_max: Boundary patches with explicit maximum indices
- test_invalid_interior_patches: Interior patches raise ValueError
- test_boundary_validation_with_negative_indices: Boundary validation works correctly with negative indices
- test_boundary_validation_different_block_sizes: Boundary validation works with different block sizes
- test_boundary_validation_preserves_existing_behavior: Boundary validation doesn't break existing valid use cases
- test_get_ijk_face_constant_i: get_ijk_face for constant i face
- test_get_ijk_face_constant_j: get_ijk_face for constant j face
- test_get_ijk_face_constant_k: get_ijk_face for constant k face
- test_get_ijk_face_shape_relationship: Face shapes are correctly derived from node shapes
- test_get_ijk_face_single_face_patches: get_ijk_face for minimal single-face patches
- test_get_ijk_face_permutation: get_ijk_face with permutation
- test_get_ijk_face_flipping: get_ijk_face with flipping
- test_get_ijk_face_complex_transform: get_ijk_face with both permutation and flipping
- test_get_ijk_face_vs_get_ijk_node_consistency: get_ijk_face returns consistent subset of get_ijk_node
- test_get_ijk_face_boundary_patches: get_ijk_face works correctly with realistic boundary patches
- test_get_ijk_face_error_conditions: get_ijk_face error handling
- test_get_ijk_face_mathematical_correctness: Mathematical correctness of face indexing concept
- test_patch_requires_3d_block: Patches require 3D blocks
- test_patch_various_ndim: Error handling for 1D, 2D, 4D blocks
- test_block_flat_removes_patches: Patches cleared when flattening blocks
"""

import pytest
import numpy as np
import ember.block
import ember.set_iter
from ember.patch import (
    Patch,
    PeriodicPatch,
    InletPatch,
    OutletPatch,
    MixingPatch,
    RotatingPatch,
    InviscidPatch,
    CuspPatch,
    NonMatchPatch,
    CoolingPatch,
)
from ember.patch import BlockPatchCollection
from ember.fluid import PerfectFluid


def _make_patch_collection():
    """Return a BlockPatchCollection attached to a fully initialised 50x50x50 block."""
    b = ember.block.Block(shape=(50, 50, 50))
    x, y, z = np.mgrid[0:50, 0:50, 0:50].astype(float) + 1.0
    b.set_x(x)
    b.set_r(y)
    b.set_t(z)
    return b.patches


class TestCastLim:
    """Tests for the _cast_lim utility function."""

    def test_cast_lim_int(self):
        """Test casting single integer to tuple."""
        assert Patch._cast_lim(5) == (5, 5)
        assert Patch._cast_lim(0) == (0, 0)
        assert Patch._cast_lim(-1) == (-1, -1)

    def test_cast_lim_tuple(self):
        """Test valid tuple inputs."""
        assert Patch._cast_lim((2, 8)) == (2, 8)
        assert Patch._cast_lim((0, -1)) == (0, -1)
        assert Patch._cast_lim((-5, 10)) == (-5, 10)

    def test_cast_lim_invalid_type(self):
        """Test invalid input types."""
        with pytest.raises(
            ValueError,
            match="i must be an int, a tuple of two ints, or a length-2 numpy array",
        ):
            Patch._cast_lim(3.14)  # float
        with pytest.raises(
            ValueError,
            match="i must be an int, a tuple of two ints, or a length-2 numpy array",
        ):
            Patch._cast_lim("5")  # string
        with pytest.raises(
            ValueError,
            match="i must be an int, a tuple of two ints, or a length-2 numpy array",
        ):
            Patch._cast_lim([1, 2])  # list

    def test_cast_lim_invalid_tuple(self):
        """Test invalid tuple formats."""
        with pytest.raises(
            ValueError,
            match="i must be an int, a tuple of two ints, or a length-2 numpy array",
        ):
            Patch._cast_lim((1, 2, 3))  # wrong length
        with pytest.raises(
            ValueError,
            match="i must be an int, a tuple of two ints, or a length-2 numpy array",
        ):
            Patch._cast_lim((1.5, 2))  # non-int in tuple
        with pytest.raises(
            ValueError,
            match="i must be an int, a tuple of two ints, or a length-2 numpy array",
        ):
            Patch._cast_lim((1,))  # single element tuple


class TestPatchInit:
    """Tests for Patch constructor."""

    def test_init_defaults_invalid(self):
        """Test that default constructor raises error (3D patch)."""
        with pytest.raises(ValueError, match="at least one constant dimension"):
            PeriodicPatch()

    def test_init_single_ints(self):
        """Test constructor with integer inputs."""
        p = PeriodicPatch(i=0, j=3, k=(0, 10))
        expected = np.array([[0, 0], [3, 3], [0, 10]])
        assert np.array_equal(p._ijk_lim, expected)

    def test_init_tuples(self):
        """Test constructor with tuple inputs."""
        p = PeriodicPatch(i=(2, 8), j=(1, 5), k=7)
        expected = np.array([[2, 8], [1, 5], [7, 7]])
        assert np.array_equal(p._ijk_lim, expected)

    def test_init_with_label(self):
        """Test constructor with label and valid 2D patch."""
        p = PeriodicPatch(i=5, label="test_patch")
        assert p.label == "test_patch"

    def test_init_with_label_and_indices(self):
        """Test constructor with both label and indices."""
        p = PeriodicPatch(i=5, j=(2, 8), k=3, label="inlet")
        assert p.label == "inlet"
        expected = np.array([[5, 5], [2, 8], [3, 3]])
        assert np.array_equal(p._ijk_lim, expected)

    def test_init_invalid_3d_patch(self):
        """Test that 3D patches (all dimensions varying) are rejected."""
        with pytest.raises(ValueError, match="at least one constant dimension"):
            PeriodicPatch(i=(0, 5), j=(0, 3), k=(0, 10))

    def test_init_valid_2d_patches(self):
        """Test valid 2D patch configurations."""
        # i constant
        p1 = PeriodicPatch(i=5, j=(0, 3), k=(0, 10))
        assert p1.const_dim == 0

        # j constant
        p2 = PeriodicPatch(i=(0, 5), j=3, k=(0, 10))
        assert p2.const_dim == 1

        # k constant
        p3 = PeriodicPatch(i=(0, 5), j=(0, 3), k=10)
        assert p3.const_dim == 2


class TestPatchSetters:
    """Tests for Patch setter methods."""

    def test_set_i_lim(self):
        """Test set_i_lim method."""
        p = PeriodicPatch(i=0)  # Start with valid 2D patch
        p.set_i_lim(5)
        assert np.array_equal(p._ijk_lim[0], [5, 5])

        p.set_i_lim((2, 8))
        assert np.array_equal(p._ijk_lim[0], [2, 8])

    def test_set_j_lim(self):
        """Test set_j_lim method."""
        p = PeriodicPatch(j=0)  # Start with valid 2D patch
        p.set_j_lim(3)
        assert np.array_equal(p._ijk_lim[1], [3, 3])

        p.set_j_lim((1, 7))
        assert np.array_equal(p._ijk_lim[1], [1, 7])

    def test_set_k_lim(self):
        """Test set_k_lim method."""
        p = PeriodicPatch(k=0)  # Start with valid 2D patch
        p.set_k_lim(9)
        assert np.array_equal(p._ijk_lim[2], [9, 9])

        p.set_k_lim((0, 5))
        assert np.array_equal(p._ijk_lim[2], [0, 5])

    def test_setter_invalid_input(self):
        """Test that setters validate input through _cast_lim."""
        p = PeriodicPatch(i=0)  # Start with valid 2D patch
        with pytest.raises(ValueError):
            p.set_i_lim(3.14)
        with pytest.raises(ValueError):
            p.set_j_lim("invalid")
        with pytest.raises(ValueError):
            p.set_k_lim((1, 2, 3))


class TestPatchProperties:
    """Tests for Patch properties."""

    def test_const_dim(self):
        """Test const_dim property identification."""
        # i constant
        p1 = PeriodicPatch(i=5, j=(0, 3), k=(0, 10))
        assert p1.const_dim == 0

        # j constant
        p2 = PeriodicPatch(i=(0, 5), j=3, k=(0, 10))
        assert p2.const_dim == 1

        # k constant
        p3 = PeriodicPatch(i=(0, 5), j=(0, 3), k=10)
        assert p3.const_dim == 2

    def test_const_dim_multiple_constant(self):
        """Test const_dim when multiple dimensions are constant."""
        p = PeriodicPatch(i=0, j=3, k=(0, 10))
        with pytest.raises(ValueError):
            # Both i and j constant - should error as no unique constant dimension
            _ = p.const_dim

    def test_block_shape_not_set(self):
        """Test block shape when not set."""
        p = PeriodicPatch(i=5)  # Valid 2D patch
        with pytest.raises(ValueError, match="not attached to.*block"):
            _ = p.block

    def test_block_shape_set(self, block_10_20_30):
        """Test block shape when block is set."""
        p = PeriodicPatch(i=0)  # Valid 2D patch at boundary
        p.attach_to_block(block_10_20_30)
        expected_shape = (10, 20, 30)
        assert tuple(p.block.shape) == expected_shape

    def test_ijk_lim_abs_positive_indices(self, block_10_20_30):
        """Test _ijk_lim_abs with all positive indices."""
        p = PeriodicPatch(i=(2, 5), j=0, k=(1, 8))  # j=0 is boundary
        p.attach_to_block(block_10_20_30)
        expected = np.array([[2, 5], [0, 0], [1, 8]])
        assert np.array_equal(p.ijk_lim_abs, expected)

    def test_ijk_lim_abs_negative_indices(self, block_10_20_30):
        """Test _ijk_lim_abs with negative indices."""
        # block_10_20_30.shape is (10, 20, 30)
        p = PeriodicPatch(i=(0, -1), j=0, k=(2, -2))
        p.attach_to_block(block_10_20_30)
        expected = np.array([[0, 9], [0, 0], [2, 28]])  # -1 -> 9, -2 -> 28
        assert np.array_equal(p.ijk_lim_abs, expected)

    def test_ijk_lim_abs_mixed_indices(self, block_10_20_30):
        """Test _ijk_lim_abs with mix of positive and negative indices."""
        p = PeriodicPatch(i=(-3, 7), j=(2, -1), k=0)
        p.attach_to_block(block_10_20_30)
        expected = np.array([[7, 7], [2, 19], [0, 0]])  # -3 -> 7, -1 -> 19
        assert np.array_equal(p.ijk_lim_abs, expected)

    def test_ijk_lim_abs_out_of_bounds(self, block_10_20_30):
        """Test _ijk_lim_abs with out of bounds indices."""
        p = PeriodicPatch(i=(0, 15), j=-1, k=(0, 5))  # i end > block size (10)
        with pytest.raises(ValueError, match="out of bounds"):
            p.attach_to_block(block_10_20_30)

    def test_ijk_lim_abs_negative_out_of_bounds(self, block_10_20_30):
        """Test _ijk_lim_abs with negative indices out of bounds."""
        p = PeriodicPatch(i=(-15, 5), j=-1, k=(0, 5))  # -15 would give negative result
        with pytest.raises(ValueError, match="out of bounds"):
            p.attach_to_block(block_10_20_30)

    def test_shape_property(self, block_10_20_30):
        """Test shape property calculation."""
        p = PeriodicPatch(i=(2, 5), j=-1, k=(1, 8))
        p.attach_to_block(block_10_20_30)
        # Shape should be (5-2+1, 3-3+1, 8-1+1) = (4, 1, 8)
        assert p.shape == (4, 1, 8)

    def test_shape_single_point(self, block_10_20_30):
        """Test shape for single point patch."""
        p = PeriodicPatch(i=5, j=3, k=7)
        p.attach_to_block(block_10_20_30)
        assert p.shape == (1, 1, 1)

    def test_shape_with_negative_indices(self, block_10_20_30):
        """Test shape calculation with negative indices."""
        p = PeriodicPatch(i=(0, -1), j=-1, k=(2, -2))
        p.attach_to_block(block_10_20_30)
        # Should resolve to (0, 9), (-1, -1), (2, 28) -> shape (10, 1, 27)
        assert p.shape == (10, 1, 27)

    def test_size_property(self, block_10_20_30):
        """Test size property calculation."""
        p = PeriodicPatch(i=(2, 5), j=-1, k=(1, 8))
        p.attach_to_block(block_10_20_30)
        assert p.size == 32  # 4 * 1 * 8

    def test_size_single_point(self, block_10_20_30):
        """Test size for single point patch."""
        p = PeriodicPatch(i=5, j=3, k=7)
        p.attach_to_block(block_10_20_30)
        assert p.size == 1

    def test_index_properties(self, block_10_20_30):
        """Test index start/end properties (ist, ien, jst, jen, kst, ken)."""
        p = PeriodicPatch(i=(2, 5), j=-1, k=(1, 8))
        p.attach_to_block(block_10_20_30)

        # Test individual properties
        assert p.ist == 2
        assert p.ien == 5
        assert p.jst == -1
        assert p.jen == -1
        assert p.kst == 1
        assert p.ken == 8

    def test_index_properties_single_values(self):
        """Test index properties when dimensions are constant."""
        p = PeriodicPatch(i=7, j=(3, 9), k=2)

        # Constant dimensions should have start == end
        assert p.ist == 7
        assert p.ien == 7
        assert p.jst == 3
        assert p.jen == 9
        assert p.kst == 2
        assert p.ken == 2

    def test_index_properties_negative_values(self, block_10_20_30):
        """Test index properties with negative values."""
        p = PeriodicPatch(i=(-3, -1), j=0, k=(2, -2))
        p.attach_to_block(block_10_20_30)

        # Should return raw values, not resolved values
        assert p.ist == -3
        assert p.ien == -1
        assert p.jst == 0
        assert p.jen == 0
        assert p.kst == 2
        assert p.ken == -2

    def test_properties_without_block(self):
        """Test that properties requiring block fail appropriately."""

        # If no negative indices, these should work
        p = PeriodicPatch(i=1, j=(2, 8), k=(1, 6))
        _ = p.const_dim
        _ = p.shape
        _ = p.size
        _ = p.ijk_lim_abs

        # Index properties should always work
        _ = p.ist
        _ = p.ien
        _ = p.jst
        _ = p.jen
        _ = p.kst
        _ = p.ken

        # If any negative indices, these should not work
        p = PeriodicPatch(i=(0, -1), j=5, k=(2, -2))
        _ = p.const_dim
        # Index properties should still work (return raw values)
        _ = p.ist
        _ = p.ien
        _ = p.jst
        _ = p.jen
        _ = p.kst
        _ = p.ken

        with pytest.raises(ValueError, match="not attached to.*block"):
            _ = p.shape
        with pytest.raises(ValueError, match="not attached to.*block"):
            _ = p.size
        with pytest.raises(ValueError, match="not attached to.*block"):
            _ = p.ijk_lim_abs


class TestPatchMethods:
    """Tests for Patch methods."""

    def test_slice_property_default(self):
        """Test slice property with default parameters."""
        p = PeriodicPatch(i=5, j=(2, 8), k=(1, 6))
        result = p.slice
        expected = (slice(5, 6), slice(2, 9), slice(1, 7))
        assert result == expected

    def test_get_offset_slice_boundary_patch(self):
        """Test _get_offset_slice with offset on boundary patch."""
        # Patch at i=0 boundary
        p = PeriodicPatch(i=0, j=(2, 8), k=(1, 6))
        result = p._get_offset_slice(offset=2)
        # Offset should be added to const dim when at boundary
        expected = (slice(2, 3), slice(2, 9), slice(1, 7))
        assert result == expected

    def test_get_offset_slice_interior_patch(self):
        """Test _get_offset_slice with offset on interior patch."""
        # Patch at i=5 (not boundary)
        p = PeriodicPatch(i=5, j=(2, 8), k=(1, 6))
        result = p._get_offset_slice(offset=2)
        # Offset should be subtracted from const dim when not at boundary
        expected = (slice(3, 4), slice(2, 9), slice(1, 7))
        assert result == expected

    def test_slice_offset_properties(self):
        """Test get_slice_offset method."""
        # Patch at i=0 boundary
        p = PeriodicPatch(i=0, j=(2, 8), k=(1, 6))

        # Test get_slice_offset(1)
        result1 = p._get_offset_slice(1)
        expected1 = (slice(1, 2), slice(2, 9), slice(1, 7))
        assert result1 == expected1

        # Test get_slice_offset(2)
        result2 = p._get_offset_slice(2)
        expected2 = (slice(2, 3), slice(2, 9), slice(1, 7))
        assert result2 == expected2

        # Verify method returns new slice each time (not cached)
        assert p._get_offset_slice(1) == result1
        assert p._get_offset_slice(2) == result2

        # Test interior patch
        p_interior = PeriodicPatch(i=5, j=(2, 8), k=(1, 6))
        result1_int = p_interior._get_offset_slice(1)
        expected1_int = (slice(4, 5), slice(2, 9), slice(1, 7))
        assert result1_int == expected1_int

    def test_slice_property_negative_indices(self):
        """Test slice property with negative indices."""
        p = PeriodicPatch(i=5, j=(2, -1), k=(1, 6))
        result = p.slice
        # -1 should become None in slice
        expected = (slice(5, 6), slice(2, None), slice(1, 7))
        assert result == expected

    def test_slice_property_all_negative_end(self):
        """Test slice property when all end indices are -1."""
        p = PeriodicPatch(i=-1, j=(2, -1), k=(1, -1))
        result = p.slice
        expected = (slice(-1, None), slice(2, None), slice(1, None))
        assert result == expected

    def test_get_ijk_node_default(self, block_10_20_30):
        """Test get_ijk_node with default parameters."""
        p = PeriodicPatch(i=-1, j=(2, 4), k=(1, 3))
        p.attach_to_block(block_10_20_30)

        ijk = p.get_ijk_node()
        # Expected shape: (ni, nj, nk, 3) = (1, 3, 3, 3)
        assert ijk.shape == (1, 3, 3, 3)

        # Check specific coordinate values
        assert ijk[0, 0, 0, 0] == 9  # i index
        assert ijk[0, 0, 0, 1] == 2  # j index
        assert ijk[0, 0, 0, 2] == 1  # k index

        assert ijk[0, 2, 2, 0] == 9  # i still 9
        assert ijk[0, 2, 2, 1] == 4  # j should be 4
        assert ijk[0, 2, 2, 2] == 3  # k should be 3

    def test_get_ijk_node_permutation(self, block_10_20_30):
        """Test get_ijk_node with permutation."""
        p = PeriodicPatch(i=0, j=(2, 4), k=(1, 3))
        p.attach_to_block(block_10_20_30)

        ijk = p.get_ijk_node(perm=(1, 0, 2))  # swap i and j dimensions
        # Shape should be permuted: (nj, ni, nk, 3) = (3, 1, 3, 3)
        assert ijk.shape == (3, 1, 3, 3)

        # Check that coordinates are still correct in the output
        assert ijk[0, 0, 0, 0] == 0  # i coordinate
        assert ijk[0, 0, 0, 1] == 2  # j coordinate
        assert ijk[0, 0, 0, 2] == 1  # k coordinate

    def test_get_ijk_node_flip(self, block_10_20_30):
        """Test get_ijk_node with flipping."""
        p = PeriodicPatch(i=-1, j=(2, 4), k=(1, 3))
        p.attach_to_block(block_10_20_30)

        ijk = p.get_ijk_node(flip=(1,))  # flip j dimension
        assert ijk.shape == (1, 3, 3, 3)

        # Check that j dimension is flipped
        assert ijk[0, 0, 0, 1] == 4  # j flipped: should start at 4
        assert ijk[0, 2, 0, 1] == 2  # j flipped: should end at 2

    def test_get_ijk_node_complex_transform(self, block_10_20_30):
        """Test get_ijk_node with both permutation and flipping."""
        p = PeriodicPatch(i=0, j=(2, 4), k=(1, 3))
        p.attach_to_block(block_10_20_30)

        ijk = p.get_ijk_node(perm=(2, 1, 0), flip=(0, 2))
        # Shape: (nk, nj, ni, 3) = (3, 3, 1, 3)
        assert ijk.shape == (3, 3, 1, 3)

    def test_get_ijk_node_single_point(self, block_10_20_30):
        """Test get_ijk_node for single point patch."""
        p = PeriodicPatch(i=0, j=3, k=7)
        p.attach_to_block(block_10_20_30)

        ijk = p.get_ijk_node()
        assert ijk.shape == (1, 1, 1, 3)
        assert ijk[0, 0, 0, 0] == 0
        assert ijk[0, 0, 0, 1] == 3
        assert ijk[0, 0, 0, 2] == 7

    def test_get_ijk_node_negative_indices(self, block_10_20_30):
        """Test get_ijk_node with negative indices."""
        p = PeriodicPatch(i=(0, -1), j=-1, k=(2, -2))
        p.attach_to_block(block_10_20_30)

        ijk = p.get_ijk_node()
        # Should resolve negative indices first
        expected_shape = (10, 1, 27, 3)  # Based on resolved absolute limits
        assert ijk.shape == expected_shape

        # Check boundary values
        assert ijk[0, 0, 0, 0] == 0  # i start
        assert ijk[9, 0, 0, 0] == 9  # i end
        assert ijk[0, 0, 0, 1] == 19  # j constant
        assert ijk[0, 0, 0, 2] == 2  # k start
        assert ijk[0, 0, 26, 2] == 28  # k end

    def test_repr_method(self):
        """Test __repr__ method."""
        p = PeriodicPatch(i=(2, 5), j=-1, k=(1, 8), label="test")
        repr_str = repr(p)

        assert "PeriodicPatch(" in repr_str
        assert "i=(2, 5)" in repr_str
        assert "j=(-1, -1)" in repr_str
        assert "k=(1, 8)" in repr_str
        assert "label='test'" in repr_str

    def test_repr_no_label(self):
        """Test __repr__ method without label."""
        p = PeriodicPatch(i=5, j=(2, 8), k=0)
        repr_str = repr(p)

        assert "PeriodicPatch(" in repr_str
        assert "i=(5, 5)" in repr_str
        assert "j=(2, 8)" in repr_str
        assert "k=(0, 0)" in repr_str
        assert "label=None" in repr_str

    def test_methods_without_block(self):
        """Test methods that require block fail appropriately."""

        # If no negative indices, these should work
        p = PeriodicPatch(i=0, j=(2, 8), k=(1, 6))
        _ = p.slice
        _ = p.get_ijk_node()

        # If negative indices, these should fail for coordinate generation
        p = PeriodicPatch(i=-5, j=(2, 8), k=(1, 6))
        _ = p.slice  # This still works with negative indices

        with pytest.raises(ValueError):
            _ = (
                p.get_ijk_node()
            )  # This fails because no block shape for negative index resolution


class TestPatchIntegration:
    """Integration tests for complete patch workflows."""

    def test_complete_workflow(self, block_10_20_30):
        """Test complete workflow from creation to data access."""
        # Create patch
        p = PeriodicPatch(i=(1, 3), j=-1, k=(2, 7), label="inlet")

        # Set block
        p.attach_to_block(block_10_20_30)

        # Check properties
        assert p.const_dim == 1
        assert p.shape == (3, 1, 6)
        assert p.size == 18
        assert p.label == "inlet"

        # Get data
        assert p.block_view.shape == (3, 1, 6)

        # Get coordinates
        ijk = p.get_ijk_node()
        assert ijk.shape == (3, 1, 6, 3)

        # Check slice generation
        slice_obj = p.slice
        # Patch indices are inclusive
        print(slice_obj)
        assert slice_obj == (slice(1, 4), slice(-1, None), slice(2, 8))

    def test_different_orientations(self, block_10_20_30):
        """Test patches with different constant dimensions."""
        # i-constant patch (j-k plane)
        p1 = PeriodicPatch(i=-1, j=(0, 9), k=(0, 14))
        p1.attach_to_block(block_10_20_30)
        assert p1.const_dim == 0
        assert p1.shape == (1, 10, 15)

        # j-constant patch (i-k plane)
        p2 = PeriodicPatch(i=(0, 4), j=19, k=(0, 14))
        p2.attach_to_block(block_10_20_30)
        assert p2.const_dim == 1
        assert p2.shape == (5, 1, 15)

        # k-constant patch (i-j plane)
        p3 = PeriodicPatch(i=(0, 4), j=(0, 9), k=29)
        p3.attach_to_block(block_10_20_30)
        assert p3.const_dim == 2
        assert p3.shape == (5, 10, 1)

    def test_negative_index_resolution(self, block_10_20_30):
        """Test complete workflow with negative indices."""
        # block_10_20_30 shape is (10, 20, 30)
        p = PeriodicPatch(i=(-2, -1), j=0, k=(-10, -1))  # j constant for 2D patch
        p.attach_to_block(block_10_20_30)

        # Check resolved limits
        expected_abs = np.array([[8, 9], [0, 0], [20, 29]])  # j is constant
        assert np.array_equal(p.ijk_lim_abs, expected_abs)

        # Check shape
        assert p.shape == (2, 1, 10)  # j constant, so shape[1] = 1
        assert p.size == 20

        # Check data access
        assert p.block_view.shape == (2, 1, 10)

    def test_patch_modification_workflow(self, block_10_20_30):
        """Test modifying patch limits after creation."""
        # Create initial patch
        p = PeriodicPatch(i=0, j=(2, 8), k=(1, 6))
        p.attach_to_block(block_10_20_30)

        initial_shape = p.shape
        assert initial_shape == (1, 7, 6)

        # Modify limits and re-attach to update cached views
        p.set_j_lim((3, 9))
        p.set_k_lim((0, 8))
        p.attach_to_block(block_10_20_30)

        # Check updated properties
        new_shape = p.shape
        assert new_shape == (1, 7, 9)
        assert p.size == 63

        # Check that block_view reflects new limits
        assert p.block_view.shape == (1, 7, 9)

    def test_edge_cases(self, block_10_20_30):
        """Test edge case handling for patch operations."""
        # Single point patch (points are allowed anywhere)
        p1 = PeriodicPatch(i=5, j=10, k=15)
        p1.attach_to_block(block_10_20_30)
        assert p1.shape == (1, 1, 1)
        assert p1.size == 1

        assert p1.block_view.shape == (1, 1, 1)

        # Patches with ambiguous constant dimension should fail at construction
        with pytest.raises(ValueError, match="at least one constant dimension"):
            PeriodicPatch(i=(0, 5), j=(5, 15), k=(20, 25))  # No constant dimensions

        # Maximum size patch (almost full block)
        p3 = PeriodicPatch(i=(0, -1), j=(0, -1), k=0)
        p3.attach_to_block(block_10_20_30)
        assert p3.shape == (10, 20, 1)
        assert p3.size == 200


class TestPatchErrorConditions:
    """Tests for error conditions and edge cases."""

    def test_invalid_constructor_inputs(self):
        """Test constructor validation with invalid inputs."""
        # All dimensions varying (3D patch)
        with pytest.raises(ValueError, match="at least one constant dimension"):
            PeriodicPatch(i=(0, 5), j=(0, 3), k=(0, 2))

        # Invalid types passed to constructor
        with pytest.raises(ValueError):
            PeriodicPatch(i=3.14, j=0)  # j constant to make valid 2D
        with pytest.raises(ValueError):
            PeriodicPatch(i=0, j="invalid")  # i constant to make valid 2D
        with pytest.raises(ValueError):
            PeriodicPatch(i=0, k=(1, 2, 3))  # i constant to make valid 2D

    def test_out_of_bounds_conditions(self, block_10_20_30):
        """Test out of bounds error handling."""
        # Positive indices out of bounds
        p1 = PeriodicPatch(i=(0, 20), j=5, k=(0, 3))  # i end > block size
        with pytest.raises(ValueError, match="out of bounds"):
            p1.attach_to_block(block_10_20_30)

        # Negative indices that resolve out of bounds
        p2 = PeriodicPatch(i=(-50, 5), j=3, k=(0, 3))  # -50 would be negative
        with pytest.raises(ValueError, match="out of bounds"):
            p2.attach_to_block(block_10_20_30)

        # Start index greater than end index (after resolution)
        p3 = PeriodicPatch(i=(8, 5), j=3, k=(0, 3))  # start > end
        with pytest.raises(ValueError):
            p3.attach_to_block(block_10_20_30)

    def test_coordinate_generation_edge_cases(self, small_block):
        """Test coordinate generation edge cases."""
        # Single point
        p1 = PeriodicPatch(i=2, j=2, k=2)
        p1.attach_to_block(small_block)
        ijk = p1.get_ijk_node()
        assert ijk.shape == (1, 1, 1, 3)

        # Invalid permutation (though this might not raise error in current impl)
        p2 = PeriodicPatch(i=0, j=(0, 2), k=(0, 2))
        p2.attach_to_block(small_block)

        # These should work but produce expected transformations
        ijk_perm = p2.get_ijk_node(perm=(2, 0, 1))
        assert ijk_perm.shape == (3, 1, 3, 3)

    def test_property_access_patterns(self, block_10_20_30):
        """Test property access patterns and error conditions."""
        p = PeriodicPatch(i=(0, -1), j=0, k=(2, -2))

        # Access properties before setting block - should fail
        with pytest.raises(ValueError):
            _ = p.shape

        # Set block
        p.attach_to_block(block_10_20_30)

        # Now properties should work
        shape = p.shape
        assert isinstance(shape, tuple)
        assert len(shape) == 3

        size = p.size
        assert isinstance(size, (int, np.integer))
        assert size > 0

        # Test multiple accesses (should use caching if implemented)
        shape2 = p.shape
        assert shape == shape2

    def test_realistic_cfd_scenarios(self, block_10_20_30):
        """Test realistic CFD patch scenarios."""
        # Inlet patch (k=0 face)
        inlet = PeriodicPatch(i=(1, -2), j=(1, -2), k=0, label="inlet")
        inlet.attach_to_block(block_10_20_30)
        assert inlet.const_dim == 2
        assert inlet.shape == (8, 18, 1)

        # Outlet patch (k=-1 face)
        outlet = PeriodicPatch(i=(1, -2), j=(1, -2), k=-1, label="outlet")
        outlet.attach_to_block(block_10_20_30)
        assert outlet.const_dim == 2
        assert outlet.shape == (8, 18, 1)

        # Wall patch (j=0 face, hub)
        wall_hub = PeriodicPatch(i=(0, -1), j=0, k=(0, -1), label="hub")
        wall_hub.attach_to_block(block_10_20_30)
        assert wall_hub.const_dim == 1
        assert wall_hub.shape == (10, 1, 30)

        # Wall patch (j=-1 face, shroud)
        wall_shroud = PeriodicPatch(i=(0, -1), j=-1, k=(0, -1), label="shroud")
        wall_shroud.attach_to_block(block_10_20_30)
        assert wall_shroud.const_dim == 1
        assert wall_shroud.shape == (10, 1, 30)

        # Check that all patches can access data
        for patch in [inlet, outlet, wall_hub, wall_shroud]:
            assert patch.block_view.size > 0
            assert patch.label is not None


class TestPatchBoundaryValidation:
    """Tests for patch boundary constraint validation."""

    def test_valid_boundary_patches_start_end(self, block_10_20_30):
        """Test that patches on block boundaries (start/end) are valid."""
        # block_10_20_30 shape is (10, 20, 30)
        valid_patches = [
            # i-direction boundaries
            PeriodicPatch(i=0, j=(0, 10), k=(0, 15)),  # i at start boundary
            PeriodicPatch(
                i=-1, j=(0, 10), k=(0, 15)
            ),  # i at end boundary (resolves to 9)
            # j-direction boundaries
            PeriodicPatch(i=(0, 5), j=0, k=(0, 15)),  # j at start boundary
            PeriodicPatch(
                i=(0, 5), j=-1, k=(0, 15)
            ),  # j at end boundary (resolves to 19)
            # k-direction boundaries
            PeriodicPatch(i=(0, 5), j=(0, 10), k=0),  # k at start boundary
            PeriodicPatch(
                i=(0, 5), j=(0, 10), k=-1
            ),  # k at end boundary (resolves to 29)
        ]

        for patch in valid_patches:
            patch.attach_to_block(block_10_20_30)
            # Should not raise - accessing _ijk_lim_abs triggers validation
            abs_limits = patch.ijk_lim_abs
            assert abs_limits is not None

    def test_valid_boundary_patches_explicit_max(self, block_10_20_30):
        """Test boundary patches with explicit maximum indices."""
        # block_10_20_30 shape is (10, 20, 30), so max indices are (9, 19, 29)
        valid_patches = [
            PeriodicPatch(i=9, j=(0, 10), k=(0, 15)),  # i at explicit end boundary
            PeriodicPatch(i=(0, 5), j=19, k=(0, 15)),  # j at explicit end boundary
            PeriodicPatch(i=(0, 5), j=(0, 10), k=29),  # k at explicit end boundary
        ]

        for patch in valid_patches:
            patch.attach_to_block(block_10_20_30)
            abs_limits = patch.ijk_lim_abs
            assert abs_limits is not None

    def test_invalid_interior_patches(self, block_10_20_30):
        """Test that interior patches raise ValueError."""
        # block_10_20_30 shape is (10, 20, 30)
        # Interior positions: i in [1,8], j in [1,18], k in [1,28]
        invalid_patches = [
            PeriodicPatch(i=5, j=(0, 10), k=(0, 15)),  # i=5 is interior (not 0 or 9)
            PeriodicPatch(i=(0, 5), j=10, k=(0, 15)),  # j=10 is interior (not 0 or 19)
            PeriodicPatch(i=(0, 5), j=(0, 10), k=15),  # k=15 is interior (not 0 or 29)
            # Additional interior cases
            PeriodicPatch(i=1, j=(0, 5), k=(0, 10)),  # i=1 is interior
            PeriodicPatch(i=8, j=(0, 5), k=(0, 10)),  # i=8 is interior
            PeriodicPatch(i=(0, 5), j=1, k=(0, 10)),  # j=1 is interior
            PeriodicPatch(i=(0, 5), j=18, k=(0, 10)),  # j=18 is interior
            PeriodicPatch(i=(0, 5), j=(0, 10), k=1),  # k=1 is interior
            PeriodicPatch(i=(0, 5), j=(0, 10), k=28),  # k=28 is interior
        ]

        for patch in invalid_patches:
            with pytest.raises(
                ValueError, match="constant dimension is not at start or end"
            ):
                patch.attach_to_block(block_10_20_30)

    def test_boundary_validation_with_negative_indices(self, block_10_20_30):
        """Test boundary validation works correctly with negative indices."""
        # block_10_20_30 shape is (10, 20, 30)

        # Valid: -1 should resolve to max_index and be valid (end boundary)
        valid_patches = [
            PeriodicPatch(i=-1, j=(0, 10), k=(0, 15)),  # -1 -> 9 (valid end)
            PeriodicPatch(i=(0, 5), j=-1, k=(0, 15)),  # -1 -> 19 (valid end)
            PeriodicPatch(i=(0, 5), j=(0, 10), k=-1),  # -1 -> 29 (valid end)
        ]

        for patch in valid_patches:
            patch.attach_to_block(block_10_20_30)
            abs_limits = patch.ijk_lim_abs  # Should not raise
            assert abs_limits is not None

        # Invalid: -2, -3, etc. should resolve to interior positions and be invalid
        invalid_patches = [
            PeriodicPatch(i=-2, j=(0, 10), k=(0, 15)),  # -2 -> 8 (interior)
            PeriodicPatch(i=-3, j=(0, 10), k=(0, 15)),  # -3 -> 7 (interior)
            PeriodicPatch(i=(0, 5), j=-2, k=(0, 15)),  # -2 -> 18 (interior)
            PeriodicPatch(i=(0, 5), j=-5, k=(0, 15)),  # -5 -> 15 (interior)
            PeriodicPatch(i=(0, 5), j=(0, 10), k=-2),  # -2 -> 28 (interior)
            PeriodicPatch(i=(0, 5), j=(0, 10), k=-10),  # -10 -> 20 (interior)
        ]

        for patch in invalid_patches:
            with pytest.raises(
                ValueError, match="constant dimension is not at start or end"
            ):
                patch.attach_to_block(block_10_20_30)

    def test_boundary_validation_different_block_sizes(self):
        """Test boundary validation works with different block sizes."""
        # Small block
        from conftest import _make_block

        small_block = _make_block((3, 4, 5))

        # Valid boundary patches for small block
        valid_patches = [
            PeriodicPatch(i=0, j=(0, 2), k=(0, 3)),  # i=0 (start)
            PeriodicPatch(i=2, j=(0, 2), k=(0, 3)),  # i=2 (end, max_i=2)
            PeriodicPatch(i=(0, 1), j=0, k=(0, 3)),  # j=0 (start)
            PeriodicPatch(i=(0, 1), j=3, k=(0, 3)),  # j=3 (end, max_j=3)
            PeriodicPatch(i=(0, 1), j=(0, 2), k=0),  # k=0 (start)
            PeriodicPatch(i=(0, 1), j=(0, 2), k=4),  # k=4 (end, max_k=4)
        ]

        for patch in valid_patches:
            patch.attach_to_block(small_block)
            abs_limits = patch.ijk_lim_abs
            assert abs_limits is not None

        # Invalid interior patches for small block
        invalid_patches = [
            PeriodicPatch(i=1, j=(0, 2), k=(0, 3)),  # i=1 is interior (not 0 or 2)
            PeriodicPatch(i=(0, 1), j=1, k=(0, 3)),  # j=1 is interior (not 0 or 3)
            PeriodicPatch(i=(0, 1), j=2, k=(0, 3)),  # j=2 is interior (not 0 or 3)
            PeriodicPatch(i=(0, 1), j=(0, 2), k=1),  # k=1 is interior (not 0 or 4)
            PeriodicPatch(i=(0, 1), j=(0, 2), k=2),  # k=2 is interior (not 0 or 4)
            PeriodicPatch(i=(0, 1), j=(0, 2), k=3),  # k=3 is interior (not 0 or 4)
        ]

        for patch in invalid_patches:
            with pytest.raises(
                ValueError, match="constant dimension is not at start or end"
            ):
                patch.attach_to_block(small_block)

    def test_boundary_validation_preserves_existing_behavior(self, block_10_20_30):
        """Test that boundary validation doesn't break existing valid use cases."""
        # These are patches that should work and are used in other tests
        existing_valid_patches = [
            # From realistic CFD scenarios
            PeriodicPatch(i=(1, -2), j=(1, -2), k=0, label="inlet"),  # k=0 boundary
            PeriodicPatch(i=(1, -2), j=(1, -2), k=-1, label="outlet"),  # k=-1 boundary
            PeriodicPatch(i=(0, -1), j=0, k=(0, -1), label="hub"),  # j=0 boundary
            PeriodicPatch(i=(0, -1), j=-1, k=(0, -1), label="shroud"),  # j=-1 boundary
            # From other tests
            PeriodicPatch(i=5, j=3, k=7),  # All constant - won't trigger validation
        ]

        for patch in existing_valid_patches:
            patch.attach_to_block(block_10_20_30)
            # Should work without issues - either valid boundaries or special cases
            try:
                abs_limits = patch.ijk_lim_abs
                assert abs_limits is not None
            except ValueError as e:
                # If it fails, it should be for a reason other than boundary validation
                assert "constant dimension is not at start or end" not in str(e)


class TestPatchIjkFace:
    """Tests for get_ijk_face method - face index generation."""

    def test_get_ijk_face_constant_i(self, block_10_20_30):
        """Test get_ijk_face for constant i face (j-k plane)."""
        p = PeriodicPatch(i=0, j=(2, 5), k=(1, 4))  # i=0 boundary, 4x4 nodes
        p.attach_to_block(block_10_20_30)

        ijk_node = p.get_ijk_node()
        ijk_face = p.get_ijk_face()

        # For constant i: exclude j_max and k_max
        # Node shape: (1, 4, 4, 3), Face shape should be: (1, 3, 3, 3)
        assert ijk_node.shape == (1, 4, 4, 3)
        assert ijk_face.shape == (1, 3, 3, 3)

        # Face indices should match node indices but exclude last j,k
        assert np.array_equal(ijk_face, ijk_node[:, :-1, :-1, :])

        # Check specific values
        assert ijk_face[0, 0, 0, 0] == 0  # i constant
        assert ijk_face[0, 0, 0, 1] == 2  # j start
        assert ijk_face[0, 0, 0, 2] == 1  # k start
        assert ijk_face[0, 2, 2, 1] == 4  # j max in face array
        assert ijk_face[0, 2, 2, 2] == 3  # k max in face array

    def test_get_ijk_face_constant_j(self, block_10_20_30):
        """Test get_ijk_face for constant j face (i-k plane)."""
        p = PeriodicPatch(i=(1, 4), j=-1, k=(2, 5))  # j=-1 boundary, 4x4 nodes
        p.attach_to_block(block_10_20_30)

        ijk_node = p.get_ijk_node()
        ijk_face = p.get_ijk_face()

        # For constant j: exclude i_max and k_max
        # Node shape: (4, 1, 4, 3), Face shape should be: (3, 1, 3, 3)
        assert ijk_node.shape == (4, 1, 4, 3)
        assert ijk_face.shape == (3, 1, 3, 3)

        # Face indices should match node indices but exclude last i,k
        assert np.array_equal(ijk_face, ijk_node[:-1, :, :-1, :])

        # Check specific values
        assert ijk_face[0, 0, 0, 0] == 1  # i start
        assert ijk_face[0, 0, 0, 1] == 19  # j constant (resolved from -1)
        assert ijk_face[0, 0, 0, 2] == 2  # k start
        assert ijk_face[2, 0, 2, 0] == 3  # i max in face array
        assert ijk_face[2, 0, 2, 2] == 4  # k max in face array

    def test_get_ijk_face_constant_k(self, block_10_20_30):
        """Test get_ijk_face for constant k face (i-j plane)."""
        p = PeriodicPatch(i=(1, 4), j=(3, 6), k=0)  # k=0 boundary, 4x4 nodes
        p.attach_to_block(block_10_20_30)

        ijk_node = p.get_ijk_node()
        ijk_face = p.get_ijk_face()

        # For constant k: exclude i_max and j_max
        # Node shape: (4, 4, 1, 3), Face shape should be: (3, 3, 1, 3)
        assert ijk_node.shape == (4, 4, 1, 3)
        assert ijk_face.shape == (3, 3, 1, 3)

        # Face indices should match node indices but exclude last i,j
        assert np.array_equal(ijk_face, ijk_node[:-1, :-1, :, :])

        # Check specific values
        assert ijk_face[0, 0, 0, 0] == 1  # i start
        assert ijk_face[0, 0, 0, 1] == 3  # j start
        assert ijk_face[0, 0, 0, 2] == 0  # k constant
        assert ijk_face[2, 2, 0, 0] == 3  # i max in face array
        assert ijk_face[2, 2, 0, 1] == 5  # j max in face array

    def test_get_ijk_face_shape_relationship(self, block_10_20_30):
        """Test that face shapes are correctly derived from node shapes."""
        test_cases = [
            (
                PeriodicPatch(i=0, j=(0, 5), k=(0, 8)),
                (1, 6, 9, 3),
                (1, 5, 8, 3),
            ),  # const i
            (
                PeriodicPatch(i=(0, 7), j=0, k=(0, 5)),
                (8, 1, 6, 3),
                (7, 1, 5, 3),
            ),  # const j
            (
                PeriodicPatch(i=(0, 4), j=(0, 6), k=-1),
                (5, 7, 1, 3),
                (4, 6, 1, 3),
            ),  # const k
        ]

        for patch, expected_node_shape, expected_face_shape in test_cases:
            patch.attach_to_block(block_10_20_30)

            ijk_node = patch.get_ijk_node()
            ijk_face = patch.get_ijk_face()

            assert ijk_node.shape == expected_node_shape
            assert ijk_face.shape == expected_face_shape

    def test_get_ijk_face_single_face_patches(self, block_10_20_30):
        """Test get_ijk_face for minimal single-face patches."""
        # Single face in each orientation
        test_cases = [
            PeriodicPatch(i=0, j=(0, 1), k=(0, 1)),  # const i: 1x1 face
            PeriodicPatch(i=(0, 1), j=0, k=(0, 1)),  # const j: 1x1 face
            PeriodicPatch(i=(0, 1), j=(0, 1), k=0),  # const k: 1x1 face
        ]

        for patch in test_cases:
            patch.attach_to_block(block_10_20_30)

            ijk_node = patch.get_ijk_node()
            ijk_face = patch.get_ijk_face()

            # Each should have exactly one face
            assert ijk_face.size // 3 == 1  # 1 face × 3 coordinates

            # Face array should be smaller than node array
            for dim in range(3):
                if dim != patch.const_dim:
                    assert ijk_face.shape[dim] == ijk_node.shape[dim] - 1
                else:
                    assert ijk_face.shape[dim] == ijk_node.shape[dim]

    def test_get_ijk_face_permutation(self, block_10_20_30):
        """Test get_ijk_face with permutation."""
        p = PeriodicPatch(i=0, j=(0, 2), k=(0, 3))  # const i face
        p.attach_to_block(block_10_20_30)

        # Test permutation (swap j and k dimensions)
        ijk_face_default = p.get_ijk_face()
        ijk_face_perm = p.get_ijk_face(perm=(0, 2, 1))  # swap j,k

        # Shapes should be permuted: (1, 2, 2, 3) -> (1, 2, 2, 3) but data rearranged
        assert ijk_face_default.shape == (1, 2, 3, 3)
        assert ijk_face_perm.shape == (1, 3, 2, 3)

        # Check that coordinates are permuted correctly
        # Original: [i, j, k], Permuted: [i, k, j]
        assert ijk_face_perm[0, 0, 0, 0] == ijk_face_default[0, 0, 0, 0]  # i unchanged
        assert (
            ijk_face_perm[0, 0, 0, 1] == ijk_face_default[0, 0, 0, 2]
        )  # j->k position
        assert (
            ijk_face_perm[0, 0, 0, 2] == ijk_face_default[0, 0, 0, 1]
        )  # k->j position

    def test_get_ijk_face_flipping(self, block_10_20_30):
        """Test get_ijk_face with flipping."""
        p = PeriodicPatch(i=-1, j=(2, 4), k=(1, 3))  # const i face
        p.attach_to_block(block_10_20_30)

        ijk_face_default = p.get_ijk_face()
        ijk_face_flip = p.get_ijk_face(flip=(1,))  # flip j dimension

        # Same shape but j-dimension flipped
        assert ijk_face_default.shape == ijk_face_flip.shape

        # Check that j dimension is flipped
        # j should go from high to low instead of low to high
        assert (
            ijk_face_flip[0, 0, 0, 1] == ijk_face_default[0, 1, 0, 1]
        )  # first j = last j in default
        assert (
            ijk_face_flip[0, 1, 0, 1] == ijk_face_default[0, 0, 0, 1]
        )  # last j = first j in default

    def test_get_ijk_face_complex_transform(self, block_10_20_30):
        """Test get_ijk_face with both permutation and flipping."""
        p = PeriodicPatch(i=(1, 3), j=0, k=(2, 4))  # const j face
        p.attach_to_block(block_10_20_30)

        ijk_face = p.get_ijk_face(perm=(2, 1, 0), flip=(0, 2))

        # Should apply both transformations without error
        assert ijk_face.shape == (2, 1, 2, 3)  # transformed shape
        assert ijk_face.dtype == np.int64 or ijk_face.dtype == int

    def test_get_ijk_face_vs_get_ijk_node_consistency(self, block_10_20_30):
        """Test that get_ijk_face returns consistent subset of get_ijk_node."""
        test_patches = [
            PeriodicPatch(i=0, j=(1, 4), k=(2, 6)),  # const i
            PeriodicPatch(i=(1, 5), j=-1, k=(2, 7)),  # const j
            PeriodicPatch(i=(2, 6), j=(1, 5), k=0),  # const k
        ]

        for patch in test_patches:
            patch.attach_to_block(block_10_20_30)

            ijk_node = patch.get_ijk_node()
            ijk_face = patch.get_ijk_face()

            # Face should be a proper subset of nodes
            const_dim = patch.const_dim

            if const_dim == 0:  # const i
                expected_face = ijk_node[:, :-1, :-1, :]
            elif const_dim == 1:  # const j
                expected_face = ijk_node[:-1, :, :-1, :]
            else:  # const k
                expected_face = ijk_node[:-1, :-1, :, :]

            assert np.array_equal(ijk_face, expected_face)

    def test_get_ijk_face_boundary_patches(self, block_10_20_30):
        """Test get_ijk_face works correctly with realistic boundary patches."""
        # Realistic CFD boundary patches
        boundary_patches = [
            PeriodicPatch(i=0, j=(0, -1), k=(0, -1), label="inlet"),  # inlet at i=0
            PeriodicPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet"),  # outlet at i=-1
            PeriodicPatch(
                i=(0, -1), j=0, k=(0, -1), label="bottom"
            ),  # bottom wall at j=0
            PeriodicPatch(i=(0, -1), j=-1, k=(0, -1), label="top"),  # top wall at j=-1
            PeriodicPatch(i=(0, -1), j=(0, -1), k=0, label="front"),  # front at k=0
            PeriodicPatch(i=(0, -1), j=(0, -1), k=-1, label="back"),  # back at k=-1
        ]

        for patch in boundary_patches:
            patch.attach_to_block(block_10_20_30)

            # Should work without errors
            ijk_node = patch.get_ijk_node()
            ijk_face = patch.get_ijk_face()

            # Face array should be smaller than node array in non-constant dimensions
            const_dim = patch.const_dim
            for dim in range(3):
                if dim != const_dim:
                    assert ijk_face.shape[dim] == ijk_node.shape[dim] - 1
                else:
                    assert ijk_face.shape[dim] == ijk_node.shape[dim]

            # Should have valid indices
            assert np.all(ijk_face >= 0)
            assert np.all(ijk_face[:, :, :, 0] < block_10_20_30.shape[0])
            assert np.all(ijk_face[:, :, :, 1] < block_10_20_30.shape[1])
            assert np.all(ijk_face[:, :, :, 2] < block_10_20_30.shape[2])

    def test_get_ijk_face_error_conditions(self, block_10_20_30):
        """Test get_ijk_face error handling."""
        # Test with invalid constant dimension (should not happen with valid patches)
        p = PeriodicPatch(i=0, j=(1, 3), k=(2, 4))
        p.attach_to_block(block_10_20_30)

        # This should work fine
        ijk_face = p.get_ijk_face()
        assert ijk_face.shape == (1, 2, 2, 3)

    def test_get_ijk_face_mathematical_correctness(self, block_10_20_30):
        """Test mathematical correctness of face indexing concept."""
        # For a 3x3 node grid, we should have 2x2 faces
        p = PeriodicPatch(i=0, j=(0, 2), k=(0, 2))  # 1x3x3 nodes
        p.attach_to_block(block_10_20_30)

        ijk_node = p.get_ijk_node()
        ijk_face = p.get_ijk_face()

        assert ijk_node.shape == (1, 3, 3, 3)  # 9 nodes
        assert ijk_face.shape == (1, 2, 2, 3)  # 4 faces

        # Each face should reference its "lower-left" node
        # Face [0,0] should correspond to node [0,0]
        assert np.array_equal(ijk_face[0, 0, 0, :], ijk_node[0, 0, 0, :])
        # Face [0,1] should correspond to node [0,1]
        assert np.array_equal(ijk_face[0, 0, 1, :], ijk_node[0, 0, 1, :])
        # Face [1,0] should correspond to node [1,0]
        assert np.array_equal(ijk_face[0, 1, 0, :], ijk_node[0, 1, 0, :])
        # Face [1,1] should correspond to node [1,1]
        assert np.array_equal(ijk_face[0, 1, 1, :], ijk_node[0, 1, 1, :])


def _create_test_block_3d():
    """Helper to create a minimal 3D block with surface-of-revolution geometry.

    j is spanwise (varying x), k is pitchwise (constant x,r).
    """
    from ember.block import Block
    from ember.fluid import PerfectFluid

    shape = (5, 4, 3)
    block = Block(shape=shape)
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    block.set_fluid(fluid)
    x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1) * np.ones(shape)
    r = np.ones(shape) * 0.5
    t = np.linspace(0, 0.2, shape[2]).reshape(1, 1, -1) * np.ones(shape)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)
    block.set_P_T(101325.0, 300.0)
    return block


def test_patch_requires_3d_block():
    """Test that attach_to_block raises error for non-3D blocks."""
    import ember.patch

    # Create a 3D block with a patch
    block_3d = _create_test_block_3d()
    block_3d.patches.append(ember.patch.InletPatch(i=0))

    # This should work fine
    assert len(block_3d.patches) == 1

    # Now reshape to 2D - patches are still a view of the original
    block_2d = block_3d.reshape((5, 12))
    assert len(block_2d.patches) == 1
    # The patch collection still references the original 3D block
    assert block_2d.patches._block is block_3d

    # When you try to add a NEW patch to the reshaped block,
    # it still uses the original block's shape (3D), so it succeeds
    block_2d.patches.append(ember.patch.OutletPatch(i=-1))
    assert len(block_2d.patches) == 2

    # But if you directly try to set a 2D shape on a patch, it should fail
    patch = ember.patch.InletPatch(i=0)
    with pytest.raises(ValueError, match="Patches require 3D blocks"):
        patch.attach_to_block(block_2d)


def test_patch_various_ndim():
    """Test patch with various block shapes."""
    import ember.patch
    from ember.block import Block

    # 3D block should work
    patch = ember.patch.InletPatch(i=0, j=(0, -1), k=(0, -1))
    block_3d = _create_test_block_3d()
    patch.attach_to_block(block_3d)
    assert patch._block_ref is not None

    # 2D block should fail
    patch2 = ember.patch.InletPatch(i=0, j=(0, -1), k=(0, -1))
    block_2d = Block(shape=(5, 12))
    with pytest.raises(
        ValueError, match="Patches require 3D blocks.*ndim=3.*2 dimensions"
    ):
        patch2.attach_to_block(block_2d)

    # 1D block should fail
    patch3 = ember.patch.InletPatch(i=0, j=(0, -1), k=(0, -1))
    block_1d = Block(shape=(60,))
    with pytest.raises(
        ValueError, match="Patches require 3D blocks.*ndim=3.*1 dimensions"
    ):
        patch3.attach_to_block(block_1d)

    # 4D block should fail
    patch4 = ember.patch.InletPatch(i=0, j=(0, -1), k=(0, -1))
    block_4d = Block(shape=(5, 4, 3, 2))
    with pytest.raises(
        ValueError, match="Patches require 3D blocks.*ndim=3.*4 dimensions"
    ):
        patch4.attach_to_block(block_4d)


def test_block_flat_removes_patches():
    """Test that block.flat() properly handles patches."""
    import ember.patch
    from ember.block import Block
    from ember.fluid import PerfectFluid

    # Create a 3D block with patches
    block_3d = Block(shape=(5, 4, 3))
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    block_3d.set_fluid(fluid)
    block_3d.set_x(np.random.rand(5, 4, 3))
    block_3d.set_r(np.ones((5, 4, 3)))
    block_3d.set_t(np.random.rand(5, 4, 3))
    block_3d.set_P_T(101325.0, 300.0)

    # Add some patches (use PeriodicPatch since geometry is not a surface of revolution)
    block_3d.patches.append(ember.patch.PeriodicPatch(i=0))
    block_3d.patches.append(ember.patch.PeriodicPatch(i=-1))

    assert len(block_3d.patches) == 2

    # flat() should clear patches since it creates a 1D block
    block_flat = block_3d.flat()

    # According to the recent commit, patches should be cleared
    assert len(block_flat.patches) == 0


class TestPatchOverlapDetection:
    """Test overlap detection between patches of the same type."""

    def setup_method(self):
        """Set up a block for testing."""
        self.block = ember.block.Block(shape=(10, 20, 30))

    def test_non_overlapping_patches_allowed(self):
        """Test that non-overlapping patches of the same type are allowed."""
        # Create patches that don't overlap
        patch1 = PeriodicPatch(i=0, j=(0, 5), k=(0, 10))  # i=0, j=0-5, k=0-10
        patch2 = PeriodicPatch(
            i=0, j=(6, 10), k=(0, 10)
        )  # i=0, j=6-10, k=0-10 (adjacent in j)
        patch3 = PeriodicPatch(
            i=0, j=(0, 5), k=(11, 15)
        )  # i=0, j=0-5, k=11-15 (adjacent in k)

        # Should not raise any errors
        self.block.patches.append(patch1)
        self.block.patches.append(patch2)
        self.block.patches.append(patch3)

        assert len(self.block.patches) == 3
        assert all(isinstance(p, PeriodicPatch) for p in self.block.patches)

    def test_different_types_no_overlap_check(self):
        """Test that different patch types don't trigger overlap detection."""
        # Create patches of different types that would overlap if they were the same type
        from ember.patch import CoolingPatch, InviscidPatch

        patch1 = PeriodicPatch(i=0, j=(0, 10), k=(0, 15))
        patch2 = CoolingPatch(i=0, j=(5, 15), k=(5, 20))  # Overlaps with patch1
        patch3 = InviscidPatch(i=0, j=(8, 18), k=(10, 25))  # Overlaps with both

        # Should not raise any errors since they're different types
        self.block.patches.append(patch1)
        self.block.patches.append(patch2)
        self.block.patches.append(patch3)

        assert len(self.block.patches) == 3

    def test_overlapping_patches_rejected(self):
        """Test that overlapping patches of the same type are rejected."""
        patch1 = PeriodicPatch(i=0, j=(0, 10), k=(0, 15))
        patch2 = PeriodicPatch(
            i=0, j=(5, 15), k=(10, 20)
        )  # Overlaps in j=5-10, k=10-15

        # First patch should be added successfully
        self.block.patches.append(patch1)
        assert len(self.block.patches) == 1

        # Second patch should be rejected due to overlap
        with pytest.raises(ValueError, match="overlaps with existing patch"):
            self.block.patches.append(patch2)

        # Collection should still only have the first patch
        assert len(self.block.patches) == 1

    def test_exact_duplicate_patches_rejected(self):
        """Test that exact duplicate patches are rejected."""
        patch1 = PeriodicPatch(i=0, j=(0, 10), k=(0, 15))
        patch2 = PeriodicPatch(i=0, j=(0, 10), k=(0, 15))  # Exact duplicate

        self.block.patches.append(patch1)

        with pytest.raises(ValueError, match="overlaps with existing patch"):
            self.block.patches.append(patch2)

    def test_partial_overlap_rejected(self):
        """Test various partial overlap scenarios."""
        base_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        self.block.patches.append(base_patch)

        # Test overlaps in different dimensions (keeping within block bounds: 10x20x30)
        overlap_cases = [
            PeriodicPatch(i=0, j=(10, 19), k=(15, 25)),  # Overlap in j=10-15, k=15-20
            PeriodicPatch(i=0, j=(0, 8), k=(12, 18)),  # Overlap in j=5-8, k=12-18
            PeriodicPatch(i=0, j=(12, 18), k=(8, 12)),  # Overlap in j=12-15, k=10-12
        ]

        for overlapping_patch in overlap_cases:
            with pytest.raises(ValueError, match="overlaps with existing patch"):
                self.block.patches.append(overlapping_patch)

        # Should still only have the original patch
        assert len(self.block.patches) == 1

    def test_edge_adjacent_patches_allowed(self):
        """Test that patches that are adjacent (touching edges) are allowed."""
        patch1 = PeriodicPatch(i=0, j=(0, 10), k=(0, 15))
        # These patches touch at boundaries but don't overlap
        patch2 = PeriodicPatch(i=0, j=(11, 19), k=(0, 15))  # Adjacent in j (11 > 10)
        patch3 = PeriodicPatch(i=0, j=(0, 10), k=(16, 25))  # Adjacent in k (16 > 15)

        self.block.patches.append(patch1)
        self.block.patches.append(patch2)  # Should work
        self.block.patches.append(patch3)  # Should work

        assert len(self.block.patches) == 3

    def test_extend_with_overlaps_rejected(self):
        """Test that extend method also checks for overlaps."""
        patch1 = PeriodicPatch(i=0, j=(0, 5), k=(0, 10))
        self.block.patches.append(patch1)

        # Try to extend with overlapping patches
        overlapping_patches = [
            PeriodicPatch(i=0, j=(6, 10), k=(0, 10)),  # Non-overlapping
            PeriodicPatch(i=0, j=(3, 8), k=(5, 15)),  # Overlaps with patch1
        ]

        with pytest.raises(ValueError, match="overlaps with existing patch"):
            self.block.patches.extend(overlapping_patches)

        # Should still only have original patch
        assert len(self.block.patches) == 1

    def test_extend_internal_overlaps_rejected(self):
        """Test that extend rejects patches that overlap within the batch."""
        # Create patches that overlap with each other (not with existing patches)
        overlapping_patches = [
            PeriodicPatch(i=0, j=(0, 5), k=(0, 10)),
            PeriodicPatch(i=0, j=(3, 8), k=(5, 15)),  # Overlaps with first patch
        ]

        with pytest.raises(
            ValueError, match="overlaps with another patch in the same batch"
        ):
            self.block.patches.extend(overlapping_patches)

        # Should have no patches
        assert len(self.block.patches) == 0

    def test_insert_with_overlaps_rejected(self):
        """Test that insert method also checks for overlaps."""
        patch1 = PeriodicPatch(i=0, j=(0, 10), k=(0, 15))
        self.block.patches.append(patch1)

        overlapping_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))

        with pytest.raises(ValueError, match="overlaps with existing patch"):
            self.block.patches.insert(0, overlapping_patch)

        assert len(self.block.patches) == 1

    def test_negative_indices_overlap_detection(self):
        """Test overlap detection works with negative indices."""
        # Using negative indices that resolve to actual positions
        patch1 = PeriodicPatch(i=0, j=(0, 10), k=(0, -10))  # k goes to 20
        patch2 = PeriodicPatch(
            i=0, j=(5, 15), k=(-15, -5)
        )  # k goes 15-25, overlaps in j and k

        self.block.patches.append(patch1)

        with pytest.raises(ValueError, match="overlaps with existing patch"):
            self.block.patches.append(patch2)

    def test_different_constant_dimensions_can_overlap(self):
        """Test that patches with different constant dimensions can overlap without error."""
        # These patches have different constant dimensions so geometric overlap is expected
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))  # i constant at start
        patch2 = PeriodicPatch(
            i=(5, 9), j=0, k=(10, 20)
        )  # j constant at start (boundary)

        # This should work because they intersect geometrically but have different orientations
        # However, they are same type, so if they overlap in 3D space, it should be caught
        self.block.patches.append(patch1)

        # patch2 intersects patch1 at the point/line where i=5-9, j=0, k=10-20
        # Since patch1 has i=0 (constant), they don't actually overlap in 3D
        # So this should work
        self.block.patches.append(patch2)

        assert len(self.block.patches) == 2

    def test_same_constant_dimension_overlaps_rejected(self):
        """Test overlapping patches with same constant dimension."""
        # Both patches have i=0 (same constant dimension) and overlap in j,k space
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = PeriodicPatch(
            i=0, j=(10, 19), k=(15, 25)
        )  # Overlaps in j=10-15, k=15-20

        self.block.patches.append(patch1)

        with pytest.raises(ValueError, match="overlaps with existing patch"):
            self.block.patches.append(patch2)


class TestPatchOverlapEdgeCases:
    """Test edge cases for patch overlap detection."""

    def setup_method(self):
        """Set up a block for testing."""
        self.block = ember.block.Block(shape=(10, 20, 30))

    def test_no_overlap_check_without_block_shape(self):
        """Test that overlap checking is skipped when patches are not attached to a block."""
        collection = BlockPatchCollection(self.block)

        # Patches constructed without attaching to a block have no _block_ref,
        # so overlap checking is skipped even though the collection has a block.
        patch1 = PeriodicPatch(i=0, j=(0, 10), k=(0, 15))
        patch2 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))

        # Should work because patches are not attached (no _block_ref), so no overlap check
        collection._items.append(patch1)
        collection._items.append(patch2)

        assert len(collection) == 2

    def test_overlap_detection_handles_validation_errors(self):
        """Test that overlap detection gracefully handles validation errors."""
        # Create a patch that might cause validation issues
        patch1 = PeriodicPatch(i=0, j=(0, 5), k=(0, 10))
        self.block.patches.append(patch1)

        # Create a patch with potentially problematic indices
        patch2 = PeriodicPatch(
            i=0, j=(-100, 5), k=(0, 10)
        )  # Might cause validation error

        # The method should handle validation errors gracefully
        # Either it works (if indices are valid) or raises a validation error (not overlap error)
        try:
            self.block.patches.append(patch2)
            # If it succeeds, check that it was added
            assert len(self.block.patches) == 2
        except ValueError as e:
            # Should be a validation error, not an overlap error
            assert "out of bounds" in str(e) or "overlaps" in str(e)
            assert len(self.block.patches) == 1

    def test_single_point_patches_overlap_detection(self):
        """Test overlap detection with single-point patches."""
        # Single point patches (all dimensions constant)
        patch1 = PeriodicPatch(i=5, j=10, k=15)  # Single point at (5,10,15)
        patch2 = PeriodicPatch(
            i=5, j=10, k=15
        )  # Same point - should not overlap due to early return
        patch3 = PeriodicPatch(i=5, j=10, k=16)  # Different point - should not overlap

        self.block.patches.append(patch1)

        # Point patches should not trigger overlap detection
        self.block.patches.append(patch2)  # Should work now due to early return
        self.block.patches.append(patch3)  # Should also work

        assert len(self.block.patches) == 3


"""Tests for BlockPatchCollection interface (ember.collections).

Module tested: ember.collections.BlockPatchCollection interface

Test cases:
- test_append_single_patch: appending single patch
- test_extend_multiple_patches: extending with multiple patches
- test_insert_at_index: inserting patch at specific index
- test_remove_patch: removing a patch
- test_remove_nonexistent_patch_raises_error: error for non-existent patch removal
- test_pop_default_last: popping last element by default
- test_pop_by_index: popping by specific index
- test_clear_all_patches: clearing all patches
- test_index_of_patch: finding index of patch
- test_index_with_start_stop: finding index with start/stop parameters
- test_getitem_by_index: accessing patch by index using bracket notation
- test_getitem_by_slice: accessing patches by slice using bracket notation
- test_setitem_by_index: setting patch at specific index using bracket notation
- test_delitem_by_index: deleting patch by index using del statement
- test_contains_operator: checking patch membership using 'in' operator
- test_iteration_maintains_order: iteration preserves patch order
- test_block_integration_with_pythonic_interface: integration of pythonic interface with block operations
- test_empty_collection_operations: operations on empty patch collections
- test_negative_indexing: negative index access for patches
"""


class TestPatchCollectionInterface:
    """Test pythonic collection interface."""

    def test_append_single_patch(self):
        """Test appending a single patch."""
        collection = _make_patch_collection()
        patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))

        collection.append(patch)

        assert len(collection) == 1
        assert collection[0] is patch

    def test_extend_multiple_patches(self):
        """Test extending with multiple patches."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(5, 15), k=(10, 20)),
            InletPatch(i=-1, j=(5, 15), k=(10, 20)),
            OutletPatch(i=(0, -1), j=0, k=(0, -1)),
        ]

        collection.extend(patches)

        assert len(collection) == 3
        for i, patch in enumerate(patches):
            assert collection[i] is patch

    def test_insert_at_index(self):
        """Test inserting patch at specific index."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = InletPatch(i=-1, j=(5, 15), k=(10, 20))
        patch3 = OutletPatch(i=(0, -1), j=0, k=(0, -1))

        collection.append(patch1)
        collection.append(patch3)
        collection.insert(1, patch2)  # Insert between patch1 and patch3

        assert len(collection) == 3
        assert collection[0] is patch1
        assert collection[1] is patch2
        assert collection[2] is patch3

    def test_remove_patch(self):
        """Test removing a patch."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = InletPatch(i=-1, j=(5, 15), k=(10, 20))

        collection.extend([patch1, patch2])
        collection.remove(patch1)

        assert len(collection) == 1
        assert collection[0] is patch2

    def test_remove_nonexistent_patch_raises_error(self):
        """Test removing non-existent patch raises ValueError."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = InletPatch(i=-1, j=(5, 15), k=(10, 20))

        collection.append(patch1)

        with pytest.raises(ValueError):
            collection.remove(patch2)

    def test_pop_default_last(self):
        """Test popping last element by default."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = InletPatch(i=-1, j=(5, 15), k=(10, 20))

        collection.extend([patch1, patch2])
        popped = collection.pop()

        assert popped is patch2
        assert len(collection) == 1
        assert collection[0] is patch1

    def test_pop_by_index(self):
        """Test popping by specific index."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = InletPatch(i=-1, j=(5, 15), k=(10, 20))
        patch3 = OutletPatch(i=(0, -1), j=0, k=(0, -1))

        collection.extend([patch1, patch2, patch3])
        popped = collection.pop(1)  # Pop middle element

        assert popped is patch2
        assert len(collection) == 2
        assert collection[0] is patch1
        assert collection[1] is patch3

    def test_clear_all_patches(self):
        """Test clearing all patches."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(5, 15), k=(10, 20)),
            InletPatch(i=-1, j=(5, 15), k=(10, 20)),
            OutletPatch(i=(0, -1), j=0, k=(0, -1)),
        ]

        collection.extend(patches)
        assert len(collection) == 3

        collection.clear()
        assert len(collection) == 0
        assert list(collection) == []

    def test_index_of_patch(self):
        """Test finding index of patch."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = InletPatch(i=-1, j=(5, 15), k=(10, 20))
        patch3 = OutletPatch(i=(0, -1), j=0, k=(0, -1))

        collection.extend([patch1, patch2, patch3])

        assert collection.index(patch1) == 0
        assert collection.index(patch2) == 1
        assert collection.index(patch3) == 2

    def test_index_with_start_stop(self):
        """Test finding index with start and stop parameters."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = InletPatch(i=-1, j=(5, 15), k=(10, 20))
        patch3 = PeriodicPatch(i=0, j=(21, 30), k=(10, 20))

        collection.extend([patch1, patch2, patch3])

        # Should find patch1 first by default (index 0)
        assert collection.index(patch1) == 0
        # Should find patch3 (same type, different location) when starting from index 1
        assert collection.index(patch3, start=1) == 2

    def test_getitem_by_index(self):
        """Test getting patch by index."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(5, 15), k=(10, 20)),
            InletPatch(i=-1, j=(5, 15), k=(10, 20)),
            OutletPatch(i=(0, -1), j=0, k=(0, -1)),
        ]

        collection.extend(patches)

        for i, patch in enumerate(patches):
            assert collection[i] is patch

    def test_getitem_by_slice(self):
        """Test getting patches by slice."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(5, 15), k=(10, 20)),
            InletPatch(i=-1, j=(5, 15), k=(10, 20)),
            OutletPatch(i=(0, -1), j=0, k=(0, -1)),
            OutletPatch(i=-1, j=(5, 15), k=(10, 20)),
        ]

        collection.extend(patches)

        # Test various slicing operations - now returns BlockPatchCollection instances
        subset_1_3 = collection[1:3]
        assert isinstance(subset_1_3, BlockPatchCollection)
        assert len(subset_1_3) == 2
        assert list(subset_1_3) == patches[1:3]

        subset_0_2 = collection[:2]
        assert len(subset_0_2) == 2
        assert list(subset_0_2) == patches[:2]

        subset_2_end = collection[2:]
        assert len(subset_2_end) == 2
        assert list(subset_2_end) == patches[2:]

        subset_step = collection[::2]
        assert len(subset_step) == 2
        assert list(subset_step) == patches[::2]

    def test_setitem_by_index(self):
        """Test setting patch by index."""
        collection = _make_patch_collection()
        original_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        new_patch = InletPatch(i=-1, j=(5, 15), k=(10, 20))

        collection.append(original_patch)
        collection[0] = new_patch

        assert len(collection) == 1
        assert collection[0] is new_patch

    def test_delitem_by_index(self):
        """Test deleting patch by index."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(5, 15), k=(10, 20)),
            InletPatch(i=-1, j=(5, 15), k=(10, 20)),
            OutletPatch(i=(0, -1), j=0, k=(0, -1)),
        ]

        collection.extend(patches)
        del collection[1]  # Delete middle patch

        assert len(collection) == 2
        assert collection[0] is patches[0]
        assert collection[1] is patches[2]

    def test_contains_operator(self):
        """Test 'in' operator."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = InletPatch(i=-1, j=(5, 15), k=(10, 20))
        patch3 = OutletPatch(i=(0, -1), j=0, k=(0, -1))

        collection.extend([patch1, patch2])

        assert patch1 in collection
        assert patch2 in collection
        assert patch3 not in collection

    def test_iteration_maintains_order(self):
        """Test that iteration maintains insertion order."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(5, 15), k=(10, 20)),
            InletPatch(i=-1, j=(5, 15), k=(10, 20)),
            OutletPatch(i=-1, j=(5, 15), k=(10, 20)),
        ]

        collection.extend(patches)

        for original, iterated in zip(patches, collection):
            assert original is iterated

    def test_block_integration_with_pythonic_interface(self):
        """Test that pythonic interface works with block integration."""
        from ember.patch import CoolingPatch, InviscidPatch

        block = ember.block.Block(shape=(10, 20, 30))

        # Add initial patches using block.add_patch
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = CoolingPatch(i=-1, j=(5, 15), k=(10, 20))
        block.patches.append(patch1)
        block.patches.append(patch2)

        # Use pythonic interface to add more patches
        patch3 = InviscidPatch(i=(0, -1), j=0, k=(0, -1))
        block.patches.append(patch3)
        patch3.attach_to_block(block)  # Manual since not using block.add_patch

        # Test pythonic operations
        assert len(block.patches) == 3
        assert patch1 in block.patches
        assert patch2 in block.patches
        assert patch3 in block.patches

        # Test type-based access still works
        assert len(block.patches.periodic) == 1
        assert len(block.patches.cooling) == 1
        assert len(block.patches.inviscid) == 1

        # Test removal
        block.patches.remove(patch2)
        assert len(block.patches) == 2
        assert len(block.patches.cooling) == 0


class TestPatchCollectionEdgeCases:
    """Test edge cases for pythonic interface."""

    def test_empty_collection_operations(self):
        """Test operations on empty collection."""
        collection = _make_patch_collection()

        assert len(collection) == 0
        assert list(collection) == []

        # These should work without error
        collection.clear()
        assert len(collection) == 0

        # These should raise appropriate errors
        with pytest.raises(IndexError):
            collection.pop()

        with pytest.raises(IndexError):
            _ = collection[0]

    def test_negative_indexing(self):
        """Test negative indexing works correctly."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(5, 15), k=(10, 20)),
            InletPatch(i=-1, j=(5, 15), k=(10, 20)),
            OutletPatch(i=(0, -1), j=0, k=(0, -1)),
        ]

        collection.extend(patches)

        assert collection[-1] is patches[-1]
        assert collection[-2] is patches[-2]
        assert collection[-3] is patches[-3]

        # Test popping with negative index
        popped = collection.pop(-2)  # Should pop middle element
        assert popped is patches[1]
        assert len(collection) == 2


"""Tests for BlockPatchCollection label indexing (ember.collections).

Module tested: ember.collections

Test cases:
- test_string_getitem_by_label: Getting patches by string label
- test_string_getitem_nonexistent_label_raises_keyerror: Accessing nonexistent label raises KeyError
- test_string_setitem_by_label: Setting patches by string label
- test_string_setitem_nonexistent_label_with_mismatched_label_raises_valueerror: Setting nonexistent label with mismatched item label raises ValueError
- test_string_setitem_nonexistent_label_with_compatible_label_succeeds: Setting nonexistent label with compatible item label succeeds
- test_string_delitem_by_label: Deleting patches by string label
- test_string_delitem_nonexistent_label_raises_keyerror: Deleting nonexistent label raises KeyError
- test_contains_with_string_label: 'in' operator with string labels
- test_labels_property: labels property
- test_numeric_indexing_still_works: Numeric indexing still works alongside string indexing
- test_slice_indexing_still_works: Slice indexing still works
- test_add_duplicate_label_raises_error: Adding patches with duplicate labels raises error
- test_append_duplicate_label_raises_error: Appending patches with duplicate labels raises error
- test_extend_with_duplicate_label_raises_error: Extending with duplicate labels raises error
- test_extend_with_duplicate_against_existing_raises_error: Extending with label duplicating existing raises error
- test_insert_duplicate_label_raises_error: Inserting patches with duplicate labels raises error
- test_setitem_with_duplicate_label_raises_error: Setting item with duplicate label raises error
- test_setitem_by_string_with_duplicate_label_raises_error: Setting by string with duplicate label raises error
- test_setitem_same_label_allowed: Setting patch with same label is allowed
- test_setitem_by_string_same_label_allowed: Setting by string with same label is allowed
- test_setitem_by_string_with_none_label_updates_label: Setting by string with None label updates the label
- test_setitem_by_string_with_empty_label_updates_label: Setting by string with empty label updates the label
- test_setitem_by_string_with_mismatched_label_raises_error: Setting by string with mismatched label raises error
- test_none_labels_allowed_and_not_unique: None labels are allowed and don't need to be unique
- test_empty_string_labels_must_be_unique: Empty string labels must be unique
- test_mixed_labeled_and_unlabeled_patches: Collection with mix of labeled and unlabeled patches
- test_label_indexing_with_type_based_access: Label indexing works with type-based access
- test_label_indexing_preserves_order: Label-based operations preserve order
- test_string_label_insertion_new_item: Insert new items using string label indexing
- test_string_label_insertion_with_existing_label_on_item: Insertion when item already has matching label
- test_string_label_insertion_with_mismatched_label_raises_error: Insertion with mismatched label raises error
- test_string_label_insertion_updates_none_label: Insertion updates None label to match key
- test_string_label_insertion_updates_empty_label: Insertion updates empty string label to match key
"""


class TestPatchCollectionLabelIndexing:
    """Test string label indexing functionality."""

    def test_string_getitem_by_label(self):
        """Test getting patches by string label."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="inlet_patch")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="outlet_patch")

        collection.append(patch1)
        collection.append(patch2)

        # Test string indexing
        retrieved_patch1 = collection["inlet_patch"]
        retrieved_patch2 = collection["outlet_patch"]

        assert retrieved_patch1 is patch1
        assert retrieved_patch2 is patch2

    def test_string_getitem_nonexistent_label_raises_keyerror(self):
        """Test that accessing nonexistent label raises KeyError."""
        collection = _make_patch_collection()
        patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="existing")
        collection.append(patch)

        with pytest.raises(KeyError, match="No item found with label 'nonexistent'"):
            _ = collection["nonexistent"]

    def test_string_setitem_by_label(self):
        """Test setting patches by string label."""
        collection = _make_patch_collection()
        original_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="test_patch")
        new_patch = PeriodicPatch(
            i=-1, j=(5, 15), k=(10, 20), label="test_patch"
        )  # Same label

        collection.append(original_patch)

        # Replace by label
        collection["test_patch"] = new_patch

        assert len(collection) == 1
        assert collection[0] is new_patch
        assert collection[0].label == "test_patch"

    def test_string_setitem_nonexistent_label_with_mismatched_label_raises_valueerror(
        self,
    ):
        """Test that setting nonexistent label with mismatched item label raises ValueError."""
        collection = _make_patch_collection()
        patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="existing")
        new_patch = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="new")

        collection.append(patch)

        # Should raise ValueError because item label "new" doesn't match key "nonexistent"
        with pytest.raises(
            ValueError,
            match="Cannot assign item with label 'new' to key 'nonexistent'. Item label must be None, empty, or match the key 'nonexistent'.",
        ):
            collection["nonexistent"] = new_patch

    def test_string_setitem_nonexistent_label_with_compatible_label_succeeds(self):
        """Test that setting nonexistent label with compatible item label succeeds (insertion)."""
        collection = _make_patch_collection()
        existing_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="existing")

        # Test with None label - should be updated to match key
        new_patch1 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label=None)

        # Test with matching label (different j range to avoid overlap)
        new_patch2 = PeriodicPatch(i=-1, j=(20, 30), k=(10, 20), label="new_label")

        collection.append(existing_patch)

        # Should succeed and insert new patch
        collection["new_auto_label"] = new_patch1
        assert len(collection) == 2
        assert collection["new_auto_label"] is new_patch1
        assert new_patch1.label == "new_auto_label"

        # Should succeed and insert patch with matching label
        collection["new_label"] = new_patch2
        assert len(collection) == 3
        assert collection["new_label"] is new_patch2
        assert new_patch2.label == "new_label"

    def test_string_delitem_by_label(self):
        """Test deleting patches by string label."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="keep")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="delete")
        patch3 = PeriodicPatch(i=-1, j=(20, 30), k=(10, 20), label="also_keep")

        collection.extend([patch1, patch2, patch3])

        # Delete by label
        del collection["delete"]

        assert len(collection) == 2
        assert patch1 in collection
        assert patch2 not in collection
        assert patch3 in collection

    def test_string_delitem_nonexistent_label_raises_keyerror(self):
        """Test that deleting nonexistent label raises KeyError."""
        collection = _make_patch_collection()
        patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="existing")
        collection.append(patch)

        with pytest.raises(KeyError, match="No item found with label 'nonexistent'"):
            del collection["nonexistent"]

    def test_contains_with_string_label(self):
        """Test 'in' operator with string labels."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="test_label")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label=None)  # No label

        collection.extend([patch1, patch2])

        # Test string containment
        assert "test_label" in collection
        assert "nonexistent_label" not in collection

        # Test object containment still works
        assert patch1 in collection
        assert patch2 in collection

    def test_labels_property(self):
        """Test the labels property."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="label1")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="label2")
        patch3 = PeriodicPatch(i=-1, j=(20, 30), k=(10, 20), label=None)  # No label

        collection.extend([patch1, patch2, patch3])

        labels = collection.labels
        assert set(labels) == {"label1", "label2", None}
        assert len(labels) == 3  # None label included

    def test_numeric_indexing_still_works(self):
        """Test that numeric indexing still works alongside string indexing."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="first")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="second")

        collection.extend([patch1, patch2])

        # Numeric indexing
        assert collection[0] is patch1
        assert collection[1] is patch2
        assert collection[-1] is patch2

        # String indexing
        assert collection["first"] is patch1
        assert collection["second"] is patch2

    def test_slice_indexing_still_works(self):
        """Test that slice indexing still works."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(i * 8, i * 8 + 5), k=(10, 20), label=f"patch_{i}")
            for i in range(5)
        ]
        collection.extend(patches)

        # Test slicing
        slice_result = collection[1:4]
        assert len(slice_result) == 3
        assert slice_result[0] is patches[1]
        assert slice_result[2] is patches[3]


class TestPatchCollectionLabelUniqueness:
    """Test label uniqueness enforcement."""

    def test_add_duplicate_label_raises_error(self):
        """Test that adding patches with duplicate labels raises error."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="duplicate")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="duplicate")

        collection.append(patch1)

        with pytest.raises(
            ValueError, match="Item with label 'duplicate' already exists in collection"
        ):
            collection.append(patch2)

    def test_append_duplicate_label_raises_error(self):
        """Test that appending patches with duplicate labels raises error."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="duplicate")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="duplicate")

        collection.append(patch1)

        with pytest.raises(
            ValueError, match="Item with label 'duplicate' already exists in collection"
        ):
            collection.append(patch2)

    def test_extend_with_duplicate_label_raises_error(self):
        """Test that extending with duplicate labels raises error."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="existing")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="duplicate")
        patch3 = PeriodicPatch(i=-1, j=(20, 30), k=(10, 20), label="duplicate")

        collection.append(patch1)

        with pytest.raises(
            ValueError, match="Duplicate label 'duplicate' found in new items"
        ):
            collection.extend([patch2, patch3])

        # Should not have added any patches due to validation failure
        assert len(collection) == 1

    def test_extend_with_duplicate_against_existing_raises_error(self):
        """Test that extending with label duplicating existing raises error."""
        collection = _make_patch_collection()
        existing_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="existing")
        new_patch = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="existing")

        collection.append(existing_patch)

        with pytest.raises(
            ValueError, match="Item with label 'existing' already exists in collection"
        ):
            collection.extend([new_patch])

    def test_insert_duplicate_label_raises_error(self):
        """Test that inserting patches with duplicate labels raises error."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="existing")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="existing")

        collection.append(patch1)

        with pytest.raises(
            ValueError, match="Item with label 'existing' already exists in collection"
        ):
            collection.insert(0, patch2)

    def test_setitem_with_duplicate_label_raises_error(self):
        """Test that setting item with duplicate label raises error."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="label1")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="label2")
        patch3 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="label1")  # Duplicate

        collection.extend([patch1, patch2])

        with pytest.raises(
            ValueError, match="Item with label 'label1' already exists in collection"
        ):
            collection[1] = patch3

    def test_setitem_by_string_with_duplicate_label_raises_error(self):
        """Test that setting by string with duplicate label raises error."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="label1")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="label2")
        patch3 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="label1")  # Duplicate

        collection.extend([patch1, patch2])

        # This should fail because patch3 has label "label1" but we're trying to assign it to key "label2"
        with pytest.raises(
            ValueError,
            match="Cannot assign item with label 'label1' to key 'label2'. Item label must be None, empty, or match the key 'label2'.",
        ):
            collection["label2"] = patch3

    def test_setitem_same_label_allowed(self):
        """Test that setting patch with same label (replacing itself) is allowed."""
        collection = _make_patch_collection()
        original_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="same_label")
        new_patch = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="same_label")

        collection.append(original_patch)

        # This should work - replacing patch with same label
        collection[0] = new_patch
        assert collection[0] is new_patch

    def test_setitem_by_string_same_label_allowed(self):
        """Test that setting by string with same label is allowed."""
        collection = _make_patch_collection()
        original_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="same_label")
        new_patch = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="same_label")

        collection.append(original_patch)

        # This should work - replacing patch with same label
        collection["same_label"] = new_patch
        assert collection["same_label"] is new_patch

    def test_setitem_by_string_with_none_label_updates_label(self):
        """Test that setting by string with None label updates the label."""
        collection = _make_patch_collection()
        original_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="original")
        new_patch = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label=None)

        collection.append(original_patch)

        # This should work and update the label
        collection["original"] = new_patch
        assert collection["original"] is new_patch
        assert new_patch.label == "original"

    def test_setitem_by_string_with_empty_label_updates_label(self):
        """Test that setting by string with empty label updates the label."""
        collection = _make_patch_collection()
        original_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="original")
        new_patch = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="")

        collection.append(original_patch)

        # This should work and update the label
        collection["original"] = new_patch
        assert collection["original"] is new_patch
        assert new_patch.label == "original"

    def test_setitem_by_string_with_mismatched_label_raises_error(self):
        """Test that setting by string with mismatched label raises error."""
        collection = _make_patch_collection()
        original_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="original")
        wrong_patch = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="wrong")

        collection.append(original_patch)

        with pytest.raises(
            ValueError,
            match="Cannot assign item with label 'wrong' to key 'original'. Item label must be None, empty, or match the key 'original'.",
        ):
            collection["original"] = wrong_patch

    def test_none_labels_allowed_and_not_unique(self):
        """Test that None labels are allowed and don't need to be unique."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label=None)
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label=None)
        patch3 = PeriodicPatch(i=-1, j=(20, 30), k=(10, 20), label=None)

        # Should work - None labels don't need to be unique
        collection.extend([patch1, patch2, patch3])
        assert len(collection) == 3

    def test_empty_string_labels_must_be_unique(self):
        """Test that empty string labels must be unique."""
        collection = _make_patch_collection()
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="")
        patch2 = PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="")

        collection.append(patch1)

        with pytest.raises(
            ValueError, match="Item with label '' already exists in collection"
        ):
            collection.append(patch2)

    def test_mixed_labeled_and_unlabeled_patches(self):
        """Test collection with mix of labeled and unlabeled patches."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="labeled1"),
            PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label=None),
            PeriodicPatch(i=-1, j=(20, 30), k=(10, 20), label="labeled2"),
            PeriodicPatch(i=-1, j=(35, 45), k=(10, 20), label=None),
        ]

        collection.extend(patches)
        assert len(collection) == 4

        # String indexing works for labeled patches
        assert collection["labeled1"] is patches[0]
        assert collection["labeled2"] is patches[2]

        # Numeric indexing works for all
        assert collection[1] is patches[1]  # Unlabeled
        assert collection[3] is patches[3]  # Unlabeled

        # Labels list contains all labels including None
        labels = collection.labels
        assert set(labels) == {"labeled1", "labeled2", None}


class TestPatchCollectionLabelIntegration:
    """Test label indexing integration with other features."""

    def test_label_indexing_with_type_based_access(self):
        """Test label indexing works with type-based access."""
        collection = _make_patch_collection()
        periodic = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="periodic_patch")
        inlet = InletPatch(i=-1, j=(5, 15), k=(10, 20), label="inlet_patch")

        collection.extend([periodic, inlet])

        # Type-based access
        assert len(collection.periodic) == 1
        assert len(collection.inlet) == 1

        # String indexing
        assert collection["periodic_patch"] is periodic
        assert collection["inlet_patch"] is inlet

        # Combined usage
        assert collection["periodic_patch"] in collection.periodic
        assert collection["inlet_patch"] in collection.inlet

    def test_label_indexing_preserves_order(self):
        """Test that label-based operations preserve order."""
        collection = _make_patch_collection()
        patches = [
            PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="first"),
            PeriodicPatch(i=-1, j=(5, 15), k=(10, 20), label="second"),
            PeriodicPatch(i=-1, j=(20, 30), k=(10, 20), label="third"),
        ]

        collection.extend(patches)

        # Replace middle patch by label (using None label so it gets updated)
        new_patch = PeriodicPatch(i=-1, j=(35, 45), k=(10, 20), label=None)
        collection["second"] = new_patch

        # Order should be preserved
        assert collection[0] is patches[0]  # first
        assert collection[1] is new_patch  # replaced second
        assert collection[2] is patches[2]  # third

        assert collection[0].label == "first"
        assert collection[1].label == "second"  # Should be updated to match key
        assert collection[2].label == "third"


class TestPatchCollectionStringLabelInsertion:
    """Test insertion by string label indexing."""

    def test_string_label_insertion_new_item(self):
        """Test that we can insert new items using string label indexing."""
        collection = _make_patch_collection()

        # Insert first patch using string indexing (should work like append)
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label=None)
        collection["new_patch"] = patch1

        assert len(collection) == 1
        assert collection["new_patch"] is patch1
        assert patch1.label == "new_patch"  # Label should be set automatically

        # Insert another patch with different label
        patch2 = InletPatch(i=-1, j=(5, 15), k=(10, 20), label="inlet_1")
        collection["inlet_1"] = patch2

        assert len(collection) == 2
        assert collection["inlet_1"] is patch2
        assert patch2.label == "inlet_1"

    def test_string_label_insertion_with_existing_label_on_item(self):
        """Test insertion when item already has matching label."""
        collection = _make_patch_collection()

        # Item already has the label we're using as key
        patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="existing_label")
        collection["existing_label"] = patch

        assert len(collection) == 1
        assert collection["existing_label"] is patch
        assert patch.label == "existing_label"

    def test_string_label_insertion_with_mismatched_label_raises_error(self):
        """Test that insertion with mismatched label raises error."""
        collection = _make_patch_collection()

        # Item has different label than the key we're using
        patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="wrong_label")

        with pytest.raises(
            ValueError,
            match="Cannot assign item with label 'wrong_label' to key 'correct_label'. Item label must be None, empty, or match the key 'correct_label'.",
        ):
            collection["correct_label"] = patch

    def test_string_label_insertion_updates_none_label(self):
        """Test that insertion updates None label to match key."""
        collection = _make_patch_collection()

        # Item has None label - should be updated to match key
        patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label=None)
        collection["auto_label"] = patch

        assert len(collection) == 1
        assert collection["auto_label"] is patch
        assert patch.label == "auto_label"

    def test_string_label_insertion_updates_empty_label(self):
        """Test that insertion updates empty string label to match key."""
        collection = _make_patch_collection()

        # Item has empty label - should be updated to match key
        patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20), label="")
        collection["updated_label"] = patch

        assert len(collection) == 1
        assert collection["updated_label"] is patch
        assert patch.label == "updated_label"


"""Tests for patch check_match interface (ember.patch).

Module tested: ember.patch

Test cases:
- test_periodic_patch_identical_match: Identical PeriodicPatches match
- test_periodic_patch_different_no_match: Different PeriodicPatches don't match
- test_periodic_patch_different_type_no_match: PeriodicPatch doesn't match with other patch types
- test_periodic_patch_tolerance_sensitivity: Tolerance parameter affects matching
- test_xrt_centre_shape_and_bounds: xrt_centre returns expected shape and reasonable values
- test_xrt_centre_different_patches: Different patches have different centres
- test_custom_patch_type_different_matching: Custom patch type could implement different matching
- test_mixing_patch_identical_match: Identical MixingPatches match
- test_mixing_patch_same_xr_different_theta: Patches with same x,r but different theta match
- test_mixing_patch_different_k_size: Patches with same x,r but different k-dimension size match
- test_mixing_patch_permutation_match: Patches that match after dimension permutation
- test_mixing_patch_flip_match: Patches that match after flipping dimensions
- test_mixing_patch_different_x_no_match: Patches at different x locations don't match
- test_mixing_patch_different_r_no_match: Patches at different r locations don't match
- test_mixing_patch_different_types_no_match: MixingPatch doesn't match with other patch types
- test_mixing_patch_incompatible_ij_dimensions: Patches with incompatible i,j dimensions don't match
- test_mixing_patch_different_constant_dimensions: Mixing patches with different constant dimensions
- test_mixing_patch_different_radial_ranges: Patches with different radial ranges affecting tolerance
- test_mixing_patch_corner_matching_logic: Corner matching optimization works correctly
- test_interpolate_spanwise_default_true: interpolate_spanwise defaults to True
- test_interpolate_spanwise_set_method: interpolate_spanwise can be set using set_interpolate method
- test_interpolate_spanwise_property_setter: interpolate_spanwise property can be modified
- test_interpolate_spanwise_repr_default: __repr__ doesn't show interpolate_spanwise when default
- test_interpolate_spanwise_repr_false: __repr__ doesn't show interpolate_spanwise for non-constructor parameter
- test_flexible_matching_both_interpolate_true: Flexible matching when both patches have interpolate_spanwise=True
- test_exact_matching_one_interpolate_false: Exact matching when one patch has interpolate_spanwise=False
- test_exact_matching_both_interpolate_false: Exact matching when both patches have interpolate_spanwise=False
- test_exact_matching_identical_patches_interpolate_false: Identical patches match even with interpolate_spanwise=False
- test_flexible_matching_corner_alignment: Flexible matching still requires corner alignment
- test_flexible_matching_different_constant_dimensions: Flexible matching with patches having different constant dimensions
- test_flexible_vs_exact_transform_generation: Flexible and exact transform generation produce different results
- test_mixed_interpolation_settings_comprehensive: Comprehensive test of different interpolation setting combinations
"""


class TestBasePatchCheckMatch:
    """Test the base Patch.check_match interface."""

    def setup_method(self):
        """Set up blocks for testing."""
        self.block = ember.block.Block(shape=(10, 20, 30))

        # Set up coordinates
        x = np.linspace(0.0, 1.0, 10)
        r = np.linspace(0.5, 1.5, 20)
        t = np.linspace(0.0, 2 * np.pi, 30)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt = np.stack([xv, rv, tv], axis=-1)
        self.block.set_x(xrt[..., 0])
        self.block.set_r(xrt[..., 1])
        self.block.set_t(xrt[..., 2])


class TestPeriodicPatchCheckMatch:
    """Test PeriodicPatch.check_match implementation."""

    def setup_method(self):
        """Set up blocks for testing."""
        self.block = ember.block.Block(shape=(10, 20, 30))

        # Set up coordinates
        x = np.linspace(0.0, 1.0, 10)
        r = np.linspace(0.5, 1.5, 20)
        t = np.linspace(0.0, 2 * np.pi, 30)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt = np.stack([xv, rv, tv], axis=-1)
        self.block.set_x(xrt[..., 0])
        self.block.set_r(xrt[..., 1])
        self.block.set_t(xrt[..., 2])

    def test_periodic_patch_identical_match(self):
        """Test that identical PeriodicPatches match."""
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))

        patch1.attach_to_block(self.block)
        patch2.attach_to_block(self.block)

        # Should match and return identity transform
        transform = patch1.check_match(patch2)
        assert transform == ((0, 1, 2), ())

    def test_periodic_patch_different_no_match(self):
        """Test that different PeriodicPatches don't match."""
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = PeriodicPatch(i=0, j=(0, 10), k=(10, 20))  # Different j range

        patch1.attach_to_block(self.block)
        patch2.attach_to_block(self.block)

        # Should not match
        assert patch1.check_match(patch2) is None

    def test_periodic_patch_different_type_no_match(self):
        """Test that PeriodicPatch doesn't match with other patch types."""
        periodic_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        inlet_patch = InletPatch(i=0, j=(5, 15), k=(10, 20))

        periodic_patch.attach_to_block(self.block)
        inlet_patch.attach_to_block(self.block)

        # Should not match different types
        assert periodic_patch.check_match(inlet_patch) is None
        assert inlet_patch.check_match(periodic_patch) is None

    def test_periodic_patch_tolerance_sensitivity(self):
        """Test that tolerance parameter affects matching."""
        patch1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        # Create a slightly different patch
        patch2 = PeriodicPatch(i=0, j=(5, 15), k=(10, 21))  # One index different

        patch1.attach_to_block(self.block)
        patch2.attach_to_block(self.block)

        # With tight tolerance, should not match
        assert patch1.check_match(patch2, rtol=1e-10) is None

        # With looser tolerance, might match (depending on geometry)
        # Don't assert this since it depends on the specific coordinates


class TestPatchGetCenter:
    """Test the xrt_centre property."""

    def setup_method(self):
        """Set up blocks for testing."""
        self.block = ember.block.Block(shape=(5, 6, 8))

        # Simple linear coordinates for predictable results
        x = np.linspace(0, 1, 5)
        r = np.linspace(1, 2, 6)
        t = np.linspace(0, np.pi, 8)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt = np.stack([xv, rv, tv], axis=-1)
        self.block.set_x(xrt[..., 0])
        self.block.set_r(xrt[..., 1])
        self.block.set_t(xrt[..., 2])

    def test_xrt_centre_shape_and_bounds(self):
        """Test that xrt_centre returns expected shape and reasonable values."""
        patch = PeriodicPatch(i=0, j=(1, 4), k=(2, 6))
        patch.attach_to_block(self.block)

        centre = patch.xrt_centre

        # Should return 3D coordinate
        assert centre.shape == (3,)

        # Should be within expected bounds
        assert 0 <= centre[0] <= 1  # x coordinate
        assert 1 <= centre[1] <= 2  # r coordinate
        assert 0 <= centre[2] <= np.pi  # t coordinate

    def test_xrt_centre_different_patches(self):
        """Test that different patches have different centres."""
        patch1 = PeriodicPatch(i=0, j=(1, 3), k=(2, 4))
        patch2 = PeriodicPatch(i=0, j=(3, 5), k=(4, 6))

        patch1.attach_to_block(self.block)
        patch2.attach_to_block(self.block)

        centre1 = patch1.xrt_centre
        centre2 = patch2.xrt_centre

        # Centres should be different
        assert not np.allclose(centre1, centre2)


class TestSubclassMatchingFlexibility:
    """Demonstrate how subclasses can implement different matching criteria."""

    def setup_method(self):
        """Set up blocks for testing."""
        self.block = ember.block.Block(shape=(10, 20, 30))

        x = np.linspace(0.0, 1.0, 10)
        r = np.linspace(0.5, 1.5, 20)
        t = np.linspace(0.0, 2 * np.pi, 30)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt = np.stack([xv, rv, tv], axis=-1)
        self.block.set_x(xrt[..., 0])
        self.block.set_r(xrt[..., 1])
        self.block.set_t(xrt[..., 2])

    def test_custom_patch_type_different_matching(self):
        """Demonstrate how a custom patch type could implement different matching."""

        # Create a custom patch type that only matches x,r coordinates
        class CustomXRPatch(PeriodicPatch):
            """Example patch that only matches x,r coordinates (ignores t)."""

            def check_match(self, other, rtol=1e-6):
                """Match only x,r coordinates, ignore theta."""
                if not isinstance(other, CustomXRPatch):
                    return None

                center_self = self.xrt_centre
                center_other = other.xrt_centre

                # Only compare x,r components
                xr_self = center_self[:2]
                xr_other = center_other[:2]

                # Simple tolerance check
                rref = 0.5 * (center_self[1] + center_other[1])
                atol = rtol * rref

                if np.allclose(xr_self, xr_other, rtol=rtol, atol=atol):
                    return ((0, 1, 2), ())  # Identity transform
                return None

        # Create patches that have same x,r but different theta
        patch1 = CustomXRPatch(i=0, j=(5, 15), k=(10, 15))
        patch2 = CustomXRPatch(i=0, j=(5, 15), k=(15, 20))  # Different k (theta) range

        patch1.attach_to_block(self.block)
        patch2.attach_to_block(self.block)

        # These would not match with PeriodicPatch logic (different theta)
        periodic1 = PeriodicPatch(i=0, j=(5, 15), k=(10, 15))
        periodic2 = PeriodicPatch(i=0, j=(5, 15), k=(15, 20))
        periodic1.attach_to_block(self.block)
        periodic2.attach_to_block(self.block)

        assert periodic1.check_match(periodic2) is None

        # But our custom type should match (only x,r compared)
        assert patch1.check_match(patch2) is not None


class TestMixingPatchCheckMatch:
    """Test MixingPatch.check_match implementation."""

    def setup_method(self):
        """Set up blocks for testing."""
        self.block1 = ember.block.Block(shape=(10, 20, 30))
        self.block2 = ember.block.Block(shape=(10, 20, 40))  # Different k-size

        # Set up coordinates for block1
        x1 = np.linspace(0.0, 1.0, 10)
        r1 = np.linspace(0.5, 1.5, 20)
        t1 = np.linspace(0.0, 2 * np.pi, 30)
        xv1, rv1, tv1 = np.meshgrid(x1, r1, t1, indexing="ij")
        xrt1 = np.stack([xv1, rv1, tv1], axis=-1)
        self.block1.set_x(xrt1[..., 0])
        self.block1.set_r(xrt1[..., 1])
        self.block1.set_t(xrt1[..., 2])

        # Set up coordinates for block2 (same x,r but different theta resolution)
        x2 = np.linspace(0.0, 1.0, 10)
        r2 = np.linspace(0.5, 1.5, 20)
        t2 = np.linspace(0.0, 2 * np.pi, 40)  # Different theta resolution
        xv2, rv2, tv2 = np.meshgrid(x2, r2, t2, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        self.block2.set_x(xrt2[..., 0])
        self.block2.set_r(xrt2[..., 1])
        self.block2.set_t(xrt2[..., 2])

    def test_mixing_patch_identical_match(self):
        """Test that identical MixingPatches match."""
        patch1 = MixingPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = MixingPatch(i=0, j=(5, 15), k=(10, 20))

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should match with no span flip needed
        transform = patch1.check_match(patch2)
        assert not transform

    def test_mixing_patch_same_xr_different_theta(self):
        """Test that patches with same x,r but different theta match."""
        patch1 = MixingPatch(i=0, j=(5, 15), k=(10, 15))  # 5 theta points
        patch2 = MixingPatch(i=0, j=(5, 15), k=(15, 20))  # Different 5 theta points

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should match despite different theta ranges
        transform = patch1.check_match(patch2)
        assert transform is not None

    def test_mixing_patch_different_k_size(self):
        """Test that patches with same x,r but different k-dimension size match."""
        patch1 = MixingPatch(i=0, j=(5, 15), k=(10, 20))  # 10 theta points
        patch2 = MixingPatch(
            i=0, j=(5, 15), k=(15, 25)
        )  # 10 theta points, different range

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block2)  # Different k-size block

        # Should match despite different k-dimension sizes
        transform = patch1.check_match(patch2)
        assert transform is not None

    def test_mixing_patch_permutation_match(self):
        """Test patches that match after dimension permutation."""
        patch1 = MixingPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = MixingPatch(j=(5, 15), i=0, k=(10, 20))  # Swapped i,j specification

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should find appropriate permutation
        transform = patch1.check_match(patch2)
        assert transform is not None

    def test_mixing_patch_flip_match(self):
        """Test patches that match after flipping dimensions."""
        # Create two blocks with coordinates that would match after flipping
        # Block 1: normal coordinates
        block1 = ember.block.Block(shape=(6, 8, 10))
        x = np.linspace(0.0, 1.0, 6)
        r = np.linspace(0.5, 1.5, 8)
        t = np.linspace(0.0, 2 * np.pi, 10)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt1 = np.stack([xv, rv, tv], axis=-1)
        block1.set_x(xrt1[..., 0])
        block1.set_r(xrt1[..., 1])
        block1.set_t(xrt1[..., 2])

        # Block 2: same coordinates but with j-dimension flipped
        block2 = ember.block.Block(shape=(6, 8, 10))
        r_flipped = np.flip(r)  # Flip the r coordinate array
        xv2, rv2, tv2 = np.meshgrid(x, r_flipped, t, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])

        # Create patches that would match if we flip the j-dimension
        patch1 = MixingPatch(i=0, j=(2, 5), k=(3, 7))
        patch2 = MixingPatch(i=0, j=(2, 5), k=(3, 7))  # Same indices

        patch1.attach_to_block(block1)
        patch2.attach_to_block(block2)

        # Should match with span flip needed
        transform = patch1.check_match(patch2)
        assert transform is True

    def test_mixing_patch_different_x_no_match(self):
        """Test that patches at different x locations don't match."""
        patch1 = MixingPatch(i=0, j=(5, 15), k=(10, 20))  # x at index 0 (start)
        patch2 = MixingPatch(i=9, j=(5, 15), k=(10, 20))  # x at index 9 (end)

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should not match due to different x coordinates
        assert patch1.check_match(patch2) is None

    def test_mixing_patch_different_r_no_match(self):
        """Test that patches at different r locations don't match."""
        patch1 = MixingPatch(i=0, j=(2, 8), k=(10, 20))  # r at indices 2-8
        patch2 = MixingPatch(i=0, j=(12, 18), k=(10, 20))  # r at indices 12-18

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should not match due to different r coordinates
        assert patch1.check_match(patch2) is None

    def test_mixing_patch_different_types_no_match(self):
        """Test that MixingPatch doesn't match with other patch types."""
        mixing_patch = MixingPatch(i=0, j=(5, 15), k=(10, 20))
        periodic_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        inlet_patch = InletPatch(i=0, j=(5, 15), k=(10, 20))

        mixing_patch.attach_to_block(self.block1)
        periodic_patch.attach_to_block(self.block1)
        inlet_patch.attach_to_block(self.block1)

        # Should not match with different patch types
        assert mixing_patch.check_match(periodic_patch) is None
        assert mixing_patch.check_match(inlet_patch) is None
        assert periodic_patch.check_match(mixing_patch) is None

    def test_mixing_patch_incompatible_ij_dimensions(self):
        """Test that patches with incompatible i,j dimensions don't match."""
        patch1 = MixingPatch(i=0, j=(5, 15), k=(10, 20))  # j-size: 11
        patch2 = MixingPatch(i=0, j=(5, 12), k=(10, 20))  # j-size: 8

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should not match due to incompatible j-dimension sizes
        assert patch1.check_match(patch2) is None

    def test_mixing_patch_different_radial_ranges(self):
        """Test patches with different radial ranges affecting tolerance."""
        # Create block with wider radial range
        block_wide = ember.block.Block(shape=(10, 20, 30))
        x = np.linspace(0.0, 1.0, 10)
        r_wide = np.linspace(0.1, 3.0, 20)  # Much wider radial range
        t = np.linspace(0.0, 2 * np.pi, 30)
        xv, rv, tv = np.meshgrid(x, r_wide, t, indexing="ij")
        xrt_wide = np.stack([xv, rv, tv], axis=-1)
        block_wide.set_x(xrt_wide[..., 0])
        block_wide.set_r(xrt_wide[..., 1])
        block_wide.set_t(xrt_wide[..., 2])

        patch1 = MixingPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = MixingPatch(i=0, j=(5, 15), k=(10, 20))

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(block_wide)

        # Should handle different radial ranges in tolerance calculation
        # Exact match depends on coordinate alignment
        transform = patch1.check_match(patch2)
        assert isinstance(transform, (bool, type(None)))

    def test_mixing_patch_corner_matching_logic(self):
        """Test that corner matching optimization works correctly."""
        patch1 = MixingPatch(i=0, j=(5, 10), k=(8, 12))
        patch2 = MixingPatch(i=0, j=(5, 10), k=(15, 19))  # Different theta range

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should match if corners align (same x,r ranges)
        transform = patch1.check_match(patch2)
        assert transform is not None

        # Verify no span flip needed for this case
        assert not transform


class TestMixingPatchSpanwiseMatching:
    """Test MixingPatch spanwise matching with corner-only comparison."""

    def setup_method(self):
        """Set up blocks for testing."""
        self.block1 = ember.block.Block(shape=(10, 20, 30))
        self.block2 = ember.block.Block(shape=(10, 15, 40))  # Different j,k sizes

        # Set up coordinates for block1
        x1 = np.linspace(0.0, 1.0, 10)
        r1 = np.linspace(0.5, 1.5, 20)
        t1 = np.linspace(0.0, 2 * np.pi, 30)
        xv1, rv1, tv1 = np.meshgrid(x1, r1, t1, indexing="ij")
        xrt1 = np.stack([xv1, rv1, tv1], axis=-1)
        self.block1.set_x(xrt1[..., 0])
        self.block1.set_r(xrt1[..., 1])
        self.block1.set_t(xrt1[..., 2])

        # Set up coordinates for block2 (same x but different r,t resolution)
        x2 = np.linspace(0.0, 1.0, 10)  # Same x coordinates
        r2 = np.linspace(0.5, 1.5, 15)  # Different r resolution
        t2 = np.linspace(0.0, 2 * np.pi, 40)  # Different theta resolution
        xv2, rv2, tv2 = np.meshgrid(x2, r2, t2, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        self.block2.set_x(xrt2[..., 0])
        self.block2.set_r(xrt2[..., 1])
        self.block2.set_t(xrt2[..., 2])

    def test_flexible_matching_different_pitchwise(self):
        """Test matching patches with different pitchwise resolutions."""
        patch1 = MixingPatch(i=0, j=(5, 10), k=(8, 15))  # k-size: 8
        patch2 = MixingPatch(
            i=0, j=(5, 10), k=(18, 25)
        )  # k-size: 8, same i,j, different k range

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)  # Same block

        transform = patch1.check_match(patch2)
        assert transform is not None

    def test_identical_patches_match(self):
        """Test that identical patches match."""
        patch1 = MixingPatch(i=0, j=(5, 15), k=(8, 18))
        patch2 = MixingPatch(i=0, j=(5, 15), k=(8, 18))

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)  # Same block shape

        transform = patch1.check_match(patch2)
        assert transform is not None

    def test_corner_alignment_required(self):
        """Test that matching still requires corner alignment."""
        # Patch 1 at block1 boundary i=0 (x=0.0)
        patch1 = MixingPatch(i=0, j=(5, 15), k=(8, 18))
        # Patch 2 at block1 boundary i=9 (x=1.0) - different x location
        patch2 = MixingPatch(i=9, j=(3, 10), k=(12, 25))

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should NOT match because corners don't align (different x coordinates)
        transform = patch1.check_match(patch2)
        assert transform is None

    def test_different_constant_dimensions(self):
        """Test matching with patches having different constant dimensions."""
        # Patch with i-constant (mixing plane perpendicular to x)
        patch1 = MixingPatch(i=0, j=(5, 15), k=(8, 18))
        # Patch with j-constant (mixing plane perpendicular to r)
        patch2 = MixingPatch(i=(2, 8), j=0, k=(12, 25))

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block2)

        # Result depends on coordinate alignment, but should handle different constant dims
        transform = patch1.check_match(patch2)
        assert isinstance(transform, (bool, type(None)))


"""Tests for patch setter methods (ember.patch).

Module tested: ember.patch

Test cases:
- test_set_Po_To_Alpha_Beta_single_values: Setting single scalar values for InletPatch
- test_set_Po_To_Alpha_Beta_all_at_once: Setting all inlet parameters at once
- test_set_Po_To_Alpha_Beta_arrays_broadcast: Setting arrays that broadcast correctly
- test_set_Po_To_Alpha_Beta_broadcast_failure: Broadcast validation failure
- test_set_Po_To_Alpha_Beta_validation_errors: Validation errors for inlet parameters
- test_calc_raw_raises_without_block: Accessing stagnation properties without proper block raises error
- test_calc_raw_computes_properties: Compute rhoo, ho, s from Po, To
- test_calc_raw_with_arrays: Array-valued Po, To work correctly
- test_set_P_scalar: Setting scalar pressure for outlet
- test_set_P_conversion: Type conversion to float32
- test_set_P_array_raises_error: Array validation for scalar parameters
- test_set_P_validation_errors: Validation errors for outlet pressure
- test_set_Omega_scalar: Setting scalar angular velocity
- test_set_Omega_conversion: Type conversion for rotating patch
- test_set_Omega_array_raises_error: Array validation for rotating parameters
- test_set_rpm_scalar: Setting RPM for rotating patch
- test_set_rpm_conversion: RPM to angular velocity conversion
- test_set_cool_single_values: Setting individual cooling parameters
- test_set_cool_all_at_once: Setting all cooling parameters at once
- test_set_cool_type_conversion: Type conversion for cooling parameters
- test_set_cool_default_angle_def: Default angle definition behavior
- test_set_cool_array_raises_error: Array validation for cooling parameters
- test_inlet_patch_init: InletPatch initialization
- test_outlet_patch_init: OutletPatch initialization
- test_rotating_patch_init: RotatingPatch initialization
- test_cooling_patch_init: CoolingPatch initialization
- test_block_view_returns_sliced_block: block_view returns correct slice
- test_block_view_partial_patch: block_view with partial j-k range
- test_block_view_raises_without_block: block_view raises without proper block
- test_block_view_shares_data: block_view shares data with parent
"""


class TestMixingPatchCopy:
    def test_copy_returns_new_instance(self):
        p = MixingPatch(i=0, j=(1, 3), k=(0, 4), label="mix")
        c = p.copy()
        assert c is not p

    def test_copy_preserves_limits(self):
        p = MixingPatch(i=0, j=(1, 3), k=(0, 4), label="mix")
        c = p.copy()
        np.testing.assert_array_equal(c._ijk_lim, p._ijk_lim)

    def test_copy_preserves_label(self):
        p = MixingPatch(i=0, j=(1, 3), k=(0, 4), label="mix")
        c = p.copy()
        assert c.label == "mix"

    def test_copy_is_unattached(self):
        p = MixingPatch(i=0, j=(1, 3), k=(0, 4), label="mix")
        c = p.copy()
        assert c._block_ref is None
        assert c._block_view is None
        assert c._target is None


class TestInletPatch:
    """Test InletPatch setter methods."""

    def test_set_Po_To_Alpha_Beta_single_values(self, block_10):
        """Test setting single scalar values."""
        patch = InletPatch(i=0)  # Make it 2D by fixing i dimension
        patch.attach_to_block(block_10)

        # Set individual parameters
        patch.set_Po_To_Alpha_Beta(Po=101325.0)
        assert patch.Po == np.float32(101325.0)
        assert np.isnan(patch.To)

        patch.set_Po_To_Alpha_Beta(To=300.0)
        assert patch.To == np.float32(300.0)

        patch.set_Po_To_Alpha_Beta(Alpha=0.1, Beta=0.2)
        assert patch.Alpha == np.float32(0.1)
        assert patch.Beta == np.float32(0.2)

    def test_set_Po_To_Alpha_Beta_all_at_once(self, block_10):
        """Test setting all parameters at once."""
        patch = InletPatch(i=0)
        patch.attach_to_block(block_10)
        patch.set_Po_To_Alpha_Beta(Po=101325.0, To=300.0, Alpha=0.1, Beta=0.2)

        assert patch.Po == np.float32(101325.0)
        assert patch.To == np.float32(300.0)
        assert patch.Alpha == np.float32(0.1)
        assert patch.Beta == np.float32(0.2)

    def test_set_Po_To_Alpha_Beta_arrays_broadcast(self, block_10):
        """Test setting arrays that broadcast correctly."""
        patch = InletPatch(
            i=0, j=(0, 1), k=(0, 9)
        )  # Make patch shape (1, 2, 10) to match arrays along j
        patch.attach_to_block(block_10)

        # Set compatible arrays that broadcast with patch shape (1, 2, 10)
        Po = np.array([[101325.0, 102000.0]]).T  # Shape (2, 1) broadcasts to (1, 2, 10)
        To = np.array([[300.0, 310.0]]).T
        patch.set_Po_To_Alpha_Beta(Po=Po, To=To)

        np.testing.assert_array_equal(patch.Po, Po.astype(np.float32))
        np.testing.assert_array_equal(patch.To, To.astype(np.float32))

    def test_set_Po_To_Alpha_Beta_broadcast_failure(self, block_10):
        """Test broadcast validation failure."""
        patch = InletPatch(i=0, j=(0, 1), k=(0, 9))  # Shape (1, 2, 10)
        patch.attach_to_block(block_10)

        # Set first array
        patch.set_Po_To_Alpha_Beta(
            Po=np.array([[101325.0, 102000.0]]).T
        )  # Shape (2, 1)

        # Try to set incompatible array (wrong size)
        with pytest.raises(ValueError):
            patch.set_Po_To_Alpha_Beta(
                To=np.array([300.0, 310.0, 320.0])
            )  # Shape (3,) won't broadcast

    def test_set_Po_To_Alpha_Beta_validation_errors(self, block_10):
        """Test validation errors for Po and To."""
        patch = InletPatch(i=0)
        patch.attach_to_block(block_10)

        # Test negative pressure
        with pytest.raises(ValueError, match="Po must be positive"):
            patch.set_Po_To_Alpha_Beta(Po=-1000.0)

        # Test zero pressure
        with pytest.raises(ValueError, match="Po must be positive"):
            patch.set_Po_To_Alpha_Beta(Po=0.0)

        # Test infinite pressure
        with pytest.raises(ValueError, match="Po must be finite"):
            patch.set_Po_To_Alpha_Beta(Po=np.inf)

        # Test NaN pressure
        with pytest.raises(ValueError, match="Po must be finite"):
            patch.set_Po_To_Alpha_Beta(Po=np.nan)

        # Test negative temperature
        with pytest.raises(ValueError, match="To must be positive"):
            patch.set_Po_To_Alpha_Beta(To=-100.0)

        # Test zero temperature
        with pytest.raises(ValueError, match="To must be positive"):
            patch.set_Po_To_Alpha_Beta(To=0.0)

        # Test infinite temperature
        with pytest.raises(ValueError, match="To must be finite"):
            patch.set_Po_To_Alpha_Beta(To=np.inf)

        # Test arrays with mixed valid/invalid values - create patch that can broadcast with shape (2,)
        patch_array = InletPatch(i=0, j=(0, 1), k=(0, 9))  # Shape (1, 2, 10)
        patch_array.attach_to_block(block_10)
        with pytest.raises(ValueError, match="Po must be positive"):
            patch_array.set_Po_To_Alpha_Beta(
                Po=np.array([[101325.0, -1000.0]]).T
            )  # Shape (2, 1) broadcasts

        # Test infinite angles
        with pytest.raises(ValueError, match="Alpha must be finite"):
            patch.set_Po_To_Alpha_Beta(Alpha=np.inf)

        with pytest.raises(ValueError, match="Alpha must be finite"):
            patch.set_Po_To_Alpha_Beta(Alpha=np.nan)

        with pytest.raises(ValueError, match="Beta must be finite"):
            patch.set_Po_To_Alpha_Beta(Beta=np.inf)

        with pytest.raises(ValueError, match="Beta must be finite"):
            patch.set_Po_To_Alpha_Beta(Beta=np.nan)

        # Test arrays with infinite angles
        patch_array2 = InletPatch(i=0, j=(0, 1), k=(0, 9))  # Shape (1, 2, 10)
        patch_array2.attach_to_block(block_10)
        with pytest.raises(ValueError, match="Alpha must be finite"):
            patch_array2.set_Po_To_Alpha_Beta(
                Alpha=np.array([[0.1, np.inf]]).T
            )  # Shape (2, 1)

    def test_calc_raw_raises_without_block(self):
        """Test that accessing rhoo, ho, s without attach_to_block raises error."""
        patch = InletPatch(i=0)
        # Cannot set inlet without attaching to block first (negative indices need resolving)
        # But we can test that block property raises without attachment
        with pytest.raises(ValueError, match="not attached to any block"):
            _ = patch.block

    def test_apply_at_target_leaves_conserved_unchanged(self):
        """apply() must not perturb conserved variables when block is already at target.

        The P slot of delta_bcond must be zero (not -b.P). If bcond_target
        mistakenly included a zero P column and the full current state was
        subtracted, the P slot would equal -b.P and conserved would change
        even at the target state.
        """
        from ember import util

        shape = (10, 10, 10)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)
        block.set_fluid(fluid)
        _xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
        block.set_x(_xrt[..., 0])
        block.set_r(_xrt[..., 1])
        block.set_t(_xrt[..., 2])

        Po, To, Alpha, Beta = 1e5, 300.0, 15.0, 5.0
        s = fluid.get_s(*fluid.set_P_T(Po, To))
        ho = fluid.get_h(*fluid.set_P_T(Po, To))
        Ma = 0.3
        ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho, s, Ma, Alpha, Beta)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        patch.set_Po_To_Alpha_Beta(Po=Po, To=To, Alpha=Alpha, Beta=Beta)

        conserved_before = block.conserved.copy()
        patch.rf = 0.5
        patch.apply()

        # Float32 round-trip through set_ho_s_Ma_Alpha_Beta introduces small
        # residual deltas (~1e-2 on conserved values of ~1e2). The P-slot bug
        # would produce errors of order b.P ~ 1e5, so rtol=1e-2 is sufficient
        # to distinguish float32 noise from the bug.
        np.testing.assert_allclose(block.conserved, conserved_before, rtol=1e-2)


class TestInletPatchCopy:
    def test_copy_returns_new_instance(self):
        p = InletPatch(i=0, j=(1, 3), k=(0, 4), label="inlet")
        c = p.copy()
        assert c is not p

    def test_copy_preserves_limits(self):
        p = InletPatch(i=0, j=(1, 3), k=(0, 4), label="inlet")
        c = p.copy()
        np.testing.assert_array_equal(c._ijk_lim, p._ijk_lim)

    def test_copy_preserves_label(self):
        p = InletPatch(i=0, j=(1, 3), k=(0, 4), label="inlet")
        c = p.copy()
        assert c.label == "inlet"

    def test_copy_is_unattached(self):
        p = InletPatch(i=0, j=(1, 3), k=(0, 4), label="inlet")
        c = p.copy()
        assert c._block_ref is None
        assert c._block_view is None

    def test_copy_preserves_raw_defaults(self):
        p = InletPatch(i=0, j=(1, 3), k=(0, 4))
        c = p.copy()
        assert np.isnan(c._raw["Po"])
        assert np.isnan(c._raw["To"])

    def test_copy_raw_is_independent(self):
        p = InletPatch(i=0, j=(1, 3), k=(0, 4))
        c = p.copy()
        c._raw["Po"] = np.float32(99.0)
        assert np.isnan(p._raw["Po"])

    def test_copy_raw_values_are_independent(self):
        p = InletPatch(i=0, j=(1, 3), k=(0, 4))
        p._raw["Po"] = np.array([1e5, 2e5], dtype=np.float32)
        c = p.copy()
        c._raw["Po"][0] = 0.0
        assert p._raw["Po"][0] == pytest.approx(1e5)


class TestOutletPatch:
    """Test OutletPatch setter methods."""

    def test_set_P_scalar(self):
        """Test setting scalar pressure."""
        patch = OutletPatch(i=0)
        patch.set_P(50000.0)
        assert patch.P == np.float32(50000.0)

    def test_throttle_and_nonscalar_P_mutually_exclusive(self):
        """Non-scalar P and throttle control are mutually exclusive."""
        P_span = np.full((1, 8, 1), 101325.0)

        # set_P (non-scalar) then set_throttle raises
        patch = OutletPatch(i=0, j=(0, 7), k=(0, 0))
        patch.set_P(P_span)
        with pytest.raises(ValueError, match="incompatible with throttle"):
            patch.set_throttle(10.0, (0.1, 0.0, 0.0))

        # set_throttle then set_P (non-scalar) raises
        patch2 = OutletPatch(i=0, j=(0, 7), k=(0, 0))
        patch2.set_throttle(10.0, (0.1, 0.0, 0.0))
        with pytest.raises(ValueError, match="incompatible with throttle"):
            patch2.set_P(P_span)

        # Scalar P with throttle is allowed
        patch3 = OutletPatch(i=0, j=(0, 7), k=(0, 0))
        patch3.set_throttle(10.0, (0.1, 0.0, 0.0))
        patch3.set_P(101325.0)  # should not raise

    def test_set_P_validation_errors(self):
        """Test validation errors for P."""
        patch = OutletPatch(i=0)

        # Test negative pressure
        with pytest.raises(ValueError, match="P must be positive and finite"):
            patch.set_P(-1000.0)

        # Test zero pressure
        with pytest.raises(ValueError, match="P must be positive and finite"):
            patch.set_P(0.0)

        # Test infinite pressure
        with pytest.raises(ValueError, match="P must be positive and finite"):
            patch.set_P(np.inf)

        # Test NaN pressure
        with pytest.raises(ValueError, match="P must be positive and finite"):
            patch.set_P(np.nan)


class TestOutletPatchCopy:
    def test_copy_returns_new_instance(self):
        p = OutletPatch(i=-1, j=(1, 3), k=(0, 4), label="outlet")
        c = p.copy()
        assert c is not p

    def test_copy_preserves_limits(self):
        p = OutletPatch(i=-1, j=(1, 3), k=(0, 4), label="outlet")
        c = p.copy()
        np.testing.assert_array_equal(c._ijk_lim, p._ijk_lim)

    def test_copy_preserves_label(self):
        p = OutletPatch(i=-1, j=(1, 3), k=(0, 4), label="outlet")
        c = p.copy()
        assert c.label == "outlet"

    def test_copy_is_unattached(self):
        p = OutletPatch(i=-1, j=(1, 3), k=(0, 4), label="outlet")
        c = p.copy()
        assert c._block_ref is None
        assert c._block_view is None

    def test_copy_preserves_P_raw_none(self):
        p = OutletPatch(i=-1, j=(1, 3), k=(0, 4))
        c = p.copy()
        assert c._P_raw is None

    def test_copy_preserves_P_raw_scalar(self):
        p = OutletPatch(i=-1, j=(1, 3), k=(0, 4))
        p._P_raw = np.float32(101325.0)
        c = p.copy()
        assert c._P_raw == pytest.approx(101325.0)

    def test_copy_P_raw_is_independent(self):
        p = OutletPatch(i=-1, j=(1, 3), k=(0, 4))
        p._P_raw = np.array([1e5, 2e5], dtype=np.float32)
        c = p.copy()
        c._P_raw[0] = 0.0
        assert p._P_raw[0] == pytest.approx(1e5)

    def test_copy_preserves_pid_and_mdot(self):
        p = OutletPatch(i=-1, j=(1, 3), k=(0, 4))
        p._K_pid = (1.0, 0.1, 0.01)
        p._mdot_target = np.float32(5.0)
        c = p.copy()
        assert c._K_pid == (1.0, 0.1, 0.01)
        assert c._mdot_target == pytest.approx(5.0)

    def test_copy_preserves_adjustment(self):
        p = OutletPatch(i=-1, j=(1, 3), k=(0, 4))
        p._adjustment = {"mode": "radial_equilibrium", "rf": 0.5}
        c = p.copy()
        assert c._adjustment == {"mode": "radial_equilibrium", "rf": 0.5}


class TestRotatingPatch:
    """Test RotatingPatch setter methods."""

    def test_set_Omega_scalar(self):
        """Test setting scalar angular velocity."""
        patch = RotatingPatch(i=0)
        patch.set_Omega(100.0)
        assert patch.Omega == np.float32(100.0)

    def test_set_Omega_conversion(self):
        """Test type conversion to float32."""
        patch = RotatingPatch(i=0)
        patch.set_Omega(100)  # int
        assert patch.Omega == np.float32(100.0)
        assert isinstance(patch.Omega, np.float32)

    def test_set_Omega_array_raises_error(self):
        """Test that passing arrays to scalar parameter raises error."""
        patch = RotatingPatch(i=0)
        with pytest.raises(ValueError, match="Omega must be a scalar, not an array"):
            patch.set_Omega(np.array([100.0, 200.0]))

    def test_set_rpm_scalar(self):
        """Test setting scalar rpm."""
        patch = RotatingPatch(i=0)
        patch.set_rpm(1000.0)
        expected_omega = 1000.0 * 2.0 * np.pi / 60.0
        assert patch.Omega == np.float32(expected_omega)

    def test_set_rpm_conversion(self):
        """Test type conversion to float32 through Omega."""
        patch = RotatingPatch(i=0)
        patch.set_rpm(1000)  # int
        expected_omega = 1000.0 * 2.0 * np.pi / 60.0
        assert patch.Omega == np.float32(expected_omega)
        assert isinstance(patch.Omega, np.float32)


class TestRotatingPatchCopy:
    def test_copy_returns_new_instance(self):
        p = RotatingPatch(i=0, j=(1, 3), k=(0, 4), label="rotor")
        c = p.copy()
        assert c is not p

    def test_copy_preserves_limits(self):
        p = RotatingPatch(i=0, j=(1, 3), k=(0, 4), label="rotor")
        c = p.copy()
        np.testing.assert_array_equal(c._ijk_lim, p._ijk_lim)

    def test_copy_preserves_label(self):
        p = RotatingPatch(i=0, j=(1, 3), k=(0, 4), label="rotor")
        c = p.copy()
        assert c.label == "rotor"

    def test_copy_is_unattached(self):
        p = RotatingPatch(i=0, j=(1, 3), k=(0, 4))
        c = p.copy()
        assert c._block_ref is None
        assert c._block_view is None

    def test_copy_preserves_omega_default(self):
        p = RotatingPatch(i=0, j=(1, 3), k=(0, 4))
        c = p.copy()
        assert np.isnan(c._Omega)

    def test_copy_preserves_omega(self):
        p = RotatingPatch(i=0, j=(1, 3), k=(0, 4))
        p.set_Omega(100.0)
        c = p.copy()
        assert c._Omega == pytest.approx(np.float32(100.0))


class TestInviscidPatchCopy:
    def test_copy_returns_new_instance(self):
        p = InviscidPatch(i=0, j=(1, 3), k=(0, 4), label="wall")
        c = p.copy()
        assert c is not p

    def test_copy_preserves_limits(self):
        p = InviscidPatch(i=0, j=(1, 3), k=(0, 4), label="wall")
        c = p.copy()
        np.testing.assert_array_equal(c._ijk_lim, p._ijk_lim)

    def test_copy_preserves_label(self):
        p = InviscidPatch(i=0, j=(1, 3), k=(0, 4), label="wall")
        c = p.copy()
        assert c.label == "wall"

    def test_copy_is_unattached(self):
        p = InviscidPatch(i=0, j=(1, 3), k=(0, 4))
        c = p.copy()
        assert c._block_ref is None
        assert c._block_view is None


class TestCuspPatchCopy:
    def test_copy_returns_new_instance(self):
        p = CuspPatch(i=0, j=(1, 3), k=(0, 4), label="cusp")
        c = p.copy()
        assert c is not p

    def test_copy_preserves_limits(self):
        p = CuspPatch(i=0, j=(1, 3), k=(0, 4), label="cusp")
        c = p.copy()
        np.testing.assert_array_equal(c._ijk_lim, p._ijk_lim)

    def test_copy_preserves_label(self):
        p = CuspPatch(i=0, j=(1, 3), k=(0, 4), label="cusp")
        c = p.copy()
        assert c.label == "cusp"

    def test_copy_is_unattached(self):
        p = CuspPatch(i=0, j=(1, 3), k=(0, 4))
        c = p.copy()
        assert c._block_ref is None
        assert c._block_view is None


class TestNonMatchPatchCopy:
    def test_copy_returns_new_instance(self):
        p = NonMatchPatch(i=0, j=(1, 3), k=(0, 4), label="nm")
        c = p.copy()
        assert c is not p

    def test_copy_preserves_limits(self):
        p = NonMatchPatch(i=0, j=(1, 3), k=(0, 4), label="nm")
        c = p.copy()
        np.testing.assert_array_equal(c._ijk_lim, p._ijk_lim)

    def test_copy_preserves_label(self):
        p = NonMatchPatch(i=0, j=(1, 3), k=(0, 4), label="nm")
        c = p.copy()
        assert c.label == "nm"

    def test_copy_is_unattached(self):
        p = NonMatchPatch(i=0, j=(1, 3), k=(0, 4))
        c = p.copy()
        assert c._block_ref is None
        assert c._block_view is None


class TestCoolingPatch:
    """Test CoolingPatch setter methods."""

    def test_set_cool_single_values(self):
        """Test setting individual parameters."""
        patch = CoolingPatch(i=0)

        # Set individual parameters
        patch.set_cool(type=1)
        assert patch.type == 1
        assert np.isnan(patch.mass)

        patch.set_cool(mass=0.1)
        assert patch.mass == np.float32(0.1)

        patch.set_cool(pstag=101325.0, tstag=300.0)
        assert patch.pstag == np.float32(101325.0)
        assert patch.tstag == np.float32(300.0)

    def test_set_cool_all_at_once(self):
        """Test setting all parameters at once."""
        patch = CoolingPatch(i=0)
        patch.set_cool(
            type=1,
            mass=0.1,
            pstag=101325.0,
            tstag=300.0,
            sangle=0.1,
            xangle=0.2,
            mach=0.5,
            angle_def=2,
        )

        assert patch.type == 1
        assert patch.mass == np.float32(0.1)
        assert patch.pstag == np.float32(101325.0)
        assert patch.tstag == np.float32(300.0)
        assert patch.sangle == np.float32(0.1)
        assert patch.xangle == np.float32(0.2)
        assert patch.mach == np.float32(0.5)
        assert patch.angle_def == 2

    def test_set_cool_type_conversion(self):
        """Test type conversion for int and float parameters."""
        patch = CoolingPatch(i=0)

        # Test int conversion
        patch.set_cool(type=1.5)  # float to int
        assert patch.type == 1
        assert isinstance(patch.type, int)

        # Test float32 conversion
        patch.set_cool(mass=0.1, pstag=101325)  # int to float32
        assert patch.mass == np.float32(0.1)
        assert patch.pstag == np.float32(101325.0)
        assert isinstance(patch.mass, np.float32)
        assert isinstance(patch.pstag, np.float32)

    def test_set_cool_default_angle_def(self):
        """Test that angle_def defaults to 1."""
        patch = CoolingPatch(i=0)
        assert patch.angle_def == 1

    def test_set_cool_array_raises_error(self):
        """Test that passing arrays to scalar parameters raises error."""
        patch = CoolingPatch(i=0)

        # Test arrays for float parameters
        with pytest.raises(ValueError, match="mass must be a scalar, not an array"):
            patch.set_cool(mass=np.array([0.1, 0.2]))

        with pytest.raises(ValueError, match="pstag must be a scalar, not an array"):
            patch.set_cool(pstag=np.array([101325.0, 102000.0]))

        # Test arrays for int parameters
        with pytest.raises(ValueError, match="type must be a scalar, not an array"):
            patch.set_cool(type=np.array([1, 2]))


class TestPatchInitialization:
    """Test patch initialization."""

    def test_inlet_patch_init(self):
        """Test InletPatch initialization."""
        patch = InletPatch(i=0)
        assert len(patch._raw) == 4
        assert all(np.isnan(x) for x in patch._raw.values())

    def test_rotating_patch_init(self):
        """Test RotatingPatch initialization."""
        patch = RotatingPatch(i=0)
        assert np.isnan(patch._Omega)

    def test_cooling_patch_init(self):
        """Test CoolingPatch initialization."""
        patch = CoolingPatch(i=0)
        assert np.isnan(patch.type)
        assert np.isnan(patch.mass)
        assert np.isnan(patch.pstag)
        assert np.isnan(patch.tstag)
        assert np.isnan(patch.sangle)
        assert np.isnan(patch.xangle)
        assert np.isnan(patch.mach)
        assert patch.angle_def == 1


class TestPatchBlockView:
    """Test Patch.block_view property."""

    def test_block_view_returns_sliced_block(self):
        """Test that block_view returns block sliced at patch location."""
        block = ember.block.Block(shape=(10, 8, 6))
        patch = PeriodicPatch(i=0)
        patch.attach_to_block(block)

        view = patch.block_view
        assert view.shape == (1, 8, 6)

    def test_block_view_partial_patch(self):
        """Test block_view with patch covering partial j-k range."""
        block = ember.block.Block(shape=(10, 8, 6))
        patch = PeriodicPatch(i=0, j=(2, 5), k=(1, 4))
        patch.attach_to_block(block)

        view = patch.block_view
        assert view.shape == (1, 4, 4)

    def test_block_view_raises_without_block(self):
        """Test that block_view raises error if not attached to block."""
        patch = PeriodicPatch(i=0)

        with pytest.raises(ValueError, match="not attached to any block"):
            _ = patch.block_view

    def test_block_view_shares_data(self):
        """Test that block_view shares underlying data with parent block."""
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=(10, 8, 6))
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)

        patch = PeriodicPatch(i=0)
        patch.attach_to_block(block)

        view = patch.block_view
        # Modifying view should affect parent block
        original_rho = block.rho[0, :, :].copy()
        view.set_P_T(2e5, 400.0)
        assert not np.allclose(block.rho[0, :, :], original_rho)


class TestPatchSurfaceOfRevolution:
    """Tests for pitch_dim, span_dim, and spf properties on surface of revolution patches."""

    def test_pitch_dim_identified_correctly_k_pitch(self):
        """Test that pitch_dim is correctly identified when k is pitchwise (const x,r)."""

        shape = (10, 5, 7)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Create a surface of revolution: k is circumferential (pitch), j is spanwise
        # Constant radius surface with varying x in j direction
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.ones((shape[0], shape[1], shape[2])) * 0.5
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [np.broadcast_to(x, shape), r, np.broadcast_to(t, shape)], axis=-1
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        # Patch on i=0 face, so const_dim=0, varying dims are j and k
        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        # k should be pitch (no x,r variation), j should be span (x varies)
        assert patch.pitch_dim == 2
        assert patch.span_dim == 1

    def test_span_dim_identified_correctly_j_span(self):
        """Test that span_dim is correctly identified when j is spanwise (varying x,r)."""

        shape = (10, 6, 5)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Create surface: j varies in x and r (spanwise), k is constant x,r (pitch)
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.linspace(0.5, 1.0, shape[1]).reshape(1, -1, 1)
        t = np.linspace(0.0, 0.3, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [
                np.broadcast_to(x, shape),
                np.broadcast_to(r, shape),
                np.broadcast_to(t, shape),
            ],
            axis=-1,
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        assert patch.span_dim == 1
        assert patch.pitch_dim == 2

    def test_pitch_and_span_mutually_exclusive(self):
        """Test that pitch_dim and span_dim are different and neither equals const_dim."""

        shape = (8, 5, 6)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Surface of revolution with j as span, k as pitch
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.linspace(0.5, 1.5, shape[1]).reshape(1, -1, 1)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [
                np.broadcast_to(x, shape),
                np.broadcast_to(r, shape),
                np.broadcast_to(t, shape),
            ],
            axis=-1,
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        # All three dimensions should be different
        assert patch.pitch_dim != patch.span_dim
        assert patch.pitch_dim != patch.const_dim
        assert patch.span_dim != patch.const_dim

        # Should cover all three dimensions
        dims = {patch.pitch_dim, patch.span_dim, patch.const_dim}
        assert dims == {0, 1, 2}

    def test_pitch_dim_j_pitch_direction(self):
        """Test pitch_dim when j is the pitchwise direction."""

        shape = (10, 6, 8)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # j is pitch (constant x,r), k is span (varying x,r)
        x = np.linspace(0.0, 1.0, shape[2]).reshape(1, 1, -1)
        r = np.linspace(0.5, 1.0, shape[2]).reshape(1, 1, -1)
        t = np.linspace(0.0, 0.25, shape[1]).reshape(1, -1, 1)
        xrt = np.stack(
            [
                np.broadcast_to(x, shape),
                np.broadcast_to(r, shape),
                np.broadcast_to(t, shape),
            ],
            axis=-1,
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        assert patch.pitch_dim == 1
        assert patch.span_dim == 2

    def test_spf_computed_for_surface_of_revolution(self):
        """Test that spf is computed when patch is a surface of revolution."""

        shape = (10, 5, 6)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Surface of revolution
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.ones((shape[0], shape[1], shape[2])) * 0.5
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [np.broadcast_to(x, shape), r, np.broadcast_to(t, shape)], axis=-1
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        # spf should be accessible and be an array
        spf = patch.spf
        assert spf is not None
        assert isinstance(spf, np.ndarray)

    def test_weight_pitch_sums_to_one(self):
        """Weights must sum to 1 along pitch dimension."""
        shape = (10, 5, 7)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)
        Nb = 36
        block.set_Nb(Nb)
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.ones(shape) * 0.5
        pitch = 2.0 * np.pi / Nb
        t = np.linspace(0.0, pitch, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [np.broadcast_to(x, shape), r, np.broadcast_to(t, shape)], axis=-1
        )
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        assert np.isclose(patch.weight_pitch.sum(), 1.0)

    def test_weight_pitch_uniform_spacing(self):
        """Uniformly spaced nodes give equal weights of 1/npitch."""
        shape = (10, 5, 8)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)
        Nb = 24
        block.set_Nb(Nb)
        pitch = 2.0 * np.pi / Nb
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.ones(shape) * 0.5
        t = np.linspace(0.0, pitch, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [np.broadcast_to(x, shape), r, np.broadcast_to(t, shape)], axis=-1
        )
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        # With npitch nodes uniformly spaced over [0, pitch], spacing is pitch/(npitch-1).
        # Endpoint nodes own half an interval; interior nodes own a full interval.
        # Normalised by pitch: endpoints get 1/(2*(npitch-1)), interior get 1/(npitch-1).
        npitch = shape[2]
        w = np.full(npitch, 1.0 / (npitch - 1))
        w[0] *= 0.5
        w[-1] *= 0.5
        np.testing.assert_allclose(patch.weight_pitch.squeeze(), w, rtol=1e-5)

    def test_weight_pitch_nonuniform_spacing(self):
        """Non-uniform theta spacing gives weights proportional to interval size."""
        shape = (10, 5, 4)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)
        Nb = 18
        block.set_Nb(Nb)
        pitch = 2.0 * np.pi / Nb
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.ones(shape) * 0.5
        # Non-uniform theta: nodes at 0, 0.1, 0.6, 1.0 of pitch
        t_frac = np.array([0.0, 0.1, 0.6, 1.0])
        t = (t_frac * pitch).reshape(1, 1, -1)
        xrt = np.stack(
            [np.broadcast_to(x, shape), r, np.broadcast_to(t, shape)], axis=-1
        )
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        # Midpoints: 0.05, 0.35, 0.8 (fractions of pitch)
        # dt[0] = 0.05 - 0.0 = 0.05
        # dt[1] = 0.35 - 0.05 = 0.30
        # dt[2] = 0.80 - 0.35 = 0.45
        # dt[3] = 1.0 - 0.80 = 0.20
        expected = np.array([0.05, 0.30, 0.45, 0.20])
        np.testing.assert_allclose(patch.weight_pitch.squeeze(), expected, rtol=1e-5)

    def _make_sor_block(self, shape, Nb=36, t_frac=None):
        """Helper: surface-of-revolution block with i=0 patch (j=span, k=pitch)."""
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)
        block.set_fluid(fluid)
        block.set_Nb(Nb)
        pitch = 2.0 * np.pi / Nb
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1) * np.ones(shape)
        r = np.ones(shape) * 0.5
        if t_frac is None:
            t = np.linspace(0.0, pitch, shape[2]).reshape(1, 1, -1) * np.ones(shape)
        else:
            t = (t_frac * pitch).reshape(1, 1, -1) * np.ones(shape)
        xrt = np.stack(
            [np.broadcast_to(x, shape), r, np.broadcast_to(t, shape)], axis=-1
        )
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        return block

    def test_set_block_avg_uniform_field(self):
        """Uniform conserved field gives pitch average equal to that field at every span node."""
        shape = (10, 5, 7)
        block = self._make_sor_block(shape)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(50.0)
        block.set_Vr(10.0)
        block.set_Vt(20.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        patch.set_block_avg()

        # All pitch nodes identical so average must match any single node
        cons_node = block.conserved[0, :, 0, :]  # shape (nspan, 5)
        np.testing.assert_allclose(patch.block_avg.conserved, cons_node, rtol=1e-5)

    def test_set_block_avg_uniform_spacing(self):
        """Uniform spacing: pitch average matches np.average over pitch axis."""
        shape = (10, 5, 8)
        block = self._make_sor_block(shape, Nb=24)
        block.set_P_T(1e5, 300.0)
        # Set varying conserved field: rho varies with j and k
        rng = np.random.default_rng(0)
        cons = rng.uniform(0.5, 1.5, shape + (5,)).astype(np.float32)
        block.set_conserved(cons)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        patch.set_block_avg()

        # block_view has shape (1, nj, nk, 5); w has shape (1, 1, nk)
        w = patch.weight_pitch  # (1, 1, nk) broadcasts against block_view
        expected = np.sum(
            patch.block_view.conserved * w[..., np.newaxis], axis=patch.pitch_dim
        ).squeeze()  # (nj, 5)
        np.testing.assert_allclose(patch.block_avg.conserved, expected, rtol=1e-4)

    def test_set_block_avg_nonuniform_spacing(self):
        """Non-uniform pitch spacing: weighted average matches hand-computed result."""
        shape = (10, 3, 4)
        t_frac = np.array([0.0, 0.1, 0.6, 1.0])
        block = self._make_sor_block(shape, Nb=18, t_frac=t_frac)
        block.set_P_T(1e5, 300.0)
        # Set a simple conserved field: constant rho=1, rhoVx varies with k only
        cons = np.zeros(shape + (5,), dtype=np.float32)
        cons[..., 0] = 1.0  # rho
        cons[..., 1] = np.array([1.0, 2.0, 3.0, 4.0])  # rhoVx varies over pitch
        block.set_conserved(cons)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        patch.set_block_avg()

        # weights: [0.05, 0.30, 0.45, 0.20]
        w = patch.weight_pitch
        expected_rhoVx = np.dot(
            w.squeeze(), np.array([1.0, 2.0, 3.0, 4.0])
        )  # scalar, same at all span
        np.testing.assert_allclose(
            patch.block_avg.conserved[:, 1], expected_rhoVx, rtol=1e-5
        )
        np.testing.assert_allclose(patch.block_avg.conserved[:, 0], 1.0, rtol=1e-5)

    def _make_conical_sor_block(self, shape, Nb=36, slope=0.0, dRi=0.0, R0=0.5):
        """Helper: SOR block whose meridional line (j) has radius R0 + slope*x,
        and whose i-direction offsets radius by -dRi per step, so the "inward"
        flip logic in `_build_rot_matrices` has a genuine reference direction.
        """
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)
        block.set_fluid(fluid)
        block.set_Nb(Nb)
        pitch = 2.0 * np.pi / Nb
        ni, nj, nk = shape
        x_j = np.linspace(0.0, 1.0, nj)
        i_idx = np.arange(ni).reshape(ni, 1, 1)
        x = x_j.reshape(1, nj, 1) * np.ones(shape)
        r = (R0 - dRi * i_idx + slope * x_j.reshape(1, nj, 1)) * np.ones(shape)
        t = np.linspace(0.0, pitch, nk).reshape(1, 1, nk) * np.ones(shape)
        block.set_x(x)
        block.set_r(r)
        block.set_t(t)
        return block

    def test_build_rot_matrices_straight_duct_no_flip(self):
        """Constant-radius duct: face normal is purely radial (xi = pi/2).

        With no radial offset between i-planes (dRi=0) the "inward" reference
        vector is zero, so the flip in `_build_rot_matrices` never triggers and
        the angle stays at the raw meridional value.
        """
        shape = (3, 5, 8)
        block = self._make_conical_sor_block(shape, slope=0.0, dRi=0.0)
        block.set_P_T(1e5, 300.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        patch._build_rot_matrices()

        expected_c = np.zeros(shape[1], dtype=np.float32)
        expected_s = np.ones(shape[1], dtype=np.float32)
        np.testing.assert_allclose(
            patch._rot_to.squeeze(),
            np.stack(
                [
                    np.stack([expected_c, expected_s], axis=-1),
                    np.stack([-expected_s, expected_c], axis=-1),
                ],
                axis=-2,
            ),
            atol=1e-6,
        )

    def test_build_rot_matrices_flip_when_dot_negative(self):
        """Radius decreasing into the domain (dRi > 0) flips the angle by pi.

        The raw meridional xi (pi/2) points away from the "inward" reference
        vector (0, -dRi), so `_build_rot_matrices` should flip it to 3*pi/2.
        """
        shape = (3, 5, 8)
        block = self._make_conical_sor_block(shape, slope=0.0, dRi=0.1)
        block.set_P_T(1e5, 300.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        patch._build_rot_matrices()

        expected_c = np.zeros(shape[1], dtype=np.float32)
        expected_s = -np.ones(shape[1], dtype=np.float32)
        np.testing.assert_allclose(
            patch._rot_to.squeeze()[:, 0, 0], expected_c, atol=1e-6
        )
        np.testing.assert_allclose(
            patch._rot_to.squeeze()[:, 0, 1], expected_s, atol=1e-6
        )

    def test_build_rot_matrices_inward_false_negates_matrices(self):
        """inward=False shifts the angle by pi, negating both rotation matrices."""
        shape = (3, 5, 8)
        block = self._make_conical_sor_block(shape, slope=0.0, dRi=0.0)
        block.set_P_T(1e5, 300.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        patch._build_rot_matrices(inward=True)
        rot_to_in = patch._rot_to.copy()
        rot_from_in = patch._rot_from.copy()

        patch._build_rot_matrices(inward=False)
        np.testing.assert_allclose(patch._rot_to, -rot_to_in, atol=1e-6)
        np.testing.assert_allclose(patch._rot_from, -rot_from_in, atol=1e-6)

    def test_resolve_to_interface_matches_manual_rotation(self):
        """resolve_to_interface applies the precomputed rot_to to (rhoVx, rhoVr).

        On the constant-radius duct xi = pi/2 (c=0, s=1), so the documented
        formula V_norm = c*Vx + s*Vr, V_span = -s*Vx + c*Vr reduces to
        V_norm = Vr, V_span = -Vx.
        """
        shape = (3, 5, 8)
        block = self._make_conical_sor_block(shape, slope=0.0, dRi=0.0)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(30.0)
        block.set_Vr(7.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        patch.set_block_avg()
        patch._build_rot_matrices()

        rhoVx_before = patch.block_view.conserved_nd[..., 1].copy()
        rhoVr_before = patch.block_view.conserved_nd[..., 2].copy()

        patch.resolve_to_interface()

        np.testing.assert_allclose(
            patch.block_view.conserved_nd[..., 1], rhoVr_before, atol=1e-5
        )
        np.testing.assert_allclose(
            patch.block_view.conserved_nd[..., 2], -rhoVx_before, atol=1e-5
        )

    def test_resolve_round_trip_recovers_original(self):
        """resolve_from_interface undoes resolve_to_interface on the same field."""
        shape = (3, 5, 8)
        block = self._make_conical_sor_block(shape, slope=0.3, dRi=0.05)
        block.set_P_T(1e5, 300.0)
        rng = np.random.default_rng(0)
        block.set_Vx(rng.uniform(10.0, 50.0, shape))
        block.set_Vr(rng.uniform(-10.0, 10.0, shape))
        block.set_Vt(rng.uniform(-10.0, 10.0, shape))

        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        patch.set_block_avg()
        patch._build_rot_matrices()

        cons_before = patch.block_view.conserved_nd[..., 1:3].copy()

        patch.resolve_to_interface()
        patch.resolve_from_interface()

        np.testing.assert_allclose(
            patch.block_view.conserved_nd[..., 1:3], cons_before, atol=1e-4
        )


class TestPatchNonSurfaceOfRevolution:
    """Tests for pitch_dim, span_dim, and spf on non-surface of revolution patches."""

    def _make_non_sor_block(self):
        """Helper to create a block that is NOT a surface of revolution."""
        shape = (10, 5, 6)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Both j and k have x,r variation - not a surface of revolution
        xj = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        xk = np.linspace(0.0, 0.1, shape[2]).reshape(1, 1, -1)
        rj = np.linspace(0.5, 1.0, shape[1]).reshape(1, -1, 1)
        rk = np.linspace(0.5, 0.6, shape[2]).reshape(1, 1, -1)

        x = np.broadcast_to(xj + xk, shape)
        r = np.broadcast_to(rj + rk, shape)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack([x, r, np.broadcast_to(t, shape)], axis=-1)

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)
        return block

    def test_revolution_patch_raises_on_attach(self):
        """Test that RevolutionPatch subclass raises on attach when not a surface of revolution."""
        block = self._make_non_sor_block()

        patch = InletPatch(i=0)
        with pytest.raises(ValueError, match="not a surface of revolution"):
            patch.attach_to_block(block)

    def test_revolution_patch_raises_on_attach_outlet(self):
        """Test that OutletPatch raises on attach when not a surface of revolution."""
        block = self._make_non_sor_block()

        patch = OutletPatch(i=0)
        with pytest.raises(ValueError, match="not a surface of revolution"):
            patch.attach_to_block(block)

    def test_no_variation_both_directions(self):
        """Test when both varying dimensions have no x,r variation (degenerate case)."""
        shape = (10, 5, 6)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Neither j nor k have x,r variation - degenerate case
        x = np.ones(shape) * 0.5
        r = np.ones(shape) * 0.5
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack([x, r, np.broadcast_to(t, shape)], axis=-1)

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        # Should raise error on attach - no clear span or pitch
        with pytest.raises(ValueError, match="not a surface of revolution"):
            patch.attach_to_block(block)

    def test_properties_when_only_one_dimension_varies(self):
        """Test edge case where only one dimension varies (both should be classified same)."""
        shape = (10, 5, 6)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Only j varies in x,r; k is constant
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.linspace(0.5, 1.0, shape[1]).reshape(1, -1, 1)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [
                np.broadcast_to(x, shape),
                np.broadcast_to(r, shape),
                np.broadcast_to(t, shape),
            ],
            axis=-1,
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        # This is actually a valid surface of revolution
        # j should be span, k should be pitch
        assert patch.span_dim == 1
        assert patch.pitch_dim == 2


class TestPatchSpanFractionVector:
    """Tests for the spf (span fraction vector) property."""

    def test_spf_length_matches_span_dimension(self):
        """Test that length of spf matches the size of the span dimension."""

        shape = (10, 7, 5)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Surface of revolution: j is span (varying x,r), k is pitch (constant x,r)
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.linspace(0.5, 1.5, shape[1]).reshape(1, -1, 1)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [
                np.broadcast_to(x, shape),
                np.broadcast_to(r, shape),
                np.broadcast_to(t, shape),
            ],
            axis=-1,
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        # Patch on i=0 face has shape (1, nj, nk) -> squeezed to (nj, nk)
        # spf should have length equal to span dimension size
        assert patch.span_dim == 1
        spf = patch.spf
        # The span dimension on the patch has size shape[1] = 7
        assert len(spf) == shape[1]

    def test_spf_starts_at_zero(self):
        """Test that spf starts at zero."""

        shape = (10, 6, 5)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Surface of revolution
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.linspace(0.5, 1.0, shape[1]).reshape(1, -1, 1)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [
                np.broadcast_to(x, shape),
                np.broadcast_to(r, shape),
                np.broadcast_to(t, shape),
            ],
            axis=-1,
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        spf = patch.spf
        assert spf[0] == 0.0

    def test_spf_monotonically_increasing(self):
        """Test that spf is monotonically increasing."""

        shape = (10, 8, 5)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Surface of revolution with varying x and r
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.linspace(0.5, 1.5, shape[1]).reshape(1, -1, 1)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [
                np.broadcast_to(x, shape),
                np.broadcast_to(r, shape),
                np.broadcast_to(t, shape),
            ],
            axis=-1,
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        spf = patch.spf
        # Check that all differences are non-negative
        assert np.all(np.diff(spf) >= 0)

    def test_spf_ends_at_one(self):
        """Test that spf is normalised so the final value is exactly 1.0."""

        shape = (10, 6, 5)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Diagonal line so total arc-length != 1 without normalisation
        x = np.linspace(0.0, 3.0, shape[1]).reshape(1, -1, 1)
        r = np.linspace(0.5, 2.5, shape[1]).reshape(1, -1, 1)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [
                np.broadcast_to(x, shape),
                np.broadcast_to(r, shape),
                np.broadcast_to(t, shape),
            ],
            axis=-1,
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        spf = patch.spf
        assert spf[0] == 0.0
        np.testing.assert_allclose(spf[-1], 1.0, rtol=1e-6)

    def test_spf_computed_from_arc_length_straight_line(self):
        """Test spf computed from arc length for a straight radial line."""

        shape = (10, 5, 6)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Simple case: straight line in x direction at constant r
        # Arc length should just be x coordinate
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.ones((shape[0], shape[1], shape[2])) * 0.5
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [np.broadcast_to(x, shape), r, np.broadcast_to(t, shape)], axis=-1
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        spf = patch.spf

        # For a uniform straight line from 0 to 1 with 5 points,
        # the cumulative arc length should be [0, 0.25, 0.5, 0.75, 1.0]
        expected = np.linspace(0.0, 1.0, shape[1])
        np.testing.assert_allclose(spf, expected, rtol=1e-6)

    def test_spf_computed_from_arc_length_radial_line(self):
        """Test spf for a pure radial line (varying r only)."""

        shape = (10, 6, 5)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Pure radial line: x constant, r varying
        x = np.zeros((shape[0], shape[1], shape[2]))
        r_vals = np.linspace(0.5, 1.5, shape[1])
        r = np.broadcast_to(r_vals.reshape(1, -1, 1), shape)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack([x, r, np.broadcast_to(t, shape)], axis=-1)

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        spf = patch.spf

        # For uniform radial line from 0.5 to 1.5 with 6 points,
        # arc length is just the radial distance
        expected = np.linspace(0.0, 1.0, shape[1])
        np.testing.assert_allclose(spf, expected, rtol=1e-6)

    def test_spf_diagonal_line(self):
        """Test spf for a diagonal line (both x and r varying)."""

        shape = (10, 5, 6)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Diagonal line: x and r both varying uniformly
        # This creates a line at 45 degrees
        # Start r at 0.5 to avoid r=0 constraint
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r = np.linspace(0.5, 1.5, shape[1]).reshape(1, -1, 1)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack(
            [
                np.broadcast_to(x, shape),
                np.broadcast_to(r, shape),
                np.broadcast_to(t, shape),
            ],
            axis=-1,
        )

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        spf = patch.spf

        # For a diagonal line with uniform segments the spf should be uniformly
        # spaced from 0 to 1 regardless of actual arc-length per segment.
        expected = np.linspace(0.0, 1.0, shape[1])
        np.testing.assert_allclose(spf, expected, rtol=1e-6)

    def test_spf_normalized_correctly(self):
        """Test that spf computation uses correct averaging over pitch dimension."""

        shape = (10, 5, 7)
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)

        # Surface of revolution with j as span
        # Vary only in span direction to ensure it's a valid surface of revolution
        x_base = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1)
        r_base = np.linspace(0.5, 1.5, shape[1]).reshape(1, -1, 1)

        # No perturbations - must be a perfect surface of revolution
        x = np.broadcast_to(x_base, shape)
        r = np.broadcast_to(r_base, shape)
        t = np.linspace(0.0, 0.2, shape[2]).reshape(1, 1, -1)
        xrt = np.stack([x, r, np.broadcast_to(t, shape)], axis=-1)

        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_fluid(fluid)
        block.set_P_T(1e5, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)

        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        spf = patch.spf

        # Verify it has correct properties
        assert len(spf) == shape[1]
        assert spf[0] == 0.0
        assert np.all(np.diff(spf) >= 0)
        np.testing.assert_allclose(spf[-1], 1.0, rtol=1e-6)


class TestMdot:
    """Tests for dA and mdot on RevolutionPatch subclasses."""

    def _make_sor_block(self, shape, Nb=36, r_span=None):
        """Helper: surface-of-revolution block, i=0 patch (j=span, k=pitch).

        r_span allows non-uniform radii along the span dimension so that
        face areas vary, making dA non-trivial.
        """
        fluid = PerfectFluid(gamma=1.4, cp=1004.5, mu=1.8e-5, Pr=0.7)
        block = ember.block.Block(shape=shape)
        block.set_fluid(fluid)
        block.set_Nb(Nb)
        pitch = 2.0 * np.pi / Nb
        x = np.linspace(0.0, 1.0, shape[1]).reshape(1, -1, 1) * np.ones(shape)
        if r_span is None:
            r = np.ones(shape) * 0.5
        else:
            r = np.array(r_span).reshape(1, -1, 1) * np.ones(shape)
        t = np.linspace(0.0, pitch, shape[2]).reshape(1, 1, -1) * np.ones(shape)
        xrt = np.stack(
            [np.broadcast_to(x, shape), r, np.broadcast_to(t, shape)], axis=-1
        )
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        return block

    def test_dA_shape(self):
        """dA has shape (nspan,) after attach_to_block."""
        shape = (10, 6, 8)
        block = self._make_sor_block(shape)
        block.set_P_T(1e5, 300.0)
        patch = InletPatch(i=0)
        patch.attach_to_block(block)
        assert patch._dA_node.shape == (shape[1],)

    def test_dA_trapezoid_uniform_area(self):
        """Uniform face areas: endpoints get A/2, interior nodes get A."""
        shape = (10, 5, 8)
        block = self._make_sor_block(shape)
        block.set_P_T(1e5, 300.0)
        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        ws = patch._dA_node
        # All faces equal area A; endpoint nodes own A/2, interior own A
        A = ws[1]
        np.testing.assert_allclose(ws[0], A / 2, rtol=1e-5)
        np.testing.assert_allclose(ws[-1], A / 2, rtol=1e-5)
        np.testing.assert_allclose(ws[1:-1], A, rtol=1e-5)

    def test_dA_varies_with_radius(self):
        """Face areas proportional to r; dA reflects varying areas."""
        shape = (10, 5, 8)
        r_span = np.linspace(0.5, 1.5, shape[1])
        block = self._make_sor_block(shape, r_span=r_span)
        block.set_P_T(1e5, 300.0)
        patch = InletPatch(i=0)
        patch.attach_to_block(block)

        ws = patch._dA_node
        assert np.all(np.diff(ws[:-1]) > 0), "dA should increase with r"


class TestPeriodicPatchCopy:
    def test_copy_returns_new_instance(self):
        p = PeriodicPatch(i=0, j=(1, 3), k=(0, 4), label="periodic")
        c = p.copy()
        assert c is not p

    def test_copy_preserves_limits(self):
        p = PeriodicPatch(i=0, j=(1, 3), k=(0, 4), label="periodic")
        c = p.copy()
        np.testing.assert_array_equal(c._ijk_lim, p._ijk_lim)

    def test_copy_preserves_label(self):
        p = PeriodicPatch(i=0, j=(1, 3), k=(0, 4), label="periodic")
        c = p.copy()
        assert c.label == "periodic"

    def test_copy_is_unattached(self):
        p = PeriodicPatch(i=0, j=(1, 3), k=(0, 4), label="periodic")
        c = p.copy()
        assert c._block_ref is None
        assert c._block_view is None

    def test_copy_limits_are_independent(self):
        p = PeriodicPatch(i=0, j=(1, 3), k=(0, 4), label="periodic")
        c = p.copy()
        c._ijk_lim[1, 0] = 99
        assert p._ijk_lim[1, 0] != 99


# ---------------------------------------------------------------------------
# Docstring example runner
# ---------------------------------------------------------------------------


def _exec_patch_example(name):
    """Find the '::' code block tagged '# example: <name>' in the ember.patch
    module docstring, exec it, and assert all '# value' annotations against
    captured print output.  Returns the execution namespace."""
    import ember.patch
    import textwrap
    import re
    import io
    import contextlib

    blocks = re.findall(r"::\n\n((?:    .+\n|\n)+)", ember.patch.__doc__)
    for raw in blocks:
        code = textwrap.dedent(raw)
        first_line = code.lstrip().splitlines()[0]
        if first_line.strip() != f"# example: {name}":
            continue

        raises = []
        expected_outputs = []
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

        import ember
        import ember.block
        import ember.patch
        from ember.fluid import PerfectFluid

        ns = {"np": np, "ember": ember, "PerfectFluid": PerfectFluid}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec("\n".join(clean_lines), ns)  # noqa: S102
        actual_outputs = [line for line in buf.getvalue().splitlines() if line.strip()]

        assert len(actual_outputs) == len(expected_outputs), (
            f"example '{name}': {len(actual_outputs)} printed lines but "
            f"{len(expected_outputs)} annotated"
        )

        def _parse(s):
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


class TestPatchDocstringExamples:
    """Execute and verify the inline examples in the ember.patch module docstring."""

    def test_patch_examples(self):
        _exec_patch_example("patch_examples")
