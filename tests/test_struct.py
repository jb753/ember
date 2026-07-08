"""Tests for StructuredData class (ember.struct).

Module tested: ember.struct

Test cases:
- test_shape_and_dimension_properties: Shape and dimension properties
- test_increment_data_slice: _increment_data method for in-place increments with slice
- test_increment_data_full_array: _increment_data method for full array increments
- test_increment_data_single_variable: _increment_data with a single variable
- test_increment_data_consecutive_variables: _increment_data with consecutive variables
- test_increment_data_non_consecutive_variables_error: Error for non-consecutive variables
- test_transpose_keeps_variable_axis_last_and_reorders_spatial_axes: Transpose functionality
- test_transpose_invalid_axes_raises: Invalid axes for transpose
- test_flip_negative_and_positive_axes_match_numpy_flip: Flip functionality
- test_flip_invalid_axis_raises: Invalid axis for flip
- test_squeeze_does_not_remove_variable_axis_for_nvar_one: Squeeze functionality
- test_flat_shape_and_values_equal_numpy_reshape: Flat reshape functionality
- test_getitem_scalar_index_and_slice_behavior: Getitem behavior
- test_get_data_by_key_returns_readonly_array_and_correct_values: Get data by key
- test_set_data_by_key_accepts_scalar_and_broadcastable_arrays_and_rejects_bad_shapes: Set data by key
- test_metadata_type_enforcement_accepts_int_and_np_floating: Metadata type enforcement
- test_size_returns_python_int_and_matches_product_of_shape: Size property
- test_copy_makes_independent_data_but_preserves_content_and_metadata_shallow_copy: Copy functionality
- test_view_shares_data_and_metadata_references: View functionality
- test_reshape_returns_new_view_and_preserves_nvar: Reshape functionality
- test_copy_does_not_affect_original: Copy independence
- test_empty_does_not_affect_original: Empty method independence
- test_3d_slicing_preserves_correct_ndim_and_shape: 3D slicing behavior
- test_basic_mean_functionality: Basic mean functionality on different axes
- test_negative_axis_indexing: Negative axis indexing verification
- test_error_conditions: Error condition handling for invalid axes
- test_metadata_preservation: Metadata preservation after mean operation
- test_edge_cases: Single dimension edge case handling
- test_variable_axis_protection: Variable axis access protection
- test_structured_data_cannot_be_instantiated_directly: StructuredData abstract behavior
- test_structured_data_with_empty_data_keys: Empty data keys validation
- test_structured_data_defaults_inheritance: Defaults inheritance behavior
- test_triangulated_flag_handling: Triangulated flag validation
- test_memory_layout_validation: Memory layout requirements
- test_property_access_edge_cases: Property access with different shapes
- test_slicing_behavior_edge_cases: Slicing behavior edge cases
- test_module_constants_and_attributes: Module constants and attributes
- test_basic_nanmean_functionality: Basic nanmean functionality on different axes
- test_nanmean_vs_mean_comparison: Nanmean vs mean comparison with no NaNs
- test_nanmean_all_nan_behavior: All-NaN behavior along axis
- test_nanmean_negative_axis: Negative axis indexing verification
- test_nanmean_error_conditions: Error condition handling for invalid axes
- test_nanmean_metadata_preservation: Metadata preservation after nanmean operation
- test_nanmean_edge_cases: Single dimension edge case with NaN handling
- test_nanmean_warning_suppression: Warning suppression for empty slice cases
- test_nanmean_variable_axis_protection: Variable axis access protection for nanmean
- test_initial_data_is_f_contiguous_and_float32: Initial data F-contiguous and float32 verification
- test_copy_preserves_f_contiguous_and_dtype: Copy operation memory layout preservation
- test_view_preserves_memory_layout: View operation memory layout preservation
- test_transpose_may_affect_contiguity: Transpose effects on memory contiguity
- test_flat_preserves_dtype: Flatten operation dtype preservation
- test_getitem_slicing_preserves_dtype: Slicing operation dtype preservation
- test_empty_creates_f_contiguous_float32: Empty creation F-contiguous float32 verification
- test_reshape_preserves_dtype_and_contiguity: Reshape operation preservation testing
- test_copy_independence_with_memory_layout: Copy independence with memory layout verification
"""

# test_structured_data.py
import numpy as np
import pytest

from ember.struct import StructuredData, cached_array
import ember.struct


# helper subclasses for tests
class ThreeVarData(StructuredData):
    _data_keys = ("a", "b", "c")


class OneVarData(StructuredData):
    _data_keys = ("q",)


def fill_threevar(obj: ThreeVarData):
    """Fill object with deterministic numbers so tests can predict results."""
    # spatial shape
    shp = obj.shape
    npts = int(np.prod(shp))
    # create a base (npoints,) then reshape to spatial shape
    base = np.arange(npts, dtype=np.float32).reshape(shp)
    obj._set_data_by_keys(("a",), base)
    obj._set_data_by_keys(("b",), base + 100.0)
    obj._set_data_by_keys(("c",), base + 200.0)
    return obj


def test_shape_and_dimension_properties():
    data = ThreeVarData(shape=(2, 3))
    assert data.nvar == 3
    assert data.shape == (2, 3)
    assert data.ndim == 2
    assert data.ni == 2
    assert data.nj == 3
    # nk should be invalid for 2D and raise AttributeError (per our agreed API)
    with pytest.raises(AttributeError):
        _ = data.nk


def test_increment_data_slice():
    """Test the _increment_data method for in-place increments with slice."""
    data = fill_threevar(ThreeVarData(shape=(3, 4)))

    # Store original values
    orig_a = data._get_data_by_keys(("a",)).copy()
    orig_b = data._get_data_by_keys(("b",)).copy()
    orig_c = data._get_data_by_keys(("c",)).copy()

    # Create increment for a specific slice
    test_slice = (slice(1, 3), slice(None))  # Rows 1-2, all columns
    delta = np.ones((2, 4, 2), dtype=np.float32)  # Shape matches slice + 2 variables
    delta[..., 0] = 10.0  # Increment for variable "a"
    delta[..., 1] = 20.0  # Increment for variable "b"

    # Apply increment to variables "a" and "b"
    data._increment_data(("a", "b"), delta, test_slice)

    # Check that variables "a" and "b" were incremented in the slice
    np.testing.assert_allclose(
        data._get_data_by_keys(("a",))[test_slice], orig_a[test_slice] + 10.0
    )
    np.testing.assert_allclose(
        data._get_data_by_keys(("b",))[test_slice], orig_b[test_slice] + 20.0
    )

    # Check that variable "c" was unchanged
    np.testing.assert_allclose(data._get_data_by_keys(("c",)), orig_c)

    # Check that regions outside the slice were unchanged for "a" and "b"
    mask = np.ones((3, 4), dtype=bool)
    mask[test_slice] = False
    np.testing.assert_allclose(data._get_data_by_keys(("a",))[mask], orig_a[mask])
    np.testing.assert_allclose(data._get_data_by_keys(("b",))[mask], orig_b[mask])


def test_increment_data_full_array():
    """Test the _increment_data method for full array increments (no slice)."""
    data = fill_threevar(ThreeVarData(shape=(3, 4)))

    # Store original values
    orig_a = data._get_data_by_keys(("a",)).copy()
    orig_b = data._get_data_by_keys(("b",)).copy()
    orig_c = data._get_data_by_keys(("c",)).copy()

    # Create increment for full array
    delta = np.ones(
        (3, 4, 2), dtype=np.float32
    )  # Shape matches full array + 2 variables
    delta[..., 0] = 5.0  # Increment for variable "a"
    delta[..., 1] = 15.0  # Increment for variable "b"

    # Apply increment to variables "a" and "b" (no slice)
    data._increment_data(("a", "b"), delta)

    # Check that variables "a" and "b" were incremented everywhere
    np.testing.assert_allclose(data._get_data_by_keys(("a",)), orig_a + 5.0)
    np.testing.assert_allclose(data._get_data_by_keys(("b",)), orig_b + 15.0)

    # Check that variable "c" was unchanged
    np.testing.assert_allclose(data._get_data_by_keys(("c",)), orig_c)


def test_increment_data_single_variable():
    """Test _increment_data with a single variable."""
    data = OneVarData(shape=(2, 3))
    data._set_data_by_keys(("q",), np.ones((2, 3), dtype=np.float32))

    orig_q = data._get_data_by_keys(("q",)).copy()

    # Increment entire array
    delta = np.full((2, 3, 1), 5.0, dtype=np.float32)
    data._increment_data(("q",), delta, (slice(None), slice(None)))

    # Check increment was applied
    np.testing.assert_allclose(data._get_data_by_keys(("q",)), orig_q + 5.0)


def test_increment_data_consecutive_variables():
    """Test _increment_data with consecutive variables."""
    data = fill_threevar(ThreeVarData(shape=(2, 2)))

    orig_a = data._get_data_by_keys(("a",)).copy()
    orig_b = data._get_data_by_keys(("b",)).copy()
    orig_c = data._get_data_by_keys(("c",)).copy()

    # Increment consecutive variables "a" and "b"
    delta = np.ones((2, 2, 2), dtype=np.float32)
    delta[..., 0] = 100.0  # For "a"
    delta[..., 1] = 200.0  # For "b"

    data._increment_data(("a", "b"), delta, (slice(None), slice(None)))

    # Check "a" and "b" were incremented
    np.testing.assert_allclose(data._get_data_by_keys(("a",)), orig_a + 100.0)
    np.testing.assert_allclose(data._get_data_by_keys(("b",)), orig_b + 200.0)

    # Check "c" was unchanged
    np.testing.assert_allclose(data._get_data_by_keys(("c",)), orig_c)


def test_increment_data_non_consecutive_variables_error():
    """Test that _increment_data raises error for non-consecutive variables."""
    data = fill_threevar(ThreeVarData(shape=(2, 2)))

    # Try to increment non-consecutive variables "a" and "c" (indices 0 and 2)
    delta = np.ones((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Variable indices must be consecutive"):
        data._increment_data(("a", "c"), delta, (slice(None), slice(None)))


def test_transpose_keeps_variable_axis_last_and_reorders_spatial_axes():
    orig = fill_threevar(ThreeVarData(shape=(2, 3)))
    # transpose spatial axes (1,0)
    out = orig.transpose(axes=(1, 0))
    # variable axis must still be last
    assert out._data.shape[-1] == orig.nvar
    # shape should be (3,2) for spatial dims
    assert out.shape == (3, 2)
    # expected using numpy full transpose on underlying array
    expected = np.transpose(orig._data, axes=(1, 0, 2))
    np.testing.assert_array_equal(out._data, expected)


def test_transpose_invalid_axes_raises():
    obj = ThreeVarData(shape=(2, 3))
    # wrong length
    with pytest.raises(ValueError):
        obj.transpose(axes=(0,))  # ndim is 2, so need 2 axes
    # invalid axis values
    with pytest.raises(ValueError):
        obj.transpose(axes=(0, 3))


def test_flip_negative_and_positive_axes_match_numpy_flip():
    orig = fill_threevar(ThreeVarData(shape=(2, 3)))
    # flip axis -1 => flip second spatial axis (index 1)
    out_neg = orig.flip(-1)
    expected_neg = np.flip(orig._data, axis=1)
    np.testing.assert_array_equal(out_neg._data, expected_neg)

    # flip axis 0
    out0 = orig.flip(0)
    expected0 = np.flip(orig._data, axis=0)
    np.testing.assert_array_equal(out0._data, expected0)


def test_flip_invalid_axis_raises():
    obj = ThreeVarData(shape=(2, 3))
    with pytest.raises(ValueError):
        obj.flip(2)  # valid spatial axes are 0 and 1
    with pytest.raises(ValueError):
        obj.flip(-3)  # negative out-of-range


def test_squeeze_does_not_remove_variable_axis_for_nvar_one():
    # create object with single variable and a singleton spatial axis
    orig = OneVarData(shape=(1, 4))
    # populate q with easy-to-check numbers
    arr = np.arange(4, dtype=np.float32).reshape((1, 4))
    orig._set_data_by_keys(("q",), arr)
    assert orig._data.shape == (1, 4, 1)  # spatial (1,4) and nvar=1

    # squeeze should remove the leading singleton spatial axis but keep variable axis
    out = orig.squeeze()
    # expected shape(np.squeeze over spatial axes only)
    expected = np.squeeze(orig._data, axis=(0,))
    np.testing.assert_array_equal(out._data, expected)
    assert out._data.shape == (4, 1)
    assert out.nvar == 1


def test_flat_shape_and_values_equal_numpy_reshape():
    orig = fill_threevar(ThreeVarData(shape=(2, 3)))
    out = orig.flat()
    # shape must be (npoints, nvar)
    assert out._data.shape == (orig.size, orig.nvar)
    # expected computed with numpy reshape on the original array
    expected = orig._data.reshape(-1, orig.nvar)
    np.testing.assert_array_equal(out._data, expected)


def test_getitem_scalar_index_and_slice_behavior():
    orig = fill_threevar(ThreeVarData(shape=(2, 3)))
    # scalar index (first spatial axis)
    item0 = orig[0]
    assert isinstance(item0, StructuredData)
    # item0 spatial shape should be (3,) because we removed the first spatial axis
    assert item0.shape == (3,)
    assert item0.nvar == orig.nvar
    # checking that values correspond to orig._data[0, :, :]
    np.testing.assert_array_equal(item0._data, orig._data[0, :, :])

    # slice on spatial axes
    slice_obj = orig[:, :2]
    assert slice_obj.shape == (2, 2)
    # compare raw arrays
    np.testing.assert_array_equal(slice_obj._data, orig._data[:, :2, :])


def test_get_data_by_key_returns_readonly_array_and_correct_values():
    orig = fill_threevar(ThreeVarData(shape=(2, 3)))
    arr = orig._get_data_by_keys(("a",))
    # shape should match spatial shape
    assert arr.shape == orig.shape
    # array should be read-only for arrays (scalar would already be read-only)
    if arr.shape != ():
        assert arr.flags.writeable is False
        with pytest.raises(ValueError):
            arr[...] = 0.0
    # content matches original slice
    np.testing.assert_array_equal(arr, orig._data[..., orig._data_inds["a"]])


def test_set_data_by_key_accepts_scalar_and_broadcastable_arrays_and_rejects_bad_shapes():
    orig = ThreeVarData(shape=(2, 3))
    # scalar broadcast
    orig._set_data_by_keys(("a",), np.array([5.0], dtype=np.float32))
    np.testing.assert_array_equal(orig._data[..., orig._data_inds["a"]], 5.0)

    # 2D array matching shape
    arr = np.arange(6, dtype=np.float32).reshape((2, 3))
    orig._set_data_by_keys(("b",), arr)
    np.testing.assert_array_equal(orig._data[..., orig._data_inds["b"]], arr)

    # incompatible shape should raise ValueError
    with pytest.raises(ValueError):
        orig._set_data_by_keys(("c",), np.ones((4,), dtype=np.float32))


def test_metadata_type_enforcement_accepts_int_and_np_floating():
    obj = ThreeVarData(shape=(1, 1))
    # Accept integer
    obj._set_metadata_by_key("m_int", 2)
    assert isinstance(obj._get_metadata_by_key("m_int"), int)

    # Accept numpy floating scalars
    obj._set_metadata_by_key("m_npfloat32", np.float32(1.23))
    assert isinstance(obj._get_metadata_by_key("m_npfloat32"), np.floating)

    obj._set_metadata_by_key("m_npfloat64", np.float64(4.56))
    assert isinstance(obj._get_metadata_by_key("m_npfloat64"), np.floating)


def test_size_returns_python_int_and_matches_product_of_shape():
    obj = ThreeVarData(shape=(2, 5))
    assert isinstance(obj.size, int)
    assert obj.size == int(np.prod(obj.shape))


def test_copy_makes_independent_data_but_preserves_content_and_metadata_shallow_copy():
    orig = fill_threevar(ThreeVarData(shape=(2, 3)))
    orig._set_metadata_by_key("meta1", np.float32(7.0))
    copied = orig.copy()
    # data arrays must be equal but not the same object (deep copy)
    np.testing.assert_array_equal(copied._data, orig._data)
    assert copied._data is not orig._data
    # metadata dict should be a different dict object (shallow-copied)
    assert copied._metadata is not orig._metadata
    assert copied._metadata == orig._metadata


def test_view_shares_data_and_metadata_references():
    orig = fill_threevar(ThreeVarData(shape=(2, 3)))
    v = orig.view()
    # view should reference the same underlying array and metadata object
    assert v._data is orig._data
    assert v._metadata is orig._metadata


def test_reshape_returns_new_view_and_preserves_nvar():
    obj = fill_threevar(ThreeVarData(shape=(2, 3)))
    old_nvar = obj.nvar
    reshaped = obj.reshape((6,))  # collapse to 1D of 6 points
    # Original object should remain unchanged
    assert obj.shape == (2, 3)
    # New reshaped object should have the new shape
    assert reshaped.shape == (6,)
    assert reshaped.nvar == old_nvar
    # data shape must match the new spatial shape
    assert reshaped._data.shape == (6, old_nvar)


def test_copy_does_not_affect_original():
    orig = fill_threevar(ThreeVarData(shape=(2, 3)))
    orig._set_metadata_by_key("test", 2)
    copy_obj = orig.copy()

    # modify copy's data
    copy_obj._data[...] = 42.0

    # original data should be unchanged
    assert not np.any(orig._data == 42.0)

    # modify copy's metadata
    copy_obj._metadata["test"] = 123
    assert orig._metadata["test"] != 123


def test_empty_does_not_affect_original():
    orig = fill_threevar(ThreeVarData(shape=(2, 3)))
    orig._set_metadata_by_key("test", 2)
    empty_obj = orig.empty(shape=(5, 5))

    # data shapes should differ
    assert empty_obj.shape != orig.shape

    # modifying empty_obj data should not change orig
    empty_obj._data[...] = 99.0
    assert not np.any(orig._data == 99.0)

    # modifying metadata in empty should not affect orig
    empty_obj._metadata["test"] = 999
    assert orig._metadata["test"] != 999


def test_3d_slicing_preserves_correct_ndim_and_shape():
    # Create a 3D StructuredData instance
    orig = fill_threevar(ThreeVarData(shape=(4, 5, 6)))

    # Test various slice combinations
    # Single index slice - reduces ndim by 1
    slice_i = orig[1]
    assert slice_i.ndim == 2
    assert slice_i.shape == (5, 6)

    # Slice with range on first axis
    slice_range_i = orig[1:3]
    assert slice_range_i.ndim == 3
    assert slice_range_i.shape == (2, 5, 6)

    # Slice with range on second axis
    slice_range_j = orig[:, 2:4]
    assert slice_range_j.ndim == 3
    assert slice_range_j.shape == (4, 2, 6)

    # Slice with range on third axis
    slice_range_k = orig[:, :, 1:5]
    assert slice_range_k.ndim == 3
    assert slice_range_k.shape == (4, 5, 4)

    # Multiple single indices - reduces ndim accordingly
    slice_ij = orig[1, 2]
    assert slice_ij.ndim == 1
    assert slice_ij.shape == (6,)

    # All single indices - reduces to 0D
    slice_ijk = orig[1, 2, 3]
    assert slice_ijk.ndim == 0
    assert slice_ijk.shape == ()

    # Mixed slicing - single index and range
    slice_mixed = orig[1, 2:4]
    assert slice_mixed.ndim == 2
    assert slice_mixed.shape == (2, 6)

    # Verify data integrity for one slice
    np.testing.assert_array_equal(slice_i._data, orig._data[1, :, :, :])
    np.testing.assert_array_equal(slice_ij._data, orig._data[1, 2, :, :])


def test_set_data_by_key_scalar_target_with_singleton_array():
    """Test that (1,) shape arrays can be set on scalar (()) targets."""
    obj = OneVarData(shape=())
    # Set with (1,) shape array - should squeeze and broadcast to ()
    val = np.array([42.0], dtype=np.float32)
    obj._set_data_by_keys(("q",), val)
    # Verify the scalar value was set correctly
    result = obj._get_data_by_keys(("q",))
    assert result.shape == ()
    np.testing.assert_allclose(result, 42.0)


def test_set_data_by_key_scalar_target_with_multidim_singleton():
    """Test that (1, 1, 1) shape arrays can be set on scalar (()) targets."""
    obj = OneVarData(shape=())
    # Set with (1, 1, 1) shape array - should squeeze to scalar
    val = np.array([[[99.0]]], dtype=np.float32)
    obj._set_data_by_keys(("q",), val)
    result = obj._get_data_by_keys(("q",))
    assert result.shape == ()
    np.testing.assert_allclose(result, 99.0)


def test_set_data_by_key_preserves_singleton_broadcasting():
    """Test that legitimate singleton-dimension broadcasting still works."""
    obj = ThreeVarData(shape=(5, 6, 7))
    # (5, 1, 7) should still broadcast to (5, 6, 7)
    val = np.ones((5, 1, 7), dtype=np.float32) * 3.14
    obj._set_data_by_keys(("a",), val)
    result = obj._get_data_by_keys(("a",))
    assert result.shape == (5, 6, 7)
    # All values should be 3.14 (broadcasted from (5,1,7))
    np.testing.assert_allclose(result, 3.14)


def test_set_data_by_key_scalar_target_rejects_incompatible():
    """Test that incompatible shapes are still rejected for scalar targets."""
    obj = OneVarData(shape=())
    # (2, 3) cannot squeeze to scalar
    val = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Cannot broadcast"):
        obj._set_data_by_keys(("q",), val)


def test_set_data_by_keys_scalar_target_with_singleton():
    """Test that _set_data_by_keys works with singleton arrays on scalar target."""
    obj = ThreeVarData(shape=())
    # Create (1, 3) array for three variables on scalar target
    val = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    obj._set_data_by_keys(("a", "b", "c"), val)

    a = obj._get_data_by_keys(("a",))
    b = obj._get_data_by_keys(("b",))
    c = obj._get_data_by_keys(("c",))

    assert a.shape == ()
    assert b.shape == ()
    assert c.shape == ()
    np.testing.assert_allclose(a, 10.0)
    np.testing.assert_allclose(b, 20.0)
    np.testing.assert_allclose(c, 30.0)


def test_set_data_by_keys_preserves_singleton_broadcasting():
    """Test that _set_data_by_keys preserves singleton-dimension broadcasting."""
    obj = ThreeVarData(shape=(5, 6, 7))
    # (5, 1, 7, 2) should broadcast to (5, 6, 7) for first two variables
    val = np.ones((5, 1, 7, 2), dtype=np.float32)
    val[..., 0] = 1.11  # for "a" and "b"
    val[..., 1] = 2.22

    obj._set_data_by_keys(("a", "b"), val)

    a = obj._get_data_by_keys(("a",))
    b = obj._get_data_by_keys(("b",))

    assert a.shape == (5, 6, 7)
    assert b.shape == (5, 6, 7)
    np.testing.assert_allclose(a, 1.11)
    np.testing.assert_allclose(b, 2.22)


def test_basic_mean_functionality():
    """Test basic mean functionality on different axes."""
    from ember.block import Block

    # Create a Block with shape (4, 3) - 4 nodes along axis 0, 3 along axis 1
    block = Block(shape=(4, 3))

    # Set test data in rho with variation along axis 0
    # Create array with values [1, 2, 3, 4] along axis 0, same for all j
    rho_data = np.zeros((4, 3), dtype=np.float32)
    for i in range(4):
        rho_data[i, :] = i + 1  # Each row has values 1, 2, 3, 4

    block._set_data_by_keys(("rho",), rho_data)

    # Test mean along axis 0 (should average over i-direction)
    mean_axis0 = block.mean(axis=0)
    expected_mean = np.mean([1, 2, 3, 4])  # = 2.5
    assert np.allclose(mean_axis0.conserved_nd[..., 0], expected_mean), (
        f"Expected {expected_mean}, got {mean_axis0.conserved_nd[..., 0]}"
    )

    # Test mean along axis 1 (should average over j-direction)
    mean_axis1 = block.mean(axis=1)
    # Each row should keep its original value since all j values are the same
    assert np.allclose(mean_axis1.conserved_nd[..., 0], [1, 2, 3, 4]), (
        f"Expected [1, 2, 3, 4], got {mean_axis1.conserved_nd[..., 0]}"
    )


def test_negative_axis_indexing():
    """Test negative axis indexing."""
    from ember.block import Block

    block = Block(shape=(3, 4))

    # Set data: each row has different values
    rho_data = np.array(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
        dtype=np.float32,
    )
    block._set_data_by_keys(("rho",), rho_data)

    # Test axis=-1 (last spatial dimension, should be equivalent to axis=1)
    mean_neg1 = block.mean(axis=-1)
    mean_pos1 = block.mean(axis=1)

    assert np.allclose(
        mean_neg1.conserved_nd[..., 0], mean_pos1.conserved_nd[..., 0]
    ), "axis=-1 should equal axis=1"

    # Test axis=-2 (should be equivalent to axis=0)
    mean_neg2 = block.mean(axis=-2)
    mean_pos0 = block.mean(axis=0)

    assert np.allclose(
        mean_neg2.conserved_nd[..., 0], mean_pos0.conserved_nd[..., 0]
    ), "axis=-2 should equal axis=0"


def test_error_conditions():
    """Test error conditions."""
    from ember.block import Block

    block = Block(shape=(3, 4))

    # Test invalid positive axis
    with pytest.raises(ValueError, match="Invalid axis 2 for data with 2 dimensions"):
        block.mean(axis=2)  # Invalid for 2D data

    # Test invalid negative axis
    with pytest.raises(ValueError, match="Invalid axis -3 for data with 2 dimensions"):
        block.mean(axis=-3)  # Invalid for 2D data


def test_edge_cases():
    """Test edge cases."""
    from ember.block import Block

    # Test single dimension (should result in scalar)
    block = Block(shape=(5,))
    rho_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    block._set_data_by_keys(("rho",), rho_data)

    scalar_result = block.mean(axis=0)
    assert scalar_result.shape == (), (
        f"Expected scalar shape (), got {scalar_result.shape}"
    )
    assert np.allclose(scalar_result.conserved_nd[..., 0], 3.0), (
        f"Expected 3.0, got {scalar_result.conserved_nd[..., 0]}"
    )


def test_variable_axis_protection():
    """Test that averaging over variable axis is forbidden."""
    from ember.block import Block

    block = Block(shape=(3, 4))
    block._set_data_by_keys(("rho",), np.ones((3, 4), dtype=np.float32))

    # The variable axis is the last axis in _data, which should not be accessible
    # Since block.ndim = 2, axis=2 would be the variable axis if it were valid
    with pytest.raises(ValueError, match="Invalid axis 2 for data with 2 dimensions"):
        block.mean(axis=2)


# Edge case tests merged from test_struct_edge_cases.py


class MockStructuredData(StructuredData):
    """Mock StructuredData class for testing abstract functionality."""

    _data_keys = ("x", "y", "z")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if hasattr(self, "_data"):
            # Initialize data arrays - only works for 2D shapes due to indexing
            self._data[:, :, 0] = np.arange(np.prod(self.shape)).reshape(
                self.shape
            )  # x
            self._data[:, :, 1] = (
                np.arange(np.prod(self.shape)).reshape(self.shape) * 2
            )  # y
            self._data[:, :, 2] = (
                np.arange(np.prod(self.shape)).reshape(self.shape) * 3
            )  # z


def test_structured_data_cannot_be_instantiated_directly():
    """Test that StructuredData is effectively abstract."""

    # This should fail - StructuredData with empty _data_keys should raise an error
    with pytest.raises(
        AssertionError, match="StructuredData must have at least one variable"
    ):
        ember.struct.StructuredData(shape=(2, 2))


def test_structured_data_with_empty_data_keys():
    """Test StructuredData behavior with empty _data_keys."""

    class EmptyKeysData(ember.struct.StructuredData):
        _data_keys = ()

    # Should fail - empty _data_keys means nvar=0 which violates assertion
    with pytest.raises(
        AssertionError, match="StructuredData must have at least one variable"
    ):
        EmptyKeysData(shape=(2, 2))


def test_structured_data_defaults_inheritance():
    """Test that _defaults are properly inherited."""

    class DefaultsData(ember.struct.StructuredData):
        _defaults = {"custom_param": 42, "string_param": "default"}
        _data_keys = ("x",)

    obj = DefaultsData(shape=(2, 2))

    # Defaults should be accessible via _get_metadata_by_key
    assert obj._get_metadata_by_key("custom_param") == 42
    assert obj._get_metadata_by_key("string_param") == "default"


def test_triangulated_flag_handling():
    """Test triangulated flag validation."""

    # Valid triangulated data (shape[1] == 3)
    obj1 = MockStructuredData(shape=(5, 3))
    obj1.set_triangulated(True)
    assert obj1.triangulated

    # Test default (False)
    obj2 = MockStructuredData(shape=(3, 3))
    assert not obj2.triangulated

    # Invalid triangulated data (shape[1] != 3)
    with pytest.raises(
        ValueError, match="Triangulated data requires shape\\[1\\] == 3"
    ):
        obj3 = MockStructuredData(shape=(3, 4))
        obj3.set_triangulated(True)


def test_memory_layout_validation():
    """Test memory layout requirements."""

    obj = MockStructuredData(shape=(3, 4))

    # Check that data is F-contiguous
    assert obj._data.flags["F_CONTIGUOUS"]
    assert obj._data.dtype == np.float32


def test_property_access_edge_cases():
    """Test property access with different shapes."""

    # Test 2D data (MockStructuredData only works with 2D due to indexing)
    obj_2d = MockStructuredData(shape=(3, 4))
    assert obj_2d.ndim == 2
    assert obj_2d.ni == 3
    assert obj_2d.nj == 4

    # Test square shape
    obj_square = MockStructuredData(shape=(5, 5))
    assert obj_square.ndim == 2
    assert obj_square.ni == 5
    assert obj_square.nj == 5


def test_slicing_behavior_edge_cases():
    """Test slicing behavior with different slice types."""

    obj = MockStructuredData(shape=(4, 5))

    # Test various slice types
    slice1 = obj[0]  # Single index
    assert slice1.shape == (5,)

    slice2 = obj[0:2]  # Range slice
    assert slice2.shape == (2, 5)

    slice3 = obj[:, 0:3]  # Multi-axis slice
    assert slice3.shape == (4, 3)


def test_module_constants_and_attributes():
    """Test module-level constants and attributes."""

    # Test that required classes exist
    assert hasattr(ember.struct, "StructuredData")

    # Test module constants
    assert hasattr(ember.struct, "f32")
    assert ember.struct.f32 == np.float32


# Nanmean tests merged from test_struct_nanmean.py


def test_basic_nanmean_functionality():
    """Test basic nanmean functionality on different axes."""
    from ember.block import Block

    # Create a Block with shape (4, 3) - 4 nodes along axis 0, 3 along axis 1
    block = Block(shape=(4, 3))

    # Set test data in rho with some NaN values
    rho_data = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, np.nan],
            [10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )

    block._set_data_by_keys(("rho",), rho_data)

    # Test nanmean along axis 0 (should average over i-direction, ignoring NaNs)
    nanmean_axis0 = block.nanmean(axis=0)

    # Expected: column 0: (1+4+7+10)/4 = 5.5
    #          column 1: (2+8+11)/3 = 7.0 (ignoring NaN)
    #          column 2: (3+6+12)/3 = 7.0 (ignoring NaN)
    expected = [5.5, 7.0, 7.0]
    assert np.allclose(nanmean_axis0.conserved_nd[..., 0], expected), (
        f"Expected {expected}, got {nanmean_axis0.conserved_nd[..., 0]}"
    )

    # Test nanmean along axis 1 (should average over j-direction, ignoring NaNs)
    nanmean_axis1 = block.nanmean(axis=1)

    # Expected: row 0: (1+2+3)/3 = 2.0
    #          row 1: (4+6)/2 = 5.0 (ignoring NaN)
    #          row 2: (7+8)/2 = 7.5 (ignoring NaN)
    #          row 3: (10+11+12)/3 = 11.0
    expected = [2.0, 5.0, 7.5, 11.0]
    assert np.allclose(nanmean_axis1.conserved_nd[..., 0], expected), (
        f"Expected {expected}, got {nanmean_axis1.conserved_nd[..., 0]}"
    )


def test_nanmean_vs_mean_comparison():
    """Test that nanmean and mean give same results when no NaNs present."""
    from ember.block import Block

    block = Block(shape=(3, 4))

    # Set data without any NaN values
    rho_data = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=np.float32,
    )
    block._set_data_by_keys(("rho",), rho_data)

    # Test that nanmean and mean give identical results
    mean_result = block.mean(axis=0)
    nanmean_result = block.nanmean(axis=0)

    assert np.allclose(
        mean_result.conserved_nd[..., 0], nanmean_result.conserved_nd[..., 0]
    ), "nanmean should equal mean when no NaNs present"


def test_nanmean_all_nan_behavior():
    """Test nanmean behavior when all values along an axis are NaN."""
    from ember.block import Block

    block = Block(shape=(3, 2))

    # Set data where one column is all NaN
    rho_data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    block._set_data_by_keys(("rho",), rho_data)

    # Test nanmean along axis 0
    nanmean_result = block.nanmean(axis=0)

    # Expected: column 0: (1+2+3)/3 = 2.0, column 1: NaN (all NaN)
    expected_0 = 2.0
    assert np.isclose(nanmean_result.conserved_nd[..., 0][0], expected_0), (
        f"Expected {expected_0}, got {nanmean_result.conserved_nd[..., 0][0]}"
    )
    assert np.isnan(nanmean_result.conserved_nd[..., 0][1]), (
        f"Expected NaN, got {nanmean_result.conserved_nd[..., 0][1]}"
    )


def test_nanmean_negative_axis():
    """Test nanmean with negative axis indexing."""
    from ember.block import Block

    block = Block(shape=(3, 4))

    # Set data with some NaN values
    rho_data = np.array(
        [
            [1.0, np.nan, 3.0, 4.0],
            [5.0, 6.0, np.nan, 8.0],
            [9.0, 10.0, 11.0, np.nan],
        ],
        dtype=np.float32,
    )
    block._set_data_by_keys(("rho",), rho_data)

    # Test axis=-1 (last spatial dimension, should be equivalent to axis=1)
    nanmean_neg1 = block.nanmean(axis=-1)
    nanmean_pos1 = block.nanmean(axis=1)

    assert np.allclose(
        nanmean_neg1.conserved_nd[..., 0],
        nanmean_pos1.conserved_nd[..., 0],
        equal_nan=True,
    ), "axis=-1 should equal axis=1"


def test_nanmean_error_conditions():
    """Test nanmean error conditions."""
    from ember.block import Block

    block = Block(shape=(3, 4))

    # Test invalid positive axis
    with pytest.raises(ValueError, match="Invalid axis 2 for data with 2 dimensions"):
        block.nanmean(axis=2)  # Invalid for 2D data

    # Test invalid negative axis
    with pytest.raises(ValueError, match="Invalid axis -3 for data with 2 dimensions"):
        block.nanmean(axis=-3)  # Invalid for 2D data


def test_nanmean_metadata_preservation():
    """Test that metadata is preserved correctly."""
    from ember.block import Block

    # Use a simple Block subclass to test metadata
    block = Block(shape=(5, 2))
    block._set_data_by_keys(("rho",), np.ones((5, 2), dtype=np.float32))

    # Test that metadata is preserved after nanmean
    avg_block = block.nanmean(axis=0)

    # Check that it's still a Block object
    assert isinstance(avg_block, Block), f"Type not preserved: {type(avg_block)}"


def test_nanmean_edge_cases():
    """Test nanmean edge cases."""
    from ember.block import Block

    # Test single dimension with NaN
    block = Block(shape=(5,))
    rho_data = np.array([1.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float32)
    block._set_data_by_keys(("rho",), rho_data)

    scalar_result = block.nanmean(axis=0)

    # Expected: (1+3+4+5)/4 = 3.25 (ignoring NaN)
    expected = 3.25
    assert scalar_result.shape == (), (
        f"Expected scalar shape (), got {scalar_result.shape}"
    )
    assert np.allclose(scalar_result.conserved_nd[..., 0], expected), (
        f"Expected {expected}, got {scalar_result.conserved_nd[..., 0]}"
    )


def test_nanmean_warning_suppression():
    """Test that nanmean suppresses 'mean of empty slice' warnings."""
    from ember.block import Block

    block = Block(shape=(3, 2))

    # Set data where one column is all NaN
    rho_data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    block._set_data_by_keys(("rho",), rho_data)

    # This should not raise any warnings
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        block.nanmean(axis=0)

        # Check if any warnings were about mean of empty slice
        empty_slice_warnings = [
            warning for warning in w if "Mean of empty slice" in str(warning.message)
        ]
        assert len(empty_slice_warnings) == 0, (
            f"Expected no 'mean of empty slice' warnings, got {len(empty_slice_warnings)}"
        )


def test_nanmean_variable_axis_protection():
    """Test that averaging over variable axis is forbidden."""
    from ember.block import Block

    block = Block(shape=(3, 4))
    block._set_data_by_keys(("rho",), np.ones((3, 4), dtype=np.float32))

    # The variable axis is the last axis in _data, which should not be accessible
    # Since block.ndim = 2, axis=2 would be the variable axis if it were valid
    with pytest.raises(ValueError, match="Invalid axis 2 for data with 2 dimensions"):
        block.nanmean(axis=2)


# Memory layout tests merged from test_struct_memory_layout.py


class MemoryTestData(StructuredData):
    """Test class with 3 variables for memory layout testing."""

    _data_keys = ("x", "y", "z")


def test_initial_data_is_f_contiguous_and_float32():
    """Test that initial _data array is F_CONTIGUOUS and float32."""
    data = MemoryTestData(shape=(3, 4, 5))

    # Check memory layout
    assert data._data.flags["F_CONTIGUOUS"], (
        "_data should be F_CONTIGUOUS after initialization"
    )
    assert not data._data.flags["C_CONTIGUOUS"], (
        "_data should not be C_CONTIGUOUS after initialization"
    )

    # Check dtype
    assert data._data.dtype == np.float32, "_data should have float32 dtype"


def test_copy_preserves_f_contiguous_and_dtype():
    """Test that copy() operation preserves F_CONTIGUOUS and float32 dtype."""
    original = MemoryTestData(shape=(3, 4, 5))

    # Fill with some data to make copy meaningful
    original._set_data_by_keys(
        ("x",), np.arange(60, dtype=np.float32).reshape((3, 4, 5))
    )
    original._set_data_by_keys(("y",), np.ones((3, 4, 5), dtype=np.float32) * 2.0)
    original._set_data_by_keys(("z",), np.ones((3, 4, 5), dtype=np.float32) * 3.0)

    # Make copy
    copied = original.copy()

    # Verify original properties
    assert original._data.flags["F_CONTIGUOUS"], "Original _data should be F_CONTIGUOUS"
    assert original._data.dtype == np.float32, "Original _data should be float32"

    # Verify copy properties
    assert copied._data.flags["F_CONTIGUOUS"], "Copied _data should be F_CONTIGUOUS"
    assert copied._data.dtype == np.float32, "Copied _data should be float32"

    # Verify they are independent arrays
    assert copied._data is not original._data, "Copy should create independent array"

    # Verify data content is preserved
    np.testing.assert_array_equal(copied._data, original._data)


def test_view_preserves_memory_layout():
    """Test that view() operation preserves memory layout properties."""
    original = MemoryTestData(shape=(4, 5, 6))

    # Create view
    viewed = original.view()

    # View should reference same array, so should have same properties
    assert viewed._data is original._data, "View should reference same _data array"
    assert viewed._data.flags["F_CONTIGUOUS"], "View _data should be F_CONTIGUOUS"
    assert viewed._data.dtype == np.float32, "View _data should be float32"


def test_transpose_may_affect_contiguity():
    """Test that transpose operations may affect memory layout."""
    original = MemoryTestData(shape=(3, 4, 5))

    # Fill with data
    original._set_data_by_keys(
        ("x",), np.arange(60, dtype=np.float32).reshape((3, 4, 5))
    )

    # Original should be F_CONTIGUOUS
    assert original._data.flags["F_CONTIGUOUS"]

    # Transpose - this may or may not preserve F_CONTIGUOUS depending on the operation
    transposed = original.transpose((2, 1, 0))  # Reverse axes

    # Dtype should always be preserved
    assert transposed._data.dtype == np.float32, (
        "Transpose should preserve float32 dtype"
    )

    # Shape should be reversed
    assert transposed.shape == (5, 4, 3), "Transpose should reverse shape"


def test_flat_preserves_dtype():
    """Test that flat() operation preserves dtype."""
    original = MemoryTestData(shape=(3, 4, 5))

    # Fill with data
    original._set_data_by_keys(("x",), np.ones((3, 4, 5), dtype=np.float32))

    flattened = original.flat()

    # Should preserve dtype
    assert flattened._data.dtype == np.float32, "Flat should preserve float32 dtype"

    # Should have correct shape
    assert flattened.shape == (60,), "Flat should have 1D shape"
    assert flattened._data.shape == (60, 3), (
        "Flat _data should have shape (npoints, nvar)"
    )


def test_getitem_slicing_preserves_dtype():
    """Test that slicing operations preserve dtype."""
    original = MemoryTestData(shape=(4, 5, 6))

    # Fill with data
    original._set_data_by_keys(
        ("x",), np.arange(120, dtype=np.float32).reshape((4, 5, 6))
    )

    # Various slice operations
    sliced = original[1:3, :, 2:5]

    # Should preserve dtype
    assert sliced._data.dtype == np.float32, "Slicing should preserve float32 dtype"

    # Should have correct shape
    assert sliced.shape == (2, 5, 3), "Slice should have correct shape"


def test_empty_creates_f_contiguous_float32():
    """Test that empty() creates arrays with correct memory layout and dtype."""
    original = MemoryTestData(shape=(2, 3))

    # Create empty with new shape
    empty = original.empty(shape=(5, 7, 9))

    # Should have F_CONTIGUOUS and float32
    assert empty._data.flags["F_CONTIGUOUS"], "Empty _data should be F_CONTIGUOUS"
    assert empty._data.dtype == np.float32, "Empty _data should be float32"

    # Should have correct shape
    assert empty.shape == (5, 7, 9), "Empty should have requested shape"


def test_reshape_preserves_dtype_and_contiguity():
    """Test that reshape preserves dtype and contiguity when possible."""
    original = MemoryTestData(shape=(3, 4))

    # Fill with data
    original._set_data_by_keys(("x",), np.arange(12, dtype=np.float32).reshape((3, 4)))

    # Reshape to compatible shape
    reshaped = original.reshape((2, 6))

    # Should preserve dtype
    assert reshaped._data.dtype == np.float32, "Reshape should preserve float32 dtype"

    # Should have new shape
    assert reshaped.shape == (2, 6), "Reshape should change shape"
    assert reshaped._data.shape == (2, 6, 3), (
        "Reshape _data should have shape (new_shape, nvar)"
    )

    # Original should be unchanged
    assert original.shape == (3, 4), "Original should remain unchanged"
    assert original._data.shape == (3, 4, 3), "Original _data should remain unchanged"


def test_copy_independence_with_memory_layout():
    """Test that copy creates truly independent arrays with correct memory properties."""
    original = MemoryTestData(shape=(2, 3))

    # Fill original with specific values
    original._set_data_by_keys(
        ("x",), np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    )
    original._set_data_by_keys(
        ("y",), np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
    )

    # Create copy
    copied = original.copy()

    # Verify both have correct memory layout
    assert original._data.flags["F_CONTIGUOUS"], "Original should be F_CONTIGUOUS"
    assert copied._data.flags["F_CONTIGUOUS"], "Copy should be F_CONTIGUOUS"
    assert original._data.dtype == np.float32, "Original should be float32"
    assert copied._data.dtype == np.float32, "Copy should be float32"

    # Modify original
    original._set_data_by_keys(("x",), np.zeros((2, 3), dtype=np.float32))

    # Verify copy is unaffected
    expected_x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    np.testing.assert_array_equal(copied._get_data_by_keys(("x",)), expected_x)

    # Modify copy
    copied._set_data_by_keys(("y",), np.ones((2, 3), dtype=np.float32) * 999)

    # Verify original is unaffected
    expected_y = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
    np.testing.assert_array_equal(original._get_data_by_keys(("y",)), expected_y)


"""Test module for ember.struct.cached_array decorator.

Tests cached_array decorator functionality with in-place array reuse for performance optimization.

Test cases:
- test_first_access_allocates_and_computes: First access allocates memory and computes result
- test_second_access_returns_cached_no_computation: Second access returns cached result without recomputation
- test_cache_invalidation_reuses_array: Cache invalidation reuses existing array memory
- test_multiple_cached_properties_independent: Multiple cached properties operate independently
- test_version_tracking_works_correctly: Version tracking correctly detects cache invalidation
- test_shape_dtype_compatibility_for_reuse: Shape and dtype compatibility validation for array reuse
- test_metadata_dependent_caching: Caching behavior with metadata dependencies
- test_clear_cache_functionality: Cache clearing functionality
- test_empty_cache_behavior: Behavior when cache is empty
- test_cached_property_result_read_only: cached_array results are not writable in-place
- test_cached_property_reuse_after_invalidation: Memory reuse validation after cache invalidation
- test_rijk_caching_public_interface: Public interface for rijk coordinate caching
- test_cached_property_no_args_never_recalculates: Cached properties with no arguments never recalculate
"""


# Global counters to track function calls and memory allocations
call_counts = {}
allocation_counts = {}


def reset_counters():
    """Reset global counters for testing."""
    global call_counts, allocation_counts
    call_counts.clear()
    allocation_counts.clear()


def track_calls(func_name):
    """Increment call counter for a function."""
    call_counts[func_name] = call_counts.get(func_name, 0) + 1


def track_allocation(func_name):
    """Increment allocation counter for a function."""
    allocation_counts[func_name] = allocation_counts.get(func_name, 0) + 1


def compute_sum_squares(data, out=None):
    """Test function that computes sum of squares with optional in-place output."""
    track_calls("sum_squares")

    # Expected output shape: reduce last dimension to scalar
    expected_shape = data.shape[:-1] + (1,)

    if out is not None and out.shape == expected_shape and out.dtype == data.dtype:
        # Reuse existing array
        result = out
    else:
        # Allocate new array
        track_allocation("sum_squares")
        result = np.empty(expected_shape, dtype=data.dtype, order="F")

    # Compute sum of squares in-place
    result[..., 0] = np.sum(data**2, axis=-1)

    return result


def compute_magnitude(data, out=None):
    """Test function that computes vector magnitude with optional in-place output."""
    track_calls("magnitude")

    # Expected output shape: reduce last dimension to scalar
    expected_shape = data.shape[:-1] + (1,)

    if out is not None and out.shape == expected_shape and out.dtype == data.dtype:
        # Reuse existing array
        result = out
    else:
        # Allocate new array
        track_allocation("magnitude")
        result = np.empty(expected_shape, dtype=data.dtype, order="F")

    # Compute magnitude in-place
    result[..., 0] = np.sqrt(np.sum(data**2, axis=-1))

    return result


def compute_scaled_sum(data, scale_factor, out=None):
    """Test function that depends on both data and metadata (scale_factor)."""
    track_calls("scaled_sum")

    # Expected output shape: reduce last dimension to scalar
    expected_shape = data.shape[:-1] + (1,)

    if out is not None and out.shape == expected_shape and out.dtype == data.dtype:
        # Reuse existing array
        result = out
    else:
        # Allocate new array
        track_allocation("scaled_sum")
        result = np.empty(expected_shape, dtype=data.dtype, order="F")

    # Compute scaled sum in-place
    result[..., 0] = scale_factor * np.sum(data, axis=-1)

    return result


class CacheTestData(StructuredData):
    """Test class for cached property functionality."""

    _data_keys = ("x", "y", "z")

    def set_xyz(self, x, y, z):
        """Set x, y, z coordinates."""
        self._set_data_by_keys(("x",), x)
        self._set_data_by_keys(("y",), y)
        self._set_data_by_keys(("z",), z)
        return self

    def set_scale_factor(self, scale_factor):
        """Set scale factor metadata."""
        self._set_metadata_by_key("scale_factor", scale_factor)
        return self

    @property
    def xyz(self):
        """Get combined xyz coordinates."""
        return self._get_data_by_keys(("x", "y", "z"))

    @property
    def scale_factor(self):
        """Get scale factor from metadata."""
        return self._get_metadata_by_key("scale_factor")

    @cached_array("x", "y", "z")
    def sum_squares(self, out=None):
        """Compute sum of squares with caching and optional in-place output."""
        return compute_sum_squares(self.xyz, out=out)

    @cached_array("x", "y")
    def magnitude_xy(self, out=None):
        """Compute magnitude of x,y components only."""
        return compute_magnitude(self.xyz[..., :2], out=out)

    @cached_array("x", "y", "z", "scale_factor")
    def scaled_sum(self, out=None):
        """Compute scaled sum that depends on both data and metadata."""
        return compute_scaled_sum(self.xyz, self.scale_factor, out=out)


class TestCachedProperty:
    """Test suite for cached_array decorator."""

    def setup_method(self):
        """Setup for each test method."""
        reset_counters()

    def test_first_access_allocates_and_computes(self):
        """Test that first access allocates memory and computes result."""
        cache_obj = CacheTestData(shape=(3, 4))

        # Set test data
        x = np.random.rand(3, 4).astype(np.float32)
        y = np.random.rand(3, 4).astype(np.float32)
        z = np.random.rand(3, 4).astype(np.float32)
        cache_obj.set_xyz(x, y, z)

        # First access
        result = cache_obj.sum_squares

        assert call_counts["sum_squares"] == 1
        assert allocation_counts["sum_squares"] == 1
        assert result.shape == (3, 4, 1)
        assert result.dtype == np.float32

        # Verify correct computation
        expected = np.sum(cache_obj.xyz**2, axis=-1, keepdims=True)
        assert np.allclose(result, expected)

    def test_second_access_returns_cached_no_computation(self):
        """Test that second access returns cached result without recomputation."""
        cache_obj = CacheTestData(shape=(2, 3))

        x = np.ones((2, 3), dtype=np.float32)
        y = np.ones((2, 3), dtype=np.float32) * 2
        z = np.ones((2, 3), dtype=np.float32) * 3
        cache_obj.set_xyz(x, y, z)

        # First access
        result1 = cache_obj.sum_squares

        # Second access
        result2 = cache_obj.sum_squares

        # Should be same object (cached)
        assert result2 is result1

        # Should not have called function again
        assert call_counts["sum_squares"] == 1
        assert allocation_counts["sum_squares"] == 1

    def test_cache_invalidation_reuses_array(self):
        """Test that cache invalidation reuses the existing array."""
        cache_obj = CacheTestData(shape=(2, 2))

        # Initial data
        x1 = np.ones((2, 2), dtype=np.float32)
        y1 = np.ones((2, 2), dtype=np.float32) * 2
        z1 = np.ones((2, 2), dtype=np.float32) * 3
        cache_obj.set_xyz(x1, y1, z1)

        # First computation
        result1 = cache_obj.sum_squares

        # Modify data to invalidate cache
        x2 = np.ones((2, 2), dtype=np.float32) * 4
        cache_obj.set_xyz(x2, y1, z1)  # Only change x

        # Second computation should reuse array
        result2 = cache_obj.sum_squares

        # Should be same object (reused array)
        assert result2 is result1

        # Should have called function twice but allocated only once
        assert call_counts["sum_squares"] == 2
        assert allocation_counts["sum_squares"] == 1

        # Values should be different
        expected1 = np.sum(np.stack([x1, y1, z1], axis=-1) ** 2, axis=-1, keepdims=True)
        expected2 = np.sum(np.stack([x2, y1, z1], axis=-1) ** 2, axis=-1, keepdims=True)

        # Check that result2 has the new values
        assert np.allclose(result2, expected2)
        assert not np.allclose(result2, expected1)

    def test_multiple_cached_properties_independent(self):
        """Test that multiple cached properties are independent."""
        cache_obj = CacheTestData(shape=(2, 2))

        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([[5, 6], [7, 8]], dtype=np.float32)
        z = np.array([[9, 10], [11, 12]], dtype=np.float32)
        cache_obj.set_xyz(x, y, z)

        # Access both properties
        sum_sq = cache_obj.sum_squares
        mag_xy = cache_obj.magnitude_xy

        assert call_counts["sum_squares"] == 1
        assert call_counts["magnitude"] == 1
        assert allocation_counts["sum_squares"] == 1
        assert allocation_counts["magnitude"] == 1

        # Modify only z (should only invalidate sum_squares, not magnitude_xy)
        z_new = z * 2
        cache_obj._set_data_by_keys(("z",), z_new)

        sum_sq_new = cache_obj.sum_squares
        mag_xy_cached = cache_obj.magnitude_xy

        # sum_squares should be recalculated, magnitude_xy should be cached
        assert call_counts["sum_squares"] == 2
        assert call_counts["magnitude"] == 1  # Still 1, not called again

        # magnitude_xy should return same cached object
        assert mag_xy_cached is mag_xy
        # sum_squares should be same object (reused array) but different values
        assert sum_sq_new is sum_sq

    def test_version_tracking_works_correctly(self):
        """Test that version tracking correctly determines cache validity."""
        cache_obj = CacheTestData(shape=(1, 1))

        x = np.array([[1]], dtype=np.float32)
        y = np.array([[2]], dtype=np.float32)
        z = np.array([[3]], dtype=np.float32)
        cache_obj.set_xyz(x, y, z)

        # Get initial versions
        initial_versions = cache_obj._get_version(("x", "y", "z"))

        # First access
        result1 = cache_obj.sum_squares

        # Verify cache entry
        assert "sum_squares" in cache_obj._store
        assert cache_obj._store["sum_squares"][0] == initial_versions
        assert cache_obj._store["sum_squares"][1] is result1

        # Modify one coordinate
        cache_obj.set_xyz(x * 2, y, z)
        new_versions = cache_obj._get_version(("x", "y", "z"))

        # Versions should be different
        assert new_versions != initial_versions

        # Access again - should detect version change and recalculate
        result2 = cache_obj.sum_squares

        # Cache should be updated with new versions
        assert cache_obj._store["sum_squares"][0] == new_versions
        assert cache_obj._store["sum_squares"][1] is result2

    def test_shape_dtype_compatibility_for_reuse(self):
        """Test that arrays are only reused when shape and dtype are compatible."""
        cache_obj = CacheTestData(shape=(2, 2))

        x = np.ones((2, 2), dtype=np.float32)
        y = np.ones((2, 2), dtype=np.float32)
        z = np.ones((2, 2), dtype=np.float32)
        cache_obj.set_xyz(x, y, z)

        # First access
        result1 = cache_obj.sum_squares
        original_shape = result1.shape
        original_dtype = result1.dtype

        # Change data (same shape/dtype) - should reuse
        cache_obj.set_xyz(x * 2, y, z)
        result2 = cache_obj.sum_squares

        assert result2 is result1  # Same object reused
        assert result2.shape == original_shape
        assert result2.dtype == original_dtype

        # Verify only one allocation despite two computations
        assert allocation_counts["sum_squares"] == 1
        assert call_counts["sum_squares"] == 2

    def test_metadata_dependent_caching(self):
        """Test caching for properties that depend on both data and metadata."""
        cache_obj = CacheTestData(shape=(2, 2))

        # Set initial data and metadata
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([[5, 6], [7, 8]], dtype=np.float32)
        z = np.array([[9, 10], [11, 12]], dtype=np.float32)
        scale_factor = 2.0

        cache_obj.set_xyz(x, y, z)
        cache_obj.set_scale_factor(scale_factor)

        # First access - should compute and cache
        result1 = cache_obj.scaled_sum

        assert call_counts["scaled_sum"] == 1
        assert allocation_counts["scaled_sum"] == 1

        # Verify correct computation: scale_factor * sum(xyz, axis=-1)
        expected = scale_factor * np.sum(cache_obj.xyz, axis=-1, keepdims=True)
        assert np.allclose(result1, expected)

        # Second access - should return cached
        result2 = cache_obj.scaled_sum
        assert result2 is result1
        assert call_counts["scaled_sum"] == 1  # No additional call

        # Modify only data (not metadata) - should recalculate and reuse array
        x_new = x * 3
        cache_obj._set_data_by_keys(("x",), x_new)

        result3 = cache_obj.scaled_sum
        assert result3 is result1  # Same array object (reused)
        assert call_counts["scaled_sum"] == 2  # Function called again
        assert allocation_counts["scaled_sum"] == 1  # No new allocation

        # Verify new values are correct
        expected_new = scale_factor * np.sum(cache_obj.xyz, axis=-1, keepdims=True)
        assert np.allclose(result3, expected_new)
        assert not np.allclose(
            result3, expected
        )  # Should be different from first result

        # Modify only metadata - should also recalculate and reuse array
        scale_factor_new = 5.0
        cache_obj.set_scale_factor(scale_factor_new)

        result4 = cache_obj.scaled_sum
        assert result4 is result1  # Same array object (reused again)
        assert call_counts["scaled_sum"] == 3  # Function called a third time
        assert allocation_counts["scaled_sum"] == 1  # Still no new allocation

        # Verify metadata change affected the result
        expected_meta_change = scale_factor_new * np.sum(
            cache_obj.xyz, axis=-1, keepdims=True
        )
        assert np.allclose(result4, expected_meta_change)

        # Test that changing unrelated metadata doesn't invalidate cache
        cache_obj._set_metadata_by_key("unrelated_param", 999)

        result5 = cache_obj.scaled_sum
        assert result5 is result1  # Same cached result
        assert call_counts["scaled_sum"] == 3  # No additional call

        print(
            f"Metadata test summary: {call_counts['scaled_sum']} calls, {allocation_counts['scaled_sum']} allocations"
        )

    def test_clear_cache_functionality(self):
        """Test that clear_cache() forces recalculation on next access."""
        cache_obj = CacheTestData(shape=(2, 2))

        # Set up test data
        x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([[5, 6], [7, 8]], dtype=np.float32)
        z = np.array([[9, 10], [11, 12]], dtype=np.float32)
        cache_obj.set_xyz(x, y, z)

        # First access - should compute and allocate
        result1 = cache_obj.sum_squares
        assert call_counts["sum_squares"] == 1
        assert allocation_counts["sum_squares"] == 1

        # Second access - should return cached (no computation)
        result2 = cache_obj.sum_squares
        assert result2 is result1  # Same cached object
        assert call_counts["sum_squares"] == 1  # No additional call

        # Clear cache
        cache_obj.clear_cache()

        # Next access should recalculate and allocate new array (cache was cleared)
        result3 = cache_obj.sum_squares
        assert result3 is not result1  # Different array object (new allocation)
        assert call_counts["sum_squares"] == 2  # Function called again
        assert allocation_counts["sum_squares"] == 2  # New allocation occurred

        # Values should be identical (data hasn't changed)
        assert np.allclose(result3, result1)

        # Test clear_cache works with multiple cached properties
        cache_obj.set_scale_factor(3.0)
        scaled_result1 = cache_obj.scaled_sum
        mag_result1 = cache_obj.magnitude_xy

        assert call_counts["scaled_sum"] == 1
        assert call_counts["magnitude"] == 1

        # Access again - should be cached
        scaled_result2 = cache_obj.scaled_sum
        mag_result2 = cache_obj.magnitude_xy
        assert scaled_result2 is scaled_result1
        assert mag_result2 is mag_result1
        assert call_counts["scaled_sum"] == 1  # Still 1
        assert call_counts["magnitude"] == 1  # Still 1

        # Clear all caches
        cache_obj.clear_cache()

        # Both should recalculate on next access
        scaled_result3 = cache_obj.scaled_sum
        mag_result3 = cache_obj.magnitude_xy

        assert call_counts["scaled_sum"] == 2  # Incremented
        assert call_counts["magnitude"] == 2  # Incremented

        # Should allocate new arrays since cache was cleared
        assert scaled_result3 is not scaled_result1
        assert mag_result3 is not mag_result1

        # Test clear_cache when no cache exists (should not error)
        empty_obj = CacheTestData(shape=(1, 1))
        empty_obj.clear_cache()  # Should not raise error

        # Should work normally after clearing empty cache
        empty_obj.set_xyz(
            np.array([[1]], dtype=np.float32),
            np.array([[2]], dtype=np.float32),
            np.array([[3]], dtype=np.float32),
        )
        result = empty_obj.sum_squares
        assert result.shape == (1, 1, 1)

    def test_empty_cache_behavior(self):
        """Test behavior when cache is empty or doesn't exist."""
        cache_obj = CacheTestData(shape=(1, 1))

        # Verify no cache entries initially
        assert "sum_squares" not in cache_obj._store

        x = np.array([[1]], dtype=np.float32)
        y = np.array([[2]], dtype=np.float32)
        z = np.array([[3]], dtype=np.float32)
        cache_obj.set_xyz(x, y, z)

        # First access should pass None as out parameter
        result = cache_obj.sum_squares

        assert result is not None
        assert result.shape == (1, 1, 1)
        assert call_counts["sum_squares"] == 1
        assert allocation_counts["sum_squares"] == 1


class ArrayCacheTestData(StructuredData):
    """Test class for cached_array read-only and reuse functionality."""

    _data_keys = ("x", "y", "z")

    def set_xyz(self, x, y, z):
        self._set_data_by_keys(("x",), x)
        self._set_data_by_keys(("y",), y)
        self._set_data_by_keys(("z",), z)
        return self

    @property
    def xyz(self):
        return self._get_data_by_keys(("x", "y", "z"))

    @cached_array("x", "y", "z")
    def sum_sq(self, out=None):
        track_calls("sum_sq")
        if out is None:
            out = np.empty(self.xyz.shape[:-1], dtype=self.xyz.dtype)
        np.sum(self.xyz**2, axis=-1, out=out)
        return out


def test_cached_property_result_read_only():
    """Test that cached_array results are not writable in-place."""
    cache_obj = ArrayCacheTestData(shape=(2, 2))
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[5, 6], [7, 8]], dtype=np.float32)
    z = np.array([[9, 10], [11, 12]], dtype=np.float32)
    cache_obj.set_xyz(x, y, z)

    result = cache_obj.sum_sq
    assert not result.flags.writeable

    with pytest.raises((ValueError, TypeError)):
        result[...] = 0


def test_cached_property_reuse_after_invalidation():
    """Test that cached_array reuses the array buffer after cache invalidation."""
    reset_counters()
    cache_obj = ArrayCacheTestData(shape=(2, 2))
    x1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y1 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    z1 = np.array([[9, 10], [11, 12]], dtype=np.float32)
    cache_obj.set_xyz(x1, y1, z1)

    result_first = cache_obj.sum_sq
    assert call_counts["sum_sq"] == 1

    # Invalidate cache
    x2 = x1 * 2
    cache_obj._set_data_by_keys(("x",), x2)

    result_second = cache_obj.sum_sq

    # Same array object reused, values updated
    assert result_second is result_first
    assert call_counts["sum_sq"] == 2
    expected = np.sum(np.stack([x2, y1, z1], axis=-1) ** 2, axis=-1)
    assert np.allclose(result_second, expected)
    assert not result_second.flags.writeable


def compute_constant_result(data, out=None):
    """Test function that computes a constant result independent of data."""
    track_calls("constant_result")

    expected_shape = (2, 2, 1)

    if out is not None and out.shape == expected_shape and out.dtype == data.dtype:
        # Reuse existing array
        result = out
    else:
        # Allocate new array
        track_allocation("constant_result")
        result = np.empty(expected_shape, dtype=data.dtype, order="F")

    # Always return the same constant value regardless of input
    result[..., 0] = 42.0

    return result


class NoArgsCacheTestData(StructuredData):
    """Test class for cached property with no arguments."""

    _data_keys = ("x", "y", "z")

    def set_xyz(self, x, y, z):
        """Set x, y, z coordinates."""
        self._set_data_by_keys(("x",), x)
        self._set_data_by_keys(("y",), y)
        self._set_data_by_keys(("z",), z)
        return self

    @property
    def xyz(self):
        """Get combined xyz coordinates."""
        return self._get_data_by_keys(("x", "y", "z"))

    @cached_array()  # No arguments - should never recalculate
    def constant_result(self, out=None):
        """Compute constant result that never depends on data changes."""
        return compute_constant_result(self.xyz, out=out)


def test_cached_property_no_args_never_recalculates():
    """Test that cached_array with no args is never recalculated."""
    reset_counters()

    cache_obj = NoArgsCacheTestData(shape=(2, 2))

    # Set initial data
    x1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y1 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    z1 = np.array([[9, 10], [11, 12]], dtype=np.float32)
    cache_obj.set_xyz(x1, y1, z1)

    # First access - should allocate and compute
    result1 = cache_obj.constant_result

    assert call_counts["constant_result"] == 1
    assert allocation_counts["constant_result"] == 1
    assert result1.shape == (2, 2, 1)
    assert np.all(result1 == 42.0)

    # Second access - should return cached result
    result2 = cache_obj.constant_result
    assert result2 is result1
    assert call_counts["constant_result"] == 1  # Still 1, no additional call

    # Change all data - should STILL return cached result (no dependencies)
    x2 = x1 * 10
    y2 = y1 * 10
    z2 = z1 * 10
    cache_obj.set_xyz(x2, y2, z2)

    # Access again - should STILL return cached result because no args means no dependencies
    result3 = cache_obj.constant_result
    assert result3 is result1  # Same cached object
    assert call_counts["constant_result"] == 1  # Still 1, never recalculated
    assert allocation_counts["constant_result"] == 1  # Still 1, never reallocated

    # Values should still be the original constant
    assert np.all(result3 == 42.0)

    # Even clearing and setting new data should not trigger recalculation
    cache_obj.set_xyz(
        np.ones((2, 2), dtype=np.float32) * 999,
        np.ones((2, 2), dtype=np.float32) * 888,
        np.ones((2, 2), dtype=np.float32) * 777,
    )

    result4 = cache_obj.constant_result
    assert result4 is result1  # Same cached object
    assert call_counts["constant_result"] == 1  # Still 1, never recalculated
    assert np.all(result4 == 42.0)
