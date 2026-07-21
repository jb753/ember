"""Tests for Grid collection behavior (ember.grid).

Module tested: ember.grid

Test cases:
- test_slice_basic: Basic slice operations
- test_slice_step: Slice with step parameter
- test_slice_negative_indices: Slice with negative indices
- test_slice_empty: Empty slice results
- test_index_tuple_list: Selection with list of indices
- test_index_tuple_tuple: Selection with tuple of indices
- test_index_tuple_single: Selection with single-element tuple/list
- test_index_tuple_duplicates: Selection with duplicate indices raises error
- test_index_tuple_negative: Index tuple with negative indices
- test_mixed_access_preserved: Existing int/str access still works
- test_sliced_collection_modifications: Modifying sliced collection items affects original
- test_sliced_collection_independence: Sliced collections are independent containers
- test_error_handling_invalid_key_type: Error handling for invalid key types
- test_error_handling_mixed_tuple: Error handling for mixed-type tuples
- test_error_handling_out_of_bounds: Error handling for out-of-bounds indices
- test_error_handling_invalid_label: Error handling for non-existent labels
- test_empty_grid_slicing: Slicing operations on empty grid
- test_slice_preserves_item_class: Sliced collections preserve item_class
- test_comprehensive_slice_scenarios: Comprehensive slice scenarios
- TestPatchRelaxationFactor: Inlet/mixing patches carry their own rf attribute
- TestApplyBconds: Grid.apply_bconds applies all BCs using patch rf
"""

import pytest
import numpy as np
import ember.block
import ember.fluid
from ember.grid import Grid
from ember.block import Block
from ember import util
from ember.patch import GridPatchCollection
from ember.patch import PeriodicPatch, InletPatch, OutletPatch, RotatingPatch


class TestLabelledListSlicing:
    """Test slicing and index tuple functionality for LabelledList subclasses."""

    @pytest.fixture
    def sample_grid(self):
        """Create a sample grid with labeled blocks for testing."""
        blocks = []
        labels = ["inlet", "rotor", "stator", "outlet", "mixing"]

        for i, label in enumerate(labels):
            block = Block(shape=(2, 2, 2))
            xrt = util.linmesh3([i, i + 1], [1.0, 2.0], [0.0, 0.1], (2, 2, 2))
            block.set_x(xrt[..., 0])
            block.set_r(xrt[..., 1])
            block.set_t(xrt[..., 2])
            block.set_label(label)
            blocks.append(block)

        return Grid(blocks)

    def test_slice_basic(self, sample_grid):
        """Test basic slice operations."""
        # Test slice[start:stop]
        subset = sample_grid[1:4]
        assert len(subset) == 3
        assert subset[0].label == "rotor"
        assert subset[1].label == "stator"
        assert subset[2].label == "outlet"

        # Verify it returns a new Grid instance
        assert isinstance(subset, Grid)
        assert subset is not sample_grid

    def test_slice_step(self, sample_grid):
        """Test slice with step parameter."""
        # Every other block
        subset = sample_grid[::2]
        assert len(subset) == 3
        assert subset[0].label == "inlet"
        assert subset[1].label == "stator"
        assert subset[2].label == "mixing"

    def test_slice_negative_indices(self, sample_grid):
        """Test slice with negative indices."""
        # Last two blocks
        subset = sample_grid[-2:]
        assert len(subset) == 2
        assert subset[0].label == "outlet"
        assert subset[1].label == "mixing"

        # All but last
        subset = sample_grid[:-1]
        assert len(subset) == 4
        assert subset[-1].label == "outlet"

    def test_slice_empty(self, sample_grid):
        """Test empty slice results."""
        # Empty slice
        subset = sample_grid[5:10]
        assert len(subset) == 0
        assert isinstance(subset, Grid)

    def test_index_tuple_list(self, sample_grid):
        """Test selection with list of indices."""
        # Select specific indices
        subset = sample_grid[[0, 2, 4]]
        assert len(subset) == 3
        assert subset[0].label == "inlet"
        assert subset[1].label == "stator"
        assert subset[2].label == "mixing"

        # Verify it returns a new Grid instance
        assert isinstance(subset, Grid)
        assert subset is not sample_grid

    def test_index_tuple_tuple(self, sample_grid):
        """Test selection with tuple of indices."""
        # Select specific indices using tuple
        subset = sample_grid[(1, 3)]
        assert len(subset) == 2
        assert subset[0].label == "rotor"
        assert subset[1].label == "outlet"

    def test_index_tuple_single(self, sample_grid):
        """Test selection with single-element tuple/list."""
        subset_list = sample_grid[[2]]
        subset_tuple = sample_grid[(2,)]

        assert len(subset_list) == 1
        assert len(subset_tuple) == 1
        assert subset_list[0].label == "stator"
        assert subset_tuple[0].label == "stator"

    def test_index_tuple_duplicates(self, sample_grid):
        """Test selection with duplicate indices raises error due to duplicate labels."""
        # Duplicate indices would create duplicate labels, which violates uniqueness constraint
        with pytest.raises(
            ValueError, match="Duplicate label .* found in initial items"
        ):
            sample_grid[[0, 1, 0]]  # Would create duplicate "inlet" label

    def test_index_tuple_negative(self, sample_grid):
        """Test index tuple with negative indices."""
        subset = sample_grid[[-1, -2]]
        assert len(subset) == 2
        assert subset[0].label == "mixing"
        assert subset[1].label == "outlet"

    def test_mixed_access_preserved(self, sample_grid):
        """Test that existing int/str access still works."""
        # Integer access
        block = sample_grid[0]
        assert block.label == "inlet"

        # String (label) access
        block = sample_grid["rotor"]
        assert block is sample_grid[1]

        # Verify these don't return Grid instances
        assert isinstance(sample_grid[0], Block)
        assert isinstance(sample_grid["rotor"], Block)

    def test_sliced_collection_modifications(self, sample_grid):
        """Test that modifying sliced collection items affects original."""
        # Get a slice
        subset = sample_grid[1:3]

        # Modify a block in the slice
        original_label = subset[0].label
        subset[0].set_label("modified_rotor")

        # Check that original grid reflects the change
        assert sample_grid[1].label == "modified_rotor"
        assert sample_grid["modified_rotor"] is subset[0]

        # Restore original label
        subset[0].set_label(original_label)

    def test_sliced_collection_independence(self, sample_grid):
        """Test that sliced collections are independent containers."""
        subset = sample_grid[1:4]
        original_len = len(sample_grid)

        # Add a new block to subset
        new_block = Block(shape=(2, 2, 2))
        new_block.set_label("new_block")
        subset.append(new_block)

        # Original grid should be unchanged
        assert len(sample_grid) == original_len
        assert len(subset) == 4
        assert "new_block" not in [b.label for b in sample_grid]

    def test_error_handling_invalid_key_type(self, sample_grid):
        """Test error handling for invalid key types."""
        with pytest.raises(TypeError, match="Invalid key type"):
            _ = sample_grid[{"invalid": "dict"}]

        with pytest.raises(TypeError, match="Invalid key type"):
            _ = sample_grid[3.14]  # float

    def test_error_handling_mixed_tuple(self, sample_grid):
        """Test error handling for mixed-type tuples."""
        with pytest.raises(TypeError, match="Invalid key type"):
            _ = sample_grid[[0, "rotor", 2]]  # Mixed int and string

    def test_error_handling_out_of_bounds(self, sample_grid):
        """Test error handling for out-of-bounds indices."""
        with pytest.raises(IndexError):
            _ = sample_grid[10]  # Single index out of bounds

        with pytest.raises(IndexError):
            _ = sample_grid[[0, 10]]  # Index tuple with out-of-bounds

    def test_error_handling_invalid_label(self, sample_grid):
        """Test error handling for non-existent labels."""
        with pytest.raises(KeyError, match="No item found with label 'nonexistent'"):
            _ = sample_grid["nonexistent"]

    def test_empty_grid_slicing(self):
        """Test slicing operations on empty grid."""
        empty_grid = Grid()

        # Empty slices
        subset = empty_grid[:]
        assert len(subset) == 0
        assert isinstance(subset, Grid)

        subset = empty_grid[[]]
        assert len(subset) == 0

        # Out of bounds should still raise errors
        with pytest.raises(IndexError):
            _ = empty_grid[0]

    def test_slice_preserves_item_class(self, sample_grid):
        """Test that sliced collections preserve item_class through Grid constructor."""
        subset = sample_grid[1:3]

        # The item class should be preserved (Grid sets it to Block in constructor)
        # Both original and subset should have the same item class
        assert subset._item_class is sample_grid._item_class

    def test_comprehensive_slice_scenarios(self, sample_grid):
        """Test comprehensive slice scenarios to ensure robustness."""
        # Full slice
        full_copy = sample_grid[:]
        assert len(full_copy) == len(sample_grid)
        assert all(a is b for a, b in zip(full_copy, sample_grid))

        # Reverse slice
        reversed_grid = sample_grid[::-1]
        assert len(reversed_grid) == len(sample_grid)
        assert reversed_grid[0].label == "mixing"
        assert reversed_grid[-1].label == "inlet"

        # Complex step slices
        subset = sample_grid[1::2]
        expected_labels = ["rotor", "outlet"]
        actual_labels = [block.label for block in subset]
        assert actual_labels == expected_labels


"""Tests for GridPatchCollection module (ember.collections).

Module tested: ember.collections.GridPatchCollection

Test cases:
- test_grid_patch_collection_instantiation: GridPatchCollection instantiation
- test_iteration_over_all_patches: Iteration over all patches in collection
- test_len_returns_total_patch_count: Length returns total patch count
- test_contains_patch_objects: Contains operator for patch objects
- test_single_index_access: Single index access to patches
- test_index_out_of_range_raises_error: Index out of range error handling
- test_slice_access: Slice access to patch collections
- test_string_indexing_not_supported: String indexing error handling
- test_invalid_index_type_raises_error: Invalid index type error handling
- test_periodic_patches_property: Periodic patches property filtering
- test_inlet_patches_property: Inlet patches property filtering
- test_outlet_patches_property: Outlet patches property filtering
- test_empty_type_collections: Empty type collections behavior
- test_no_append_method: Verification that append method is not available
- test_no_extend_method: Verification that extend method is not available
- test_no_insert_method: Verification that insert method is not available
- test_no_remove_method: Verification that remove method is not available
- test_no_pop_method: Verification that pop method is not available
- test_no_clear_method: Verification that clear method is not available
- test_no_setitem_support: Verification that setitem is not supported
- test_no_delitem_support: Verification that delitem is not supported
- test_empty_grid_collection: Behavior with empty grid collections
- test_grid_with_blocks_but_no_patches: Grid with blocks but no patches
- test_repr_with_mixed_patches: String representation with mixed patch types
- test_repr_with_empty_collection: String representation of empty collection
- test_grid_patches_property_returns_collection: Grid patches property returns collection
- test_grid_patches_property_updates_with_blocks: Grid patches property updates with blocks
- test_multiple_grids_independent_patch_collections: Multiple grids have independent patch collections
"""


class TestGridPatchCollectionBasics:
    """Test basic GridPatchCollection functionality."""

    def setup_method(self):
        """Set up test grid with multiple blocks containing various patches."""
        # Create three blocks
        shape1 = (5, 6, 7)
        shape2 = (8, 9, 10)
        shape3 = (4, 5, 6)
        self.block1 = ember.block.Block(shape=shape1)
        self.block2 = ember.block.Block(shape=shape2)
        self.block3 = ember.block.Block(shape=shape3)

        # Add fluid and coordinates to blocks (required for operations)
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        for block, shape in [
            (self.block1, shape1),
            (self.block2, shape2),
            (self.block3, shape3),
        ]:
            block.set_fluid(fluid)
            xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
            block.set_x(xrt[..., 0])
            block.set_r(xrt[..., 1])
            block.set_t(xrt[..., 2])

        # Add patches to block1
        self.block1_patches = [
            PeriodicPatch(i=0, j=(1, 3), k=(2, 4), label="block1_periodic1"),
            PeriodicPatch(i=-1, j=(1, 3), k=(2, 4), label="block1_periodic2"),
            InletPatch(i=0, j=(0, -1), k=(0, -1), label="block1_inlet"),
        ]
        for patch in self.block1_patches:
            self.block1.patches.append(patch)

        # Add patches to block2
        self.block2_patches = [
            OutletPatch(i=-1, j=(2, 6), k=(3, 7), label="block2_outlet"),
            PeriodicPatch(i=(1, 6), j=0, k=(1, 8), label="block2_wall1"),
            PeriodicPatch(i=(1, 6), j=-1, k=(1, 8), label="block2_wall2"),
        ]
        for patch in self.block2_patches:
            self.block2.patches.append(patch)

        # Add patches to block3
        self.block3_patches = [
            PeriodicPatch(i=(1, 2), j=0, k=(1, 4), label="block3_periodic"),
            InletPatch(i=0, j=(1, 3), k=(1, 4), label="block3_inlet"),
        ]
        for patch in self.block3_patches:
            self.block3.patches.append(patch)

        # Create grid with all blocks
        self.grid = Grid([self.block1, self.block2, self.block3])
        self.grid_patches = self.grid.patches

    def test_grid_patch_collection_instantiation(self):
        """Test that GridPatchCollection can be created and basic properties work."""
        assert isinstance(self.grid_patches, GridPatchCollection)
        assert len(self.grid_patches) == 8  # Total patches across all blocks

    def test_iteration_over_all_patches(self):
        """Test iteration over all patches from all blocks."""
        all_patches = list(self.grid_patches)
        expected_patches = (
            self.block1_patches + self.block2_patches + self.block3_patches
        )

        assert len(all_patches) == len(expected_patches)

        # Check that all expected patches are present
        for expected_patch in expected_patches:
            assert expected_patch in all_patches

    def test_len_returns_total_patch_count(self):
        """Test that len() returns total patch count across all blocks."""
        expected_total = (
            len(self.block1.patches)
            + len(self.block2.patches)
            + len(self.block3.patches)
        )
        assert len(self.grid_patches) == expected_total
        assert len(self.grid_patches) == 8

    def test_contains_patch_objects(self):
        """Test that __contains__ works for patch objects."""
        # Test patches that should be found
        for patch in self.block1_patches:
            assert patch in self.grid_patches
        for patch in self.block2_patches:
            assert patch in self.grid_patches
        for patch in self.block3_patches:
            assert patch in self.grid_patches

        # Test patch that shouldn't be found
        external_patch = PeriodicPatch(i=0, j=(0, 1), k=(0, 1), label="external")
        assert external_patch not in self.grid_patches


class TestGridPatchCollectionIndexing:
    """Test indexing and slicing functionality."""

    def setup_method(self):
        """Set up test grid for indexing tests."""
        # Create two blocks with known patches
        shape = (5, 5, 5)
        self.block1 = ember.block.Block(shape=shape)
        self.block2 = ember.block.Block(shape=shape)

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
        self.block1.set_fluid(fluid)
        self.block1.set_x(xrt[..., 0])
        self.block1.set_r(xrt[..., 1])
        self.block1.set_t(xrt[..., 2])
        self.block2.set_fluid(fluid)
        self.block2.set_x(xrt[..., 0])
        self.block2.set_r(xrt[..., 1])
        self.block2.set_t(xrt[..., 2])

        # Block1 patches (indices 0, 1, 2)
        self.patch1 = PeriodicPatch(i=0, j=(0, 2), k=(0, 2), label="patch1")
        self.patch2 = InletPatch(i=-1, j=(0, 2), k=(0, 2), label="patch2")
        self.patch3 = OutletPatch(i=(0, 2), j=0, k=(0, 2), label="patch3")

        self.block1.patches.append(self.patch1)
        self.block1.patches.append(self.patch2)
        self.block1.patches.append(self.patch3)

        # Block2 patches (indices 3, 4)
        self.patch4 = OutletPatch(i=0, j=(1, 3), k=(1, 3), label="patch4")
        self.patch5 = PeriodicPatch(i=-1, j=(1, 3), k=(1, 3), label="patch5")

        self.block2.patches.append(self.patch4)
        self.block2.patches.append(self.patch5)

        self.grid = Grid([self.block1, self.block2])
        self.grid_patches = self.grid.patches

    def test_single_index_access(self):
        """Test accessing patches by single index."""
        assert self.grid_patches[0] is self.patch1
        assert self.grid_patches[1] is self.patch2
        assert self.grid_patches[2] is self.patch3
        assert self.grid_patches[3] is self.patch4
        assert self.grid_patches[4] is self.patch5

        # Test negative indexing
        assert self.grid_patches[-1] is self.patch5
        assert self.grid_patches[-2] is self.patch4

    def test_index_out_of_range_raises_error(self):
        """Test that out of range indices raise IndexError."""
        with pytest.raises(IndexError, match="patch index out of range"):
            _ = self.grid_patches[5]

        with pytest.raises(IndexError, match="patch index out of range"):
            _ = self.grid_patches[-6]

    def test_slice_access(self):
        """Test accessing patches by slice."""
        # Test various slicing patterns
        first_three = self.grid_patches[0:3]
        assert first_three == [self.patch1, self.patch2, self.patch3]

        last_two = self.grid_patches[3:5]
        assert last_two == [self.patch4, self.patch5]

        all_patches = self.grid_patches[:]
        assert len(all_patches) == 5

        # Test step slicing
        every_other = self.grid_patches[::2]
        assert every_other == [self.patch1, self.patch3, self.patch5]

    def test_string_indexing_not_supported(self):
        """Test that string indexing raises TypeError."""
        with pytest.raises(TypeError, match="string access not supported"):
            _ = self.grid_patches["patch1"]

    def test_invalid_index_type_raises_error(self):
        """Test that invalid index types raise TypeError."""
        with pytest.raises(TypeError, match="Index must be int or slice"):
            _ = self.grid_patches[1.5]

        with pytest.raises(TypeError, match="Index must be int or slice"):
            _ = self.grid_patches[["patch1"]]


class TestGridPatchCollectionTypeAccess:
    """Test patch type property access (periodic, inlet, outlet)."""

    def setup_method(self):
        """Set up grid with patches of all types."""
        shape = (10, 10, 10)
        self.block1 = ember.block.Block(shape=shape)
        self.block2 = ember.block.Block(shape=shape)

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
        self.block1.set_fluid(fluid)
        self.block1.set_x(xrt[..., 0])
        self.block1.set_r(xrt[..., 1])
        self.block1.set_t(xrt[..., 2])
        self.block2.set_fluid(fluid)
        self.block2.set_x(xrt[..., 0])
        self.block2.set_r(xrt[..., 1])
        self.block2.set_t(xrt[..., 2])

        # Add various patch types to block1
        self.periodic1 = PeriodicPatch(i=0, j=(2, 7), k=(2, 7), label="periodic1")
        self.periodic2 = PeriodicPatch(i=-1, j=(2, 7), k=(2, 7), label="periodic2")
        self.inlet1 = InletPatch(i=(1, 8), j=0, k=(1, 8), label="inlet1")
        self.outlet1 = OutletPatch(i=(1, 8), j=-1, k=(1, 8), label="outlet1")

        self.block1.patches.append(self.periodic1)
        self.block1.patches.append(self.periodic2)
        self.block1.patches.append(self.inlet1)
        self.block1.patches.append(self.outlet1)

        # Add patches to block2
        self.inlet2 = InletPatch(i=0, j=(3, 6), k=(3, 6), label="inlet2")

        self.block2.patches.append(self.inlet2)

        self.grid = Grid([self.block1, self.block2])
        self.grid_patches = self.grid.patches

    def test_periodic_patches_property(self):
        """Test accessing periodic patches across all blocks."""
        periodic_patches = self.grid_patches.periodic

        assert len(periodic_patches) == 2
        assert self.periodic1 in periodic_patches
        assert self.periodic2 in periodic_patches

        # Ensure only PeriodicPatch objects are returned
        for patch in periodic_patches:
            assert isinstance(patch, PeriodicPatch)

    def test_inlet_patches_property(self):
        """Test accessing inlet patches across all blocks."""
        inlet_patches = self.grid_patches.inlet

        assert len(inlet_patches) == 2
        assert self.inlet1 in inlet_patches
        assert self.inlet2 in inlet_patches

        # Ensure only InletPatch objects are returned
        for patch in inlet_patches:
            assert isinstance(patch, InletPatch)

    def test_outlet_patches_property(self):
        """Test accessing outlet patches across all blocks."""
        outlet_patches = self.grid_patches.outlet

        assert len(outlet_patches) == 1
        assert self.outlet1 in outlet_patches

        # Ensure only OutletPatch objects are returned
        for patch in outlet_patches:
            assert isinstance(patch, OutletPatch)

    def test_empty_type_collections(self):
        """Test type properties when no patches of that type exist."""
        # Create grid with only periodic patches
        shape = (5, 5, 5)
        block = ember.block.Block(shape=shape)
        block.set_fluid(
            ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        )
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        periodic = PeriodicPatch(i=0, j=(1, 3), k=(1, 3), label="only_periodic")
        block.patches.append(periodic)

        grid = Grid([block])
        grid_patches = grid.patches

        # Should have periodic patches
        assert len(grid_patches.periodic) == 1

        # Should have empty lists for other types
        assert len(grid_patches.inlet) == 0
        assert len(grid_patches.outlet) == 0
        assert isinstance(grid_patches.inlet, list)
        assert isinstance(grid_patches.outlet, list)


class TestGridPatchCollectionReadOnly:
    """Test that GridPatchCollection is properly read-only."""

    def setup_method(self):
        """Set up minimal grid for read-only tests."""
        shape = (5, 5, 5)
        self.block = ember.block.Block(shape=shape)
        self.block.set_fluid(
            ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        )
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
        self.block.set_x(xrt[..., 0])
        self.block.set_r(xrt[..., 1])
        self.block.set_t(xrt[..., 2])

        patch = PeriodicPatch(i=0, j=(1, 3), k=(1, 3), label="test_patch")
        self.block.patches.append(patch)

        self.grid = Grid([self.block])
        self.grid_patches = self.grid.patches

    def test_no_append_method(self):
        """Test that append method doesn't exist."""
        assert not hasattr(self.grid_patches, "append")

    def test_no_extend_method(self):
        """Test that extend method doesn't exist."""
        assert not hasattr(self.grid_patches, "extend")

    def test_no_insert_method(self):
        """Test that insert method doesn't exist."""
        assert not hasattr(self.grid_patches, "insert")

    def test_no_remove_method(self):
        """Test that remove method doesn't exist."""
        assert not hasattr(self.grid_patches, "remove")

    def test_no_pop_method(self):
        """Test that pop method doesn't exist."""
        assert not hasattr(self.grid_patches, "pop")

    def test_no_clear_method(self):
        """Test that clear method doesn't exist."""
        assert not hasattr(self.grid_patches, "clear")

    def test_no_setitem_support(self):
        """Test that item assignment is not supported."""
        assert not hasattr(self.grid_patches, "__setitem__")

    def test_no_delitem_support(self):
        """Test that item deletion is not supported."""
        assert not hasattr(self.grid_patches, "__delitem__")


class TestGridPatchCollectionEmptyGrid:
    """Test GridPatchCollection with empty grid."""

    def test_empty_grid_collection(self):
        """Test GridPatchCollection behavior with empty grid."""
        empty_grid = Grid([])
        empty_patches = empty_grid.patches

        assert len(empty_patches) == 0
        assert list(empty_patches) == []
        assert len(empty_patches.periodic) == 0
        assert len(empty_patches.inlet) == 0
        assert len(empty_patches.outlet) == 0

    def test_grid_with_blocks_but_no_patches(self):
        """Test GridPatchCollection with blocks that have no patches."""
        block1 = ember.block.Block(shape=(5, 5, 5))
        block2 = ember.block.Block(shape=(6, 6, 6))

        # Set fluid but don't add any patches
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block1.set_fluid(fluid)
        block2.set_fluid(fluid)

        grid = Grid([block1, block2])
        grid_patches = grid.patches

        assert len(grid_patches) == 0
        assert list(grid_patches) == []
        assert len(grid_patches.periodic) == 0
        assert len(grid_patches.inlet) == 0
        assert len(grid_patches.outlet) == 0


class TestGridPatchCollectionStringRepresentation:
    """Test string representation of GridPatchCollection."""

    def test_repr_with_mixed_patches(self):
        """Test __repr__ with various patch types."""
        shape = (10, 10, 10)
        block = ember.block.Block(shape=shape)
        block.set_fluid(
            ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        )
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        # Add various patch types
        block.patches.append(PeriodicPatch(i=0, j=(1, 5), k=(1, 5), label="periodic1"))
        block.patches.append(PeriodicPatch(i=-1, j=(1, 5), k=(1, 5), label="periodic2"))
        block.patches.append(PeriodicPatch(i=(1, 5), j=(1, 5), k=0, label="periodic3"))
        block.patches.append(InletPatch(i=(1, 5), j=0, k=(1, 5), label="inlet1"))
        block.patches.append(OutletPatch(i=(1, 5), j=-1, k=(1, 5), label="outlet1"))

        grid = Grid([block])
        grid_patches = grid.patches

        repr_str = repr(grid_patches)

        # Check that all counts are included
        assert "total=5" in repr_str
        assert "periodic=3" in repr_str
        assert "inlet=1" in repr_str
        assert "outlet=1" in repr_str
        assert "GridPatchCollection" in repr_str

    def test_repr_with_empty_collection(self):
        """Test __repr__ with empty collection."""
        empty_grid = Grid([])
        empty_patches = empty_grid.patches

        repr_str = repr(empty_patches)

        assert "total=0" in repr_str
        assert "periodic=0" in repr_str
        assert "inlet=0" in repr_str
        assert "outlet=0" in repr_str


class TestGridPatchCollectionIntegration:
    """Test integration with Grid class."""

    def test_grid_patches_property_returns_collection(self):
        """Test that grid.patches returns GridPatchCollection."""
        grid = Grid([])
        assert isinstance(grid.patches, GridPatchCollection)

    def test_grid_patches_property_updates_with_blocks(self):
        """Test that grid.patches reflects changes when blocks are added."""
        grid = Grid([])
        assert len(grid.patches) == 0

        # Add block with patches
        shape = (5, 5, 5)
        block = ember.block.Block(shape=shape)
        block.set_fluid(
            ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        )
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape)
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.patches.append(PeriodicPatch(i=0, j=(1, 3), k=(1, 3), label="test"))

        grid.append(block)
        assert len(grid.patches) == 1
        assert len(grid.patches.periodic) == 1

    def test_multiple_grids_independent_patch_collections(self):
        """Test that different grids have independent patch collections."""
        shape1 = (5, 5, 5)
        shape2 = (6, 6, 6)
        block1 = ember.block.Block(shape=shape1)
        block2 = ember.block.Block(shape=shape2)

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block1.set_fluid(fluid)
        _xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape1)
        block1.set_x(_xrt1[..., 0])
        block1.set_r(_xrt1[..., 1])
        block1.set_t(_xrt1[..., 2])
        block2.set_fluid(fluid)
        _xrt2 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], shape2)
        block2.set_x(_xrt2[..., 0])
        block2.set_r(_xrt2[..., 1])
        block2.set_t(_xrt2[..., 2])

        patch1 = PeriodicPatch(i=0, j=(1, 3), k=(1, 3), label="patch1")
        patch2 = InletPatch(i=0, j=(1, 4), k=(1, 4), label="patch2")

        block1.patches.append(patch1)
        block2.patches.append(patch2)

        grid1 = Grid([block1])
        grid2 = Grid([block2])

        # Each grid should have its own patches
        assert len(grid1.patches) == 1
        assert len(grid2.patches) == 1
        assert patch1 in grid1.patches
        assert patch2 in grid2.patches
        assert patch1 not in grid2.patches
        assert patch2 not in grid1.patches


def _make_block():
    """Create a minimal valid Block for testing."""
    block = Block(shape=(2, 2, 2))
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)
    _xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (2, 2, 2))
    block.set_x(_xrt[..., 0])
    block.set_r(_xrt[..., 1])
    block.set_t(_xrt[..., 2])
    return block


class TestApplyRotation:
    """Tests for Grid.apply_rotation()."""

    def test_stationary_sets_omega(self):
        """Stationary row sets Omega on block."""
        block = _make_block()
        grid = Grid([block])
        grid.apply_rotation(["stationary"], [0.0])
        assert block.Omega == 0.0

    def test_stationary_adds_no_patches(self):
        """Stationary row does not add any rotating patches."""
        block = _make_block()
        grid = Grid([block])
        n_before = len(block.patches)
        grid.apply_rotation(["stationary"], [0.0])
        assert len(block.patches) == n_before

    def test_tip_gap_sets_omega(self):
        """tip_gap row sets Omega on block."""
        block = _make_block()
        grid = Grid([block])
        grid.apply_rotation(["tip_gap"], [500.0])
        assert block.Omega == 500.0

    def test_tip_gap_adds_five_rotating_patches(self):
        """tip_gap row adds exactly 5 RotatingPatch objects."""
        block = _make_block()
        grid = Grid([block])
        n_before = len(block.patches)
        grid.apply_rotation(["tip_gap"], [500.0])
        new_patches = block.patches[n_before:]
        assert len(new_patches) == 5
        assert all(isinstance(p, RotatingPatch) for p in new_patches)

    def test_tip_gap_patch_faces(self):
        """tip_gap patches cover i=0, i=-1, j=0, k=0, k=-1."""
        block = _make_block()
        grid = Grid([block])
        n_before = len(block.patches)
        grid.apply_rotation(["tip_gap"], [500.0])
        new_patches = block.patches[n_before:]
        # Each patch has one constant dim; collect (const_dim, start_val) pairs
        face_ids = {(p.const_dim, p._ijk_lim[p.const_dim, 0]) for p in new_patches}
        assert face_ids == {(0, 0), (0, -1), (1, 0), (2, 0), (2, -1)}

    def test_tip_gap_patches_have_correct_omega(self):
        """tip_gap rotating patches carry the specified Omega."""
        block = _make_block()
        grid = Grid([block])
        n_before = len(block.patches)
        grid.apply_rotation(["tip_gap"], [300.0])
        new_patches = block.patches[n_before:]
        assert all(p.Omega == 300.0 for p in new_patches)

    def test_shroud_sets_omega(self):
        """shroud row sets Omega on block."""
        block = _make_block()
        grid = Grid([block])
        grid.apply_rotation(["shroud"], [1000.0])
        assert block.Omega == 1000.0

    def test_shroud_adds_six_rotating_patches(self):
        """shroud row adds exactly 6 RotatingPatch objects."""
        block = _make_block()
        grid = Grid([block])
        n_before = len(block.patches)
        grid.apply_rotation(["shroud"], [1000.0])
        new_patches = block.patches[n_before:]
        assert len(new_patches) == 6
        assert all(isinstance(p, RotatingPatch) for p in new_patches)

    def test_shroud_patch_faces(self):
        """shroud patches cover all six faces."""
        block = _make_block()
        grid = Grid([block])
        n_before = len(block.patches)
        grid.apply_rotation(["shroud"], [1000.0])
        new_patches = block.patches[n_before:]
        face_ids = {(p.const_dim, p._ijk_lim[p.const_dim, 0]) for p in new_patches}
        assert face_ids == {(0, 0), (0, -1), (1, 0), (1, -1), (2, 0), (2, -1)}

    def test_shroud_patches_have_correct_omega(self):
        """shroud rotating patches carry the specified Omega."""
        block = _make_block()
        grid = Grid([block])
        n_before = len(block.patches)
        grid.apply_rotation(["shroud"], [200.0])
        new_patches = block.patches[n_before:]
        assert all(p.Omega == 200.0 for p in new_patches)

    def test_mismatched_lengths_raises(self):
        """Mismatched row_types and Omega lengths raise AssertionError."""
        block = _make_block()
        grid = Grid([block])
        with pytest.raises(AssertionError):
            grid.apply_rotation(["stationary", "tip_gap"], [0.0])

    def test_unknown_row_type_raises(self):
        """Unknown row type raises ValueError."""
        block = _make_block()
        grid = Grid([block])
        with pytest.raises(ValueError, match="Unknown row type"):
            grid.apply_rotation(["rotating"], [100.0])

    def test_multi_block_single_row(self):
        """Blocks connected by periodic patches (same row) both get Omega and patches."""
        block1 = _make_block()
        block2 = _make_block()
        # Connect the two blocks into one row via periodic patches
        block1.patches.append(PeriodicPatch(i=0, label="p1"))
        block2.patches.append(PeriodicPatch(i=0, label="p2"))
        grid = Grid([block1, block2])
        grid.apply_rotation(["shroud"], [400.0])
        for block in [block1, block2]:
            assert block.Omega == 400.0
            rotating = [p for p in block.patches if isinstance(p, RotatingPatch)]
            assert len(rotating) == 6


def _make_flow_block(shape=(5, 6, 8), Nb=30):
    """Block with a uniform flow field and matching pitch, ready for BC apply."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block = Block(shape=shape)
    block.set_fluid(fluid)
    xrt = util.linmesh3([0.0, 1.0], [1.0, 2.0], [0.0, 2 * np.pi / Nb], shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_Nb(Nb)
    block.set_P_rho(1e5 * np.ones(shape), 1.2 * np.ones(shape))
    block.set_Vx(50.0 * np.ones(shape))
    block.set_Vr(np.zeros(shape))
    block.set_Vt(np.zeros(shape))
    return block


class TestPatchRelaxationFactor:
    """Inlet/mixing patches carry their own relaxation factor."""

    def test_inlet_rf_default(self):
        """InletPatch.rf defaults to 1.0, i.e. no relaxation.

        The characteristic solve in apply() makes the face velocity a
        well-conditioned target, so taking it in full is correct; rf < 1 is a
        startup-lag knob rather than a stability crutch.
        """
        assert InletPatch(i=0).rf == 1.0

    def test_inlet_rf_settable_and_copied(self):
        """rf is settable and preserved across a patch copy."""
        block = _make_flow_block()
        patch = InletPatch(i=0)
        block.patches.append(patch)
        patch.attach_to_block(block)
        patch.rf = 0.3
        copy = block.copy()
        assert copy.patches.inlet[0].rf == 0.3


class TestApplyBconds:
    """Tests for Grid.apply_bconds()."""

    def _grid_with_inlet_and_seam(self):
        b1 = _make_flow_block()
        b2 = _make_flow_block()
        # Curved axial pressure profile so the pressure extrapolated to the
        # inlet face from the first two interior nodes differs from the face
        # value; otherwise the rf relaxation term vanishes. The profile must be
        # non-linear: the patch extrapolates as 2*P_1 - P_2, which reproduces a
        # linear field exactly and would leave nothing for rf to relax.
        i_idx = np.arange(b1.shape[0]).reshape(-1, 1, 1)
        P_grad = 1e5 * (1.0 + 0.02 * i_idx**2) * np.ones(b1.shape)
        b1.set_P_rho(P_grad, 1.2 * np.ones(b1.shape))
        b1.patches.append(PeriodicPatch(i=0, j=(1, 4), k=(2, 5)))
        b2.patches.append(PeriodicPatch(i=0, j=(1, 4), k=(2, 5)))
        inlet = InletPatch(i=0)
        b1.patches.append(inlet)
        grid = Grid([b1, b2])
        inlet.attach_to_block(b1)
        inlet.set_Po_To_Alpha_Beta(Po=1.1e5, To=300.0, Alpha=0.0, Beta=0.0)
        return grid, inlet

    def test_uses_patch_rf(self):
        """The inlet rf attribute drives the apply; different rf -> different state."""
        grid0, inlet0 = self._grid_with_inlet_and_seam()
        inlet0.rf = 0.0
        grid0.apply_bconds()
        cons_rf0 = grid0[0].conserved.copy()

        grid1, inlet1 = self._grid_with_inlet_and_seam()
        inlet1.rf = 1.0
        grid1.apply_bconds()
        cons_rf1 = grid1[0].conserved.copy()

        assert not np.allclose(cons_rf0, cons_rf1)
