"""Tests for Block.copy() patch independence.

This module tests that Block.copy() creates independent deep copies of patches,
ensuring that modifications to patches in the copied block do not affect the
original block.
"""

import numpy as np
import pytest
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.patch import (
    InletPatch,
    OutletPatch,
    PeriodicPatch,
    RotatingPatch,
)


@pytest.fixture
def simple_block():
    """Create a simple block with coordinates and fluid."""
    block = Block(shape=(5, 5, 5))
    x = np.linspace(0, 1, 5)
    r = np.linspace(0.5, 1.5, 5)
    t = np.linspace(0, 0.1, 5)
    xrt = np.stack(np.meshgrid(x, r, t, indexing="ij"), axis=-1)
    block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    block.set_fluid(fluid)
    block.set_P_T(1e5, 300.0)

    return block


class TestBlockCopyPatchCollectionIndependence:
    """Tests for patch collection independence after Block.copy()."""

    def test_patch_collections_are_different_objects(self, simple_block):
        """Test that copied block has a different patch collection object."""
        patch = InletPatch(i=0, label="inlet")
        patch.attach_to_block(simple_block)
        patch.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        simple_block.patches.append(patch)

        block_copy = simple_block.copy()

        # Patch collections should be different objects
        assert simple_block.patches is not block_copy.patches
        assert id(simple_block.patches) != id(block_copy.patches)

    def test_adding_patch_to_copy_does_not_affect_original(self, simple_block):
        """Test that adding a patch to the copy doesn't affect original."""
        patch1 = InletPatch(i=0, label="inlet")
        patch1.attach_to_block(simple_block)
        patch1.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        simple_block.patches.append(patch1)

        block_copy = simple_block.copy()

        # Add a new patch to the copy
        patch2 = OutletPatch(i=-1, label="outlet")
        patch2.attach_to_block(block_copy)
        patch2.set_P(9e4)
        block_copy.patches.append(patch2)

        # Original should still have only 1 patch
        assert len(simple_block.patches) == 1
        assert len(block_copy.patches) == 2

    def test_removing_patch_from_copy_does_not_affect_original(self, simple_block):
        """Test that removing a patch from the copy doesn't affect original."""
        patch1 = InletPatch(i=0, label="inlet")
        patch1.attach_to_block(simple_block)
        patch1.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        simple_block.patches.append(patch1)

        patch2 = OutletPatch(i=-1, label="outlet")
        patch2.attach_to_block(simple_block)
        patch2.set_P(9e4)
        simple_block.patches.append(patch2)

        block_copy = simple_block.copy()

        # Remove a patch from the copy
        block_copy.patches.pop()

        # Original should still have 2 patches
        assert len(simple_block.patches) == 2
        assert len(block_copy.patches) == 1


class TestBlockCopyIndividualPatchIndependence:
    """Tests for individual patch object independence after Block.copy()."""

    def test_individual_patches_are_different_objects(self, simple_block):
        """Test that individual patches are different objects after copy."""
        patch = InletPatch(i=0, label="inlet")
        patch.attach_to_block(simple_block)
        patch.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        simple_block.patches.append(patch)

        block_copy = simple_block.copy()

        # Individual patches should be different objects
        assert simple_block.patches[0] is not block_copy.patches[0]
        assert id(simple_block.patches[0]) != id(block_copy.patches[0])

    def test_modifying_patch_properties_in_copy_does_not_affect_original(
        self, simple_block
    ):
        """Test that modifying patch properties in copy doesn't affect original."""
        patch = OutletPatch(i=-1, label="outlet")
        patch.attach_to_block(simple_block)
        patch.set_P(9e4)
        simple_block.patches.append(patch)

        block_copy = simple_block.copy()

        # Modify the patch in the copy
        block_copy.patches[0].set_P(8e4)

        # Original patch should be unchanged
        assert simple_block.patches[0].P == 9e4
        assert block_copy.patches[0].P == 8e4

    def test_modifying_patch_indices_in_copy_does_not_affect_original(
        self, simple_block
    ):
        """Test that modifying patch indices in copy doesn't affect original."""
        patch = InletPatch(i=0, j=(0, 2), k=(0, 2), label="inlet")
        patch.attach_to_block(simple_block)
        patch.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        simple_block.patches.append(patch)

        block_copy = simple_block.copy()

        # Modify the patch indices in the copy
        block_copy.patches[0].set_j_lim((1, 3))
        block_copy.patches[0].set_k_lim((1, 3))

        # Original patch should be unchanged
        assert simple_block.patches[0].jst == 0
        assert simple_block.patches[0].jen == 2
        assert simple_block.patches[0].kst == 0
        assert simple_block.patches[0].ken == 2

        # Copy should have modified indices
        assert block_copy.patches[0].jst == 1
        assert block_copy.patches[0].jen == 3
        assert block_copy.patches[0].kst == 1
        assert block_copy.patches[0].ken == 3

    def test_modifying_inlet_patch_boundary_conditions_in_copy(self, simple_block):
        """Test that modifying inlet BC in copy doesn't affect original."""
        patch = InletPatch(i=0, label="inlet")
        patch.attach_to_block(simple_block)
        patch.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        simple_block.patches.append(patch)

        block_copy = simple_block.copy()

        # Modify the inlet BC in the copy
        block_copy.patches[0].set_Po_To_Alpha_Beta(Po=1.2e5, To=350, Alpha=10, Beta=5)

        # Original patch should be unchanged
        assert simple_block.patches[0].Po == 1e5
        assert simple_block.patches[0].To == 300
        assert simple_block.patches[0].Alpha == 0
        assert simple_block.patches[0].Beta == 0

        # Copy should have modified BC
        assert block_copy.patches[0].Po == 1.2e5
        assert block_copy.patches[0].To == 350
        assert block_copy.patches[0].Alpha == 10
        assert block_copy.patches[0].Beta == 5

    def test_modifying_rotating_patch_omega_in_copy(self, simple_block):
        """Test that modifying rotating patch Omega in copy doesn't affect original."""
        patch = RotatingPatch(i=0, label="rotating")
        patch.attach_to_block(simple_block)
        patch.set_Omega(100.0)
        simple_block.patches.append(patch)

        block_copy = simple_block.copy()

        # Modify Omega in the copy
        block_copy.patches[0].set_Omega(200.0)

        # Original patch should be unchanged
        assert simple_block.patches[0].Omega == 100.0
        assert block_copy.patches[0].Omega == 200.0


class TestBlockCopyPatchLabels:
    """Tests for patch label independence after Block.copy()."""

    def test_modifying_patch_label_in_copy_does_not_affect_original(self, simple_block):
        """Test that modifying patch label in copy doesn't affect original."""
        patch = InletPatch(i=0, label="inlet")
        patch.attach_to_block(simple_block)
        patch.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        simple_block.patches.append(patch)

        block_copy = simple_block.copy()

        # Modify the label in the copy
        block_copy.patches[0].set_label("modified_inlet")

        # Original patch should be unchanged
        assert simple_block.patches[0].label == "inlet"
        assert block_copy.patches[0].label == "modified_inlet"


class TestBlockCopyMultiplePatches:
    """Tests for multiple patch independence after Block.copy()."""

    def test_copy_with_multiple_patch_types(self, simple_block):
        """Test that copying works correctly with multiple patch types."""
        # Add multiple different patch types
        inlet = InletPatch(i=0, label="inlet")
        inlet.attach_to_block(simple_block)
        inlet.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        simple_block.patches.append(inlet)

        outlet = OutletPatch(i=-1, label="outlet")
        outlet.attach_to_block(simple_block)
        outlet.set_P(9e4)
        simple_block.patches.append(outlet)

        periodic = PeriodicPatch(j=0, label="periodic")
        periodic.attach_to_block(simple_block)
        simple_block.patches.append(periodic)

        block_copy = simple_block.copy()

        # All patches should be different objects
        for i in range(len(simple_block.patches)):
            assert simple_block.patches[i] is not block_copy.patches[i]

        # Modify all patches in the copy
        block_copy.patches[0].set_Po_To_Alpha_Beta(Po=1.1e5, To=310, Alpha=5, Beta=0)
        block_copy.patches[1].set_P(8.5e4)
        block_copy.patches[2].set_label("periodic_modified")

        # Original patches should be unchanged
        assert simple_block.patches[0].Po == 1e5
        assert simple_block.patches[0].To == 300
        assert simple_block.patches[1].P == 9e4
        assert simple_block.patches[2].label == "periodic"


class TestBlockCopyEmptyPatches:
    """Tests for Block.copy() with no patches."""

    def test_copy_block_with_no_patches(self, simple_block):
        """Test that copying a block with no patches works correctly."""
        block_copy = simple_block.copy()

        # Both should have empty patch collections
        assert len(simple_block.patches) == 0
        assert len(block_copy.patches) == 0

        # Patch collections should still be different objects
        assert simple_block.patches is not block_copy.patches

    def test_adding_patch_to_copy_of_empty_block(self, simple_block):
        """Test that adding a patch to copy of empty block doesn't affect original."""
        block_copy = simple_block.copy()

        # Add a patch to the copy
        patch = InletPatch(i=0, label="inlet")
        patch.attach_to_block(block_copy)
        patch.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        block_copy.patches.append(patch)

        # Original should still have no patches
        assert len(simple_block.patches) == 0
        assert len(block_copy.patches) == 1


class TestBlockCopyPatchBlockReference:
    """Tests for patch block reference after Block.copy()."""

    def test_copied_patches_reference_copied_block(self, simple_block):
        """Test that patches in copied block reference the copied block."""
        patch = InletPatch(i=0, label="inlet")
        patch.attach_to_block(simple_block)
        patch.set_Po_To_Alpha_Beta(Po=1e5, To=300, Alpha=0, Beta=0)
        simple_block.patches.append(patch)

        block_copy = simple_block.copy()

        # The patch collection should reference the copied block
        assert block_copy.patches._block is block_copy
        assert simple_block.patches._block is simple_block
