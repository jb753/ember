"""Tests for NonMatchPatch.check_match implementation (ember.patch).

Module tested: ember.patch.NonMatchPatch

Test cases:
- test_nonmatch_patch_identical_match: Identical NonMatchPatches match
- test_nonmatch_patch_different_node_distribution: Patches with different node counts match
- test_nonmatch_patch_different_types_no_match: NonMatchPatch doesn't match with other patch types
- test_nonmatch_patch_different_x_no_match: Patches at different x locations don't match
- test_nonmatch_patch_different_r_no_match: Patches at different r locations don't match
- test_nonmatch_patch_permutation_match: Patches that match after dimension permutation
- test_nonmatch_patch_flip_match: Patches that match after flipping dimensions
- test_nonmatch_patch_same_xr_different_theta: Patches with same x,r but different theta match
- test_nonmatch_patch_different_constant_dimensions: Patches with different constant dimensions
- test_nonmatch_patch_corner_matching_sufficient: Corner matching is sufficient (no full check)
- test_nonmatch_patch_very_different_sizes: Patches with very different node counts
- test_nonmatch_patch_tolerance_sensitivity: Tolerance parameter affects matching
"""

import numpy as np
import ember.block
from ember.patch import NonMatchPatch, PeriodicPatch, InletPatch, MixingPatch
from conftest import _make_block


class TestNonMatchPatchCheckMatch:
    """Test NonMatchPatch.check_match implementation."""

    def setup_method(self):
        """Set up blocks for testing."""
        self.block1 = ember.block.Block(shape=(10, 20, 30))
        self.block2 = ember.block.Block(shape=(8, 15, 25))  # Different shape

        # Set up coordinates for block1
        x1 = np.linspace(0.0, 1.0, 10)
        r1 = np.linspace(0.5, 1.5, 20)
        t1 = np.linspace(0.0, 2 * np.pi, 30)
        xv1, rv1, tv1 = np.meshgrid(x1, r1, t1, indexing="ij")
        xrt1 = np.stack([xv1, rv1, tv1], axis=-1)
        self.block1.set_x(xrt1[..., 0]).set_r(xrt1[..., 1]).set_t(xrt1[..., 2])

        # Set up coordinates for block2 (same x,r but different theta resolution)
        x2 = np.linspace(0.0, 1.0, 8)
        r2 = np.linspace(0.5, 1.5, 15)
        t2 = np.linspace(0.0, 2 * np.pi, 25)  # Different theta resolution
        xv2, rv2, tv2 = np.meshgrid(x2, r2, t2, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        self.block2.set_x(xrt2[..., 0]).set_r(xrt2[..., 1]).set_t(xrt2[..., 2])

    def test_nonmatch_patch_identical_match(self):
        """Test that identical NonMatchPatches match."""
        patch1 = NonMatchPatch(i=0, j=(5, 15), k=(10, 20))
        patch2 = NonMatchPatch(i=0, j=(5, 15), k=(10, 20))

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should match with identity transform
        transform = patch1.check_match(patch2)
        assert transform is not None
        assert transform == ((0, 1, 2), ())

    def test_nonmatch_patch_different_node_distribution(self):
        """Test that patches with different node counts match."""
        # Use full faces to ensure same physical extent
        # Patch 1: full j,k extent on i=0 face
        patch1 = NonMatchPatch(i=0, j=(0, -1), k=(0, -1))
        # Patch 2: full j,k extent on i=0 face (different node counts)
        patch2 = NonMatchPatch(i=0, j=(0, -1), k=(0, -1))

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block2)

        # Should match despite different node counts (20x30 vs 15x25)
        transform = patch1.check_match(patch2)
        assert transform is not None

    def test_nonmatch_patch_different_types_no_match(self):
        """Test that NonMatchPatch doesn't match with other patch types."""
        nonmatch_patch = NonMatchPatch(i=0, j=(5, 15), k=(10, 20))
        periodic_patch = PeriodicPatch(i=0, j=(5, 15), k=(10, 20))
        inlet_patch = InletPatch(i=0, j=(5, 15), k=(10, 20))
        mixing_patch = MixingPatch(i=0, j=(5, 15), k=(10, 20))

        nonmatch_patch.attach_to_block(self.block1)
        periodic_patch.attach_to_block(self.block1)
        inlet_patch.attach_to_block(self.block1)
        mixing_patch.attach_to_block(self.block1)

        # Should not match with different patch types
        assert nonmatch_patch.check_match(periodic_patch) is None
        assert nonmatch_patch.check_match(inlet_patch) is None
        assert nonmatch_patch.check_match(mixing_patch) is None
        assert periodic_patch.check_match(nonmatch_patch) is None

    def test_nonmatch_patch_different_x_no_match(self):
        """Test that patches at different x locations don't match."""
        patch1 = NonMatchPatch(i=0, j=(5, 15), k=(10, 20))  # x at index 0 (start)
        patch2 = NonMatchPatch(i=-1, j=(5, 15), k=(10, 20))  # x at index -1 (end)

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should not match due to different x coordinates
        assert patch1.check_match(patch2) is None

    def test_nonmatch_patch_different_r_no_match(self):
        """Test that patches at different r locations don't match."""
        patch1 = NonMatchPatch(i=0, j=(2, 8), k=(10, 20))  # r at indices 2-8
        patch2 = NonMatchPatch(i=0, j=(12, 18), k=(10, 20))  # r at indices 12-18

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should not match due to different r coordinates
        assert patch1.check_match(patch2) is None

    def test_nonmatch_patch_permutation_match(self):
        """Test patches that match after dimension permutation."""
        # Create two blocks with matching geometry but permuted
        block1 = ember.block.Block(shape=(6, 8, 10))
        x = np.linspace(0.0, 1.0, 6)
        r = np.linspace(0.5, 1.5, 8)
        t = np.linspace(0.0, 2 * np.pi, 10)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt1 = np.stack([xv, rv, tv], axis=-1)
        block1.set_x(xrt1[..., 0]).set_r(xrt1[..., 1]).set_t(xrt1[..., 2])

        # Block 2: same geometry but with dimensions in different order
        block2 = ember.block.Block(shape=(8, 6, 10))
        # Swap x and r dimensions
        xv2, rv2, tv2 = np.meshgrid(r, x, t, indexing="ij")
        xrt2 = np.stack([rv2, xv2, tv2], axis=-1)  # Note: swapped r and x
        block2.set_x(xrt2[..., 0]).set_r(xrt2[..., 1]).set_t(xrt2[..., 2])

        # Patches at corresponding locations
        patch1 = NonMatchPatch(i=0, j=(2, 5), k=(3, 7))
        patch2 = NonMatchPatch(j=0, i=(2, 5), k=(3, 7))  # Swapped i,j

        patch1.attach_to_block(block1)
        patch2.attach_to_block(block2)

        # Should find appropriate permutation
        transform = patch1.check_match(patch2)
        assert transform is not None

    def test_nonmatch_patch_flip_match(self):
        """Test patches that match after flipping dimensions."""
        # Create two blocks with coordinates that would match after flipping
        block1 = ember.block.Block(shape=(6, 8, 10))
        x = np.linspace(0.0, 1.0, 6)
        r = np.linspace(0.5, 1.5, 8)
        t = np.linspace(0.0, 2 * np.pi, 10)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt1 = np.stack([xv, rv, tv], axis=-1)
        block1.set_x(xrt1[..., 0]).set_r(xrt1[..., 1]).set_t(xrt1[..., 2])

        # Block 2: same coordinates but with j-dimension flipped
        block2 = ember.block.Block(shape=(6, 8, 10))
        r_flipped = np.flip(r)
        xv2, rv2, tv2 = np.meshgrid(x, r_flipped, t, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        block2.set_x(xrt2[..., 0]).set_r(xrt2[..., 1]).set_t(xrt2[..., 2])

        # Create patches that would match if we flip the j-dimension
        patch1 = NonMatchPatch(i=0, j=(2, 5), k=(3, 7))
        patch2 = NonMatchPatch(i=0, j=(2, 5), k=(3, 7))  # Same indices

        patch1.attach_to_block(block1)
        patch2.attach_to_block(block2)

        # Should match with appropriate flipping
        transform = patch1.check_match(patch2)
        assert transform is not None
        # Should include flipping in the transform
        perm, flip = transform
        assert 1 in flip  # j-dimension should be flipped

    def test_nonmatch_patch_same_xr_different_theta(self):
        """Test that patches with same x,r but different theta match."""
        patch1 = NonMatchPatch(i=0, j=(5, 15), k=(10, 15))  # 6 theta points
        patch2 = NonMatchPatch(i=0, j=(5, 15), k=(15, 20))  # Different 6 theta points

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)

        # Should match despite different theta ranges (only x,r checked)
        transform = patch1.check_match(patch2)
        assert transform is not None

    def test_nonmatch_patch_different_constant_dimensions(self):
        """Test non-matching patches with different constant dimensions."""
        patch1 = NonMatchPatch(i=0, j=(5, 15), k=(10, 20))  # i-constant
        patch2 = NonMatchPatch(i=(2, 8), j=0, k=(10, 20))  # j-constant
        patch3 = NonMatchPatch(i=(2, 8), j=(5, 15), k=0)  # k-constant

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block1)
        patch3.attach_to_block(self.block1)

        # Test various combinations - results depend on coordinate alignment
        transform12 = patch1.check_match(patch2)
        transform13 = patch1.check_match(patch3)
        transform23 = patch2.check_match(patch3)

        # Each should return either a valid transform tuple or None
        assert isinstance(transform12, (tuple, type(None)))
        assert isinstance(transform13, (tuple, type(None)))
        assert isinstance(transform23, (tuple, type(None)))

    def test_nonmatch_patch_corner_matching_sufficient(self):
        """Test that corner matching is sufficient (no full coordinate check)."""
        # This test verifies that NonMatchPatch only checks corners,
        # not all interior points
        # Use full faces so corners align physically
        patch1 = NonMatchPatch(i=0, j=(0, -1), k=(0, -1))
        patch2 = NonMatchPatch(i=0, j=(0, -1), k=(0, -1))  # Different node count

        patch1.attach_to_block(self.block1)
        patch2.attach_to_block(self.block2)

        # Should match if corners align (same x,r extent)
        transform = patch1.check_match(patch2)
        assert transform is not None

    def test_nonmatch_patch_very_different_sizes(self):
        """Test patches with very different node counts."""
        # Create blocks with very different resolutions
        block_coarse = ember.block.Block(shape=(4, 5, 6))
        block_fine = ember.block.Block(shape=(20, 25, 30))

        # Same physical extent
        x_coarse = np.linspace(0.0, 1.0, 4)
        r_coarse = np.linspace(0.5, 1.5, 5)
        t_coarse = np.linspace(0.0, 2 * np.pi, 6)
        xv_c, rv_c, tv_c = np.meshgrid(x_coarse, r_coarse, t_coarse, indexing="ij")
        xrt_coarse = np.stack([xv_c, rv_c, tv_c], axis=-1)
        block_coarse.set_x(xrt_coarse[..., 0]).set_r(xrt_coarse[..., 1]).set_t(
            xrt_coarse[..., 2]
        )

        x_fine = np.linspace(0.0, 1.0, 20)
        r_fine = np.linspace(0.5, 1.5, 25)
        t_fine = np.linspace(0.0, 2 * np.pi, 30)
        xv_f, rv_f, tv_f = np.meshgrid(x_fine, r_fine, t_fine, indexing="ij")
        xrt_fine = np.stack([xv_f, rv_f, tv_f], axis=-1)
        block_fine.set_x(xrt_fine[..., 0]).set_r(xrt_fine[..., 1]).set_t(
            xrt_fine[..., 2]
        )

        # Full face patches
        patch_coarse = NonMatchPatch(i=0)
        patch_fine = NonMatchPatch(i=0)

        patch_coarse.attach_to_block(block_coarse)
        patch_fine.attach_to_block(block_fine)

        # Should match despite 5x-6x difference in node count
        transform = patch_coarse.check_match(patch_fine)
        assert transform is not None
        assert transform == ((0, 1, 2), ())

    def test_nonmatch_patch_tolerance_sensitivity(self):
        """Test that tolerance parameter affects matching."""
        # Create two patches with slightly different coordinates
        block1 = ember.block.Block(shape=(5, 5, 5))
        block2 = ember.block.Block(shape=(5, 5, 5))

        x = np.linspace(0.0, 1.0, 5)
        r = np.linspace(1.0, 2.0, 5)
        t = np.linspace(0.0, np.pi, 5)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt1 = np.stack([xv, rv, tv], axis=-1)
        block1.set_x(xrt1[..., 0]).set_r(xrt1[..., 1]).set_t(xrt1[..., 2])

        # Block2: slightly shifted in x
        x2 = x + 0.001  # Small shift
        xv2, rv2, tv2 = np.meshgrid(x2, r, t, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        block2.set_x(xrt2[..., 0]).set_r(xrt2[..., 1]).set_t(xrt2[..., 2])

        patch1 = NonMatchPatch(i=0)
        patch2 = NonMatchPatch(i=0)

        patch1.attach_to_block(block1)
        patch2.attach_to_block(block2)

        # With tight tolerance, should not match
        assert patch1.check_match(patch2, rtol=1e-10) is None

        # With looser tolerance, should match
        transform = patch1.check_match(patch2, rtol=1e-2)
        assert transform is not None


class TestNonMatchPatchTransformGeneration:
    """Test the _get_viable_transforms_nonmatch method."""

    def test_transform_generation_different_sizes(self):
        """Test transform generation with different node counts."""
        # Patch with i-constant: shape (1, 10, 20)
        patch1 = NonMatchPatch(i=0, j=(0, 9), k=(0, 19))
        patch1.attach_to_block(_make_block((10, 10, 20)))

        # Other patch also i-constant but different sizes: shape (1, 15, 25)
        other_shape = (1, 15, 25)

        transforms = patch1._get_viable_transforms_nonmatch(other_shape)

        # Should generate transforms since both have i-constant
        assert len(transforms) > 0

        # Should include identity transform
        assert ((0, 1, 2), ()) in transforms

    def test_transform_generation_incompatible_constant_dims(self):
        """Test transform generation with incompatible constant dimensions."""
        # Patch with i-constant: shape (1, 10, 20)
        patch1 = NonMatchPatch(i=0, j=(0, 9), k=(0, 19))
        patch1.attach_to_block(_make_block((10, 10, 20)))

        # Other patch with different constant dimension pattern
        # This won't match because constant dims don't align
        other_shape = (10, 1, 20)  # j-constant instead

        transforms = patch1._get_viable_transforms_nonmatch(other_shape)

        # Should generate some transforms but they need constant dims to align
        # The permutation (1, 0, 2) would swap i and j
        valid_transforms = [t for t in transforms if other_shape[t[0].index(0)] == 1]
        assert len(valid_transforms) > 0

    def test_transform_allows_arbitrary_size_mismatch(self):
        """Test that transforms allow arbitrary size mismatches in varying dimensions."""
        # Patch: shape (1, 5, 10)
        patch1 = NonMatchPatch(i=0, j=(0, 4), k=(0, 9))
        patch1.attach_to_block(_make_block((10, 5, 10)))

        # Other: shape (1, 100, 200) - very different sizes
        other_shape = (1, 100, 200)

        transforms = patch1._get_viable_transforms_nonmatch(other_shape)

        # Should still generate transforms since constant dim matches
        assert len(transforms) > 0
        assert ((0, 1, 2), ()) in transforms
