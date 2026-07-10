"""Tests for GridConnectivity (ember.grid).

Module tested: ember.grid

Test cases:
- test_grid_connectivity_instantiation: GridConnectivity can be created
- test_grid_connectivity_property_access: Accessing connectivity through manager
- test_periodic_property_empty_grid: Pair method with empty grid
- test_periodic_property_no_periodic_patches: Pair method with blocks that have no periodic patches
- test_single_periodic_connection: Connectivity with one pair of matching periodic patches
- test_multiple_periodic_connections_same_blocks: Connectivity with multiple periodic patches on same blocks
- test_three_block_periodic_connectivity: Connectivity across three blocks
- test_non_matching_periodic_patches: Non-matching periodic patches raise ValueError
- test_single_periodic_patch_no_connection: Single periodic patch has no connections
- test_pairing_is_cached: Pair method caches; clear() invalidates
- test_clear_reflects_grid_changes: Pair method reflects changes after clear()
- test_connectivity_not_pickled: EMB round-trip drops cache and re-pairs lazily
- test_empty_grid_connectivity: Connectivity with completely empty grid
- test_grid_with_non_periodic_patches_only: Connectivity with grid containing only non-periodic patches
- test_grid_connectivity_instantiation_with_patch_class: GridConnectivity can be instantiated with patch class
- test_repeated_access_is_cached: Repeated access returns cached instance until clear()
- test_connectivity_manager_pair_method: GridConnectivityManager.pair() method
"""

import numpy as np
import pytest
import ember.block
import ember.fluid
from ember.grid import Grid, GridConnectivity
from ember.patch import PeriodicPatch


class TestGridConnectivityBasics:
    """Test basic GridConnectivity functionality."""

    def setup_method(self):
        """Set up test grid with multiple blocks."""
        # Create two blocks with periodic patches that should connect
        self.block1 = ember.block.Block(shape=(5, 6, 7))
        self.block2 = ember.block.Block(shape=(5, 6, 7))

        # Add fluid to blocks (required for some operations)
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        self.block1.set_fluid(fluid)
        self.block2.set_fluid(fluid)

        # Set identical coordinates for both blocks to enable matching
        x = np.linspace(0, 1, 5)
        r = np.linspace(1, 2, 6)
        t = np.linspace(0, np.pi, 7)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt = np.stack([xv, rv, tv], axis=-1)

        self.block1.set_x(xrt[..., 0])
        self.block1.set_r(xrt[..., 1])
        self.block1.set_t(xrt[..., 2])
        self.block2.set_x(xrt[..., 0])
        self.block2.set_r(xrt[..., 1])
        self.block2.set_t(xrt[..., 2])

    def test_grid_connectivity_instantiation(self):
        """Test that GridConnectivity can be created."""
        grid = Grid([self.block1, self.block2])
        connectivity = grid.connectivity.periodic
        assert isinstance(connectivity, GridConnectivity)

    def test_grid_connectivity_property_access(self):
        """Test accessing connectivity through manager."""
        grid = Grid([self.block1, self.block2])
        connectivity = grid.connectivity.periodic
        assert isinstance(connectivity, GridConnectivity)

    def test_periodic_property_empty_grid(self):
        """Test pair method with empty grid."""
        empty_grid = Grid([])
        connectivity = empty_grid.connectivity.periodic
        assert connectivity.pair() == {}

    def test_periodic_property_no_periodic_patches(self):
        """Test pair method with blocks that have no periodic patches."""
        grid = Grid([self.block1, self.block2])
        connectivity = grid.connectivity.periodic
        assert connectivity.pair() == {}


class TestPeriodicConnectivity:
    """Test periodic patch connectivity detection."""

    def setup_method(self):
        """Set up test blocks with periodic patches."""
        self.block1 = ember.block.Block(shape=(5, 6, 8))
        self.block2 = ember.block.Block(shape=(5, 6, 8))

        # Set fluid
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        self.block1.set_fluid(fluid)
        self.block2.set_fluid(fluid)

        # Set identical coordinates to enable patch matching
        x = np.linspace(0, 1, 5)
        r = np.linspace(1, 2, 6)
        t = np.linspace(0, 2 * np.pi, 8)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt = np.stack([xv, rv, tv], axis=-1)

        self.block1.set_x(xrt[..., 0])
        self.block1.set_r(xrt[..., 1])
        self.block1.set_t(xrt[..., 2])
        self.block2.set_x(xrt[..., 0])
        self.block2.set_r(xrt[..., 1])
        self.block2.set_t(xrt[..., 2])

    def test_single_periodic_connection(self):
        """Test connectivity with one pair of matching periodic patches."""
        # Add matching periodic patches to both blocks
        patch1 = PeriodicPatch(i=0, j=(1, 4), k=(2, 6), label="patch1")
        patch2 = PeriodicPatch(i=0, j=(1, 4), k=(2, 6), label="patch2")  # Identical

        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)

        grid = Grid([self.block1, self.block2])
        connectivity = grid.connectivity.periodic

        matches = connectivity.pair()
        assert len(matches) == 2  # Both directions should be in dict

        # Check forward direction: (0,0) -> (1,0)
        assert (0, 0) in matches
        matching_patch, transform = matches[(0, 0)]
        assert matching_patch == (1, 0)
        assert transform is not None

        # Check reverse direction: (1,0) -> (0,0)
        assert (1, 0) in matches
        matching_patch_rev, reverse_transform = matches[(1, 0)]
        assert matching_patch_rev == (0, 0)
        assert reverse_transform is not None

    def test_multiple_periodic_connections_same_blocks(self):
        """Test connectivity with multiple periodic patches on same blocks."""
        # Add multiple matching periodic patch pairs
        patch1a = PeriodicPatch(i=0, j=(1, 3), k=(2, 4), label="patch1a")
        patch1b = PeriodicPatch(i=-1, j=(1, 3), k=(2, 4), label="patch1b")

        patch2a = PeriodicPatch(
            i=0, j=(1, 3), k=(2, 4), label="patch2a"
        )  # Matches patch1a
        patch2b = PeriodicPatch(
            i=-1, j=(1, 3), k=(2, 4), label="patch2b"
        )  # Matches patch1b

        self.block1.patches.append(patch1a)
        self.block1.patches.append(patch1b)
        self.block2.patches.append(patch2a)
        self.block2.patches.append(patch2b)

        grid = Grid([self.block1, self.block2])
        connectivity = grid.connectivity.periodic

        matches = connectivity.pair()
        assert len(matches) == 4  # 2 pairs × 2 directions each

        # Check that all expected patch pairs are present
        expected_keys = {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert set(matches.keys()) == expected_keys

        # Verify bidirectional connectivity
        # patch1a (0,0) <-> patch2a (1,0)
        assert matches[(0, 0)][0] == (1, 0)
        assert matches[(1, 0)][0] == (0, 0)

        # patch1b (0,1) <-> patch2b (1,1)
        assert matches[(0, 1)][0] == (1, 1)
        assert matches[(1, 1)][0] == (0, 1)

        # Check transforms are present
        for key in matches:
            _, transform = matches[key]
            assert transform is not None

    def test_three_block_periodic_connectivity(self):
        """Test connectivity across three blocks."""
        block3 = ember.block.Block(shape=(5, 6, 8))
        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block3.set_fluid(fluid)

        # Set same coordinates
        x = np.linspace(0, 1, 5)
        r = np.linspace(1, 2, 6)
        t = np.linspace(0, 2 * np.pi, 8)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt = np.stack([xv, rv, tv], axis=-1)
        block3.set_x(xrt[..., 0])
        block3.set_r(xrt[..., 1])
        block3.set_t(xrt[..., 2])

        # Add matching patches to all three blocks
        patch1 = PeriodicPatch(i=0, j=(2, 4), k=(3, 5), label="p1")
        patch2 = PeriodicPatch(i=0, j=(2, 4), k=(3, 5), label="p2")
        patch3 = PeriodicPatch(i=0, j=(2, 4), k=(3, 5), label="p3")

        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)
        block3.patches.append(patch3)

        grid = Grid([self.block1, self.block2, block3])
        connectivity = grid.connectivity.periodic

        # With 3 identical patches, one will be left unpaired and should raise an error
        with pytest.raises(ValueError, match="Unmatched:"):
            _ = connectivity.pair()

    def test_non_matching_periodic_patches(self):
        """Test that non-matching periodic patches raise ValueError."""
        # Add non-matching periodic patches
        patch1 = PeriodicPatch(i=0, j=(1, 3), k=(2, 4), label="patch1")
        patch2 = PeriodicPatch(
            i=0, j=(2, 4), k=(3, 5), label="patch2"
        )  # Different position

        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)

        grid = Grid([self.block1, self.block2])
        connectivity = grid.connectivity.periodic

        # Should raise error for unmatched patches
        with pytest.raises(ValueError, match="Unmatched:"):
            _ = connectivity.pair()

    def test_single_periodic_patch_no_connection(self):
        """Test that single periodic patch has no connections."""
        patch1 = PeriodicPatch(i=0, j=(1, 4), k=(2, 5), label="lonely_patch")
        self.block1.patches.append(patch1)

        grid = Grid([self.block1])
        connectivity = grid.connectivity.periodic

        pairs = connectivity.pair()
        assert len(pairs) == 0


class TestGridConnectivityUpdate:
    """Test update functionality of GridConnectivity."""

    def setup_method(self):
        """Set up test blocks."""
        self.block1 = ember.block.Block(shape=(5, 6, 8))
        self.block2 = ember.block.Block(shape=(5, 6, 8))

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        self.block1.set_fluid(fluid)
        self.block2.set_fluid(fluid)

        # Set coordinates
        x = np.linspace(0, 1, 5)
        r = np.linspace(1, 2, 6)
        t = np.linspace(0, 2 * np.pi, 8)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt = np.stack([xv, rv, tv], axis=-1)

        self.block1.set_x(xrt[..., 0])
        self.block1.set_r(xrt[..., 1])
        self.block1.set_t(xrt[..., 2])
        self.block2.set_x(xrt[..., 0])
        self.block2.set_r(xrt[..., 1])
        self.block2.set_t(xrt[..., 2])

    def test_pairing_is_cached(self):
        """pair() is cached; topology changes are only seen after clear()."""
        # Initial setup with no patches
        grid = Grid([self.block1, self.block2])
        connectivity = grid.connectivity.periodic

        # First call should return empty (and is cached)
        assert len(connectivity.pair()) == 0

        # Add patches after first access
        patch1 = PeriodicPatch(i=0, j=(1, 4), k=(2, 5), label="p1")
        patch2 = PeriodicPatch(i=0, j=(1, 4), k=(2, 5), label="p2")
        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)

        # Cached empty result is retained until explicitly invalidated
        assert len(connectivity.pair()) == 0

        # After clearing the cache the new connectivity is detected
        grid.connectivity.clear()
        assert len(grid.connectivity.periodic.pair()) == 2  # Both directions

    def test_clear_reflects_grid_changes(self):
        """pair() reflects removed patches after the cache is cleared."""
        # Start with matching patches
        patch1 = PeriodicPatch(i=0, j=(1, 4), k=(2, 5), label="p1")
        patch2 = PeriodicPatch(i=0, j=(1, 4), k=(2, 5), label="p2")
        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)

        grid = Grid([self.block1, self.block2])
        connectivity = grid.connectivity.periodic

        # Should have one connection (both directions)
        assert len(connectivity.pair()) == 2

        # Remove patches
        self.block1.patches.clear()
        self.block2.patches.clear()

        # Cached result retained until invalidated, then reflects the change
        assert len(connectivity.pair()) == 2
        grid.connectivity.clear()
        assert len(grid.connectivity.periodic.pair()) == 0

    def test_connectivity_not_pickled(self, tmp_path):
        """EMB round-trip drops the cached manager and re-pairs lazily."""
        patch1 = PeriodicPatch(i=0, j=(1, 4), k=(2, 5), label="p1")
        patch2 = PeriodicPatch(i=0, j=(1, 4), k=(2, 5), label="p2")
        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)

        grid = Grid([self.block1, self.block2])
        # Populate the cache before writing
        assert len(grid.connectivity.periodic.pair()) == 2
        assert grid._connectivity is not None

        emb = tmp_path / "roundtrip.emb"
        grid.write_emb(str(emb))

        reloaded = Grid.read_emb(str(emb))
        # Cache is not carried through the pickle
        assert reloaded._connectivity is None
        # ...and rebuilds cleanly on access
        assert len(reloaded.connectivity.periodic.pair()) == 2


class TestGridConnectivityEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_grid_connectivity(self):
        """Test connectivity with completely empty grid."""
        empty_grid = Grid([])
        connectivity = empty_grid.connectivity.periodic

        assert len(connectivity.pair()) == 0

    def test_grid_with_non_periodic_patches_only(self):
        """Test connectivity with grid containing only non-periodic patches."""
        block = ember.block.Block(shape=(5, 5, 5))
        block.set_fluid(
            ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        )

        # Add non-periodic patches (use non-revolution types since block has no xrt)
        from ember.patch import CoolingPatch, InviscidPatch, PeriodicPatch

        block.patches.append(CoolingPatch(i=0, j=(1, 3), k=(1, 3)))
        block.patches.append(InviscidPatch(i=-1, j=(1, 3), k=(1, 3)))
        block.patches.append(PeriodicPatch(i=(1, 3), j=0, k=(1, 3)))

        grid = Grid([block])
        connectivity = grid.connectivity.periodic

        # Should have no periodic connections
        assert len(connectivity.pair()) == 0


class TestGridConnectivityIntegration:
    """Test integration with Grid class."""

    def test_grid_connectivity_instantiation_with_patch_class(self):
        """Test that GridConnectivity can be instantiated with patch class."""
        grid = Grid([])
        connectivity = grid.connectivity.periodic
        assert isinstance(connectivity, GridConnectivity)
        assert connectivity.patch_class == PeriodicPatch

    def test_repeated_access_is_cached(self):
        """Repeated access returns the same cached GridConnectivity instance."""
        grid = Grid([])

        connectivity1 = grid.connectivity.periodic
        connectivity2 = grid.connectivity.periodic

        # Cached: same instance each time, until clear()
        assert connectivity1 is connectivity2
        assert isinstance(connectivity1, GridConnectivity)

        grid.connectivity.clear()
        assert grid.connectivity.periodic is not connectivity1

    def test_connectivity_manager_pair_method(self):
        """Test GridConnectivityManager.pair() method."""
        # Create blocks with periodic patches
        block1 = ember.block.Block(shape=(5, 5, 5))
        block2 = ember.block.Block(shape=(5, 5, 5))

        fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
        block1.set_fluid(fluid)
        block2.set_fluid(fluid)

        # Set identical coordinates
        x = np.linspace(0, 1, 5)
        r = np.linspace(1, 2, 5)
        t = np.linspace(0, np.pi, 5)
        xv, rv, tv = np.meshgrid(x, r, t, indexing="ij")
        xrt = np.stack([xv, rv, tv], axis=-1)

        block1.set_x(xrt[..., 0])
        block1.set_r(xrt[..., 1])
        block1.set_t(xrt[..., 2])
        block2.set_x(xrt[..., 0])
        block2.set_r(xrt[..., 1])
        block2.set_t(xrt[..., 2])

        patch1 = PeriodicPatch(i=0, j=(1, 3), k=(1, 3), label="p1")
        patch2 = PeriodicPatch(i=0, j=(1, 3), k=(1, 3), label="p2")
        block1.patches.append(patch1)
        block2.patches.append(patch2)

        grid = Grid([block1, block2])

        # Test manager's pair method
        all_matches = grid.connectivity.pair()
        periodic_matches = grid.connectivity.periodic.pair()

        # Should be the same since we only have periodic patches
        assert all_matches == periodic_matches
        assert len(all_matches) == 2


class TestGridMixingConnectivity:
    """Test mixing patch connectivity functionality."""

    def setup_method(self):
        """Set up test grid with mixing patches."""

        # Create two blocks with compatible coordinates
        self.block1 = ember.block.Block(shape=(5, 8, 12))
        self.block2 = ember.block.Block(shape=(5, 8, 16))  # Different k-size

        # Set up coordinates for block1
        x1 = np.linspace(0.0, 1.0, 5)
        r1 = np.linspace(0.5, 1.5, 8)
        t1 = np.linspace(0.0, 2 * np.pi, 12)
        xv1, rv1, tv1 = np.meshgrid(x1, r1, t1, indexing="ij")
        xrt1 = np.stack([xv1, rv1, tv1], axis=-1)
        self.block1.set_x(xrt1[..., 0])
        self.block1.set_r(xrt1[..., 1])
        self.block1.set_t(xrt1[..., 2])

        # Set up coordinates for block2 (same x,r but different theta resolution)
        x2 = np.linspace(0.0, 1.0, 5)
        r2 = np.linspace(0.5, 1.5, 8)
        t2 = np.linspace(0.0, 2 * np.pi, 16)  # Different theta resolution
        xv2, rv2, tv2 = np.meshgrid(x2, r2, t2, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        self.block2.set_x(xrt2[..., 0])
        self.block2.set_r(xrt2[..., 1])
        self.block2.set_t(xrt2[..., 2])

        # Create grid
        self.grid = Grid()
        self.block1.set_label("block1")
        self.block2.set_label("block2")
        self.grid.append(self.block1)
        self.grid.append(self.block2)

    def test_mixing_connectivity_manager_property(self):
        """Test that mixing connectivity manager is accessible."""
        assert hasattr(self.grid.connectivity, "mixing")
        mixing_manager = self.grid.connectivity.mixing
        assert mixing_manager is not None

        # Should be a GridConnectivity instance
        from ember.grid import GridConnectivity

        assert isinstance(mixing_manager, GridConnectivity)

    def test_mixing_patch_pairing(self):
        """Test basic mixing patch pairing functionality."""
        from ember.patch import MixingPatch

        # Create two blocks with the same coordinates to ensure patches can match
        # Make both blocks have identical x,r coordinates
        x = np.linspace(0.0, 1.0, 5)
        r = np.linspace(0.5, 1.5, 8)
        t1 = np.linspace(0.0, 2 * np.pi, 12)
        t2 = np.linspace(0.0, 2 * np.pi, 16)

        # Block1 coordinates
        xv1, rv1, tv1 = np.meshgrid(x, r, t1, indexing="ij")
        xrt1 = np.stack([xv1, rv1, tv1], axis=-1)
        self.block1.set_x(xrt1[..., 0])
        self.block1.set_r(xrt1[..., 1])
        self.block1.set_t(xrt1[..., 2])

        # Block2 coordinates (same x,r)
        xv2, rv2, tv2 = np.meshgrid(x, r, t2, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        self.block2.set_x(xrt2[..., 0])
        self.block2.set_r(xrt2[..., 1])
        self.block2.set_t(xrt2[..., 2])

        # Add matching mixing patches at block boundaries
        patch1 = MixingPatch(i=4, j=(2, 6), k=(3, 8))  # End boundary of block1
        patch2 = MixingPatch(i=4, j=(2, 6), k=(5, 10))  # End boundary of block2

        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)

        # Get mixing patch pairs
        pairs = self.grid.connectivity.mixing.pair()

        # Should find one pair
        assert len(pairs) == 2  # Both directions

        # Check that patches are paired correctly
        assert (0, 0) in pairs  # block1, patch0
        assert (1, 0) in pairs  # block2, patch0

        # Check bidirectional pairing
        pair_info_1 = pairs[(0, 0)]
        pair_info_2 = pairs[(1, 0)]

        assert pair_info_1[0] == (1, 0)  # Points to block2, patch0
        assert pair_info_2[0] == (0, 0)  # Points to block1, patch0

        # Both should have transform information
        assert pair_info_1[1] is not None
        assert pair_info_2[1] is not None

    def test_combined_connectivity_pairing(self):
        """Test that grid.connectivity.pair() includes mixing patches."""
        from ember.patch import MixingPatch

        # Create matching coordinates like in the main pairing test
        x = np.linspace(0.0, 1.0, 5)
        r = np.linspace(0.5, 1.5, 8)
        t1 = np.linspace(0.0, 2 * np.pi, 12)
        t2 = np.linspace(0.0, 2 * np.pi, 16)

        # Set same x,r coordinates
        xv1, rv1, tv1 = np.meshgrid(x, r, t1, indexing="ij")
        xrt1 = np.stack([xv1, rv1, tv1], axis=-1)
        self.block1.set_x(xrt1[..., 0])
        self.block1.set_r(xrt1[..., 1])
        self.block1.set_t(xrt1[..., 2])

        xv2, rv2, tv2 = np.meshgrid(x, r, t2, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        self.block2.set_x(xrt2[..., 0])
        self.block2.set_r(xrt2[..., 1])
        self.block2.set_t(xrt2[..., 2])

        # Add matching mixing patches
        patch1 = MixingPatch(i=4, j=(2, 6), k=(3, 8))
        patch2 = MixingPatch(i=4, j=(2, 6), k=(5, 10))  # Same i,j coordinates

        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)

        # Get all patch pairs (should include mixing)
        all_pairs = self.grid.connectivity.pair()

        # Should include the mixing patch pairs
        assert len(all_pairs) >= 2  # At least the mixing patches
        assert (0, 0) in all_pairs
        assert (1, 0) in all_pairs

    def test_mixing_patches_different_k_sizes(self):
        """Test mixing patches with different k-dimension sizes."""
        from ember.patch import MixingPatch

        # Set up same coordinates like in main test
        x = np.linspace(0.0, 1.0, 5)
        r = np.linspace(0.5, 1.5, 8)
        t1 = np.linspace(0.0, 2 * np.pi, 12)
        t2 = np.linspace(0.0, 2 * np.pi, 16)

        xv1, rv1, tv1 = np.meshgrid(x, r, t1, indexing="ij")
        xrt1 = np.stack([xv1, rv1, tv1], axis=-1)
        self.block1.set_x(xrt1[..., 0])
        self.block1.set_r(xrt1[..., 1])
        self.block1.set_t(xrt1[..., 2])

        xv2, rv2, tv2 = np.meshgrid(x, r, t2, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        self.block2.set_x(xrt2[..., 0])
        self.block2.set_r(xrt2[..., 1])
        self.block2.set_t(xrt2[..., 2])

        # Create patches with same x,r but different k ranges/sizes
        patch1 = MixingPatch(i=4, j=(2, 6), k=(3, 8))  # 5 theta points
        patch2 = MixingPatch(i=4, j=(2, 6), k=(4, 12))  # 8 theta points, same i,j

        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)

        # Should still pair successfully
        pairs = self.grid.connectivity.mixing.pair()
        assert len(pairs) == 2
        assert (0, 0) in pairs
        assert (1, 0) in pairs

    def test_mixing_patches_no_match(self):
        """Test that non-matching mixing patches raise error."""
        from ember.patch import MixingPatch

        # Create patches with different x,r coordinates
        patch1 = MixingPatch(i=4, j=(2, 6), k=(3, 8))  # End of block1
        patch2 = MixingPatch(i=0, j=(1, 5), k=(5, 10))  # Different j range

        self.block1.patches.append(patch1)
        self.block2.patches.append(patch2)

        # Should raise error for unmatched patches
        with pytest.raises(ValueError, match="Unmatched"):
            self.grid.connectivity.mixing.pair()

    def test_mixing_connectivity_with_no_patches(self):
        """Test mixing connectivity when no mixing patches exist."""
        # Don't add any mixing patches
        pairs = self.grid.connectivity.mixing.pair()

        # Should return empty dictionary
        assert pairs == {}

    def test_mixing_connectivity_single_patch(self):
        """Test mixing connectivity with only one mixing patch."""
        from ember.patch import MixingPatch

        # Add only one mixing patch
        patch1 = MixingPatch(i=4, j=(2, 6), k=(3, 8))
        self.block1.patches.append(patch1)

        # Should return empty dictionary (no pairs possible)
        pairs = self.grid.connectivity.mixing.pair()
        assert pairs == {}

    def test_mixing_separate_from_periodic(self):
        """Test that mixing and periodic patches are handled separately."""
        from ember.patch import MixingPatch

        # Set up same coordinates
        x = np.linspace(0.0, 1.0, 5)
        r = np.linspace(0.5, 1.5, 8)
        t1 = np.linspace(0.0, 2 * np.pi, 12)
        t2 = np.linspace(0.0, 2 * np.pi, 16)

        xv1, rv1, tv1 = np.meshgrid(x, r, t1, indexing="ij")
        xrt1 = np.stack([xv1, rv1, tv1], axis=-1)
        self.block1.set_x(xrt1[..., 0])
        self.block1.set_r(xrt1[..., 1])
        self.block1.set_t(xrt1[..., 2])

        xv2, rv2, tv2 = np.meshgrid(x, r, t2, indexing="ij")
        xrt2 = np.stack([xv2, rv2, tv2], axis=-1)
        self.block2.set_x(xrt2[..., 0])
        self.block2.set_r(xrt2[..., 1])
        self.block2.set_t(xrt2[..., 2])

        # Add mixing patches
        mixing1 = MixingPatch(i=4, j=(2, 6), k=(3, 8))
        mixing2 = MixingPatch(i=4, j=(2, 6), k=(5, 10))  # Same i,j

        # Add periodic patches at k boundaries (block1 has k=0-11, block2 has k=0-15)
        periodic1 = PeriodicPatch(k=0, i=(1, 4), j=(2, 6))
        periodic2 = PeriodicPatch(
            k=15, i=(1, 4), j=(2, 6)
        )  # Matching periodic at other block's end

        self.block1.patches.extend([mixing1, periodic1])
        self.block2.patches.extend([mixing2, periodic2])

        # Get separate connectivity
        mixing_pairs = self.grid.connectivity.mixing.pair()
        periodic_pairs = self.grid.connectivity.periodic.pair()

        # Should have pairs for both types
        assert len(mixing_pairs) == 2  # Mixing patches paired
        assert len(periodic_pairs) == 2  # Periodic patches paired

        # Combined should have all pairs
        all_pairs = self.grid.connectivity.pair()
        assert len(all_pairs) == 4  # Both mixing and periodic pairs
