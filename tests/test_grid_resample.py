"""Tests for Grid resampling functionality (ember.grid).

Module tested: ember.grid

Test cases:
- test_resample_single_block: Resampling a grid with a single block
- test_resample_multiple_blocks: Resampling a grid with multiple blocks
- test_resample_with_patches: Resampling preserves patches correctly
- test_resample_scalar_factor: Resampling with scalar factor applied to all dimensions
- test_resample_tuple_factors: Resampling with different factors per dimension
- test_resample_empty_grid: Resampling an empty grid
- test_resample_preserves_block_metadata: Resampling preserves block metadata
- test_resample_method_chaining: Resample returns grid for method chaining
- test_resample_returns_new_grid: Resample returns a new Grid, does not mutate original
- test_resample_maintains_grid_structure: Resampling maintains grid structure and ordering
- test_resample_turbomachinery_grid: Resampling a typical turbomachinery grid setup
"""

import numpy as np
import pytest
from ember.grid import Grid
from ember.block import Block
from ember.block_restart import BlockRestart, make_restart, apply_restart
from ember.patch import InletPatch, OutletPatch, PeriodicPatch, MixingPatch
from ember.fluid import PerfectFluid
from ember import util


def _make_state_block(shape, P=101325.0, T=300.0, Vx=50.0, mu_turb=1e-4):
    """Create a block with fluid, coordinates and uniform flow state."""
    block = Block(shape=shape)
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    block.set_fluid(fluid)
    ni, nj, nk = shape
    x = np.linspace(0.0, 1.0, ni).reshape(-1, 1, 1) * np.ones(shape)
    r = np.linspace(0.5, 1.0, nj).reshape(1, -1, 1) * np.ones(shape)
    t = np.linspace(0.0, 0.1, nk).reshape(1, 1, -1) * np.ones(shape)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)
    block.set_P_T(P, T)
    block.set_Vx(Vx)
    block.set_Vr(0.0)
    block.set_Vt(0.0)
    block.set_mu_turb(mu_turb * np.ones(shape))
    return block


class TestGridResample:
    """Test cases for Grid.resample method."""

    def test_resample_single_block(self):
        """Test resampling a grid with a single block."""
        # Create a simple 3x3x3 block
        block = Block(shape=(3, 3, 3))
        xrt = util.linmesh3([0.0, 2.0], [1.0, 3.0], [0.0, 1.0], (3, 3, 3))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        # Create grid
        grid = Grid([block])
        original_shape = grid[0].shape

        # Resample with factor 2.0 (should increase resolution)
        resampled_grid = grid.resample(2.0)

        # Check that the method returns a new grid
        assert resampled_grid is not grid

        # Check that block was resampled (should have more points)
        new_shape = resampled_grid[0].shape
        assert all(new_shape[i] >= original_shape[i] for i in range(3))

        # Check that coordinates are still within bounds
        x_bounds = [xrt[..., 0].min(), xrt[..., 0].max()]
        r_bounds = [xrt[..., 1].min(), xrt[..., 1].max()]
        t_bounds = [xrt[..., 2].min(), xrt[..., 2].max()]

        new_x = resampled_grid[0].x
        new_r = resampled_grid[0].r
        new_t = resampled_grid[0].t

        assert new_x.min() >= x_bounds[0] and new_x.max() <= x_bounds[1]
        assert new_r.min() >= r_bounds[0] and new_r.max() <= r_bounds[1]
        assert new_t.min() >= t_bounds[0] and new_t.max() <= t_bounds[1]

    def test_resample_multiple_blocks(self):
        """Test resampling a grid with multiple blocks."""
        # Create multiple blocks with different shapes
        block1 = Block(shape=(4, 3, 2))
        block2 = Block(shape=(3, 4, 3))

        xrt1 = util.linmesh3([0.0, 1.0], [1.0, 2.0], [0.0, 0.5], (4, 3, 2))
        xrt2 = util.linmesh3([1.0, 2.0], [2.0, 3.0], [0.1, 0.6], (3, 4, 3))

        block1.set_x(xrt1[..., 0])
        block1.set_r(xrt1[..., 1])
        block1.set_t(xrt1[..., 2])
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])

        # Create grid
        grid = Grid([block1, block2])
        original_shapes = [block.shape for block in grid]

        # Resample with factor 0.7 (should decrease resolution)
        grid = grid.resample(0.7)

        # Check that all blocks were resampled
        new_shapes = [block.shape for block in grid]

        for i, (orig_shape, new_shape) in enumerate(zip(original_shapes, new_shapes)):
            # Should have fewer points for downsampling
            assert all(new_shape[j] <= orig_shape[j] for j in range(3)), (
                f"Block {i} not properly downsampled"
            )

        # Check grid still has same number of blocks
        assert len(grid) == 2

    def test_resample_with_patches(self):
        """Test resampling preserves patches correctly."""
        # Create block with patches
        block = Block(shape=(5, 4, 3))
        xrt = util.linmesh3([0.0, 2.0], [1.0, 2.0], [0.0, 1.0], (5, 4, 3))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        # Add patches
        inlet_patch = InletPatch(i=0, j=(1, 2), k=(0, 1), label="inlet")
        outlet_patch = OutletPatch(i=-1, j=(1, 2), k=(0, 1), label="outlet")
        periodic_patch = PeriodicPatch(i=(1, 3), j=0, k=(0, -1), label="periodic")

        block.patches.extend([inlet_patch, outlet_patch, periodic_patch])

        # Create grid
        grid = Grid([block])
        original_patch_count = len(grid[0].patches)

        # Resample
        grid = grid.resample(1.5)

        # Check patches are preserved
        assert len(grid[0].patches) == original_patch_count

        # Check patch labels are preserved
        labels = [p.label for p in grid[0].patches]
        assert "inlet" in labels
        assert "outlet" in labels
        assert "periodic" in labels

        # Check patch types are preserved
        types = [type(p).__name__ for p in grid[0].patches]
        assert "InletPatch" in types
        assert "OutletPatch" in types
        assert "PeriodicPatch" in types

    def test_resample_scalar_factor(self):
        """Test resampling with scalar factor applied to all dimensions."""
        block = Block(shape=(4, 4, 4))
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 1.0], (4, 4, 4))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        grid = Grid([block])

        # Resample with scalar factor
        factor = 1.8
        grid = grid.resample(factor)

        new_shape = grid[0].shape

        # All dimensions should be affected by the same factor
        # (exact new size depends on block.resample implementation)
        assert new_shape[0] > 4 and new_shape[1] > 4 and new_shape[2] > 4

    def test_resample_tuple_factors(self):
        """Test resampling with different factors per dimension."""
        block = Block(shape=(6, 4, 3))
        xrt = util.linmesh3([0.0, 2.0], [0.5, 1.5], [0.0, 0.5], (6, 4, 3))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        grid = Grid([block])
        original_shape = grid[0].shape

        # Resample with different factors per dimension
        factors = (0.5, 1.0, 2.0)  # decrease i, keep j same, increase k
        grid = grid.resample(factors)

        new_shape = grid[0].shape

        # Check directional effects (approximate due to resampling algorithm)
        assert new_shape[0] <= original_shape[0], "i-direction should be downsampled"
        assert new_shape[2] >= original_shape[2], "k-direction should be upsampled"

    def test_resample_empty_grid(self):
        """Test resampling an empty grid."""
        grid = Grid([])

        # Should not raise an error
        result = grid.resample(2.0)

        # Should return a new grid
        assert result is not grid
        assert len(result) == 0

    def test_resample_preserves_block_metadata(self):
        """Test that resampling preserves block metadata."""
        block = Block(shape=(3, 3, 3))
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 1.0], (3, 3, 3))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        # Add some metadata
        block._metadata["test_key"] = "test_value"
        block._metadata["number"] = 42

        grid = Grid([block])

        # Resample
        grid = grid.resample(1.2)

        # Check metadata is preserved
        assert grid[0]._metadata["test_key"] == "test_value"
        assert grid[0]._metadata["number"] == 42

    def test_resample_method_chaining(self):
        """Test that resample returns a grid that supports method chaining."""
        block = Block(shape=(3, 3, 3))
        xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 1.0], (3, 3, 3))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        grid = Grid([block])

        # Method chaining should work and return a Grid
        result = grid.resample(1.1).resample(0.9)

        assert isinstance(result, Grid)
        assert len(result) == 1

    def test_resample_returns_new_grid(self):
        """Test that resample returns a new Grid and does not mutate the original."""
        block = Block(shape=(4, 3, 2))
        xrt = util.linmesh3([0.0, 2.0], [1.0, 3.0], [0.0, 1.0], (4, 3, 2))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        grid = Grid([block])
        original_shape = grid[0].shape

        # Resample
        resampled = grid.resample(1.5)

        # Should be a different object
        assert resampled is not grid
        assert len(resampled) == 1

        # Original grid should be unchanged
        assert grid[0].shape == original_shape

    def test_resample_maintains_grid_structure(self):
        """Test resampling maintains grid structure and ordering."""
        block1 = Block(shape=(3, 3, 3))
        block2 = Block(shape=(3, 3, 3))

        xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 1.0], (3, 3, 3))
        xrt2 = util.linmesh3([1.0, 2.0], [0.5, 1.5], [0.0, 1.0], (3, 3, 3))

        block1.set_x(xrt1[..., 0])
        block1.set_r(xrt1[..., 1])
        block1.set_t(xrt1[..., 2])
        block2.set_x(xrt2[..., 0])
        block2.set_r(xrt2[..., 1])
        block2.set_t(xrt2[..., 2])

        # Create grid
        grid = Grid([block1, block2])

        # Store original coordinate bounds for verification
        orig_x1_bounds = [block1.x.min(), block1.x.max()]
        orig_x2_bounds = [block2.x.min(), block2.x.max()]

        # Resample
        grid = grid.resample(1.2)

        # Check grid structure is preserved
        assert len(grid) == 2

        # Check coordinate bounds are preserved
        new_x1_bounds = [grid[0].x.min(), grid[0].x.max()]
        new_x2_bounds = [grid[1].x.min(), grid[1].x.max()]

        np.testing.assert_allclose(new_x1_bounds, orig_x1_bounds, rtol=1e-5)
        np.testing.assert_allclose(new_x2_bounds, orig_x2_bounds, rtol=1e-5)


class TestGridResampleValues:
    """Tests that verify interpolated coordinate values after resampling."""

    def test_resample_linear_mesh_corner_values(self):
        """Corner coordinates must be exactly preserved after resampling."""
        block = Block(shape=(5, 4, 3))
        xrt = util.linmesh3([0.0, 4.0], [1.0, 4.0], [0.0, 2.0], (5, 4, 3))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        grid = Grid([block])

        resampled = grid.resample(2.0)
        b = resampled[0]

        # All eight corners must be exactly preserved
        np.testing.assert_allclose(b.x[0, 0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(b.x[-1, 0, 0], 4.0, atol=1e-5)
        np.testing.assert_allclose(b.r[0, 0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(b.r[0, -1, 0], 4.0, atol=1e-5)
        np.testing.assert_allclose(b.t[0, 0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(b.t[0, 0, -1], 2.0, atol=1e-5)

    def test_resample_linear_mesh_interior_values(self):
        """For a linear mesh, every resampled coordinate must lie on the original
        linear function — i.e. x = a + b*i_frac, etc."""
        ni, nj, nk = 5, 5, 5
        block = Block(shape=(ni, nj, nk))
        xrt = util.linmesh3([0.0, 1.0], [2.0, 3.0], [10.0, 20.0], (ni, nj, nk))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        grid = Grid([block])

        resampled = grid.resample(2.0)
        b = resampled[0]

        ni_new, nj_new, nk_new = b.shape

        # x varies only with i, r only with j, t only with k on a linmesh
        # Check every point against the expected linear interpolant
        i_frac = np.arange(ni_new) / (ni_new - 1)
        j_frac = np.arange(nj_new) / (nj_new - 1)
        k_frac = np.arange(nk_new) / (nk_new - 1)

        expected_x = i_frac[:, None, None] * np.ones((1, nj_new, nk_new))
        expected_r = 2.0 + j_frac[None, :, None] * np.ones((ni_new, 1, nk_new))
        expected_t = 10.0 + k_frac[None, None, :] * 10.0 * np.ones((ni_new, nj_new, 1))

        np.testing.assert_allclose(b.x, expected_x, atol=1e-4)
        np.testing.assert_allclose(b.r, expected_r, atol=1e-4)
        np.testing.assert_allclose(b.t, expected_t, atol=1e-4)

    def test_resample_downsample_preserves_linear_values(self):
        """Downsampling a linear mesh should also give exactly linear values."""
        ni, nj, nk = 9, 9, 9
        block = Block(shape=(ni, nj, nk))
        xrt = util.linmesh3([0.0, 8.0], [1.0, 9.0], [0.0, 8.0], (ni, nj, nk))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        grid = Grid([block])

        resampled = grid.resample(0.5)
        b = resampled[0]

        ni_new, nj_new, nk_new = b.shape

        i_frac = np.arange(ni_new) / (ni_new - 1)
        j_frac = np.arange(nj_new) / (nj_new - 1)
        k_frac = np.arange(nk_new) / (nk_new - 1)

        expected_x = i_frac[:, None, None] * 8.0 * np.ones((1, nj_new, nk_new))
        expected_r = 1.0 + j_frac[None, :, None] * 8.0 * np.ones((ni_new, 1, nk_new))
        expected_t = k_frac[None, None, :] * 8.0 * np.ones((ni_new, nj_new, 1))

        np.testing.assert_allclose(b.x, expected_x, atol=1e-4)
        np.testing.assert_allclose(b.r, expected_r, atol=1e-4)
        np.testing.assert_allclose(b.t, expected_t, atol=1e-4)

    def test_resample_downsamples_wdist_at_coincident_nodes(self):
        """Downsampling must carry wall distance onto the coarse grid, sampling it
        exactly at the node-coincident coarse nodes.

        ``resample(0.5)`` halves each dimension node-coincidentally (a 9-node edge
        becomes the 5 even-indexed fine nodes), so the coarse wall distance must
        equal the fine field at those nodes. This is the property full-multigrid
        relies on when it resamples a grid carrying ``wdist`` -- e.g. before a TS3
        run, whose input file requires wall distance on every level. A non-linear
        field is used so the check is only satisfied by exact node-coincident
        sampling; a linear field would pass even under interpolation.
        """
        ni, nj, nk = 9, 9, 9
        block = Block(shape=(ni, nj, nk))
        block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
        xrt = util.linmesh3([0.0, 8.0], [1.0, 9.0], [0.0, 8.0], (ni, nj, nk))
        x, r, t = xrt[..., 0], xrt[..., 1], xrt[..., 2]
        block.set_x(x)
        block.set_r(r)
        block.set_t(t)

        # Strictly-positive (set_wdist requires it), non-linear wall-distance field.
        wdist_fine = (1.0 + 0.1 * (x**2 + r**2 + t**2)).astype(np.float32)
        block.set_wdist(wdist_fine)

        coarse = Grid([block]).resample(0.5)[0]

        # 9 -> 5 per dimension, with coarse node i coincident with fine node 2*i.
        assert coarse.shape == (5, 5, 5)
        np.testing.assert_allclose(coarse.x, x[::2, ::2, ::2], rtol=1e-5)
        np.testing.assert_allclose(
            coarse.wdist, wdist_fine[::2, ::2, ::2], rtol=1e-5, atol=1e-6
        )

    def test_resample_nonlinear_mesh_interpolation(self):
        """Interior points of a curved mesh must match bilinear interpolation
        from the original grid — checked at a single known interior point."""
        ni, nj, nk = 3, 3, 3
        block = Block(shape=(ni, nj, nk))

        # Build a mesh where x = i^2 / (ni-1)^2, others linear
        i_idx = np.arange(ni)
        j_idx = np.arange(nj)
        k_idx = np.arange(nk)
        x = (i_idx[:, None, None] / (ni - 1)) ** 2 * np.ones((1, nj, nk))
        r = 1.0 + j_idx[None, :, None] / (nj - 1) * np.ones((ni, 1, nk))
        t = k_idx[None, None, :] / (nk - 1) * np.ones((ni, nj, 1))

        xrt = np.stack([x, r, t], axis=-1).astype(np.float32)
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        grid = Grid([block])

        resampled = grid.resample(2.0)
        b = resampled[0]

        # Corners must be exactly preserved
        np.testing.assert_allclose(b.x[0, 0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(b.x[-1, 0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(b.r[0, 0, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(b.r[0, -1, 0], 2.0, atol=1e-5)
        np.testing.assert_allclose(b.t[0, 0, 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(b.t[0, 0, -1], 1.0, atol=1e-5)

        # The middle resampled i-index maps to original index 1 (a grid point), so x = (1/2)^2 = 0.25
        mid_i = b.shape[0] // 2
        np.testing.assert_allclose(b.x[mid_i, 0, 0], 0.25, atol=1e-4)


class TestGridResampleIntegration:
    """Integration tests for Grid.resample with real-world scenarios."""

    def test_resample_turbomachinery_grid(self):
        """Test resampling a typical turbomachinery grid setup."""
        # Create a simplified turbomachinery-like grid
        ni, nj, nk = 10, 8, 15  # Axial, radial, circumferential

        block = Block(shape=(ni, nj, nk))

        # Create cylindrical-like coordinates
        x_vals = np.linspace(0.0, 1.0, ni)
        r_vals = np.linspace(0.5, 1.0, nj)
        t_vals = np.linspace(0.0, 2 * np.pi / nk, nk)  # Small circumferential sector

        x, r, t = np.meshgrid(x_vals, r_vals, t_vals, indexing="ij")
        xrt = np.stack([x, r, t], axis=-1)
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        # Add typical turbomachinery patches
        inlet = InletPatch(i=0, j=(1, -2), k=(1, -2), label="inlet")
        outlet = OutletPatch(i=-1, j=(1, -2), k=(1, -2), label="outlet")
        hub = PeriodicPatch(i=(1, -2), j=0, k=(1, -2), label="hub")
        shroud = PeriodicPatch(i=(1, -2), j=-1, k=(1, -2), label="shroud")

        block.patches.extend([inlet, outlet, hub, shroud])

        grid = Grid([block])

        # Resample with different factors for each direction
        # Higher resolution in circumferential direction
        factors = (1.0, 1.0, 2.0)  # Keep axial/radial, increase circumferential
        grid = grid.resample(factors)

        # Check that resampling worked
        new_shape = grid[0].shape
        assert new_shape[2] > nk, "Circumferential resolution should increase"

        # Check patches are still there and functional
        assert len(grid[0].patches) == 4
        patch_labels = [p.label for p in grid[0].patches]
        assert all(
            label in patch_labels for label in ["inlet", "outlet", "hub", "shroud"]
        )

        # Check that patch shapes can still be accessed (tests patch resampling)
        for patch in grid[0].patches:
            shape = patch.shape  # This would fail if patch indices weren't updated
            assert all(s > 0 for s in shape), f"Invalid patch shape: {shape}"


class TestGridInterpFrom:
    """Tests for Grid.interp_from method."""

    def test_same_shape_copies_conserved(self):
        """Same-shape interp_from copies conserved variables exactly."""
        src = _make_state_block((4, 4, 4), P=2e5, T=350.0, Vx=100.0, mu_turb=2e-4)
        tgt = _make_state_block((4, 4, 4))

        Grid([tgt]).interp_from(Grid([src]))

        np.testing.assert_allclose(tgt.conserved, src.conserved, rtol=1e-5)
        np.testing.assert_allclose(tgt.mu_turb, src.mu_turb, rtol=1e-5)

    def test_same_shape_different_fluid_references(self):
        """Same-shape copy still correct when src and tgt have different reference scales."""
        src = _make_state_block((4, 4, 4), P=2e5, T=350.0)

        # Target with a different fluid (different V_ref, rho_ref)
        tgt = Block(shape=(4, 4, 4))
        fluid2 = PerfectFluid(cp=1004.0, gamma=1.35, mu=2e-5, Pr=0.71)
        tgt.set_fluid(fluid2)
        ni, nj, nk = (4, 4, 4)
        x = np.linspace(0.0, 1.0, ni).reshape(-1, 1, 1) * np.ones((4, 4, 4))
        r = np.linspace(0.5, 1.0, nj).reshape(1, -1, 1) * np.ones((4, 4, 4))
        t = np.linspace(0.0, 0.1, nk).reshape(1, 1, -1) * np.ones((4, 4, 4))
        tgt.set_x(x)
        tgt.set_r(r)
        tgt.set_t(t)
        tgt.set_P_T(101325.0, 300.0)
        tgt.set_mu_turb(np.ones((4, 4, 4)) * 1e-4)

        Grid([tgt]).interp_from(Grid([src]))

        # Dimensional conserved variables must match
        np.testing.assert_allclose(tgt.conserved, src.conserved, rtol=1e-4)
        np.testing.assert_allclose(tgt.mu_turb, src.mu_turb, rtol=1e-4)

    def test_upsample_constant_field(self):
        """Upsampling a uniform state gives the same uniform state everywhere."""
        P, T, Vx, mu = 2e5, 400.0, 80.0, 3e-4
        src = _make_state_block((3, 3, 3), P=P, T=T, Vx=Vx, mu_turb=mu)
        tgt = _make_state_block((6, 6, 6), P=101325.0, T=300.0)

        Grid([tgt]).interp_from(Grid([src]))

        ref = src.conserved[0, 0, 0]
        np.testing.assert_allclose(
            tgt.conserved, np.broadcast_to(ref, tgt.conserved.shape), rtol=1e-4
        )
        np.testing.assert_allclose(tgt.mu_turb, mu, rtol=1e-4)

    def test_downsample_constant_field(self):
        """Downsampling a uniform state gives the same uniform state everywhere."""
        P, T, Vx, mu = 1.5e5, 320.0, 50.0, 5e-4
        src = _make_state_block((8, 8, 8), P=P, T=T, Vx=Vx, mu_turb=mu)
        tgt = _make_state_block((3, 3, 3), P=101325.0, T=300.0)

        Grid([tgt]).interp_from(Grid([src]))

        ref = src.conserved[0, 0, 0]
        np.testing.assert_allclose(
            tgt.conserved, np.broadcast_to(ref, tgt.conserved.shape), rtol=1e-4
        )
        np.testing.assert_allclose(tgt.mu_turb, mu, rtol=1e-4)

    def test_interpolation_linear_field(self):
        """Interpolated values lie on the original linear field."""
        ni, nj, nk = 5, 5, 5
        src = Block(shape=(ni, nj, nk))
        fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
        src.set_fluid(fluid)
        x = np.linspace(0.0, 1.0, ni).reshape(-1, 1, 1) * np.ones((ni, nj, nk))
        r = np.linspace(0.5, 1.0, nj).reshape(1, -1, 1) * np.ones((ni, nj, nk))
        t = np.linspace(0.0, 0.1, nk).reshape(1, 1, -1) * np.ones((ni, nj, nk))
        src.set_x(x)
        src.set_r(r)
        src.set_t(t)

        # Linear axial velocity field: Vx varies linearly with i-index
        i_frac = np.linspace(0.0, 1.0, ni).reshape(-1, 1, 1) * np.ones((ni, nj, nk))
        Vx_src = 50.0 + 100.0 * i_frac
        src.set_P_T(101325.0, 300.0)
        src.set_Vx(Vx_src)
        src.set_Vr(np.zeros((ni, nj, nk)))
        src.set_Vt(np.zeros((ni, nj, nk)))
        src.set_mu_turb(np.zeros((ni, nj, nk)))

        ni2, nj2, nk2 = 9, 5, 5
        tgt = Block(shape=(ni2, nj2, nk2))
        tgt.set_fluid(fluid)
        x2 = np.linspace(0.0, 1.0, ni2).reshape(-1, 1, 1) * np.ones((ni2, nj2, nk2))
        r2 = np.linspace(0.5, 1.0, nj2).reshape(1, -1, 1) * np.ones((ni2, nj2, nk2))
        t2 = np.linspace(0.0, 0.1, nk2).reshape(1, 1, -1) * np.ones((ni2, nj2, nk2))
        tgt.set_x(x2)
        tgt.set_r(r2)
        tgt.set_t(t2)
        tgt.set_P_T(101325.0, 300.0)
        tgt.set_mu_turb(np.zeros((ni2, nj2, nk2)))

        Grid([tgt]).interp_from(Grid([src]))

        i_frac2 = np.linspace(0.0, 1.0, ni2).reshape(-1, 1, 1) * np.ones(
            (ni2, nj2, nk2)
        )
        expected_Vx = 50.0 + 100.0 * i_frac2
        np.testing.assert_allclose(tgt.Vx, expected_Vx, atol=1e-3)

    def test_multi_block_grid(self):
        """interp_from works block-by-block on a multi-block grid."""
        src1 = _make_state_block((4, 4, 4), P=2e5, T=350.0)
        src2 = _make_state_block((4, 4, 4), P=3e5, T=400.0)
        tgt1 = _make_state_block((6, 6, 6))
        tgt2 = _make_state_block((6, 6, 6))

        Grid([tgt1, tgt2]).interp_from(Grid([src1, src2]))

        # Both target blocks should have the uniform state from their source
        ref1 = src1.conserved[0, 0, 0]
        ref2 = src2.conserved[0, 0, 0]
        np.testing.assert_allclose(
            tgt1.conserved, np.broadcast_to(ref1, tgt1.conserved.shape), rtol=1e-4
        )
        np.testing.assert_allclose(
            tgt2.conserved, np.broadcast_to(ref2, tgt2.conserved.shape), rtol=1e-4
        )

    def test_critical_index_mismatch_raises(self):
        """Mismatched number of critical indices raises ValueError."""
        # src inlet covers j=(1, 3) -> critical j indices [0, 1, 3, 5]
        src = _make_state_block((4, 6, 4))
        src.patches.append(InletPatch(i=0, j=(1, 3), k=(0, -1)))

        # tgt has no patches -> critical j indices [0, 7] only (2 vs 4)
        tgt = _make_state_block((4, 8, 4))

        import pytest

        with pytest.raises(ValueError, match="critical indices"):
            Grid([tgt]).interp_from(Grid([src]))

    def test_critical_index_aware_coords(self):
        """Patch j-boundary in src lands exactly at the corresponding boundary in tgt."""
        fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)

        # src: 5 j-points, inlet patch j=(1,3) -> critical j indices [0, 1, 3, 4]
        # Set a linearly varying Vx field along j so we can check alignment
        ni, nj, nk = 4, 5, 4
        src = Block(shape=(ni, nj, nk))
        src.set_fluid(fluid)
        x = np.linspace(0.0, 1.0, ni).reshape(-1, 1, 1) * np.ones((ni, nj, nk))
        r = np.linspace(0.5, 1.0, nj).reshape(1, -1, 1) * np.ones((ni, nj, nk))
        t = np.linspace(0.0, 0.1, nk).reshape(1, 1, -1) * np.ones((ni, nj, nk))
        src.set_x(x)
        src.set_r(r)
        src.set_t(t)
        j_frac = np.linspace(0.0, 1.0, nj).reshape(1, -1, 1) * np.ones((ni, nj, nk))
        src.set_P_T(101325.0, 300.0)
        src.set_Vx(50.0 + 100.0 * j_frac)
        src.set_Vr(np.zeros((ni, nj, nk)))
        src.set_Vt(np.zeros((ni, nj, nk)))
        src.set_mu_turb(np.zeros((ni, nj, nk)))
        src.patches.append(InletPatch(i=0, j=(1, 3), k=(0, -1)))

        # tgt: 9 j-points, inlet patch j=(2,6) -> critical j indices [0, 2, 6, 8]
        # j=2 in tgt maps to j=1 in src, j=6 in tgt maps to j=3 in src
        ni2, nj2, nk2 = 4, 9, 4
        tgt = Block(shape=(ni2, nj2, nk2))
        tgt.set_fluid(fluid)
        x2 = np.linspace(0.0, 1.0, ni2).reshape(-1, 1, 1) * np.ones((ni2, nj2, nk2))
        r2 = np.linspace(0.5, 1.0, nj2).reshape(1, -1, 1) * np.ones((ni2, nj2, nk2))
        t2 = np.linspace(0.0, 0.1, nk2).reshape(1, 1, -1) * np.ones((ni2, nj2, nk2))
        tgt.set_x(x2)
        tgt.set_r(r2)
        tgt.set_t(t2)
        tgt.set_P_T(101325.0, 300.0)
        tgt.set_mu_turb(np.zeros((ni2, nj2, nk2)))
        tgt.patches.append(InletPatch(i=0, j=(2, 6), k=(0, -1)))

        Grid([tgt]).interp_from(Grid([src]))

        # tgt j=2 should exactly match src j=1
        np.testing.assert_allclose(tgt.Vx[:, 2, :], src.Vx[:, 1, :], atol=1e-3)
        # tgt j=6 should exactly match src j=3
        np.testing.assert_allclose(tgt.Vx[:, 6, :], src.Vx[:, 3, :], atol=1e-3)


class TestGridApplyGuessRestart:
    """Tests for Grid.apply_guess_restart method.

    apply_guess_restart takes a list of BlockRestart objects, transfers only
    conserved variables (no mu_turb), and interpolates in uniform
    index space (no critical-index awareness).
    """

    def test_same_shape_copies_exactly(self):
        """Same-shape conserved array is copied without modification."""
        src = _make_state_block((4, 4, 4), P=2e5, T=350.0, Vx=100.0)
        tgt = _make_state_block((4, 4, 4))

        Grid([tgt]).apply_guess_restart([BlockRestart(src.conserved)])

        np.testing.assert_allclose(tgt.conserved, src.conserved, rtol=1e-5)

    def test_upsample_constant_field(self):
        """Upsampling a uniform conserved field gives the same uniform state everywhere."""
        src = _make_state_block((3, 3, 3), P=2e5, T=400.0, Vx=80.0)
        tgt = _make_state_block((6, 6, 6), P=101325.0, T=300.0)

        Grid([tgt]).apply_guess_restart([BlockRestart(src.conserved)])

        ref = src.conserved[0, 0, 0]
        np.testing.assert_allclose(
            tgt.conserved, np.broadcast_to(ref, tgt.conserved.shape), rtol=1e-4
        )

    def test_downsample_constant_field(self):
        """Downsampling a uniform conserved field gives the same uniform state everywhere."""
        src = _make_state_block((8, 8, 8), P=1.5e5, T=320.0, Vx=50.0)
        tgt = _make_state_block((3, 3, 3), P=101325.0, T=300.0)

        Grid([tgt]).apply_guess_restart([BlockRestart(src.conserved)])

        ref = src.conserved[0, 0, 0]
        np.testing.assert_allclose(
            tgt.conserved, np.broadcast_to(ref, tgt.conserved.shape), rtol=1e-4
        )

    def test_multi_block(self):
        """apply_guess_restart works block-by-block for multiple blocks."""
        src1 = _make_state_block((4, 4, 4), P=2e5, T=350.0)
        src2 = _make_state_block((4, 4, 4), P=3e5, T=400.0)
        tgt1 = _make_state_block((6, 6, 6))
        tgt2 = _make_state_block((6, 6, 6))

        Grid([tgt1, tgt2]).apply_guess_restart(
            [BlockRestart(src1.conserved), BlockRestart(src2.conserved)]
        )

        ref1 = src1.conserved[0, 0, 0]
        ref2 = src2.conserved[0, 0, 0]
        np.testing.assert_allclose(
            tgt1.conserved, np.broadcast_to(ref1, tgt1.conserved.shape), rtol=1e-4
        )
        np.testing.assert_allclose(
            tgt2.conserved, np.broadcast_to(ref2, tgt2.conserved.shape), rtol=1e-4
        )

    def test_does_not_transfer_mu_turb(self):
        """apply_guess_restart does not touch mu_turb on the target block."""
        src = _make_state_block((4, 4, 4), mu_turb=5e-4)
        tgt = _make_state_block((4, 4, 4), mu_turb=1e-4)
        mu_before = tgt.mu_turb.copy()

        Grid([tgt]).apply_guess_restart([BlockRestart(src.conserved)])

        np.testing.assert_array_equal(tgt.mu_turb, mu_before)

    def test_get_restart_round_trip(self):
        """apply_restart(b2, make_restart(Grid([b1]))[0]) reproduces b1's conserved state."""
        src = _make_state_block((4, 4, 4), P=2e5, T=350.0, Vx=100.0)
        tgt = _make_state_block((4, 4, 4))

        apply_restart(tgt, make_restart(Grid([src]))[0])

        np.testing.assert_allclose(tgt.conserved, src.conserved, rtol=1e-5)


class TestBlockRestartImmutability:
    """BlockRestart must be a true snapshot: copy on construction, read-only."""

    def test_constructor_copies_array(self):
        arr = np.zeros((2, 2, 2, 5))
        r = BlockRestart(arr)
        assert r.conserved is not arr

    def test_field_assignment_is_blocked(self):
        import dataclasses

        r = BlockRestart(np.zeros((2, 2, 2, 5)))
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.conserved = np.ones((2, 2, 2, 5))

    def test_array_contents_are_read_only(self):
        r = BlockRestart(np.zeros((2, 2, 2, 5)))
        with pytest.raises(ValueError):
            r.conserved[0, 0, 0, 0] = 1.0

    def test_source_mutation_does_not_leak(self):
        arr = np.zeros((2, 2, 2, 5))
        r = BlockRestart(arr)
        arr[0, 0, 0, 0] = 99.0
        assert r.conserved[0, 0, 0, 0] == 0.0

    def test_outlet_and_mixing_are_frozen(self):
        a = np.ones((2, 2, 2))
        b = np.ones((1, 2, 1, 5))
        r = BlockRestart(np.zeros((2, 2, 2, 5)), outlet=(a,), mixing=(b,))
        assert r.outlet[0] is not a
        assert r.mixing[0] is not b
        with pytest.raises(ValueError):
            r.outlet[0][0, 0, 0] = 0.0
        with pytest.raises(ValueError):
            r.mixing[0][0, 0, 0, 0] = 0.0


def _outlet_block_with_target(shape, P=2e5, T=350.0):
    """Build a block carrying a single OutletPatch on the i=-1 face."""
    block = _make_state_block(shape, P=P, T=T)
    outlet = OutletPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet")
    block.patches.append(outlet)
    outlet.set_P(P)
    outlet.update_target()
    return block, outlet


def _mixing_block_with_target(shape, P=2e5, T=350.0):
    """Build a block carrying a single MixingPatch on the j=0 face."""
    block = _make_state_block(shape, P=P, T=T)
    mixing = MixingPatch(i=(0, -1), j=0, k=(0, -1), label="mixing")
    block.patches.append(mixing)
    mixing.attach_to_block(block)
    mixing.set_target()
    return block, mixing


class TestRestartOutlet:
    """OutletPatch _P_target_nd round-trip via BlockRestart."""

    def test_pure_round_trip(self):
        src, src_outlet = _outlet_block_with_target((4, 4, 4), P=2e5)
        # Stamp a non-uniform pattern onto _P_target_nd to verify shape preservation.
        src_outlet._P_target_nd = (
            src_outlet._P_target_nd
            * (
                1.0
                + 0.1
                * np.linspace(-1, 1, src_outlet._P_target_nd.size).reshape(
                    src_outlet._P_target_nd.shape
                )
            )
        ).astype(np.float32)

        tgt, tgt_outlet = _outlet_block_with_target((4, 4, 4), P=2e5)
        apply_restart(tgt, make_restart(Grid([src]))[0])

        np.testing.assert_allclose(
            tgt_outlet._P_target_nd, src_outlet._P_target_nd, rtol=1e-5
        )

    def test_shape_preserved_across_mean_change(self):
        src, src_outlet = _outlet_block_with_target((4, 4, 4), P=2e5)
        # Non-uniform _P_target_nd
        prof = (
            1.0
            + 0.05
            * np.linspace(-1, 1, src_outlet._P_target_nd.size).reshape(
                src_outlet._P_target_nd.shape
            )
        ).astype(np.float32)
        src_outlet._P_target_nd = (src_outlet._P_target_nd * prof).astype(np.float32)

        tgt, tgt_outlet = _outlet_block_with_target((4, 4, 4), P=2e5)
        # Change destination mean target pressure
        tgt_outlet.set_P(3e5)
        tgt_outlet.update_target()  # rebuilds _P_target_nd at the new mean

        # Ratio of stored shape (P_target_nd / P_raw_nd) must be preserved.
        src_ratio = src_outlet._P_target_nd / (src_outlet._P_raw / src.fluid.P_ref)

        apply_restart(tgt, make_restart(Grid([src]))[0])
        tgt_ratio = tgt_outlet._P_target_nd / (tgt_outlet._P_raw / tgt.fluid.P_ref)

        np.testing.assert_allclose(tgt_ratio, src_ratio, rtol=1e-5)
        # And the mean reflects the destination (3e5 / P_ref).
        expected_mean_nd = 3e5 / tgt.fluid.P_ref
        actual_mean_nd = tgt_outlet._P_target_nd.mean()
        np.testing.assert_allclose(actual_mean_nd, expected_mean_nd, rtol=5e-2)

    def test_shape_interpolation(self):
        src, src_outlet = _outlet_block_with_target((4, 4, 4), P=2e5)
        # Force _P_target_nd to a uniform value on src
        src_outlet._P_target_nd[...] = src_outlet._P_raw / src.fluid.P_ref

        tgt, tgt_outlet = _outlet_block_with_target((6, 6, 6), P=2e5)
        apply_restart(tgt, make_restart(Grid([src]))[0])

        expected = src_outlet._P_raw / tgt.fluid.P_ref
        np.testing.assert_allclose(
            tgt_outlet._P_target_nd,
            np.full(tgt_outlet._P_target_nd.shape, expected),
            rtol=1e-4,
        )


class TestRestartMixing:
    """MixingPatch _target round-trip via BlockRestart."""

    def test_round_trip_across_different_refs(self):
        # Source and destination use different reference scales
        # (rho_ref, V_ref, Rgas_ref). Stored values must round-trip
        # through dimensionalize / re-nondimensionalize.
        src, src_mix = _mixing_block_with_target((4, 4, 4), P=2e5, T=350.0)
        tgt, tgt_mix = _mixing_block_with_target((4, 4, 4), P=2e5, T=350.0)

        # Re-create fluids with non-default reference scales that differ
        # between src and tgt, then re-set them on the blocks.
        src_fluid = PerfectFluid(
            cp=1005.0,
            gamma=1.4,
            mu=1.8e-5,
            Pr=0.72,
            rho_ref=1.2,
            V_ref=200.0,
            Rgas_ref=287.0,
        )
        tgt_fluid = PerfectFluid(
            cp=1005.0,
            gamma=1.4,
            mu=1.8e-5,
            Pr=0.72,
            rho_ref=2.5,
            V_ref=350.0,
            Rgas_ref=287.0,
        )
        src.set_fluid(src_fluid)
        tgt.set_fluid(tgt_fluid)
        src.set_P_T(2e5, 350.0)
        src.set_Vx(50.0)
        src.set_Vr(0.0)
        src.set_Vt(0.0)
        tgt.set_P_T(2e5, 350.0)
        tgt.set_Vx(50.0)
        tgt.set_Vr(0.0)
        tgt.set_Vt(0.0)
        # Re-set _target on both since underlying refs changed.
        src_mix.set_target()
        tgt_mix.set_target()

        # Sanity: refs really do differ.
        assert src.fluid.rho_ref != tgt.fluid.rho_ref
        assert src.fluid.V_ref != tgt.fluid.V_ref

        # _target is a conserved-variable stack, so it scales like one.
        def _refs(block):
            f = block.fluid
            return np.array(
                [
                    f.rho_ref,
                    f.rho_ref * f.V_ref,
                    f.rho_ref * f.V_ref,
                    f.rho_ref * block.L_ref * f.V_ref,
                    f.rho_ref * f.V_ref**2,
                ],
                dtype=np.float32,
            )

        # Capture the source's _target in dimensional units.
        src_dim = src_mix._target * _refs(src)

        apply_restart(tgt, make_restart(Grid([src]))[0])

        # Destination's nondim _target = source's dimensional / dst refs.
        tgt_refs = _refs(tgt)
        np.testing.assert_allclose(tgt_mix._target, src_dim / tgt_refs, rtol=1e-5)

    def test_shape_interpolation(self):
        src, src_mix = _mixing_block_with_target((4, 4, 4), P=2e5)
        # Force uniform _target across the span axis.
        src_mix._target[...] = src_mix._target.mean(
            axis=src_mix.span_dim, keepdims=True
        )
        original = src_mix._target.copy()  # uniform along span

        tgt, tgt_mix = _mixing_block_with_target((6, 6, 6), P=2e5)

        apply_restart(tgt, make_restart(Grid([src]))[0])

        # Uniform field must remain uniform after interpolation; the per-component
        # value must match the source's.
        for c in range(5):
            expected = original[..., c].mean()
            np.testing.assert_allclose(
                tgt_mix._target[..., c],
                np.full(tgt_mix._target[..., c].shape, expected),
                rtol=1e-4,
            )
