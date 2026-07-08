"""Tests for I/O functionality across all file formats.

Tests file input/output operations for EMB, Plot3D, and TS3 formats including
roundtrip verification, error handling, and format-specific features.

Test cases:
- test_emb_basic_roundtrip: Basic grid coordinates roundtrip verification
- test_emb_multiblock_roundtrip: Multiple blocks with different shapes roundtrip
- test_emb_with_patches_roundtrip: Inlet, outlet, and periodic patches preservation
- test_emb_with_flow_data_roundtrip: Complete flow field data roundtrip
- test_emb_empty_grid: Empty grid error handling
- test_emb_invalid_file: Non-existent file error handling
- test_grid_emb_methods_interface: Grid EMB method interface verification
- test_emb_patch_block_shape_restoration: Patch block shape restoration after deserialization
- test_emb_compression_roundtrip: Compression functionality roundtrip testing
- test_write_plot3d_multiblock_roundtrip: Write Plot3D with multiblock grid and verify coordinates roundtrip
- test_write_plot3d_empty_grid: Empty grid raises appropriate error
- test_plot3d_fvbnd_roundtrip: Combined Plot3D and FVBND roundtrip with patches
- test_block_set_xyz_roundtrip: Block.set_xyz correctly converts Cartesian to polar coordinates
- test_grid_write_plot3d_with_fvbnd: Grid.write_plot3d() method with FVBND generation
- test_grid_write_plot3d_coordinates_only: Grid.write_plot3d() method with coordinates only
- test_grid_write_plot3d_empty_grid: Grid.write_plot3d() raises error for empty grid
- test_grid_write_plot3d_interface: Grid.write_plot3d() method interface and parameters
- test_write_ts3_multiblock_roundtrip: Grid.write_ts3 with multiblock grid and verify roundtrip
- test_write_ts3_empty_grid: Empty grid raises appropriate error
- test_ts3_basic_roundtrip: Basic TS3 Grid methods roundtrip without patches
- test_ts3_geometry_only_non_strict: TS3 Grid write in non-strict mode with geometry only
- test_ts3_geometry_only_strict_mode: Strict mode rejects geometry-only grids
- test_grid_ts3_methods_interface: Grid TS3 methods have correct interface
- test_ts3_multiblock_with_patches_roundtrip: TS3 round-trip with multiblock grid including patches
"""

import numpy as np
import pytest
import os
import inspect
from ember.grid import Grid
from ember.block import Block
from ember.patch import InletPatch, OutletPatch, PeriodicPatch, MixingPatch
from ember.plot3d import write_plot3d, read_plot3d, write_fvbnd
from ember.block_util import to_tm3
from ember.fluid import PerfectFluid
from ember import util


# EMB (Ember Binary) Format Tests


def test_emb_basic_roundtrip(tmp_path):
    """Test EMB round-trip with basic grid coordinates."""
    # Create a simple single block grid
    block = Block(shape=(3, 4, 5))

    # Set coordinates
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Create grid
    original_grid = Grid([block])

    # Write EMB file
    emb_file = tmp_path / "test_basic.emb"
    original_grid.write_emb(str(emb_file))

    # Verify file was created
    assert emb_file.exists()

    # Read back
    reconstructed_grid = Grid.read_emb(str(emb_file))

    # Verify structure
    assert len(reconstructed_grid) == 1, "Should have 1 block"

    # Verify coordinates match
    orig_block = original_grid[0]
    recon_block = reconstructed_grid[0]

    assert orig_block.shape == recon_block.shape, "Block shape mismatch"
    np.testing.assert_allclose(
        recon_block.xrt,
        orig_block.xrt,
        rtol=1e-12,
        atol=1e-15,
        err_msg="Block coordinates don't match after EMB roundtrip",
    )


def test_emb_multiblock_roundtrip(tmp_path):
    """Test EMB round-trip with multiple blocks of different shapes."""
    # Create multiblock grid with different shaped blocks
    block1 = Block(shape=(3, 4, 5))
    block2 = Block(shape=(4, 3, 6))
    block3 = Block(shape=(2, 5, 3))

    # Set coordinates for each block
    xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    xrt2 = util.linmesh3([1.0, 2.0], [1.5, 2.5], [0.1, 0.3], (4, 3, 6))
    xrt3 = util.linmesh3([2.0, 2.5], [0.8, 1.8], [0.0, 0.15], (2, 5, 3))

    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])
    block3.set_x(xrt3[..., 0])
    block3.set_r(xrt3[..., 1])
    block3.set_t(xrt3[..., 2])

    # Create original grid
    original_grid = Grid([block1, block2, block3])

    # Write EMB file
    emb_file = tmp_path / "test_multiblock.emb"
    original_grid.write_emb(str(emb_file))

    # Verify file was created
    assert emb_file.exists()

    # Read back
    reconstructed_grid = Grid.read_emb(str(emb_file))

    # Verify grid structure
    assert len(reconstructed_grid) == 3, "Should have 3 blocks"

    # Compare each block's coordinates
    for i, (orig_block, recon_block) in enumerate(
        zip(original_grid, reconstructed_grid)
    ):
        assert orig_block.shape == recon_block.shape, f"Block {i + 1} shape mismatch"

        np.testing.assert_allclose(
            recon_block.xrt,
            orig_block.xrt,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Block {i + 1} coordinates don't match after EMB roundtrip",
        )


def test_emb_with_patches_roundtrip(tmp_path):
    """Test EMB round-trip preserving inlet, outlet, and periodic patches."""
    # Create two blocks
    block1 = Block(shape=(5, 6, 7))
    block2 = Block(shape=(4, 5, 6))

    # Set coordinates
    xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.3], (5, 6, 7))
    xrt2 = util.linmesh3([1.0, 2.0], [1.0, 2.0], [0.1, 0.4], (4, 5, 6))

    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])

    # Add patches to blocks
    inlet1 = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet_1")
    outlet1 = OutletPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet_1")
    periodic1 = PeriodicPatch(i=(0, -1), j=(0, -1), k=0, label="periodic_1")
    periodic2 = PeriodicPatch(i=(0, -1), j=(0, -1), k=-1, label="periodic_2")

    # Add patches to blocks first so they can resolve their shapes
    block1.patches.append(inlet1)
    block1.patches.append(periodic1)
    block2.patches.append(outlet1)
    block2.patches.append(periodic2)

    # Now set inlet properties after patch is attached to block
    inlet1.set_Po_To_Alpha_Beta(
        Po=np.ones(inlet1.shape) * 101325.0,
        To=np.ones(inlet1.shape) * 288.15,
        Alpha=np.zeros(inlet1.shape),
        Beta=np.ones(inlet1.shape) * 10.0,
    )

    outlet1.set_P(120000.0)

    # Create grid
    original_grid = Grid([block1, block2])

    # Write EMB file
    emb_file = tmp_path / "test_patches.emb"
    original_grid.write_emb(str(emb_file))

    # Read back
    reconstructed_grid = Grid.read_emb(str(emb_file))

    # Verify structure
    assert len(reconstructed_grid) == 2, "Should have 2 blocks"

    # Verify patches were preserved
    assert len(reconstructed_grid[0].patches) == 2, "Block 1 should have 2 patches"
    assert len(reconstructed_grid[1].patches) == 2, "Block 2 should have 2 patches"

    # Check patch types
    block1_patches = reconstructed_grid[0].patches
    block2_patches = reconstructed_grid[1].patches

    assert len([p for p in block1_patches if isinstance(p, InletPatch)]) == 1
    assert len([p for p in block1_patches if isinstance(p, PeriodicPatch)]) == 1
    assert len([p for p in block2_patches if isinstance(p, OutletPatch)]) == 1
    assert len([p for p in block2_patches if isinstance(p, PeriodicPatch)]) == 1

    # Verify patch labels and properties were preserved
    recon_inlet = [p for p in block1_patches if isinstance(p, InletPatch)][0]
    recon_outlet = [p for p in block2_patches if isinstance(p, OutletPatch)][0]

    # Check basic patch properties that are preserved in serialization
    assert recon_inlet.label == inlet1.label, "Inlet patch label not preserved"
    assert recon_outlet.label == outlet1.label, "Outlet patch label not preserved"

    # Test that patch shapes work after deserialization (requires block shape restoration)
    assert recon_inlet.shape == inlet1.shape, "Inlet patch shape not preserved"
    assert recon_outlet.shape == outlet1.shape, "Outlet patch shape not preserved"

    # Test that absolute limits work (also requires block shape restoration)
    np.testing.assert_array_equal(
        recon_inlet.ijk_lim_abs,
        inlet1.ijk_lim_abs,
        err_msg="Inlet patch absolute limits not preserved",
    )
    np.testing.assert_array_equal(
        recon_outlet.ijk_lim_abs,
        outlet1.ijk_lim_abs,
        err_msg="Outlet patch absolute limits not preserved",
    )

    # Verify coordinates match
    for i, (orig_block, recon_block) in enumerate(
        zip(original_grid, reconstructed_grid)
    ):
        np.testing.assert_allclose(
            recon_block.xrt,
            orig_block.xrt,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Block {i} coordinates don't match",
        )


def test_emb_with_flow_data_roundtrip(tmp_path):
    """Test EMB round-trip with complete flow field data."""
    # Create block with flow data
    block = Block(shape=(3, 4, 5))

    # Set coordinates
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set up fluid and flow data
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    block.set_fluid(fluid)
    block.set_rpm(3600.0)
    block.set_Nb(24)

    # Set conserved variables
    rho = np.ones(block.shape, dtype=np.float32) * 1.2
    rhoVx = np.ones(block.shape, dtype=np.float32) * 150.0
    rhoVr = np.zeros(block.shape, dtype=np.float32)
    rhoVt = np.ones(block.shape, dtype=np.float32) * 100.0
    rhoe = np.ones(block.shape, dtype=np.float32) * 2.5e5

    conserved = np.stack([rho, rhoVx, rhoVr, rhoVt, rhoe], axis=-1)
    block.set_conserved(conserved)
    block.set_wdist(np.ones(block.shape, dtype=np.float32) * 0.001)

    # Create grid
    original_grid = Grid([block])

    # Write EMB file
    emb_file = tmp_path / "test_flow_data.emb"
    original_grid.write_emb(str(emb_file))

    # Read back
    reconstructed_grid = Grid.read_emb(str(emb_file))

    # Verify structure
    assert len(reconstructed_grid) == 1

    orig_block = original_grid[0]
    recon_block = reconstructed_grid[0]

    # Verify flow properties
    assert orig_block.rpm == recon_block.rpm, "RPM not preserved"
    assert orig_block.Nb == recon_block.Nb, "Number of blades not preserved"

    # Verify fluid properties
    # Use dummy rho and u values since these methods need state parameters
    rho_test, u_test = 1.0, 1000.0
    assert orig_block.fluid.get_cp(rho_test, u_test) == recon_block.fluid.get_cp(
        rho_test, u_test
    ), "Fluid cp not preserved"
    assert orig_block.fluid.get_gamma(rho_test, u_test) == recon_block.fluid.get_gamma(
        rho_test, u_test
    ), "Fluid gamma not preserved"

    # Verify conserved variables
    np.testing.assert_allclose(
        recon_block.conserved,
        orig_block.conserved,
        rtol=1e-12,
        err_msg="Conserved variables not preserved",
    )

    # Verify wall distance
    np.testing.assert_allclose(
        recon_block.wdist,
        orig_block.wdist,
        rtol=1e-12,
        err_msg="Wall distance not preserved",
    )


def test_emb_empty_grid():
    """Test that empty grid raises appropriate error."""
    grid = Grid()
    with pytest.raises(
        ValueError, match="Cannot write EMB file: grid contains no blocks"
    ):
        grid.write_emb("test.emb")


def test_emb_invalid_file():
    """Test error handling for invalid EMB files."""
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        Grid.read_emb("nonexistent.emb")


def test_grid_emb_methods_interface():
    """Test that Grid EMB methods have correct interface."""
    # Test class method exists
    assert hasattr(Grid, "read_emb"), "Grid should have read_emb class method"
    assert callable(Grid.read_emb), "Grid.read_emb should be callable"

    # Test instance method exists
    grid = Grid()
    assert hasattr(grid, "write_emb"), "Grid should have write_emb instance method"
    assert callable(grid.write_emb), "Grid.write_emb should be callable"


def test_emb_patch_block_shape_restoration(tmp_path):
    """Test that patch block shapes are properly restored after EMB deserialization."""
    # Create blocks with patches that use negative indices
    block1 = Block(shape=(5, 6, 7))
    block2 = Block(shape=(4, 5, 8))

    # Set coordinates
    xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.3], (5, 6, 7))
    xrt2 = util.linmesh3([1.0, 2.0], [1.0, 2.0], [0.1, 0.4], (4, 5, 8))
    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])

    # Add patches with negative indices (require block shape for resolution)
    inlet_patch = InletPatch(i=0, j=(1, -2), k=(2, -1), label="inlet_negative")
    outlet_patch = OutletPatch(i=-1, j=(0, -1), k=(1, -2), label="outlet_negative")
    periodic_patch = PeriodicPatch(i=(1, -1), j=0, k=(2, -2), label="periodic_negative")

    # Add patches to blocks (this sets their block shapes)
    block1.patches.append(inlet_patch)
    block2.patches.append(outlet_patch)
    block2.patches.append(periodic_patch)

    # Verify patches can access their shapes before serialization
    assert inlet_patch.shape == (1, 4, 5), (
        "Inlet patch shape incorrect before serialization"
    )
    assert outlet_patch.shape == (1, 5, 6), (
        "Outlet patch shape incorrect before serialization"
    )
    assert periodic_patch.shape == (3, 1, 5), (
        "Periodic patch shape incorrect before serialization"
    )

    # Create grid and perform EMB round-trip
    original_grid = Grid([block1, block2])
    emb_file = tmp_path / "test_patch_shapes.emb"
    original_grid.write_emb(str(emb_file))
    reconstructed_grid = Grid.read_emb(str(emb_file))

    # Verify structure is preserved
    assert len(reconstructed_grid) == 2, "Grid structure not preserved"
    assert len(reconstructed_grid[0].patches) == 1, "Block 1 patches not preserved"
    assert len(reconstructed_grid[1].patches) == 2, "Block 2 patches not preserved"

    # Get reconstructed patches
    recon_inlet = reconstructed_grid[0].patches[0]
    recon_outlet = reconstructed_grid[1].patches[0]
    recon_periodic = reconstructed_grid[1].patches[1]

    # Critical test: patches should be able to access their shapes after deserialization
    # This will fail if block shapes weren't restored properly
    assert recon_inlet.shape == (1, 4, 5), "Reconstructed inlet patch shape incorrect"
    assert recon_outlet.shape == (1, 5, 6), "Reconstructed outlet patch shape incorrect"
    assert recon_periodic.shape == (3, 1, 5), (
        "Reconstructed periodic patch shape incorrect"
    )

    # Test absolute limit access (also requires block shape)
    np.testing.assert_array_equal(
        recon_inlet.ijk_lim_abs,
        inlet_patch.ijk_lim_abs,
        err_msg="Inlet patch absolute limits don't match",
    )
    np.testing.assert_array_equal(
        recon_outlet.ijk_lim_abs,
        outlet_patch.ijk_lim_abs,
        err_msg="Outlet patch absolute limits don't match",
    )
    np.testing.assert_array_equal(
        recon_periodic.ijk_lim_abs,
        periodic_patch.ijk_lim_abs,
        err_msg="Periodic patch absolute limits don't match",
    )

    # Test that block shape property works
    assert list(recon_inlet.block.shape) == [5, 6, 7], (
        "Inlet patch block shape incorrect"
    )
    assert list(recon_outlet.block.shape) == [4, 5, 8], (
        "Outlet patch block shape incorrect"
    )
    assert list(recon_periodic.block.shape) == [4, 5, 8], (
        "Periodic patch block shape incorrect"
    )


def test_emb_compression_roundtrip(tmp_path):
    """Test EMB compression functionality with roundtrip."""
    # Create a test grid with some complexity to get reasonable compression
    blocks = []
    for i in range(2):
        block = Block(shape=(10, 10, 10))
        xrt = util.linmesh3([i, i + 1], [0.5, 1.5], [0.0, 0.2], (10, 10, 10))
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])

        # Add some flow data
        fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
        conserved = np.random.random((10, 10, 10, 5)).astype(np.float32)
        block.set_fluid(fluid)
        block.set_conserved(conserved)
        blocks.append(block)

    original_grid = Grid(blocks)

    # Write both compressed and uncompressed versions
    uncompressed_file = tmp_path / "test_uncompressed.emb"
    compressed_file = tmp_path / "test_compressed.emb"

    original_grid.write_emb(str(uncompressed_file), compress=False)
    original_grid.write_emb(str(compressed_file), compress=True)

    # Verify both files were created
    assert uncompressed_file.exists()
    assert compressed_file.exists()

    # Check that compression actually reduced file size
    uncompressed_size = os.path.getsize(uncompressed_file)
    compressed_size = os.path.getsize(compressed_file)
    assert compressed_size < uncompressed_size, "Compressed file should be smaller"

    # Check that we get reasonable compression (at least 10% reduction)
    compression_ratio = compressed_size / uncompressed_size
    assert compression_ratio < 0.9, f"Poor compression: {compression_ratio:.2f}"

    # Read both versions back and verify they are identical
    grid_uncompressed = Grid.read_emb(str(uncompressed_file))
    grid_compressed = Grid.read_emb(str(compressed_file))

    # Verify structure is the same
    assert len(grid_uncompressed) == len(grid_compressed) == 2

    # Verify data is identical across both versions
    for i in range(len(grid_uncompressed)):
        block_uncomp = grid_uncompressed[i]
        block_comp = grid_compressed[i]

        assert block_uncomp.shape == block_comp.shape
        np.testing.assert_allclose(
            block_uncomp.xrt,
            block_comp.xrt,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Block {i} coordinates differ between compressed/uncompressed",
        )
        np.testing.assert_allclose(
            block_uncomp.conserved,
            block_comp.conserved,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Block {i} flow data differs between compressed/uncompressed",
        )


# Plot3D Format Tests


def test_write_plot3d_multiblock_roundtrip(tmp_path):
    """Test write_plot3d with multiblock grid and verify coordinates roundtrip correctly."""
    # Create multiblock grid with different shaped blocks
    block1 = Block(shape=(3, 4, 5))
    block2 = Block(shape=(4, 3, 6))
    block3 = Block(shape=(2, 5, 3))

    # Set coordinates for each block
    xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    xrt2 = util.linmesh3([1.0, 2.0], [1.5, 2.5], [0.1, 0.3], (4, 3, 6))
    xrt3 = util.linmesh3([2.0, 2.5], [0.8, 1.8], [0.0, 0.15], (2, 5, 3))

    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])
    block3.set_x(xrt3[..., 0])
    block3.set_r(xrt3[..., 1])
    block3.set_t(xrt3[..., 2])

    # Create original grid
    original_grid = Grid([block1, block2, block3])

    # Write to file using new interface
    output_file = tmp_path / "test_multiblock.dat"
    write_plot3d(original_grid, str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Read back using new interface
    reconstructed_grid = read_plot3d(str(output_file))

    # Verify grid structure
    assert len(reconstructed_grid) == 3, "Should have 3 blocks"

    # Compare each block's coordinates
    for i, (orig_block, recon_block) in enumerate(
        zip(original_grid, reconstructed_grid)
    ):
        assert orig_block.shape == recon_block.shape, f"Block {i + 1} shape mismatch"

        # Compare Cartesian coordinates (should match within numerical precision)
        # Note: Using relaxed tolerance due to coordinate conversion round-off errors
        for attr in ("x", "y", "z"):
            np.testing.assert_allclose(
                getattr(recon_block, attr),
                getattr(orig_block, attr),
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Block {i + 1} {attr} coordinates don't match after roundtrip",
            )

        # Also compare polar coordinates for completeness
        np.testing.assert_allclose(
            recon_block.xrt,
            orig_block.xrt,
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"Block {i + 1} polar coordinates don't match after roundtrip",
        )


def test_write_plot3d_empty_grid():
    """Test that empty grid raises appropriate error."""
    grid = Grid()
    with pytest.raises(
        ValueError, match="Cannot write Plot3D file: grid contains no blocks"
    ):
        write_plot3d(grid, "test.dat")


def test_plot3d_fvbnd_roundtrip(tmp_path):
    """Test combined Plot3D and FVBND roundtrip with patches."""
    # Create a grid with blocks and patches
    block1 = Block(shape=(5, 6, 7))
    block2 = Block(shape=(4, 5, 6))

    # Set coordinates
    xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.3], (5, 6, 7))
    xrt2 = util.linmesh3([1.0, 2.0], [1.0, 2.0], [0.1, 0.4], (4, 5, 6))

    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])

    # Add patches to blocks
    inlet1 = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet_1")
    outlet1 = OutletPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet_1")
    mixing1 = MixingPatch(i=(0, -1), j=0, k=(0, -1), label="mixing_1")
    periodic1 = PeriodicPatch(i=(0, -1), j=(0, -1), k=0, label="periodic_1")

    block1.patches.append(inlet1)
    block1.patches.append(mixing1)
    block2.patches.append(outlet1)
    block2.patches.append(periodic1)

    # Create grid
    original_grid = Grid([block1, block2])

    # Write both files
    p3d_file = tmp_path / "test_roundtrip.p3d"
    fvbnd_file = tmp_path / "test_roundtrip.fvbnd"

    write_plot3d(original_grid, str(p3d_file))
    write_fvbnd(original_grid, str(fvbnd_file), iregion=0)

    # Read back using combined interface
    reconstructed_grid = Grid.read_plot3d(str(p3d_file), str(fvbnd_file))

    # Verify structure
    assert len(reconstructed_grid) == 2, "Should have 2 blocks"
    assert len(reconstructed_grid.patches) == 4, "Should have 4 patches total"
    assert len(reconstructed_grid.patches.inlet) == 1, "Should have 1 inlet patch"
    assert len(reconstructed_grid.patches.outlet) == 1, "Should have 1 outlet patch"
    assert len(reconstructed_grid.patches.mixing) == 1, "Should have 1 mixing patch"
    assert len(reconstructed_grid.patches.periodic) == 1, "Should have 1 periodic patch"

    # Verify coordinates match
    for i, (orig_block, recon_block) in enumerate(
        zip(original_grid, reconstructed_grid)
    ):
        for attr in ("x", "y", "z"):
            np.testing.assert_allclose(
                getattr(recon_block, attr),
                getattr(orig_block, attr),
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Block {i} {attr} coordinates don't match",
            )


def test_grid_write_plot3d_with_fvbnd(tmp_path):
    """Test Grid.write_plot3d() method with FVBND generation."""
    # Create a grid with blocks and patches
    block1 = Block(shape=(5, 6, 7))
    block2 = Block(shape=(4, 5, 6))

    # Set coordinates
    xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.3], (5, 6, 7))
    xrt2 = util.linmesh3([1.0, 2.0], [1.0, 2.0], [0.1, 0.4], (4, 5, 6))

    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])

    # Add patches to blocks
    inlet1 = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet_1")
    outlet1 = OutletPatch(i=-1, j=(0, -1), k=(0, -1), label="outlet_1")
    mixing1 = MixingPatch(i=(0, -1), j=0, k=(0, -1), label="mixing_1")
    periodic1 = PeriodicPatch(i=(0, -1), j=(0, -1), k=0, label="periodic_1")

    block1.patches.append(inlet1)
    block1.patches.append(mixing1)
    block2.patches.append(outlet1)
    block2.patches.append(periodic1)

    # Create grid
    grid = Grid([block1, block2])

    # Write both files using Grid method
    p3d_file = tmp_path / "test_grid_method.p3d"
    fvbnd_file = tmp_path / "test_grid_method.fvbnd"

    grid.write_plot3d(str(p3d_file), str(fvbnd_file), iregion=1)

    # Verify both files were created
    assert p3d_file.exists()
    assert fvbnd_file.exists()

    # Read back using combined interface to verify
    reconstructed_grid = Grid.read_plot3d(str(p3d_file), str(fvbnd_file))

    # Verify structure
    assert len(reconstructed_grid) == 2, "Should have 2 blocks"
    assert len(reconstructed_grid.patches) == 4, "Should have 4 patches total"

    # Verify coordinates match
    for i, (orig_block, recon_block) in enumerate(zip(grid, reconstructed_grid)):
        for attr in ("x", "y", "z"):
            np.testing.assert_allclose(
                getattr(recon_block, attr),
                getattr(orig_block, attr),
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"Block {i} {attr} coordinates don't match",
            )


def test_grid_write_plot3d_coordinates_only(tmp_path):
    """Test Grid.write_plot3d() method with coordinates only (no FVBND)."""
    # Create simple grid
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    grid = Grid([block])

    # Write Plot3D file only (no FVBND)
    p3d_file = tmp_path / "test_coords_only.p3d"
    grid.write_plot3d(str(p3d_file))

    # Verify file was created
    assert p3d_file.exists()

    # Read back
    reconstructed_grid = Grid.read_plot3d(str(p3d_file))

    # Verify structure and coordinates
    assert len(reconstructed_grid) == 1
    for attr in ("x", "y", "z"):
        np.testing.assert_allclose(
            getattr(reconstructed_grid[0], attr),
            getattr(grid[0], attr),
            rtol=1e-6,
            atol=1e-8,
        )


def test_grid_write_plot3d_empty_grid():
    """Test that Grid.write_plot3d() raises error for empty grid."""
    grid = Grid()
    with pytest.raises(
        ValueError, match="Cannot write Plot3D file: grid contains no blocks"
    ):
        grid.write_plot3d("test.p3d")


def test_grid_write_plot3d_interface():
    """Test Grid.write_plot3d() method interface and parameters."""
    # Test method exists
    grid = Grid()
    assert hasattr(grid, "write_plot3d"), (
        "Grid should have write_plot3d instance method"
    )
    assert callable(grid.write_plot3d), "Grid.write_plot3d should be callable"

    # Test that method signature accepts expected parameters
    sig = inspect.signature(grid.write_plot3d)
    expected_params = {"p3d_filename", "fvbnd_filename", "flip_k", "iregion"}
    actual_params = set(sig.parameters.keys())
    assert expected_params == actual_params, (
        f"Expected parameters {expected_params}, got {actual_params}"
    )

    # Test default values
    params = sig.parameters
    assert params["fvbnd_filename"].default is None
    assert params["flip_k"].default is True
    assert params["iregion"].default == 0


# TS3 Format Tests


def test_write_ts3_multiblock_roundtrip(tmp_path):
    """Test Grid.write_ts3 with multiblock grid and verify roundtrip."""
    # Create multiblock grid with different shaped blocks
    block1 = Block(shape=(3, 4, 5))
    block2 = Block(shape=(4, 3, 6))
    block3 = Block(shape=(2, 5, 3))

    # Set coordinates for each block
    xrt1 = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    xrt2 = util.linmesh3([1.0, 2.0], [1.5, 2.5], [0.1, 0.3], (4, 3, 6))
    xrt3 = util.linmesh3([2.0, 2.5], [0.8, 1.8], [0.0, 0.15], (2, 5, 3))

    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])
    block3.set_x(xrt3[..., 0])
    block3.set_r(xrt3[..., 1])
    block3.set_t(xrt3[..., 2])

    # Create fluid and set flow data for TS3 requirements
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    for block in [block1, block2, block3]:
        block.set_fluid(fluid)
        block.set_rpm(0.0)
        block.set_Nb(1)

        # A non-trivial, physically meaningful state: P/T vary along i and all
        # three velocity components are non-zero, so the flow-field roundtrip is
        # actually exercised (not just a uniform field).
        ramp = np.linspace(0.0, 1.0, block.shape[0]).reshape(-1, 1, 1)
        P = 1.0e5 + 2.0e4 * ramp * np.ones(block.shape)
        T = 300.0 + 50.0 * ramp * np.ones(block.shape)
        block.set_P_T(P, T)
        block.set_Vxrt(
            np.stack(
                [
                    120.0 * np.ones(block.shape),
                    10.0 * np.ones(block.shape),
                    -30.0 * np.ones(block.shape),
                ],
                axis=-1,
            )
        )
        block.set_wdist(np.ones(block.shape, dtype=np.float32) * 0.01)

    # Create original grid
    original_grid = Grid([block1, block2, block3])

    # Write to file using Grid.write_ts3 method
    output_file = tmp_path / "test_multiblock.ts3"
    original_grid.write_ts3(str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Read back using Grid.read_ts3 method
    reconstructed_grid = Grid.read_ts3(str(output_file))

    # Verify grid structure
    assert len(reconstructed_grid) == 3, "Should have 3 blocks"

    # Compare each block's coordinates
    for i, (orig_block, recon_block) in enumerate(
        zip(original_grid, reconstructed_grid)
    ):
        assert orig_block.shape == recon_block.shape, f"Block {i + 1} shape mismatch"

        # Compare coordinates (relaxed tolerance for HDF5 round-off)
        np.testing.assert_allclose(
            recon_block.xrt,
            orig_block.xrt,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Block {i + 1} coordinates don't match after roundtrip",
        )

        # Compare the dimensional flow field. Density and the three momenta are
        # datum-independent and must round-trip directly; P and T cover the
        # energy, which is written on TS3's u = cv * T datum and converted back
        # on read. f4 storage limits accuracy to ~1e-4 relative.
        for name in ("rho", "rhoVx", "rhoVr", "rhorVt"):
            np.testing.assert_allclose(
                getattr(recon_block, name),
                getattr(orig_block, name),
                rtol=1e-4,
                atol=1e-3,
                err_msg=f"Block {i + 1} {name} doesn't match after roundtrip",
            )
        np.testing.assert_allclose(
            recon_block.P,
            orig_block.P,
            rtol=1e-4,
            err_msg=f"Block {i + 1} P doesn't match after roundtrip",
        )
        np.testing.assert_allclose(
            recon_block.T,
            orig_block.T,
            rtol=1e-4,
            err_msg=f"Block {i + 1} T doesn't match after roundtrip",
        )


def test_write_ts3_empty_grid():
    """Test that empty grid raises appropriate error."""
    grid = Grid()
    with pytest.raises(
        ValueError, match="Cannot write TS3 file: grid contains no blocks"
    ):
        grid.write_ts3("test.ts3")


def test_write_ts3_no_fluid_geometry_only_roundtrip(tmp_path):
    """A grid with no fluid round-trips as geometry only, fabricating no fluid.

    Gas properties default to NaN in DEFAULT_AV, so write omits them; their
    absence is the "no fluid" signal on read, which then leaves the block
    fluid-less rather than inventing default gas properties.
    """
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_rpm(0.0)
    block.set_Nb(1)
    grid = Grid([block])

    output_file = tmp_path / "no_fluid.hdf5"
    # Non-strict (default) write must succeed without a fluid.
    grid.write_ts3(str(output_file))

    recon = Grid.read_ts3(str(output_file))
    assert len(recon) == 1

    # Coordinates survive.
    np.testing.assert_allclose(recon[0].xrt, block.xrt, rtol=1e-5, atol=1e-6)

    # No fluid was fabricated: accessing fluid properties raises.
    with pytest.raises(ValueError, match="Working fluid must be set"):
        recon[0].fluid


def test_write_ts3_no_fluid_strict_raises():
    """Strict write of a fluid-less grid still raises."""
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_rpm(0.0)
    block.set_Nb(1)
    grid = Grid([block])

    with pytest.raises(ValueError, match="Requires working fluid set in strict mode"):
        grid.write_ts3("unused.hdf5", strict=True)


def test_ts3_basic_roundtrip(tmp_path):
    """Test basic TS3 Grid methods roundtrip without patches."""
    # Create a simple single block grid
    block = Block(shape=(3, 4, 5))

    # Set coordinates
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set up fluid and minimal flow data
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    block.set_fluid(fluid)
    block.set_rpm(0.0)
    block.set_Nb(1)

    # Set simple conserved variables
    rho = np.ones(block.shape, dtype=np.float32)
    rhoVx = np.zeros(block.shape, dtype=np.float32)
    rhoVr = np.zeros(block.shape, dtype=np.float32)
    rhoVt = np.zeros(block.shape, dtype=np.float32)
    rhoe = np.ones(block.shape, dtype=np.float32) * 1e5

    conserved = np.stack([rho, rhoVx, rhoVr, rhoVt, rhoe], axis=-1)
    block.set_conserved(conserved)
    block.set_wdist(np.ones(block.shape, dtype=np.float32) * 0.01)

    # Create grid
    original_grid = Grid([block])

    # Write TS3 file using Grid method
    ts3_file = tmp_path / "test_basic.ts3"
    original_grid.write_ts3(str(ts3_file))

    # Read back using Grid method
    reconstructed_grid = Grid.read_ts3(str(ts3_file))

    # Verify structure
    assert len(reconstructed_grid) == 1, "Should have 1 block"

    # Verify coordinates match
    orig_block = original_grid[0]
    recon_block = reconstructed_grid[0]

    assert orig_block.shape == recon_block.shape, "Block shape mismatch"
    np.testing.assert_allclose(
        recon_block.xrt,
        orig_block.xrt,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Block coordinates don't match",
    )


def test_ts3_geometry_only_non_strict(tmp_path):
    """Test TS3 Grid write in non-strict mode with geometry only (no flow data)."""
    # Create a simple block with only coordinates
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set minimal required properties for TS3
    block.set_rpm(0.0)
    block.set_Nb(1)

    # Create grid without setting any flow data
    grid = Grid([block])

    # Should work in non-strict mode
    ts3_file = tmp_path / "test_geometry_only.ts3"
    grid.write_ts3(str(ts3_file), strict=False)

    # Verify file was created
    assert ts3_file.exists()


def test_ts3_geometry_only_strict_mode(tmp_path):
    """Test that strict mode rejects geometry-only grids."""
    # Create a simple block with only coordinates
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    # Set minimal required properties for TS3
    block.set_rpm(0.0)
    block.set_Nb(1)

    # Create grid without setting any flow data
    grid = Grid([block])

    # Should fail in strict mode
    ts3_file = tmp_path / "test_strict_fail.ts3"
    with pytest.raises(ValueError):
        grid.write_ts3(str(ts3_file), strict=True)


def test_grid_ts3_methods_interface():
    """Test that Grid TS3 methods have correct interface."""
    # Test class method exists
    assert hasattr(Grid, "read_ts3"), "Grid should have read_ts3 class method"
    assert callable(Grid.read_ts3), "Grid.read_ts3 should be callable"

    # Test instance method exists
    grid = Grid()
    assert hasattr(grid, "write_ts3"), "Grid should have write_ts3 instance method"
    assert callable(grid.write_ts3), "Grid.write_ts3 should be callable"


def test_ts3_multiblock_with_patches_roundtrip(tmp_path):
    """Test TS3 round-trip with multiblock grid including inlet, outlet, and periodic patches."""
    # Create a three-block turbomachinery configuration
    # Block 1: Inlet block with inlet patch
    # Block 2: Middle block with periodic patches
    # Block 3: Outlet block with outlet patch

    # Create blocks with different shapes
    block1 = Block(shape=(5, 6, 8))  # Inlet block
    block2 = Block(shape=(6, 6, 8))  # Middle block
    block3 = Block(shape=(5, 6, 8))  # Outlet block

    # Set coordinates for each block to represent a turbomachinery passage
    # Block 1: x=0 to 0.5
    x1 = np.linspace(0.0, 0.5, 5)
    r1 = np.linspace(0.8, 1.2, 6)
    t1 = np.linspace(0.0, np.pi / 4, 8)
    xv1, rv1, tv1 = np.meshgrid(x1, r1, t1, indexing="ij")
    xrt1 = np.stack([xv1, rv1, tv1], axis=-1)

    # Block 2: x=0.4 to 0.9 (overlapping with block 1)
    x2 = np.linspace(0.4, 0.9, 6)
    r2 = np.linspace(0.8, 1.2, 6)
    t2 = np.linspace(0.0, np.pi / 4, 8)
    xv2, rv2, tv2 = np.meshgrid(x2, r2, t2, indexing="ij")
    xrt2 = np.stack([xv2, rv2, tv2], axis=-1)

    # Block 3: x=0.8 to 1.3 (overlapping with block 2)
    x3 = np.linspace(0.8, 1.3, 5)
    r3 = np.linspace(0.8, 1.2, 6)
    t3 = np.linspace(0.0, np.pi / 4, 8)
    xv3, rv3, tv3 = np.meshgrid(x3, r3, t3, indexing="ij")
    xrt3 = np.stack([xv3, rv3, tv3], axis=-1)

    # Set coordinates
    block1.set_x(xrt1[..., 0])
    block1.set_r(xrt1[..., 1])
    block1.set_t(xrt1[..., 2])
    block2.set_x(xrt2[..., 0])
    block2.set_r(xrt2[..., 1])
    block2.set_t(xrt2[..., 2])
    block3.set_x(xrt3[..., 0])
    block3.set_r(xrt3[..., 1])
    block3.set_t(xrt3[..., 2])

    # Create fluid and set flow properties
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)

    # Set up all blocks with fluid and flow data
    for i, block in enumerate([block1, block2, block3]):
        block.set_fluid(fluid)
        block.set_rpm(3600.0)  # Rotating at 3600 RPM
        block.set_Nb(24)  # 24 blades

        # Set realistic conserved variables for compressor flow
        rho = np.ones(block.shape, dtype=np.float32) * 1.2  # kg/m³
        rhoVx = np.ones(block.shape, dtype=np.float32) * 150.0 * rho  # Axial velocity
        rhoVr = np.zeros(block.shape, dtype=np.float32)  # Radial velocity
        rhoVt = (
            np.ones(block.shape, dtype=np.float32) * 100.0 * rho
        )  # Tangential velocity
        # Internal energy for realistic temperature
        rhoe = np.ones(block.shape, dtype=np.float32) * 2.5e5

        conserved = np.stack([rho, rhoVx, rhoVr, rhoVt, rhoe], axis=-1)
        block.set_conserved(conserved)
        block.set_wdist(np.ones(block.shape, dtype=np.float32) * 0.001)

    # Add inlet patch to block 1 (i=0 face)
    inlet_patch = InletPatch(i=0, j=(1, 5), k=(1, 7), label="inlet")
    inlet_patch.set_Po_To_Alpha_Beta(
        Po=np.ones((1, 5, 7)) * 101325.0,  # Total pressure (Pa)
        To=np.ones((1, 5, 7)) * 288.15,  # Total temperature (K)
        Alpha=np.zeros((1, 5, 7)),  # Yaw angle
        Beta=np.ones((1, 5, 7)) * 15.0,  # Pitch angle (degrees)
    )
    block1.patches.append(inlet_patch)

    # Add outlet patch to block 3 (i=-1 face)
    outlet_patch = OutletPatch(i=-1, j=(1, 5), k=(1, 7), label="outlet")
    outlet_patch.set_P(120000.0)  # Static pressure (Pa)
    block3.patches.append(outlet_patch)

    # Add periodic patches to all blocks (k=0 and k=-1 faces)
    for i, block in enumerate([block1, block2, block3]):
        # Lower periodic boundary (k=0)
        periodic_low = PeriodicPatch(
            i=(1, block.shape[0] - 1),
            j=(1, block.shape[1] - 1),
            k=0,
            label=f"periodic_low_block{i}",
        )
        block.patches.append(periodic_low)

        # Upper periodic boundary (k=-1)
        periodic_high = PeriodicPatch(
            i=(1, block.shape[0] - 1),
            j=(1, block.shape[1] - 1),
            k=-1,
            label=f"periodic_high_block{i}",
        )
        block.patches.append(periodic_high)

    # Create original grid
    original_grid = Grid([block1, block2, block3])

    # Write to TS3 file
    output_file = tmp_path / "test_multiblock_patches.ts3"
    original_grid.write_ts3(str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Read back the grid
    reconstructed_grid = Grid.read_ts3(str(output_file))

    # Verify grid structure
    assert len(reconstructed_grid) == 3, "Should have 3 blocks"

    # Compare each block's basic properties
    for i, (orig_block, recon_block) in enumerate(
        zip(original_grid, reconstructed_grid)
    ):
        assert orig_block.shape == recon_block.shape, f"Block {i} shape mismatch"
        assert orig_block.rpm == recon_block.rpm, f"Block {i} rpm mismatch"
        assert orig_block.Nb == recon_block.Nb, f"Block {i} Nb mismatch"

        # Compare coordinates (relaxed tolerance for HDF5 round-off)
        np.testing.assert_allclose(
            recon_block.xrt,
            orig_block.xrt,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Block {i} coordinates don't match after roundtrip",
        )

    # Verify patches were preserved
    # Block 1 should have inlet patch + 2 periodic patches
    assert len(reconstructed_grid[0].patches) == 3
    inlet_patches = [
        p for p in reconstructed_grid[0].patches if isinstance(p, InletPatch)
    ]
    periodic_patches_b1 = [
        p for p in reconstructed_grid[0].patches if isinstance(p, PeriodicPatch)
    ]
    assert len(inlet_patches) == 1, "Block 1 should have 1 inlet patch"
    assert len(periodic_patches_b1) == 2, "Block 1 should have 2 periodic patches"

    # Block 2 should have 2 periodic patches
    assert len(reconstructed_grid[1].patches) == 2
    periodic_patches_b2 = [
        p for p in reconstructed_grid[1].patches if isinstance(p, PeriodicPatch)
    ]
    assert len(periodic_patches_b2) == 2, "Block 2 should have 2 periodic patches"

    # Block 3 should have outlet patch + 2 periodic patches
    assert len(reconstructed_grid[2].patches) == 3
    outlet_patches = [
        p for p in reconstructed_grid[2].patches if isinstance(p, OutletPatch)
    ]
    periodic_patches_b3 = [
        p for p in reconstructed_grid[2].patches if isinstance(p, PeriodicPatch)
    ]
    assert len(outlet_patches) == 1, "Block 3 should have 1 outlet patch"
    assert len(periodic_patches_b3) == 2, "Block 3 should have 2 periodic patches"

    # Verify inlet patch properties were preserved
    recon_inlet = inlet_patches[0]
    np.testing.assert_allclose(
        recon_inlet.Po,
        inlet_patch.Po,
        rtol=1e-5,
        err_msg="Inlet total pressure not preserved",
    )
    np.testing.assert_allclose(
        recon_inlet.To,
        inlet_patch.To,
        rtol=1e-5,
        err_msg="Inlet total temperature not preserved",
    )
    np.testing.assert_allclose(
        recon_inlet.Beta,
        inlet_patch.Beta,
        rtol=1e-5,
        err_msg="Inlet pitch angle not preserved",
    )

    # Verify outlet patch properties were preserved
    recon_outlet = outlet_patches[0]
    np.testing.assert_allclose(
        recon_outlet.P,
        outlet_patch.P,
        rtol=1e-5,
        err_msg="Outlet pressure not preserved",
    )

    # Verify periodic patch connectivity is maintained
    original_connectivity = original_grid.connectivity.periodic.pair()
    reconstructed_connectivity = reconstructed_grid.connectivity.periodic.pair()

    # Should have same number of connections (both directions counted)
    assert len(original_connectivity) == len(reconstructed_connectivity), (
        "Periodic connectivity not preserved"
    )


def test_ts3_write_preserves_block_variables(tmp_path):
    """Regression: writing TS3 must preserve rpm/Nb block variables.

    Previously set_bv reset self.bv[bid] to defaults on every call (it used a
    dict-style membership guard against a list), so a post-get_blocks set_bv
    discarded the rpm/nblade/fblade/fracann set earlier by get_blocks, silently
    writing a rotating block as stationary with a single blade.
    """
    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])

    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    block.set_fluid(fluid)
    block.set_rpm(3600.0)
    block.set_Nb(24)

    rho = np.ones(block.shape, dtype=np.float32)
    rhoVx = np.zeros(block.shape, dtype=np.float32)
    rhoVr = np.zeros(block.shape, dtype=np.float32)
    rhoVt = np.zeros(block.shape, dtype=np.float32)
    rhoe = np.ones(block.shape, dtype=np.float32) * 1e5
    block.set_conserved(np.stack([rho, rhoVx, rhoVr, rhoVt, rhoe], axis=-1))
    block.set_wdist(np.ones(block.shape, dtype=np.float32) * 0.01)

    original_grid = Grid([block])

    ts3_file = tmp_path / "test_bv.ts3"
    original_grid.write_ts3(str(ts3_file))

    reconstructed_grid = Grid.read_ts3(str(ts3_file))

    recon_block = reconstructed_grid[0]
    assert recon_block.rpm == 3600.0, "rpm wiped by set_bv"
    assert recon_block.Nb == 24, "Nb wiped by set_bv"


def _make_inlet_writer():
    """A TS3Writer populated from a single-block grid with an inlet patch."""
    from ember.ts3 import TS3Writer

    block = Block(shape=(3, 4, 5))
    xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    inlet = InletPatch(i=0, j=(0, -1), k=(0, -1), label="inlet")
    block.patches.append(inlet)
    inlet.set_Po_To_Alpha_Beta(
        Po=np.full(inlet.shape, 1.0e5),
        To=np.full(inlet.shape, 300.0),
        Alpha=np.zeros(inlet.shape),
        Beta=np.zeros(inlet.shape),
    )
    writer = TS3Writer()
    writer.get(Grid([block]), strict=False)
    return writer


def test_set_pv_casts_existing_key():
    """set_pv updates an existing patch variable, casting to its type."""
    writer = _make_inlet_writer()
    writer.set_pv(0, 0, rfin=0.25)
    assert writer.pv[0][0]["rfin"] == pytest.approx(0.25)


def test_set_pv_rejects_unknown_name():
    """set_pv rejects a name not already present on the patch."""
    writer = _make_inlet_writer()
    with pytest.raises(Exception, match="Unknown patch variable"):
        writer.set_pv(0, 0, not_a_pv=1.0)


def test_set_pv_before_get_raises():
    """set_pv before get() raises a clear error."""
    from ember.ts3 import TS3Writer

    with pytest.raises(RuntimeError, match="before get"):
        TS3Writer().set_pv(0, 0, rfin=0.25)


# ---------------------------------------------------------------------------
# block_util.to_tm3
# ---------------------------------------------------------------------------


def _make_tri_block():
    """Return a minimal triangulated block via triangulate_to_unstructured."""
    import ember.block as _eb
    import ember.fluid as _ef
    from ember.cut import triangulate_to_unstructured

    shape = (5, 6)  # 2D block required by triangulate_to_unstructured
    b = _eb.Block(shape=shape)
    fluid = _ef.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)
    x = np.linspace(0.0, 1.0, shape[0])
    r = np.linspace(0.5, 1.5, shape[1])
    x_mesh, r_mesh = np.meshgrid(x, r, indexing="ij")
    _xrt = np.stack([x_mesh, r_mesh, np.zeros_like(x_mesh)], axis=-1)
    b.set_x(_xrt[..., 0])
    b.set_r(_xrt[..., 1])
    b.set_t(_xrt[..., 2])
    b.set_P_T(101325.0, 300.0)
    return triangulate_to_unstructured(b)


def test_to_tm3_roundtrip(tmp_path):
    """to_tm3 writes a readable binary file with correct vertex and triangle counts."""
    tri = _make_tri_block()
    ntri = tri.shape[0]
    values = np.ones(tri.shape, dtype=np.float32)
    path = tmp_path / "out.tm3"
    to_tm3(tri, str(path), Ma=values)

    raw = np.frombuffer(path.read_bytes(), dtype=np.int32)
    # header: nsteps=1, nsurfaces=1, name(96 bytes=24 int32), nverts, ntris, nprops
    nsteps = raw[0]
    nsurfaces = raw[1]
    # surface name occupies 96 bytes = 24 int32 words
    nverts = raw[2 + 24]
    ntris_file = raw[2 + 24 + 1]
    assert nsteps == 1
    assert nsurfaces == 1
    assert nverts == ntri * 3
    assert ntris_file == ntri


def test_to_tm3_requires_triangulated(tmp_path):
    """to_tm3 raises ValueError on a non-triangulated block."""
    b = Block(shape=(3, 4, 5))
    _xrt = util.linmesh3([0.0, 1.0], [0.5, 1.5], [0.0, 0.2], (3, 4, 5))
    b.set_x(_xrt[..., 0])
    b.set_r(_xrt[..., 1])
    b.set_t(_xrt[..., 2])
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    b.set_fluid(fluid)
    with pytest.raises(ValueError, match="triangulated"):
        to_tm3(b, str(tmp_path / "out.tm3"), Ma=np.ones(b.shape))


def test_to_tm3_requires_one_kwarg(tmp_path):
    """to_tm3 raises ValueError when given zero or two property kwargs."""
    tri = _make_tri_block()
    path = str(tmp_path / "out.tm3")
    with pytest.raises(ValueError, match="exactly one keyword argument"):
        to_tm3(tri, path)
    with pytest.raises(ValueError, match="exactly one keyword argument"):
        to_tm3(tri, path, Ma=np.ones(tri.shape), P=np.ones(tri.shape))


def test_to_tm3_shape_mismatch(tmp_path):
    """to_tm3 raises ValueError when property array shape does not match block."""
    tri = _make_tri_block()
    wrong = np.ones((tri.shape[0] + 1, tri.shape[1]))
    with pytest.raises(ValueError, match="shape"):
        to_tm3(tri, str(tmp_path / "out.tm3"), Ma=wrong)
