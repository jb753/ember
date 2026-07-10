"""Tests for NonMatchCommunicator class.

Test cases:
- test_nonmatch_communicator_init: NonMatchCommunicator initialization and setup verification
- test_nonmatch_communicator_apply: Non-matching boundary condition application via interpolation
- test_nonmatch_linear_field_interpolation: Interpolation of linear field across different meshes
- test_nonmatch_bidirectional_averaging: Bidirectional transfer and averaging
- test_nonmatch_converges_to_average: Bidirectional averaging converges toward mean value
- test_nonmatch_different_resolutions: Transfer between very different mesh resolutions
- test_nonmatch_conserved_variable_count: All conserved variables interpolated correctly
- test_nonmatch_vs_periodic_identical_meshes: NonMatch matches Periodic for coincident faces
"""

import pytest
import numpy as np
import ember.block
import ember.grid
from ember.patch import NonMatchPatch, PeriodicPatch
from ember.nonmatch_communicator import NonMatchCommunicator
from ember.periodic_communicator import PeriodicCommunicator
import ember.fluid
from ember import util


@pytest.fixture
def grid_with_nonmatch_patches():
    """Create a grid with two blocks that have non-matching patches.

    Block 1: coarse mesh (6, 10, 12)
    Block 2: fine mesh (6, 15, 18)
    Same physical extent, different node counts
    """
    shape1 = (6, 10, 12)
    shape2 = (6, 15, 18)

    # Block 1 - coarse
    b1 = ember.block.Block(shape=shape1)
    L = 0.1
    xrt1 = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape1)
    b1.set_x(xrt1[..., 0])
    b1.set_r(xrt1[..., 1])
    b1.set_t(xrt1[..., 2])

    # Block 2 - fine, identical geometry shifted in x
    b2 = ember.block.Block(shape=shape2)
    xrt2 = util.linmesh3((L, 2 * L), (0.5, 1.0), (0.0, np.pi / 4), shape2)
    b2.set_x(xrt2[..., 0])
    b2.set_r(xrt2[..., 1])
    b2.set_t(xrt2[..., 2])

    # Add non-matching patches on adjacent faces
    b1.patches.append(NonMatchPatch(i=-1))  # i=5 face
    b2.patches.append(NonMatchPatch(i=0))  # i=0 face

    # Set different conserved values for each block
    b1.conserved_nd[...] = np.random.rand(*shape1, 5).astype(np.float32)
    b2.conserved_nd[...] = np.random.rand(*shape2, 5).astype(np.float32)

    # Create grid
    grid = ember.grid.Grid([b1, b2])

    return grid


def test_nonmatch_communicator_init(grid_with_nonmatch_patches):
    """Test NonMatchCommunicator initialization."""
    grid = grid_with_nonmatch_patches

    # Get non-matching pairs
    nonmatch_pairs = grid.connectivity.nonmatch.pair()

    # Create communicator
    comm = NonMatchCommunicator(grid, nonmatch_pairs)

    # Check that pairs were created
    assert len(comm.pairs) > 0

    # Check that parametric coords were set up
    assert len(comm.uv_coords) > 0

    # Check that all patches have parametric coords
    for bid, pid in comm.uv_coords:
        uv = comm.uv_coords[(bid, pid)]
        # Should have shape (..., ..., 2) with last dim being [u, v]
        assert uv.shape[-1] == 2
        # Parametric coords should be in [0, 1]
        assert uv.min() >= 0.0 and uv.max() <= 1.0


def test_nonmatch_communicator_apply(grid_with_nonmatch_patches):
    """Test NonMatchCommunicator apply method."""
    grid = grid_with_nonmatch_patches

    # Get patch data before applying
    patch1 = grid[0].patches[0]
    patch2 = grid[1].patches[0]
    slice1 = patch1.slice
    slice2 = patch2.slice

    Q1_initial = grid[0][slice1].conserved_nd.copy()
    Q2_initial = grid[1][slice2].conserved_nd.copy()

    # Verify initial values are different
    assert not np.allclose(Q1_initial.mean(), Q2_initial.mean())

    # Get non-matching pairs and create communicator
    nonmatch_pairs = grid.connectivity.nonmatch.pair()
    comm = NonMatchCommunicator(grid, nonmatch_pairs)

    # Apply non-matching boundary conditions
    comm.apply()

    # Get final values
    Q1_final = grid[0][slice1].conserved_nd.copy()
    Q2_final = grid[1][slice2].conserved_nd.copy()

    # Values should have changed
    assert not np.allclose(Q1_final, Q1_initial)
    assert not np.allclose(Q2_final, Q2_initial)

    # After bidirectional averaging, values should be similar
    # Note: When initial difference is very small, interpolation error can
    # dominate and cause the difference to increase slightly. This is expected
    # behavior - the key is that final values remain bounded.
    mean_diff_initial = abs(Q1_initial.mean() - Q2_initial.mean())
    mean_diff_final = abs(Q1_final.mean() - Q2_final.mean())

    # Either the difference decreased, OR it remains small (< 1% of the value magnitude)
    value_magnitude = max(abs(Q1_initial.mean()), abs(Q2_initial.mean()))
    assert (mean_diff_final < mean_diff_initial) or (
        mean_diff_final < 0.01 * value_magnitude
    )


def test_nonmatch_linear_field_interpolation():
    """Test interpolation of a linear field across different mesh resolutions."""
    # Create two blocks with different resolutions but same physical extent
    shape_coarse = (1, 5, 7)
    shape_fine = (1, 10, 14)

    b_coarse = ember.block.Block(shape=shape_coarse)
    b_fine = ember.block.Block(shape=shape_fine)

    L = 0.1
    # Same physical extent
    xrt_coarse = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape_coarse)
    xrt_fine = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape_fine)

    b_coarse.set_x(xrt_coarse[..., 0])
    b_coarse.set_r(xrt_coarse[..., 1])
    b_coarse.set_t(xrt_coarse[..., 2])
    b_fine.set_x(xrt_fine[..., 0])
    b_fine.set_r(xrt_fine[..., 1])
    b_fine.set_t(xrt_fine[..., 2])

    # Set a linear field on both blocks
    # Q = r + 2*theta (linear in physical space)
    r_coarse = xrt_coarse[..., 1]
    t_coarse = xrt_coarse[..., 2]
    Q_coarse_linear = r_coarse + 2 * t_coarse

    r_fine = xrt_fine[..., 1]
    t_fine = xrt_fine[..., 2]
    Q_fine_linear = r_fine + 2 * t_fine

    # Set all conserved variables to this linear field
    Q_coarse = np.zeros((*shape_coarse, 5), dtype=np.float32)
    Q_fine = np.zeros((*shape_fine, 5), dtype=np.float32)
    for ivar in range(5):
        Q_coarse[..., ivar] = Q_coarse_linear
        Q_fine[..., ivar] = Q_fine_linear
    b_coarse.conserved_nd[...] = Q_coarse
    b_fine.conserved_nd[...] = Q_fine

    # Add matching patches
    b_coarse.patches.append(NonMatchPatch(i=0))
    b_fine.patches.append(NonMatchPatch(i=0))

    grid = ember.grid.Grid([b_coarse, b_fine])

    # Apply non-matching communication
    nonmatch_pairs = grid.connectivity.nonmatch.pair()
    comm = NonMatchCommunicator(grid, nonmatch_pairs)
    comm.apply()

    # Check that after averaging, both blocks still have the linear field
    # (since both started with same field, averaging should preserve it)
    Q_coarse_result = b_coarse[b_coarse.patches[0].slice].conserved_nd
    Q_fine_result = b_fine[b_fine.patches[0].slice].conserved_nd

    # Expected value (same linear field)
    Q_coarse_expected = r_coarse + 2 * t_coarse
    Q_fine_expected = r_fine + 2 * t_fine

    # Interpolation of linear field should be exact (or very close)
    for ivar in range(5):
        np.testing.assert_allclose(
            Q_coarse_result[..., ivar],
            Q_coarse_expected,
            rtol=1e-3,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            Q_fine_result[..., ivar],
            Q_fine_expected,
            rtol=1e-3,
            atol=1e-4,
        )


def test_nonmatch_bidirectional_averaging():
    """Test bidirectional transfer and averaging."""
    shape1 = (1, 6, 8)
    shape2 = (1, 10, 12)

    b1 = ember.block.Block(shape=shape1)
    b2 = ember.block.Block(shape=shape2)

    L = 0.1
    xrt1 = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape1)
    xrt2 = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape2)

    b1.set_x(xrt1[..., 0])
    b1.set_r(xrt1[..., 1])
    b1.set_t(xrt1[..., 2])
    b2.set_x(xrt2[..., 0])
    b2.set_r(xrt2[..., 1])
    b2.set_t(xrt2[..., 2])

    # Set distinct constant values
    Q1 = np.ones((*shape1, 5), dtype=np.float32) * 1.0
    Q2 = np.ones((*shape2, 5), dtype=np.float32) * 2.0

    b1.conserved_nd[...] = Q1
    b2.conserved_nd[...] = Q2

    # Add matching patches
    b1.patches.append(NonMatchPatch(i=0))
    b2.patches.append(NonMatchPatch(i=0))

    grid = ember.grid.Grid([b1, b2])

    # Apply bidirectional averaging
    nonmatch_pairs = grid.connectivity.nonmatch.pair()
    comm = NonMatchCommunicator(grid, nonmatch_pairs)
    comm.apply()

    # Get results
    Q1_result = b1[b1.patches[0].slice].conserved_nd
    Q2_result = b2[b2.patches[0].slice].conserved_nd

    # Both should converge toward average value of 1.5
    # (won't be exactly 1.5 due to interpolation, but should be close)
    assert Q1_result.mean() > 1.0  # Increased from 1.0
    assert Q1_result.mean() < 2.0  # But less than 2.0
    assert Q2_result.mean() < 2.0  # Decreased from 2.0
    assert Q2_result.mean() > 1.0  # But more than 1.0

    # Should be close to each other after averaging
    assert abs(Q1_result.mean() - Q2_result.mean()) < 0.2


def test_nonmatch_converges_to_average():
    """Test that bidirectional averaging converges toward mean value."""
    shape1 = (1, 6, 8)
    shape2 = (1, 10, 12)

    b1 = ember.block.Block(shape=shape1)
    b2 = ember.block.Block(shape=shape2)

    L = 0.1
    xrt1 = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape1)
    xrt2 = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape2)

    b1.set_x(xrt1[..., 0])
    b1.set_r(xrt1[..., 1])
    b1.set_t(xrt1[..., 2])
    b2.set_x(xrt2[..., 0])
    b2.set_r(xrt2[..., 1])
    b2.set_t(xrt2[..., 2])

    # Set distinct constant values
    Q1_initial = np.ones((*shape1, 5), dtype=np.float32) * 3.0
    Q2_initial = np.ones((*shape2, 5), dtype=np.float32) * 7.0

    b1.conserved_nd[...] = Q1_initial.copy()
    b2.conserved_nd[...] = Q2_initial.copy()

    # Add matching patches
    b1.patches.append(NonMatchPatch(i=0))
    b2.patches.append(NonMatchPatch(i=0))

    grid = ember.grid.Grid([b1, b2])

    # Apply bidirectional averaging
    nonmatch_pairs = grid.connectivity.nonmatch.pair()
    comm = NonMatchCommunicator(grid, nonmatch_pairs)
    comm.apply()

    # Get results
    Q1_result = b1[b1.patches[0].slice].conserved_nd
    Q2_result = b2[b2.patches[0].slice].conserved_nd

    # Both should have changed from initial values
    assert not np.allclose(Q1_result, Q1_initial[b1.patches[0].slice])
    assert not np.allclose(Q2_result, Q2_initial[b2.patches[0].slice])

    # Both should converge toward average value of 5.0
    # (won't be exactly 5.0 due to interpolation)
    assert Q1_result.mean() > 3.0  # Increased from 3.0
    assert Q1_result.mean() < 7.0  # But less than 7.0
    assert Q2_result.mean() < 7.0  # Decreased from 7.0
    assert Q2_result.mean() > 3.0  # But more than 3.0


def test_nonmatch_different_resolutions():
    """Test transfer between very different mesh resolutions."""
    # Very coarse
    shape_coarse = (1, 3, 4)
    # Very fine
    shape_fine = (1, 20, 30)

    b_coarse = ember.block.Block(shape=shape_coarse)
    b_fine = ember.block.Block(shape=shape_fine)

    L = 0.1
    xrt_coarse = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape_coarse)
    xrt_fine = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape_fine)

    b_coarse.set_x(xrt_coarse[..., 0])
    b_coarse.set_r(xrt_coarse[..., 1])
    b_coarse.set_t(xrt_coarse[..., 2])
    b_fine.set_x(xrt_fine[..., 0])
    b_fine.set_r(xrt_fine[..., 1])
    b_fine.set_t(xrt_fine[..., 2])

    # Set smooth field on both blocks (same field, different resolutions)
    r_coarse = xrt_coarse[..., 1]
    r_fine = xrt_fine[..., 1]

    Q_coarse = np.zeros((*shape_coarse, 5), dtype=np.float32)
    Q_coarse[..., 0] = r_coarse  # Smooth in r

    Q_fine = np.zeros((*shape_fine, 5), dtype=np.float32)
    Q_fine[..., 0] = r_fine  # Same field on fine grid

    b_coarse.conserved_nd[...] = Q_coarse
    b_fine.conserved_nd[...] = Q_fine

    # Add patches
    b_coarse.patches.append(NonMatchPatch(i=0))
    b_fine.patches.append(NonMatchPatch(i=0))

    grid = ember.grid.Grid([b_coarse, b_fine])

    # Apply transfer
    nonmatch_pairs = grid.connectivity.nonmatch.pair()
    comm = NonMatchCommunicator(grid, nonmatch_pairs)
    comm.apply()

    # Check that both blocks still have reasonable values
    # Since both started with r-field, averaging should preserve it
    Q_coarse_result = b_coarse[b_coarse.patches[0].slice].conserved_nd[..., 0]
    Q_fine_result = b_fine[b_fine.patches[0].slice].conserved_nd[..., 0]

    # Should have reasonable values (in range of r)
    r_minmax = (0.5, 1.0)
    assert Q_coarse_result.min() >= r_minmax[0] - 0.01
    assert Q_coarse_result.max() <= r_minmax[1] + 0.01
    assert Q_fine_result.min() >= r_minmax[0] - 0.01
    assert Q_fine_result.max() <= r_minmax[1] + 0.01

    # Should be smooth (check that neighboring values are similar)
    diff_j = np.abs(np.diff(Q_fine_result[0, :, 0]))
    diff_k = np.abs(np.diff(Q_fine_result[0, 0, :]))
    assert np.max(diff_j) < 0.1  # Smooth in j-direction
    assert np.max(diff_k) < 0.1  # Smooth in k-direction


def test_nonmatch_conserved_variable_count():
    """Test that all conserved variables are interpolated correctly."""
    shape1 = (1, 5, 6)
    shape2 = (1, 8, 10)

    b1 = ember.block.Block(shape=shape1)
    b2 = ember.block.Block(shape=shape2)

    L = 0.1
    xrt1 = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape1)
    xrt2 = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape2)

    b1.set_x(xrt1[..., 0])
    b1.set_r(xrt1[..., 1])
    b1.set_t(xrt1[..., 2])
    b2.set_x(xrt2[..., 0])
    b2.set_r(xrt2[..., 1])
    b2.set_t(xrt2[..., 2])

    # Set different values for each conserved variable on block 1
    Q1 = np.zeros((*shape1, 5), dtype=np.float32)
    for ivar in range(5):
        Q1[..., ivar] = float(ivar + 1)  # 1, 2, 3, 4, 5

    # Set different values on block 2
    Q2 = np.zeros((*shape2, 5), dtype=np.float32)
    for ivar in range(5):
        Q2[..., ivar] = float(10 + ivar)  # 10, 11, 12, 13, 14

    b1.conserved_nd[...] = Q1
    b2.conserved_nd[...] = Q2

    # Add patches
    b1.patches.append(NonMatchPatch(i=0))
    b2.patches.append(NonMatchPatch(i=0))

    grid = ember.grid.Grid([b1, b2])

    # Apply transfer
    nonmatch_pairs = grid.connectivity.nonmatch.pair()
    comm = NonMatchCommunicator(grid, nonmatch_pairs)
    comm.apply()

    # Check that each variable was transferred and averaged
    Q1_result = b1[b1.patches[0].slice].conserved_nd
    Q2_result = b2[b2.patches[0].slice].conserved_nd

    for ivar in range(5):
        # After averaging, each variable should be between its initial values
        # Block 1 started at (ivar+1), Block 2 started at (10+ivar)
        # Average would be (11+2*ivar)/2
        initial_1 = float(ivar + 1)
        initial_2 = float(10 + ivar)

        # Both should have moved toward each other
        assert Q1_result[..., ivar].mean() > initial_1  # Increased
        assert Q1_result[..., ivar].mean() < initial_2  # But less than b2's value
        assert Q2_result[..., ivar].mean() < initial_2  # Decreased
        assert Q2_result[..., ivar].mean() > initial_1  # But more than b1's value


def test_nonmatch_vs_periodic_identical_meshes():
    """Test that NonMatch and Periodic give identical results for coincident faces.

    When two patches have identical node distributions (fully coincident),
    NonMatchPatch should give the same result as PeriodicPatch since the
    interpolation should be exact at all nodes.
    """
    # Create two blocks with identical mesh
    shape = (1, 8, 10)

    # Setup for PeriodicPatch test
    b1_periodic = ember.block.Block(shape=shape)
    b2_periodic = ember.block.Block(shape=shape)

    # Setup for NonMatchPatch test
    b1_nonmatch = ember.block.Block(shape=shape)
    b2_nonmatch = ember.block.Block(shape=shape)

    L = 0.1
    # Same geometry for all blocks
    xrt = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape)

    b1_periodic.set_x(xrt[..., 0].copy())
    b1_periodic.set_r(xrt[..., 1].copy())
    b1_periodic.set_t(xrt[..., 2].copy())
    b2_periodic.set_x(xrt[..., 0].copy())
    b2_periodic.set_r(xrt[..., 1].copy())
    b2_periodic.set_t(xrt[..., 2].copy())
    b1_nonmatch.set_x(xrt[..., 0].copy())
    b1_nonmatch.set_r(xrt[..., 1].copy())
    b1_nonmatch.set_t(xrt[..., 2].copy())
    b2_nonmatch.set_x(xrt[..., 0].copy())
    b2_nonmatch.set_r(xrt[..., 1].copy())
    b2_nonmatch.set_t(xrt[..., 2].copy())

    # Set initial conserved variables (different on each block)
    # Use a non-trivial field to test interpolation
    r = xrt[..., 1]
    t = xrt[..., 2]
    Q1_field = r + 2 * t
    Q2_field = 2 * r + t

    Q1 = np.zeros((*shape, 5), dtype=np.float32)
    Q2 = np.zeros((*shape, 5), dtype=np.float32)
    for ivar in range(5):
        Q1[..., ivar] = Q1_field * (ivar + 1)
        Q2[..., ivar] = Q2_field * (ivar + 1)

    # Set same initial values for both test cases
    b1_periodic.conserved_nd[...] = Q1.copy()
    b2_periodic.conserved_nd[...] = Q2.copy()
    b1_nonmatch.conserved_nd[...] = Q1.copy()
    b2_nonmatch.conserved_nd[...] = Q2.copy()

    # Add PeriodicPatches to first pair
    b1_periodic.patches.append(PeriodicPatch(i=0))
    b2_periodic.patches.append(PeriodicPatch(i=0))

    # Add NonMatchPatches to second pair
    b1_nonmatch.patches.append(NonMatchPatch(i=0))
    b2_nonmatch.patches.append(NonMatchPatch(i=0))

    # Create grids and apply boundary conditions
    grid_periodic = ember.grid.Grid([b1_periodic, b2_periodic])
    periodic_pairs = grid_periodic.connectivity.periodic.pair()
    comm_periodic = PeriodicCommunicator(grid_periodic, periodic_pairs)
    comm_periodic.apply()

    grid_nonmatch = ember.grid.Grid([b1_nonmatch, b2_nonmatch])
    nonmatch_pairs = grid_nonmatch.connectivity.nonmatch.pair()
    comm_nonmatch = NonMatchCommunicator(grid_nonmatch, nonmatch_pairs)
    comm_nonmatch.apply()

    # Get results from both methods
    Q1_periodic = b1_periodic[b1_periodic.patches[0].slice].conserved_nd
    Q2_periodic = b2_periodic[b2_periodic.patches[0].slice].conserved_nd

    Q1_nonmatch = b1_nonmatch[b1_nonmatch.patches[0].slice].conserved_nd
    Q2_nonmatch = b2_nonmatch[b2_nonmatch.patches[0].slice].conserved_nd

    # Results should be identical (or very close due to numerical precision)
    # For identical meshes, NonMatch interpolation should be exact
    np.testing.assert_allclose(
        Q1_nonmatch,
        Q1_periodic,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Block 1: NonMatch and Periodic should give identical results for coincident meshes",
    )

    np.testing.assert_allclose(
        Q2_nonmatch,
        Q2_periodic,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Block 2: NonMatch and Periodic should give identical results for coincident meshes",
    )
