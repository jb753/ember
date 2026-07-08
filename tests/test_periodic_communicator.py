"""Tests for PeriodicCommunicator class.

Test cases:
- test_periodic_communicator_init: PeriodicCommunicator initialization and setup verification
- test_periodic_communicator_apply: Periodic boundary condition application and averaging
- test_periodic_apply_all_perm_flip: All permutation and flip combinations testing
"""

import pytest
import numpy as np
import ember.block
import ember.grid
from ember.patch import PeriodicPatch
import ember.patch
from ember.periodic_communicator import PeriodicCommunicator
import ember.fluid
from ember import util
import itertools


@pytest.fixture
def grid_with_periodic_patches():
    """Create a grid with two blocks that have matching periodic patches."""
    shape = (6, 7, 8)

    # Block 1
    b1 = ember.block.Block(shape=shape)
    L = 0.1
    xrt = util.linmesh3((0.0, L), (0.5, 1.0), (0.0, np.pi / 4), shape)
    b1.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])

    # Block 2 - identical geometry shifted in x
    b2 = b1.copy()
    b2.set_x(b2.x + L)

    # Add periodic patches on i=0 and i=-1 faces
    b1.patches.append(PeriodicPatch(i=-1))
    b2.patches.append(PeriodicPatch(i=0))

    # Set different conserved values for each block to test averaging
    b1.conserved_nd[...] = np.random.rand(*shape, 5)
    b2.conserved_nd[...] = np.random.rand(*shape, 5)

    # Create grid
    grid = ember.grid.Grid([b1, b2])

    return grid


def test_periodic_communicator_init(grid_with_periodic_patches):
    """Test PeriodicCommunicator initialization."""
    grid = grid_with_periodic_patches

    # Get periodic pairs
    periodic_pairs = grid.connectivity.periodic.pair()

    # Create communicator
    comm = PeriodicCommunicator(grid, periodic_pairs)

    # Check that pairs were created
    assert len(comm.pairs) > 0

    # Check that ijk indices were set up
    assert len(comm.ijk_node_flat) > 0


def test_periodic_communicator_apply(grid_with_periodic_patches):
    """Test PeriodicCommunicator apply method."""
    grid = grid_with_periodic_patches

    # Get patch slices and coordinates
    slices = [p.slice for p in grid.patches.periodic]
    xrt_coords = [b.xrt[s] for b, s in zip(grid, slices)]

    # Calculate sorting indices based on xrt product for order-independent comparison
    sort_indices = [
        np.argsort((xrt[..., 0] * xrt[..., 1] * xrt[..., 2]).ravel())
        for xrt in xrt_coords
    ]

    # Get conserved values before applying periodic BCs, flattened and sorted
    cons_initial = [
        b.conserved_nd[s].reshape(-1, 5)[sort_idx]
        for b, s, sort_idx in zip(grid, slices, sort_indices)
    ]

    # Verify initial values are different
    assert not np.allclose(cons_initial[0], cons_initial[1])

    # Get periodic pairs and create communicator
    periodic_pairs = grid.connectivity.periodic.pair()
    comm = PeriodicCommunicator(grid, periodic_pairs)

    # Apply periodic boundary conditions
    comm.apply()

    # Get final values after averaging, flattened and sorted
    cons_final = [
        b.conserved_nd[s].reshape(-1, 5)[sort_idx]
        for b, s, sort_idx in zip(grid, slices, sort_indices)
    ]

    # Calculate expected average and verify
    cons_avg = 0.5 * (cons_initial[0] + cons_initial[1])
    assert np.allclose(cons_final[0], cons_avg)
    assert np.allclose(cons_final[1], cons_avg)


def test_periodic_apply_all_perm_flip():
    """Test PeriodicCommunicator apply method with all permutations and flips."""

    # Loop over all permutations and flips
    for perm in itertools.permutations((0, 1, 2)):
        for flip in itertools.chain.from_iterable(
            itertools.combinations((0, 1, 2), r) for r in range(4)
        ):
            _test_single_perm_flip(perm, flip)


def _test_single_perm_flip(perm, flip):
    fluid = ember.fluid.PerfectFluid(gamma=1.4, cp=1005, Pr=0.71, mu=1.8e-5)
    shape = (7, 7, 7)

    # First block has straightforward indexing
    b1 = ember.block.Block(shape=shape)
    L = 0.1
    xrt1 = util.linmesh3([0.0, L], [0.9, 1.0], [0.0, 0.1], shape)
    b1.set_x(xrt1[..., 0]).set_r(xrt1[..., 1]).set_t(xrt1[..., 2])
    b1.set_fluid(fluid)
    b1.set_conserved(np.random.rand(*shape, 5) + 1)

    # Second block is shifted in x is perm/flip
    xrt2 = util.apply_perm_flip(xrt1, perm, flip)
    b2 = b1.copy()
    b2.set_x(xrt2[..., 0]).set_r(xrt2[..., 1]).set_t(xrt2[..., 2]).set_x(b2.x + L)
    b2.set_conserved(np.random.rand(*shape, 5) + 2)

    dim_i_new = perm.index(0)
    flip_i_new = dim_i_new in flip
    if dim_i_new == 0:
        if flip_i_new:
            patch = ember.patch.PeriodicPatch(i=-1)
        else:
            patch = ember.patch.PeriodicPatch(i=0)
    elif dim_i_new == 1:
        if flip_i_new:
            patch = ember.patch.PeriodicPatch(j=-1)
        else:
            patch = ember.patch.PeriodicPatch(j=0)
    else:
        if flip_i_new:
            patch = ember.patch.PeriodicPatch(k=-1)
        else:
            patch = ember.patch.PeriodicPatch(k=0)

    b1.patches.append(ember.patch.PeriodicPatch(i=-1))
    b2.patches.append(patch)

    grid = ember.grid.Grid([b1, b2])

    # Get patch slices and coordinates
    slices = [p.slice for p in grid.patches.periodic]

    # Calculate sorting indices based on xrt product for order-independent comparison
    sort_indices = [
        np.argsort(np.prod(b[s].xrt + 0.01, axis=-1).ravel())
        for s, b in zip(slices, grid)
    ]

    # Get conserved values before applying periodic BCs, flattened and sorted
    cons_initial = [
        b[s].conserved.reshape(-1, 5)[sort_idx]
        for b, s, sort_idx in zip(grid, slices, sort_indices)
    ]

    # Verify initial values are different
    assert not np.allclose(cons_initial[0], cons_initial[1])

    # Apply periodic boundary conditions
    comm = PeriodicCommunicator(grid, grid.connectivity.periodic.pair())
    comm.apply()

    # Get final values after averaging, flattened and sorted
    cons_final = [
        b[s].conserved[s].reshape(-1, 5)[sort_idx]
        for b, s, sort_idx in zip(grid, slices, sort_indices)
    ]

    # Calculate expected average and verify
    cons_avg = 0.5 * (cons_initial[0] + cons_initial[1])
    assert np.allclose(cons_final[0], cons_avg)
    assert np.allclose(cons_final[1], cons_avg)
