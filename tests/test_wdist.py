"""Tests for Grid.calculate_wdist() and the node-smearing logic in
ember.fluxes.get_ijk_wall_distance.

## Node smearing

A node is a wall if *every* face touching it is a wall. The number of faces
touching a node depends on its topological position:

  interior  thresh=0  always non-wall (inside the block)
  face      thresh=8  4 i/j/k-faces touch it; all must be non-wall
  edge      thresh=4  2 faces touch it; both must be non-wall
  corner    thresh=3  3 faces touch it; all must be non-wall

Consequence: making one face of a block permeable frees the face-centre
nodes of that face but leaves edge and corner nodes as walls (they are
still touched by the adjacent wall faces).

## calculate_wdist integration

PERMEABLE_TYPES patches (Inlet, Outlet, Periodic, Mixing, NonMatch) remove a
face from the wall set. Nodes strictly interior to that face get a wall
distance comparable to the cell spacing. Corner/edge nodes of the permeable
face are shared with adjacent wall faces and remain near-zero. All other patch
types (Inviscid, Rotating, Cooling, Cusp, plain wall) leave the face as a
wall, so all nodes on it stay at near-zero distance (float32 KDTree noise,
< 1e-5).

Test cases:
Node smearing (via get_ijk_wall_distance):
- test_smear_no_patches_all_boundary_nodes_are_walls
- test_smear_interior_nodes_never_walls
- test_smear_face_centre_freed_by_single_permeable_face
- test_smear_face_centre_node_needs_all_four_touching_faces_free
- test_smear_edge_node_freed_only_when_both_adjacent_faces_permeable
- test_smear_corner_node_freed_only_when_all_three_adjacent_faces_permeable

calculate_wdist integration:
- test_no_patches_all_walls
- test_inlet_removes_wall
- test_outlet_removes_wall
- test_periodic_removes_wall
- test_mixing_removes_wall
- test_nonmatch_removes_wall
- test_inviscid_keeps_wall
- test_rotating_keeps_wall
- test_cooling_keeps_wall
- test_cusp_keeps_wall
- test_all_permeable_raises
"""

import numpy as np
import pytest

import ember.block
from ember import util
from ember.grid import Grid
from ember.patch import (
    InletPatch,
    OutletPatch,
    PeriodicPatch,
    MixingPatch,
    NonMatchPatch,
    InviscidPatch,
    RotatingPatch,
    CoolingPatch,
    CuspPatch,
)

# Threshold separating "is a wall" (near-zero KDTree noise on coincident points)
# from "not a wall" (true distance to the nearest remaining wall face).
_WALL_THRESHOLD = 1e-5


@pytest.fixture
def block():
    """Simple 3D block with uniform Cartesian-like geometry."""
    shape = (5, 4, 4)
    xrt = util.linmesh3([0.0, 0.1], [0.95, 1.05], [0.0, 0.1], shape)
    b = ember.block.Block(shape=shape)
    b.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    return b


# ---------------------------------------------------------------------------
# Node smearing: tests via block._wall_nodes (no KDTree involved)
#
# Threshold rules:
#   interior  thresh=0  always non-wall
#   face      thresh=8  all 4 touching faces must be non-wall
#   edge      thresh=4  both touching faces must be non-wall
#   corner    thresh=3  all 3 touching faces must be non-wall
# ---------------------------------------------------------------------------


def test_smear_no_patches_all_boundary_nodes_are_walls(block):
    """With no patches every boundary node is a wall node."""
    wall = block._wall_nodes
    assert wall[0, :, :].all()
    assert wall[-1, :, :].all()
    assert wall[:, 0, :].all()
    assert wall[:, -1, :].all()
    assert wall[:, :, 0].all()
    assert wall[:, :, -1].all()


def test_smear_interior_nodes_never_walls(block):
    """Interior nodes (not on any boundary face) are never wall nodes."""
    assert not block._wall_nodes[1:-1, 1:-1, 1:-1].any()


def test_smear_face_centre_freed_by_single_permeable_face(block):
    """Nodes interior to a permeable face (not on any edge/corner) become non-wall."""
    block.patches.append(InletPatch(i=0))
    wall = block._wall_nodes
    # Interior of the i=0 face: only one face (i=0) touches these nodes
    # and it is now permeable, so they are no longer walls
    assert not wall[0, 1:-1, 1:-1].any()
    # The rest of the i=0 boundary (edges/corners) still touches other walls
    assert wall[0, 0, :].all()
    assert wall[0, -1, :].all()
    assert wall[0, :, 0].all()
    assert wall[0, :, -1].all()


def test_smear_face_centre_node_needs_all_four_touching_faces_free(block):
    """A partial patch covering only part of a face does not free its uncovered nodes."""
    ni, nj, nk = block.shape
    # Patch covers only the first half of the j/k extent on i=0
    block.patches.append(InletPatch(i=0, j=(0, nj // 2 - 1), k=(0, nk // 2 - 1)))
    wall = block._wall_nodes
    # Nodes inside the patch footprint (interior, not on any edge) are freed
    assert not wall[0, 1 : nj // 2 - 1, 1 : nk // 2 - 1].any()
    # Nodes outside the patch footprint on the same face remain walls
    assert wall[0, nj // 2 :, :].all()


def test_smear_edge_node_freed_only_when_both_adjacent_faces_permeable(block):
    """An edge node (shared by two faces) is freed only when both faces are permeable."""
    # Edge at i=0, j=0 is shared by the i=0 and j=0 faces
    block.patches.append(InletPatch(i=0))
    # With only i=0 permeable the i=0,j=0 edge nodes are still walls
    assert block._wall_nodes[0, 0, :].all()

    # Now also make j=0 permeable
    block.patches.append(PeriodicPatch(j=0))
    # Interior of the shared edge (not at the corners) is now freed
    assert not block._wall_nodes[0, 0, 1:-1].any()


def test_smear_corner_node_freed_only_when_all_three_adjacent_faces_permeable(block):
    """A corner node is freed only when all three faces meeting at it are permeable."""
    # Corner (0, 0, 0) is touched by i=0, j=0, k=0 faces
    block.patches.append(InletPatch(i=0))
    block.patches.append(PeriodicPatch(j=0))
    # Two of three faces permeable: corner is still a wall
    assert block._wall_nodes[0, 0, 0]

    block.patches.append(NonMatchPatch(k=0))
    # All three faces permeable: corner is freed
    assert not block._wall_nodes[0, 0, 0]


# ---------------------------------------------------------------------------
# No patches: all faces are walls so every boundary node stays near-zero
# ---------------------------------------------------------------------------


def test_no_patches_all_walls(block):
    """With no patches every boundary face has near-zero wdist (all are walls)."""
    Grid([block]).calculate_wdist()
    assert np.all(block.wdist[0, :, :] < _WALL_THRESHOLD)
    assert np.all(block.wdist[-1, :, :] < _WALL_THRESHOLD)
    assert np.all(block.wdist[:, 0, :] < _WALL_THRESHOLD)
    assert np.all(block.wdist[:, -1, :] < _WALL_THRESHOLD)
    assert np.all(block.wdist[:, :, 0] < _WALL_THRESHOLD)
    assert np.all(block.wdist[:, :, -1] < _WALL_THRESHOLD)


# ---------------------------------------------------------------------------
# PERMEABLE_TYPES: interior nodes of the permeable face get large wdist;
# the opposite face stays a wall.
# ---------------------------------------------------------------------------


def test_inlet_removes_wall(block):
    """InletPatch at i=0 removes that face from the wall set."""
    block.patches.append(InletPatch(i=0))
    Grid([block]).calculate_wdist()
    # Interior nodes of the permeable face are no longer on any wall
    assert np.all(block.wdist[0, 1:-1, 1:-1] > _WALL_THRESHOLD)
    # Opposite face remains a wall
    assert np.all(block.wdist[-1, :, :] < _WALL_THRESHOLD)


def test_outlet_removes_wall(block):
    """OutletPatch at i=-1 removes that face from the wall set."""
    block.patches.append(OutletPatch(i=-1))
    Grid([block]).calculate_wdist()
    assert np.all(block.wdist[-1, 1:-1, 1:-1] > _WALL_THRESHOLD)
    assert np.all(block.wdist[0, :, :] < _WALL_THRESHOLD)


def test_periodic_removes_wall(block):
    """PeriodicPatch at j=0 removes that face from the wall set."""
    block.patches.append(PeriodicPatch(j=0))
    Grid([block]).calculate_wdist()
    assert np.all(block.wdist[1:-1, 0, 1:-1] > _WALL_THRESHOLD)
    assert np.all(block.wdist[:, -1, :] < _WALL_THRESHOLD)


def test_mixing_removes_wall(block):
    """MixingPatch at j=-1 removes that face from the wall set."""
    block.patches.append(MixingPatch(j=-1))
    Grid([block]).calculate_wdist()
    assert np.all(block.wdist[1:-1, -1, 1:-1] > _WALL_THRESHOLD)
    assert np.all(block.wdist[:, 0, :] < _WALL_THRESHOLD)


def test_nonmatch_removes_wall(block):
    """NonMatchPatch at i=0 removes that face from the wall set."""
    block.patches.append(NonMatchPatch(i=0))
    Grid([block]).calculate_wdist()
    assert np.all(block.wdist[0, 1:-1, 1:-1] > _WALL_THRESHOLD)
    assert np.all(block.wdist[-1, :, :] < _WALL_THRESHOLD)


# ---------------------------------------------------------------------------
# Non-permeable patch types: all nodes on the face stay near-zero
# ---------------------------------------------------------------------------


def test_inviscid_keeps_wall(block):
    """InviscidPatch is in SLIP_TYPES but not PERMEABLE_TYPES: face stays a wall."""
    block.patches.append(InviscidPatch(i=0))
    Grid([block]).calculate_wdist()
    assert np.all(block.wdist[0, :, :] < _WALL_THRESHOLD)


def test_rotating_keeps_wall(block):
    """RotatingPatch does not remove a face from the wall set."""
    block.patches.append(RotatingPatch(i=0))
    Grid([block]).calculate_wdist()
    assert np.all(block.wdist[0, :, :] < _WALL_THRESHOLD)


def test_cooling_keeps_wall(block):
    """CoolingPatch does not remove a face from the wall set."""
    block.patches.append(CoolingPatch(i=0))
    Grid([block]).calculate_wdist()
    assert np.all(block.wdist[0, :, :] < _WALL_THRESHOLD)


def test_cusp_removes_wall(block):
    """CuspPatch is permeable: interior nodes of the k-face are no longer nearest-wall."""
    block.patches.append(CuspPatch(k=0))
    Grid([block]).calculate_wdist()
    assert np.all(block.wdist[1:-1, 1:-1, 0] > _WALL_THRESHOLD)
    assert np.all(block.wdist[:, :, -1] < _WALL_THRESHOLD)


def test_cusp_invalid_on_i_face(block):
    """CuspPatch on an i-face raises ValueError."""
    with pytest.raises(ValueError, match="constant-k face"):
        block.patches.append(CuspPatch(i=0))


def test_cusp_invalid_on_j_face(block):
    """CuspPatch on a j-face raises ValueError."""
    with pytest.raises(ValueError, match="constant-k face"):
        block.patches.append(CuspPatch(j=0))


# ---------------------------------------------------------------------------
# Edge case: no walls at all
# ---------------------------------------------------------------------------


def test_all_permeable_raises(block):
    """Grid with all six faces permeable raises ValueError (no wall nodes)."""
    block.patches.append(InletPatch(i=0))
    block.patches.append(OutletPatch(i=-1))
    block.patches.append(PeriodicPatch(j=0))
    block.patches.append(MixingPatch(j=-1))
    block.patches.append(NonMatchPatch(k=0))
    block.patches.append(NonMatchPatch(k=-1))
    with pytest.raises(ValueError, match="No wall nodes"):
        Grid([block]).calculate_wdist()
