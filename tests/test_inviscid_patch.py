"""Tests for InviscidPatch wall detection behavior.

Module tested: ember.patch.InviscidPatch and wall detection integration

Test cases:
- test_inviscid_patch_creation: Basic InviscidPatch functionality
- test_inviscid_patch_wall_detection: InviscidPatch in slip but not permeable
- test_inviscid_patch_vs_wall: Verify InviscidPatch eliminates friction but not distance walls
- test_inviscid_patch_vs_inlet: Compare InviscidPatch behavior with InletPatch
- test_inviscid_patch_demonstration: Formal demonstration of InviscidPatch behavior
"""

import numpy as np
import pytest
import ember.patch
import ember.block
from ember import util
from ember.fluid import PerfectFluid


def test_inviscid_patch_creation():
    """Test basic InviscidPatch creation and properties."""
    patch = ember.patch.InviscidPatch(i=0, j=(0, -1), k=(0, -1), label="test_inviscid")

    assert patch._collection_name == "inviscid"
    assert patch.label == "test_inviscid"
    assert patch.const_dim == 0  # i-direction patch

    # Check it exists in SLIP_TYPES but not PERMEABLE_TYPES
    assert ember.patch.InviscidPatch in ember.patch.SLIP_TYPES
    assert ember.patch.InviscidPatch not in ember.patch.PERMEABLE_TYPES


def test_inviscid_patch_wall_detection():
    """Test that InviscidPatch appears in appropriate wall detection lists."""
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    block = ember.block.Block(shape=(5, 5, 5))
    block.set_fluid(fluid)

    inviscid_patch = ember.patch.InviscidPatch(i=0, label="inviscid")
    block.patches["inviscid"] = inviscid_patch

    slip_patches = block.patches.slip
    permeable_patches = block.patches.permeable

    # InviscidPatch should be in slip but NOT in permeable
    assert len(slip_patches) == 1
    assert isinstance(slip_patches[0], ember.patch.InviscidPatch)
    assert len(permeable_patches) == 0


@pytest.fixture
def block():
    """Create a test block with simple geometry."""
    shape = ni, nj, nk = (6, 4, 4)
    b = ember.block.Block(shape=shape)
    L = 0.1
    rm = 1.0
    dr = 0.1
    pitch = 0.1
    xrt = util.linmesh3([0, L], [rm - dr / 2, rm + dr / 2], [0.0, pitch], shape)
    b.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    return b


def test_inviscid_patch_vs_wall(block):
    """Test that InviscidPatch eliminates friction walls but not distance walls."""
    # Add InviscidPatch to i=0 before accessing cached ijk_wall_visc
    block.patches["inviscid"] = ember.patch.InviscidPatch(i=0, label="inviscid")

    # Friction: InviscidPatch is in slip -> i=0 face becomes free (1.0)
    ws = block.ijk_wall_visc
    assert np.all(ws["walli1"] == 1.0)  # i=0 no longer a friction wall
    assert np.all(ws["wallni"] == 0.0)  # i=-1 still friction wall
    assert np.all(ws["wallj1"] == 0.0)
    assert np.all(ws["wallnj"] == 0.0)
    assert np.all(ws["wallk1"] == 0.0)
    assert np.all(ws["wallnk"] == 0.0)

    # Distance: InviscidPatch is NOT in permeable -> i=0 face stays a wall
    assert block._wall_nodes[0, :, :].all()
    assert block._wall_nodes[-1, :, :].all()
    assert block._wall_nodes[:, 0, :].all()
    assert block._wall_nodes[:, -1, :].all()
    assert block._wall_nodes[:, :, 0].all()
    assert block._wall_nodes[:, :, -1].all()


def test_inviscid_patch_vs_inlet():
    """Compare InviscidPatch behavior with InletPatch for wall detection."""
    shape = (5, 4, 3)
    xrt = util.linmesh3([0, 0.1], [0.95, 1.05], [0.0, 0.1], shape)

    block_inlet = ember.block.Block(shape=shape)
    block_inlet.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    block_inlet.patches["inlet"] = ember.patch.InletPatch(i=0, label="inlet")

    block_inviscid = ember.block.Block(shape=shape)
    block_inviscid.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    block_inviscid.patches["inviscid"] = ember.patch.InviscidPatch(
        i=0, label="inviscid"
    )

    # Both InletPatch and InviscidPatch are in SLIP_TYPES -> both free the i=0 friction face
    assert np.all(block_inlet.ijk_wall_visc["walli1"] == 1.0)
    assert np.all(block_inviscid.ijk_wall_visc["walli1"] == 1.0)

    # InletPatch is in PERMEABLE_TYPES -> frees interior nodes of i=0 from distance walls
    # InviscidPatch is NOT -> i=0 nodes stay as distance walls
    inlet_walls_at_i0 = block_inlet._wall_nodes[0, :, :].sum()
    inviscid_walls_at_i0 = block_inviscid._wall_nodes[0, :, :].sum()

    assert inlet_walls_at_i0 < inviscid_walls_at_i0
    assert block_inviscid._wall_nodes[0, :, :].all()


def test_inviscid_patch_demonstration():
    """Formal test demonstrating InviscidPatch wall detection behavior."""
    shape = (5, 4, 3)
    xrt = util.linmesh3([0, 0.1], [0.95, 1.05], [0.0, 0.1], shape)

    # Block with no patches: i=0 is both a friction and distance wall
    block_bare = ember.block.Block(shape=shape)
    block_bare.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    assert np.all(block_bare.ijk_wall_visc["walli1"] == 0.0), (
        "i=0 should initially be friction wall"
    )
    assert block_bare._wall_nodes[0, :, :].all(), (
        "i=0 should initially be distance wall"
    )

    # Block with InviscidPatch at i=0
    block = ember.block.Block(shape=shape)
    block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    block.patches["inviscid"] = ember.patch.InviscidPatch(i=0, label="inviscid")

    # InviscidPatch ELIMINATES friction walls (it is in SLIP_TYPES)
    assert np.all(block.ijk_wall_visc["walli1"] == 1.0), (
        "InviscidPatch should eliminate friction walls at i=0"
    )
    assert np.all(block.ijk_wall_visc["wallni"] == 0.0), (
        "i=-1 should remain friction wall"
    )

    # InviscidPatch PRESERVES distance walls (it is NOT in PERMEABLE_TYPES)
    assert block._wall_nodes[0, :, :].all(), (
        "InviscidPatch should preserve distance walls at i=0"
    )
    assert block._wall_nodes[-1, :, :].all(), "i=-1 should remain distance wall"
