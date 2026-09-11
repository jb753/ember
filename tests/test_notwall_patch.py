"""Tests for NotWallPatch, the collapsed-face marker.

Module tested: ember.notwall.NotWallPatch and its classification

A degenerate block has a face with no area, where the grid lines have
converged to a line or a point. It carries no flux and so needs no boundary
condition, but it must not be taken for a wall: every unpatched face is one,
and a wall there would seed the wall-distance search with a line of zero
distance through the middle of the flow.

Test cases:
- test_notwall_is_permeable_not_slip: how the one classification point sees it
- test_notwall_is_not_a_wall: the marker does its job on a real block
- test_collection_lists_notwall_separately: the accessor picks it out of
  permeable, which holds it alongside faces flow really does pass through
"""

import ember.block
import ember.patch
from ember.fluid import PerfectFluid


def test_notwall_is_permeable_not_slip():
    """It is classified where a face is told from a wall, and nowhere else."""
    patch = ember.patch.NotWallPatch(i=0, label="tip")

    assert patch.label == "tip"
    assert patch.const_dim == 0
    assert ember.patch.NotWallPatch in ember.patch.PERMEABLE_TYPES
    # SLIP_TYPES is a superset of PERMEABLE_TYPES, and a face with no area
    # applies no friction for the same reason it carries no flux.
    assert ember.patch.NotWallPatch in ember.patch.SLIP_TYPES


def test_notwall_is_not_a_wall():
    """The point of the marker: a collapsed face stops counting as a wall."""
    block = ember.block.Block(shape=(5, 5, 5))
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7))

    block.patches["tip"] = ember.patch.NotWallPatch(i=0, label="tip")

    assert len(block.patches.permeable) == 1
    assert len(block.patches.slip) == 1


def test_collection_lists_notwall_separately():
    """Permeable holds it with faces flow passes through; notwall does not."""
    block = ember.block.Block(shape=(5, 5, 5))
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7))

    block.patches["tip"] = ember.patch.NotWallPatch(i=0, label="tip")
    block.patches["cusp"] = ember.patch.CuspPatch(k=0, label="cusp")

    notwall = block.patches.notwall

    assert len(notwall) == 1
    assert notwall[0].label == "tip"
    assert len(block.patches.permeable) == 2
