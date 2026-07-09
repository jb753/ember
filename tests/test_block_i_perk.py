"""Tests for Block.i_perk.

Test cases:
- test_i_perk_no_periodic_patch: no k-face PeriodicPatch gives (0, 0)
- test_i_perk_upstream_and_downstream: k-face PeriodicPatches restricted to
  upstream/downstream i-ranges give the corresponding 1-based bounds
"""

import ember.block
from ember import util
from ember.periodic import PeriodicPatch


def _make_block(shape, extents=((0.0, 1.0), (0.5, 1.5), (0.0, 0.2))):
    b = ember.block.Block(shape=shape)
    xrt = util.linmesh3(*extents, shape)
    b.set_x(xrt[..., 0])
    b.set_r(xrt[..., 1])
    b.set_t(xrt[..., 2])
    return b


def test_i_perk_no_periodic_patch():
    b = _make_block((9, 3, 5))
    assert b.i_perk == (0, 0)


def test_i_perk_upstream_and_downstream():
    shape = (9, 3, 5)
    b = _make_block(shape)

    # Upstream k-periodic interval over nodes i=0..2 (1-based end 3).
    b.patches.append(PeriodicPatch(i=(0, 2), k=0))
    # Downstream k-periodic interval over nodes i=6..8 (1-based start 7).
    b.patches.append(PeriodicPatch(i=(6, 8), k=-1))

    assert b.i_perk == (3, 7)
