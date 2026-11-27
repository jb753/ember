"""Tests for Grid.snap_periodic (ember.grid).

Module tested: ember.grid

Test cases:
- test_a_passage_closes_on_itself_a_pitch_apart: a block's own periodic faces
  end a whole pitch apart in theta, and equal in x and r
- test_each_face_moves_half_way: the two faces meet at their mean, and the
  distance moved is reported
- test_interior_nodes_do_not_move: only nodes on paired patches change
- test_corners_shared_by_several_pairs_agree: nodes joined through a chain of
  pairs, across an interface and a periodic face, all end up at one point
- test_a_grid_without_pairs_is_left_alone: no pairs, nothing moves, zero back
"""

import numpy as np

from ember import util
from ember.block import Block
from ember.grid import Grid
from ember.patch import PeriodicPatch

NB = 20
PITCH = 2 * np.pi / NB


def _block(x_lim, shape=(4, 3, 5)):
    """A block one pitch wide in theta, paired with itself across its k faces."""
    block = Block(shape=shape)
    block.set_Nb(NB)
    xrt = util.linmesh3(x_lim, [0.5, 1.0], [0.0, PITCH], shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.patches.append(PeriodicPatch(i=(0, -1), j=(0, -1), k=0, label="k0"))
    block.patches.append(PeriodicPatch(i=(0, -1), j=(0, -1), k=-1, label="nk"))
    return block


def _nudge(block, k, dx=0.0, dt=0.0):
    """Move the nodes of face k by small amounts, as a mesher's error might."""
    x = np.array(block.x, dtype=np.float64)
    t = np.array(block.t, dtype=np.float64)
    x[:, :, k] += dx
    t[:, :, k] += dt
    block.set_x(x)
    block.set_t(t)


def _float32_step(value):
    return float(np.spacing(np.float32(np.max(np.abs(value)))))


def test_a_passage_closes_on_itself_a_pitch_apart():
    block = _block([0.0, 1.0])
    _nudge(block, -1, dx=2e-6, dt=4e-6)
    Grid([block]).snap_periodic()

    np.testing.assert_array_equal(block.x[:, :, 0], block.x[:, :, -1])
    np.testing.assert_array_equal(block.r[:, :, 0], block.r[:, :, -1])
    gap = np.float64(block.t[:, :, -1]) - np.float64(block.t[:, :, 0])
    assert np.abs(gap - PITCH).max() <= 2 * _float32_step(block.t)


def test_each_face_moves_half_way():
    block = _block([0.0, 1.0])
    x_lo = np.array(block.x[:, :, 0], dtype=np.float64)
    _nudge(block, -1, dx=2e-6)
    moved = Grid([block]).snap_periodic()

    np.testing.assert_allclose(
        block.x[:, :, 0], x_lo + 1e-6, atol=_float32_step(block.x)
    )
    assert abs(moved - 1e-6) < 2 * _float32_step(block.x)


def test_interior_nodes_do_not_move():
    block = _block([0.0, 1.0])
    before = np.array(block.xrt[:, :, 1:-1])
    _nudge(block, -1, dx=2e-6, dt=4e-6)
    Grid([block]).snap_periodic()
    np.testing.assert_array_equal(block.xrt[:, :, 1:-1], before)


def test_corners_shared_by_several_pairs_agree():
    # Two passages end to end in i, joined by an interface; each is also
    # periodic with itself. A node on the interface at k = 0 is joined to its
    # copy in the other block, and to both their copies a pitch away.
    up = _block([0.0, 1.0])
    dn = _block([1.0, 2.0])
    up.patches.append(PeriodicPatch(i=-1, j=(0, -1), k=(0, -1), label="out"))
    dn.patches.append(PeriodicPatch(i=0, j=(0, -1), k=(0, -1), label="in"))
    _nudge(up, -1, dt=3e-6)
    _nudge(dn, 0, dx=1e-6)
    x = np.array(dn.x, dtype=np.float64)
    x[0] += 2e-6
    dn.set_x(x)
    Grid([up, dn]).snap_periodic()

    corners = [up.xrt[-1, :, 0], up.xrt[-1, :, -1], dn.xrt[0, :, 0], dn.xrt[0, :, -1]]
    for c in corners[1:]:
        np.testing.assert_array_equal(c[:, :2], corners[0][:, :2])
    shift = [0.0, PITCH, 0.0, PITCH]
    t0 = np.float64(corners[0][:, 2])
    for c, s in zip(corners, shift):
        assert np.abs(np.float64(c[:, 2]) - s - t0).max() <= 2 * _float32_step(up.t)


def test_a_grid_without_pairs_is_left_alone():
    block = _block([0.0, 1.0])
    block.patches.clear()
    before = np.array(block.xrt)
    assert Grid([block]).snap_periodic() == 0.0
    np.testing.assert_array_equal(block.xrt, before)
