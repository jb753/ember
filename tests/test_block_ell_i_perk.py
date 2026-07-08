"""Tests for Block.ell and Block.i_perk.

Test cases:
- test_ell_shape: ell has the documented nodal shape (ni, nj, nk, 3)
- test_ell_isotropic_cell_is_uniform: a near-cubic-cell block (large radius,
  so curvature is negligible) gives weight ~1 in every direction
- test_i_perk_no_periodic_patch: no k-face PeriodicPatch gives (0, 0)
- test_i_perk_upstream_and_downstream: k-face PeriodicPatches restricted to
  upstream/downstream i-ranges give the corresponding 1-based bounds
"""

import numpy as np

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


def test_ell_shape():
    shape = (3, 4, 5)
    b = _make_block(shape)

    assert b.ell.shape == (*shape, 3)


def test_ell_isotropic_cell_is_uniform():
    # dx == dr, and r is large relative to the r-extent so the arc length
    # r * dtheta is nearly constant across the block: cells are near-cubic,
    # so the Martinelli weights should be isotropic (all ~1) everywhere.
    shape = (4, 4, 4)
    n_cell = shape[0] - 1
    r0 = 1000.0
    dx = dr = 1.0 / n_cell
    dt = dx / (r0 + 0.5 * (shape[0] - 1) * dr / n_cell)
    b = _make_block(
        shape, extents=((0.0, 1.0), (r0, r0 + n_cell * dr), (0.0, n_cell * dt))
    )

    np.testing.assert_allclose(b.ell, 1.0, rtol=2e-3)


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
