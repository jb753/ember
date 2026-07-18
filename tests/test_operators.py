"""Tests for node/cell distribution operators (ember.fortran).

Module tested: ember.fortran

Test cases:
- test_node_to_cell: Node to cell distribution operations
- test_cell_to_node: Cell to node distribution operations
"""

import numpy as np
import ember.fortran


f32 = np.float32


def test_node_to_cell():
    """Check averaging of nodal values to cell centers."""

    # Make an ijk grid
    ni = 97
    nj = 65
    nk = 73

    # Generate a grid of indices
    iv = np.linspace(0.0, ni - 1.0, ni)
    jv = np.linspace(0.0, nj - 1.0, nj)
    kv = np.linspace(0.0, nk - 1.0, nk)
    i, j, k = np.meshgrid(iv, jv, kv, indexing="ij")

    i = np.asfortranarray(np.expand_dims(i, -1), dtype=f32)
    j = np.asfortranarray(np.expand_dims(j, -1), dtype=f32)
    k = np.asfortranarray(np.expand_dims(k, -1), dtype=f32)

    # Uniform should stay uniform
    xn = np.ones_like(i)
    ni, nj, nk, nv = i.shape
    shape_cell = (ni - 1, nj - 1, nk - 1, nv)
    xc = np.zeros(shape_cell, order="F", dtype=f32)
    ember.fortran.node_to_cell(xn, xc)
    assert np.allclose(xn[:-1, :-1, :-1, :], xc)

    # Discrepancy should be exactly half for linear variation in each direction
    ic = np.zeros(shape_cell, order="F", dtype=f32)
    ember.fortran.node_to_cell(i, ic)
    assert np.allclose(ic - i[:-1, :-1, :-1, :], 0.5)

    jc = np.zeros(shape_cell, order="F", dtype=f32)
    ember.fortran.node_to_cell(j, jc)
    assert np.allclose(jc - j[:-1, :-1, :-1, :], 0.5)

    kc = np.zeros(shape_cell, order="F", dtype=f32)
    ember.fortran.node_to_cell(k, kc)
    assert np.allclose(kc - k[:-1, :-1, :-1, :], 0.5)


def test_cell_to_node():
    """Distribute a linear ramp from cell centers to nodes."""

    # Make an ijk grid
    ni = 97
    nj = 65
    nk = 73

    # Generate a grid of indices
    iv = np.linspace(0.0, ni - 1.0, ni)
    jv = np.linspace(0.0, nj - 1.0, nj)
    kv = np.linspace(0.0, nk - 1.0, nk)
    i, j, k = np.meshgrid(iv, jv, kv, indexing="ij")

    i = np.asfortranarray(np.expand_dims(i, -1), dtype=f32)
    j = np.asfortranarray(np.expand_dims(j, -1), dtype=f32)
    k = np.asfortranarray(np.expand_dims(k, -1), dtype=f32)

    # Uniform should stay uniform
    xc = np.ones_like(i)
    ni, nj, nk, nv = xc.shape
    shape_node = (ni + 1, nj + 1, nk + 1, nv)
    xn = np.zeros(shape_node, order="F", dtype=f32)
    ember.fortran.cell_to_node(xc, xn)
    assert np.allclose(xc, 1.0)

    # Check linear variation in each direction
    # Should have no change at boundaries
    # Offset of 1/2 along the ramping direction

    inode = np.zeros(shape_node, order="F", dtype=f32)
    ember.fortran.cell_to_node(i, inode)
    assert np.allclose(inode[0, :-1, :-1], i[0, :, :])
    assert np.allclose(inode[-1, :-1, :-1], i[-1, :, :])
    assert np.allclose(inode[1:-1, :-1, :-1] - i[:-1, :, :], 0.5)

    jnode = np.zeros(shape_node, order="F", dtype=f32)
    ember.fortran.cell_to_node(j, jnode)
    assert np.allclose(jnode[:-1, 0, :-1], j[:, 0, :])
    assert np.allclose(jnode[:-1, -1, :-1], j[:, -1, :])
    assert np.allclose(jnode[:-1, 1:-1, :-1] - j[:, :-1, :], 0.5)

    knode = np.zeros(shape_node, order="F", dtype=f32)
    ember.fortran.cell_to_node(k, knode)
    assert np.allclose(knode[:-1, :-1, 0], k[:, :, 0])
    assert np.allclose(knode[:-1, :-1, -1], k[:, :, -1])
    assert np.allclose(knode[:-1, :-1, 1:-1] - k[:, :, :-1], 0.5)


if __name__ == "__main__":
    test_node_to_cell()
    test_cell_to_node()
