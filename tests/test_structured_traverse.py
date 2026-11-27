"""Tests for structured traverse cuts (ember.cut.structured_traverse).

Module tested: ember.cut, ember._traverse, and the kernels of
_fortran/traverse.f90

Test cases:
- test_crossing_counts_match_numpy: kernel counts of sign changes per line
  along each axis, and the positive nodes, agree with numpy
- test_crossing_patch_matches_numpy: kernel interpolation to the nth crossing
  along each axis agrees with a numpy walk
- test_polyline_arc_matches_numpy: kernel arc lengths agree with a direct
  projection, beyond both ends included
- test_a_ring_comes_back_on_the_blocks_own_lines: one block a pitch wide,
  shape (nj, nk), each node interpolated along an i-edge
- test_linear_fields_are_exact: an inclined cut reproduces fields linear in
  x, r and theta
- test_two_blocks_across_an_interface_make_one_sheet: matches the same
  annulus meshed as one block
- test_theta_leaning_more_than_a_pitch: closes round one pitch, theta rising
  along every row
- test_a_blade_is_refused: a passage less than a pitch wide does not close
- test_a_cut_that_misses_is_refused: no block crossed
- test_union_labels_match_connected_components: kernel groups agree with a
  breadth-first search, labelled by least member
- test_edge_topology_of_a_disc_and_an_annulus: Euler characteristic and
  boundary loops of two small meshes
- test_fill_hole_reproduces_a_linear_field: kernel slicing of sloped rows
  through triangles, every interior node hit
- test_a_hole_between_chains_is_filled: three blocks round the pitch, the
  middle one split streamwise and crossed part-way; its sides exact, fields
  linear in x and r reproduced
- test_the_seam_bounds_a_cut_with_no_clean_block: two blocks one behind the
  other, crossed part-way by an inclined cut; closes round one pitch
- test_an_unsnapped_ragged_grid_is_refused: an interface out by much more
  than round-off leaves a gap, and the error points to Grid.snap_periodic
- test_untwist_slides_rows_into_straight_columns: grid lines twisted round
  the annulus, every column at one theta, at twice the density, the hub
  row's own nodes kept exactly
- test_untwist_keeps_periodic_linear_fields_exact: fields linear in x and r
  come through the interpolation along the rows exactly
- test_slide_rows_does_not_overshoot_a_step: the monotone cubic stays within
  the data either side of a jump
"""

import numpy as np
import pytest

import ember.cut
import ember.fluid
import ember.fortran
from ember import util
from ember.block import Block
from ember.periodic import PeriodicPatch

NB = 12
PITCH = 2 * np.pi / NB


def _linear(x, r, t, ct=1.0):
    """Conserved fields linear in the coordinates, density positive; with
    `ct` zero they are the same a pitch on, as a periodic flow is."""
    return np.stack(
        [
            1.0 + 0.1 * x + 0.2 * r + 0.3 * ct * t,
            100.0 + 5.0 * x - 3.0 * r + 2.0 * ct * t,
            20.0 - x + 4.0 * r + ct * t,
            10.0 + 2.0 * x + r - 5.0 * ct * t,
            2.5e5 + 1e3 * x + 2e3 * r + 5e2 * ct * t,
        ],
        axis=-1,
    )


def _block(x_lim, t_lim, shape=(9, 6, 11), lean=0.0, ct=1.0):
    """An annular block, theta turning by `lean` pitches from hub to casing."""
    block = Block(shape=shape)
    block.set_Nb(NB)
    block.set_fluid(ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72))
    xrt = util.linmesh3(x_lim, [0.5, 1.0], t_lim, shape)
    span = (xrt[..., 1] - 0.5) / 0.5
    xrt[..., 2] += lean * PITCH * span
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_conserved(_linear(*np.moveaxis(xrt, -1, 0), ct=ct))
    return block


def _faces(block, i=(), k=()):
    """Periodic patches on the given i and k faces: interfaces and the
    passage's periodic faces alike."""
    for f in i:
        block.patches.append(PeriodicPatch(i=f, j=(0, -1), k=(0, -1)))
    for f in k:
        block.patches.append(PeriodicPatch(i=(0, -1), j=(0, -1), k=f))
    return block


def _numpy_counts(pos):
    return [(np.diff(pos.astype(int), axis=ax) != 0).sum(axis=ax) for ax in range(3)]


def test_crossing_counts_match_numpy():
    rng = np.random.default_rng(1)
    a = np.asfortranarray(rng.standard_normal((7, 5, 6)), dtype=np.float32)
    shape = a.shape
    sgn = np.empty(shape, dtype=np.int8, order="F")
    cnt = [
        np.empty(s, dtype=np.int32, order="F")
        for s in ((shape[1], shape[2]), (shape[0], shape[2]), (shape[0], shape[1]))
    ]
    prev_j = np.empty(shape[0], dtype=np.int8)
    prev_k = np.empty(shape[:2], dtype=np.int8, order="F")
    n_pos = ember.fortran.crossing_counts(
        a, a, 1.0, 0.0, 0.0, sgn, *cnt, prev_j, prev_k
    )
    pos = a >= 0
    assert n_pos == pos.sum()
    np.testing.assert_array_equal(sgn, pos.astype(np.int8))
    for got, want in zip(cnt, _numpy_counts(pos)):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("ax", [0, 1, 2])
def test_crossing_patch_matches_numpy(ax):
    rng = np.random.default_rng(2)
    shape = (6, 7, 5)
    # Two crossings on every line along ax: a bump in that coordinate
    s = np.linspace(-1.0, 1.0, shape[ax])
    s = s.reshape([-1 if n == ax else 1 for n in range(3)])
    a = np.asfortranarray(0.5 - s**2 + 0.05 * rng.random(shape), dtype=np.float32)
    data = np.asfortranarray(rng.random((*shape, 3)), dtype=np.float32)
    sgn = np.asfortranarray(a >= 0, dtype=np.int8)
    others = [n for n in range(3) if n != ax]
    n1, n2 = shape[others[0]], shape[others[1]]
    for nth in (1, 2):
        out = np.empty((n1, n2, 3), dtype=np.float32, order="F")
        edge = np.empty((n1, n2), dtype=np.int32, order="F")
        cnt = np.empty((n1, n2), dtype=np.int32, order="F")
        ember.fortran.crossing_patch(
            a, a, 1.0, 0.0, 0.0, sgn, data, ax + 1, nth, 0, 0, out, edge, cnt
        )
        A = np.moveaxis(a, ax, -1).astype(float)
        D = np.moveaxis(data, ax, -2).astype(float)
        want = np.empty((n1, n2, 3))
        for p in range(n1):
            for q in range(n2):
                e = np.nonzero(np.diff((A[p, q] >= 0).astype(int)))[0][nth - 1]
                lo, hi = (e, e + 1) if A[p, q, e] < 0 else (e + 1, e)
                f = A[p, q, lo] / (A[p, q, lo] - A[p, q, hi])
                want[p, q] = D[p, q, lo] + (D[p, q, hi] - D[p, q, lo]) * f
        np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-6)


def test_polyline_arc_matches_numpy():
    xr = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    xq = np.array(
        [[-1.0, 0.5], [0.5, -0.2], [1.3, 1.0], [0.8, 3.0], [2.0, 0.2]],
        dtype=np.float32,
    )
    arc = np.empty(len(xq), dtype=np.float32)
    ember.fortran.polyline_arc(np.asfortranarray(xr), np.asfortranarray(xq), arc)
    np.testing.assert_allclose(arc, [-1.0, 0.5, 2.0, 4.0, 1.2], atol=1e-6)


def test_a_ring_comes_back_on_the_blocks_own_lines():
    block = _block([0.0, 1.0], [0.0, PITCH])
    ni, nj, nk = block.shape
    x_cut = 0.37
    sheet = ember.cut.structured_traverse(
        block, [[x_cut, 0.4], [x_cut, 1.1]], untwist=0
    )

    assert sheet.shape == (nj, nk)
    x = np.asarray(block.x)[:, 0, 0]
    i0 = np.searchsorted(x, x_cut) - 1
    f = (x_cut - x[i0]) / (x[i0 + 1] - x[i0])
    data = np.asarray(block._data, dtype=float)
    want = data[i0] + (data[i0 + 1] - data[i0]) * f
    np.testing.assert_allclose(sheet._data, want, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sheet.t[:, -1] - sheet.t[:, 0], PITCH, rtol=1e-5)


def test_linear_fields_are_exact():
    block = _block([0.0, 1.0], [0.0, PITCH], lean=0.3)
    sheet = ember.cut.structured_traverse(block, [[0.3, 0.4], [0.6, 1.1]], untwist=0)
    want = _linear(
        np.asarray(sheet.x, float),
        np.asarray(sheet.r, float),
        np.asarray(sheet.t, float),
    )
    np.testing.assert_allclose(sheet.conserved, want, rtol=2e-5)


def test_two_blocks_across_an_interface_make_one_sheet():
    shape = (9, 6, 13)
    whole = _block([0.0, 1.0], [0.0, PITCH], shape=shape, ct=0.0)
    split = 5 * PITCH / 12  # the node at k = 5
    lower = _block([0.0, 1.0], [0.0, split], shape=(9, 6, 6), ct=0.0)
    upper = _block([0.0, 1.0], [split, PITCH], shape=(9, 6, 8), ct=0.0)
    xr_cut = [[0.42, 0.4], [0.42, 1.1]]

    one = ember.cut.structured_traverse(whole, xr_cut, untwist=0)
    two = ember.cut.structured_traverse([upper, lower], xr_cut, untwist=0)

    assert two.shape == (6, 13)
    np.testing.assert_allclose(two.xrt, one.xrt, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(two.conserved, one.conserved, rtol=1e-5)


def test_theta_leaning_more_than_a_pitch():
    block = _block([0.0, 1.0], [0.0, PITCH], lean=1.5)
    sheet = ember.cut.structured_traverse(block, [[0.55, 0.4], [0.55, 1.1]])
    t = np.asarray(sheet.t, float)
    assert (np.diff(t, axis=1) > 0).all()
    np.testing.assert_allclose(t[:, -1] - t[:, 0], PITCH, rtol=1e-5)
    r = np.asarray(sheet.r, float)
    assert (np.diff(r[:, 0]) > 0).all()


def test_a_blade_is_refused():
    block = _block([0.0, 1.0], [0.0, 0.8 * PITCH])
    with pytest.raises(ValueError, match="close round the pitch"):
        ember.cut.structured_traverse(block, [[0.5, 0.4], [0.5, 1.1]])


def test_a_cut_that_misses_is_refused():
    block = _block([0.0, 1.0], [0.0, PITCH])
    with pytest.raises(ValueError, match="does not cross"):
        ember.cut.structured_traverse(block, [[2.0, 0.4], [2.0, 1.1]])


def test_union_labels_match_connected_components():
    a = np.array([0, 3, 5, 6, 2], dtype=np.int32)
    b = np.array([3, 1, 6, 4, 2], dtype=np.int32)
    n = 8
    label = np.empty(n, dtype=np.int32)
    parent = np.empty(n, dtype=np.int32)
    n_label = ember.fortran.union_labels(a, b, label, parent)
    # Groups {0, 1, 3}, {2}, {4, 5, 6}, {7}, numbered by least member
    np.testing.assert_array_equal(label, [0, 0, 1, 0, 2, 2, 2, 3])
    assert n_label == 4


def _topology(faces, nv):
    fm = np.asfortranarray(np.asarray(faces, dtype=np.int32).T)
    nf = fm.shape[1]
    out = [
        np.empty(nv + 1, np.int32),
        np.empty(3 * nf, np.int32),
        np.empty(3 * nf, np.int32),
        np.empty((2, nv), np.int32, order="F"),
        np.empty(nv, np.int32),
        np.empty(3 * nf, np.int32),
        np.empty(3 * nf + 1, np.int32),
    ]
    n_edge, _, n_loop, n_used, status = ember.fortran.edge_topology(
        fm, *out, nv, nf, 3 * nf
    )
    loops, start = out[5], out[6]
    return (
        n_used - n_edge + nf,
        [sorted(loops[start[q] - 1 : start[q + 1] - 1]) for q in range(n_loop)],
        status,
    )


def test_edge_topology_of_a_disc_and_an_annulus():
    # A square of two triangles: one loop round its four corners
    chi, loops, status = _topology([[0, 1, 2], [0, 2, 3]], 4)
    assert (chi, status) == (1, 0)
    assert loops == [[0, 1, 2, 3]]

    # A ring of four quads between an inner and an outer square
    quads = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    chi, loops, status = _topology(tris, 8)
    assert (chi, status) == (0, 0)
    assert sorted(loops) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_fill_hole_reproduces_a_linear_field():
    # Triangles over the unit square in (zeta, theta), data linear in both
    nz, nt = 7, 9
    z, t = np.meshgrid(np.linspace(0, 1, nz), np.linspace(0, 1, nt), indexing="ij")
    idx = np.arange(nz * nt).reshape(nz, nt)
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1)
    q = q.reshape(-1, 4)
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    var = np.stack([1 + 2 * z.ravel() - t.ravel(), 3 * t.ravel()], axis=-1)

    # Sloped rows from the left side to the right, nodes clustered
    n, m = 6, 11
    zl = np.linspace(0.0, 1.0, n)
    zr = np.clip(zl + 0.05 * np.sin(np.pi * zl), 0, 1)
    tl, tr = np.full(n, 0.1), np.full(n, 0.9)
    u = np.linspace(0, 1, m) ** 1.5
    T = tl[:, None] + u[None] * (tr - tl)[:, None]
    out = np.zeros((n, m, 2), dtype=np.float32, order="F")
    hits = np.zeros((n, m), dtype=np.int32, order="F")
    f32 = np.float32
    ember.fortran.fill_hole(
        z.ravel().astype(f32),
        t.ravel().astype(f32),
        np.asfortranarray(var, dtype=f32),
        np.asfortranarray(faces.T, dtype=np.int32),
        *(v.astype(f32) for v in (zl, tl, zr, tr)),
        np.asfortranarray(T.T, dtype=f32),
        f32(1e-5),
        out,
        hits,
    )
    assert (hits[1:-1] > 0).all()
    Z = zl[:, None] + u[None] * (zr - zl)[:, None]
    want = np.stack([1 + 2 * Z - T, 3 * T], axis=-1)
    np.testing.assert_allclose(out[1:-1], want[1:-1], atol=1e-5)


def _three_round_the_pitch():
    """Blocks A, B and C round the pitch, B split streamwise into two."""
    shape_ac, shape_b = (9, 6, 5), (5, 6, 4)
    third = PITCH / 3
    A = _faces(_block([0.0, 1.0], [0.0, third], shape_ac, ct=0.0), k=(0, -1))
    C = _faces(_block([0.0, 1.0], [2 * third, PITCH], shape_ac, ct=0.0), k=(0, -1))
    B1 = _faces(
        _block([0.0, 0.5], [third, 2 * third], shape_b, ct=0.0), i=(-1,), k=(0, -1)
    )
    B2 = _faces(
        _block([0.5, 1.0], [third, 2 * third], shape_b, ct=0.0), i=(0,), k=(0, -1)
    )
    return [A, B1, B2, C]


def test_a_hole_between_chains_is_filled():
    blocks = _three_round_the_pitch()
    sheet = ember.cut.structured_traverse(blocks, [[0.4, 0.4], [0.6, 1.1]], untwist=0)

    t = np.asarray(sheet.t, float)
    assert (np.diff(t, axis=1) > 0).all()
    np.testing.assert_allclose(t[:, -1] - t[:, 0], PITCH, rtol=1e-5)
    want = _linear(np.asarray(sheet.x, float), np.asarray(sheet.r, float), t, ct=0.0)
    np.testing.assert_allclose(sheet.conserved, want, rtol=1e-4)

    # Columns from the clean blocks A and C are those of the same annulus
    # meshed as one block, whose nodes in theta coincide with theirs
    alone = ember.cut.structured_traverse(
        _block([0.0, 1.0], [0.0, PITCH], (9, 6, 13), ct=0.0),
        [[0.4, 0.4], [0.6, 1.1]],
        untwist=0,
    )
    k = t[0] / (PITCH / 12)
    exact = (np.abs(k - np.round(k)) < 1e-4) & ((np.round(k) <= 4) | (np.round(k) >= 8))
    assert exact.sum() == 10
    np.testing.assert_allclose(
        sheet._data[:, exact], alone._data[:, np.round(k[exact]).astype(int)], rtol=1e-5
    )


def test_the_seam_bounds_a_cut_with_no_clean_block():
    front = _faces(
        _block([0.0, 0.5], [0.0, PITCH], (5, 6, 9), ct=0.0), i=(-1,), k=(0, -1)
    )
    back = _faces(
        _block([0.5, 1.0], [0.0, PITCH], (5, 6, 9), ct=0.0), i=(0,), k=(0, -1)
    )
    sheet = ember.cut.structured_traverse([front, back], [[0.35, 0.4], [0.65, 1.1]])

    t = np.asarray(sheet.t, float)
    assert (np.diff(t, axis=1) > 0).all()
    np.testing.assert_allclose(t[:, -1] - t[:, 0], PITCH, rtol=1e-5)
    want = _linear(np.asarray(sheet.x, float), np.asarray(sheet.r, float), t, ct=0.0)
    np.testing.assert_allclose(sheet.conserved, want, rtol=1e-4)


def test_an_unsnapped_ragged_grid_is_refused():
    blocks = _three_round_the_pitch()
    # The interface between the two halves of B out by a tenth of a cell
    x = np.array(blocks[2].x, dtype=np.float64)
    x[0] += 0.05
    blocks[2].set_x(x)
    with pytest.raises(ValueError, match="snap_periodic"):
        ember.cut.structured_traverse(blocks, [[0.4, 0.4], [0.6, 1.1]])


def test_untwist_slides_rows_into_straight_columns():
    block = _block([0.0, 1.0], [0.0, PITCH], lean=1.5, ct=0.0)
    xr_cut = [[0.55, 0.4], [0.55, 1.1]]
    twisted = ember.cut.structured_traverse(block, xr_cut, untwist=0)
    flat = ember.cut.structured_traverse(block, xr_cut, untwist=2)

    nj, nk = twisted.shape
    assert flat.shape == (nj, 2 * (nk - 1) + 1)
    t = np.asarray(flat.t, float)
    np.testing.assert_allclose(t, np.broadcast_to(t[:1], t.shape), atol=1e-6)
    np.testing.assert_allclose(t[0, -1] - t[0, 0], PITCH, rtol=1e-6)
    assert (np.diff(t, axis=1) > 0).all()
    # The hub row does not move: its own nodes are every other column
    np.testing.assert_allclose(flat._data[0, ::2], twisted._data[0], rtol=1e-6)


def test_untwist_keeps_periodic_linear_fields_exact():
    block = _block([0.0, 1.0], [0.0, PITCH], lean=0.7, ct=0.0)
    sheet = ember.cut.structured_traverse(block, [[0.3, 0.4], [0.6, 1.1]], untwist=2)
    want = _linear(np.asarray(sheet.x, float), np.asarray(sheet.r, float), 0.0, ct=0.0)
    np.testing.assert_allclose(sheet.conserved, want, rtol=2e-5)


def test_slide_rows_does_not_overshoot_a_step():
    m = 9
    tn = np.linspace(0.0, 1.0, m, dtype=np.float32)[None]
    tn = np.asfortranarray(
        np.repeat(tn, 2, axis=0) + np.array([[0.0], [0.03]], np.float32)
    )
    step = np.where(np.arange(m) < 4, 1.0, 2.0).astype(np.float32)
    step[-1] = step[0]
    data = np.zeros((2, m, 2), dtype=np.float32, order="F")
    data[..., 0] = tn * 2 * np.pi
    data[..., 1] = step
    out = np.empty((2, 4 * (m - 1) + 1, 2), dtype=np.float32, order="F")
    ember.fortran.slide_rows(tn, data, 0.0, 2 * np.pi, 1, 4, out)
    assert out[..., 1].min() >= 1.0 - 1e-6
    assert out[..., 1].max() <= 2.0 + 1e-6
