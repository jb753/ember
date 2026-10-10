"""Tests for the elliptic grid generation kernels.

Module tested: ember.elliptic

Test cases:
- The stencil coefficients and right-hand side match the equations written out
  directly in numpy, on random stencils with pitch shifts
- A uniform Cartesian grid gives the five-point Laplacian and no source
- A row of the system is satisfied by the grid it was frozen at, for a grid
  whose spacing is held by the control functions
- Node numbers out of range are refused
- Winslow sweeps match the same steps written out in numpy: on a ring of
  cells round a circle, its wall nodes sliding round it, with singular,
  sliding and corner-patch rows besides; and on random stencils with pitch
  shifts
"""

import numpy as np
import pytest

from ember.elliptic import ttm_stencil, winslow_sweeps


def reference(X, g, s, PQ, pitch):
    """The equations written out in numpy, independently of the kernel."""
    V = X[g].copy()
    V[..., 1] += s * pitch
    E, N, W, S = V[:, 1], V[:, 3], V[:, 5], V[:, 7]
    r_xi, r_eta = 0.5 * (E - W), 0.5 * (N - S)
    a = np.sum(r_eta**2, axis=-1)
    b = np.sum(r_xi * r_eta, axis=-1)
    c = np.sum(r_xi**2, axis=-1)
    raw = np.stack((-2 * (a + c), a, -b / 2, c, b / 2, a, -b / 2, c, b / 2), axis=1)
    diag = -raw[:, 0:1]
    w = -raw / diag
    src = (a[:, None] * PQ[:, 0:1] * r_xi + c[:, None] * PQ[:, 1:2] * r_eta) / diag
    src[:, 1] -= np.sum(w * s, axis=1) * pitch
    return w, src


def grid_stencils(ni, nk):
    """Interior stencils of one (ni, nk) block numbered i fastest, and its nodes."""
    num = np.arange(ni * nk).reshape(nk, ni).T
    offsets = [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]
    ii, kk = np.meshgrid(np.arange(1, ni - 1), np.arange(1, nk - 1), indexing="ij")
    g = np.stack([num[ii + di, kk + dk].ravel() for di, dk in offsets], axis=1)
    return g


def test_matches_reference_on_random_stencils():
    rng = np.random.default_rng(0)
    n_node, n_reg, pitch = 200, 150, 0.7
    X = rng.standard_normal((n_node, 2))
    g = rng.integers(0, n_node, (n_reg, 9))
    s = rng.integers(-1, 2, (n_reg, 9))
    s[:, 0] = 0
    PQ = rng.uniform(-2.0, 2.0, (n_reg, 2))

    w, src = ttm_stencil(X, g, s, PQ, pitch)
    w_ref, src_ref = reference(X, g, s, PQ, pitch)
    np.testing.assert_allclose(w, w_ref, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(src, src_ref, rtol=1e-12, atol=1e-14)


def test_uniform_grid_is_the_laplacian():
    ni, nk = 6, 5
    x, y = np.meshgrid(np.arange(ni), np.arange(nk), indexing="ij")
    X = np.stack((x.T.ravel(), y.T.ravel()), axis=1).astype(float)
    g = grid_stencils(ni, nk)
    PQ = np.zeros((len(g), 2))

    w, src = ttm_stencil(X, g, np.zeros_like(g), PQ)
    expect = np.array([1.0, -0.25, 0.0, -0.25, 0.0, -0.25, 0.0, -0.25, 0.0])
    np.testing.assert_allclose(w, np.broadcast_to(expect, w.shape), atol=1e-15)
    np.testing.assert_allclose(src, 0.0, atol=1e-15)


def test_grid_with_held_spacing_satisfies_its_rows():
    """A geometric expansion in eta, held by its own Q, is a fixed point.

    In one dimension `r_etaeta + Q r_eta = 0` holds the points where they are
    for `Q = -r_eta . r_etaeta / |r_eta|^2`, so with that Q every row is
    satisfied by the grid it was frozen at.
    """
    ni, nk = 5, 7
    yv = np.cumsum(np.concatenate(([0.0], 1.2 ** np.arange(nk - 1))))
    x, y = np.meshgrid(np.arange(ni, dtype=float), yv, indexing="ij")
    X = np.stack((x.T.ravel(), y.T.ravel()), axis=1)
    g = grid_stencils(ni, nk)
    V = X[g]
    r_eta = 0.5 * (V[:, 3] - V[:, 7])
    r_ee = V[:, 3] - 2.0 * V[:, 0] + V[:, 7]
    Q = -np.sum(r_eta * r_ee, axis=1) / np.sum(r_eta**2, axis=1)
    PQ = np.stack((np.zeros_like(Q), Q), axis=1)

    w, src = ttm_stencil(X, g, np.zeros_like(g), PQ)
    lhs = np.einsum("nk,nkd->nd", w, X[g])
    np.testing.assert_allclose(lhs, src, atol=1e-12)


def test_rejects_out_of_range_nodes():
    X = np.zeros((4, 2))
    g = np.array([[0, 1, 2, 3, 0, 1, 2, 3, 4]])
    with pytest.raises(ValueError, match="node numbers"):
        ttm_stencil(X, g, np.zeros_like(g), np.zeros((1, 2)))


#
# WINSLOW SWEEPS
#


def nearest_on_line(line, s_line, p):
    """Arc length of the point on a closed line (n, 2) nearest `p`, by brute
    force over its samples and then the segment either side."""
    m = len(line) - 1
    k = int(np.argmin(np.sum((line[:m] - p) ** 2, axis=1)))
    best, s = np.inf, s_line[k]
    for ka in (k - 1, k):
        kb = ka + 1
        if ka < 0:
            ka, kb = m - 1, m
        a, b = line[ka], line[kb]
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
        d = np.sum((a + t * (b - a) - p) ** 2)
        if d < best:
            best, s = d, s_line[ka] + t * (s_line[kb] - s_line[ka])
    return s


def reference_sweeps(X, g, s, pitch, irregular, slide, corner, wall, n_sweep):
    """The sweeps written out in numpy, independently of the kernel."""
    X = np.array(X, dtype=np.float64)
    nodes_i, ptr_i, col_i, w_i, c_i = irregular
    nodes_s, ptr_s, col_s, w_s, c_s = slide
    cp_node, cp_corner, cp_dy, cp_f = corner
    wn0, wn1, dy0, dy1, line, s_line, floor, _, _ = wall
    S = s_line[-1]
    n_w = len(wn0)
    for _ in range(n_sweep):
        V = X[g].copy()
        V[..., 1] += s * pitch
        E, NE, N, NW, W, SW, So, SE = (V[:, k] for k in range(1, 9))
        r_xi, r_eta = 0.5 * (E - W), 0.5 * (N - So)
        a = np.sum(r_eta**2, axis=-1)[:, None]
        b = np.sum(r_xi * r_eta, axis=-1)[:, None]
        c = np.sum(r_xi**2, axis=-1)[:, None]
        new = (a * (E + W) + c * (N + So) - 0.5 * b * (NE - SE + SW - NW)) / (
            2 * (a + c)
        )
        irr = np.array(
            [
                w_i[ptr_i[n] : ptr_i[n + 1]] @ X[col_i[ptr_i[n] : ptr_i[n + 1]]]
                for n in range(len(nodes_i))
            ]
        ).reshape(-1, 2)
        irr[:, 1] += c_i
        X[g[:, 0]] = new
        X[nodes_i] = irr
        ys = np.array(
            [
                w_s[ptr_s[n] : ptr_s[n + 1]] @ X[col_s[ptr_s[n] : ptr_s[n + 1]], 1]
                for n in range(len(nodes_s))
            ]
        )
        X[nodes_s, 1] = ys + c_s
        xc = X[cp_corner].copy()
        xc[:, 1] += cp_dy
        X[cp_node] = xc + (X[cp_node] - xc) * np.asarray(cp_f)[:, None]
        if n_w:
            p0 = X[wn0] + np.stack((np.zeros(n_w), dy0), axis=1)
            p1 = X[wn1] + np.stack((np.zeros(n_w), dy1), axis=1)
            s0 = np.array([nearest_on_line(line, s_line, p) for p in p0])
            s1 = np.array([nearest_on_line(line, s_line, p) for p in p1])
            half = n_w // 2
            for q in (s0, s1):
                q[:half] = np.where(q[:half] > 0.75 * S, q[:half] - S, q[:half])
                q[half:] = np.where(q[half:] < 0.25 * S, q[half:] + S, q[half:])
                q[-1] = q[0] + S
            before = np.concatenate(([s0[-2] - S], s0[:-1]))
            after = np.concatenate((s0[1:], [s0[1] + S]))
            s_new = np.clip(s1, 0.5 * (before + s0), 0.5 * (s0 + after))
            s_new = np.maximum.accumulate(s_new - floor) + floor
            s_new[-1] = s_new[0] + S
            s_new = np.minimum.accumulate((s_new - floor)[::-1])[::-1] + floor
            sm = np.mod(s_new, S)
            foot = np.stack([np.interp(sm, s_line, line[:, d]) for d in (0, 1)], axis=1)
            foot[:, 1] -= dy0
            X[wn0] = foot
    return X


def ring_case(ni=24, nk=5, n_extra=30, seed=1):
    """A ring of cells round the unit circle, its wall sliding round it, and
    extra nodes with singular, sliding and corner-patch rows."""
    rng = np.random.default_rng(seed)
    theta = 2 * np.pi * np.arange(ni) / ni
    radius = 1.0 + 0.15 * np.arange(nk) ** 1.2
    # Off the circle a little, and skewed, for the sweeps to have work to do.
    th = theta[:, None] + 0.04 * rng.standard_normal((ni, nk)) * (np.arange(nk) > 0)
    rr = radius[None, :] + 0.02 * rng.standard_normal((ni, nk)) * (np.arange(nk) > 0)
    ring = np.stack((rr * np.cos(th), rr * np.sin(th)), axis=-1).reshape(-1, 2)
    num = np.arange(ni * nk).reshape(ni, nk)
    extra = rng.uniform(-3, 3, (n_extra, 2))
    X = np.concatenate((ring, extra))
    x0 = ni * nk

    offsets = [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]
    g = np.array(
        [
            [num[(i + di) % ni, k + dk] for di, dk in offsets]
            for i in range(ni)
            for k in range(1, nk - 1)
        ]
    )
    s = np.zeros_like(g)

    def rows(nodes, width):
        ptr = np.arange(len(nodes) + 1) * width
        col = rng.integers(0, len(X), len(nodes) * width)
        w = rng.uniform(0.1, 0.4, len(nodes) * width)
        return nodes, ptr, col, w, rng.uniform(-0.1, 0.1, len(nodes))

    irregular = rows(x0 + np.arange(0, 5), 5)
    slide = rows(x0 + np.arange(5, 10), 3)
    corner = (
        x0 + np.arange(10, 20),
        x0 + np.arange(20, 30),
        rng.uniform(-0.2, 0.2, 10),
        rng.uniform(0.6, 1.0, 10),
    )

    # The wall: the circle, finely sampled, from angle zero round once.
    t = np.linspace(0.0, 2 * np.pi, 4001)
    line = np.stack((np.cos(t), np.sin(t)), axis=1)
    line[-1] = line[0]
    s_line = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
    )
    wn0 = np.append(num[:, 0], num[0, 0])
    wn1 = np.append(num[:, 1], num[0, 1])
    s_built = np.append(theta, 2 * np.pi)
    floor = np.concatenate(([0.0], np.cumsum(0.1 * np.diff(s_built))))
    seg = [np.argmin(np.sum((line[:-1] - X[n]) ** 2, axis=1)) for n in wn0]
    seg_out = [np.argmin(np.sum((line[:-1] - X[n]) ** 2, axis=1)) for n in wn1]
    zero = np.zeros(len(wn0))
    wall = (wn0, wn1, zero, zero, line, s_line, floor, seg, seg_out)
    return X, g, s, irregular, slide, corner, wall


@pytest.mark.parametrize("n_sweep", [1, 5])
def test_sweeps_match_reference_on_a_ring(n_sweep):
    X, g, s, irregular, slide, corner, wall = ring_case()
    out, _, _ = winslow_sweeps(X, g, s, 0.0, irregular, slide, corner, wall, n_sweep)
    ref = reference_sweeps(X, g, s, 0.0, irregular, slide, corner, wall, n_sweep)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_sweeps_match_reference_on_random_stencils_with_pitch():
    rng = np.random.default_rng(2)
    n_node, n_reg, pitch = 200, 150, 0.7
    X = rng.standard_normal((n_node, 2))
    g = rng.integers(0, n_node, (n_reg, 9))
    g[:, 0] = rng.permutation(n_node)[:n_reg]
    s = rng.integers(-1, 2, (n_reg, 9))
    s[:, 0] = 0
    none = (
        np.zeros(0, int),
        np.zeros(1, int),
        np.zeros(0, int),
        np.zeros(0),
        np.zeros(0),
    )
    no_corner = (np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0))
    no_wall = (
        np.zeros(0, int),
        np.zeros(0, int),
        np.zeros(0),
        np.zeros(0),
        np.zeros((2, 2)),
        np.zeros(2),
        np.zeros(0),
        np.zeros(0, int),
        np.zeros(0, int),
    )
    out, _, _ = winslow_sweeps(X, g, s, pitch, none, none, no_corner, no_wall, 1)
    ref = reference_sweeps(X, g, s, pitch, none, none, no_corner, no_wall, 1)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_sweeps_refuse_node_numbers_out_of_range():
    X, g, s, irregular, slide, corner, wall = ring_case()
    g = g.copy()
    g[0, 1] = len(X)
    with pytest.raises(ValueError):
        winslow_sweeps(X, g, s, 0.0, irregular, slide, corner, wall, 1)
