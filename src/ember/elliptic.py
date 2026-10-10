"""Kernels for elliptic grid generation.

ember does not generate grids, but a mesher that smooths them by solving the
Thompson-Thames-Mastin equations spends most of its time forming their stencils,
and this is where the compiled code lives. The kernels take node numbers and
coordinates only, so they serve any block structure a mesher assembles.
"""

import numpy as np

import ember.fortran


def ttm_stencil(X, stencil_g, stencil_s, PQ, pitch=0.0):
    r"""Nine-point stencils of the Thompson-Thames-Mastin equations.

    The equations in Thomas-Middlecoff form,

    .. math::
        \alpha (r_{\xi\xi} + P r_\xi) - 2 \beta r_{\xi\eta}
        + \gamma (r_{\eta\eta} + Q r_\eta) = 0,

    with :math:`\alpha = |r_\eta|^2`, :math:`\beta = r_\xi \cdot r_\eta` and
    :math:`\gamma = |r_\xi|^2`, are frozen at the coordinates `X` and
    discretised by central differences, with the control function term lagged
    onto the right-hand side.

    Parameters
    ----------
    X : array, shape (n_node, 2)
        Node coordinates :math:`(x, y)`.
    stencil_g : int array, shape (n_reg, 9)
        Node numbers of each stencil, counted from zero, in the order centre,
        E, NE, N, NW, W, SW, S, SE.
    stencil_s : int array, shape (n_reg, 9)
        Pitch shift of each stencil node in whole periods of `y`, for a stencil
        that straddles a periodic boundary.
    PQ : array, shape (n_reg, 2)
        Control functions :math:`P, Q` at each stencil centre.
    pitch : float
        Period in `y`, or zero for none.

    Returns
    -------
    w : array, shape (n_reg, 9)
        Stencil coefficients, normalised to one at the centre.
    src : array, shape (n_reg, 2)
        Right-hand side of each row, so that
        ``sum(w[n] * X[stencil_g[n]]) == src[n]`` is the discrete equation,
        with the neighbours' pitch shifts already moved into the `y` component.

    """
    stencil_g = np.asfortranarray(stencil_g, dtype=np.int32)
    stencil_s = np.asfortranarray(stencil_s, dtype=np.int32)
    n_node = len(X)
    if stencil_g.size and (stencil_g.min() < 0 or stencil_g.max() >= n_node):
        raise ValueError(f"Stencil node numbers must lie in [0, {n_node}).")
    return ember.fortran.ttm_stencil(
        np.asfortranarray(X, dtype=np.float64),
        stencil_g,
        stencil_s,
        np.asfortranarray(PQ, dtype=np.float64),
        float(pitch),
    )


def winslow_sweeps(
    X, stencil_g, stencil_s, pitch, irregular, slide, corner, wall, n_sweep
):
    r"""Point-Jacobi sweeps of Winslow's equations, in single precision.

    Each sweep moves every regular node to the solution of its nine-point
    stencil of

    .. math::
        \alpha r_{\xi\xi} - 2 \beta r_{\xi\eta} + \gamma r_{\eta\eta} = 0,

    frozen at the coordinates the sweep starts from, and every singular point
    to a weighted row of the same; then every sliding node in `y` to a
    weighted row of the coordinates after that; then draws every node near a
    corner towards it; then slides every wall node along a closed line to the
    foot of the perpendicular from the node outside it, no further than half
    way to either neighbour, its gaps to them kept above a floor.

    Parameters
    ----------
    X : array, shape (n_node, 2)
        Node coordinates :math:`(x, y)`.
    stencil_g, stencil_s : int arrays, shape (n_reg, 9)
        Node numbers of each regular node's stencil, counted from zero, centre
        first then E, NE, N, NW, W, SW, S, SE, and the pitch shift of each.
    pitch : float
        Period in `y`, or zero for none.
    irregular : tuple
        Singular points ``(nodes, indptr, indices, weights, const)``: each
        node to ``weights @ X[indices]`` over its row of the compressed sparse
        rows, plus `const` in `y`.
    slide : tuple
        Sliding nodes ``(nodes, indptr, indices, weights, const)``, the same
        in `y` only.
    corner : tuple
        ``(nodes, corners, dy, f)``: each node to ``c + (X[node] - c) * f``,
        `c` its corner seen shifted `dy` in `y`.
    wall : tuple
        ``(nodes, outside, dy, dy_outside, line, s_line, floor, seg, seg_out)``:
        the wall nodes in order round a closed ring, the last the first a turn
        on, and the node outside each, both seen shifted in `y`; the closed
        line (n_line, 2) they lie on, its last point its first, and arc
        lengths along it; the least cumulative arc length of each node past
        the first; and the samples of the line each node and the node outside
        it were nearest the sweep before.
    n_sweep : int
        Number of sweeps.

    Returns
    -------
    X : array, shape (n_node, 2)
        Swept coordinates, single precision.
    seg, seg_out : int arrays
        Samples nearest each wall node and the node outside it, to start the
        next call from.

    """
    f32 = np.float32
    i32 = np.int32

    # f2py takes an array whose dimension is zero to be one long, so an
    # empty group is passed as one dummy entry, its count given as zero.
    def ints(a):
        a = np.ascontiguousarray(a, dtype=i32)
        return a if a.size else np.zeros(1, dtype=i32)

    def reals(a):
        a = np.ascontiguousarray(a, dtype=f32)
        return a if a.size else np.zeros(1, dtype=f32)

    def doubles(a):
        a = np.ascontiguousarray(a, dtype=np.float64)
        return a if a.size else np.zeros(1)

    Xf = np.array(X, dtype=f32, order="F")
    nodes_i, ptr_i, col_i, w_i, c_i = irregular
    nodes_s, ptr_s, col_s, w_s, c_s = slide
    cp_node, cp_corner, cp_dy, cp_f = corner
    wn0, wn1, dy0, dy1, line, s_line, floor, seg0, seg1 = wall
    n_w = len(wn0)
    seg0 = ints(seg0).copy()
    seg1 = ints(seg1).copy()
    n_node = len(Xf)
    stencil_g = np.asfortranarray(stencil_g, dtype=i32)
    for g in (stencil_g, nodes_i, col_i, nodes_s, col_s, cp_node, cp_corner, wn0, wn1):
        g = np.asarray(g)
        if g.size and (g.min() < 0 or g.max() >= n_node):
            raise ValueError(f"Node numbers must lie in [0, {n_node}).")
    ember.fortran.winslow_sweeps(
        Xf,
        stencil_g,
        np.asfortranarray(stencil_s, dtype=i32),
        ints(nodes_i),
        ints(ptr_i) if len(nodes_i) else np.zeros(2, dtype=i32),
        ints(col_i),
        reals(w_i),
        reals(c_i),
        ints(nodes_s),
        ints(ptr_s) if len(nodes_s) else np.zeros(2, dtype=i32),
        ints(col_s),
        reals(w_s),
        reals(c_s),
        ints(cp_node),
        ints(cp_corner),
        reals(cp_dy),
        reals(cp_f),
        ints(wn0),
        ints(wn1),
        reals(dy0),
        reals(dy1),
        np.asfortranarray(line, dtype=np.float64),
        np.ascontiguousarray(s_line, dtype=np.float64),
        doubles(floor),
        seg0,
        seg1,
        float(pitch),
        int(n_sweep),
        len(nodes_i),
        len(nodes_s),
        len(cp_node),
        n_w,
    )
    return Xf, seg0[:n_w], seg1[:n_w]
