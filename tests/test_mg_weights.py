"""Tests for the geometry-aware multigrid prolongation weights.

The multigrid coarse correction is carried back to the fine grid by a separable
factor-2 linear interpolation. Which pair of coarse cells a target falls between
is index arithmetic (``mg_bracket2x``, or ``mg_bracket2x_node`` for the final
hop); where it sits BETWEEN them is geometry, and that is what
:attr:`ember.block.Block.weight_mgrid` supplies.

The final hop targets the fine NODES, so hop 1 is node-shaped while the
coarse->coarse hops stay cell-shaped. Its bracket is the one that does not
always contain its target, which is why its weights may leave [0, 1].

Covered here:

- the packed layout Python builds is byte-for-byte the one ``mg_weight_offsets``
  in ``scree.f90`` expects, which is the only thing keeping the two in step
- on a mesh uniform in physical space the weights collapse onto the index
  weights the kernel used to compute for itself
- on a stretched mesh they do not, and they reproduce the fine node's true
  position between the coarse centroids -- the property the whole change buys
- the interior excursion outside [0, 1] is real but small, and the ends stay
  clamped so the flat extrapolation survives
- ``mg_interp_i2x``/``mg_interp_i2x_node`` fed the index weights reproduce their
  hardcoded blends
"""

import numpy as np
import pytest

import ember
from ember import fortran, util
from ember.block import _mg_index_bracket_node, _mg_n_hops, _mg_weight_lengths

SHAPE = (17, 17, 17)  # 16 cells a side, so three factor-2 hops exist
LENGTH = 0.05
NB = 157


def _block(xv=None):
    """Single block on a duct-like annular mesh.

    ``xv`` overrides the axial node distribution, which is the one direction
    whose cell centroids are exactly the mean of their nodes however the
    annulus curves -- so a stretch there is a clean test of the projection.
    """
    pitch = 2.0 * np.pi / NB
    xrt = util.linmesh3([0.0, LENGTH], [1.0, 1.0 + LENGTH], [0.0, pitch], SHAPE)
    block = ember.block.Block(shape=SHAPE)
    x = xrt[..., 0] if xv is None else np.broadcast_to(xv[:, None, None], SHAPE)
    block.set_x(np.ascontiguousarray(x, dtype=np.float32))
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_Nb(NB)
    return block


def _stretched_x(n, ER=1.2):
    """Geometrically stretched node positions on [0, LENGTH]."""
    d = ER ** np.arange(n - 1, dtype=np.float64)
    x = np.concatenate([[0.0], np.cumsum(d)])
    return (LENGTH * x / x[-1]).astype(np.float32)


def _index_w1(nf):
    """The weight ``mg_bracket2x`` used to return, for one direction."""
    i = np.arange(1, nf + 1)
    t = (i - 0.5) / 2.0 + 0.5
    icl = np.floor(t).astype(int)
    nc = nf // 2
    return np.where((icl < 1) | (icl >= nc), 0.0, t - icl)


def _index_w1_node(nn, nc):
    """The uniform-mesh weight for a node direction: injection, then midpoint.

    Node 2m sits at coarse cell m's centre and node 2m+1 at the interface, so on
    a mesh uniform in physical space the weights are 0 and 0.5 alternating, with
    the two ends clamped by the flat extrapolation.
    """
    i = np.arange(1, nn + 1)
    lo = np.clip(i // 2, 1, max(nc - 1, 1))
    return np.clip((i - 2 * lo) / 2.0, 0.0, 1.0)


def _hop1_wi(block):
    """The i-direction weights of hop 1 (the final hop onto the fine nodes)."""
    ni, nj, nk = block.shape
    pwi = block.weight_mgrid[0]
    shape = (ni, (nj - 1) // 2, (nk - 1) // 2)
    return pwi[: np.prod(shape)].reshape(shape, order="F")


def test_packed_layout_matches_the_kernel():
    """Python's packing must be the layout mg_weight_offsets reads.

    The two are independent transcriptions of one layout, and nothing else
    checks them against each other: a drift here would silently feed each hop
    another hop's weights.
    """
    ni, nj, nk = SHAPE
    n_hops = _mg_n_hops(SHAPE)
    offwi, offwj, offwk = fortran.mg_weight_offsets(ni, nj, nk, n_hops)

    for m in range(n_hops):
        # Offsets are the running totals of the hops before this one.
        expected = _mg_weight_lengths(SHAPE, m)
        assert (offwi[m], offwj[m], offwk[m]) == expected

    # And each direction's array is exactly its hops, nothing over or under.
    sizes = tuple(w.size for w in _block().weight_mgrid)
    assert sizes == _mg_weight_lengths(SHAPE, n_hops)


def test_hops_stop_where_the_cells_stop_dividing():
    """Only the hops the cell counts admit are built.

    _validate_mg guarantees divisibility down to n_levels and no further, so a
    grid that halves twice must never be asked for a third hop.
    """
    assert _mg_n_hops((17, 17, 17)) == 3  # 16 = 2**4, capped by MAX_MG_LEVELS
    assert _mg_n_hops((13, 17, 17)) == 2  # 12 -> 6 -> 3, odd after two
    assert _mg_n_hops((10, 17, 17)) == 0  # 9 is odd immediately
    assert _mg_n_hops((17, 17, 9)) == 3  # 8 = 2**3, so three hops again
    assert _mg_n_hops((17, 17, 5)) == 2  # 4 -> 2 -> 1, odd after two


def test_uniform_mesh_gives_back_the_index_weights():
    """Uniform in physical space is the one case the old index weights got right.

    For the node-targeted final hop those weights alternate 0 (the node at a
    coarse cell centre, so injection) and 1/2 (the node on a coarse interface).

    Not exactly zero difference: cell volume in an annulus goes as r, so a
    block's volume-weighted centroid sits slightly outboard of its arithmetic
    midpoint. That is the residual seen here, and it is three orders below the
    error clustering introduces (see the test below).
    """
    wi = _hop1_wi(_block())
    expected = _index_w1_node(SHAPE[0], (SHAPE[0] - 1) // 2)[:, None, None]

    np.testing.assert_allclose(wi, np.broadcast_to(expected, wi.shape), atol=1e-6)


def test_stretched_mesh_departs_from_the_index_weights():
    """On a clustered mesh the index weight is simply the wrong place."""
    wi = _hop1_wi(_block(_stretched_x(SHAPE[0])))
    index = np.broadcast_to(
        _index_w1_node(SHAPE[0], (SHAPE[0] - 1) // 2)[:, None, None], wi.shape
    )

    assert np.abs(wi - index).max() > 0.02


def test_interior_weights_leave_the_unit_interval_but_barely():
    """The node bracket does not always contain its node, and must not pretend to.

    A coarse centroid is the volume-weighted mean of its two children, so on a
    stretched mesh it sits off the node between them and the node falls just
    outside the pair its index picks. Clamping there -- which is what the
    cell-targeted hops do -- would flatten the correction at every second node,
    so the interior weight is left to extrapolate. The excursion is bounded by
    the clustering and stays far inside a well-conditioned blend.
    """
    wi = _hop1_wi(_block(_stretched_x(SHAPE[0])))[:, 0, 0]

    assert wi.min() < 0.0  # the excursion is real
    assert wi.min() > -0.1  # and small: coefficients stay inside about [0, 1.1]
    assert wi.max() <= 1.0
    # The ends stay clamped, which is what keeps the flat extrapolation past the
    # outer coarse centroids.
    assert wi[0] == 0.0 and wi[-1] == 1.0


def test_weights_place_the_fine_node_where_it_really_is():
    """The weight must reproduce the fine NODE position from the coarse pair.

    This is the property the change buys, stated directly: interpolating the
    coarse cell POSITION with (lo, hi, w) has to give back the node position.
    With the index weights it does not, on any mesh that is not uniform -- which
    is what the assertion at the end pins.
    """
    ni = SHAPE[0]
    block = _block(_stretched_x(ni))
    wi = _hop1_wi(block)[:, 0, 0]

    # Fine cell centroids in x, and their volume-weighted coarse blocks, built
    # the way the ladder does. x is separable here, so one column says it all.
    x = np.asarray(block.x, dtype=np.float64)
    xn = x[:, 0, 0]
    xc = 0.5 * (xn[:-1] + xn[1:])
    vol = np.asarray(block.vol, dtype=np.float64).sum(axis=(1, 2))
    pair = slice(None, None, 2), slice(1, None, 2)
    wsum = vol[pair[0]] + vol[pair[1]]
    xcoarse = (xc[pair[0]] * vol[pair[0]] + xc[pair[1]] * vol[pair[1]]) / wsum

    lo, hi = _mg_index_bracket_node(ni, (ni - 1) // 2)
    got = xcoarse[lo] * (1.0 - wi) + xcoarse[hi] * wi

    # Only the nodes outside the outer coarse centroids are clamped, and on this
    # mesh that is one node at each end; everything else is the claim.
    inner = (xn > xcoarse[0]) & (xn < xcoarse[-1])
    np.testing.assert_allclose(got[inner], xn[inner], rtol=1e-5)

    index = _index_w1_node(ni, (ni - 1) // 2)
    with pytest.raises(AssertionError):
        got_index = xcoarse[lo] * (1.0 - index) + xcoarse[hi] * index
        np.testing.assert_allclose(got_index[inner], xn[inner], rtol=1e-5)


def test_interp_i2x_reproduces_its_old_blend():
    """Fed the index weights the kernel must emit what it hardcoded before.

    mg_interp_i2x used to carry the literals 0.75/0.25, so handing it those
    same values through the weight array must not move the answer. The
    comparison is to float32 rounding rather than exact: the reference below is
    evaluated by numpy, which does not fuse the multiply-add the way the
    compiled kernel does.
    """
    nci, nfi = 8, 16
    rng = np.random.default_rng(0)
    cin = np.asfortranarray(rng.standard_normal(nci), dtype=np.float32)
    w = np.asfortranarray(_index_w1(nfi), dtype=np.float32)

    got = np.zeros(nfi, dtype=np.float32, order="F")
    fortran.mg_interp_i2x(cin, got, w)

    want = np.empty(nfi, dtype=np.float32)
    want[0] = cin[0]
    for m in range(1, nci):
        want[2 * m - 1] = cin[m - 1] * np.float32(0.75) + cin[m] * np.float32(0.25)
        want[2 * m] = cin[m - 1] * np.float32(0.25) + cin[m] * np.float32(0.75)
    want[nfi - 1] = cin[nci - 1]

    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_interp_i2x_node_blends_its_pair():
    """The node interpolator must use the pair (m, m+1) for nodes 2m and 2m+1.

    Fed the uniform-mesh weights it is injection at the coarse centres and the
    midpoint on the interfaces, with both ends flat -- which is the vertex
    prolongation the fused final hop is supposed to be.
    """
    nci, nfi = 8, 17
    rng = np.random.default_rng(0)
    cin = np.asfortranarray(rng.standard_normal(nci), dtype=np.float32)
    w = np.asfortranarray(_index_w1_node(nfi, nci), dtype=np.float32)

    got = np.zeros(nfi, dtype=np.float32, order="F")
    fortran.mg_interp_i2x_node(cin, got, w)

    want = np.empty(nfi, dtype=np.float32)
    want[0] = cin[0]
    for m in range(1, nci):
        want[2 * m - 1] = cin[m - 1]  # node at coarse centre m: injection
        want[2 * m] = cin[m - 1] * np.float32(0.5) + cin[m] * np.float32(0.5)
    want[nfi - 2] = cin[nci - 1]  # last coarse centre, still injection
    want[nfi - 1] = cin[nci - 1]  # past it, flat

    np.testing.assert_allclose(got, want, rtol=1e-6)


def _prolong_only(block, coarse):
    """Run the fused final hop with the fine term switched off.

    ``scale = 0`` kills ``scale*dt_vol*q``, so what lands on ``cons`` is the
    prolonged coarse correction and nothing else.
    """
    ni, nj, nk = block.shape
    ncj, nck = (nj - 1) // 2, (nk - 1) // 2

    def Z(*shape):
        return np.asfortranarray(np.zeros(shape, dtype=np.float32))

    cons = Z(ni, nj, nk, 5)
    pwi, pwj, pwk = block.weight_mgrid
    shapes = ((ni, ncj, nck), (ni, nj, nck), (ni, nj, nk))
    wi, wj, wk = (
        np.asfortranarray(p[: np.prod(sh)].reshape(sh, order="F"))
        for p, sh in zip((pwi, pwj, pwk), shapes)
    )
    fortran.mg_prolong2x_fine_scatter(
        src=np.asfortranarray(
            np.repeat(coarse[..., None], 5, axis=-1), dtype=np.float32
        ),
        base=Z(ni, nj, nk, 5),
        cons=cons,
        scale=0.0,
        dt_vol=Z(ni - 1, nj - 1, nk - 1),
        q=Z(ni - 1, nj - 1, nk - 1, 5),
        aplane=Z(ni, ncj),
        bb=Z(ni, nj, nck, 5),
        cbuf=Z(ni, nj, 5),
        rbuf=Z(ni - 1, nj - 1, 5, 2),
        wi=wi,
        wj=wj,
        wk=wk,
    )
    return cons[..., 0]


@pytest.mark.parametrize("ER", [1.0, 1.2, 1.35])
def test_prolonged_linear_correction_lands_on_the_nodes(ER):
    """A correction linear in space must arrive at the nodes unchanged.

    The whole point of targeting the nodes: linear interpolation reproduces a
    linear field exactly, and it does so at the node only if that is where the
    interpolant is evaluated. Interpolating onto the fine cell centres and
    averaging back out -- what this used to do -- evaluates it at the mean of
    the eight surrounding centroids instead, which a stretched mesh moves off
    the node by ``(h_next - h_prev)/4``.

    The gradient is along x alone, the one direction this annular mesh makes
    exactly separable, so the residual here is float32 and nothing else; a
    gradient with radial and tangential parts also picks up the annulus
    curvature that :func:`test_uniform_mesh_gives_back_the_index_weights`
    already documents.
    """
    block = _block(_stretched_x(SHAPE[0], ER) if ER > 1.0 else None)
    nodes, levels = ember.block._mg_centroid_ladder(
        block._xrt_nd, block.vol_nd, _mg_n_hops(SHAPE)
    )
    origin = nodes.reshape(-1, 3).mean(axis=0)
    got = _prolong_only(block, (levels[1] - origin)[..., 0].astype(np.float32))
    want = (nodes - origin)[..., 0]

    # The outer half coarse cell is deliberately flat, so compare the interior.
    inner = (slice(2, -2),) * 3
    assert np.abs(got[inner] - want[inner]).max() / np.ptp(want) < 1e-6
