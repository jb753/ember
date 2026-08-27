"""Tests for the geometry-aware multigrid prolongation weights.

The multigrid coarse correction is carried back to the fine grid by a separable
factor-2 linear interpolation. Which pair of coarse cells a fine cell falls
between is index arithmetic (``mg_bracket2x``); where it sits BETWEEN them is
geometry, and that is what :attr:`ember.block.Block.weight_mgrid` supplies.

Covered here:

- the packed layout Python builds is byte-for-byte the one ``mg_weight_offsets``
  in ``scree.f90`` expects, which is the only thing keeping the two in step
- on a mesh uniform in physical space the weights collapse onto the index
  weights the kernel used to compute for itself
- on a stretched mesh they do not, and they reproduce the fine cell's true
  position between the coarse centroids -- the property the whole change buys
- ``mg_interp_i2x`` fed the index weights reproduces its old hardcoded blend
"""

import numpy as np
import pytest

import ember
from ember import fortran, util
from ember.block import _mg_index_bracket, _mg_n_hops, _mg_weight_lengths

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


def _hop1_wi(block):
    """The i-direction weights of hop 1 (the final hop onto the fine grid)."""
    ni, nj, nk = block.shape
    pwi = block.weight_mgrid[0]
    shape = (ni - 1, (nj - 1) // 2, (nk - 1) // 2)
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

    Not exactly zero difference: cell volume in an annulus goes as r, so a
    block's volume-weighted centroid sits slightly outboard of its arithmetic
    midpoint. That is the residual seen here, and it is three orders below the
    error clustering introduces (see the test below).
    """
    wi = _hop1_wi(_block())
    expected = _index_w1(SHAPE[0] - 1)[:, None, None]

    np.testing.assert_allclose(wi, np.broadcast_to(expected, wi.shape), atol=1e-6)


def test_stretched_mesh_departs_from_the_index_weights():
    """On a clustered mesh the index weight is simply the wrong place."""
    wi = _hop1_wi(_block(_stretched_x(SHAPE[0])))
    index = np.broadcast_to(_index_w1(SHAPE[0] - 1)[:, None, None], wi.shape)

    assert np.abs(wi - index).max() > 0.05
    assert wi.min() >= 0.0 and wi.max() <= 1.0


def test_weights_place_the_fine_cell_where_it_really_is():
    """The weight must reproduce the fine centroid from the coarse pair.

    This is the property the change buys, stated directly: interpolating the
    coarse cell POSITION with (lo, hi, w) has to give back the fine cell
    position. With the index weights it does not, on any mesh that is not
    uniform -- which is what the assertion at the end pins.
    """
    ni = SHAPE[0]
    block = _block(_stretched_x(ni))
    wi = _hop1_wi(block)[:, 0, 0]

    # Fine cell centroids in x, and their volume-weighted coarse blocks, built
    # the way the ladder does. x is separable here, so one column says it all.
    x = np.asarray(block.x, dtype=np.float64)
    xc = 0.5 * (x[:-1, 0, 0] + x[1:, 0, 0])
    vol = np.asarray(block.vol, dtype=np.float64).sum(axis=(1, 2))
    pair = slice(None, None, 2), slice(1, None, 2)
    wsum = vol[pair[0]] + vol[pair[1]]
    xcoarse = (xc[pair[0]] * vol[pair[0]] + xc[pair[1]] * vol[pair[1]]) / wsum

    lo, hi = _mg_index_bracket(ni - 1, (ni - 1) // 2)
    got = xcoarse[lo] * (1.0 - wi) + xcoarse[hi] * wi

    # The two end cells clamp onto a single coarse centroid (lo == hi), so they
    # cannot reproduce anything but that centroid; the interior is the claim.
    np.testing.assert_allclose(got[1:-1], xc[1:-1], rtol=1e-5)

    index = _index_w1(ni - 1)
    with pytest.raises(AssertionError):
        got_index = xcoarse[lo] * (1.0 - index) + xcoarse[hi] * index
        np.testing.assert_allclose(got_index[1:-1], xc[1:-1], rtol=1e-5)


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
