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
- every weight stays a bounded blend, on a sheared mesh clustered off the
  direction being resolved -- the shape that made the LISA rotor diverge
- each pass is anchored where the last one landed, so the interpolant is
  evaluated at the node it claims to be
- ``mg_interp_i2x``/``mg_interp_i2x_node`` fed the index weights reproduce their
  hardcoded blends
"""

import numpy as np
import pytest

import ember
from ember import fortran, util
from ember.block import (
    MG_W_HI,
    MG_W_LO,
    _mg_centroid_ladder,
    _mg_index_bracket_node,
    _mg_n_hops,
    _mg_project,
    _mg_weight_lengths,
)

SHAPE = (17, 17, 17)  # 16 cells a side, so three factor-2 hops exist
LENGTH = 0.05
NB = 157


def _block(xv=None, ER_r=1.0, shear=0.0):
    """Single block on a duct-like annular mesh.

    ``xv`` overrides the axial node distribution, which is the one direction
    whose cell centroids are exactly the mean of their nodes however the
    annulus curves -- so a stretch there is a clean test of the projection.

    ``ER_r`` clusters the RADIAL nodes towards the hub and ``shear`` tilts the
    j lines in x by ``shear*(r - r_hub)``. Both default off, leaving the mesh
    every other test here is written against. The pair together is what makes a
    coarse j edge short AND gives it an axial component, which is the only mesh
    on which the j-pass projection is ill-conditioned -- see
    :func:`test_weights_are_a_bounded_blend`.
    """
    pitch = 2.0 * np.pi / NB
    xrt = util.linmesh3([0.0, LENGTH], [1.0, 1.0 + LENGTH], [0.0, pitch], SHAPE)
    r = xrt[..., 1]
    if ER_r > 1.0:
        rv = 1.0 + _stretch(SHAPE[1], ER_r) * LENGTH
        r = np.broadcast_to(rv[None, :, None], SHAPE).astype(np.float32)
    x = xrt[..., 0] if xv is None else np.broadcast_to(xv[:, None, None], SHAPE)
    x = x + shear * (r - 1.0)
    block = ember.block.Block(shape=SHAPE)
    block.set_x(np.ascontiguousarray(x, dtype=np.float32))
    block.set_r(np.ascontiguousarray(r, dtype=np.float32))
    block.set_t(xrt[..., 2])
    block.set_Nb(NB)
    return block


def _stretch(n, ER):
    """Geometrically stretched node positions on [0, 1]."""
    d = ER ** np.arange(n - 1, dtype=np.float64)
    x = np.concatenate([[0.0], np.cumsum(d)])
    return x / x[-1]


def _stretched_x(n, ER=1.2):
    """Geometrically stretched node positions on [0, LENGTH]."""
    return (LENGTH * _stretch(n, ER)).astype(np.float32)


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
    return _hop1_weights(block)[0]


def _hop1_weights(block):
    """All three directions of hop 1, unpacked to the shapes the kernel reads.

    Hop 1 is packed first in each direction's array, so its slice is the first
    ``_mg_weight_lengths(shape, 1)`` entries.
    """
    ni, nj, nk = block.shape
    ncj, nck = (nj - 1) // 2, (nk - 1) // 2
    shapes = ((ni, ncj, nck), (ni, nj, nck), (ni, nj, nk))
    return tuple(
        w[: np.prod(sh)].reshape(sh, order="F").astype(np.float64)
        for w, sh in zip(block.weight_mgrid, shapes)
    )


def _pass_gain(w):
    """Worst-case gain of one blend pass, ``max(|1-w| + |w|)``.

    A blend ``(1-w)*a + w*b`` cannot amplify its two inputs by more than this,
    and it is exactly 1 while w stays in [0, 1]. The three passes compose, so
    the product is what the final hop can do to a coarse correction before it
    is added to a node at full weight.
    """
    return float((np.abs(1.0 - w) + np.abs(w)).max())


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


# A sheared mesh clustered in r is the one that breaks the j pass, and neither
# ingredient does it alone: the clustering is what makes a coarse j edge short,
# the shear is what gives it an axial component for an i-offset to project
# onto. Measured gains in j on this block, before the anchoring fix:
#
#   ER_r=1.0 shear=0.0 -> 1.0     ER_r=1.3 shear=2.0 -> 10.3
#   ER_r=1.3 shear=0.0 -> 1.1     ER_r=1.4 shear=4.0 -> 13.1
#   ER_r=1.0 shear=2.0 -> 1.4
#
# Every other test in this file stretches x alone, on a mesh that is otherwise
# separable -- and there the coarse j edge is purely radial while the i-offset
# is purely axial, so the bad projection is identically zero. That is why this
# went unseen until the LISA rotor diverged on it.
SKEWED = [(1.0, 0.0), (1.3, 0.0), (1.3, 2.0), (1.4, 4.0)]


@pytest.mark.parametrize("ER_r,shear", SKEWED)
def test_weights_are_a_bounded_blend(ER_r, shear):
    """No mesh may turn the final hop into an amplifier.

    The correction this hop produces is added to the node at full weight, so
    the gain of the three passes composed is the factor by which a coarse
    correction can arrive larger than it left. A blend is only a blend while
    its coefficient is bounded; past that it is an extrapolation of arbitrary
    reach, and on the LISA rotor it reached -25.

    The projection cannot be trusted to stay bounded on its own -- it divides
    by the square of a coarse edge that wall clustering makes arbitrarily
    short -- so ``_mg_project`` clamps it, and this is the assertion that the
    clamp is load-bearing rather than decorative.
    """
    ws = _hop1_weights(_block(_stretched_x(SHAPE[0]), ER_r=ER_r, shear=shear))

    for w in ws:
        assert w.min() >= MG_W_LO
        assert w.max() <= MG_W_HI

    # Which bounds the composed gain, the number that actually matters.
    cap = (1.0 + 2.0 * max(-MG_W_LO, MG_W_HI - 1.0)) ** 3
    assert np.prod([_pass_gain(w) for w in ws]) <= cap


@pytest.mark.parametrize("ER_r,shear", SKEWED)
def test_each_pass_is_anchored_where_the_last_one_landed(ER_r, shear):
    """Pushing the coarse POSITIONS through the three passes must give the nodes.

    The three-dimensional statement of what
    :func:`test_weights_place_the_fine_node_where_it_really_is` pins in i
    alone. Pass J blends two values pass I has already placed at node ``i``, so
    its weight has to measure along the segment joining THOSE positions. Anchor
    it at the coarse centroids of some representative i index instead and the
    target sits off its own segment by up to half a coarse i cell, which on a
    sheared mesh projects onto the j edge and divides by its length.

    Interpolating position rather than field is the direct test: linear
    interpolation of a linear quantity is exact, and position is the most
    linear quantity there is, so what comes out is where the interpolant is
    actually being evaluated.
    """
    block = _block(_stretched_x(SHAPE[0]), ER_r=ER_r, shear=shear)
    nodes, levels = _mg_centroid_ladder(block._xrt_nd, block.vol_nd, _mg_n_hops(SHAPE))
    Pc = levels[1]
    got = _prolong_positions(Pc, nodes.shape[:3], *_hop1_weights(block))

    assert _misplacement(got, nodes, Pc) < ANCHOR_TOL


def test_anchoring_at_the_bracket_index_misplaces_the_interpolant():
    """The counter-case: what the test above rules out, ruled out by name.

    On the sheared mesh the old anchoring puts the interpolant an eighth of a
    coarse cell from the node it claims to evaluate at -- five times the bar,
    and on the LISA rotor enough to make the blend coefficient -25.
    """
    block = _block(_stretched_x(SHAPE[0]), ER_r=1.3, shear=2.0)
    nodes, levels = _mg_centroid_ladder(block._xrt_nd, block.vol_nd, _mg_n_hops(SHAPE))
    Pc = levels[1]
    old = _prolong_positions(Pc, nodes.shape[:3], *_hop1_weights_old(block, Pc))

    assert _misplacement(old, nodes, Pc) > 2.0 * ANCHOR_TOL


# How far the interpolant may sit from the node it is evaluated at, as a
# fraction of a coarse cell. The anchored weights reach 2.5% on the worst mesh
# here, the old ones 12.6%: a whole coarse cell is what a mis-anchored pass is
# wrong by, so the scale is right and the margin is real.
ANCHOR_TOL = 0.05


def _misplacement(got, nodes, Pc):
    """Node-position error of a prolongation, in coarse cells.

    The outer two node planes are excluded: past the outer coarse centroids the
    weights are deliberately flat, so the interpolant is meant to sit off the
    node there.
    """
    inner = (slice(2, -2),) * 3
    cell = max(np.abs(np.diff(Pc, axis=ax)).max() for ax in (0, 1, 2))
    return np.abs(got[inner] - nodes[inner]).max() / cell


def _prolong_positions(Pc, node_shape, wi, wj, wk):
    """Run the final hop's three passes on the coarse centroid POSITIONS."""
    ni, nj, nk = node_shape
    nci, ncj, nck = Pc.shape[:3]
    il, ih = _mg_index_bracket_node(ni, nci)
    jl, jh = _mg_index_bracket_node(nj, ncj)
    kl, kh = _mg_index_bracket_node(nk, nck)
    a = Pc[il] * (1.0 - wi[..., None]) + Pc[ih] * wi[..., None]
    a[0] = Pc[0]  # the kernel's hardcoded cout(1) = cin(1)
    b = a[:, jl] * (1.0 - wj[..., None]) + a[:, jh] * wj[..., None]
    return b[:, :, kl] * (1.0 - wk[..., None]) + b[:, :, kh] * wk[..., None]


def _hop1_weights_old(block, Pc):
    """Hop 1's weights as they were built before the anchoring fix.

    Kept here, and nowhere else, so the counter-case above is a statement about
    a specific alternative rather than about nothing in particular: every pass
    anchored at ``Pc[il]``, the coarse centroids of the bracket's own index.
    """
    nodes, _ = _mg_centroid_ladder(block._xrt_nd, block.vol_nd, _mg_n_hops(SHAPE))
    ni, nj, nk = nodes.shape[:3]
    nci, ncj, nck = Pc.shape[:3]
    il, ih = _mg_index_bracket_node(ni, nci)
    jl, jh = _mg_index_bracket_node(nj, ncj)
    kl, kh = _mg_index_bracket_node(nk, nck)
    jmid = np.minimum(np.arange(1, ncj + 1) * 2 - 1, nj - 1)
    kmid = np.minimum(np.arange(1, nck + 1) * 2 - 1, nk - 1)

    def ends(lo, hi, nc):
        return lo == 0, hi == nc - 1

    ilo, ihi = ends(il, ih, nci)
    wi = _mg_project(
        nodes[:, jmid][:, :, kmid], Pc[il], Pc[ih], ilo[:, None, None], ihi[:, None, None]
    )
    Pci = Pc[il]
    jlo, jhi = ends(jl, jh, ncj)
    wj = _mg_project(
        nodes[:, :, kmid],
        Pci[:, jl],
        Pci[:, jh],
        jlo[None, :, None],
        jhi[None, :, None],
    )
    Pcij = Pci[:, jl]
    klo, khi = ends(kl, kh, nck)
    wk = _mg_project(
        nodes, Pcij[:, :, kl], Pcij[:, :, kh], klo[None, None, :], khi[None, None, :]
    )
    return wi, wj, wk


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


def _prolong_only(block, coarse, weights=None):
    """Run the fused final hop with the fine term switched off.

    ``scale = 0`` kills ``scale*dt_vol*q``, so what lands on ``cons`` is the
    prolonged coarse correction and nothing else. ``weights`` overrides the
    block's own, which is how an alternative set can be put through the real
    kernel rather than a numpy restatement of it.
    """
    ni, nj, nk = block.shape
    ncj, nck = (nj - 1) // 2, (nk - 1) // 2

    def Z(*shape):
        return np.asfortranarray(np.zeros(shape, dtype=np.float32))

    cons = Z(ni, nj, nk, 5)
    wi, wj, wk = (
        np.asfortranarray(w, dtype=np.float32)
        for w in (weights if weights is not None else _hop1_weights(block))
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


@pytest.mark.parametrize("ER_r,shear", SKEWED)
def test_prolonged_linear_correction_survives_shear(ER_r, shear):
    """The same claim on the mesh that broke it, against the anchoring it replaced.

    The test above takes its gradient along x, which this annular mesh makes
    exactly separable -- so it can demand round-off and does. Shear costs the
    separability: each pass then measures along a segment the next one tilts,
    and a three-component gradient picks up the annulus curvature too, so what
    is left is a real error rather than float32 noise. Bounding it is still
    worth doing, because a mis-anchored pass shows up here as multiples of it:
    the anchoring this replaced reaches 2.8% on the ER_r=1.3, shear=2 mesh
    against 0.6% here, which is what the bar below is set to separate.
    """
    block = _block(_stretched_x(SHAPE[0]), ER_r=ER_r, shear=shear)
    nodes, levels = ember.block._mg_centroid_ladder(
        block._xrt_nd, block.vol_nd, _mg_n_hops(SHAPE)
    )
    Pc = levels[1]
    origin = nodes.reshape(-1, 3).mean(axis=0)
    grad = np.array([0.3, 1.0, -0.7])  # nothing lines up with a mesh direction
    coarse = ((Pc - origin) @ grad).astype(np.float32)
    want = (nodes - origin) @ grad

    inner = (slice(2, -2),) * 3
    got = _prolong_only(block, coarse)

    assert np.abs(got[inner] - want[inner]).max() / np.ptp(want) < 0.015
