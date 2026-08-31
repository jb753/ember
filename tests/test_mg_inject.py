"""Tests for the multigrid prolongation, which is plain INJECTION.

Every fine cell under a coarse block takes that block's correction unaltered,
and the correction -- a cell quantity like the fine term -- rides the same
``cell_to_node`` scatter. This replaced a cascaded trilinear prolongation whose
final hop targeted the fine nodes through geometry-derived weights; see
``docs/dev/plan_piecewise_constant_mgrid.md``.

What these pin, in the order the design document
(``docs/dev/plan_piecewise_constant_mgrid.md`` section 3) states them:

1. the state increment from the correction is block-uniform before the scatter,
   and after it is that block value on the nodes interior to a block and the
   mean of two blocks on the faces between them;
2. the block-sum restriction is exactly the transpose of the injection, with no
   scaling, on any mesh -- of the CELL-TO-CELL pair, which is not the operator
   the solver applies (that one has the scatter composed onto it);
3. the DC gain is ``fac_mgrid * sum_l b_l * expon_mgrid**-(l-1)`` fine terms,
   in closed form.

Point 4 of that list -- the correction vanishes with the residual, so the steady
state is unchanged -- is structural: ``corr_all`` is linear in ``q`` and every
test here differences against a ``fac_mgrid = 0`` run that relies on it.
"""

from pathlib import Path

import numpy as np
import pytest

import ember.block
import ember.grid  # noqa: F401  binds ember.fortran
import ember.solver
from ember import util
from ember.block import Block
from ember.fluid import PerfectFluid

NP = 5
CFL = 0.4
ALPHA = 0.5
FAC_MGRID = 0.3
EXPON_MGRID = 2.0
N_LEVELS = 3

# 16 cells a side: a multiple of 2**N_LEVELS, as ember.solver._validate_mg
# requires and as the injection index (i+1)/2 depends on (there is no bracket
# and no clamp to fall back on).
SHAPE = (17, 17, 17)

GOLDEN_FILE = Path(__file__).parent / "data" / "mg_inject_golden.npz"


# ---------------------------------------------------------------------------
# Fixtures and reference implementations
# ---------------------------------------------------------------------------


def _make_block(shape, cluster=False):
    """A block with real geometry, so ``vol_nd`` is nonzero and non-uniform.

    ``cluster`` bunches the j nodes towards j=0, which is what makes ``dtblk``
    differ block to block and so is what lets a test tell injection from
    interpolation.
    """
    block = Block(shape=shape)
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
    ni, nj, nk = shape
    xrt = util.linmesh3((0.0, 0.3), (0.5, 0.9), (0.0, 0.2), shape)
    if cluster:
        s = np.linspace(0.0, 1.0, nj) ** 2.5
        r = 0.5 + 0.4 * s
        xrt[..., 1] = r.reshape(1, -1, 1)
    block.set_x(np.ascontiguousarray(xrt[..., 0]))
    block.set_r(np.ascontiguousarray(xrt[..., 1]))
    block.set_t(np.ascontiguousarray(xrt[..., 2]))
    block.set_P_T(101325.0, 300.0)
    return block


def _seed(block, residual, dt_vol):
    """Poke a synthetic residual and timestep onto a block."""
    block.residual_nd.flags.writeable = True
    block.residual_nd[...] = residual
    block.residual_nd.flags.writeable = False
    block.dt_vol_nd.flags.writeable = True
    block.dt_vol_nd[...] = dt_vol
    block.dt_vol_nd.flags.writeable = False


def _rk_increment(shape, residual, dt_vol, *, fac_mgrid, cluster=False):
    """One RK stage's node increment ``cons - snapshot``."""
    block = _make_block(shape, cluster=cluster)
    _seed(block, residual, dt_vol)
    grid = ember.grid.Grid([block])
    block.store[...] = block.conserved_nd
    snapshot = np.array(block.store, dtype=np.float64)
    ember.solver.advance_rk_stage_mg(
        grid,
        ALPHA,
        CFL,
        fac_mgrid,
        N_LEVELS,
        expon_mgrid=EXPON_MGRID,
        sf_irs=0.0,
    )
    return np.array(block.conserved_nd, dtype=np.float64) - snapshot, block


def _block_sums(field, b):
    """Sum ``field`` over every ``b**3`` block of cells, keeping the trailing axis."""
    ni, nj, nk = field.shape[:3]
    r = field.reshape(ni // b, b, nj // b, b, nk // b, b, -1)
    return r.sum(axis=(1, 3, 5))


def _reference_correction_cells(residual, dt_vol, vol, scale, fac_mgrid, n_levels):
    """The coarse correction as a CELL field, from (PC1), in float64.

    Independent of the kernel: hierarchical restriction written out as flat
    block sums, the volume-weighted harmonic mean for the coarse timestep, and
    the correction of level ``l`` broadcast unaltered over its ``b**3`` cells.
    """
    residual = np.asarray(residual, dtype=np.float64)
    dt_vol = np.asarray(dt_vol, dtype=np.float64)[..., None]
    vol = np.asarray(vol, dtype=np.float64)[..., None]
    out = np.zeros(residual.shape)
    for lvl in range(1, n_levels + 1):
        b = 2**lvl
        cres = _block_sums(residual, b)
        dtblk = _block_sums(vol, b) / _block_sums(vol / dt_vol, b)
        coef = scale * fac_mgrid / b**2 * EXPON_MGRID ** -(lvl - 1)
        corr = coef * dtblk * cres
        out += np.repeat(np.repeat(np.repeat(corr, b, 0), b, 1), b, 2)
    return out


def _scatter_to_nodes(cell):
    """``cell_to_node``: every node takes the mean of the cells it touches.

    A partition of unity everywhere -- 8 cells in the interior, 4 on a face, 2
    on an edge, 1 at a corner -- which is why a block-constant cell field comes
    through it unchanged inside a block.
    """
    ni, nj, nk = (n + 1 for n in cell.shape[:3])
    acc = np.zeros((ni, nj, nk, cell.shape[3]))
    cnt = np.zeros((ni, nj, nk, 1))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                acc[di : di + ni - 1, dj : dj + nj - 1, dk : dk + nk - 1] += cell
                cnt[di : di + ni - 1, dj : dj + nj - 1, dk : dk + nk - 1] += 1.0
    return acc / cnt


@pytest.fixture
def synthetic():
    """A residual and timestep varying strongly between ADJACENT cells.

    A smooth ramp will not do: any prolongation reproduces a linear field, so a
    ramp cannot tell injection from interpolation. That mistake has been made
    once already in this subsystem.
    """
    ni, nj, nk = SHAPE
    rng = np.random.default_rng(20260830)
    residual = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))
    dt_vol = 0.2 + 1.6 * rng.random((ni - 1, nj - 1, nk - 1))
    return residual, dt_vol


# ---------------------------------------------------------------------------
# 4.1  The increment is uniform over a coarse block
# ---------------------------------------------------------------------------


def test_the_increment_is_uniform_over_a_coarse_block(synthetic):
    """The property the whole scheme rests on, in its post-scatter form.

    The pre-scatter cell increment is never materialised -- the correction is
    fused into the rolling-plane scatter -- and the post-scatter increment is
    NOT block-constant, because block-face nodes are averaged. So the assertion
    is made against the reference cell field pushed through the same scatter,
    plus the direct statement at the nodes interior to a block.
    """
    residual, dt_vol = synthetic
    got, block = _rk_increment(
        SHAPE, residual, dt_vol, fac_mgrid=FAC_MGRID, cluster=True
    )
    off, _ = _rk_increment(
        SHAPE, residual, dt_vol, fac_mgrid=0.0, cluster=True
    )
    corr = got - off

    cells = _reference_correction_cells(
        residual, dt_vol, block.vol_nd, ALPHA * CFL, FAC_MGRID, N_LEVELS
    )
    want = _scatter_to_nodes(cells)
    scale = np.abs(want).max()
    assert scale > 0.0
    np.testing.assert_allclose(corr, want, rtol=2e-4, atol=1e-5 * scale)

    # The block-uniformity itself, stated directly. Cells 2a and 2a+1 form
    # level-1 block a, so node 2a+1 -- the only node fed by two cells of one
    # block in every direction -- carries that block's correction unaltered,
    # with nothing averaged into it.
    np.testing.assert_allclose(
        corr[1::2, 1::2, 1::2],
        cells[::2, ::2, ::2],
        rtol=2e-4,
        atol=1e-5 * scale,
    )
    # And the block faces are where the one-cell smoothing lives: node 2a is
    # fed by cells 2a-1 and 2a, which belong to blocks a-1 and a.
    np.testing.assert_allclose(
        corr[2:-2:2, 1::2, 1::2],
        0.5 * (cells[1:-2:2, ::2, ::2] + cells[2:-1:2, ::2, ::2]),
        rtol=2e-4,
        atol=1e-5 * scale,
    )


# ---------------------------------------------------------------------------
# 4.2  Restriction is the transpose of injection
# ---------------------------------------------------------------------------


def _dense_injection(nc, np_):
    """The injection matrix P as (n_fine, n_coarse), from ``mg_inject_acc``."""
    nf = 2 * nc
    cols = []
    for c in range(nc**3):
        src = np.asfortranarray(np.zeros((nc, nc, nc, np_), dtype=np.float32))
        src.reshape(-1, np_, order="F")[c, 0] = 1.0
        out = np.asfortranarray(np.zeros((nf, nf, nf, np_), dtype=np.float32))
        ember.fortran.mg_inject_acc(src=src, out=out)
        cols.append(out[..., 0].reshape(-1, order="F").copy())
    return np.stack(cols, axis=1)


def _dense_restriction(nc, np_):
    """The block-sum matrix R as (n_coarse, n_fine), from the production kernel.

    Driven through ``rk_mg_noirs`` rather than ``mg_restrict_levels``, whose
    dummy-procedure smoother f2py exposes as a Python callback. ``dt_vol`` and
    ``vol`` are 1 and ``fmgrid`` is chosen so that ``coef_1 * dtblk_1 == 1``,
    which leaves ``corr_all`` holding the bare block sum: the transfer, with
    nothing of the timestep weighting in it.
    """
    nf = 2 * nc
    ni = nj = nk = nf + 1
    rows = []
    for f in range(nf**3):
        residual = np.asfortranarray(np.zeros((nf, nf, nf, np_), dtype=np.float32))
        residual.reshape(-1, np_, order="F")[f, 0] = 1.0
        ones3 = np.asfortranarray(np.ones((nf, nf, nf), dtype=np.float32))

        def Z(*shape):
            return np.asfortranarray(np.zeros(shape, dtype=np.float32))

        n_corr, n_tri = ember.solver._mg_coarse_scratch_sizes(
            ni, nj, nk, 1, np=np_
        )
        corr_all = Z(n_corr)
        ember.fortran.rk_mg_noirs(
            cons=Z(ni, nj, nk, np_),
            snapshot=Z(ni, nj, nk, np_),
            residual=residual,
            dt_vol=ones3,
            vol=ones3,
            alpha=1.0,
            cfl=1.0,
            fmgrid=4.0,  # coef_1 = 1*1*4/2**2 = 1
            expon_mgrid=EXPON_MGRID,
            sf_irs=0.0,
            n_levels=1,
            rbuf=Z(nf, nf, np_, 2),
            dtblk=Z(nc, nc, nc),
            rawbuf=Z(nc, nc, nc, np_),
            sdt=Z(nc, nc, nc),
            sv=Z(nc, nc, nc),
            corr_all=corr_all,
                triw=Z(n_tri),
            rfac=Z(np_),
            dampin=0.0,
        )
        rows.append(corr_all.reshape(nc, nc, nc, np_, order="F")[..., 0].ravel("F"))
    return np.stack(rows, axis=1)


def test_restriction_is_the_transpose_of_injection():
    """``R == P.T`` exactly, with no scaling and no geometry.

    This is close to true by construction, and that is the point: it is the
    property two earlier branches spent their effort engineering, and here it
    falls out of doing less. Nothing in either operator reads the mesh.
    """
    nc = 4
    P = _dense_injection(nc, 1)
    R = _dense_restriction(nc, 1)
    assert P.shape == (8**3, 4**3)
    # The pattern is the transpose EXACTLY -- no entry is anywhere the other is
    # not -- and every nonzero carries the same value, so there is no per-entry
    # weight, normalisation or geometry anywhere in the pair. That value is
    # coef_1 * dtblk_1, driven to 1 above; it belongs to the scaling, not to
    # the transfer, and lands one ulp off it under -Ofast.
    np.testing.assert_array_equal(R != 0.0, P.T != 0.0)
    nonzero = R[R != 0.0]
    assert nonzero.min() == nonzero.max()
    np.testing.assert_allclose(R, P.T, rtol=1e-6, atol=0.0)
    # Every fine cell belongs to exactly one block, and every block owns eight.
    np.testing.assert_array_equal(P.sum(axis=1), np.ones(8**3, dtype=np.float32))
    np.testing.assert_array_equal(P.sum(axis=0), 8.0 * np.ones(4**3, dtype=np.float32))


def test_the_applied_transfer_is_not_the_adjoint_pair():
    """Scope of the claim above, asserted rather than only documented.

    What the solver applies is ``S . I``: injection with the cell->node scatter
    composed onto it. ``S`` is row-stochastic -- which is what carries a
    block-constant correction through unchanged, and is the whole of 2.1's
    argument -- but it is not a permutation, so the applied prolongation is not
    the transpose of the block-sum restriction. The exact adjointness belongs
    to the cell-to-cell pair and to nothing else.
    """
    n = 4  # cells a side
    S = np.stack(
        [
            _scatter_to_nodes(e.reshape(n, n, n, 1))[..., 0].ravel()
            for e in np.eye(n**3)
        ],
        axis=1,
    )
    np.testing.assert_allclose(S.sum(axis=1), 1.0)
    # Interior nodes average eight cells, so S is emphatically not a permutation.
    assert (S > 0).sum(axis=1).max() == 8


# ---------------------------------------------------------------------------
# 4.3  A constant residual gives the calibrated gain
# ---------------------------------------------------------------------------


def test_a_constant_residual_gives_the_calibrated_gain():
    """The DC gain, in closed form.

    ``dt_vol`` is held CONSTANT so the closed form is exact: the coarse
    timestep is the volume-weighted harmonic mean ``sum(vol)/sum(vol/dt_vol)``,
    which for constant ``dt_vol`` collapses to ``dt_vol`` on any mesh whatever
    the volumes are, leaving the level correction uniform.

    The closed form is ``fac_mgrid * sum_l b_l * expon_mgrid**-(l-1)`` fine
    terms, which at ``expon_mgrid=2, n_levels=3`` is ``2+2+2 = 6`` -- the three
    levels contribute equally.
    """
    ni, nj, nk = SHAPE
    r_const, dt_const = 0.75, 1.25
    residual = np.full((ni - 1, nj - 1, nk - 1, NP), r_const)
    dt_vol = np.full((ni - 1, nj - 1, nk - 1), dt_const)

    off, _ = _rk_increment(SHAPE, residual, dt_vol, fac_mgrid=0.0)
    on, _ = _rk_increment(SHAPE, residual, dt_vol, fac_mgrid=FAC_MGRID)

    levels = sum(2**lvl * EXPON_MGRID ** -(lvl - 1) for lvl in range(1, N_LEVELS + 1))
    assert levels == pytest.approx(6.0)
    want = ALPHA * CFL * FAC_MGRID * dt_const * r_const * levels

    np.testing.assert_allclose(on - off, want, rtol=1e-5)


# ---------------------------------------------------------------------------
# 4.4  Off is off
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4.5  The correction reaches conserved, on both integrators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sf_irs", [0.0, 1.0])
def test_the_correction_reaches_conserved_rk(synthetic, sf_irs):
    residual, dt_vol = synthetic

    def run(fac_mgrid):
        block = _make_block(SHAPE)
        _seed(block, residual, dt_vol)
        grid = ember.grid.Grid([block])
        block.store[...] = block.conserved_nd
        ember.solver.advance_rk_stage_mg(
            grid,
            ALPHA,
            CFL,
            fac_mgrid,
            N_LEVELS,
            expon_mgrid=EXPON_MGRID,
            sf_irs=sf_irs,
        )
        return np.array(block.conserved_nd, dtype=np.float64)

    off, on = run(0.0), run(FAC_MGRID)
    assert np.abs(on - off).max() > 1e-6 * np.abs(off).max()


@pytest.mark.parametrize("sf_irs", [0.0, 1.0])
def test_the_correction_reaches_conserved_scree(synthetic, sf_irs):
    residual, dt_vol = synthetic

    def run(fac_mgrid):
        block = _make_block(SHAPE)
        _seed(block, residual, dt_vol)
        grid = ember.grid.Grid([block])
        ember.solver.scree_step(
            grid,
            CFL,
            fac_mgrid=fac_mgrid,
            expon_mgrid=EXPON_MGRID,
            n_levels=N_LEVELS,
            sf_irs=sf_irs,
        )
        return np.array(block.conserved_nd, dtype=np.float64)

    off, on = run(0.0), run(FAC_MGRID)
    assert np.abs(on - off).max() > 1e-6 * np.abs(off).max()


def test_the_two_integrators_agree_on_the_correction(synthetic):
    """scree and RK build the same coarse push at ``alpha = 1``.

    scree takes one full-weight step, so ``coef_l`` is the RK formula at
    ``alpha = 1``; the fine terms differ (scree's is the Denton-lagged
    ``2*residual - store``) but the correction is built from the same ``q`` by
    the same call, so differencing each against its own ``fac_mgrid = 0`` run
    isolates two things that must match.
    """
    residual, dt_vol = synthetic

    def rk(fac_mgrid):
        block = _make_block(SHAPE)
        _seed(block, residual, dt_vol)
        grid = ember.grid.Grid([block])
        block.store[...] = block.conserved_nd
        ember.solver.advance_rk_stage_mg(
            grid,
            1.0,
            CFL,
            fac_mgrid,
            N_LEVELS,
            expon_mgrid=EXPON_MGRID,
        )
        return np.array(block.conserved_nd, dtype=np.float64)

    def scree(fac_mgrid):
        block = _make_block(SHAPE)
        _seed(block, residual, dt_vol)
        grid = ember.grid.Grid([block])
        # store (the Denton history) is zero, so q = 2*residual: halve the
        # residual and the two schemes see the same q.
        ember.solver.scree_step(
            grid,
            CFL,
            fac_mgrid=fac_mgrid,
            expon_mgrid=EXPON_MGRID,
            n_levels=N_LEVELS,
        )
        return np.array(block.conserved_nd, dtype=np.float64)

    d_rk = rk(FAC_MGRID) - rk(0.0)
    d_scree = scree(FAC_MGRID) - scree(0.0)
    # scree's q is 2*residual where RK's is residual, so its push is doubled.
    # atol carries the float32 cancellation of differencing two O(1) states.
    np.testing.assert_allclose(
        d_scree, 2.0 * d_rk, rtol=1e-4, atol=1e-5 * np.abs(d_rk).max()
    )


# ---------------------------------------------------------------------------
# 4.6  Golden and arena
# ---------------------------------------------------------------------------


def _golden_increment():
    """The assembled coarse increment the golden pins: fixed grid, fixed data."""
    ni, nj, nk = SHAPE
    rng = np.random.default_rng(11)
    residual = rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))
    dt_vol = 0.2 + 1.6 * rng.random((ni - 1, nj - 1, nk - 1))
    on, _ = _rk_increment(
        SHAPE, residual, dt_vol, fac_mgrid=FAC_MGRID, cluster=True
    )
    off, _ = _rk_increment(
        SHAPE, residual, dt_vol, fac_mgrid=0.0, cluster=True
    )
    return (on - off).astype(np.float32)


@pytest.mark.skipif(not GOLDEN_FILE.exists(), reason="golden not generated")
def test_golden_coarse_increment():
    """Numbers, not properties: 4.1-4.3 pin the shape of the answer, this the answer.

    Regenerate after an intentional change:

        uv run python tests/test_mg_inject.py
    """
    want = np.load(GOLDEN_FILE)["increment"]
    np.testing.assert_allclose(_golden_increment(), want, rtol=1e-6, atol=1e-12)


def test_the_buffer_list_matches_the_shapes():
    """MG_COARSE_NAMES and mg_coarse_shapes must not drift apart.

    They are zipped together at the carve, so a name added to one and not the
    other silently mislabels every buffer after it.
    """
    names = ember.solver.MG_COARSE_NAMES
    shapes = ember.solver.mg_coarse_shapes(*SHAPE, N_LEVELS)
    assert len(names) == len(shapes) == 6
    assert set(names) == {
        "dtblk", "rawbuf", "sdt", "sv", "corr_all", "triw",
    }


def test_the_multigrid_carve_fits_the_arena():
    """The multigrid phase, increment buffer included, fits what the block allocates."""
    ni, nj, nk = SHAPE
    block = _make_block(SHAPE)
    need = (ni - 1) * (nj - 1) * NP * 2 + sum(
        int(np.prod(s)) for s in ember.solver.mg_coarse_shapes(ni, nj, nk, N_LEVELS)
    )
    assert need <= block.scratch.size


if __name__ == "__main__":
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_FILE, increment=_golden_increment())
    print(f"wrote {GOLDEN_FILE}")
