"""Tests for the negative-feedback change limiter (multall's ``DAMP``) applied
to the ASSEMBLED increment -- ``damp_increment`` and the copy of the same
arithmetic fused into ``mg_fine_scatter``'s rolling buffer.

Correctness contract:

- ``dampin <= 0`` is the exact identity: the state stays zero and the march is
  bitwise what it was before the limiter existed.
- The state is lagged one call, so the FIRST call of a march is unlimited
  whatever ``dampin`` is; the limiter engages from the second call on.
- The stored state is scale-invariant, so a lag carried across RK stages of
  differing ``alpha`` is not mis-scaled by the ratio of stage coefficients.
- The limiter acts DOWNSTREAM of the multigrid restriction. This is the whole
  point of the position (multall damps at ``tblock-p-2_3_1.f:7736``, after the
  block sums at 7710-7713); the limiter removed in ember ``7b4fd71`` sat
  upstream of the restriction and broke the box sum's conservation. Here it
  must leave the residual history untouched.
- A cell far above its block mean is pulled back towards ``dampin`` times it.
"""

import numpy as np

import ember.grid  # noqa: F401  binds ember.fortran
import ember.solver

NP = 5
CFL = 0.4
FAC_MGRID = 0.2

# Shape divisible by 2**3, small enough to be fast.
NI, NJ, NK = 17, 17, 17
N_LEVELS = 3


def _make_inputs(ni, nj, nk, seed):
    """Fortran-ordered inputs for one scree_plain / scree_mg_noirs call."""
    rng = np.random.default_rng(seed)

    def F(a):
        return np.asfortranarray(a.astype(np.float32))

    return dict(
        residual=F(rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))),
        dt_vol=F(0.5 + rng.random((ni - 1, nj - 1, nk - 1))),
        vol=F(0.5 + rng.random((ni - 1, nj - 1, nk - 1))),
        store=F(rng.standard_normal((ni - 1, nj - 1, nk - 1, NP))),
        cons=F(rng.standard_normal((ni, nj, nk, NP))),
    )


def _Z(*shape):
    return np.asfortranarray(np.zeros(shape, dtype=np.float32))


def _run_plain(inp, rfac, dampin, cfl=CFL, ni=NI, nj=NJ, nk=NK):
    """One scree_plain call. ``rfac`` is carried in and updated in place."""
    cons = np.asfortranarray(inp["cons"].copy())
    store = np.asfortranarray(inp["store"].copy())
    ember.fortran.scree_plain(
        cons=cons,
        residual=inp["residual"],
        store=store,
        dt_vol=inp["dt_vol"],
        cfl=cfl,
        tmp=_Z(ni - 1, nj - 1, nk - 1, NP),
        rfac=rfac,
        dampin=dampin,
    )
    return cons, store


def _run_mg(inp, rfac, dampin, cfl=CFL, ni=NI, nj=NJ, nk=NK, n_levels=N_LEVELS):
    """One scree_mg_noirs call. ``rfac`` is carried in and updated in place."""
    cons = np.asfortranarray(inp["cons"].copy())
    store = np.asfortranarray(inp["store"].copy())
    nc1i, nc1j, nc1k = (ni - 1) // 2, (nj - 1) // 2, (nk - 1) // 2
    n_corr, n_tri = ember.solver._mg_coarse_scratch_sizes(ni, nj, nk, n_levels)
    ember.fortran.scree_mg_noirs(
        cons=cons,
        residual=inp["residual"],
        store=store,
        dt_vol=inp["dt_vol"],
        vol=inp["vol"],
        cfl=cfl,
        fmgrid=FAC_MGRID,
        expon_mgrid=2.0,
        sf_irs=0.0,
        n_levels=n_levels,
        rbuf=_Z(ni - 1, nj - 1, NP, 2),
        dtblk=_Z(nc1i, nc1j, nc1k),
        rawbuf=_Z(nc1i, nc1j, nc1k, NP),
        sdt=_Z(nc1i, nc1j, nc1k),
        sv=_Z(nc1i, nc1j, nc1k),
        corr_all=_Z(n_corr),
        triw=_Z(n_tri),
        rfac=rfac,
        dampin=dampin,
    )
    return cons, store


def test_dampin_zero_is_the_identity():
    """dampin <= 0 must never leave the zero state, on either path.

    That zero is what makes the per-cell scaling ``dU/(1 + |dU|*0)`` exactly
    ``dU``, so an undamped march is bitwise the pre-limiter one and the
    production kernels stay branch-free.
    """
    inp = _make_inputs(NI, NJ, NK, seed=1)
    for runner in (_run_plain, _run_mg):
        rfac = _Z(NP)
        for _ in range(3):
            runner(inp, rfac, dampin=0.0)
            assert np.all(rfac == 0.0)


def test_first_call_is_unlimited():
    """The state is lagged, so the first call of a march cannot be limited.

    Whatever dampin is, rfac starts at zero and the first call's scaling is the
    identity -- it only accumulates the mean for the next call.

    Equal to float32 rounding rather than bitwise, and the gap is instructive:
    dampin=0 skips the limiter traversal entirely while dampin=1 runs it with
    rfac == 0, and merely reading the increment back changes how the compiler
    contracts the multiply-add that built it (~1 ULP). That is the same effect
    the guard in scree.f90 exists to avoid, seen from the other side. The
    clipping itself is worth far more than 1 ULP -- see
    test_limiter_engages_and_shrinks_the_increment, where the second call moves
    the result by a visible fraction.
    """
    inp = _make_inputs(NI, NJ, NK, seed=2)
    for runner in (_run_plain, _run_mg):
        cons_off, _ = runner(inp, _Z(NP), dampin=0.0)
        cons_on, _ = runner(inp, _Z(NP), dampin=1.0)
        np.testing.assert_allclose(
            cons_on, cons_off, rtol=1e-6, atol=1e-6 * np.abs(cons_off).max()
        )


def test_limiter_engages_and_shrinks_the_increment():
    """From the second call on, a positive dampin must shrink the increment.

    The soft-clip ``dU/(1 + |dU|/(dampin*mean))`` is a contraction of every
    cell, so the whole nodal change must come down.
    """
    inp = _make_inputs(NI, NJ, NK, seed=3)
    for runner in (_run_plain, _run_mg):
        rfac_off, rfac_on = _Z(NP), _Z(NP)
        runner(inp, rfac_off, dampin=0.0)  # priming call
        runner(inp, rfac_on, dampin=1.0)
        cons_off, _ = runner(inp, rfac_off, dampin=0.0)
        cons_on, _ = runner(inp, rfac_on, dampin=1.0)
        assert not np.allclose(cons_on, cons_off)
        dU_off = np.abs(cons_off - inp["cons"])
        dU_on = np.abs(cons_on - inp["cons"])
        assert dU_on.max() < dU_off.max()


def test_stored_state_is_scale_invariant():
    """rfac must not depend on the march coefficient it was measured at.

    The increment is linear in ``scale`` (cfl for scree, alpha*cfl for an RK
    stage) and so is sum|dU|, so ``ncell*scale/(dampin*sum|dU|)`` is not. This
    is what lets one lagged state serve four RK stages whose alpha differ by 4x
    without alternately over- and under-damping around the cycle.
    """
    inp = _make_inputs(NI, NJ, NK, seed=4)
    for runner in (_run_plain, _run_mg):
        rfac_1, rfac_2 = _Z(NP), _Z(NP)
        runner(inp, rfac_1, dampin=25.0, cfl=CFL)
        runner(inp, rfac_2, dampin=25.0, cfl=2.0 * CFL)
        assert np.all(rfac_1 > 0.0)
        np.testing.assert_allclose(rfac_1, rfac_2, rtol=1e-5)


def test_limiter_leaves_the_residual_history_alone():
    """The limiter must act downstream of the restriction, not on the residual.

    ember's removed limiter (7b4fd71) rescaled the residual before the box sum
    saw it, which destroyed the extensivity the restriction depends on. This one
    touches only the assembled increment, so the rolled history must come out
    exactly equal to the residual regardless of dampin.
    """
    inp = _make_inputs(NI, NJ, NK, seed=5)
    for runner in (_run_plain, _run_mg):
        rfac = _Z(NP)
        runner(inp, rfac, dampin=1.0)  # priming call, so the next one limits
        _, store_out = runner(inp, rfac, dampin=1.0)
        np.testing.assert_array_equal(store_out, inp["residual"])


def test_outlier_is_clipped_towards_dampin_times_the_mean():
    """A cell far above its block mean saturates, as multall's feedback does.

    ``dU/(1 + |dU|/(dampin*mean))`` tends to ``dampin*mean`` as |dU| grows, so a
    single spike orders of magnitude above the mean must come back to about
    that bound rather than through it.
    """
    dampin = 4.0
    inp = _make_inputs(NI, NJ, NK, seed=6)
    # One cell's residual set enormous, in the first conserved variable only.
    residual = np.asfortranarray(inp["residual"].copy())
    residual[5, 5, 5, 0] = 1.0e4
    inp = dict(inp, residual=residual)

    rfac = _Z(NP)
    _run_plain(inp, rfac, dampin=dampin)  # priming call: accumulates the mean
    # rfac = ncell*cfl/(dampin*sum|dU|), so the saturation bound dampin*mean
    # expressed on this call's scale is simply 1/(rfac/cfl).
    bound = CFL / rfac[0]
    cons, _ = _run_plain(inp, rfac, dampin=dampin)
    dU = np.abs(cons - inp["cons"])[..., 0]
    # cell_to_node averages the limited cell increment onto the nodes, which
    # can only reduce it further, so the bound holds with margin.
    assert dU.max() <= bound * (1.0 + 1e-4)
