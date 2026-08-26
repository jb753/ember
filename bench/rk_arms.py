"""Arm definitions for the RK stage advance, `advance_rk_stage_mg`'s kernels.

WHY THIS EXISTS. A whole-step line profile of the default configuration
(1M-cell duct, n_stage=4, cfl=2, n_levels=3, sf_resid=1.0) puts
`advance_rk_stage_mg` at 21% of a serial step and 22.5% of an 8-rank
socket-contended one -- the LARGEST single phase under contention, ahead of
`set_residual`. It had never been in this harness: every tracked result in
bench/results/ is a residual, IRS or viscous number. This module is its
counterpart of `residual_arms.py`/`irs_arms.py`, so
`bench_prod_baseline.py --kernel rk` can time it under the protocol the
README mandates.

The arms are CONFIGURATIONS of production, not competing implementations of
one thing, and they are here to attribute the phase's cost rather than to
race:

  rk       rk_mg_irs    -- production as the solver defaults run it:
                          n_levels=3 coarse levels, coarse-IRS on
  rknoirs  rk_mg_noirs  -- the same stage with the coarse residual left
                          unsmoothed, which is what sf_resid=0 dispatches.
                          The difference prices the coarse smoother.
  rkplain  rk_plain     -- fine RK term and scatter only, the n_levels=0
                          fast path. The difference from `rknoirs` prices
                          the whole restrict/prolong machinery.

They compute different things, so there is no cross-arm value gate; the gate
that matters for the instrument is IDEMPOTENCE. Each kernel reads the frozen
step-top snapshot and writes `cons`, so rep n of a timed loop must produce
exactly what rep 1 did -- otherwise the reps are not measuring one operation.
`check_correctness` asserts that bitwise, per arm, and also that no arm
leaves subnormals behind in `cons` (they are slow on this hardware and would
bias every later rep).
"""

import numpy as np

import ember.fortran
import ember.solver
from ember import util

from residual_arms import build_case, swirl  # noqa: F401  (re-export)

RK_ARMS = ("rk", "rknoirs", "rkplain")

ENTRY = {
    "rk": "rk_mg_irs",
    "rknoirs": "rk_mg_noirs",
    "rkplain": "rk_plain",
}

# The solver defaults these arms are measured at (ember.solver.Solver), so the
# timing describes the configuration a real run marches in rather than one
# invented here. cfl matches the whole-step profile that motivated the module.
ALPHA = 1.0  # final RK stage: alpha = 1/(n_stage - k) with k = n_stage-1
CFL = 2.0
N_LEVELS = 3
FAC_MGRID = 0.2
EXPON_MGRID = 1.414
SF_IRS = 1.0


def seed_stage(b):
    """Populate what a stage reads: the snapshot, the residual and dt_vol.

    `build_case` already leaves dt_vol and (via update_sources) the fields the
    residual is built from; this evaluates the residual itself and freezes the
    step-top snapshot exactly as `rk_step` does before its first stage.
    """
    import ember.grid  # noqa: F401  (build_case returned a live grid)

    b.store[...] = b.conserved_nd
    return b


def build_kwargs(b):
    """Production's carve, taken from `advance_rk_stage_mg` itself.

    The multigrid scratch and the increment buffer are live at the same time,
    so they come from ONE carve -- see `ember.solver.mg_coarse_shapes`.
    """
    ni, nj, nk = b.shape
    rbuf, *mg_bufs = util.carve_view(
        b.scratch,
        (ni - 1, nj - 1, 5, 2),
        *ember.solver.mg_coarse_shapes(ni, nj, nk, N_LEVELS),
    )
    mg_scratch = dict(zip(ember.solver.MG_COARSE_NAMES, mg_bufs))
    common = dict(
        cons=b.conserved_nd,
        snapshot=b.store,
        residual=b.residual_nd,
        dt_vol=b.dt_vol_nd,
        alpha=ALPHA,
        cfl=CFL,
    )
    mg = dict(
        common,
        vol=b.vol_nd,
        fmgrid=FAC_MGRID,
        expon_mgrid=EXPON_MGRID,
        n_levels=N_LEVELS,
        rbuf=rbuf,
        **mg_scratch,
    )
    private = dict(
        rk=dict(mg, sf_irs=SF_IRS),
        # rk_mg_noirs still takes sf_irs in its signature (it passes the
        # no-op smoother to the shared engine), so the argument is present
        # and inert rather than absent.
        rknoirs=dict(mg, sf_irs=SF_IRS),
        # The plain path takes a full-volume increment instead of the rolling
        # pair, and none of the coarse scratch.
        rkplain=dict(
            common, tmp=util.carve_view(b.scratch, (ni - 1, nj - 1, nk - 1, 5))
        ),
    )
    return private


def callers_rk(b, active_arms=RK_ARMS):
    """One zero-argument callable per active arm, each writing into `cons`."""
    private = build_kwargs(b)
    out = {}
    for name in active_arms:
        fn = getattr(ember.fortran, ENTRY[name], None)
        if fn is None:
            continue
        out[name] = lambda fn=fn, kw=private[name]: fn(**kw)
    return out


def check_correctness(b, active_arms=RK_ARMS):
    """Gate the instrument: every arm must be idempotent and subnormal-free.

    Returns one entry per arm. `bitwise` False means repeated reps are timing
    different states and every number from this kernel is suspect.
    """
    results = {}
    for name, fn in callers_rk(b, active_arms).items():
        fn()
        first = np.array(b.conserved_nd, copy=True)
        fn()
        again = np.array(b.conserved_nd, copy=True)
        tiny = np.abs(first[first != 0])
        results[name] = dict(
            bitwise=bool(np.array_equal(first, again)),
            finite=bool(np.all(np.isfinite(first))),
            n_subnormal=int(np.count_nonzero(tiny < np.finfo(np.float32).tiny)),
        )
        b.conserved_nd[...] = b.store
    return results


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ncell", type=int, default=1_000_000)
    args = ap.parse_args()

    grid, b = build_case(args.ncell)
    swirl(b)
    grid.update_residual()
    seed_stage(b)
    arms = [a for a in RK_ARMS if getattr(ember.fortran, ENTRY[a], None)]
    print(f"grid {b.shape[0]} x {b.shape[1]} x {b.shape[2]}  ncell={args.ncell}")
    print(f"arms in this build: {arms}")
    print("\ninstrument gate (idempotence over a repeated call):")
    for name, r in check_correctness(b, arms).items():
        print(
            f"  {name:8s} idempotent={r['bitwise']}  finite={r['finite']}  "
            f"subnormals={r['n_subnormal']}"
        )


if __name__ == "__main__":
    main()
