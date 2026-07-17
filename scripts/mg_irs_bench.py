"""A/B benchmark for the coarse-grid IRS smoother in the RK multigrid path.

Background (docs/dev/viscous_kernels.md section 14): the coarse-level implicit
residual smoothing in ``rk_mg_irs``/``scree_mg_irs`` (``scree.f90``, via the
shared ``mg_coarse_correction`` engine) was handed the untiled
``smooth_residual_tri`` smoother, whose i-direction Thomas solve is a scalar
recurrence along the unit-stride axis (the link-stage ``-fopt-info-vec`` report
shows it as "no vectype"). The fine grid already uses the transpose-tiled
``smooth_residual_tri_tiled`` (identical maths, i-solve vectorised over BJ=8
lanes). This benchmark measures swapping the coarse path to the tiled variant.

It times, through the real production call path, two things:

1. **Full RK stage** (``solver.advance_rk_stage_mg``, which carves the coarse
   scratch via ``solver._mg_coarse_carve`` and dispatches the Fortran kernel):
     * ``irs``   -- ``sf_irs > 0`` -> ``rk_mg_irs`` (the CHANGED kernel);
     * ``noirs`` -- ``sf_irs = 0`` -> ``rk_mg_noirs`` (UNCHANGED by the swap:
       the co-measured cross-build drift gauge, per the doc's gauge method).
   Report ``irs`` deltas gauge-corrected against ``noirs`` at the same size/rep.

2. **Isolated coarse smoother** (within a single build, both f2py entries always
   exist): ``smooth_residual_tri`` vs ``smooth_residual_tri_tiled`` at the actual
   coarse-level shapes the MG restriction produces (level 1/2/3 = fine/2, /4, /8),
   so the i-solve tiling win is visible without the rest of the stage.

Protocol mirrors ``scripts/bench_viscous.py`` and viscous_kernels.md section 4:
one pinned core, ``OMP_NUM_THREADS=1``, warmup, round-robin interleave, median +
min ns/cell. A/B: ``make compile`` each side of a ``git stash`` of the scree.f90
swap; the ``noirs`` column is the drift gauge. Sizes are NODE dims; cell dims
(node-1) must be divisible by ``2**n_levels`` (=8 for n_levels=3) so multigrid
divides evenly (``solver._validate_mg``).

Run:  uv run python scripts/mg_irs_bench.py [--label L] [--csv F] [--reps N]
"""

import argparse
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.sched_setaffinity(0, {int(os.environ.get("BENCH_CPU", "2"))})

import numpy as np  # noqa: E402

import ember.fortran  # noqa: E402
import ember.grid  # noqa: E402
from ember import solver, util  # noqa: E402

# Reuse the real single-block grid fixture and the timing/report/CSV helpers
# from the viscous bench rather than duplicating them.
from bench_viscous import build_grid, report, time_variants  # noqa: E402

NP = 5
# Node dims; cell dims (node-1) divisible by 8 for n_levels=3.
SIZES = [(49, 33, 33), (65, 49, 49), (81, 65, 65), (97, 97, 97), (129, 97, 97)]
N_LEVELS = 3
CFL = 0.4
FAC_MGRID = 0.2
ALPHA = 1.0
SF_IRS = 0.5

# The swap (untiled vs tiled coarse smoother) is internal to rk_mg_irs and
# bitwise-invisible at runtime -- the f2py signature is identical on both sides
# -- so the A/B side cannot be auto-detected. Pass --label explicitly per build.


def seed_rk_fields(block, seed):
    """Populate the fields advance_rk_stage_mg consumes (store, dt_vol, residual).

    Geometry/vol come from build_grid; the RK stage reads only store, residual,
    dt_vol and vol and writes conserved -- no P/T dependence -- and its streaming
    loops have no data-dependent branches, so representative random fills give a
    faithful cost. Mirrors rk_step's ``block.store[...] = conserved_nd`` seed.
    """
    rng = np.random.default_rng(seed)
    block.store[...] = block.conserved_nd
    dt_vol = block.dt_vol_nd
    dt_vol.flags.writeable = True
    dt_vol[...] = (0.5 + rng.random(dt_vol.shape)).astype(np.float32)
    resid = block.residual_nd
    resid.flags.writeable = True
    resid[...] = rng.standard_normal(resid.shape).astype(np.float32)


def make_stage_call(grid, sf_irs):
    """Closure: one production advance_rk_stage_mg stage at the given sf_irs.

    sf_irs > 0 dispatches rk_mg_irs (the changed kernel); sf_irs == 0 dispatches
    rk_mg_noirs (the unchanged gauge). Both route through _mg_coarse_carve.
    """

    def call():
        solver.advance_rk_stage_mg(grid, ALPHA, CFL, FAC_MGRID, N_LEVELS, sf_irs)

    return call


def make_smoother_call(kernel, nci, ncj, nck, seed):
    """Closure: one coarse-shape IRS smooth on a fresh random dU each call.

    dU is rebuilt per call (cheap vs the solve) so repeated calls do not smooth
    an already-smoothed field into a degenerate one; the arithmetic cost per call
    is identical across reps. work sizes the Thomas coefficients only.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((nci, ncj, nck, NP)).astype(np.float32)
    dU = np.asfortranarray(base.copy())
    work = np.zeros(2 * (nci + ncj + nck), dtype=np.float32)

    def call():
        dU[...] = base
        kernel(du=dU, sf=SF_IRS, work=work, ni=nci + 1, nj=ncj + 1, nk=nck + 1)

    return call


def coarse_shapes(ni, nj, nk, n_levels):
    """(nci, ncj, nck) coarse cell dims for each MG level (fine/2, /4, ...)."""
    return [
        ((ni - 1) // (2**l), (nj - 1) // (2**l), (nk - 1) // (2**l))
        for l in range(1, n_levels + 1)
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="run", help="A/B side label (base|cand)")
    ap.add_argument("--csv", default=None, help="append results to this CSV")
    ap.add_argument("--reps", type=int, default=None, help="override rep count")
    args = ap.parse_args()

    print(f"# label={args.label}  n_levels={N_LEVELS}  sf_irs={SF_IRS}")
    print(f"{'label':10s} {'size':>11s} {'kernel':15s} {'variant':>8s} "
          f"{'med':>8s} {'min':>8s}  (ns/cell)")

    rows = []
    for size in SIZES:
        grid = build_grid(size)
        block = list(grid)[0]
        ni, nj, nk = block.shape
        cells = (ni - 1) * (nj - 1) * (nk - 1)
        seed_rk_fields(block, seed=abs(hash(size)) % (2**31))

        # --- Full RK stage: changed kernel (irs) vs unchanged gauge (noirs). ---
        stage_variants = {
            "irs": make_stage_call(grid, SF_IRS),
            "noirs": make_stage_call(grid, 0.0),
        }
        reps = args.reps or max(30, int(0.5e9 / (cells * 40)))
        report(rows, args.label, size, cells, "rk_stage",
               time_variants(stage_variants, reps))

        # --- Isolated coarse smoother at each MG level's shape. ---
        for lvl, (nci, ncj, nck) in enumerate(coarse_shapes(ni, nj, nk, N_LEVELS), 1):
            ccells = nci * ncj * nck
            sm_variants = {
                "untiled": make_smoother_call(
                    ember.fortran.smooth_residual_tri, nci, ncj, nck, seed=lvl),
                "tiled": make_smoother_call(
                    ember.fortran.smooth_residual_tri_tiled, nci, ncj, nck, seed=lvl),
            }
            sreps = args.reps or max(50, int(0.3e9 / (ccells * 30)))
            report(rows, args.label, (nci, ncj, nck), ccells,
                   f"smooth_l{lvl}", time_variants(sm_variants, sreps))

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a") as f:
            if new:
                f.write("label,size,kernel,variant,med_ns_per_cell,min_ns_per_cell\n")
            for row in rows:
                f.write(",".join(str(x) for x in row) + "\n")
        print(f"# appended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
