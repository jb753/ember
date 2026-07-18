"""A/B benchmark for the RK multigrid stage (docs/dev/viscous_kernels.md 14-15).

Times the whole per-block RK multigrid stage through the real production call
path ``solver.advance_rk_stage_mg`` (which carves the coarse scratch via
``solver._mg_coarse_carve`` and dispatches the Fortran kernels), in three
variants:

  * ``irs``   -- ``sf_irs > 0``  -> ``rk_mg_irs``   (coarse IRS smoothing on);
  * ``noirs`` -- ``sf_irs = 0``  -> ``rk_mg_noirs`` (smoothing off);
  * ``plain`` -- ``fac_mgrid = 0`` -> ``rk_plain``  (multigrid off).

For an A/B of a change that touches the multigrid kernels, co-measure whichever
variant the change leaves *unchanged* as the cross-build drift gauge and report
deltas gauge-corrected against it (e.g. the section-15 scatter fusion touched
both irs and noirs, so ``plain`` is the gauge).

Protocol mirrors ``scripts/bench_viscous.py`` and viscous_kernels.md section 4:
one pinned core, ``OMP_NUM_THREADS=1``, warmup, round-robin interleave, median +
min ns/cell. A/B: ``make compile`` each side of the change. Sizes are NODE dims;
cell dims (node-1) must be divisible by ``2**n_levels`` (=8 for n_levels=3) so
multigrid divides evenly (``solver._validate_mg``).

Run:  uv run python scripts/mg_irs_bench.py [--label L] [--csv F] [--reps N]
"""

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.sched_setaffinity(0, {int(os.environ.get("BENCH_CPU", "2"))})

import numpy as np  # noqa: E402

from ember import solver  # noqa: E402

# Reuse the real single-block grid fixture and the timing/report/CSV helpers
# from the viscous bench rather than duplicating them.
from bench_viscous import build_grid, report, time_variants  # noqa: E402

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


def make_stage_call(grid, sf_irs, fac_mgrid=FAC_MGRID):
    """Closure: one production advance_rk_stage_mg stage.

    fac_mgrid > 0: sf_irs > 0 dispatches rk_mg_irs, sf_irs == 0 dispatches
    rk_mg_noirs (both use the fused final-hop+scatter -- the changed kernels).
    fac_mgrid == 0 collapses to rk_plain (multigrid off), which is *unchanged*
    by the fusion and serves as the co-measured cross-build drift gauge.
    """

    def call():
        solver.advance_rk_stage_mg(grid, ALPHA, CFL, fac_mgrid, N_LEVELS, sf_irs)

    return call


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

        # --- Full RK stage. irs + noirs are the fused (changed) kernels; plain
        # (fac_mgrid=0 -> rk_plain) is unchanged by the fusion and is the
        # co-measured cross-build drift gauge for this A/B. ---
        stage_variants = {
            "irs": make_stage_call(grid, SF_IRS),
            "noirs": make_stage_call(grid, 0.0),
            "plain": make_stage_call(grid, 0.0, fac_mgrid=0.0),
        }
        reps = args.reps or max(30, int(0.5e9 / (cells * 40)))
        report(rows, args.label, size, cells, "rk_stage",
               time_variants(stage_variants, reps))

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
