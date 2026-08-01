#!/usr/bin/env -S uv run
"""A/B the residual post-passes: production damp_residual vs the merged one.

Motivation (docs/dev/viscous_kernels.md section 21): `set_residual` itself is
at DRAM bandwidth and has ~101 B/cell of compulsory traffic, but the post-
passes that run straight afterwards in `Grid.update_residual` move MORE than
that on an array which never leaves the pipeline. `damp_residual` is the worst
offender: it loops the component index outside the (i,j,k) nest, so five
components become TEN full-volume sweeps over dU (five to reduce, five to
scale), each 20 B/cell, plus dt_vol read five times over.

`damp_residual_merged` collapses those to two sweeps. This measures that
directly, and also times the IRS smoother alongside for context (unchanged in
both arms -- it doubles as the in-process drift gauge, since a fused-damp
build should not move it).

Same protocol as tools/bench_residual_variants.py: both entry points live in
one .so, compared round-robin interleaved in one process, serial and 100-rank
saturated on sapphire. Correctness is gated before any timing.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

import ember.fortran
from ember import util
from ember.cases import build_duct_grid

# dampin=2 is the low end of multall's recommended 2..100, so the soft-clip is
# actually active rather than a no-op; sf=1.0 matches the duct CFL sweeps.
DAMPIN = 2.0
SF = 1.0


def build_case(ncell):
    grid = build_duct_grid(ncell)
    grid.update_cached_conserved()
    grid.apply_bconds()
    grid.update_sources(False, 0.0)
    grid.update_timestep(rf=1.0)
    grid.update_residual()
    return grid, grid[0]


def snapshot(b):
    """A pristine copy of dU to restore before each timed call.

    Both damp variants mutate dU in place and the soft-clip is not
    idempotent, so without restoring, later reps would operate on
    progressively damped data and measure something else.
    """
    return np.array(b.residual_nd, copy=True)


def check_correctness(b, ref_du):
    """Compare merged damp against production damp on identical input."""
    ni, nj, nk = b.shape
    du = b.residual_nd
    du.flags.writeable = True

    du[...] = ref_du
    ember.fortran.damp_residual(
        du=du, dt_vol=b.dt_vol_nd, dampin=DAMPIN, ni=ni, nj=nj, nk=nk
    )
    prod = np.array(du, copy=True)

    du[...] = ref_du
    ember.fortran.damp_residual_merged(
        du=du, dt_vol=b.dt_vol_nd, dampin=DAMPIN, ni=ni, nj=nj, nk=nk
    )
    merged = np.array(du, copy=True)

    scale = float(np.abs(prod).max())
    diff = float(np.abs(merged - prod).max())
    bitwise = bool(np.array_equal(merged, prod))
    print(f"  damp: |dU|_max = {scale:.6e}")
    print(
        f"  merged vs prod: max_abs_diff = {diff:.6e} "
        f"({diff / scale if scale else 0.0:.3e} of scale)  bitwise={bitwise}"
    )
    return dict(bitwise=bitwise, max_abs_diff=diff, rel=diff / scale if scale else 0.0)


def time_variants(b, ref_du, reps, warmup):
    ni, nj, nk = b.shape
    ncell = (ni - 1) * (nj - 1) * (nk - 1)
    du = b.residual_nd
    du.flags.writeable = True
    nwork = 2 * ((ni - 1) + (nj - 1) + (nk - 1))
    work = util.carve_view(b.scratch, (nwork,))

    def call_prod():
        ember.fortran.damp_residual(
            du=du, dt_vol=b.dt_vol_nd, dampin=DAMPIN, ni=ni, nj=nj, nk=nk
        )

    def call_merged():
        ember.fortran.damp_residual_merged(
            du=du, dt_vol=b.dt_vol_nd, dampin=DAMPIN, ni=ni, nj=nj, nk=nk
        )

    def call_irs():
        ember.fortran.smooth_residual_tri_tiled(
            du=du, sf=SF, work=work, ni=ni, nj=nj, nk=nk
        )

    variants = {"damp_prod": call_prod, "damp_merged": call_merged, "irs_gauge": call_irs}

    for _ in range(warmup):
        for fn in variants.values():
            du[...] = ref_du
            fn()

    samples = {n: [] for n in variants}
    for _ in range(reps):
        for name, fn in variants.items():
            du[...] = ref_du  # untimed restore
            t0 = time.perf_counter()
            fn()
            samples[name].append((time.perf_counter() - t0) / ncell * 1e9)

    return {
        n: dict(median=statistics.median(s), min=min(s)) for n, s in samples.items()
    }


def wait_until(t):
    while True:
        rem = t - time.time()
        if rem <= 0:
            return
        time.sleep(min(rem, 0.05))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["serial", "saturated"], default="serial")
    ap.add_argument("--ncell", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--start-delay", type=float, default=300.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    if not hasattr(ember.fortran, "damp_residual_merged"):
        print("damp_residual_merged not in this build", file=sys.stderr)
        return 1

    grid, b = build_case(args.ncell)
    ref_du = snapshot(b)

    check = {}
    if rank == 0:
        print(f"grid {b.ni} x {b.nj} x {b.nk}  dampin={DAMPIN} sf={SF}")
        print("\ncorrectness gate:")
        check = check_correctness(b, ref_du)

    if args.mode == "saturated":
        start = float(os.environ.get("EMBER_BENCH_START", "0")) or (
            time.time() + args.start_delay
        )
        wait_until(start)

    res = time_variants(b, ref_du, args.reps, args.warmup)
    base = res["damp_prod"]["median"]
    for n, r in res.items():
        line = f"rank {rank:3d}  {n:12s} median {r['median']:7.3f} ns/cell  min {r['min']:7.3f}"
        if n == "damp_merged":
            line += f"   vs damp_prod: {(r['median'] / base - 1) * 100:+.1f}%"
        print(line, flush=True)

    if args.json:
        with open(args.json, "a") as fh:
            fh.write(
                json.dumps(
                    dict(rank=rank, mode=args.mode, results=res, correctness=check)
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
