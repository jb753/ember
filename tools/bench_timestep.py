#!/usr/bin/env -S uv run
"""End-to-end timestep benchmark: does a kernel win survive the whole solver?

Compiler flags are GLOBAL -- they change viscous, IRS, multigrid and the
boundary conditions as well as set_residual. A flag adopted on kernel evidence
alone could be speeding the residual and slowing everything else, and the
kernel benchmark would never see it. This times `Solver.run`, so the answer is
in the currency that matters.

Replication is at the LAUNCH, as everywhere else in this study: a fresh
process re-draws page placement, allocation alignment and thermal state, and
those dominate the within-launch scatter (see bench_rep_convergence.py).

Reported as ns per cell per step, so it is directly comparable with the
kernel numbers from bench_prod_baseline.py.

Usage:
    taskset -c 0 uv run python tools/bench_timestep.py --launches 5 --steps 20
    uv run python tools/bench_timestep.py --analyze tools/bench_timestep.jsonl
"""

import argparse
import json
import statistics
import sys
import time


def one_launch(ncell, steps, dampin, sf_resid):
    from ember.cases import build_duct_grid
    from ember.solver import Solver

    grid = build_duct_grid(ncell)
    b = grid[0]
    ni, nj, nk = b.shape
    ncells = (ni - 1) * (nj - 1) * (nk - 1)

    # One short run to warm up (first-touch, any lazy caches), then the timed one.
    Solver(n_step=2, dampin=dampin, sf_resid=sf_resid, n_step_log=10**9).run(grid)

    grid = build_duct_grid(ncell)
    solver = Solver(n_step=steps, dampin=dampin, sf_resid=sf_resid, n_step_log=10**9)
    t0 = time.perf_counter()
    solver.run(grid)
    dt = time.perf_counter() - t0
    return dt / steps / ncells * 1e9, (ni, nj, nk), ncells


def analyze(path):
    rows = [json.loads(l) for l in open(path)]
    for label in sorted({r["label"] for r in rows}):
        v = sorted(r["ns_per_cell_step"] for r in rows if r["label"] == label)
        m = statistics.median(v)
        half = (max(v) - min(v)) / 2 / m * 100
        print(f"  {label:<24} {m:8.2f} ns/cell/step   half-range {half:5.2f}%   "
              f"({len(v)} launches)")
    labels = sorted({r["label"] for r in rows})
    if len(labels) == 2:
        a, b = (statistics.median([r["ns_per_cell_step"] for r in rows
                                   if r["label"] == L]) for L in labels)
        print(f"\n  {labels[1]} vs {labels[0]}: {(b / a - 1) * 100:+.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ncell", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--launches", type=int, default=5)
    ap.add_argument("--dampin", type=float, default=25.0)
    ap.add_argument("--sf-resid", type=float, default=0.5)
    ap.add_argument("--label", default="build")
    ap.add_argument("--json", default="tools/bench_timestep.jsonl")
    ap.add_argument("--analyze", default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return 0

    for L in range(args.launches):
        v, shape, ncells = one_launch(args.ncell, args.steps, args.dampin,
                                      args.sf_resid)
        print(f"  launch {L}  {v:8.2f} ns/cell/step", flush=True)
        with open(args.json, "a") as fh:
            fh.write(json.dumps(dict(label=args.label, launch=L, shape=list(shape),
                                     ncell=ncells, steps=args.steps,
                                     ns_per_cell_step=v)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
