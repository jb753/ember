#!/usr/bin/env -S uv run
"""How many reps does one launch need to pin down a kernel time to +/-1%?

The A/B harness reports the median of 50 reps and treats it as the answer.
That is an assumption, never tested: 50 reps could be far more than needed
(wasted time) or far fewer (a median that still moves when you re-run it).
This measures it directly.

Method. One process, one arm, one grid. Time N_TOTAL reps and keep EVERY
sample rather than a summary. Then ask, offline: if I had run only n reps,
how much would my answer have moved?

That question is answered with a CONTIGUOUS-BLOCK bootstrap -- windows of n
consecutive reps drawn from the trace -- not an i.i.d. resample. Consecutive
reps are correlated (frequency drift, thermal ramp, background noise arrives
in bursts), so shuffling would break exactly the structure that makes short
runs unreliable and would understate the n required. A contiguous window of
length n is the faithful analogue of "run n reps and take the median".

Reported per n: the central 95% interval of the window estimator, as a
percentage of the full-trace median. The smallest n whose interval fits
inside +/-1% is the answer.

Both estimators the harness offers are scored:
  median -- what the harness reports
  min    -- often preferred for kernel timing, since interference only ever
            adds time; here it is checked rather than assumed.

NOTE this bounds WITHIN-launch precision only. It says nothing about
launch-to-launch variation, which is a separate and larger effect (physical
page placement, allocation alignment, thermal state at start) and needs
repeat launches to measure -- see docs/dev/plan_nodal_primitives.md.

Usage:
    taskset -c 0 uv run python bench/bench_rep_convergence.py \
        --ncell 1000000 --reps 2000 --arm prod --out bench/results/bench_repconv.npz
    uv run python bench/bench_rep_convergence.py --analyze bench/results/bench_repconv.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def collect(ncell, arm, reps, warmup, flush):
    """Time `reps` calls of one arm, keeping every sample."""
    from residual_arms import DAMPIN, build_case, callers, flush_llc

    grid, b = build_case(ncell)
    du = b.residual_nd
    du.flags.writeable = True
    fn = callers(b, du, DAMPIN)[arm]

    for _ in range(warmup):
        if flush:
            flush_llc()
        fn()

    ni, nj, nk = b.shape
    ncell_actual = (ni - 1) * (nj - 1) * (nk - 1)
    samples = np.empty(reps)
    for i in range(reps):
        if flush:
            flush_llc()  # untimed
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) / ncell_actual * 1e9
    return samples, (ni, nj, nk), ncell_actual


def block_bootstrap(samples, n, estimator, n_windows=4000, rng=None):
    """Spread of `estimator` over contiguous windows of length n."""
    rng = rng or np.random.default_rng(0)
    hi = len(samples) - n
    if hi <= 0:
        return None
    starts = rng.integers(0, hi + 1, size=min(n_windows, hi + 1))
    win = np.lib.stride_tricks.sliding_window_view(samples, n)[starts]
    return estimator(win, axis=1)


def analyze(path):
    d = np.load(path)
    s = d["samples"]
    full = float(np.median(s))
    print(f"grid {tuple(d['shape'])}  ncell={int(d['ncell'])}  arm={str(d['arm'])}  "
          f"flush={bool(d['flush'])}  reps={len(s)}")
    print(f"\nper-rep distribution (ns/cell):")
    print(f"  median {full:8.3f}   mean {s.mean():8.3f}   std {s.std():7.3f} "
          f"({s.std() / full * 100:5.2f}% of median)")
    for q in (0, 1, 5, 50, 95, 99, 100):
        print(f"  p{q:<3d}  {np.percentile(s, q):8.3f}  "
              f"({(np.percentile(s, q) / full - 1) * 100:+6.2f}%)")

    # Drift: is the trace stationary? Quartile means + a straight-line fit.
    q = np.array_split(s, 4)
    print("\ndrift across the run (quartile medians, % vs full median):")
    print("  " + "  ".join(f"Q{i + 1} {(np.median(x) / full - 1) * 100:+.2f}%"
                           for i, x in enumerate(q)))
    slope = np.polyfit(np.arange(len(s)), s, 1)[0]
    print(f"  linear slope {slope * len(s) / full * 100:+.2f}% over the whole trace")

    print("\ncontiguous-window bootstrap: central 95% interval of the estimator,"
          "\nas % of the full-trace median (want the half-width <= 1%)\n")
    print(f"{'n reps':>7} | {'median: 2.5%':>12} {'97.5%':>8} {'half-width':>11} "
          f"| {'min: 2.5%':>10} {'97.5%':>8} {'half-width':>11}")
    print("-" * 78)
    answer = {}
    for n in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000):
        row = [f"{n:>7} |"]
        for name, est in (("median", np.median), ("min", np.min)):
            v = block_bootstrap(s, n, est)
            if v is None:
                row.append(f"{'--':>12} {'--':>8} {'--':>11} |")
                continue
            lo, hi = np.percentile(v, [2.5, 97.5])
            lo_p, hi_p = (lo / full - 1) * 100, (hi / full - 1) * 100
            half = (hi_p - lo_p) / 2
            row.append(f"{lo_p:>+11.2f}% {hi_p:>+7.2f}% {half:>10.2f}%" +
                       (" |" if name == "median" else ""))
            if name not in answer and half <= 1.0:
                answer[name] = n
        print(" ".join(row))

    print()
    for name in ("median", "min"):
        if name in answer:
            print(f"  {name:>6}: {answer[name]} reps suffice for +/-1% within one launch")
        else:
            print(f"  {name:>6}: NOT within +/-1% even at the longest window tested")

    # Plot to PDF (never raster).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(s, lw=0.4)
    ax[0].axhline(full, color="k", ls="--", lw=0.8, label="median")
    ax[0].set_xlabel("rep index")
    ax[0].set_ylabel("ns/cell")
    ax[0].set_title("per-rep trace")
    ax[0].legend()
    for name, est in (("median", np.median), ("min", np.min)):
        half = []
        for n in ns:
            v = block_bootstrap(s, n, est)
            if v is None:
                half.append(np.nan)
                continue
            lo, hi = np.percentile(v, [2.5, 97.5])
            half.append((hi - lo) / 2 / full * 100)
        ax[1].plot(ns, half, marker="o", label=name)
    ax[1].axhline(1.0, color="r", ls="--", lw=0.8, label="1% target")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("reps per launch")
    ax[1].set_ylabel("95% half-width (% of median)")
    ax[1].set_title("estimator precision vs rep count")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    out = str(path).replace(".npz", ".pdf")
    fig.savefig(out)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ncell", type=int, default=1_000_000)
    ap.add_argument("--arm", default="prod")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--out", default="bench/results/bench_repconv.npz")
    ap.add_argument("--analyze", default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return 0

    flush = not args.no_flush
    samples, shape, ncell = collect(args.ncell, args.arm, args.reps, args.warmup, flush)
    np.savez(args.out, samples=samples, shape=np.array(shape), ncell=ncell,
             arm=args.arm, flush=flush)
    print(f"wrote {args.out}  ({len(samples)} reps, median "
          f"{np.median(samples):.3f} ns/cell)")
    analyze(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
