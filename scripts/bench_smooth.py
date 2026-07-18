"""Benchmark the constant-coefficient smoother ``smooth3d_const``.

A/B protocol per docs/dev/viscous_kernels.md section 4:

  * build via the real path (make compile) before running; never benchmark a
    standalone compile;
  * measure through the real Python call path (f2py marshaling included);
  * single-thread, pinned to one core, warmed up, median (and min) ns/cell over
    many reps;
  * sweep block sizes across the cache boundary and report ns/cell per size.

Signature-agnostic: it detects whether the kernel takes the rolling-buffer
``xs(ni,nj,kr)`` (the fused candidate) or the full ``xs(ni,nj,nk)`` (the
all-at-once baseline) from the f2py docstring, so the same script runs on both
sides of a git-stash A/B.

Usage:

    uv run python scripts/bench_smooth.py --label cand --csv cand.csv

Pinning is done in-process via sched_setaffinity (default core 2, override with
BENCH_CPU=<n>). OMP_NUM_THREADS is forced to 1 before numpy loads.
"""

import argparse
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.sched_setaffinity(0, {int(os.environ.get("BENCH_CPU", "2"))})

import numpy as np  # noqa: E402

import ember.fortran  # noqa: E402

# The fused kernel takes a rolling buffer with a ``kr`` plane count; the baseline
# takes a full (ni,nj,nk) work array. Detected from the f2py signature.
ROLLING = "kr" in ember.fortran.smooth3d_const.__doc__

SIZES = [(48, 32, 32), (64, 48, 48), (80, 64, 64), (96, 96, 96), (128, 96, 96)]
NP = 5  # conserved variables
SF4, SF2 = 0.02, 0.01


def make_work(ni, nj, nk):
    """Allocate the kernel's work array, sized for whichever signature is built."""
    kr = min(6, nk) if ROLLING else nk
    return np.zeros((ni, nj, kr), order="F", dtype=np.float32)


def bench_one(shape, reps=25, warmup=5):
    ni, nj, nk = shape
    rng = np.random.default_rng(0)
    x0 = np.asfortranarray(rng.random((ni, nj, nk, NP)), dtype=np.float32)
    x = np.empty_like(x0)
    xs = make_work(ni, nj, nk)
    ncell = ni * nj * nk * NP

    def call():
        ember.fortran.smooth3d_const(
            x=x, sf4=np.float32(SF4), sf2=np.float32(SF2), xs=xs
        )

    for _ in range(warmup):
        x[...] = x0
        call()

    times = []
    for _ in range(reps):
        x[...] = x0  # re-init outside timing (kernel smooths in place)
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)

    ns_med = statistics.median(times) / ncell * 1e9
    ns_min = min(times) / ncell * 1e9
    return ns_med, ns_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    kind = "rolling" if ROLLING else "baseline"
    print(f"# smooth3d_const bench ({kind}), label={args.label}, "
          f"core={os.sched_getaffinity(0)}")
    print(f"{'size':>12} {'ns/cell(med)':>13} {'ns/cell(min)':>13}")
    rows = []
    for shape in SIZES:
        ns_med, ns_min = bench_one(shape)
        tag = "x".join(map(str, shape))
        print(f"{tag:>12} {ns_med:13.3f} {ns_min:13.3f}")
        rows.append((args.label, kind, tag, ns_med, ns_min))

    if args.csv:
        import csv

        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["label", "kind", "size", "ns_med", "ns_min"])
            w.writerows(rows)
        print(f"# wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
