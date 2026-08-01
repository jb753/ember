#!/usr/bin/env -S uv run
"""Sweep the IRS i-solve's transpose-tile width BJ.

Per-direction timing (docs section 23) shows the i-solve is 68% of the IRS
smoother's cost -- 9.0 of 13.5 ns/cell, more than the j- and k-solves
combined. Its transpose gather/scatter reads dU with stride nci (~1 KB), so
it touches a new cache line per element; BJ sets how much solve work
amortises that traffic.

Production fixes BJ = 8, commented "AVX = 8 float32 lanes" -- correct for
the AVX2 machine the doc's earlier sections used, but this is Sapphire
Rapids (AVX-512, 16 float32 lanes), so BJ = 8 fills half a zmm register.

BJ only groups independent j-lines, so every width must be bitwise
identical; that is asserted before any timing.
"""
import argparse, json, os, statistics, sys, time
import numpy as np
import ember.fortran as F
from ember import util
from ember.cases import build_duct_grid

VARIANTS = {
    "BJ=8(prod)": "smooth_residual_tri_tiled",
    "BJ=16": "smooth_residual_tri_bj16",
    "BJ=32": "smooth_residual_tri_bj32",
    "BJ=64": "smooth_residual_tri_bj64",
    "BJ=128": "smooth_residual_tri_bj128",
}
SF = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["serial", "saturated"], default="serial")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--start-delay", type=float, default=300.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", "0"))

    g = build_duct_grid(1_000_000)
    g.update_cached_conserved(); g.apply_bconds()
    g.update_sources(False, 0.0); g.update_timestep(rf=1.0); g.update_residual()
    b = g[0]; ni, nj, nk = b.shape
    N = (ni - 1) * (nj - 1) * (nk - 1)
    du = b.residual_nd; du.flags.writeable = True
    ref = np.array(du, copy=True)
    work = util.carve_view(b.scratch, (2 * ((ni - 1) + (nj - 1) + (nk - 1)),))
    names = [n for n, e in VARIANTS.items() if hasattr(F, e)]
    calls = {n: getattr(F, VARIANTS[n]) for n in names}

    if rank == 0:
        print(f"grid {ni} x {nj} x {nk}  sf={SF}\ncorrectness (must be bitwise):")
        base = None
        for n in names:
            du[...] = ref
            calls[n](du=du, sf=SF, work=work, ni=ni, nj=nj, nk=nk)
            o = np.array(du, copy=True)
            if base is None:
                base = o; print(f"  {n:12s} (reference)"); continue
            print(f"  {n:12s} bitwise={np.array_equal(o, base)}  "
                  f"maxdiff={np.abs(o - base).max():.3e}")

    if a.mode == "saturated":
        t0 = float(os.environ.get("EMBER_BENCH_START", "0")) or (time.time() + a.start_delay)
        while time.time() < t0:
            time.sleep(min(t0 - time.time(), 0.05))

    for _ in range(4):
        for n in names:
            du[...] = ref; calls[n](du=du, sf=SF, work=work, ni=ni, nj=nj, nk=nk)
    s = {n: [] for n in names}
    for _ in range(a.reps):
        for n in names:
            du[...] = ref
            t = time.perf_counter()
            calls[n](du=du, sf=SF, work=work, ni=ni, nj=nj, nk=nk)
            s[n].append((time.perf_counter() - t) / N * 1e9)
    res = {n: dict(median=statistics.median(v), min=min(v)) for n, v in s.items()}
    p = res[names[0]]["median"]
    for n in names:
        print(f"rank {rank:3d}  {n:12s} median {res[n]['median']:7.3f}  "
              f"min {res[n]['min']:7.3f}   {100*(res[n]['median']/p-1):+6.1f}%", flush=True)
    if a.json:
        with open(a.json, "a") as fh:
            fh.write(json.dumps(dict(rank=rank, mode=a.mode, results=res)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
