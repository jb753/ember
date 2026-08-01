#!/usr/bin/env -S uv run
"""A/B the production set_residual against an idiomatic rewrite.

Both kernels are compiled into the *same* .so (setup.py globs
src/ember/_fortran/*.f90, so residual_cand.f90's `set_residual_clean` sits
alongside residual.f90's `set_residual` -- the pattern residual_old_bench.f90
already established). That means the two are compared **within one process,
round-robin interleaved**, so there is no cross-build LTO drift and none of the
gauge-kernel correction every A/B in docs/dev/viscous_kernels.md sections 7-15
had to apply. It also means a surprising result can be re-checked against the
exact binary that produced it.

The question being answered is not "which is faster" but "what does idiomatic
Fortran cost on this kernel under current ifort" -- the clean variant losing is
a perfectly good result, and prices the hand-optimisation.

Two regimes, both on build_duct_grid(1_000_000) (ni=273, nj=65, nk=57):

  serial     one process, the diagnostic number. Pair with `perf stat` to
             settle the compute-vs-memory-bound question section 12.2 raised
             and never answered.
  saturated  one rank of many (default 100) launched by srun, each holding its
             own 1M-cell grid, all timing the same window. This is the
             production-representative regime -- ember runs 100+ ranks per
             sapphire node, where DRAM bandwidth is contended -- and it has
             never been measured for this kernel.

In saturated mode ranks rendezvous on a wall-clock start time rather than an
MPI barrier: the ranks are independent processes with no communicator, and a
"spin until T" gate is enough to overlap the timed windows without adding an
mpi4py dependency. Each rank prints its own row; aggregate across ranks with
--collect.

Note: at ni=273, ni*nj = 17745 is not a multiple of 1024, so set_residual's
conditional anti-aliasing pad (njp = nj+1) is INACTIVE here. Draw no
conclusions about the pad from these runs.
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

# Variants to compare. Name -> f2py entry point. Production first; every entry
# must accept the same kwargs (see build_kwargs) apart from the per-variant
# overlay below.
VARIANTS = {
    "prod": "set_residual",
    "clean": "set_residual_clean",
    "tiled": "set_residual_tiled",
    "naive": "set_residual_naive",
    "consa": "set_residual_consa",
}

# Tile dimensions swept for the tiled variant, as (IB, JB, KB) cells. Sized
# so the per-tile working set (9 node quantities + two partial averages)
# stays inside a 2 MB L2; see docs section 18.
TILE_SIZES = [(96, 16, 8), (64, 16, 8), (48, 24, 8), (32, 32, 8), (128, 16, 4)]
DEFAULT_TILE = (96, 16, 8)

# Per-variant kwargs the others do not take. Production still carries the
# inert `kb` dummy (its slab loop is a pure re-nesting of `do k = 1, nk-1`);
# the clean rewrite drops it, so it must not be passed one.
EXTRA_KWARGS = {
    "prod": ("kb", "planes", "rows", "njp"),
    "clean": ("planes", "rows", "njp"),
    "tiled": ("qn", "ai", "t1", "fa", "fl", "wf", "ib", "jb", "kbt"),
    "naive": ("flow_i", "flow_j", "flow_k"),
    # consa is production's driver verbatim, so it takes the same
    # rolling buffers and the same (inert) kb/njp dummies.
    "consa": ("planes", "rows", "njp", "kb"),
}


def build_case(ncell):
    """Build the duct grid and drive it to the state set_residual consumes."""
    grid = build_duct_grid(ncell)
    grid.update_cached_conserved()
    grid.apply_bconds()
    grid.update_sources(False, 0.0)  # seeds F_body, tau_q_halo, mu_turb
    grid.update_timestep(rf=1.0)
    return grid, grid[0]


def build_kwargs(b, du, tile=DEFAULT_TILE):
    """Superset of kwargs across all variants, writing into the caller's dU.

    This is a superset, not a common set: each variant's private arguments
    (production's `kb`/`planes`/`rows`, the tiled kernel's tile scratch) are
    stripped for the others by variant_kwargs(). Building one dict and
    subtracting keeps the call sites from drifting apart.

    All scratch is carved zero-copy from block.tau_q_halo, exactly as the real
    caller does. The tiled kernel's six flat buffers are sized for the largest
    tile; the production/clean rolling buffers are carved after them so the two
    sets never alias within one call.
    """
    ni, nj, nk = b.shape
    i_cusp_start, i_cusp_end = b.i_cusp
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
    IB, JB, KB = tile
    ntile = (IB + 1) * (JB + 1) * (KB + 1)
    planes, rows, qn, ai, t1, fa, fl, wf = util.carve_view(
        b.tau_q_halo,
        (ni, njp, 5, 2),
        (ni, 5, 3),
        (ntile * 9,),
        (ntile * 9,),
        (ntile * 9,),
        (ntile * 9,),
        (ntile * 5,),
        (ntile,),
    )
    # The naive kernel's three full-volume face-flow arrays are ~56 MB at 1M
    # cells -- more than tau_q_halo holds alone, so flow_i/flow_j come from
    # tau_q_halo (carved from its start, aliasing the tile scratch above,
    # which is fine: no variant uses both) and flow_k from block.scratch.
    flow_i, flow_j = util.carve_view(
        b.tau_q_halo, (ni, nj - 1, nk - 1, 5), (ni - 1, nj, nk - 1, 5)
    )
    flow_k = util.carve_view(b.scratch, (ni - 1, nj - 1, nk, 5))
    return dict(
        flow_i=flow_i,
        flow_j=flow_j,
        flow_k=flow_k,
        qn=qn,
        ai=ai,
        t1=t1,
        fa=fa,
        fl=fl,
        wf=wf,
        ib=IB,
        jb=JB,
        kbt=KB,
        cons=b.conserved_nd,
        p=b.P_nd,
        p_offset=b.P_offset_nd,
        r=b.r_nd,
        omega=b.Omega_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        du=du,
        f_body=b.F_body_nd,
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_nd,
        ho=b.ho_nd,
        planes=planes,
        rows=rows,
        **b.ijk_wall_conv,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
        # PRODUCTION ONLY -- stripped for every other variant. kb is inert
        # even for production (its slab loop is a pure re-nesting of
        # `do k = 1, nk-1`), so the value is immaterial; the clean rewrite
        # drops the dummy entirely and must not be passed one.
        kb=nk - 1,
        njp=njp,
        ni=ni,
        nj=nj,
        nk=nk,
    )


def variant_kwargs(name, kwargs):
    """Strip the kwargs a given variant's signature does not accept."""
    keep_extra = EXTRA_KWARGS.get(name, ())
    drop = {k for extras in EXTRA_KWARGS.values() for k in extras} - set(keep_extra)
    return {k: v for k, v in kwargs.items() if k not in drop}


def available_variants(sweep_tiles=False):
    """Labels to benchmark, in order, restricted to what this build exposes.

    With sweep_tiles, the tiled kernel contributes one label per tile geometry
    ("tiled@96x16x8", ...) instead of a single default-tile entry.
    """
    labels = []
    for n, entry in VARIANTS.items():
        if not hasattr(ember.fortran, entry):
            continue
        if n == "tiled" and sweep_tiles:
            labels += [f"tiled@{a}x{b}x{c}" for a, b, c in TILE_SIZES]
        else:
            labels.append(n)
    return labels


def check_correctness(b, names):
    """Compare every variant's dU against production's, before timing anything.

    Each variant writes into its own zeroed scratch dU, so this isolates the
    kernel's own output. Reports max-abs-diff both raw and against the field
    scale, plus exact equality -- printed, not just asserted, because the
    interesting cases (sub-ulp reassociation vs. a real bug) differ by orders
    of magnitude and the number is what tells them apart.
    """
    ref = None
    results = {}
    for name in names:
        base, _, geom = name.partition("@")
        tile = tuple(int(v) for v in geom.split("x")) if geom else DEFAULT_TILE
        du = np.zeros_like(b.residual_nd)
        kwargs = variant_kwargs(base, build_kwargs(b, du, tile))
        getattr(ember.fortran, VARIANTS[base])(**kwargs)
        if ref is None:
            ref = du
            scale = float(np.abs(ref).max())
            print(f"  {name:8s}  (reference)  |dU|_max = {scale:.6e}")
            results[name] = dict(bitwise=True, max_abs_diff=0.0, rel=0.0)
            continue
        diff = float(np.abs(du - ref).max())
        bitwise = bool(np.array_equal(du, ref))
        rel = diff / scale if scale > 0 else 0.0
        print(
            f"  {name:8s}  max_abs_diff = {diff:.6e}  "
            f"({rel:.3e} of scale)  bitwise={bitwise}"
        )
        results[name] = dict(bitwise=bitwise, max_abs_diff=diff, rel=rel)
    return results


def time_variants(b, names, reps, warmup):
    """Round-robin interleaved timing: A,B,A,B,... not A*n then B*n.

    Interleaving is the point of the same-.so protocol -- it cancels drift
    (thermal, frequency, interference) that would otherwise land entirely on
    whichever variant ran second.
    """
    ncell = (b.ni - 1) * (b.nj - 1) * (b.nk - 1)
    du = b.residual_nd
    du.flags.writeable = True
    # Each label gets its own kwargs. "tiled@IBxJBxKB" labels carry their own
    # tile geometry (and hence their own scratch carve); everything else uses
    # the default tile, which it ignores anyway.
    calls = {}
    kwargs = {}
    for n in names:
        base, _, geom = n.partition("@")
        tile = tuple(int(v) for v in geom.split("x")) if geom else DEFAULT_TILE
        kwargs[n] = variant_kwargs(base, build_kwargs(b, du, tile))
        calls[n] = getattr(ember.fortran, VARIANTS[base])

    for _ in range(warmup):
        for name in names:
            calls[name](**kwargs[name])

    samples = {n: [] for n in names}
    for _ in range(reps):
        for name in names:
            t0 = time.perf_counter()
            calls[name](**kwargs[name])
            samples[name].append((time.perf_counter() - t0) / ncell * 1e9)

    return {
        n: dict(
            median=statistics.median(s),
            min=min(s),
            ncell=ncell,
        )
        for n, s in samples.items()
    }


def wait_until(start_epoch):
    """Spin-then-sleep until the shared start time, so ranks overlap."""
    while True:
        remaining = start_epoch - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["serial", "saturated"], default="serial")
    ap.add_argument(
        "--sweep-tiles",
        action="store_true",
        help="benchmark the tiled kernel at every geometry in TILE_SIZES",
    )
    ap.add_argument("--ncell", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--start-delay",
        type=float,
        default=120.0,
        help="saturated mode: seconds allowed for every rank to finish "
        "building its grid before the shared timed window opens",
    )
    ap.add_argument(
        "--json",
        default=None,
        help="append one JSON result line per rank to this file",
    )
    args = ap.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    nranks = int(os.environ.get("SLURM_NTASKS", "1"))

    names = available_variants(args.sweep_tiles)
    if rank == 0:
        missing = {v for v in VARIANTS
                   if not any(n.split("@")[0] == v for n in names)}
        print(f"variants present: {names}")
        if missing:
            print(f"variants MISSING from this build (skipped): {sorted(missing)}")
        print(f"mode={args.mode}  ranks={nranks}  reps={args.reps}")

    grid, b = build_case(args.ncell)
    if rank == 0:
        print(f"grid {b.ni} x {b.nj} x {b.nk}")
        print("\ncorrectness gate (vs production, zeroed scratch dU):")
        check = check_correctness(b, names)
        if not all(v["bitwise"] for v in check.values()):
            print(
                "  note: non-bitwise variants are expected if they reassociate;\n"
                "  judge by 'of scale' (~1e-7 is float32 ulp noise, not a bug)."
            )
    else:
        check = {}

    if args.mode == "saturated":
        # Every rank built its own grid; give the slowest one time to finish,
        # then open the timed window simultaneously so the DRAM contention is
        # real rather than staggered.
        start = float(os.environ.get("EMBER_BENCH_START", "0")) or (
            time.time() + args.start_delay
        )
        wait_until(start)

    results = time_variants(b, names, args.reps, args.warmup)

    base = names[0]
    for name in names:
        r = results[name]
        line = (
            f"rank {rank:3d}  {name:8s}  "
            f"median {r['median']:7.3f} ns/cell   min {r['min']:7.3f}"
        )
        if name != base:
            dm = (r["median"] / results[base]["median"] - 1) * 100
            dn = (r["min"] / results[base]["min"] - 1) * 100
            line += f"   vs {base}: median {dm:+.1f}%  min {dn:+.1f}%"
        print(line, flush=True)

    if args.json:
        row = dict(
            rank=rank,
            nranks=nranks,
            mode=args.mode,
            ncell=args.ncell,
            shape=[b.ni, b.nj, b.nk],
            results={n: results[n] for n in names},
            correctness={n: check.get(n) for n in names},
        )
        with open(args.json, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
