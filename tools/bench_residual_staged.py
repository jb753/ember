#!/usr/bin/env -S uv run
"""A/B production set_residual against the multall/multall staged design.

multall evaluates the five conserved-variable residuals in FIVE passes:
SET_FLUX stages the face mass fluxes FIMAS/FJMAS/FKMAS once, then SUMFLUX is
called once per variable. Production ember does it in ONE fused sweep, holding
the shared face mass flux in a register and consuming it for all five
components before discarding it.

Three arms, all in the same .so (setup.py globs _fortran/*.f90), compared
round-robin interleaved in one process so there is no cross-build LTO drift:

  prod    set_residual         -- one fused sweep
  staged  set_residual_staged  -- stage mdot into fi/fj/fk, five narrow passes
  split   set_residual_split   -- five narrow passes, mdot recomputed inline

`split` is the attribution control: without it a loss cannot be attributed to
the 5-way split rather than to the staging itself.

Two regimes. `serial` is the diagnostic number. `contended` is one of several
independent processes each holding its own grid, all timing the same window,
rendezvousing on a wall-clock start time (no MPI needed). On this box the
contended arm is meant to be run with every rank pinned to a core of ONE
socket, which does saturate that socket's memory controller -- see
tools/run_residual_staged.sh. It is NOT the 100-rank sapphire regime and must
not be reported as "saturated".

Predicted per-cell compulsory traffic (docs section 25): prod ~152 B/cell,
staged ~384, split ~372 -- so ~2.5x, with the serial gap much smaller because
staging removes four fifths of the reciprocals and r-divides.
"""

import argparse
import json
import os
import statistics
import sys
import time

import ember.fortran as F
import numpy as np

from ember import util
from ember.cases import build_duct_grid

# dampin=2 is the low end of multall's recommended 2..100, so the soft-clip is
# actually active rather than a no-op. The strict bitwise gate runs at
# dampin=0: the avg(m) reduction visits each component in (k,j,i) order in
# every arm, so even the damped result should agree, but 0 isolates the sweep.
DAMPIN = 2.0

ARMS = ("prod", "staged", "split", "multall")

# The `multall` arm's nine face-area component arrays are grid GEOMETRY: built
# once at startup in a real port (multall sets AIX/AIR/AIT up in FIND_AREAS),
# never rebuilt per step. Splitting ember's component-first dA into them
# therefore happens outside the timed region, which is faithful, not a cheat.
# Cached per block id so a repeated build_kwargs call does not re-transpose.
_DA_SOA = {}


def build_case(ncell):
    """Build the duct grid and drive it to the state set_residual consumes."""
    grid = build_duct_grid(ncell)
    grid.update_cached_conserved()
    grid.apply_bconds()
    grid.update_sources(False, 0.0)  # seeds F_body, tau_q_halo, mu_turb
    grid.update_timestep(rf=1.0)
    return grid, grid[0]


def build_kwargs(b):
    """Common kwargs plus each arm's private scratch, all carved zero-copy.

    Every buffer for a given call comes from ONE carve_view call, which
    guarantees the spans are disjoint -- block.tau_q_halo's docstring forbids
    aliasing two arrays into the same kernel call, and this is how
    bench_residual_variants.py already satisfies that.
    """
    ni, nj, nk = b.shape
    i_cusp_start, i_cusp_end = b.i_cusp
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj

    # Production's 5-wide rolling buffers, the two narrow pairs, the staged
    # mass-flux volumes and the split arm's mass-flux scratch.
    (
        planes5,
        rows5,
        planes,
        rows,
        mrows,
        mplanes,
        fi,
        fj,
        fk,
    ) = util.carve_view(
        b.tau_q_halo,
        (ni, njp, 5, 2),
        (ni, 5, 3),
        (ni, njp, 2),
        (ni, 3),
        (ni, 3),
        (ni - 1, nj - 1, 2),
        (ni, nj - 1, nk - 1),
        (ni - 1, nj, nk - 1),
        (ni - 1, nj - 1, nk),
    )
    for name, arr in (("fi", fi), ("planes5", planes5), ("mplanes", mplanes)):
        assert arr.flags["F_CONTIGUOUS"] and not arr.flags["OWNDATA"], name

    # SoA geometry + the two staged nodal primitives for the multall arm.
    # rowt/rvt are solution-dependent, so the KERNEL recomputes them every
    # call; only their storage is allocated here, as any block scratch is.
    key = id(b)
    if key not in _DA_SOA:
        soa = {}
        for d, src in (("i", b.dAi_nd), ("j", b.dAj_nd), ("k", b.dAk_nd)):
            for c in range(3):
                soa[f"da{d}{c + 1}"] = np.asfortranarray(src[c])
        soa["rowt"] = np.zeros((ni, nj, nk), dtype=b.r_nd.dtype, order="F")
        soa["rvt"] = np.zeros((ni, nj, nk), dtype=b.r_nd.dtype, order="F")
        _DA_SOA[key] = soa

    common = dict(
        cons=b.conserved_nd,
        p=b.P_nd,
        p_offset=b.P_offset_nd,
        r=b.r_nd,
        omega=b.Omega_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        f_body=b.F_body_nd,
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_nd,
        ho=b.ho_nd,
        dt_vol=b.dt_vol_nd,
        **b.ijk_wall_conv,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
        njp=njp,
        ni=ni,
        nj=nj,
        nk=nk,
    )
    private = dict(
        # kb is inert for production (its slab loop is a pure re-nesting of
        # `do k = 1, nk-1`); the two new arms drop the dummy entirely.
        prod=dict(planes=planes5, rows=rows5, kb=nk - 1),
        staged=dict(planes=planes, rows=rows, fi=fi, fj=fj, fk=fk),
        split=dict(planes=planes, rows=rows, mrows=mrows, mplanes=mplanes),
        # The AoS dai/daj are stripped for this arm in callers(); dak stays,
        # because the cusp seam correction is production's own routine,
        # called unmodified by every arm, and it wants AoS.
        multall=dict(planes=planes, rows=rows, fi=fi, fj=fj, fk=fk, **_DA_SOA[key]),
    )
    return common, private


def callers(b, du, dampin):
    """One zero-argument callable per available arm, writing into `du`."""
    common, private = build_kwargs(b)
    entry = dict(
        prod=F.set_residual,
        staged=getattr(F, "set_residual_staged", None),
        split=getattr(F, "set_residual_split", None),
        multall=getattr(F, "set_residual_multall", None),
    )
    out = {}
    for name in ARMS:
        fn = entry[name]
        if fn is None:
            continue
        base = dict(common)
        if name == "multall":
            # This arm takes the nine SoA components instead; only dak
            # survives, for the shared cusp correction.
            del base["dai"], base["daj"]
        kw = dict(base, **private[name], du=du, dampin=dampin)
        out[name] = lambda fn=fn, kw=kw: fn(**kw)
    return out


def swirl(b):
    """Give the duct cross-stream momentum, so j/k mass fluxes are non-zero.

    build_duct_grid is axially straight and perturbs Vx only, so dAj(1) and
    dAk(1) vanish and the j- and k-face mass fluxes are EXACTLY zero. Timing is
    unaffected (the work happens either way, there are no data-dependent
    branches), but a correctness gate on that state cannot see an error in
    mflux_jface_row / mflux_kface_plane at all -- zero times anything is zero.

    So the gate runs on a state with rho*Vr and rho*r*Vt seeded to ~5% of
    rho*Vx. It need not be a converged or even physical flow field; it only has
    to make every term in every helper non-degenerate. Returns the original
    conserved block so the caller can restore it before timing.
    """
    cons = b.conserved_nd
    cons.flags.writeable = True
    saved = np.array(cons, copy=True)
    rng = np.random.default_rng(0)
    scale = 0.05 * float(np.abs(cons[..., 1]).max())
    for m in (2, 3):
        cons[..., m] += scale * rng.standard_normal(cons.shape[:3]).astype(cons.dtype)
    # Essential, not hygiene: the `multall` arm reads the NODAL Vx/Vr/Vt/ho
    # arrays where production re-derives them from cons, so a direct write to
    # conserved_nd that skipped this would leave the two arms solving
    # different states and the gate would report a bug that is not there.
    b.update_cached_conserved()
    return saved


def check_correctness(b, du, ref):
    """Compare each arm against production on identical input.

    dU is intent(inout) but every element is assigned before it is read, so no
    input restore is needed between arms -- `ref` only exists to prove that.
    """
    results = {}
    for dampin in (0.0, DAMPIN):
        fns = callers(b, du, dampin)
        du[...] = ref
        fns["prod"]()
        base = np.array(du, copy=True)
        scale = float(np.abs(base).max())
        for name, fn in fns.items():
            if name == "prod":
                continue
            du[...] = ref
            fn()
            got = np.array(du, copy=True)
            diff = float(np.abs(got - base).max())
            bitwise = bool(np.array_equal(got, base))
            key = f"{name}@dampin={dampin:g}"
            results[key] = dict(
                bitwise=bitwise,
                max_abs_diff=diff,
                rel=diff / scale if scale else 0.0,
            )
            print(
                f"  {key:22s} max_abs_diff = {diff:.6e} "
                f"({diff / scale if scale else 0.0:.3e} of scale)  bitwise={bitwise}"
            )
    return results


def time_arms(b, du, reps, warmup):
    ni, nj, nk = b.shape
    ncell = (ni - 1) * (nj - 1) * (nk - 1)
    fns = callers(b, du, DAMPIN)

    for _ in range(warmup):
        for fn in fns.values():
            fn()

    samples = {n: [] for n in fns}
    for _ in range(reps):
        for name, fn in fns.items():
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
    ap.add_argument("--mode", choices=["serial", "contended"], default="serial")
    ap.add_argument("--ncell", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--start-delay", type=float, default=20.0)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    # No SLURM here: the driver script hands each local process its index.
    rank = int(os.environ.get("EMBER_BENCH_RANK", "0"))

    missing = [n for n in ARMS[1:] if not hasattr(F, "set_residual_" + n)]
    if missing:
        print(f"not in this build: {missing}", file=sys.stderr)
        return 1

    grid, b = build_case(args.ncell)
    du = b.residual_nd
    du.flags.writeable = True
    ref = np.array(du, copy=True)

    check = {}
    if rank == 0:
        print(
            f"grid {b.ni} x {b.nj} x {b.nk}  ncell={args.ncell}  cusp={b.i_cusp[0] > 0}"
        )
        print("\ncorrectness gate (swirled state, so j/k mass fluxes are non-zero):")
        saved = swirl(b)
        try:
            check = check_correctness(b, du, ref)
        finally:
            b.conserved_nd[...] = saved
            b.update_cached_conserved()
    if args.check_only:
        return 0

    if args.mode == "contended":
        start = float(os.environ.get("EMBER_BENCH_START", "0")) or (
            time.time() + args.start_delay
        )
        wait_until(start)

    res = time_arms(b, du, args.reps, args.warmup)
    for n, rr in res.items():
        line = f"rank {rank:3d}  {n:8s} median {rr['median']:8.3f} ns/cell  min {rr['min']:8.3f}"
        if n != "prod":
            line += (
                f"   vs prod: {(rr['median'] / res['prod']['median'] - 1) * 100:+.1f}%"
            )
        print(line, flush=True)

    if args.json:
        with open(args.json, "a") as fh:
            fh.write(
                json.dumps(
                    dict(
                        rank=rank,
                        mode=args.mode,
                        ncell=args.ncell,
                        shape=list(b.shape),
                        results=res,
                        correctness=check,
                    )
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
