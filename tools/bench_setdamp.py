#!/usr/bin/env -S uv run
"""A/B set_residual+damp_residual against the fused set_residual_damp.

The change limiter splits into a global reduction (block mean of
|dU*dt_vol|) and a pointwise scaling. The reduction only reads dU, and
set_residual already has each dU value in a register as it writes it -- so
folding the reduction into set_residual removes a full-volume dU read,
~20 B/cell, about 23% of the dU-path traffic.

Both arms compute the same thing (verified bitwise), so this measures the
fusion alone. Note the fused form necessarily runs damp BEFORE any IRS,
where production runs IRS first -- a numerics change that this harness does
not evaluate.
"""
import argparse, json, os, statistics, sys, time
import numpy as np
import ember.fortran as F
from ember import util
from ember.cases import build_duct_grid

DAMPIN = 2.0


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
    g.update_sources(False, 0.0); g.update_timestep(rf=1.0)
    b = g[0]; ni, nj, nk = b.shape
    N = (ni - 1) * (nj - 1) * (nk - 1)
    ics, ice = b.i_cusp
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
    planes, rows = util.carve_view(b.tau_q_halo, (ni, njp, 5, 2), (ni, 5, 3))
    kw = dict(cons=b.conserved_nd, p=b.P_nd, p_offset=b.P_offset_nd, r=b.r_nd,
              omega=b.Omega_nd, dai=b.dAi_nd, daj=b.dAj_nd, dak=b.dAk_nd,
              f_body=b.F_body_nd, vx=b.Vx_nd, vr=b.Vr_nd, vt=b.Vt_nd, ho=b.ho_nd,
              planes=planes, rows=rows, **b.ijk_wall_conv,
              i_cusp_start=ics, i_cusp_end=ice, kb=nk - 1, njp=njp,
              ni=ni, nj=nj, nk=nk)
    du = b.residual_nd; du.flags.writeable = True

    def unfused():
        F.set_residual(du=du, **kw)
        F.damp_residual(du=du, dt_vol=b.dt_vol_nd, dampin=DAMPIN, ni=ni, nj=nj, nk=nk)

    def fused():
        F.set_residual_damp(du=du, dt_vol=b.dt_vol_nd, dampin=DAMPIN, **kw)

    def fused_ivdep():
        F.set_residual_damp_ivdep(du=du, dt_vol=b.dt_vol_nd, dampin=DAMPIN, **kw)

    def fused_split():
        F.set_residual_damp_split(du=du, dt_vol=b.dt_vol_nd, dampin=DAMPIN, **kw)

    if rank == 0:
        x = np.zeros_like(b.residual_nd); du[...] = x; unfused(); ref = np.array(du, copy=True)
        du[...] = x; fused(); got = np.array(du, copy=True)
        d = float(np.abs(got - ref).max()); sc = float(np.abs(ref).max())
        print(f"grid {ni}x{nj}x{nk}  cusp={ics > 0}  dampin={DAMPIN}")
        print(f"correctness: maxdiff {d:.4e} ({d / sc if sc else 0:.3e} of scale) "
              f"bitwise={np.array_equal(got, ref)}")

    if a.mode == "saturated":
        t0 = float(os.environ.get("EMBER_BENCH_START", "0")) or (time.time() + a.start_delay)
        while time.time() < t0:
            time.sleep(min(t0 - time.time(), 0.05))

    variants = {"unfused": unfused, "fused": fused}
    for nm, fn in (("fused_ivdep", fused_ivdep), ("fused_split", fused_split)):
        if hasattr(F, "set_residual_damp_" + nm.split("_")[1]):
            variants[nm] = fn
    for _ in range(4):
        for fn in variants.values():
            fn()
    s = {n: [] for n in variants}
    for _ in range(a.reps):
        for n, fn in variants.items():
            t = time.perf_counter(); fn()
            s[n].append((time.perf_counter() - t) / N * 1e9)
    res = {n: dict(median=statistics.median(v), min=min(v)) for n, v in s.items()}
    base = res["unfused"]["median"]
    for n in variants:
        line = f"rank {rank:3d}  {n:8s} median {res[n]['median']:7.3f}  min {res[n]['min']:7.3f}"
        if n != "unfused":
            line += f"   vs unfused: {100 * (res[n]['median'] / base - 1):+.1f}%"
        print(line, flush=True)
    if a.json:
        with open(a.json, "a") as fh:
            fh.write(json.dumps(dict(rank=rank, mode=a.mode, results=res)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
