"""Benchmark the viscous body-force kernels and Grid.update_sources.

A/B protocol per docs/dev/viscous_kernels.md section 4:

  * build via the real path (make compile) before running; never benchmark a
    standalone compile;
  * measure through the real Python call path (f2py marshaling included);
  * single-thread, pinned to one core, warmed up, median (and min) ns/cell
    over many reps, kb variants interleaved round-robin to cancel drift;
  * sweep block sizes across the cache boundary and report ns/cell per size.

Works against the fused kernel (kb argument, rolling planes/rows scratch),
the tiled kernel (kb argument, slab-sized flow_scratch) and the pre-tiling
baseline (full-volume flow_scratch), detected from the f2py signature, so the
same script serves both sides of a git-stash A/B.

Usage:

    uv run python scripts/bench_viscous.py --label tiled --csv tiled.csv

Pinning is done in-process via sched_setaffinity (default core 2, override
with BENCH_CPU=<n>). OMP_NUM_THREADS is forced to 1 before numpy loads.
"""

import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.sched_setaffinity(0, {int(os.environ.get("BENCH_CPU", "2"))})

import numpy as np  # noqa: E402

import ember.block  # noqa: E402
import ember.fortran  # noqa: E402
import ember.grid  # noqa: E402
from ember import util  # noqa: E402
from ember.fluid import PerfectFluid  # noqa: E402
from ember.periodic import PeriodicPatch  # noqa: E402

# The tiled kernels take kb / slab-sized flow scratch; the baselines take
# full-volume flow scratch and no kb. Detected per kernel so the same script
# runs on both sides of a git-stash A/B.
TILED = "kb" in ember.fortran.set_visc_force.__doc__
FUSED = "planes" in ember.fortran.set_visc_force.__doc__
TILED_RESID = "kb" in ember.fortran.set_residual.__doc__
FUSED_RESID = "planes" in ember.fortran.set_residual.__doc__
# Production slab-depth knob (renamed _KB_VISC -> _KB_SLAB when set_residual
# was tiled).
KB_ATTR = "_KB_SLAB" if hasattr(ember.grid, "_KB_SLAB") else "_KB_VISC"

SIZES = [(48, 32, 32), (64, 48, 48), (80, 64, 64), (96, 96, 96), (128, 96, 96)]
KBS = [1, 2, 4, 8, 16]


def build_grid(shape):
    """Single-block theta-periodic grid with a swirling sheared flow.

    Mirrors the fixture of tests/test_viscous_phases_golden.py, generalized to
    `shape`, so every branch of both kernels sees physical values (i/j faces
    are walls, k is periodic, nonzero rotation).
    """
    nb = 36
    pitch = 2.0 * np.pi / nb

    block = ember.block.Block(shape=shape)
    block.set_Nb(nb)
    xrt = util.linmesh3((0.0, 0.15), (0.5, 0.9), (0.0, pitch), shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))

    x, r, t = block.x, block.r, block.t
    r_span = float(r.max() - r.min())
    temp = (
        300.0
        + 20.0 * (r - r.min()) / r_span
        + 8.0 * np.sin(2.0 * np.pi * x / float(x.max()))
    ).astype(np.float32)
    block.set_P_T(101325.0, temp)

    vx = (
        100.0
        + 20.0 * np.sin(4.0 * np.pi * t / pitch + np.pi / 4.0)
        + 10.0 * (r - r.min()) / r_span
    ).astype(np.float32)
    vr = (5.0 * np.cos(2.0 * np.pi * t / pitch)).astype(np.float32)
    vt = (40.0 + 15.0 * np.sin(2.0 * np.pi * x / float(x.max()))).astype(np.float32)
    block.set_Vx(vx)
    block.set_Vr(vr)
    block.set_Vt(vt)
    block.set_Omega(50.0)

    wdist = 0.008 * (1.0 + np.sin(np.pi * (r - r.min()) / r_span))
    block.set_wdist(wdist.astype(np.float32))

    block.patches.append(PeriodicPatch(k=0))
    block.patches.append(PeriodicPatch(k=-1))
    return ember.grid.Grid([block])


def phase1_call(block):
    """One production-path set_tau_q_soa call (as in Grid.update_sources)."""
    halo = block.tau_q_halo
    mu_turb = block._get_data_by_keys(
        ("mu_turb",), raise_uninit=False, writeable=True
    )
    ember.fortran.set_tau_q_soa(
        cons=block.conserved_nd,
        t=block.T_nd,
        mu=block.mu_nd,
        cp=block.cp_nd,
        pr_lam=block.fluid._Pr,
        pr_turb=1.0,
        xlength=block.xlen_sq_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        r=block.r_nd,
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        tau_cell=halo[..., 0:6],
        q_cell=halo[..., 6:9],
        mu_turb=mu_turb,
    )
    block._versions["mu_turb"] += 1


def make_phase2_call(block, kb):
    """Closure for one production-path set_visc_force call at slab depth kb."""
    ni, nj, nk = block.shape
    halo = block.tau_q_halo
    if FUSED:
        planes, rows = util.carve_view(block.scratch, (ni, nj, 4, 2), (ni, 4, 3))
        assert planes.flags["F_CONTIGUOUS"] and rows.flags["F_CONTIGUOUS"]
        assert not planes.flags["OWNDATA"] and not rows.flags["OWNDATA"]
        extra = {"planes": planes, "rows": rows, "kb": kb}
    elif TILED:
        flow_scratch = util.carve_view(block.scratch, (ni, nj, kb + 1, 4))
        assert flow_scratch.flags["F_CONTIGUOUS"]
        assert not flow_scratch.flags["OWNDATA"]
        extra = {"flow_scratch": flow_scratch, "kb": kb}
    else:
        extra = {"flow_scratch": block.scratch[..., 0:4]}
    # update_sources leaves F_body_nd locked read-only; unlock before taking
    # the fvisc view (the e2e timing below manages the lock itself).
    block.F_body_nd.flags.writeable = True
    fvisc = block.F_body_nd[..., 1:]
    i_cusp_start, i_cusp_end = block.i_cusp

    def call():
        ember.fortran.set_visc_force(
            cons=block.conserved_nd,
            vol=block.vol_nd,
            dai=block.dAi_nd,
            daj=block.dAj_nd,
            dak=block.dAk_nd,
            omega_block=block.Omega_nd,
            r=block.r_nd,
            mu=block.mu_nd,
            fvisc=fvisc,
            vx=block.Vx_nd,
            vr=block.Vr_nd,
            vt=block.Vt_rel_nd,
            tau_cell=halo[..., 0:6],
            q_cell=halo[..., 6:9],
            **extra,
            **block.ijk_wall_visc,
            **block.Omega_wall_nd,
            i_cusp_start=i_cusp_start,
            i_cusp_end=i_cusp_end,
        )

    return call


def make_residual_call(block, kb):
    """Closure for one production-path set_residual call at slab depth kb.

    kb is ignored by the fused kernel (no slab depth); the caller's variant
    loop collapses to one entry there.
    """
    ni, nj, nk = block.shape
    if FUSED_RESID:
        # Conditional anti-aliasing pad, mirroring Grid.update_residual.
        njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
        planes, rows = util.carve_view(
            block.tau_q_halo, (ni, njp, 5, 2), (ni, 5, 3)
        )
        assert planes.flags["F_CONTIGUOUS"] and rows.flags["F_CONTIGUOUS"]
        assert not planes.flags["OWNDATA"] and not rows.flags["OWNDATA"]
        extra = {"planes": planes, "rows": rows, "kb": kb, "njp": njp}
    elif TILED_RESID:
        flow_i = util.carve_view(block.scratch, (ni, nj, kb, 5))
        flow_jk = util.carve_view(block.tau_q_halo, (ni, nj, kb + 1, 10))
        extra = {"flow_i": flow_i, "flow_jk": flow_jk, "kb": kb}
    else:
        flow_jk = util.carve_view(block.tau_q_halo, (ni, nj, nk, 10))
        extra = {"flow_i": block.scratch, "flow_jk": flow_jk}
    # update_residual leaves residual_nd locked read-only; unlock before
    # taking the du view (the view keeps its own writeable flag).
    block.residual_nd.flags.writeable = True
    du = block.residual_nd
    i_cusp_start, i_cusp_end = block.i_cusp

    def call():
        ember.fortran.set_residual(
            cons=block.conserved_nd,
            p=block.P_nd,
            p_offset=block.P_offset_nd,
            r=block.r_nd,
            omega=block.Omega_nd,
            dai=block.dAi_nd,
            daj=block.dAj_nd,
            dak=block.dAk_nd,
            du=du,
            f_body=block.F_body_nd,
            vx=block.Vx_nd,
            vr=block.Vr_nd,
            vt=block.Vt_nd,
            vt_rel=block.Vt_rel_nd,
            ho=block.ho_nd,
            **extra,
            **block.ijk_wall_conv,
            i_cusp_start=i_cusp_start,
            i_cusp_end=i_cusp_end,
            ni=ni,
            nj=nj,
            nk=nk,
        )

    return call


def time_variants(variants, reps, warmup=3):
    """Round-robin timing of {name: callable}; returns {name: [dt_ns, ...]}."""
    for call in variants.values():
        for _ in range(warmup):
            call()
    out = {name: [] for name in variants}
    for _ in range(reps):
        for name, call in variants.items():
            t0 = time.perf_counter_ns()
            call()
            out[name].append(time.perf_counter_ns() - t0)
    return out


def report(rows, label, size, cells, kernel, timings):
    for name, dts in timings.items():
        med = statistics.median(dts) / cells
        best = min(dts) / cells
        rows.append((label, "x".join(map(str, size)), kernel, name, med, best))
        print(
            f"{label:10s} {'x'.join(map(str, size)):>11s} {kernel:15s} "
            f"{name:>8s} {med:8.2f} {best:8.2f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="tiled" if TILED else "baseline")
    ap.add_argument("--csv", default=None, help="append results to this CSV")
    ap.add_argument("--reps", type=int, default=None, help="override rep count")
    args = ap.parse_args()

    print(f"# variant={args.label} tiled={TILED} cpu={sorted(os.sched_getaffinity(0))}")
    print(f"{'label':10s} {'size':>11s} {'kernel':15s} {'kb':>8s} "
          f"{'med ns':>8s} {'min ns':>8s}")

    rows = []
    for size in SIZES:
        grid = build_grid(size)
        block = grid[0]
        ni, nj, nk = size
        cells = (ni - 1) * (nj - 1) * (nk - 1)
        # Aim for ~0.5 s per variant per kernel at ~100 ns/cell.
        reps = args.reps or max(5, min(60, int(5e9 / (cells * 100 * 3))))

        # Warm the field caches (first update_sources builds them all).
        grid.update_sources(inviscid=False, gain_filt=0.0)
        grid.update_residual()

        # Phase 1: no kb dependence.
        timings = time_variants({"-": lambda: phase1_call(block)}, reps)
        report(rows, args.label, size, cells, "set_tau_q_soa", timings)

        # Phase 2: kb sweep (tiled) or single unblocked variant (baseline).
        kbs = [kb for kb in KBS if kb <= nk - 1] if TILED else [nk - 1]
        if TILED and (nk - 1) not in kbs:
            kbs.append(nk - 1)  # single-slab reference
        variants = {str(kb): make_phase2_call(block, kb) for kb in kbs}
        timings = time_variants(variants, reps)
        report(rows, args.label, size, cells, "set_visc_force", timings)

        # End to end, production path; kb via the module constant.
        def make_e2e(kb):
            def call():
                setattr(ember.grid, KB_ATTR, kb)
                grid.update_sources(inviscid=False, gain_filt=0.0)
            return call

        variants = {str(kb): make_e2e(kb) for kb in kbs} if TILED else {
            "-": lambda: grid.update_sources(inviscid=False, gain_filt=0.0)
        }
        timings = time_variants(variants, max(3, reps // 2))
        report(rows, args.label, size, cells, "update_sources", timings)

        # Inviscid residual kernel: kb sweep (tiled) or single unblocked
        # variant (baseline).
        kbs_r = [kb for kb in KBS if kb <= nk - 1] if TILED_RESID else [nk - 1]
        if TILED_RESID and (nk - 1) not in kbs_r:
            kbs_r.append(nk - 1)  # single-slab reference
        variants = (
            {str(kb): make_residual_call(block, kb) for kb in kbs_r}
            if TILED_RESID
            else {"-": make_residual_call(block, nk - 1)}
        )
        timings = time_variants(variants, reps)
        report(rows, args.label, size, cells, "set_residual", timings)

        # update_residual end to end (no IRS / damping), production path.
        def make_e2e_resid(kb):
            def call():
                setattr(ember.grid, KB_ATTR, kb)
                grid.update_residual()
            return call

        variants = (
            {str(kb): make_e2e_resid(kb) for kb in kbs_r}
            if TILED_RESID
            else {"-": lambda: grid.update_residual()}
        )
        timings = time_variants(variants, max(3, reps // 2))
        report(rows, args.label, size, cells, "update_residual", timings)
        sys.stdout.flush()

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a") as f:
            if new:
                f.write("label,size,kernel,kb,med_ns_per_cell,min_ns_per_cell\n")
            for row in rows:
                f.write(",".join(str(x) for x in row) + "\n")
        print(f"# appended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
