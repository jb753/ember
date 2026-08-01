#!/usr/bin/env -S uv run
"""Sweep the k-slab tiling depth (kb) for set_visc_force (and, if the
pre-fusion comparison kernel is built, set_residual_old).

set_residual still *takes* a kb argument, but kb is inert: its i/j/k passes
were fused into a single k-loop with no remaining slab boundary, which
removed the second touch that the old two-pass-per-slab structure needed
slab blocking to protect (see that kernel's header comment in residual.f90).
The surviving `do k0 = 1, nk-1, kb` nest is a pure re-nesting of
`do k = 1, nk-1` -- pa/pb and the k=1 face prime sit outside both loops and
there is no per-slab prologue or epilogue -- so sweeping kb here would
measure nothing. Benchmarked once instead, at a single fixed cost, for
reference against the still-genuinely-tiled kernels.

set_visc_force still has three separate direction sweeps per slab (i, j, k),
each re-reading overlapping nodal input within the slab's k-range, so kb
still does real cache blocking there. Production fixes it to
ember.grid._KB_SLAB (currently 8); this script measures whether that is
actually the wall-time optimum on a given grid/machine.

set_residual_old (residual_old_bench.f90, kept only for this comparison) is
the pre-fusion kernel with the same genuinely-tiled two-pass structure
set_residual used to have, included here to show what slab depth actually
bought before the fusion.
"""

import time

import ember.fortran
from ember import util
from ember.cases import build_duct_grid

N = 100


def bench_residual(b):
    ni, nj, nk = b.shape
    i_cusp_start, i_cusp_end = b.i_cusp
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
    planes, rows = util.carve_view(b.tau_q_halo, (ni, njp, 5, 2), (ni, 5, 3))
    du = b.residual_nd
    du.flags.writeable = True

    kwargs = dict(
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
        # kb is still a required dummy argument, but it is inert: the slab
        # loop it drives is a pure re-nesting of `do k = 1, nk-1` (pa/pb and
        # the k=1 face prime sit outside both loops, and there is no per-slab
        # prologue/epilogue). Passed here only to satisfy the signature.
        kb=nk - 1,
        njp=njp,
        ni=ni,
        nj=nj,
        nk=nk,
    )

    t0 = time.perf_counter()
    for _ in range(N):
        ember.fortran.set_residual(**kwargs)
    return (time.perf_counter() - t0) / N * 1e3


def bench_residual_old(b, kb):
    """Sweep target: set_residual_old, the pre-fusion, genuinely
    two-pass-per-slab kernel kept in residual_old_bench.f90 for this
    comparison -- shows what kb tiling actually bought before the fusion
    that removed it from production set_residual.
    """
    ni, nj, nk = b.shape
    i_cusp_start, i_cusp_end = b.i_cusp
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
    planes, rows = util.carve_view(b.tau_q_halo, (ni, njp, 5, 2), (ni, 5, 3))
    du = b.residual_nd
    du.flags.writeable = True

    kwargs = dict(
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
        kb=kb,
        njp=njp,
        ni=ni,
        nj=nj,
        nk=nk,
    )

    t0 = time.perf_counter()
    for _ in range(N):
        ember.fortran.set_residual_old(**kwargs)
    return (time.perf_counter() - t0) / N * 1e3


def bench_visc_force(b, kb):
    halo = b.tau_q_halo
    tau_cell = halo[..., 0:6]
    q_cell = halo[..., 6:9]
    ni, nj, nk = b.shape
    i_cusp_start, i_cusp_end = b.i_cusp
    planes, rows = util.carve_view(b.scratch, (ni, nj, 4, 2), (ni, 4, 3))
    fbody = b.F_body_nd
    fbody.flags.writeable = True

    kwargs = dict(
        cons=b.conserved_nd,
        cons_cell=b.conserved_cell_nd,
        vol=b.vol_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        omega_block=b.Omega_nd,
        r=b.r_nd,
        mu=b.mu_nd,
        p=b.P_nd,
        p_offset=b.P_offset_nd,
        fvisc=fbody[..., 1:],
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_rel_nd,
        tau_cell=tau_cell,
        q_cell=q_cell,
        planes=planes,
        rows=rows,
        kb=kb,
        **b.ijk_wall_visc,
        **b.Omega_wall_nd,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
    )

    t0 = time.perf_counter()
    for _ in range(N):
        fbody.fill(0.0)
        ember.fortran.set_visc_force(**kwargs)
    return (time.perf_counter() - t0) / N * 1e3


def main():
    grid = build_duct_grid(1_000_000)
    b = grid[0]
    ni, nj, nk = b.shape
    print(f"grid {ni} x {nj} x {nk}  (nk-1={nk - 1} cells in k)")

    grid.update_cached_conserved()
    grid.apply_bconds()
    grid.update_sources(False, 0.0)  # seeds tau_q_halo, mu_turb, F_body, etc
    grid.update_timestep(rf=1.0)

    t_r = bench_residual(b)
    print(f"\nset_residual (no longer kb-tiled): {t_r:.4f} ms/call")

    kb_values = [
        k for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, nk - 1] if 1 <= k <= nk - 1
    ]
    kb_values = sorted(set(kb_values))

    has_old = hasattr(ember.fortran, "set_residual_old")
    if not has_old:
        print(
            "\n(set_residual_old not built -- skipping pre-fusion comparison; "
            "see residual_old_bench.f90)"
        )

    header = f"{'kb':>4}  {'set_visc_force (ms)':>20}"
    if has_old:
        header += f"  {'set_residual_old (ms)':>22}"
    print(f"\n{header}")

    results_v, results_ro = {}, {}
    for kb in kb_values:
        t_v = bench_visc_force(b, kb)
        results_v[kb] = t_v
        row = f"{kb:>4}  {t_v:>20.4f}"
        if has_old:
            t_ro = bench_residual_old(b, kb)
            results_ro[kb] = t_ro
            row += f"  {t_ro:>22.4f}"
        print(row)

    best_v = min(results_v, key=results_v.get)
    print(f"\nbest kb for set_visc_force:   {best_v}  ({results_v[best_v]:.4f} ms)")
    print(f"current _KB_SLAB=8 -> visc_force {results_v.get(8, float('nan')):.4f} ms")
    if has_old:
        best_ro = min(results_ro, key=results_ro.get)
        print(
            f"best kb for set_residual_old: {best_ro}  ({results_ro[best_ro]:.4f} ms), "
            f"vs kb=nk-1: {results_ro[nk - 1]:.4f} ms "
            f"({(results_ro[nk - 1] / results_ro[best_ro] - 1) * 100:+.1f}% vs best)"
        )


if __name__ == "__main__":
    main()
