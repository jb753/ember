"""Correctness check and timing benchmark for advance_rk_stage_mg_fused_irs.

Compares the experimental coarse-IRS kernel against the production
advance_rk_stage_mg_fused_opt on a 97^3 node (96^3 cell) grid:

1. Correctness: sf_irs=0.0 must match _opt to within float rounding order
   (see tests/test_mg_irs.py for the same check at a smaller, CI-friendly size).
2. Benchmark: mean time per call for _opt vs _irs, both at sf_irs=0.0 (isolates
   the new subroutine's own overhead) and at a representative sf_irs>0 (the
   real cost of smoothing).

Run with: uv run python scripts/mg_irs_bench.py
"""

import logging
import time

import numpy as np

import ember.grid  # noqa: F401  binds ember.fortran
import ember

logging.disable(logging.CRITICAL)

NP = 5
NI = NJ = NK = 97  # 96^3 cells, divisible by 2**5
N_LEVELS = 3
N_REPEAT = 30


def _make_inputs(ni, nj, nk, seed):
    rng = np.random.default_rng(seed)
    residual = np.asfortranarray(
        rng.standard_normal((ni - 1, nj - 1, nk - 1, NP)).astype(np.float32)
    )
    dt_vol = np.asfortranarray(
        (0.5 + rng.random((ni - 1, nj - 1, nk - 1))).astype(np.float32)
    )
    snapshot = np.asfortranarray(
        rng.standard_normal((ni, nj, nk, NP)).astype(np.float32)
    )
    return residual, dt_vol, snapshot


def _make_scratch(ni, nj, nk):
    nc1i, nc1j, nc1k = (ni - 1) // 2, (nj - 1) // 2, (nk - 1) // 2
    tmp = np.asfortranarray(np.zeros((ni - 1, nj - 1, nk - 1, NP), dtype=np.float32))
    corr = np.asfortranarray(np.zeros((nc1i, nc1j, nc1k, NP), dtype=np.float32))
    aplane = np.asfortranarray(np.zeros((ni - 1, nc1j), dtype=np.float32))
    bb = np.asfortranarray(np.zeros((ni - 1, nj - 1, nc1k, NP), dtype=np.float32))
    return tmp, corr, aplane, bb


def _mg_irs_scratch_sizes(ni, nj, nk, n_levels):
    """Element counts for advance_rk_stage_mg_fused_irs's flat packed scratch.

    Allocated once by the caller (see _build_kwargs) and reused every call --
    no per-step allocation.
    """
    n_res = n_tri = 0
    for lvl in range(1, n_levels + 1):
        b = 2**lvl
        nib, njb, nkb = (ni - 1) // b, (nj - 1) // b, (nk - 1) // b
        n_res += nib * njb * nkb * NP
        n_tri += 2 * (nib + njb + nkb)
    return n_res, n_tri


def _build_kwargs(residual, dt_vol, snapshot, ni, nj, nk, n_levels, sf_irs=None):
    cons = np.asfortranarray(snapshot.copy())
    tmp, corr, aplane, bb = _make_scratch(ni, nj, nk)
    kwargs = dict(
        cons=cons,
        snapshot=snapshot,
        residual=residual,
        dt_vol=dt_vol,
        alpha=1.0,
        cfl=0.4,
        fmgrid=0.2,
        n_levels=n_levels,
        tmp=tmp,
        corr=corr,
        aplane=aplane,
        bb=bb,
    )
    if sf_irs is not None:
        n_res, n_tri = _mg_irs_scratch_sizes(ni, nj, nk, n_levels)
        kwargs["sf_irs"] = sf_irs
        kwargs["coarse_res_buf"] = np.asfortranarray(
            np.zeros(max(n_res, 1), dtype=np.float32)
        )
        kwargs["tri_work_buf"] = np.asfortranarray(
            np.zeros(max(n_tri, 1), dtype=np.float32)
        )
    return kwargs


def _run(kernel, residual, dt_vol, snapshot, ni, nj, nk, n_levels, sf_irs=None):
    kwargs = _build_kwargs(residual, dt_vol, snapshot, ni, nj, nk, n_levels, sf_irs)
    kernel(**kwargs)
    return kwargs["cons"]


def _time_call(
    kernel, residual, dt_vol, snapshot, ni, nj, nk, n_levels, sf_irs, n_repeat
):
    # Buffers built once, outside the timed loop, and reused every call --
    # matches how a persistent Block's scratch would be reused step to step.
    kwargs = _build_kwargs(residual, dt_vol, snapshot, ni, nj, nk, n_levels, sf_irs)
    # Warm-up call (page faults, first-touch, branch predictor).
    kernel(**kwargs)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        kernel(**kwargs)
    elapsed = time.perf_counter() - t0
    return elapsed / n_repeat


def main():
    residual, dt_vol, snapshot = _make_inputs(NI, NJ, NK, seed=0)

    print(f"Grid: {NI}x{NJ}x{NK} nodes ({NI - 1}^3 cells), n_levels={N_LEVELS}")
    print()

    # --- Correctness: sf_irs=0.0 must match production _opt. ---
    cons_opt = _run(
        ember.fortran.advance_rk_stage_mg_fused_opt,
        residual,
        dt_vol,
        snapshot,
        NI,
        NJ,
        NK,
        N_LEVELS,
    )
    cons_irs = _run(
        ember.fortran.advance_rk_stage_mg_fused_irs,
        residual,
        dt_vol,
        snapshot,
        NI,
        NJ,
        NK,
        N_LEVELS,
        sf_irs=0.0,
    )
    max_abs_diff = np.abs(cons_opt - cons_irs).max()
    np.testing.assert_allclose(cons_opt, cons_irs, rtol=1e-5, atol=1e-6)
    print(f"Correctness OK: sf_irs=0.0 matches _opt (max abs diff {max_abs_diff:.3e})")
    print()

    # --- Benchmark. ---
    t_opt = _time_call(
        ember.fortran.advance_rk_stage_mg_fused_opt,
        residual,
        dt_vol,
        snapshot,
        NI,
        NJ,
        NK,
        N_LEVELS,
        None,
        N_REPEAT,
    )
    t_irs_0 = _time_call(
        ember.fortran.advance_rk_stage_mg_fused_irs,
        residual,
        dt_vol,
        snapshot,
        NI,
        NJ,
        NK,
        N_LEVELS,
        0.0,
        N_REPEAT,
    )
    t_irs_smooth = _time_call(
        ember.fortran.advance_rk_stage_mg_fused_irs,
        residual,
        dt_vol,
        snapshot,
        NI,
        NJ,
        NK,
        N_LEVELS,
        0.5,
        N_REPEAT,
    )

    print(f"{'kernel':40s} {'ms/call':>10s} {'vs _opt':>10s}")
    print(f"{'_opt (production)':40s} {t_opt * 1e3:10.3f} {'1.00x':>10s}")
    print(
        f"{'_irs sf_irs=0.0 (overhead only)':40s} {t_irs_0 * 1e3:10.3f} {t_irs_0 / t_opt:9.2f}x"
    )
    print(
        f"{'_irs sf_irs=0.5 (smoothing cost)':40s} {t_irs_smooth * 1e3:10.3f} {t_irs_smooth / t_opt:9.2f}x"
    )


if __name__ == "__main__":
    main()
