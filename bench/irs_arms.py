"""Arm definitions for the implicit residual smoothing (IRS) study.

Production's `smooth_residual_tri_tiled` (src/ember/_fortran/residual.f90) had
never been in this harness -- every tracked result in bench/results/ is a
`set_residual` number. This module is the IRS counterpart of
`residual_arms.py`: it builds a comparable case, seeds a realistic residual
field, and exposes one zero-argument callable per arm, so
`bench_prod_baseline.py --kernel irs` can time it under exactly the protocol
the README mandates (one arm per process, rank-barriered, launch-replicated).

Arms:

  irs     smooth_residual_tri_tiled  -- production
  irsna   smooth_residual_tri_na     -- production with the four alias-versioned
                                        j/k recurrence loops hoisted into
                                        contained subroutines taking separate
                                        dummies (opt-report fix 1)
  irsnat  smooth_residual_tri_nat    -- irsna plus the i-solve tile carried in
                                        `work` instead of an automatic array,
                                        removing the alloca (fix 1 + fix 2)

Both arms are bitwise identical to production by construction -- same
operations, same order, same operands. `check_correctness` gates on bitwise,
not a tolerance; anything else is a bug in the arm.

WHY REPEATED IN-PLACE CALLS ARE SAFE HERE. IRS runs in place on dU, so rep n
smooths the output of rep n-1 and the field is not the same on every rep. That
would be a fatal instrument flaw for a data-dependent kernel. This one has no
data-dependent branch and no data-dependent memory access -- the work is
identical whatever dU holds -- so the only way the drift could bias a timing is
by producing subnormals, which are slow on this hardware. It cannot: the
factored Neumann operator preserves constant fields EXACTLY, so the block mean
of each component survives every application and the stored values stay O(mean)
while only the fluctuation about it decays. `check_denormals` asserts that
empirically after the timed run rather than trusting the argument, and
`--reps`-dependence of the median would show it too.
"""

import numpy as np

from ember import util

from residual_arms import (  # noqa: F401  (re-export)
    DAMPIN,
    build_case,
    build_kwargs,
    callers,
)

# Jameson IRS coefficient. Production's default is sf_resid=0.0 (IRS off), so
# there is no "production value" to inherit; 0.5 is the middle of the usual
# 0.4-0.8 range and makes the smoother meaningfully active for the correctness
# gate. Timing does not depend on it (no data-dependent control flow), so this
# choice is a correctness-gate choice, not a benchmarking one.
SF = 0.5

IRS_ARMS = (
    "irs",
    "irsna",
    "irsnat",
    "irstr",
    "irsi",
    "irsj",
    "irsk",
    "irsijk",
)

ENTRY = {
    "irs": "smooth_residual_tri_tiled",
    "irsna": "smooth_residual_tri_na",
    "irsnat": "smooth_residual_tri_nat",
    "irstr": "smooth_residual_tri_tr",
    # Per-direction diagnostic arms (residual_irs_dirs.f90): the same kernel
    # with whole direction solves switched off, so the three can be timed
    # apart. `irsijk` runs all three and exists as the control -- it should
    # match `irs`, and if it does not, the switches themselves cost something
    # and none of the other three means anything.
    "irsi": "smooth_residual_tri_dirs",
    "irsj": "smooth_residual_tri_dirs",
    "irsk": "smooth_residual_tri_dirs",
    "irsijk": "smooth_residual_tri_dirs",
}

# Direction switches for the diagnostic arms above.
_DIRS = {
    "irsi": (1, 0, 0),
    "irsj": (0, 1, 0),
    "irsk": (0, 0, 1),
    "irsijk": (1, 1, 1),
}

# Arms that carry the i-solve tile inside `work` and take its length as an
# argument, so they need the longer buffer.
_TILED_WORK = ("irsnat",)

# Must match `integer, parameter :: BJ` in both residual.f90 and
# bench/subroutines/residual_irs_noalias.f90. Only used to size `work` for the
# irsnat arm; a mismatch would be caught immediately by an out-of-bounds write
# in the debug build, but keep it in step by hand.
BJ = 32


def work_len(shape, tiled=False):
    """Length of the Thomas-coefficient buffer, per the kernel's own layout."""
    ni, nj, nk = shape
    nci, ncj, nck = ni - 1, nj - 1, nk - 1
    n = 2 * (nci + ncj + nck)
    return n + BJ * nci if tiled else n


def seed_du(b):
    """Fill block.residual_nd with a real residual field, once, untimed.

    Rule 3: a real port evaluates set_residual immediately before IRS every
    step, so the residual is emphatically NOT a hoistable input -- but it is
    the *previous kernel's* output, not this one's, and pricing it inside the
    IRS timing window would measure set_residual. It is therefore built once
    here and the smoother is timed on it.

    Uses the production residual (with the change limiter active, dampin as in
    the set_residual study) rather than random data, so the field has the
    spectral content IRS actually sees.
    """
    du = b.residual_nd
    du.flags.writeable = True
    callers(b, du, DAMPIN)["prod"]()
    return du


def swirl_state(b):
    """Non-degenerate state for the correctness gate.

    Rule 5 applies one level removed: build_duct_grid is axially straight, so
    the j/k face mass fluxes vanish and the seeded residual's j/k structure is
    degenerate -- which is exactly the structure the j- and k-direction Thomas
    solves act on. Reusing residual_arms.swirl() gives the residual cross-
    stream content before it is handed to the smoother.
    """
    from residual_arms import swirl

    return swirl(b)


def callers_irs(b, du, sf=SF, active_arms=IRS_ARMS):
    """One zero-argument callable per active IRS arm, operating in place on `du`."""
    import ember.fortran as F

    ni, nj, nk = b.shape
    out = {}
    for name in active_arms:
        fn = getattr(F, ENTRY[name], None)
        if fn is None:
            continue
        tiled = name in _TILED_WORK
        nwork = work_len((ni, nj, nk), tiled)
        # Carved from block.scratch, exactly as grid.py does: nodal
        # (ni,nj,nk,5) and so vastly oversized for either length. Free at this
        # point in a real step -- set_residual stages its face flows in
        # tau_q_halo and the march reuses scratch only afterwards.
        work = util.carve_view(b.scratch, (nwork,))
        kw = dict(du=du, sf=sf, work=work, ni=ni, nj=nj, nk=nk)
        if tiled:
            kw["nwork"] = nwork
        if name in _DIRS:
            kw["do_i"], kw["do_j"], kw["do_k"] = _DIRS[name]
        out[name] = lambda fn=fn, kw=kw: fn(**kw)
    return out


# Arms that compute the whole three-direction operator and so must agree with
# production bitwise. The per-direction diagnostics deliberately do not --
# skipping a solve is the point of them -- so gating them against `irs` would
# report a bug that is not one.
FULL_ARMS = ("irs", "irsna", "irsnat", "irstr", "irsijk")


# Arms for the FUSED-LIMITER study (Phase 3). The saving spans two kernels --
# set_residual's trailing scaling pass and the smoother's i-solve -- so neither
# can be timed alone; these arms time the pair.
UPDATE_ARMS = ("unfused", "fused")


def callers_update(b, du, dampin=DAMPIN, sf=SF):
    """Time set_residual + IRS as a unit, unfused vs fused.

    Both arms are two f2py calls against pre-built kwargs, so the Python
    scaffolding either side of the kernels is identical and does not favour
    one. `unfused` is the historical sequence: set_residual applies the change
    limiter itself over the whole volume, then the standalone three-direction
    smoother runs. `fused` defers the limiter's scaling to the smoother, which
    applies it inside its i-solve gather -- one full-volume read/write pair
    fewer. The two are bitwise identical (tests/test_irs_fused_damp.py).
    """
    import ember.fortran as F

    ni, nj, nk = b.shape
    common, private = build_kwargs(b)
    kw_damped = dict(common, **private["prod"], du=du, dampin=dampin)
    kw_plain = dict(common, **private["prod"], du=du, dampin=0.0)
    work = util.carve_view(b.scratch, (work_len((ni, nj, nk)),))
    shape = dict(ni=ni, nj=nj, nk=nk)

    def unfused():
        F.set_residual(**kw_damped)
        F.smooth_residual_tri_tiled(du=du, sf=sf, work=work, **shape)

    def fused():
        ravg = F.set_residual(**kw_plain)
        F.smooth_residual_scale_tri(
            du=du, dt_vol=b.dt_vol_nd, ravg=ravg, dampin=dampin, sf=sf,
            work=work, **shape
        )

    return dict(unfused=unfused, fused=fused)


def check_correctness(b, du, ref, active_arms=FULL_ARMS):
    """Bitwise-gate every arm against production on identical input.

    dU is intent(inout) and the smoother is in place, so unlike the
    set_residual gate this one MUST restore the input before each arm.
    """
    fns = callers_irs(b, du, SF, active_arms)
    du[...] = ref
    fns["irs"]()
    base = np.array(du, copy=True)
    results = {}
    for name, fn in fns.items():
        if name == "irs":
            continue
        du[...] = ref
        fn()
        got = np.array(du, copy=True)
        diff = np.abs(got - base)
        # In ulps of the quantity being differenced, not of the final result.
        scale = np.abs(base)
        with np.errstate(divide="ignore", invalid="ignore"):
            ulps = np.where(scale > 0, diff / np.spacing(scale), 0.0)
        results[name] = dict(
            bitwise=bool(np.array_equal(got, base)),
            max_abs=float(diff.max()),
            max_ulp=float(np.nanmax(ulps)),
        )
    return results


def check_denormals(du):
    """Fraction of dU entries that are subnormal or zero, after a timed run.

    The instrument's one real hazard (see module docstring): repeated in-place
    smoothing decays the fluctuation about a preserved mean, and a field that
    had decayed into subnormals would time differently from a real one.
    """
    a = np.abs(np.asarray(du))
    tiny = np.finfo(a.dtype).tiny
    nz = a > 0
    return dict(
        frac_subnormal=float(((a > 0) & (a < tiny)).mean()),
        frac_zero=float((a == 0).mean()),
        median_mag=float(np.median(a[nz])) if nz.any() else 0.0,
    )


def main():
    """Standalone Gate-2 correctness pre-flight. Times nothing."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--ncell", type=int, default=300_000)
    ap.add_argument("--sf", type=float, default=SF)
    args = ap.parse_args()

    grid, b = build_case(args.ncell)
    swirl_state(b)
    du = seed_du(b)
    ref = np.array(du, copy=True)

    ni, nj, nk = b.shape
    print(f"shape {ni}x{nj}x{nk}, ncell={(ni - 1) * (nj - 1) * (nk - 1)}, sf={args.sf}")
    print(f"seeded |dU| max {np.abs(ref).max():.6e}  mean {np.abs(ref).mean():.6e}")

    bad = 0
    for name, r in check_correctness(b, du, ref).items():
        ok = "BITWISE" if r["bitwise"] else f"DIFFERS max {r['max_abs']:.3e}"
        print(f"  {name:>8}  {ok}  (max {r['max_ulp']:.2f} ulp)")
        bad += not r["bitwise"]
    return 1 if bad else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
