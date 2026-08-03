"""Shared case-builder and arm definitions for the `set_residual` A/B study.

multall evaluates the five conserved-variable residuals in FIVE passes:
SET_FLUX stages the face mass fluxes FIMAS/FJMAS/FKMAS once, then SUMFLUX is
called once per variable. Production ember does it in ONE fused sweep, holding
the shared face mass flux in a register and consuming it for all five
components before discarding it.

Eight arms, all in the same .so (setup.py globs _fortran/*.f90 -- build with
EMBER_ARMS=all or EMBER_ARMS=<comma-list> to get the non-production ones back
in, see setup.py's BENCHMARK_ONLY):

  prod    set_residual         -- one fused sweep
  staged  set_residual_staged  -- stage mdot into fi/fj/fk, five narrow passes
  split   set_residual_split   -- five narrow passes, mdot recomputed inline
  multall  set_residual_multall  -- the faithful multall/multall design: staged
                                  nodal primitives + SoA geometry + five passes
  nodal   set_residual_nodal   -- production's fused sweep reading the nodal
                                  primitives instead of deriving them from cons
  tbaos   set_residual_multall_aos -- the multall design on ember's own AoS
                                  dA(3,i,j,k) geometry
  prodsoa set_residual_prod_soa   -- production's kernel on SoA geometry
  rinv    set_residual_rinv       -- production's kernel plus a staged 1/r

This module holds only what every driver needs to build a comparable case and
call each arm: the grid/state setup, the scratch-carving kwargs builder, the
correctness gate, and an LLC-flush helper. It is deliberately NOT a timing
driver -- see bench/README.md's "what's in this directory" for why the old
round-robin-in-one-process timing loop that used to live here was dropped
(it's the harness bug documented in the "Two independent faults" section:
sweeping the arm set moved a ranking by double digits). Use
bench_prod_baseline.py (one arm per process, barriered, launch-replicated)
for real timing.

See docs/dev/plan_nodal_primitives.md for the nodal/tbaos attribution study
these arms were built to support.
"""

import numpy as np

from ember import util
from ember.cases import build_duct_grid

# dampin=2 is the low end of multall's recommended 2..100, so the soft-clip is
# actually active rather than a no-op. The strict bitwise gate runs at
# dampin=0: the avg(m) reduction visits each component in (k,j,i) order in
# every arm, so even the damped result should agree, but 0 isolates the sweep.
DAMPIN = 2.0

ARMS = ("prod", "staged", "split", "multall", "nodal", "tbaos", "prodsoa", "rinv")

# Arm name -> entry point in the .so. Not derivable from the arm name: `tbaos`
# is short for the ladder tables, but the kernel is set_residual_multall_aos so
# that its file sorts next to the arm it varies.
ENTRY = {
    "prod": "set_residual",
    "staged": "set_residual_staged",
    "split": "set_residual_split",
    "multall": "set_residual_multall",
    "nodal": "set_residual_nodal",
    "tbaos": "set_residual_multall_aos",
    "prodsoa": "set_residual_prod_soa",
    "rinv": "set_residual_rinv",
}

# The `multall` arm's nine face-area component arrays are grid GEOMETRY: built
# once at startup in a real port (multall sets AIX/AIR/AIT up in FIND_AREAS),
# never rebuilt per step. Splitting ember's component-first dA into them
# therefore happens outside the timed region, which is faithful, not a cheat.
# Cached per block id so a repeated build_kwargs call does not re-transpose.
_DA_SOA = {}


# This box: 20 MB L3 per socket. 48 MB is comfortably past it even when a
# rank has the socket to itself, and the buffer is per-process so contended
# runs evict for each other too -- which is what a real timestep does.
FLUSH_MB = 48
_FLUSH = None


def flush_llc():
    """Evict the LLC between timed calls, UNTIMED.

    Without this, an arm's time depends on which OTHER arms are in the
    round-robin: they run between its reps and evict its working set. That is
    not a detail -- sweeping the arm set moved `multall` vs `prod` at 1M
    contended from -15.5% (four arms) to -2.1% (two arms), a 13-point swing
    with the binary, size and rank count all fixed. Streaming a buffer past
    the LLC before every call makes every arm start from the same cold state,
    so the measurement depends on the kernel and not on its neighbours.

    Cold start is also the honest condition for the sizes we care about: a
    1M-cell block streams ~152 MB through set_residual, and production calls
    it with a whole timestep (IRS, viscous, bconds) in between.
    """
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = np.ones(FLUSH_MB * 1024 * 1024 // 4, dtype=np.float32)
    _FLUSH += 1.0


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
    aliasing two arrays into the same kernel call.
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
        # 1/r for the `rinv` arm. Built once here because r is grid
        # geometry -- rebuilt only on adapt, never per step -- so a real
        # port would build it alongside the face areas (Rule 3).
        soa["rinv"] = np.asfortranarray(1.0 / b.r_nd, dtype=b.r_nd.dtype)
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
        # NB explicit filter, not a bare **_DA_SOA splat: that cache also
        # holds `rinv`, which is another arm's input and is not in this
        # kernel's signature.
        multall=dict(
            planes=planes,
            rows=rows,
            fi=fi,
            fj=fj,
            fk=fk,
            **{k: v for k, v in _DA_SOA[key].items() if k != "rinv"},
        ),
        # `nodal` is production's own sweep with the nodal primitives read
        # instead of derived, so it takes production's 5-wide carve and kb
        # unchanged: no new scratch, no SoA geometry, nothing else to vary.
        nodal=dict(planes=planes5, rows=rows5, kb=nk - 1),
        # `tbaos` is the multall design on ember's own AoS dA, so it takes
        # the same staging scratch as `multall` (sharing rowt/rvt is safe --
        # both arms recompute them from cons on every call) but reads
        # dai/daj/dak from `common` instead of the nine SoA components.
        tbaos=dict(
            planes=planes,
            rows=rows,
            fi=fi,
            fj=fj,
            fk=fk,
            rowt=_DA_SOA[key]["rowt"],
            rvt=_DA_SOA[key]["rvt"],
        ),
        # `prodsoa` is production's kernel on the same nine component arrays:
        # production's 5-wide carve and kb, plus the SoA geometry, and no
        # staging scratch of any kind.
        prodsoa=dict(
            planes=planes5,
            rows=rows5,
            kb=nk - 1,
            **{k: v for k, v in _DA_SOA[key].items() if k.startswith("da")},
        ),
        # `rinv` is production's kernel plus one static geometry array: same
        # 5-wide carve, same kb, AoS dA untouched.
        rinv=dict(
            planes=planes5, rows=rows5, kb=nk - 1, rinv=_DA_SOA[key]["rinv"]
        ),
    )
    return common, private


def callers(b, du, dampin, active_arms=ARMS):
    """One zero-argument callable per active arm, writing into `du`."""
    import ember.fortran as F

    common, private = build_kwargs(b)
    entry = {name: getattr(F, sym, None) for name, sym in ENTRY.items()}
    out = {}
    for name in active_arms:
        fn = entry[name]
        if fn is None:
            continue
        base = dict(common)
        if name in ("multall", "prodsoa"):
            # These arms take the nine SoA components instead; only dak
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


def check_correctness(b, du, ref, active_arms=ARMS):
    """Compare each arm against production on identical input.

    dU is intent(inout) but every element is assigned before it is read, so no
    input restore is needed between arms -- `ref` only exists to prove that.
    """
    results = {}
    for dampin in (0.0, DAMPIN):
        fns = callers(b, du, dampin, active_arms)
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


if __name__ == "__main__":
    # Fast Gate-2 correctness pre-flight, standalone: build the swirled
    # state, run every arm this build exposes against production, print the
    # deviations. This is deliberately the ONLY thing this module runs
    # standalone -- for timing, use bench_prod_baseline.py.
    import argparse

    import ember.fortran as F

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ncell", type=int, default=300_000)
    args = ap.parse_args()

    active = [a for a in ARMS if a == "prod" or getattr(F, ENTRY[a], None)]
    grid, b = build_case(args.ncell)
    du = b.residual_nd
    du.flags.writeable = True
    ref = np.array(du, copy=True)

    print(f"grid {b.ni} x {b.nj} x {b.nk}  ncell={args.ncell}  cusp={b.i_cusp[0] > 0}")
    print(f"arms in this build: {active}")
    print("\ncorrectness gate (swirled state, so j/k mass fluxes are non-zero):")
    saved = swirl(b)
    try:
        check_correctness(b, du, ref, active)
    finally:
        b.conserved_nd[...] = saved
        b.update_cached_conserved()
