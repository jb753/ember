"""Arm definitions for the viscous study: `set_visc_force` and `set_tau_q_soa`.

The viscous kernels (src/ember/_fortran/viscous.f90) carry a long adopted
history in bench/README.md -- k-slab blocking, rolling-buffer fusion -- but
none of it was ever measured on THIS harness: every tracked result in
bench/results/ is a `set_residual` or IRS number. This module is the viscous
counterpart of `residual_arms.py` and `irs_arms.py`, so
`bench_prod_baseline.py --kernel visc` (or `--kernel tauq`) can time them under
the protocol the README mandates: one arm per process, rank-barriered before
every call, replicated at the launch.

Two kernels, timed separately because they are separate calls with a grid-wide
periodic halo exchange between them (grid.py's update_sources):

  tauq    set_tau_q_soa    -- phase 1, cell stress tensor and heat flux
  visc    set_visc_force   -- phase 2, face fluxes accumulated into fvisc

The `visc` arm set is where the fvisc-fusion study lives: production touches
fvisc four times (the i-sweep assigns it, the j- and k-sweeps read-modify-write
it, the trailing polar-source pass RMWs component 2), and the rolling buffers
needed to collapse that to a single store already exist in the kernel.

WHY REPEATED CALLS NEED A RESTORE HOOK. Unlike `set_residual`, `set_visc_force`
is NOT idempotent: its entry pass scales the tau/q halo slots by (2*wall - 1)
IN PLACE, so rep n reads rep n-1's halos. On the duct case this happens to be
harmless twice over, and both reasons are accidents of the case rather than
properties of the kernel:

  - block.ijk_wall_visc casts a boolean, so the mask is exactly 0.0 or 1.0 and
    the factor is exactly -1 or +1: reps flip a sign, they do not decay. A
    blended mask (0 < wall < 1) would contract the halos geometrically toward
    subnormal, which IS a timing bias -- the hazard irs_arms.check_denormals
    exists for.
  - the flip is invisible in fvisc anyway, because the halo tau/q only reaches
    the boundary faces of the wall-adjacent cells, and those cells are
    multiplied by the same zero mask at the end of the kernel. Verified: four
    consecutive calls with no restore are bitwise identical here.

So `halo_restorer` is insurance, not a repair. It snapshots the six halo faces
(O(surface), not O(volume): ~0.4% of the buffer at 1M cells, microseconds
against a call of tens of milliseconds) and restores them before each call, so
every rep does bitwise-identical work on bitwise-identical input regardless of
which case is being run. The driver calls it OUTSIDE the timed window and
before the barrier.

fvisc itself needs no restore: the i-direction sweep assigns every cell before
anything reads it, so each call re-initialises its own output.
"""

import numpy as np

from ember import util
from ember.grid import _KB_SLAB

from residual_arms import build_case, swirl  # noqa: F401  (re-export)

VISC_ARMS = ("visc", "viscij", "viscijk", "viscpol", "viscpol2")
TAUQ_ARMS = ("tauq",)

ENTRY = {
    "visc": "set_visc_force",
    # The fvisc-fusion ladder (bench/subroutines/viscous_fused.f90). Each arm
    # removes one more visit to fvisc; all three sum each cell's contributions
    # in production's order and so should gate bitwise.
    "viscij": "set_visc_force_ij",  # i fused into j:   4 touches -> 3
    "viscijk": "set_visc_force_ijk",  # k fused in too:   4 touches -> 2
    "viscpol": "set_visc_force_pol",  # polar fused too:  4 touches -> 1
    # viscpol with the i=1/i=ni-1 sheet moved out of the O(surface)
    # pass and into the fused store, where it is unit-stride.
    "viscpol2": "set_visc_force_pol2",
    "tauq": "set_tau_q_soa",
}


def seed_tau_q(grid, b):
    """Run phase 1 so tau/q hold a real field for phase 2, once, untimed.

    Rule 3: a real port evaluates set_tau_q_soa immediately before
    set_visc_force every step, so tau/q are emphatically not a hoistable
    input -- but they are the PREVIOUS kernel's output, not this one's, and
    pricing them inside the set_visc_force window would time set_tau_q_soa.
    Built once here, exactly as grid.py's update_sources builds them, including
    the periodic seam exchange that runs between the two phases.

    Must be called after swirl(): tau/q are a function of the state, so seeding
    them before the state is perturbed would leave phase 2 reading a tau/q
    field that does not belong to the velocities it also reads.
    """
    import ember.fortran as F

    halo = b.tau_q_halo
    F.set_tau_q_soa(
        cons=b.conserved_nd,
        t=b.T_nd,
        mu=b.mu_nd,
        cp=b.cp_nd,
        pr_lam=b.fluid._Pr,
        pr_turb=1.0,
        xlength=b.xlen_sq_nd,
        vol=b.vol_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        r=b.r_nd,
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_rel_nd,
        tau_cell=halo[..., 0:6],
        q_cell=halo[..., 6:9],
        mu_turb=b._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True),
    )
    b._versions["mu_turb"] += 1
    grid.connectivity.periodic.exchange_halos()
    return halo


def halo_restorer(b):
    """Snapshot the tau/q halo faces; return a zero-argument restore callable.

    See the module docstring for why this exists. Only the six outer faces of
    the (ni+1, nj+1, nk+1, 9) buffer are ever touched by set_visc_force's entry
    scaling, so this is O(surface): at 1M cells it is ~0.4% of the buffer and
    a few microseconds against a call of tens of milliseconds.
    """
    halo = b.tau_q_halo
    ni, nj, nk = b.shape
    faces = (
        (np.s_[0, :, :, :], None),
        (np.s_[ni, :, :, :], None),
        (np.s_[:, 0, :, :], None),
        (np.s_[:, nj, :, :], None),
        (np.s_[:, :, 0, :], None),
        (np.s_[:, :, nk, :], None),
    )
    saved = [(sl, np.array(halo[sl], copy=True)) for sl, _ in faces]

    def restore():
        for sl, val in saved:
            halo[sl] = val

    return restore


def callers_visc(b, active_arms=VISC_ARMS):
    """One zero-argument callable per active set_visc_force arm."""
    import ember.fortran as F

    ni, nj, nk = b.shape
    halo = b.tau_q_halo
    i_cusp_start, i_cusp_end = b.i_cusp
    # Rolling face-flow buffers carved from block.scratch exactly as grid.py
    # does -- NOT from tau_q_halo, which is this kernel's tau/q input and
    # whose docstring forbids aliasing a second array into the same call.
    planes, rows = util.carve_view(b.scratch, (ni, nj, 4, 2), (ni, 4, 3))
    kw = dict(
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
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_rel_nd,
        tau_cell=halo[..., 0:6],
        q_cell=halo[..., 6:9],
        planes=planes,
        rows=rows,
        kb=min(_KB_SLAB, nk - 1),
        **b.ijk_wall_visc,
        **b.Omega_wall_nd,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
    )
    # F_body_nd is served read-only; the kernel's fvisc dummy is its
    # components 2-5 (mass excluded), a contiguous trailing slice.
    b.F_body_nd.flags.writeable = True
    fvisc = b.F_body_nd[..., 1:]

    out = {}
    for name in active_arms:
        fn = getattr(F, ENTRY[name], None)
        if fn is None:
            continue
        out[name] = lambda fn=fn: fn(fvisc=fvisc, **kw)
    return out


def callers_tauq(b, active_arms=TAUQ_ARMS):
    """One zero-argument callable per active set_tau_q_soa arm.

    Phase 1 is idempotent (every owned cell and every halo slot it reads is
    assigned before it is read), so it needs no restore hook.
    """
    import ember.fortran as F

    halo = b.tau_q_halo
    mu_turb = b._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True)
    kw = dict(
        cons=b.conserved_nd,
        t=b.T_nd,
        mu=b.mu_nd,
        cp=b.cp_nd,
        pr_lam=b.fluid._Pr,
        pr_turb=1.0,
        xlength=b.xlen_sq_nd,
        vol=b.vol_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        r=b.r_nd,
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_rel_nd,
        tau_cell=halo[..., 0:6],
        q_cell=halo[..., 6:9],
        mu_turb=mu_turb,
    )
    out = {}
    for name in active_arms:
        fn = getattr(F, ENTRY[name], None)
        if fn is None:
            continue
        out[name] = lambda fn=fn: fn(**kw)
    return out


def check_correctness(b, active_arms=VISC_ARMS):
    """Compare each set_visc_force arm against production on identical input.

    The input restore is mandatory here, not decorative: without it arm n
    would see the halo signs arm n-1 left behind (module docstring), and every
    arm after the first would be gated against a different tau/q field.
    """
    restore = halo_restorer(b)
    fns = callers_visc(b, active_arms)
    fvisc = b.F_body_nd

    restore()
    fns["visc"]()
    base = np.array(fvisc, copy=True)
    scale = float(np.abs(base).max())

    results = {}
    for name, fn in fns.items():
        if name == "visc":
            continue
        restore()
        fn()
        got = np.array(fvisc, copy=True)
        diff = np.abs(got - base)
        # Ulps AT THE FIELD SCALE, not pointwise. fvisc is a small difference
        # of large face flows, so a cell whose own value is ~1e-15 has an
        # enormous pointwise ulp count for an absolute deviation that is
        # nothing -- an early version of this gate reported 131072 ulps for a
        # deviation of 1 ulp of the largest value in the field. The README's
        # rule is to quantify in ulps of the quantities being DIFFERENCED; the
        # face flows are not visible from here, and they are larger than the
        # residual, so ulps of max|fvisc| is the conservative stand-in.
        results[name] = dict(
            bitwise=bool(np.array_equal(got, base)),
            max_abs=float(diff.max()),
            rel=float(diff.max() / scale) if scale else 0.0,
            max_ulp=float(diff.max() / np.spacing(np.float32(scale))) if scale else 0.0,
        )
    return results


def main():
    """Standalone Gate-2 correctness pre-flight. Times nothing."""
    import argparse

    import ember.fortran as F

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--ncell", type=int, default=300_000)
    args = ap.parse_args()

    grid, b = build_case(args.ncell)
    ni, nj, nk = b.shape
    active = [a for a in VISC_ARMS if a == "visc" or getattr(F, ENTRY[a], None)]
    print(f"grid {ni} x {nj} x {nk}  ncell={args.ncell}  cusp={b.i_cusp[0] > 0}")
    print(f"arms in this build: {active}")

    # Rule 5: build_duct_grid is axially straight, so Vr = Vt = 0 and the
    # wall function's Vt_slip, tau(6)'s swirl term and the polar source's
    # rho*Vt^2 are all degenerate. Swirl before seeding tau/q, since tau/q
    # are a function of the state.
    swirl(b)
    seed_tau_q(grid, b)

    print("\ncorrectness gate (swirled state, so the swirl terms are non-zero):")
    bad = 0
    for name, r in check_correctness(b, active).items():
        ok = "BITWISE" if r["bitwise"] else f"DIFFERS max {r['max_abs']:.3e}"
        print(f"  {name:>8}  {ok}  ({r['rel']:.3e} of scale, {r['max_ulp']:.2f} ulp)")
        bad += not r["bitwise"]
    return 1 if bad else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
