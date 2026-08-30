#!/usr/bin/env -S uv run
"""Clustered square-duct baseline: RK4 + IRS + multigrid, one fixed case.

Runs the duct case that the duct stability sweep calls RK4/IRS/multigrid --
``n_stage=4``, ``sf_resid=1.0``, ``fac_mgrid=0.4``, ``n_levels=3``,
``expon_mgrid=2.0``, ``sigma=0.05``, no selective-frequency damping -- and
reports convergence (energy residual, mass flow error, entropy rise). Every
one of those is a flag, so the scheme is a set of DEFAULTS, not hard-coded,
and ``--cfl`` in particular is meant to be swept. The mesh is NOT: a million
cells at ``nj=73``, ``nk=65`` are fixed in this file, so every run of this
script is the same case on the same grid.

The grid is the clustered baseline, not a uniform mesh: the cross-stream
expansion ratio is solved for by :func:`ember.cases.er_for_duct_yplus` so the
first cell off the wall lands at ``--yplus`` (30 by default, above the
sublayer edge where the wall function changes branch), rather than being
picked by hand.

Distinct from tools/run_duct.py, which is the general-purpose duct sweep
driver over a uniform-by-default grid with no BC-relaxation, per-level MG
decay, or residual-growth settings. Convergence is judged exactly as there:
exit 1 if the run diverges, exit 2 if it runs but fails the verdict, exit 0
only when it converges.
"""

import argparse
import logging
import sys
import time

import numpy as np

from ember.cases import build_duct_grid, er_for_duct_yplus
import ember.solver

logging.disable(logging.CRITICAL)  # silence per-step convergence logging

# The duct baseline's fixed mesh, deliberately not flags: nj and nk carry the
# clustered wall layer, and both cell counts are multiples of 8 as n_levels=3
# multigrid requires. ncell is unchanged from a million, so ni falls to keep
# the node count where it is.
NCELL = 1_000_000
NJ = 73
NK = 65


def run(args):
    grid_settings = dict(
        ncell=NCELL,
        nj=NJ,
        nk=NK,
        perturb_vx=args.perturb_vx,
        perturb_seed=args.perturb_seed,
        ho_frac=args.ho_frac,
        s_frac=args.s_frac,
        vx_ramp=args.vx_ramp,
    )
    # The uniform mesh is the control for anything that looks like a wall-layer
    # effect: same node count, same everything else, no clustering at all -- so
    # the y+ solve is skipped rather than solved and thrown away.
    if args.uniform:
        print("uniform cross-stream mesh (no wall clustering)")
        grid = build_duct_grid(cluster=False, **grid_settings)
    else:
        ER = er_for_duct_yplus(args.yplus, **grid_settings)
        print(f"y+={args.yplus} -> solved ER = {ER:.5f}")
        grid = build_duct_grid(cluster=True, ER=ER, **grid_settings)
    b = grid[0]
    n_nodes = b.ni * b.nj * b.nk
    print(f"Grid = {b.ni} x {b.nj} x {b.nk}  ({n_nodes} nodes)")
    print(
        f"CFL={args.cfl}, n_stage={args.n_stage}, n_levels={args.n_levels}, "
        f"fac_mgrid={args.fac_mgrid}, expon_mgrid={args.expon_mgrid}, "
        f"mgrid_pwc={args.mgrid_pwc}, "
        f"sf_resid={args.sf_resid}, sigma={args.sigma}, n_step={args.n_step}"
    )

    conf = ember.solver.Solver(
        n_step=args.n_step,
        n_step_log=args.n_step_log,
        n_step_avg=1,
        cfl=args.cfl,
        n_stage=args.n_stage,
        n_levels=args.n_levels,
        fac_mgrid=args.fac_mgrid,
        expon_mgrid=args.expon_mgrid,
        mgrid_pwc=args.mgrid_pwc,
        sf_resid=args.sf_resid,
        sf4=args.sf4,
        sf2=args.sf2,
        inviscid=args.inviscid,
        rf_inlet=args.sigma,
        rf_outlet=args.sigma,
    )

    try:
        t0 = time.perf_counter()
        hist = conf.run(grid)
        wall = time.perf_counter() - t0
    except (RuntimeError, FloatingPointError) as exc:
        print(f"Diverged ({type(exc).__name__}: {exc})")
        sys.exit(1)

    # Written BEFORE the divergence verdict, and Solver.run hands back a
    # history already trimmed to the records it logged, so a blow-up leaves a
    # readable .cnv of the steps that led into it rather than nothing at all --
    # at --n-step-log 1 that is a step-by-step account of the divergence.
    if args.write_hist:
        hist.write_cnv(args.write_hist)
        print(f"Wrote {args.write_hist}")

    # Also before the verdict, and for the same reason: the field a diverging
    # run left behind is the thing worth looking at, not a missing file.
    if args.write_emb:
        grid.write_emb(args.write_emb, compress=True)
        print(f"Wrote {args.write_emb}")

    # ember.solver.run catches a NaN blow-up internally (Grid.check_nan) and
    # breaks its step loop early rather than re-raising, so a diverged run does
    # not surface as an exception here.
    if hist.diverged:
        print(f"Diverged (after {hist.i_log + 1} convergence records)")
        sys.exit(1)
    i_step = hist.i_step
    per_node_step = wall / args.n_step / n_nodes * 1e6
    print(f"{wall:.3f}s  {per_node_step:.3f} us/node/step")

    # Convergence verdict: require the energy residual to fall 1 decade from
    # its peak; the slope criterion is disabled (slope=0).
    res_e = hist.residual[:, 4]
    decades = float(np.log10(res_e.max() / res_e[-1]))
    converged = hist.check_convergence(decay=1.0)
    print(f"Converged={converged}  (energy residual fell {decades:.2f} decades)")

    # Falling a decade is necessary but not sufficient. Multigrid on a
    # clustered grid does not fail by diverging: the residual falls, clears the
    # bar, bottoms out, and only then climbs, so a run can satisfy
    # check_convergence while being unstable -- the reason this run is long
    # enough (10000 steps by default) for such a climb to show up at all.
    growth = float(res_e[-1] / res_e.min())
    print(f"Residual growth = {growth:.2f}x above its minimum")
    if args.growth_max > 0.0 and growth > args.growth_max:
        print(
            f"Residual climbed {growth:.2f}x above its minimum "
            f"(> {args.growth_max}) -- not a converged run"
        )
        converged = False

    # Settling: where the entropy rise zeta stopped moving (within 1% of its
    # total swing), a "solution output settled" marker distinct from the
    # residual-decade bar above. wall time subtracts time[0], the startup offset.
    idx = hist.find_settling_record()
    settle_step = int(i_step[idx])
    settle_ms = float(hist.time[idx] - hist.time[0])
    print(
        f"zeta settled to <1% of range by step {settle_step} of {args.n_step} "
        f"({settle_ms:.0f} ms)"
    )

    if not converged:
        sys.exit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-step", type=int, default=10000)
    p.add_argument(
        "--n-step-log",
        type=int,
        default=25,
        help="Steps between convergence records (sets settling-step resolution)",
    )
    p.add_argument("--cfl", type=float, default=5.0)
    p.add_argument("--n-stage", type=int, default=4, help="RK stages (0 = scree)")
    p.add_argument(
        "--n-levels", type=int, default=3, help="MG coarse levels (0 = no MG)"
    )
    p.add_argument(
        "--fac-mgrid", type=float, default=0.4, help="MG coarse correction fraction"
    )
    p.add_argument(
        "--expon-mgrid",
        type=float,
        default=2.0,
        help="Per-level MG decay base (coef_l ~ expon_mgrid**-(l-1)); the "
        "factor-2 per level the duct multigrid results were established at, "
        "not Solver's own default of 1.414",
    )
    p.add_argument(
        "--mgrid-pwc",
        action="store_true",
        help="Prolong the coarse correction by plain injection (piecewise "
        "constant over each coarse block) instead of the cascaded trilinear "
        "interpolation. Not comparable to the cascade at matched --fac-mgrid: "
        "injection does not attenuate the mid-band, so its optimum sits lower "
        "-- sweep each and compare the best of the two curves",
    )
    p.add_argument(
        "--sf-resid", type=float, default=1.0, help="IRS residual smoothing factor"
    )
    p.add_argument(
        "--sf4",
        type=float,
        default=0.008,
        help="Fourth-difference artificial dissipation, scaled by cfl "
        "(the filter is unstable above sf4*cfl = 1/16)",
    )
    p.add_argument(
        "--sf2",
        type=float,
        default=0.002,
        help="Second-difference artificial dissipation, scaled by cfl "
        "(the filter is unstable above sf2*cfl = 1/4)",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Under-relaxation factor on the inlet/outlet NonReflectingPatch "
        "characteristic BC correction -- sets both Solver.rf_inlet and "
        "Solver.rf_outlet",
    )
    p.add_argument(
        "--growth-max",
        type=float,
        default=2.0,
        metavar="F",
        help="Fail a run whose energy residual ends more than this factor "
        "above its own minimum, even when it cleared the 1-decade bar -- the "
        "signature of a run that bottomed out and then climbed. Exits 2, as "
        "for any non-converged run. Pass 0 to judge on the decade bar alone",
    )
    p.add_argument(
        "--yplus",
        type=float,
        default=30.0,
        help="Wall-unit distance of the first node off the wall; the "
        "clustering expansion ratio is solved for to hit it",
    )
    p.add_argument(
        "--uniform",
        action="store_true",
        help="Space the cross-stream mesh uniformly instead of clustering it "
        "to the walls; --yplus is then unused",
    )
    p.add_argument("--inviscid", action="store_true", help="Disable viscous terms")
    p.add_argument(
        "--perturb-vx", type=float, default=0.01, help="Axial-velocity ripple amplitude"
    )
    p.add_argument("--perturb-seed", type=int, default=0, help="Velocity-ripple seed")
    p.add_argument(
        "--ho-frac", type=float, default=0.01, help="Stagnation-enthalpy IC offset"
    )
    p.add_argument("--s-frac", type=float, default=0.01, help="Entropy IC offset")
    p.add_argument(
        "--vx-ramp",
        type=float,
        default=0.01,
        help="Streamwise Vx ramp (outlet vs inlet)",
    )
    p.add_argument(
        "--write-emb",
        metavar="EMBFILE",
        help="Write the marched grid to this EMB file (read back with "
        "Grid.read_emb), diverged or not",
    )
    p.add_argument(
        "--write-hist",
        metavar="CNVFILE",
        help="Write the convergence history to this CNV file (read back with "
        "ConvergenceHistory.read_cnv)",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
