#!/usr/bin/env -S uv run
"""Square-duct baseline: configurable CFL / MG / IRS run.

Runs the square-duct case and reports convergence (energy residual, mass flow
error, entropy rise). The case itself is assembled by
``ember.cases.build_duct_grid``; this script is a thin CLI wrapper that drives
it and the solver. With --plot writes a 3-panel figure to the given path.
"""

import argparse
import logging
import sys
import time

from ember.cases import build_duct_grid
import ember.solver

logging.disable(logging.CRITICAL)  # silence per-step convergence logging


def run(args):
    grid = build_duct_grid(
        args.ncell,
        cluster=args.cluster,
        ER=args.ER,
        perturb_vx=args.perturb_vx,
        perturb_seed=args.perturb_seed,
        ho_frac=args.ho_frac,
        s_frac=args.s_frac,
        vx_ramp=args.vx_ramp,
    )
    b = grid[0]
    n_nodes = b.ni * b.nj * b.nk
    print(f"Grid = {b.ni} x {b.nj} x {b.nk}  ({n_nodes} nodes)")
    print(
        f"CFL={args.cfl}, n_stage={args.n_stage}, n_levels={args.n_levels}, "
        f"fac_mgrid={args.fac_mgrid}, sf_resid={args.sf_resid}, n_step={args.n_step}"
    )

    conf = ember.solver.SolverConfig(
        n_step=args.n_step,
        n_step_log=100,
        n_step_avg=1,
        cfl=args.cfl,
        n_stage=args.n_stage,
        n_levels=args.n_levels,
        fac_mgrid=args.fac_mgrid,
        sf_resid=args.sf_resid,
        inviscid=args.inviscid,
    )

    try:
        t0 = time.perf_counter()
        hist = ember.solver.run(grid, conf)
        wall = time.perf_counter() - t0
    except (RuntimeError, FloatingPointError) as exc:
        print(f"Diverged ({type(exc).__name__}: {exc})")
        sys.exit(1)

    # ember.solver.run catches a NaN blow-up internally (Grid.check_nan) and
    # breaks its step loop early rather than re-raising, so a diverged run does
    # not surface as an exception here.
    if hist.diverged:
        print(f"Diverged (after {hist.i_log + 1} convergence records)")
        sys.exit(1)
    i_step = hist.i_step
    per_node_step = wall / args.n_step / n_nodes * 1e6
    print(f"{wall:.3f}s  {per_node_step:.3f} us/node/step")

    if not args.plot:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_res, ax_err, ax_s) = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True)

    ax_res.semilogy(i_step, hist.residual[:, 4], marker=".", ms=3, lw=1.0)
    ax_res.set_ylabel(r"$|\Delta(\rho e)|$")
    ax_res.set_title("Energy residual (semilog)")
    ax_res.grid(True, which="both", alpha=0.3)

    ax_err.axhline(0.0, color="0.6", lw=0.8)
    ax_err.plot(i_step, hist.err_mdot, marker=".", ms=3, lw=1.0)
    ax_err.set_ylabel(r"$(\dot m_\mathrm{out} - \dot m_\mathrm{in}) / \bar{\dot m}$")
    ax_err.set_title("Mass flow error")
    ax_err.grid(True, alpha=0.3)

    ax_s.plot(i_step, hist.zeta, marker=".", ms=3, lw=1.0)
    ax_s.set_ylabel(r"$\zeta = s_\mathrm{out} - s_\mathrm{in}$")
    ax_s.set_title("Entropy rise")
    ax_s.set_xlabel("i_step")
    ax_s.grid(True, alpha=0.3)

    fig.suptitle(
        f"CFL={args.cfl}, {args.n_stage}-stage RK, n_levels={args.n_levels}, "
        f"fac_mgrid={args.fac_mgrid}, {args.n_step} steps",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(args.plot)
    print(f"Wrote {args.plot}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-step", type=int, default=100)
    p.add_argument("--n-stage", type=int, default=4)
    p.add_argument("--cfl", type=float, default=3.0)
    p.add_argument(
        "--n-levels", type=int, default=0, help="MG coarse levels (0 = no MG)"
    )
    p.add_argument(
        "--fac-mgrid", type=float, default=0.2, help="MG coarse correction fraction"
    )
    p.add_argument(
        "--sf-resid", type=float, default=0.0, help="IRS residual smoothing factor"
    )
    p.add_argument("--ncell", type=int, default=int(1e6), help="Target cell count")
    p.add_argument("--inviscid", action="store_true", help="Disable viscous terms")
    p.add_argument(
        "--cluster",
        action="store_true",
        help="Cluster the cross-stream mesh towards the walls (default: uniform)",
    )
    p.add_argument(
        "--ER", type=float, default=1.05, help="Wall-clustering expansion ratio"
    )
    p.add_argument(
        "--perturb-vx", type=float, default=0.01, help="Axial-velocity ripple amplitude"
    )
    p.add_argument("--perturb-seed", type=int, default=0, help="Velocity-ripple seed")
    p.add_argument(
        "--ho-frac", type=float, default=0.01, help="Stagnation-enthalpy IC offset"
    )
    p.add_argument("--s-frac", type=float, default=0.01, help="Entropy IC offset")
    p.add_argument(
        "--vx-ramp", type=float, default=0.01, help="Streamwise Vx ramp (outlet vs inlet)"
    )
    p.add_argument("--plot", metavar="PATH", help="Write 3-panel figure to this path")
    run(p.parse_args())


if __name__ == "__main__":
    main()
