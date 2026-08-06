#!/usr/bin/env -S uv run
"""Square duct throttled to a target mass flow: closed-loop convergence.

Drives ``ember.outlet.OutletPatch.set_throttle`` end to end. The duct is built
by ``ember.cases.build_duct_grid`` and would naturally pass some mass flow at
the prescribed exit pressure; this script measures that flow, asks the outlet
for a fraction of it, and reports whether the controller gets there.

The point of the exercise is that the gains are dimensionless and should not
need touching: --Kp and --Ki exist to show what happens when they are, not
because a case is expected to need them. See ``OutletPatch.set_throttle`` for
why Kp = 1 is the notional Newton step and the defaults are half of it.

Too slow for the test suite, which covers the controller arithmetic directly;
this is the one that marches.
"""

import argparse
import logging
import sys

import numpy as np

from ember import average
from ember.cases import build_duct_grid
import ember.solver

logging.disable(logging.CRITICAL)  # silence per-step convergence logging


def run(args):
    grid = build_duct_grid(args.ncell, cluster=args.cluster)
    b = grid[0]
    outlet = grid.patches.outlet[0]

    # The flow the duct passes at the prescribed exit pressure, before the
    # throttle touches it. Per passage, which is what the throttle controls on.
    mdot_natural = float(average.flow_mass(outlet.block_view.squeeze()))
    mdot_target = args.mdot_frac * mdot_natural
    outlet.set_throttle(mdot_target, Kp=args.Kp, Ki=args.Ki)

    print(f"Grid = {b.ni} x {b.nj} x {b.nk}")
    print(
        f"mdot natural = {mdot_natural:.5f} kg/s, target = {mdot_target:.5f} kg/s "
        f"({args.mdot_frac:.3f} of natural)"
    )
    print(f"Kp={args.Kp}, Ki={args.Ki}")
    print(
        f"cfl={args.cfl}, n_stage={args.n_stage}, n_levels={args.n_levels}, "
        f"fac_mgrid={args.fac_mgrid}, sf_resid={args.sf_resid}, n_step={args.n_step}"
    )

    conf = ember.solver.Solver(
        n_step=args.n_step,
        n_step_log=args.n_step_log,
        n_step_avg=1,
        cfl=args.cfl,
        n_stage=args.n_stage,
        n_levels=args.n_levels,
        fac_mgrid=args.fac_mgrid,
        sf_resid=args.sf_resid,
        inviscid=args.inviscid,
    )

    try:
        hist = conf.run(grid)
    except (RuntimeError, FloatingPointError) as exc:
        print(f"Diverged ({type(exc).__name__}: {exc})")
        sys.exit(1)

    if hist.diverged:
        print(f"Diverged (after {hist.i_log + 1} convergence records)")
        sys.exit(1)

    i_step = hist.i_step
    # Columns of the compound property: setpoint, measurement, total correction.
    mdot = np.asarray(hist.throttle[:, 1], dtype=float)
    dP = np.asarray(hist.throttle[:, 2], dtype=float)
    err = mdot / mdot_target - 1.0

    print(f"mdot final   = {mdot[-1]:.5f} kg/s  ({err[-1] * 100:+.3f}% of target)")
    print(f"dP throttle  = {dP[-1]:+.1f} Pa on {float(outlet.P_nd.mean()):.4f} nondim")

    # The controller has done its job when the standing error is gone; the
    # tolerance is on the mass flow, not on the residual, since a march can be
    # well converged at the wrong operating point.
    hit = abs(err[-1]) < args.tol
    print(f"Reached target within {args.tol * 100:.2f}%: {hit}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_m, ax_p, ax_r) = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True)

        ax_m.axhline(0.0, color="0.6", lw=0.8)
        ax_m.plot(i_step, err * 100.0, marker=".", ms=3, lw=1.0)
        ax_m.set_ylabel(r"$\dot m / \dot m_\mathrm{target} - 1$ [\%]")
        ax_m.set_title("Mass flow error against the throttle setpoint")
        ax_m.grid(True, alpha=0.3)

        ax_p.plot(i_step, dP, marker=".", ms=3, lw=1.0, label="total")
        ax_p.plot(i_step, hist.dP_P, lw=1.0, ls="--", label=r"$K_p$ term")
        ax_p.plot(i_step, hist.dP_I, lw=1.0, ls=":", label=r"$K_i$ term")
        ax_p.set_ylabel(r"$\Delta p_\mathrm{throttle}$ [Pa]")
        ax_p.set_title("Pressure correction the controller is holding")
        ax_p.grid(True, alpha=0.3)
        ax_p.legend()

        ax_r.semilogy(i_step, hist.residual[:, 4], marker=".", ms=3, lw=1.0)
        ax_r.set_ylabel(r"$|\Delta(\rho e)|$")
        ax_r.set_title("Energy residual (semilog)")
        ax_r.set_xlabel("i_step")
        ax_r.grid(True, which="both", alpha=0.3)

        fig.suptitle(
            f"Throttled to {args.mdot_frac:.3f} of natural flow, "
            f"Kp={args.Kp}, Ki={args.Ki}, {args.n_step} steps",
            y=0.995,
        )
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"Wrote {args.plot}")

    if args.write_hist:
        hist.write_cnv(args.write_hist)
        print(f"Wrote {args.write_hist}")

    if not hit:
        sys.exit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mdot-frac",
        type=float,
        default=0.95,
        help="Target mass flow as a fraction of what the duct passes untouched",
    )
    p.add_argument("--Kp", type=float, default=0.5, help="Proportional gain (-)")
    p.add_argument("--Ki", type=float, default=0.002, help="Integral gain (-)")
    p.add_argument(
        "--tol",
        type=float,
        default=0.005,
        help="Fractional mass flow error counted as having reached the target",
    )
    p.add_argument("--n-step", type=int, default=1000)
    p.add_argument("--n-step-log", type=int, default=25)
    p.add_argument("--n-stage", type=int, default=4)
    p.add_argument("--cfl", type=float, default=5.0)
    p.add_argument(
        "--n-levels", type=int, default=2, help="MG coarse levels (0 = no MG)"
    )
    p.add_argument(
        "--fac-mgrid", type=float, default=0.2, help="MG coarse correction fraction"
    )
    p.add_argument(
        "--sf-resid", type=float, default=1.0, help="IRS residual smoothing factor"
    )
    p.add_argument("--ncell", type=int, default=500000, help="Target cell count")
    p.add_argument("--inviscid", action="store_true", help="Disable viscous terms")
    p.add_argument(
        "--cluster",
        action="store_true",
        help="Cluster the cross-stream mesh towards the walls (default: uniform)",
    )
    p.add_argument(
        "--plot", metavar="PATH", help="Write a 3-panel PDF figure to this path"
    )
    p.add_argument(
        "--write-hist",
        metavar="CNVFILE",
        help="Write the convergence history to this CNV file",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
