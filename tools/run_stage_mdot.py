#!/usr/bin/env -S uv run
"""Mixing-plane mass flow forcing: does it converge faster?

Marches the two-row case of ``ember.cases.build_stage_grid`` twice from the same
off-design start -- once with ``Solver.gain_mdot`` at zero and once at the gain
under test -- and reports how many steps each took to close the mass flow
transient. The two rows begin at different mass flows, which is the state a
mixing plane is otherwise slow to clear: without forcing the plane can only find
the common level by integrating the flux mismatch at ``rf_exchange``, and the
information has to convect the length of a row to do it.

The forcing is Holmes (2008) Eq. 20-22, with the target taken as the mean of the
inlet and outlet mass flows rather than prescribed. It ramps itself out below
``MixingCommunicator.eps_deadband``, so both runs must converge to the same
answer; the script checks that as well as the step counts, because a
"convergence acceleration" that moves the answer is not one.

Too slow for the test suite, which covers the arithmetic directly; this is the
one that marches.
"""

import argparse
import logging
import sys

import numpy as np

from ember.cases import build_stage_grid
import ember.solver

logging.disable(logging.CRITICAL)  # silence per-step convergence logging


def mix_gap(hist):
    """Fractional mass flow mismatch across the mixing plane, per record.

    Stations run ``[row0_up, row0_dn, row1_up, row1_dn]``, so the two sides of
    the single plane are stations 1 and 2. This is what
    ``ConvergenceHistory.format_message`` prints as the mix error; it is not
    exposed as a property, so form it here.
    """
    mdot = hist.mdot_nd
    up, dn = mdot[:, 1], mdot[:, 2]
    return (dn - up) / (0.5 * (up + dn))


def steps_to(i_step, series, tol):
    """First step at which ``|series|`` falls below ``tol`` and stays there.

    Returns None if it never does, which is a legitimate outcome and is reported
    as such rather than being papered over with the last value.
    """
    below = np.abs(series) < tol
    if not below.any() or not below[-1]:
        return None
    # Walk back from the end to find where it last came good, so a curve that
    # dips through the tolerance early and comes back out is not credited.
    i = len(below) - 1
    while i > 0 and below[i - 1]:
        i -= 1
    return int(i_step[i])


def march(args, gain):
    """Build a fresh grid and march it at the given forcing gain."""
    grid = build_stage_grid(
        args.ncell,
        cluster=args.cluster,
        rhoVm_frac_up=args.frac_up,
        rhoVm_frac_dn=args.frac_dn,
        Alpha=args.alpha,
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
        rf_exchange=args.rf_exchange,
        lead_exchange=args.lead,
        gain_mdot=gain,
    )
    try:
        hist = conf.run(grid)
    except (RuntimeError, FloatingPointError) as exc:
        print(f"Diverged at gain_mdot={gain} ({type(exc).__name__}: {exc})")
        sys.exit(1)
    if hist.diverged:
        print(f"Diverged at gain_mdot={gain} (after {hist.i_log + 1} records)")
        sys.exit(1)
    return grid, hist


def run(args):
    grid = build_stage_grid(
        args.ncell,
        cluster=args.cluster,
        rhoVm_frac_up=args.frac_up,
        rhoVm_frac_dn=args.frac_dn,
        Alpha=args.alpha,
    )
    shapes = " + ".join("x".join(str(n) for n in b.shape) for b in grid)
    print(f"Grid = {shapes}  (Nb = {[b.Nb for b in grid]})")
    print(
        f"Initial mass flow: upstream row {args.frac_up:.3f} of design, "
        f"downstream row {args.frac_dn:.3f}"
    )
    print(
        f"cfl={args.cfl}, n_stage={args.n_stage}, n_levels={args.n_levels}, "
        f"n_step={args.n_step}, gain_mdot={args.gain}"
    )

    results = {}
    for label, gain in (("off", 0.0), ("on", args.gain)):
        grid, hist = march(args, gain)
        results[label] = (grid, hist)

    print(f"\n{'':14s}{'off':>14s}{'on':>14s}")
    rows = []
    for label in ("off", "on"):
        hist = results[label][1]
        i_step = hist.i_step
        rows.append(
            (
                steps_to(i_step, mix_gap(hist), args.tol),
                steps_to(i_step, hist.err_mdot, args.tol),
                float(np.abs(mix_gap(hist))[-1]),
                float(np.abs(hist.err_mdot)[-1]),
                float(hist.residual[-1, 4]),
            )
        )
    names = [
        f"mix gap < {args.tol:g}",
        f"err_mdot < {args.tol:g}",
        "final |mix gap|",
        "final |err_mdot|",
        "final residual",
    ]
    fmt = lambda v: ("never" if v is None else f"{v}") if isinstance(v, (int, type(None))) else f"{v:.3e}"
    for i, name in enumerate(names):
        print(f"{name:14s}{fmt(rows[0][i]):>14s}{fmt(rows[1][i]):>14s}")

    # A faster path to the same answer is the claim; check the second half of it
    # too. The forcing is zero below eps_deadband, so the two runs must agree on
    # the converged mass flow to well within that.
    #
    # Only meaningful once both runs have actually got there. Comparing two
    # unconverged states says nothing about where either is heading, and a run
    # that is merely further along its transient would otherwise be reported as
    # having moved the answer.
    m_off = float(results["off"][1].mdot_nd[-1, -1])
    m_on = float(results["on"][1].mdot_nd[-1, -1])
    drift = abs(m_on - m_off) / abs(m_off)
    settled = rows[0][0] is not None and rows[1][0] is not None
    verdict = f"differ by {drift:.2e}" if settled else "neither run settled yet"
    print(f"\nExit mass flow off={m_off:.6f}  on={m_on:.6f}  ({verdict})")

    if settled and rows[1][0] > 0:
        print(f"Steps to close the mixing-plane gap: {rows[0][0]/rows[1][0]:.2f}x faster")
    elif rows[1][0] is not None:
        print("Only the forced run closed the gap within n_step.")
    else:
        print("Neither run closed the gap within n_step; march for longer.")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_g, ax_m, ax_r) = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True)
        styles = {"off": dict(ls="--", color="0.45"), "on": dict(ls="-", color="C0")}

        for label in ("off", "on"):
            hist = results[label][1]
            i_step = hist.i_step
            name = f"gain_mdot = {0.0 if label == 'off' else args.gain}"
            ax_g.semilogy(
                i_step, np.abs(mix_gap(hist)), lw=1.2, label=name, **styles[label]
            )
            ax_m.semilogy(
                i_step, np.abs(hist.err_mdot), lw=1.2, label=name, **styles[label]
            )
            ax_r.semilogy(
                i_step, hist.residual[:, 4], lw=1.2, label=name, **styles[label]
            )

        ax_g.axhline(args.tol, color="0.8", lw=0.8)
        ax_g.set_ylabel(r"$|\dot m_\mathrm{dn} - \dot m_\mathrm{up}| / \bar{\dot m}$")
        ax_g.set_title("Mass flow mismatch across the mixing plane")

        ax_m.axhline(args.tol, color="0.8", lw=0.8)
        ax_m.set_ylabel(r"$|\dot m_\mathrm{out} - \dot m_\mathrm{in}| / \bar{\dot m}$")
        ax_m.set_title("Mass flow mismatch across the whole machine")

        ax_r.set_ylabel(r"$|\Delta(\rho e)|$")
        ax_r.set_title("Energy residual")
        ax_r.set_xlabel("i_step")

        for ax in (ax_g, ax_m, ax_r):
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

        fig.suptitle(
            f"Mixing-plane mass flow forcing: rows started at "
            f"{args.frac_up:.2f} and {args.frac_dn:.2f} of design",
            y=0.995,
        )
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"Wrote {args.plot}")

    if args.write_hist:
        for label in ("off", "on"):
            path = args.write_hist.replace(".cnv", f"_{label}.cnv")
            results[label][1].write_cnv(path)
            print(f"Wrote {path}")

    if settled and drift > args.tol:
        print("Forcing moved the converged answer; that is a failure, not a speed-up.")
        sys.exit(2)
    if not settled:
        sys.exit(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gain", type=float, default=0.5, help="Forcing gain under test (-)"
    )
    p.add_argument(
        "--frac-up",
        type=float,
        default=0.75,
        help="Upstream row initial mass flow, as a fraction of design",
    )
    p.add_argument(
        "--frac-dn",
        type=float,
        default=1.15,
        help="Downstream row initial mass flow, as a fraction of design",
    )
    p.add_argument(
        "--alpha", type=float, default=0.0, help="Yaw angle through both rows [deg]"
    )
    p.add_argument(
        "--tol",
        type=float,
        default=0.005,
        help="Fractional mass flow error counted as closed",
    )
    p.add_argument("--n-step", type=int, default=2000)
    p.add_argument("--n-step-log", type=int, default=10)
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
    p.add_argument(
        "--rf-exchange",
        type=float,
        default=0.01,
        help="Cross-plane exchange relaxation, applied to both arms",
    )
    p.add_argument(
        "--lead",
        type=float,
        default=0.0,
        help="Phase lead on the exchange in steps, applied to both arms",
    )
    p.add_argument("--ncell", type=int, default=200000, help="Target cell count")
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
        help="Write both convergence histories, suffixed _off and _on",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
