#!/usr/bin/env -S uv run
"""Inlet reflectivity: domain-truncation study, non-reflecting vs standard inlet.

Measures the thing a steady non-reflecting inflow condition is actually for --
being able to put the inflow plane close to a blade row without corrupting the
solution near it.

A pitchwise-varying static pressure at the outlet stands in for the potential
field of a downstream blade row. In subsonic flow its harmonics are evanescent:
the wave parameter of Saxer and Giles (their Eq. 22) is imaginary, so each
pitchwise harmonic decays upstream over roughly a pitch divided by two pi, and a
disturbance seeded further upstream than that never reaches the inflow plane at
all. The inlet condition therefore only matters when the plane sits within a
few decay lengths of the source, which is exactly the regime this script sets
up: the same physical problem is solved on a short domain and on a long one,
and the two solutions are compared over their common region. A perfectly
non-reflecting inlet gives the same answer on both.

Note that the *pitchwise mean* is prescribed by any inlet that fixes stagnation
quantities and flow angles, so the ``m = 0`` mode is reflecting by construction
for both conditions here -- Saxer and Giles make this point in their Section
III. Only the harmonics are absorbed, which is why the source below is purely
harmonic.

Not part of ``make test``: it drives a few thousand solver steps per
configuration.

Examples
--------
::

    uv run tools/run_nrbc_reflection.py
    uv run tools/run_nrbc_reflection.py --sigma 0.05 0.1 0.2 --plot nrbc.png
"""

import argparse
import logging

import numpy as np

import ember.block
import ember.fluid
import ember.grid
import ember.patch
import ember.solver
from ember import util

logging.disable(logging.CRITICAL)  # silence per-step convergence logging

_P = 1.0e5
_T = 300.0
_VX = 100.0
_R_HUB = 2.0
_SPAN = 0.125
_NB = int(np.round(2.0 * np.pi * _R_HUB / _SPAN))
_PITCH = 2.0 * np.pi / _NB
_ARC = _R_HUB * _PITCH  # pitch as a length, the natural axial decay scale


def build_grid(n_pitch, inlet_kind, amp, sigma, dx_frac=0.04, nj=9, nk=25):
    """Uniform duct ending at x = 0 with a pitchwise-varying exit pressure.

    Parameters
    ----------
    n_pitch : float
        Domain length in pitches; the inflow plane sits at ``x = -n_pitch*arc``.
    inlet_kind : str
        ``"standard"`` for :class:`~ember.inlet.InletPatch`, ``"nrbc"`` for
        :class:`~ember.inlet_nonreflecting.NonReflectingInletPatch`.
    amp : float
        Relative amplitude of the exit static pressure harmonic.
    sigma : float
        Relaxation factor for the non-reflecting inlet.
    """
    length = n_pitch * _ARC
    ni = int(round(length / (dx_frac * _ARC))) + 1

    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=1.8e-4,
        Pr=1.0,
        T_dtm=_T,
        Rgas_ref=287.0,
        rho_ref=1.1,
        V_ref=_VX,
    )

    xrt = util.linmesh3(
        [-length, 0.0],
        [_R_HUB, _R_HUB + _SPAN],
        [-_PITCH / 2.0, _PITCH / 2.0],
        (ni, nj, nk),
    )
    block = ember.block.Block(shape=(ni, nj, nk))
    block.set_xrt(xrt)
    block.set_Nb(_NB)
    block.set_fluid(fluid)
    block.set_P_T(_P, _T)
    block.set_Vx(_VX)
    block.set_Vr(0.0)
    block.set_Vt(0.0)

    # Uniform inflow state, read off a scalar block so it stays dimensional.
    ref = ember.block.Block(shape=())
    ref.set_fluid(fluid)
    ref.set_x(np.array([0.0]))
    ref.set_r(np.array([_R_HUB]))
    ref.set_t(np.array([0.0]))
    ref.set_P_T(_P, _T)
    ref.set_Vx(_VX)
    ref.set_Vr(0.0)
    ref.set_Vt(0.0)

    if inlet_kind == "nrbc":
        inlet = ember.patch.NonReflectingInletPatch(i=0)
    else:
        inlet = ember.patch.InletPatch(i=0)
    outlet = ember.patch.OutletPatch(i=-1)
    block.patches.extend(
        [
            inlet,
            outlet,
            ember.patch.PeriodicPatch(k=0),
            ember.patch.PeriodicPatch(k=-1),
            ember.patch.InviscidPatch(j=0),
            ember.patch.InviscidPatch(j=-1),
        ]
    )

    if inlet_kind == "nrbc":
        inlet.set_ho_s_Alpha_Beta(ho=float(ref.ho), s=float(ref.s), Alpha=0.0, Beta=0.0)
        inlet.sigma = sigma
    else:
        inlet.set_Po_To_Alpha_Beta(float(ref.Po), float(ref.To), 0.0, 0.0)

    # Purely harmonic exit pressure: the mean is prescribed by both inlets, so
    # only the harmonic part exercises the non-reflecting theory.
    theta = block.t[-1]
    outlet.set_P(_P * (1.0 + amp * np.cos(2.0 * np.pi * theta / _PITCH)))

    grid = ember.grid.Grid([block])
    grid.set_L_ref(_SPAN)
    grid.calculate_wdist()
    grid.connectivity.periodic.pair()
    return grid


def profile_at(grid, x_target):
    """Pitchwise static pressure profile at the node plane nearest x_target."""
    block = grid[0]
    x = block.x[:, 0, 0]
    i = int(np.argmin(np.abs(x - x_target)))
    return block.P[i].mean(axis=0), float(x[i])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--short", type=float, default=0.5, help="short domain, pitches"
    )
    parser.add_argument("--long", type=float, default=4.0, help="long domain, pitches")
    parser.add_argument(
        "--probe",
        type=float,
        default=0.2,
        help="probe station, pitches upstream of exit",
    )
    parser.add_argument("--amp", type=float, default=0.02)
    parser.add_argument("--n-step", type=int, default=2000)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--n-stage", type=int, default=4)
    parser.add_argument("--sigma", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    conf = ember.solver.Solver(
        n_step=args.n_step,
        n_step_log=args.n_step,
        n_stage=args.n_stage,
        cfl=args.cfl,
        n_step_avg=1,
    )
    x_probe = -args.probe * _ARC

    def solve(n_pitch, kind, sigma):
        grid = build_grid(n_pitch, kind, args.amp, sigma)
        conf.run(grid)
        return profile_at(grid, x_probe)

    # Reference: the same physics with the inflow plane far enough away that the
    # evanescent harmonics have died before reaching it.
    reference, x_ref = solve(args.long, "standard", 0.0)
    scale = float(np.ptp(reference))
    print(
        f"reference: long domain ({args.long:g} pitches), probe at x/arc={x_ref / _ARC:.3f}"
    )
    print(f"           peak-to-peak pressure variation {scale:.1f} Pa\n")

    cases = [("InletPatch", "standard", 0.0)]
    cases += [(f"NRBC sigma={s:g}", "nrbc", s) for s in args.sigma]

    results = {}
    width = max(len(label) for label, _, _ in cases)
    print(f"{'short-domain inlet':<{width}}  error vs reference")
    for label, kind, sigma in cases:
        got, _ = solve(args.short, kind, sigma)
        results[label] = got
        err = float(np.abs(got - reference).max()) / scale
        print(f"{label:<{width}}  {err:8.1%}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        theta = np.linspace(-0.5, 0.5, len(reference))
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(
            theta, reference, "k-", lw=2, label=f"reference ({args.long:g} pitches)"
        )
        for label, got in results.items():
            ax.plot(theta, got, "--", label=label)
        ax.set_xlabel(r"pitchwise position $\theta / P$")
        ax.set_ylabel("static pressure [Pa]")
        ax.set_title(f"probe {args.probe:g} pitches upstream of exit")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
