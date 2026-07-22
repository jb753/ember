#!/usr/bin/env -S uv run
"""Boundary reflectivity: domain-truncation study, non-reflecting vs standard.

Measures the thing a steady non-reflecting boundary condition is actually for --
being able to put the boundary plane close to a blade row without corrupting the
solution near it. Either end of a duct can be put under test with ``--boundary``.

With ``--boundary inlet`` a pitchwise-varying static pressure at the outlet
stands in for the potential field of a downstream blade row, and the inflow
plane is moved toward it. In subsonic flow the harmonics of that field are
evanescent: the wave parameter of Saxer and Giles (their Eq. 22) is imaginary, so
each pitchwise harmonic decays upstream over roughly a pitch divided by two pi,
and a disturbance seeded further upstream than that never reaches the inflow
plane at all. The inlet condition therefore only matters when the plane sits
within a few decay lengths of the source, which is exactly the regime this script
sets up: the same physical problem is solved on a short domain and on a long one,
and the two solutions are compared over their common region. A perfectly
non-reflecting inlet gives the same answer on both.

With ``--boundary outlet`` the study is mirrored. The source is a
pitchwise-varying inflow yaw angle standing in for the wake and potential field
of an upstream blade row, and the outflow plane is moved toward it. The vorticity
it sheds convects to the exit whatever the domain length, so a reflecting outlet
answers it with a pressure wave; that wave is evanescent and decays going
upstream, so again only a boundary within a few decay lengths of the source
corrupts it.

Note that the *pitchwise mean* is prescribed by any inlet that fixes stagnation
quantities and flow angles, and by any outlet that fixes pressure, so the
``m = 0`` mode is reflecting by construction for both conditions here -- Saxer and
Giles make this point in their Section III. Only the harmonics are absorbed,
which is why the source below is purely harmonic.

Not part of ``make test``: it drives a few thousand solver steps per
configuration.

Examples
--------
::

    uv run tools/run_nrbc_reflection.py
    uv run tools/run_nrbc_reflection.py --boundary outlet --plot nrbc_outlet.pdf
    uv run tools/run_nrbc_reflection.py --sigma 0.05 0.1 0.2 --plot nrbc.pdf
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

# Default source amplitude for each study: a relative static pressure harmonic
# at the exit, or an inflow yaw angle harmonic in degrees.
_AMP_DEFAULT = {"inlet": 0.02, "outlet": 5.0}

# The patch class used at the boundary under test, standard or non-reflecting.
_PATCH_TYPES = {
    "inlet": {
        "standard": ember.patch.InletPatch,
        "nrbc": ember.patch.NonReflectingInletPatch,
    },
    "outlet": {
        "standard": ember.patch.OutletPatch,
        "nrbc": ember.patch.NonReflectingOutletPatch,
    },
}


def build_grid(n_pitch, boundary, kind, amp, sigma, dx_frac=0.04, nj=9, nk=25):
    """Uniform duct with a pitchwise-varying source at the end not under test.

    Parameters
    ----------
    n_pitch : float
        Domain length in pitches; the boundary under test sits ``n_pitch``
        arc lengths from the source.
    boundary : str
        ``"inlet"`` or ``"outlet"``, which end of the duct is under test.
    kind : str
        ``"standard"`` for the reflecting patch, ``"nrbc"`` for the
        non-reflecting one.
    amp : float
        Source amplitude: relative static pressure harmonic at the exit for the
        inlet study, inflow yaw angle harmonic in degrees for the outlet study.
    sigma : float
        Relaxation factor for the non-reflecting patch.
    """
    length = n_pitch * _ARC
    ni = int(round(length / (dx_frac * _ARC))) + 1
    # The source sits at x = 0 either way, with the domain upstream of it for
    # the inlet study and downstream of it for the outlet study.
    x_lim = [-length, 0.0] if boundary == "inlet" else [0.0, length]

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
        x_lim,
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

    inlet_kind = kind if boundary == "inlet" else "standard"
    outlet_kind = kind if boundary == "outlet" else "standard"
    inlet = _PATCH_TYPES["inlet"][inlet_kind](i=0)
    outlet = _PATCH_TYPES["outlet"][outlet_kind](i=-1)
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

    # Purely harmonic source: the mean is prescribed by every condition here, so
    # only the harmonic part exercises the non-reflecting theory.
    Alpha = 0.0
    P_out = _P
    if boundary == "inlet":
        P_out = _P * (1.0 + amp * np.cos(2.0 * np.pi * block.t[-1] / _PITCH))
    else:
        Alpha = amp * np.cos(2.0 * np.pi * block.t[0] / _PITCH)

    if inlet_kind == "nrbc":
        inlet.set_Po_To(float(ref.Po), float(ref.To))
        inlet.set_Alpha(Alpha)
        inlet.set_Beta(0.0)
        inlet.sigma = sigma
    else:
        inlet.set_Po_To_Alpha_Beta(float(ref.Po), float(ref.To), Alpha, 0.0)

    outlet.set_P(P_out)
    if outlet_kind == "nrbc":
        outlet.sigma = sigma

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
        "--boundary",
        choices=("inlet", "outlet"),
        default="inlet",
        help="which boundary is under test",
    )
    parser.add_argument(
        "--short", type=float, default=0.5, help="short domain, pitches"
    )
    parser.add_argument("--long", type=float, default=4.0, help="long domain, pitches")
    parser.add_argument(
        "--probe",
        type=float,
        default=0.2,
        help="probe station, pitches from the source, inside the domain",
    )
    parser.add_argument(
        "--amp",
        type=float,
        default=None,
        help="source amplitude; exit pressure fraction (inlet study) or inflow "
        "yaw angle in degrees (outlet study)",
    )
    parser.add_argument("--n-step", type=int, default=2000)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--n-stage", type=int, default=4)
    parser.add_argument("--sigma", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    amp = _AMP_DEFAULT[args.boundary] if args.amp is None else args.amp

    conf = ember.solver.Solver(
        n_step=args.n_step,
        n_step_log=args.n_step,
        n_stage=args.n_stage,
        cfl=args.cfl,
        n_step_avg=1,
    )
    # The probe sits a fixed distance inside the domain from the source at
    # x = 0, in the region of interest common to the short and long domains.
    sign = -1.0 if args.boundary == "inlet" else 1.0
    source = "exit" if args.boundary == "inlet" else "inlet"

    def solve(n_pitch, kind, sigma):
        grid = build_grid(n_pitch, args.boundary, kind, amp, sigma)
        conf.run(grid)
        return profile_at(grid, sign * args.probe * _ARC)

    # Reference: the same physics with the boundary under test far enough away
    # that the evanescent harmonics have died before reaching it.
    reference, x_ref = solve(args.long, "standard", 0.0)
    scale = float(np.ptp(reference))
    print(
        f"reference: long domain ({args.long:g} pitches), probe at x/arc={x_ref / _ARC:.3f}"
    )
    print(f"           peak-to-peak pressure variation {scale:.1f} Pa\n")

    standard = _PATCH_TYPES[args.boundary]["standard"].__name__
    cases = [(standard, "standard", 0.0)]
    cases += [(f"NRBC sigma={s:g}", "nrbc", s) for s in args.sigma]

    results = {}
    width = max(len(label) for label, _, _ in cases)
    print(f"{'short-domain ' + args.boundary:<{width}}  error vs reference")
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
        ax.set_title(f"probe {args.probe:g} pitches from the {source}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
