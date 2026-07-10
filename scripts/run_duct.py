#!/usr/bin/env -S uv run
"""Inviscid square-duct baseline: configurable CFL / MG / IRS run.

Runs the square-duct case and reports convergence (energy residual, mass flow
error, entropy rise). With --plot writes a 3-panel figure to
scripts/baseline_cfl3_mg.pdf.
"""

import argparse
import logging
import sys
import time

import numpy as np

import ember.block
import ember.grid
import ember.fluid
import ember.patch
import ember.solver
import ember.set_iter
from ember import util

logging.disable(logging.CRITICAL)  # silence per-step convergence logging

# ---------------------------------------------------------------------------
# Fixed IC perturbation parameters
# ---------------------------------------------------------------------------
HO_FRAC = 0.01  # ho raised by 1% of local dynamic enthalpy
S_FRAC = 0.01  # s raised by entropy equivalent of that offset
PERTURB_VX = 0.01
PERTURB_SEED = 0


def build_grid(ncell):
    """Assemble the inviscid square-duct grid."""
    side = 0.1
    r_mid_ratio = 5.0
    length_ratio = 3.0
    ER = 1.05
    nj = 65
    nk = 57
    Ma_bulk = 0.3
    Po = 1e5
    To = 300.0

    r_mid = r_mid_ratio * side
    r_low = r_mid - 0.5 * side
    r_high = r_mid + 0.5 * side
    length = length_ratio * side

    n_half = (nj + 1) // 2
    eta_half = util.cluster(n_half, ER, 1.0)
    ds_mid = 0.5 * side * float(eta_half[-1] - eta_half[-2])

    Nb = round(2.0 * np.pi * r_mid / (nk * ds_mid)) * 2
    pitch = 2.0 * np.pi / Nb

    ni = ncell // (nj * nk)
    ni = ((ni - 1 + 4) // 8) * 8 + 1

    # Below this the duct is too short to march: a handful of streamwise cells
    # gives the inlet and outlet patches no interior between them, and ni=1
    # (ncell < nj*nk) yields zero cells, which the Fortran kernels reject.
    if ni < 25:
        raise ValueError(
            f"ncell={ncell} gives only ni={ni} streamwise nodes "
            f"(nj={nj}, nk={nk}); need ni >= 25, i.e. ncell >= {25 * nj * nk}"
        )

    xrt = util.linmesh3(
        [0.0, length], [r_low, r_high], [-0.5 * pitch, 0.5 * pitch], (ni, nj, nk)
    )
    block = ember.block.Block(shape=(ni, nj, nk))
    block.set_xrt(xrt)
    block.set_Nb(Nb)

    fluid = ember.fluid.PerfectFluid(
        cp=1005.0, gamma=1.4, mu=1.0e-3, Pr=0.72, T_dtm=400.0
    )
    block.set_fluid(fluid)

    rho_o, e_o = fluid.set_P_T(Po, To)
    ho = fluid.get_h(rho_o, e_o)
    so = fluid.get_s(rho_o, e_o)
    a_o = fluid.get_a(rho_o, e_o)
    Vbar = Ma_bulk * a_o
    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho, so, Ma_bulk, 0.0, 0.0)

    U = Vbar / np.inf
    Omega = U / r_mid
    block.set_Omega(Omega)
    block.set_Vt(Omega * block.r)

    block.patches["inlet"] = ember.patch.InletPatch(i=0)
    block.patches["outlet"] = ember.patch.OutletPatch(i=-1)

    Po_in = block.Po[0].mean()
    To_in = block.To[0].mean()
    Alpha_in = block.Alpha[0].mean()
    P_out = block.P[-1].mean()
    T_out = block.T[-1].mean()
    block.patches["inlet"].set_Po_To_Alpha_Beta(Po_in, To_in, Alpha_in, 0.0)
    block.patches["outlet"].set_P(P_out)
    block.patches["outlet"].set_backflow(ho, so, 0.0, 0.0)

    rng = np.random.default_rng(PERTURB_SEED)
    Vx = block.Vx
    block.set_Vx(
        Vx * (1.0 + PERTURB_VX * rng.standard_normal(Vx.shape)).astype(Vx.dtype)
    )

    grid = ember.grid.Grid([block])
    grid.set_L_ref(side)
    grid.set_fluid(
        fluid.change_datum(P_out, T_out).change_ref(rho_o, Vbar, block.Rgas.mean())
    )
    grid.calculate_wdist()
    return grid


def build_grid_ic(ncell):
    """build_grid, then add the deterministic ho/entropy perturbation and a
    linear +1% streamwise Vx ramp."""
    grid = build_grid(ncell)
    block = grid[0]

    V = np.asarray(block.V)
    ho = np.asarray(block.ho)
    s = np.asarray(block.s)
    T = np.asarray(block.T)
    h_static = ho - 0.5 * V**2

    dh = HO_FRAC * 0.5 * V**2
    ds = S_FRAC * 0.5 * V**2 / T
    block.set_h_s(h_static + dh, s + ds)  # velocity preserved

    # Linear +1% ramp in Vx: inlet unchanged, outlet +1%.
    # set_Vx keeps the static (rho, e) state fixed and rewrites momentum + energy.
    Vx = np.asarray(block.Vx)
    ni = Vx.shape[0]
    ramp = np.linspace(1.0, 1.01, ni, dtype=Vx.dtype)
    block.set_Vx(Vx * ramp[:, None, None])
    return grid


def run(args):
    grid = build_grid_ic(args.ncell)
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
    p.add_argument("--plot", metavar="PATH", help="Write 3-panel figure to this path")
    run(p.parse_args())


if __name__ == "__main__":
    main()
