r"""
Square-duct multigrid convergence
==================================

This example solves for the viscous flow through a square duct.

"""

import time

import numpy as np
import matplotlib.pyplot as plt

import ember.block
import ember.grid
import ember.fluid
import ember.patch
import ember.solver
import ember.set_iter
from ember import util

# %%
# Case definition
# ---------------
#
# The duct cross-section is a square of side ``SIDE``, bent onto a mean
# radius ``R_MID_RATIO * SIDE`` and repeated around the annulus at ``Nb``
# passages. ``Nb`` is chosen so that the arc subtended by one passage at the
# mean radius is as close to ``SIDE`` as an integer passage count allows,
# making the cross-section square. The streamwise mesh is uniform, while the
# two cross-stream directions are clustered towards both walls at expansion
# ratio ``ER``, so the near-wall spacing is not prescribed: it falls out of
# ``SIDE``, ``NJ``, and ``NK``. The mean flow is fixed by a bulk Mach number
# at a given stagnation state.

SIDE = 0.1
R_MID_RATIO = 5.0
LENGTH_RATIO = 3.0
ER = 1.05
NI, NJ, NK = 121, 65, 57
MA_BULK = 0.3
PO = 1e5
TO = 300.0

R_MID = R_MID_RATIO * SIDE


def make_block(fluid):
    """Build the duct block with its mesh, fluid, and uniform mean flow."""
    Nb = round(2.0 * np.pi * R_MID / SIDE)  # pitch subtends ~SIDE at mean radius
    pitch = 2.0 * np.pi / Nb
    r_low = R_MID - 0.5 * SIDE
    length = LENGTH_RATIO * SIDE

    xv = np.linspace(0.0, length, NI, dtype=np.float32)
    rv = r_low + SIDE * util.cluster_symmetric(NJ, ER)
    tv = pitch * (util.cluster_symmetric(NK, ER) - 0.5)
    xm, rm, tm = np.meshgrid(xv, rv, tv, indexing="ij")
    xrt = np.stack((xm, rm, tm), axis=-1).astype(np.float32)

    block = ember.block.Block(shape=(NI, NJ, NK))
    block.set_xrt(xrt)
    block.set_Nb(Nb)
    block.set_fluid(fluid)

    rho_o, e_o = fluid.set_P_T(PO, TO)
    ho = fluid.get_h(rho_o, e_o)
    so = fluid.get_s(rho_o, e_o)
    a_o = fluid.get_a(rho_o, e_o)
    Vbar = MA_BULK * a_o
    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho, so, MA_BULK, 0.0, 0.0)

    U = Vbar / np.inf
    Omega = U / R_MID
    block.set_Omega(Omega)
    block.set_Vt(Omega * block.r)

    return block, rho_o, ho, so, Vbar


# %%
# Boundary conditions
# -------------------
#
# The inlet fixes stagnation conditions and swirl angle, and the outlet fixes
# static pressure with a backflow state for any transient reverse flow.


def set_boundary_conditions(block, ho, so):
    """Attach inlet and outlet patches consistent with the mean flow."""
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

    return P_out, T_out


# %%
# Initial condition
# ------------------
#
# The march is started from the mean flow above, perturbed with a small
# random velocity ripple (to seed any instability) plus a deterministic
# streamwise ramp in enthalpy, entropy, and velocity so the solution has to
# do some work to relax back to the boundary-condition-consistent state.


def add_velocity_noise(block, seed=0):
    """Perturb the axial velocity with a 1% random ripple."""
    rng = np.random.default_rng(seed)
    Vx = block.Vx
    block.set_Vx(Vx * (1.0 + 0.01 * rng.standard_normal(Vx.shape)).astype(Vx.dtype))


def make_grid(block, fluid, rho_o, Vbar, P_out, T_out):
    """Wrap the block in a grid, set the non-dimensionalisation, and get wdist."""
    grid = ember.grid.Grid([block])
    grid.set_L_ref(SIDE)
    grid.set_fluid(
        fluid.change_datum(P_out, T_out).change_ref(rho_o, Vbar, block.Rgas.mean())
    )
    grid.calculate_wdist()
    return grid


def add_thermodynamic_ramp(block):
    """Offset ho and s, then ramp Vx by +1% along the duct."""
    V = np.asarray(block.V)
    ho_field = np.asarray(block.ho)
    s_field = np.asarray(block.s)
    T_field = np.asarray(block.T)
    h_static = ho_field - 0.5 * V**2

    ho_frac = 0.01  # ho raised by 1% of local dynamic enthalpy
    s_frac = 0.01  # s raised by entropy equivalent of that offset
    dh = ho_frac * 0.5 * V**2
    ds = s_frac * 0.5 * V**2 / T_field
    block.set_h_s(h_static + dh, s_field + ds)  # velocity preserved

    Vx = np.asarray(block.Vx)
    ramp = np.linspace(1.0, 1.01, NI, dtype=Vx.dtype)  # +1% streamwise Vx ramp
    block.set_Vx(Vx * ramp[:, None, None])


# %%
# Cross-section mesh
# ------------------
#
# A constant-``i`` slice through the block, drawn in the Cartesian ``y``-``z``
# plane. The passage count chosen above makes the cross-section square to
# within the rounding of ``Nb`` to an integer, and the geometric expansion
# away from each wall is visible as the thinning of cells towards all four
# sides.
#
# A constant-``k`` slice, drawn in the meridional ``x``-``r`` plane, shows the
# streamwise mesh (uniform) and the clustering towards the two ``r`` walls.


def plot_mesh(block, i=0):
    """Draw the constant-i grid slice in the y-z plane."""
    y = np.asarray(block.y)[i]
    z = np.asarray(block.z)[i]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.plot(z, y, color="k", lw=0.2)  # lines of constant j
    ax.plot(z.T, y.T, color="k", lw=0.2)  # lines of constant k
    ax.set_title(f"Cross-section mesh at $i={i}$ ({NJ} x {NK} nodes)")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    plt.show()


def plot_mesh_k(block, k=0):
    """Draw the constant-k grid slice in the meridional x-r plane."""
    x = np.asarray(block.x)[:, :, k]
    r = np.asarray(block.r)[:, :, k]

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(x, r, color="k", lw=0.2)  # lines of constant j
    ax.plot(x.T, r.T, color="k", lw=0.2)  # lines of constant i
    ax.set_title(f"Meridional mesh at $k={k}$ ({NI} x {NJ} nodes)")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    plt.show()


# %%
# Multigrid march
# ---------------
#
# ``ember.solver.run`` advances the flow field with an explicit
# Runge-Kutta scheme accelerated by a two-level multigrid correction.


def solve(grid, n_stage, cfl, sf_resid, fac_mgrid):
    """March the flow field and return the trimmed convergence history.

    ``n_stage=0`` selects Denton's basic scree march, and ``n_stage>=1`` a
    Jameson multi-stage Runge-Kutta step. ``sf_resid`` is the implicit
    residual smoothing factor, which relaxes the explicit stability limit and
    so admits a larger ``cfl``. ``fac_mgrid`` scales the two-level multigrid
    correction and is tuned separately per scheme.

    The history is trimmed to the steps actually logged, so a march that
    diverged part-way carries no trailing unfilled records.
    """
    n_step = 500
    conf = ember.solver.SolverConfig(
        n_step=n_step,
        n_step_log=50,
        n_step_avg=1,
        cfl=cfl,
        n_stage=n_stage,
        n_levels=2,
        fac_mgrid=fac_mgrid,
        sf_resid=sf_resid,
        inviscid=False,
    )
    tic = time.perf_counter()
    hist = ember.solver.run(grid, conf)
    wall = time.perf_counter() - tic
    hist = hist.trim()

    # A diverged march breaks out of the step loop early, so the step count it
    # actually reached is unknown to within one logging interval. Quoting a
    # per-step cost against the requested n_step would flatter it.
    tag = (
        f"n_stage={n_stage}, cfl={cfl}, sf_resid={sf_resid}, "
        f"fac_mgrid={fac_mgrid}: {wall:.1f} s"
    )
    if hist.diverged:
        print(f"{tag}, DIVERGED after >={int(hist.i_step[-1])} of {n_step} steps")
    else:
        block = grid[0]
        n_node = block.ni * block.nj * block.nk
        print(
            f"{tag}, {wall / n_step * 1e3:.1f} ms/step, "
            f"{wall / n_step / n_node * 1e6:.3f} us/node/step"
        )

    return hist


def build_case():
    """Assemble a fresh grid at the perturbed initial condition.

    ``ember.solver.run`` marches the grid in place, so each of the schemes
    compared below needs its own copy to start from the same state.
    """
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=1.0e-3,
        Pr=0.72,
        T_dtm=400.0,
    )
    block, rho_o, ho, so, Vbar = make_block(fluid)
    P_out, T_out = set_boundary_conditions(block, ho, so)
    add_velocity_noise(block)
    grid = make_grid(block, fluid, rho_o, Vbar, P_out, T_out)
    add_thermodynamic_ramp(grid[0])
    return grid


# %%
# Convergence history
# -------------------
#
# Two marches of the same initial condition, differing in the time
# integrator, the CFL number, and the multigrid scaling factor. The two
# multigrid levels are held fixed for both.

CASES = (
    ("scree, CFL=0.4", dict(n_stage=0, cfl=0.4, sf_resid=0.0, fac_mgrid=0.4)),
    ("RK4, CFL=4.0", dict(n_stage=4, cfl=4.0, sf_resid=0.0, fac_mgrid=0.0)),
    ("RK4+RS, CFL=8.0", dict(n_stage=4, cfl=8.0, sf_resid=1.0, fac_mgrid=0.0)),
)


def plot_history(results):
    """Overlay energy residual, mass flow error, and entropy rise per scheme."""
    fig, (ax_res, ax_err, ax_s) = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True)
    ax_err.axhline(0.0, color="0.6", lw=0.8)

    for label, hist in results:
        i_step = hist.i_step
        drhoe = hist.residual[:, 4]
        ax_res.semilogy(i_step, drhoe, marker=".", ms=3, lw=1.0, label=label)
        ax_err.plot(i_step, hist.err_mdot, marker=".", ms=3, lw=1.0, label=label)
        ax_s.plot(i_step, hist.zeta, marker=".", ms=3, lw=1.0, label=label)

    ax_res.set_ylabel(r"$|\Delta(\rho e)|$")
    ax_res.set_title("Energy residual (semilog)")
    ax_res.grid(True, which="both", alpha=0.3)
    ax_res.legend()

    ax_err.set_ylabel(r"$(\dot m_\mathrm{out} - \dot m_\mathrm{in}) / \bar{\dot m}$")
    ax_err.set_title("Mass flow error")
    ax_err.grid(True, alpha=0.3)

    ax_s.set_ylabel(r"$\zeta = s_\mathrm{out} - s_\mathrm{in}$")
    ax_s.set_title("Entropy rise")
    ax_s.set_xlabel("i_step")
    ax_s.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# %%
# Run the case
# ------------

grid = build_case()

# %%
# The cross-section mesh of the assembled grid, before any marching.

plot_mesh(grid[0])

# %%
# The meridional mesh of the same grid.

plot_mesh_k(grid[0])

# %%
# Now march the flow field once per scheme and plot how each converges.

results = [(label, solve(build_case(), **kwargs)) for label, kwargs in CASES]
plot_history(results)
