r"""
Square-duct multigrid convergence
==================================

This example compares time-marching schemes on the viscous flow through a
square duct: Denton's basic scree march, a Jameson multi-stage Runge-Kutta
step, and the same with implicit residual smoothing. The duct case itself --
mesh, mean flow, boundary conditions, and the perturbed initial condition -- is
assembled by :func:`ember.cases.build_duct_grid`; here we focus on how the three
schemes converge from the same starting state.

"""

import time

import numpy as np
import matplotlib.pyplot as plt

import ember.solver
from ember.cases import build_duct_grid

# %%
# Case definition
# ---------------
#
# :func:`ember.cases.build_duct_grid` meshes the duct (streamwise-uniform, with
# the two cross-stream directions clustered towards both walls), fixes the mean
# flow from a bulk Mach number at a given stagnation state, attaches inlet and
# outlet patches consistent with that mean flow, and perturbs the initial
# condition with a small random velocity ripple plus a deterministic streamwise
# ramp in enthalpy, entropy, and velocity, so the solution has to do some work
# to relax back to the boundary-condition-consistent state. ``ncell`` sets the
# target cell count; the value below yields a ``121 x 65 x 57`` block.

NCELL = 450_000


# %%
# Cross-section mesh
# ------------------
#
# A constant-``i`` slice through the block, drawn in the Cartesian ``y``-``z``
# plane. The passage count makes the cross-section square to within the rounding
# of ``Nb`` to an integer, and the geometric expansion away from each wall is
# visible as the thinning of cells towards all four sides.
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
    ax.set_title(f"Cross-section mesh at $i={i}$ ({block.nj} x {block.nk} nodes)")
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
    ax.set_title(f"Meridional mesh at $k={k}$ ({block.ni} x {block.nj} nodes)")
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
    """March the flow field and return the convergence history.

    ``n_stage=0`` selects Denton's basic scree march, and ``n_stage>=1`` a
    Jameson multi-stage Runge-Kutta step. ``sf_resid`` is the implicit
    residual smoothing factor, which relaxes the explicit stability limit and
    so admits a larger ``cfl``. ``fac_mgrid`` scales the two-level multigrid
    correction and is tuned separately per scheme.
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


# %%
# Convergence history
# -------------------
#
# Two marches of the same initial condition, differing in the time
# integrator, the CFL number, and the multigrid scaling factor. The two
# multigrid levels are held fixed for both.

CASES = (
    ("scree, CFL=0.4", dict(n_stage=0, cfl=0.4, sf_resid=0.0, fac_mgrid=0.4)),
    ("RK4, CFL=4.0", dict(n_stage=4, cfl=4.0, sf_resid=0.0, fac_mgrid=0.4)),
    ("RK4+RS, CFL=8.0", dict(n_stage=4, cfl=8.0, sf_resid=1.0, fac_mgrid=0.4)),
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
#
# ``ember.solver.run`` marches the grid in place, so each scheme is given its
# own freshly built grid to start from the same state.

grid = build_duct_grid(ncell=NCELL)

# %%
# The cross-section mesh of the assembled grid, before any marching.

plot_mesh(grid[0])

# %%
# The meridional mesh of the same grid.

plot_mesh_k(grid[0])

# %%
# Now march the flow field once per scheme and plot how each converges.

results = [(label, solve(build_duct_grid(ncell=NCELL), **kwargs)) for label, kwargs in CASES]
plot_history(results)
