r"""
Square-duct multigrid convergence
==================================

This example solves the inviscid flow through a square duct with an
``ember.scree`` explicit multigrid march, and plots the convergence history.
It is the same case driven by ``scripts/run_duct.py``, with the command-line
options replaced by fixed values below.

Unlike the other examples in this gallery, the grid here is large enough
(around 90000 nodes) that the multigrid solve takes on the order of ten
seconds, so it is **not** executed on every documentation build (see
``examples/README.txt``). Run ``make docs-full`` to regenerate this page's
output after changing the case.
"""

import numpy as np
import matplotlib.pyplot as plt

import ember.block
import ember.grid
import ember.fluid
import ember.patch
import ember.scree
import ember.set_iter
from ember import util

# %%
# Grid and mean flow
# ------------------
#
# The duct cross-section is a square of side ``side``, bent onto a mean
# radius ``r_mid_ratio * side`` and repeated around the annulus at ``Nb``
# passages so the pitch matches the near-wall grid spacing. The mean flow is
# fixed by a bulk Mach number at a given stagnation state.

side = 0.1
r_mid_ratio = 5.0
length_ratio = 3.0
ER = 1.05
ni, nj, nk = 25, 65, 57
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

xrt = util.linmesh3(
    [0.0, length], [r_low, r_high], [-0.5 * pitch, 0.5 * pitch], (ni, nj, nk)
)
block = ember.block.Block(shape=(ni, nj, nk))
block.set_xrt(xrt)
block.set_Nb(Nb)

fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.0e-3, Pr=0.72, T_dtm=400.0)
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

# %%
# Boundary conditions
# -------------------
#
# The inlet fixes stagnation conditions and swirl angle, and the outlet fixes
# static pressure with a backflow state for any transient reverse flow.

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

# %%
# Initial condition
# ------------------
#
# The march is started from the mean flow above, perturbed with a small
# random velocity ripple (to seed any instability) plus a deterministic
# streamwise ramp in enthalpy, entropy, and velocity so the solution has to
# do some work to relax back to the boundary-condition-consistent state.

rng = np.random.default_rng(0)
Vx = block.Vx
block.set_Vx(Vx * (1.0 + 0.01 * rng.standard_normal(Vx.shape)).astype(Vx.dtype))

grid = ember.grid.Grid([block])
grid.set_L_ref(side)
grid.set_fluid(
    fluid.change_datum(P_out, T_out).change_ref(rho_o, Vbar, block.Rgas.mean())
)
grid.calculate_wdist()

block = grid[0]
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
ramp = np.linspace(1.0, 1.01, ni, dtype=Vx.dtype)  # +1% streamwise Vx ramp
block.set_Vx(Vx * ramp[:, None, None])

# %%
# Multigrid march
# ---------------
#
# ``ember.scree.loop`` advances the flow field with an explicit
# Runge-Kutta scheme accelerated by a three-level multigrid correction.

conf = ember.scree.ScreeConfig(
    n_step=300,
    n_step_log=10,
    n_step_avg=1,
    cfl=3.0,
    n_stage=4,
    n_levels=3,
    fac_mgrid=0.2,
    sf_resid=0.0,
    inviscid=False,
)
hist = ember.scree.loop(grid, conf)
n = hist.i_log + 1
i_step = np.asarray([hist.i_step[i] for i in range(n)], dtype=float)

# %%
# Convergence history
# -------------------
#
# The energy residual should decay several orders of magnitude, while the
# inlet-to-outlet mass flow error and entropy rise settle to their converged
# values.

fig, (ax_res, ax_err, ax_s) = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True)

drhoe = np.abs(np.asarray(hist.residual, dtype=float)[:n, 4])
m = np.isfinite(drhoe) & (drhoe > 0)
ax_res.semilogy(i_step[m], drhoe[m], marker=".", ms=3, lw=1.0)
ax_res.set_ylabel(r"$|\Delta(\rho e)|$")
ax_res.set_title("Energy residual (semilog)")
ax_res.grid(True, which="both", alpha=0.3)

err = np.asarray(hist.err_mdot, dtype=float)[:n]
me = np.isfinite(err)
ax_err.axhline(0.0, color="0.6", lw=0.8)
ax_err.plot(i_step[me], err[me], marker=".", ms=3, lw=1.0)
ax_err.set_ylabel(r"$(\dot m_\mathrm{out} - \dot m_\mathrm{in}) / \bar{\dot m}$")
ax_err.set_title("Mass flow error")
ax_err.grid(True, alpha=0.3)

zeta = np.asarray(hist.zeta, dtype=float)[:n]
mz = np.isfinite(zeta)
ax_s.plot(i_step[mz], zeta[mz], marker=".", ms=3, lw=1.0)
ax_s.set_ylabel(r"$\zeta = s_\mathrm{out} - s_\mathrm{in}$")
ax_s.set_title("Entropy rise")
ax_s.set_xlabel("i_step")
ax_s.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
