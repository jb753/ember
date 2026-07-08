"""Turbulent-diffusion limit in the spectral timestep.

``Grid.update_timestep`` (via ``set_timestep_spectral``) sets the unscaled
volumetric timestep to ``1 / max(lam_conv, lam_diff)`` where ``lam_conv`` is the
existing convective spectral-radius sum and ``lam_diff =
(mu_turb/rho)*sum_d||dA_d||^2/vol`` is a directional turbulent-diffusion spectral
radius over the same per-cell face areas. These tests pin the two regimes:

  * inviscid / no turbulence (``mu_turb = 0``): the diffusion radius vanishes and
    the timestep reduces exactly to the convective value; and

  * viscous (``mu_turb > 0``): the timestep matches the closed-form
    ``1 / max(lam_conv, lam_diff)`` and is never larger than the convective one.
"""

import numpy as np

import ember.block
import ember.grid
from ember import util
from ember.fluid import PerfectFluid

SHAPE = (5, 5, 5)


def _build_grid():
    """Single-block grid with a smooth sheared flow; mu_turb left at its 0 init."""
    block = ember.block.Block(shape=SHAPE)
    block.set_Nb(36)
    xrt = util.linmesh3((0.0, 0.1), (0.5, 1.0), (0.0, 0.2), SHAPE)
    block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
    block.set_P_T(101325.0, 300.0)
    block.set_wdist(np.zeros_like(block.r))
    Vx = 100.0 + 20.0 * np.sin(2.0 * np.pi * block.r)
    block.set_Vx(Vx.astype(np.float32))
    block.set_Vr(np.zeros_like(Vx, dtype=np.float32))
    block.set_Vt(np.zeros_like(Vx, dtype=np.float32))
    return ember.grid.Grid([block])


def test_inviscid_timestep_is_convective_only():
    """With mu_turb = 0 the diffusion radius vanishes: pure convective timestep."""
    grid = _build_grid()

    # Fresh grid: mu_turb is the benign 0 init (never set), so the initial
    # timestep must be finite -- no NaN leaking from an unset field.
    grid.update_timestep(rf=1.0)
    dt_conv = grid[0].dt_vol_nd.copy()
    assert np.all(np.isfinite(dt_conv))
    assert np.all(dt_conv > 0.0)

    # An explicit zero fill reproduces the init result bit-for-bit.
    grid[0].set_mu_turb(np.zeros(SHAPE, dtype=np.float32))
    grid.update_timestep(rf=1.0)
    np.testing.assert_array_equal(grid[0].dt_vol_nd, dt_conv)


def test_viscous_timestep_matches_max_of_spectral_radii():
    """mu_turb > 0 gives dt_vol = 1/max(lam_conv, lam_diff), <= convective."""
    grid = _build_grid()
    block = grid[0]

    # Convective reference (mu_turb = 0); rf=1 fully overwrites, no blend, so
    # lam_conv is recovered exactly as 1/dt_conv.
    grid.update_timestep(rf=1.0)
    dt_conv = block.dt_vol_nd.copy()
    lam_conv = 1.0 / dt_conv

    # A large uniform turbulent viscosity so the diffusion radius dominates in
    # at least some cells (thin/edge cells first).
    mu_turb = 50.0
    block.set_mu_turb(np.full(SHAPE, mu_turb, dtype=np.float32))
    grid.update_timestep(rf=1.0)
    dt_visc = block.dt_vol_nd

    # Closed-form diffusion radius from the same cell-centred fields the kernel
    # uses: sum over directions of |S_d|^2, each S_d the mean of the two opposing
    # face-area vectors, then the max-of-radii timestep.
    rho = block.conserved_cell_nd[..., 0]
    vol = block.vol_nd
    Si = 0.5 * (block.dAi_nd[:, :-1] + block.dAi_nd[:, 1:])
    Sj = 0.5 * (block.dAj_nd[:, :, :-1] + block.dAj_nd[:, :, 1:])
    Sk = 0.5 * (block.dAk_nd[:, :, :, :-1] + block.dAk_nd[:, :, :, 1:])
    s2 = (Si**2).sum(0) + (Sj**2).sum(0) + (Sk**2).sum(0)
    lam_diff = mu_turb * s2 / (rho * vol)
    expected = 1.0 / np.maximum(lam_conv, lam_diff)

    np.testing.assert_allclose(dt_visc, expected, rtol=1e-5)

    # Adding a diffusion radius can only shrink the step, and must shrink it
    # somewhere for a viscosity this large.
    assert np.all(dt_visc <= dt_conv * (1.0 + 1e-6))
    assert np.any(dt_visc < dt_conv)
