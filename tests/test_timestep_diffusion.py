"""Turbulent-diffusion limit in the spectral timestep.

``Grid.update_timestep`` (via ``set_timestep_spectral``) sets the unscaled
volumetric timestep to ``1 / max(lam_conv, lam_diff)`` where ``lam_conv`` is the
convective spectral radius (max over the three directions) and ``lam_diff =
(mu_turb/rho)*max_d||dA_d||^2/vol`` is a directional turbulent-diffusion spectral
radius over the same per-cell face areas (both convective and diffusion radii
take the max over the three directions, not the sum). These tests pin the two
regimes:

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
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
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
    # uses: max over directions of |S_d|^2, each S_d the mean of the two opposing
    # face-area vectors, then the max-of-radii timestep.
    rho = block.conserved_cell_nd[..., 0]
    vol = block.vol_nd
    Si = 0.5 * (block.dAi_nd[:, :-1] + block.dAi_nd[:, 1:])
    Sj = 0.5 * (block.dAj_nd[:, :, :-1] + block.dAj_nd[:, :, 1:])
    Sk = 0.5 * (block.dAk_nd[:, :, :, :-1] + block.dAk_nd[:, :, :, 1:])
    s2 = np.maximum.reduce([(Si**2).sum(0), (Sj**2).sum(0), (Sk**2).sum(0)])
    lam_diff = mu_turb * s2 / (rho * vol)
    expected = 1.0 / np.maximum(lam_conv, lam_diff)

    np.testing.assert_allclose(dt_visc, expected, rtol=1e-5)

    # Adding a diffusion radius can only shrink the step, and must shrink it
    # somewhere for a viscosity this large.
    assert np.all(dt_visc <= dt_conv * (1.0 + 1e-6))
    assert np.any(dt_visc < dt_conv)


def _avg_cell(x):
    """8-node corner average to cell centres, matching the kernel's avg_cell."""
    return 0.125 * (
        x[:-1, :-1, :-1] + x[1:, :-1, :-1] + x[:-1, 1:, :-1] + x[1:, 1:, :-1]
        + x[:-1, :-1, 1:] + x[1:, :-1, 1:] + x[:-1, 1:, 1:] + x[1:, 1:, 1:]
    )


def test_convective_timestep_matches_naive_directional_max():
    """Inviscid dt_vol equals a naive max-of-directional-radii reference.

    With the max normalisation, ``set_timestep_spectral`` sets
    ``dt_vol = 1 / max_d(|V.S_d| + a||S_d||)`` over the three directions (with
    ``mu_turb = 0``). This reimplements that formula directly in NumPy from the
    block's cell-centred nondimensional fields on a uniform grid and checks the
    Fortran kernel reproduces it -- and, crucially, that it is the *max* and not
    the Blazek *sum* of the directional radii.
    """
    grid = _build_grid()
    block = grid[0]

    grid.update_timestep(rf=1.0)  # mu_turb = 0 init -> convective only
    dt = np.asarray(block.dt_vol_nd)

    a = _avg_cell(np.asarray(block.a_nd))
    r = _avg_cell(np.asarray(block.r_nd))
    cons = np.asarray(block.conserved_cell_nd)
    rho = cons[..., 0]
    Vx = cons[..., 1] / rho
    Vr = cons[..., 2] / rho
    Vt = cons[..., 3] / (rho * r)
    Vt_rel = Vt - float(block.Omega_nd) * r

    # Per-direction spectral radius from the mean of the two opposing face-area
    # vectors, exactly as the kernel builds S_d.
    dAi, dAj, dAk = (
        np.asarray(block.dAi_nd),
        np.asarray(block.dAj_nd),
        np.asarray(block.dAk_nd),
    )
    Si = 0.5 * (dAi[:, :-1] + dAi[:, 1:])
    Sj = 0.5 * (dAj[:, :, :-1] + dAj[:, :, 1:])
    Sk = 0.5 * (dAk[:, :, :, :-1] + dAk[:, :, :, 1:])

    def lam(S):
        Sx, Sr, St = S
        return np.abs(Vx * Sx + Vr * Sr + Vt_rel * St) + a * np.sqrt(
            Sx**2 + Sr**2 + St**2
        )

    lam_i, lam_j, lam_k = lam(Si), lam(Sj), lam(Sk)
    lam_max = np.maximum(np.maximum(lam_i, lam_j), lam_k)
    np.testing.assert_allclose(dt, 1.0 / lam_max, rtol=1e-5)

    # Pin that it is the max and NOT the sum: on this grid the directional sum is
    # strictly larger somewhere, so the summed timestep would be detectably
    # smaller -- guards against a regression to the Blazek sum-of-radii form.
    lam_sum = lam_i + lam_j + lam_k
    assert np.any(lam_sum > lam_max * (1.0 + 1e-3))
    assert not np.allclose(dt, 1.0 / lam_sum, rtol=1e-3)
