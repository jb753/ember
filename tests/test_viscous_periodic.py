"""Periodic-seam transparency for the viscous body force.

A single block is made periodic in theta (the k direction) and carries an axial
velocity that varies sinusoidally in theta with *two* full wavelengths across
the pitch.  The field is smooth and exactly periodic across the seam, so the
viscous halo exchange should make the seam transparent:

  (A) toggling the exchange on/off may change only the two seam-adjacent
      k-cells -- never an interior cell; and

  (B) because the pitch holds two wavelengths, every phase appears twice (once
      next to the seam, once in the interior), so the force on a seam cell must
      equal the force on its interior twin half a domain away.

The seam (theta = 0) is placed at a generic phase (pi/4 offset) so that a
non-transparent boundary cannot be masked by the field being even or odd about
the seam.
"""

import numpy as np

import ember.block
import ember.fortran
import ember.grid
from ember import util
from ember.fluid import PerfectFluid
from ember.periodic import PeriodicPatch
from ember.periodic_communicator import PeriodicCommunicator


def _build_periodic_block():
    Nb = 36
    pitch = 2.0 * np.pi / Nb
    shape = (5, 5, 9)  # 8 theta cells -> 2 wavelengths => twins 4 cells apart

    block = ember.block.Block(shape=shape)
    block.set_Nb(Nb)
    xrt = util.linmesh3((0.0, 0.1), (0.5, 1.0), (0.0, pitch), shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
    block.set_P_T(101325.0, 300.0)
    # Laminar: zero wall distance => zero mixing length (no turbulent viscosity).
    block.set_wdist(np.zeros_like(block.r))

    # Two wavelengths across the pitch, phase-shifted by pi/4 so the seam sits
    # at a generic point of the force profile.
    Vx = 100.0 + 20.0 * np.sin(4.0 * np.pi * block.t / pitch + np.pi / 4.0)
    block.set_Vx(Vx.astype(np.float32))
    # Initialise the remaining momenta (the viscous wrappers read the public
    # Vr_nd / Vt_rel_nd getters, which require every momentum component set).
    block.set_Vr(np.zeros_like(Vx, dtype=np.float32))
    block.set_Vt(np.zeros_like(Vx, dtype=np.float32))

    block.patches.append(PeriodicPatch(k=0))
    block.patches.append(PeriodicPatch(k=-1))
    return block


def _fvisc_x(block, comm):
    """x-momentum viscous body force, cell-centred, with/without exchange."""
    # F_body_nd is a read-only cached buffer; this test owns its lifecycle here.
    block.F_body_nd.flags.writeable = True
    block.F_body_nd.fill(0.0)

    # First viscous phase: tau/q per cell (mirrors Grid.update_sources).
    halo = block.tau_q_halo
    tau_cell = halo[..., 0:6]
    q_cell = halo[..., 6:9]
    mu_turb = block._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True)
    ember.fortran.set_tau_q_soa(
        cons=block.conserved_nd,
        t=block.T_nd,
        mu=block.mu_nd,
        cp=block.cp_nd,
        pr_lam=block.fluid._Pr,
        pr_turb=0.9,
        xlength=block.xlen_sq_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        r=block.r_nd,
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        tau_cell=tau_cell,
        q_cell=q_cell,
        mu_turb=mu_turb,
    )
    block._versions["mu_turb"] += 1

    if comm is not None:
        comm.exchange_halos()

    # Second viscous phase: face fluxes from tau/q, accumulated into F_body_nd.
    i_cusp_start, i_cusp_end = block.i_cusp
    ni, nj, nk = block.shape
    kb = min(8, nk - 1)  # mirrors the ember.grid._KB_VISC production clamp
    flow_scratch = util.carve_view(block.scratch, (ni, nj, kb + 1, 4))
    ember.fortran.set_visc_force(
        cons=block.conserved_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        omega_block=block.Omega_nd,
        r=block.r_nd,
        mu=block.mu_nd,
        fvisc=block.F_body_nd[..., 1:],
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        tau_cell=tau_cell,
        q_cell=q_cell,
        flow_scratch=flow_scratch,
        kb=kb,
        **block.ijk_wall_visc,
        **block.Omega_wall_nd,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
    )

    block.F_body_nd.flags.writeable = False
    return block.F_body_nd[..., 1].copy()


def test_viscous_periodic_seam_transparent():
    block = _build_periodic_block()
    grid = ember.grid.Grid([block])
    comm = PeriodicCommunicator(grid, grid.connectivity.periodic.pair())

    fx_exchange = _fvisc_x(block, comm)
    fx_noexchange = _fvisc_x(block, None)

    nk_cell = block.shape[2] - 1  # 8 theta cells
    half = nk_cell // 2  # 4 cells = one wavelength
    seam_cells = {0, nk_cell - 1}  # the two k-cells adjacent to the seam

    # Inspect a surviving interior (i, j) column: the wall-adjacent i/j cells
    # are zeroed for the viscous force, so the physics there is purely theta.
    fe = fx_exchange[1, 1, :]
    fn = fx_noexchange[1, 1, :]

    tol = 1e-6 * np.max(np.abs(fn))

    # (A) Only the seam-adjacent cells may differ when the exchange is toggled.
    changed = set(np.nonzero(np.abs(fe - fn) > tol)[0].tolist())
    assert changed <= seam_cells, (
        f"exchange changed non-seam cells {sorted(changed - seam_cells)}; "
        "the periodic halo exchange is leaking into the interior"
    )

    # (B) Two wavelengths => each seam cell equals its interior twin once the
    # seam is transparent (cell k matches cell k + half).
    np.testing.assert_allclose(fe[:half], fe[half:], rtol=0, atol=tol)
