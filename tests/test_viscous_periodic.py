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

The same block gates the seam-free fused arm
(``bench/subroutines/viscous_tauq_selfk.f90``), which claims a block periodic
to itself in k needs no halo exchange at all because it can read its own far
cell plane instead.  That claim is exactly (A) and (B) above with the exchange
deleted rather than toggled, so it is tested here rather than by a tolerance
comparison in the bench: a wrong far-plane index moves the seam cells by tens
of percent, which an ulp gate on a 300k-cell case would report in the same
breath as compiler reassociation.  Skipped unless the arm is in the build
(``EMBER_BENCH_KERNELS=viscous_tauq_selfk``).
"""

import numpy as np
import pytest

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
        kappa=block.kappa_nd,
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
    kb = min(8, nk - 1)  # mirrors the ember.grid._KB_SLAB production clamp
    planes, rows = util.carve_view(block.scratch, (ni, nj, 4, 2), (ni, 4, 3))
    ember.fortran.set_visc_force(
        cons=block.conserved_nd,
        cons_cell=block.conserved_cell_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        omega_block=block.Omega_nd,
        r=block.r_nd,
        mu=block.mu_nd,
        p=block.P_nd,
        p_offset=block.P_offset_nd,
        fvisc=block.F_body_nd[..., 1:],
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        tau_cell=tau_cell,
        q_cell=q_cell,
        planes=planes,
        rows=rows,
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


def _fvisc_x_fused(block, entry="set_visc_force_tqf_selfk", n_tq=4,
                   between_phases=None):
    """x-momentum viscous body force from a fused arm, with NO exchange.

    ``between_phases`` is called after set_tau_q_soa and before the fused
    kernel.  It exists so a test can perturb the halo at exactly the point the
    exchange would have run -- set_tau_q_soa rewrites every halo slot on its
    way out (its "+edge" fill), so anything done to the halo BEFORE it is
    simply erased.

    Phase 1 still runs, for two reasons that are not the k seam: it fills the
    i/j halo edge slots the fused kernel reads on every k plane (via
    load_halo_ijedge), and it writes the mu_turb the caller compares.  What is
    deleted is the ``comm.exchange_halos()`` between the phases -- the thing
    this arm exists to remove.
    """
    block.F_body_nd.flags.writeable = True
    block.F_body_nd.fill(0.0)

    halo = block.tau_q_halo
    tau_cell = halo[..., 0:6]
    q_cell = halo[..., 6:9]
    mu_turb = block._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True)
    ember.fortran.set_tau_q_soa(
        cons=block.conserved_nd,
        t=block.T_nd,
        mu=block.mu_nd,
        cp=block.cp_nd,
        kappa=block.kappa_nd,
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

    # NO exchange_halos here. That is the whole point. Anything the caller
    # wants done in its place happens now, after phase 1 has finished writing
    # the halo and before the fused kernel reads it.
    if between_phases is not None:
        between_phases()

    ni, nj, nk = block.shape
    planes, rows = util.carve_view(block.scratch, (ni, nj, 4, 2), (ni, 4, 3))
    # Four rolling/saved tau/q planes. Allocated rather than carved from
    # block.scratch: at this block size scratch holds 1125 floats and the arm
    # needs 1556. A real integration wants its own buffer anyway.
    tq = np.zeros((ni + 1, nj + 1, 9, n_tq), dtype=np.float32, order="F")
    i_cusp_start, i_cusp_end = block.i_cusp
    getattr(ember.fortran, entry)(
        cons=block.conserved_nd,
        cons_cell=block.conserved_cell_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        omega_block=block.Omega_nd,
        r=block.r_nd,
        mu=block.mu_nd,
        p=block.P_nd,
        p_offset=block.P_offset_nd,
        fvisc=block.F_body_nd[..., 1:],
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        t=block.T_nd,
        cp=block.cp_nd,
        kappa=block.kappa_nd,
        pr_turb=0.9,
        xlength=block.xlen_sq_nd,
        mu_turb=mu_turb,
        tau_cell=tau_cell,
        q_cell=q_cell,
        tq=tq,
        planes=planes,
        rows=rows,
        kb=min(8, nk - 1),
        **block.ijk_wall_visc,
        **block.Omega_wall_nd,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
    )
    block.F_body_nd.flags.writeable = False
    return block.F_body_nd[..., 1].copy()


@pytest.mark.skipif(
    not hasattr(ember.fortran, "set_visc_force_tqf_selfk"),
    reason="bench arm viscous_tauq_selfk not in this build",
)
def test_viscous_selfk_seam_free_matches_exchange():
    """The seam-free arm reproduces the exchanged answer without exchanging.

    This is the claim the arm is built on: for a block periodic to ITSELF in
    k, the halo exchange is a copy from the block's own far cell plane, so the
    kernel can read that plane directly and skip the exchange entirely.

    Gated against `fx_exchange` rather than against the no-exchange reference,
    because those two differ by tens of percent AT the seam -- a kernel that
    quietly fell back to the zero-gradient seam would sail through a
    comparison with `fx_noexchange` and fail this one.
    """
    block = _build_periodic_block()
    grid = ember.grid.Grid([block])
    comm = PeriodicCommunicator(grid, grid.connectivity.periodic.pair())

    fx_exchange = _fvisc_x(block, comm)
    fx_noexchange = _fvisc_x(block, None)
    fx_selfk = _fvisc_x_selfk(block)

    scale = np.max(np.abs(fx_exchange))
    tol = 1e-6 * scale

    # The seam must actually matter here, or the test proves nothing: if the
    # exchange were a no-op on this case, a kernel that ignored the seam
    # entirely would pass. So require the exchange to move fvisc by at least
    # 100x the tolerance the comparison below runs at -- that ratio IS the
    # test's discriminating power.
    #
    # The margin is deliberately modest because this block is built to be
    # smooth and exactly periodic across the seam (that is what makes claim
    # (B) meaningful), so the exchange only has the float32 truncation of a
    # continuous field to correct: ~5e-4 of scale here, against a 1e-6
    # comparison. A case with a discontinuity at the seam would separate them
    # much further, but would not support the twin test.
    seam_gap = np.max(np.abs(fx_exchange - fx_noexchange))
    assert seam_gap > 100.0 * tol, (
        f"the exchange moves fvisc by only {seam_gap:.3e} ({seam_gap / scale:.1e} "
        "of scale) on this case, so it cannot discriminate a seam-free kernel"
    )

    # The whole field, not just the interior column (A) inspects: the seam-free
    # arm reproduces production everywhere, seam cells included.
    np.testing.assert_allclose(fx_selfk, fx_exchange, rtol=0, atol=tol)

    # And it must not merely be reproducing the zero-gradient seam.
    assert np.max(np.abs(fx_selfk - fx_noexchange)) > 100.0 * tol


def _fvisc_x_selfk(block, between_phases=None):
    """The seam-free arm: four tau/q planes, no exchange."""
    return _fvisc_x_fused(block, "set_visc_force_tqf_selfk", 4, between_phases)


def _fvisc_x_tqf(block, between_phases=None):
    """The PARENT fused arm, which does read the k halo. Reference for the
    poison test below: same call shape, two rolling tau/q planes not four."""
    return _fvisc_x_fused(block, "set_visc_force_tqf", 2, between_phases)


@pytest.mark.skipif(
    not hasattr(ember.fortran, "set_visc_force_tqf_selfk"),
    reason="bench arm viscous_tauq_selfk not in this build",
)
def test_viscous_selfk_never_reads_the_k_halo():
    """The seam-free arm is independent of the k halo slots, provably.

    The transparency test above shows the arm gets the right answer without an
    exchange.  This shows *why*: it never reads the exchanged slots at all.
    Poisoning the two k-direction halo planes with NaN leaves its output
    bitwise unchanged, while the same poison floods the parent arm's fvisc
    with NaN.

    That is the claim "there is nothing to exchange" reduced to something a
    single assertion can establish, and unlike a timing comparison it cannot
    be confounded by codegen.  Only the k planes are poisoned: the i/j halo
    edges are unrelated to the k seam and both arms legitimately read them.
    """
    block = _build_periodic_block()
    grid = ember.grid.Grid([block])
    comm = PeriodicCommunicator(grid, grid.connectivity.periodic.pair())
    _fvisc_x(block, comm)  # seed the halo, including the i/j edges

    ni, nj, nk = block.shape
    halo = block.tau_q_halo
    clean = np.array(halo, copy=True)

    def poison():
        # Only the two k-direction halo planes. The i/j edges are unrelated to
        # the k seam and both arms legitimately read them.
        halo[1:ni, 1:nj, 0, :] = np.nan
        halo[1:ni, 1:nj, nk, :] = np.nan

    reference = _fvisc_x_selfk(block)
    poisoned = _fvisc_x_selfk(block, between_phases=poison)

    # The poison must still be in place when the kernel returns -- the fused
    # arms take tau_cell/q_cell as intent(in), so nothing should have cleared
    # it, and if phase 1 had erased it (it rewrites every halo slot on its way
    # out) this test would be asserting nothing at all.
    assert np.isnan(halo[1:ni, 1:nj, 0, :]).all()

    assert not np.isnan(poisoned).any()
    np.testing.assert_array_equal(poisoned, reference)

    # Trap armed: the same poison at the same point floods the PARENT arm,
    # which does read those slots. Without this, a kernel that had somehow
    # stopped reading tau_cell entirely would pass on a technicality.
    if hasattr(ember.fortran, "set_visc_force_tqf"):
        assert np.isnan(_fvisc_x_tqf(block, between_phases=poison)).any(), (
            "poisoning the k halo left set_visc_force_tqf unaffected, so the "
            "poison is not reaching the slots this test is about"
        )
    halo[...] = clean
