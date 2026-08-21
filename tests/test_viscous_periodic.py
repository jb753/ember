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


def _build_periodic_block(i_lims=None):
    """The k-periodic test block. ``i_lims`` splits the seam into subsets.

    Default is one PeriodicPatch spanning each whole k face. Passing a list of
    inclusive ``(start, end)`` i-ranges instead puts one patch per range, so
    the seam mixes periodic and wall along a single face -- the H-mesh shape,
    and the case that catches an exchange which assumes a patch covers its
    whole face.
    """
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

    for i_lim in (i_lims or [(0, -1)]):
        block.patches.append(PeriodicPatch(k=0, i=i_lim))
        block.patches.append(PeriodicPatch(k=-1, i=i_lim))
    return block


def _build_two_block_periodic():
    """Two blocks each spanning half a pitch, periodic to EACH OTHER in k.

    Both k seams cross blocks: the inner faces coincide at theta = 0 and the
    outer faces sit one pitch apart. A single self-periodic block cannot test
    a one-directional exchange -- both ends are the same block, so filling
    only one side still leaves every value right. This can.
    """
    Nb = 36
    pitch = 2.0 * np.pi / Nb
    shape = (5, 5, 5)

    def half(t0, t1):
        blk = ember.block.Block(shape=shape)
        blk.set_Nb(Nb)
        xrt = util.linmesh3((0.0, 0.1), (0.5, 1.0), (t0, t1), shape)
        blk.set_x(xrt[..., 0])
        blk.set_r(xrt[..., 1])
        blk.set_t(xrt[..., 2])
        blk.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
        blk.set_P_T(101325.0, 300.0)
        blk.set_wdist(np.zeros_like(blk.r))
        Vx = 100.0 + 20.0 * np.sin(2.0 * np.pi * blk.t / pitch + np.pi / 4.0)
        blk.set_Vx(Vx.astype(np.float32))
        blk.set_Vr(np.zeros_like(Vx, dtype=np.float32))
        blk.set_Vt(np.zeros_like(Vx, dtype=np.float32))
        blk.patches.append(PeriodicPatch(k=0))
        blk.patches.append(PeriodicPatch(k=-1))
        return blk

    return half(-pitch / 2, 0.0), half(0.0, pitch / 2)


def _fill_faces(block):
    """Run the O(surface) boundary producer into ``block.tau_q_faces``."""
    f = block.tau_q_faces
    ember.fortran.set_tau_q_faces(
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
        f_i1=f[0], f_ini=f[1], f_j1=f[2], f_jnj=f[3], f_k1=f[4], f_knk=f[5],
        **block.ijk_wall_visc,
    )
    return f


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
    # One carve for the whole viscous phase, so these cannot land on
    # top of the tau/q volume at the arena's head.
    _, _, planes, rows = ember.block._carve_viscous(block)
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
    # One carve for the whole viscous phase, so these cannot land on
    # top of the tau/q volume at the arena's head.
    _, _, planes, rows = ember.block._carve_viscous(block)
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


def test_exchange_faces_fills_both_blocks():
    """Both sides of a cross-block periodic pair get their halo layer filled.

    ``PeriodicCommunicator._prune_pairs`` reduces each pair to one key, and
    ``swap_by_ijk`` could exploit that because it swapped in place -- one call
    filled both sides. ``copy_faces_by_ij`` copies, so ``exchange_faces`` must
    issue both directions itself, and forgetting one is invisible on a
    self-periodic block: there both ends live in the same block, so the values
    still come out right. This case is two blocks, so it is not invisible.
    """
    up, dn = _build_two_block_periodic()
    grid = ember.grid.Grid([up, dn])
    comm = PeriodicCommunicator(grid, grid.connectivity.periodic.pair())

    for block in (up, dn):
        _fill_faces(block)

    # k1 is index 4 in the tau_q_faces tuple, knk index 5.
    before = [np.array(b.tau_q_faces[i][:, :, :, 1], copy=True)
              for b in (up, dn) for i in (4, 5)]
    comm.exchange_faces()
    after = [b.tau_q_faces[i][:, :, :, 1] for b in (up, dn) for i in (4, 5)]

    labels = ["up.k1", "up.knk", "dn.k1", "dn.knk"]
    unfilled = [n for n, b, a in zip(labels, before, after)
                if np.array_equal(b, a)]
    assert not unfilled, f"exchange_faces left {unfilled} untouched"

    # Each halo layer must be exactly its partner's owned layer -- a copy, so
    # bitwise, unlike anything that goes through the producer twice.
    np.testing.assert_array_equal(
        up.tau_q_faces[4][:, :, :, 1], dn.tau_q_faces[5][:, :, :, 0])
    np.testing.assert_array_equal(
        up.tau_q_faces[5][:, :, :, 1], dn.tau_q_faces[4][:, :, :, 0])
    np.testing.assert_array_equal(
        dn.tau_q_faces[4][:, :, :, 1], up.tau_q_faces[5][:, :, :, 0])
    np.testing.assert_array_equal(
        dn.tau_q_faces[5][:, :, :, 1], up.tau_q_faces[4][:, :, :, 0])


def test_exchange_faces_respects_subset_patches():
    """A patch covering part of a face exchanges exactly that part.

    Patches are not required to span their face -- an H-mesh puts two of them
    on each k face with the blade between. An exchange that assumed whole-face
    coverage, or that merged the patches on a face into one index list, would
    pass the full-span case and fail here.
    """
    # i nodes 0..4, so cells 0..3: periodic over the first and last cell only,
    # wall over the two in between.
    block = _build_periodic_block(i_lims=[(0, 1), (3, 4)])
    grid = ember.grid.Grid([block])
    comm = PeriodicCommunicator(grid, grid.connectivity.periodic.pair())
    _fill_faces(block)

    k1 = block.tau_q_faces[4]
    before = np.array(k1[:, :, :, 1], copy=True)
    comm.exchange_faces()

    # Which face cells the exchange touched, against which the wall mask calls
    # non-wall. They must be the same set: everything periodic and nothing else.
    changed = np.any(k1[:, :, :, 1] != before, axis=1)
    periodic = np.asarray(block.ijk_wall_visc["wallk1"])[:, :, 0] > 0
    assert periodic.any() and not periodic.all(), (
        "this case must mix periodic and wall on one face or it proves nothing"
    )
    np.testing.assert_array_equal(changed, periodic)

    # And the wall part must still hold the producer's -edge ghost. The mask
    # is per face cell, so it broadcasts across the component axis in between.
    wall = ~periodic[:, None, :]
    np.testing.assert_array_equal(
        np.where(wall, k1[:, :, :, 1], 0.0),
        np.where(wall, -k1[:, :, :, 0], 0.0),
    )


def _fvisc_x_faces(block, comm):
    """x-momentum viscous force through the surface-buffer path, end to end.

    Produces the halo with the O(surface) boundary kernel, exchanges it, and
    consumes it -- never touching the full-volume tau/q buffer, which is
    poisoned here to prove exactly that.
    """
    block.F_body_nd.flags.writeable = True
    block.F_body_nd.fill(0.0)

    _fill_faces(block)
    if comm is not None:
        comm.exchange_faces()

    # The claim is that this path needs nothing from the volume buffer, so
    # leave it holding nothing usable.
    block.tau_q_halo[...] = np.nan

    ni, nj, nk = block.shape
    # One carve for the whole viscous phase, so these cannot land on
    # top of the tau/q volume at the arena's head.
    _, _, planes, rows = ember.block._carve_viscous(block)
    tq = np.zeros((ni + 1, nj + 1, 9, 2), dtype=np.float32, order="F")
    f = block.tau_q_faces
    i_cusp_start, i_cusp_end = block.i_cusp
    ember.fortran.set_visc_force_tqf_faces(
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
        mu_turb=block._get_data_by_keys(
            ("mu_turb",), raise_uninit=False, writeable=True
        ),
        f_i1=f[0], f_ini=f[1], f_j1=f[2], f_jnj=f[3], f_k1=f[4], f_knk=f[5],
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
    not hasattr(ember.fortran, "set_visc_force_tqf_faces"),
    reason="bench arm viscous_tauq_faces not in this build",
)
@pytest.mark.parametrize("i_lims", [None, [(0, 1), (3, 4)]])
def test_viscous_faces_path_matches_production(i_lims):
    """The surface-buffer path reproduces production without the volume buffer.

    Run with the seam spanning the whole face and with it split into two
    subsets, because the subset case is what catches an exchange that assumed
    whole-face coverage.

    The tolerance is looser than the fused arms' own gates (~0.05 ulp) for a
    reason worth stating: ``set_tau_q_faces`` evaluates the producer per cell
    where ``set_tau_q_soa`` evaluates it a row at a time, and the two round
    differently. The gap concentrates in ``q(3)``, whose six-term sum collapses
    on an orthogonal mesh to a difference of two nearly-equal temperature sums,
    so its relative error is amplified by the cancellation. Every other
    component agrees to ~4e-7 of the field scale.
    """
    block = _build_periodic_block(i_lims=i_lims)
    grid = ember.grid.Grid([block])
    comm = PeriodicCommunicator(grid, grid.connectivity.periodic.pair())

    reference = _fvisc_x(block, comm)
    scale = np.max(np.abs(reference))
    got = _fvisc_x_faces(block, comm)

    assert not np.isnan(got).any(), (
        "the faces path produced NaN, so it read the poisoned volume buffer"
    )
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-5 * scale)


@pytest.mark.skipif(
    not hasattr(ember.fortran, "set_visc_force_tqf_faces"),
    reason="bench arm viscous_tauq_faces not in this build",
)
def test_viscous_faces_path_serves_a_sealed_block():
    """No periodic patches at all, and the faces path still works.

    This is the point of the surface buffers over the seam-free arm they
    replace: ``set_visc_force_tqf_selfk`` needs a block periodic to itself in
    k and refuses anything else, while this path is indifferent. A block whose
    every face is a wall exercises the ``(2*wall - 1)`` seeding with no
    exchange involved at all.
    """
    block = _build_periodic_block()
    for patch in list(block.patches.periodic):
        block.patches.remove(patch)
    block.clear_cache()

    assert not block.patches.periodic
    reference = _fvisc_x(block, None)
    got = _fvisc_x_faces(block, None)
    scale = np.max(np.abs(reference))
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-5 * scale)
