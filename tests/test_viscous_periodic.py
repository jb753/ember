"""Periodic-seam transparency for the viscous body force.

A single block is made periodic in theta (the k direction) and carries an axial
velocity that varies sinusoidally in theta with *two* full wavelengths across
the pitch.  The field is smooth and exactly periodic across the seam, so the
viscous seam exchange should make the seam transparent:

  (A) toggling the exchange on/off may change only the two seam-adjacent
      k-cells -- never an interior cell; and

  (B) because the pitch holds two wavelengths, every phase appears twice (once
      next to the seam, once in the interior), so the force on a seam cell must
      equal the force on its interior twin half a domain away.

The seam (theta = 0) is placed at a generic phase (pi/4 offset) so that a
non-transparent boundary cannot be masked by the field being even or odd about
the seam.

What crosses the seam is O(surface): ``set_tau_q_faces`` writes each boundary
face's own tau/q into layer 0 of its buffer and a ``(2*wall - 1)`` ghost into
layer 1, and ``exchange_faces`` overwrites layer 1 with the partner's layer 0
wherever a patch connects. So these tests gate the whole seam mechanism, and
the two-block tests below gate the direction handling that a block periodic to
itself cannot: with both ends the same block, filling only one side still
leaves every value right.
"""
import numpy as np

import ember.block
import ember.fortran
import ember.grid
from ember import util
from ember.fluid import PerfectFluid
from ember.periodic import PeriodicPatch
from ember.periodic_communicator import PeriodicCommunicator

import viscous_util

# Turbulent Prandtl number for the pair. Any fixed value does: the fixture
# is laminar (zero wall distance, so zero mixing length) and it only
# multiplies a term that is zero here.
PR_TURB = 0.9


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


def _fvisc_x(block, comm):
    """x-momentum viscous body force, cell-centred, with/without exchange.

    ``comm=None`` skips the exchange, which leaves the seam faces reading the
    ``+edge`` ghost their own producer seeded instead of the neighbour's edge
    cell -- the toggle test (A) is the difference between the two.
    """
    viscous_util.run_pair(block, PR_TURB, comm=comm)
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
        viscous_util.fill_faces(block, PR_TURB)

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
    viscous_util.fill_faces(block, PR_TURB)

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
