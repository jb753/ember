"""``set_residual``'s cusp k-face correction, on a block that actually has one.

The correction (``correct_cusp_kface_du`` in ``_fortran/residual.f90``) rebuilds
the two seam k-faces from seam-averaged mass flow, velocity and pressure, and
is deferred: the rolling sweep leaves the raw one-sided fluxes in the two seam
cells and this pass adds the difference. Every grid the rest of the suite builds
has ``i_cusp == (0, 0)``, which skips the whole thing --- ``test_viscous_cusp
_seam`` says so of the viscous kernel's copy, and it is just as true here, so
without this module the path has no numerical coverage at all.

These tests do not pin values; they pin that the correction fires, and that it
lands where the deferred scheme says it can land and nowhere else.
"""

import numpy as np
import pytest

import ember.block
import ember.fortran
from ember import util
from test_viscous_cusp_seam import _build_block


def _run_residual(block, i_cusp, j_cusp=None):
    """``set_residual`` as Grid.update_residual drives it, cusp span given.

    ``j_cusp`` defaults to the block's own, so a caller that cares only about
    the streamwise span need not spell it.
    """
    if j_cusp is None:
        j_cusp = block.j_cusp
    ni, nj, nk = block.shape
    block.update_primitive()
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
    planes, rows = util.carve_view(block.scratch, (ni, njp, 5, 2), (ni, 5, 3))
    du = block.residual_nd
    du.flags.writeable = True
    du.fill(0.0)
    ember.fortran.set_residual(
        cons=block.conserved_nd,
        p=block.P_nd,
        p_offset=block.P_offset_nd,
        r=block.r_nd,
        omega=block.Omega_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        du=du,
        f_body=block.F_body_nd,
        planes=planes,
        rows=rows,
        **block.ijk_wall_conv,
        i_cusp_start=i_cusp[0],
        i_cusp_end=i_cusp[1],
        j_cusp_start=j_cusp[0],
        j_cusp_end=j_cusp[1],
        kb=min(8, nk - 1),
        njp=njp,
        ni=ni,
        nj=nj,
        nk=nk,
    )
    du.flags.writeable = False
    return np.array(du)


@pytest.fixture
def cusped():
    """The cusp-seam fixture's block, which is the one grid with i_cusp != 0."""
    block = _build_block()
    assert block.i_cusp != (0, 0), "fixture lost its cusp; these tests prove nothing"
    return block


def test_cusp_correction_fires(cusped):
    """Switching the cusp span on changes the residual.

    ``i_cusp = (0, 0)`` is the disabled form the whole rest of the suite runs,
    so this is the only thing separating the corrected kernel from the one
    every other test exercises.
    """
    off = _run_residual(cusped, (0, 0))
    on = _run_residual(cusped, cusped.i_cusp)
    assert not np.array_equal(off, on), "cusp span made no difference to dU"
    scale = float(np.abs(off).max())
    assert np.abs(on - off).max() > 1e-4 * scale, (
        "cusp correction is within float noise of doing nothing"
    )


def test_cusp_correction_touches_only_the_seam_planes(cusped):
    """It reaches the two seam cell planes and no interior one.

    The correction couples the k=1 and k=nk faces, so the cells it may alter
    are the first and last in k. An interior cell moving would mean the
    deferred pass had reached past the seam it is meant to repair.
    """
    off = _run_residual(cusped, (0, 0))
    on = _run_residual(cusped, cusped.i_cusp)
    moved = np.abs(on - off) > 0.0
    nk_cell = off.shape[2]
    assert moved[:, :, 0].any() and moved[:, :, nk_cell - 1].any()
    assert not moved[:, :, 1 : nk_cell - 1].any(), (
        "cusp correction altered an interior k plane"
    )


def test_cusp_correction_honours_a_partial_j_span(cusped):
    """A cusp over part of the span corrects that part and nothing else.

    A blade that stops short of the hub or casing leaves a gap whose k faces
    carry no cusp, so the correction must skip those j. Both halves are pinned
    exactly: inside the span the partial run must reproduce the full-span run
    bit for bit, outside it the uncorrected run, because the deferred pass is
    a per-(i, j) update with no coupling along j.
    """
    nj = cusped.shape[1]
    jst, jen = 3, 7  # 1-based nodes, i.e. cells 3..6 -> 0-based 2:6
    assert 1 < jst and jen < nj, "span must be interior for this test to bite"

    off = _run_residual(cusped, (0, 0))
    full = _run_residual(cusped, cusped.i_cusp, (1, nj))
    part = _run_residual(cusped, cusped.i_cusp, (jst, jen))

    inside = slice(jst - 1, jen - 1)
    assert np.array_equal(part[:, inside], full[:, inside]), (
        "partial span did not reproduce the full span inside its own j range"
    )
    outside = np.ones(off.shape[1], dtype=bool)
    outside[inside] = False
    assert np.array_equal(part[:, outside], off[:, outside]), (
        "partial span corrected cells outside its j range"
    )
    assert not np.array_equal(part[:, inside], off[:, inside]), (
        "partial span made no difference where it should have"
    )


def test_full_j_span_is_the_block_default(cusped):
    """Spelling the full span explicitly is the same run as the block's own."""
    nj = cusped.shape[1]
    assert cusped.j_cusp == (1, nj)
    assert np.array_equal(
        _run_residual(cusped, cusped.i_cusp),
        _run_residual(cusped, cusped.i_cusp, (1, nj)),
    )
