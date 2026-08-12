"""Sign of the cusp-seam correction in ``set_visc_force``.

Motivated by the turbine-case regression bisected to ``2381658745`` "Fuse
polar source into set_visc_force and drop its separate negation pass" (see
ember-paper's ``turbine/run_final-divergence-findings.md``).

That commit made three changes, not the two its message enumerates:

1. the i/j/k face-difference accumulates were flipped from low-minus-high to
   high-minus-low,
2. the polar source was fused into the kernel's final pass, and
3. **the trailing full-array negation pass was deleted.**

(3) is the one with no test coverage. That pass did not only negate the face
accumulates -- it negated *everything* already accumulated into ``fvisc`` at
that point, and there is exactly one other such term: the cusp-seam
correction, which replaces each seam cell's one-sided k-face flux with the
average of the two fluxes meeting at the modelled trailing edge. The
correction itself was left untouched by the commit, so the sign flip in (1)
must be mirrored in it -- ``fcorr`` is not a face difference, it is a
*replacement delta*, and its own internal high-minus-low ordering is
irrelevant to how it composes with the accumulation.

The derivation, for the low seam cell ``kc = 1`` (the high seam cell
``kc = nk-1`` gives the same answer, which is why one ``fcorr`` serves both):

* pre-flip, the cell held ``flux(1) - flux(2)``, so swapping ``flux(1)`` for
  ``avg = 0.5*(flux(1) + flux(nk))`` means adding ``avg - flux(1) = +fcorr``,
  after which the global negation flipped accumulate and correction together;
* post-flip, the cell holds ``flux(2) - flux(1)`` already in the residual's
  sign convention, so the swap means adding ``flux(1) - avg = -fcorr``.

The kernel still adds ``+fcorr`` (``viscous.f90``'s cusp block, whose comment
argues the correction "needs no adjustment" because its sign already matches
the accumulate's -- precisely the reason it does need one). If that is the
bug, each seam cell carries an anti-diffusive error of ``2*fcorr`` every
step.

No existing test can see this: the correction is gated behind
``i_cusp_start > 0``, which is non-zero only for a block carrying a
:class:`~ember.cusp.CuspPatch`, and no viscous test builds one --
``test_viscous_periodic`` uses :class:`~ember.periodic.PeriodicPatch`, a
different mechanism, and ``test_viscous_phases_golden`` (regenerated inside
the suspect commit) and ``test_viscous_polar_mu_limit`` both run with
``i_cusp = (0, 0)``. The branch is dead code under the suite. The LISA
turbine reaches it: ``turbine/setup.py`` calls
``grid.connectivity.cusp.pair()``.

Method: run ``set_visc_force`` twice on identical state, once with the
correction disabled (``i_cusp_start = 0``) and once with it spanning the
block, and compare the difference against ``fcorr`` recomputed independently
in numpy from the same tau/q field. Because the two runs differ *only* by
that branch, the difference isolates the correction exactly, and every other
term -- including the fused polar source -- cancels.
"""

import numpy as np

import ember.block
import ember.fortran
from ember import util
from ember.cusp import CuspPatch
from ember.fluid import PerfectFluid
from ember.inviscid import InviscidPatch

SHAPE = (7, 9, 9)
NB = 36
PR_TURB = 1.0
MU = 1.8e-5  # production magnitude, as in test_viscous_phases_golden


def _build_block():
    """Rotating, swirling, sheared single block with a cusp on both k faces.

    Geometry and flow field follow ``test_viscous_polar_mu_limit``'s fixture.
    The i/j faces are frictionless so that every wall mask is 1.0: the wall
    zeroing pass then cannot mask off the seam cells the correction lands on,
    and the entry halo scaling by ``(2*wall - 1)`` is the identity, which is
    what lets the numpy reference below read the tau/q field as
    ``set_tau_q_soa`` left it. Both properties are asserted, not assumed.

    The k faces carry cusp patches rather than periodic ones: that is what
    makes ``block.i_cusp`` non-zero, and it is the configuration the LISA
    turbine's blade rows actually run in.
    """
    pitch = 2.0 * np.pi / NB

    block = ember.block.Block(shape=SHAPE)
    block.set_Nb(NB)
    xrt = util.linmesh3((0.0, 0.15), (0.5, 0.9), (0.0, pitch), SHAPE)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=MU, Pr=0.72))

    x, r, t = block.x, block.r, block.t
    r_span = float(r.max() - r.min())

    Temp = (
        300.0
        + 20.0 * (r - r.min()) / r_span
        + 8.0 * np.sin(2.0 * np.pi * x / float(x.max()))
    ).astype(np.float32)
    block.set_P_T(101325.0, Temp)

    # Deliberately NOT periodic in theta: a field that matched across the seam
    # would give flow(k=1) == flow(k=nk), hence fcorr == 0, and the test would
    # pass whatever sign the kernel used. The 1.5 wavelengths and the linear
    # theta ramp on Vt both break that symmetry.
    Vx = (
        100.0
        + 20.0 * np.sin(3.0 * np.pi * t / pitch + np.pi / 4.0)
        + 10.0 * (r - r.min()) / r_span
    ).astype(np.float32)
    Vr = (5.0 * np.cos(3.0 * np.pi * t / pitch)).astype(np.float32)
    Vt = (
        40.0
        + 15.0 * np.sin(2.0 * np.pi * x / float(x.max()))
        + 12.0 * (t - t.min()) / pitch
    ).astype(np.float32)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    block.set_Omega(50.0)

    wdist = 0.008 * (1.0 + np.sin(np.pi * (r - r.min()) / r_span))
    block.set_wdist(wdist.astype(np.float32))

    block.patches.append(CuspPatch(k=0))
    block.patches.append(CuspPatch(k=-1))
    block.patches.append(InviscidPatch(i=0))
    block.patches.append(InviscidPatch(i=-1))
    block.patches.append(InviscidPatch(j=0))
    block.patches.append(InviscidPatch(j=-1))
    return block


def _set_tau_q(block):
    """Viscous phase 1, as Grid.update_sources runs it. Returns tau/q views."""
    halo = block.tau_q_halo
    halo.fill(0.0)
    tau_cell = halo[..., 0:6]
    q_cell = halo[..., 6:9]
    mu_turb = block._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True)
    ember.fortran.set_tau_q_soa(
        cons=block.conserved_nd,
        t=block.T_nd,
        mu=block.mu_nd,
        cp=block.cp_nd,
        pr_lam=block.fluid._Pr,
        pr_turb=PR_TURB,
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
    return tau_cell, q_cell


def _run_visc_force(block, i_cusp):
    """Viscous phase 2 with the cusp correction spanning ``i_cusp``.

    tau/q are rebuilt from scratch on every call: ``set_visc_force`` scales
    the tau/q boundary halos in place on entry, so a second call on the same
    buffer would otherwise see a twice-scaled field.
    """
    tau_cell, q_cell = _set_tau_q(block)

    fbody = block.F_body_nd
    fbody.flags.writeable = True
    fbody.fill(0.0)

    ni, nj, nk = block.shape
    kb = min(8, nk - 1)
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
        fvisc=fbody[..., 1:],
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
        i_cusp_start=i_cusp[0],
        i_cusp_end=i_cusp[1],
    )
    fbody.flags.writeable = False
    # Components 1: of F_body_nd are set_visc_force's own fvisc(:,:,:,1:4).
    return np.array(fbody[..., 1:], dtype=np.float64)


def _kface_flow(block, tau_cell, q_cell, kf):
    """Raw viscous flux through k-face plane ``kf`` (1-based), all (i, j).

    An expression-for-expression numpy transcription of ``kface_flow`` in
    ``viscous.f90`` -- the same helper the kernel's cusp correction calls --
    so the reference is independent of how the correction composes with the
    accumulation, which is the thing under test. Returns (ni-1, nj-1, 4).
    """
    ni, nj, _ = block.shape
    kk = kf - 1  # Fortran 1-based face plane -> numpy index

    # tau_cell(i+1, j+1, kf, m) for i = 1..ni-1, j = 1..nj-1 (halo-indexed).
    def _cell_avg(fld, m):
        return 0.5 * (
            fld[1:ni, 1:nj, kk, m].astype(np.float64)
            + fld[1:ni, 1:nj, kk + 1, m].astype(np.float64)
        )

    tauf = [_cell_avg(tau_cell, m) for m in range(6)]
    qf = [_cell_avg(q_cell, m) for m in range(3)]

    def _node_avg(fld):
        f = np.asarray(fld, dtype=np.float64)
        return 0.25 * (
            f[:-1, :-1, kk] + f[1:, :-1, kk] + f[:-1, 1:, kk] + f[1:, 1:, kk]
        )

    Vf = [_node_avg(f) for f in (block.Vx_nd, block.Vr_nd, block.Vt_rel_nd)]
    rf = _node_avg(block.r_nd)
    Vabs = Vf[2] + float(block.Omega_nd) * rf

    dA = np.asarray(block.dAk_nd, dtype=np.float64)
    dA1, dA2, dA3 = dA[0, :, :, kk], dA[1, :, :, kk], dA[2, :, :, kk]

    flow = np.empty(tauf[0].shape + (4,), dtype=np.float64)
    flow[..., 0] = tauf[0] * dA1 + tauf[3] * dA2 + tauf[4] * dA3
    flow[..., 1] = tauf[3] * dA1 + tauf[1] * dA2 + tauf[5] * dA3
    flow[..., 2] = (tauf[4] * dA1 + tauf[5] * dA2 + tauf[2] * dA3) * rf
    wvisc = (
        Vf[0] * tauf[0] + Vf[1] * tauf[3] + Vabs * tauf[4],
        Vf[0] * tauf[3] + Vf[1] * tauf[1] + Vabs * tauf[5],
        Vf[0] * tauf[4] + Vf[1] * tauf[5] + Vabs * tauf[2],
    )
    flow[..., 3] = (
        (wvisc[0] - qf[0]) * dA1
        + (wvisc[1] - qf[1]) * dA2
        + (wvisc[2] - qf[2]) * dA3
    )
    return flow


def test_fixture_leaves_every_wall_mask_free():
    """The reference below assumes it: with any mask at 0 the seam cells would
    be zeroed after the correction (hiding it) and the entry halo scaling by
    (2*wall - 1) would no longer be the identity."""
    block = _build_block()
    for name, mask in block.ijk_wall_visc.items():
        assert np.all(np.asarray(mask) == 1.0), f"{name} is not all-free"


def test_fixture_reaches_the_cusp_branch():
    """A block with no cusp patch gives i_cusp = (0, 0) and skips the whole
    correction -- which is exactly why the rest of the suite never sees it."""
    block = _build_block()
    ni = block.shape[0]
    assert block.i_cusp == (1, ni), block.i_cusp


def test_cusp_seam_correction_sign():
    """The seam correction must carry the residual's sign convention.

    Isolates the correction as the difference between two otherwise identical
    kernel runs, and checks it against ``-fcorr`` -- the replacement delta
    derived in the module docstring. A failure at exactly ``-1x`` the expected
    value is the sign flip that commit ``2381658745`` left behind.
    """
    block = _build_block()

    f_off = _run_visc_force(block, (0, 0))
    f_on = _run_visc_force(block, block.i_cusp)
    delta = f_on - f_off

    tau_cell, q_cell = _set_tau_q(block)
    _, _, nk = block.shape
    flow1 = _kface_flow(block, tau_cell, q_cell, 1)
    flownk = _kface_flow(block, tau_cell, q_cell, nk)
    fcorr = 0.5 * (flownk - flow1)
    # Cell kc=1 holds flux(2) - flux(1) and cell kc=nk-1 holds
    # flux(nk) - flux(nk-1); swapping the seam face for the average of the two
    # is -fcorr in both (see module docstring).
    expected_seam = -fcorr

    # Vacuity guards: the correction must be resolvable against the field it
    # is added to, and it must not be trivially zero.
    scale = np.abs(f_off).max()
    assert np.abs(expected_seam).max() > 1e-3 * scale, (
        "seam flux mismatch is too small to test -- the fixture has become "
        "effectively periodic across the seam"
    )

    # Only the two seam cell planes may move at all.
    interior = delta[:, :, 1 : nk - 2, :]
    assert np.abs(interior).max() == 0.0, (
        "the cusp correction touched non-seam cells"
    )

    tol = 1e-5 * scale
    for kc, label in ((0, "low"), (nk - 2, "high")):
        got = delta[:, :, kc, :]
        ratio = np.median(got[np.abs(expected_seam) > tol]
                          / expected_seam[np.abs(expected_seam) > tol])
        np.testing.assert_allclose(
            got,
            expected_seam,
            rtol=0,
            atol=tol,
            err_msg=(
                f"{label}-k seam cell: correction applied with the wrong "
                f"magnitude or sign (median got/expected = {ratio:+.4f}; "
                "-1 means the sign flip of 2381658745 was never mirrored "
                "into the cusp block)"
            ),
        )
