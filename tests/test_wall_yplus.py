"""``ember.block_util.wall_yplus`` against an independent numpy reference.

``wall_yplus_field`` (``_fortran/viscous.f90``) shares its Re/skin-friction
core (``wall_core``) with ``set_visc_force``'s own wall function (``wall_func``)
-- see that file's module docstring for the extraction -- so this test exists
to gate the NEW half of that split (``wall_yplus``/``wall_yplus_iface/jface/
kface``/``wall_yplus_field``), not to re-litigate ``wall_func`` itself, which
``tests/test_viscous_cusp_seam.py`` and the viscous golden tests already cover.

Method: an expression-for-expression numpy transcription of ``wall_core`` and
the ``iface``/``jface``/``kface`` corner-averaging helpers it is built on (same
spirit as ``test_viscous_cusp_seam.py``'s ``_kface_flow`` transcription),
compared against ``ember.block_util.wall_yplus(block)`` on all six wall faces
of a single rotating, sheared block whose i-faces are frictionless
(:class:`~ember.inviscid.InviscidPatch`) and whose j/k faces are left as the
default no-slip walls -- exercising both the "zero on non-wall cells" path
and the "nonzero, correctly-valued on wall cells" path in one fixture.

Being a transcription, this pins the formula by duplicating it: it cannot say
whether ``wall_core`` computes the right thing, only whether it still computes
the same thing. ``tests/test_wall_function_limits.py`` is the complement,
asserting the physical identities the formula has to satisfy. It is also the
only test that reaches the ``Re < RE_SMALL`` branch -- this fixture's
production ``MU`` puts every face at ``Re ~ 1e5-1e7``, so ``_wall_core_np``'s
``small`` arm below is never evaluated here.
"""

import numpy as np

import ember.block
import ember.block_util
from ember import util
from ember.fluid import PerfectFluid
from ember.inviscid import InviscidPatch

SHAPE = (7, 9, 9)
NB = 36
MU = 1.8e-5  # production magnitude, as in test_viscous_cusp_seam/phases_golden

# Curve-fit constants and Re threshold, transcribed from wall_core in
# viscous.f90 -- must match exactly, this is the thing under test.
RE_SMALL = 127.53373025
A1, A2, A3 = -1.767e-3, 3.177e-2, 2.5614e-1


def _build_block():
    """Rotating, swirling, sheared single block; frictionless i-faces, no-slip
    j/k-faces (the default -- no patch needed to make a face a solid wall).

    Geometry and flow field follow ``test_viscous_cusp_seam.py``'s fixture.
    Deliberately non-degenerate in every velocity component and in r, so
    every term in ``wall_core`` (V, d, Re, both cf branches if they occur,
    r for the swirl-work term) is actually exercised.
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

    block.patches.append(InviscidPatch(i=0))
    block.patches.append(InviscidPatch(i=-1))
    # j and k faces carry no patch -> default no-slip walls.
    return block


def _wall_core_np(r, dA0, dA1, dA2, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt):
    """numpy transcription of viscous_helpers' wall_core, through y+.

    float64 throughout (the block's own arrays are float32; cast up before
    the divide-heavy tail so this reference isn't itself precision-limited).
    """
    r = r.astype(np.float64)
    dA0, dA1, dA2 = (a.astype(np.float64) for a in (dA0, dA1, dA2))
    vol = vol.astype(np.float64)
    rho = rho.astype(np.float64)
    Vx, Vr_, Vt = (a.astype(np.float64) for a in (Vx, Vr, Vt))

    Vt_slip = Vt - (Omega_wall - Omega_block) * r
    V = np.sqrt(Vx**2 + Vr_**2 + Vt_slip**2 + 1e-9)
    dA_mag = np.sqrt(dA0**2 + dA1**2 + dA2**2)
    d = vol / dA_mag
    Re = rho * V * d / mu

    small = Re < RE_SMALL
    lnRe = np.log(np.where(small, 1.0, Re))  # dummy value where small: unused there
    cf_small = 2.0 / np.where(small, Re, 1.0)  # dummy where not small: unused there
    cf_big = A1 + A2 / lnRe + A3 / lnRe**2
    cf = np.where(small, cf_small, cf_big)

    return Re * np.sqrt(cf * 0.5)


def _wall_yplus_reference(block):
    """Build the six reference face arrays from the block's raw numpy state,
    following wall_func_iface/jface/kface's exact index conventions (see
    each call site in set_visc_force)."""
    r = np.asarray(block.r, dtype=np.float64)
    rho = np.asarray(block.rho, dtype=np.float64)
    Vx = np.asarray(block.Vx, dtype=np.float64)
    Vr = np.asarray(block.Vr, dtype=np.float64)
    Vt = np.asarray(block.Vt_rel, dtype=np.float64)
    vol = np.asarray(block.vol, dtype=np.float64)
    dAj = np.asarray(block.dAj, dtype=np.float64)  # (3, ni-1, nj, nk-1)
    dAk = np.asarray(block.dAk, dtype=np.float64)  # (3, ni-1, nj-1, nk)
    Omega_block = float(block.Omega)
    mu = MU

    def _mean4(a, b, c, d):
        return 0.25 * (a + b + c + d)

    # ---- k faces: kface(x,i,j,k) sums (i,j)/(i+1,j)/(i,j+1)/(i+1,j+1) at k ----
    def kface_ref(k_r, k_flow, vol_k):
        rf = _mean4(r[:-1, :-1, k_r], r[1:, :-1, k_r], r[:-1, 1:, k_r], r[1:, 1:, k_r])
        Vxf = _mean4(Vx[:-1, :-1, k_flow], Vx[1:, :-1, k_flow], Vx[:-1, 1:, k_flow], Vx[1:, 1:, k_flow])
        Vrf = _mean4(Vr[:-1, :-1, k_flow], Vr[1:, :-1, k_flow], Vr[:-1, 1:, k_flow], Vr[1:, 1:, k_flow])
        Vtf = _mean4(Vt[:-1, :-1, k_flow], Vt[1:, :-1, k_flow], Vt[:-1, 1:, k_flow], Vt[1:, 1:, k_flow])
        rhof = _mean4(rho[:-1, :-1, k_flow], rho[1:, :-1, k_flow], rho[:-1, 1:, k_flow], rho[1:, 1:, k_flow])
        vol_ = vol[:, :, vol_k]
        dA0, dA1, dA2 = dAk[0, :, :, k_r], dAk[1, :, :, k_r], dAk[2, :, :, k_r]
        omega_wall = Omega_block  # no RotatingPatch in this fixture
        return _wall_core_np(rf, dA0, dA1, dA2, vol_, Omega_block, omega_wall, mu, rhof, Vxf, Vrf, Vtf)

    yplus_k1 = kface_ref(k_r=0, k_flow=1, vol_k=0)
    yplus_nk = kface_ref(k_r=-1, k_flow=-2, vol_k=-1)

    # ---- j faces: jface(x,i,j,k) sums (i,j)/(i+1,j)/(i,j+1)/(i+1,j+1)-ish
    # at fixed j, varying i,k ----
    def jface_ref(j_r, j_flow, vol_j):
        rf = _mean4(r[:-1, j_r, :-1], r[1:, j_r, :-1], r[:-1, j_r, 1:], r[1:, j_r, 1:])
        Vxf = _mean4(Vx[:-1, j_flow, :-1], Vx[1:, j_flow, :-1], Vx[:-1, j_flow, 1:], Vx[1:, j_flow, 1:])
        Vrf = _mean4(Vr[:-1, j_flow, :-1], Vr[1:, j_flow, :-1], Vr[:-1, j_flow, 1:], Vr[1:, j_flow, 1:])
        Vtf = _mean4(Vt[:-1, j_flow, :-1], Vt[1:, j_flow, :-1], Vt[:-1, j_flow, 1:], Vt[1:, j_flow, 1:])
        rhof = _mean4(rho[:-1, j_flow, :-1], rho[1:, j_flow, :-1], rho[:-1, j_flow, 1:], rho[1:, j_flow, 1:])
        vol_ = vol[:, vol_j, :]
        dA0, dA1, dA2 = dAj[0, :, j_r, :], dAj[1, :, j_r, :], dAj[2, :, j_r, :]
        omega_wall = Omega_block
        return _wall_core_np(rf, dA0, dA1, dA2, vol_, Omega_block, omega_wall, mu, rhof, Vxf, Vrf, Vtf)

    yplus_j1 = jface_ref(j_r=0, j_flow=1, vol_j=0)
    yplus_nj = jface_ref(j_r=-1, j_flow=-2, vol_j=-1)

    # i faces are frictionless (InviscidPatch) -> expect exact zero, not
    # compared against a reference formula.
    ni, nj, nk = block.shape
    yplus_i1 = np.zeros((nj - 1, nk - 1))
    yplus_ni = np.zeros((nj - 1, nk - 1))

    return {
        "yplus_i1": yplus_i1,
        "yplus_j1": yplus_j1,
        "yplus_k1": yplus_k1,
        "yplus_ni": yplus_ni,
        "yplus_nj": yplus_nj,
        "yplus_nk": yplus_nk,
    }


def test_shapes_match_ijk_wall_visc():
    """Each face array's shape matches its own face plane, i.e. the
    corresponding ``ijk_wall_visc`` array's shape with its broadcastable
    leading/middle singleton axis (added there for splatting into the
    Fortran call, see ``Block.ijk_wall_visc``'s docstring) squeezed out --
    ``wall_yplus_field``'s outputs are plain rank-2 Fortran arrays."""
    block = _build_block()
    got = ember.block_util.wall_yplus(block)
    wall = block.ijk_wall_visc
    assert got["yplus_i1"].shape == np.squeeze(np.asarray(wall["walli1"])).shape
    assert got["yplus_ni"].shape == np.squeeze(np.asarray(wall["wallni"])).shape
    assert got["yplus_j1"].shape == np.squeeze(np.asarray(wall["wallj1"])).shape
    assert got["yplus_nj"].shape == np.squeeze(np.asarray(wall["wallnj"])).shape
    assert got["yplus_k1"].shape == np.squeeze(np.asarray(wall["wallk1"])).shape
    assert got["yplus_nk"].shape == np.squeeze(np.asarray(wall["wallnk"])).shape


def test_zero_on_frictionless_i_faces():
    """The i-faces carry InviscidPatch (frictionless): walli1/wallni == 1.0
    (free), so wall_yplus_field must leave those two arrays at their initial
    zero fill, bitwise -- exercising the "non-wall cells stay zero" path."""
    block = _build_block()
    wall = block.ijk_wall_visc
    assert np.all(np.asarray(wall["walli1"]) == 1.0)
    assert np.all(np.asarray(wall["wallni"]) == 1.0)

    got = ember.block_util.wall_yplus(block)
    assert np.all(got["yplus_i1"] == 0.0)
    assert np.all(got["yplus_ni"] == 0.0)


def test_nonzero_on_default_noslip_jk_faces():
    """j/k faces carry no patch -> default no-slip walls (wallj1/wallk1 ==
    0.0), so wall_yplus_field must actually compute y+ there, not leave the
    zero fill -- exercising the "wall cells get a real value" path before
    the numeric-agreement check below."""
    block = _build_block()
    wall = block.ijk_wall_visc
    assert np.all(np.asarray(wall["wallj1"]) == 0.0)
    assert np.all(np.asarray(wall["wallk1"]) == 0.0)

    got = ember.block_util.wall_yplus(block)
    assert np.all(got["yplus_j1"] > 0.0)
    assert np.all(got["yplus_k1"] > 0.0)


def test_matches_reference_on_wall_faces():
    block = _build_block()
    got = ember.block_util.wall_yplus(block)
    ref = _wall_yplus_reference(block)

    # Vacuity guard: the reference must not be trivially zero/degenerate.
    for key in ("yplus_j1", "yplus_nj", "yplus_k1", "yplus_nk"):
        assert np.all(ref[key] > 0.0), f"{key} reference is non-positive"

    for key, ref_val in ref.items():
        got_val = np.asarray(got[key], dtype=np.float64)
        np.testing.assert_allclose(
            got_val,
            ref_val,
            rtol=1e-4,
            atol=1e-6,
            err_msg=f"{key}: wall_yplus_field disagrees with the numpy reference",
        )
