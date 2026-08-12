"""Shared fixtures for the non-reflecting boundary condition tests.

Used by test_nonreflecting.py (the shared machinery), test_inlet.py
and test_outlet.py. Not a test module itself.
"""

import numpy as np

from ember import block_util, perturbation, util
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.patch import InletPatch, OutletPatch

# Reference scales chosen so nondimensional quantities are O(1); with the
# default unit references P_nd would be O(1e5) and a small perturbation would
# fall below float32 resolution.
FLUID = PerfectFluid(
    cp=1005.0,
    gamma=1.4,
    mu=1.8e-4,
    Pr=1.0,
    T_dtm=500.0,
    Rgas_ref=287.0,
    rho_ref=1.1,
    V_ref=100.0,
)
N_BLADE = 20
PITCH = 2.0 * np.pi / N_BLADE
P_MEAN = 1.0e5
T_MEAN = 300.0
VX_MEAN = 100.0
VT_MEAN = 60.0

# The patch class and the face it lives on, keyed by the kind of boundary. Both
# faces of the same block work: the flow runs along +x either way, so the
# interior is on the +x side of the i=0 face and the -x side of the i=-1 face.
PATCH_KINDS = {
    "inlet": (InletPatch, 0),
    "outlet": (OutletPatch, -1),
}


def pitch_coords(npitch, stretch):
    """Pitchwise node angles, optionally stretched away from uniform spacing."""
    u = np.linspace(0.0, 1.0, npitch)
    # Monotonic for |stretch| < 1 and fixes both end points, so the patch still
    # spans exactly one pitch with the periodic node repeated.
    return PITCH * (u + stretch * np.sin(2.0 * np.pi * u) / (2.0 * np.pi))


# Meridional origin the duct is built about, and its extent along and across
# the flow. R0 is large enough that r stays positive at every orientation.
X0, R0 = 0.0, 0.5
LEN_M, LEN_N = 0.1, 0.4


def make_block(
    ni=5,
    nspan=7,
    npitch=17,
    *,
    span_dim=1,
    Vx=VX_MEAN,
    Vr=0.0,
    Vt=VT_MEAN,
    P=P_MEAN,
    T=T_MEAN,
    stretch=0.0,
    chi=0.0,
    bow=0.0,
):
    """Build an annular duct at any meridional orientation.

    The duct is laid out in its own meridional frame -- ``m`` along the flow
    and ``n`` across it, both in the ``(x, r)`` plane -- and then turned through
    ``chi`` about the origin, which is what carries the end faces from constant
    :math:`x` to any surface of revolution. Its velocities are given and turned
    in the same frame, so a caller works in duct coordinates throughout and the
    orientation is the only thing that changes between one case and the next.

    Parameters
    ----------
    span_dim : int
        Which block axis is spanwise; the other of j and k is pitchwise.
    Vx, Vr, Vt : float
        Velocity along the duct, across it, and in the pitch direction [m/s].
        The first two are turned with the geometry, so ``Vx`` is the
        through-flow at any ``chi`` and ``Vr`` the cross-flow.
    stretch : float
        Pitchwise node stretching; see :func:`pitch_coords`.
    chi : float
        Angle of the duct axis from :math:`+x` [deg]. ``0`` gives the straight
        axial duct with constant-x end faces, ``90`` and ``270`` radial ducts
        flowing out and in, ``180`` an axial duct running backwards, and
        anything else a conical one.
    bow : float
        Curvature of the end faces, as a displacement along the duct axis
        applied to mid-span and tapering to zero at either end [m]. Nonzero
        makes the face normal turn from hub to tip, so the frame angle varies
        along the span instead of being one number for the face.
    """
    assert span_dim in (1, 2)
    nj, nk = (nspan, npitch) if span_dim == 1 else (npitch, nspan)
    shape = (ni, nj, nk)

    # Duct coordinates: m along the flow, n across it, bowed so that the
    # constant-m surfaces are curved rather than flat.
    m_vec = np.linspace(0.0, LEN_M, ni)
    n_vec = np.linspace(-0.5 * LEN_N, 0.5 * LEN_N, nspan)
    m = m_vec[:, None] + bow * np.sin(np.pi * np.linspace(0.0, 1.0, nspan))[None, :]
    n = np.broadcast_to(n_vec[None, :], m.shape)

    # Turned into the meridional plane. At chi = 0 this is the identity on the
    # duct coordinates, so the straight axial duct comes back unchanged.
    c, s = np.cos(np.radians(chi)), np.sin(np.radians(chi))
    x_mn = X0 + m * c - n * s
    r_mn = R0 + m * s + n * c

    t_vec = pitch_coords(npitch, stretch)
    if span_dim == 1:
        x = x_mn[:, :, None] * np.ones(shape)
        r = r_mn[:, :, None] * np.ones(shape)
        t = t_vec[None, None, :] * np.ones(shape)
    else:
        x = x_mn[:, None, :] * np.ones(shape)
        r = r_mn[:, None, :] * np.ones(shape)
        t = t_vec[None, :, None] * np.ones(shape)

    block = Block(shape=shape)
    block.set_fluid(FLUID)
    block.set_Nb(N_BLADE)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)
    block.set_P_T(P, T)
    block.set_Vx(Vx * np.ones(shape))
    block.set_Vr(Vr * np.ones(shape))
    block.set_Vt(Vt * np.ones(shape))
    # The velocities were given in duct coordinates, so turn them with it.
    if chi:
        _, rot_from = util.rotation_matrices(np.radians(chi))
        block_util.resolve_from_interface(block, rot_from)
    return block


def turn(Vm, Vn, chi):
    """Duct-frame meridional velocity pair as (Vx, Vr) in the machine frame."""
    c, s = np.cos(np.radians(chi)), np.sin(np.radians(chi))
    return Vm * c - Vn * s, Vm * s + Vn * c


def reference_state(Vx=VX_MEAN, Vr=0.0, Vt=VT_MEAN, P=P_MEAN, T=T_MEAN):
    """Scalar block holding an intended boundary state, for reading off targets."""
    block = Block(shape=())
    block.set_fluid(FLUID)
    block.set_x(np.array([0.0]))
    block.set_r(np.array([0.5]))
    block.set_t(np.array([0.0]))
    block.set_P_T(P, T)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)
    return block


def attached(kind="inlet", sigma=1.0, target=None, **kwargs):
    """Block with a non-reflecting patch of the given kind on the matching face.

    The prescribed boundary state defaults to whatever flow the block was built
    with, so the patch starts at its own fixed point; pass ``target`` as a dict
    of :func:`reference_state` arguments to prescribe something else and create
    a deliberate mismatch.
    """
    patch_type, i_face = PATCH_KINDS[kind]
    block = make_block(**kwargs)
    patch = patch_type(i=i_face, label=f"{kind}_nrbc")
    block.patches.append(patch)

    if target is None:
        target = {
            "Vx": kwargs.get("Vx", VX_MEAN),
            "Vr": kwargs.get("Vr", 0.0),
            "Vt": kwargs.get("Vt", VT_MEAN),
            "P": kwargs.get("P", P_MEAN),
            "T": kwargs.get("T", T_MEAN),
        }
    # The target is given in duct coordinates like the block's own velocities,
    # so turn it the same way. What the setters then see is a machine-frame
    # state, which is the frame the inlet angles are defined in.
    target = dict(target)
    target["Vx"], target["Vr"] = turn(
        target.get("Vx", VX_MEAN), target.get("Vr", 0.0), kwargs.get("chi", 0.0)
    )
    state = reference_state(**target)
    if kind == "inlet":
        patch.set_ho_s(float(state.ho), float(state.s))
        patch.set_Alpha(float(state.Alpha))
        patch.set_Beta(float(state.Beta))
    else:
        patch.set_P(float(state.P))
    patch.sigma = sigma
    return block, patch


def face_prim(patch):
    """Primitive vector on the patch face, nondimensional, in interface coordinates.

    Components 1 and 2 are the velocity along the frame axis and the one in the
    surface, which on a face of constant x are Vx and Vr. Read inside the
    patch's own rotation window, so it is the same state the condition works
    in and can be differenced against the reference the condition froze.
    """
    with patch._resolved():
        b = patch.block_view
        return np.stack((b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1)


def face_chic(patch):
    """Characteristic deviation of the face from the frozen reference state."""
    p2c = patch._span_bcast(perturbation.primitive_to_chic(patch.block_avg))
    return util.matvec(p2c, face_prim(patch) - patch._ref["prim"])


def seed_chic(patch, wave):
    """Write a face state whose characteristic deviation from the mean is wave.

    Written in interface coordinates, like :func:`face_prim` reads them, so the
    same ``wave`` deposits the same physical disturbance whatever the face
    orientation.
    """
    c2p = patch._span_bcast(perturbation.chic_to_primitive(patch.block_avg))
    prim = patch._ref["prim"] + util.matvec(c2p, wave)
    with patch._resolved():
        b = patch.block_view
        rho_nd, u_nd = b.fluid.set_P_rho(prim[..., 4], prim[..., 0])
        b.set_rho_u_Vxrt_nd(rho_nd, u_nd, prim[..., 1], prim[..., 2], prim[..., 3])


def harmonic(patch, field):
    """Pitchwise-varying part of a patch-shaped field."""
    return field - patch._pitch_mean(field)
