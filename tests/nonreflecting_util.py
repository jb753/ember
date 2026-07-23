"""Shared fixtures for the non-reflecting boundary condition tests.

Used by test_nonreflecting.py (the shared machinery), test_inlet_nonreflecting.py
and test_outlet_nonreflecting.py. Not a test module itself.
"""

import numpy as np

from ember import perturbation, util
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.patch import NonReflectingInletPatch, NonReflectingOutletPatch

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
    "inlet": (NonReflectingInletPatch, 0),
    "outlet": (NonReflectingOutletPatch, -1),
}


def pitch_coords(npitch, stretch):
    """Pitchwise node angles, optionally stretched away from uniform spacing."""
    u = np.linspace(0.0, 1.0, npitch)
    # Monotonic for |stretch| < 1 and fixes both end points, so the patch still
    # spans exactly one pitch with the periodic node repeated.
    return PITCH * (u + stretch * np.sin(2.0 * np.pi * u) / (2.0 * np.pi))


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
):
    """Build a block with constant-x end faces; span_dim selects the spanwise axis."""
    assert span_dim in (1, 2)
    nj, nk = (nspan, npitch) if span_dim == 1 else (npitch, nspan)
    shape = (ni, nj, nk)

    r_vec = np.linspace(0.3, 0.7, nspan)
    t_vec = pitch_coords(npitch, stretch)

    x = np.linspace(0.0, 0.1, ni)[:, None, None] * np.ones(shape)
    if span_dim == 1:
        r = r_vec[None, :, None] * np.ones(shape)
        t = t_vec[None, None, :] * np.ones(shape)
    else:
        r = r_vec[None, None, :] * np.ones(shape)
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
    return block


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
    """Primitive vector [rho, Vx, Vr, Vt, P] on the patch face, nondimensional."""
    b = patch.block_view
    return np.stack((b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1)


def face_chic(patch):
    """Characteristic deviation of the face from the frozen reference state."""
    p2c = patch._span_bcast(perturbation.primitive_to_chic(patch.block_avg))
    return util.matvec(p2c, face_prim(patch) - patch._ref["prim"])


def seed_chic(patch, wave):
    """Write a face state whose characteristic deviation from the mean is wave."""
    c2p = patch._span_bcast(perturbation.chic_to_primitive(patch.block_avg))
    prim = patch._ref["prim"] + util.matvec(c2p, wave)
    b = patch.block_view
    rho_nd, u_nd = b.fluid.set_P_rho(prim[..., 4], prim[..., 0])
    b.set_rho_u_Vxrt_nd(rho_nd, u_nd, prim[..., 1], prim[..., 2], prim[..., 3])


def harmonic(patch, field):
    """Pitchwise-varying part of a patch-shaped field."""
    return field - patch._pitch_mean(field)
