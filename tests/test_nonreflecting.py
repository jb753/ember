"""Tests for the machinery shared by both non-reflecting boundary conditions.

Module tested: ember.nonreflecting.NonReflectingPatch

Every test that exercises behaviour a subclass can influence -- the geometry
checks, the mean-state guards, the target bookkeeping, the relaxation -- is
parametrised over both the inflow and the outflow condition. The Hilbert
transform numerics are run once: the matrix is built from the block geometry
alone by base-class code that no subclass touches, and
:func:`test_attach_detects_axes` already covers both patches building it.

Test cases:
- Attachment: axis detection, constant-x plane, interior side, whole pitch
- Hilbert transform: annihilation of the mean, modal response, structural
  properties, agreement with an independent DFT, non-uniform pitch spacing,
  bounded norm
- Shared behaviour: characteristic round trip, relaxation linearity, copy
  semantics, unset targets, collection membership, class member order
- Guards: backflow, axially supersonic, absolutely supersonic
"""

import importlib
import inspect

import numpy as np
import pytest

from ember import perturbation, util
from ember.patch import PERMEABLE_TYPES
from nonreflecting_util import (
    PATCH_KINDS,
    PITCH,
    P_MEAN,
    VT_MEAN,
    VX_MEAN,
    attached,
    face_prim,
    make_block,
    pitch_coords,
)

# Target that mismatches the flow the block is built with, per kind, for the
# tests that need the patch to actually do something.
MISMATCH = {"inlet": {"Vt": VT_MEAN + 30.0}, "outlet": {"P": P_MEAN * 1.02}}


@pytest.fixture(params=list(PATCH_KINDS))
def kind(request):
    """The kind of non-reflecting boundary, "inlet" or "outlet"."""
    return request.param


def _bare(kind, **kwargs):
    """Block and an unattached patch of the given kind, for attachment tests."""
    patch_type, i_face = PATCH_KINDS[kind]
    return make_block(**kwargs), patch_type, i_face


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("span_dim", [1, 2])
def test_attach_detects_axes(kind, span_dim):
    """Span and pitch axes are detected and the Hilbert matrix is square."""
    _, patch = attached(kind, span_dim=span_dim)
    assert patch.span_dim != patch.pitch_dim
    assert patch.const_dim == 0
    npitch = patch.shape[patch.pitch_dim]
    assert patch._hilbert.shape == (npitch, npitch)


def test_attach_rejects_canted_plane(kind):
    """A boundary plane that is not constant-x is refused."""
    block, patch_type, i_face = _bare(kind)
    # Skew the face in x so it is no longer a plane of constant x.
    x = block.x.copy()
    x[i_face] += 0.01 * np.linspace(0.0, 1.0, x.shape[1])[:, None]
    block.set_x(x)
    with pytest.raises(ValueError, match="constant x"):
        block.patches.append(patch_type(i=i_face))


def test_attach_rejects_interior_on_wrong_side(kind):
    """The interior must lie on the side the flow direction implies."""
    block, patch_type, i_face = _bare(kind)
    # The other end face of the same block has its interior on the wrong side.
    wrong = -1 if i_face == 0 else 0
    side = r"\+x" if kind == "inlet" else "-x"
    with pytest.raises(NotImplementedError, match=side):
        block.patches.append(patch_type(i=wrong))


def test_attach_rejects_partial_pitch(kind):
    """A patch covering only part of the pitch cannot define the transform."""
    block, patch_type, i_face = _bare(kind, npitch=17)
    with pytest.raises(ValueError, match="whole pitch"):
        block.patches.append(patch_type(i=i_face, k=(0, 8)))


# ---------------------------------------------------------------------------
# Hilbert transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stretch", [0.0, 0.4])
def test_hilbert_annihilates_constant(stretch):
    """The transform kills the pitch mean, so it cannot corrupt the mean mode."""
    _, patch = attached(stretch=stretch)
    npitch = patch.shape[patch.pitch_dim]
    assert np.abs(patch._hilbert @ np.ones(npitch)).max() < 1e-5


def test_hilbert_modal_response():
    """H[cos] = +sin and H[sin] = -cos for every resolved mode.

    This pins the sign convention against Giles Eq. 5.16/5.19; with the sign
    reversed both boundary conditions would amplify rather than absorb, and no
    other test would catch it.
    """
    _, patch = attached(npitch=17)
    theta = pitch_coords(17, 0.0)
    m_max = (17 - 1 - 1) // 2
    for m in range(1, m_max + 1):
        phase = 2.0 * np.pi * m * theta / PITCH
        assert np.abs(patch._hilbert @ np.cos(phase) - np.sin(phase)).max() < 1e-5
        assert np.abs(patch._hilbert @ np.sin(phase) + np.cos(phase)).max() < 1e-5


def test_hilbert_structure():
    """Uniform spacing gives a real antisymmetric matrix with wrapped end rows."""
    _, patch = attached(npitch=17)
    hilbert = patch._hilbert
    assert hilbert.dtype == np.float32
    # Antisymmetric up to the node weights, which are uniform except for the
    # half weights on the repeated periodic node.
    interior = hilbert[1:-1, 1:-1]
    assert np.abs(interior + interior.T).max() < 1e-5
    # The repeated periodic node must transform identically at both ends.
    assert np.abs(hilbert[0] - hilbert[-1]).max() < 1e-5


def test_hilbert_matches_dft():
    """The matrix reproduces an independent DFT implementation of Giles Eq. 5.17.

    Two independent implementations of the same equation: the collapse to a
    Hilbert transform is only valid if it agrees with the transform pair
    written out directly.
    """
    _, patch = attached(npitch=17)
    npitch = 17
    n_dist = npitch - 1
    m_max = (n_dist - 1) // 2

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(npitch)
    signal[-1] = signal[0]  # periodic node repeated, as on the patch
    distinct = signal[:-1]

    # Giles Eq. 5.16 and 5.19-5.20 with sign(m) weighting, written directly.
    # His analysis sum carries exp(+2i pi jk/N), which is numpy's ifft, not fft.
    coeff = np.fft.ifft(distinct)
    expect = np.zeros(n_dist)
    for m in range(1, m_max + 1):
        phase = np.exp(-2j * np.pi * m * np.arange(n_dist) / n_dist)
        expect += 2.0 * np.real(1j * coeff[m] * phase)

    got = (patch._hilbert @ signal)[:-1]
    assert np.abs(got - expect).max() < 1e-4 * np.abs(expect).max()


@pytest.mark.parametrize("stretch", [0.2, 0.4])
def test_hilbert_nonuniform_resolves_low_modes(stretch):
    """Stretched pitch spacing stays accurate for the resolved harmonics.

    "Resolved" means at least four nodes per wavelength at the *coarsest*
    local spacing; above that the non-uniform quadrature degrades, which is a
    documented limitation rather than a bug.
    """
    npitch = 33
    _, patch = attached(npitch=npitch, stretch=stretch)
    theta = pitch_coords(npitch, stretch)
    m_resolved = int(PITCH / (4.0 * np.diff(theta).max()))
    assert m_resolved >= 3, "test mesh is too coarse to be meaningful"
    for m in range(1, m_resolved + 1):
        phase = 2.0 * np.pi * m * theta / PITCH
        err = np.abs(patch._hilbert @ np.cos(phase) - np.sin(phase)).max()
        assert err < 0.05, f"mode {m} error {err}"


@pytest.mark.parametrize("stretch", [0.0, 0.4])
@pytest.mark.parametrize("npitch", [17, 33])
def test_hilbert_norm_is_bounded(stretch, npitch):
    """The transform cannot amplify, whatever the pitchwise spacing.

    The discrete Hilbert transform grows only logarithmically with the node
    count, so an unresolved harmonic can leave the boundary slightly
    reflecting but never unstable.
    """
    _, patch = attached(npitch=npitch, stretch=stretch)
    assert np.abs(patch._hilbert).sum(axis=1).max() < 4.0


# ---------------------------------------------------------------------------
# Shared behaviour
# ---------------------------------------------------------------------------


def test_chic_round_trip(kind):
    """The frozen characteristic transform pair is an exact inverse."""
    _, patch = attached(kind)
    patch.update_soln()
    avg = patch.block_avg
    p2c = perturbation.primitive_to_chic(avg)
    c2p = perturbation.chic_to_primitive(avg)
    identity = util.matmat(c2p, p2c)
    expect = np.broadcast_to(np.eye(5, dtype=np.float32), identity.shape)
    assert np.abs(identity - expect).max() < 1e-5


def test_sigma_scales_the_correction_linearly(kind):
    """Halving sigma halves the applied change.

    Measured in primitives, which is where the relaxation is applied; the
    conserved variables are a nonlinear function of these so their change
    carries a second-order term.
    """
    changes = {}
    for sigma in (1.0, 0.5):
        _, patch = attached(kind, sigma=sigma, target=MISMATCH[kind])
        patch.update_soln()
        before = face_prim(patch)
        patch.apply()
        changes[sigma] = face_prim(patch) - before
    ratio = np.abs(changes[0.5]).max() / np.abs(changes[1.0]).max()
    assert ratio == pytest.approx(0.5, abs=1e-3)


def test_copy_preserves_targets_and_drops_caches(kind):
    """copy() carries the boundary condition but not geometry-derived caches."""
    _, patch = attached(kind, sigma=0.25)
    patch.update_soln()
    clone = patch.copy()
    np.testing.assert_array_equal(clone._target, patch._target)
    np.testing.assert_array_equal(clone._target_set, patch._target_set)
    assert clone._target is not patch._target
    assert clone.sigma == patch.sigma
    assert clone._hilbert is None
    assert clone._ref is None
    assert clone._prim_prev is None


def test_apply_without_targets_raises(kind):
    """Applying before the boundary state is set is an error, not a silent NaN."""
    block, patch_type, i_face = _bare(kind)
    patch = patch_type(i=i_face)
    block.patches.append(patch)
    with pytest.raises(ValueError, match="missing boundary condition values"):
        patch.apply()


def test_collection_and_permeable(kind):
    """The patch is discoverable and counts as a permeable, non-wall face."""
    block, patch = attached(kind)
    collection = getattr(block.patches, f"{kind}_nonreflecting")
    assert collection == [patch]
    # Its own collection, not the reflecting one, whose consumers poke
    # attributes private to those patches.
    assert getattr(block.patches, kind) == []
    assert isinstance(patch, PERMEABLE_TYPES)


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("ember.nonreflecting", "NonReflectingPatch"),
        ("ember.inlet_nonreflecting", "NonReflectingInletPatch"),
        ("ember.outlet_nonreflecting", "NonReflectingOutletPatch"),
        ("ember.mixing_nonreflecting", "NonReflectingMixingPatch"),
    ],
)
def test_class_member_order(module_name, class_name):
    """Class members follow the repository ordering convention."""
    from conftest import assert_class_member_order

    module = importlib.import_module(module_name)
    assert_class_member_order(inspect.getsource(module), class_name)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("Vx", [VX_MEAN, -10.0])
def test_split_follows_the_flow_direction(kind, Vx):
    """The four splits of the table in ember.nonreflecting, checked one by one.

    A characteristic is outgoing when its wave speed carries it out of the
    domain. So the acoustic split is fixed by the geometry -- c_up outgoing when
    the interior is on the +x side, c_down when it is on the -x side -- and the
    three convective characteristics follow the flow. This is the load-bearing
    test for the whole family: get a column wrong here and a boundary condition
    silently overwrites information the interior owns, or keeps stale
    information the interior should have replaced.
    """
    _, patch = attached(kind, Vx=Vx, target={})
    patch.update_soln()

    entering = Vx * patch._sign_interior > 0.0
    assert patch._entering.all() == entering
    acoustic_out = 0 if patch._sign_interior > 0 else 1
    expect = np.zeros(5, dtype=bool)
    expect[acoustic_out] = True
    if not entering:
        expect[2:] = True
    mask = np.broadcast_to(patch._mask_out, patch.shape + (5,))
    np.testing.assert_array_equal(mask[0, 0, 0], expect)


def test_reversed_mean_does_not_raise(kind):
    """Reversal is the other row of the table, not an error."""
    _, patch = attached(kind, Vx=-10.0, target={})
    patch.update_soln()
    patch.apply()
    assert np.isfinite(face_prim(patch)).all()


def test_axially_supersonic_raises(kind):
    """An axially supersonic mean state is not implemented."""
    _, patch = attached(kind, Vx=500.0, Vt=0.0, target={})
    with pytest.raises(NotImplementedError, match="axially"):
        patch.update_soln()


def test_absolutely_supersonic_raises(kind):
    """A supersonic but axially subsonic mean state is not implemented."""
    _, patch = attached(kind, Vx=100.0, Vt=400.0, target={})
    with pytest.raises(NotImplementedError, match="supersonic mean state"):
        patch.update_soln()
