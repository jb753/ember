"""Tests for OutletPatch.

Module tested: ember.outlet.OutletPatch
"""

import numpy as np
import pytest
from ember.block import Block
from ember.fluid import PerfectFluid
from ember.outlet import OutletPatch


P0 = 101325.0
P1 = 95000.0
_FLUID = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
_Nb = 20
_PITCH = 2.0 * np.pi / _Nb


def _make_block(ni, nj, nk, *, span_dim, Vt):
    """Build an i-const outlet block with radial span along span_dim.

    span_dim=1: j varies in r, k is pitchwise
    span_dim=2: k varies in r, j is pitchwise
    """
    assert span_dim in (1, 2)
    nspan = nj if span_dim == 1 else nk
    npitch = nk if span_dim == 1 else nj
    r_vec = np.linspace(0.3, 0.7, nspan)
    t_vec = np.linspace(0.0, _PITCH, npitch)

    x = np.linspace(0.0, 0.1, ni)[:, None, None] * np.ones((ni, nj, nk))
    if span_dim == 1:
        r = r_vec[None, :, None] * np.ones((ni, nj, nk))
        t = t_vec[None, None, :] * np.ones((ni, nj, nk))
    else:
        r = r_vec[None, None, :] * np.ones((ni, nj, nk))
        t = t_vec[None, :, None] * np.ones((ni, nj, nk))

    block = Block(shape=(ni, nj, nk))
    block.set_fluid(_FLUID)
    block.set_Nb(_Nb)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)
    block.set_P_T(P0, 300.0)
    block.set_Vx(50.0 * np.ones((ni, nj, nk)))
    block.set_Vr(np.zeros((ni, nj, nk)))
    block.set_Vt(Vt * np.ones((ni, nj, nk)))
    return block


@pytest.fixture(params=["span_j", "span_k"])
def swirl_block(request):
    """i-const outlet block with swirl; parametrized over spanwise orientation."""
    span_dim = 1 if request.param == "span_j" else 2
    return _make_block(3, 8, 12, span_dim=span_dim, Vt=50.0)


@pytest.fixture
def attached_patch(swirl_block):
    """OutletPatch with P0 set and attached to swirl_block."""
    patch = OutletPatch(i=-1, j=(0, -1), k=(0, -1))
    patch.set_P(P0)
    patch.attach_to_block(swirl_block)
    return patch


# ---------------------------------------------------------------------------
# _span_bcast initialisation (orientation-independent, tested once with span_j)
# ---------------------------------------------------------------------------


def test_set_P_before_attach():
    """set_P then attach_to_block: _span_bcast has correct shape and is uniform P0."""
    block = _make_block(2, 8, 12, span_dim=1, Vt=50.0)
    patch = OutletPatch(i=-1, j=(0, -1), k=(0, -1))
    patch.set_P(P0)
    patch.attach_to_block(block)

    assert (
        np.broadcast_to(patch._span_bcast(patch._P_raw), patch.block_view.shape).shape
        == patch.block_view.P.shape
    )
    assert (
        np.broadcast_to(patch._span_bcast(patch._P_raw), patch.block_view.shape).dtype
        == np.float32
    )
    assert np.all(
        np.isclose(
            np.broadcast_to(patch._span_bcast(patch._P_raw), patch.block_view.shape),
            P0,
            rtol=1e-5,
        )
    )


def test_set_P_after_attach():
    """attach_to_block then set_P: _span_bcast has correct shape and is uniform P0."""
    block = _make_block(2, 8, 12, span_dim=1, Vt=50.0)
    patch = OutletPatch(i=-1, j=(0, -1), k=(0, -1))
    patch.attach_to_block(block)
    patch.set_P(P0)

    assert (
        np.broadcast_to(patch._span_bcast(patch._P_raw), patch.block_view.shape).shape
        == patch.block_view.P.shape
    )
    assert (
        np.broadcast_to(patch._span_bcast(patch._P_raw), patch.block_view.shape).dtype
        == np.float32
    )
    assert np.all(
        np.isclose(
            np.broadcast_to(patch._span_bcast(patch._P_raw), patch.block_view.shape),
            P0,
            rtol=1e-5,
        )
    )


# ---------------------------------------------------------------------------
# apply (parametrized via attached_patch)
# ---------------------------------------------------------------------------


def test_apply_no_shape_error(attached_patch):
    """apply() runs without error."""
    attached_patch.apply()


# ---------------------------------------------------------------------------
# set_P reset (parametrized via attached_patch)
# ---------------------------------------------------------------------------


def test_set_P_resets_bcast(attached_patch):
    """set_P resets _span_bcast to a uniform new value."""
    attached_patch.set_P(P1)

    assert (
        np.broadcast_to(
            attached_patch._span_bcast(attached_patch._P_raw),
            attached_patch.block_view.shape,
        ).shape
        == attached_patch.block_view.P.shape
    )
    assert np.all(
        np.isclose(
            np.broadcast_to(
                attached_patch._span_bcast(attached_patch._P_raw),
                attached_patch.block_view.shape,
            ),
            P1,
            rtol=1e-5,
        )
    )


def test_apply_default_no_adjustment(attached_patch):
    """With default empty adjustment, apply() does not change _span_bcast."""
    P_before = np.broadcast_to(
        attached_patch._span_bcast(attached_patch._P_raw),
        attached_patch.block_view.shape,
    ).copy()
    attached_patch.apply()
    assert np.array_equal(
        np.broadcast_to(
            attached_patch._span_bcast(attached_patch._P_raw),
            attached_patch.block_view.shape,
        ),
        P_before,
    )


# ---------------------------------------------------------------------------
# set_adjustment
# ---------------------------------------------------------------------------


def test_set_adjustment_stores_values():
    """set_adjustment stores K_dyn, radial_equilibrium and rf."""
    patch = OutletPatch(i=-1, j=(0, -1), k=(0, -1))
    patch.set_P(P0)
    patch.set_adjustment(K_dyn=0.5, radial_equilibrium=False, rf=0.2)
    assert patch._adjustment == {"K_dyn": 0.5, "radial_equilibrium": False, "rf": 0.2}


def test_set_adjustment_incompatible_with_nonscalar_P():
    """set_adjustment raises ValueError when P is non-scalar (span-varying)."""
    block = _make_block(3, 5, 4, span_dim=1, Vt=0.0)
    patch = OutletPatch(i=-1, j=(0, -1), k=(0, -1))
    patch.attach_to_block(block)
    # span-varying P: shape (1, 5, 1) broadcasts over patch shape (1, 5, 4)
    patch.set_P(np.full((1, 5, 1), P0))
    with pytest.raises(ValueError, match="incompatible"):
        patch.set_adjustment()


# ---------------------------------------------------------------------------
# set_backflow
# ---------------------------------------------------------------------------


def test_set_backflow_stores_nondim_snapshot(swirl_block):
    """set_backflow stores a nondimensionalised 4-element snapshot."""
    patch = OutletPatch(i=-1, j=(0, -1), k=(0, -1))
    patch.set_P(P0)
    patch.attach_to_block(swirl_block)
    fluid = swirl_block.fluid
    ho, s, Vr, Vt = 400000.0, 1500.0, 5.0, 10.0
    patch.set_backflow(ho, s, Vr, Vt)
    snap = patch._inout_snapshot
    assert snap.shape == (4,)
    assert np.isclose(snap[0], ho / fluid.u_ref)
    assert np.isclose(snap[1], s / fluid.Rgas_ref)
    assert np.isclose(snap[2], Vr / fluid.V_ref)
    assert np.isclose(snap[3], Vt / fluid.V_ref)
    assert patch._backflow_enabled


def test_set_backflow_rejects_non_scalar(swirl_block):
    """set_backflow raises TypeError if any argument is not a scalar."""
    patch = OutletPatch(i=-1, j=(0, -1), k=(0, -1))
    patch.set_P(P0)
    patch.attach_to_block(swirl_block)
    with pytest.raises(TypeError, match="scalar"):
        patch.set_backflow(np.array([400000.0]), 1500.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# smooth_pitch_121: periodic 1-2-1 entropy smoother
# ---------------------------------------------------------------------------


def test_smooth_pitch_121_preserves_constant(attached_patch):
    """A pitch-uniform field passes through unchanged."""
    field = np.ones(attached_patch.block_view.shape, dtype=np.float32) * 2.5
    out = attached_patch.smooth_pitch_121(field, alpha=1.0)
    np.testing.assert_allclose(out, field)


def test_smooth_pitch_121_kills_sawtooth(attached_patch):
    """alpha=1 annihilates the pitch Nyquist (sawtooth) mode."""
    shape = attached_patch.block_view.shape
    pd = attached_patch.pitch_dim
    npitch = shape[pd]
    assert npitch % 2 == 0, "test needs even pitch count for a clean sawtooth"
    # Build [-1, 1, -1, 1, ...] along pitch_dim, broadcast over the rest
    saw_1d = np.where(np.arange(npitch) % 2 == 0, -1.0, 1.0).astype(np.float32)
    bshape = [1, 1, 1]
    bshape[pd] = npitch
    field = np.broadcast_to(saw_1d.reshape(bshape), shape).astype(np.float32)
    out = attached_patch.smooth_pitch_121(field, alpha=1.0)
    np.testing.assert_allclose(out, 0.0, atol=1e-7)


def test_smooth_pitch_121_alpha_blend(attached_patch):
    """alpha in (0, 1) blends raw field with the smoothed one."""
    rng = np.random.default_rng(0)
    field = rng.standard_normal(attached_patch.block_view.shape).astype(np.float32)
    alpha = np.float32(0.95)
    full = attached_patch.smooth_pitch_121(field, alpha=1.0)
    blended = attached_patch.smooth_pitch_121(field, alpha=alpha)
    np.testing.assert_allclose(blended, alpha * full + (1.0 - alpha) * field, rtol=1e-6)


def test_smooth_pitch_121_alpha_zero_is_noop(attached_patch):
    """alpha=0 leaves the field unchanged."""
    rng = np.random.default_rng(1)
    field = rng.standard_normal(attached_patch.block_view.shape).astype(np.float32)
    out = attached_patch.smooth_pitch_121(field, alpha=0.0)
    assert out is field
