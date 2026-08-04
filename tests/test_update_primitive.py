"""Eager fused evaluation of the primitive cache, ``Block.update_primitive``.

The six primitives ``_Vxrt_nd_uninit``, ``_halfVsq_nd_uninit``,
``_u_nd_uninit``, ``P_nd``, ``ho_nd`` and ``T_nd`` are normally pulled lazily
by ``@cached_array``, one numpy pass at a time. ``update_primitive`` fills all
six in two fused passes -- a Fortran kinematic kernel and the fluid's batched
``get_P_h_T`` -- and publishes them into the same ``_store`` entries with the
same data-key versions.

Two properties matter and are pinned here:

  * it agrees with the lazy path (not bitwise -- one is fused ``-Ofast``
    Fortran, the other op-by-op numpy -- so agreement is quantified); and
  * afterwards the lazy properties are cache HITS, because the whole point is
    to turn a recompute into a lookup. A silent fallback would still be
    correct, just slow, so nothing else in the suite would catch it.
"""

import numpy as np

import ember.block
import ember.grid
from ember import util
from ember.fluid import PerfectFluid

SHAPE = (9, 7, 5)

# THE TEST STATE MUST SIT AWAY FROM THE DATUM. u = e - halfVsq, and
# PerfectFluid's datum defaults to T_dtm = 300 K, so a block at T = 300 has
# u = 0 identically: the subtraction is then pure cancellation and every bit of
# u is noise, with the two paths disagreeing by 100% of nothing. T = 600 K puts
# u three orders of magnitude above the kinetic head, where the comparison
# means something. (Same trap as the bench harness's swirl(): ask what the test
# state actually exercises before trusting a gate.)
TOL_ULP = dict.fromkeys(
    ("P_nd", "ho_nd", "T_nd", "_Vxrt_nd_uninit", "_halfVsq_nd_uninit",
     "_u_nd_uninit"),
    8.0,
)

PRIMITIVES = tuple(TOL_ULP)


def _build_block():
    """Single block with swirl, so no velocity component is identically zero."""
    block = ember.block.Block(shape=SHAPE)
    block.set_Nb(36)
    xrt = util.linmesh3((0.0, 0.1), (0.5, 1.0), (0.0, 0.2), SHAPE)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
    block.set_P_T(101325.0, 600.0)
    block.set_wdist(np.zeros_like(block.r))
    Vx = 100.0 + 20.0 * np.sin(2.0 * np.pi * block.r)
    block.set_Vx(Vx.astype(np.float32))
    # Non-zero Vr and Vt: with either identically zero, halfVsq would not
    # exercise every term and the kinematic kernel's Vt = rhorVt/rho/r path
    # would be untested.
    block.set_Vr((0.05 * Vx).astype(np.float32))
    block.set_Vt((0.10 * Vx).astype(np.float32))
    return block


def _invalidate(block):
    """Bump the tracked data versions, as writing conserved_nd would."""
    for key in ("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe"):
        block._versions[key] += 1


def _ulps(got, ref):
    """Deviation in ulps of the field scale (not pointwise)."""
    scale = float(np.abs(ref).max())
    if scale == 0.0:
        return 0.0
    return float(np.abs(got - ref).max() / np.spacing(np.float32(scale)))


def test_matches_the_lazy_path():
    """The fused kernels agree with the op-by-op numpy properties."""
    block = _build_block()
    ref = {n: np.array(getattr(block, n), copy=True) for n in PRIMITIVES}

    _invalidate(block)
    block.update_primitive()

    for name in PRIMITIVES:
        got = np.asarray(getattr(block, name))
        assert got.shape == ref[name].shape
        assert np.all(np.isfinite(got)), name
        assert _ulps(got, ref[name]) <= TOL_ULP[name], (
            f"{name}: {_ulps(got, ref[name]):.1f} ulp of scale "
            f"exceeds {TOL_ULP[name]}"
        )


def test_leaves_the_properties_cached():
    """After update_primitive, every primitive is a cache hit, not a recompute.

    Checked by version stamp rather than by timing: the stored version tuple
    must already equal the one the property's own decorator would compute, and
    the returned array must be the very buffer update_primitive published.
    """
    block = _build_block()
    _invalidate(block)
    block.update_primitive()

    for name in PRIMITIVES:
        assert name in block._store, f"{name} was not published"
        stored_versions, stored_arr = block._store[name]
        # Reading the property must hand back the identical object -- a
        # recompute would allocate into it afresh and, more to the point, would
        # mean the version stamp did not match.
        assert getattr(block, name) is stored_arr, f"{name} recomputed"
        assert not stored_arr.flags.writeable, f"{name} left writeable"
        assert stored_versions == block._store[name][0]


def test_is_idempotent_and_returns_early():
    """A second call is a no-op: same buffers, same contents, same versions."""
    block = _build_block()
    _invalidate(block)
    block.update_primitive()
    first = {n: block._store[n] for n in PRIMITIVES}
    before = {n: np.array(block._store[n][1], copy=True) for n in PRIMITIVES}

    block.update_primitive()

    for name in PRIMITIVES:
        versions, arr = block._store[name]
        assert versions == first[name][0], f"{name} version changed"
        assert arr is first[name][1], f"{name} buffer replaced"
        np.testing.assert_array_equal(arr, before[name])


def test_invalidation_still_works():
    """Mutating conserved state makes the properties recompute, as before."""
    block = _build_block()
    block.update_primitive()
    P_before = np.array(block.P_nd, copy=True)

    cons = block.conserved_nd
    cons.flags.writeable = True
    cons[..., 4] *= np.float32(1.05)  # more energy -> more pressure
    block._versions["rhoe"] += 1

    assert np.all(np.asarray(block.P_nd) > P_before), (
        "P_nd did not pick up the change in rhoe"
    )


def test_fluid_batched_getter_matches_the_singles():
    """PerfectFluid.get_P_h_T agrees with get_P / get_h / get_T."""
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    rng = np.random.default_rng(0)
    rho = np.asfortranarray(1.0 + 0.2 * rng.standard_normal(SHAPE)).astype(np.float32)
    u = np.asfortranarray(2.0 + 0.5 * rng.standard_normal(SHAPE)).astype(np.float32)

    outs = [np.zeros(SHAPE, np.float32, order="F") for _ in range(3)]
    P, h, T = fluid.get_P_h_T(rho, u, *outs)

    assert (P, h, T) == tuple(outs), "did not write into the supplied buffers"
    for got, ref in ((P, fluid.get_P(rho, u)),
                     (h, fluid.get_h(rho, u)),
                     (T, fluid.get_T(rho, u))):
        assert _ulps(got, ref) <= 4.0


def test_fluid_batched_getter_falls_back():
    """Non-float32, non-contiguous, or missing outputs take the numpy path.

    The kernel writes in place, so anything f2py would have to copy must not
    reach it -- the result would be silently discarded.
    """
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    rho = np.full(SHAPE, 1.2, dtype=np.float64, order="F")
    u = np.full(SHAPE, 2.0, dtype=np.float64, order="F")

    # float64, with outputs supplied
    outs = [np.zeros(SHAPE, np.float64, order="F") for _ in range(3)]
    P, _, _ = fluid.get_P_h_T(rho, u, *outs)
    np.testing.assert_allclose(P, fluid.get_P(rho, u))

    # no outputs supplied at all
    P, h, T = fluid.get_P_h_T(rho.astype(np.float32), u.astype(np.float32))
    np.testing.assert_allclose(P, fluid.get_P(rho, u), rtol=1e-6)

    # non-contiguous views
    r32 = np.asfortranarray(rho.astype(np.float32))[::2]
    u32 = np.asfortranarray(u.astype(np.float32))[::2]
    outs = [np.zeros_like(r32) for _ in range(3)]
    P, _, _ = fluid.get_P_h_T(r32, u32, *outs)
    np.testing.assert_allclose(P, fluid.get_P(r32, u32), rtol=1e-6)
