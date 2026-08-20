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
import ember.fortran
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
    ("P_nd", "ho_nd", "T_nd", "_Vxrt_nd_uninit", "_halfVsq_nd_uninit", "_u_nd_uninit"),
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
            f"{name}: {_ulps(got, ref[name]):.1f} ulp of scale exceeds {TOL_ULP[name]}"
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
    for got, ref in (
        (P, fluid.get_P(rho, u)),
        (h, fluid.get_h(rho, u)),
        (T, fluid.get_T(rho, u)),
    ):
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


# ---------------------------------------------------------------------------
# The real-gas kernel behind RealFluid.get_P_h_T
# ---------------------------------------------------------------------------


def _real_fluid_and_state(shape=SHAPE, dtype=np.float32, order="F"):
    """A fitted RealFluid and a state filling the middle of its box.

    States are drawn from the fluid's own bounds rather than written out: the
    box moves with the datum and the reference scales, so literals would drift
    out of range and the kernel would be compared on an extrapolation.
    """
    from conftest import VanDerWaals, fit_real_fluid

    fluid = fit_real_fluid(VanDerWaals(), (1.0, 150.0), (3.0e5, 5.0e5))
    rng = np.random.default_rng(0)

    def _span(lim):
        lo = lim[0] + 0.25 * (lim[1] - lim[0])
        hi = lim[0] + 0.75 * (lim[1] - lim[0])
        arr = rng.uniform(lo, hi, shape).astype(dtype)
        return np.asfortranarray(arr) if order == "F" else np.ascontiguousarray(arr)

    return fluid, _span(fluid.rho_lim_nd), _span(fluid.u_lim_nd)


def test_real_fluid_batched_getter_matches_the_singles():
    """RealFluid.get_P_h_T agrees with get_P / get_h / get_T.

    Not bitwise. The kernel builds the Legendre basis by its three-term
    recurrence and contracts it, where numpy runs a Clenshaw sweep per surface;
    one basis serves all four contractions that way, which is most of the point
    of the kernel, and the price is a few ulp.
    """
    fluid, rho, u = _real_fluid_and_state()

    outs = [np.zeros(SHAPE, np.float32, order="F") for _ in range(3)]
    P, h, T = fluid.get_P_h_T(rho, u, *outs)

    assert (P, h, T) == tuple(outs), "did not write into the supplied buffers"
    for name, got, ref in (
        ("P", P, fluid.get_P(rho, u)),
        ("h", h, fluid.get_h(rho, u)),
        ("T", T, fluid.get_T(rho, u)),
    ):
        assert _ulps(got, ref) <= 8.0, (
            f"{name}: {_ulps(got, ref):.1f} ulp of scale exceeds 8"
        )


def test_real_fluid_batched_getter_is_no_less_accurate():
    """The kernel is as close to the modelled gas as the numpy path is.

    Agreement with numpy alone would not catch the two drifting together, and
    it is the wrong reference in any case: numpy is not the truth here, the
    van der Waals gas the coefficients were fitted to is. Both paths carry the
    fit residual, which dwarfs either one's arithmetic, so the test is that the
    kernel adds nothing measurable on top of what numpy already gives away.
    """
    from conftest import VanDerWaals

    fluid, rho, u = _real_fluid_and_state()
    outs = [np.zeros(SHAPE, np.float32, order="F") for _ in range(3)]
    P_k, _, T_k = fluid.get_P_h_T(rho, u, *outs)

    # The analytic model works in absolute internal energy; the fluid measures
    # it from the datum, so shift back before evaluating the reference.
    u_abs = u + (3.0e5 - fluid.u_lim_nd[0])
    model = VanDerWaals()

    for name, kernel, numpy_path, exact in (
        ("P", P_k, fluid.get_P(rho, u), model.get_P(rho, u_abs)),
        ("T", T_k, fluid.get_T(rho, u), model.get_T(rho, u_abs)),
    ):
        err_kernel = float(np.abs(kernel - exact).max())
        err_numpy = float(np.abs(numpy_path - exact).max())
        assert err_kernel <= 1.05 * err_numpy + np.spacing(
            np.float32(np.abs(exact).max())
        ), f"{name}: kernel err {err_kernel:.3e} worse than numpy {err_numpy:.3e}"


# Coefficients with no gas behind them, for the term-by-term kernel test
# below. Shapes mirror what differentiating a (5, 4) entropy surface gives:
# one order shorter in x for scx, one shorter in y for scy and sly.
_SCX = np.asfortranarray(
    np.array(
        [
            [-0.9, 0.20, -0.05, 0.02],
            [0.30, -0.10, 0.04, -0.01],
            [-0.08, 0.03, -0.01, 0.005],
            [0.02, -0.01, 0.004, -0.002],
        ],
        dtype=np.float32,
    )
)
_SCY = np.asfortranarray(
    np.array(
        [
            [0.60, 0.15, -0.04],
            [0.20, -0.08, 0.02],
            [-0.06, 0.03, -0.01],
            [0.02, -0.01, 0.004],
            [-0.005, 0.002, -0.001],
        ],
        dtype=np.float32,
    )
)
_SL = np.array([0.80, -0.25, 0.10, -0.04], dtype=np.float32)
_SLY = np.array([0.50, -0.20, 0.08], dtype=np.float32)
_HATS = tuple(np.float32(v) for v in (0.5, -1.5, 0.4, -0.2))  # xa, xb, ya, yb


def _kernel_reference(rho, u, scx, scy, sl, sly, xa, xb, ya, yb):
    """P, h and T from coefficient arrays, written out in numpy."""
    leg = np.polynomial.legendre
    x, y = rho * xa + xb, u * ya + yb
    lnr = np.log(rho)
    s_r = leg.legval2d(x, y, scx) * xa + leg.legval(y, sl) / rho
    s_u = (leg.legval2d(x, y, scy) + leg.legval(y, sly) * lnr) * ya
    T = 1.0 / s_u
    P = -(rho**2) * T * s_r
    return P, u + P / rho, T


def test_real_kernel_evaluates_every_term():
    """The kernel against the formula, on coefficients that exercise all four.

    Needed because no fitted gas can do this job. The multiplier on log(rho) is
    the compressibility factor at zero density, and every real gas is ideal
    there, so it comes out constant in internal energy: across van der Waals
    fits from ideal to strongly attracting, ``Sl`` is Rgas to the digit and
    ``Sl_y`` never rises above 1e-8 of it. A test driven by a fitted fluid
    therefore cannot see the ``sly`` contraction at all -- it was checked, and
    perturbing that term by 1% left every such assertion passing.

    So the coefficients here describe no gas. They are chosen only so that all
    four contractions land at the same order, which the assertions below pin,
    and the reference is the formula itself rather than RealFluid's numpy path.
    """
    rng = np.random.default_rng(1)
    shape = (7, 5, 3)
    # Ranges chosen to keep the normalised coordinates inside [-1, 1], where
    # the Legendre basis is orthogonal and a fitted surface would live.
    rho = np.asfortranarray(rng.uniform(2.0, 5.0, shape)).astype(np.float32)
    u = np.asfortranarray(rng.uniform(0.5, 3.0, shape)).astype(np.float32)
    xa, xb, ya, yb = _HATS

    # Pin that this state exercises what it claims to: the log terms must be
    # comparable to the polynomial ones, or the test would pass with the
    # ``sl`` and ``sly`` loops deleted.
    leg = np.polynomial.legendre
    x, y = rho * xa + xb, u * ya + yb
    for label, poly, log in (
        ("s_r", leg.legval2d(x, y, _SCX) * xa, leg.legval(y, _SL) / rho),
        ("s_u", leg.legval2d(x, y, _SCY), leg.legval(y, _SLY) * np.log(rho)),
    ):
        ratio = float(np.abs(log).max() / np.abs(poly).max())
        assert 0.1 < ratio < 10.0, f"{label}: log term is {ratio:.2e} of the polynomial"

    outs = [np.zeros(shape, np.float32, order="F") for _ in range(3)]
    ember.fortran.set_p_h_t_real(
        rho=np.ravel(rho, order="A"),
        u=np.ravel(u, order="A"),
        scx=_SCX,
        scy=_SCY,
        sl=_SL,
        sly=_SLY,
        xa=xa,
        xb=xb,
        ya=ya,
        yb=yb,
        p=np.ravel(outs[0], order="A"),
        h=np.ravel(outs[1], order="A"),
        t=np.ravel(outs[2], order="A"),
    )

    ref = _kernel_reference(rho, u, _SCX, _SCY, _SL, _SLY, *_HATS)
    for name, got, want in zip(("P", "h", "T"), outs, ref):
        assert _ulps(got, want) <= 16.0, (
            f"{name}: {_ulps(got, want):.1f} ulp of scale exceeds 16"
        )


def test_real_fluid_batched_getter_falls_back():
    """Anything the kernel cannot write into takes the numpy path.

    The kernel writes in place, so a call f2py would have to copy must not
    reach it -- the result would be silently discarded. Unlike PerfectFluid,
    the fallback here is RealFluid's own fused numpy body rather than the
    unfused base, so it stays a single pass over the surface.
    """
    fluid, rho64, u64 = _real_fluid_and_state(dtype=np.float64)

    # float64, with outputs supplied
    outs = [np.zeros(SHAPE, np.float64, order="F") for _ in range(3)]
    P, _, _ = fluid.get_P_h_T(rho64, u64, *outs)
    np.testing.assert_allclose(P, fluid.get_P(rho64, u64))

    # no outputs supplied at all
    r32, u32 = rho64.astype(np.float32), u64.astype(np.float32)
    P, h, T = fluid.get_P_h_T(r32, u32)
    np.testing.assert_allclose(P, fluid.get_P(r32, u32), rtol=1e-6)

    # non-contiguous views
    rv, uv = np.asfortranarray(r32)[::2], np.asfortranarray(u32)[::2]
    outs = [np.zeros_like(rv) for _ in range(3)]
    P, _, _ = fluid.get_P_h_T(rv, uv, *outs)
    np.testing.assert_allclose(P, fluid.get_P(rv, uv), rtol=1e-6)
