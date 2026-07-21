"""Correctness harness for the constant-coefficient smoother ``smooth3d_const``.

``smooth3d_const`` (``src/ember/_fortran/smooth_const.f90``) applies blended
2nd/4th-order artificial dissipation with fixed, isotropic ``sf2``/``sf4``
factors. Unlike the adaptive ``smooth3d`` (covered by ``test_smooth.py``) it has
no shock sensor, no CFL scaling and no length weighting, and it uses biased
4th-order differences at the boundaries so it is cubic-exact *everywhere*
(interior and edges).

These tests are the correctness gate for optimising the kernel:

- polynomial exactness (zero / constant / linear / cubic) validates the whole
  stencil set, including the biased boundary formulas;
- ``test_matches_numpy_reference`` checks arbitrary random data against an
  independent, separable NumPy reimplementation of the exact per-direction
  stencils -- so a refactor that changes results anywhere is caught, within the
  float32 reassociation tolerance the ``-Ofast`` build is allowed.
"""

import numpy as np
import pytest

from ember import fortran

typ = np.float32


def run(x, sf4, sf2):
    """Call ``smooth3d_const`` on a copy of ``x`` and return the result.

    ``x`` is any array shaped ``(ni, nj, nk, np)``; it is copied to an
    F-contiguous float32 array (the kernel writes it in place) and a matching
    ``(ni, nj, nk)`` work array is supplied.
    """
    xf = np.asfortranarray(x, dtype=typ)
    xs = np.zeros(xf.shape[:3], order="F", dtype=typ)
    fortran.smooth3d_const(x=xf, sf4=typ(sf4), sf2=typ(sf2), xs=xs)
    return xf


def _dir_ops(a, axis):
    """Return ``(s2, s4)``: the 2nd- and 4th-order smoothing targets along
    ``axis``, matching the kernel's central-interior / biased-boundary stencils.

    Central (index 2..n-3):
        s2 = (a[-1] + a[+1]) / 2
        s4 = (-a[-2] + 4a[-1] + 4a[+1] - a[+2]) / 6
    Boundary rows come from M = S^T S (S = 2nd differences where they fit), so
    the 4th-order term degrades to 2nd order at the two nodes nearest each face:
        p=0: s2 = 2a1 - a2,          s4 = a0 - (a0 - 2a1 + a2)/6
        p=1: s2 = (a0 + a2) / 2,     s4 = a1 - (-2a0 + 5a1 - 4a2 + a3)/6
    High boundary is the mirror image.
    """
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    s2 = np.empty_like(a)
    s4 = np.empty_like(a)

    # Central interior
    s2[2 : n - 2] = 0.5 * (a[1 : n - 3] + a[3 : n - 1])
    s4[2 : n - 2] = (
        -a[0 : n - 4] + 4.0 * a[1 : n - 3] + 4.0 * a[3 : n - 1] - a[4:n]
    ) / 6.0

    # Low boundary: rows 0 and 1 of M = S^T S
    s2[0] = 2.0 * a[1] - a[2]
    s4[0] = a[0] - (a[0] - 2.0 * a[1] + a[2]) / 6.0
    s2[1] = 0.5 * (a[0] + a[2])
    s4[1] = a[1] - (-2.0 * a[0] + 5.0 * a[1] - 4.0 * a[2] + a[3]) / 6.0

    # High boundary: mirror image
    s2[n - 1] = 2.0 * a[n - 2] - a[n - 3]
    s4[n - 1] = a[n - 1] - (a[n - 1] - 2.0 * a[n - 2] + a[n - 3]) / 6.0
    s2[n - 2] = 0.5 * (a[n - 3] + a[n - 1])
    s4[n - 2] = (
        a[n - 2] - (-2.0 * a[n - 1] + 5.0 * a[n - 2] - 4.0 * a[n - 3] + a[n - 4]) / 6.0
    )

    return np.moveaxis(s2, 0, axis), np.moveaxis(s4, 0, axis)


def reference(x, sf4, sf2):
    """Independent, separable NumPy reference for ``smooth3d_const`` (float64).

    The smoother is a sum of three independent 1-D operators applied to the
    original field, so each direction can be computed with whole-array slicing.
    """
    x = np.asarray(x, dtype=np.float64)
    sum_sf = 3.0 * (sf2 + sf4)
    s2i, s4i = _dir_ops(x, 0)
    s2j, s4j = _dir_ops(x, 1)
    s2k, s4k = _dir_ops(x, 2)
    return (1.0 - sum_sf) * x + sf2 * (s2i + s2j + s2k) + sf4 * (s4i + s4j + s4k)


# ------------------------------------------------------------------
# Polynomial exactness
# ------------------------------------------------------------------


def test_zero_smoothing_identity():
    """sf2 = sf4 = 0 leaves the field bitwise unchanged."""
    rng = np.random.default_rng(0)
    x = rng.random((9, 11, 7, 3)).astype(typ)
    out = run(x, sf4=0.0, sf2=0.0)
    assert np.array_equal(out, np.asfortranarray(x))


def test_constant_preserved():
    """A constant field is unchanged by any smoothing factor."""
    for sf2 in (0.1, 0.25):
        for sf4 in (0.1, 0.25):
            x = np.full((10, 8, 6, 2), 3.5, dtype=typ)
            out = run(x, sf4=sf4, sf2=sf2)
            assert np.allclose(out, 3.5, atol=1e-5)
            assert not np.isnan(out).any()


def test_linear_preserved_everywhere():
    """A linear field is preserved at every node, including boundaries."""
    ni, nj, nk = 10, 12, 8
    iv, jv, kv = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij")
    f = (1.0 + iv + 2.0 * jv - 3.0 * kv).astype(typ)[..., None]
    for sf2, sf4 in ((0.2, 0.0), (0.0, 0.2), (0.15, 0.15)):
        out = run(f, sf4=sf4, sf2=sf2)
        assert np.allclose(out, f, atol=1e-3), f"linear failed sf2={sf2} sf4={sf4}"


def test_cubic_preserved_interior():
    """4th-order smoothing preserves a cubic in the interior.

    Not at the boundary: with M = S^T S the two nodes nearest each face apply a
    2nd difference, which a cubic does not survive. That is the deliberate price
    of a symmetric positive semi-definite M (see the sf4 section below); the
    earlier cubic-exact-everywhere closure was defective and amplified.
    """
    ni, nj, nk = 10, 12, 8
    iv, jv, kv = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij")
    # Cubic in each direction; centred to keep float32 magnitudes modest.
    f = ((iv - 4.5) ** 3 + 2.0 * (jv - 5.5) ** 3 - (kv - 3.5) ** 3 + 1.0).astype(typ)[
        ..., None
    ]
    out = run(f, sf4=0.2, sf2=0.0)
    core = (slice(2, -2), slice(2, -2), slice(2, -2))
    # Relative check: cubic magnitudes are O(100), so scale the tolerance.
    assert np.allclose(out[core], f[core], rtol=2e-4, atol=5e-2), (
        "cubic not preserved in interior"
    )


# ------------------------------------------------------------------
# Reference match on arbitrary data
# ------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(9, 11, 7, 5), (5, 5, 5, 1), (16, 8, 12, 3)])
def test_matches_numpy_reference(shape):
    """Random multi-component data matches the independent NumPy reference."""
    rng = np.random.default_rng(shape[0])
    x = rng.random(shape).astype(typ)
    sf4, sf2 = 0.15, 0.2
    out = run(x, sf4=sf4, sf2=sf2)
    ref = reference(x, sf4, sf2)
    assert np.allclose(out, ref, rtol=1e-4, atol=2e-4), (
        f"max abs diff {np.abs(out - ref).max():.2e}"
    )


def _run_kr(x, sf4, sf2, kr):
    """Call the kernel with an explicit rolling-buffer depth ``kr``."""
    xf = np.asfortranarray(x, dtype=typ)
    ni, nj, nk = xf.shape[:3]
    xs = np.zeros((ni, nj, kr), order="F", dtype=typ)
    fortran.smooth3d_const(x=xf, sf4=typ(sf4), sf2=typ(sf2), xs=xs)
    return xf


def test_kr_independent():
    """The result is bitwise independent of the rolling-buffer depth ``kr``.

    ``kr = nk`` never recycles a slot (no lag collisions); the production
    ``kr = min(6, nk)`` rolls. Identical output proves the slot bookkeeping and
    the lagged writeback/tail flush are correct regardless of depth.
    """
    rng = np.random.default_rng(7)
    for nk in (5, 6, 7, 11):
        x = rng.random((10, 8, nk, 2)).astype(typ)
        ref = _run_kr(x, 0.15, 0.2, kr=nk)
        for kr in {min(6, nk), nk, nk + 1}:
            got = _run_kr(x, 0.15, 0.2, kr=kr)
            assert np.array_equal(got, ref), f"kr={kr} differs at nk={nk}"


# ------------------------------------------------------------------
# 2nd-order diffusivity at the i/j/k boundaries
# ------------------------------------------------------------------
#
# Polynomial exactness above pins down what the stencils *preserve*; it says
# nothing about the sign of what they do to everything else. The boundary
# 2nd-order stencil is a one-parameter family: linear exactness forces
#
#     s2[0] = (1+c)*x0 - 2c*x1 + c*x2   ->   contribution = sf2 * c * d2(x1)
#
# so every linear-exact choice is some multiple of the second difference about
# the *first interior* node, and only the scalar c is free. The kernel uses
# ``s2[0] = 2*x1 - x2`` (c = -1). Because the difference is centred inboard,
# damping the boundary node requires c < 0: at c = 0 the node is left untouched
# and never decays, and at c = +1/2 (the "interior-consistent" sign) a boundary
# spike *grows*. The tests below are what distinguishes those cases -- they fail
# for c >= 0 while every polynomial-exactness test above still passes.
#
# sf4 is held at zero throughout. The 4th-order operator alone is neutrally
# stable rather than contracting (rho = 1 with a defective mode), so it has no
# strict-decay property to assert; mixed sf2+sf4 converges but not monotonically.

# Realistic block dimensions, distinct per axis so a transposed index shows up.
NI, NJ, NK = 49, 33, 25


def _axis_field(axis, values):
    """A block varying as ``values`` along ``axis`` and constant across the other two.

    Constants are preserved exactly by every stencil, biased boundary formulas
    included, so the other two directions contribute exactly zero and the 3-D
    kernel reduces to the pure 1-D operator for ``axis``. That isolates one pair
    of faces at a time.
    """
    shape = [1, 1, 1]
    shape[axis] = len(values)
    a = np.asarray(values, dtype=typ).reshape(shape)
    return np.broadcast_to(a, (NI, NJ, NK)).copy()[..., None]


def _axis_line(y, axis):
    """Extract the 1-D profile along ``axis`` from a field built by ``_axis_field``."""
    return np.moveaxis(y[..., 0], axis, 0)[:, 0, 0].astype(np.float64)


def _linear_residual(y, axis):
    """L2 distance of the profile along ``axis`` from the nearest linear field.

    ``span{1, x}`` is exactly the eigenvalue-1 eigenspace of the 1-D sf2
    operator (dimension 2 at every n), so this residual is the component the
    smoother must strictly remove.
    """
    a = _axis_line(y, axis)
    basis = np.vstack([np.ones(a.size), np.arange(a.size)]).T
    coef, *_ = np.linalg.lstsq(basis, a, rcond=None)
    return float(np.linalg.norm(a - basis @ coef))


@pytest.mark.parametrize("sf2", [0.05, 0.25, 1.0 / 3.0])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sf2_boundary_spike_decays(axis, sf2):
    """A spike sitting *on* a boundary node decays under repeated smoothing.

    Covers both ends of all three axes (six faces). This is the direct probe of
    the boundary row: at c = 0 the amplitude would sit at 1.0 forever, and at
    c = +1/2 it would grow.
    """
    n = (NI, NJ, NK)[axis]
    for idx in (0, n - 1):
        values = np.zeros(n)
        values[idx] = 1.0
        y = _axis_field(axis, values)
        amp = [1.0]
        for _ in range(40):
            y = run(y, sf4=0.0, sf2=sf2)
            amp.append(abs(_axis_line(y, axis)[idx]))
        rise = np.diff(amp).max()
        assert rise < 0.0, (
            f"boundary spike not strictly decaying: axis={axis} idx={idx} "
            f"sf2={sf2} worst step {rise:+.2e}"
        )
        assert amp[-1] < 0.5 * amp[0]


@pytest.mark.parametrize("sf2", [0.05, 0.25, 1.0 / 3.0])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sf2_contracts_to_linear(axis, sf2):
    """Repeated smoothing strictly contracts random data towards a linear field.

    Strict per-pass decrease of the residual is the statement of diffusivity;
    it holds despite the operator being non-normal at the boundaries (no
    transient growth for sf4 = 0).
    """
    n = (NI, NJ, NK)[axis]
    rng = np.random.default_rng(axis)
    y = _axis_field(axis, rng.random(n).astype(typ))
    res = [_linear_residual(y, axis)]
    for _ in range(60):
        y = run(y, sf4=0.0, sf2=sf2)
        res.append(_linear_residual(y, axis))
    rise = np.diff(res).max()
    assert rise < 0.0, (
        f"residual not strictly decreasing: axis={axis} sf2={sf2} "
        f"worst step {rise:+.2e}"
    )
    assert res[-1] < res[0]


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 4th-order diffusivity at the i/j/k boundaries
# ------------------------------------------------------------------
#
# The same contraction framework as above, with the fixed subspace widened from
# span{1, x} to span{1, x, x^2, x^3} because the 4th-order operator annihilates
# cubics rather than linears.
#
# These tests were written against the previous closure, which shared one biased
# 4th difference between the two nodes nearest each face. That made the boundary
# block defective (rho = 1 with a defective eigenvalue), so repeated application
# amplified the residual like sf4*sqrt(k) instead of damping it -- at n = 5 all
# five nodes read the identical five points, giving A = I - N with N^2 = 0 and
# hence A^k = I - k*N. They failed 10/12 at the time.
#
# Building the 4th-order term as M = S^T S fixed it: M is symmetric positive
# semi-definite, so ||A^k||_2 = 1 exactly, for every k and every sf4 <= 0.75.
#
# sf2 is held at zero: mixing in sf2 masks the growth behind the 2nd-order
# contraction, which is why the exactness suite never caught this.

_SF4_PASSES = 400


def _cubic_residual(y, axis):
    """L2 distance of the profile along ``axis`` from the nearest cubic.

    ``span{1, x, x^2, x^3}`` is the null space of the 4th-order operator, so
    this is the component a diffusive smoother would have to remove. The
    abscissa is normalised to [0, 1] to keep the Vandermonde well conditioned.
    """
    a = _axis_line(y, axis)
    t = np.arange(a.size, dtype=np.float64) / a.size
    basis = np.vstack([t**p for p in range(4)]).T
    coef, *_ = np.linalg.lstsq(basis, a, rcond=None)
    return float(np.linalg.norm(a - basis @ coef))


def _sf4_residual_history(axis, sf4, passes=_SF4_PASSES):
    """Cubic residual after each of ``passes`` applications, normalised to r[0]."""
    n = (NI, NJ, NK)[axis]
    rng = np.random.default_rng(axis)
    y = _axis_field(axis, rng.random(n).astype(typ))
    res = [_cubic_residual(y, axis)]
    for _ in range(passes):
        y = run(y, sf4=sf4, sf2=0.0)
        res.append(_cubic_residual(y, axis))
    return np.array(res) / res[0]


@pytest.mark.parametrize("sf4", [0.05, 0.25])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sf4_residual_does_not_grow_late(axis, sf4):
    """Late-time cubic residual must not grow: r[400] <= r[100].

    This is the direct signature of the defective mode -- the early transient
    decays, then the residual turns around and climbs linearly in the pass
    count. Currently fails on all three axes at both sf4 values.
    """
    res = _sf4_residual_history(axis, sf4)
    growth = res[400] / res[100]
    assert growth <= 1.0, (
        f"cubic residual growing late: axis={axis} sf4={sf4} "
        f"r[400]/r[100] = {growth:.3f}"
    )


@pytest.mark.parametrize("sf4", [0.05, 0.25])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sf4_residual_bounded(axis, sf4):
    """The cubic residual must never exceed its starting value.

    A diffusive operator cannot amplify the component it is supposed to remove.
    Currently fails at sf4 = 0.25 (reaching ~3.4x on the i axis).
    """
    res = _sf4_residual_history(axis, sf4)
    peak = res.max()
    assert peak <= 1.0, (
        f"cubic residual amplified: axis={axis} sf4={sf4} "
        f"peak r[k]/r[0] = {peak:.3f} at pass {int(res.argmax())}"
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sf4_boundary_spike_decays(axis):
    """A spike on a boundary node decays monotonically under 4th-order smoothing.

    The magnitude bound is loose on purpose. Under M = S^T S the wall applies a
    2nd difference, so a boundary spike damps more slowly than under the old
    cubic-exact closure (0.23 -> 0.67 after 40 passes at sf4 = 0.25). That
    slowdown is the accepted price of a non-amplifying operator, so the bound
    was retuned; the strict monotone decrease below is the real assertion.
    """
    n = (NI, NJ, NK)[axis]
    for idx in (0, n - 1):
        values = np.zeros(n)
        values[idx] = 1.0
        y = _axis_field(axis, values)
        amp = [1.0]
        for _ in range(40):
            y = run(y, sf4=0.25, sf2=0.0)
            amp.append(abs(_axis_line(y, axis)[idx]))
        assert np.diff(amp).max() < 0.0, f"axis={axis} idx={idx} not decaying"
        assert amp[-1] < 0.8 * amp[0]


@pytest.mark.parametrize("sf4", [0.01, 0.05, 0.25, 0.75])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_sf4_operator_symmetric_and_non_amplifying(axis, sf4):
    """The 1-D sf4 operator is symmetric and never amplifies, at any pass count.

    Probing with unit vectors recovers the kernel's 1-D operator exactly (the
    other two directions contribute zero on a field constant across them). M =
    S^T S makes it symmetric positive semi-definite, hence ||A^k||_2 = 1 for all
    k -- the guarantee the previous closure could not provide. This is the
    regression gate: it fails the moment a boundary row stops being the mirror
    of the corresponding interior column.
    """
    n = (NI, NJ, NK)[axis]
    op = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        op[:, j] = _axis_line(run(_axis_field(axis, e), sf4=sf4, sf2=0.0), axis)
    asym = np.abs(op - op.T).max()
    assert asym < 1e-6, f"operator not symmetric: axis={axis} sf4={sf4} {asym:.2e}"
    for k in (1, 10, 100, 1000):
        gain = np.linalg.svd(np.linalg.matrix_power(op, k), compute_uv=False)[0]
        assert gain < 1.0 + 1e-3, (
            f"amplifying at k={k}: axis={axis} sf4={sf4} ||A^k||={gain:.4f}"
        )
