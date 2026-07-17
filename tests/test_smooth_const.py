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
    Low boundary (shared biased 4th difference d4 = (a0-4a1+6a2-4a3+a4)/6):
        p=0: s2 = 2a1 - a2,          s4 = a0 - d4
        p=1: s2 = (a0 + a2) / 2,     s4 = a1 - d4
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

    # Low boundary (forward-biased from index 0)
    d4_lo = (a[0] - 4.0 * a[1] + 6.0 * a[2] - 4.0 * a[3] + a[4]) / 6.0
    s2[0] = 2.0 * a[1] - a[2]
    s4[0] = a[0] - d4_lo
    s2[1] = 0.5 * (a[0] + a[2])
    s4[1] = a[1] - d4_lo

    # High boundary (backward-biased from index n-1)
    d4_hi = (
        a[n - 5] - 4.0 * a[n - 4] + 6.0 * a[n - 3] - 4.0 * a[n - 2] + a[n - 1]
    ) / 6.0
    s2[n - 1] = 2.0 * a[n - 2] - a[n - 3]
    s4[n - 1] = a[n - 1] - d4_hi
    s2[n - 2] = 0.5 * (a[n - 3] + a[n - 1])
    s4[n - 2] = a[n - 2] - d4_hi

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
    return (
        (1.0 - sum_sf) * x
        + sf2 * (s2i + s2j + s2k)
        + sf4 * (s4i + s4j + s4k)
    )


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
    iv, jv, kv = np.meshgrid(
        np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij"
    )
    f = (1.0 + iv + 2.0 * jv - 3.0 * kv).astype(typ)[..., None]
    for sf2, sf4 in ((0.2, 0.0), (0.0, 0.2), (0.15, 0.15)):
        out = run(f, sf4=sf4, sf2=sf2)
        assert np.allclose(out, f, atol=1e-3), f"linear failed sf2={sf2} sf4={sf4}"


def test_cubic_preserved_everywhere():
    """4th-order smoothing preserves a cubic at every node (biased edges)."""
    ni, nj, nk = 10, 12, 8
    iv, jv, kv = np.meshgrid(
        np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij"
    )
    # Cubic in each direction; centred to keep float32 magnitudes modest.
    f = (
        (iv - 4.5) ** 3
        + 2.0 * (jv - 5.5) ** 3
        - (kv - 3.5) ** 3
        + 1.0
    ).astype(typ)[..., None]
    out = run(f, sf4=0.2, sf2=0.0)
    # Relative check: cubic magnitudes are O(100), so scale the tolerance.
    assert np.allclose(out, f, rtol=2e-4, atol=5e-2), "cubic not preserved"


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
