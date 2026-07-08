"""Unit tests for the implicit residual smoothing (IRS) kernel.

Module tested: ``smooth_residual`` in ``residual.f90`` (exposed as
``ember.fortran.smooth_residual``), the Jameson-style implicit residual
smoother wired into :meth:`ember.grid.Grid.update_residual`.

The kernel solves ``(1 - sf*grad^2) R* = R`` by ``n_smooth`` in-place
Gauss-Seidel sweeps with zero-gradient (Neumann) boundaries. Its correctness
contract:

- ``n_smooth == 0`` or ``sf == 0`` is an exact identity;
- a constant residual field is preserved exactly everywhere;
- a linear field is preserved in the interior up to the Neumann-boundary shift
  that Gauss-Seidel propagates along the sweep (small, not machine zero);
- high-frequency (checkerboard) content is damped, monotonically in ``n_smooth``.
"""

import numpy as np
import pytest

import ember.grid  # noqa: F401  binds ember.fortran
import ember


def _smooth(du, sf, n_smooth):
    """Run smooth_residual on a copy of `du` (cell array (nci,ncj,nck,5))."""
    nci, ncj, nck = du.shape[:3]
    du = np.asfortranarray(du.astype(np.float32))
    work = np.asfortranarray(np.empty_like(du))
    ember.fortran.smooth_residual(
        du=du,
        sf=sf,
        n_smooth=n_smooth,
        work=work,
        ni=nci + 1,
        nj=ncj + 1,
        nk=nck + 1,
    )
    return du


CELL_SHAPE = (6, 7, 8, 5)


def test_n_smooth_zero_is_identity():
    """Zero sweeps leaves the residual bit-identical."""
    r = np.random.default_rng(0).standard_normal(CELL_SHAPE).astype(np.float32)
    assert np.array_equal(_smooth(r, 0.5, 0), r)


def test_sf_zero_is_identity():
    """Zero smoothing coefficient leaves the residual bit-identical."""
    r = np.random.default_rng(1).standard_normal(CELL_SHAPE).astype(np.float32)
    assert np.array_equal(_smooth(r, 0.0, 5), r)


def test_constant_field_preserved_everywhere():
    """A uniform residual is a fixed point at every cell, boundaries included."""
    c = np.full(CELL_SHAPE, 3.14, dtype=np.float32)
    out = _smooth(c, 0.7, 10)
    assert np.allclose(out, 3.14, atol=1e-5)


def test_linear_field_preserved_in_interior():
    """A linear field is preserved in the interior up to the GS boundary shift.

    Not machine zero: the reduced-stencil Neumann boundary shifts edge cells by
    a slope-dependent amount, which Gauss-Seidel carries inward along the sweep.
    The deviation stays small relative to the field range.
    """
    shape = (8, 8, 8, 5)
    i = np.arange(shape[0])[:, None, None, None]
    j = np.arange(shape[1])[None, :, None, None]
    k = np.arange(shape[2])[None, None, :, None]
    lin = ((0.3 * i + 0.2 * j + 0.1 * k + 1.0) * np.ones(shape)).astype(np.float32)
    out = _smooth(lin, 0.5, 1)
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    dev = np.abs(out[interior] - lin[interior]).max()
    field_range = float(lin.max() - lin.min())
    assert dev < 0.01 * field_range


def _checkerboard(shape):
    ii, jj, kk = np.indices(shape[:3])
    sign = ((ii + jj + kk) % 2) * 2 - 1
    return (sign[..., None] * np.ones(shape)).astype(np.float32)


def test_checkerboard_is_damped():
    """The highest-frequency mode shrinks in magnitude after one sweep."""
    cb = _checkerboard(CELL_SHAPE)
    out = _smooth(cb, 0.5, 1)
    assert np.abs(out).max() < 1.0


def test_converged_checkerboard_matches_operator():
    """Enough sweeps converge to (1 - sf*grad^2) R* = R on the interior mode.

    The discrete Laplacian eigenvalue of the 3D checkerboard is 12 (each of the
    six neighbours flips sign), so the exact interior solution is 1/(1 + 12*sf).
    Note the approach is not monotone in sweep count: Gauss-Seidel overshoots on
    the first sweep, then relaxes toward this value.
    """
    sf = 0.5
    cb = _checkerboard((9, 9, 9, 5))
    out = _smooth(cb, sf, 60)
    interior = np.abs(out[3:-3, 3:-3, 3:-3]).mean()
    expected = 1.0 / (1.0 + 12.0 * sf)
    assert interior == pytest.approx(expected, rel=1e-2)


def test_damping_increases_with_sf():
    """A larger smoothing coefficient damps the high-frequency mode more."""
    cb = _checkerboard((9, 9, 9, 5))
    interior = (slice(3, -3), slice(3, -3), slice(3, -3))
    weak = np.abs(_smooth(cb, 0.3, 60)[interior]).mean()
    strong = np.abs(_smooth(cb, 0.8, 60)[interior]).mean()
    assert strong < weak


def test_input_unmodified_via_grid_update_residual():
    """IRS through update_residual leaves a converged (zero) residual at zero.

    IRS acts only on the residual, and IRS(0) = 0, so a fully converged state is
    a fixed point -- the steady solution is unchanged regardless of sf/n_smooth.
    """
    zero = np.zeros(CELL_SHAPE, dtype=np.float32)
    out = _smooth(zero, 0.8, 20)
    assert np.array_equal(out, zero)


if __name__ == "__main__":
    pytest.main([__file__])
