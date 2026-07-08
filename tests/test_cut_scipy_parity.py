"""Parity checks for the Fortran + pykdtree replacement of scipy ``griddata``.

``ember.cut.interpolate_to_structured`` used to call ``scipy.interpolate.griddata``
twice (linear, then a nearest fallback). Those are now a single-precision Fortran
kernel (``ember.fortran.tri_interp_linear``) plus ``pykdtree`` for the nearest
fallback. scipy is no longer a dependency, so this module is self-skipping: it runs
only where scipy happens to be installed and never blocks the scipy-free suite.

The kernel interpolates on supplied triangles whereas ``griddata`` re-triangulates
its input cloud. To compare like with like, we feed the kernel scipy's *own*
Delaunay simplices, so both sides use an identical triangulation and must agree to
float32 round-off even for nonlinear fields. The nearest fallback is compared
directly against ``griddata(method="nearest")``.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.interpolate import griddata  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

import ember.fortran  # noqa: E402


def _fa(x):
    return np.asfortranarray(x, dtype=np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_linear_matches_griddata_same_triangulation(seed):
    """Kernel linear interp == griddata linear on a shared Delaunay triangulation."""
    rng = np.random.default_rng(seed)
    n_src, n_var, n_tgt = 60, 3, 200
    pts = rng.uniform(0.0, 1.0, size=(n_src, 2))

    # A deliberately nonlinear field: only an identical triangulation makes the
    # two interpolants agree, so this exercises the barycentric blend itself.
    vals = np.column_stack(
        [
            np.sin(3.0 * pts[:, 0]) * np.cos(2.0 * pts[:, 1]),
            pts[:, 0] ** 2 - pts[:, 1] ** 2,
            1.0 + pts[:, 0] + 0.5 * pts[:, 1],
        ]
    ).astype(np.float64)

    tri = Delaunay(pts)
    tri_xy = pts[tri.simplices]  # (ntri, 3, 2)
    tri_var = vals[tri.simplices]  # (ntri, 3, nvar)

    # Targets strictly inside the hull so both methods return a linear value.
    targets = rng.uniform(0.15, 0.85, size=(n_tgt, 2))

    got = ember.fortran.tri_interp_linear(_fa(tri_xy), _fa(tri_var), _fa(targets))
    ref = griddata(pts, vals, targets, method="linear")

    both = np.isfinite(got[:, 0]) & np.isfinite(ref[:, 0])
    assert both.sum() > n_tgt // 2  # most interior targets resolved by both
    assert np.allclose(got[both], ref[both], rtol=1e-4, atol=1e-5)
    assert n_var == tri_var.shape[2]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nearest_fallback_matches_griddata(seed):
    """pykdtree nearest fill == griddata(method="nearest")."""
    from pykdtree.kdtree import KDTree

    rng = np.random.default_rng(seed)
    src = rng.uniform(0.0, 1.0, size=(80, 2))
    vals = rng.standard_normal((80, 4))
    targets = rng.uniform(-0.2, 1.2, size=(300, 2))  # spills outside the hull

    _, idx = KDTree(np.ascontiguousarray(src, np.float32)).query(
        np.ascontiguousarray(targets, np.float32)
    )
    got = vals[idx.ravel()]
    ref = griddata(src, vals, targets, method="nearest")

    assert np.allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_full_pipeline_matches_scipy_on_linear_field():
    """End-to-end interpolate_to_structured agrees with a scipy griddata path.

    Both reproduce an affine field exactly, so the full geometry pipeline plus the
    new kernel must match the analytic field (and hence any scipy reference) to
    float32 round-off.
    """
    import ember.block
    import ember.fluid
    import ember.grid
    from ember import util
    from ember.cut import unstructured, interpolate_to_structured

    Nb = 8
    pitch = 2.0 * np.pi / Nb
    shape = (9, 9, 9)
    block = ember.block.Block(shape=shape)
    xrt = util.linmesh3([0.0, 1.0], [1.0, 2.0], [0.0, pitch], shape)
    x, r = xrt[..., 0], xrt[..., 1]
    block.set_x(x).set_r(r).set_t(xrt[..., 2])
    block.set_fluid(ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72))
    coeffs = [
        (1.0, 0.5, 0.3),
        (100.0, 20.0, -10.0),
        (50.0, -5.0, 8.0),
        (25.0, 3.0, 4.0),
        (2.5e5, 1.0e4, -2.0e4),
    ]
    cons = np.empty((*shape, 5))
    for k, (c0, cx, cr) in enumerate(coeffs):
        cons[..., k] = c0 + cx * x + cr * r
    block.set_conserved(cons)
    block.set_Nb(Nb)
    grid = ember.grid.Grid([block])

    unstr = unstructured(grid, np.array([[0.1, 1.1], [0.9, 1.9]]))
    res = interpolate_to_structured(unstr, (60, 24), periodic=True)
    d = res._data
    x_t, r_t = d[..., 0], d[..., 1]
    for k, (c0, cx, cr) in enumerate(coeffs):
        expected = c0 + cx * x_t + cr * r_t
        assert np.allclose(d[..., 3 + k], expected, rtol=1e-4, atol=1e-4 * abs(c0))
