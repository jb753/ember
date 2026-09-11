"""Correctness harness for the adaptive smoother ``smooth3d_adaptive``.

``smooth3d_adaptive`` (``src/ember/_fortran/smooth.f90``) applies the same
stencil family and the same biased boundary closures as ``smooth3d_const``
(covered by ``test_smooth_const.py``), but the 2nd-order factor is a per-node,
per-direction JST shock sensor rather than a constant, and the 4th-order factor
is clipped against it::

    sf2n = max(sf2P * nu_P, sf2T * nu_T)
    sf4n = max(sf4 - sf2n, 0)

The strategy here is to pin the sensor to a *known constant* and then demand an
exact match against ``smooth3d_const``, which is independently verified against
a NumPy reference in ``test_smooth_const.py``. Two sensor states bracket the
blend:

- **uniform P, T** -- ``nu = 0`` exactly, so ``sf2n = 0`` and ``sf4n = sf4``.
  The operator must collapse onto ``smooth3d_const(sf2=0, sf4=sf4)``: pure
  4th-order, the 4th-order boundary rows included.
- **checkerboard P** -- ``nu = |B| / A`` exactly and uniformly, at every node
  including the faces (the shifted boundary stencil sees the same alternation).
  Choosing ``sf4 < sf2n`` clips ``sf4n`` to zero, so the operator must collapse
  onto ``smooth3d_const(sf2=sf2n, sf4=0)``: pure 2nd-order.

Between them these fix both limbs of the blend and every boundary row. The
remaining tests cover the properties that hold for an *arbitrary* sensor field,
which is where the reworked boundary treatment differs from the old one: linear
fields are now preserved at every node including the faces.
"""

import numpy as np
import pytest

from ember import fortran

typ = np.float32

# Checkerboard amplitudes chosen so the sensor is exact in float32:
# numerator 4B = 1.0, denominator 4A = 4.0, nu = 0.25 with no rounding.
CHECKER_A = 1.0
CHECKER_B = 0.25
CHECKER_NU = 0.25


def run_adaptive(x, P, T, sf4, sf2P, sf2T):
    """Call ``smooth3d_adaptive`` on a copy of ``x`` and return the result."""
    xf = np.asfortranarray(x, dtype=typ)
    shape3 = xf.shape[:3]
    sf2n = np.zeros(shape3 + (3,), order="F", dtype=typ)
    dx = np.zeros(shape3, order="F", dtype=typ)
    fortran.smooth3d_adaptive(
        x=xf,
        p=np.asfortranarray(P, dtype=typ),
        t=np.asfortranarray(T, dtype=typ),
        sf4=typ(sf4),
        sf2p=typ(sf2P),
        sf2t=typ(sf2T),
        sf2n=sf2n,
        dx=dx,
    )
    return xf


def run_const(x, sf4, sf2):
    """Call ``smooth3d_const`` on a copy of ``x`` and return the result."""
    xf = np.asfortranarray(x, dtype=typ)
    xs = np.zeros(xf.shape[:3], order="F", dtype=typ)
    fortran.smooth3d_const(x=xf, sf4=typ(sf4), sf2=typ(sf2), xs=xs)
    return xf


def uniform_field(shape3, value=1.0):
    """A strictly positive constant field: sensor reads exactly zero."""
    return np.full(shape3, value, dtype=typ)


def checkerboard_field(shape3):
    """``A + B (-1)^(i+j+k)``: sensor reads exactly ``B/A`` in all 3 directions.

    The alternation is seen identically by the centred interior stencil and by
    the shifted boundary stencil (``1,2,3`` and ``n-2,n-1,n``), so the sensor is
    uniform over the whole volume -- faces, edges and corners included.
    """
    ni, nj, nk = shape3
    i, j, k = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij")
    sign = np.where((i + j + k) % 2 == 0, 1.0, -1.0)
    return (CHECKER_A + CHECKER_B * sign).astype(typ)


def random_state(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(typ)


SHAPE = (7, 6, 8, 5)
SHAPE3 = SHAPE[:3]


# ----------------------------------------------------------------------
# Collapse onto smooth3d_const at the two ends of the blend
# ----------------------------------------------------------------------


@pytest.mark.parametrize("sf4", [0.0, 0.1, 0.25, 0.5])
def test_uniform_sensor_matches_const_pure_4th(sf4):
    """Uniform P and T: sf2n = 0, sf4n = sf4, so this is const(sf2=0, sf4)."""
    x = random_state(SHAPE, seed=1)
    P = uniform_field(SHAPE3, 1.0)
    T = uniform_field(SHAPE3, 300.0)

    got = run_adaptive(x, P, T, sf4=sf4, sf2P=0.5, sf2T=0.5)
    want = run_const(x, sf4=sf4, sf2=0.0)

    assert np.allclose(got, want, atol=1e-5), (
        f"adaptive with a dead sensor diverged from const(sf2=0, sf4={sf4})"
    )


@pytest.mark.parametrize("sf4", [0.0, 0.05, 0.1])
def test_saturated_sensor_matches_const_pure_2nd(sf4):
    """Checkerboard P: sf2n = 0.25 uniformly, and sf4 < sf2n clips sf4n to 0.

    So the operator must be exactly const(sf2=0.25, sf4=0).
    """
    x = random_state(SHAPE, seed=2)
    P = checkerboard_field(SHAPE3)
    T = uniform_field(SHAPE3, 300.0)

    got = run_adaptive(x, P, T, sf4=sf4, sf2P=1.0, sf2T=0.0)
    want = run_const(x, sf4=0.0, sf2=CHECKER_NU)

    assert np.allclose(got, want, atol=1e-5), (
        f"adaptive with a saturated sensor diverged from const(sf2={CHECKER_NU}, sf4=0)"
    )


def test_sensor_reads_checkerboard_exactly():
    """Guard the premise of the test above: nu is B/A, uniformly, everywhere.

    Driven through the kernel rather than asserted on the helper: with sf4=0
    and sf2P=1, a single pass must equal const(sf2=B/A, sf4=0). If the sensor
    were not uniform (for instance if the boundary rows read a different
    curvature) this equality would fail at the faces.
    """
    x = random_state(SHAPE, seed=3)
    P = checkerboard_field(SHAPE3)
    T = uniform_field(SHAPE3, 300.0)

    got = run_adaptive(x, P, T, sf4=0.0, sf2P=1.0, sf2T=0.0)
    want = run_const(x, sf4=0.0, sf2=CHECKER_NU)

    assert np.allclose(got, want, atol=1e-5), "sensor is not uniform over the volume"


def test_temperature_sensor_drives_the_blend():
    """The T limb alone must saturate the sensor when P is uniform.

    Same checkerboard, moved from P to T, with the weights swapped: the result
    must be identical to the P-driven case, since sf2n is an elementwise max.
    """
    x = random_state(SHAPE, seed=4)
    checker = checkerboard_field(SHAPE3)
    flat = uniform_field(SHAPE3, 1.0)

    by_p = run_adaptive(x, checker, flat, sf4=0.1, sf2P=1.0, sf2T=0.0)
    by_t = run_adaptive(x, flat, checker, sf4=0.1, sf2P=0.0, sf2T=1.0)

    assert np.allclose(by_p, by_t, atol=1e-6), "T sensor limb does not match the P limb"


# ----------------------------------------------------------------------
# Properties that hold for an arbitrary (spatially varying) sensor
# ----------------------------------------------------------------------


def varying_sensor_fields(shape3, seed=10):
    """A strictly positive, non-smooth P/T pair: sensor varies node to node."""
    rng = np.random.default_rng(seed)
    P = (1.0 + 0.3 * rng.standard_normal(shape3)).astype(typ)
    T = (300.0 + 50.0 * rng.standard_normal(shape3)).astype(typ)
    return np.abs(P) + typ(0.1), np.abs(T) + typ(1.0)


@pytest.mark.parametrize("value", [0.0, 1.0, -2.5])
def test_constant_preserved_everywhere(value):
    """Every row has zero sum, so a constant survives for any sensor field."""
    x = np.full(SHAPE, value, dtype=typ)
    P, T = varying_sensor_fields(SHAPE3)

    got = run_adaptive(x, P, T, sf4=0.3, sf2P=0.5, sf2T=0.5)

    assert np.allclose(got, x, atol=1e-6), f"constant {value} not preserved"


@pytest.mark.parametrize("sf4", [0.0, 0.1, 0.3, 0.6])
def test_linear_preserved_everywhere(sf4):
    """A linear field is preserved at every node, faces included.

    This is the property the old boundary treatment lacked: it targeted the
    first interior neighbour on the outermost plane, which is a *first*
    difference and pulls a linear profile inward every pass. The biased rows
    adopted from ``smooth3d_const`` are second and fourth differences of the
    unshifted stencil, so they annihilate linears at every node.
    """
    ni, nj, nk, npr = SHAPE
    i, j, k = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij")
    f = (2.0 + 3.0 * i - 1.5 * j + 0.75 * k).astype(typ)
    x = np.repeat(f[:, :, :, None], npr, axis=3)
    P, T = varying_sensor_fields(SHAPE3)

    got = run_adaptive(x, P, T, sf4=sf4, sf2P=0.5, sf2T=0.5)

    assert np.allclose(got, x, atol=1e-3), f"linear not preserved, sf4={sf4}"


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_linear_preserved_on_the_faces_specifically(axis):
    """Pin the face planes of the linear test, which is where the old rows failed.

    A plain ramp along ``axis`` with a saturated sensor: the outermost two
    planes normal to ``axis`` must be untouched to float32 noise.
    """
    n = SHAPE[axis]
    ramp = np.arange(n, dtype=typ) * typ(4.0)
    shape = [1, 1, 1, 1]
    shape[axis] = n
    x = np.broadcast_to(ramp.reshape(shape), SHAPE).astype(typ)
    P = checkerboard_field(SHAPE3)
    T = uniform_field(SHAPE3, 300.0)

    got = run_adaptive(x, P, T, sf4=0.1, sf2P=1.0, sf2T=0.0)

    face = [slice(None)] * 4
    for plane in (0, 1, n - 2, n - 1):
        face[axis] = plane
        assert np.allclose(got[tuple(face)], x[tuple(face)], atol=1e-3), (
            f"linear ramp disturbed on plane {plane} of axis {axis}"
        )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cubic_preserved_interior_when_sensor_dead(axis):
    """With sf2n = 0 the interior rows are pure 4th differences: cubic-exact.

    Only the interior is checked. The boundary rows degrade to 2nd-order
    dissipation by construction (inherited from ``smooth3d_const``), so a cubic
    is deliberately *not* reproduced on the two planes nearest each face.
    """
    n = SHAPE[axis]
    t = np.arange(n, dtype=np.float64)
    cubic = (t**3 - 4.0 * t**2 + 2.0 * t + 1.0).astype(typ)
    shape = [1, 1, 1, 1]
    shape[axis] = n
    x = np.broadcast_to(cubic.reshape(shape), SHAPE).astype(typ)
    P = uniform_field(SHAPE3, 1.0)
    T = uniform_field(SHAPE3, 300.0)

    got = run_adaptive(x, P, T, sf4=0.5, sf2P=0.5, sf2T=0.5)

    interior = [slice(None)] * 4
    interior[axis] = slice(2, n - 2)
    sl = tuple(interior)
    assert np.allclose(got[sl], x[sl], rtol=1e-3, atol=1e-2), (
        f"cubic not preserved in the interior along axis {axis}"
    )


def test_components_are_independent():
    """The operator acts componentwise: np separate fields, one shared sensor."""
    x = random_state(SHAPE, seed=7)
    P, T = varying_sensor_fields(SHAPE3)
    kw = dict(sf4=0.3, sf2P=0.5, sf2T=0.5)

    together = run_adaptive(x, P, T, **kw)
    for ip in range(SHAPE[3]):
        alone = run_adaptive(x[:, :, :, ip : ip + 1], P, T, **kw)
        assert np.allclose(together[:, :, :, ip], alone[:, :, :, 0], atol=1e-6), (
            f"component {ip} depends on the others"
        )


def test_sensor_localises_second_order_damping():
    """A sensor that fires only locally must leave the smooth region on 4th order.

    P is uniform except for a zig-zag confined to a slab of i. Outside that slab
    the result must match the globally-dead-sensor case, i.e. const(sf2=0, sf4).
    """
    x = random_state(SHAPE, seed=8)
    ni = SHAPE[0]
    P = uniform_field(SHAPE3, 1.0)
    P[3, :, :] = typ(1.5)
    T = uniform_field(SHAPE3, 300.0)
    sf4 = 0.3

    got = run_adaptive(x, P, T, sf4=sf4, sf2P=1.0, sf2T=0.0)
    pure4 = run_const(x, sf4=sf4, sf2=0.0)

    # The i-sensor is nonzero only at i in {2,3,4} (the 3-point curvature
    # stencils that straddle the spike), so its 5-point stencil can only reach
    # i in {0..6}. Planes beyond that must be untouched by the sensor.
    assert np.allclose(got[ni - 1, :, :, :], pure4[ni - 1, :, :, :], atol=1e-5), (
        "sensor leaked into the far-field plane"
    )
    assert not np.allclose(got[3, :, :, :], pure4[3, :, :, :], atol=1e-5), (
        "sensor did not fire at the spike"
    )
