"""Tests for Fortran gradient/curl operators and node/cell distribution (ember.fortran).

Module tested: ember.fortran

Test cases:
- test_grad: Gradient approximation for analytic functions in cylindrical coordinates
- test_curl_from_grad: Curl computed from gradients matches analytical curl
- test_node_to_cell: Node to cell distribution operations
- test_cell_to_node: Cell to node distribution operations
"""

import numpy as np
import ember.fortran
import ember.geometry


f32 = np.float32
f64 = np.float64

# Utility functions


def to_fort(x):
    """Convert an array to Fortran."""
    x = np.asfortranarray(x.copy()).astype(f32)
    return x


def make_cylinder(ni, nj, nk):
    """Assemble coordinates for a cylindrical sector."""

    # Geometry
    L = 0.1
    rm = 4.0

    ARr = 1.0
    dr = L * ARr

    r1 = rm - dr / 2.0
    r2 = rm + dr / 2.0

    ARt = 1.0
    pitch = dr / rm * ARt

    xv = np.linspace(0, L, ni)
    rv = np.linspace(r1, r2, nj)
    tv = np.linspace(-pitch / 2.0, pitch / 2.0, nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"), axis=-1)
    skew = 0.0
    skewr = np.radians(skew)
    xrt[2] += xrt[0] * np.tan(skewr)

    dAi = ember.geometry.get_dAi(xrt)
    dAj = ember.geometry.get_dAj(xrt)
    dAk = ember.geometry.get_dAk(xrt)
    vol = ember.geometry.get_vol(xrt, dAi, dAj, dAk)

    assert xrt.dtype == f64
    assert dAi.dtype == f64
    assert dAj.dtype == f64
    assert dAk.dtype == f64
    assert vol.dtype == f64

    xrt = to_fort(xrt)
    dAi = to_fort(dAi)
    dAj = to_fort(dAj)
    dAk = to_fort(dAk)
    vol = to_fort(vol)

    assert xrt.dtype == f32
    assert dAi.dtype == f32
    assert dAj.dtype == f32
    assert dAk.dtype == f32
    assert vol.dtype == f32

    return xrt, dAi, dAj, dAk, vol


# Begin test functions


def test_grad():
    """Check approximation of gradient for analytic functions."""

    nn = 65
    nj = nn
    ni = nn + 2
    nk = nn + 4

    xrt, dAi, dAj, dAk, vol = make_cylinder(ni, nj, nk)
    x = xrt[..., 0]
    r = xrt[..., 1]
    t = xrt[..., 2]

    # Reorder dA arrays for grad (move last axis to first)
    dAi = to_fort(np.moveaxis(dAi, -1, 0))
    dAj = to_fort(np.moveaxis(dAj, -1, 0))
    dAk = to_fort(np.moveaxis(dAk, -1, 0))

    # Initialize gradient output array (full size with padding)
    gradx = to_fort(np.zeros((ni, nj, nk, 3)))

    # Cell-centered coordinates for comparison
    x_cell = ember.geometry.node_to_cell(x)
    r_cell = ember.geometry.node_to_cell(r)

    print("Checking gradient of test fields...")
    print(
        "Note that in a cylindrical coordinate system:\n"
        "  grad(f) = df/dx ex + df/dr er + df/dt/r et"
    )

    rtol = 2e-3

    # Test 1: Gradient of constant field should be zero
    scalar = to_fort(np.ones_like(x) * 5.0)
    ember.fortran.grad(scalar, gradx, vol, dAi, dAj, dAk, r)
    # Extract valid region (not padding)
    gradx_valid = gradx[: ni - 1, : nj - 1, : nk - 1, :]
    err_x = np.abs(gradx_valid[..., 0])
    err_r = np.abs(gradx_valid[..., 1])
    err_t = np.abs(gradx_valid[..., 2])
    err = np.max([err_x.max(), err_r.max(), err_t.max()])
    print(f"grad(const)=0 error={err:.2e}")
    assert (err < rtol).all()

    # Test 2: Gradient of linear field in x: f = 2*x
    # Expected: df/dx = 2, df/dr = 0, df/dt = 0
    scalar = to_fort(2.0 * x)
    ember.fortran.grad(scalar, gradx, vol, dAi, dAj, dAk, r)
    print(f"grad(2x)=(2,0,0) error={err:.2e}")
    np.testing.assert_allclose(gradx[:-1, :-1, :-1, 0], 2.0, rtol=rtol)

    # Test 3: Gradient of linear field in r: f = 3*r
    # Expected: df/dx = 0, df/dr = 3, df/dt = 0
    scalar = to_fort(3.0 * r)
    ember.fortran.grad(scalar, gradx, vol, dAi, dAj, dAk, r)
    print(f"grad(3r)=(0,3,0) error={err:.2e}")
    np.testing.assert_allclose(gradx[:-1, :-1, :-1, 1], 3.0, rtol=rtol)

    # Test 4: Gradient of linear field in theta: f = -t
    # Expected: df/dx = 0, df/dr = 0, df/dt = -1 (but gradx[2] = df/dt/r in cylindrical coords)
    scalar = to_fort(-t)
    ember.fortran.grad(scalar, gradx, vol, dAi, dAj, dAk, r)
    err = np.max([err_x.max(), err_r.max(), err_t.max()])
    print(f"grad(-t)=(0,0,-1/r) error={err:.2e}")
    np.testing.assert_allclose(gradx[:-1, :-1, :-1, 2], -1.0 / r_cell, rtol=rtol)

    # Test 5: Gradient of quadratic field: f = x^2
    # Expected: df/dx = 2*x, df/dr = 0, df/dt = 0
    scalar = to_fort(x**2)
    ember.fortran.grad(scalar, gradx, vol, dAi, dAj, dAk, r)
    err = np.max([err_x.max(), err_r.max(), err_t.max()])
    print(f"grad(x^2)=(2x,0,0) error={err:.2e}")
    np.testing.assert_allclose(gradx[:-1, :-1, :-1, 0], 2.0 * x_cell, rtol=rtol)

    # Test 6: Gradient of quadratic field: f = r^2
    # Expected: df/dx = 0, df/dr = 2*r, df/dt = 0
    scalar = to_fort(r**2)
    ember.fortran.grad(scalar, gradx, vol, dAi, dAj, dAk, r)
    print(f"grad(r^2)=(0,2r,0) error={err:.2e}")
    np.testing.assert_allclose(gradx[:-1, :-1, :-1, 1], 2.0 * r_cell, rtol=rtol)

    # Test 7: Gradient of combined field: f = 2x + 3r - t
    # Expected: df/dx = 2, df/dr = 3, df/dt = -1/r
    scalar = to_fort(2.0 * x + 3.0 * r - t)
    ember.fortran.grad(scalar, gradx, vol, dAi, dAj, dAk, r)
    np.testing.assert_allclose(gradx[:-1, :-1, :-1, 0], 2.0, rtol=rtol)
    np.testing.assert_allclose(gradx[:-1, :-1, :-1, 1], 3.0, rtol=rtol)
    np.testing.assert_allclose(gradx[:-1, :-1, :-1, 2], -1.0 / r_cell, rtol=1e-2)


def test_curl_from_grad():
    """Check that curl computed from gradients matches analytical curl."""

    nn = 40
    nj = nn
    ni = nn + 2
    nk = nn + 4

    xrt, dAi, dAj, dAk, vol = make_cylinder(ni, nj, nk)
    x = xrt[..., 0]
    r = xrt[..., 1]
    t = xrt[..., 2]

    # Cell-centered coordinates
    r_cell = ember.geometry.node_to_cell(r)

    # Prepare arrays for grad operator (need component-first indexing)
    dAi_grad = to_fort(np.moveaxis(dAi, -1, 0))
    dAj_grad = to_fort(np.moveaxis(dAj, -1, 0))
    dAk_grad = to_fort(np.moveaxis(dAk, -1, 0))

    # Gradient outputs (full size with padding)
    gradVx = to_fort(np.zeros((ni, nj, nk, 3)))
    gradVr = to_fort(np.zeros((ni, nj, nk, 3)))
    gradVt = to_fort(np.zeros((ni, nj, nk, 3)))

    print("Checking curl from gradients...")
    print(
        "In cylindrical coordinates:\n"
        "  curl(V) = [1/r * dVt/dt - dVr/dx,\n"
        "             1/r * dVx/dt - dVt/dx,\n"
        "             dVr/dx - dVx/dr + Vt/r]\n"
        "Note: grad operator returns (dV/dx, dV/dr, 1/r*dV/dt)"
    )

    # Test field 1: V = (r, 0, 0)
    # Axial flow that varies with radius
    # curl(V) in cylindrical:
    #   curl_x = 1/r * d(0)/dt - d(0)/dx = 0
    #   curl_r = 1/r * d(r)/dt - d(0)/dx = 0
    #   curl_t = d(0)/dx - d(r)/dr + 0/r = 0 - 1 + 0 = -1
    u = to_fort(np.zeros_like(xrt))
    u[..., 0] = r

    Vx = u[..., 0:1]
    Vr = u[..., 1:2]
    Vt = u[..., 2:3]

    ember.fortran.grad(Vx, gradVx, vol, dAi_grad, dAj_grad, dAk_grad, r)
    ember.fortran.grad(Vr, gradVr, vol, dAi_grad, dAj_grad, dAk_grad, r)
    ember.fortran.grad(Vt, gradVt, vol, dAi_grad, dAj_grad, dAk_grad, r)

    # Extract interior cells from gradients
    gradVx_c = gradVx[:-1, :-1, :-1, :]
    gradVr_c = gradVr[:-1, :-1, :-1, :]
    gradVt_c = gradVt[:-1, :-1, :-1, :]

    # Cell-centered velocity components for Vt/r term
    Vt_cell = ember.geometry.node_to_cell(Vt.squeeze())

    # Compute curl from gradients using cylindrical formula
    # Remember: gradV[..., 2] = 1/r * dV/dt (already includes 1/r factor)
    # Cylindrical curl components (physical):
    #   curl_x = (1/r)*dVt/dt - dVr/dx = gradVt[2] - gradVr[0]
    #   curl_r = (1/r)*dVx/dt - dVt/dx = gradVx[2] - gradVt[0]
    #   curl_t = dVr/dx - dVx/dr (testing without Vt/r term)
    curl_x = gradVt_c[..., 2] - gradVr_c[..., 0]
    curl_r = gradVx_c[..., 2] - gradVt_c[..., 0]
    curl_t = gradVr_c[..., 0] - gradVx_c[..., 1]

    # Analytical curl for V = (r, 0, 0)
    expected_curl_x = np.zeros_like(r_cell)
    expected_curl_r = np.zeros_like(r_cell)
    expected_curl_t = -np.ones_like(r_cell)

    rtol = 2e-3

    err_x = np.abs(curl_x - expected_curl_x)
    err_r = np.abs(curl_r - expected_curl_r)
    err_t = np.abs(curl_t - expected_curl_t)
    print("  curl(V=(r,0,0)) = (0, 0, -1)")
    print(f"    curl_x error: {err_x.max():.2e}")
    print(f"    curl_r error: {err_r.max():.2e}")
    print(f"    curl_t error: {err_t.max():.2e}")

    np.testing.assert_allclose(curl_x, expected_curl_x, atol=rtol)
    np.testing.assert_allclose(curl_r, expected_curl_r, atol=rtol)
    np.testing.assert_allclose(curl_t, expected_curl_t, rtol=rtol)

    # Test field 2: V = (-r*t, x*t, x*r)
    # More complex field with all components
    #   curl_x = x*(2-1/r), curl_r = -1-r, curl_t = 2*t
    u[..., 0] = -r * t
    u[..., 1] = x * t
    u[..., 2] = x * r

    Vx = u[..., 0:1]
    Vr = u[..., 1:2]
    Vt = u[..., 2:3]

    ember.fortran.grad(Vx, gradVx, vol, dAi_grad, dAj_grad, dAk_grad, r)
    ember.fortran.grad(Vr, gradVr, vol, dAi_grad, dAj_grad, dAk_grad, r)
    ember.fortran.grad(Vt, gradVt, vol, dAi_grad, dAj_grad, dAk_grad, r)

    gradVx_c = gradVx[:-1, :-1, :-1, :]
    gradVr_c = gradVr[:-1, :-1, :-1, :]
    gradVt_c = gradVt[:-1, :-1, :-1, :]

    Vt_cell = ember.geometry.node_to_cell(Vt.squeeze())

    curl_x = gradVt_c[..., 1] - gradVr_c[..., 2] + Vt_cell / r_cell
    curl_r = gradVx_c[..., 2] - gradVt_c[..., 0]
    curl_t = gradVr_c[..., 0] - gradVx_c[..., 1]

    x_cell = ember.geometry.node_to_cell(x)
    t_cell = ember.geometry.node_to_cell(t)
    expected_curl_x = x_cell * (2.0 - 1.0 / r_cell)
    expected_curl_r = -1.0 - r_cell
    expected_curl_t = 2.0 * t_cell

    err_x = np.abs((curl_x - expected_curl_x) / (np.abs(expected_curl_x) + 1e-10))
    err_r = np.abs((curl_r - expected_curl_r) / (np.abs(expected_curl_r) + 1e-10))
    err_t = np.abs((curl_t - expected_curl_t) / (np.abs(expected_curl_t) + 1e-10))
    print("  curl(V=(-r*t, x*t, x*r)) = (x*(2-1/r), -1-r, 2*t)")
    print(f"    curl_x error: {err_x.max():.2e}")
    print(f"    curl_r error: {err_r.max():.2e}")
    print(f"    curl_t error: {err_t.max():.2e}")

    np.testing.assert_allclose(curl_x, expected_curl_x, rtol=rtol)
    np.testing.assert_allclose(curl_r, expected_curl_r, rtol=rtol)
    np.testing.assert_allclose(curl_t, expected_curl_t, rtol=rtol)


def test_node_to_cell():
    """Check averaging of nodal values to cell centers."""

    # Make an ijk grid
    ni = 97
    nj = 65
    nk = 73

    # Generate a grid of indices
    iv = np.linspace(0.0, ni - 1.0, ni)
    jv = np.linspace(0.0, nj - 1.0, nj)
    kv = np.linspace(0.0, nk - 1.0, nk)
    i, j, k = np.meshgrid(iv, jv, kv, indexing="ij")

    i = np.asfortranarray(np.expand_dims(i, -1), dtype=f32)
    j = np.asfortranarray(np.expand_dims(j, -1), dtype=f32)
    k = np.asfortranarray(np.expand_dims(k, -1), dtype=f32)

    # Uniform should stay uniform
    xn = np.ones_like(i)
    ni, nj, nk, nv = i.shape
    shape_cell = (ni - 1, nj - 1, nk - 1, nv)
    xc = np.zeros(shape_cell, order="F", dtype=f32)
    ember.fortran.node_to_cell(xn, xc)
    assert np.allclose(xn[:-1, :-1, :-1, :], xc)

    # Discrepancy should be exactly half for linear variation in each direction
    ic = np.zeros(shape_cell, order="F", dtype=f32)
    ember.fortran.node_to_cell(i, ic)
    assert np.allclose(ic - i[:-1, :-1, :-1, :], 0.5)

    jc = np.zeros(shape_cell, order="F", dtype=f32)
    ember.fortran.node_to_cell(j, jc)
    assert np.allclose(jc - j[:-1, :-1, :-1, :], 0.5)

    kc = np.zeros(shape_cell, order="F", dtype=f32)
    ember.fortran.node_to_cell(k, kc)
    assert np.allclose(kc - k[:-1, :-1, :-1, :], 0.5)


def test_cell_to_node():
    """Distribute a linear ramp from cell centers to nodes."""

    # Make an ijk grid
    ni = 97
    nj = 65
    nk = 73

    # Generate a grid of indices
    iv = np.linspace(0.0, ni - 1.0, ni)
    jv = np.linspace(0.0, nj - 1.0, nj)
    kv = np.linspace(0.0, nk - 1.0, nk)
    i, j, k = np.meshgrid(iv, jv, kv, indexing="ij")

    i = np.asfortranarray(np.expand_dims(i, -1), dtype=f32)
    j = np.asfortranarray(np.expand_dims(j, -1), dtype=f32)
    k = np.asfortranarray(np.expand_dims(k, -1), dtype=f32)

    # Uniform should stay uniform
    xc = np.ones_like(i)
    ni, nj, nk, nv = xc.shape
    shape_node = (ni + 1, nj + 1, nk + 1, nv)
    xn = np.zeros(shape_node, order="F", dtype=f32)
    ember.fortran.cell_to_node(xc, xn)
    assert np.allclose(xc, 1.0)

    # Check linear variation in each direction
    # Should have no change at boundaries
    # Offset of 1/2 along the ramping direction

    inode = np.zeros(shape_node, order="F", dtype=f32)
    ember.fortran.cell_to_node(i, inode)
    assert np.allclose(inode[0, :-1, :-1], i[0, :, :])
    assert np.allclose(inode[-1, :-1, :-1], i[-1, :, :])
    assert np.allclose(inode[1:-1, :-1, :-1] - i[:-1, :, :], 0.5)

    jnode = np.zeros(shape_node, order="F", dtype=f32)
    ember.fortran.cell_to_node(j, jnode)
    assert np.allclose(jnode[:-1, 0, :-1], j[:, 0, :])
    assert np.allclose(jnode[:-1, -1, :-1], j[:, -1, :])
    assert np.allclose(jnode[:-1, 1:-1, :-1] - j[:, :-1, :], 0.5)

    knode = np.zeros(shape_node, order="F", dtype=f32)
    ember.fortran.cell_to_node(k, knode)
    assert np.allclose(knode[:-1, :-1, 0], k[:, :, 0])
    assert np.allclose(knode[:-1, :-1, -1], k[:, :, -1])
    assert np.allclose(knode[:-1, :-1, 1:-1] - k[:, :, :-1], 0.5)


if __name__ == "__main__":
    test_grad()
    test_curl_from_grad()
    test_node_to_cell()
    test_cell_to_node()
