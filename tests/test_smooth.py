"""Test module for ember smoothing operations.

Tests smoothing functionality for numerical stability and solution quality.

Test cases:
- test_v2_zero: Zero smoothing factor should change nothing
- test_v2_const: Constant fields should be preserved
- test_v2_linear: Linear fields should be preserved
- test_v2_cubic: Cubic polynomial fields should be preserved
- test_v2_converge: Convergence properties of smoothing operators
"""

from ember import fortran
import numpy as np

np.random.seed(3)

typ = np.float32


# Utility functions


def to_fort(x):
    """Convert to Fortran-contiguous array."""
    return np.asfortranarray(x, dtype=typ)


def make_cfl_3d(shape, ndim=3):
    """Create CFL array of ones for 3D case."""
    ni, nj, nk, nvar = shape
    return to_fort(np.ones((ni - 1, nj - 1, nk - 1, 5)))


def make_ijk():
    """Assembly ijk 3D arrays."""
    ni = 10
    nj = 12
    nk = 8

    # Generate a grid of indices
    iv = np.linspace(0.0, ni - 1.0, ni)
    jv = np.linspace(0.0, nj - 1.0, nj)
    kv = np.linspace(0.0, nk - 1.0, nk)
    i, j, k = np.meshgrid(iv, jv, kv, indexing="ij")

    return i, j, k


def get_L_P(x):
    """Generate isotropic cell lengths and test pressure fields for 3D case."""
    # Equal weights in each direction
    shape = x.shape
    L = to_fort(np.ones(shape[:-1] + (3,)))
    # Uniform pressure for 4th-order only
    P4 = to_fort(np.ones(shape[:-1]))
    # Wobbly pressure for 2nd-order only
    P2 = to_fort(np.ones(shape[:-1]))
    P2[::2, ::2, ::2] = 2.0
    return L, P2, P4


def uniform_T(x):
    """Uniform temperature field — neutralises the T-sensor in tests that
    target the pressure-only behaviour."""
    return to_fort(np.ones(x.shape[:-1]))


def make_v2_work_arrays(shape):
    """Preallocate working arrays for smooth3d_v2."""
    ni, nj, nk, nvar = shape
    sf2n = np.zeros((ni, nj, nk, 3), order="F", dtype=typ)
    sf2n_t = np.zeros((ni, nj, nk, 3), order="F", dtype=typ)
    cfln = np.zeros((ni, nj, nk, 5), order="F", dtype=typ)
    dx = np.zeros((ni, nj, nk), order="F", dtype=typ)
    return sf2n, sf2n_t, cfln, dx


# ============================================================
# Tests for smooth3d_v2
# ============================================================


def test_v2_zero():
    """Zero smoothing factor should change nothing."""
    shape = (5, 6, 7, 3)
    X = np.random.random_sample(shape)
    Xs = to_fort(X)
    L, P2, P4 = get_L_P(Xs)
    T = uniform_T(Xs)
    cfl = make_cfl_3d(shape)
    sf2n, sf2n_t, cfln, dx = make_v2_work_arrays(shape)

    fortran.smooth3d(
        Xs,
        P2,
        T,
        L,
        cfl,
        sf2p=0.0,
        sf2t=0.0,
        sf4=0.0,
        sf2min=0.0,
        sf2n=sf2n,
        sf2n_t=sf2n_t,
        cfln=cfln,
        dx=dx,
    )
    assert np.allclose(X, Xs), "Field changed with zero smoothing (P2)"

    Xs = to_fort(X)
    fortran.smooth3d(
        Xs,
        P4,
        T,
        L,
        cfl,
        sf2p=0.0,
        sf2t=0.0,
        sf4=0.0,
        sf2min=0.0,
        sf2n=sf2n,
        sf2n_t=sf2n_t,
        cfln=cfln,
        dx=dx,
    )
    assert np.allclose(X, Xs), "Field changed with zero smoothing (P4)"


def test_v2_const():
    """A constant value should stay constant after smoothing."""
    for sf2 in (0.1, 0.2):
        for sf4 in (0.1, 0.2):
            X = np.ones((10, 15, 20, 1), order="F", dtype=typ)
            L, P2, P4 = get_L_P(X)
            T = uniform_T(X)
            cfl = make_cfl_3d(X.shape)
            sf2n, sf2n_t, cfln, dx = make_v2_work_arrays(X.shape)

            fortran.smooth3d(
                X,
                P2,
                T,
                L,
                cfl,
                sf2p=sf2,
                sf2t=0.0,
                sf4=sf4,
                sf2min=0.0,
                sf2n=sf2n,
                sf2n_t=sf2n_t,
                cfln=cfln,
                dx=dx,
            )
            assert np.allclose(X, 1.0), (
                f"Constant not preserved for sf2={sf2}, sf4={sf4}"
            )
            assert not np.isnan(X).any()


def test_v2_linear():
    """Smoothing a linear function should introduce no error in the interior.

    Boundary nodes use the nearest interior value as target rather than a
    linear extrapolation, so they are excluded from the check.
    """
    i, j, k = make_ijk()
    f = i + 2.0 * j - 2.0 * (k - 5) + 1.0
    f = np.expand_dims(f, -1)

    fs = to_fort(f.copy())
    L, P2, P4 = get_L_P(fs)
    T = uniform_T(fs)
    cfl = make_cfl_3d(fs.shape)
    sf2n, sf2n_t, cfln, dx = make_v2_work_arrays(fs.shape)

    interior = np.s_[1:-1, 1:-1, 1:-1, :]

    # 2nd-order only
    fortran.smooth3d(
        fs,
        P2,
        T,
        L,
        cfl,
        sf2p=0.1,
        sf2t=0.0,
        sf4=0.0,
        sf2min=0.0,
        sf2n=sf2n,
        sf2n_t=sf2n_t,
        cfln=cfln,
        dx=dx,
    )
    assert np.allclose(f[interior], fs[interior]), (
        "Linear function not preserved with 2nd-order"
    )

    fs = to_fort(f.copy())
    fortran.smooth3d(
        fs,
        P4,
        T,
        L,
        cfl,
        sf2p=0.0,
        sf2t=0.0,
        sf4=0.1,
        sf2min=0.0,
        sf2n=sf2n,
        sf2n_t=sf2n_t,
        cfln=cfln,
        dx=dx,
    )
    assert np.allclose(f[interior], fs[interior]), (
        "Linear function not preserved with 4th-order"
    )


def test_v2_cubic():
    """Fourth-order smoothing a cubic function should introduce no error."""
    i, j, k = make_ijk()

    # Define a cubic test function
    f = 2.0 * i**3 + (j**2 - 2.0 * j) - (k - 5) ** 3 + 1.0
    f = np.expand_dims(f, -1)

    fs = to_fort(f.copy())
    L, P2, P4 = get_L_P(fs)
    T = uniform_T(fs)
    cfl = make_cfl_3d(fs.shape)
    sf2n, sf2n_t, cfln, dx = make_v2_work_arrays(fs.shape)

    fortran.smooth3d(
        fs,
        P4,
        T,
        L,
        cfl,
        sf2p=0.0,
        sf2t=0.0,
        sf4=0.1,
        sf2min=0.0,
        sf2n=sf2n,
        sf2n_t=sf2n_t,
        cfln=cfln,
        dx=dx,
    )
    # Exclude boundaries where stencils may be asymmetric
    assert np.allclose(f[2:-2, 2:-2, 2:-2, :], fs[2:-2, 2:-2, 2:-2, :]), (
        "Cubic preservation failed in interior"
    )


# ============================================================
# Tests for smooth3d_hmesh (k-periodic H-mesh variant)
# ============================================================


def test_hmesh_k_continuous():
    """Wrapped k-smoothing is continuous across the k=1/k=nk seam.

    On a block fully periodic to self in k, the smoother must commute with a
    roll in k (translation invariance). A seam at k=1/k=nk would break this.
    """
    ni, nj, nk = 9, 7, 8
    per = nk - 1  # reduced periodic k-domain (k=nk duplicates k=1)
    rng = np.random.default_rng(0)

    def make_full(reduced):
        # Append the duplicate plane so F[..., nk-1] == F[..., 0].
        return to_fort(np.concatenate([reduced, reduced[:, :, :1]], axis=2))

    # k-periodic X (3 comps) and P; T uniform. Roll X and P together.
    X0r = rng.random((ni, nj, per, 3)).astype(typ)
    P0r = (1.0 + 0.3 * rng.random((ni, nj, per))).astype(typ)
    Xrr = np.roll(X0r, 1, axis=2)
    Prr = np.roll(P0r, 1, axis=2)

    def run(Xred, Pred):
        X = make_full(Xred)
        P = make_full(Pred)
        T = to_fort(np.ones((ni, nj, nk)))
        L = to_fort(np.ones((ni, nj, nk, 3)))
        cfl = to_fort(np.ones((ni - 1, nj - 1, nk - 1, 5)))
        sf2n, sf2n_t, cfln, dx = make_v2_work_arrays((ni, nj, nk, 3))
        fortran.smooth3d_hmesh(
            X,
            P,
            T,
            L,
            cfl,
            sf2p=0.2,
            sf2t=0.0,
            sf4=0.1,
            sf2min=0.01,
            sf2n=sf2n,
            sf2n_t=sf2n_t,
            cfln=cfln,
            dx=dx,
            i_le=ni,  # whole block periodic in k
            i_te=1,
        )
        return X

    S0 = run(X0r, P0r)
    Sr = run(Xrr, Prr)

    # (A) continuity preserved: duplicate plane stays equal after smoothing
    assert np.allclose(S0[:, :, 0], S0[:, :, nk - 1], atol=1e-5), (
        "k=1/k=nk planes diverged -> seam in smoothing"
    )
    # (B) translation invariance on the reduced domain -> no seam
    assert np.allclose(Sr[:, :, :per], np.roll(S0[:, :, :per], 1, axis=2), atol=1e-5), (
        "smoother does not commute with a k-roll -> seam at k=1/k=nk"
    )


def test_hmesh_k_continuous_mixed():
    """Translation invariance with all three i-segments non-empty.

    A mixed block exercises upstream-wrapped, bladed-clipped, and
    downstream-wrapped segments at once. The smoother commutes with a k-roll
    only in the periodic i-regions (i<=i_le and i>=i_te); the bladed region
    uses clipped k-stencils and is deliberately not roll-invariant. Each node's
    update depends only on the input field within one sweep, so the bladed
    treatment does not leak into the periodic columns -> the periodic-region
    check is valid even with a clipped region present.
    """
    ni, nj, nk = 14, 7, 8
    i_le, i_te = 4, 11  # periodic 1-based i=1..4 and i=11..14; bladed i=5..10
    per = nk - 1
    rng = np.random.default_rng(2)

    def make_full(reduced):
        return to_fort(np.concatenate([reduced, reduced[:, :, :1]], axis=2))

    X0r = rng.random((ni, nj, per, 3)).astype(typ)
    P0r = (1.0 + 0.3 * rng.random((ni, nj, per))).astype(typ)
    Xrr = np.roll(X0r, 1, axis=2)
    Prr = np.roll(P0r, 1, axis=2)

    def run(Xred, Pred):
        X = make_full(Xred)
        P = make_full(Pred)
        T = to_fort(np.ones((ni, nj, nk)))
        L = to_fort(np.ones((ni, nj, nk, 3)))
        cfl = to_fort(np.ones((ni - 1, nj - 1, nk - 1, 5)))
        sf2n, sf2n_t, cfln, dx = make_v2_work_arrays((ni, nj, nk, 3))
        fortran.smooth3d_hmesh(
            X,
            P,
            T,
            L,
            cfl,
            sf2p=0.2,
            sf2t=0.0,
            sf4=0.1,
            sf2min=0.01,
            sf2n=sf2n,
            sf2n_t=sf2n_t,
            cfln=cfln,
            dx=dx,
            i_le=i_le,
            i_te=i_te,
        )
        return X

    S0 = run(X0r, P0r)
    Sr = run(Xrr, Prr)

    def rolled_match(region):
        return np.allclose(
            Sr[region][:, :, :per],
            np.roll(S0[region][:, :, :per], 1, axis=2),
            atol=1e-5,
        )

    # Periodic regions: must be k-roll invariant (no seam).
    assert rolled_match(np.s_[:i_le]), "upstream periodic region not k-roll invariant"
    assert rolled_match(np.s_[i_te - 1 :]), (
        "downstream periodic region not k-roll invariant"
    )
    # Bladed region: clipped k-stencils -> must NOT be roll invariant, else the
    # test is not actually distinguishing the wrapped path from the clipped one.
    assert not rolled_match(np.s_[i_le : i_te - 1]), (
        "bladed region unexpectedly k-roll invariant (clipped path not exercised)"
    )


def test_hmesh_bladed_matches_smooth3d():
    """In the bladed (non-periodic) i-region, smooth3d_hmesh equals smooth3d."""
    ni, nj, nk = 12, 7, 8
    i_le, i_te = 3, 9  # periodic: i<=3 and i>=9; bladed (1-based) i=4..8
    rng = np.random.default_rng(1)

    X = to_fort(rng.random((ni, nj, nk, 3)).astype(typ))
    P = to_fort((1.0 + 0.3 * rng.random((ni, nj, nk))).astype(typ))
    T = to_fort(np.ones((ni, nj, nk)))
    L = to_fort(np.ones((ni, nj, nk, 3)))
    cfl = to_fort(np.ones((ni - 1, nj - 1, nk - 1, 5)))

    kw = dict(sf2p=0.2, sf2t=0.0, sf4=0.1, sf2min=0.01)

    Xa = to_fort(X.copy())
    sf2n, sf2n_t, cfln, dx = make_v2_work_arrays((ni, nj, nk, 3))
    fortran.smooth3d(Xa, P, T, L, cfl, sf2n=sf2n, sf2n_t=sf2n_t, cfln=cfln, dx=dx, **kw)

    Xb = to_fort(X.copy())
    sf2n, sf2n_t, cfln, dx = make_v2_work_arrays((ni, nj, nk, 3))
    fortran.smooth3d_hmesh(
        Xb,
        P,
        T,
        L,
        cfl,
        sf2n=sf2n,
        sf2n_t=sf2n_t,
        cfln=cfln,
        dx=dx,
        i_le=i_le,
        i_te=i_te,
        **kw,
    )

    # Bladed region (1-based i=i_le+1..i_te-1 -> 0-based i_le..i_te-2).
    bladed = np.s_[i_le : i_te - 1, :, :, :]
    assert np.allclose(Xa[bladed], Xb[bladed], atol=1e-6), (
        "bladed-region result diverged from generic smooth3d"
    )
