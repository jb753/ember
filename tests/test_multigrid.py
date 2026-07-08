"""Tests for multigrid operations."""

import numpy as np

import ember.fortran


def test_prolongate_correction_constant():
    """Test that prolongate_correction preserves constant corrections exactly.

    Prolongation is correction-based: if we add a constant to coarse grid values,
    the same constant should be added to all fine grid values.
    """
    # Small grid for testing
    ni_coarse, nj_coarse, nk_coarse = 3, 3, 3
    ni_fine, nj_fine, nk_fine = 5, 5, 5
    ncomp = 5

    # Create random initial data on fine grid
    np.random.seed(42)
    fine = np.random.randn(ni_fine, nj_fine, nk_fine, ncomp).astype(np.float32)
    fine = np.asfortranarray(fine)

    # Build constant correction on coarse grid
    constants = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    correction = np.zeros(
        (ni_coarse, nj_coarse, nk_coarse, ncomp), dtype=np.float32, order="F"
    )
    for n in range(ncomp):
        correction[:, :, :, n] = constants[n]

    # Save original fine grid to compute actual correction applied
    fine_original = fine.copy(order="F")

    # Call Fortran prolongation routine
    ember.fortran.prolongate_correction(
        correction,
        fine,
        1.0,
        ni_coarse,
        nj_coarse,
        nk_coarse,
        ni_fine,
        nj_fine,
        nk_fine,
        ncomp,
    )

    # Compute actual correction applied
    correction_applied = fine - fine_original

    # Check all nodes - with trilinear interpolation, ALL nodes should get constant correction
    print("\nChecking ALL nodes:")
    for n in range(ncomp):
        actual = correction_applied[:, :, :, n]
        expected = constants[n]
        print(
            f"  Component {n}: expected {expected}, got mean={actual.mean():.6f}, std={actual.std():.6f}, min={actual.min():.6f}, max={actual.max():.6f}"
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"All nodes should get correction {expected} for component {n}",
        )


def test_prolongate_correction_linear():
    """Test that prolongate_correction preserves linear corrections exactly.

    Linear functions should be reproduced exactly by trilinear interpolation.
    """
    # Grid for testing (must satisfy: ni_fine = 2*ni_coarse - 1 for 2:1 refinement)
    ni_coarse, nj_coarse, nk_coarse = 5, 5, 5
    ni_fine, nj_fine, nk_fine = 9, 9, 9
    ncomp = 5

    # Create random initial data on fine grid
    np.random.seed(42)
    fine = np.random.randn(ni_fine, nj_fine, nk_fine, ncomp).astype(np.float32)
    fine = np.asfortranarray(fine)

    # Create coordinate grids that match the 2:1 refinement pattern
    # For 2:1 refinement: fine index if = 2*ic - 1 (Fortran indexing)
    # So fine coordinate = (if + 1) / 2 in coarse grid spacing
    i_coarse = np.arange(1, ni_coarse + 1, dtype=np.float32)
    j_coarse = np.arange(1, nj_coarse + 1, dtype=np.float32)
    k_coarse = np.arange(1, nk_coarse + 1, dtype=np.float32)

    i_fine_fortran = np.arange(1, ni_fine + 1, dtype=np.float32)
    j_fine_fortran = np.arange(1, nj_fine + 1, dtype=np.float32)
    k_fine_fortran = np.arange(1, nk_fine + 1, dtype=np.float32)

    # Convert to coordinates (in coarse grid spacing)
    i_fine = (i_fine_fortran + 1) / 2.0
    j_fine = (j_fine_fortran + 1) / 2.0
    k_fine = (k_fine_fortran + 1) / 2.0

    I_coarse, J_coarse, K_coarse = np.meshgrid(
        i_coarse, j_coarse, k_coarse, indexing="ij"
    )
    I_fine, J_fine, K_fine = np.meshgrid(i_fine, j_fine, k_fine, indexing="ij")

    # Linear correction coefficients (different for each component)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    c = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

    # Build linear correction on coarse grid
    correction = np.zeros(
        (ni_coarse, nj_coarse, nk_coarse, ncomp), dtype=np.float32, order="F"
    )
    for n in range(ncomp):
        correction[:, :, :, n] = (
            a[n] + b[n] * I_coarse + c[n] * J_coarse + d[n] * K_coarse
        )

    # Expected correction on fine grid
    correction_expected = np.zeros(
        (ni_fine, nj_fine, nk_fine, ncomp), dtype=np.float32, order="F"
    )
    for n in range(ncomp):
        correction_expected[:, :, :, n] = (
            a[n] + b[n] * I_fine + c[n] * J_fine + d[n] * K_fine
        )

    # Save original fine grid
    fine_original = fine.copy(order="F")

    # Call Fortran prolongation routine
    ember.fortran.prolongate_correction(
        correction,
        fine,
        1.0,
        ni_coarse,
        nj_coarse,
        nk_coarse,
        ni_fine,
        nj_fine,
        nk_fine,
        ncomp,
    )

    # Compute actual correction applied
    correction_applied = fine - fine_original

    # Linear corrections should be reproduced exactly (within floating point precision)
    print("\nChecking linear correction interpolation:")
    for n in range(ncomp):
        actual = correction_applied[:, :, :, n]
        expected = correction_expected[:, :, :, n]
        max_error = np.abs(actual - expected).max()
        print(f"  Component {n}: max error = {max_error:.2e}")
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Linear correction should be preserved for component {n}",
        )


def test_restrict_node():
    """Test that restrict_node samples every other node correctly.

    Restriction should simply copy every other node from fine to coarse,
    equivalent to coarse = fine[::2, ::2, ::2, :].
    """
    # Grid for testing (ni_fine = 2*ni_coarse - 1 for 2:1 refinement)
    ni_fine, nj_fine, nk_fine = 9, 9, 9
    ni_coarse, nj_coarse, nk_coarse = 5, 5, 5
    ncomp = 5

    # Create random data on fine grid
    np.random.seed(42)
    fine = np.random.randn(ni_fine, nj_fine, nk_fine, ncomp).astype(np.float32)
    fine = np.asfortranarray(fine)

    # Create coarse grid array
    coarse = np.zeros(
        (ni_coarse, nj_coarse, nk_coarse, ncomp), dtype=np.float32, order="F"
    )

    # Call Fortran restriction routine
    ember.fortran.restrict_node(
        fine, coarse, ni_fine, nj_fine, nk_fine, ni_coarse, nj_coarse, nk_coarse, ncomp
    )

    # Expected result: sample every other node
    expected = fine[::2, ::2, ::2, :].copy(order="F")

    # Verify the restriction is correct
    np.testing.assert_allclose(
        coarse,
        expected,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Restriction should sample every other node",
    )


def test_prolongate_correction_asymmetric():
    """Test prolongate_correction with asymmetric grid dimensions.

    This test verifies the interpolation works correctly when grid dimensions
    differ in each direction (e.g., 5x7x9 coarse -> 9x13x17 fine).
    """
    # Asymmetric grid for testing
    ni_coarse, nj_coarse, nk_coarse = 5, 7, 9
    ni_fine, nj_fine, nk_fine = 9, 13, 17
    ncomp = 5

    # Create random initial data on fine grid
    np.random.seed(42)
    fine = np.random.randn(ni_fine, nj_fine, nk_fine, ncomp).astype(np.float32)
    fine = np.asfortranarray(fine)

    # Build constant correction on coarse grid
    constants = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    correction = np.zeros(
        (ni_coarse, nj_coarse, nk_coarse, ncomp), dtype=np.float32, order="F"
    )
    for n in range(ncomp):
        correction[:, :, :, n] = constants[n]

    # Save original fine grid
    fine_original = fine.copy(order="F")

    # Call Fortran prolongation routine
    ember.fortran.prolongate_correction(
        correction,
        fine,
        1.0,
        ni_coarse,
        nj_coarse,
        nk_coarse,
        ni_fine,
        nj_fine,
        nk_fine,
        ncomp,
    )

    # Compute actual correction applied
    correction_applied = fine - fine_original

    # Check all nodes get constant correction
    print("\nChecking ALL nodes with asymmetric grid:")
    for n in range(ncomp):
        actual = correction_applied[:, :, :, n]
        expected = constants[n]
        print(
            f"  Component {n}: expected {expected}, got mean={actual.mean():.6f}, std={actual.std():.6f}"
        )
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"All nodes should get correction {expected} for component {n}",
        )


def test_prolongate_correction_boundary_exact():
    """Test that boundary fine nodes get exact coarse values (no interpolation).

    Fine grid boundary nodes that coincide with coarse grid nodes should
    receive exactly the coarse correction value, not an interpolated value.
    """
    # Asymmetric grid for testing
    ni_coarse, nj_coarse, nk_coarse = 4, 5, 6
    ni_fine, nj_fine, nk_fine = 7, 9, 11
    ncomp = 3

    # Create zeros on fine grid
    fine = np.zeros((ni_fine, nj_fine, nk_fine, ncomp), dtype=np.float32, order="F")

    # Build correction with distinct values at corners
    correction = np.zeros(
        (ni_coarse, nj_coarse, nk_coarse, ncomp), dtype=np.float32, order="F"
    )
    corner_value = 10.0
    correction[0, 0, 0, :] = corner_value
    correction[-1, 0, 0, :] = corner_value * 2
    correction[0, -1, 0, :] = corner_value * 3
    correction[0, 0, -1, :] = corner_value * 4

    # Call Fortran prolongation routine
    ember.fortran.prolongate_correction(
        correction,
        fine,
        1.0,
        ni_coarse,
        nj_coarse,
        nk_coarse,
        ni_fine,
        nj_fine,
        nk_fine,
        ncomp,
    )

    # Check corners get exact values (no interpolation)
    np.testing.assert_allclose(
        fine[0, 0, 0, :],
        corner_value,
        rtol=1e-6,
        err_msg="Corner (0,0,0) should get exact coarse value",
    )
    np.testing.assert_allclose(
        fine[-1, 0, 0, :],
        corner_value * 2,
        rtol=1e-6,
        err_msg="Corner (-1,0,0) should get exact coarse value",
    )
    np.testing.assert_allclose(
        fine[0, -1, 0, :],
        corner_value * 3,
        rtol=1e-6,
        err_msg="Corner (0,-1,0) should get exact coarse value",
    )
    np.testing.assert_allclose(
        fine[0, 0, -1, :],
        corner_value * 4,
        rtol=1e-6,
        err_msg="Corner (0,0,-1) should get exact coarse value",
    )


def test_restrict_cell():
    """Test that restrict_cell sums 2x2x2 blocks of fine cells.

    Each coarse cell should equal the sum of the 8 fine cells it contains.
    """
    # Grid for testing (cells = nodes - 1)
    ni_fine, nj_fine, nk_fine = 9, 9, 9  # 8x8x8 cells
    ni_coarse, nj_coarse, nk_coarse = 5, 5, 5  # 4x4x4 cells
    ncomp = 5

    # Create random cell data on fine grid in Fortran order: (ni-1, nj-1, nk-1, ncomp)
    np.random.seed(42)
    fine = np.asfortranarray(
        np.random.randn(ni_fine - 1, nj_fine - 1, nk_fine - 1, ncomp).astype(np.float32)
    )

    # Create coarse grid array in Fortran order: (ni-1, nj-1, nk-1, ncomp)
    coarse = np.zeros(
        (ni_coarse - 1, nj_coarse - 1, nk_coarse - 1, ncomp),
        dtype=np.float32,
        order="F",
    )

    # Call Fortran restriction routine (pass node counts, not cell counts)
    ember.fortran.restrict_cell(
        fine, coarse, ni_fine, nj_fine, nk_fine, ni_coarse, nj_coarse, nk_coarse, ncomp
    )

    # Verify that each coarse cell equals sum of 2x2x2 fine cells
    for ic in range(ni_coarse - 1):
        for jc in range(nj_coarse - 1):
            for kc in range(nk_coarse - 1):
                # Fine cell indices (Python 0-based)
                if_start = 2 * ic
                jf_start = 2 * jc
                kf_start = 2 * kc

                # Sum 2x2x2 block of fine cells
                # Array is (ni-1, nj-1, nk-1, ncomp) so sum over spatial axes
                expected = np.sum(
                    fine[
                        if_start : if_start + 2,
                        jf_start : jf_start + 2,
                        kf_start : kf_start + 2,
                        :,
                    ],
                    axis=(0, 1, 2),
                )

                # Check coarse cell matches
                np.testing.assert_allclose(
                    coarse[ic, jc, kc, :],
                    expected,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"Coarse cell ({ic},{jc},{kc}) should equal sum of fine cells",
                )
