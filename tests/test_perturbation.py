"""Tests for perturbation analysis matrices.

Test cases:
- test_conserved_to_primitive_linearization: Conserved-to-primitive transformation linearization
- test_matrices_are_inverses: Matrix inverse relationship verification
- test_primitive_chic_matrices_are_inverses: Primitive-chic matrix inverse verification
- test_primitive_flux_matrices_are_inverses: Primitive-flux matrix inverse verification
- test_primitive_bcond_matrices_are_inverses: Primitive-bcond matrix inverse verification
- test_primitive_to_flux_linearization: Primitive-to-flux transformation linearization
- test_primitive_to_chic_linearization: Primitive-to-chic transformation linearization
- test_flux_conserved_matrices_consistency: Flux-conserved matrix consistency
- test_primitive_to_bcond_linearization: Primitive-to-bcond transformation linearization
- test_matrix_uniformity_all_dimensions: Matrix uniformity across dimensions
"""

import numpy as np
import pytest
import ember.block
import ember.fluid
from ember import util, perturbation


@pytest.fixture
def scalar_blocks():
    """Create two scalar blocks with small perturbation between them."""
    fluid = ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-4, Pr=0.8)

    # Base state
    block1 = ember.block.Block(shape=())
    block1.set_x(np.array([0.05]))
    block1.set_r(np.array([0.85]))
    block1.set_t(np.array([0.05]))
    block1.set_fluid(fluid)
    block1.set_Vx(100.0)
    block1.set_Vr(80.0)
    block1.set_Vt(50.0)
    block1.set_P_rho(1.2e5, 295.0)

    # Small perturbation to conserved variables
    eps = 1e-3
    dcons = util.get_atol(block1.conserved, block1.r.mean(), rtol=eps)

    # Perturbed state
    block2 = block1.copy()
    block2.set_conserved(block1.conserved + dcons)

    # Set appropriate tolerance for primitive variables
    prim1 = np.stack((block1.rho, block1.Vx, block1.Vr, block1.Vt, block1.P), axis=-1)
    prim2 = np.stack((block2.rho, block2.Vx, block2.Vr, block2.Vt, block2.P), axis=-1)
    atol = prim1 - prim2 * 1e-2

    return block1, block2, atol


def test_conserved_to_primitive_linearization(scalar_blocks):
    """Test that conserved_to_primitive matrix correctly linearizes the transformation."""
    block1, block2, atol = scalar_blocks

    # Calculate finite difference in conserved variables
    dcons_actual = block2.conserved - block1.conserved

    # Calculate finite difference in primitive variables
    dprim_actual = np.array(
        [
            block2.rho - block1.rho,
            block2.Vx - block1.Vx,
            block2.Vr - block1.Vr,
            block2.Vt - block1.Vt,
            block2.P - block1.P,
        ]
    )

    # Use linearization matrix to predict primitive delta
    c2p = perturbation.conserved_to_primitive(block1)
    dprim_calc = util.matvec(c2p, dcons_actual)

    err_rel = np.abs(dprim_calc - dprim_actual) / atol
    assert (err_rel < 1.0).all(), f"Relative error too large: {err_rel}"


def test_matrices_are_inverses(scalar_blocks):
    """Test that primitive_to_conserved and conserved_to_primitive are inverses."""
    block1 = scalar_blocks[0]

    p2c = perturbation.primitive_to_conserved(block1)
    c2p = perturbation.conserved_to_primitive(block1)

    # Test both directions
    identity1 = util.matmat(p2c, c2p)
    identity2 = util.matmat(c2p, p2c)

    np.testing.assert_allclose(identity1, np.eye(5), atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(identity2, np.eye(5), atol=1e-2, rtol=0.0)


def test_primitive_chic_matrices_are_inverses(scalar_blocks):
    """Test that primitive_to_chic and chic_to_primitive are inverses."""
    block1 = scalar_blocks[0]

    p2chic = perturbation.primitive_to_chic(block1)
    chic2p = perturbation.chic_to_primitive(block1)

    # Test both directions
    identity1 = util.matmat(p2chic, chic2p)
    identity2 = util.matmat(chic2p, p2chic)

    np.testing.assert_allclose(identity1, np.eye(5), atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(identity2, np.eye(5), atol=1e-5, rtol=0.0)


def test_primitive_flux_matrices_are_inverses(scalar_blocks):
    """Test that primitive_to_flux and flux_to_primitive are inverses."""
    block1 = scalar_blocks[0]

    p2flux = perturbation.primitive_to_flux(block1)
    flux2p = perturbation.flux_to_primitive(block1)

    # Test both directions
    identity1 = util.matmat(p2flux, flux2p)
    identity2 = util.matmat(flux2p, p2flux)

    np.testing.assert_allclose(identity2, np.eye(5), atol=0.02, rtol=0.0)
    np.testing.assert_allclose(identity1, np.eye(5), atol=0.1, rtol=0.0)


def test_primitive_bcond_matrices_are_inverses(scalar_blocks):
    """Test that primitive_to_bcond and bcond_to_primitive are inverses."""
    block1 = scalar_blocks[0]

    p2bcond = perturbation.primitive_to_bcond(block1)
    bcond2p = perturbation.bcond_to_primitive(block1)

    # Test both directions
    identity1 = util.matmat(p2bcond, bcond2p)
    identity2 = util.matmat(bcond2p, p2bcond)

    np.testing.assert_allclose(identity1, np.eye(5), atol=1e-2, rtol=0.0)
    np.testing.assert_allclose(identity2, np.eye(5), atol=1e-2, rtol=0.0)


def test_primitive_to_flux_linearization(scalar_blocks):
    """Test that primitive_to_flux matrix correctly linearizes the transformation."""
    block1, block2, atol = scalar_blocks

    # Calculate finite difference in primitive variables
    dprim_actual = np.array(
        [
            block2.rho - block1.rho,
            block2.Vx - block1.Vx,
            block2.Vr - block1.Vr,
            block2.Vt - block1.Vt,
            block2.P - block1.P,
        ]
    )

    # Calculate finite difference in flux variables [rhoVx, rhoVx^2+P, rhoVxVr, rhoVxrVt, rhoVx*ho]
    # Based on the matrix definition in primitive_to_flux
    flux1 = np.array(
        [
            block1.rhoVx,
            block1.Vx**2 * block1.rho + block1.P,  # rhoVx^2 + P = Vx^2 * rho + P
            block1.Vx * block1.Vr * block1.rho,  # rhoVxVr
            block1.Vx * block1.r * block1.Vt * block1.rho,  # rhoVxrVt
            block1.rhoVx * block1.ho,
        ]
    )
    flux2 = np.array(
        [
            block2.rhoVx,
            block2.Vx**2 * block2.rho + block2.P,
            block2.Vx * block2.Vr * block2.rho,
            block2.Vx * block2.r * block2.Vt * block2.rho,
            block2.rhoVx * block2.ho,
        ]
    )
    dflux_actual = flux2 - flux1

    flux2p = perturbation.flux_to_primitive(block1)
    dprim_calc = util.matvec(flux2p, dflux_actual)

    err_rel = np.abs(dprim_calc - dprim_actual) / atol
    assert (err_rel < 1.0).all(), f"Relative error too large: {err_rel}"


def test_primitive_to_chic_linearization(scalar_blocks):
    """Test that primitive_to_chic matrix correctly linearizes the transformation."""
    block1, block2, atol = scalar_blocks

    # Calculate finite difference in primitive variables
    dprim_actual = np.array(
        [
            block2.rho - block1.rho,
            block2.Vx - block1.Vx,
            block2.Vr - block1.Vr,
            block2.Vt - block1.Vt,
            block2.P - block1.P,
        ]
    )

    # Calculate finite difference in characteristic variables
    # Characteristic variables are typically acoustic waves: [entropy, upstream acoustic, radial acoustic, tangential acoustic, downstream acoustic]
    rhoa1 = block1.rho * block1.a
    rhoa2 = block2.rho * block2.a

    chic1 = np.array(
        [
            -block1.Vx * rhoa1 + block1.P,  # upstream acoustic
            block1.Vx * rhoa1 + block1.P,  # downstream acoustic
            block1.Vr * rhoa1,  # radial acoustic
            block1.Vt * rhoa1,  # tangential acoustic
            block1.P - block1.a**2 * block1.rho,  # entropy
        ]
    )
    chic2 = np.array(
        [
            -block2.Vx * rhoa2 + block2.P,
            block2.Vx * rhoa2 + block2.P,
            block2.Vr * rhoa2,
            block2.Vt * rhoa2,
            block2.P - block2.a**2 * block2.rho,
        ]
    )
    dchic_actual = chic2 - chic1

    chic2p = perturbation.chic_to_primitive(block1)
    dprim_calc = util.matvec(chic2p, dchic_actual)
    err_rel = np.abs(dprim_calc - dprim_actual) / atol
    assert (err_rel < 1.0).all(), f"Relative error too large: {err_rel}"


def test_flux_conserved_matrices_consistency(scalar_blocks):
    """Test that new flux_to_conserved and conserved_to_flux are consistent with chained transformations."""
    block1 = scalar_blocks[0]

    # Test flux_to_conserved consistency
    p2c = perturbation.primitive_to_conserved(block1)
    f2p = perturbation.flux_to_primitive(block1)
    f2c_direct = perturbation.flux_to_conserved(block1)
    f2c_chained = util.matmat(p2c, f2p)

    np.testing.assert_allclose(f2c_direct, f2c_chained, atol=1e-12, rtol=0.0)

    # Test conserved_to_flux consistency
    p2f = perturbation.primitive_to_flux(block1)
    c2p = perturbation.conserved_to_primitive(block1)
    c2f_direct = perturbation.conserved_to_flux(block1)
    c2f_chained = util.matmat(p2f, c2p)

    np.testing.assert_allclose(c2f_direct, c2f_chained, atol=1e-12, rtol=0.0)


def test_primitive_to_bcond_linearization(scalar_blocks):
    """Test that primitive_to_bcond matrix correctly linearizes the transformation."""
    block1, block2, atol = scalar_blocks

    # Calculate finite difference in primitive variables
    dprim_actual = np.array(
        [
            block2.rho - block1.rho,
            block2.Vx - block1.Vx,
            block2.Vr - block1.Vr,
            block2.Vt - block1.Vt,
            block2.P - block1.P,
        ]
    )

    # Calculate finite difference in boundary condition variables [ho, s, tanAlpha, sinBeta, P]
    bcond1 = np.array(
        [
            block1.ho,
            block1.s,
            block1.tanAlpha,
            block1.sinBeta,
            block1.P,
        ]
    )
    bcond2 = np.array(
        [
            block2.ho,
            block2.s,
            block2.tanAlpha,
            block2.sinBeta,
            block2.P,
        ]
    )
    dbcond_actual = bcond2 - bcond1

    bcond2p = perturbation.bcond_to_primitive(block1)
    dprim_calc = util.matvec(bcond2p, dbcond_actual)
    err_rel = np.abs(dprim_calc - dprim_actual) / atol
    assert (err_rel < 1.0).all(), f"Relative error too large: {err_rel}"


def test_primitive_mix_matrices_are_inverses(scalar_blocks):
    """Test that primitive_to_mix and mix_to_primitive are inverses."""
    block1 = scalar_blocks[0]

    p2mix = perturbation.primitive_to_mix(block1)
    mix2p = perturbation.mix_to_primitive(block1)

    identity1 = util.matmat(p2mix, mix2p)
    identity2 = util.matmat(mix2p, p2mix)

    np.testing.assert_allclose(identity1, np.eye(5), atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(identity2, np.eye(5), atol=1e-5, rtol=0.0)


def test_primitive_to_mix_linearization(scalar_blocks):
    """Test that mix_to_primitive matrix correctly linearizes the transformation."""
    block1, block2, atol = scalar_blocks

    mix1 = np.array([block1.ho, block1.s, block1.Vr, block1.Vt, block1.P])
    mix2 = np.array([block2.ho, block2.s, block2.Vr, block2.Vt, block2.P])
    dmix_actual = mix2 - mix1

    dprim_actual = np.array(
        [
            block2.rho - block1.rho,
            block2.Vx - block1.Vx,
            block2.Vr - block1.Vr,
            block2.Vt - block1.Vt,
            block2.P - block1.P,
        ]
    )

    mix2p = perturbation.mix_to_primitive(block1)
    dprim_calc = util.matvec(mix2p, dmix_actual)
    err_rel = np.abs(dprim_calc - dprim_actual) / atol
    assert (err_rel < 1.0).all(), f"Relative error too large: {err_rel}"


def test_mix_to_conserved_matches_product(scalar_blocks):
    """Test that mix_to_conserved equals primitive_to_conserved @ mix_to_primitive."""
    block1 = scalar_blocks[0]

    m2c_direct = perturbation.mix_to_conserved(block1)
    m2c_chained = util.matmat(
        perturbation.primitive_to_conserved(block1),
        perturbation.mix_to_primitive(block1),
    )

    np.testing.assert_allclose(m2c_direct, m2c_chained, atol=1e-2, rtol=0.0)


def test_mix_to_conserved_linearization(scalar_blocks):
    """Test that mix_to_conserved correctly maps a mix perturbation to conserved."""
    block1, block2, _ = scalar_blocks

    dprim = np.array(
        [
            block2.rho - block1.rho,
            block2.Vx - block1.Vx,
            block2.Vr - block1.Vr,
            block2.Vt - block1.Vt,
            block2.P - block1.P,
        ]
    )
    p2mix = perturbation.primitive_to_mix(block1)
    dmix = util.matvec(p2mix, dprim)

    m2c = perturbation.mix_to_conserved(block1)
    dcons_calc = util.matvec(m2c, dmix)

    p2c = perturbation.primitive_to_conserved(block1)
    dcons_ref = util.matvec(p2c, dprim)

    np.testing.assert_allclose(dcons_calc, dcons_ref, atol=0.0, rtol=1e-3)


def test_chic_to_mix_matches_product(scalar_blocks):
    """Test that chic_to_mix equals primitive_to_mix @ chic_to_primitive."""
    block1 = scalar_blocks[0]

    c2m_direct = perturbation.chic_to_mix(block1)
    c2m_chained = util.matmat(
        perturbation.primitive_to_mix(block1),
        perturbation.chic_to_primitive(block1),
    )

    np.testing.assert_allclose(c2m_direct, c2m_chained, atol=1e-4, rtol=0.0)


def test_chic_to_mix_linearization(scalar_blocks):
    """Test that chic_to_mix correctly maps a chic perturbation to mix."""
    block1, block2, _ = scalar_blocks

    dprim = np.array(
        [
            block2.rho - block1.rho,
            block2.Vx - block1.Vx,
            block2.Vr - block1.Vr,
            block2.Vt - block1.Vt,
            block2.P - block1.P,
        ]
    )
    p2chic = perturbation.primitive_to_chic(block1)
    dchic = util.matvec(p2chic, dprim)

    c2m = perturbation.chic_to_mix(block1)
    dmix_calc = util.matvec(c2m, dchic)

    p2mix = perturbation.primitive_to_mix(block1)
    dmix_ref = util.matvec(p2mix, dprim)

    np.testing.assert_allclose(dmix_calc, dmix_ref, atol=0.0, rtol=1e-5)


def _ref_scales(block):
    """Return dict of reference scale vectors for all variable spaces."""
    rho_ref = block.fluid.rho_ref
    V_ref = block.fluid.V_ref
    L_ref = block.L_ref
    P_ref = block.fluid.P_ref
    T_ref = block.fluid.T_ref
    u_ref = V_ref**2  # = fluid.u_ref
    Rgas_ref = V_ref**2 / T_ref

    prim_ref = np.array([rho_ref, V_ref, V_ref, V_ref, P_ref])
    cons_ref = np.array(
        [
            rho_ref,
            rho_ref * V_ref,
            rho_ref * V_ref,
            rho_ref * V_ref * L_ref,
            rho_ref * V_ref**2,
        ]
    )
    flux_ref = np.array(
        [
            rho_ref * V_ref,
            rho_ref * V_ref**2,
            rho_ref * V_ref**2,
            rho_ref * V_ref**2 * L_ref,
            rho_ref * V_ref**3,
        ]
    )
    chic_ref = np.array([P_ref, P_ref, rho_ref * V_ref**2, rho_ref * V_ref**2, P_ref])
    bcond_ref = np.array([u_ref, Rgas_ref, 1.0, 1.0, P_ref])
    mix_ref = np.array([u_ref, Rgas_ref, V_ref, V_ref, P_ref])

    return {
        "prim": prim_ref,
        "cons": cons_ref,
        "flux": flux_ref,
        "chic": chic_ref,
        "bcond": bcond_ref,
        "mix": mix_ref,
    }


def _to_dimensional(J_nd, row_ref, col_ref):
    """Convert nondimensional Jacobian to dimensional: J_dim[i,j] = row_ref[i] / col_ref[j] * J_nd[i,j]."""
    return J_nd * (row_ref[:, None] / col_ref[None, :])


def test_matrix_reference_invariance():
    """Test that all Jacobians give same physical result with different reference scales.

    The nondimensional Jacobian differs between reference scales, but converting
    to dimensional should give the same physical matrix.
    """
    # Base state with default references
    fluid_base = ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-4, Pr=0.8)
    block_base = ember.block.Block(shape=())
    block_base.set_x(np.array([0.05]))
    block_base.set_r(np.array([0.85]))
    block_base.set_t(np.array([0.05]))
    block_base.set_fluid(fluid_base)
    block_base.set_Vx(100.0)
    block_base.set_Vr(80.0)
    block_base.set_Vt(50.0)
    block_base.set_P_rho(1.2e5, 295.0)

    # Same physical state with different references
    fluid_ref = ember.fluid.PerfectFluid(
        cp=1105.0,
        gamma=1.3,
        mu=1.8e-4,
        Pr=0.8,
        rho_ref=1.5,
        V_ref=200.0,
    )
    block_ref = ember.block.Block(shape=())
    block_ref.set_x(np.array([0.05]))
    block_ref.set_r(np.array([0.85]))
    block_ref.set_t(np.array([0.05]))
    block_ref.set_fluid(fluid_ref)
    block_ref.set_Vx(100.0)
    block_ref.set_Vr(80.0)
    block_ref.set_Vt(50.0)
    block_ref.set_P_rho(1.2e5, 295.0)
    block_ref.set_L_ref(0.05)

    # Verify dimensional state matches (float32 precision)
    np.testing.assert_allclose(block_base.rho, block_ref.rho, rtol=1e-4)
    np.testing.assert_allclose(block_base.Vx, block_ref.Vx, rtol=1e-4)
    np.testing.assert_allclose(block_base.P, block_ref.P, rtol=1e-4)

    refs_base = _ref_scales(block_base)
    refs_ref = _ref_scales(block_ref)

    # (label, function, row_space, col_space)
    jacobians = [
        ("primitive_to_conserved", perturbation.primitive_to_conserved, "cons", "prim"),
        ("conserved_to_primitive", perturbation.conserved_to_primitive, "prim", "cons"),
        ("primitive_to_chic", perturbation.primitive_to_chic, "chic", "prim"),
        ("chic_to_primitive", perturbation.chic_to_primitive, "prim", "chic"),
        ("primitive_to_flux", perturbation.primitive_to_flux, "flux", "prim"),
        ("flux_to_primitive", perturbation.flux_to_primitive, "prim", "flux"),
        ("primitive_to_bcond", perturbation.primitive_to_bcond, "bcond", "prim"),
        ("bcond_to_primitive", perturbation.bcond_to_primitive, "prim", "bcond"),
        ("primitive_to_mix", perturbation.primitive_to_mix, "mix", "prim"),
        ("mix_to_primitive", perturbation.mix_to_primitive, "prim", "mix"),
        ("mix_to_conserved", perturbation.mix_to_conserved, "cons", "mix"),
        ("chic_to_mix", perturbation.chic_to_mix, "mix", "chic"),
        ("flux_to_conserved", perturbation.flux_to_conserved, "cons", "flux"),
        ("conserved_to_flux", perturbation.conserved_to_flux, "flux", "cons"),
    ]

    for label, fn, row_space, col_space in jacobians:
        J_base = fn(block_base)
        J_ref = fn(block_ref)
        J_dim_base = _to_dimensional(J_base, refs_base[row_space], refs_base[col_space])
        J_dim_ref = _to_dimensional(J_ref, refs_ref[row_space], refs_ref[col_space])
        # atol scales with matrix magnitude to handle float32 near-zero entries
        scale = max(np.abs(J_dim_base).max(), np.abs(J_dim_ref).max(), 1.0)
        np.testing.assert_allclose(
            J_dim_base,
            J_dim_ref,
            rtol=1e-3,
            atol=scale * 1e-5,
            err_msg=f"{label} dimensional Jacobian is not invariant to reference values",
        )


def test_matrix_uniformity_all_dimensions():
    """Test that all perturbation matrices work with 1D, 2D, and 3D blocks and have uniform values for uniform flow fields."""
    fluid = ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-4, Pr=0.8)

    # Test shapes: scalar (0D), 1D, 2D, 3D
    test_shapes = [(), (10,), (5, 8), (4, 6, 3)]

    for shape in test_shapes:
        # Create block with uniform flow field
        block = ember.block.Block(shape=shape)
        block.set_x(np.array([0.05]))
        block.set_r(np.array([0.85]))
        block.set_t(np.array([0.05]))
        block.set_fluid(fluid)
        block.set_Vx(100.0)
        block.set_Vr(80.0)
        block.set_Vt(50.0)
        block.set_P_rho(1.2e5, 295.0)

        # Test all transformation matrices exist and have correct shape
        matrices_to_test = {
            "primitive_to_conserved": perturbation.primitive_to_conserved,
            "conserved_to_primitive": perturbation.conserved_to_primitive,
            "primitive_to_chic": perturbation.primitive_to_chic,
            "chic_to_primitive": perturbation.chic_to_primitive,
            "primitive_to_flux": perturbation.primitive_to_flux,
            "flux_to_primitive": perturbation.flux_to_primitive,
            "primitive_to_bcond": perturbation.primitive_to_bcond,
            "bcond_to_primitive": perturbation.bcond_to_primitive,
            "primitive_to_mix": perturbation.primitive_to_mix,
            "mix_to_primitive": perturbation.mix_to_primitive,
            "mix_to_conserved": perturbation.mix_to_conserved,
            "chic_to_mix": perturbation.chic_to_mix,
        }

        for matrix_name, fn in matrices_to_test.items():
            matrix = fn(block)

            # Check matrix has correct shape: block.shape + (5, 5) (trailing dimensions)
            expected_shape = shape + (5, 5)
            assert matrix.shape == expected_shape, (
                f"{matrix_name} matrix has shape {matrix.shape}, expected {expected_shape} for block shape {shape}"
            )

            # For uniform flow field, the 5x5 matrix values should be identical at all grid points
            if len(shape) > 0:  # Skip uniformity check for scalar blocks
                # Get first 5x5 matrix as reference
                ref_matrix = matrix[(0,) * len(shape)]

                # Check all 5x5 matrices are identical across all grid points
                it = np.nditer(np.zeros(shape), flags=["multi_index"])
                for _ in it:
                    idx = it.multi_index
                    current_matrix = matrix[idx]
                    np.testing.assert_allclose(
                        current_matrix,
                        ref_matrix,
                        rtol=1e-10,
                        atol=1e-15,
                        err_msg=f"{matrix_name} matrix values are not uniform for block shape {shape} at grid point {idx}",
                    )
