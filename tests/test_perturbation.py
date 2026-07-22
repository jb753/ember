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


# Relative size of the finite-difference perturbation. Small enough that the
# second-order truncation error stays well under _LIN_RTOL, large enough that
# differencing float32 storage does not swamp it (the noise floor is about
# 6e-8 / _LIN_EPS in relative terms).
_LIN_EPS = 1e-3

# How closely a Jacobian must reproduce the finite-difference delta, as a
# fraction of the delta a one-_LIN_EPS change in that component would produce.
_LIN_RTOL = 5e-2


@pytest.fixture(params=range(5), ids=["d_rho", "d_Vx", "d_Vr", "d_Vt", "d_P"])
def scalar_blocks(request):
    """Two scalar blocks differing by a small change in one primitive variable.

    Perturbing a single primitive at a time means each linearization test
    probes one column of the Jacobian on its own, so a wrong entry cannot be
    masked by cancellation against the others.

    The returned tolerance is per component and proportional to the
    perturbation, not to the state: a Jacobian must predict each component of
    the delta to within :data:`_LIN_RTOL` of what a one-:data:`_LIN_EPS` change
    in that component would be. Scaling the tolerance by the state instead
    would admit any Jacobian at all, and scaling by the delta alone would be
    unusable for the components a single-variable perturbation leaves exactly
    zero.
    """
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

    prim1 = np.array([block1.rho, block1.Vx, block1.Vr, block1.Vt, block1.P]).ravel()

    # Perturbed state: one primitive variable moved by a small relative step.
    prim2 = prim1.copy()
    prim2[request.param] *= 1.0 + _LIN_EPS

    block2 = block1.copy()
    block2.set_P_rho(prim2[4], prim2[0])  # thermodynamics, velocity preserved
    block2.set_Vx(prim2[1])
    block2.set_Vr(prim2[2])
    block2.set_Vt(prim2[3])

    atol = _LIN_RTOL * _LIN_EPS * np.abs(prim1)

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

    # Characteristic variables are defined as perturbations about a reference
    # state with the coefficients frozen there (Giles 1988, Eq. 5.7), so the
    # delta must be formed with block1's rho, a throughout. Evaluating rho*a on
    # each state separately would add a d(rho*a) term that primitive_to_chic
    # does not model and that does not vanish with the step size, which shows
    # up whenever the perturbation changes the speed of sound.
    rhoa = block1.rho * block1.a
    asq = block1.a**2

    def _chic(block):
        return np.array(
            [
                -block.Vx * rhoa + block.P,  # upstream acoustic
                block.Vx * rhoa + block.P,  # downstream acoustic
                block.Vr * rhoa,  # radial acoustic
                block.Vt * rhoa,  # tangential acoustic
                block.P - asq * block.rho,  # entropy
            ]
        )

    dchic_actual = _chic(block2) - _chic(block1)

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

    # A linear identity, so only float32 round-off separates the two sides. A
    # single-variable perturbation leaves some components exactly zero, which a
    # pure rtol cannot express, hence the atol scaled to the vector magnitude.
    np.testing.assert_allclose(
        dcons_calc,
        dcons_ref,
        atol=1e-4 * np.abs(dcons_ref).max(),
        rtol=1e-3,
    )


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


def test_chic_to_bcond_and_chic_to_mix_share_rows(scalar_blocks):
    """chic_to_bcond and chic_to_mix agree exactly on rows 0, 1 and 4.

    The two differ only in which pair of transverse quantities they carry --
    (tanAlpha, sinBeta) against (Vr, Vt) -- so their stagnation enthalpy,
    entropy and static pressure rows are the same expressions. Several callers
    depend on that:

    * :class:`ember.outlet_nonreflecting.NonReflectingOutletPatch` hardcodes
      ``dp/dc_up = 1/2`` from row 4, so it is correct under either set;
    * :meth:`ember.inlet_nonreflecting.NonReflectingInletPatch._calc_reference_extra`
      builds its local Newton system and both coupling columns from rows 0:2,
      so those are the same matrices whichever set the patch prescribes;
    * :meth:`ember.mixing_communicator.MixingCommunicator._write_targets`
      splits the interface jump on the assumption that row 4 is the pressure.

    If this ever stops holding, the outflow side of the non-reflecting mixing
    plane and the harmonic systems of the inflow side would silently disagree
    between the two variable sets.
    """
    block1 = scalar_blocks[0]

    c2b = perturbation.chic_to_bcond(block1)
    c2m = perturbation.chic_to_mix(block1)

    for row in (0, 1, 4):
        np.testing.assert_array_equal(
            c2b[..., row, :],
            c2m[..., row, :],
            err_msg=f"chic_to_bcond and chic_to_mix differ in row {row}",
        )

    # And rows 2-3 genuinely do differ, so the test above is not vacuous.
    assert not np.array_equal(c2b[..., 2:4, :], c2m[..., 2:4, :])


def test_chic_to_bcond_matches_product(scalar_blocks):
    """Test that chic_to_bcond equals primitive_to_bcond @ chic_to_primitive."""
    block1 = scalar_blocks[0]

    c2b_direct = perturbation.chic_to_bcond(block1)
    c2b_chained = util.matmat(
        perturbation.primitive_to_bcond(block1),
        perturbation.chic_to_primitive(block1),
    )

    np.testing.assert_allclose(c2b_direct, c2b_chained, atol=1e-6, rtol=0.0)


def test_chic_to_bcond_linearization(scalar_blocks):
    """Test that chic_to_bcond correctly maps a chic perturbation to bcond."""
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

    c2b = perturbation.chic_to_bcond(block1)
    dbcond_calc = util.matvec(c2b, dchic)

    p2b = perturbation.primitive_to_bcond(block1)
    dbcond_ref = util.matvec(p2b, dprim)

    np.testing.assert_allclose(dbcond_calc, dbcond_ref, atol=0.0, rtol=1e-5)


def test_primitive_to_bcond_finite_difference():
    """Test primitive_to_bcond against finite differences of the block properties.

    The angle rows are measured against the meridional speed, matching
    Block.tanAlpha and Block.sinBeta. The existing linearization tests compare
    against a tolerance of order the values themselves, which is too loose to
    detect a wrong velocity scale in these derivatives.
    """
    fluid = ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-4, Pr=0.8)

    def _block(Vx, Vr, Vt):
        b = ember.block.Block(shape=())
        b.set_x(np.array([0.05]))
        b.set_r(np.array([0.85]))
        b.set_t(np.array([0.05]))
        b.set_fluid(fluid)
        b.set_Vx(Vx)
        b.set_Vr(Vr)
        b.set_Vt(Vt)
        b.set_P_rho(1.2e5, 1.4)
        return b

    V = (100.0, 80.0, 50.0)
    jac = perturbation.primitive_to_bcond(_block(*V))

    h = 1.0
    for col, axis in enumerate((0, 1, 2), start=1):
        step = [0.0, 0.0, 0.0]
        step[axis] = h
        plus = _block(*(v + d for v, d in zip(V, step)))
        minus = _block(*(v - d for v, d in zip(V, step)))
        d_tanAlpha = (float(plus.tanAlpha) - float(minus.tanAlpha)) / (2.0 * h)
        d_sinBeta = (float(plus.sinBeta) - float(minus.sinBeta)) / (2.0 * h)
        assert jac[2, col] == pytest.approx(d_tanAlpha, rel=1e-3)
        assert jac[3, col] == pytest.approx(d_sinBeta, rel=1e-3, abs=1e-9)


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
        ("chic_to_bcond", perturbation.chic_to_bcond, "bcond", "chic"),
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
            "chic_to_bcond": perturbation.chic_to_bcond,
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
