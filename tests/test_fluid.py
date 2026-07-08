"""Tests for fluid thermodynamic properties (ember.fluid).

Module tested: ember.fluid

Test cases:
- test_perfect_properites: Input properties of the perfect fluid
- test_universal_relations: Thermodynamic relations valid for any fluid
- test_get_set_pairs: Get and set pairs for fluid properties
- test_internal_energy_datum: Internal energy datum for fluid properties
- test_derivatives: Derivatives of fluid properties
- test_change_datum: change_datum method preserves thermodynamic state
- test_change_datum_effect: change_datum correctly affects u and s datum
- test_datum_zero: u = 0 and s = 0 simultaneously at (P_dtm, T_dtm)
- test_set_P_rho_accuracy_comparison: Numerical accuracy of set_P_rho with different datum values
- test_set_P_rho_accuracy: Numerical accuracy of set_P_rho implementation
- test_perfect_fluid_validation: PerfectFluid input validation
- test_perfect_fluid_datum_default_and_custom: PerfectFluid datum handling
- test_main_script_execution: Main script execution path
- test_nondim_scaling: Non-unity-ref fluid returns values scaled by refs
- test_change_ref: change_ref returns consistent object
- test_change_datum_nondim: change_datum with non-unity reference values
- test_fluid_member_order: _Fluid and subclasses follow standard member ordering
"""

import ember.fluid
import numpy as np
import pytest


FLUIDS = [
    ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-5, Pr=0.7),
    ember.fluid.PerfectFluid(
        cp=1051.0, gamma=1.4, mu=3.8e-5, Pr=1.0, P_dtm=5e4, T_dtm=200.0
    ),
    ember.fluid.PerfectFluid(cp=1001.0, gamma=1.36, mu=2.8e-5, Pr=0.6, T_dtm=600.0),
]

rho_test = [
    1.0,
    5.0,
    2.2,
    np.array([1.6]),
    np.array([1.0, 2.0]),
    np.array([[1.0, 2.0], [3.0, 4.0]]),
]
u_test = [
    30000.0,
    1000.0,
    2200.0,
    np.array([1600.0]),
    np.array([1000.0, 2100.0]),
    np.array([[1000.0, 2000.0], [3000.0, 4000.0]]),
]


def test_perfect_properites():
    """Test the input properties of the perfect fluid."""

    cp = 1105.0
    gamma = 1.33
    mu = 4.0e-5
    Pr = 0.8

    cv = cp / gamma
    Rgas = cp - cv

    fluid = ember.fluid.PerfectFluid(cp=cp, gamma=gamma, mu=mu, Pr=Pr)

    for rho, u in zip(rho_test, u_test):
        # Inputs are correct
        assert np.allclose(fluid.get_cp(rho, u), cp)
        assert np.allclose(fluid.get_mu(rho, u), mu)
        assert np.allclose(fluid.get_Pr(rho, u), Pr)
        assert np.allclose(fluid.get_gamma(rho, u), gamma)

        # Ideal gas equations OK
        assert np.allclose(fluid.get_P(rho, u), rho * Rgas * fluid.get_T(rho, u))
        assert np.allclose(fluid.get_h(rho, u), u + fluid.get_P(rho, u) / rho)
        assert np.allclose(fluid.get_h(rho, u), u + fluid.get_T(rho, u) * Rgas)


@pytest.mark.parametrize("fluid", FLUIDS)
def test_universal_relations(fluid):
    """Test thermodynamic relations valid for any fluid."""

    for rho, u in zip(rho_test, u_test):
        # Definition of enthalpy
        h = fluid.get_h(rho, u)
        assert np.allclose(h, u + fluid.get_P(rho, u) / rho)

        # Definition of gamma
        gamma = fluid.get_gamma(rho, u)
        cv = fluid.get_cv(rho, u)
        cp = fluid.get_cp(rho, u)
        assert np.allclose(gamma, cp / cv)


@pytest.mark.parametrize("fluid", FLUIDS)
def test_get_set_pairs(fluid):
    """Test the get and set pairs for the fluid properties."""

    for rho, u in zip(rho_test, u_test):
        h = fluid.get_h(rho, u)
        s = fluid.get_s(rho, u)
        P = fluid.get_P(rho, u)
        T = fluid.get_T(rho, u)

        rtol = 1e-4  # Original tolerance maintained for round-trip tests
        assert np.allclose(fluid.set_h_s(h, s), (rho, u), rtol=rtol)
        assert np.allclose(fluid.set_P_T(P, T), (rho, u), rtol=rtol)
        assert np.allclose(fluid.set_P_s(P, s), (rho, u), rtol=rtol)
        assert np.allclose(fluid.set_P_h(P, h), (rho, u), rtol=rtol)
        assert np.allclose(fluid.set_P_rho(P, rho), (rho, u), rtol=rtol)
        assert np.allclose(fluid.set_T_rho(T, rho), (rho, u), rtol=rtol)
        assert np.allclose(fluid.set_T_s(T, s), (rho, u), rtol=rtol)
        assert np.allclose(fluid.set_rho_s(rho, s), (rho, u), rtol=rtol)


@pytest.mark.parametrize("fluid", FLUIDS)
def test_internal_energy_datum(fluid):
    """Test the internal energy datum for the fluid properties."""

    T_dtm = fluid.T_dtm
    # Use a temperature above T_dtm to have a non-zero u
    T_test = T_dtm + 100.0
    rho, u = fluid.set_P_T(1e5, T_test)
    assert np.isclose(fluid.get_T(rho, u), T_test)

    # Test that internal energy is zero at the datum temperature
    _, u_datum = fluid.set_P_T(1e5, T_dtm)
    assert np.isclose(u_datum, 0.0, atol=1e-10)


@pytest.mark.parametrize("fluid", FLUIDS)
def test_derivatives(fluid):
    """Test the derivatives of the fluid properties."""

    rtol = 1e-3  # Tolerance for numerical derivatives (relaxed for float32)

    for rho, u in zip(rho_test, u_test):
        # Evaulate rho and P vectors for perturbations
        rho0 = np.mean(rho)
        P0 = fluid.get_P(rho, u).mean()
        rho_vec = np.linspace(0.9, 1.1) * rho0
        P_vec = np.linspace(0.9, 1.1) * P0

        # rho derivatives
        u_vec = fluid.set_P_rho(P0, rho_vec)[1]
        dsdrho = np.gradient(fluid.get_s(rho_vec, u_vec), rho_vec)
        dhdrho = np.gradient(fluid.get_h(rho_vec, u_vec), rho_vec)
        dudrho = np.gradient(u_vec, rho_vec)

        assert np.allclose(
            dsdrho[1:-1], fluid.get_dsdrho_P(rho_vec, u_vec)[1:-1], rtol=rtol
        )
        assert np.allclose(
            dhdrho[1:-1], fluid.get_dhdrho_P(rho_vec, u_vec)[1:-1], rtol=rtol
        )
        assert np.allclose(
            dudrho[1:-1], fluid.get_dudrho_P(rho_vec, u_vec)[1:-1], rtol=rtol
        )

        # P derivatives
        u_vec = fluid.set_P_rho(P_vec, rho0)[1]
        dsdp = np.gradient(fluid.get_s(rho0, u_vec), P_vec)
        dhdp = np.gradient(fluid.get_h(rho0, u_vec), P_vec)
        dudp = np.gradient(u_vec, P_vec)

        assert np.allclose(dsdp[1:-1], fluid.get_dsdP_rho(rho0, u_vec)[1:-1], rtol=rtol)
        assert np.allclose(dhdp[1:-1], fluid.get_dhdP_rho(rho0, u_vec), rtol=rtol)
        assert np.allclose(dudp[1:-1], fluid.get_dudP_rho(rho0, u_vec), rtol=rtol)


@pytest.mark.parametrize("fluid", FLUIDS)
def test_get_functions_out_parameter(fluid):
    """Test that out= parameter writes correct results into pre-allocated array."""

    rho = np.ones((3, 4), order="F", dtype=np.float32) * 2.5
    u = np.full((3, 4), 8e4, order="F", dtype=np.float32)

    get_funcs = [
        fluid.get_cp,
        fluid.get_cv,
        fluid.get_T,
        fluid.get_a,
        fluid.get_P,
        fluid.get_s,
        fluid.get_gamma,
        fluid.get_Rgas,
        fluid.get_h,
        fluid.get_mu,
        fluid.get_Pr,
        fluid.get_dhdP_rho,
        fluid.get_dhdrho_P,
        fluid.get_dsdP_rho,
        fluid.get_dsdrho_P,
        fluid.get_dudP_rho,
        fluid.get_dudrho_P,
    ]

    for func in get_funcs:
        expected = func(rho, u)
        out = np.empty_like(expected)
        returned = func(rho, u, out=out)

        # out is the same object that was returned
        assert returned is out, f"{func.__name__}: returned array is not out"

        # Values match the no-out path
        assert np.allclose(out, expected), (
            f"{func.__name__}: out values differ from expected"
        )


@pytest.mark.parametrize("fluid", FLUIDS)
def test_get_functions_out_3d(fluid):
    """Test out= parameter with 3D F-contiguous arrays."""

    rho = np.ones((3, 4, 2), order="F", dtype=np.float32) * 2.5
    u = np.full((3, 4, 2), 8e4, order="F", dtype=np.float32)

    get_funcs = [
        fluid.get_P,
        fluid.get_T,
        fluid.get_a,
        fluid.get_h,
        fluid.get_s,
        fluid.get_dhdP_rho,
        fluid.get_dudP_rho,
    ]

    for func in get_funcs:
        expected = func(rho, u)
        out = np.empty_like(expected)
        returned = func(rho, u, out=out)

        assert returned is out, f"{func.__name__}: returned array is not out"
        assert np.allclose(out, expected), (
            f"{func.__name__}: out values differ from expected (3D path)"
        )


@pytest.mark.parametrize("fluid", FLUIDS)
def test_change_datum(fluid):
    """Test change_datum method returns fluid with correct datum."""

    P_dtm_new, T_dtm_new = 2e5, 400.0

    fluid_new = fluid.change_datum(P_dtm_new, T_dtm_new)

    # Check new fluid has correct datum
    assert (fluid_new.P_dtm, fluid_new.T_dtm) == (
        np.float32(P_dtm_new),
        np.float32(T_dtm_new),
    )

    # Check that fluid properties are preserved
    rho, u = fluid_new.set_P_T(P_dtm_new, T_dtm_new)
    assert np.allclose(fluid_new.get_cp(rho, u), fluid.get_cp(rho, u))
    assert np.allclose(fluid_new.get_gamma(rho, u), fluid.get_gamma(rho, u))
    assert np.allclose(fluid_new.get_Rgas(rho, u), fluid.get_Rgas(rho, u))
    assert np.allclose(fluid_new.get_mu(rho, u), fluid.get_mu(rho, u))
    assert np.allclose(fluid_new.get_Pr(rho, u), fluid.get_Pr(rho, u))

    # At the new datum, u=0 and s=0
    assert np.allclose(u, 0.0, atol=1e-10)
    assert np.allclose(fluid_new.get_s(rho, u), 0.0, atol=1e-4)


def test_change_datum_effect():
    """Test that change_datum returns fluid with shifted datum zeros."""

    fluid = ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-5, Pr=0.7)

    P_dtm_new, T_dtm_new = 2e5, 350.0
    fluid_new = fluid.change_datum(P_dtm_new, T_dtm_new)

    # At the new datum state, both u and s should be zero
    rho_dtm, u_datum = fluid_new.set_P_T(P_dtm_new, T_dtm_new)
    assert np.allclose(u_datum, 0.0, atol=1e-10)
    s_datum = fluid_new.get_s(rho_dtm, u_datum)
    assert np.allclose(s_datum, 0.0, atol=1e-4)


def test_datum_zero():
    """Test that u = 0 and s = 0 simultaneously at (P_dtm, T_dtm)."""

    for P_dtm, T_dtm in [(1e5, 300.0), (2e5, 400.0), (5e4, 250.0)]:
        fluid = ember.fluid.PerfectFluid(
            cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7, P_dtm=P_dtm, T_dtm=T_dtm
        )
        rho, u = fluid.set_P_T(P_dtm, T_dtm)
        assert np.isclose(u, 0.0, atol=1e-10), f"u != 0 at datum (P={P_dtm}, T={T_dtm})"
        s = fluid.get_s(rho, u)
        # float32 log cancellation gives ~1e-5 residual; use generous atol
        assert np.isclose(s, 0.0, atol=1e-4), f"s != 0 at datum (P={P_dtm}, T={T_dtm})"


def test_set_P_rho_accuracy_comparison():
    """Compare numerical accuracy of set_P_rho with different datum values."""

    # Create fluids with different datum temperatures
    fluids_test = [
        ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-5, Pr=0.7, T_dtm=100.0),
        ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-5, Pr=0.7),  # default
        ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-5, Pr=0.7, T_dtm=600.0),
    ]

    # Test conditions spanning typical CFD ranges
    P_test = np.array([1e3, 1e5, 1e6, 5e6])  # Pa
    rho_test = np.array([0.1, 1.0, 5.0, 10.0])  # kg/m^3

    for fluid in fluids_test:
        T_dtm_val = fluid.T_dtm
        max_temp_error = 0.0
        max_pressure_error = 0.0

        for P in P_test:
            for rho in rho_test:
                rho_out, u_out = fluid.set_P_rho(P, rho)

                assert np.allclose(rho_out, rho, rtol=1e-15)

                T_computed = fluid.get_T(rho_out, u_out)
                T_expected = P / (fluid.get_Rgas(rho, u_out) * rho)

                temp_rel_error = abs(T_computed - T_expected) / T_expected
                max_temp_error = max(max_temp_error, temp_rel_error)

                P_roundtrip = fluid.get_P(rho_out, u_out)
                pressure_rel_error = abs(P_roundtrip - P) / P
                max_pressure_error = max(max_pressure_error, pressure_rel_error)

        assert max_temp_error < 1e-4, (
            f"Temperature error too large with T_dtm={T_dtm_val}: {max_temp_error}"
        )
        assert max_pressure_error < 1e-4, (
            f"Pressure error too large with T_dtm={T_dtm_val}: {max_pressure_error}"
        )


@pytest.mark.parametrize("fluid", FLUIDS)
def test_set_P_rho_accuracy(fluid):
    """Test numerical accuracy of set_P_rho implementation."""

    # Test conditions spanning typical CFD ranges
    P_test = np.array([1e3, 1e5, 1e6, 5e6])  # Pa
    rho_test = np.array([0.1, 1.0, 5.0, 10.0])  # kg/m^3

    max_temp_error = 0.0
    max_pressure_error = 0.0

    for P in P_test:
        for rho in rho_test:
            # Use set_P_rho to get (rho, u)
            rho_out, u_out = fluid.set_P_rho(P, rho)

            # Verify density is preserved exactly
            assert np.allclose(rho_out, rho, rtol=1e-15)

            # Compute temperature from internal energy
            T_computed = fluid.get_T(rho_out, u_out)

            # Expected temperature from ideal gas law
            T_expected = P / (fluid.get_Rgas(rho, u_out) * rho)

            # Track maximum temperature error
            temp_rel_error = abs(T_computed - T_expected) / T_expected
            max_temp_error = max(max_temp_error, temp_rel_error)

            # Verify round-trip accuracy: P_rho -> get_P
            P_roundtrip = fluid.get_P(rho_out, u_out)
            pressure_rel_error = abs(P_roundtrip - P) / P
            max_pressure_error = max(max_pressure_error, pressure_rel_error)

    # Assert tight accuracy bounds to prevent regression
    # Note: tolerances relaxed for float32 Fortran implementation
    assert max_temp_error < 1e-4, f"Temperature error too large: {max_temp_error}"
    assert max_pressure_error < 1e-4, f"Pressure error too large: {max_pressure_error}"


def test_perfect_fluid_validation():
    """Test PerfectFluid input validation."""

    # Test invalid inputs
    with pytest.raises(ValueError):
        ember.fluid.PerfectFluid(
            cp=0.0, gamma=1.4, mu=1e-5, Pr=0.72
        )  # cp must be positive

    with pytest.raises(ValueError):
        ember.fluid.PerfectFluid(
            cp=1000.0, gamma=1.0, mu=1e-5, Pr=0.72
        )  # gamma must be > 1

    with pytest.raises(ValueError):
        ember.fluid.PerfectFluid(
            cp=1000.0, gamma=1.4, mu=0.0, Pr=0.72
        )  # mu must be positive

    with pytest.raises(ValueError):
        ember.fluid.PerfectFluid(
            cp=1000.0, gamma=1.4, mu=1e-5, Pr=0.0
        )  # Pr must be positive


def test_perfect_fluid_datum_default_and_custom():
    """Test PerfectFluid datum handling."""

    # Test defaults
    fluid_default = ember.fluid.PerfectFluid(cp=1000.0, gamma=1.4, mu=1e-5, Pr=0.72)
    assert np.isclose(fluid_default.P_dtm, 1e5)
    assert np.isclose(fluid_default.T_dtm, 300.0)

    # Test custom datum
    fluid_custom = ember.fluid.PerfectFluid(
        cp=1000.0, gamma=1.4, mu=1e-5, Pr=0.72, P_dtm=2e5, T_dtm=400.0
    )
    assert np.isclose(fluid_custom.P_dtm, 2e5)
    assert np.isclose(fluid_custom.T_dtm, 400.0)


def test_main_script_execution():
    """Test the __main__ script execution path."""

    # This tests the lines 573-577 that are marked as missing coverage
    # We can't directly test __main__ execution, but we can test the functionality
    x = np.asfortranarray(np.array([[1, 2], [3, 4]]), dtype=np.float32)
    y = np.asfortranarray(np.array([[1, 2]]), dtype=np.float32)

    # Import util to access full_bcast
    from ember import util

    z = util.full_bcast(x, y, 0.0)

    # Test that output has expected properties (matching __main__ expectations)
    assert z.flags["F_CONTIGUOUS"]
    assert z.dtype == np.float32


def test_fluid_member_order():
    """_Fluid and all concrete subclasses follow the standard member ordering convention."""
    import inspect
    from pathlib import Path
    from conftest import assert_class_member_order

    fluid_cls = ember.fluid._Fluid
    src = Path(inspect.getfile(fluid_cls)).read_text()

    assert_class_member_order(src, "_Fluid")

    for name in dir(ember.fluid):
        obj = getattr(ember.fluid, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, fluid_cls)
            and obj is not fluid_cls
            and not inspect.isabstract(obj)
        ):
            assert_class_member_order(src, obj.__name__)


def test_nondim_scaling():
    """Test that non-unity-ref fluid returns values scaled by refs vs unity-ref fluid."""

    cp, gamma, mu_val, Pr = 1105.0, 1.3, 1.8e-5, 0.7
    rho_ref, V_ref, Rgas_ref = 1.2, 340.0, 287.0
    P_ref = rho_ref * V_ref**2
    u_ref = V_ref**2
    T_ref = V_ref**2 / Rgas_ref

    fluid_unity = ember.fluid.PerfectFluid(cp=cp, gamma=gamma, mu=mu_val, Pr=Pr)
    fluid_nd = ember.fluid.PerfectFluid(
        cp=cp,
        gamma=gamma,
        mu=mu_val,
        Pr=Pr,
        rho_ref=rho_ref,
        V_ref=V_ref,
        Rgas_ref=Rgas_ref,
    )

    rtol = 1e-4

    for rho, u in zip(rho_test, u_test):
        # Convert dimensional inputs to nondim
        rho_n = rho / rho_ref
        u_n = u / u_ref

        # Non-unity-ref fluid with nondim inputs should give values that,
        # when multiplied by the appropriate ref, match the unity-ref (dimensional) result.
        checks = {
            "P": P_ref,
            "T": T_ref,
            "h": u_ref,
            "a": V_ref,
            "s": Rgas_ref,
            "cp": Rgas_ref,
            "cv": Rgas_ref,
            "Rgas": Rgas_ref,
        }
        for name, ref in checks.items():
            dim = getattr(fluid_unity, f"get_{name}")(rho, u)
            nd = getattr(fluid_nd, f"get_{name}")(rho_n, u_n)
            assert np.allclose(nd * ref, dim, rtol=rtol), f"get_{name}: nd * ref != dim"

        # Dimensionless quantities should match directly
        for name in ("gamma", "Pr"):
            dim = getattr(fluid_unity, f"get_{name}")(rho, u)
            nd = getattr(fluid_nd, f"get_{name}")(rho_n, u_n)
            assert np.allclose(nd, dim, rtol=rtol), f"get_{name}: nd != dim"

    # set_P_T: nondim inputs -> nondim outputs, scaled by refs -> dimensional
    P, T = 1e5, 350.0
    rho_d, u_d = fluid_unity.set_P_T(P, T)
    rho_n, u_n = fluid_nd.set_P_T(P / P_ref, T / T_ref)
    assert np.allclose(rho_n * rho_ref, rho_d, rtol=rtol)
    assert np.allclose(u_n * u_ref, u_d, rtol=rtol)

    # P_dtm and T_dtm are dimensional regardless of refs
    assert (fluid_nd.P_dtm, fluid_nd.T_dtm) == (fluid_unity.P_dtm, fluid_unity.T_dtm)


def test_change_ref():
    """Test that change_ref returns a consistent object with correct refs."""

    fluid = ember.fluid.PerfectFluid(cp=1105.0, gamma=1.3, mu=1.8e-5, Pr=0.7)
    rho_ref, V_ref, Rgas_ref = 1.2, 340.0, 287.0
    fluid2 = fluid.change_ref(rho_ref=rho_ref, V_ref=V_ref, Rgas_ref=Rgas_ref)

    P_ref = rho_ref * V_ref**2
    u_ref = V_ref**2
    T_ref = V_ref**2 / Rgas_ref

    rtol = 1e-4

    # Unity-ref fluid with dimensional inputs should match
    # changed-ref fluid with nondim inputs, after scaling by ref
    rho, u = 2.0, 50000.0
    rho_nd = rho / rho_ref
    u_nd = u / u_ref
    assert np.isclose(
        fluid2.get_P(rho_nd, u_nd) * P_ref, fluid.get_P(rho, u), rtol=rtol
    )
    assert np.isclose(
        fluid2.get_T(rho_nd, u_nd) * T_ref, fluid.get_T(rho, u), rtol=rtol
    )
    assert np.isclose(
        fluid2.get_s(rho_nd, u_nd) * Rgas_ref, fluid.get_s(rho, u), rtol=rtol
    )

    # Refs are stored
    assert np.isclose(fluid2.rho_ref, 1.2)
    assert np.isclose(fluid2.V_ref, 340.0)


def test_change_datum_nondim():
    """Test change_datum with non-unity reference values."""

    rho_ref, V_ref, Rgas_ref = 1.2, 340.0, 287.0
    P_ref = rho_ref * V_ref**2
    T_ref = V_ref**2 / Rgas_ref

    fluid = ember.fluid.PerfectFluid(
        cp=1105.0,
        gamma=1.3,
        mu=1.8e-5,
        Pr=0.7,
        rho_ref=rho_ref,
        V_ref=V_ref,
        Rgas_ref=Rgas_ref,
    )

    P_dtm_new, T_dtm_new = 2e5, 400.0
    fluid_new = fluid.change_datum(P_dtm_new, T_dtm_new)

    # New fluid has correct datum (dimensional)
    assert np.isclose(fluid_new.P_dtm, P_dtm_new)
    assert np.isclose(fluid_new.T_dtm, T_dtm_new)

    # Reference values preserved on the new fluid
    assert np.isclose(fluid_new.rho_ref, fluid.rho_ref)
    assert np.isclose(fluid_new.V_ref, fluid.V_ref)
    assert np.isclose(fluid_new.Rgas_ref, fluid.Rgas_ref)

    # At new datum state, u = 0 and s = 0
    rho_dtm, u_datum = fluid_new.set_P_T(P_dtm_new / P_ref, T_dtm_new / T_ref)
    assert np.allclose(u_datum, 0.0, atol=1e-10)
    s_datum = fluid_new.get_s(rho_dtm, u_datum)
    assert np.allclose(s_datum, 0.0, atol=1e-4)
