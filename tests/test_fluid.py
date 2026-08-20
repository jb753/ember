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
- test_real_fluid_reports_a_solve_that_produced_no_number: a nan solve raises
- test_real_fluid_rejects_density_outside_the_fit_box: given rho is checked
- test_real_fluid_accepts_density_on_the_box_bounds: the bounds are in domain
- test_real_fluid_rejects_more_beta_than_alpha_can_carry: beta size validated
- test_last_nonzero_rows: column trip counts for the sparse contraction
- test_real_fluid_column_counts_cover_every_nonzero: counts miss no coefficient
- test_real_fluid_solves_stop_when_they_stop_improving: solves stop at the floor
- test_real_fluid_partials2_kernel_matches_numpy: stacked kernel vs numpy
- test_real_fluid_partials2_falls_back_off_float32: float64 keeps its precision
- test_real_partials2_kernel_evaluates_every_term: kernel vs formula, all terms
- test_real_fluid_returns_the_state_it_verified: answer matches what was checked
- test_real_f_fu_kernel_evaluates_every_term: scalar-solve kernel vs formula
"""

import dataclasses

import ember.fluid
import ember.fortran
import numpy as np
import pytest


@dataclasses.dataclass(frozen=True)
class FluidCase:
    """A fluid bundled with sample points that lie inside its valid domain.

    The generic tests below run against every equation of state, but they cannot
    share one set of sample states: a :class:`~ember.fluid.RealFluid` is only
    defined inside the box its coefficients were fitted over, whereas a perfect
    gas is defined everywhere. So the states travel with the fluid instead of
    sitting in module globals.

    Attributes
    ----------
    name : str
        Identifier used as the pytest parameter id.
    fluid : ember.fluid._Fluid
        The equation of state under test.
    rho, u : tuple
        Paired sample densities [kg/m^3] and internal energies [J/kg], covering
        scalars and arrays of assorted shape to exercise broadcasting.
    rho_pt, u_pt : float
        A single in-domain state, filled into pre-allocated arrays by the
        ``out=`` tests.
    P_pt : float
        In-domain pressure used with the datum temperature.
    T_off : float
        Temperature offset from ``T_dtm`` that stays in domain.
    datum_new : tuple
        ``(P, T)`` for :meth:`~ember.fluid._Fluid.change_datum`, in domain.
    P_acc, rho_acc : tuple
        Pressure and density sweeps for the ``set_P_rho`` accuracy test.
    u_atol : float
        Absolute tolerance on internal energy [J/kg]. Internal energy is
        measured from the datum and so passes through zero there, where a
        relative comparison says nothing. A perfect gas inverts analytically and
        hits the datum exactly; a real gas gets there by Newton iteration and
        stalls at the precision of the arithmetic, so it needs real headroom.
    """

    name: str
    fluid: object
    rho: tuple
    u: tuple
    rho_pt: float
    u_pt: float
    P_pt: float
    T_off: float
    datum_new: tuple
    P_acc: tuple
    rho_acc: tuple
    u_atol: float


# Sample states shared by the perfect-gas cases, which are valid everywhere.
_RHO_PERFECT = (
    1.0,
    5.0,
    2.2,
    np.array([1.6]),
    np.array([1.0, 2.0]),
    np.array([[1.0, 2.0], [3.0, 4.0]]),
)
_U_PERFECT = (
    30000.0,
    1000.0,
    2200.0,
    np.array([1600.0]),
    np.array([1000.0, 2100.0]),
    np.array([[1000.0, 2000.0], [3000.0, 4000.0]]),
)


def _perfect_case(name, **kwargs):
    """Build a FluidCase for a perfect gas, which is valid over all states."""
    return FluidCase(
        name=name,
        fluid=ember.fluid.PerfectFluid(**kwargs),
        rho=_RHO_PERFECT,
        u=_U_PERFECT,
        rho_pt=2.5,
        u_pt=8e4,
        P_pt=1e5,
        T_off=100.0,
        datum_new=(2e5, 400.0),
        P_acc=(1e3, 1e5, 1e6, 5e6),
        rho_acc=(0.1, 1.0, 5.0, 10.0),
        u_atol=1e-10,
    )


def _real_case(name, rho_lim, u_lim):
    """Build a FluidCase for a RealFluid fitted to an analytic van der Waals gas.

    Every sample point is derived from the fluid's own valid domain rather than
    written out as literals. The fit box is meaningless outside its bounds, and
    the bounds move when the datum does, so hard-coded states would silently
    drift out of range.
    """
    from conftest import VanDerWaals, fit_real_fluid

    fluid = fit_real_fluid(VanDerWaals(), rho_lim, u_lim)
    rho_box, u_box = fluid.rho_lim_nd, fluid.u_lim_nd

    def _inside(frac):
        """A state at a given fraction across the box, 0 being the low corner."""
        return (
            rho_box[0] + frac * (rho_box[1] - rho_box[0]),
            u_box[0] + frac * (u_box[1] - u_box[0]),
        )

    # Not 0.5. fit_real_fluid centres the datum in the box, so the exact
    # middle is the state where u is zero by construction -- and a test that
    # sweeps a fraction either side of its sample state then sweeps a fraction
    # of nothing, leaving np.gradient to divide by a spacing of zero. It only
    # ever worked because the datum landed an ulp off centre; anything that
    # moves it by an ulp, in the solve or in the fit, lands on it exactly.
    rho_mid, u_mid = _inside(0.55)
    rho_lo, u_lo = _inside(0.3)
    rho_hi, u_hi = _inside(0.7)

    # set_P_rho is exercised over the cross product of these two sweeps, so
    # every pressure must be reachable at every density in it. That reachable
    # interval is the intersection of the per-density pressure ranges, not a
    # pair of box corners, and it is only non-empty when the density band is
    # narrow relative to the energy span: pressure climbs about as fast with one
    # as with the other, so breadth in density has to be paid for in energy.
    rho_acc = np.linspace(_inside(0.45)[0], _inside(0.55)[0], 3)
    u_min, u_max = _inside(0.05)[1], _inside(0.95)[1]
    P_lo = max(float(fluid.get_P(r, u_min)) for r in rho_acc)
    P_hi = min(float(fluid.get_P(r, u_max)) for r in rho_acc)
    assert P_hi > P_lo, (
        f"{name}: no pressure is reachable at every density in the sweep "
        f"({P_lo:.4g} > {P_hi:.4g}); narrow the density band."
    )
    P_acc = np.linspace(P_lo, P_hi, 3)

    return FluidCase(
        name=name,
        fluid=fluid,
        rho=(rho_mid, rho_lo, np.array([rho_hi]), np.full((2, 2), rho_mid)),
        u=(u_mid, u_lo, np.array([u_hi]), np.full((2, 2), u_mid)),
        rho_pt=rho_mid,
        u_pt=u_mid,
        P_pt=float(fluid.P_dtm),
        # A temperature step small enough that the resulting energy stays well
        # inside the box: cv is order 1e3, so 20 K moves u by a fifth of it.
        T_off=20.0,
        datum_new=(
            float(fluid.get_P(rho_hi, u_hi)),
            float(fluid.get_T(rho_hi, u_hi)),
        ),
        P_acc=tuple(P_acc),
        rho_acc=tuple(rho_acc),
        # Newton stalls around 1e-7 relative in single precision; this is
        # about four times the worst observed miss over the box.
        u_atol=1e-6 * 0.5 * (u_box[1] - u_box[0]),
    )


CASES = [
    _perfect_case("perfect-ga1.3", cp=1105.0, gamma=1.3, mu=1.8e-5, Pr=0.7),
    _perfect_case(
        "perfect-lowdtm",
        cp=1051.0,
        gamma=1.4,
        mu=3.8e-5,
        Pr=1.0,
        P_dtm=5e4,
        T_dtm=200.0,
    ),
    _perfect_case(
        "perfect-hidtm", cp=1001.0, gamma=1.36, mu=2.8e-5, Pr=0.6, T_dtm=600.0
    ),
    _real_case("real-vdw", rho_lim=(1.0, 150.0), u_lim=(3.0e5, 5.0e5)),
]

CASE_IDS = [case.name for case in CASES]

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


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_universal_relations(case):
    """Test thermodynamic relations valid for any fluid."""

    fluid = case.fluid
    rtol = 1e-5  # exact algebraic identities, loosened only for float32

    for rho, u in zip(case.rho, case.u):
        P = fluid.get_P(rho, u)
        T = fluid.get_T(rho, u)

        # Definition of enthalpy
        h = fluid.get_h(rho, u)
        assert np.allclose(h, u + P / rho)

        # Definition of the isentropic exponent, a^2 = gamma p / rho. For a
        # perfect gas gamma is also cp/cv, but for a real gas the two differ and
        # it is this one that governs acoustics -- see get_gamma's docstring.
        a = fluid.get_a(rho, u)
        assert np.allclose(a**2, fluid.get_gamma(rho, u) * P / rho, rtol=rtol)

        # Thermodynamic consistency, dh = T ds + dp/rho, evaluated at constant
        # density and then at constant pressure. These cross-check the entropy
        # path against the enthalpy path, so they catch an equation of state
        # whose s and h have drifted out of step with each other.
        #
        # Note the trap: dhdP_rho = dudP_rho + 1/rho follows straight from
        # h = u + p/rho and is how these derivatives are usually implemented,
        # so asserting *that* would be a tautology. The T ds form below is
        # independent of the route taken.
        assert np.allclose(
            fluid.get_dhdP_rho(rho, u),
            T * fluid.get_dsdP_rho(rho, u) + 1.0 / rho,
            rtol=rtol,
        )
        assert np.allclose(
            fluid.get_dhdrho_P(rho, u),
            T * fluid.get_dsdrho_P(rho, u),
            rtol=rtol,
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_universal_relations_numerical(case):
    """Thermodynamic identities that need a numerical derivative to check."""

    fluid = case.fluid
    rtol = 1e-3  # numerical derivatives, as in test_derivatives

    for rho, u in zip(case.rho, case.u):
        rho0 = np.mean(rho)
        u0 = np.mean(u)

        # (ds/du)_rho = 1/T, the definition of temperature.
        u_vec = np.linspace(0.9, 1.1) * u0
        dsdu = np.gradient(fluid.get_s(rho0, u_vec), u_vec)
        T_vec = fluid.get_T(rho0, u_vec)
        assert np.allclose(dsdu[1:-1], 1.0 / T_vec[1:-1], rtol=rtol)

        # a^2 = (dP/drho)_s, marching along an isentrope through the state.
        s0 = fluid.get_s(rho0, u0)
        rho_vec = np.linspace(0.9, 1.1) * rho0
        u_isen = fluid.set_rho_s(rho_vec, s0)[1]
        dPdrho = np.gradient(fluid.get_P(rho_vec, u_isen), rho_vec)
        a_vec = fluid.get_a(rho_vec, u_isen)
        assert np.allclose(dPdrho[1:-1], a_vec[1:-1] ** 2, rtol=rtol)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_perfect_gamma_is_cp_over_cv(case):
    """For a perfect gas only, the isentropic exponent equals cp/cv.

    Not a universal relation: :meth:`~ember.fluid._Fluid.get_gamma` returns the
    isentropic exponent, which for a real gas differs from the ratio of specific
    heats. The two coincide when cp and cv are constant.
    """

    fluid = case.fluid
    if not isinstance(fluid, ember.fluid.PerfectFluid):
        pytest.skip("cp/cv equals the isentropic exponent only for a perfect gas")

    for rho, u in zip(case.rho, case.u):
        gamma = fluid.get_gamma(rho, u)
        cv = fluid.get_cv(rho, u)
        cp = fluid.get_cp(rho, u)
        assert np.allclose(gamma, cp / cv)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_get_set_pairs(case):
    """Test the get and set pairs for the fluid properties."""

    fluid = case.fluid
    for rho, u in zip(case.rho, case.u):
        h = fluid.get_h(rho, u)
        s = fluid.get_s(rho, u)
        P = fluid.get_P(rho, u)
        T = fluid.get_T(rho, u)

        rtol = 1e-4  # Original tolerance maintained for round-trip tests

        def _check(name, got):
            # Density and energy are checked separately: density is strictly
            # positive so a relative test is sound, while energy is measured
            # from the datum and vanishes there, needing an absolute floor.
            rho_got, u_got = got
            assert np.allclose(rho_got, rho, rtol=rtol), f"set_{name}: density"
            assert np.allclose(u_got, u, rtol=rtol, atol=case.u_atol), (
                f"set_{name}: internal energy"
            )

        _check("h_s", fluid.set_h_s(h, s))
        _check("P_T", fluid.set_P_T(P, T))
        _check("P_s", fluid.set_P_s(P, s))
        _check("P_h", fluid.set_P_h(P, h))
        _check("P_rho", fluid.set_P_rho(P, rho))
        _check("T_rho", fluid.set_T_rho(T, rho))
        _check("T_s", fluid.set_T_s(T, s))
        _check("rho_s", fluid.set_rho_s(rho, s))


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_internal_energy_datum(case):
    """Test the internal energy datum for the fluid properties."""

    fluid = case.fluid
    T_dtm = fluid.T_dtm
    # Use a temperature above T_dtm to have a non-zero u
    T_test = T_dtm + case.T_off
    rho, u = fluid.set_P_T(case.P_pt, T_test)
    assert np.isclose(fluid.get_T(rho, u), T_test)

    # Test that internal energy is zero at the datum temperature
    _, u_datum = fluid.set_P_T(case.P_pt, T_dtm)
    assert np.isclose(u_datum, 0.0, atol=case.u_atol)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_derivatives(case):
    """Test the derivatives of the fluid properties."""

    fluid = case.fluid
    rtol = 1e-3  # Tolerance for numerical derivatives (relaxed for float32)

    for rho, u in zip(case.rho, case.u):
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

        # Evaluate the analytic derivatives at the same trimmed states as the
        # numerical ones, rather than slicing their result. A perfect gas
        # returns a scalar here -- these derivatives do not depend on u at all,
        # so with a scalar density there is nothing to index.
        u_mid = u_vec[1:-1]
        assert np.allclose(dsdp[1:-1], fluid.get_dsdP_rho(rho0, u_mid), rtol=rtol)
        assert np.allclose(dhdp[1:-1], fluid.get_dhdP_rho(rho0, u_mid), rtol=rtol)
        assert np.allclose(dudp[1:-1], fluid.get_dudP_rho(rho0, u_mid), rtol=rtol)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_get_functions_out_parameter(case):
    """Test that out= parameter writes correct results into pre-allocated array."""

    fluid = case.fluid
    rho = np.full((3, 4), case.rho_pt, order="F", dtype=np.float32)
    u = np.full((3, 4), case.u_pt, order="F", dtype=np.float32)

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


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_get_functions_out_3d(case):
    """Test out= parameter with 3D F-contiguous arrays."""

    fluid = case.fluid
    rho = np.full((3, 4, 2), case.rho_pt, order="F", dtype=np.float32)
    u = np.full((3, 4, 2), case.u_pt, order="F", dtype=np.float32)

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


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_change_datum(case):
    """Test change_datum method returns fluid with correct datum."""

    fluid = case.fluid
    P_dtm_new, T_dtm_new = case.datum_new

    fluid_new = fluid.change_datum(P_dtm_new, T_dtm_new)

    # Check new fluid has correct datum
    assert (fluid_new.P_dtm, fluid_new.T_dtm) == (
        np.float32(P_dtm_new),
        np.float32(T_dtm_new),
    )

    # Check that fluid properties are preserved. They must be compared at the
    # same *physical* state, reached through each fluid's own datum: shifting
    # the datum re-labels internal energy, so feeding both fluids the same u
    # would ask them about different conditions. That distinction is invisible
    # for a perfect gas, whose properties are constant, and decisive for a real
    # one, whose are not.
    rho, u = fluid_new.set_P_T(P_dtm_new, T_dtm_new)
    rho_old, u_old = fluid.set_P_T(P_dtm_new, T_dtm_new)
    rtol = 1e-4
    for name in ("cp", "gamma", "Rgas", "mu", "Pr"):
        new_val = getattr(fluid_new, f"get_{name}")(rho, u)
        old_val = getattr(fluid, f"get_{name}")(rho_old, u_old)
        assert np.allclose(new_val, old_val, rtol=rtol), f"get_{name} not preserved"

    # At the new datum, u=0 and s=0
    assert np.allclose(u, 0.0, atol=case.u_atol)
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


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_set_P_rho_accuracy(case):
    """Test numerical accuracy of set_P_rho implementation.

    Density preservation and the pressure round-trip hold for any equation of
    state. The ideal gas law check that used to live here moved to
    :func:`test_perfect_set_P_rho_ideal_gas_law`, since it is only true when
    the compressibility factor is one.
    """

    fluid = case.fluid
    P_test = np.array(case.P_acc)  # Pa
    rho_test = np.array(case.rho_acc)  # kg/m^3

    max_pressure_error = 0.0

    for P in P_test:
        for rho in rho_test:
            # Use set_P_rho to get (rho, u)
            rho_out, u_out = fluid.set_P_rho(P, rho)

            # Verify density is preserved exactly
            assert np.allclose(rho_out, rho, rtol=1e-15)

            # Verify round-trip accuracy: P_rho -> get_P
            P_roundtrip = fluid.get_P(rho_out, u_out)
            pressure_rel_error = abs(P_roundtrip - P) / P
            max_pressure_error = max(max_pressure_error, pressure_rel_error)

    # Assert tight accuracy bounds to prevent regression
    # Note: tolerances relaxed for float32 Fortran implementation
    assert max_pressure_error < 1e-4, f"Pressure error too large: {max_pressure_error}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_perfect_set_P_rho_ideal_gas_law(case):
    """For a perfect gas only, set_P_rho lands on the ideal gas temperature.

    Split out of :func:`test_set_P_rho_accuracy`: a real gas satisfies
    ``p = Z rho R T`` with ``Z != 1``, so recovering T from ``p/(rho R)`` is a
    perfect-gas assertion, not a universal one.
    """

    fluid = case.fluid
    if not isinstance(fluid, ember.fluid.PerfectFluid):
        pytest.skip("p = rho R T holds only for a perfect gas")

    max_temp_error = 0.0
    for P in np.array(case.P_acc):
        for rho in np.array(case.rho_acc):
            _, u_out = fluid.set_P_rho(P, rho)
            T_computed = fluid.get_T(rho, u_out)
            T_expected = P / (fluid.get_Rgas(rho, u_out) * rho)
            max_temp_error = max(
                max_temp_error, abs(T_computed - T_expected) / T_expected
            )

    assert max_temp_error < 1e-4, f"Temperature error too large: {max_temp_error}"


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


# ---------------------------------------------------------------------------
# RealFluid refuses states its coefficients do not describe
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_fluid():
    """A RealFluid fitted to the analytic van der Waals gas of the suite."""
    from conftest import VanDerWaals, fit_real_fluid

    return fit_real_fluid(VanDerWaals(), (1.0, 150.0), (3.0e5, 5.0e5))


def test_real_fluid_reports_a_solve_that_produced_no_number(real_fluid):
    """A solve that diverges to nan is reported rather than returned.

    The perfect-gas guess for a wildly out-of-range enthalpy takes the log of a
    negative temperature, and the nan that produces survives every Newton step.
    Nothing downstream would notice: a nan compares false against a tolerance,
    so the acceptance test has to ask whether the answer is a number at all
    rather than only whether it is close enough.
    """
    with pytest.raises(RuntimeError, match="did not return a finite"):
        real_fluid.set_h_s(np.array([-1e9]), np.array([0.0]))


@pytest.mark.parametrize("frac", [-0.5, 1.5], ids=["below", "above"])
@pytest.mark.parametrize("method", ["set_rho_s", "set_P_rho", "set_T_rho"])
def test_real_fluid_rejects_density_outside_the_fit_box(real_fluid, method, frac):
    """Density is an input to these solves, so it has to be checked on the way in.

    Only internal energy is iterated here, and it is clipped to the box; the
    density is whatever the caller passed. Outside the box the fitted surface
    still returns a number, and the residual in the other property can still be
    driven to zero against it, so the converged-solve check cannot see the
    problem -- the state is self-consistent with an extrapolation of the fit.
    """
    rho_lo, rho_hi = real_fluid.rho_lim_nd
    rho = np.array([rho_lo + frac * (rho_hi - rho_lo)])

    # The second argument is a property the state must match; take it from a
    # state well inside the box, so only the density is out of range.
    u_mid = 0.5 * (real_fluid.u_lim_nd[0] + real_fluid.u_lim_nd[1])
    rho_mid = 0.5 * (rho_lo + rho_hi)
    val = {
        "set_rho_s": real_fluid.get_s(rho_mid, u_mid),
        "set_P_rho": real_fluid.get_P(rho_mid, u_mid),
        "set_T_rho": real_fluid.get_T(rho_mid, u_mid),
    }[method]
    args = (rho, val) if method == "set_rho_s" else (val, rho)

    with pytest.raises(RuntimeError, match="outside the fit box"):
        getattr(real_fluid, method)(*args)


def test_real_fluid_accepts_density_on_the_box_bounds(real_fluid):
    """The bounds themselves are inside the fitted domain, not outside it."""
    u_lo, u_hi = real_fluid.u_lim_nd
    # Away from the midpoint, which sits at the datum where u passes through
    # zero and a relative comparison would say nothing.
    u_want = u_lo + 0.3 * (u_hi - u_lo)
    for rho in real_fluid.rho_lim_nd:
        s = real_fluid.get_s(rho, u_want)
        _, u = real_fluid.set_rho_s(np.array([rho]), s)
        assert np.allclose(u, u_want, atol=1e-5 * (u_hi - u_lo))


def test_real_fluid_rejects_more_beta_than_alpha_can_carry():
    """An isochor polynomial longer than the surface can hold is an error.

    The entropy surface has one column per internal-energy order in ``alpha``,
    so a longer ``beta`` has nowhere to go. Dropping the excess would leave a
    fluid whose entropy is missing its energy dependence -- every temperature
    and pressure it returns would be wrong, in a way no later call can detect.
    """
    with pytest.raises(ValueError, match="beta"):
        ember.fluid.RealFluid(
            # Z = 1 needs a single term, so alpha has one column while beta,
            # fitted at a higher order, has nine.
            alpha=[[1.0]],
            beta=np.arange(9.0) + 1.0,
            rho_lim=(0.5, 5.0),
            u_lim=(1e5, 3e5),
            rho_isochor=1.0,
            Rgas=287.0,
            mu=1.8e-5,
            Pr=0.72,
        )


def test_last_nonzero_rows():
    """Column trip counts for the kernel's sparse contraction.

    The kernel shortens each column's loop to this, so a count that is too
    small silently drops real coefficients. Trailing zeros are what a
    total-order fit leaves and are meant to be skipped; an interior zero is
    not, because everything below it still has to be visited.
    """
    coef = np.array(
        [
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0],
            [7.0, 0.0, 0.0, 0.0],
        ]
    )
    got = ember.fluid._last_nonzero_rows(coef)
    np.testing.assert_array_equal(got, [4, 2, 1, 0])
    assert got.dtype == np.int32

    # An interior zero must not truncate the column above it.
    interior = np.array([[1.0], [0.0], [3.0], [0.0]])
    np.testing.assert_array_equal(ember.fluid._last_nonzero_rows(interior), [3])

    # A dense surface gives back its full extent, so nothing is skipped.
    dense = np.ones((5, 3))
    np.testing.assert_array_equal(ember.fluid._last_nonzero_rows(dense), [5, 5, 5])


def test_real_fluid_column_counts_cover_every_nonzero():
    """A fitted fluid's counts reach every nonzero coefficient it holds.

    The saving is only sound if the terms below each count are the only ones
    that matter, so this checks the fit's own surfaces rather than a
    constructed example: nothing at or below the count may be missed, and the
    count must actually be shorter than the dense extent or there is no saving
    to have.
    """
    from conftest import VanDerWaals, fit_real_fluid

    fluid = fit_real_fluid(VanDerWaals(), (1.0, 150.0), (3.0e5, 5.0e5))
    for coef, counts in (
        (fluid._Sc_x, fluid._nzx),
        (fluid._Sc_y, fluid._nzy),
    ):
        for b, count in enumerate(counts):
            assert not coef[count:, b].any(), f"column {b} has a nonzero past {count}"
        assert counts.sum() < coef.size, "a total-order fit should leave zeros"


@pytest.mark.parametrize("method", ["set_P_rho", "set_h_s", "set_P_T", "set_rho_s"])
def test_real_fluid_solves_stop_when_they_stop_improving(method):
    """A converged solve stops iterating instead of running to the cap.

    Newton reaches the float32 floor here in two or three steps, and then the
    step cannot get any smaller: asking for one finer than float32 resolves
    means the size test never fires and the solve pays the full iteration
    limit. The answer was always right, just forty iterations late, so nothing
    but a count of the surface evaluations can see it.

    The array has to be patch-sized to show it. The step is a maximum over
    every node, so it is the noisiest one that decides when the loop ends, and
    the more nodes there are the worse that sample gets: the same solve takes
    4 evaluations over 64 nodes, 7 over 2304, and the full 41 over 9409. A
    small test array hides the whole thing.
    """
    from conftest import VanDerWaals, fit_real_fluid

    fluid = fit_real_fluid(VanDerWaals(), (1.0, 150.0), (3.0e5, 5.0e5), order=8)
    rng = np.random.default_rng(0)

    def _span(lim):
        lo = lim[0] + 0.3 * (lim[1] - lim[0])
        hi = lim[0] + 0.7 * (lim[1] - lim[0])
        return rng.uniform(lo, hi, (97, 97)).astype(np.float32)

    rho, u = _span(fluid.rho_lim_nd), _span(fluid.u_lim_nd)
    args = {
        "set_P_rho": (fluid.get_P(rho, u), rho),
        "set_rho_s": (rho, fluid.get_s(rho, u)),
        "set_h_s": (fluid.get_h(rho, u), fluid.get_s(rho, u)),
        "set_P_T": (fluid.get_P(rho, u), fluid.get_T(rho, u)),
    }[method]

    calls = []
    original = ember.fluid.RealFluid._partials2
    ember.fluid.RealFluid._partials2 = lambda self, r, e: (
        calls.append(1),
        original(self, r, e),
    )[1]
    try:
        rho_got, u_got = getattr(fluid, method)(*args)
    finally:
        ember.fluid.RealFluid._partials2 = original

    # Generous: the point is that it stops, not exactly when. A solve running
    # to the limit spends _NEWTON_ITER + 1 here.
    assert len(calls) <= 12, f"{method} took {len(calls)} surface evaluations"

    # And it still lands on the state it was asked for.
    assert np.allclose(rho_got, rho, rtol=1e-4)
    assert np.allclose(u_got, u, atol=1e-4 * (fluid.u_lim_nd[1] - fluid.u_lim_nd[0]))


def test_real_fluid_partials2_kernel_matches_numpy():
    """The six-surface kernel agrees with the numpy evaluation it replaces.

    Both read the same coefficients, but the kernel reads them stacked and
    padded, with a per-column count deciding where each surface ends. A
    mistake in that bookkeeping -- a surface written to the wrong slice, a
    count taken from the wrong column -- would show up here and almost nowhere
    else, since the six partials only ever surface through derived quantities.
    """
    from conftest import VanDerWaals, fit_real_fluid

    fluid = fit_real_fluid(VanDerWaals(), (1.0, 150.0), (3.0e5, 5.0e5), order=8)
    rng = np.random.default_rng(0)

    def _span(lim):
        lo = lim[0] + 0.25 * (lim[1] - lim[0])
        hi = lim[0] + 0.75 * (lim[1] - lim[0])
        return rng.uniform(lo, hi, (13, 11)).astype(np.float32)

    rho, u = _span(fluid.rho_lim_nd), _span(fluid.u_lim_nd)

    got = fluid._partials2(rho, u)
    # float64 inputs take the numpy path, which is the reference.
    want = fluid._partials2(rho.astype(np.float64), u.astype(np.float64))

    names = ("s", "s_r", "s_u", "s_rr", "s_ru", "s_uu")
    for name, g, w in zip(names, got, want):
        scale = float(np.abs(w).max())
        assert np.allclose(g, w, rtol=1e-4, atol=1e-6 * scale), (
            f"{name}: max |kernel - numpy| = {float(np.abs(g - w).max()):.3e} "
            f"against a scale of {scale:.3e}"
        )


def test_real_fluid_partials2_falls_back_off_float32():
    """Anything not float32 takes the numpy path rather than being cast.

    The kernel is single precision throughout, so quietly narrowing a float64
    caller's state to reach it would hand back an answer worse than the one
    numpy would have given, with nothing to show for it.
    """
    from conftest import VanDerWaals, fit_real_fluid

    fluid = fit_real_fluid(VanDerWaals(), (1.0, 150.0), (3.0e5, 5.0e5), order=8)
    rho = np.full((4, 4), 0.5 * sum(fluid.rho_lim_nd), dtype=np.float64)
    u = np.full((4, 4), 0.3 * sum(fluid.u_lim_nd), dtype=np.float64)

    def _must_not_run(**kwargs):
        raise AssertionError("kernel called with float64 state")

    original = ember.fortran.set_partials2_real
    ember.fortran.set_partials2_real = _must_not_run
    try:
        out = fluid._partials2(rho, u)
    finally:
        ember.fortran.set_partials2_real = original

    assert all(np.isfinite(a).all() for a in out)
    assert all(a.dtype == np.float64 for a in out)


def test_real_partials2_kernel_evaluates_every_term():
    """The six-surface kernel against the formula, on all-significant terms.

    A fitted gas cannot do this job, for the same reason it could not for the
    getter kernel and more so: the multiplier on log(rho) is the
    compressibility factor at zero density, which is one for every real gas, so
    its first derivative in energy is ~1e-8 of it and its second smaller
    still. Perturbing the ``Sl_yy`` term of ``s_uu`` by 2% leaves every
    fluid-level assertion in this file passing -- it was tried.

    So these coefficients describe no gas. They are chosen only to put each
    surface and its log partner at the same order, which the ratios below pin,
    and the reference is the formula rather than RealFluid's numpy path. The
    counts are dense, which exercises the unpadded case the fitted fluids do
    not reach.
    """
    leg = np.polynomial.legendre
    rng = np.random.default_rng(3)
    nx, ny = 5, 4
    decay = 0.5 ** np.arange(nx)[:, None, None]
    sc2 = np.asfortranarray(
        (rng.uniform(-1.0, 1.0, (nx, ny, 6)) * decay).astype(np.float32)
    )
    sc1 = np.asfortranarray(rng.uniform(-1.0, 1.0, (ny, 3)).astype(np.float32))
    nz2 = np.asfortranarray(np.full((ny, 6), nx, dtype=np.int32))
    xa, xb, ya, yb = (np.float32(v) for v in (0.5, -1.5, 0.4, -0.2))

    shape = (6, 5)
    rho = rng.uniform(2.0, 5.0, shape).astype(np.float32)
    u = rng.uniform(0.5, 3.0, shape).astype(np.float32)
    x, y, lnr = rho * xa + xb, u * ya + yb, np.log(rho)

    c = [leg.legval2d(x, y, sc2[:, :, m]) for m in range(6)]
    M, My, Myy = (leg.legval(y, sc1[:, m]) for m in range(3))
    want = (
        c[0] + M * lnr,
        c[1] * xa + M / rho,
        (c[2] + My * lnr) * ya,
        c[3] * xa * xa - M / rho**2,
        (c[4] * xa + My / rho) * ya,
        (c[5] + Myy * lnr) * ya * ya,
    )

    names = ("s", "s_r", "s_u", "s_rr", "s_ru", "s_uu")
    for name, poly, log in zip(
        names,
        (c[0], c[1] * xa, c[2], c[3] * xa * xa, c[4] * xa, c[5]),
        (M * lnr, M / rho, My * lnr, M / rho**2, My / rho, Myy * lnr),
    ):
        ratio = float(np.abs(log).max() / np.abs(poly).max())
        assert 0.1 < ratio < 10.0, f"{name}: log term is {ratio:.2e} of the polynomial"

    outs = [np.zeros(rho.size, dtype=np.float32) for _ in range(6)]
    ember.fortran.set_partials2_real(
        rho=np.ravel(rho, order="A"),
        u=np.ravel(u, order="A"),
        sc2=sc2,
        nz2=nz2,
        sc1=sc1,
        xa=xa,
        xb=xb,
        ya=ya,
        yb=yb,
        s=outs[0],
        s_r=outs[1],
        s_u=outs[2],
        s_rr=outs[3],
        s_ru=outs[4],
        s_uu=outs[5],
    )

    for name, got, ref in zip(names, outs, want):
        scale = float(np.abs(ref).max())
        err = float(np.abs(got.reshape(shape) - ref).max())
        assert err <= 8.0 * np.spacing(np.float32(scale)), (
            f"{name}: max error {err:.3e} against a scale of {scale:.3e}"
        )


@pytest.mark.parametrize("method", ["set_P_rho", "set_rho_s", "set_h_s", "set_P_T"])
def test_real_fluid_returns_the_state_it_verified(method):
    """The state handed back is the one the acceptance test looked at.

    The loop stops before applying its last correction rather than after, so
    the residual it checked belongs to the answer it returns. Measuring one
    state and returning another would be almost right -- the discarded step is
    at the arithmetic's floor -- but "almost" is what the acceptance test
    exists to rule out, and it would cost a full walk of every surface to
    re-measure. Recomputing the property here from the returned pair is the
    only way to see which state was actually checked.
    """
    from conftest import VanDerWaals, fit_real_fluid

    fluid = fit_real_fluid(VanDerWaals(), (1.0, 150.0), (3.0e5, 5.0e5), order=8)
    rng = np.random.default_rng(0)

    def _span(lim):
        lo = lim[0] + 0.3 * (lim[1] - lim[0])
        hi = lim[0] + 0.7 * (lim[1] - lim[0])
        return rng.uniform(lo, hi, (48, 48)).astype(np.float32)

    rho, u = _span(fluid.rho_lim_nd), _span(fluid.u_lim_nd)
    targets = {
        "set_P_rho": (("P",), (fluid.get_P(rho, u), rho)),
        "set_rho_s": (("s",), (rho, fluid.get_s(rho, u))),
        "set_h_s": (("h", "s"), (fluid.get_h(rho, u), fluid.get_s(rho, u))),
        "set_P_T": (("P", "T"), (fluid.get_P(rho, u), fluid.get_T(rho, u))),
    }
    props, args = targets[method]

    rho_got, u_got = getattr(fluid, method)(*args)

    # The values asked for, in the order the setter takes them.
    wanted = {
        "set_P_rho": (args[0],),
        "set_rho_s": (args[1],),
        "set_h_s": args,
        "set_P_T": args,
    }[method]

    for prop, want in zip(props, wanted):
        got = getattr(fluid, f"get_{prop}")(rho_got, u_got)
        scale = np.abs(want) + fluid._floor(prop)
        worst = float(np.max(np.abs(got - want) / scale))
        assert worst <= fluid._VERIFY_RTOL, (
            f"{method}: returned state misses {prop} by {worst:.3e}, "
            f"beyond the {fluid._VERIFY_RTOL:.0e} it was accepted at"
        )


@pytest.mark.parametrize(
    "prop,which,sel",
    [("s", 1, [1, 3]), ("T", 2, [3, 6]), ("P", 3, [2, 3, 5, 6])],
)
def test_real_f_fu_kernel_evaluates_every_term(prop, which, sel):
    """The scalar-solve kernel against the formula, on all-significant terms.

    It walks only the surfaces the matched property needs -- two of the six for
    entropy and temperature, four for pressure -- so a wrong index in the
    selection or a wrong line in the closing algebra gives a wrong derivative,
    and a wrong derivative only shows up in a fitted fluid as a solve taking an
    extra iteration or refusing to converge. The reference here is the formula.

    As with the other kernels these coefficients describe no gas: the log terms
    of a fitted one carry the compressibility factor at zero density, which is
    one for every real gas, so its energy derivatives vanish and the terms
    holding them cannot be checked through a fluid at all.
    """
    leg = np.polynomial.legendre
    rng = np.random.default_rng(5)
    nx, ny = 5, 4
    decay = 0.5 ** np.arange(nx)[:, None, None]
    sc2 = np.asfortranarray(
        (rng.uniform(-1.0, 1.0, (nx, ny, 6)) * decay).astype(np.float32)
    )
    sc1 = np.asfortranarray(rng.uniform(-1.0, 1.0, (ny, 3)).astype(np.float32))
    nz2 = np.asfortranarray(np.full((ny, 6), nx, dtype=np.int32))
    sel = np.asfortranarray(np.array(sel, dtype=np.int32))
    xa, xb, ya, yb = (np.float32(v) for v in (0.5, -1.5, 0.4, -0.2))

    shape = (6, 5)
    rho = rng.uniform(2.0, 5.0, shape).astype(np.float32)
    u = rng.uniform(0.5, 3.0, shape).astype(np.float32)
    x, y, lnr = rho * xa + xb, u * ya + yb, np.log(rho)

    def surface(idx):
        return leg.legval2d(x, y, sc2[:, :, idx - 1])

    M, My, Myy = (leg.legval(y, sc1[:, m]) for m in range(3))
    if prop == "s":
        want = (surface(1) + M * lnr, (surface(3) + My * lnr) * ya)
    elif prop == "T":
        s_u = (surface(3) + My * lnr) * ya
        s_uu = (surface(6) + Myy * lnr) * ya * ya
        T = 1.0 / s_u
        want = (T, -T * T * s_uu)
    else:
        s_r = surface(2) * xa + M / rho
        s_u = (surface(3) + My * lnr) * ya
        s_ru = (surface(5) * xa + My / rho) * ya
        s_uu = (surface(6) + Myy * lnr) * ya * ya
        T = 1.0 / s_u
        T_u = -T * T * s_uu
        want = (-(rho**2) * T * s_r, -(rho**2) * (T_u * s_r + T * s_ru))

    # Pin that the log terms carry real weight here, or the test would pass
    # with the one-dimensional contractions deleted.
    for label, poly, log in (("M", surface(1), M * lnr), ("My", surface(3), My * lnr)):
        ratio = float(np.abs(log).max() / np.abs(poly).max())
        assert 0.1 < ratio < 10.0, f"{label}: log term is {ratio:.2e} of the polynomial"

    f = np.zeros(rho.size, dtype=np.float32)
    f_u = np.zeros(rho.size, dtype=np.float32)
    ember.fortran.set_f_fu_real(
        rho=np.ravel(rho, order="A"),
        u=np.ravel(u, order="A"),
        sc2=sc2,
        nz2=nz2,
        sel=sel,
        sc1=sc1,
        xa=xa,
        xb=xb,
        ya=ya,
        yb=yb,
        which=which,
        f=f,
        f_u=f_u,
    )

    for name, got, ref in zip(("f", "f_u"), (f, f_u), want):
        scale = float(np.abs(ref).max())
        err = float(np.abs(got.reshape(shape) - ref).max())
        assert err <= 8.0 * np.spacing(np.float32(scale)), (
            f"{prop} {name}: max error {err:.3e} against a scale of {scale:.3e}"
        )
