"""Tests for real-gas coefficient fitting (ember.realgas_fit).

Module tested: ember.realgas_fit

The fitting pipeline turns a table of thermodynamic properties into the
Legendre coefficient arrays that :class:`ember.fluid.RealFluid` evaluates. It is
checked here against analytic equations of state, so the suite needs no CoolProp
and no reference data files.

Test cases:
- test_hat_maps_box_to_unit_interval: box bounds map onto [-1, 1]
- test_hat_is_affine_invariant: pre-scaling the data leaves the hats unchanged
- test_order_matrix_counts: total-order and tensor-grid basis sizes
- test_legfit2d_recovers_known_polynomial: exact fit of a Legendre polynomial
- test_entropy_integral_matches_quadrature: analytic integral vs numerical
- test_entropy_integral_vanishes_at_isochor: definite integral is zero at rho0
- test_entropy_integral_near_low_density: accuracy where ln(rho) is worst
- test_ideal_gas_alpha_is_unit: Z = 1 is represented exactly
- test_ideal_gas_equivalence: RealFluid reproduces the PerfectFluid it was fit to
- test_van_der_waals_equivalence: RealFluid reproduces an analytic real gas
- test_sample_coolprop_reports_states_on_the_model: sampler output matches source
- test_sample_coolprop_drops_unusable_states: two-phase and failed states dropped
- test_sample_coolprop_rejects_an_empty_box: a box with nothing usable raises
- test_fit_rejects_an_isochor_outside_the_box: bad rho_isochor raises
"""

import sys
import types

import numpy as np
import pytest

import ember.fluid
import ember.realgas_fit as rgf
from conftest import VanDerWaals


# ---------------------------------------------------------------------------
# Analytic equations of state used as ground truth for the fits
# ---------------------------------------------------------------------------


# Air-like perfect gas and a dense-vapour-like van der Waals gas.
PERFECT = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
VDW = VanDerWaals()

# Fit boxes. The perfect-gas box is written in absolute internal energy, i.e.
# on PerfectFluid's own datum, which is what the sampler below produces.
RHO_LIM_PERFECT = (0.5, 8.0)
U_LIM_PERFECT = (5.0e4, 4.0e5)
RHO_LIM_VDW = (1.0, 150.0)
U_LIM_VDW = (3.0e5, 5.0e5)

ORDER = 10


def _sample_box(rho_lim, u_lim, ni=40, inset=0.0):
    """Uniform tensor grid of states over a fit box, flattened.

    ``inset`` shrinks the box by a fraction at each edge, to keep comparison
    points away from the bounds where a least-squares fit is always weakest.
    """

    def _axis(lim):
        pad = inset * (lim[1] - lim[0])
        return np.linspace(lim[0] + pad, lim[1] - pad, ni)

    rho_g, u_g = np.meshgrid(_axis(rho_lim), _axis(u_lim), indexing="ij")
    return rho_g.ravel(), u_g.ravel()


def _sample_fluid(fluid, ni=11, inset=0.05):
    """Sample states inside a RealFluid's own valid domain.

    Must go through ``rho_lim_nd``/``u_lim_nd`` rather than the box handed to
    the fitter: the constructor locates the datum by inverting the surface and
    then measures internal energy from it, so the same box sits at different
    numbers once a datum is applied.
    """
    return _sample_box(fluid.rho_lim_nd, fluid.u_lim_nd, ni=ni, inset=inset)


def _fit_perfect(order=ORDER):
    """Fit the pipeline to perfect-gas data and return (FitResult, box)."""
    rho, u = _sample_box(RHO_LIM_PERFECT, U_LIM_PERFECT)
    return rgf.fit(
        rho=rho,
        u=u,
        P=PERFECT.get_P(rho, u),
        T=PERFECT.get_T(rho, u),
        s=PERFECT.get_s(rho, u),
        Rgas=float(PERFECT.get_Rgas(1.0, 1.0)),
        rho_lim=RHO_LIM_PERFECT,
        u_lim=U_LIM_PERFECT,
        rho_isochor=float(np.mean(RHO_LIM_PERFECT)),
        order=order,
    )


def _fit_vdw(order=ORDER):
    """Fit the pipeline to van der Waals data."""
    rho, u = _sample_box(RHO_LIM_VDW, U_LIM_VDW)
    return rgf.fit(
        rho=rho,
        u=u,
        P=VDW.get_P(rho, u),
        T=VDW.get_T(rho, u),
        s=VDW.get_s(rho, u),
        Rgas=VDW.Rgas,
        rho_lim=RHO_LIM_VDW,
        u_lim=U_LIM_VDW,
        rho_isochor=float(np.mean(RHO_LIM_VDW)),
        order=order,
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_hat_maps_box_to_unit_interval():
    """The box bounds map onto exactly -1 and +1."""

    lim = (0.01, 260.0)
    assert np.isclose(rgf.hat(lim[0], lim), -1.0)
    assert np.isclose(rgf.hat(lim[1], lim), 1.0)
    assert np.isclose(rgf.hat(np.mean(lim), lim), 0.0)

    # Round-trip through the inverse
    x = np.linspace(*lim, 17)
    assert np.allclose(rgf.unhat(rgf.hat(x, lim), lim), x)


def test_hat_is_affine_invariant():
    """Pre-scaling and offsetting the data leaves the normalised coordinate alone.

    This is why the fit box cannot be replaced by ember's datum and reference
    scales, and equally why applying them before fitting would buy nothing: any
    affine change of units is absorbed exactly by the min/max normalisation.
    """

    x = np.linspace(3.0, 91.0, 23)
    lim = (x.min(), x.max())

    for scale, offset in ((2.5, 0.0), (1.0, -17.0), (0.03, 400.0)):
        xp = scale * x + offset
        lim_p = (xp.min(), xp.max())
        assert np.allclose(rgf.hat(xp, lim_p), rgf.hat(x, lim))


# ---------------------------------------------------------------------------
# Basis and fitting
# ---------------------------------------------------------------------------


def test_order_matrix_counts():
    """Total-order and tensor-grid bases retain the expected term counts."""

    order = 3

    tensor = rgf.order_matrix(order, basis="tensor-grid")
    assert tensor.shape == (order + 1, order + 1)
    assert tensor.sum() == (order + 1) ** 2

    total = rgf.order_matrix(order, basis="total-order")
    assert total.shape == (order + 1, order + 1)
    # Terms with i + j <= order only: (n+1)(n+2)/2
    assert total.sum() == (order + 1) * (order + 2) // 2
    assert total[0, 0] and total[order, 0] and total[0, order]
    assert not total[order, order]

    with pytest.raises(ValueError):
        rgf.order_matrix(order, basis="not-a-basis")


def test_legfit2d_recovers_known_polynomial():
    """A fit to data generated from Legendre coefficients recovers them."""

    rng = np.random.default_rng(17)
    coef = np.zeros((4, 4))
    mask = rgf.order_matrix(3, basis="total-order")
    coef[mask] = rng.standard_normal(mask.sum())

    x = np.linspace(-1.0, 1.0, 31)
    y = np.linspace(-1.0, 1.0, 29)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    z = np.polynomial.legendre.legval2d(xg, yg, coef)

    fitted, info = rgf.legfit2d(
        xg.ravel(), yg.ravel(), z.ravel(), order=3, basis="total-order"
    )

    assert np.allclose(fitted, coef, atol=1e-10)
    assert info.rmse < 1e-10
    assert info.R2 > 1.0 - 1e-12


# ---------------------------------------------------------------------------
# The entropy integral -- the highest-risk derivation
# ---------------------------------------------------------------------------


def _quad_entropy_integral(alpha, c, x0, x, y, n=400):
    """Reference integral(Z dln rho) from x0 to x, by Gauss-Legendre quadrature.

    Integrating in ``t = ln(x + c)`` rather than in ``x`` removes the
    ``1/(x + c)`` factor, leaving a smooth integrand that Gauss-Legendre
    resolves to near machine precision with a modest number of nodes. That
    matters because a fit box reaching down to low density puts the singularity
    just outside the interval, where a fixed-step rule converges far too slowly
    to be a credible reference.
    """
    t0, t1 = np.log(x0 + c), np.log(x + c)
    nodes, weights = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (t1 - t0) * nodes + 0.5 * (t1 + t0)
    xs = np.exp(t) - c
    Z = np.polynomial.legendre.legval2d(xs, np.full_like(xs, y), alpha)
    return 0.5 * (t1 - t0) * np.sum(weights * Z)


@pytest.mark.parametrize("c", [1.00008, 1.418, 3.0])
def test_entropy_integral_matches_quadrature(c):
    """The closed-form entropy integral agrees with numerical quadrature."""

    rng = np.random.default_rng(3)
    alpha = np.zeros((4, 4))
    mask = rgf.order_matrix(3, basis="total-order")
    alpha[mask] = rng.standard_normal(mask.sum())

    x0 = 0.0  # isochor at the box centre
    D, Lam = rgf.entropy_integral(alpha, c, x0)

    for y in (-0.9, -0.2, 0.5, 1.0):
        for x in (-0.95, -0.5, 0.0, 0.37, 1.0):
            got = np.polynomial.legendre.legval2d(
                x, y, D
            ) + np.polynomial.legendre.legval(y, Lam) * np.log(x + c)
            expect = _quad_entropy_integral(alpha, c, x0, x, y)
            assert np.isclose(got, expect, rtol=1e-8, atol=1e-10), (
                f"c={c} x={x} y={y}: {got} != {expect}"
            )


def test_entropy_integral_vanishes_at_isochor():
    """The definite integral is zero at the reference isochor, for every energy."""

    rng = np.random.default_rng(5)
    alpha = rng.standard_normal((4, 4))
    c, x0 = 1.2, -0.3
    D, Lam = rgf.entropy_integral(alpha, c, x0)

    y = np.linspace(-1.0, 1.0, 11)
    val = np.polynomial.legendre.legval2d(
        np.full_like(y, x0), y, D
    ) + np.polynomial.legendre.legval(y, Lam) * np.log(x0 + c)
    assert np.allclose(val, 0.0, atol=1e-12)


def test_entropy_integral_near_low_density():
    """Accuracy holds at the low-density edge, where ln(rho) is nearly singular.

    For a box reaching down to rho ~ 0 the logarithmic singularity at x = -c
    sits just outside the domain, which is the worst case for this term.
    """

    rho_lim = (0.01, 260.0)
    rho_m = 0.5 * (rho_lim[1] + rho_lim[0])
    rho_f = 0.5 * (rho_lim[1] - rho_lim[0])
    c = rho_m / rho_f

    rng = np.random.default_rng(11)
    alpha = np.zeros((3, 3))
    alpha[rgf.order_matrix(2, basis="total-order")] = rng.standard_normal(6)

    D, Lam = rgf.entropy_integral(alpha, c, 0.0)

    x = rgf.hat(0.02, rho_lim)  # just above the low-density bound
    for y in (-1.0, 0.0, 1.0):
        got = np.polynomial.legendre.legval2d(x, y, D) + np.polynomial.legendre.legval(
            y, Lam
        ) * np.log(x + c)
        expect = _quad_entropy_integral(alpha, c, 0.0, x, y, n=800)
        assert np.isclose(got, expect, rtol=1e-11)


# ---------------------------------------------------------------------------
# End-to-end equivalence against analytic equations of state
# ---------------------------------------------------------------------------


def test_ideal_gas_alpha_is_unit():
    """For a perfect gas Z = 1 exactly, so only the constant term survives.

    Independent of the logarithm approximation that limits the entropy fit, so
    it is a sharp check on the normalisation and basis machinery alone.
    """

    result = _fit_perfect()
    alpha = result.kwargs["alpha"]

    assert np.isclose(alpha[0, 0], 1.0, atol=1e-10)
    rest = alpha.copy()
    rest[0, 0] = 0.0
    assert np.allclose(rest, 0.0, atol=1e-10)
    assert result.rmse_Z < 1e-12


def test_ideal_gas_equivalence():
    """A RealFluid fitted to perfect-gas data reproduces that perfect gas."""

    result = _fit_perfect()

    # The entropy fit approximates a logarithm, so it is not exact. Drive the
    # tolerance from the measured residual rather than guessing, and assert the
    # residual separately so a fitting regression cannot hide inside a loose
    # tolerance. Temperature comes from the fit's derivative, which converges
    # one order slower than its value, hence the margin.
    # The entropy fit approximates a logarithm and so is never exact. Assert the
    # residual separately, so a fitting regression is caught rather than hidden
    # inside a loose comparison tolerance.
    assert result.rmse_s < 1e-7, f"entropy fit residual {result.rmse_s}"

    # Two tolerances, because error grows every time the surface is
    # differentiated. Measured at this order: P, T, s and h land within 4e-7,
    # the speed of sound and isentropic exponent within 3e-6, and the specific
    # heats -- which need second derivatives -- within 1.1e-5. Each bound below
    # carries roughly twenty times that margin.
    rtol = {name: 1e-5 for name in ("P", "T", "s", "h")}
    rtol.update({name: 1e-4 for name in ("a", "gamma", "cp", "cv")})

    # The datum must lie inside the fit box, since it is located by inverting
    # the surface. T = 400 K puts u near the middle of U_LIM_PERFECT.
    fluid = ember.fluid.RealFluid(
        mu=1.8e-5,
        Pr=0.72,
        P_dtm=1e5,
        T_dtm=400.0,
        **result.kwargs,
    )
    ref = PERFECT.change_datum(1e5, 400.0)
    # Both fluids now share a datum, so states line up and every property can
    # be compared without an offset.
    rho, u = _sample_fluid(fluid)

    for name in ("P", "T", "s", "h", "a", "cp", "cv", "gamma"):
        got = getattr(fluid, f"get_{name}")(rho, u)
        expect = getattr(ref, f"get_{name}")(rho, u)
        tol = rtol[name]
        # Entropy and enthalpy vanish at the datum, so a purely relative
        # comparison is meaningless there; scale the floor by the spread.
        atol = tol * float(np.abs(expect).max())
        assert np.allclose(got, expect, rtol=tol, atol=atol), (
            f"get_{name} disagrees: max rel "
            f"{np.abs(got - expect).max() / max(np.abs(expect).max(), 1e-30):.2e}"
        )


def test_van_der_waals_equivalence():
    """A RealFluid fitted to van der Waals data reproduces that gas.

    Unlike the ideal-gas case the compressibility factor varies across the box,
    so this is the test that actually exercises the entropy integral.
    """

    result = _fit_vdw()

    # Unlike the ideal gas, Z is a genuine two-dimensional surface here, so its
    # residual is a real measurement rather than a formality. Assert both
    # residuals separately from the property comparison, so a fitting regression
    # cannot hide inside a loose tolerance.
    assert result.rmse_Z < 1e-6, f"compressibility fit residual {result.rmse_Z}"
    assert result.rmse_s < 1e-6, f"entropy fit residual {result.rmse_s}"
    rtol = 1e-4

    fluid = ember.fluid.RealFluid(
        mu=1.0e-5,
        Pr=1.0,
        P_dtm=1e6,
        T_dtm=300.0,
        **result.kwargs,
    )

    rho, u = _sample_fluid(fluid)
    # The analytic model works in absolute internal energy; the fluid measures
    # it from the datum, so shift back before evaluating the reference.
    u_abs = u + _vdw_u_datum(fluid)

    # P and T are absolute, so they compare directly against the analytic model.
    assert np.allclose(fluid.get_P(rho, u), VDW.get_P(rho, u_abs), rtol=rtol)
    assert np.allclose(fluid.get_T(rho, u), VDW.get_T(rho, u_abs), rtol=rtol)

    # Entropy carries a datum offset, so only differences are meaningful --
    # compare against the analytic model relative to one state.
    s_got = fluid.get_s(rho, u) - fluid.get_s(rho[0], u[0])
    s_exp = VDW.get_s(rho, u_abs) - VDW.get_s(rho[0], u_abs[0])
    assert np.allclose(s_got, s_exp, atol=rtol * np.abs(s_exp).max())


def _vdw_u_datum(fluid):
    """Absolute internal energy corresponding to u = 0 on a fitted fluid."""
    # u_lim_nd is the fit box shifted onto the fluid's datum, so the offset is
    # the difference between the two representations of the same bound.
    return U_LIM_VDW[0] - fluid.u_lim_nd[0]


# ---------------------------------------------------------------------------
# Sampling reference data
# ---------------------------------------------------------------------------

_R_UNIVERSAL = 8.314462618  # J/mol/K

# Sentinel for CoolProp's DmassUmass_INPUTS flag. Its value is meaningless
# here; what matters is that the sampler passes that flag and not another.
_DMASS_UMASS_INPUTS = object()


class _StubState:
    """Stand-in for ``CoolProp.AbstractState``, backed by :data:`VDW`.

    Only the handful of methods the sampler calls are implemented. Two bands of
    the box are made awkward on purpose: states denser than ``rho_dome`` report
    a vapour quality inside the two-phase dome, and states below ``rho_fail``
    refuse to converge. Those are the two cases the sampler exists to drop, and
    with the real library they can only be reached by finding a fluid and a box
    that happen to straddle saturation -- which would test CoolProp's data
    rather than the sampler's own logic.
    """

    rho_dome = 100.0
    rho_fail = 5.0

    def __init__(self, backend, fluid_name):
        self.backend = backend
        self.fluid_name = fluid_name
        self._rho = None
        self._u = None

    def gas_constant(self):
        return _R_UNIVERSAL

    def molar_mass(self):
        return _R_UNIVERSAL / VDW.Rgas

    def update(self, inputs, rho, u):
        assert inputs is _DMASS_UMASS_INPUTS
        if rho < self.rho_fail:
            raise ValueError("state did not converge")
        self._rho, self._u = rho, u

    def Q(self):
        # Outside the dome CoolProp reports a negative quality, not an error.
        return 0.5 if self._rho > self.rho_dome else -1.0

    def p(self):
        return VDW.get_P(self._rho, self._u)

    def T(self):
        return VDW.get_T(self._rho, self._u)

    def smass(self):
        return VDW.get_s(self._rho, self._u)


@pytest.fixture
def stub_coolprop(monkeypatch):
    """Make ``from CoolProp import CoolProp`` find :class:`_StubState`.

    Returns the stub module, so a test can read the thresholds it will apply
    without reaching around the fixture for them.
    """
    pkg = types.ModuleType("CoolProp")
    mod = types.ModuleType("CoolProp.CoolProp")
    mod.AbstractState = _StubState
    mod.DmassUmass_INPUTS = _DMASS_UMASS_INPUTS
    pkg.CoolProp = mod
    monkeypatch.setitem(sys.modules, "CoolProp", pkg)
    monkeypatch.setitem(sys.modules, "CoolProp.CoolProp", mod)
    return mod


def _expected_keep(state, rho_lim, u_lim, ni):
    """Mask of the sampled grid that ``state`` is willing to report."""
    rho_g, _ = np.meshgrid(
        np.linspace(*rho_lim, ni), np.linspace(*u_lim, ni), indexing="ij"
    )
    rho = rho_g.ravel()
    return (rho >= state.rho_fail) & (rho <= state.rho_dome)


def test_sample_coolprop_reports_states_on_the_model(stub_coolprop):
    """Sampled properties are those of the fluid, at the states returned."""
    out = rgf.sample_coolprop("VanDerWaals", RHO_LIM_VDW, U_LIM_VDW, ni=12)

    # Rgas is specific, so the sampler has to divide the universal constant by
    # the molar mass rather than pass either one straight through.
    assert out["Rgas"] == pytest.approx(VDW.Rgas)

    rho, u = out["rho"], out["u"]
    assert np.allclose(out["P"], VDW.get_P(rho, u))
    assert np.allclose(out["T"], VDW.get_T(rho, u))
    assert np.allclose(out["s"], VDW.get_s(rho, u))

    # Every array describes the same states, and there are some.
    assert rho.size == u.size == out["P"].size == out["T"].size == out["s"].size
    assert rho.size > 0
    assert stub_coolprop.AbstractState is _StubState


def test_sample_coolprop_drops_unusable_states(stub_coolprop):
    """Two-phase and non-converging states are dropped, and nothing else is.

    A nan left in the table would be quietly fatal to the least-squares fit
    downstream, and dropping valid states would thin the fit box without
    saying so, so both directions are checked.
    """
    ni = 12
    state = stub_coolprop.AbstractState
    out = rgf.sample_coolprop("VanDerWaals", RHO_LIM_VDW, U_LIM_VDW, ni=ni)
    keep = _expected_keep(state, RHO_LIM_VDW, U_LIM_VDW, ni)

    assert keep.sum() < keep.size, "box does not span the states meant to be cut"
    assert out["rho"].size == keep.sum()
    assert (out["rho"] >= state.rho_fail).all()
    assert (out["rho"] <= state.rho_dome).all()
    for key in ("P", "T", "s"):
        assert np.isfinite(out[key]).all()


def test_sample_coolprop_rejects_an_empty_box(stub_coolprop):
    """A box lying wholly inside the dome raises rather than fitting nothing."""
    rho_dome = stub_coolprop.AbstractState.rho_dome
    with pytest.raises(ValueError, match="No valid single-phase states"):
        rgf.sample_coolprop(
            "VanDerWaals", (2.0 * rho_dome, 3.0 * rho_dome), U_LIM_VDW, ni=5
        )


def test_fit_rejects_an_isochor_outside_the_box():
    """A reference isochor off the box is refused where the mistake was made.

    The density integral is taken about this isochor, so a non-positive one
    puts a negative number inside a logarithm and every entropy coefficient
    comes back nan. RealFluid rejects such an isochor, but only once the
    coefficients reach it -- by which point the residuals reported by the fit,
    the thing a caller inspects to decide the fit is good, are nan too.
    """
    ni = 12
    rho_g, u_g = np.meshgrid(
        np.linspace(*RHO_LIM_VDW, ni), np.linspace(*U_LIM_VDW, ni), indexing="ij"
    )
    rho, u = rho_g.ravel(), u_g.ravel()
    kwargs = dict(
        rho=rho,
        u=u,
        P=VDW.get_P(rho, u),
        T=VDW.get_T(rho, u),
        s=VDW.get_s(rho, u),
        Rgas=VDW.Rgas,
        rho_lim=RHO_LIM_VDW,
        u_lim=U_LIM_VDW,
        order=6,
    )

    for rho_isochor in (-5.0, 0.0, 10.0 * RHO_LIM_VDW[1]):
        with pytest.raises(ValueError, match="rho_isochor"):
            rgf.fit(rho_isochor=rho_isochor, **kwargs)

    # The bounds themselves are valid isochors, and still fit cleanly.
    for rho_isochor in RHO_LIM_VDW:
        assert np.isfinite(rgf.fit(rho_isochor=rho_isochor, **kwargs).rmse_s)
