r"""Fit real-gas equation of state from tabulated properties.

This module turns a table of thermodynamic properties into the polynomial
coefficient arrays required by :class:`ember.fluid.RealFluid`, following the
entropy-based formulation of :cite:t:`Wheeler2024`.

The user should perform the fitting once, offline, and then pass the resulting
coefficients to :class:`ember.fluid.RealFluid` at simulation runtime. Only
:func:`sample_coolprop` needs a lazy import of CoolProp, so the rest of the
code can be used without it.

The method requires two polynomial fits: compressibility factor as a
two-dimensional surface in density and internal energy, and entropy along a
reference isochor as a one-dimensional function of internal energy. Optionally,
viscosity and thermal conductivity are fitted as two further two-dimensional
surfaces over density and internal energy.

Polynomials are fitted in coordinates scaled onto :math:`[-1, 1]` by the bounds
of a fit box. The reference isochor passes through the centre of the box, and viscosity and conductivity are normalised by their values at the box centre.

Example usage
=============

With CoolProp installed, the pipeline is a single call:

.. code-block:: python

    from ember.realgas_fit import fit, sample_coolprop
    from ember.fluid import RealFluid

    result = fit(
        **sample_coolprop(
            "CO2", rho_lim=(2.0, 150.0), u_lim=(3.5e5, 5.0e5)
        )
    )  # fit the equation of state and transport surfaces

    fluid = RealFluid(**result.kwargs)               # hand to the solver
    print(result.info_Z.rmse, result.info_s.rmse)    # inspect the fit residuals


"""

import dataclasses

import numpy as np

from ember import _realgas_poly as _poly

_leg = np.polynomial.legendre


@dataclasses.dataclass(frozen=True)
class FitInfo:
    """Quality of a single least-squares polynomial fit.

    Attributes
    ----------
    rmse : float
        Root-mean-square residual, in the units of the fitted quantity.
    R2 : float
        Coefficient of determination, one for a perfect fit.
    """

    rmse: float
    R2: float


@dataclasses.dataclass(frozen=True)
class FitResult:
    """Fitted coefficients together with the residuals that bound their accuracy.

    Attributes
    ----------
    kwargs : dict
        Keyword arguments defining the equation of state, ready to splat into
        :class:`ember.fluid.RealFluid`. Contains ``alpha``, ``beta``,
        ``rho_lim``, ``u_lim``, ``Rgas``, ``delta``, ``gamma``, ``mu_c`` and
        ``kappa_c``.
    info_Z : FitInfo
        Residual of the compressibility surface [--].
    info_s : FitInfo
        Residual of the entropy fit, in units of the gas constant [--].
    info_mu, info_kappa : FitInfo
        Residuals of the two transport surfaces, relative to the value each is
        normalised by [--], so that they read as fractional errors.

    """

    kwargs: dict
    info_Z: FitInfo
    info_s: FitInfo
    info_mu: FitInfo
    info_kappa: FitInfo


def _fit_info(y, y_fit):
    """Residual statistics of a fit."""
    resid = np.asarray(y) - np.asarray(y_fit)
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    R2 = 1.0 if ss_tot == 0.0 else 1.0 - float(np.sum(resid**2)) / ss_tot
    return FitInfo(rmse=rmse, R2=R2)


def _fit_normalised(x, y, z, order, basis, name):
    """Fit a surface and divide it through by its value at the box centre.

    Least squares is linear in its target, so scaling the coefficients of a fit
    of ``z`` is exactly the fit of ``z`` scaled --- there is no second fit here,
    and no interpolation of the sample data to find the centre value. It also
    leaves the returned surface exactly one at the centre, which is what makes
    that point an anchor a caller can rely on rather than a fit residual away.

    Returns
    -------
    coef : ndarray
        Normalised Legendre coefficients, of order unity.
    centre : float
        Value of the fitted surface at the centre of the box, in the units of
        ``z``.
    info : FitInfo
        Residual statistics, scaled to match ``coef``.
    """
    coef, info = _legfit2d(x, y, z, order, basis)
    centre = float(_leg.legval2d(0.0, 0.0, coef))
    if not centre > 0.0:
        raise ValueError(
            f"The fitted {name} surface is {centre} at the centre of the fit "
            "box, so it cannot be normalised by that value. Check the sample "
            "data for the wrong sign or the wrong units."
        )
    return coef / centre, centre, FitInfo(rmse=info.rmse / centre, R2=info.R2)


def fit(
    rho,
    u,
    P,
    T,
    s,
    mu,
    kappa,
    Rgas,
    rho_lim,
    u_lim,
    order=8,
    basis="total-order",
):
    r"""Fit an equation of state to tabulated thermodynamic properties.

    Fits the compressibility factor as a surface in density and internal energy,
    then recovers the entropy variation along the reference isochor by
    subtracting the analytic density integral from the tabulated entropy. What
    remains is a function of internal energy alone, so a one-dimensional fit
    closes the model.

    The sample points need not lie on a grid, but they must all be inside the
    box given by ``rho_lim`` and ``u_lim``, and must avoid the two-phase region,
    where the properties are not smooth and the fit would be poisoned.
    :func:`sample_coolprop` masks it, and returns every argument below but the
    two that select the basis, so ``fit(**sample_coolprop(...))`` is the whole
    pipeline.

    Parameters
    ----------
    rho : array_like
        Sample densities [kg/m³].
    u : array_like
        Sample specific internal energies [J/kg], on any datum.
    P : array_like
        Pressure at the sample states [Pa].
    T : array_like
        Temperature at the sample states [K].
    s : array_like
        Specific entropy at the sample states [J/kg/K], on any datum.
    mu : array_like
        Dynamic viscosity at the sample states [kg/m/s].
    kappa : array_like
        Thermal conductivity at the sample states [W/m/K].
    Rgas : float
        Specific gas constant [J/kg/K].
    rho_lim : tuple
        ``(min, max)`` density bounds of the fit box [kg/m³].
    u_lim : tuple
        ``(min, max)`` internal energy bounds of the fit box [J/kg], on the same
        datum as ``u``.
    order : int, optional
        Maximum polynomial order in each variable.
    basis : {'total-order', 'tensor-grid'}, optional
        Which order combinations to retain. ``'total-order'`` (the default)
        keeps only terms whose orders in the two variables sum to at most
        ``order``; ``'tensor-grid'`` keeps every combination up to ``order`` in
        each. The default drops the worst-conditioned high-order-in-both corners
        for roughly half the terms.

    Returns
    -------
    result : FitResult
        Coefficients ready for :class:`ember.fluid.RealFluid`, with the fit
        residuals that bound their accuracy.

    """
    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    x = _poly.hat(rho, rho_lim)
    y = _poly.hat(u, u_lim)

    # Compressibility surface. Dimensionless and O(1), so it fits far better
    # than entropy would directly, and it is what the density integral needs.
    Z = np.asarray(P, dtype=float) / (rho * Rgas * np.asarray(T, dtype=float))
    alpha, info_Z = _legfit2d(x, y, Z, order, basis)

    # Subtract the analytic density integral from the tabulated entropy. The
    # remainder is a function of internal energy alone -- exactly, for an
    # equation of state of this form -- so a one-dimensional fit closes it. Any
    # error in alpha leaks in here as a weak density dependence, which the fit
    # averages out rather than amplifies.
    #
    # Zero density maps to x = -c, where the log in the integral is singular.
    c = -float(_poly.hat(0.0, rho_lim))
    D, Lam = _poly.entropy_integral(alpha, c)
    integral = _leg.legval2d(x, y, D) + _leg.legval(y, Lam) * np.log(x + c)
    beta_target = np.asarray(s, dtype=float) / Rgas + integral

    beta = _leg.legfit(y, beta_target, order)
    info_s = _fit_info(beta_target, _leg.legval(y, beta))

    # Transport surfaces, fitted over the same points in the same coordinates
    # and normalised at the centre of the box. They are ordinary least-squares
    # fits: nothing couples them to the entropy surface or to each other.
    delta, mu_c, info_mu = _fit_normalised(x, y, mu, order, basis, "viscosity")
    gamma, kappa_c, info_kappa = _fit_normalised(
        x, y, kappa, order, basis, "conductivity"
    )

    kwargs = {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "gamma": gamma,
        "rho_lim": tuple(float(v) for v in rho_lim),
        "u_lim": tuple(float(v) for v in u_lim),
        "Rgas": float(Rgas),
        "mu_c": mu_c,
        "kappa_c": kappa_c,
    }
    return FitResult(
        kwargs=kwargs,
        info_Z=info_Z,
        info_s=info_s,
        info_mu=info_mu,
        info_kappa=info_kappa,
    )


def _legfit2d(x, y, z, order, basis="total-order"):
    """Least-squares Legendre fit of a function of two normalised variables.

    Parameters
    ----------
    x, y : array_like
        Normalised sample coordinates, in ``[-1, 1]``.
    z : array_like
        Values to fit.
    order : int
        Maximum polynomial order in each variable.
    basis : {'total-order', 'tensor-grid'}, optional
        Which combinations of orders to retain.

    Returns
    -------
    coef : ndarray
        Legendre coefficients, shape ``(order + 1, order + 1)``, zero outside
        the retained basis.
    info : FitInfo
        Residual statistics.

    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    mask = _poly.order_matrix(order, basis)
    # legvander2d lays terms out as P_i(x)*P_j(y) at flat index i*(order+1)+j,
    # which is C order -- the same as ravelling the mask.
    V = _leg.legvander2d(x, y, [order, order])[:, mask.ravel()]
    coef_flat, *_ = np.linalg.lstsq(V, z, rcond=None)

    coef = np.zeros((order + 1, order + 1))
    coef[mask] = coef_flat
    return coef, _fit_info(z, _leg.legval2d(x, y, coef))


def sample_coolprop(fluid_name, rho_lim, u_lim, ni=100):
    """Sample thermodynamic properties over a fit box using CoolProp.

    States that fail to converge or fall inside the two-phase dome are dropped:
    properties are not smooth across saturation, and including such points would
    poison the fit far more than any choice of basis or order. So are states
    whose transport properties CoolProp declines to report, which for a fluid
    with no transport model at all is every one of them --- said so, rather
    than reported as an empty box.

    Parameters
    ----------
    fluid_name : str
        Fluid name in the CoolProp database.
    rho_lim : tuple
        ``(min, max)`` density bounds [kg/m³].
    u_lim : tuple
        ``(min, max)`` internal energy bounds [J/kg], on CoolProp's datum.
    ni : int, optional
        Number of sample points along each axis.

    Returns
    -------
    dict
        Arrays ``rho``, ``u``, ``P``, ``T``, ``s``, ``mu`` and ``kappa`` at the
        surviving states, ``Rgas``, the specific gas constant [J/kg/K], and the
        ``rho_lim`` and ``u_lim`` that were sampled. That is every argument :func:`fit` needs,
        so the whole pipeline is ``fit(**sample_coolprop(...))``.

        The box is passed on rather than left to the caller to repeat because
        nothing downstream could catch it being repeated wrongly: a fit taken
        over a box other than the sampled one puts the normalised coordinates
        outside ``[-1, 1]``, where the Legendre basis loses its orthogonality
        and the fit its conditioning -- and returns coefficients that are
        wrong without being nan.

    """
    try:
        # Deliberately not at module scope: CoolProp is an optional extra,
        # and every consumer of a fitted equation of state must work
        # without it. noqa: PLC0415
        from CoolProp import CoolProp as CP  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "sample_coolprop requires CoolProp; install ember with the 'fit' "
            "extra, for example `uv pip install -e '.[fit]'`."
        ) from err

    state = CP.AbstractState("HEOS", fluid_name)
    Rgas = state.gas_constant() / state.molar_mass()

    rho_g, u_g = np.meshgrid(
        np.linspace(*rho_lim, ni), np.linspace(*u_lim, ni), indexing="ij"
    )
    rho_flat = rho_g.ravel()
    u_flat = u_g.ravel()

    P = np.full(rho_flat.shape, np.nan)
    T = np.full(rho_flat.shape, np.nan)
    s = np.full(rho_flat.shape, np.nan)
    mu = np.full(rho_flat.shape, np.nan)
    kappa = np.full(rho_flat.shape, np.nan)
    for k, (rho_k, u_k) in enumerate(zip(rho_flat, u_flat)):
        try:
            state.update(CP.DmassUmass_INPUTS, rho_k, u_k)
            quality = state.Q()
            if 0.0 <= quality <= 1.0:
                continue  # inside the two-phase dome
            P[k] = state.p()
            T[k] = state.T()
            s[k] = state.smass()
        except ValueError:
            continue  # state did not converge; leave as nan
        # Separately, because a fluid can have a perfectly good equation of
        # state and no transport model, and the two failures want telling
        # apart below.
        try:
            mu[k] = state.viscosity()
            kappa[k] = state.conductivity()
        except ValueError:
            continue

    thermodynamic = np.isfinite(P) & np.isfinite(T) & np.isfinite(s)
    keep = thermodynamic & np.isfinite(mu) & np.isfinite(kappa)
    if not keep.any():
        if thermodynamic.any():
            raise ValueError(
                f"CoolProp reports no transport properties for {fluid_name!r} "
                "anywhere in the given box, so the viscosity and conductivity "
                "surfaces cannot be fitted."
            )
        raise ValueError(
            f"No valid single-phase states for {fluid_name!r} in the given box; "
            "check rho_lim and u_lim."
        )

    return {
        "rho": rho_flat[keep],
        "u": u_flat[keep],
        "P": P[keep],
        "T": T[keep],
        "s": s[keep],
        "mu": mu[keep],
        "kappa": kappa[keep],
        "Rgas": Rgas,
        "rho_lim": (float(rho_lim[0]), float(rho_lim[1])),
        "u_lim": (float(u_lim[0]), float(u_lim[1])),
    }
