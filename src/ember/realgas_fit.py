r"""Fit real-gas equation of state coefficients from tabulated properties.

This module turns a table of thermodynamic properties into the Legendre
coefficient arrays that :class:`ember.fluid.RealFluid` evaluates, following the
entropy-based formulation of Wheeler (2024), *Computers and Fluids* 268:106088.

It is an offline tool. Fitting happens once, in a user script, and the
coefficients are then passed to a fluid that the solver evaluates; nothing here
runs in a hot path. Only :func:`sample_coolprop` needs CoolProp, and it imports
it lazily, so the rest of the module -- and every consumer of the coefficients
-- works with numpy alone.

The method
==========

The compressibility factor is fitted as a two-dimensional polynomial surface in
density and internal energy,

.. math:: Z(\rho, u) = \frac{p}{\rho R T}

and entropy follows by integrating it along an isochor and then in density,

.. math::

    \frac{s}{R} = \sum_k \beta_k P_k(\hat{u})
                - \int_{\rho_0}^{\rho} Z \, \mathrm{d}\ln\rho

Because temperature and pressure are then *derived* from this one entropy
surface rather than fitted independently, the resulting equation of state is
thermodynamically consistent by construction. See :class:`ember.fluid.RealFluid`
for the evaluation side.

.. _normalised-coordinates:

Normalised coordinates
======================

Polynomials are fitted in coordinates scaled onto :math:`[-1, 1]` by the bounds
of the fit box,

.. math::

    \hat{\rho} = \frac{\rho - \rho_m}{\rho_f}, \qquad
    \hat{u} = \frac{u - u_m}{u_f}

where :math:`\rho_m, \rho_f` are the midpoint and half-width of the density
bounds. Legendre polynomials are orthogonal only on :math:`[-1, 1]`, so this
min/max normalisation -- rather than the mean and standard deviation used in the
original paper, which do not bound the coordinate -- is what keeps the fit well
conditioned.

The normalisation is affine, so it absorbs any change of units or datum in the
input data exactly: pre-scaling the table before fitting changes nothing. That
is also why the fit box cannot be replaced by ember's reference scales, which
are pure scalings with no offset and are reset at runtime from the flow.

"""

import dataclasses

import numpy as np

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
        ``rho_lim``, ``u_lim``, ``rho_isochor`` and ``Rgas``.
    info_Z : FitInfo
        Residual of the compressibility surface [--].
    info_s : FitInfo
        Residual of the entropy fit, in units of the gas constant [--].

    """

    kwargs: dict
    info_Z: FitInfo
    info_s: FitInfo

    @property
    def rmse_Z(self):
        """Root-mean-square residual of the compressibility fit [--]."""
        return self.info_Z.rmse

    @property
    def rmse_s(self):
        """Root-mean-square residual of the entropy fit, in units of R [--]."""
        return self.info_s.rmse


def _fit_info(y, y_fit):
    """Residual statistics of a fit."""
    resid = np.asarray(y) - np.asarray(y_fit)
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    R2 = 1.0 if ss_tot == 0.0 else 1.0 - float(np.sum(resid**2)) / ss_tot
    return FitInfo(rmse=rmse, R2=R2)


def entropy_integral(alpha, c, x0):
    r"""Closed-form coefficients of the density integral of a fitted surface.

    Evaluates the definite integral of the compressibility surface along a
    density path at fixed internal energy,

    .. math::

        I(\hat\rho, \hat u)
            = \int_{\rho_0}^{\rho} Z \, \frac{\mathrm{d}\rho}{\rho}

    which supplies the density dependence of entropy. Since
    :math:`\rho \propto \hat\rho + c`, the measure is
    :math:`\mathrm{d}\ln\rho = \mathrm{d}\hat\rho / (\hat\rho + c)`, and each
    basis term integrates in closed form. Polynomial division by
    :math:`\hat\rho + c` leaves a constant remainder,

    .. math::

        P_i(x) = (x + c)\,Q_i(x) + P_i(-c)

    so that

    .. math::

        \int \frac{P_i(x)}{x + c}\,\mathrm{d}x
            = \int Q_i(x)\,\mathrm{d}x + P_i(-c)\ln(x + c)

    and the whole integral splits into a polynomial part plus a single
    logarithmic term whose coefficient depends only on internal energy. The
    division and integration are done in the Legendre basis throughout, so no
    monomial coefficients are ever formed.

    Parameters
    ----------
    alpha : array_like
        Two-dimensional Legendre coefficients of :math:`Z(\hat\rho, \hat u)`.
    c : float
        Ratio of the density midpoint to half-width, so that
        :math:`\rho \propto \hat\rho + c`. The logarithmic singularity sits at
        :math:`\hat\rho = -c`, just outside the box when the lower density bound
        approaches zero.
    x0 : float
        Normalised density of the reference isochor, where the integral is zero.

    Returns
    -------
    D : ndarray
        Two-dimensional Legendre coefficients of the polynomial part.
    Lam : ndarray
        One-dimensional Legendre coefficients, in normalised internal energy, of
        the multiplier on :math:`\ln(\hat\rho + c)`.

    Notes
    -----
    The two outputs reconstruct the integral as

    .. code-block:: text

        I = legval2d(x, y, D) + legval(y, Lam)*log(x + c)

    which is zero at ``x = x0`` for every ``y``.

    """
    alpha = np.atleast_2d(np.asarray(alpha, dtype=float))
    m = alpha.shape[0] - 1

    # Split each density basis polynomial into a quotient, which integrates to a
    # polynomial, and a constant remainder, which integrates to the log term.
    divisor = np.array([c, 1.0])  # (x + c) in the Legendre basis
    G = np.zeros((m + 1, m + 2))  # antiderivative of the quotient, per order
    r = np.zeros(m + 1)  # remainder P_i(-c), per order
    for i in range(m + 1):
        basis = np.zeros(i + 1)
        basis[i] = 1.0
        quo, rem = _leg.legdiv(basis, divisor)
        r[i] = rem[0] if rem.size else 0.0
        Gi = _leg.legint(quo)
        G[i, : Gi.size] = Gi

    # Contract over the density order: the polynomial part inherits the
    # antiderivatives, the log multiplier inherits the remainders.
    D = G.T @ alpha
    Lam = alpha.T @ r

    # Shift so the definite integral vanishes on the reference isochor. Both
    # corrections are functions of internal energy alone, so they fold into the
    # constant-density row of D, where the density basis polynomial is one.
    Px0 = _leg.legvander(np.asarray(float(x0)), D.shape[0] - 1).ravel()
    D = D.copy()
    D[0, :] -= Px0 @ D + Lam * np.log(x0 + c)

    return D, Lam


def fit(
    rho,
    u,
    P,
    T,
    s,
    Rgas,
    rho_lim,
    u_lim,
    rho_isochor,
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
    where the properties are not smooth and the fit would be poisoned. See
    :func:`sample_coolprop`, which masks it.

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
    Rgas : float
        Specific gas constant [J/kg/K].
    rho_lim : tuple
        ``(min, max)`` density bounds of the fit box [kg/m³].
    u_lim : tuple
        ``(min, max)`` internal energy bounds of the fit box [J/kg], on the same
        datum as ``u``.
    rho_isochor : float
        Density of the isochor the entropy integral starts from [kg/m³].
    order : int, optional
        Maximum polynomial order in each variable.
    basis : {'total-order', 'tensor-grid'}, optional
        Which combinations of orders to retain; see :func:`order_matrix`.

    Returns
    -------
    result : FitResult
        Coefficients ready for :class:`ember.fluid.RealFluid`, with the fit
        residuals that bound their accuracy.

    """
    # The density integral is taken about this isochor, so an isochor off the
    # box puts a non-positive number under a logarithm and every entropy
    # coefficient comes back nan -- along with the residuals a caller would
    # inspect to decide whether the fit is any good.
    if not (rho_isochor > 0.0 and rho_lim[0] <= rho_isochor <= rho_lim[1]):
        raise ValueError(
            f"rho_isochor={rho_isochor} must be positive and lie within "
            f"rho_lim={tuple(rho_lim)}."
        )

    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    x = hat(rho, rho_lim)
    y = hat(u, u_lim)

    # Compressibility surface. Dimensionless and O(1), so it fits far better
    # than entropy would directly, and it is what the density integral needs.
    Z = np.asarray(P, dtype=float) / (rho * Rgas * np.asarray(T, dtype=float))
    alpha, info_Z = legfit2d(x, y, Z, order, basis)

    # Subtract the analytic density integral from the tabulated entropy. The
    # remainder is a function of internal energy alone -- exactly, for an
    # equation of state of this form -- so a one-dimensional fit closes it. Any
    # error in alpha leaks in here as a weak density dependence, which the fit
    # averages out rather than amplifies.
    c, x0 = _log_shift(rho_lim, rho_isochor)
    D, Lam = entropy_integral(alpha, c, x0)
    integral = _leg.legval2d(x, y, D) + _leg.legval(y, Lam) * np.log(x + c)
    beta_target = np.asarray(s, dtype=float) / Rgas + integral

    beta, info_s = legfit1d(y, beta_target, order)

    kwargs = {
        "alpha": alpha,
        "beta": beta,
        "rho_lim": tuple(float(v) for v in rho_lim),
        "u_lim": tuple(float(v) for v in u_lim),
        "rho_isochor": float(rho_isochor),
        "Rgas": float(Rgas),
    }
    return FitResult(kwargs=kwargs, info_Z=info_Z, info_s=info_s)


def hat(x, lim):
    r"""Map a variable from its box bounds onto :math:`[-1, 1]`.

    See :ref:`normalised-coordinates`.

    Parameters
    ----------
    x : array_like
        Values to normalise.
    lim : tuple
        ``(min, max)`` bounds of the fit box.

    Returns
    -------
    x_hat : ndarray
        Normalised coordinate, ``-1`` at ``lim[0]`` and ``+1`` at ``lim[1]``.

    """
    lo, hi = float(lim[0]), float(lim[1])
    if not hi > lo:
        raise ValueError(f"Box bounds must be increasing, got {lim}")
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return (np.asarray(x, dtype=float) - mid) / half


def legfit1d(x, z, order):
    """Least-squares Legendre fit of a function of one normalised variable.

    Parameters
    ----------
    x : array_like
        Normalised sample coordinates, in ``[-1, 1]``.
    z : array_like
        Values to fit.
    order : int
        Maximum polynomial order.

    Returns
    -------
    coef : ndarray
        Legendre coefficients, shape ``(order + 1,)``.
    info : FitInfo
        Residual statistics.

    """
    coef = _leg.legfit(np.asarray(x, dtype=float), np.asarray(z, dtype=float), order)
    return coef, _fit_info(z, _leg.legval(x, coef))


def legfit2d(x, y, z, order, basis="total-order"):
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
        Which combinations of orders to retain; see :func:`order_matrix`.

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

    mask = order_matrix(order, basis)
    # legvander2d lays terms out as P_i(x)*P_j(y) at flat index i*(order+1)+j,
    # which is C order -- the same as ravelling the mask.
    V = _leg.legvander2d(x, y, [order, order])[:, mask.ravel()]
    coef_flat, *_ = np.linalg.lstsq(V, z, rcond=None)

    coef = np.zeros((order + 1, order + 1))
    coef[mask] = coef_flat
    return coef, _fit_info(z, _leg.legval2d(x, y, coef))


def _log_shift(rho_lim, rho_isochor):
    """Return ``(c, x0)`` for the entropy integral over a given density box."""
    lo, hi = float(rho_lim[0]), float(rho_lim[1])
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid / half, (float(rho_isochor) - mid) / half


def order_matrix(order, basis="total-order"):
    """Boolean mask of the polynomial orders retained by a basis.

    A ``tensor-grid`` basis keeps every combination of orders up to ``order`` in
    each variable, giving ``(order + 1)**2`` terms. A ``total-order`` basis keeps
    only those whose orders sum to at most ``order``, giving roughly half as
    many. The smaller basis is usually the better trade: the dropped terms are
    the high-order-in-both corners, which contribute least and are the worst
    conditioned.

    Parameters
    ----------
    order : int
        Maximum polynomial order.
    basis : {'total-order', 'tensor-grid'}, optional
        Which combinations to retain.

    Returns
    -------
    mask : ndarray of bool
        Shape ``(order + 1, order + 1)``, true where the term is retained.

    """
    i, j = np.mgrid[0 : order + 1, 0 : order + 1]
    if basis == "tensor-grid":
        return np.ones((order + 1, order + 1), dtype=bool)
    if basis == "total-order":
        return (i + j) <= order
    raise ValueError(f"basis must be 'total-order' or 'tensor-grid', got {basis!r}")


def sample_coolprop(fluid_name, rho_lim, u_lim, ni=100):
    """Sample thermodynamic properties over a fit box using CoolProp.

    CoolProp is an optional dependency, imported here rather than at module
    level so that evaluating a fitted equation of state never requires it.
    Install it with the ``fit`` extra.

    States that fail to converge or fall inside the two-phase dome are dropped:
    properties are not smooth across saturation, and including such points would
    poison the fit far more than any choice of basis or order.

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
        Arrays ``rho``, ``u``, ``P``, ``T``, ``s`` at the surviving states, and
        ``Rgas``, the specific gas constant [J/kg/K], ready to pass to
        :func:`fit`.

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

    keep = np.isfinite(P) & np.isfinite(T) & np.isfinite(s)
    if not keep.any():
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
        "Rgas": Rgas,
    }


def unhat(x_hat, lim):
    """Map a normalised coordinate back onto its box bounds.

    Inverse of :func:`hat`.

    Parameters
    ----------
    x_hat : array_like
        Normalised coordinate, in ``[-1, 1]``.
    lim : tuple
        ``(min, max)`` bounds of the fit box.

    Returns
    -------
    x : ndarray
        Values in the original units.

    """
    lo, hi = float(lim[0]), float(lim[1])
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return np.asarray(x_hat, dtype=float) * half + mid
