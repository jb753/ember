r"""Legendre-basis polynomial helpers shared by the real-gas fit and evaluation.

Pure coordinate and polynomial routines with no dependency on the fit-result
types or on :class:`ember.fluid.RealFluid`: the box normalisation, the retained
order mask, and the closed-form density integral that links the compressibility
surface to the entropy surface. Both :mod:`ember.realgas_fit` (which produces
the coefficients) and :class:`ember.fluid.RealFluid` (which evaluates them) use
these; keeping them here avoids a dependency either way between those modules.

Not part of the public API.
"""

import numpy as np

_leg = np.polynomial.legendre


def hat(x, lim):
    r"""Map a variable from its fit-box bounds onto :math:`[-1, 1]`.

    Legendre polynomials are orthogonal only on :math:`[-1, 1]`, so every fit is
    done in this coordinate. The mapping is affine, so pre-scaling the input
    data (a change of units or datum) leaves the normalised coordinate
    unchanged.

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


def entropy_integral(alpha, c):
    r"""Closed-form coefficients of the density integral of a fitted surface.

    Evaluates the definite integral of the compressibility surface along a
    density path at fixed internal energy,

    .. math::

        I(\hat\rho, \hat u)
            = \int_{\rho_0}^{\rho} Z \, \frac{\mathrm{d}\rho}{\rho}

    which supplies the density dependence of entropy, from the reference isochor
    at the centre of the fit box up to the density of interest. Since
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

    which is zero at ``x = 0`` for every ``y``.

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

    # Shift so the definite integral vanishes on the reference isochor, the
    # centre of the box at x = 0. Both corrections are functions of internal
    # energy alone, so they fold into the constant-density row of D, where the
    # density basis polynomial is one.
    P0 = _leg.legvander(np.asarray(0.0), D.shape[0] - 1).ravel()
    D = D.copy()
    D[0, :] -= P0 @ D + Lam * np.log(c)

    return D, Lam
