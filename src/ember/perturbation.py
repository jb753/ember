r"""Jacobians between the flow variable sets used throughout EMBER.

Six five-component variable sets, each labelled by a single calligraphic
letter, describe the same flow state; boundary conditions, the
characteristic treatment and the mixing planes all need to move small
perturbations between them:

* **primitive**

  .. math::
      \mathcal{P} = [\rho, V_x, V_r, V_\theta, p]^\mathrm{T}

* **conserved** -- what :meth:`~ember.block.Block.set_conserved` stores

  .. math::
      \mathcal{U} = [\rho, \rho V_x, \rho V_r, \rho r V_\theta, \rho e]^\mathrm{T}

* **flux** -- the x-direction advective flux of each conserved variable

  .. math::
      \mathcal{F} = [\rho V_x,\ \rho V_x^2 + p,\ \rho V_x V_r,\
      \rho V_x r V_\theta,\ \rho V_x h_0]^\mathrm{T}

* **chic** -- one-dimensional characteristic variables of the Euler equations

  .. math::
      \mathcal{C} = [c_\mathrm{up}, c_\mathrm{down}, c_r, c_t, c_s]^\mathrm{T}

* **bcond** -- what a subsonic inflow prescribes; see
  :class:`~ember.patch.InletPatch`

  .. math::
      \mathcal{B} = [h_0, s, \tan\alpha, \sin\beta, p]^\mathrm{T}

* **mix** -- what a mixing plane exchanges; see
  :class:`~ember.patch.MixingPatch`

  .. math::
      \mathcal{M} = [h_0, s, V_r, V_\theta, p]^\mathrm{T}

Every function below is named ``X_to_Y`` and returns the Jacobian
:math:`\mathbf{J}_{\mathcal{X}\to\mathcal{Y}}`, evaluated pointwise from a
:class:`~ember.block.Block`'s current state and returned as a batch of 5x5
matrices stacked on the trailing two axes:

.. math::
    \delta\mathcal{Y} = \mathbf{J}_{\mathcal{X}\to\mathcal{Y}}\,\delta\mathcal{X}

Two conventions keep this cheap. First, whenever both a Jacobian and its
inverse are needed, the inverse is not obtained by numerically inverting the
forward 5x5 matrix at every point -- it is worked out analytically in closed
form and implemented as its own function. :func:`conserved_to_primitive`,
for instance, is the closed-form inverse of :func:`primitive_to_conserved`,
and every other ``Y_to_X`` in this module pairs with an ``X_to_Y`` the same
way. Second, every Jacobian between two sets neither of which is primitive
necessarily passes through primitive variables,
:math:`\mathbf{J}_{\mathcal{X}\to\mathcal{Y}} = \mathbf{J}_{\mathcal{P}\to\mathcal{Y}}\,\mathbf{J}_{\mathcal{X}\to\mathcal{P}}`,
and where that composite sits on a hot path -- the characteristic and
mixing-plane boundary conditions evaluate one every timestep -- the product
is worked out analytically once and implemented as its own fused function
rather than composed from two separate calls. For example
:func:`chic_to_bcond` evaluates
:math:`\mathbf{J}_{\mathcal{C}\to\mathcal{B}} = \mathbf{J}_{\mathcal{P}\to\mathcal{B}}\,\mathbf{J}_{\mathcal{C}\to\mathcal{P}}`
directly, rather than calling :func:`primitive_to_bcond` and
:func:`chic_to_primitive` and multiplying the results at every timestep.
Elsewhere, where the extra multiply is not performance-critical, a composite
Jacobian is simply the product of two separately evaluated calls --
:func:`flux_to_conserved` and :func:`conserved_to_flux`, for instance, are
literally :func:`primitive_to_conserved` :math:`\cdot` :func:`flux_to_primitive`
and :func:`primitive_to_flux` :math:`\cdot` :func:`conserved_to_primitive`
multiplied together at call time.

All Jacobians are evaluated in the nondimensional space the block stores its
state in, using :attr:`~ember.block.Block.conserved_nd`,
:attr:`~ember.block.Block.r_nd` and the fluid's ``_nd`` thermodynamic
derivative properties.
"""

import numpy as np
from ember import util
import ember.fortran


def _stack_matrix(*args, shape, out=None):
    """Stack nested iterables into a matrix with trailing matrix dimensions.

    Parameters
    ----------
    args : nested iterables length [nrow][ncol]
        Variables to stack, where args[i][j] contains the (i,j) matrix element.
        Use None for zero entries to skip the copy.
    shape : tuple
        Grid shape for the batch dimensions.
    out : array, optional
        Preallocated output array of shape (*shape, nrow, ncol). If None, a new
        array is allocated.

    Returns
    -------
    out : Array, shape (*shape, nrow, ncol)
        A composite matrix variable with matrix dimensions in trailing axes.
        Uses f32 precision and Fortran ordering for optimal performance.
    """
    nrow = len(args)
    ncol = len(args[0])

    if out is None:
        out = np.empty(shape + (nrow, ncol), dtype=np.float32, order="F")
    out.fill(0.0)
    for i in range(nrow):
        for j in range(ncol):
            v = args[i][j]
            if v is not None:
                out[..., i, j] = v
    return out


def primitive_to_conserved(block, out=None):
    r"""Jacobian of conserved variables with respect to primitive variables.

    .. math::
        \delta\mathcal{U} = \mathbf{J}_{\mathcal{P}\to\mathcal{U}}\,\delta\mathcal{P}

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{P}\to\mathcal{U}}`, stacked on the
        trailing two axes.
    """
    b = block
    if out is None:
        out = np.empty(b.shape + (5, 5), dtype=np.float32, order="F")

    if b.ndim == 1:
        ember.fortran.primitive_to_conserved(
            b.conserved_nd, b.r_nd, b.dudrho_P_nd, b.dudP_rho_nd, out
        )
        return out

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    e = q[..., 4] / rho
    drhoe_drho_P = e + rho * b.dudrho_P_nd
    drhoe_dP_rho = rho * b.dudP_rho_nd

    out.fill(0.0)
    out[..., 0, 0] = 1.0
    out[..., 1, 0] = Vx
    out[..., 1, 1] = rho
    out[..., 2, 0] = Vr
    out[..., 2, 2] = rho
    out[..., 3, 0] = r * Vt
    out[..., 3, 3] = r * rho
    out[..., 4, 0] = drhoe_drho_P
    out[..., 4, 1] = q[..., 1]
    out[..., 4, 2] = q[..., 2]
    out[..., 4, 3] = rho * Vt
    out[..., 4, 4] = drhoe_dP_rho
    return out


def conserved_to_primitive(block):
    r"""Jacobian of primitive variables with respect to conserved variables.

    Analytical inverse of :func:`primitive_to_conserved`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{U}\to\mathcal{P}}`, stacked on the
        trailing two axes.
    """
    b = block

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    e = q[..., 4] / rho
    Vsq = Vx**2 + Vr**2 + Vt**2
    drhoe_drho_P = e + rho * b.dudrho_P_nd
    drhoe_dP_rho = rho * b.dudP_rho_nd

    out = _stack_matrix(
        (1.0, None, None, None, None),
        (-Vx, 1.0, None, None, None),
        (-Vr, None, 1.0, None, None),
        (-Vt, None, None, 1.0 / r, None),
        (
            (Vsq - drhoe_drho_P),
            -Vx,
            -Vr,
            -Vt / r,
            1.0,
        ),
        shape=b.shape,
    )
    # Divide rows 1-3 of each matrix by rho (need to add dimensions for broadcasting)
    out[..., 1:4, :] /= rho[..., None, None]
    # Divide last row by drhoe_dP_rho
    out[..., -1, :] /= drhoe_dP_rho[..., None]

    return out


def primitive_to_chic(block, out=None):
    r"""Jacobian of one-dimensional characteristic variables with respect to primitive variables.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{P}\to\mathcal{C}}`, stacked on the
        trailing two axes.
    """
    b = block
    if out is None:
        out = np.empty(b.shape + (5, 5), dtype=np.float32, order="F")

    rho = b.conserved_nd[..., 0]
    a = b.a_nd

    if b.ndim == 1:
        ember.fortran.primitive_to_chic(rho, a, out)
        return out

    rhoa = rho * a
    asq = a**2

    out.fill(0.0)
    out[..., 0, 1] = -rhoa
    out[..., 0, 4] = 1.0
    out[..., 1, 1] = rhoa
    out[..., 1, 4] = 1.0
    out[..., 2, 2] = rhoa
    out[..., 3, 3] = rhoa
    out[..., 4, 0] = -asq
    out[..., 4, 4] = 1.0
    return out


def chic_to_primitive(block, out=None):
    r"""Jacobian of primitive variables with respect to one-dimensional characteristic variables.

    Analytical inverse of :func:`primitive_to_chic`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{C}\to\mathcal{P}}`, stacked on the
        trailing two axes.
    """
    b = block
    if out is None:
        out = np.empty(b.shape + (5, 5), dtype=np.float32, order="F")

    rho = b.conserved_nd[..., 0]
    a = b.a_nd

    if b.ndim == 1:
        ember.fortran.chic_to_primitive(rho, a, out)
        return out

    asq_recip = 1.0 / a**2
    rhoa_recip = 1.0 / (rho * a)

    out.fill(0.0)
    out[..., 0, 0] = 0.5 * asq_recip
    out[..., 0, 1] = 0.5 * asq_recip
    out[..., 0, 4] = -asq_recip
    out[..., 1, 0] = -0.5 * rhoa_recip
    out[..., 1, 1] = 0.5 * rhoa_recip
    out[..., 2, 2] = rhoa_recip
    out[..., 3, 3] = rhoa_recip
    out[..., 4, 0] = 0.5
    out[..., 4, 1] = 0.5
    return out


def primitive_to_flux(block):
    r"""Jacobian of the x-direction advective flux with respect to primitive variables.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{P}\to\mathcal{F}}`, stacked on the
        trailing two axes.
    """
    b = block

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    rhoVx = q[..., 1]
    rhoVr = q[..., 2]
    rhorVt = q[..., 3]
    Vx = rhoVx / rho
    Vr = rhoVr / rho
    Vt = rhorVt / (rho * r)
    ho = b.ho_nd
    VxVr = Vx * Vr
    VxrVt = Vx * r * Vt
    VxVx = Vx**2
    dE_drho = Vx * ho + rhoVx * b.dhdrho_P_nd
    dE_dVx = rho * ho + rhoVx * Vx

    return _stack_matrix(
        (Vx, rho, None, None, None),
        (VxVx, 2.0 * rhoVx, None, None, 1.0),
        (VxVr, rhoVr, rhoVx, None, None),
        (VxrVt, rhorVt, None, rhoVx * r, None),
        (dE_drho, dE_dVx, rhoVx * Vr, rhoVx * Vt, rhoVx * b.dhdP_rho_nd),
        shape=b.shape,
    )


def flux_to_primitive(block, out=None):
    r"""Jacobian of primitive variables with respect to the x-direction advective flux.

    Analytical inverse of :func:`primitive_to_flux`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{F}\to\mathcal{P}}`, stacked on the
        trailing two axes.
    """
    b = block
    if out is None:
        out = np.empty(b.shape + (5, 5), dtype=np.float32, order="F")

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    ho = b.ho_nd
    dhdP_rho = b.dhdP_rho_nd
    dhdrho_P = b.dhdrho_P_nd

    # Fast Fortran path for 1D blocks only
    if b.ndim == 1:
        ember.fortran.flux_to_primitive(q, r, ho, dhdP_rho, dhdrho_P, out)
        return out

    VxVx = Vx * Vx
    Vsq = Vr * Vr + Vt * Vt
    dhdP_rho_rho = dhdP_rho * rho
    dhdrho_rho = dhdrho_P * rho
    # D = Vx^2*(dhdP_rho*rho - 1) + dhdrho_P*rho
    D_inv = 1.0 / (VxVx * (dhdP_rho_rho - 1.0) + dhdrho_rho)
    Vx_inv = 1.0 / Vx
    rho_inv = 1.0 / rho
    rhoVx_inv = rho_inv * Vx_inv
    r_inv = 1.0 / r

    D_inv_Vx = D_inv * Vx_inv
    D_inv_rho = D_inv * rho_inv

    out.fill(0.0)

    # Row 0: d(rho)/d(flux)
    out[..., 0, 0] = (Vsq - ho + VxVx * (2.0 * dhdP_rho_rho - 1.0)) * D_inv_Vx
    out[..., 0, 1] = -dhdP_rho_rho * D_inv
    out[..., 0, 2] = -Vr * D_inv_Vx
    out[..., 0, 3] = -Vt * r_inv * D_inv_Vx
    out[..., 0, 4] = D_inv_Vx

    # Row 1: d(Vx)/d(flux)
    out[..., 1, 0] = (-Vsq + ho + dhdrho_rho - VxVx * dhdP_rho_rho) * D_inv_rho
    out[..., 1, 1] = Vx * dhdP_rho * D_inv
    out[..., 1, 2] = Vr * D_inv_rho
    out[..., 1, 3] = Vt * r_inv * D_inv_rho
    out[..., 1, 4] = -D_inv_rho

    # Row 2: d(Vr)/d(flux)
    out[..., 2, 0] = -Vr * rhoVx_inv
    out[..., 2, 2] = rhoVx_inv

    # Row 3: d(Vt)/d(flux)
    out[..., 3, 0] = -Vt * rhoVx_inv
    out[..., 3, 3] = r_inv * rhoVx_inv

    # Row 4: d(P)/d(flux)
    out[..., 4, 0] = Vx * (Vsq + VxVx - 2.0 * dhdrho_rho - ho) * D_inv
    out[..., 4, 1] = (dhdrho_rho - VxVx) * D_inv
    out[..., 4, 2] = -Vx * Vr * D_inv
    out[..., 4, 3] = -Vx * Vt * r_inv * D_inv
    out[..., 4, 4] = Vx * D_inv

    return out


def primitive_to_bcond(block, out=None):
    r"""Jacobian of a subsonic inflow's prescribed quantities with respect to primitive variables.

    Both angles are measured against the meridional velocity magnitude
    :math:`V_m = \sqrt{V_x^2 + V_r^2}`, as in :attr:`~ember.block.Block.tanAlpha`
    and :attr:`~ember.block.Block.sinBeta`, so every angle derivative below
    carries :math:`V_m` and not the total speed. ``sinBeta`` is used in place
    of ``tanBeta`` to avoid the singularity at :math:`\beta = 90^\circ`;
    :math:`\sin\beta = V_r/V_m`, with derivatives verified numerically.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{P}\to\mathcal{B}}`, stacked on the
        trailing two axes.
    """
    b = block

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    Vm_sq = Vx**2 + Vr**2
    Vm = np.sqrt(Vm_sq)
    Vm_cb = Vm * Vm_sq
    tanAlpha = Vt / Vm
    dtanAl_dVx = -tanAlpha * Vx / Vm_sq
    dtanAl_dVr = -tanAlpha * Vr / Vm_sq
    dtanAl_dVt = 1.0 / Vm

    dsinBe_dVx = -Vr * Vx / Vm_cb
    dsinBe_dVr = Vx**2 / Vm_cb

    return _stack_matrix(
        (b.dhdrho_P_nd, Vx, Vr, Vt, b.dhdP_rho_nd),
        (b.dsdrho_P_nd, None, None, None, b.dsdP_rho_nd),
        (None, dtanAl_dVx, dtanAl_dVr, dtanAl_dVt, None),
        (None, dsinBe_dVx, dsinBe_dVr, None, None),
        (None, None, None, None, 1.0),
        shape=b.shape,
        out=out,
    )


def bcond_to_primitive(block, out=None):
    r"""Jacobian of primitive variables with respect to a subsonic inflow's prescribed quantities.

    Analytical inverse of :func:`primitive_to_bcond`.

    :math:`V_m` is the meridional speed the two angles are measured against
    and :math:`V^2 = V_m^2 + V_\theta^2` the total speed squared. The entropy
    row of the forward matrix eliminates :math:`\delta\rho` in favour of
    :math:`\delta s` and :math:`\delta p`, leaving a stagnation-enthalpy
    residual

    .. math::
        \delta b_0 = \delta h_0
            - \frac{\partial h_0/\partial\rho|_p}{\partial s/\partial\rho|_p}\,\delta s
            + \text{cross}\cdot\delta p

    Substituting that into the enthalpy row leaves the meridional velocity
    projection :math:`\delta u = V_x\,\delta V_x + V_r\,\delta V_r` as the
    only remaining unknown, solved for as

    .. math::
        \delta u = \frac{V_m^2}{V^2}
            \left(\delta b_0 - V_\theta V_m\,\delta\tan\alpha\right)

    which is then split between :math:`\delta V_x` and :math:`\delta V_r`
    using :math:`\delta\sin\beta`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{B}\to\mathcal{P}}`, stacked on the
        trailing two axes.
    """
    b = block
    if out is None:
        out = np.empty(b.shape + (5, 5), dtype=np.float32, order="F")

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    dhdrho_P = b.dhdrho_P_nd
    dhdP_rho = b.dhdP_rho_nd
    dsdrho_P = b.dsdrho_P_nd
    dsdP_rho = b.dsdP_rho_nd

    if b.ndim == 1:
        ember.fortran.bcond_to_primitive(
            q, r, dhdrho_P, dhdP_rho, dsdrho_P, dsdP_rho, out
        )
        return out

    Vm_sq = Vx**2 + Vr**2
    Vsq = Vm_sq + Vt**2
    Vm = np.sqrt(Vm_sq)
    dsdrho_inv = 1.0 / dsdrho_P
    cross = (dhdrho_P * dsdP_rho - dhdP_rho * dsdrho_P) * dsdrho_inv
    Vx_Vsq = Vx / Vsq
    Vr_Vsq = Vr / Vsq
    Vt_Vsq = Vt / Vsq
    Vt_Vm = Vt * Vm

    out.fill(0.0)
    out[..., 0, 1] = dsdrho_inv
    out[..., 0, 4] = -dsdP_rho * dsdrho_inv
    out[..., 1, 0] = Vx_Vsq
    out[..., 1, 1] = -dhdrho_P * Vx_Vsq * dsdrho_inv
    out[..., 1, 2] = -Vx_Vsq * Vt_Vm
    out[..., 1, 3] = -Vr * Vm / Vx
    out[..., 1, 4] = cross * Vx_Vsq
    out[..., 2, 0] = Vr_Vsq
    out[..., 2, 1] = -dhdrho_P * Vr_Vsq * dsdrho_inv
    out[..., 2, 2] = -Vr_Vsq * Vt_Vm
    out[..., 2, 3] = Vm
    out[..., 2, 4] = cross * Vr_Vsq
    out[..., 3, 0] = Vt_Vsq
    out[..., 3, 1] = -dhdrho_P * Vt_Vsq * dsdrho_inv
    out[..., 3, 2] = Vm * Vm_sq / Vsq
    out[..., 3, 4] = cross * Vt_Vsq
    out[..., 4, 4] = 1.0
    return out


def primitive_to_mix(block, out=None):
    r"""Jacobian of a mixing plane's exchanged quantities with respect to primitive variables.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{P}\to\mathcal{M}}`, stacked on the
        trailing two axes.
    """
    b = block
    if out is None:
        out = np.empty(b.shape + (5, 5), dtype=np.float32, order="F")

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    dhdrho_P = b.dhdrho_P_nd
    dhdP_rho = b.dhdP_rho_nd
    dsdrho_P = b.dsdrho_P_nd
    dsdP_rho = b.dsdP_rho_nd

    if b.ndim == 1:
        ember.fortran.primitive_to_mix(
            q, r, dhdrho_P, dhdP_rho, dsdrho_P, dsdP_rho, out
        )
        return out

    out.fill(0.0)
    out[..., 0, 0] = dhdrho_P
    out[..., 0, 1] = Vx
    out[..., 0, 2] = Vr
    out[..., 0, 3] = Vt
    out[..., 0, 4] = dhdP_rho
    out[..., 1, 0] = dsdrho_P
    out[..., 1, 4] = dsdP_rho
    out[..., 2, 2] = 1.0
    out[..., 3, 3] = 1.0
    out[..., 4, 4] = 1.0
    return out


def mix_to_primitive(block, out=None):
    r"""Jacobian of primitive variables with respect to a mixing plane's exchanged quantities.

    Analytical inverse of :func:`primitive_to_mix`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{M}\to\mathcal{P}}`, stacked on the
        trailing two axes.
    """
    b = block
    if out is None:
        out = np.empty(b.shape + (5, 5), dtype=np.float32, order="F")

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    dhdrho_P = b.dhdrho_P_nd
    dhdP_rho = b.dhdP_rho_nd
    dsdrho_P = b.dsdrho_P_nd
    dsdP_rho = b.dsdP_rho_nd

    if b.ndim == 1:
        ember.fortran.mix_to_primitive(
            q, r, dhdrho_P, dhdP_rho, dsdrho_P, dsdP_rho, out
        )
        return out

    Vx_inv = 1.0 / Vx
    dsdrho_inv = 1.0 / dsdrho_P
    cross = (dhdrho_P * dsdP_rho - dhdP_rho * dsdrho_P) * dsdrho_inv

    out.fill(0.0)
    out[..., 0, 1] = dsdrho_inv
    out[..., 0, 4] = -dsdP_rho * dsdrho_inv
    out[..., 1, 0] = Vx_inv
    out[..., 1, 1] = -dhdrho_P * Vx_inv * dsdrho_inv
    out[..., 1, 2] = -Vr * Vx_inv
    out[..., 1, 3] = -Vt * Vx_inv
    out[..., 1, 4] = cross * Vx_inv
    out[..., 2, 2] = 1.0
    out[..., 3, 3] = 1.0
    out[..., 4, 4] = 1.0
    return out


def mix_to_conserved(block, out=None):
    r"""Jacobian of conserved variables with respect to a mixing plane's exchanged quantities.

    Analytically fused product
    :math:`\mathbf{J}_{\mathcal{M}\to\mathcal{U}}
    = \mathbf{J}_{\mathcal{P}\to\mathcal{U}}\,\mathbf{J}_{\mathcal{M}\to\mathcal{P}}`,
    i.e. :func:`primitive_to_conserved` :math:`\cdot`
    :func:`mix_to_primitive`, rather than composing the two Jacobians at
    call time.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{M}\to\mathcal{U}}`, stacked on the
        trailing two axes.
    """
    b = block

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    e = q[..., 4] / rho
    dhdrho_P = b.dhdrho_P_nd
    dhdP_rho = b.dhdP_rho_nd
    dsdP_rho = b.dsdP_rho_nd
    dsdrho_P = b.dsdrho_P_nd
    drhoe_drho_P = e + rho * b.dudrho_P_nd
    drhoe_dP_rho = rho * b.dudP_rho_nd

    dsdrho_inv = 1.0 / dsdrho_P
    Vx_inv = 1.0 / Vx
    rho_Vx_inv = rho * Vx_inv
    VxVx = Vx**2
    cross = dhdrho_P * dsdP_rho - dhdP_rho * dsdrho_P

    return _stack_matrix(
        # Row 0: d(rho)/d(mix)
        (None, dsdrho_inv, None, None, -dsdP_rho * dsdrho_inv),
        # Row 1: d(rhoVx)/d(mix)
        (
            rho_Vx_inv,
            (VxVx - dhdrho_P * rho) * Vx_inv * dsdrho_inv,
            -Vr * rho_Vx_inv,
            -Vt * rho_Vx_inv,
            (rho * cross - VxVx * dsdP_rho) * Vx_inv * dsdrho_inv,
        ),
        # Row 2: d(rhoVr)/d(mix)
        (None, Vr * dsdrho_inv, rho, None, -Vr * dsdP_rho * dsdrho_inv),
        # Row 3: d(rhorVt)/d(mix)
        (
            None,
            Vt * r * dsdrho_inv,
            None,
            r * rho,
            -Vt * dsdP_rho * r * dsdrho_inv,
        ),
        # Row 4: d(rhoe)/d(mix)
        (
            rho,
            (drhoe_drho_P - dhdrho_P * rho) * dsdrho_inv,
            None,
            None,
            (drhoe_dP_rho * dsdrho_P - drhoe_drho_P * dsdP_rho + rho * cross)
            * dsdrho_inv,
        ),
        shape=b.shape,
        out=out,
    )


def chic_to_bcond(block, out=None):
    r"""Jacobian of a subsonic inflow's prescribed quantities with respect to characteristic variables.

    Analytically fused product
    :math:`\mathbf{J}_{\mathcal{C}\to\mathcal{B}}
    = \mathbf{J}_{\mathcal{P}\to\mathcal{B}}\,\mathbf{J}_{\mathcal{C}\to\mathcal{P}}`,
    i.e. :func:`primitive_to_bcond` :math:`\cdot` :func:`chic_to_primitive`.
    Rows 0-3 against the four incoming characteristic columns form the
    square system a non-reflecting inlet solves to drive its boundary
    condition residuals to zero; see :class:`~ember.patch.InletPatch`. The
    angle derivatives are as in :func:`primitive_to_bcond`, both measured
    against the meridional speed.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{C}\to\mathcal{B}}`, stacked on the
        trailing two axes.
    """
    b = block

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    a = b.a_nd
    dhdrho_P = b.dhdrho_P_nd
    dhdP_rho = b.dhdP_rho_nd
    dsdrho_P = b.dsdrho_P_nd
    dsdP_rho = b.dsdP_rho_nd
    asq_recip = 1.0 / a**2
    rhoa_recip = 1.0 / (rho * a)
    half_asq = asq_recip / 2.0
    half_rhoa_recip = rhoa_recip / 2.0

    Vm_sq = Vx**2 + Vr**2
    Vm = np.sqrt(Vm_sq)
    Vm_cb = Vm * Vm_sq
    tanAlpha = Vt / Vm
    dtanAl_dVx = -tanAlpha * Vx / Vm_sq
    dtanAl_dVr = -tanAlpha * Vr / Vm_sq
    dtanAl_dVt = 1.0 / Vm
    dsinBe_dVx = -Vr * Vx / Vm_cb
    dsinBe_dVr = Vx**2 / Vm_cb

    # Common sub-expressions for row 0
    half_dhdP = dhdP_rho / 2.0
    half_dhdrho_asq = dhdrho_P * half_asq

    # Common sub-expressions for row 1
    half_dsdP = dsdP_rho / 2.0
    half_dsdrho_asq = dsdrho_P * half_asq

    # The two acoustic characteristics enter every velocity derivative through
    # dVx = (c_down - c_up) / (2 rho a), so their columns share a magnitude and
    # differ only in sign.
    half_dtanAl = dtanAl_dVx * half_rhoa_recip
    half_dsinBe = dsinBe_dVx * half_rhoa_recip

    return _stack_matrix(
        # Row 0: d(ho)/d(chic)
        (
            half_dhdrho_asq - Vx * half_rhoa_recip + half_dhdP,
            half_dhdrho_asq + Vx * half_rhoa_recip + half_dhdP,
            Vr * rhoa_recip,
            Vt * rhoa_recip,
            -dhdrho_P * asq_recip,
        ),
        # Row 1: d(s)/d(chic)
        (
            half_dsdrho_asq + half_dsdP,
            half_dsdrho_asq + half_dsdP,
            None,
            None,
            -dsdrho_P * asq_recip,
        ),
        # Row 2: d(tanAlpha)/d(chic)
        (
            -half_dtanAl,
            half_dtanAl,
            dtanAl_dVr * rhoa_recip,
            dtanAl_dVt * rhoa_recip,
            None,
        ),
        # Row 3: d(sinBeta)/d(chic)
        (
            -half_dsinBe,
            half_dsinBe,
            dsinBe_dVr * rhoa_recip,
            None,
            None,
        ),
        # Row 4: d(P)/d(chic)
        (0.5, 0.5, None, None, None),
        shape=b.shape,
        out=out,
    )


def chic_to_mix(block, out=None):
    r"""Jacobian of a mixing plane's exchanged quantities with respect to characteristic variables.

    Analytically fused product
    :math:`\mathbf{J}_{\mathcal{C}\to\mathcal{M}}
    = \mathbf{J}_{\mathcal{P}\to\mathcal{M}}\,\mathbf{J}_{\mathcal{C}\to\mathcal{P}}`,
    i.e. :func:`primitive_to_mix` :math:`\cdot` :func:`chic_to_primitive`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{C}\to\mathcal{M}}`, stacked on the
        trailing two axes.
    """
    b = block

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    a = b.a_nd
    dhdrho_P = b.dhdrho_P_nd
    dhdP_rho = b.dhdP_rho_nd
    dsdrho_P = b.dsdrho_P_nd
    dsdP_rho = b.dsdP_rho_nd
    asq_recip = 1.0 / a**2
    rhoa_recip = 1.0 / (rho * a)
    half_asq = asq_recip / 2.0

    # Common sub-expressions for row 0
    half_dhdP = dhdP_rho / 2.0
    half_dhdrho_asq = dhdrho_P * half_asq

    # Common sub-expressions for row 1
    half_dsdP = dsdP_rho / 2.0
    half_dsdrho_asq = dsdrho_P * half_asq

    return _stack_matrix(
        # Row 0: d(ho)/d(chic)
        (
            half_dhdrho_asq - Vx * rhoa_recip / 2.0 + half_dhdP,
            half_dhdrho_asq + Vx * rhoa_recip / 2.0 + half_dhdP,
            Vr * rhoa_recip,
            Vt * rhoa_recip,
            -dhdrho_P * asq_recip,
        ),
        # Row 1: d(s)/d(chic)
        (
            half_dsdrho_asq + half_dsdP,
            half_dsdrho_asq + half_dsdP,
            None,
            None,
            -dsdrho_P * asq_recip,
        ),
        # Row 2: d(Vr)/d(chic)
        (None, None, rhoa_recip, None, None),
        # Row 3: d(Vt)/d(chic)
        (None, None, None, rhoa_recip, None),
        # Row 4: d(P)/d(chic)
        (0.5, 0.5, None, None, None),
        shape=b.shape,
        out=out,
    )


def flux_to_conserved(block):
    r"""Jacobian of conserved variables with respect to the x-direction advective flux.

    Computed as :func:`primitive_to_conserved` :math:`\cdot`
    :func:`flux_to_primitive`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{F}\to\mathcal{U}}`, stacked on the
        trailing two axes.
    """
    return util.matmat(primitive_to_conserved(block), flux_to_primitive(block))


def conserved_to_flux(block):
    r"""Jacobian of the x-direction advective flux with respect to conserved variables.

    Computed as :func:`primitive_to_flux` :math:`\cdot`
    :func:`conserved_to_primitive`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{U}\to\mathcal{F}}`, stacked on the
        trailing two axes.
    """
    return util.matmat(primitive_to_flux(block), conserved_to_primitive(block))
