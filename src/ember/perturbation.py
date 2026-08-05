r"""Analytic Jacobians between the flow variable sets used throughout EMBER.

Six five-component variable sets describe the same flow state, and boundary
conditions, the characteristic treatment and the mixing planes all need to
move small perturbations between them:

* **primitive** -- :math:`[\rho, V_x, V_r, V_\theta, P]`
* **conserved** -- :math:`[\rho, \rho V_x, \rho V_r, \rho r V_\theta, \rho e]`,
  what :meth:`~ember.block.Block.set_conserved` stores
* **flux** -- the x-direction advective flux of each conserved variable,
  :math:`[\rho V_x, \rho V_x^2 + P, \rho V_x V_r, \rho V_x r V_\theta, \rho V_x h_0]`
* **chic** -- Giles' one-dimensional characteristic variables
  :math:`[c_\mathrm{up}, c_\mathrm{down}, c_r, c_t, c_s]`
* **bcond** -- :math:`[h_0, s, \tan\alpha, \sin\beta, P]`, what a subsonic
  inflow prescribes; see :class:`~ember.inlet.InletPatch`
* **mix** -- :math:`[h_0, s, V_r, V_\theta, P]`, what a mixing plane exchanges;
  see :class:`~ember.mixing_nonreflecting.NonReflectingMixingPatch`

Each function below is named ``x_to_y`` and returns the Jacobian
:math:`\partial y/\partial x`, evaluated pointwise from a
:class:`~ember.block.Block`'s current state and returned as a batch of 5x5
matrices stacked on the trailing two axes. An inverse transformation between
two of these sets is computed analytically rather than by matrix inversion
(:func:`conserved_to_primitive` is the analytical inverse of
:func:`primitive_to_conserved`, and so on); a transformation between two sets
neither of which is primitive is fused into a single evaluation rather than
composed from two Jacobians at call time (:func:`chic_to_bcond`, for
instance, is the fused product of :func:`primitive_to_bcond` and
:func:`chic_to_primitive`).

All Jacobians are evaluated in the nondimensional space the block stores its
state in, using :attr:`~ember.block.Block.conserved_nd`,
:attr:`~ember.block.Block.r_nd` and the fluid's ``_nd`` thermodynamic
derivative properties.
"""

import numpy as np
from ember import util
import ember.fortran


def primitive_to_conserved(block, out=None):
    r"""Jacobian of conserved variables with respect to primitive variables.

    .. math::
        \delta\mathcal{U} = \mathbf{J}_{\mathcal{P}\mathcal{U}} \delta\mathcal{P}

    for primitive :math:`\delta\mathcal{P} = [\delta\rho, \delta V_x, \delta
    V_r, \delta V_\theta, \delta P]^T` and conserved :math:`\delta\mathcal{U}
    = [\delta\rho, \delta(\rho V_x), \delta(\rho V_r), \delta(\rho r
    V_\theta), \delta(\rho e)]^T`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{P}\mathcal{U}}`, stacked on the trailing
        two axes.
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
        :math:`\mathbf{J}_{\mathcal{U}\mathcal{P}}`, stacked on the trailing
        two axes.
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

    out = util.stack_matrix(
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
    r"""Jacobian of Giles' characteristic variables with respect to primitive variables.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathcal{P}\mathrm{c}}`, stacked on the trailing
        two axes.
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
    r"""Jacobian of primitive variables with respect to Giles' characteristic variables.

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
        :math:`\mathbf{J}_{\mathrm{c}\mathcal{P}}`, stacked on the trailing
        two axes.
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
        :math:`\mathbf{J}_{\mathcal{P}\mathrm{F}}`, stacked on the trailing
        two axes.
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

    return util.stack_matrix(
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
        :math:`\mathbf{J}_{\mathrm{F}\mathcal{P}}`, stacked on the trailing
        two axes.
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
        :math:`\mathbf{J}_{\mathcal{P}\mathrm{b}}`, stacked on the trailing
        two axes.
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

    return util.stack_matrix(
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
    and :math:`V^2 = V_m^2 + V_\theta^2` the total speed squared. Eliminating
    the velocity components from the forward matrix leaves the meridional
    projection :math:`u = V_x\,\delta V_x + V_r\,\delta V_r` as the only
    coupling, with

    .. math::
        u = \frac{V_m^2}{V^2}\left(b_0 - V_\theta V_m\,\delta\tan\alpha\right),
        \qquad
        b_0 = \delta h_0 - \frac{\partial h_0/\partial\rho|_P}{\partial s/\partial\rho|_P}\,\delta s
              + \text{cross}\cdot\delta P

    the stagnation enthalpy residual once density has been eliminated via the
    entropy row.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathrm{b}\mathcal{P}}`, stacked on the trailing
        two axes.
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
        :math:`\mathbf{J}_{\mathcal{P}\mathrm{m}}`, stacked on the trailing
        two axes.
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
        :math:`\mathbf{J}_{\mathrm{m}\mathcal{P}}`, stacked on the trailing
        two axes.
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

    Analytically fused product :func:`primitive_to_conserved`
    :math:`\cdot` :func:`mix_to_primitive`, rather than composing the two at
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
        :math:`\mathbf{J}_{\mathrm{m}\mathcal{U}}`, stacked on the trailing
        two axes.
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

    return util.stack_matrix(
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

    Analytically fused product :func:`primitive_to_bcond` :math:`\cdot`
    :func:`chic_to_primitive`. Rows 0-3 against the four incoming
    characteristic columns form the square system a non-reflecting inlet
    solves to drive its boundary condition residuals to zero; see
    :class:`~ember.inlet.InletPatch`. The angle derivatives are as in
    :func:`primitive_to_bcond`, both measured against the meridional speed.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathrm{c}\mathrm{b}}`, stacked on the trailing
        two axes.
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

    return util.stack_matrix(
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

    Analytically fused product :func:`primitive_to_mix` :math:`\cdot`
    :func:`chic_to_primitive`.

    Parameters
    ----------
    block : Block
        Block whose current state the Jacobian is evaluated at.
    out : ndarray, optional
        Pre-allocated output array, shape ``(*block.shape, 5, 5)``.

    Returns
    -------
    jac : ndarray, shape (..., 5, 5)
        :math:`\mathbf{J}_{\mathrm{c}\mathrm{m}}`, stacked on the trailing
        two axes.
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

    return util.stack_matrix(
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
        :math:`\mathbf{J}_{\mathrm{F}\mathcal{U}}`, stacked on the trailing
        two axes.
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
        :math:`\mathbf{J}_{\mathcal{U}\mathrm{F}}`, stacked on the trailing
        two axes.
    """
    return util.matmat(primitive_to_flux(block), conserved_to_primitive(block))
