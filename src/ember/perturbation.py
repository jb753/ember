"""Perturbation analysis and Jacobian matrices for linearized CFD operations.

This module provides analytical Jacobian matrices for transformations between primitive and
conserved flow variables, enabling linearized perturbation analysis for stability studies,
adjoint methods, and sensitivity analysis. The functions compute exact Jacobian matrices that
transform small perturbations between different variable sets (density, velocity, pressure
vs conserved mass, momentum, energy) accounting for thermodynamic relations specific to the
working fluid model. All Jacobians are computed pointwise on structured grids and returned
as stacked matrix arrays suitable for efficient batch linear algebra operations. This module
is essential for implementing Newton methods, linear stability analysis, and gradient-based
optimization in turbomachinery flow solvers.

All Jacobians operate in nondimensional space, using conserved_nd, r_nd, and _nd
thermodynamic derivative properties from the block.
"""

import numpy as np
from ember import util
import ember.fortran


def primitive_to_conserved(block, out=None):
    r"""Jacobian matrix for primitive to conserved variable transformation.

    Transforms perturbations in primitive variables :math:`\delta\mathcal{P}` to
    conserved variables :math:`\delta\mathcal{U}`:

    .. math::
        \delta\mathcal{U} = \mathbf{J}_{\mathcal{P}\mathcal{U}} \delta\mathcal{P}

    Where:
        :math:`\delta\mathcal{P} = [\delta\rho, \delta V_x, \delta V_r, \delta V_\theta, \delta p]^T`
        :math:`\delta\mathcal{U} = [\delta\rho, \delta(\rho V_x), \delta(\rho V_r), \delta(\rho r V_\theta), \delta(\rho e)]^T`

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for conserved to primitive variable transformation.

    Analytical inverse of primitive_to_conserved.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for primitive to characteristic variable transformation.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for characteristic to primitive variable transformation.

    Analytical inverse of primitive_to_chic.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for primitive to flux variable transformation.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for flux to primitive variable transformation.

    Analytical inverse of primitive_to_flux.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for primitive to boundary condition variable transformation.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
    """
    b = block

    q = b.conserved_nd
    r = b.r_nd
    rho = q[..., 0]
    Vx = q[..., 1] / rho
    Vr = q[..., 2] / rho
    Vt = q[..., 3] / (rho * r)
    # Both angles are measured against the meridional velocity magnitude, as in
    # Block.tanAlpha and Block.sinBeta, so every derivative below carries Vm
    # (meridional) and not the total speed.
    Vm_sq = Vx**2 + Vr**2
    Vm = np.sqrt(Vm_sq)
    Vm_cb = Vm * Vm_sq
    tanAlpha = Vt / Vm
    dtanAl_dVx = -tanAlpha * Vx / Vm_sq
    dtanAl_dVr = -tanAlpha * Vr / Vm_sq
    dtanAl_dVt = 1.0 / Vm

    # Use sinBeta instead of tanBeta to avoid singularity at Beta=90
    # sinBeta = Vr / sqrt(Vx^2 + Vr^2), derivatives verified numerically
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
    r"""Jacobian matrix for boundary condition to primitive variable transformation.

    Analytical inverse of primitive_to_bcond.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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

    # Vm is the meridional speed the two angles are measured against; Vsq is the
    # total speed squared. Eliminating the velocity components from the forward
    # matrix leaves the meridional projection u = Vx dVx + Vr dVr as the only
    # coupling, and u = Vm^2 (b0 - Vt Vm dtanAlpha) / Vsq with
    # b0 = dho - (dhdrho_P/dsdrho_P) ds + cross dP the stagnation enthalpy
    # residual once density has been eliminated via the entropy row.
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
    r"""Jacobian matrix for primitive to mix variable transformation.

    mix = [ho, s, Vr, Vt, P] where ho is stagnation enthalpy and s is entropy.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for mix to primitive variable transformation.

    Analytical inverse of primitive_to_mix.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for mix to conserved variable transformation.

    Analytically fused product of primitive_to_conserved @ mix_to_primitive.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for characteristic to boundary condition variable transformation.

    Analytically fused product of primitive_to_bcond @ chic_to_primitive.

    bcond = [ho, s, tanAlpha, sinBeta, P] are the four quantities a subsonic
    inflow specifies plus the static pressure; chic = [c_up, c_down, c_r, c_t,
    c_s] are the characteristic variables. Rows 0-3 against the four incoming
    characteristic columns form the square system a non-reflecting inlet solves
    to drive its boundary condition residuals to zero; see
    :class:`~ember.inlet_nonreflecting.NonReflectingInletPatch`.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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

    # Angle derivatives as in primitive_to_bcond: both measured against the
    # meridional speed.
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
    r"""Jacobian matrix for characteristic to mix variable transformation.

    Analytically fused product of primitive_to_mix @ chic_to_primitive.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
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
    r"""Jacobian matrix for flux to conserved variable transformation.

    Computed as primitive_to_conserved @ flux_to_primitive.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
    """
    return util.matmat(primitive_to_conserved(block), flux_to_primitive(block))


def conserved_to_flux(block):
    r"""Jacobian matrix for conserved to flux variable transformation.

    Computed as primitive_to_flux @ conserved_to_primitive.

    Returns:
        Array with shape (..., 5, 5) with matrices stacked on trailing dimensions
    """
    return util.matmat(primitive_to_flux(block), conserved_to_primitive(block))
