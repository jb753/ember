"""Flux computation and boundary condition application using Fortran kernels.

This module provides high-performance flux calculations for finite volume CFD simulations
using Fortran-accelerated routines. Functions compute inviscid fluxes using Roe's approximate
Riemann solver, viscous fluxes with gradient-based stress and heat transfer terms, and
artificial dissipation for numerical stability. The module handles boundary condition
enforcement for walls, inlets, outlets, and periodic patches, including specialized treatment
for wall distance calculations and wall friction models. Key utilities include classification
of patch types for different physical treatments (impermeable walls vs frictionless boundaries)
and efficient batch processing of multi-block grids. All flux routines are optimized for
structured grids in polar coordinates and support both 2D and 3D domains.
"""

import numpy as np

f32 = np.float32


def get_flux_node(block):
    """Calculate flux for 0D/1D blocks using pure Python.

    For blocks without face areas, computes flux vector at nodes.

    Parameters
    ----------
    block : Block
        The block to calculate fluxes for (ndim 0 or 1)

    Returns
    -------
    Array
        0D blocks: (3, 5) - flux vector [x, r, t] × [rho, rhoVx, rhoVr, rhorVt, rhoe]
        1D blocks: (ni-1, 3, 5) - flux at each i-face
    """
    if block.ndim > 1:
        raise ValueError(f"get_flux_node only for 0D/1D blocks, got {block.ndim}D")

    # Compute mass flux vector at nodes: rho * V
    mass_flux = block.Vxrt * block.rho[..., None]  # Shape: (..., 3)

    # Conserved variables per unit mass at nodes
    cons_per_mass = np.stack(
        (
            np.ones_like(block.rho),  # mass per unit mass
            block.Vxrt[..., 0],  # axial momentum per unit mass
            block.Vxrt[..., 1],  # radial momentum per unit mass
            block.Vxrt[..., 2] * block.r,  # angular momentum per unit mass
            block.ho,  # stagnation enthalpy (energy per unit mass)
        ),
        axis=-1,
    )  # Shape: (..., 5)

    if block.ndim == 0:
        # For 0D: return flux vector at single point
        # flux[direction, conserved_var] = mass_flux[direction] * cons_per_mass[conserved_var]
        # mass_flux is scalar (3,), cons_per_mass is scalar (5,)
        flux = mass_flux[:, None] * cons_per_mass[None, :]  # (3, 5)

        # Add pressure contributions
        P = block.P.item() if hasattr(block.P, "item") else block.P
        r = block.r.item() if hasattr(block.r, "item") else block.r
        flux[0, 1] += P  # x-momentum flux
        flux[1, 2] += P  # r-momentum flux
        flux[2, 3] += P * r  # angular momentum flux (with radius factor)
        flux[2, 4] += block.Omega * r * P  # energy flux (with rotation)

    else:  # ndim == 1
        # For 1D: compute flux at each i-face by averaging adjacent nodes
        mass_flux_avg = (mass_flux[:-1] + mass_flux[1:]) / 2  # (ni-1, 3)
        cons_per_mass_avg = (cons_per_mass[:-1] + cons_per_mass[1:]) / 2  # (ni-1, 5)

        # flux[i-face, direction, conserved_var]
        flux = mass_flux_avg[..., None] * cons_per_mass_avg[:, None, :]  # (ni-1, 3, 5)

        # Add pressure contributions (averaged at faces)
        P_avg = (block.P[:-1] + block.P[1:]) / 2  # (ni-1,)
        r_avg = (block.r[:-1] + block.r[1:]) / 2  # (ni-1,)
        flux[:, 0, 1] += P_avg  # x-momentum flux
        flux[:, 1, 2] += P_avg  # r-momentum flux
        flux[:, 2, 3] += P_avg * r_avg  # angular momentum flux
        flux[:, 2, 4] += block.Omega * r_avg * P_avg  # energy flux

    return flux


def get_flux_quad(block):
    """Calculate flux for 2D structured quadrilateral blocks using pure Python.

    Computes flux at cell centers by averaging from corner nodes.

    Parameters
    ----------
    block : Block
        The block to calculate fluxes for (ndim 2, structured)

    Returns
    -------
    Array
        Shape (ni-1, nj-1, 3, 5) - flux at each quad cell
    """
    if block.ndim != 2:
        raise ValueError(f"get_flux_quad only for 2D blocks, got {block.ndim}D")
    if block.triangulated:
        raise ValueError(
            "get_flux_quad only for structured blocks, use get_flux_tri for triangulated"
        )

    # Compute mass flux vector at nodes: rho * V
    mass_flux = block.Vxrt * block.rho[..., None]  # Shape: (ni, nj, 3)

    # Conserved variables per unit mass at nodes
    cons_per_mass = np.stack(
        (
            np.ones_like(block.rho),  # mass per unit mass
            block.Vxrt[..., 0],  # axial momentum per unit mass
            block.Vxrt[..., 1],  # radial momentum per unit mass
            block.Vxrt[..., 2] * block.r,  # angular momentum per unit mass
            block.ho,  # stagnation enthalpy (energy per unit mass)
        ),
        axis=-1,
    )  # Shape: (ni, nj, 5)

    # Average from 4 corner nodes to cell centers
    # For a quad with corners (i,j), (i+1,j), (i,j+1), (i+1,j+1)
    mass_flux_avg = (
        mass_flux[:-1, :-1]
        + mass_flux[1:, :-1]
        + mass_flux[:-1, 1:]
        + mass_flux[1:, 1:]
    ) / 4  # (ni-1, nj-1, 3)

    cons_per_mass_avg = (
        cons_per_mass[:-1, :-1]
        + cons_per_mass[1:, :-1]
        + cons_per_mass[:-1, 1:]
        + cons_per_mass[1:, 1:]
    ) / 4  # (ni-1, nj-1, 5)

    # Compute flux tensor
    # flux[i, j, direction, conserved_var]
    flux = (
        mass_flux_avg[..., None] * cons_per_mass_avg[..., None, :]
    )  # (ni-1, nj-1, 3, 5)

    # Add pressure contributions (averaged at cell centers)
    P_avg = (
        block.P[:-1, :-1] + block.P[1:, :-1] + block.P[:-1, 1:] + block.P[1:, 1:]
    ) / 4  # (ni-1, nj-1)

    r_avg = (
        block.r[:-1, :-1] + block.r[1:, :-1] + block.r[:-1, 1:] + block.r[1:, 1:]
    ) / 4  # (ni-1, nj-1)

    flux[:, :, 0, 1] += P_avg  # x-momentum flux
    flux[:, :, 1, 2] += P_avg  # r-momentum flux
    flux[:, :, 2, 3] += P_avg * r_avg  # angular momentum flux
    flux[:, :, 2, 4] += block.Omega * r_avg * P_avg  # energy flux

    return flux


def get_mass_flux(block):
    """Mass-flux vector rho*V at cell centres, shape (..., 3).

    This is exactly ``get_flux(block)[..., 0]`` (the mass column of the full
    conserved-flux tensor) but computed directly, for callers that only need the
    mass flux -- :func:`ember.average.flow_mass` and
    :func:`ember.average.mass_average` on the convergence-logging hot path. For a
    2D structured block it skips the ``cons_per_mass`` stack, the (..., 3, 5)
    outer product and the pressure terms that :func:`get_flux_quad` builds and
    then discards, cutting the allocation and work by ~5x. The result is
    bit-identical: the mass column's ``cons_per_mass`` factor is ``ones`` (so its
    4-corner average is exactly 1.0) and the pressure terms only touch the
    momentum/energy columns.

    Triangulated and 0D/1D blocks fall back to the general
    :func:`get_flux`, which is cold post-processing rather than the logged path.
    """
    if block.ndim == 2 and not block.triangulated:
        # 4-corner average of nodal rho*V onto quad-cell centres.
        mass_flux = block.Vxrt * block.rho[..., None]  # (ni, nj, 3)
        return (
            mass_flux[:-1, :-1]
            + mass_flux[1:, :-1]
            + mass_flux[:-1, 1:]
            + mass_flux[1:, 1:]
        ) / 4  # (ni-1, nj-1, 3)
    return get_flux(block)[..., 0]


def get_flux_tri(block):
    """Calculate flux for triangulated 2D blocks using pure Python.

    Parameters
    ----------
    block : Block
        The block to calculate fluxes for (ndim 2, triangulated)

    Returns
    -------
    Array
        Shape (ntri, 3, 5) - flux at each triangle edge
    """
    if block.ndim != 2:
        raise ValueError(f"get_flux_tri only for 2D blocks, got {block.ndim}D")
    if not block.triangulated:
        raise ValueError(
            "get_flux_tri only for triangulated blocks, use get_flux_quad for structured"
        )
    if block.shape[1] != 3:
        raise ValueError(
            f"Triangulated blocks must have shape (ntri, 3), got {block.shape}"
        )

    # Compute mass flux at nodes
    mass_flux = block.Vxrt * block.rho[..., None]

    # Conserved variables per unit mass at nodes
    cons_per_mass = np.stack(
        (
            np.ones_like(block.rho),  # mass per unit mass
            block.Vxrt[..., 0],  # axial momentum per unit mass
            block.Vxrt[..., 1],  # radial momentum per unit mass
            block.Vxrt[..., 2] * block.r,  # angular momentum per unit mass
            block.ho,  # energy per unit mass
        ),
        axis=-1,
    )

    # Triangle-averaged mass flux and conserved quantities per unit mass
    mass_flux_avg = mass_flux.mean(axis=1)  # (ntri, 3)
    cons_per_mass_avg = cons_per_mass.mean(axis=1)  # (ntri, 5)

    # Compute flux by broadcasting and multiplying
    flux = mass_flux_avg[..., None] * cons_per_mass_avg[:, None, :]

    # Add pressure contributions
    p_avg = block.P.mean(axis=1)  # (ntri,)
    r_avg = block.r.mean(axis=1)  # (ntri,)
    flux[:, 0, 1] += p_avg  # x-momentum flux
    flux[:, 1, 2] += p_avg  # r-momentum flux
    flux[:, 2, 3] += p_avg * r_avg  # angular momentum flux (with radius factor)
    flux[:, 2, 4] += block.Omega * r_avg * p_avg  # energy flux (with rotation)

    return flux


def get_flux(block):
    """Calculate flux for 0D, 1D, or 2D blocks using pure Python.

    Dispatcher that delegates to appropriate pure Python implementation based on
    block dimensionality and triangulation.

    Parameters
    ----------
    block : Block
        The block to calculate fluxes for (ndim 0, 1, or 2)

    Returns
    -------
    Array
        0D blocks: (3, 5) - flux vector
        1D blocks: (ni-1, 3, 5) - flux at each i-face
        2D structured: (ni-1, nj-1, 3, 5) - flux at each quad cell
        2D triangulated: (ntri, 3, 5) - flux at each triangle

    Raises
    ------
    ValueError
        If block is 3D (use Fortran implementation via SolverBlock instead)
    """
    if block.ndim == 3:
        raise ValueError(
            "For 3D blocks, use fluxi, fluxj, fluxk directly instead of flux"
        )

    # Dispatch to appropriate pure Python implementation
    if block.triangulated:
        return get_flux_tri(block)
    elif block.ndim <= 1:
        return get_flux_node(block)
    else:  # ndim == 2, structured
        return get_flux_quad(block)
