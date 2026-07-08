r"""Functions for integration and averaging over 2D blocks.

This module implements averaging methods for reducing spatially-varying 2D flow fields
to representative scalar quantities, essential for comparing CFD results with mean-line design
points and experimental measurements. The module provides area averaging, mass-flux averaging,
and mixed-out averaging methods following turbomachinery conventions. As discussed by
:cite:t:`Cumpsty2005`, dimensional reduction from 2D to 0D inherently loses information, requiring
careful selection of which flow properties to conserve. :cite:t:`Burdett2022` demonstrate
that mixed-out averages minimize sensitivity to streamwise location variations. All averaging
functions operate on 2D Block objects and support both absolute and relative reference frames
for turbomachinery with rotating components, enabling consistent performance metric extraction
across blade rows.
"""

import numpy as np

from ember import util
from ember import perturbation
from ember import fluxes as ember_fluxes
from ember import set_iter
from ember.geometry import node_to_face_2d


def _get_axes(axes, triangulated):
    """Helper to validate axes argument."""
    if triangulated:
        if axes is not None:
            raise ValueError("For triangulated grids, axes must be None.")
        axes = (0,)
    else:
        axes = (0, 1) if axes is None else axes
        if len(axes) > 2 or any(ax not in (0, 1) for ax in axes) or len(axes) == 0:
            raise ValueError(
                "For structured grids, axes must be a tuple of two integers from (0, 1), got {axes}."
            )
    return axes


def _get_dA(block):
    """Return face area array, dispatching to dA_tri or dA_quad based on block.triangulated."""
    return block.dA_tri if block.triangulated else block.dA_quad


def _integrate_scalar(scalar_face, dA_face, axes):
    """Helper function to integrate scalar over faces.

    Agnostic to quadrilateral vs triangulated faces."""
    assert scalar_face.ndim == dA_face.ndim - 1
    assert dA_face.shape[-1] == 3
    assert scalar_face.shape == dA_face.shape[:-1]
    return np.sum(scalar_face * util.vecnorm(dA_face), axis=axes)


def _integrate_vector(vector_face, dA_face, axes):
    """Helper function to integrate vector flux over faces.

    Agnostic to quadrilateral vs triangulated faces."""
    return np.sum(util.dot(vector_face, dA_face), axis=axes)


def _node_to_face(nodal_data, triangulated):
    """Distribute nodal data to faces."""
    if triangulated:
        # Average over triangle vertices
        return np.mean(nodal_data, axis=1)
    else:
        # Use 2D node_to_face distribution
        return node_to_face_2d(nodal_data)


def flow_mass(block, axes=None):
    r"""Integrate mass flow through faces of a 2D block.

    Calculates the mass flow over the block faces,

    .. math::

        \dot{m} = \int \rho \mathbf{V}\cdot\mathrm{d}\mathbf{A} \,.

    Parameters
    ----------
    block : Block, shape (ni, nj) or (ntri, 3)
        2D structured or triangulated block.
    axes : tuple of int, default (0, 1)
        For structured grids, axes over which to sum the mass flow;
        for triangulated grids, should be None to sum over all faces.

    Returns
    -------
    mass_flow: float
        Mass flow rate through the block.

    """

    mass_flux_face = ember_fluxes.get_mass_flux(block)
    axes = _get_axes(axes, block.triangulated)
    return _integrate_vector(mass_flux_face, _get_dA(block), axes)


def flow_conserved(block, axes=None):
    r"""Integrate conserved flows through faces of a 2D block.

    Calculates the conserved flows over the block faces,

    .. math::

        \int \mathcal{F}\cdot\mathrm{d}\mathbf{A} \,,

    where the conserved flux tensor :math:`\mathcal{F}` carries the fluxes of
    mass, axial momentum, radial momentum, angular momentum and stagnation
    enthalpy (energy),

    .. math::

        \mathcal{F} = \rho \mathbf{V}
        \begin{bmatrix} 1 \\ V_x \\ V_r \\ r V_\theta \\ h_0 \end{bmatrix}
        + p
        \begin{bmatrix}
            \mathbf{0} \\ \mathbf{e}_x \\ \mathbf{e}_r
            \\ r\,\mathbf{e}_\theta \\ \Omega r\,\mathbf{e}_\theta
        \end{bmatrix} \,.

    Parameters
    ----------
    block : Block, shape (ni, nj) or (ntri, 3)
        2D structured or triangulated block.
    axes : tuple of int, default (0, 1)
        For structured grids, axes over which to sum the flows;
        for triangulated grids, should be None to sum over all faces.

    Returns
    -------
    flow_conserved : Array shape (5,)
        Integrated conserved flows
    """
    axes = _get_axes(axes, block.triangulated)
    return util.dot_conserved(ember_fluxes.get_flux(block), _get_dA(block), axes)


def mass_average(scalar_node, block, axes=None):
    r"""Take mass-weighted average of a 2D nodal scalar field.

    Calculates the mass-weighted average of the scalar field :math:`\phi`,

    .. math::

        \bar{\phi} = \frac{\int \phi\, \rho \mathbf{V}\cdot\mathrm{d}\mathbf{A}}
                          {\int \rho \mathbf{V}\cdot\mathrm{d}\mathbf{A}} \,.

    Parameters
    ----------
    scalar_node : Array, shape (ni, nj) or (ntri, 3)
        Scalar field values at grid nodes
    block : Block, shape (ni, nj) or (ntri, 3)
        2D structured or triangulated block
    axes : tuple of int, default (0, 1)
        For structured grids, axes over which to average;
        for triangulated grids, should be None to average over all faces

    Returns
    -------
    avg_scalar :  float
        Mass-weighted average value

    Raises
    ------
    ValueError
        If the net mass flux through the block is zero
    """
    axes = _get_axes(axes, block.triangulated)
    mass_flux_vector = ember_fluxes.get_mass_flux(block)  # Shape (..., 3)
    scalar_face = _node_to_face(
        scalar_node, block.triangulated
    )  # Shape (...,) - scalar at faces

    # Calculate scalar mass flow through each face: phi * (rho V . dA)
    mass_flow_scalar = util.dot(
        mass_flux_vector, _get_dA(block)
    )  # Shape (...,) - scalar mass flow
    scalar_mass_flow = (
        scalar_face * mass_flow_scalar
    )  # Shape (...,) - scalar-weighted mass flow

    numerator = np.sum(scalar_mass_flow, axis=axes)
    denominator = flow_mass(block, axes)

    # Check for zero net mass flux
    if np.abs(denominator) < 1e-14:
        raise ValueError(
            "Net mass flux through the block is zero. "
            "Mass averaging requires non-zero net mass flux. "
            "Consider using area_average() instead."
        )

    return numerator / denominator


def area_average(scalar_node, block, axes=None):
    r"""Take area-weighted average of a 2D nodal scalar field.

    Calculates the area-weighted average of the scalar field :math:`\phi`,

    .. math::

        \bar{\phi} = \frac{\int \phi\, \mathrm{d}A}{\int \mathrm{d}A} \,.

    Parameters
    ----------
    scalar_node : Array, shape (ni, nj) or (ntri, 3)
        Scalar field values at grid nodes
    block : Block, shape (ni, nj) or (ntri, 3)
        2D structured or triangulated block
    axes : tuple of int, default (0, 1)
        For structured grids, axes over which to average;
        for triangulated grids, should be None to average over all faces

    Returns
    -------
    avg_scalar : float
        Area-weighted average value
    """
    axes = _get_axes(axes, block.triangulated)
    scalar_face = _node_to_face(scalar_node, block.triangulated)
    numerator = _integrate_scalar(scalar_face, _get_dA(block), axes)
    denominator = _integrate_scalar(np.ones_like(scalar_face), _get_dA(block), axes)
    return numerator / denominator


def total_area(block):
    r"""Compute total vector area of a 2D block.

    Calculates the total vector area as the integral of the face area vectors,

    .. math::

        \mathbf{A} = \int \mathrm{d}\mathbf{A} \,.

    Parameters
    ----------
    block : Block, shape (ni, nj) or (ntri, 3)
        2D structured or triangulated block

    Returns
    -------
    A : Array, shape (3,)
        Total area of the cut [m^2] in polar coordinates (Ax, Ar, At)
    """
    assert block.ndim == 2
    axes = _get_axes(None, block.triangulated)
    return np.sum(_get_dA(block), axis=axes)


def mix_out(block, AR=1.0):
    r"""Mix out a 2D cut to uniformity, optionally through a contracted area.

    The mixed-out state is the uniform flow that, passed through the total
    area :math:`\mathbf{A} = \int \mathrm{d}\mathbf{A}`, carries the same
    conserved flows as the non-uniform cut. Its conserved variables
    :math:`\mathcal{U}` are found by solving

    .. math::

        \mathcal{F}(\mathcal{U})\cdot\mathbf{A}
        = \int \mathcal{F}\cdot\mathrm{d}\mathbf{A} \,,

    for the five conserved flows (mass, axial and radial momentum, angular
    momentum and energy), where :math:`\mathcal{F}` is the flux tensor of
    :func:`flow_conserved`. The five equations are solved iteratively by
    Newton steps on :math:`\mathcal{U}`. Mixing to uniformity generates
    entropy, so the result has higher entropy than the original state.

    The optional area ratio ``AR`` then contracts (``AR<1``) or expands the
    uniform state isentropically from :math:`\mathbf{A}` to ``AR`` times
    :math:`\mathbf{A}`, conserving mass, stagnation enthalpy, entropy and
    angular momentum :math:`r V_\theta` (at fixed radius) while holding the
    pitch angle :math:`\beta`. This second step is reversible, so the mixing
    loss is independent of ``AR`` and ``AR=1`` recovers the plain mix-out. The
    contraction stays on the mixed-out sub/supersonic branch and raises
    :class:`RuntimeError` if it would choke.

    Parameters
    ----------
    block : Block, shape (ni, nj) or (ntri, 3)
        2D block, can be structured or triangulated.
    AR : float, default 1.0
        Area ratio for the isentropic contraction applied after mixing out.
        ``AR=1`` retains the constant-area mix-out; ``AR<1`` contracts the
        uniform state to area ``AR * A``.

    Returns
    -------
    mix : Block, shape ()
        New scalar block with mixed-out uniform state.

    """

    # Calculate total area and conserved quantities
    A = total_area(block)
    flow = flow_conserved(block)

    # Ensure that mass flow is positive
    if flow[0] <= 0.0:
        A *= -1.0
        flow *= -1.0

    # Do not allow significant projected area in theta direction
    assert np.all(np.abs(A[2]) < 1e-6 * np.linalg.norm(A[:2])), (
        f"Block has significant projected area in theta direction: A={A}"
    )
    A = A[:2]  # Drop theta component
    A_ref = np.linalg.norm(A)

    # Allocate a scalar block for the mixed-out state
    mix = block.empty()

    # Mixed out coordinates
    rmix = util.rms([block.r.min(), block.r.max()])
    xmix = util.bounds(block.x).mean()
    tmix = util.bounds(block.t).mean()
    mix.set_x(xmix)
    mix.set_r(rmix)
    mix.set_t(tmix)

    # Initial guess for conserved variables simple mean
    mix.set_conserved(block.conserved.mean(axis=(0, 1)))

    # Get absolute tolerance for flows
    rho_ref = mix.rho
    V_ref = mix.V
    rhoV_ref = rho_ref * V_ref
    rhoVsq_ref = rho_ref * V_ref**2
    de_ref = rho_ref * V_ref**3
    atol = (
        np.array(
            [
                rhoV_ref,
                rhoVsq_ref,
                rhoVsq_ref,
                rhoVsq_ref * rmix,
                de_ref,
            ]
        )
        * A_ref
        * 1e-4
    )

    # Reference scales for converting between dimensional and nondimensional
    # Jacobians operate in ND space, so we scale err_flux to ND, apply J, scale back
    _rho_ref = mix.fluid.rho_ref
    _V_ref = mix.fluid.V_ref
    _L_ref = mix.L_ref
    _rhoV_ref = _rho_ref * _V_ref
    _flux_ref = np.array(
        [
            _rhoV_ref,
            _rhoV_ref * _V_ref,
            _rhoV_ref * _V_ref,
            _rhoV_ref * _V_ref * _L_ref,
            _rhoV_ref * _V_ref**2,
        ]
    )
    _cons_ref = np.array(
        [
            _rho_ref,
            _rhoV_ref,
            _rhoV_ref,
            _rhoV_ref * _L_ref,
            _rhoV_ref * _V_ref,
        ]
    )

    # Iteratively adjust conserved variables to match total flow
    rf = 0.1
    max_iter = 10000
    err_flow = np.inf
    for niter in range(max_iter):
        # Calculate current fluxes and flows (xr system)
        flux_mix = ember_fluxes.get_flux(mix)[:2, :]
        flow_mix = util.dot_conserved(flux_mix, A, axes=())
        err_flow = flow - flow_mix
        err_flux = err_flow / A_ref

        # Update error
        if np.all(np.abs(err_flow) < atol):
            break

        # Resolve to interface-aligned velocities
        Beta = mix.Beta
        util.resolve_to_interface(mix, Beta)

        # Calculate Jacobian of conserved/flux transformation (nondimensional)
        f2c = perturbation.flux_to_conserved(mix)

        # Scale flux error to ND, apply ND Jacobian, scale correction back to dim
        dcons = util.matvec(f2c, err_flux / _flux_ref) * _cons_ref

        # Apply relaxation to avoid overshoot
        mix.set_conserved(mix.conserved + rf * dcons)

        if mix.rho < 0.0:
            print(
                f"  NEGATIVE DENSITY at iter {niter}: rho={mix.rho:.6g}, dcons={dcons}"
            )
            raise Exception("Negative density")

        # Resolve back to physical velocities
        util.resolve_from_interface(mix, Beta)

    if (np.abs(err_flow) >= atol).any():
        print(f"  FAILED after {max_iter} iters: err_flow={err_flow}, atol={atol}")
        print(f"  final conserved: {mix.conserved}")
        raise RuntimeError(
            f"Failed to converge mixing after {max_iter} iterations, err_flow={err_flow}, atol={atol}"
        )

    # Optionally contract the uniform mixed-out state isentropically from area
    # A to AR*A. This is a reversible second step that leaves the mixing loss
    # (computed above at the true area) unchanged.
    if AR != 1.0:
        # Only subsonic meridional flow is supported by the contraction solver.
        if mix.Vm >= mix.a:
            raise NotImplementedError(
                "Isentropic contraction (AR != 1) is only supported for "
                f"subsonic meridional flow (got Vm={mix.Vm:.6g} >= a={mix.a:.6g})."
            )

        # Meridional mass flux through the contracted area AR*A. Mass is
        # conserved, so rho*Vm scales as 1/AR.
        rhoVm_target = mix.rho * mix.Vm / AR

        set_iter.set_ho_s_rhoVm_Vt_Beta(
            mix,
            mix.ho,
            mix.s,
            rhoVm_target,
            mix.Vt,
            Beta=mix.Beta,
        )

    return mix
