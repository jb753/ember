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
from ember import block_util
from ember import perturbation
from ember import fluxes as ember_fluxes
from ember import set_iterative


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


def _get_dA_nd(block):
    """Nondimensional counterpart of :func:`_get_dA`."""
    return block.dA_tri_nd if block.triangulated else block.dA_quad_nd


def _dot_conserved(flux, dA, axes):
    """Specialized dot product for conserved variable fluxes."""
    return np.sum(
        np.einsum(
            "...ij,...i->...j",
            flux,
            dA,
        ),
        axis=axes,
    )


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


def _node_to_face_2d(nodal_data):
    r"""Average nodal values to face centres for 2D data.

    For a :math:`(n_i, n_j)` array of nodal values, the face-centred value is

    .. math::

        \bar{q}_{i,j} = \tfrac{1}{4}\bigl(
            q_{i,j} + q_{i+1,j} + q_{i+1,j+1} + q_{i,j+1}\bigr)

    Parameters
    ----------
    nodal_data : Array, shape (ni, nj, ...)
        Values at grid nodes.

    Returns
    -------
    Array, shape (ni-1, nj-1, ...)
        Values averaged to face centres.
    """
    return 0.25 * (
        nodal_data[:-1, :-1, ...]
        + nodal_data[1:, :-1, ...]
        + nodal_data[1:, 1:, ...]
        + nodal_data[:-1, 1:, ...]
    )


def _node_to_face(nodal_data, triangulated):
    """Distribute nodal data to faces."""
    if triangulated:
        # Average over triangle vertices
        return np.mean(nodal_data, axis=1)
    else:
        # Use 2D node_to_face distribution
        return _node_to_face_2d(nodal_data)


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
    return (
        flow_conserved_nd(block, axes) * ember_fluxes.flux_ref(block) * block.L_ref**2
    )


def flow_conserved_nd(block, axes=None):
    r"""Nondimensional counterpart of :func:`flow_conserved`, shape (5,).

    Scaled by ``fluxes.flux_ref(block) * block.L_ref**2``, this is
    :func:`flow_conserved`. Working directly in this form lets :func:`mix_out`
    run its Newton iteration without any reference-scale bookkeeping, since
    the Jacobian from :mod:`ember.perturbation` is nondimensional too.
    """
    axes = _get_axes(axes, block.triangulated)
    return _dot_conserved(ember_fluxes.get_flux_nd(block), _get_dA_nd(block), axes)


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
    return total_area_nd(block) * block.L_ref**2


def total_area_nd(block):
    r"""Nondimensional counterpart of :func:`total_area`, shape (3,).

    Total vector area divided by :math:`L_\mathrm{ref}^2`.
    """
    assert block.ndim == 2
    axes = _get_axes(None, block.triangulated)
    return np.sum(_get_dA_nd(block), axis=axes)


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

    # Calculate total area and conserved quantities. The whole Newton loop
    # below runs nondimensionally, matching the ND Jacobian from
    # ember.perturbation, so no reference scales appear in it at all.
    A = total_area_nd(block)
    flow = flow_conserved_nd(block)

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
    rmix = np.sqrt(np.mean(np.array([block.r.min(), block.r.max()]) ** 2))
    xmix = util.extent(block.x).mean()
    tmix = util.extent(block.t).mean()
    mix.set_x(xmix)
    mix.set_r(rmix)
    mix.set_t(tmix)

    # Initial guess for conserved variables simple mean, written direct on the
    # ND storage. This skips set_conserved's density/finite validation, which
    # a mean of valid input data cannot trip, and the loop's own mix.rho check
    # still guards against Newton-step divergence below.
    mix.conserved_nd[...] = block.conserved_nd.mean(axis=(0, 1))
    mix.update_cached_conserved()

    # Get absolute tolerance for flows, on the same ND scale as the flows
    rho_ref = mix.rho_nd
    V_ref = mix.V_nd
    rhoV_ref = rho_ref * V_ref
    rhoVsq_ref = rho_ref * V_ref**2
    de_ref = rho_ref * V_ref**3
    atol = (
        np.array(
            [
                rhoV_ref,
                rhoVsq_ref,
                rhoVsq_ref,
                rhoVsq_ref * block.r_mid_nd,
                de_ref,
            ]
        )
        * A_ref
        * 1e-4
    )

    # Iteratively adjust conserved variables to match total flow
    rf = 0.1
    max_iter = 10000
    n_stall = 50
    err_flow = np.inf
    best_err = np.inf
    best_iter = 0
    for niter in range(max_iter):
        # Calculate current fluxes and flows (xr system)
        flux_mix = ember_fluxes.get_flux_nd(mix)[:2, :]
        flow_mix = _dot_conserved(flux_mix, A, axes=())
        err_flow = flow - flow_mix
        err_flux = err_flow / A_ref

        # Stop once the residual stops improving, not the moment it first drops
        # below atol. Breaking at the atol crossing makes the answer depend on
        # which iterate happens to land inside the tolerance ball first, and
        # that in turn depends on the last bit of the input ordering -- mixing
        # a cut and its k-axis-reversed twin then differ by ~3x the atol scale.
        # Iterating on to the float32 fixed point costs ~2x the iterations and
        # brings that difference down to the storage floor (~2e-6). atol
        # survives below as the scale that makes the five residuals comparable
        # and as the post-loop divergence check.
        err_scaled = np.max(np.abs(err_flow) / atol)
        if err_scaled < best_err * (1.0 - 1e-6):
            best_err = err_scaled
            best_iter = niter
        elif niter - best_iter > n_stall:
            break

        # Resolve to interface-aligned velocities. Beta moves every iteration
        # as mix's state does, so the matrix can't be hoisted out of the loop
        # -- but building both directions from it here, once, means the
        # to/from pair below shares one sine and cosine rather than each
        # re-deriving its own.
        Beta = mix.Beta
        rot_to, rot_from = util.rotation_matrices(np.radians(Beta))
        block_util.resolve_to_interface(mix, rot_to)

        # Calculate Jacobian of conserved/flux transformation (nondimensional)
        f2c = perturbation.flux_to_conserved(mix)

        # Apply the ND Jacobian to the (already ND) flux error to get the ND
        # conserved correction directly
        dcons_nd = util.matvec(f2c, err_flux)

        # Apply relaxation to avoid overshoot, direct on the ND storage
        mix.conserved_nd[...] += rf * dcons_nd
        mix.update_cached_conserved()

        if mix.rho < 0.0:
            print(
                f"  NEGATIVE DENSITY at iter {niter}: rho={mix.rho:.6g}, dcons_nd={dcons_nd}"
            )
            raise Exception("Negative density")

        # Resolve back to physical velocities
        block_util.resolve_from_interface(mix, rot_from)

    if (np.abs(err_flow) >= atol).any():
        print(f"  FAILED after {niter + 1} iters: err_flow={err_flow}, atol={atol}")
        print(f"  final conserved: {mix.conserved}")
        raise RuntimeError(
            f"Failed to converge mixing after {niter + 1} iterations, err_flow={err_flow}, atol={atol}"
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

        set_iterative.set_ho_s_rhoVm_Vt_Beta(
            mix,
            mix.ho,
            mix.s,
            rhoVm_target,
            mix.Vt,
            Beta=mix.Beta,
        )

    return mix
