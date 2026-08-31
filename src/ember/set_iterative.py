r"""Iterative setter functions for :class:`~ember.block.Block` objects.

This module provides iterative solvers for initialising
:class:`~ember.block.Block` objects with flow fields from combined
thermodynamic and kinematic variable sets that do not have a closed-form
solution for the corresponding conserved variables. For example, stagnation
pressure depends on both velocity and static pressure. These functions run a
fixed-point iteration on density to solve the ensuing implicit equations, then
modify the block in place with the converged flow field. Like the non-iterative
setters such as :func:`~ember.block.Block.set_P_T`, inputs must be
broadcastable to the block shape.

The prescribed stagnation enthalpy and flow angles are in the absolute
(stationary) frame; the velocity magnitude is closed by a Mach number, a
meridional mass flux, or a fixed swirl.

.. autosummary::

   set_ho_s_Ma_Alpha_Beta
   set_ho_s_rhoVm_Alpha_Beta
   set_ho_s_rhoVm_Vt_Beta

For rotating rows: rothalpy (or relative total conditions) and relative flow
angles are prescribed, and the energy balance is closed with the absolute
velocity through the local blade speed :math:`U = r\,\Omega`. The block must
carry its radial coordinate before these are called (angular velocity defaults
to zero if not set).

.. autosummary::

   set_I_s_Ma_rel_Alpha_rel_Beta
   set_Po_To_Ma_rel_Alpha_rel_Beta

Example usage
=============

Initialise a block from stagnation conditions, a relative Mach number that
ramps along the block, and flow angles:

.. code-block:: python

    import numpy as np
    import ember.block
    import ember.fluid
    import ember.set_iterative

    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    block = ember.block.Block(shape=(4,))
    block.set_fluid(fluid)
    block.set_r(1.0)  # blade speed is U = r * Omega, so the radius is required
    block.set_Omega(100.0)

    # Absolute Po [Pa] and To [K], relative Mach, yaw and pitch [deg]. Scalars
    # broadcast across the block; Ma_rel here is a profile.
    ember.set_iterative.set_Po_To_Ma_rel_Alpha_rel_Beta(
        block, Po=1.0e5, To=300.0, Ma_rel=np.linspace(0.2, 0.6, 4),
        Alpha_rel=0.0, Beta=0.0,
    )
    # block.Po ~ 1e5, block.To ~ 300, block.Ma_rel ~ [0.2 ... 0.6];
    # block.Ma (absolute) differs from Ma_rel by the wheel speed

"""

import numpy as np

from ember.util import angles_to_components

f32 = np.float32

# Default convergence parameters for iterative setters
_DEFAULT_MAX_ITER = 200
_DEFAULT_TOL = 1e-6


def _iterate_ho_s_rho(
    block,
    ho_nd,
    s_nd,
    Alpha,
    Beta,
    velocity_constraint_func,
    max_iter=200,
    tol=1e-6,
    relative_frame=False,
):
    """Iterator for stagnation enthalpy constraints (nondimensional internals).

    Solves: ho = h_static + 0.5*V^2, s = constant

    Parameters
    ----------
    block : Block
        The block object to modify.
    ho_nd : Array
        Nondimensional stagnation enthalpy [--].
    s_nd : Array
        Nondimensional entropy [--].
    Alpha : Array
        Yaw angle [deg]. Absolute by default, relative if relative_frame=True.
    Beta : Array
        Pitch angle [deg].
    velocity_constraint_func : Callable
        Function that takes (rho_nd, u_nd) and returns nondimensional velocity magnitude.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    relative_frame : bool, default=False
        If True, Alpha is relative yaw angle and velocity constraint is relative.
        Energy equation uses absolute velocity for stagnation enthalpy.

    Returns
    -------
    Block
        The modified block object.
    """
    # Convert inputs to arrays
    ho_nd = np.asarray(ho_nd, dtype=f32)
    s_nd = np.asarray(s_nd, dtype=f32)
    Alpha = np.asarray(Alpha, dtype=f32)
    Beta = np.asarray(Beta, dtype=f32)

    # Initial guess: ho - 0.25*V^2 (conservative energy estimate)
    # Note: Using full kinetic energy (ho - 0.5*V^2) underestimates h_static at high Mach
    # due to compressibility effects, so we use 0.25 factor for better initial guess
    # V represents relative velocity (equals absolute velocity when Omega=0)
    rho_stag, u_stag = block.fluid.set_h_s(ho_nd, s_nd)
    V_est = velocity_constraint_func(rho_stag, u_stag)
    h_static_est = ho_nd - f32(0.25) * V_est**2

    rho, u = block.fluid.set_h_s(h_static_est, s_nd)

    # Iterative solution with under-relaxation
    for _ in range(max_iter):
        rho_old = rho.copy()

        # Get velocity magnitude from constraint (relative velocity)
        V_rel = velocity_constraint_func(rho, u)

        # Decompose relative velocity using flow angles
        # (Alpha is relative angle when relative_frame=True, absolute when relative_frame=False)
        Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha, Beta)

        # Energy balance calculation
        # We always solve: ho = h_static + 0.5*V_absolute^2 (absolute frame stagnation enthalpy)
        # where ho is the absolute stagnation enthalpy and V_absolute is the absolute velocity

        if relative_frame:
            # Rotating frame: Convert relative tangential velocity to absolute
            # Velocity triangle: Vt_absolute = Vt_relative + U (blade speed)
            U_nd = block.r_nd * block.Omega_nd  # Nondimensional blade speed
            Vt_abs = Vt_rel + U_nd

            # Absolute velocity magnitude squared
            # V_absolute^2 = Vx^2 + Vr^2 + Vt_absolute^2
            V_abs_sq = Vx**2 + Vr**2 + Vt_abs**2

            # Solve for static enthalpy from absolute stagnation enthalpy
            # ho = h_static + 0.5*V_absolute^2  =>  h_static = ho - 0.5*V_absolute^2
            h_static = ho_nd - f32(0.5) * V_abs_sq
            Vt = Vt_abs  # Store absolute tangential velocity for final state
        else:
            # Stationary frame (Omega=0): relative velocity equals absolute velocity
            # V_absolute = V_relative since U = 0
            Vt = Vt_rel
            V_sq = Vx**2 + Vr**2 + Vt**2

            # Energy balance: ho = h_static + 0.5*V_absolute^2
            h_static = ho_nd - f32(0.5) * V_sq

        # Update thermodynamic state
        rho_new, u_new = block.fluid.set_h_s(h_static, s_nd)

        # Under-relaxation for stability and convergence
        rho = f32(0.5) * rho_new + f32(0.5) * rho
        u = f32(0.5) * u_new + f32(0.5) * u

        # Check convergence
        residual_rho = np.abs(rho_new - rho_old)
        relative_rho = np.maximum(residual_rho / rho_new, residual_rho / rho_old)
        if np.all(relative_rho < tol):
            # Use final non-relaxed values for output
            rho, u = rho_new, u_new
            break
    else:
        raise RuntimeError(
            f"_iterate_ho_s_rho did not converge after {max_iter} iterations"
        )

    # Set final state: store nondim rho/u directly, redimensionalise velocities
    V_ref = block.fluid.V_ref
    block._set_rho_u_nd(rho, u)
    block.set_Vx(Vx * V_ref)
    block.set_Vr(Vr * V_ref)
    block.set_Vt(Vt * V_ref)

    return block


def _iterate_I_s_rho(
    block,
    I_nd,
    s_nd,
    Alpha_rel,
    Beta,
    velocity_constraint_func,
    max_iter=200,
    tol=1e-6,
):
    """Iterator for rothalpy constraints (rotating frame, nondimensional internals).

    Solves: I = h_static + 0.5*V_abs^2 - U*Vt_abs, s = constant

    Parameters
    ----------
    block : Block
        The block object to modify.
    I_nd : Array
        Nondimensional rothalpy [--].
    s_nd : Array
        Nondimensional entropy [--].
    Alpha_rel : Array
        Relative yaw angle [deg].
    Beta : Array
        Pitch angle [deg].
    velocity_constraint_func : Callable
        Function that takes (rho_nd, u_nd) and returns nondimensional relative velocity magnitude.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    Block
        The modified block object.
    """
    # Convert inputs to arrays
    I_nd = np.asarray(I_nd, dtype=f32)  # noqa: E741
    s_nd = np.asarray(s_nd, dtype=f32)
    Alpha_rel = np.asarray(Alpha_rel, dtype=f32)
    Beta = np.asarray(Beta, dtype=f32)

    # Get nondimensional circumferential velocity (blade speed U* = r* Omega*)
    U_nd = block.r_nd * block.Omega_nd

    # Initial guess from rothalpy conditions
    rho, u = block.fluid.set_h_s(I_nd, s_nd)

    # Iterative solution
    for _ in range(max_iter):
        rho_old = rho.copy()

        # Get relative velocity magnitude from constraint
        V_rel = velocity_constraint_func(rho, u)

        # Decompose relative velocity using relative flow angles
        Vx, Vr, Vt_rel = angles_to_components(V_rel, Alpha_rel, Beta)

        # Convert to absolute tangential velocity
        Vt_abs = Vt_rel + U_nd

        # Calculate static enthalpy using rothalpy relation
        # I = h_static + 0.5*V_abs^2 - U*Vt_abs
        V_abs_sq = Vx**2 + Vr**2 + Vt_abs**2
        h_static = I_nd - f32(0.5) * V_abs_sq + U_nd * Vt_abs

        # Update thermodynamic state
        rho, u = block.fluid.set_h_s(h_static, s_nd)

        # Check convergence
        if np.allclose(rho, rho_old, rtol=tol):
            break
    else:
        raise RuntimeError(
            f"_iterate_I_s_rho did not converge after {max_iter} iterations"
        )

    # Set final state: store nondim rho/u directly, redimensionalise velocities
    V_ref = block.fluid.V_ref
    block._set_rho_u_nd(rho, u)
    block.set_Vx(Vx * V_ref)
    block.set_Vr(Vr * V_ref)
    block.set_Vt(Vt_abs * V_ref)

    return block


def set_ho_s_Ma_Alpha_Beta(
    block,
    ho,
    s,
    Ma,
    Alpha=0.0,
    Beta=0.0,
    max_iter=_DEFAULT_MAX_ITER,
    tol=_DEFAULT_TOL,
):
    """Set the flow field from stagnation enthalpy, entropy, absolute Mach number and flow angles.

    Entropy is held constant while density is iterated until the static state
    and the velocity implied by the Mach number satisfy the stagnation
    enthalpy. Velocities are in the absolute (stationary) frame.

    Parameters
    ----------
    block : Block
        Block to modify in place.
    ho : array-like
        Stagnation enthalpy [J/kg]. Must broadcast to block shape.
    s : array-like
        Specific entropy [J/kg/K]. Must broadcast to block shape.
    Ma : array-like
        Absolute Mach number [-]. Must broadcast to block shape.
    Alpha : array-like, default 0.0
        Absolute yaw angle [deg]. Must broadcast to block shape.
    Beta : array-like, default 0.0
        Pitch angle [deg]. Must broadcast to block shape.
    max_iter : int, default 200
        Maximum number of iterations before a :class:`RuntimeError` is raised.
    tol : float, default 1e-6
        Relative convergence tolerance on density.
    """

    # Nondimensionalise inputs
    fluid = block.fluid
    ho_nd = ho / fluid.u_ref
    s_nd = s / fluid.Rgas_ref

    def velocity_constraint_func(rho, u):
        """Calculate nondimensional absolute velocity magnitude from Mach constraint."""
        a = block.fluid.get_a(rho, u)
        return Ma * a

    return _iterate_ho_s_rho(
        block,
        ho_nd,
        s_nd,
        Alpha,
        Beta,
        velocity_constraint_func,
        max_iter,
        tol,
        relative_frame=False,
    )


def set_ho_s_rhoVm_Alpha_Beta(
    block,
    ho,
    s,
    rhoVm,
    Alpha=0.0,
    Beta=0.0,
    max_iter=_DEFAULT_MAX_ITER,
    tol=_DEFAULT_TOL,
):
    """Set the flow field from stagnation enthalpy, entropy, meridional mass flux and flow angles.

    Entropy is held constant while density is iterated until the static state
    and the velocity implied by the meridional mass flux satisfy the stagnation
    enthalpy. Velocities are in the absolute (stationary) frame.

    Parameters
    ----------
    block : Block
        Block to modify in place.
    ho : array-like
        Stagnation enthalpy [J/kg]. Must broadcast to block shape.
    s : array-like
        Specific entropy [J/kg/K]. Must broadcast to block shape.
    rhoVm : array-like
        Meridional mass flux (momentum density) [kg/m^2/s]. Must broadcast to
        block shape.
    Alpha : array-like, default 0.0
        Absolute yaw angle [deg]. Must broadcast to block shape.
    Beta : array-like, default 0.0
        Pitch angle [deg]. Must broadcast to block shape.
    max_iter : int, default 200
        Maximum number of iterations before a :class:`RuntimeError` is raised.
    tol : float, default 1e-6
        Relative convergence tolerance on density.
    """

    # Nondimensionalise inputs
    fluid = block.fluid
    ho_nd = ho / fluid.u_ref
    s_nd = s / fluid.Rgas_ref
    rhoVm_nd = rhoVm / fluid.rhoV_ref

    def velocity_constraint_func(rho, u):
        """Calculate nondimensional absolute velocity magnitude from momentum density constraint."""
        return rhoVm_nd / rho / np.cos(np.radians(Alpha))

    return _iterate_ho_s_rho(
        block,
        ho_nd,
        s_nd,
        Alpha,
        Beta,
        velocity_constraint_func,
        max_iter,
        tol,
        relative_frame=False,
    )


def set_ho_s_rhoVm_Vt_Beta(
    block,
    ho,
    s,
    rhoVm,
    Vt,
    Beta=0.0,
    max_iter=_DEFAULT_MAX_ITER,
    tol=_DEFAULT_TOL,
):
    r"""Set the flow field from stagnation enthalpy, entropy, meridional mass flux and fixed swirl.

    The stagnation enthalpy, entropy and tangential velocity are held fixed
    while the meridional velocity and static state adjust to carry the
    prescribed meridional mass flux :math:`\rho V_m`. Holding :math:`V_\theta`
    rather than the yaw angle conserves angular momentum at fixed radius, unlike
    :func:`set_ho_s_rhoVm_Alpha_Beta`.

    Only the subsonic meridional branch is solved. :math:`\rho V_m` is maximised
    at meridional Mach number unity; a :class:`RuntimeError` is raised if the
    requested value exceeds that sonic maximum.

    Parameters
    ----------
    block : Block
        Block to modify in place.
    ho : array-like
        Stagnation enthalpy [J/kg]. Must broadcast to block shape.
    s : array-like
        Specific entropy [J/kg/K]. Must broadcast to block shape.
    rhoVm : array-like
        Meridional mass flux (momentum density) [kg/m^2/s]. Must broadcast to
        block shape.
    Vt : array-like
        Tangential (swirl) velocity, held fixed [m/s]. Must broadcast to block
        shape.
    Beta : array-like, default 0.0
        Pitch angle [deg]. Must broadcast to block shape.
    max_iter : int, default 200
        Maximum number of iterations before a :class:`RuntimeError` is raised.
    tol : float, default 1e-6
        Relative convergence tolerance on density.
    """

    # Nondimensionalise inputs
    fluid = block.fluid
    ho_nd = np.asarray(ho / fluid.u_ref, dtype=f32)
    s_nd = np.asarray(s / fluid.Rgas_ref, dtype=f32)
    rhoVm_nd = np.asarray(rhoVm / fluid.rhoV_ref, dtype=f32)
    Vt_nd = np.asarray(Vt / fluid.V_ref, dtype=f32)
    Beta = np.asarray(Beta, dtype=f32)

    # Meridional stagnation enthalpy: kinetic energy of the fixed swirl is
    # always present, so the meridional component sees a reduced stagnation
    # enthalpy ho_m = ho - 0.5*Vt^2. Note ho is referenced to the fluid datum
    # (u = s = 0 at the datum state), so ho_m may be negative without being
    # unphysical; feasibility is governed by the sonic limit below, not by the
    # sign of ho_m.
    ho_m_nd = ho_nd - f32(0.5) * Vt_nd**2

    # The meridional mass flux rho*Vm is maximised at meridional Mach number
    # unity. Locate the sonic state (Vm = a) to check for choking. At the sonic
    # point, h_static = ho_m - 0.5*a^2 with a^2 = a(h_static, s)^2, solved by
    # simple fixed-point iteration.
    rho_sonic, u_sonic = fluid.set_h_s(ho_m_nd, s_nd)
    for _ in range(max_iter):
        a_sonic = fluid.get_a(rho_sonic, u_sonic)
        h_sonic = ho_m_nd - f32(0.5) * a_sonic**2
        rho_new, u_new = fluid.set_h_s(h_sonic, s_nd)
        if np.all(np.abs(rho_new - rho_sonic) <= tol * rho_new):
            rho_sonic, u_sonic = rho_new, u_new
            break
        rho_sonic = f32(0.5) * rho_new + f32(0.5) * rho_sonic
        u_sonic = f32(0.5) * u_new + f32(0.5) * u_sonic
    Vm_sonic = fluid.get_a(rho_sonic, u_sonic)
    rhoVm_max = rho_sonic * Vm_sonic

    if np.any(rhoVm_nd > rhoVm_max):
        raise RuntimeError(
            "Meridional flow chokes: requested mass flux exceeds the sonic "
            f"maximum (rhoVm={rhoVm_nd * fluid.rhoV_ref}, "
            f"rhoVm_max={rhoVm_max * fluid.rhoV_ref} [kg/m^2/s])."
        )

    # Seed the subsonic branch from the stagnation state (low velocity, high
    # density). The density fixed-point iteration below is stable on, and
    # converges to, the subsonic root.
    rho, u = fluid.set_h_s(ho_m_nd, s_nd)

    # Iterate density to satisfy mass flux and energy with fixed swirl.
    for _ in range(max_iter):
        rho_old = rho.copy()

        # Meridional velocity from the mass-flux constraint
        Vm = rhoVm_nd / rho

        # Energy balance for the meridional component
        h_static = ho_m_nd - f32(0.5) * Vm**2
        rho_new, u_new = fluid.set_h_s(h_static, s_nd)

        # Under-relaxation for stability
        rho = f32(0.5) * rho_new + f32(0.5) * rho
        u = f32(0.5) * u_new + f32(0.5) * u

        residual_rho = np.abs(rho_new - rho_old)
        relative_rho = np.maximum(residual_rho / rho_new, residual_rho / rho_old)
        if np.all(relative_rho < tol):
            rho, u = rho_new, u_new
            break
    else:
        raise RuntimeError(
            f"set_ho_s_rhoVm_Vt_Beta did not converge after {max_iter} iterations"
        )

    # Decompose the converged meridional velocity by the pitch angle and apply
    # the fixed swirl. Vt is set in dimensional form below.
    Vm = rhoVm_nd / rho
    beta_rad = np.radians(Beta)
    Vx = Vm * np.cos(beta_rad)
    Vr = Vm * np.sin(beta_rad)

    V_ref = fluid.V_ref
    block._set_rho_u_nd(rho, u)
    block.set_Vx(Vx * V_ref)
    block.set_Vr(Vr * V_ref)
    block.set_Vt(Vt_nd * V_ref)

    return block


def set_I_s_Ma_rel_Alpha_rel_Beta(
    block,
    I,
    s,
    Ma_rel,
    Alpha_rel,
    Beta,
    max_iter=_DEFAULT_MAX_ITER,
    tol=_DEFAULT_TOL,
):
    r"""Set the flow field from rothalpy, entropy, relative Mach number and relative flow angles.

    Entropy is held constant while density is iterated until the static state
    and the relative velocity satisfy the rothalpy
    :math:`I = h + \tfrac12 V^2 - U V_\theta`, where :math:`V_\theta` is the
    absolute tangential velocity and :math:`U = r\,\Omega` the blade speed. Set
    the block radius (and rotation rate, if rotating) first.

    Parameters
    ----------
    block : Block
        Block to modify in place.
    I : array-like
        Rothalpy [J/kg]. Must broadcast to block shape.
    s : array-like
        Specific entropy [J/kg/K]. Must broadcast to block shape.
    Ma_rel : array-like
        Relative-frame Mach number [-]. Must broadcast to block shape.
    Alpha_rel : array-like
        Relative yaw angle [deg]. Must broadcast to block shape.
    Beta : array-like
        Pitch angle [deg]. Must broadcast to block shape.
    max_iter : int, default 200
        Maximum number of iterations before a :class:`RuntimeError` is raised.
    tol : float, default 1e-6
        Relative convergence tolerance on density.
    """

    # Nondimensionalise inputs
    fluid = block.fluid
    I_nd = I / fluid.u_ref
    s_nd = s / fluid.Rgas_ref

    def velocity_constraint_func(rho, u):
        """Calculate nondimensional relative velocity magnitude from relative Mach constraint."""
        a = block.fluid.get_a(rho, u)
        return Ma_rel * a

    return _iterate_I_s_rho(
        block, I_nd, s_nd, Alpha_rel, Beta, velocity_constraint_func, max_iter, tol
    )


def set_Po_To_Ma_rel_Alpha_rel_Beta(
    block,
    Po,
    To,
    Ma_rel,
    Alpha_rel,
    Beta,
    max_iter=_DEFAULT_MAX_ITER,
    tol=_DEFAULT_TOL,
):
    r"""Set the flow field from absolute total conditions, relative Mach number and relative flow angles.

    The total pressure and temperature fix the absolute stagnation enthalpy and
    entropy. Entropy is held constant while density is iterated until the static
    state and the relative velocity satisfy
    :math:`h_0 = h + \tfrac12 V^2`, with :math:`V` the absolute velocity and
    blade speed :math:`U = r\,\Omega`. Set the block radius (and rotation rate,
    if rotating) first.

    Parameters
    ----------
    block : Block
        Block to modify in place.
    Po : array-like
        Absolute total pressure [Pa]. Must be positive and broadcast to block
        shape.
    To : array-like
        Absolute total temperature [K]. Must be positive and broadcast to block
        shape.
    Ma_rel : array-like
        Relative-frame Mach number [-]. Must broadcast to block shape.
    Alpha_rel : array-like
        Relative yaw angle [deg]. Must broadcast to block shape.
    Beta : array-like
        Pitch angle [deg]. Must broadcast to block shape.
    max_iter : int, default 200
        Maximum number of iterations before a :class:`RuntimeError` is raised.
    tol : float, default 1e-6
        Relative convergence tolerance on density.

    Raises
    ------
    ValueError
        If the block radius has not been set.
    """
    # Check if coordinates are initialized (required for relative frame)
    if not block._versions.get("r", False):
        raise ValueError(
            "Radial coordinates (r) must be initialized before using relative frame quantities. "
            "Use block.set_x() / block.set_r() to set coordinates first."
        )

    # Nondimensionalise and compute stagnation properties
    fluid = block.fluid
    Po_nd = Po / fluid.P_ref
    To_nd = To / fluid.T_ref
    rho_stag, u_stag = fluid.set_P_T(Po_nd, To_nd)
    ho_nd = fluid.get_h(rho_stag, u_stag)  # Nondimensional stagnation enthalpy
    s_nd = fluid.get_s(rho_stag, u_stag)  # Nondimensional stagnation entropy

    def velocity_constraint_func(rho, u):
        """Calculate nondimensional relative velocity magnitude from relative Mach constraint."""
        a = block.fluid.get_a(rho, u)
        return Ma_rel * a

    # Use stagnation enthalpy iterator in relative frame mode
    return _iterate_ho_s_rho(
        block,
        ho_nd,
        s_nd,
        Alpha_rel,
        Beta,
        velocity_constraint_func,
        max_iter,
        tol,
        relative_frame=True,
    )
