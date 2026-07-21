"""Inlet boundary condition patch for EMBER CFD.

InletPatch enforces stagnation pressure, stagnation temperature, and flow angles
at an inflow face. Static pressure is not directly imposed; the face state is
instead found by solving the outgoing acoustic characteristic together with the
imposed stagnation condition, which fixes the velocity magnitude.

See Also
--------
ember.patch.Patch : Base class for all patches
ember.patch.OutletPatch : Outlet boundary condition
"""

import numpy as np
from ember.basepatch import RevolutionPatch


class InletPatch(RevolutionPatch):
    """Inflow boundary condition.

    Enforces prescribed stagnation pressure :math:`p_0`, stagnation temperature
    :math:`T_0`, yaw angle :math:`\\alpha`, and pitch angle :math:`\\beta` at the
    face. Stagnation enthalpy and entropy are derived from :math:`p_0` and
    :math:`T_0` and imposed directly. Flow direction is set from :math:`\\alpha`
    and :math:`\\beta`.

    Static pressure is not imposed directly. Each call to :meth:`apply` solves
    the outgoing acoustic characteristic carried to the face from the interior
    simultaneously with the imposed stagnation state, which determines the
    velocity magnitude. The result is relaxed toward :math:`V_\\mathrm{soln}`;
    call :meth:`update_soln` once per timestep (before the Runge-Kutta stages)
    to advance it.

    All four boundary condition values must be set via
    :meth:`set_Po_To_Alpha_Beta` before :meth:`apply` is called.
    """

    _collection_name = "inlet"

    # Newton controls for the characteristic solve in apply(). The iteration is
    # warm-started from the previous face velocity and typically exits after one
    # pass, so the cap is a safety net rather than a working limit.
    _MAX_ITER = 10
    _TOL = 1e-6

    def _copy(self, c):
        c._raw = {k: np.copy(v) for k, v in self._raw.items()}
        # _target_nd and _V_nd_max are derived from _raw and block_view.shape,
        # so they must be recomputed on the new block rather than copied.
        c._V_nd_soln = np.copy(self._V_nd_soln) if self._V_nd_soln is not None else None
        c.rf = self.rf

    def _setup(self):
        super()._setup()
        self._raw = {"Po": np.nan, "To": np.nan, "Alpha": np.nan, "Beta": np.nan}
        self._target_nd = (
            None  # (ho_nd, s_nd, cosBetacosAlpha, sinBetacosAlpha, sinAlpha)
        )
        self._V_nd_max = None
        self._V_nd_soln = None
        # Relaxation factor for the velocity update, read by apply(). Unity is
        # the correct answer once the characteristic solve makes the target
        # well-conditioned; rf < 1 only adds lag for startup transients.
        self.rf = 1.0

    def _calc_target(self):
        """Compute nondimensional target tuple from Po, To, Alpha, Beta."""
        fluid = self.block.fluid
        rhoo_nd, uo_nd = fluid.set_P_T(self.Po / fluid.P_ref, self.To / fluid.T_ref)
        ho_nd = fluid.get_h(rhoo_nd, uo_nd)
        s_nd = fluid.get_s(rhoo_nd, uo_nd)

        def _broadcast(arr):
            return np.asfortranarray(
                np.broadcast_to(arr, self.block_view.shape).astype(np.float32)
            )

        Alpha_rad = np.radians(self.Alpha)
        Beta_rad = np.radians(self.Beta)
        cosAlpha = np.cos(Alpha_rad)
        cosBeta = np.cos(Beta_rad)

        # Ceiling on the reconstructed velocity: the isentropic expansion of the
        # stagnation state down to a token fraction of To. The Newton iterate in
        # apply() is clamped to this so set_h_s never sees a static enthalpy
        # below the zero-temperature limit. It sits near Ma ~ 14, far outside any
        # physical solution, and exists only to keep transients finite.
        h_floor_nd = fluid.get_h(*fluid.set_T_s(0.01 * self.To / fluid.T_ref, s_nd))
        self._V_nd_max = _broadcast(np.sqrt(2.0 * (ho_nd - h_floor_nd)))
        self._target_nd = (
            _broadcast(ho_nd),
            _broadcast(s_nd),
            _broadcast(cosBeta * cosAlpha),
            _broadcast(np.sin(Beta_rad) * cosAlpha),
            _broadcast(np.sin(Alpha_rad)),
        )

    def _solve_V_nd(self, ho_nd, s_nd, Z_nd, R_nd, a_nd):
        """Newton solve for the face velocity along the prescribed direction.

        Warm-started from :attr:`_V_nd_soln`, so in steady marching this exits
        after a single pass. Convergence is measured against the interior
        acoustic speed ``a_nd``, a strictly positive velocity scale, so the test
        is dimensionless without risking a division by zero.

        Each iterate is clamped into the range where the root is well defined:
        below at the subsonic side of the reverse-running acoustic wave, since
        :math:`g' = -\\rho(V + a)` changes sign at :math:`V = -a` and would
        otherwise break monotonicity; above at :attr:`_V_nd_max`, keeping the
        static enthalpy handed to ``set_h_s`` above the zero-temperature limit.
        Neither bound should bind in a converged solution.
        """
        fluid = self.block.fluid
        V_nd = self._V_nd_soln
        for _ in range(self._MAX_ITER):
            rho_nd, u_nd = fluid.set_h_s(ho_nd - 0.5 * V_nd**2, s_nd)
            g_nd = fluid.get_P(rho_nd, u_nd) - Z_nd * V_nd - R_nd
            dV_nd = g_nd / (rho_nd * (V_nd + fluid.get_a(rho_nd, u_nd)))
            V_nd = np.clip(V_nd + dV_nd, -0.9 * a_nd, self._V_nd_max)
            if np.all(np.abs(dV_nd) <= self._TOL * a_nd):
                break
        return V_nd

    def _face_V_nd(self):
        """Inlet-face velocity projected onto the prescribed flow direction."""
        b = self.block_view
        _, _, cosBcosA, sinBcosA, sinA = self._target_nd
        return b.Vx_nd * cosBcosA + b.Vr_nd * sinBcosA + b.Vt_nd * sinA

    def set_Po_To_Alpha_Beta(self, Po=None, To=None, Alpha=None, Beta=None):
        """Set inlet boundary condition values.

        Each argument is independent; omitted arguments retain their current
        value. All four must be set before :meth:`apply` can be called.
        Stagnation pressure and stagnation temperature must be positive and
        finite. Yaw and pitch angles are in degrees. Each value accepts a
        scalar or an array that broadcasts to :attr:`~ember.basepatch.Patch.shape`.

        Parameters
        ----------
        Po : float or array, optional
            Prescribed stagnation pressure [Pa].
        To : float or array, optional
            Prescribed stagnation temperature [K].
        Alpha : float or array, optional
            Prescribed inflow yaw angle [deg].
        Beta : float or array, optional
            Prescribed inflow pitch angle [deg].
        """
        kwargs = {"Po": Po, "To": To, "Alpha": Alpha, "Beta": Beta}
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        if kwargs:
            broadcasted = np.broadcast_arrays(*kwargs.values(), np.ones(self.shape))
            if broadcasted[0].shape != self.shape:
                raise ValueError(
                    f"Inputs broadcast to {broadcasted[0].shape}, exceeding patch shape {self.shape}"
                )

        for key, val in kwargs.items():
            arr = np.asarray(val)
            if not np.isfinite(arr).all():
                raise ValueError(f"{key} must be finite")
            if key in ("Po", "To") and not (arr > 0).all():
                raise ValueError(f"{key} must be positive")
            self._raw[key] = arr.astype(np.float32)

        self._target_nd = None
        self._V_nd_max = None
        self._V_nd_soln = None

    def apply(self):
        """Impose inlet boundary conditions on the patch.

        Stagnation enthalpy, entropy, and trigonometric flow-direction factors
        derived from :attr:`Po`, :attr:`To`, :attr:`Alpha`, and :attr:`Beta` are
        cached on the first call and combined with a relaxed static pressure to
        reconstruct the velocity vector, which is stored via
        :py:meth:`~ember.block.Block.set_rho_u_Vxrt_nd`.

        Static pressure is not prescribed. The face state instead satisfies two
        conditions simultaneously: the outgoing acoustic characteristic carried
        to the face from the interior, and the imposed stagnation state. Writing
        :math:`V` for the velocity component along the prescribed flow
        direction, the :math:`u - c` characteristic carries the invariant

        .. math::
            R = p_\\mathrm{interior} - \\rho a\\, V_\\mathrm{interior}

        with :math:`p_\\mathrm{interior}` and :math:`V_\\mathrm{interior}`
        linearly extrapolated to the face from the first two interior layers
        (:math:`X = 2 X_1 - X_2`) and the impedance :math:`\\rho a` evaluated at
        the first interior layer. The face velocity is then the root of

        .. math::
            g(V) = p_\\mathrm{isen}(V) - \\rho a\\, V - R = 0

        where :math:`p_\\mathrm{isen}` is the static pressure reached by
        expanding the stagnation state isentropically to velocity :math:`V`.
        Along an isentrope :math:`\\mathrm{d}h = \\mathrm{d}p/\\rho` and
        :math:`\\mathrm{d}h/\\mathrm{d}V = -V`, so
        :math:`g'(V) = -\\rho (V + a)` exactly, giving the Newton step

        .. math::
            V \\leftarrow V + \\frac{g(V)}{\\rho (V + a)}

        which is monotone, and hence globally convergent, across the whole
        subsonic range :math:`-a < V < a`.

        Taking the velocity as the primary variable is what makes this
        well-conditioned at low Mach number. Inverting an imposed pressure
        through the steady isentropic relation instead implies an impedance of
        :math:`\\rho u` where the wave carries :math:`\\rho a`, wrong by a factor
        of the Mach number, so the pressure-to-velocity gain grows as
        :math:`1/(\\gamma M^2)`. Here the sensitivity is
        :math:`1/(\\rho(V + a))` — bounded, and independent of Mach number.

        Note this remains a *reflecting* boundary: the incoming characteristic is
        untouched and :attr:`Po`, :attr:`To`, :attr:`Alpha` and :attr:`Beta` stay
        hard-imposed. A stagnation reservoir does reflect acoustics; the point is
        that it now does so with the correct impedance.

        The converged velocity is relaxed toward :math:`V_\\mathrm{soln}` using
        :attr:`rf` as a convex weight, and the reference advanced by
        :meth:`update_soln` once per timestep. Because the target is
        well-conditioned, ``rf = 1`` is the correct setting rather than an
        aggressive one.

        Backflow (:math:`V < 0`) is representable and needs no special case: the
        Newton solve stays monotone there, so a reversed face resolves naturally
        rather than being forced back to inflow.
        """
        b = self.block_view
        if self._target_nd is None:
            self._calc_target()
        ho_nd, s_nd, cosBcosA, sinBcosA, sinA = self._target_nd

        # Linearly extrapolate the outgoing characteristic state to the face
        # from the first two interior layers (X_face = 2*X_1 - X_2), matching
        # OutletPatch and MixingPatch. The interior velocity is projected onto
        # the prescribed flow direction, which is well defined without any face
        # normal because that direction is imposed.
        b1 = self.block_view_offset_1
        b2 = self.block_view_offset_2
        P_interior_nd = 2.0 * b1.P_nd - b2.P_nd
        V_interior_nd = (
            (2.0 * b1.Vx_nd - b2.Vx_nd) * cosBcosA
            + (2.0 * b1.Vr_nd - b2.Vr_nd) * sinBcosA
            + (2.0 * b1.Vt_nd - b2.Vt_nd) * sinA
        )

        # Acoustic impedance, a coefficient rather than a state, so layer 1
        # without extrapolation is accurate enough.
        a1_nd = b1.a_nd
        Z_nd = b1.rho_nd * a1_nd
        R_nd = P_interior_nd - Z_nd * V_interior_nd

        if self._V_nd_soln is None:
            self._V_nd_soln = self._face_V_nd()

        V_nd = self._solve_V_nd(ho_nd, s_nd, Z_nd, R_nd, a1_nd)
        V_new_nd = self._V_nd_soln + self.rf * (V_nd - self._V_nd_soln)
        rho_nd, u_nd = b.fluid.set_h_s(ho_nd - 0.5 * V_new_nd**2, s_nd)
        b.set_rho_u_Vxrt_nd(
            rho_nd,
            u_nd,
            V_new_nd * cosBcosA,
            V_new_nd * sinBcosA,
            V_new_nd * sinA,
        )

    def update_soln(self):
        """Update :math:`V_\\mathrm{soln}` from the current inlet-face velocity.

        Should be called once per timestep before the Runge-Kutta stages so
        that each stage's relaxation in :meth:`apply` is anchored to the
        start-of-step velocity rather than drifting across stages.
        """
        if self._target_nd is None:
            self._calc_target()
        self._V_nd_soln = self._face_V_nd()

    @property
    def Alpha(self):
        r"""Prescribed inflow yaw angle :math:`\alpha` [deg]; broadcasts to :attr:`~ember.basepatch.Patch.shape`. See :py:attr:`~ember.block.Block.Alpha`."""
        return self._raw["Alpha"]

    @property
    def Beta(self):
        r"""Prescribed inflow pitch angle :math:`\beta` [deg]; broadcasts to :attr:`~ember.basepatch.Patch.shape`. See :py:attr:`~ember.block.Block.Beta`."""
        return self._raw["Beta"]

    @property
    def Po(self):
        r"""Prescribed inflow stagnation pressure :math:`p_0` [Pa]; broadcasts to :attr:`~ember.basepatch.Patch.shape`. See :py:attr:`~ember.block.Block.Po`."""
        return self._raw["Po"]

    @property
    def To(self):
        r"""Prescribed inflow stagnation temperature :math:`T_0` [K]; broadcasts to :attr:`~ember.basepatch.Patch.shape`. See :py:attr:`~ember.block.Block.To`."""
        return self._raw["To"]
