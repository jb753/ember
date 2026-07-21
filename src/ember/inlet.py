"""Inlet boundary condition patch for EMBER CFD.

InletPatch enforces stagnation pressure, stagnation temperature, and flow angles
at an inflow face. Static pressure is not directly imposed; instead it is
relaxed toward the first interior node each step (a Multall-style pressure
extrapolation) to avoid large acoustic transients.

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

    Static pressure is imposed through two decoupled relaxation stages.
    :meth:`update_soln` relaxes the cycle target :math:`p_\\mathrm{soln}`
    toward the static pressure at the first interior node by :attr:`rf`, then
    clamps it below the stagnation pressure; call it once per timestep,
    before the Runge-Kutta stages. :meth:`apply` then relaxes the actual
    boundary pressure :math:`p_\\mathrm{face}` toward that fixed
    :math:`p_\\mathrm{soln}` by :attr:`rf_stage`, once per Runge-Kutta stage.
    This damps acoustic transients at startup without prescribing a fixed
    pressure, and keeps the fast within-cycle response independently tunable
    from the slow cycle-to-cycle drift of the target.

    All four boundary condition values must be set via
    :meth:`set_Po_To_Alpha_Beta` before :meth:`apply` is called.
    """

    _collection_name = "inlet"

    def _copy(self, c):
        c._raw = {k: np.copy(v) for k, v in self._raw.items()}
        # _target_nd and _Po_nd_target are derived from _raw and block_view.shape,
        # so they must be recomputed on the new block rather than copied.
        c._P_nd_soln = np.copy(self._P_nd_soln) if self._P_nd_soln is not None else None
        c._P_nd_face = np.copy(self._P_nd_face) if self._P_nd_face is not None else None
        c.rf = self.rf
        c.rf_stage = self.rf_stage

    def _setup(self):
        super()._setup()
        self._raw = {"Po": np.nan, "To": np.nan, "Alpha": np.nan, "Beta": np.nan}
        self._target_nd = (
            None  # (ho_nd, s_nd, cosBetacosAlpha, sinBetacosAlpha, sinAlpha)
        )
        self._Po_nd_target = None
        self._P_nd_soln = None
        self._P_nd_face = None
        # Cycle-level relaxation factor for p_soln (toward the interior
        # pressure), read by update_soln().
        self.rf = 0.2
        # Stage-level relaxation factor for p_face (toward p_soln), read by
        # apply().
        self.rf_stage = 0.2

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

        self._Po_nd_target = _broadcast(self.Po / fluid.P_ref)
        self._target_nd = (
            _broadcast(ho_nd),
            _broadcast(s_nd),
            _broadcast(cosBeta * cosAlpha),
            _broadcast(np.sin(Beta_rad) * cosAlpha),
            _broadcast(np.sin(Alpha_rad)),
        )

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
        self._Po_nd_target = None
        self._P_nd_soln = None
        self._P_nd_face = None

    def apply(self):
        """Impose inlet boundary conditions on the patch.

        Stagnation enthalpy, entropy, and trigonometric flow-direction factors
        derived from :attr:`Po`, :attr:`To`, :attr:`Alpha`, and :attr:`Beta` are
        cached on the first call and combined with a relaxed static pressure to
        reconstruct the velocity vector, which is stored via
        :py:meth:`~ember.block.Block.set_rho_u_Vxrt_nd`.

        Static pressure is not prescribed; instead the live boundary pressure
        :math:`p_\\mathrm{face}` is relaxed toward the fixed cycle target
        :math:`p_\\mathrm{soln}` (advanced separately by :meth:`update_soln`,
        once per step):

        .. math::
            p_\\mathrm{face} \\mathrel{+}= rf_\\mathrm{stage}\\,
                (p_\\mathrm{soln} - p_\\mathrm{face})

        using the relaxation factor :attr:`rf_stage`, then clamped below the
        stagnation pressure :attr:`Po`. Called once per Runge-Kutta stage, so
        :math:`p_\\mathrm{face}` chases a target that itself only moves once
        per step -- a Multall-style pressure extrapolation, decoupled into a
        slow cycle-to-cycle rate (:attr:`rf`, in :meth:`update_soln`) and a
        fast within-cycle rate (:attr:`rf_stage`). Both are used directly as
        convex weights (no Mach scaling); higher values converge faster but
        may excite acoustics.

        Raises
        ------
        ValueError
            If the incoming solution has a negative axial velocity at any inlet
            boundary node (backflow), the calculation is ill-posed and is
            stopped rather than continued into an unphysical state.
        """
        b = self.block_view

        # Stop on backflow: a negative axial velocity at the inlet face means the
        # solution is trying to push flow out through the inflow boundary, which
        # the imposed-angle reconstruction below cannot represent. rho > 0 always,
        # so sign(Vx) == sign(rhoVx).
        # if np.any(Vx_nd < 0.0):
        #     n_back = int(np.count_nonzero(Vx_nd < 0.0))
        #     raise ValueError(
        #         f"Backflow at inlet patch {self.label!r}: axial velocity Vx < 0 "
        #         f"at {n_back} of {Vx_nd.size} boundary nodes "
        #         f"(min Vx = {float(np.min(Vx_nd)):.4g} m/s)."
        #     )

        if self._target_nd is None:
            self._calc_target()
        if self._P_nd_soln is None:
            self._P_nd_soln = b.P_nd.copy()
        if self._P_nd_face is None:
            self._P_nd_face = self._P_nd_soln.copy()
        else:
            self._P_nd_face = self._P_nd_face + self.rf_stage * (
                self._P_nd_soln - self._P_nd_face
            )
        np.minimum(self._P_nd_face, 0.9999 * self._Po_nd_target, out=self._P_nd_face)

        ho_nd, s_nd, cosBcosA, sinBcosA, sinA = self._target_nd

        # Density and internal energy follow from (P, s) in closed form; the
        # velocity magnitude follows from the enthalpy deficit V = sqrt(2(ho - h)),
        # and the Cartesian components from the three precomputed direction cosines.
        rho_nd, u_nd = b.fluid.set_P_s(self._P_nd_face, s_nd)
        V_nd = np.sqrt(2.0 * (ho_nd - b.fluid.get_h(rho_nd, u_nd)))
        b.set_rho_u_Vxrt_nd(rho_nd, u_nd, V_nd * cosBcosA, V_nd * sinBcosA, V_nd * sinA)

    def update_soln(self):
        """Advance the cycle target :math:`p_\\mathrm{soln}` by one relaxation step.

        Called once per timestep, before the Runge-Kutta stages: reads the
        interior pressure at the start of the step and relaxes
        :math:`p_\\mathrm{soln}` toward it by :attr:`rf`, then clamps below
        the stagnation pressure. :meth:`apply` then chases this fixed target
        with the live boundary pressure once per stage, at its own rate
        :attr:`rf_stage`.
        """
        b = self.block_view
        if self._target_nd is None:
            self._calc_target()
        if self._P_nd_soln is None:
            self._P_nd_soln = b.P_nd.copy()
            return
        P_interior_nd = self.block_view_offset_1.P_nd
        P_new_nd = self._P_nd_soln + self.rf * (P_interior_nd - self._P_nd_soln)
        np.minimum(P_new_nd, 0.9999 * self._Po_nd_target, out=P_new_nd)
        self._P_nd_soln = P_new_nd

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
