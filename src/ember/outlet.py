"""Outlet boundary condition patch for EMBER CFD.

OutletPatch enforces static pressure at the outflow boundary.
Optionally supports throttle control with PID feedback to maintain target mass flow rate.

See Also
--------
ember.patch.Patch : Base class for all patches
ember.patch.InletPatch : Inlet boundary condition
"""

import numpy as np
from ember.basepatch import RevolutionPatch


class OutletPatch(RevolutionPatch):
    """Outflow boundary condition.

    Enforces a prescribed static pressure :math:`p` at the face. Entropy and
    all three velocity components are linearly (two-point) extrapolated from
    the first two interior layers, ``X_face = 2 * X_1 - X_2``, so the outgoing
    entropy and vorticity characteristics are carried to the boundary with
    their interior gradient.

    Three optional features extend the constant uniform pressure condition:

    - **Throttle** :meth:`set_throttle`: a PID controller adjusts the
      pressure each step to drive the patch mass flow toward a target value.
    - **Spanwise adjustment** :meth:`set_adjustment`: adds zero-mean radial
      equilibrium and dynamic head profiles to the uniform pressure target.
    - **Backflow handling** :meth:`set_backflow`: when reversed flow is
      detected, affected nodes are treated as inlets; see
      :class:`~ember.inlet.InletPatch` for the density relaxation scheme.

    Static pressure must be set via :meth:`set_P` before :meth:`apply` is
    called.
    """

    _collection_name = "outlet"
    _cfl_pid = np.float32(1.0)  # dt = _cfl_pid * L_ref / V_ref

    def _copy(self, c):
        c._P_raw = np.copy(self._P_raw) if self._P_raw is not None else None
        c._P_target_nd = (
            np.copy(self._P_target_nd) if self._P_target_nd is not None else None
        )
        c._K_pid = self._K_pid
        c._mdot_target = self._mdot_target
        c._adjustment = self._adjustment.copy()
        c._err_int = self._err_int
        c._err_prev = self._err_prev
        c._dP_P = self._dP_P
        c._dP_I = self._dP_I
        c._dP_D = self._dP_D
        c._inout_snapshot = (
            np.copy(self._inout_snapshot) if self._inout_snapshot is not None else None
        )
        c._backflow_enabled = self._backflow_enabled

    def _setup(self):
        super()._setup()
        self._P_raw = None  # user-supplied value, dimensional [Pa]
        self._P_target_nd = None  # full-shape nondim target
        self._K_pid = (0.0, 0.0, 0.0)
        self._mdot_target = np.float32(0.0)
        self._adjustment = {}
        self._err_int = np.float32(0.0)
        self._err_prev = np.float32(0.0)
        self._dP_P = np.float32(0.0)
        self._dP_I = np.float32(0.0)
        self._dP_D = np.float32(0.0)
        self._mdot = np.float32(0.0)
        self._P_last_nd = None
        self._inout_snapshot = None
        self._backflow_enabled = False
        self._rho_nd_soln = None

    def _span_bcast(self, arr):
        """Reshape a 1-D span array to broadcast over the full 3-D block shape."""
        shape = [1, 1, 1]
        shape[self.span_dim] = -1
        return arr.reshape(shape)

    def set_adjustment(self, K_dyn=0.0, radial_equilibrium=True, rf=0.1):
        r"""Configure nonuniform adjustments to outlet pressure.

        The pressure at each node is:

        .. math::

            p = p_\mathrm{out} + \Delta p_\mathrm{throttle} + \Delta p_\mathrm{nonunif}

        :math:`\Delta p_\mathrm{nonunif} = \Delta p_\mathrm{dyn} + \Delta p_\mathrm{swirl}` is
        computed from the boundary solution each step and the total nonuniformity is
        relaxed toward the new value:

        .. math::

            \Delta p_\mathrm{nonunif}^\mathrm{new} =
                \mathit{rf} \, \Delta p_\mathrm{nonunif}
                + (1 - \mathit{rf}) \, \Delta p_\mathrm{nonunif}^\mathrm{old}

        :math:`\overline{\Box}` denotes the area-weighted mean. When
        ``radial_equilibrium=True`` it is the pitch mean at each spanwise location,
        preserving spanwise variation; when ``radial_equilibrium=False`` it is the
        annulus mean.

        **Dynamic head offset** :math:`\Delta p_\mathrm{dyn}`:

        Mimics a physical gauze or screen to suppress backflow by driving the exit
        velocity toward a more uniform profile. :math:`\tfrac{1}{2}\rho V_m^2` is the
        meridional dynamic head. The offset has zero net area average by construction:

        .. math::

            \Delta p_\mathrm{dyn} = K_\mathrm{dyn}
                \left(\tfrac{1}{2}\rho V_m^2 - \overline{\tfrac{1}{2}\rho V_m^2}\right)

        **Swirl offset** :math:`\Delta p_\mathrm{swirl}` (``radial_equilibrium=True``):

        Swirling flow exiting the domain carries a centrifugal pressure gradient.
        Enforcing a uniform pressure in opposition to this gradient would induce
        unnatural streamline curvature; instead the boundary pressure is adjusted
        to satisfy radial equilibrium. Pitch-averaged flow quantities are taken
        from the first interior layer (the offset-1 slice), the same source the
        outlet uses to extrapolate the boundary state:

        .. math::

            \frac{dp_\mathrm{swirl}}{dr} =
                \frac{\overline{\rho V_\theta}\;\overline{V_\theta}}{r}

        where :math:`\overline{\Box}` is the pitch mean. This product-of-means
        form matches the Multall ``EXBCONDS`` radial-equilibrium treatment.
        Integrated from hub to tip, then broadcast uniformly in the pitch
        direction. The integration starts from zero at the hub, so the profile is
        zero at the hub, i.e. :math:`p_\mathrm{out}` is enforced at the hub.

        Parameters
        ----------
        K_dyn : float, optional
            Scale factor for the dynamic head offset. Default 0 (off).
        radial_equilibrium : bool, optional
            Include the radial equilibrium offset and use pitch-area-averaging
            for the dynamic head offset. Default True.
        rf : float, optional
            Relaxation factor applied to the combined offset each step.
            Default 0.1.
        """
        if self._P_raw is not None and self._P_raw.ndim > 0 and self._P_raw.size > 1:
            raise ValueError("Adjustment is incompatible with non-scalar P")
        self._adjustment = {
            "K_dyn": float(K_dyn),
            "radial_equilibrium": bool(radial_equilibrium),
            "rf": float(rf),
        }
        return self

    def set_backflow(self, ho, s, Vr, Vt):
        """Enable reversed-flow handling at the outlet.

        Once enabled, :meth:`apply` detects cells with reversed axial flow and
        treats them as inlets: stagnation enthalpy, entropy, and transverse
        velocities are imposed from the values given here, and density is relaxed
        from the interior — exactly as in :class:`~ember.inlet.InletPatch`. If
        reversed flow is detected without this method having been called,
        :meth:`apply` raises a :exc:`RuntimeError`.

        All four arguments must be scalars.

        Parameters
        ----------
        ho : float
            Stagnation enthalpy [J/kg].
        s : float
            Specific entropy [J/(kg K)].
        Vr : float
            Radial velocity [m/s].
        Vt : float
            Tangential velocity [m/s].
        """
        for name, val in (("ho", ho), ("s", s), ("Vr", Vr), ("Vt", Vt)):
            if not np.isscalar(val):
                raise TypeError(
                    f"set_backflow: {name} must be a scalar, got {type(val).__name__}"
                )

        fluid = self.block.fluid
        self._inout_snapshot = np.array(
            [
                ho / fluid.u_ref,
                s / fluid.Rgas_ref,
                Vr / fluid.V_ref,
                Vt / fluid.V_ref,
            ],
            dtype=np.float32,
        )
        self._backflow_enabled = True
        return self

    def set_P(self, P):
        r"""Set the prescribed outlet static pressure :math:`p_\mathrm{out}`.

        Accepts a scalar or an array that broadcasts to
        :attr:`~ember.basepatch.Patch.shape`. Must be positive and finite.
        Non-scalar values are incompatible with :meth:`set_throttle` and
        :meth:`set_adjustment`.

        Parameters
        ----------
        P : float or array
            Prescribed static pressure :math:`p_\mathrm{out}` [Pa].
        """
        P_arr = np.asarray(P, dtype=np.float32)
        if not (np.all(np.isfinite(P_arr)) and np.all(P_arr > 0)):
            raise ValueError("P must be positive and finite")
        if self._mdot_target > 0 and P_arr.ndim > 0 and P_arr.size > 1:
            raise ValueError("Non-scalar P is incompatible with throttle control")
        if P_arr.ndim > 0 and P_arr.size > 1:
            broadcasted = np.broadcast_arrays(P_arr, np.ones(self.shape))
            if broadcasted[0].shape != self.shape:
                raise ValueError(
                    f"P shape {P_arr.shape} is not broadcast-compatible with patch shape {self.shape}"
                )
            self._adjustment = {}
            self._mdot_target = np.float32(0.0)
            self._err_int = np.float32(0.0)
            self._err_prev = np.float32(0.0)
            self._dP_P = np.float32(0.0)
            self._dP_I = np.float32(0.0)
            self._dP_D = np.float32(0.0)
        self._P_raw = P_arr
        self._P_target_nd = None
        return self

    def set_throttle(self, mdot_target, K_pid):
        r"""Set the mass flow target and PID gains for throttle control.

        Each time step :meth:`update_target` computes a pressure correction
        :math:`\Delta p_\mathrm{throttle}` and adds it to
        :math:`p_\mathrm{out}`:

        .. math::

            \Delta p_\mathrm{throttle} = K_p\, \varepsilon
                + K_i \int \varepsilon\, d\tau
                + K_d \frac{d\varepsilon}{d\tau},
            \qquad \varepsilon = \dot{m} - \dot{m}_\mathrm{target}

        where :math:`\tau` is pseudo-time, advanced each step by
        :math:`d\tau = \mathrm{cfl} \cdot L_\mathrm{ref} / V_\mathrm{ref}`.

        Incompatible with non-scalar :attr:`P`.

        Parameters
        ----------
        mdot_target : float
            Target mass flow rate :math:`\dot{m}_\mathrm{target}` [kg/s].
            Must be positive and finite.
        K_pid : tuple of float
            PID gains :math:`(K_p, K_i, K_d)` with units
            [Pa/(kg/s)], [Pa/kg], [Pa·s/kg].
        """
        if not np.isscalar(mdot_target):
            raise TypeError("mdot_target must be a scalar")
        if not (np.isfinite(mdot_target) and mdot_target > 0):
            raise ValueError("mdot_target must be positive and finite")
        if not (isinstance(K_pid, (tuple, list)) and len(K_pid) == 3):
            raise TypeError("K_pid must be a tuple of three floats")
        if self._P_raw is not None and self._P_raw.ndim > 0 and self._P_raw.size > 1:
            raise ValueError("Non-scalar P is incompatible with throttle control")
        self._mdot_target = np.float32(mdot_target)
        self._K_pid = tuple(np.float32(k) for k in K_pid)
        self._err_int = np.float32(0.0)
        self._err_prev = np.float32(0.0)
        self._dP_P = np.float32(0.0)
        self._dP_I = np.float32(0.0)
        self._dP_D = np.float32(0.0)

    def get_throttle_stats(self):
        """Return a dict of throttle state for convergence logging.

        Keys
        ----
        mdot_target : float
            Mass flow setpoint [kg/s]; zero when throttle is inactive.
        mdot_throttle : float
            Mass flow measured at the patch on the last call to :meth:`update_target` [kg/s].
        P_throttle : float
            Total PID pressure correction :math:`\\Delta p_\\mathrm{throttle}` [Pa].
        dP_P : float
            Proportional contribution [Pa].
        dP_I : float
            Integral contribution [Pa].
        dP_D : float
            Derivative contribution [Pa].
        """
        return dict(
            mdot_target=self._mdot_target,
            mdot_throttle=self._mdot,
            P_throttle=self._dP_P + self._dP_I + self._dP_D,
            dP_P=self._dP_P,
            dP_I=self._dP_I,
            dP_D=self._dP_D,
        )

    def apply(self, rf=1.0):
        """Impose outlet boundary conditions on the patch.

        Forward-flow cells have static pressure imposed, with entropy and all
        three velocity components linearly extrapolated from the first two
        interior layers (``X_face = 2 * X_1 - X_2``) and stored via
        :py:meth:`~ember.block.Block.set_rho_u_Vxrt_nd`. If backflow handling has
        been enabled via :meth:`set_backflow`, reversed-flow cells are treated as
        inlets; see :class:`~ember.inlet.InletPatch`. If reversed flow is detected
        without :meth:`set_backflow` having been called, a :py:exc:`RuntimeError`
        is raised.

        Parameters
        ----------
        rf : float
            Relaxation factor for the reversed-cell density update.
        """
        P_ref = self.block.fluid.P_ref
        b = self.block_view

        if self._P_target_nd is None:
            self._P_target_nd = np.broadcast_to(self._P_raw / P_ref, b.shape).astype(
                np.float32
            )

        # Linear (two-point) extrapolation of entropy and all three velocity
        # components from the first two interior layers: the outgoing entropy
        # and vorticity characteristics carry these quantities from the
        # interior to the boundary, and the gradient between the offset-1 and
        # offset-2 slices is projected one layer further to the face,
        # X_face = 2 * X_1 - X_2.
        b1 = self.block_view_offset_1
        b2 = self.block_view_offset_2

        # Detect backflow from the interior layer-1 axial momentum, the
        # physical signal of flow entering the domain. This must be read
        # BEFORE the extrapolation below overwrites the boundary face's own
        # momentum: detecting from the post-extrapolation face couples the
        # test to the value apply() just authored, and since flagged faces
        # are forced to Vx < 0, that imposed reversal feeds back into the
        # interior and re-triggers the test, latching cells permanently in
        # backflow.
        inflow = b1.conserved_nd[..., 1] < 0

        # set_rho_u_Vxrt_nd takes nondimensional velocity, so extrapolate the
        # nondimensional components (Vx_nd = Vx / V_ref, linear so the division
        # commutes with the two-point extrapolation).
        V_ref = b.fluid.V_ref
        s_extrap = 2.0 * b1.s_nd - b2.s_nd
        Vx_extrap = (2.0 * b1.Vx - b2.Vx) / V_ref
        Vr_extrap = (2.0 * b1.Vr - b2.Vr) / V_ref
        Vt_extrap = (2.0 * b1.Vt - b2.Vt) / V_ref
        rho_nd, u_nd = b.fluid.set_P_s(self._P_target_nd, s_extrap)
        b.set_rho_u_Vxrt_nd(rho_nd, u_nd, Vx_extrap, Vr_extrap, Vt_extrap)
        if inflow.any():
            if not self._backflow_enabled:
                n_rev = int(inflow.sum())
                raise RuntimeError(
                    f"Reversed flow detected at outlet patch '{self.label}' "
                    f"({n_rev} cells with interior Vx < 0) but backflow handling is "
                    "not enabled. Call set_backflow() on the patch after "
                    "initialising the block to capture the snapshot to "
                    "impose on reversed cells."
                )

            if self._rho_nd_soln is None:
                self._rho_nd_soln = b.rho_nd.copy()

            ho_snap = self._inout_snapshot[..., 0]
            s_snap = self._inout_snapshot[..., 1]
            Vr_snap = self._inout_snapshot[..., 2]
            Vt_snap = self._inout_snapshot[..., 3]

            # Cap relaxed density at the value where static enthalpy equals
            # ho_snap - 0.5*(Vr_snap^2 + Vt_snap^2) so the Vx sqrt in the
            # inflow reconstruction below stays non-negative on inflow cells.
            fluid = self.block.fluid
            h_max_nd = ho_snap - 0.5 * (Vr_snap**2 + Vt_snap**2)
            rho_cap_nd = fluid.set_h_s(h_max_nd, s_snap)[0]
            rho_new_nd = self._rho_nd_soln + np.minimum(rf * np.abs(b.Max), 0.8) * (
                b.rho_nd - self._rho_nd_soln
            )
            np.minimum(rho_new_nd, 0.9999 * rho_cap_nd, out=rho_new_nd)

            cons_saved = b.conserved_nd.copy()
            ho_saved = b.ho_nd.copy()
            s_saved = b.s_nd.copy()
            rho_saved = b.rho_nd.copy()
            Vr_saved = b.Vr_nd.copy()
            Vt_saved = b.Vt_nd.copy()

            # See mixing.py for the full rationale on np.errstate here.
            # Density and internal energy follow from (rho, s) in closed form;
            # the axial velocity is then recovered from the energy equation,
            # Vx = sqrt(2*(ho - h(rho, s)) - Vr^2 - Vt^2), over the whole array
            # before any masking. On inflow cells the rho_cap clamp keeps the
            # radicand non-negative. On outflow cells we feed in (ho_saved,
            # s_saved, rho_saved, Vr_saved, Vt_saved) which is whatever the
            # pressure BC just produced; small EOS round-trip drift can push the
            # radicand slightly negative, producing NaN Vx and a numpy "invalid
            # value in sqrt" warning. Those NaNs land in outflow cells of
            # b.conserved_nd, but the np.where below immediately overwrites
            # every outflow cell from cons_saved, so the NaNs never escape
            # this block. Suppress the spurious warning here; if the inflow
            # branch ever produced a NaN it would survive past the np.where
            # and show up as NaN in subsequent solver state.
            ho_arg = np.where(inflow, ho_snap, ho_saved)
            s_arg = np.where(inflow, s_snap, s_saved)
            rho_arg = np.where(inflow, rho_new_nd, rho_saved)
            Vr_arg = np.where(inflow, Vr_snap, Vr_saved)
            Vt_arg = np.where(inflow, Vt_snap, Vt_saved)
            rho_nd, u_nd = b.fluid.set_rho_s(rho_arg, s_arg)
            with np.errstate(invalid="ignore"):
                Vx_arg = np.sqrt(
                    2.0 * (ho_arg - b.fluid.get_h(rho_nd, u_nd)) - Vr_arg**2 - Vt_arg**2
                )
                b.set_rho_u_Vxrt_nd(rho_nd, u_nd, Vx_arg, Vr_arg, Vt_arg)

            # Setter emits Vx >= 0; reversed flow at the outlet needs Vx < 0.
            b.conserved_nd[..., 1] = np.where(
                inflow, -b.conserved_nd[..., 1], b.conserved_nd[..., 1]
            )

            # Restore outflow cells from the pressure-BC result.
            mask = inflow[..., None]
            b.conserved_nd[...] = np.where(mask, b.conserved_nd, cons_saved)
            b.update_cached_conserved()

    def update_soln(self):
        """Update :math:`\\rho_\\mathrm{soln}` from the current interior density.

        Should be called once per timestep before the Runge-Kutta stages so
        that density relaxation in reversed-flow cells during :meth:`apply`
        is anchored to the start-of-step density.
        """
        self._rho_nd_soln = self.block_view.rho_nd.copy()

    def update_target(self, cfl=None):
        r"""Recompute the pressure target for the current timestep.

        Runs the throttle PID (if active) and applies the spanwise adjustment
        (if configured). Should be called once per outer timestep before the
        Runge-Kutta stages.

        Parameters
        ----------
        cfl : float, optional
            CFL number controlling the pseudo-time step
            :math:`d\tau = \mathrm{cfl} \cdot L_\mathrm{ref} / V_\mathrm{ref}`,
            where :math:`L_\mathrm{ref}` is :py:attr:`~ember.block.Block.L_ref` and
            :math:`V_\mathrm{ref}` is the fluid reference velocity.
            Defaults to the internal ``_cfl_pid`` value if not provided.
        """
        P_ref = self.block.fluid.P_ref

        # Pitch-average the boundary face when needed by the throttle. The
        # radial-equilibrium offset below pitch-averages the offset-1 interior
        # slice directly and does not depend on block_avg.
        if self._mdot_target:
            self.set_block_avg()

        # PID throttle
        if self._mdot_target:
            mflux = self.block_avg.conserved[:, 1]
            self._mdot = np.sum(mflux * self._dA_node)
            err = self._mdot - self._mdot_target
            Kp, Ki, Kd = self._K_pid
            _cfl = cfl if cfl is not None else self._cfl_pid
            dt = _cfl * self.block.L_ref / self.block.fluid.V_ref
            self._err_int += err * dt
            self._dP_P = np.float32(Kp * err)
            self._dP_I = np.float32(Ki * self._err_int)
            self._dP_D = np.float32(Kd * (err - self._err_prev) / dt)
            self._err_prev = err

        dP_throttle_nd = np.float32((self._dP_P + self._dP_I + self._dP_D) / P_ref)

        # Nondimensional base pressure
        P_raw_nd = self._P_raw / P_ref

        # Build full-shape nondim target from scalar level
        P_target_nd = np.broadcast_to(
            P_raw_nd + dP_throttle_nd, self.block_view.shape
        ).astype(np.float32)

        # Apply pressure adjustment if set_adjustment was called
        if self._adjustment:
            K_dyn = self._adjustment["K_dyn"]
            do_re = self._adjustment["radial_equilibrium"]
            rf = self._adjustment["rf"]

            # --- Dynamic head offset ---
            # Proportional to local meridional dynamic head.
            # With radial_equilibrium: pitch-area-average removed at each span station
            # (zero pitch-avg at every span, no spanwise redistribution).
            # Without radial_equilibrium: overall area-average removed instead.
            if K_dyn != 0.0:
                q_nd = 0.5 * self.block_view.rho_nd * self.block_view.Vm_nd**2
                dyn_raw = K_dyn * q_nd
                if do_re:
                    dA_pitch_sum = self._dA_full.sum(axis=self.pitch_dim, keepdims=True)
                    dyn_span_avg = (dyn_raw * self._dA_full).sum(
                        axis=self.pitch_dim, keepdims=True
                    ) / dA_pitch_sum
                    dyn_offset = (dyn_raw - dyn_span_avg).astype(np.float32)
                else:
                    dyn_global_avg = np.sum(dyn_raw * self._dA_full) / np.sum(
                        self._dA_full
                    )
                    dyn_offset = (dyn_raw - dyn_global_avg).astype(np.float32)
            else:
                dyn_offset = np.zeros(self.block_view.shape, dtype=np.float32)

            # --- Radial equilibrium offset ---
            # Centrifugal pressure rise integrated along span; anchored to hub.
            # Flow quantities are pitch-averaged over the first interior layer
            # (the offset-1 slice), the same source apply() extrapolates the
            # boundary state from. The integrand reproduces the Multall
            # EXBCONDS IPOUT=3 form: pitch-mean(rho*Vt) * pitch-mean(Vt) / r.
            if do_re:
                b1 = self.block_view_offset_1
                w = self.weight_pitch
                pd = self.pitch_dim
                rhoVt_mean = np.sum(b1.rho_nd * b1.Vt_nd * w, axis=pd).squeeze()
                Vt_mean = np.sum(b1.Vt_nd * w, axis=pd).squeeze()
                r_nd = np.sum(b1.r_nd * w, axis=pd).squeeze()
                assert not np.any(np.isnan(r_nd)), (
                    "radial_equilibrium: r_nd contains NaN"
                )
                assert not np.any(np.isnan(rhoVt_mean)), (
                    "radial_equilibrium: rho*Vt contains NaN"
                )
                assert not np.any(np.isnan(Vt_mean)), (
                    "radial_equilibrium: Vt contains NaN"
                )
                dPdr_nd = rhoVt_mean * Vt_mean / r_nd
                dr_nd = np.diff(r_nd)
                P_re_span_nd = np.empty(len(r_nd))
                P_re_span_nd[0] = 0.0
                P_re_span_nd[1:] = np.cumsum(0.5 * (dPdr_nd[:-1] + dPdr_nd[1:]) * dr_nd)
                # Anchor to the hub: integration starts at the hub with a zero
                # offset there, so the prescribed p_out is enforced at the hub
                # and the profile rises centrifugally toward the tip.
                re_offset = np.broadcast_to(
                    self._span_bcast(P_re_span_nd), self.block_view.shape
                ).astype(np.float32)
            else:
                re_offset = np.zeros(self.block_view.shape, dtype=np.float32)

            # Combined offset: dyn part has zero area-average; re part is zero
            # at the hub. Both have zero pitch-area-average at every span.
            combined = dyn_offset + re_offset

            # Relax
            if self._P_last_nd is None:
                self._P_last_nd = combined.copy()
            P_profile_nd = rf * combined + (1 - rf) * self._P_last_nd
            self._P_last_nd = P_profile_nd.copy()

            P_target_nd += P_profile_nd

        self._P_target_nd = P_target_nd

    @property
    def K_pid(self):
        """PID gains ``(Kp, Ki, Kd)`` for throttle control; set via :meth:`set_throttle`."""
        return self._K_pid

    @property
    def mdot_target(self):
        """Mass flow target [kg/s] for throttle control; zero when throttle is inactive."""
        return self._mdot_target

    @property
    def P(self):
        r"""Prescribed outlet static pressure :math:`p_\mathrm{out}` [Pa]. See :py:attr:`~ember.block.Block.P`."""
        return self._P_raw
