"""Mixing plane boundary condition patch for EMBER CFD.

MixingPatch provides circumferentially-averaged connections between blade rows in
multi-stage turbomachinery. Conserves mass, momentum, and energy flows across the
mixing plane while allowing circumferential variations on each side.

See Also
--------
ember.patch.Patch : Base class for all patches
ember.patch.PeriodicPatch : For circumferentially periodic boundaries
"""

import numpy as np
from ember import util
from ember.basepatch import RevolutionPatch


class MixingPatch(RevolutionPatch):
    """Circumferentially averaged connection between blade rows.

    Connects two block faces at a mixing plane by exchanging pitch-averaged flow
    state. Inflow nodes are treated as in :class:`~ember.inlet.InletPatch`;
    outflow nodes as in :class:`~ember.outlet.OutletPatch`. Circumferential
    variations on each side are permitted; only the pitch average crosses the
    interface.

    Both sides must have the same spanwise resolution. Pitchwise resolution may
    differ.
    """

    _collection_name = "mixing"

    def _setup(self):
        super()._setup()
        self._flux_avg = None
        self._target = None
        self._P_nd_soln = None
        self._P_nd_face = None
        # Cycle-level relaxation factor for p_soln (toward the interior
        # pressure), read by update_soln().
        self.rf = 0.1
        # Stage-level relaxation factor for p_face (toward p_soln), read by
        # apply().
        self.rf_stage = 0.1

    def _copy(self, c):
        c.rf = self.rf
        c.rf_stage = self.rf_stage

    def set_flux_avg(self):
        """Compute pitch-averaged node fluxes and store in :attr:`flux_avg_nd`.

        Called by :class:`~ember.mixing_communicator.MixingCommunicator` before
        reading :attr:`flux_avg_nd` to form the cross-plane flux difference.
        Must be called with the block in interface coordinates so that
        ``Vx`` is the face-normal velocity and ``Vr`` is the spanwise velocity.
        """
        import ember.fortran as ft

        b = self.block_view
        cons = b.conserved_nd
        w = self.weight_pitch.ravel()
        ni, nj, nk = b.shape
        if self.pitch_dim == 0:
            dest = self._flux_avg.reshape(nj, nk, 5)
            ft.flux_avg_i(cons, b.P_nd, b.ho_nd, w, dest)
        elif self.pitch_dim == 1:
            dest = self._flux_avg.reshape(ni, nk, 5)
            ft.flux_avg_j(cons, b.P_nd, b.ho_nd, w, dest)
        else:
            dest = self._flux_avg.reshape(ni, nj, 5)
            ft.flux_avg_k(cons, b.P_nd, b.ho_nd, w, dest)

    def set_target(self, target=None):
        """Set the pitch-uniform inout target from the current block state or an explicit array.

        :class:`~ember.mixing_communicator.MixingCommunicator` calls this to
        write back the updated cross-plane target after each exchange. If
        ``target`` is omitted, the pitch mean of the current block state is used
        to initialise the target before the first exchange.

        Parameters
        ----------
        target : array of shape ``(nspan, 5)``, optional
            Nondimensional ``[ho, s, Vr, Vt, P]`` target values.
        """
        self._check_attached()
        shape = [1, 1, 1]
        shape[self.span_dim] = self._block_view.shape[self.span_dim]
        if self._target is None:
            self._target = util.zeros((*shape, 5))
        if target is None:
            # Read primitives directly and pitch-average. Going via
            # block_view.mean() returns a meaned-block whose derived primitive
            # properties read as zero before the conserved cache is primed,
            # which would silently write zero targets.
            b = self.block_view
            pd = self.pitch_dim
            self._target[..., 0] = b.ho_nd.mean(axis=pd, keepdims=True)
            self._target[..., 1] = b.s_nd.mean(axis=pd, keepdims=True)
            self._target[..., 2] = b.Vr_nd.mean(axis=pd, keepdims=True)
            self._target[..., 3] = b.Vt_nd.mean(axis=pd, keepdims=True)
            self._target[..., 4] = b.P_nd.mean(axis=pd, keepdims=True)
        else:
            self._target[...] = target.reshape((*shape, 5))

    def get_target(self):
        """Return the inout target as a nondimensional ``(nspan, 5)`` array of ``[ho, s, Vr, Vt, P]``.

        Read by :class:`~ember.mixing_communicator.MixingCommunicator` during
        the exchange to form the average target across the mixing plane.
        """
        self._check_attached()
        if self._target is None:
            self.set_target()
        return self._target.squeeze()

    def _P_cap_nd(self):
        """Pressure at which all remaining kinetic energy in the target is axial.

        Going above this cap forces the ``Vx`` sqrt in the inflow branch of
        :meth:`apply` negative, so both :meth:`apply` and :meth:`update_soln`
        clamp against it. Recomputed on every call rather than cached because
        the target's ``ho``/``Vr``/``Vt`` are refreshed by
        :class:`~ember.mixing_communicator.MixingCommunicator` every
        Runge-Kutta stage.
        """
        if self._target is None:
            self.set_target()
        ho_nd = self._target[..., 0]
        s_nd = self._target[..., 1]
        Vr_nd = self._target[..., 2]
        Vt_nd = self._target[..., 3]
        fluid = self.block.fluid
        h_max_nd = ho_nd - 0.5 * (Vr_nd**2 + Vt_nd**2)
        return fluid.get_P(*fluid.set_h_s(h_max_nd, s_nd))

    def apply(self):
        """Impose mixing-plane boundary conditions on the patch.

        Called each Runge-Kutta stage after
        :class:`~ember.mixing_communicator.MixingCommunicator` has updated the
        target via :meth:`set_target`. Inflow nodes are treated as in
        :class:`~ember.inlet.InletPatch`: stagnation enthalpy, entropy, and
        transverse velocities are imposed from the target and static pressure
        is :math:`p_\\mathrm{face}`, relaxed here (once per stage, by
        :attr:`rf_stage`) toward the fixed cycle target :math:`p_\\mathrm{soln}`
        -- itself relaxed toward the first interior layer once per step by
        :meth:`update_soln`, at its own rate :attr:`rf` -- both clamped
        against :meth:`_P_cap_nd`. Outflow nodes are
        treated as in :class:`~ember.outlet.OutletPatch`: target static
        pressure is imposed and entropy plus all three velocity components are
        linearly extrapolated from the first two interior layers.
        """
        b = self.block_view
        b1 = self.block_view_offset_1

        # Self-seed from the current block state if nothing has ever set the
        # target yet (mirrors the _P_nd_soln guard below).
        if self._target is None:
            self.set_target()

        inout_target = np.broadcast_to(self._target, b.shape + (5,))
        P_nd = inout_target[..., 4]

        # Inlet side: stage-level relaxation, mirroring InletPatch. p_face
        # chases the fixed cycle target p_soln at rate rf_stage, once per
        # Runge-Kutta stage.
        if self._P_nd_soln is None:
            self._P_nd_soln = b.P_nd.copy()
        if self._P_nd_face is None:
            self._P_nd_face = self._P_nd_soln.copy()
        else:
            self._P_nd_face = self._P_nd_face + self.rf_stage * (
                self._P_nd_soln - self._P_nd_face
            )

        np.minimum(self._P_nd_face, 0.9999 * self._P_cap_nd(), out=self._P_nd_face)
        P_new_nd = self._P_nd_face

        # Build target array with relaxed pressure replacing the target P
        target_inlet = inout_target.copy()
        target_inlet[..., 4] = P_new_nd

        # Detect reversed flow from the interior layer-1 axial momentum, the
        # physical signal of flow direction at the plane. Reading the boundary
        # face instead would couple the test to the value apply() authored on
        # the previous step: upstream-block inflow cells are forced to Vx < 0
        # below, so a face-based test would re-read that imposed reversal next
        # step and latch the cell permanently in backflow regardless of what
        # the interior is doing. Layer 1 is never written by apply().
        inflow = b1.conserved_nd[..., 1] > 0
        # Hack to detect which side of the mixer we are on
        xm = self.block.x.mean()
        xp = self.block_view.x.mean()
        upstream = xm < xp
        if upstream:
            inflow = ~inflow

        outflow = ~inflow

        if inflow.any():
            # Inlet branch: impose the relaxed-pressure target on inflow cells.
            # Block.masked confines the setter to the inflow cells, snapshotting
            # and restoring everything else, so outflow cells are untouched.
            #
            # Density and internal energy follow from (P, s) in closed form;
            # the axial velocity is then recovered from the energy equation,
            # Vx = sqrt(2*(ho - h(P, s)) - Vr^2 - Vt^2), with Vr and Vt fixed
            # from the neighbouring block.
            #
            # The sqrt is evaluated over the whole array before the mask
            # rollback. On inflow cells the P_cap clamp above keeps the radicand
            # non-negative; on the cells that get rolled back, small EOS /
            # conserved->primitive round-trip drift can push the radicand
            # slightly negative, yielding a NaN Vx and a numpy "invalid value in
            # sqrt" warning. Those NaNs only land in cells the mask then
            # restores, so they never escape this block; the errstate suppresses
            # the spurious warning. A NaN surviving into subsequent solver state
            # would be the real failure to watch for.
            ho_nd = target_inlet[..., 0]
            s_nd = target_inlet[..., 1]
            Vr_nd = target_inlet[..., 2]
            Vt_nd = target_inlet[..., 3]
            P_in_nd = target_inlet[..., 4]
            rho_nd, u_nd = b.fluid.set_P_s(P_in_nd, s_nd)
            with np.errstate(invalid="ignore"):
                Vx_nd = np.sqrt(
                    2.0 * (ho_nd - b.fluid.get_h(rho_nd, u_nd)) - Vr_nd**2 - Vt_nd**2
                )
                b.masked(inflow).set_rho_u_Vxrt_nd(rho_nd, u_nd, Vx_nd, Vr_nd, Vt_nd)
            # Setter always produces Vx >= 0. On the upstream block, inflow
            # cells are reversed flow re-entering through the mixing plane and
            # need Vx < 0. Downstream block inflow cells already have the
            # right sign. Writing conserved_nd directly bypasses cache
            # invalidation, so flush it explicitly.
            if upstream:
                b.conserved_nd[..., 1] = np.where(
                    inflow, -b.conserved_nd[..., 1], b.conserved_nd[..., 1]
                )
                b.update_cached_conserved()

        if outflow.any():
            # Outlet branch: impose target P on outflow cells. Block.masked
            # confines the setters to the outflow cells, leaving inflow cells
            # (already authored above) untouched.
            #
            # Linear extrapolation of entropy and all three velocity
            # components from the first two interior layers: the outgoing
            # entropy and vorticity characteristics carry these quantities
            # from the interior to the boundary, so X_face = 2*X_1 - X_2
            # transports them to the face with second-order accuracy.
            # set_rho_u_Vxrt_nd takes nondimensional velocity, so extrapolate the
            # nondimensional components (Vx_nd = Vx / V_ref, linear so the
            # division commutes with the two-point extrapolation).
            b2 = self.block_view_offset_2
            V_ref = b.fluid.V_ref
            s_extrap = 2.0 * b1.s_nd - b2.s_nd
            Vx_extrap = (2.0 * b1.Vx - b2.Vx) / V_ref
            Vr_extrap = (2.0 * b1.Vr - b2.Vr) / V_ref
            Vt_extrap = (2.0 * b1.Vt - b2.Vt) / V_ref
            rho_nd, u_nd = b.fluid.set_P_s(P_nd, s_extrap)
            with np.errstate(invalid="ignore"):
                b.masked(outflow).set_rho_u_Vxrt_nd(
                    rho_nd, u_nd, Vx_extrap, Vr_extrap, Vt_extrap
                )

    def attach_to_block(self, block):
        """Attach to block and allocate internal storage.

        Safe to call repeatedly; an existing ``_target`` of the correct
        shape is preserved so targets set before re-attachment are not lost.
        ``_target`` itself is left unallocated (``None``) until first used --
        :meth:`set_target` and :meth:`get_target` seed it lazily from the
        current block state -- so a fresh attach never masks the "nothing has
        set a real target yet" state behind a zero-filled array.
        """
        super().attach_to_block(block)
        self._build_rot_matrices(inward=True)

        nspan = self._block_view.shape[self.span_dim]
        self._flux_avg = util.zeros((nspan, 5))

        shape = [1, 1, 1]
        shape[self.span_dim] = self._block_view.shape[self.span_dim]
        target_shape = (*shape, 5)
        if self._target is not None and self._target.shape != target_shape:
            self._target = None

    def check_match(self, other, rtol=1e-5):
        """Check if this MixingPatch matches another for pairing purposes.

        MixingPatch matching requires x,r coordinates to match at all spanwise
        nodes (pitch-averaged), ignoring theta. Patches must have the same
        spanwise resolution but can have different pitchwise resolutions.

        Parameters
        ----------
        other : Patch
            The other patch to compare with
        rtol : float, optional
            Relative tolerance for matching

        Returns
        -------
        bool or None
            None if patches do not match. False if they match with no spanwise
            flip needed. True if they match but other's span must be reversed.
            Always test with ``is not None``; do not use as a bare truthiness
            check since False is a valid match result.
        """
        if not isinstance(other, MixingPatch):
            return None

        if self.shape[self.span_dim] != other.shape[other.span_dim]:
            return None

        return self._check_match_xr(other, rtol)

    def update_soln(self):
        r"""Advance the cycle target :math:`p_\mathrm{soln}` by one relaxation step.

        Called once per timestep, before the Runge-Kutta stages: relaxes
        :math:`p_\mathrm{soln}` toward the first interior layer's pressure at
        the start of the step by :attr:`rf`. :meth:`apply` then chases this
        fixed target with the live boundary pressure :math:`p_\mathrm{face}`
        once per stage, at its own rate :attr:`rf_stage` (reapplying the cap
        against the target as refreshed that stage); see
        :class:`~ember.inlet.InletPatch`.
        """
        self._check_attached()
        b = self.block_view
        if self._P_nd_soln is None:
            self._P_nd_soln = b.P_nd.copy()
            return
        P_interior_nd = self.block_view_offset_1.P_nd
        P_new_nd = self._P_nd_soln + self.rf * (P_interior_nd - self._P_nd_soln)
        np.minimum(P_new_nd, 0.9999 * self._P_cap_nd(), out=P_new_nd)
        self._P_nd_soln = P_new_nd

    @property
    def flux_avg_nd(self):
        """Pitch-averaged flux array, shape ``(nspan, 5)``; populated by :meth:`set_flux_avg` and read by :class:`~ember.mixing_communicator.MixingCommunicator` to form the cross-plane flux difference."""
        self._check_attached()
        return self._flux_avg
