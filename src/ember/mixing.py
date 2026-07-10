"""Mixing plane boundary condition patch for EMBER CFD.

MixingPatch provides circumferentially-averaged connections between blade rows in
multi-stage turbomachinery. Conserves mass, momentum, and energy flows across the
mixing plane while allowing circumferential variations on each side.

See Also
--------
ember.patch.Patch : Base class for all patches
ember.patch.PeriodicPatch : For circumferentially periodic boundaries
"""

import itertools

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
        # Relaxation factor for the inflow pressure update, read by apply().
        self.rf = 1.0

    def _copy(self, c):
        c.rf = self.rf

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

    def apply(self):
        """Impose mixing-plane boundary conditions on the patch.

        Called each Runge-Kutta stage after
        :class:`~ember.mixing_communicator.MixingCommunicator` has updated the
        target via :meth:`set_target`. Inflow nodes are treated as in
        :class:`~ember.inlet.InletPatch`: stagnation enthalpy, entropy, and
        transverse velocities are imposed from the target and static pressure is
        relaxed toward a linear extrapolation of the first two interior layers
        using the relaxation factor :attr:`rf` (used directly as the convex
        weight toward the first interior layer, no Mach scaling). Outflow nodes
        are treated as in :class:`~ember.outlet.OutletPatch`: target static
        pressure is imposed and entropy plus all three velocity components are
        linearly extrapolated from the first two interior layers.
        """
        b = self.block_view

        # Self-seed from the current block state if nothing has ever set the
        # target yet (mirrors the _P_nd_soln guard below).
        if self._target is None:
            self.set_target()

        inout_target = np.broadcast_to(self._target, b.shape + (5,))
        ho_nd = inout_target[..., 0]
        s_nd = inout_target[..., 1]
        Vr_nd = inout_target[..., 2]
        Vt_nd = inout_target[..., 3]
        P_nd = inout_target[..., 4]

        # Inlet side: static-pressure relaxation, mirroring InletPatch.
        if self._P_nd_soln is None:
            self.update_soln()

        # Cap relaxed pressure at the pressure where static enthalpy equals
        # ho - 0.5*(Vr^2 + Vt^2), i.e. all remaining kinetic energy is axial.
        # Going above this cap forces the Vx sqrt in the inflow branch below
        # negative. Recomputed each call because the conservation-adjustment
        # routine mutates Vr/Vt between calls.
        fluid = self.block.fluid
        h_max_nd = ho_nd - 0.5 * (Vr_nd**2 + Vt_nd**2)
        P_cap_nd = fluid.get_P(*fluid.set_h_s(h_max_nd, s_nd))

        # Linearly extrapolate static pressure from the first two interior
        # layers (P_face = 2*P_1 - P_2) and relax toward it, mirroring
        # InletPatch: rf is used directly as the convex weight (no Mach
        # scaling), anchored to the start-of-step face pressure.
        b1 = self.block_view_offset_1
        b2 = self.block_view_offset_2
        P_interior_nd = 2.0 * b1.P_nd - b2.P_nd
        P_new_nd = self._P_nd_soln + self.rf * (P_interior_nd - self._P_nd_soln)
        np.minimum(P_new_nd, 0.9999 * P_cap_nd, out=P_new_nd)

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

        perm = [0, 0, 0]
        perm[self.const_dim] = other.const_dim
        perm[self.span_dim] = other.span_dim
        perm[self.pitch_dim] = other.pitch_dim
        perm = tuple(perm)

        flip_axes = [ax for ax in (self.span_dim, self.pitch_dim) if self.shape[ax] > 1]
        flip_candidates = [
            combo
            for r in range(len(flip_axes) + 1)
            for combo in itertools.combinations(flip_axes, r)
        ]

        for flip in flip_candidates:
            if not self._compare_coords(
                other, (perm, flip), corners_only=True, xr_only=True, rtol=rtol
            ):
                continue

            span_flipped = self.span_dim in flip
            other_spf = 1.0 - other.spf[::-1] if span_flipped else other.spf
            if np.allclose(self.spf, other_spf, atol=1e-4, rtol=0):
                return span_flipped
            else:
                print("spf mismatch with flip", flip)
                print(self.spf[(0, -1),])
                print(other_spf[(0, -1),])
                err = np.abs(self.spf - other_spf)
                print("max abs error:", err.max())
                print("mean abs error:", err.mean())

        return None

    def update_soln(self):
        r"""Update :math:`p_\mathrm{soln}` from the current face static pressure.

        Should be called once per timestep before the Runge-Kutta stages so
        that the inflow pressure relaxation in :meth:`apply` is anchored to the
        start-of-step pressure; see :class:`~ember.inlet.InletPatch`.
        """
        self._check_attached()
        b = self.block_view
        self._P_nd_soln = b.P_nd.copy()

    @property
    def flux_avg_nd(self):
        """Pitch-averaged flux array, shape ``(nspan, 5)``; populated by :meth:`set_flux_avg` and read by :class:`~ember.mixing_communicator.MixingCommunicator` to form the cross-plane flux difference."""
        self._check_attached()
        return self._flux_avg
