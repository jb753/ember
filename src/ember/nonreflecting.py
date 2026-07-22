r"""Shared machinery for the steady non-reflecting boundary conditions of EMBER CFD.

:class:`NonReflectingPatch` holds everything the non-reflecting inflow and
outflow conditions of :cite:t:`Giles1988` (his Chapter 5), as extended to three
dimensions by :cite:t:`Saxer1993`, have in common: the pitchwise Hilbert
transform their harmonic relations are written with, the frozen pitchwise-mean
reference state the characteristic variables are perturbations about, and the
split of the boundary node into the characteristics the interior march owns and
those the boundary condition owns.

The two conditions are mirror images. Of the five characteristics at an axially
subsonic boundary, four propagate downstream (entropy, both vorticity waves, the
downstream-running pressure wave) and one, the upstream-running pressure wave,
propagates upstream; so an inflow plane has four incoming characteristics and
one outgoing, and an outflow plane has one incoming and four outgoing. Each
Runge-Kutta stage the outgoing characteristics are read from the boundary node
exactly as the interior scheme left them and are never overwritten, so a wave
reaching the boundary passes through it. The incoming characteristics are
discarded and rebuilt from the prescribed mean state and the non-reflecting
relations, then applied under-relaxed by :attr:`~NonReflectingPatch.sigma`.

See Also
--------
ember.basepatch.RevolutionPatch : Base class providing the pitchwise geometry
ember.inlet_nonreflecting.NonReflectingInletPatch : Subsonic inflow
ember.outlet_nonreflecting.NonReflectingOutletPatch : Subsonic outflow
ember.perturbation.chic_to_bcond : Jacobian the characteristic solves are built on
"""

from abc import abstractmethod

import numpy as np

from ember import perturbation, util
from ember.basepatch import RevolutionPatch


class NonReflectingPatch(RevolutionPatch):
    r"""Base class for the steady non-reflecting inflow and outflow conditions.

    Subclasses supply four class attributes and two hooks. The attributes are
    :attr:`_desc`, a description used in error messages; :attr:`_idx_out`, the
    characteristic components the interior march is allowed to set;
    :attr:`_sign_interior`, which side of the face the interior lies on; and
    :attr:`_target_setters`, mapping each prescribed quantity to the setter that
    fills it. The hooks are :meth:`_calc_reference_extra`, which adds the
    Jacobians and coefficients that condition needs to the frozen reference
    state, and :meth:`_calc_dchic`, which returns the change required in the
    incoming characteristics.

    :meth:`apply` is called once per Runge-Kutta stage and :meth:`update_soln`
    once per timestep, the latter refreshing the reference state to match Giles'
    definition of the characteristic variables as perturbations about the
    time-level-:math:`n` average.

    Both conditions are restricted to a constant-:math:`x` plane with the flow
    running in the :math:`+x` direction and an axially subsonic, absolutely
    subsonic mean state; each restriction is checked and raises.
    """

    # Relative tolerance for the geometric checks made at attach time.
    _rtol_geom = 1e-4

    # Description of the patch used in error messages; lower case, so it can be
    # capitalised where it starts a sentence.
    _desc = None

    # Characteristic components carried from the marched face rather than from
    # this patch's own previous output: [0] at an inflow plane, [1, 2, 3, 4] at
    # an outflow plane.
    _idx_out = None

    # +1 if the interior lies on the +x side of the face, -1 if on the -x side.
    _sign_interior = None

    # Prescribed quantity -> the setter, or setters, that fill it.
    _target_setters = None

    def _calc_hilbert(self):
        r"""Build the pitchwise Hilbert transform matrix.

        The non-reflecting relations are written per pitchwise Fourier mode
        :math:`m` in terms of the wave parameter (Giles Eq. 5.18, Saxer Eq. 15)

        .. math::
            \beta = i\,\mathrm{sign}(m)\sqrt{1 - M^2},

        which depends on the mode only through :math:`\mathrm{sign}(m)`. At an
        inflow plane the relation for the tangential vorticity characteristic is
        (Giles Eq. 5.17, Saxer Eq. 56)

        .. math::
            \hat{c}_t = -\frac{\beta + M_t}{1 + M_n}\hat{c}_\mathrm{up},

        and at an outflow plane the relation for the upstream-running pressure
        characteristic is (Giles Eq. 5.32, Saxer Eq. 57)

        .. math::
            \hat{c}_\mathrm{up} = \frac{2M_n}{\beta - M_t}\hat{c}_t
                - \frac{\beta + M_t}{\beta - M_t}\hat{c}_\mathrm{down}.

        In both, splitting :math:`\beta` from the real terms separates a local
        term from a Hilbert transform along the pitch, and no Fourier transform
        need be taken at run time; for the inflow relation,

        .. math::
            c_t = -\frac{M_t}{1 + M_n}c_\mathrm{up}
                  + \frac{\sqrt{1 - M^2}}{1 + M_n}\mathcal{H}[c_\mathrm{up}].

        The sign of the Hilbert term deserves care. Giles writes
        :math:`\beta = i\,\mathrm{sign}(k)\sqrt{1-M^2}` for the continuous
        wavenumber, but his transform pair (analysis with
        :math:`\exp(+2\pi ijk/N)`, synthesis with :math:`\exp(-2\pi ijk/N)`)
        represents the field as :math:`\exp(-ily)`, so the discrete mode index
        carries the opposite sign to the continuous wavenumber. Getting it
        backwards turns the condition from absorbing into amplifying. The
        physical check is steady potential flow upstream of a blade row:
        :math:`(1-M^2)\phi_{xx} + \phi_{yy} = 0` admits
        :math:`\phi \sim \exp(\mu x + ily)` with
        :math:`\mu = |l|/\sqrt{1-M^2}`, decaying upstream, and with
        :math:`p' = -\bar{\rho}\bar{u}u'` this gives
        :math:`c_t/c_\mathrm{up} = -i\,\mathrm{sign}(l)\sqrt{1-M^2}/(1+M)`.

        Evaluating the analysis and synthesis sums directly with the node
        weights :attr:`~ember.basepatch.RevolutionPatch.weight_pitch` gives

        .. math::
            \mathcal{H}_{ab} = -2\sum_{m=1}^{M} w_b
                \sin\left(\frac{2\pi m(\theta_b - \theta_a)}{P}\right),

        which needs no assumption of uniform pitchwise spacing: the weights are
        a quadrature rule that already sums to one and already splits the
        duplicated periodic end node into two half weights. Modes are truncated
        at :math:`M = (N-1)//2` over the :math:`N` distinct nodes, excluding the
        Nyquist mode whose sign is ambiguous, as Giles does. Built in double
        precision and stored single.

        On a uniform mesh the quadrature is exact and the matrix reproduces the
        discrete Hilbert transform to round-off. On a stretched mesh it stays
        accurate for harmonics resolved by the *coarsest* local spacing and
        degrades progressively above that, so the highest resolved harmonics of
        a strongly stretched pitch are absorbed only approximately. The
        operator norm stays O(1) either way, so the failure mode is a boundary
        that reflects a little at the shortest wavelengths, never one that
        amplifies them.
        """
        block = self.block
        pitch = float(block.pitch)

        # Pitchwise node angles, taken at the first span station as
        # weight_pitch does; check the distribution is the same at every span
        # station, since one matrix is shared by all of them.
        t_sp = self.block_view.t.transpose(self._std_perm).squeeze(axis=0)
        t_rel = t_sp - t_sp[:, :1]
        if np.abs(t_rel - t_rel[0]).max() > self._rtol_geom * pitch:
            raise ValueError(
                f"{self._desc.capitalize()} {self.label!r} requires the same "
                "pitchwise node distribution at every span station."
            )

        w = self.weight_pitch.ravel().astype(np.float64)
        if abs(w.sum() - 1.0) > self._rtol_geom:
            raise ValueError(
                f"{self._desc.capitalize()} {self.label!r} must span a whole "
                f"pitch: node weights sum to {w.sum():.6g}, expected 1."
            )

        t0 = (t_rel[0] - t_rel[0, 0]).astype(np.float64)
        # A patch spanning the whole pitch repeats the periodic node at both
        # ends; it must not be counted twice when truncating the mode range.
        wraps = abs(t0[-1] - pitch) < self._rtol_geom * pitch
        n_dist = len(t0) - 1 if wraps else len(t0)
        m_max = (n_dist - 1) // 2
        if m_max < 1:
            raise ValueError(
                f"{self._desc.capitalize()} {self.label!r} needs at least 3 "
                f"pitchwise nodes to resolve one harmonic, got {n_dist}."
            )

        phi = 2.0 * np.pi * (t0[None, :] - t0[:, None]) / pitch
        hilbert = np.zeros_like(phi)
        for m in range(1, m_max + 1):
            hilbert -= 2.0 * w[None, :] * np.sin(m * phi)

        # On a uniform mesh the sums above already annihilate a constant and
        # return a mean-free field, because the trapezoidal rule is exact for
        # the periodic integrands. Non-uniform spacing leaves a quadrature
        # error in both, which would let the transform leak into the pitchwise
        # mean and double-count against the mean-mode solve. Project it out:
        # the first correction is rank one in the weighted mean of the input,
        # so it does not disturb the mean-free fields apply() actually passes,
        # and the second removes the weighted mean of the output.
        hilbert -= hilbert.sum(axis=1, keepdims=True) * w[None, :]
        hilbert -= (w[:, None] * hilbert).sum(axis=0, keepdims=True)
        self._hilbert = hilbert.astype(np.float32)

    def _calc_reference(self):
        """Freeze the pitchwise-mean state and everything derived from it.

        Called once per timestep from :meth:`update_soln`. Giles defines the
        characteristic variables as perturbations about the average flow at time
        level ``n``, so the reference state and every Jacobian evaluated on it
        are held fixed for all the Runge-Kutta stages of a step.
        """
        self.set_block_avg()
        avg = self.block_avg

        a_nd = avg.a_nd
        Mn = avg.Vx_nd / a_nd
        Mt = avg.Vt_nd / a_nd
        Msq = Mn**2 + Mt**2

        if np.any(avg.Vx_nd <= 0.0):
            raise ValueError(
                f"Backflow at {self._desc} {self.label!r}: the pitchwise-mean "
                f"axial velocity is negative at "
                f"{int(np.count_nonzero(avg.Vx_nd <= 0.0))} of {avg.Vx_nd.size} "
                "span stations, so the characteristic split is invalid."
            )
        if np.any(Mn >= 1.0):
            raise NotImplementedError(
                f"{self._desc.capitalize()} {self.label!r} is axially "
                f"supersonic (max axial Mach {float(np.max(Mn)):.4g}); only an "
                "axially subsonic mean state is implemented."
            )
        if np.any(Msq >= 1.0):
            raise NotImplementedError(
                f"{self._desc.capitalize()} {self.label!r} has a supersonic "
                f"mean state (max Mach {float(np.sqrt(np.max(Msq))):.4g}); the "
                "supersonic branch of the wave parameter is not implemented."
            )

        self._ref = {
            "prim": self._span_bcast(
                np.stack(
                    (avg.rho_nd, avg.Vx_nd, avg.Vr_nd, avg.Vt_nd, avg.P_nd), axis=-1
                )
            ),
            "p2c": self._span_bcast(perturbation.primitive_to_chic(avg)),
            "c2p": self._span_bcast(perturbation.chic_to_primitive(avg)),
        }
        # The wave parameter magnitude, sqrt(1 - M^2). Both the axial and the
        # tangential Mach number enter it, but not the radial one: Saxer's
        # quasi-3D theory treats each span station as a two-dimensional cascade
        # (his Eq. 15).
        self._ref.update(self._calc_reference_extra(avg, Mn, Mt, np.sqrt(1.0 - Msq)))

    @abstractmethod
    def _calc_dchic(self, dchic, prim):
        """Return the change required in the incoming characteristics.

        Parameters
        ----------
        dchic : array
            Characteristic deviation of the face from the reference state,
            outgoing components as the interior march left them and incoming
            components as this patch last set them, shape ``(*shape, 5)``.
        prim : array
            The primitive face state ``dchic`` describes, so residuals are taken
            on the state about to be corrected rather than on whatever is
            currently stored in the block, shape ``(*shape, 5)``.

        Returns
        -------
        array
            Change in the characteristic variables, zero in the outgoing
            components, shape ``(*shape, 5)``. Applied under-relaxed by
            :attr:`sigma`.
        """

    @abstractmethod
    def _calc_reference_extra(self, avg, Mn, Mt, wave):
        """Return the reference-state entries specific to this condition.

        Parameters
        ----------
        avg : Block
            Pitchwise-mean state, one node per span station.
        Mn, Mt : array
            Axial and tangential Mach number of the mean state, per span
            station.
        wave : array
            Wave parameter magnitude :math:`\\sqrt{1 - M^2}`, per span station.

        Returns
        -------
        dict
            Entries to merge into the frozen reference state, each already
            broadcast over the patch shape by :meth:`_span_bcast`.
        """

    def _broadcast_target(self, name, value):
        """Check a prescribed value against the patch shape and make it a target array."""
        arr = np.asarray(value)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
        try:
            arr = np.broadcast_to(arr, self.block_view.shape)
        except ValueError:
            raise ValueError(
                f"{name} of shape {arr.shape} does not broadcast to patch "
                f"shape {self.shape}"
            ) from None
        return np.asfortranarray(arr.astype(np.float32))

    def _check_plane(self):
        """Validate the restrictions of the first implementation."""
        block = self.block
        x = self.block_view.x
        Lref = max(np.ptp(block.x), np.ptp(block.r))
        if np.ptp(x) > self._rtol_geom * Lref:
            raise ValueError(
                f"{self._desc.capitalize()} {self.label!r} must lie on a plane "
                f"of constant x (spread {float(np.ptp(x)):.4g} over reference "
                f"length {Lref:.4g}); canted planes are not implemented."
            )
        x_interior = block.x[self._get_offset_slice(1)].mean()
        if self._sign_interior * (x_interior - x.mean()) <= 0.0:
            side = "+x" if self._sign_interior > 0 else "-x"
            verb = "enters" if self._sign_interior > 0 else "leaves"
            raise NotImplementedError(
                f"{self._desc.capitalize()} {self.label!r} must have its "
                f"interior on the {side} side, so that flow {verb} along +x."
            )

    def _copy(self, c):
        for name in self._target_setters:
            val = getattr(self, name)
            setattr(c, name, None if val is None else np.copy(val))
        c.sigma = self.sigma
        # _hilbert and _ref both derive from the block geometry or solution, so
        # they are rebuilt on the new block rather than copied. The targets are
        # copied nondimensionalised, so the new block must share the reference
        # scales of the old one; every block of a grid does.

    def _pitch_mean(self, field):
        """Weighted pitchwise mean of a patch-shaped field, keeping dimensions."""
        return (field * self.weight_pitch).sum(axis=self.pitch_dim, keepdims=True)

    def _raise_unset(self):
        """Report which parts of the prescribed boundary state are still missing."""
        unset = {
            k: v for k, v in self._target_setters.items() if getattr(self, k) is None
        }
        raise ValueError(
            f"{self._desc.capitalize()} {self.label!r} is missing boundary "
            f"condition values {list(unset)}; call "
            f"{', '.join(dict.fromkeys(unset.values()))} first."
        )

    def _setup(self):
        super()._setup()
        # Prescribed boundary state, stored as the nondimensional patch arrays
        # apply() takes its residuals against. None until set.
        for name in self._target_setters:
            setattr(self, name, None)
        self._hilbert = None
        self._ref = None
        # Face state this patch last authored. The incoming characteristics are
        # carried from here rather than from the marched face; see apply().
        self._prim_prev = None
        # Under-relaxation of the characteristic correction, Giles Eq. 5.25,
        # needed for wellposedness. He suggests 1/N for N pitchwise nodes,
        # applied once per timestep; ember applies this once per Runge-Kutta
        # stage, so the effective rate per step is larger by the stage count.
        self.sigma = 0.05

    def _span_bcast(self, arr):
        """Reshape a span-indexed array to broadcast over the patch shape."""
        shape = [1, 1, 1] + list(arr.shape[1:])
        shape[self.span_dim] = arr.shape[0]
        return arr.reshape(shape)

    def _transform_pitch(self, field):
        """Apply the Hilbert matrix along the pitch axis of a patch-shaped field."""
        return np.moveaxis(
            np.tensordot(field, self._hilbert, axes=([self.pitch_dim], [1])),
            -1,
            self.pitch_dim,
        )

    def apply(self):
        r"""Impose the non-reflecting condition on the patch.

        Called once per Runge-Kutta stage. The face state is decomposed into
        characteristic deviations from the frozen reference state of
        :meth:`update_soln`, the outgoing characteristics are left exactly as the
        interior march deposited them, and :meth:`_calc_dchic` supplies the
        change in the incoming ones, applied under-relaxed by :attr:`sigma`.

        The interior march updates all five characteristics at the boundary
        node, but only the outgoing ones carry legitimate information from
        inside the domain. Giles discards the scheme's increments in the
        incoming characteristics outright (his Eq. 5.25 applies only the
        boundary-condition change), and so must this: keeping them and merely
        under-relaxing on top lets the interior drive the incoming
        characteristics, which is unstable, and worse the smaller
        :attr:`sigma` is. So the outgoing characteristics are taken from the
        marched face and the incoming ones from this patch's own previous
        output.

        Because the characteristic transform is linear with frozen
        coefficients, the update is exactly ``prim += sigma * c2p @ dchic`` with
        zero outgoing components, so no reflection is introduced by the
        reconstruction itself.
        """
        if any(getattr(self, name) is None for name in self._target_setters):
            self._raise_unset()
        if self._ref is None:
            self._calc_reference()

        b = self.block_view
        ref = self._ref

        prim_marched = np.stack((b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1)
        if self._prim_prev is None:
            self._prim_prev = prim_marched.copy()

        dchic = util.matvec(ref["p2c"], self._prim_prev - ref["prim"])
        dchic[..., self._idx_out] = util.matvec(ref["p2c"], prim_marched - ref["prim"])[
            ..., self._idx_out
        ]
        prim = ref["prim"] + util.matvec(ref["c2p"], dchic)

        prim_new = prim + self.sigma * util.matvec(
            ref["c2p"], self._calc_dchic(dchic, prim)
        )
        self._prim_prev = prim_new
        rho_nd, u_nd = b.fluid.set_P_rho(prim_new[..., 4], prim_new[..., 0])
        b.set_rho_u_Vxrt_nd(
            rho_nd, u_nd, prim_new[..., 1], prim_new[..., 2], prim_new[..., 3]
        )

    def attach_to_block(self, block):
        """Attach to a block, validate the boundary plane and build the transform."""
        super().attach_to_block(block)

        if self._block_ref is None:
            return

        self._check_plane()
        self._calc_hilbert()

    def update_soln(self):
        """Refresh the frozen reference state; call once per timestep.

        Re-derives the pitchwise-mean state and every Jacobian evaluated on it,
        which :meth:`apply` then holds fixed across the Runge-Kutta stages of
        the step.
        """
        self._calc_reference()
