r"""Non-reflecting subsonic inlet boundary condition for EMBER CFD.

:class:`NonReflectingInletPatch` prescribes stagnation enthalpy, entropy and the
two flow angles at an inflow face while letting outgoing acoustic waves leave
the domain, after the steady non-reflecting theory of :cite:t:`Giles1988` (his
Sections 5.3-5.4) extended to three dimensions by :cite:t:`Saxer1993`.

The condition works in characteristic variables. Of the five characteristics at
an axially subsonic inflow plane, four are incoming (entropy, two vorticity
waves, the downstream-running pressure wave) and one, the upstream-running
pressure wave, is outgoing. Each stage the outgoing characteristic is read from
the boundary node exactly as the interior scheme left it and is never
overwritten, so an acoustic wave reaching the inflow plane passes through it.
The four incoming characteristics are driven toward the prescribed inflow state
in two parts:

* the **pitchwise mean** of each incoming characteristic is set by requiring the
  mean stagnation enthalpy, entropy, yaw and pitch angles to take their
  prescribed values (Giles Eq. 5.13-5.15, Saxer Eq. 9);
* the **pitchwise harmonics** are set from the harmonics of the outgoing
  characteristic by the non-reflecting relation (Giles Eq. 5.17), except that
  entropy and stagnation enthalpy are additionally held uniform along the pitch
  (Giles Eq. 5.22-5.24). Giles adopts that last constraint because a
  straightforward implementation of the linear theory leaves second-order
  variations in entropy and stagnation enthalpy that would be comparable with
  the losses of a viscous calculation.

Unlike :class:`~ember.inlet.InletPatch` this patch shares no state or code with
the pressure-relaxation inlet, and the inflow state is prescribed in the
"natural" variables of the characteristic residuals -- stagnation enthalpy and
entropy rather than stagnation pressure and temperature -- so no thermodynamic
inversion happens inside the boundary condition.

See Also
--------
ember.patch.Patch : Base class for all patches
ember.inlet.InletPatch : Reflecting inlet with pressure relaxation
ember.perturbation.chic_to_bcond : Jacobian the characteristic solves are built on
"""

import numpy as np

from ember import perturbation, util
from ember.basepatch import RevolutionPatch


class NonReflectingInletPatch(RevolutionPatch):
    r"""Non-reflecting subsonic inflow boundary condition.

    Prescribes stagnation enthalpy :math:`h_0`, entropy :math:`s`, yaw angle
    :math:`\alpha` and pitch angle :math:`\beta` as pitchwise-mean quantities,
    while absorbing outgoing acoustic waves rather than reflecting them. All
    four must be set before :meth:`apply` is called, via :meth:`set_ho_s` or
    :meth:`set_Po_To` together with :meth:`set_Alpha` and :meth:`set_Beta`.
    Each setter converts and stores its target nondimensionally in
    :attr:`ho_nd`, :attr:`s_nd`, :attr:`tanAlpha` and :attr:`sinBeta`, the form
    :meth:`apply` takes its residuals against, so the patch must already be
    attached to a block whose fluid is set.

    Each Runge-Kutta stage :meth:`apply` forms the characteristic deviation of
    the face state from the frozen pitchwise-mean reference state, computes the
    change required in the four incoming characteristics, and applies it
    under-relaxed by :attr:`sigma`. The outgoing characteristic is carried
    through untouched. :meth:`update_soln` refreshes the reference state once
    per timestep, matching Giles' definition of the characteristic variables as
    perturbations about the time-level-:math:`n` average.

    The first cut is restricted to a constant-:math:`x` inflow plane with the
    flow entering in the :math:`+x` direction and an axially subsonic,
    absolutely subsonic mean state; each restriction is checked and raises.
    """

    _collection_name = "inlet_nonreflecting"

    # Relative tolerance for the geometric checks made at attach time.
    _rtol_geom = 1e-4

    # A mean-mode Jacobian is treated as singular when its determinant falls
    # this far below the Hadamard bound (the product of its row norms).
    _rtol_det = 1e-6

    def _calc_hilbert(self):
        r"""Build the pitchwise Hilbert transform matrix.

        The non-reflecting relation for the tangential vorticity characteristic
        is, per pitchwise Fourier mode :math:`m` (Giles Eq. 5.17),

        .. math::
            \hat{c}_t = -\frac{\beta + M_t}{1 + M_n}\hat{c}_\mathrm{up},
            \qquad \beta = i\,\mathrm{sign}(m)\sqrt{1 - M^2}.

        Only :math:`\mathrm{sign}(m)` depends on the mode, so splitting
        :math:`\beta` from :math:`M_t` separates a local term from a Hilbert
        transform along the pitch, and no Fourier transform need be taken at
        run time:

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
                f"Non-reflecting inlet patch {self.label!r} requires the same "
                "pitchwise node distribution at every span station."
            )

        w = self.weight_pitch.ravel().astype(np.float64)
        if abs(w.sum() - 1.0) > self._rtol_geom:
            raise ValueError(
                f"Non-reflecting inlet patch {self.label!r} must span a whole "
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
                f"Non-reflecting inlet patch {self.label!r} needs at least 3 "
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
        characteristic variables as perturbations about the average inflow at
        time level ``n``, so the reference state and every Jacobian evaluated on
        it are held fixed for all the Runge-Kutta stages of a step.
        """
        self.set_block_avg()
        avg = self.block_avg

        a_nd = avg.a_nd
        Mn = avg.Vx_nd / a_nd
        Mt = avg.Vt_nd / a_nd
        Msq = Mn**2 + Mt**2

        if np.any(avg.Vx_nd <= 0.0):
            raise ValueError(
                f"Backflow at non-reflecting inlet patch {self.label!r}: the "
                f"pitchwise-mean axial velocity is negative at "
                f"{int(np.count_nonzero(avg.Vx_nd <= 0.0))} of {avg.Vx_nd.size} "
                "span stations, so the characteristic split is invalid."
            )
        if np.any(Mn >= 1.0):
            raise NotImplementedError(
                f"Non-reflecting inlet patch {self.label!r} is axially "
                f"supersonic (max axial Mach {float(np.max(Mn)):.4g}); only "
                "subsonic axial inflow is implemented."
            )
        if np.any(Msq >= 1.0):
            raise NotImplementedError(
                f"Non-reflecting inlet patch {self.label!r} has a supersonic "
                f"mean state (max Mach {float(np.sqrt(np.max(Msq))):.4g}); the "
                "supersonic branch of the wave parameter is not implemented."
            )

        c2b = perturbation.chic_to_bcond(avg)
        # Rows [ho, s, tanAlpha, sinBeta] against the four incoming
        # characteristic columns [c_down, c_r, c_t, c_s]: the square system
        # whose solution zeroes the mean boundary condition residuals.
        jac_mean = np.ascontiguousarray(c2b[..., 0:4, 1:5])
        det = np.linalg.det(jac_mean)
        hadamard = np.prod(np.linalg.norm(jac_mean, axis=-1), axis=-1)
        if np.any(np.abs(det) < self._rtol_det * hadamard):
            raise ValueError(
                f"Non-reflecting inlet patch {self.label!r} has a singular mean "
                "characteristic Jacobian; the inflow state is degenerate "
                "(reversed or extreme swirl)."
            )

        # Local system: stagnation enthalpy and entropy against the entropy and
        # downstream-running pressure characteristics, the two left free once
        # the vorticity characteristics are fixed by the non-reflecting theory.
        # Columns 1 and 4 of a length-5 axis are c_down and c_s.
        jac_local = np.ascontiguousarray(c2b[..., 0:2, 1::3])

        self._ref = {
            "prim": self._span_bcast(
                np.stack(
                    (avg.rho_nd, avg.Vx_nd, avg.Vr_nd, avg.Vt_nd, avg.P_nd), axis=-1
                )
            ),
            "p2c": self._span_bcast(perturbation.primitive_to_chic(avg)),
            "c2p": self._span_bcast(perturbation.chic_to_primitive(avg)),
            "inv_mean": self._span_bcast(util.inv(jac_mean)),
            "inv_local": self._span_bcast(util.inv(jac_local)),
            "couple_r": self._span_bcast(np.ascontiguousarray(c2b[..., 0:2, 2])),
            "couple_t": self._span_bcast(np.ascontiguousarray(c2b[..., 0:2, 3])),
            "coef_local": self._span_bcast(-Mt / (1.0 + Mn)),
            "coef_hilbert": self._span_bcast(np.sqrt(1.0 - Msq) / (1.0 + Mn)),
        }

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
                f"Non-reflecting inlet patch {self.label!r} must lie on a plane "
                f"of constant x (spread {float(np.ptp(x)):.4g} over reference "
                f"length {Lref:.4g}); canted inflow planes are not implemented."
            )
        x_interior = block.x[self._get_offset_slice(1)].mean()
        if x_interior <= x.mean():
            raise NotImplementedError(
                f"Non-reflecting inlet patch {self.label!r} must have its "
                "interior on the +x side, so that flow enters along +x."
            )

    def _copy(self, c):
        for name in ("ho_nd", "s_nd", "tanAlpha", "sinBeta"):
            val = getattr(self, name)
            setattr(c, name, None if val is None else np.copy(val))
        c.sigma = self.sigma
        # _hilbert and _ref both derive from the block geometry or solution, so
        # they are rebuilt on the new block rather than copied. The targets are
        # copied nondimensionalised, so the new block must share the reference
        # scales of the old one; every block of a grid does.

    def _bcond_from_prim(self, prim):
        """Boundary condition quantities (ho, s, tanAlpha, sinBeta) of a primitive state.

        Evaluated without writing to the block, so :meth:`apply` can take
        residuals on the state it is about to correct rather than on whatever
        is currently stored.
        """
        fluid = self.block_view.fluid
        rho_nd, u_nd = fluid.set_P_rho(prim[..., 4], prim[..., 0])
        Vx, Vr, Vt = prim[..., 1], prim[..., 2], prim[..., 3]
        Vm = np.sqrt(Vx**2 + Vr**2)
        ho_nd = fluid.get_h(rho_nd, u_nd) + 0.5 * (Vx**2 + Vr**2 + Vt**2)
        return ho_nd, fluid.get_s(rho_nd, u_nd), Vt / Vm, Vr / Vm

    def _pitch_mean(self, field):
        """Weighted pitchwise mean of a patch-shaped field, keeping dimensions."""
        return (field * self.weight_pitch).sum(axis=self.pitch_dim, keepdims=True)

    def _raise_unset(self):
        """Report which parts of the prescribed inflow state are still missing."""
        needed = {
            "ho_nd": "set_ho_s or set_Po_To",
            "s_nd": "set_ho_s or set_Po_To",
            "tanAlpha": "set_Alpha",
            "sinBeta": "set_Beta",
        }
        unset = {k: v for k, v in needed.items() if getattr(self, k) is None}
        raise ValueError(
            f"Non-reflecting inlet patch {self.label!r} is missing boundary "
            f"condition values {list(unset)}; call "
            f"{', '.join(dict.fromkeys(unset.values()))} first."
        )

    def _setup(self):
        super()._setup()
        # Prescribed inflow state, stored as the nondimensional patch arrays
        # apply() takes its residuals against. None until set.
        self.ho_nd = None
        self.s_nd = None
        self.tanAlpha = None
        self.sinBeta = None
        self._hilbert = None
        self._ref = None
        # Face state this patch last authored. The incoming characteristics are
        # carried from here rather than from the marched face; see apply().
        self._prim_prev = None
        # Under-relaxation of the characteristic correction, Giles Eq. 5.25,
        # needed for wellposedness. He suggests 1/N for N pitchwise nodes,
        # applied once per timestep; ember applies this once per Runge-Kutta
        # stage, so the effective rate per step is larger by the stage count.
        self.sigma = 0.1

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

    def set_Alpha(self, Alpha):
        r"""Prescribe the inflow yaw angle.

        Parameters
        ----------
        Alpha : float or array
            Prescribed inflow yaw angle :math:`\alpha` [deg], measured from the
            meridional plane; must satisfy :math:`|\alpha| < 90`. A scalar or an
            array that broadcasts to :attr:`~ember.basepatch.Patch.shape`.
        """
        if not (np.abs(np.asarray(Alpha)) < 90.0).all():
            raise ValueError("Alpha must be within +/-90 degrees exclusive")
        self.tanAlpha = self._broadcast_target(
            "Alpha", np.tan(np.radians(np.asarray(Alpha, dtype=np.float32)))
        )

    def set_Beta(self, Beta):
        r"""Prescribe the inflow pitch angle.

        Parameters
        ----------
        Beta : float or array
            Prescribed inflow pitch angle :math:`\beta` [deg]; must satisfy
            :math:`|\beta| \leq 90`. A scalar or an array that broadcasts to
            :attr:`~ember.basepatch.Patch.shape`.
        """
        if not (np.abs(np.asarray(Beta)) <= 90.0).all():
            raise ValueError("Beta must be within +/-90 degrees inclusive")
        self.sinBeta = self._broadcast_target(
            "Beta", np.sin(np.radians(np.asarray(Beta, dtype=np.float32)))
        )

    def set_ho_s(self, ho, s):
        r"""Prescribe the inflow stagnation enthalpy and entropy.

        Both are measured from the fluid datum state where :math:`u = s = 0` at
        :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`, the same convention as
        :py:attr:`~ember.block.Block.ho` and :py:attr:`~ember.block.Block.s`;
        only differences are physically meaningful, so these are not
        :math:`c_p T_0` and :math:`c_p \log(\ldots)`. Use :meth:`set_Po_To` to
        prescribe a stagnation state instead.

        Parameters
        ----------
        ho : float or array
            Prescribed stagnation enthalpy :math:`h_0` [J/kg]. A scalar or an
            array that broadcasts to :attr:`~ember.basepatch.Patch.shape`.
        s : float or array
            Prescribed entropy :math:`s` [J/kg/K].
        """
        fluid = self.block.fluid
        self.ho_nd = self._broadcast_target("ho", np.asarray(ho) / fluid.u_ref)
        self.s_nd = self._broadcast_target("s", np.asarray(s) / fluid.Rgas_ref)

    def set_Po_To(self, Po, To):
        r"""Prescribe the inflow stagnation pressure and temperature.

        Converted here, once, to the stagnation enthalpy and entropy of
        :meth:`set_ho_s` using the fluid of the block this patch is attached to;
        only the result is stored, so a later change of fluid does not
        re-convert.

        Parameters
        ----------
        Po : float or array
            Prescribed stagnation pressure :math:`p_0` [Pa]; must be positive.
            A scalar or an array that broadcasts to
            :attr:`~ember.basepatch.Patch.shape`.
        To : float or array
            Prescribed stagnation temperature :math:`T_0` [K]; must be positive.
        """
        fluid = self.block.fluid

        for name, val in (("Po", Po), ("To", To)):
            arr = np.asarray(val)
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite")
            if not (arr > 0.0).all():
                raise ValueError(f"{name} must be positive")

        # get_h and get_s return nondimensional values already, so the targets
        # are formed without a round trip through dimensional ho and s.
        rhoo_nd, uo_nd = fluid.set_P_T(
            np.asarray(Po) / fluid.P_ref, np.asarray(To) / fluid.T_ref
        )
        self.ho_nd = self._broadcast_target("Po and To", fluid.get_h(rhoo_nd, uo_nd))
        self.s_nd = self._broadcast_target("Po and To", fluid.get_s(rhoo_nd, uo_nd))

    def apply(self):
        r"""Impose the non-reflecting inflow condition on the patch.

        Called once per Runge-Kutta stage. The face state is decomposed into
        characteristic deviations from the frozen reference state of
        :meth:`update_soln`; the outgoing (upstream-running pressure)
        characteristic is left exactly as the interior march deposited it, and
        the change in the four incoming characteristics is assembled from three
        contributions:

        #. a pitchwise-mean change that zeroes the mean residuals in
           :math:`(h_0, s, \tan\alpha, \sin\beta)` in one Newton step against
           ``ember.perturbation.chic_to_bcond`` (Giles Eq. 5.13-5.15);
        #. harmonic changes setting the tangential vorticity characteristic from
           the outgoing characteristic by the non-reflecting relation, and the
           radial vorticity harmonics to zero (Giles Eq. 5.17, Saxer Eq. 56);
        #. harmonic changes in the entropy and downstream-running pressure
           characteristics that hold entropy and stagnation enthalpy uniform
           along the pitch (Giles Eq. 5.22-5.24).

        Their sum is applied under-relaxed by :attr:`sigma`. Because the
        characteristic transform is linear with frozen coefficients, the update
        is exactly ``prim += sigma * c2p @ dchic`` with a zero outgoing
        component, so no reflection is introduced by the reconstruction itself.
        """
        if (
            self.ho_nd is None
            or self.s_nd is None
            or self.tanAlpha is None
            or self.sinBeta is None
        ):
            self._raise_unset()
        if self._ref is None:
            self._calc_reference()

        b = self.block_view
        ref = self._ref

        prim_marched = np.stack((b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1)
        if self._prim_prev is None:
            self._prim_prev = prim_marched.copy()

        # The interior march updates all five characteristics at the boundary
        # node, but only the outgoing one carries legitimate information from
        # inside the domain. Giles discards the scheme's increments in the four
        # incoming characteristics outright (his Eq. 5.25 applies only the
        # boundary-condition change), and so must this: keeping them and merely
        # under-relaxing on top lets the interior drive the incoming
        # characteristics, which is unstable, and worse the smaller sigma is.
        # So take the outgoing characteristic from the marched face and the
        # incoming ones from this patch's own previous output.
        dchic = util.matvec(ref["p2c"], self._prim_prev - ref["prim"])
        dchic[..., 0] = util.matvec(ref["p2c"], prim_marched - ref["prim"])[..., 0]
        prim = ref["prim"] + util.matvec(ref["c2p"], dchic)

        ho_nd, s_nd, tanAlpha, sinBeta = self._bcond_from_prim(prim)

        # Mean mode: one Newton step on the four prescribed quantities. The
        # residual is evaluated on the state about to be corrected, only the
        # Jacobian is frozen -- a modified Newton step. Reading the residual
        # from the frozen reference too would leave it up to n_stage stages
        # stale, so repeated stages would re-apply one correction rather than
        # converge on it.
        resid_mean = np.stack(
            (
                self._pitch_mean(ho_nd - self.ho_nd),
                self._pitch_mean(s_nd - self.s_nd),
                self._pitch_mean(tanAlpha - self.tanAlpha),
                self._pitch_mean(sinBeta - self.sinBeta),
            ),
            axis=-1,
        )
        dchic_mean = -util.matvec(ref["inv_mean"], resid_mean)

        # Harmonics: the non-reflecting relation for the tangential vorticity
        # characteristic, and no radial vorticity harmonics.
        c_up = dchic[..., 0]
        c_up_harm = c_up - self._pitch_mean(c_up)
        c_t_ideal = ref["coef_local"] * c_up_harm + ref[
            "coef_hilbert"
        ] * self._transform_pitch(c_up_harm)
        c_t = dchic[..., 3]
        c_r = dchic[..., 2]
        dchic_t = c_t_ideal - (c_t - self._pitch_mean(c_t))
        dchic_r = -(c_r - self._pitch_mean(c_r))

        # Harmonics of entropy and stagnation enthalpy driven to zero, given the
        # vorticity changes just fixed.
        resid_local = np.stack(
            (ho_nd - self._pitch_mean(ho_nd), s_nd - self._pitch_mean(s_nd)),
            axis=-1,
        )
        resid_local = (
            resid_local
            + ref["couple_t"] * dchic_t[..., np.newaxis]
            + ref["couple_r"] * dchic_r[..., np.newaxis]
        )
        dchic_local = -util.matvec(ref["inv_local"], resid_local)

        dchic_new = np.zeros_like(dchic)
        dchic_new[..., 1] = dchic_mean[..., 0] + dchic_local[..., 0]
        dchic_new[..., 2] = dchic_mean[..., 1] + dchic_r
        dchic_new[..., 3] = dchic_mean[..., 2] + dchic_t
        dchic_new[..., 4] = dchic_mean[..., 3] + dchic_local[..., 1]

        prim_new = prim + self.sigma * util.matvec(ref["c2p"], dchic_new)
        self._prim_prev = prim_new
        rho_nd, u_nd = b.fluid.set_P_rho(prim_new[..., 4], prim_new[..., 0])
        b.set_rho_u_Vxrt_nd(
            rho_nd, u_nd, prim_new[..., 1], prim_new[..., 2], prim_new[..., 3]
        )

    def attach_to_block(self, block):
        """Attach to a block, validate the inflow plane and build the transform."""
        super().attach_to_block(block)

        if self._block_ref is None:
            return

        self._check_plane()
        self._calc_hilbert()

    def update_soln(self):
        """Refresh the frozen reference state; call once per timestep.

        Re-derives the pitchwise-mean inflow state and every Jacobian evaluated
        on it, which :meth:`apply` then holds fixed across the Runge-Kutta
        stages of the step.
        """
        self._calc_reference()
