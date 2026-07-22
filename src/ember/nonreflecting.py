r"""Shared machinery for the steady non-reflecting boundary conditions of EMBER CFD.

:class:`NonReflectingPatch` holds everything the non-reflecting inflow and
outflow conditions of :cite:t:`Giles1988` (his Chapter 5), as extended to three
dimensions by :cite:t:`Saxer1993`, have in common: the pitchwise Hilbert
transform their harmonic relations are written with, the frozen pitchwise-mean
reference state the characteristic variables are perturbations about, the split
of the boundary node into the characteristics the interior march owns and those
the boundary condition owns, and the prescribed boundary state itself.

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

Whatever a condition prescribes, it prescribes as one nondimensional five-vector
per span station, :attr:`~NonReflectingPatch._target`, in the space
:attr:`~NonReflectingPatch._chic_to_target` maps characteristics into. Rows 0, 1
and 4 are stagnation enthalpy, entropy and static pressure in every such space;
only rows 2 and 3 distinguish them, the mix variables
:math:`(V_r, V_\theta)` of :func:`~ember.perturbation.chic_to_mix` from the
angles :math:`(\tan\alpha, \sin\beta)` of
:func:`~ember.perturbation.chic_to_bcond`. One span station rather than one node
loses nothing: every target is read only through
:meth:`~NonReflectingPatch._pitch_mean` of its own residual, which is linear, so
the pitch mean of a prescribed profile is all that was ever imposed.

Which rows a condition prescribes follows from how many characteristics are
incoming, and the pair is written down as a *split*: four incoming
characteristics take rows 0-3, the quantities an inflow plane sets; one takes
row 4, the static pressure an outflow plane sets.

See Also
--------
ember.basepatch.RevolutionPatch : Base class providing the pitchwise geometry
ember.inlet_nonreflecting.NonReflectingInletPatch : Subsonic inflow
ember.outlet_nonreflecting.NonReflectingOutletPatch : Subsonic outflow
ember.perturbation.chic_to_mix : Jacobian the characteristic solves are built on
"""

from abc import abstractmethod

import numpy as np

from ember import perturbation, util
from ember.basepatch import RevolutionPatch


class _TargetRow:
    """Read-only view of one row of a patch's prescribed target vector.

    A descriptor rather than a plain attribute, so that the named rows stay
    views on :attr:`~NonReflectingPatch._target` with nothing to re-link when a
    patch is copied or unpickled, and so that a name the patch's target space
    does not carry raises rather than quietly returning whatever that row holds:
    an inflow condition working in angles has no ``Vr_nd``, and one working in
    mix variables has no ``tanAlpha``.

    Resolution is by name against :attr:`~NonReflectingPatch._target_names` of
    the instance, not by a fixed index, because the row order is a property of
    the target space and the classes do not share one.
    """

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        names = obj._target_names
        if self._name not in names:
            raise AttributeError(
                f"{type(obj).__name__} prescribes {list(names)}, so it has no "
                f"{self._name!r}"
            )
        obj._check_attached()
        return obj._target[..., names.index(self._name)]


class NonReflectingPatch(RevolutionPatch):
    r"""Base class for the steady non-reflecting inflow and outflow conditions.

    Subclasses supply the class attributes below and two hooks. The attributes
    are :attr:`_desc`, a description used in error messages; :attr:`_split_fwd`
    and :attr:`_split_rev`, the characteristic/target splits of a forward and a
    reversed mean state; :attr:`_sign_interior`, which side of the face the
    interior lies on; :attr:`_chic_to_target` and :attr:`_target_names`, naming
    the space the prescribed target lives in; and :attr:`_target_setters`,
    mapping each prescribed row to the setter that fills it. The hooks are
    :meth:`_calc_reference_extra`, which adds the coefficients that condition's
    harmonic relations need to the frozen reference state, and
    :meth:`_calc_dchic`, which returns the change required in the incoming
    characteristics.

    :meth:`apply` is called once per Runge-Kutta stage and :meth:`update_soln`
    once per timestep, the latter refreshing the reference state to match Giles'
    definition of the characteristic variables as perturbations about the
    time-level-:math:`n` average.

    Both conditions are restricted to a constant-:math:`x` plane with the flow
    running in the :math:`+x` direction and an axially subsonic, absolutely
    subsonic mean state; each restriction is checked and raises. A span station
    whose mean flow has reversed raises too, since the split above is no longer
    the right one there -- unless the subclass declares a :attr:`_split_rev` and
    says through :meth:`_calc_reversed` that this station is using it.
    """

    # A mean-mode Jacobian is treated as singular when its determinant falls
    # this far below the Hadamard bound (the product of its row norms).
    _rtol_det = 1e-6

    # Relative tolerance for the geometric checks made at attach time.
    _rtol_geom = 1e-4

    # Description of the patch used in error messages; lower case, so it can be
    # capitalised where it starts a sentence.
    _desc = None

    # Jacobian from characteristic to target variables. Its last row must be
    # the static pressure and its first four the quantities an inflow
    # prescribes; the mean-mode solves are written against those row positions
    # and not against any particular set. Mix variables here, the set that
    # stays conditioned through zero axial velocity; a physical inlet overrides
    # to the angles, which suit it better and nothing else.
    _chic_to_target = staticmethod(perturbation.chic_to_mix)

    # Names of the rows of _target, in order, and the attribute names the
    # _TargetRow descriptors below publish them under.
    _target_names = ("ho_nd", "s_nd", "Vr_nd", "Vt_nd", "P_nd")

    # Characteristic/target split of a forward mean state, as (incoming
    # characteristic columns, the target rows prescribed against them). The
    # complement of the columns is what the interior march owns: [0] at an
    # inflow plane, [1, 2, 3, 4] at an outflow plane.
    _split_fwd = None

    # The same under a reversed mean state, or None for a condition that cannot
    # carry one and raises on it instead.
    _split_rev = None

    # Rows filled from the pitchwise mean of the face when nothing has
    # prescribed them; see _seed_target.
    _target_seeded = ()

    # +1 if the interior lies on the +x side of the face, -1 if on the -x side.
    _sign_interior = None

    # Prescribed target row -> the setter, or setters, that fill it. Rows absent
    # from this mapping are not required of the user, either because they are
    # seeded or because something else fills them.
    _target_setters = None

    ho_nd = _TargetRow()
    s_nd = _TargetRow()
    Vr_nd = _TargetRow()
    Vt_nd = _TargetRow()
    tanAlpha = _TargetRow()
    sinBeta = _TargetRow()
    P_nd = _TargetRow()

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

    def _calc_dchic_mean(self, target, split, inv):
        """One modified Newton step on the prescribed pitchwise-mean quantities.

        The residual is evaluated on the state about to be corrected and only
        the Jacobian is frozen. Reading the residual from the frozen reference
        too would leave it up to ``n_stage`` stages stale, so repeated stages
        would re-apply one correction rather than converge on it.

        Parameters
        ----------
        target : tuple of array
            The five target-space quantities of the face state, as
            :meth:`_target_from_prim` returns them.
        split : tuple
            ``(incoming characteristic columns, prescribed target rows)``. The
            two are the same length, so the system is square.
        inv : array
            Inverse of that system's Jacobian, from :meth:`_calc_inv_jac`.

        Returns
        -------
        array
            Change in each incoming characteristic, in the column order of
            ``split``, shape ``(*span_shape, len(cols))``.
        """
        rows = split[1]
        resid = np.stack(
            [self._pitch_mean(target[row] - self._target[..., row]) for row in rows],
            axis=-1,
        )
        return -util.matvec(inv, resid)

    def _calc_inv_jac(self, c2t, split, where=""):
        """Invert the mean-mode Jacobian of one split, checking it is not singular.

        Parameters
        ----------
        c2t : array
            Characteristic-to-target Jacobian on the mean state, shape
            ``(nspan, 5, 5)``.
        split : tuple
            ``(incoming characteristic columns, prescribed target rows)``.
        where : str, optional
            Clause appended to the error message, naming the stations the
            system belongs to.

        Returns
        -------
        array
            Inverse, broadcast over the patch shape by :meth:`_span_bcast`.
        """
        cols, rows = split
        jac = np.ascontiguousarray(c2t[..., rows, :][..., cols])
        det = np.linalg.det(jac)
        hadamard = np.prod(np.linalg.norm(jac, axis=-1), axis=-1)
        if np.any(np.abs(det) < self._rtol_det * hadamard):
            raise ValueError(
                f"{self._desc.capitalize()} {self.label!r} has a singular mean "
                f"characteristic Jacobian{where}; the prescribed state is "
                "degenerate (reversed or extreme swirl)."
            )
        return self._span_bcast(util.inv(jac))

    def _calc_mask_out(self):
        """Boolean mask of the characteristic components the interior march owns.

        The complement of the incoming columns of :attr:`_split_fwd`, with any
        station :meth:`_calc_reversed` flagged carrying the complement of
        :attr:`_split_rev` instead.

        Returns
        -------
        array
            Boolean, broadcastable against ``(*shape, 5)``. A bare length-5
            mask while no station is reversed; one entry per span station once
            one is.
        """
        mask = self._mask_from_split(self._split_fwd)
        if self._split_rev is None or not self._reversed.any():
            return mask
        return np.where(
            self._span_bcast(self._reversed)[..., np.newaxis],
            self._mask_from_split(self._split_rev),
            mask,
        )

    def _calc_override(self, prim):
        """Face state to write, given the state the characteristic solve produced.

        A hook for a condition that has to depart from its own linear theory at
        some nodes, the identity here. Called after :attr:`_prim_prev` has been
        stored, so whatever it returns reaches the block without displacing the
        state the characteristic solve carries into the next stage.

        Parameters
        ----------
        prim : array
            Primitive face state the characteristic solve produced, shape
            ``(*shape, 5)``.

        Returns
        -------
        array
            Primitive face state to write, same shape.
        """
        return prim

    def _calc_reference(self):
        """Freeze the pitchwise-mean state and everything derived from it.

        Called once per timestep from :meth:`update_soln`. Giles defines the
        characteristic variables as perturbations about the average flow at time
        level ``n``, so the reference state and every Jacobian evaluated on it
        are held fixed for all the Runge-Kutta stages of a step.
        """
        self.set_block_avg()
        avg = self.block_avg

        # Fill any target row nothing has prescribed, before the first solve
        # reads it. A no-op after the first call: the seed is frozen, not
        # re-derived each step; see _seed_target.
        self._seed_target()

        a_nd = avg.a_nd
        Mn = avg.Vx_nd / a_nd
        Mt = avg.Vt_nd / a_nd
        Msq = Mn**2 + Mt**2

        # Which span stations, if any, this condition treats as reversed.
        # Frozen for the step alongside everything else here, so the
        # characteristic split cannot change between Runge-Kutta stages, and
        # settled before the guard below so a handled station does not raise.
        self._reversed = self._calc_reversed(avg)
        self._mask_out = self._calc_mask_out()

        unhandled = (avg.Vx_nd <= 0.0) & ~self._reversed
        if np.any(unhandled):
            raise ValueError(
                f"Backflow at {self._desc} {self.label!r}: the pitchwise-mean "
                f"axial velocity is negative at "
                f"{int(np.count_nonzero(unhandled))} of {avg.Vx_nd.size} "
                "span stations, so the characteristic split is invalid."
            )
        # Tested on the magnitude, so a station running backwards fast enough
        # to be axially supersonic is caught too: there the upstream-running
        # pressure wave turns outgoing and even the reversed split is wrong.
        if np.any(np.abs(Mn) >= 1.0):
            raise NotImplementedError(
                f"{self._desc.capitalize()} {self.label!r} is axially "
                f"supersonic (max axial Mach {float(np.max(np.abs(Mn))):.4g}); "
                "only an axially subsonic mean state is implemented."
            )
        if np.any(Msq >= 1.0):
            raise NotImplementedError(
                f"{self._desc.capitalize()} {self.label!r} has a supersonic "
                f"mean state (max Mach {float(np.sqrt(np.max(Msq))):.4g}); the "
                "supersonic branch of the wave parameter is not implemented."
            )

        c2t = self._chic_to_target(avg)
        self._ref = {
            "prim": self._span_bcast(
                np.stack(
                    (avg.rho_nd, avg.Vx_nd, avg.Vr_nd, avg.Vt_nd, avg.P_nd), axis=-1
                )
            ),
            "p2c": self._span_bcast(perturbation.primitive_to_chic(avg)),
            "c2p": self._span_bcast(perturbation.chic_to_primitive(avg)),
            "inv_fwd": self._calc_inv_jac(c2t, self._split_fwd),
        }
        if self._split_rev is not None:
            # Built at every station, reversed or not. Every entry of the
            # reversed system stays finite and its determinant vanishes only at
            # an axial Mach number the checks above have already excluded, so
            # there is nothing to gain by building it conditionally and a
            # branch to lose.
            self._ref["inv_rev"] = self._calc_inv_jac(
                c2t, self._split_rev, " at a reversed span station"
            )
        # The wave parameter magnitude, sqrt(1 - M^2). Both the axial and the
        # tangential Mach number enter it, but not the radial one: Saxer's
        # quasi-3D theory treats each span station as a two-dimensional cascade
        # (his Eq. 15).
        self._ref.update(
            self._calc_reference_extra(avg, c2t, Mn, Mt, np.sqrt(1.0 - Msq))
        )

    def _calc_reversed(self, avg):
        """Span stations whose mean flow runs backwards through the face.

        None of them here, so a reversed mean state raises as ill-posed. A
        condition that can carry one declares a :attr:`_split_rev` and returns
        those stations instead, and owns whatever :meth:`_calc_dchic` does
        there.

        Parameters
        ----------
        avg : Block
            Pitchwise-mean state, one node per span station.

        Returns
        -------
        array
            Boolean, shape ``(nspan,)``.
        """
        return np.zeros(avg.shape, dtype=bool)

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
    def _calc_reference_extra(self, avg, c2t, Mn, Mt, wave):
        """Return the reference-state entries specific to this condition.

        The mean-mode Jacobians are not among them: the base class builds those
        from :attr:`_split_fwd` and :attr:`_split_rev`. What is left is the
        coefficients of this condition's harmonic relations.

        Parameters
        ----------
        avg : Block
            Pitchwise-mean state, one node per span station.
        c2t : array
            Characteristic-to-target Jacobian on that state, shape
            ``(nspan, 5, 5)``, passed rather than recomputed.
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
        c._target = None if self._target is None else np.copy(self._target)
        c._target_set = self._target_set.copy()
        c.sigma = self.sigma
        # _hilbert and _ref both derive from the block geometry or solution, so
        # they are rebuilt on the new block rather than copied. The target is
        # copied nondimensionalised, so the new block must share the reference
        # scales of the old one; every block of a grid does.

    def _ho_s_from_prim(self, prim):
        """Stagnation enthalpy and entropy of a primitive state.

        Evaluated without writing to the block, so a residual is taken on the
        state about to be corrected rather than on whatever is currently
        stored. Rows 0 and 1 of every target space these conditions use.
        """
        fluid = self.block_view.fluid
        rho_nd, u_nd = fluid.set_P_rho(prim[..., 4], prim[..., 0])
        Vx, Vr, Vt = prim[..., 1], prim[..., 2], prim[..., 3]
        ho_nd = fluid.get_h(rho_nd, u_nd) + 0.5 * (Vx**2 + Vr**2 + Vt**2)
        return ho_nd, fluid.get_s(rho_nd, u_nd)

    @staticmethod
    def _mask_from_split(split):
        """Length-5 boolean mask of the characteristics a split leaves outgoing."""
        mask = np.ones(5, dtype=bool)
        mask[list(split[0])] = False
        return mask

    def _pitch_mean(self, field):
        """Weighted pitchwise mean of a patch-shaped field, keeping dimensions."""
        return (field * self.weight_pitch).sum(axis=self.pitch_dim, keepdims=True)

    def _raise_unset(self):
        """Report which parts of the prescribed boundary state are still missing."""
        unset = {
            row: setter
            for row, setter in self._target_setters.items()
            if not self._target_set[row]
        }
        raise ValueError(
            f"{self._desc.capitalize()} {self.label!r} is missing boundary "
            f"condition values {[self._target_names[row] for row in unset]}; "
            f"call {', '.join(dict.fromkeys(unset.values()))} first."
        )

    def _seed_target(self):
        """Fill any seeded target row nothing has prescribed, once.

        The rows of :attr:`_target_seeded` are taken from the pitchwise mean of
        the face as it stands the first time a solve needs them -- the initial
        condition, in a run that starts from one -- and frozen there. Freezing
        matters: a row re-derived from the face every step would drive its
        residual to zero and leave the quantity with no condition on it at all.

        Read off :attr:`~ember.basepatch.Patch.block_view` and pitch-averaged
        here rather than taken from
        :attr:`~ember.basepatch.RevolutionPatch.block_avg`, which a mixing
        exchange overwrites with the symmetrised cross-plane average, or from
        ``block_view.mean()``, whose derived properties read as zero before the
        conserved cache is primed.
        """
        rows = [row for row in self._target_seeded if not self._target_set[row]]
        if not rows:
            return
        b = self.block_view
        target = self._target_from_prim(
            np.stack((b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1)
        )
        for row in rows:
            self._target[..., row] = self._pitch_mean(target[row])
            self._target_set[row] = True

    def _set_target_row(self, row, name, value):
        """Check a prescribed value against the patch shape and store it in a target row.

        The value is pitch-averaged on the way in. That is not an
        approximation: every target is read only through the pitch mean of its
        own residual, so the mean of a prescribed profile is all that was ever
        imposed.
        """
        arr = np.asarray(value)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
        try:
            bcast = np.broadcast_to(arr, self.block_view.shape)
        except ValueError:
            raise ValueError(
                f"{name} of shape {arr.shape} does not broadcast to patch "
                f"shape {self.shape}"
            ) from None
        self._target[..., row] = self._pitch_mean(bcast)
        self._target_set[row] = True

    def _setup(self):
        super()._setup()
        # Prescribed boundary state, one nondimensional five-vector per span
        # station in the space _chic_to_target maps into, allocated on attach;
        # and which of its rows have been filled.
        self._target = None
        self._target_set = np.zeros(5, dtype=bool)
        self._hilbert = None
        self._ref = None
        # Characteristic split and the reversed span stations behind it, both
        # rebuilt every timestep by _calc_reference before anything reads them.
        self._mask_out = None
        self._reversed = None
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

    def _target_from_prim(self, prim):
        r"""The five target-space quantities of a primitive state.

        Mix variables :math:`[h_0, s, V_r, V_\theta, p]` here, matching the
        default :attr:`_chic_to_target`. A condition prescribing a different
        set overrides this and that together; everything else is written
        against the two of them rather than against any particular set.
        """
        ho_nd, s_nd = self._ho_s_from_prim(prim)
        return ho_nd, s_nd, prim[..., 2], prim[..., 3], prim[..., 4]

    def _target_shape(self):
        """Shape of the stored target: one span-indexed vector of five."""
        shape = [1, 1, 1]
        shape[self.span_dim] = self._block_view.shape[self.span_dim]
        return (*shape, 5)

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

        The result is stored as the state to carry into the next stage before
        :meth:`_calc_override` is given the chance to change what actually
        reaches the block, so a condition that has to depart from its own
        linear theory somewhere does not thereby corrupt the characteristic
        state it is still solving on.
        """
        if not self._target_set[list(self._target_setters)].all():
            self._raise_unset()
        if self._ref is None:
            self._calc_reference()

        b = self.block_view
        ref = self._ref

        prim_marched = np.stack((b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1)
        if self._prim_prev is None:
            self._prim_prev = prim_marched.copy()

        dchic_prev = util.matvec(ref["p2c"], self._prim_prev - ref["prim"])
        dchic_marched = util.matvec(ref["p2c"], prim_marched - ref["prim"])
        dchic = np.where(self._mask_out, dchic_marched, dchic_prev)
        prim = ref["prim"] + util.matvec(ref["c2p"], dchic)

        prim_new = prim + self.sigma * util.matvec(
            ref["c2p"], self._calc_dchic(dchic, prim)
        )
        self._prim_prev = prim_new
        prim_write = self._calc_override(prim_new)
        rho_nd, u_nd = b.fluid.set_P_rho(prim_write[..., 4], prim_write[..., 0])
        b.set_rho_u_Vxrt_nd(
            rho_nd, u_nd, prim_write[..., 1], prim_write[..., 2], prim_write[..., 3]
        )

    def attach_to_block(self, block):
        """Attach to a block, validate the boundary plane and build the transform.

        Safe to call repeatedly; a target of the right shape survives
        re-attachment, and one of the wrong shape is dropped along with the
        record of what had been set, so it is re-prescribed or re-seeded rather
        than silently misread.
        """
        super().attach_to_block(block)

        if self._block_ref is None:
            return

        self._check_plane()
        self._calc_hilbert()

        shape = self._target_shape()
        if self._target is None or self._target.shape != shape:
            self._target = util.zeros(shape)
            self._target_set = np.zeros(5, dtype=bool)

    def update_soln(self):
        """Refresh the frozen reference state; call once per timestep.

        Re-derives the pitchwise-mean state and every Jacobian evaluated on it,
        which :meth:`apply` then holds fixed across the Runge-Kutta stages of
        the step.
        """
        self._calc_reference()
