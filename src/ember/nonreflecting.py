r"""Shared machinery for the steady non-reflecting boundary conditions of EMBER CFD.

:class:`NonReflectingPatch` implements the steady non-reflecting inflow and
outflow conditions of :cite:t:`Giles1988` (his Chapter 5), as extended to three
dimensions by :cite:t:`Saxer1993`, as one condition. Its subclasses supply a
side of the face, a set of variables to prescribe and the setters that fill
them; the characteristic treatment itself lives here.

Of the five characteristics at an axially subsonic boundary, four propagate
downstream (entropy, both vorticity waves, the downstream-running pressure wave)
and one, the upstream-running pressure wave, propagates upstream. A
characteristic is *outgoing* -- owned by the interior march, read from the
boundary node exactly as the scheme left it and never overwritten, so a wave
reaching the boundary passes through -- when its wave speed carries it out of
the domain, :math:`\lambda\,n_x < 0` for the inward face normal
:attr:`~NonReflectingPatch._sign_interior`. The rest are *incoming*: discarded
rather than taken from the march, and rebuilt once a timestep from the
prescribed mean state and the non-reflecting relations, under-relaxed by
:attr:`~NonReflectingPatch.sigma`.

Working that out for the wave speeds
:math:`[V_x - a,\, V_x + a,\, V_x,\, V_x,\, V_x]` of
:math:`[c_\mathrm{up}, c_\mathrm{down}, c_r, c_t, c_s]` gives four splits, not
two, and they have a simple structure: **the acoustic split is fixed by the
geometry** -- :math:`c_\mathrm{up}` is always outgoing when the interior lies on
the :math:`+x` side, :math:`c_\mathrm{down}` always outgoing when it lies on the
:math:`-x` side -- **and the three convective characteristics follow the flow**,
incoming at a span station the flow enters and outgoing at one it leaves.

============ ================== =================== ================
 normal       mean flow          incoming            prescribed rows
============ ================== =================== ================
 :math:`+1`   entering           ``[1, 2, 3, 4]``    ``[0, 1, 2, 3]``
 :math:`+1`   leaving            ``[1]``             ``[4]``
 :math:`-1`   leaving            ``[0]``             ``[4]``
 :math:`-1`   entering           ``[0, 2, 3, 4]``    ``[0, 1, 2, 3]``
============ ================== =================== ================

So a station the flow enters prescribes the four quantities an inflow sets, and
one the flow leaves prescribes static pressure. Reversal is not a special case
needing a guard: it is the other row of the table, and every face carries it at
every span station.

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

The harmonic relations are the one place the two directions genuinely differ,
and each patch needs only one of them. Giles and Saxer derive them for mean flow
along :math:`+x`, so a relation applies only where :math:`V_x > 0` -- and there,
entering implies an inward normal of :math:`+1` and leaving one of :math:`-1`.
A patch's normal is fixed, so exactly one relation is ever live on it and the
other kind of station takes zeroed harmonics, which is the honest thing to do
where the derivation does not hold.

See Also
--------
ember.basepatch.RevolutionPatch : Base class providing the pitchwise geometry
ember.inlet_nonreflecting.NonReflectingInletPatch : Subsonic inflow
ember.outlet_nonreflecting.NonReflectingOutletPatch : Subsonic outflow
ember.mixing_nonreflecting.NonReflectingMixingPatch : Either side of an interface
ember.perturbation.chic_to_mix : Jacobian the characteristic solves are built on
"""

import numpy as np

from ember import perturbation, util
from ember.basepatch import RevolutionPatch
from ember.outlet import calc_backflow_rho


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
    r"""The steady non-reflecting boundary condition.

    Subclasses supply :attr:`_desc`, a description used in error messages;
    :attr:`_sign_interior`, the inward face normal, or ``None`` to take whatever
    the geometry gives; :attr:`_chic_to_target` and :attr:`_target_names`, naming
    the space the prescribed target lives in; :attr:`_target_setters`, mapping
    each required target row to the setter that fills it; and
    :attr:`_target_seeded`, the rows taken from the flow when nothing prescribes
    them. They add the setters themselves and nothing else: the characteristic
    treatment, both harmonic relations and the reversed-flow handling are all
    here.

    :meth:`update_soln` and :meth:`advance` are called once per timestep, the
    first refreshing the reference state to match Giles' definition of the
    characteristic variables as perturbations about the time-level-:math:`n`
    average, the second taking the condition's one under-relaxed step on it.
    :meth:`apply` is called once per Runge-Kutta stage and only imposes what
    those two settled, so the rate of the condition does not scale with the
    stage count.

    The condition is restricted to a constant-:math:`x` plane and to an axially
    subsonic, absolutely subsonic mean state; each restriction is checked and
    raises. Neither restriction concerns the *direction* of the flow through the
    face: a span station whose mean has reversed simply takes the other
    characteristic split, and drives the quantities that split prescribes toward
    rows of the same target.
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

    # A span station stops being treated as one the flow enters once the
    # interior velocity out through the face climbs above this fraction of the
    # mean speed of sound. It starts being treated as one at zero, so the gap
    # between the two is the hysteresis that stops a station chattering between
    # the two splits.
    _frac_rev_off = 0.02

    # Whether to impose the entering state on individual nodes the interior is
    # pushing flow in through, within a station the flow leaves; see
    # _calc_override. Off for a condition whose target rows 2-3 are angles
    # rather than velocities, which cannot express the state to impose.
    _nodal_backflow = True

    # Relaxation factor for the density of such a node.
    _rf_backflow = 1.0

    # Inward face normal: +1 if the interior lies on the +x side of the face,
    # -1 if on the -x side. None lets the geometry decide at attach time, which
    # is what a patch that can sit on either side of a plane wants; a value
    # here is validated against the geometry instead.
    _sign_interior = None

    # Rows filled from the pitchwise mean of the face when nothing has
    # prescribed them; see _seed_target.
    _target_seeded = ()

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

    def _backflow(self):
        """The entering state as a tuple of four span-indexed arrays.

        Rows 0-3 of the target, in the ``[ho, s, Vr, Vt]`` order
        :func:`~ember.outlet.calc_backflow_rho` and :meth:`_calc_override` read
        them in -- so only meaningful in a mix target space, which is why
        :attr:`_nodal_backflow` gates its only caller. Each has a pitch axis of
        length one, so they broadcast against the patch-shaped face state.
        """
        return tuple(self._target[..., row] for row in range(4))

    def _calc_dchic(self, dchic, prim):
        """Change in the incoming characteristics, taken station by station.

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
        target = self._target_from_prim(prim)
        # A face whose stations are all of one kind, which is every face until
        # something reverses, evaluates one branch. Only a genuinely mixed face
        # pays for both, and each carries a pitchwise Hilbert transform.
        if self._entering.all():
            return self._calc_dchic_entering(dchic, target)
        if not self._entering.any():
            return self._calc_dchic_leaving(dchic, target)
        return np.where(
            self._span_bcast(self._entering)[..., np.newaxis],
            self._calc_dchic_entering(dchic, target),
            self._calc_dchic_leaving(dchic, target),
        )

    def _calc_dchic_entering(self, dchic, target):
        r"""Change in the four incoming characteristics where the flow enters.

        The pitchwise mean of each is set by requiring the four prescribed
        quantities of rows 0-3 to take their target values, in one modified
        Newton step (Giles Eq. 5.13-5.15, Saxer Eq. 9).

        The harmonics depend on which way the station is entering. With the
        interior on the :math:`+x` side the mean flow runs along :math:`+x` and
        Giles' inflow relations apply: the tangential vorticity characteristic
        follows from the outgoing acoustic one (Giles Eq. 5.17, Saxer Eq. 56),
        the radial vorticity harmonics are driven to zero, and entropy and
        stagnation enthalpy are held uniform along the pitch (Giles
        Eq. 5.22-5.24) through the two characteristics left free once the
        vorticity ones are fixed. Giles adopts that last constraint because a
        straightforward implementation of the linear theory leaves second-order
        variations in entropy and stagnation enthalpy that would be comparable
        with the losses of a viscous calculation.

        With the interior on the :math:`-x` side the flow through an entering
        station runs along :math:`-x`, where none of that was derived. The
        relation reads the tangential vorticity characteristic, which is itself
        incoming there; nothing is well posed enough to absorb. So the harmonics
        of all four are driven to zero instead: what is imposed is a uniform
        inflow, and the one wave still leaving is carried through untouched, so
        acoustics are not trapped by the choice.
        """
        ref = self._ref
        cols = self._split_entering[0]
        dchic_mean = self._calc_dchic_mean(
            target, self._split_entering, ref["inv_entering"]
        )
        dchic_new = np.zeros_like(dchic)

        if self._sign_interior < 0:
            for k, col in enumerate(cols):
                c = dchic[..., col]
                dchic_new[..., col] = dchic_mean[..., k] - (c - self._pitch_mean(c))
            return dchic_new

        # The non-reflecting relation for the tangential vorticity
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
        ho_nd, s_nd = target[0], target[1]
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

        dchic_new[..., 1] = dchic_mean[..., 0] + dchic_local[..., 0]
        dchic_new[..., 2] = dchic_mean[..., 1] + dchic_r
        dchic_new[..., 3] = dchic_mean[..., 2] + dchic_t
        dchic_new[..., 4] = dchic_mean[..., 3] + dchic_local[..., 1]
        return dchic_new

    def _calc_dchic_leaving(self, dchic, target):
        r"""Change in the single incoming characteristic where the flow leaves.

        Its pitchwise mean follows from the prescribed static pressure of row 4
        (Giles Eq. 5.29-5.30, Saxer Eq. D.31). Row 4 of every target space is
        :math:`\partial p/\partial c = \tfrac{1}{2}` against either acoustic
        characteristic, so the Newton step comes out as
        :math:`\delta \bar{c} = -2(\bar{p} - p_\mathrm{target})`.

        Its harmonics follow the non-reflecting relation of Giles Eq. 5.32 and
        Saxer Eq. 57 when the interior lies on the :math:`-x` side, so that the
        flow through a leaving station runs along :math:`+x` as that relation
        assumes, and are driven to zero otherwise -- the mirror of the entering
        case, and for the same reason.

        Nothing corresponding to Giles' uniform entropy and stagnation enthalpy
        constraint is needed here: both are carried out of the domain by the
        outgoing characteristics rather than prescribed, so the second-order
        variations that constraint exists to suppress never enter.
        """
        ref = self._ref
        col = self._split_leaving[0][0]
        dchic_mean = self._calc_dchic_mean(
            target, self._split_leaving, ref["inv_leaving"]
        )
        c = dchic[..., col]
        c_harm = c - self._pitch_mean(c)
        dchic_new = np.zeros_like(dchic)

        if self._sign_interior > 0:
            dchic_new[..., col] = dchic_mean[..., 0] - c_harm
            return dchic_new

        # Harmonics, from the two outgoing characteristics the relation couples
        # to. Both are taken mean-free so this cannot disturb the mean mode.
        c_t = dchic[..., 3]
        c_down = dchic[..., 1]
        c_t_harm = c_t - self._pitch_mean(c_t)
        c_down_harm = c_down - self._pitch_mean(c_down)
        c_up_ideal = (
            ref["coef_t"] * c_t_harm
            + ref["coef_t_hilbert"] * self._transform_pitch(c_t_harm)
            + ref["coef_down"] * c_down_harm
            + ref["coef_down_hilbert"] * self._transform_pitch(c_down_harm)
        )

        dchic_new[..., col] = dchic_mean[..., 0] + c_up_ideal - c_harm
        return dchic_new

    def _calc_dchic_mean(self, target, split, inv):
        """One modified Newton step on the prescribed pitchwise-mean quantities.

        The residual is evaluated on the state about to be corrected and only
        the Jacobian is frozen, so successive timesteps converge on the target
        rather than re-applying one correction against a reference that is
        already a step out of date.

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

    def _calc_entering(self, avg):
        """Span stations the mean flow enters through, with hysteresis.

        Worked in the inward-normal velocity :math:`V_x n_x`, positive where
        flow comes into the domain, so the same test serves a face of either
        orientation. A station starts being treated as entering as soon as
        anything says it is and stops only once the interior is clearly leaving,
        by :attr:`_frac_rev_off` of the mean speed of sound, so a station
        hovering about zero settles into one split rather than alternating
        between them.

        The test reads the first interior layer as well as the face. The
        interior is the physical signal, and the only one that can release a
        station once this condition is imposing an inflow on the face; but the
        face is what the reference state is built from, so a face that has gone
        backwards has to be carried whatever the interior is doing.

        Parameters
        ----------
        avg : Block
            Pitchwise-mean state, one node per span station.

        Returns
        -------
        array
            Boolean, shape ``(nspan,)``.
        """
        sign = self._sign_interior
        cons = self.block_view_offset_1.conserved_nd
        u_int = sign * (
            self._pitch_mean(cons[..., 1]) / self._pitch_mean(cons[..., 0])
        ).reshape(-1)
        u_face = sign * avg.Vx_nd

        on = (u_int > 0.0) | (u_face >= 0.0)
        off = (u_int < -self._frac_rev_off * avg.a_nd) & (u_face < 0.0)
        prev = self._entering
        if prev is None or prev.shape != on.shape:
            return on
        return np.where(prev, ~off, on)

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

    def _calc_inv_jac(self, c2t, split, where):
        """Invert the mean-mode Jacobian of one split, checking it is not singular.

        Parameters
        ----------
        c2t : array
            Characteristic-to-target Jacobian on the mean state, shape
            ``(nspan, 5, 5)``.
        split : tuple
            ``(incoming characteristic columns, prescribed target rows)``.
        where : str
            Clause naming the stations the system belongs to, for the error
            message.

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
                f"characteristic Jacobian for a span station the flow {where}; "
                "the mean state is degenerate (extreme swirl)."
            )
        return self._span_bcast(util.inv(jac))

    def _calc_mask_out(self):
        """Boolean mask of the characteristic components the interior march owns.

        The complement of the incoming columns of whichever split each span
        station is on.

        Returns
        -------
        array
            Boolean, broadcastable against ``(*shape, 5)``. A bare length-5
            mask while every station is on one split; one entry per span station
            once they are mixed.
        """
        mask_entering = self._mask_from_split(self._split_entering)
        if self._entering.all():
            return mask_entering
        mask_leaving = self._mask_from_split(self._split_leaving)
        if not self._entering.any():
            return mask_leaving
        return np.where(
            self._span_bcast(self._entering)[..., np.newaxis],
            mask_entering,
            mask_leaving,
        )

    def _calc_override(self, prim):
        """Impose the entering state on nodes the interior is pushing flow in through.

        The node-level counterpart of the station-level split: within a station
        the mean flow leaves there is no split to change, since the split is a
        property of that mean and the Hilbert transform couples every node of
        the station to every other. So this is frankly a limiter on the linear
        theory rather than an extension of it, and it is kept out of the state
        the solve carries forward.

        Off unless :attr:`_nodal_backflow` is set, since the state imposed is
        rows 0-3 read as ``[ho, s, Vr, Vt]``, which a target space carrying
        angles in rows 2-3 cannot express.
        """
        if not self._nodal_backflow:
            return prim

        # Detected from the interior layer, the physical signal of flow
        # entering the domain, and never from the face: this method authors
        # that face, and a face-based test would latch every node it flagged
        # permanently into backflow. Stations the characteristic solve is
        # already carrying as entering are left to it rather than treated
        # twice, once here and once there.
        cons_x = self.block_view_offset_1.conserved_nd[..., 1]
        inflow = cons_x * self._sign_interior > 0.0
        if self._entering.any():
            inflow = inflow & ~self._span_bcast(self._entering)
        if not inflow.any():
            return prim

        b = self.block_view
        fluid = b.fluid
        if self._rho_nd_soln is None:
            self._rho_nd_soln = b.rho_nd.copy()

        backflow = self._backflow()
        ho_snap, s_snap, Vr_snap, Vt_snap = backflow
        rho_nd, u_nd = fluid.set_rho_s(
            calc_backflow_rho(
                fluid,
                backflow,
                self._rho_nd_soln,
                prim[..., 0],
                b.Max,
                self._rf_backflow,
            ),
            s_snap,
        )

        # The cap inside calc_backflow_rho holds the radicand non-negative over
        # the whole face, not only on the flagged nodes, so the sqrt is sound
        # everywhere it is evaluated; the errstate is float32 insurance for
        # nodes sitting on the cap itself, which can land a few ulp below zero.
        with np.errstate(invalid="ignore"):
            Vx_nd = self._sign_interior * np.sqrt(
                2.0 * (ho_snap - fluid.get_h(rho_nd, u_nd)) - Vr_snap**2 - Vt_snap**2
            )

        prim_back = np.empty_like(prim)
        prim_back[..., 0] = rho_nd
        prim_back[..., 1] = Vx_nd
        prim_back[..., 2] = Vr_snap
        prim_back[..., 3] = Vt_snap
        prim_back[..., 4] = fluid.get_P(rho_nd, u_nd)
        return np.where(inflow[..., np.newaxis], prim_back, prim)

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

        # Which way the flow runs through each span station. Frozen for the step
        # alongside everything else here, so the characteristic split cannot
        # change between Runge-Kutta stages.
        self._entering = self._calc_entering(avg)
        self._mask_out = self._calc_mask_out()

        # Tested on the magnitude, so a station running backwards fast enough
        # to be axially supersonic is caught too: there one of the two acoustic
        # characteristics changes direction and even the reversed split is
        # wrong.
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
            # Both built at every station, whichever split it is on. Neither
            # goes singular anywhere the guards above admit, so there is nothing
            # to gain by building them conditionally and a branch to lose.
            "inv_entering": self._calc_inv_jac(c2t, self._split_entering, "enters"),
            "inv_leaving": self._calc_inv_jac(c2t, self._split_leaving, "leaves"),
        }
        # The wave parameter magnitude, sqrt(1 - M^2). Both the axial and the
        # tangential Mach number enter it, but not the radial one: Saxer's
        # quasi-3D theory treats each span station as a two-dimensional cascade
        # (his Eq. 15). Only the relation this face's orientation makes live is
        # built; see the module docstring.
        wave = np.sqrt(1.0 - Msq)
        if self._sign_interior > 0:
            self._ref.update(self._calc_ref_entering(c2t, Mn, Mt, wave))
        else:
            self._ref.update(self._calc_ref_leaving(Mn, Mt, wave))

    def _calc_ref_entering(self, c2t, Mn, Mt, wave):
        """Coefficients of Giles Eq. 5.17 and of the uniform ho/s solve."""
        # Stagnation enthalpy and entropy against the entropy and
        # downstream-running pressure characteristics, the two left free once
        # the vorticity characteristics are fixed by the non-reflecting theory.
        # Columns 1 and 4 of a length-5 axis are c_down and c_s. Rows 0 and 1
        # are ho and s in every target space, so this system and the two
        # coupling columns below are the same matrices whatever
        # _chic_to_target is.
        jac_local = np.ascontiguousarray(c2t[..., 0:2, 1::3])

        return {
            "inv_local": self._span_bcast(util.inv(jac_local)),
            "couple_r": self._span_bcast(np.ascontiguousarray(c2t[..., 0:2, 2])),
            "couple_t": self._span_bcast(np.ascontiguousarray(c2t[..., 0:2, 3])),
            "coef_local": self._span_bcast(-Mt / (1.0 + Mn)),
            "coef_hilbert": self._span_bcast(wave / (1.0 + Mn)),
        }

    def _calc_ref_leaving(self, Mn, Mt, wave):
        r"""Coefficients of the rationalised Giles Eq. 5.32, per span station.

        Since :math:`(\beta - M_t)(-\beta - M_t) = 1 - M_n^2` is real and
        mode-independent, rationalising the relation splits it into local terms
        and Hilbert transforms along the pitch, and no Fourier transform need be
        taken at run time:

        .. math::
            \left(1 - M_n^2\right) c_\mathrm{up} =
                -2 M_n M_t\, c_t
                + 2 M_n \sqrt{1 - M^2}\, \mathcal{H}[c_t]
                + \left(M_t^2 - 1 + M^2\right) c_\mathrm{down}
                - 2 M_t \sqrt{1 - M^2}\, \mathcal{H}[c_\mathrm{down}].

        Two limits check it: without swirl it reduces to
        :math:`c_\mathrm{up} = -c_\mathrm{down}
        + 2M_n\mathcal{H}[c_t]/\sqrt{1-M^2}`, a zero harmonic pressure
        perturbation for pure acoustics; and the steady potential mode
        downstream of the plane, :math:`\phi \sim e^{-\mu x}\cos(l\theta)` with
        :math:`\mu = |l|/\sqrt{1-M^2}`, satisfies it exactly.
        """
        # 1 - Mn^2 is the product of the wave-parameter denominator and its
        # conjugate; it is bounded away from zero by the axially subsonic check
        # in the caller.
        denom = 1.0 - Mn**2
        return {
            "coef_t": self._span_bcast(-2.0 * Mn * Mt / denom),
            "coef_t_hilbert": self._span_bcast(2.0 * Mn * wave / denom),
            "coef_down": self._span_bcast((Mt**2 - wave**2) / denom),
            "coef_down_hilbert": self._span_bcast(-2.0 * Mt * wave / denom),
        }

    def _calc_split(self, entering):
        """The characteristic/target split of a span station, from the table above.

        Parameters
        ----------
        entering : bool
            Whether the mean flow comes into the domain through the station.

        Returns
        -------
        tuple
            ``(incoming characteristic columns, prescribed target rows)``, the
            two the same length so the mean-mode system is square.
        """
        # The acoustic that runs against the inward normal is the one the
        # interior owns; the other is incoming whichever way the flow runs.
        acoustic = 1 if self._sign_interior > 0 else 0
        if entering:
            return sorted([acoustic, 2, 3, 4]), [0, 1, 2, 3]
        return [acoustic], [4]

    def _check_plane(self):
        """Validate the boundary plane and settle which side the interior is on."""
        block = self.block
        x = self.block_view.x
        Lref = max(np.ptp(block.x), np.ptp(block.r))
        if np.ptp(x) > self._rtol_geom * Lref:
            raise ValueError(
                f"{self._desc.capitalize()} {self.label!r} must lie on a plane "
                f"of constant x (spread {float(np.ptp(x)):.4g} over reference "
                f"length {Lref:.4g}); canted planes are not implemented."
            )

        # A constant-x plane has an inward normal of exactly (+/-1, 0), so the
        # sign is the whole of it.
        offset = block.x[self._get_offset_slice(1)].mean() - x.mean()
        if offset == 0.0:
            raise ValueError(
                f"{self._desc.capitalize()} {self.label!r} cannot tell which "
                "side its interior lies on: the first interior layer is in the "
                "same plane as the face."
            )
        sign = 1 if offset > 0.0 else -1

        # Read off the class, not the instance, so that a patch whose class
        # leaves the side to the geometry can be re-attached to the other side
        # of a plane rather than validated against its own previous answer.
        fixed = type(self)._sign_interior
        if fixed is not None and fixed != sign:
            side = "+x" if fixed > 0 else "-x"
            verb = "enters" if fixed > 0 else "leaves"
            raise NotImplementedError(
                f"{self._desc.capitalize()} {self.label!r} must have its "
                f"interior on the {side} side, so that flow {verb} along +x."
            )
        self._sign_interior = sign

    def _copy(self, c):
        c._target = None if self._target is None else np.copy(self._target)
        c._target_set = self._target_set.copy()
        c.sigma = self.sigma
        # _hilbert, _ref, _sign_interior and the two splits all derive from the
        # block geometry or solution, so they are rebuilt on the new block
        # rather than copied. The target is copied nondimensionalised, so the
        # new block must share the reference scales of the old one; every block
        # of a grid does.

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
        # The two splits, settled at attach time from the inward normal, and
        # which span stations are on which, rebuilt every timestep by
        # _calc_reference before anything reads it.
        self._split_entering = None
        self._split_leaving = None
        self._mask_out = None
        self._entering = None
        # Face state this patch last authored. The incoming characteristics are
        # carried from here rather than from the marched face; see apply().
        self._prim_prev = None
        # Start-of-step density the reversed-node relaxation runs from, taken
        # by update_soln.
        self._rho_nd_soln = None
        # Under-relaxation of the characteristic correction, Giles Eq. 5.25,
        # needed for wellposedness. He suggests 1/N for N pitchwise nodes,
        # applied once per timestep, and advance() takes it exactly once per
        # timestep, so the two are in the same units: set it to 1/N and it is
        # 1/N. The bound is not about the transform amplifying -- it cannot,
        # its norm grows only logarithmically -- but about how far the
        # pitchwise-nonlocal harmonic relations may spread information in one
        # application while the explicit interior march moves it one cell.
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

    def _recombine(self):
        r"""The face state this patch stands behind, given the marched interior.

        The interior march updates all five characteristics at the boundary
        node, but only the outgoing ones carry legitimate information from
        inside the domain. Giles discards the scheme's increments in the
        incoming characteristics outright (his Eq. 5.25 applies only the
        boundary-condition change), and so must this: keeping them and merely
        under-relaxing on top lets the interior drive the incoming
        characteristics, which is unstable, and worse the smaller
        :attr:`sigma` is. So the outgoing characteristics are taken from the
        marched face and the incoming ones from this patch's own last output.

        Because the characteristic transform is linear with frozen
        coefficients, this reconstruction introduces no reflection of its own.

        Returns
        -------
        dchic : array
            Characteristic deviation of that state from the reference,
            shape ``(*shape, 5)``.
        prim : array
            The same state in primitives, shape ``(*shape, 5)``.
        """
        b = self.block_view
        ref = self._ref

        prim_marched = np.stack((b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1)
        if self._prim_prev is None:
            self._prim_prev = prim_marched.copy()

        dchic_prev = util.matvec(ref["p2c"], self._prim_prev - ref["prim"])
        dchic_marched = util.matvec(ref["p2c"], prim_marched - ref["prim"])
        dchic = np.where(self._mask_out, dchic_marched, dchic_prev)
        return dchic, ref["prim"] + util.matvec(ref["c2p"], dchic)

    def advance(self):
        r"""Take the boundary condition's one step; call once per timestep.

        :meth:`_calc_dchic` supplies the change in the incoming characteristics
        and :attr:`sigma` scales it, which is exactly Giles' Eq. 5.25
        correction. This is the whole of a timestep's boundary-condition
        change: :meth:`apply` only imposes the result, once per stage.

        Per timestep and not per stage because the harmonic relations couple
        every pitchwise node to every other through the Hilbert transform, so
        one application can spread information across the whole pitch while the
        explicit interior march moves it one cell. Giles' :math:`1/N` for
        :math:`N` pitchwise nodes is the restriction that keeps the two in step,
        and it is a bound per timestep; taking the step once per stage
        multiplied the rate by the stage count and left :attr:`sigma` dependent
        on the integrator.

        A no-op until something has been prescribed, so that a patch missing a
        setter still reports it from :meth:`apply` rather than from here.
        """
        if not self._target_set[list(self._target_setters)].all():
            return
        if self._ref is None:
            self._calc_reference()

        dchic, prim = self._recombine()
        self._prim_prev = prim + self.sigma * util.matvec(
            self._ref["c2p"], self._calc_dchic(dchic, prim)
        )

    def apply(self):
        r"""Impose the non-reflecting condition on the patch.

        Called once per Runge-Kutta stage, and imposes only: the outgoing
        characteristics are re-read from the marched face every stage so a wave
        reaching the boundary still passes through within the step, while the
        incoming ones are the state :meth:`update_soln` last authored. The
        :attr:`sigma`-relaxed correction that advances that state is taken there,
        once per timestep, not here.

        :meth:`_calc_override` is given the chance to change what actually
        reaches the block, and its result is deliberately not carried back into
        :attr:`_prim_prev`, so a condition that has to depart from its own linear
        theory somewhere does not thereby corrupt the characteristic state it is
        still solving on.
        """
        if not self._target_set[list(self._target_setters)].all():
            self._raise_unset()
        if self._ref is None:
            self._calc_reference()

        b = self.block_view
        _, prim = self._recombine()
        prim_write = self._calc_override(prim)
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
        self._split_entering = self._calc_split(True)
        self._split_leaving = self._calc_split(False)
        self._calc_hilbert()

        shape = self._target_shape()
        if self._target is None or self._target.shape != shape:
            self._target = util.zeros(shape)
            self._target_set = np.zeros(5, dtype=bool)

    def update_soln(self):
        """Refresh the frozen reference state; call once per timestep.

        Re-derives the pitchwise-mean state and every Jacobian evaluated on it,
        which :meth:`apply` then holds fixed across the Runge-Kutta stages of
        the step. Snapshots the density first, so a reversed node's density is
        relaxed from the start-of-step value rather than from whatever the last
        stage happened to leave.

        Pairs with :meth:`advance`, which takes the boundary condition's own
        step on the reference this leaves behind.
        """
        self._rho_nd_soln = self.block_view.rho_nd.copy()
        self._calc_reference()
