r"""Non-reflecting subsonic outlet boundary condition for EMBER CFD.

:class:`NonReflectingOutletPatch` prescribes the pitchwise-mean static pressure
at an outflow face while letting outgoing waves leave the domain, after the
steady non-reflecting theory of :cite:t:`Giles1988` (his Section 5.6) extended
to three dimensions by :cite:t:`Saxer1993` (his Section 5.4.5 and Appendix D.3).

The condition is the mirror image of
:class:`~ember.inlet_nonreflecting.NonReflectingInletPatch` and is simpler than
it. Of the five characteristics at an axially subsonic outflow plane, four are
outgoing (entropy, two vorticity waves, the downstream-running pressure wave)
and only the upstream-running pressure wave is incoming, so a single
characteristic has to be set:

* its **pitchwise mean** follows from the prescribed exit static pressure
  (Giles Eq. 5.29-5.30, Saxer Eq. D.31);
* its **pitchwise harmonics** follow from the harmonics of the outgoing
  tangential vorticity and downstream-running pressure characteristics by the
  non-reflecting relation (Giles Eq. 5.32, Saxer Eq. 57).

Nothing corresponding to Giles' uniform entropy and stagnation enthalpy
constraint (his Eq. 5.22-5.24) is needed here: entropy and stagnation enthalpy
are carried out of the domain by the outgoing characteristics rather than
prescribed, so the second-order variations that constraint exists to suppress
never enter.

Unlike :class:`~ember.outlet.OutletPatch` this patch imposes the prescribed
pressure only on the pitchwise mean at each span station, and it neither
extrapolates from the interior nor offers throttle control. It does share that
patch's radial-equilibrium adjustment, through the common
:func:`~ember.outlet.calc_radial_equilibrium`; a swirling exit flow needs one,
and prescribing the pitchwise mean at every span station holds the exit plane
off radial equilibrium just as firmly as prescribing it node by node.

It shares that patch's backflow handling too, but reversal means something
different to a condition written in characteristic variables. The split above
holds only while the mean flow runs forward: reverse a span station and four of
its five characteristics turn incoming, so one prescribed quantity becomes four
and the exit pressure stops being one of them. That station is therefore
switched to an inflow condition of its own, in the mix variables
:math:`[h_0, s, V_r, V_\theta]` this patch's target already carries, rather than
the angles of :class:`~ember.inlet_nonreflecting.NonReflectingInletPatch`,
whose mean-mode solve goes singular at exactly the axial velocity where this is
wanted. Reversal confined to nodes within a station whose mean still runs
forward has no such reading -- the split belongs to the mean, and the Hilbert
transform couples the whole pitch -- and is handled node by node as a limiter
instead.

Those four quantities are always available, so a reversed mean is always
carried and never raises. :meth:`NonReflectingOutletPatch.set_backflow`
prescribes them; left alone, they are seeded once from the pitchwise mean of the
exit plane as it stands at the first timestep and frozen there, which for a run
started from a design guess is the design exit state.

See Also
--------
ember.nonreflecting.NonReflectingPatch : Base class holding the shared machinery
ember.outlet.OutletPatch : Reflecting outlet with prescribed nodal pressure
ember.inlet_nonreflecting.NonReflectingInletPatch : The inflow counterpart
ember.perturbation.chic_to_mix : Jacobian the characteristic solves are built on
"""

import numpy as np

from ember.nonreflecting import NonReflectingPatch
from ember.outlet import calc_backflow_rho, calc_radial_equilibrium


class NonReflectingOutletPatch(NonReflectingPatch):
    r"""Non-reflecting subsonic outflow boundary condition.

    Prescribes the static pressure :math:`p` as a pitchwise-mean quantity at
    each span station, while absorbing outgoing waves rather than reflecting
    them. It must be set before
    :meth:`~ember.nonreflecting.NonReflectingPatch.apply` is called, via
    :meth:`set_P`, which stores its target nondimensionally in :attr:`P_nd`, so
    the patch must already be attached to a block whose fluid is set.

    Each Runge-Kutta stage the characteristic deviation of the face state from
    the frozen pitchwise-mean reference state is formed, and the change required
    in the single incoming characteristic :math:`c_\mathrm{up}` is assembled
    from two contributions.

    The **mean mode** is one Newton step on the prescribed pressure. Row 4 of
    :func:`~ember.perturbation.chic_to_mix` gives
    :math:`\partial p/\partial c_\mathrm{up} = \tfrac{1}{2}` exactly, so the
    step the base class's
    :meth:`~ember.nonreflecting.NonReflectingPatch._calc_dchic_mean` takes comes
    out as (Giles Eq. 5.30, Saxer Eq. D.31)

    .. math::
        \delta \bar{c}_\mathrm{up} = -2\left(\bar{p} - p_\mathrm{out}\right).

    Giles takes the residual against the flux-averaged pressure; the mean here
    is the weighted pitch mean of
    :attr:`~ember.basepatch.RevolutionPatch.weight_pitch`, the same average the
    inflow condition takes its residuals against.

    The **harmonics** follow the non-reflecting relation, per pitchwise Fourier
    mode :math:`m` (Giles Eq. 5.32, Saxer Eq. 57),

    .. math::
        \hat{c}_\mathrm{up} = \frac{2M_n}{\beta - M_t}\hat{c}_t
            - \frac{\beta + M_t}{\beta - M_t}\hat{c}_\mathrm{down},
        \qquad \beta = i\,\mathrm{sign}(m)\sqrt{1 - M^2}.

    Since :math:`(\beta - M_t)(-\beta - M_t) = 1 - M_n^2` is real and
    mode-independent, rationalising splits the relation into local terms and
    Hilbert transforms along the pitch, and no Fourier transform need be taken
    at run time:

    .. math::
        \left(1 - M_n^2\right) c_\mathrm{up} =
            -2 M_n M_t\, c_t
            + 2 M_n \sqrt{1 - M^2}\, \mathcal{H}[c_t]
            + \left(M_t^2 - 1 + M^2\right) c_\mathrm{down}
            - 2 M_t \sqrt{1 - M^2}\, \mathcal{H}[c_\mathrm{down}],

    using the pitchwise Hilbert transform :math:`\mathcal{H}` built by the base
    class, whose Fourier multiplier is :math:`-i\,\mathrm{sign}(m)`. Two limits
    check the result:
    without swirl it reduces to
    :math:`c_\mathrm{up} = -c_\mathrm{down} + 2M_n\mathcal{H}[c_t]/\sqrt{1-M^2}`,
    a zero harmonic pressure perturbation for pure acoustics; and the steady
    potential mode downstream of the plane,
    :math:`\phi \sim e^{-\mu x}\cos(l\theta)` with
    :math:`\mu = |l|/\sqrt{1-M^2}`, satisfies it exactly.

    The sum of the two contributions is applied under-relaxed by :attr:`sigma`;
    the four outgoing characteristics are carried through untouched.

    :meth:`set_adjustment` adds a spanwise radial-equilibrium profile to the
    prescribed pressure, re-derived from the solution once per timestep by
    :meth:`update_target`. Without it the prescribed pressure is imposed on
    every span station alike, which for a swirling exit flow fights the
    centrifugal pressure gradient the flow is trying to establish.

    Reversed flow is carried at the span station through a switch of the
    characteristic split and at the node through a limiter, toward the state
    :meth:`set_backflow` prescribes or, failing that, the seeded one; see that
    method and the module docstring.
    """

    _collection_name = "outlet_nonreflecting"

    _desc = "non-reflecting outlet patch"

    # A span station is released from reversed handling once its mean axial
    # velocity climbs back above this fraction of the mean speed of sound. It
    # enters at zero, so the gap between the two is the hysteresis that stops a
    # station chattering between the two splits.
    _frac_rev_off = 0.02

    # Everything but the upstream-running pressure wave leaves an outflow
    # plane, so that one wave carries the one quantity an outflow prescribes,
    # row 4 of the target.
    _split_fwd = ([0], [4])

    # Where the mean flow has reversed the wave speeds are Vx - a < 0,
    # Vx + a > 0 and Vx < 0, so only the downstream-running pressure wave still
    # leaves and the other four turn incoming, carrying the four quantities an
    # inflow prescribes. Note this is not the inflow condition's split: that
    # patch has its interior on the other side of the face, and it is the
    # geometry, not the flow, that decides which sign of wave speed points out
    # of the domain.
    _split_rev = ([0, 2, 3, 4], [0, 1, 2, 3])

    # Relaxation factor for the density of a reversed node.
    _rf_backflow = 1.0

    _sign_interior = -1

    _target_setters = {4: "set_P"}

    # The backflow state. Not required of the user: seeded from the exit plane
    # at the first timestep if set_backflow is never called.
    _target_seeded = (0, 1, 2, 3)

    def _backflow(self):
        """The prescribed backflow state as a tuple of four span-indexed arrays.

        Rows 0-3 of the target, in the ``[ho, s, Vr, Vt]`` order
        :func:`~ember.outlet.calc_backflow_rho` and the reversed-station solve
        both read them in. Each has a pitch axis of length one, so they
        broadcast against the patch-shaped face state.
        """
        return tuple(self._target[..., row] for row in range(4))

    def _calc_dchic(self, dchic, prim):
        """Change in the incoming characteristics, taken station by station."""
        target = self._target_from_prim(prim)
        dchic_new = self._calc_dchic_forward(dchic, target)
        if self._reversed.any():
            rev = self._span_bcast(self._reversed)[..., np.newaxis]
            dchic_new = np.where(
                rev, self._calc_dchic_reversed(dchic, target), dchic_new
            )
        return dchic_new

    def _calc_dchic_forward(self, dchic, target):
        """Change in the incoming characteristic; see the class docstring."""
        ref = self._ref

        # Mean mode, a one-by-one system on the prescribed pressure.
        dchic_mean = self._calc_dchic_mean(target, self._split_fwd, ref["inv_fwd"])

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

        c_up = dchic[..., 0]
        dchic_new = np.zeros_like(dchic)
        dchic_new[..., 0] = (
            dchic_mean[..., 0] + c_up_ideal - (c_up - self._pitch_mean(c_up))
        )
        return dchic_new

    def _calc_dchic_reversed(self, dchic, target):
        r"""Change in the incoming characteristics at a reversed span station.

        Four characteristics are incoming there, so four quantities have to be
        set, and rows 0-3 of the target hold exactly four: the station is driven
        toward that state in one modified Newton step, as at the inflow.

        The harmonics of all four are driven to zero rather than through the
        non-reflecting relation. That relation is derived for forward mean flow
        and reads the tangential vorticity characteristic, which is itself
        incoming at a reversed station; nothing there is well posed enough to
        absorb. What is imposed is a uniform inflow, and the one wave still
        leaving the station, the downstream-running pressure wave, is carried
        through untouched, so acoustics are not trapped by the choice.
        """
        dchic_mean = self._calc_dchic_mean(
            target, self._split_rev, self._ref["inv_rev"]
        )

        dchic_new = np.zeros_like(dchic)
        for k, col in enumerate(self._split_rev[0]):
            c = dchic[..., col]
            dchic_new[..., col] = dchic_mean[..., k] - (c - self._pitch_mean(c))
        return dchic_new

    def _calc_override(self, prim):
        """Impose the backflow state on nodes the interior is pushing flow in through.

        The node-level counterpart of the station-level switch: within a
        station whose mean is still forward there is no characteristic split to
        change, since the split is a property of that mean and the Hilbert
        transform couples every node of the station to every other. So this is
        frankly a limiter on the linear theory rather than an extension of it,
        and it is kept out of the state the solve carries forward.
        """
        # Detected from the interior layer, the physical signal of flow
        # entering the domain, and never from the face: this method authors
        # that face, and a face-based test would latch every node it flagged
        # permanently into backflow. Stations the characteristic solve is
        # already carrying as reversed are left to it rather than treated
        # twice, once here and once there.
        inflow = self.block_view_offset_1.conserved_nd[..., 1] < 0.0
        if self._reversed.any():
            inflow = inflow & ~self._span_bcast(self._reversed)
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
            Vx_nd = -np.sqrt(
                2.0 * (ho_snap - fluid.get_h(rho_nd, u_nd)) - Vr_snap**2 - Vt_snap**2
            )

        prim_back = np.empty_like(prim)
        prim_back[..., 0] = rho_nd
        prim_back[..., 1] = Vx_nd
        prim_back[..., 2] = Vr_snap
        prim_back[..., 3] = Vt_snap
        prim_back[..., 4] = fluid.get_P(rho_nd, u_nd)
        return np.where(inflow[..., np.newaxis], prim_back, prim)

    def _calc_reference_extra(self, avg, c2t, Mn, Mt, wave):
        """Coefficients of the rationalised harmonic relation, per span station."""
        # 1 - Mn^2 is the product of the wave-parameter denominator and its
        # conjugate; it is bounded away from zero by the axially subsonic check
        # in the caller. Every coefficient here stays finite under a reversed
        # mean too, so a station the solve is carrying as reversed is left to
        # compute its own alongside the rest and simply never read.
        denom = 1.0 - Mn**2
        return {
            "coef_t": self._span_bcast(-2.0 * Mn * Mt / denom),
            "coef_t_hilbert": self._span_bcast(2.0 * Mn * wave / denom),
            "coef_down": self._span_bcast((Mt**2 - wave**2) / denom),
            "coef_down_hilbert": self._span_bcast(-2.0 * Mt * wave / denom),
        }

    def _calc_reversed(self, avg):
        """Span stations to carry as reversed this step, with hysteresis.

        A station enters reversed handling when its mean axial velocity goes
        negative and leaves only once that velocity is back above
        :attr:`_frac_rev_off` of the mean speed of sound, so a station hovering
        about zero settles into one split rather than alternating between them.

        The entry test reads the first interior layer as well as the face. The
        interior is the physical signal, and the only one that can release a
        station once this condition is imposing reversed flow on the face; but
        the face is what the reference state is built from, so a face that has
        gone backwards has to be carried whatever the interior is doing. Taken
        together they also make the base class's backflow guard unreachable
        here: a station with a non-positive mean face velocity satisfies the
        entry test and fails the release test, so it is flagged either way.
        """
        cons = self.block_view_offset_1.conserved_nd
        Vx_int = (
            self._pitch_mean(cons[..., 1]) / self._pitch_mean(cons[..., 0])
        ).reshape(-1)

        on = (Vx_int < 0.0) | (avg.Vx_nd <= 0.0)
        off = (Vx_int > self._frac_rev_off * avg.a_nd) & (avg.Vx_nd > 0.0)
        prev = self._reversed
        if prev is None or prev.shape != on.shape:
            return on
        return np.where(prev, ~off, on)

    def _copy(self, c):
        super()._copy(c)
        c._P_raw = None if self._P_raw is None else np.copy(self._P_raw)
        c._P_level_nd = None if self._P_level_nd is None else np.copy(self._P_level_nd)
        c._adjustment = self._adjustment.copy()
        # _P_last_nd is derived from the solution, so it is rebuilt by
        # update_target on the new block rather than copied, as _ref is. So are
        # _rho_nd_soln and the reversed stations.

    def _setup(self):
        super()._setup()
        # The prescribed pressure as passed, kept so set_adjustment can tell
        # whether it varies along the span.
        self._P_raw = None
        # The prescribed pressure level, nondimensional and pitch-averaged.
        # Held apart from the target row it feeds, which update_target
        # overwrites with the level plus the spanwise adjustment.
        self._P_level_nd = None
        # Empty means no spanwise adjustment; see set_adjustment.
        self._adjustment = {}
        # Relaxation state of that adjustment, built by update_target.
        self._P_last_nd = None
        # Start-of-step density the reversed-node relaxation runs from, taken
        # by update_soln.
        self._rho_nd_soln = None

    def set_adjustment(self, radial_equilibrium=True, rf=0.1):
        r"""Configure the spanwise adjustment to the prescribed pressure.

        Swirling flow leaving a blade row carries a centrifugal radial pressure
        gradient. Prescribing one pressure at every span station fights it and
        induces unnatural streamline curvature, so the adjustment adds the
        profile satisfying :math:`dp/dr = \overline{\rho V_\theta}\,
        \overline{V_\theta}/r`, integrated from the hub, where the prescribed
        pressure is then the value enforced. It is re-derived from the solution
        by :meth:`update_target` once per timestep and relaxed toward the new
        value:

        .. math::

            \Delta p^\mathrm{new} = \mathit{rf}\,\Delta p
                + (1 - \mathit{rf})\,\Delta p^\mathrm{old}

        Off unless this method is called, and incompatible with a non-scalar
        :meth:`set_P`, which would prescribe a spanwise profile of its own and
        double count.

        Unlike :meth:`~ember.outlet.OutletPatch.set_adjustment` there is no
        dynamic-head term. That offset has zero pitchwise mean at every span
        station by construction, and this patch imposes nothing but pitchwise
        means, so it would be annihilated exactly.

        Parameters
        ----------
        radial_equilibrium : bool, optional
            Include the radial equilibrium offset. Default True; False
            configures an adjustment that adjusts nothing, and is accepted only
            so the signature stays a subset of the reflecting outlet's.
        rf : float, optional
            Relaxation factor applied to the profile each step. Default 0.1.
        """
        if self._P_raw is not None and self._P_raw.ndim > 0 and self._P_raw.size > 1:
            raise ValueError("Adjustment is incompatible with non-scalar P")
        self._adjustment = {
            "radial_equilibrium": bool(radial_equilibrium),
            "rf": float(rf),
        }
        self._P_last_nd = None

    def set_backflow(self, ho, s, Vr, Vt):
        r"""Prescribe the state imposed where the flow reverses at the outlet.

        Reversal is carried at two levels, and both draw on these four
        quantities.

        A **span station** whose mean has reversed is genuinely an inflow plane,
        and is treated as one. Four of its five characteristics turn incoming,
        so four quantities have to be prescribed, and the four given here are
        exactly they; the one wave still leaving, the downstream-running
        pressure wave, is carried through from the interior as always. The
        prescribed static pressure is not imposed at such a station: pressure is
        what the free wave carries there. If a large part of the span ends up
        reversed the exit level is no longer under control, and the boundary
        wants moving downstream rather than the condition made cleverer.

        A **node** whose interior neighbour is pushing flow inward, at a station
        whose mean is still forward, is overwritten with the same four
        quantities and a density relaxed from the interior, exactly as
        :meth:`ember.outlet.OutletPatch.set_backflow` does it. There is no
        characteristic split to change at that level -- the split belongs to the
        station's mean, and the Hilbert transform couples every node of a
        station to every other -- so this one is a limiter on the linear theory,
        applied to what reaches the block and kept out of the state the solve
        carries forward.

        Calling this is optional. Left alone, the same four rows are seeded once
        from the pitchwise mean of the exit plane at the first timestep and
        frozen there; see
        :meth:`~ember.nonreflecting.NonReflectingPatch._seed_target`.

        Parameters
        ----------
        ho : float or array
            Stagnation enthalpy [J/kg]. A scalar or an array that broadcasts to
            :attr:`~ember.basepatch.Patch.shape`, of which only the pitchwise
            mean at each span station is imposed. Unlike the reflecting
            outlet's version a spanwise profile is accepted, since the seed
            this replaces is one.
        s : float or array
            Specific entropy [J/(kg K)].
        Vr : float or array
            Radial velocity [m/s].
        Vt : float or array
            Tangential velocity [m/s].

        See Also
        --------
        ember.outlet.OutletPatch.set_backflow : The reflecting outlet's version
        """
        fluid = self.block.fluid
        self._set_target_row(0, "ho", np.asarray(ho) / fluid.u_ref)
        self._set_target_row(1, "s", np.asarray(s) / fluid.Rgas_ref)
        self._set_target_row(2, "Vr", np.asarray(Vr) / fluid.V_ref)
        self._set_target_row(3, "Vt", np.asarray(Vt) / fluid.V_ref)

    def set_P(self, P):
        r"""Prescribe the outlet static pressure.

        Imposed on the pitchwise mean at each span station, not node by node,
        so a value varying along the pitch is averaged before use. With
        :meth:`set_adjustment` configured this is the hub value and the
        spanwise profile follows from radial equilibrium; without it, a
        spanwise array prescribes the profile directly.

        Parameters
        ----------
        P : float or array
            Prescribed static pressure :math:`p_\mathrm{out}` [Pa]; must be
            positive and finite. A scalar or any array that broadcasts to
            :attr:`~ember.basepatch.Patch.shape`.
        """
        arr = np.asarray(P)
        if not np.isfinite(arr).all():
            raise ValueError("P must be finite")
        if not (arr > 0.0).all():
            raise ValueError("P must be positive")
        if self._adjustment and arr.ndim > 0 and arr.size > 1:
            raise ValueError("Non-scalar P is incompatible with the adjustment")
        self._P_raw = arr
        # The level is the whole target until update_target folds in the
        # spanwise adjustment, so a patch driven directly rather than by the
        # solver loop still has the prescribed pressure imposed.
        self._set_target_row(4, "P", arr / self.block.fluid.P_ref)
        self._P_level_nd = np.copy(self._target[..., 4])

    def reset_target(self):
        """Drop the solution-derived part of the pressure target.

        Called by :class:`~ember.block.Block` when the reference scales change,
        since the spanwise adjustment :meth:`update_target` built is an integral
        over a solution and a geometry expressed in the old ones. The prescribed
        level stands until the next :meth:`update_target` re-derives the
        profile.

        The seeded backflow rows are deliberately left alone: they are the state
        the exit plane started from, taken once and frozen, and nothing
        distinguishes a seeded row from one :meth:`set_backflow` prescribed.
        """
        self._P_last_nd = None
        if self._P_level_nd is not None:
            self._target[..., 4] = self._P_level_nd

    def update_soln(self):
        """Snapshot the density the backflow relaxation runs from, then refresh the reference.

        Should be called once per outer timestep before the Runge-Kutta stages,
        as :meth:`ember.grid.Grid.update_bconds` does, so a reversed node's
        density is relaxed from the start-of-step value rather than from
        whatever the last stage happened to leave.
        """
        self._rho_nd_soln = self.block_view.rho_nd.copy()
        super().update_soln()

    def update_target(self):
        """Recompute the pressure target for the current timestep.

        Applies the spanwise adjustment of :meth:`set_adjustment`, if
        configured. Should be called once per outer timestep before the
        Runge-Kutta stages; :meth:`ember.grid.Grid.update_bconds` does so.
        """
        if self._P_level_nd is None:
            # Nothing prescribed yet; apply() reports the missing setter.
            return

        if not self._adjustment:
            self._target[..., 4] = self._P_level_nd
            return

        if self._adjustment["radial_equilibrium"]:
            profile = self._span_bcast(calc_radial_equilibrium(self).astype(np.float32))
        else:
            profile = np.zeros_like(self._P_level_nd)

        # Relax, seeding the history with the first profile so the target starts
        # where the flow is rather than crawling out from zero.
        if self._P_last_nd is None:
            self._P_last_nd = profile.copy()
        rf = self._adjustment["rf"]
        self._P_last_nd = rf * profile + (1.0 - rf) * self._P_last_nd
        self._target[..., 4] = self._P_level_nd + self._P_last_nd
