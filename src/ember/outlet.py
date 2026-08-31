r"""Subsonic outlet boundary condition.

:class:`OutletPatch` prescribes the pitchwise-mean static pressure at an outflow
face while letting outgoing waves leave the domain, after the steady
non-reflecting theory of :cite:t:`Giles1988` (his Section 5.6) extended to three
dimensions by :cite:t:`Saxer1993` (his Section 5.4.5 and Appendix D.3).

The characteristic treatment is entirely
:class:`~ember.patch.NonReflectingPatch`'s; what this class adds is an
interior on the upstream side of the face and the pressure a physical exit
plane is held at. Of the five characteristics at a subsonic outflow face four are
outgoing and only the upstream-running pressure wave is incoming, so a single
quantity is prescribed: the pitchwise mean of the static pressure at each span
station.

That pressure is imposed on the pitchwise mean alone, never node by node, and
nothing is extrapolated from the interior. :meth:`OutletPatch.set_throttle`
leaves all of that alone and only chooses the level, moving it each timestep
until the patch passes a prescribed mass flow, so a characteristic can be swept
by mass flow rather than by back pressure. A swirling exit flow still needs a
radial-equilibrium adjustment, supplied by :func:`calc_radial_equilibrium`:
prescribing the pitchwise mean at every span station holds the exit plane off
radial equilibrium just as firmly as prescribing it node by node would.

Reversal means something particular to a condition written in characteristic
variables. Reverse a span station and four of its five characteristics turn
incoming, so one prescribed quantity becomes four and the exit pressure stops
being one of them: the base class switches that station to the inflow split and
drives it to rows 0-3 of the target, in the mix variables
:math:`[h_0, s, V_s, V_\theta]` this patch works in, where :math:`V_s` is the
velocity in the exit surface.
:meth:`OutletPatch.set_backflow_ho_s` and its companions prescribe them; left
alone, they are seeded once from the pitchwise mean of the exit plane as it
stands at the first timestep and frozen there, which for a run started from a
design guess is the design exit state. :math:`V_s` is the exception: it is
pinned at zero rather than prescribed or seeded, so backflow enters normal to
the exit surface whatever that surface's orientation. Reversal confined to nodes within a
station whose mean still runs forward is handled node by node as a limiter
instead, applied by the base class.

See Also
--------
ember.patch.NonReflectingPatch : The condition itself
ember.patch.InletPatch : The inflow counterpart
ember.perturbation.chic_to_mix : Jacobian the mean-mode solves are built on
"""

import numpy as np

from ember import average
from ember.nonreflecting import NonReflectingPatch, replayable


def calc_radial_equilibrium(patch):
    r"""Spanwise static pressure profile in centrifugal radial equilibrium.

    Swirling flow leaving a blade row carries a centrifugal radial pressure
    gradient,

    .. math::

        \frac{dp}{dr} = \frac{\overline{\rho V_\theta}\;\overline{V_\theta}}{r}

    where :math:`\overline{\Box}` is the pitch mean. This product-of-means form
    matches the Multall ``EXBCONDS`` radial-equilibrium treatment. Flow
    quantities are pitch-averaged over the first interior layer (the offset-1
    slice) rather than the boundary face, so the profile is driven by the
    interior rather than by whatever the boundary condition last wrote there.

    Integrated from hub to tip and anchored to zero at the hub, so a patch that
    adds this to a prescribed pressure enforces that pressure at the hub.

    Parameters
    ----------
    patch : ember.patch.RevolutionPatch
        Outflow patch to read the interior layer and pitch weights from.

    Returns
    -------
    array
        Nondimensional pressure offset at each span station, shape
        ``(nspan,)``, zero at the hub. The caller broadcasts it over the patch
        shape and adds it to its own prescribed pressure.

    See Also
    --------
    ember.patch.OutletPatch.set_adjustment : Enables this on the outlet
    """
    b1 = patch.block_view_offset_1
    w = patch.weight_pitch
    pd = patch.pitch_dim
    rhoVt_mean = np.sum(b1.rho_nd * b1.Vt_nd * w, axis=pd).squeeze()
    Vt_mean = np.sum(b1.Vt_nd * w, axis=pd).squeeze()
    r_nd = np.sum(b1.r_nd * w, axis=pd).squeeze()
    # No NaN guard here: a diverged march can feed NaN into these means, but it
    # should propagate through (into the pressure target, then the conserved
    # state) and be caught by the solver's own Grid.check_nan() on its next
    # pass, the same graceful path every other divergence takes. Raising here
    # instead pre-empts that with a hard, uncaught crash.
    dPdr_nd = rhoVt_mean * Vt_mean / r_nd
    dr_nd = np.diff(r_nd)
    P_re_span_nd = np.empty(len(r_nd))
    P_re_span_nd[0] = 0.0
    P_re_span_nd[1:] = np.cumsum(0.5 * (dPdr_nd[:-1] + dPdr_nd[1:]) * dr_nd)
    return P_re_span_nd


class OutletPatch(NonReflectingPatch):
    r"""Subsonic outflow boundary condition.

    Prescribes the static pressure :math:`p` as a pitchwise-mean quantity at
    each span station, while absorbing outgoing waves rather than reflecting
    them. It must be set before
    :meth:`~ember.patch.NonReflectingPatch.apply` is called, via
    :meth:`set_P`, which stores its target nondimensionally in :attr:`P_nd`, so
    the patch must already be attached to a block whose fluid is set.

    Giles takes the mean-mode residual against the flux-averaged pressure; the
    mean here is the weighted pitch mean of
    :attr:`~ember.patch.RevolutionPatch.weight_pitch`, the same average
    every other residual in the family is taken against.

    :meth:`set_adjustment` adds a spanwise radial-equilibrium profile to the
    prescribed pressure, re-derived from the solution once per timestep by
    :meth:`update_target`. Without it the prescribed pressure is imposed on
    every span station alike, which for a swirling exit flow fights the
    centrifugal pressure gradient the flow is trying to establish.

    :meth:`set_throttle` turns the prescribed pressure into a starting point
    rather than the condition: a proportional-integral controller moves the
    level each timestep until the patch passes a target mass flow. The pressure
    is still what the boundary imposes -- the throttle only chooses which
    pressure -- so the characteristic treatment is untouched by it.

    :meth:`set_backflow_ho_s` (or :meth:`set_backflow_Po_To`) and
    :meth:`set_backflow_Vt` prescribe the inflow state a reversed span station
    is driven to; see those methods and the module docstring. Its meridional
    direction is not prescribed and cannot be: backflow comes in normal to the
    exit surface, so the row that would carry it is pinned at zero. Unlike an
    angle, a meridional velocity cannot be resolved onto a face of arbitrary
    orientation without knowing the normal component, which is what the
    reversed-station solve derives from :math:`h_0` -- so there is nothing
    consistent for a setter to mean.
    """

    _desc = "outlet patch"

    _sign_interior = -1

    _target_setters = {4: "set_P"}

    # The inflow state a reversed station is driven to. Not required of the
    # user: seeded from the exit plane if the set_backflow_* setters are never
    # called. Row 2, the velocity in the surface, is not seeded and not
    # settable; attach_to_block pins it at zero.
    _target_seeded = (0, 1, 3)

    def _copy(self, c):
        super()._copy(c)
        c._P_raw = None if self._P_raw is None else np.copy(self._P_raw)
        c._P_level_nd = None if self._P_level_nd is None else np.copy(self._P_level_nd)
        c._adjustment = self._adjustment.copy()
        c._mdot_target = self._mdot_target
        c._Kp = self._Kp
        c._Ki = self._Ki
        # Carried, not reset: a throttle that has already wound its integral out
        # to an operating point hands it to the copy, so a patch following its
        # block onto a finer multigrid level resumes rather than starts again.
        c._eps_int = self._eps_int
        c._mdot = self._mdot
        # _P_last_nd is derived from the solution, so it is rebuilt by
        # update_target on the new block rather than copied, as _ref is.

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
        # None means no throttle and the prescribed pressure stands as set.
        self._mdot_target = None
        self._Kp = 0.0
        self._Ki = 0.0
        # The controller's whole memory: the running sum of the fractional
        # mass-flow error, and the mass flow last measured. Every throttle
        # quantity reported by get_throttle_stats is derived from these two, so
        # neither the split correction terms nor a previous error is stored.
        self._eps_int = 0.0
        self._mdot = 0.0

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

        There is no dynamic-head term. Such an offset has zero pitchwise mean at
        every span station by construction, and this patch imposes nothing but
        pitchwise means, so it would be annihilated exactly.

        Parameters
        ----------
        radial_equilibrium : bool, optional
            Include the radial equilibrium offset. Default True; False
            configures an adjustment that adjusts nothing.
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

    @replayable
    def set_backflow_ho_s(self, ho, s):
        r"""Prescribe the stagnation enthalpy and entropy imposed where the exit flow reverses.

        Reversal is carried at two levels, and both draw on the four backflow
        quantities this and its companion setters prescribe.

        A **span station** whose mean has reversed is genuinely an inflow plane
        and is treated as one. Four of its five characteristics turn incoming,
        so four quantities have to be prescribed, and the four backflow rows are
        exactly they; the one wave still leaving, the downstream-running
        pressure wave, is carried through from the interior as always. The
        prescribed static pressure is not imposed at such a station: pressure is
        what the free wave carries there. If a large part of the span ends up
        reversed the exit level is no longer under control, and the boundary
        wants moving downstream rather than the condition made cleverer.

        A **node** whose interior neighbour is pushing flow inward, at a station
        whose mean is still forward, is overwritten with the same four
        quantities and a density, the one quantity those four leave free, taken
        from the interior: relaxed from its start-of-step value toward the
        current one at a rate that falls away with the local axial Mach number,
        and capped to keep the axial velocity real. There is no
        characteristic split to change at that level -- the split belongs to the
        station's mean, and the Hilbert transform couples every node of a
        station to every other -- so this one is a limiter on the linear theory,
        applied to what reaches the block and kept out of the state the solve
        carries forward.

        The rows that can be prescribed are set independently, by this method
        or :meth:`set_backflow_Po_To` for the thermodynamic pair and by
        :meth:`set_backflow_Vt` for the swirl, so a run can prescribe one and
        leave the rest seeded. The meridional direction is not among them: the
        backflow comes in normal to the exit surface, so the velocity in the
        surface is pinned at zero. See the class docstring.

        Both quantities here are measured from the fluid datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`, the same
        convention as :py:attr:`~ember.block.Block.ho` and
        :py:attr:`~ember.block.Block.s`.

        Calling any of the four is optional. Left alone, the rows are seeded
        once from the pitchwise mean of the exit plane at the first timestep and
        frozen there.

        Parameters
        ----------
        ho : float or array
            Stagnation enthalpy [J/kg]. A scalar or a spanwise profile. Only
            the pitchwise mean at each span station is imposed.
        s : float or array
            Specific entropy [J/(kg K)].

        See Also
        --------
        ember.patch.InletPatch.set_backflow_P : The mirror of this, prescribing
            the pressure an inflow face falls back on
        """
        fluid = self.block.fluid
        self._set_target_row(0, "ho", np.asarray(ho) / fluid.u_ref)
        self._set_target_row(1, "s", np.asarray(s) / fluid.Rgas_ref)

    @replayable
    def set_backflow_Po_To(self, Po, To):
        r"""Prescribe the backflow stagnation state as pressure and temperature.

        Converted to the stagnation enthalpy and entropy of
        :meth:`set_backflow_ho_s` using the fluid of the block this patch is
        attached to, and writing the same two target rows. The prescription is
        what survives, not the conversion: a later change of fluid re-converts
        the pressure and temperature given here against the new one.

        See :meth:`set_backflow_ho_s` for what the backflow rows do.

        Parameters
        ----------
        Po : float or array
            Stagnation pressure :math:`p_0` [Pa]; must be positive and finite.
            A scalar or a spanwise profile, of which only the pitchwise mean at
            each span station is imposed.
        To : float or array
            Stagnation temperature :math:`T_0` [K]; must be positive and finite.
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
        self._set_target_row(0, "Po and To", fluid.get_h(rhoo_nd, uo_nd))
        self._set_target_row(1, "Po and To", fluid.get_s(rhoo_nd, uo_nd))

    @replayable
    def set_backflow_Vt(self, Vt):
        r"""Prescribe the tangential velocity imposed where the exit flow reverses.

        See :meth:`set_backflow_ho_s` for what the backflow rows do.

        Parameters
        ----------
        Vt : float or array
            Tangential velocity :math:`V_\theta` [m/s]. A scalar or a spanwise
            profile, of which only the pitchwise mean at each span station is
            imposed.
        """
        self._set_target_row(3, "Vt", np.asarray(Vt) / self.block.fluid.V_ref)

    @replayable
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
            :attr:`~ember.patch.Patch.shape`.
        """
        arr = np.asarray(P)
        if not np.isfinite(arr).all():
            raise ValueError("P must be finite")
        if not (arr > 0.0).all():
            raise ValueError("P must be positive")
        if self._adjustment and arr.ndim > 0 and arr.size > 1:
            raise ValueError("Non-scalar P is incompatible with the adjustment")
        if self._mdot_target is not None and arr.ndim > 0 and arr.size > 1:
            raise ValueError("Non-scalar P is incompatible with the throttle")
        self._P_raw = arr
        # The level is the whole target until update_target folds in the
        # spanwise adjustment, so a patch driven directly rather than by the
        # solver loop still has the prescribed pressure imposed.
        self._set_target_row(4, "P", arr / self.block.fluid.P_ref)
        self._P_level_nd = np.copy(self._target[..., 4])

    def set_throttle(self, mdot_target, Kp=0.5, Ki=0.002):
        r"""Throttle the outlet to a target mass flow.

        Turns :meth:`set_P` from the condition into a starting point. Each
        timestep :meth:`update_target` measures the mass flow through the patch
        and a proportional-integral controller moves the prescribed level until
        the two agree:

        .. math::

            \varepsilon = \frac{\dot m - \dot m_\mathrm{target}}
                               {\dot m_\mathrm{target}}, \qquad
            \frac{\Delta p_\mathrm{throttle}}{p_\mathrm{ref}} =
                K_p\, \varepsilon
                + K_i \sum \varepsilon \,\mathrm{cfl}

        the sum running over timesteps. Raising the back pressure reduces the
        flow, so the sign is as written: a mass flow above target pushes the
        pressure up. What the boundary imposes is still a pressure, and the
        characteristic treatment is untouched -- the throttle only chooses which
        pressure.

        **The gains are dimensionless and should not need tuning.** For a duct
        or blade row passing :math:`\dot m \sim A\sqrt{2\rho(p_0 - p)}`, the
        steady sensitivity of mass flow to exit pressure is

        .. math::

            \frac{d\dot m}{\dot m} = -\frac{dp}{2q}, \qquad
            q = \tfrac{1}{2}\rho V_m^2

        so a correction of :math:`2q\,\varepsilon` cancels the error outright: a
        Newton step whose natural scale is the exit dynamic head. That is
        exactly the scale the nondimensionalisation already works in, since
        :math:`p_\mathrm{ref} = \rho_\mathrm{ref} V_\mathrm{ref}^2` with
        :math:`V_\mathrm{ref}` a typical convection velocity. Hence the
        correction above is formed nondimensionally with no scale factor
        written anywhere, and :math:`K_p = 1` is the notional Newton step.

        The default is half that, because a pure Newton step overshoots and
        rings: the mass flow answers a change in exit pressure only after a wave
        has crossed the domain. Proportional action alone would then settle at a
        standing droop, since it can hold a correction only in proportion to an
        error, and the correction wanted at the target is not zero; the integral
        is what removes it. Because the scale is a fixed reference quantity
        rather than the dynamic head of the current solution, neither gain
        depends on the flow field or on how good the initial guess was.

        **The integral is weighted by the CFL, not by the step.** The
        proportional term is memoryless and safe under any lag: it holds a fixed
        correction until the flow answers. The integral is not -- over the steps
        the domain takes to respond it keeps piling on correction for an error
        it has already acted on -- so its gain has to be paced against how much
        ground each step covers. Under local timestepping that is the Courant
        number, so a march at twice the CFL needs half as many steps and
        :math:`K_i \sum \varepsilon\,\mathrm{cfl}` keeps one gain valid across a
        CFL sweep. :meth:`ember.grid.Grid.update_bconds` passes the march's cfl
        down; nothing is held on the patch.

        The step count also scales with mesh density, and with whatever
        multigrid and residual smoothing are doing, and none of that is
        knowable from here: a patch can count the cells along its own normal
        but not along the flow path, which for a multi-block machine, or a
        patch that is not on a streamwise face, is not the same number.
        Refining the mesh may therefore want :math:`K_i` revisited. Changing the
        CFL does not.

        The price of the fixed pressure scale is a loop gain of
        :math:`p_\mathrm{ref} / 2q = (\rho_\mathrm{ref}/\rho)
        (V_\mathrm{ref}/V_m)^2` rather than exactly one, so the gains do assume
        the reference scales are representative of the flow. A ``V_ref`` far
        from the exit velocity moves the loop gain by its square, and is the
        one case where these want retuning.

        Only one outlet patch in a grid may be throttled; the solver refuses a
        grid carrying more. Two patches
        driving independent controllers at the same target would each apply the
        full correction for an error they share.

        Parameters
        ----------
        mdot_target : float or None
            Target mass flow :math:`\dot m_\mathrm{target}` [kg/s], through one
            passage rather than the whole annulus, matching what
            :func:`ember.average.flow_mass` returns for this patch. Must be
            positive and finite. Pass None to clear the throttle, which
            **reverts** the boundary to the pressure :meth:`set_P` prescribed:
            the controller's correction is derived from the gains and the error
            sum, so clearing those puts it back at zero on the next
            :meth:`update_target`. Clearing is therefore the inverse of setting,
            and does not depend on the order it is done in relative to
            :meth:`set_P`.

            To keep the operating point the controller found rather than
            revert to the one that was asked for, re-prescribe it first::

                patch.set_P(patch.P_throttle)
                patch.set_throttle(None)

            which is what a run does when it throttles to find an operating
            point and then holds it. See :attr:`P_throttle`.
        Kp : float, optional
            Proportional gain, dimensionless. Default 0.5, half the Newton step
            of 1 above.
        Ki : float, optional
            Integral gain, dimensionless. Default 0.002, from a sweep on a
            square duct at ``cfl=5``: ten times that rings, with six crossings
            of the target at a period of 360 steps, and a third of it never
            arrives. Since the integral only has to remove the droop, erring low
            costs settling time while erring high costs stability.

        See Also
        --------
        ember.patch.OutletPatch.get_throttle_stats : Controller state, as
            logged to the convergence history
        """
        if mdot_target is None:
            self._mdot_target = None
            self._Kp = 0.0
            self._Ki = 0.0
            self._eps_int = 0.0
            self._mdot = 0.0
            return

        if not np.isscalar(mdot_target):
            raise TypeError("mdot_target must be a scalar")
        if not (np.isfinite(mdot_target) and mdot_target > 0.0):
            raise ValueError("mdot_target must be positive and finite")
        if self._P_raw is not None and self._P_raw.ndim > 0 and self._P_raw.size > 1:
            raise ValueError("Non-scalar P is incompatible with the throttle")

        self._mdot_target = float(mdot_target)
        self._Kp = float(Kp)
        self._Ki = float(Ki)
        self._eps_int = 0.0
        self._mdot = 0.0

    def get_throttle_stats(self):
        r"""Return the throttle state, for the convergence history.

        All six values are zero when no throttle is set, which is what
        :meth:`ember.grid.Grid.get_convergence` records for a grid whose outlet
        holds a plain pressure.

        Only the measured mass flow and the running error sum are stored; the correction
        terms below are derived here from them and the gains, so there is no
        second copy of the controller state to keep in step. The values are
        those of the last :meth:`update_target`, so under
        ``Grid.update_bconds(freeze=True)``, which skips it, they stay at the
        last step the controller actually acted on.

        Returns
        -------
        dict
            ``mdot_target`` the setpoint [kg/s]; ``mdot_throttle`` the mass flow
            last measured at the patch [kg/s]; ``dP_throttle`` the total
            correction :math:`\Delta p_\mathrm{throttle}` [Pa]; ``dP_P`` and
            ``dP_I`` its proportional and integral parts [Pa]; ``dP_D`` always
            zero, the controller being PI. The derivative column is retained so
            the ``.cnv`` record layout stays readable in both directions.
        """
        if self._mdot_target is None:
            return dict(
                mdot_target=0.0,
                mdot_throttle=0.0,
                dP_throttle=0.0,
                dP_P=0.0,
                dP_I=0.0,
                dP_D=0.0,
            )

        P_ref = float(self.block.fluid.P_ref)
        eps = self._mdot / self._mdot_target - 1.0
        dP_P = self._Kp * eps * P_ref
        dP_I = self._Ki * self._eps_int * P_ref
        return dict(
            mdot_target=self._mdot_target,
            mdot_throttle=self._mdot,
            dP_throttle=dP_P + dP_I,
            dP_P=dP_P,
            dP_I=dP_I,
            dP_D=0.0,
        )

    def attach_to_block(self, block):
        """Attach to a block and pin the in-surface backflow velocity at zero.

        Done here rather than in a setter because it is not a prescription the
        user makes but a property of the condition; and after the base class,
        which is what allocates the target this writes into. Re-pinned on every
        attach, since a target rebuilt at a new shape comes back zeroed and
        unset.
        """
        super().attach_to_block(block)

        if self._block_ref is None:
            return

        self._target[..., 2] = 0.0
        self._target_set[2] = True

    def update_ref_scales(self):
        r"""Re-derive the prescribed pressure and drop the spanwise adjustment.

        The base class replays :meth:`set_P`, which rebuilds both the level and
        the target row it feeds. What is left is the adjustment relaxation
        state, an integral over a solution and a geometry expressed in the old
        scales: it is dropped, and the next :meth:`update_target` re-derives the
        profile from the rescaled solution.

        The throttle needs nothing done to it. Its error is a ratio of two
        dimensional mass flows and its correction is nondimensional by
        construction, so neither has a dimensional original to return to and the
        wound-out integral carries over intact. The pressure that correction
        comes to does follow the new :math:`p_\mathrm{ref}`, which is the same
        rule the gains obey in the first place; see :meth:`set_throttle`.
        """
        super().update_ref_scales()
        self._P_last_nd = None

    def update_target(self, cfl=1.0):
        r"""Recompute the pressure target for the current timestep.

        Advances the throttle of :meth:`set_throttle` and applies the spanwise
        adjustment of :meth:`set_adjustment`, if either is configured. Should be
        called once per outer timestep before the Runge-Kutta stages;
        :meth:`ember.grid.Grid.update_bconds` does so. The throttle's integral
        advances once per call, so calling this per Runge-Kutta stage instead
        would scale the integral gain with the stage count.

        The two adjustments are orthogonal and simply add: the throttle moves
        the level of the pitchwise-mean pressure, and radial equilibrium shapes
        its spanwise profile about that level.

        Parameters
        ----------
        cfl : float, optional
            CFL number of the march, weighting the throttle's integral so that
            one :math:`K_i` holds across a CFL sweep; see :meth:`set_throttle`.
            Passed down by :meth:`ember.grid.Grid.update_bconds` rather than
            held on the patch, so it is always the number the march is actually
            running at. Default 1, integrating per call, which is all a patch
            stepped by hand outside a march can mean. Unused without a throttle.
        """
        if self._P_level_nd is None:
            # Nothing prescribed yet; apply() reports the missing setter.
            return

        dP_nd = 0.0
        if self._mdot_target is not None:
            # The mass flow is read from the marched face rather than the
            # pitchwise mean in block_avg, so a canted or radial exit plane is
            # integrated as rho*V.dA rather than assumed axial.
            self._mdot = float(average.flow_mass(self.block_view.squeeze()))
            eps = self._mdot / self._mdot_target - 1.0
            # Weighted by the CFL, not by the step: a march at twice the CFL
            # covers the same ground in half the steps, so integrating per step
            # would make the same Ki twice as hot.
            self._eps_int += eps * cfl
            dP_nd = self._Kp * eps + self._Ki * self._eps_int

        if not self._adjustment:
            self._target[..., 4] = self._P_level_nd + dP_nd
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
        self._target[..., 4] = self._P_level_nd + dP_nd + self._P_last_nd

    @property
    def Ki(self):
        """Integral gain of the throttle, dimensionless.

        Zero when no throttle is set; see :attr:`Kp`.
        """
        return self._Ki

    @property
    def Kp(self):
        """Proportional gain of the throttle, dimensionless.

        Zero when no throttle is set, since :meth:`set_throttle` clears both
        gains along with the setpoint; test :attr:`mdot_target` against None to
        tell an unthrottled patch from one deliberately given a zero gain. Set
        through :meth:`set_throttle`, which documents what the value means.
        """
        return self._Kp

    @property
    def mdot_target(self):
        """Throttle setpoint [kg/s], or None when the patch holds a pressure.

        Read by :meth:`ember.grid.Grid.get_convergence`, and by the solver's
        throttle validation, to find the throttled outlet; set through
        :meth:`set_throttle`.
        """
        return self._mdot_target

    @property
    def P(self):
        """Outlet static pressure field as imposed [Pa], shaped like the patch.

        The whole prescription, node by node, read back from :attr:`P_nd`. Not
        the inverse of :meth:`set_P` and not a level: a scalar passed to
        :meth:`set_P` comes back broadcast over the face, :meth:`set_throttle`
        moves what is here away from what was passed, and
        :meth:`set_adjustment` shapes it along the span about a level that is
        then the hub value rather than any average of this.

        For the single number the boundary is holding at, which is what a
        caller recording an operating point wants, see :attr:`P_throttle`.
        """
        return self.P_nd * self.block.fluid.P_ref

    @property
    def P_throttle(self):
        r"""Static pressure level the throttle has arrived at [Pa].

        The inverse of :meth:`set_P`, moved by whatever the controller has done
        to it:

        .. math::

            p_\mathrm{throttle} = p_\mathrm{out} + \Delta p_\mathrm{throttle}

        so with no throttle set this is simply the prescribed pressure, and
        with one it is the operating point the controller has reached. Scalar
        whenever a throttle is set, :meth:`set_throttle` having refused a
        non-scalar prescription; otherwise whatever shape :meth:`set_P` was
        given.

        This is the number to record when a run throttles to an operating point
        and something else has to reproduce it later, and the number to
        re-prescribe to hold that point:

        .. code-block:: python

            patch.set_P(patch.P_throttle)   # keep what the controller found
            patch.set_throttle(None)

        Held apart from :attr:`P` because that is the imposed field rather than
        its level, and the two differ under :meth:`set_adjustment` --- see
        :func:`calc_radial_equilibrium`, whose profile is anchored at the hub,
        so the level is the hub value and not the mean of :attr:`P`.

        See Also
        --------
        ember.patch.OutletPatch.get_throttle_stats : The correction alone, with
            the rest of the controller state
        """
        if self._P_raw is None:
            raise ValueError(
                f"No pressure has been prescribed on this {self._desc}; call "
                "set_P before reading P_throttle."
            )

        return self._P_raw + self.get_throttle_stats()["dP_throttle"]
