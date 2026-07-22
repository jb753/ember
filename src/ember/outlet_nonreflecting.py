r"""Non-reflecting subsonic outlet boundary condition for EMBER CFD.

:class:`NonReflectingOutletPatch` prescribes the pitchwise-mean static pressure
at an outflow face while letting outgoing waves leave the domain, after the
steady non-reflecting theory of :cite:t:`Giles1988` (his Section 5.6) extended
to three dimensions by :cite:t:`Saxer1993` (his Section 5.4.5 and Appendix D.3).

The characteristic treatment is entirely
:class:`~ember.nonreflecting.NonReflectingPatch`'s; what this class adds is an
interior on the :math:`-x` side and the pressure a physical exit plane is held
at. Of the five characteristics at an axially subsonic outflow plane four are
outgoing and only the upstream-running pressure wave is incoming, so a single
quantity is prescribed: the pitchwise mean of the static pressure at each span
station.

Unlike :class:`~ember.outlet.OutletPatch` this patch imposes that pressure only
on the pitchwise mean at each span station, and it neither extrapolates from the
interior nor offers throttle control. It does share that patch's
radial-equilibrium adjustment, through the common
:func:`~ember.outlet.calc_radial_equilibrium`; a swirling exit flow needs one,
and prescribing the pitchwise mean at every span station holds the exit plane
off radial equilibrium just as firmly as prescribing it node by node.

It shares that patch's backflow handling too, but reversal means something
different to a condition written in characteristic variables. Reverse a span
station and four of its five characteristics turn incoming, so one prescribed
quantity becomes four and the exit pressure stops being one of them: the base
class switches that station to the inflow split and drives it to rows 0-3 of the
target, in the mix variables :math:`[h_0, s, V_r, V_\theta]` this patch works
in. :meth:`NonReflectingOutletPatch.set_backflow` prescribes them; left alone,
they are seeded once from the pitchwise mean of the exit plane as it stands at
the first timestep and frozen there, which for a run started from a design guess
is the design exit state. Reversal confined to nodes within a station whose mean
still runs forward is handled node by node as a limiter instead, by the base
class's ``_calc_override``.

See Also
--------
ember.nonreflecting.NonReflectingPatch : The condition itself
ember.outlet.OutletPatch : Reflecting outlet with prescribed nodal pressure
ember.inlet_nonreflecting.NonReflectingInletPatch : The inflow counterpart
ember.perturbation.chic_to_mix : Jacobian the mean-mode solves are built on
"""

import numpy as np

from ember.nonreflecting import NonReflectingPatch
from ember.outlet import calc_radial_equilibrium


class NonReflectingOutletPatch(NonReflectingPatch):
    r"""Non-reflecting subsonic outflow boundary condition.

    Prescribes the static pressure :math:`p` as a pitchwise-mean quantity at
    each span station, while absorbing outgoing waves rather than reflecting
    them. It must be set before
    :meth:`~ember.nonreflecting.NonReflectingPatch.apply` is called, via
    :meth:`set_P`, which stores its target nondimensionally in :attr:`P_nd`, so
    the patch must already be attached to a block whose fluid is set.

    Giles takes the mean-mode residual against the flux-averaged pressure; the
    mean here is the weighted pitch mean of
    :attr:`~ember.basepatch.RevolutionPatch.weight_pitch`, the same average
    every other residual in the family is taken against.

    :meth:`set_adjustment` adds a spanwise radial-equilibrium profile to the
    prescribed pressure, re-derived from the solution once per timestep by
    :meth:`update_target`. Without it the prescribed pressure is imposed on
    every span station alike, which for a swirling exit flow fights the
    centrifugal pressure gradient the flow is trying to establish.

    :meth:`set_backflow` prescribes the inflow state a reversed span station is
    driven to; see that method and the module docstring.
    """

    _collection_name = "outlet_nonreflecting"

    _desc = "non-reflecting outlet patch"

    _sign_interior = -1

    _target_setters = {4: "set_P"}

    # The inflow state a reversed station is driven to. Not required of the
    # user: seeded from the exit plane if set_backflow is never called.
    _target_seeded = (0, 1, 2, 3)

    def _copy(self, c):
        super()._copy(c)
        c._P_raw = None if self._P_raw is None else np.copy(self._P_raw)
        c._P_level_nd = None if self._P_level_nd is None else np.copy(self._P_level_nd)
        c._adjustment = self._adjustment.copy()
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
        r"""Prescribe the inflow state imposed where the exit flow reverses.

        Reversal is carried at two levels, and both draw on these four
        quantities.

        A **span station** whose mean has reversed is genuinely an inflow plane
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
        ember.inlet_nonreflecting.NonReflectingInletPatch.set_backflow_P : The
            mirror of this, prescribing the pressure an inflow face falls back on
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
