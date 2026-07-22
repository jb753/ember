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
extrapolates from the interior nor offers throttle control or backflow
handling. It does share that patch's radial-equilibrium adjustment, through the
common :func:`~ember.outlet.calc_radial_equilibrium`; a swirling exit flow needs
one, and prescribing the pitchwise mean at every span station holds the exit
plane off radial equilibrium just as firmly as prescribing it node by node.

See Also
--------
ember.nonreflecting.NonReflectingPatch : Base class holding the shared machinery
ember.outlet.OutletPatch : Reflecting outlet with prescribed nodal pressure
ember.inlet_nonreflecting.NonReflectingInletPatch : The inflow counterpart
ember.perturbation.chic_to_bcond : Jacobian the characteristic solves are built on
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

    Each Runge-Kutta stage the characteristic deviation of the face state from
    the frozen pitchwise-mean reference state is formed, and the change required
    in the single incoming characteristic :math:`c_\mathrm{up}` is assembled
    from two contributions.

    The **mean mode** is one Newton step on the prescribed pressure. Row 4 of
    ``ember.perturbation.chic_to_bcond`` gives
    :math:`\partial p/\partial c_\mathrm{up} = \tfrac{1}{2}` exactly, so the
    step is (Giles Eq. 5.30, Saxer Eq. D.31)

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
    """

    _collection_name = "outlet_nonreflecting"

    _desc = "non-reflecting outlet patch"

    # Everything but the upstream-running pressure wave leaves an outflow plane.
    _idx_out = [1, 2, 3, 4]

    _sign_interior = -1

    _target_setters = {"P_nd": "set_P"}

    def _calc_dchic(self, dchic, prim):
        """Change in the incoming characteristic; see the class docstring."""
        ref = self._ref

        # Mean mode. The residual is evaluated on the state about to be
        # corrected, only the Jacobian is frozen -- a modified Newton step, as
        # at the inflow. Here the Jacobian is the constant dp/dc_up = 1/2, so
        # the step is written out rather than solved for.
        #
        # update_target folds the spanwise adjustment into _P_target_nd once per
        # timestep. Until it has run -- there is no adjustment configured, or
        # this patch is being driven directly rather than by the solver loop --
        # the prescribed level is the whole target.
        P_target = self.P_nd if self._P_target_nd is None else self._P_target_nd
        resid_mean = self._pitch_mean(prim[..., 4] - P_target)
        dchic_mean = -2.0 * resid_mean

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
        dchic_new[..., 0] = dchic_mean + c_up_ideal - (c_up - self._pitch_mean(c_up))
        return dchic_new

    def _calc_reference_extra(self, avg, Mn, Mt, wave):
        """Coefficients of the rationalised harmonic relation, per span station."""
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

    def _copy(self, c):
        super()._copy(c)
        c._P_raw = None if self._P_raw is None else np.copy(self._P_raw)
        c._adjustment = self._adjustment.copy()
        # _P_last_nd and _P_target_nd are derived from the solution, so they are
        # rebuilt by update_target on the new block rather than copied, as _ref
        # is.

    def _setup(self):
        super()._setup()
        # The prescribed pressure as passed, kept so set_adjustment can tell
        # whether it varies along the span.
        self._P_raw = None
        # Empty means no spanwise adjustment; see set_adjustment.
        self._adjustment = {}
        # Relaxation state and the combined target, both built by update_target.
        self._P_last_nd = None
        self._P_target_nd = None

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
        self._P_target_nd = None

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
        self.P_nd = self._broadcast_target("P", arr / self.block.fluid.P_ref)
        self._P_target_nd = None

    def update_target(self):
        """Recompute the pressure target for the current timestep.

        Applies the spanwise adjustment of :meth:`set_adjustment`, if
        configured. Should be called once per outer timestep before the
        Runge-Kutta stages; :meth:`ember.grid.Grid.update_bconds` does so.
        """
        if self.P_nd is None:
            # Nothing prescribed yet; apply() reports the missing setter.
            return

        if not self._adjustment:
            self._P_target_nd = self.P_nd
            return

        if self._adjustment["radial_equilibrium"]:
            profile = self._span_bcast(calc_radial_equilibrium(self).astype(np.float32))
        else:
            profile = np.zeros_like(self.P_nd)

        # Relax, seeding the history with the first profile so the target starts
        # where the flow is rather than crawling out from zero.
        if self._P_last_nd is None:
            self._P_last_nd = profile.copy()
        rf = self._adjustment["rf"]
        self._P_last_nd = rf * profile + (1.0 - rf) * self._P_last_nd
        self._P_target_nd = self.P_nd + self._P_last_nd
