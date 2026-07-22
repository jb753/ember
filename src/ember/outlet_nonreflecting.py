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
extrapolates from the interior nor offers throttle control, spanwise pressure
adjustment or backflow handling.

See Also
--------
ember.nonreflecting.NonReflectingPatch : Base class holding the shared machinery
ember.outlet.OutletPatch : Reflecting outlet with prescribed nodal pressure
ember.inlet_nonreflecting.NonReflectingInletPatch : The inflow counterpart
ember.perturbation.chic_to_bcond : Jacobian the characteristic solves are built on
"""

import numpy as np

from ember.nonreflecting import NonReflectingPatch


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
        resid_mean = self._pitch_mean(prim[..., 4] - self.P_nd)
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

    def set_P(self, P):
        r"""Prescribe the outlet static pressure.

        Imposed on the pitchwise mean at each span station, not node by node,
        so a value varying along the pitch is averaged before use; pass a
        spanwise profile to prescribe one, for instance to satisfy radial
        equilibrium in a swirling exit flow.

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
        self.P_nd = self._broadcast_target("P", arr / self.block.fluid.P_ref)
