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
  characteristic by the non-reflecting relation (Giles Eq. 5.17, Saxer Eq. 56),
  except that entropy and stagnation enthalpy are additionally held uniform
  along the pitch (Giles Eq. 5.22-5.24). Giles adopts that last constraint
  because a straightforward implementation of the linear theory leaves
  second-order variations in entropy and stagnation enthalpy that would be
  comparable with the losses of a viscous calculation.

Unlike :class:`~ember.inlet.InletPatch` this patch shares no state or code with
the pressure-relaxation inlet, and the inflow state is prescribed in the
"natural" variables of the characteristic residuals -- stagnation enthalpy and
entropy rather than stagnation pressure and temperature -- so no thermodynamic
inversion happens inside the boundary condition.

This is the one non-reflecting condition that works in the angle variables of
:func:`~ember.perturbation.chic_to_bcond` rather than the mix variables the base
class defaults to; see the class docstring for why the angles suit a physical
inlet and nothing else.

See Also
--------
ember.nonreflecting.NonReflectingPatch : Base class holding the shared machinery
ember.inlet.InletPatch : Reflecting inlet with pressure relaxation
ember.outlet_nonreflecting.NonReflectingOutletPatch : The outflow counterpart
ember.perturbation.chic_to_bcond : Jacobian the characteristic solves are built on
"""

import numpy as np

from ember import perturbation, util
from ember.nonreflecting import NonReflectingPatch


class NonReflectingInletPatch(NonReflectingPatch):
    r"""Non-reflecting subsonic inflow boundary condition.

    Prescribes stagnation enthalpy :math:`h_0`, entropy :math:`s`, yaw angle
    :math:`\alpha` and pitch angle :math:`\beta` as pitchwise-mean quantities,
    while absorbing outgoing acoustic waves rather than reflecting them. All
    four must be set before :meth:`~ember.nonreflecting.NonReflectingPatch.apply`
    is called, via :meth:`set_ho_s` or :meth:`set_Po_To` together with
    :meth:`set_Alpha` and :meth:`set_Beta`. Each setter converts its target and
    stores it nondimensionally in the corresponding row of the prescribed
    target, published as :attr:`ho_nd`, :attr:`s_nd`, :attr:`tanAlpha` and
    :attr:`sinBeta`, so the patch must already be attached to a block whose
    fluid is set.

    The angles are what makes this condition different from every other
    non-reflecting one. A physical inlet knows its flow angles and not its
    velocity magnitude, so :math:`(\tan\alpha, \sin\beta)` are the right
    variables here, and :attr:`_chic_to_target` is
    :func:`~ember.perturbation.chic_to_bcond` rather than the base class's
    :func:`~ember.perturbation.chic_to_mix`. Only rows 2 and 3 of the two
    Jacobians differ; rows 0, 1 and 4 are identical.

    Each Runge-Kutta stage the characteristic deviation of the face state from
    the frozen pitchwise-mean reference state is formed, and the change required
    in the four incoming characteristics is assembled from three contributions:

    #. a pitchwise-mean change that zeroes the mean residuals in
       :math:`(h_0, s, \tan\alpha, \sin\beta)` in one Newton step
       (Giles Eq. 5.13-5.15), taken by the base class's
       :meth:`~ember.nonreflecting.NonReflectingPatch._calc_dchic_mean`;
    #. harmonic changes setting the tangential vorticity characteristic from
       the outgoing characteristic by the non-reflecting relation, and the
       radial vorticity harmonics to zero (Giles Eq. 5.17, Saxer Eq. 56);
    #. harmonic changes in the entropy and downstream-running pressure
       characteristics that hold entropy and stagnation enthalpy uniform along
       the pitch (Giles Eq. 5.22-5.24).

    Their sum is applied under-relaxed by :attr:`sigma`; the outgoing
    characteristic is carried through untouched.
    """

    _collection_name = "inlet_nonreflecting"

    _desc = "non-reflecting inlet patch"

    # Only the upstream-running pressure wave leaves an inflow plane, so the
    # other four characteristics are incoming and carry the four quantities an
    # inflow prescribes, rows 0-3 of the target.
    _split_fwd = ([1, 2, 3, 4], [0, 1, 2, 3])

    _sign_interior = 1

    # Angles rather than the base class's transverse velocities; see the class
    # docstring. Row 4, the static pressure, is the same in both spaces and is
    # not prescribed here.
    _chic_to_target = staticmethod(perturbation.chic_to_bcond)

    _target_names = ("ho_nd", "s_nd", "tanAlpha", "sinBeta", "P_nd")

    _target_setters = {
        0: "set_ho_s or set_Po_To",
        1: "set_ho_s or set_Po_To",
        2: "set_Alpha",
        3: "set_Beta",
    }

    def _target_from_prim(self, prim):
        """The target-space quantities (ho, s, tanAlpha, sinBeta, P) of a primitive state.

        The angles in place of the base class's transverse velocities, measured
        against the meridional speed as
        :func:`~ember.perturbation.chic_to_bcond` differentiates them.
        """
        ho_nd, s_nd = self._ho_s_from_prim(prim)
        Vx, Vr, Vt = prim[..., 1], prim[..., 2], prim[..., 3]
        Vm = np.sqrt(Vx**2 + Vr**2)
        return ho_nd, s_nd, Vt / Vm, Vr / Vm, prim[..., 4]

    def _calc_dchic(self, dchic, prim):
        """Change in the four incoming characteristics; see the class docstring."""
        ref = self._ref
        target = self._target_from_prim(prim)
        ho_nd, s_nd = target[0], target[1]

        # Mean mode: one Newton step on the four prescribed quantities.
        dchic_mean = self._calc_dchic_mean(target, self._split_fwd, ref["inv_fwd"])

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
        return dchic_new

    def _calc_reference_extra(self, avg, c2t, Mn, Mt, wave):
        """Jacobian of the local Newton step, and the harmonic coefficients."""
        # Local system: stagnation enthalpy and entropy against the entropy and
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

    def set_Alpha(self, Alpha):
        r"""Prescribe the inflow yaw angle.

        Parameters
        ----------
        Alpha : float or array
            Prescribed inflow yaw angle :math:`\alpha` [deg], measured from the
            meridional plane; must satisfy :math:`|\alpha| < 90`. A scalar or an
            array that broadcasts to :attr:`~ember.basepatch.Patch.shape`, of
            which only the pitchwise mean at each span station is imposed.
        """
        if not (np.abs(np.asarray(Alpha)) < 90.0).all():
            raise ValueError("Alpha must be within +/-90 degrees exclusive")
        self._set_target_row(
            2, "Alpha", np.tan(np.radians(np.asarray(Alpha, dtype=np.float32)))
        )

    def set_Beta(self, Beta):
        r"""Prescribe the inflow pitch angle.

        Parameters
        ----------
        Beta : float or array
            Prescribed inflow pitch angle :math:`\beta` [deg]; must satisfy
            :math:`|\beta| \leq 90`. A scalar or an array that broadcasts to
            :attr:`~ember.basepatch.Patch.shape`, of which only the pitchwise
            mean at each span station is imposed.
        """
        if not (np.abs(np.asarray(Beta)) <= 90.0).all():
            raise ValueError("Beta must be within +/-90 degrees inclusive")
        self._set_target_row(
            3, "Beta", np.sin(np.radians(np.asarray(Beta, dtype=np.float32)))
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
            array that broadcasts to :attr:`~ember.basepatch.Patch.shape`, of
            which only the pitchwise mean at each span station is imposed.
        s : float or array
            Prescribed entropy :math:`s` [J/kg/K].
        """
        fluid = self.block.fluid
        self._set_target_row(0, "ho", np.asarray(ho) / fluid.u_ref)
        self._set_target_row(1, "s", np.asarray(s) / fluid.Rgas_ref)

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
            :attr:`~ember.basepatch.Patch.shape`, of which only the pitchwise
            mean at each span station is imposed.
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
        self._set_target_row(0, "Po and To", fluid.get_h(rhoo_nd, uo_nd))
        self._set_target_row(1, "Po and To", fluid.get_s(rhoo_nd, uo_nd))
