r"""Subsonic inlet boundary condition for EMBER CFD.

:class:`InletPatch` prescribes stagnation enthalpy, entropy and the two flow
angles at an inflow face while letting outgoing acoustic waves leave the domain,
after the steady non-reflecting theory of :cite:t:`Giles1988` (his Sections
5.3-5.4) extended to three dimensions by :cite:t:`Saxer1993`.

The characteristic treatment is entirely
:class:`~ember.patch.NonReflectingPatch`'s; what this class adds is an
interior on the downstream side of the face and the variables a physical inlet
knows. Of the five characteristics at a subsonic inflow face four are incoming
(entropy, two vorticity waves, the downstream-running pressure wave) and one,
the upstream-running pressure wave, is outgoing, so four quantities are
prescribed: the pitchwise means of :math:`h_0`, :math:`s`, :math:`\tan\alpha`
and :math:`\sin\beta`.

The inflow state is prescribed in the "natural" variables of the characteristic
residuals -- stagnation enthalpy and entropy rather than stagnation pressure and
temperature -- so no thermodynamic inversion happens inside the boundary
condition. :meth:`InletPatch.set_Po_To` converts a stagnation state on the way
in for callers who would rather give one.

This is the one condition in the family that works in the angle variables of
:func:`~ember.perturbation.chic_to_bcond` rather than the mix variables the base
class defaults to; see the class docstring for why the angles suit a physical
inlet and nothing else.

See Also
--------
ember.patch.NonReflectingPatch : The condition itself
ember.patch.OutletPatch : The outflow counterpart
ember.perturbation.chic_to_bcond : Jacobian this patch's mean-mode solve is built on
"""

import numpy as np

from ember import perturbation
from ember.nonreflecting import NonReflectingPatch, replayable


class InletPatch(NonReflectingPatch):
    r"""Subsonic inflow boundary condition.

    Prescribes stagnation enthalpy :math:`h_0`, entropy :math:`s`, yaw angle
    :math:`\alpha` and pitch angle :math:`\beta` as pitchwise-mean quantities,
    while absorbing outgoing acoustic waves rather than reflecting them. All
    four must be set before :meth:`~ember.patch.NonReflectingPatch.apply`
    is called, via :meth:`set_ho_s` or :meth:`set_Po_To` together with
    :meth:`set_Alpha` and :meth:`set_Beta`. Each setter converts its target and
    stores it nondimensionally in the corresponding row of the prescribed
    target, published as :attr:`ho_nd`, :attr:`s_nd`, :attr:`tanAlpha` and
    :attr:`sinBeta`, so the patch must already be attached to a block whose
    fluid is set.

    The angles are what makes this condition different from every other in the
    family. A physical inlet knows its flow angles and not its
    velocity magnitude, so :math:`(\tan\alpha, \sin\beta)` are the right
    variables here, and the target space's Jacobian is
    :func:`~ember.perturbation.chic_to_bcond` rather than the base class's
    :func:`~ember.perturbation.chic_to_mix`. Only rows 2 and 3 of the two
    Jacobians differ; rows 0, 1 and 4 are identical.

    A span station whose mean flow has reversed becomes an outflow, and the base
    class drives it to the static pressure of row 4 instead. Nothing need be
    configured for that: :meth:`set_backflow_P` prescribes the pressure and,
    left alone, it is seeded from the inflow plane at the first timestep. The
    angle rows are not solved there -- row 4 is static pressure in both target
    spaces -- so the factor of :math:`V_x` those rows carry never takes the
    solve singular.
    """

    _desc = "inlet patch"

    _sign_interior = 1

    # Angles rather than the base class's transverse velocities; see the class
    # docstring.
    _chic_to_target = staticmethod(perturbation.chic_to_bcond)

    _target_names = ("ho_nd", "s_nd", "tanAlpha", "sinBeta", "P_nd")

    _target_setters = {
        0: "set_ho_s or set_Po_To",
        1: "set_ho_s or set_Po_To",
        2: "set_Alpha",
        3: "set_Beta",
    }

    # The static pressure imposed at a station whose inflow has reversed. Not
    # required: seeded from the face if set_backflow_P is never called.
    _target_seeded = (4,)

    # The node-level limiter imposes rows 0-3 read as [ho, s, Vr, Vt], which
    # this patch's angle rows cannot express. It is also the wrong shape of
    # problem here: a node the flow enters at an inflow face is the normal
    # case, not the pathology the limiter exists for.
    _nodal_backflow = False

    def _target_from_prim(self, prim):
        r"""The target-space quantities (ho, s, tanAlpha, sinBeta, P) of a primitive state.

        The angles in place of the base class's transverse velocities, measured
        against the meridional speed as
        :func:`~ember.perturbation.chic_to_bcond` differentiates them.

        Read in the interface frame, where rows 1 and 2 of ``prim`` are the
        velocity along the frame axis and the one in the surface. The yaw angle
        is unaffected -- :math:`V_m = \sqrt{V_n^2 + V_s^2} = \sqrt{V_x^2 +
        V_r^2}` is invariant under the rotation -- while the pitch angle that
        comes out is measured from the frame axis rather than from :math:`x`.
        The two differ by exactly :attr:`~ember.patch.RevolutionPatch.chi_node`,
        which is where :meth:`set_Beta` puts it back.
        """
        ho_nd, s_nd = self._ho_s_from_prim(prim)
        Vn, Vs, Vt = prim[..., 1], prim[..., 2], prim[..., 3]
        Vm = np.sqrt(Vn**2 + Vs**2)
        return ho_nd, s_nd, Vt / Vm, Vs / Vm, prim[..., 4]

    def _stagnation_state(self):
        """Nondimensional stagnation density and internal energy of the prescribed inflow.

        Shared by :attr:`Po` and :attr:`To` so the two do not each repeat the
        equation-of-state solve :func:`~ember.fluid.PerfectFluid.set_h_s`
        performs.
        """
        fluid = self.block.fluid
        return fluid.set_h_s(self.ho_nd, self.s_nd)

    @replayable
    def set_Alpha(self, Alpha):
        r"""Prescribe the inflow yaw angle.

        Parameters
        ----------
        Alpha : float or array
            Prescribed inflow yaw angle :math:`\alpha` [deg], measured from the
            meridional plane; must satisfy :math:`|\alpha| < 90`. A scalar or an
            array that broadcasts to :attr:`~ember.patch.Patch.shape`, of
            which only the pitchwise mean at each span station is imposed.
        """
        if not (np.abs(np.asarray(Alpha)) < 90.0).all():
            raise ValueError("Alpha must be within +/-90 degrees exclusive")
        self._set_target_row(
            2, "Alpha", np.tan(np.radians(np.asarray(Alpha, dtype=np.float32)))
        )

    @replayable
    def set_backflow_P(self, P):
        r"""Prescribe the static pressure imposed where the inflow reverses.

        A span station whose pitchwise-mean flow has turned round is an outflow:
        four of its five characteristics leave the domain and only one enters,
        so one quantity is prescribed and it is static pressure, not the inflow
        state. This is that pressure. The other four rows are not imposed at
        such a station -- they are what the outgoing waves carry there.

        Calling this is optional. Left alone, the row is seeded once from the
        pitchwise mean of the inflow plane at the first timestep and frozen
        there; see ``NonReflectingPatch._seed_target``. If a large
        part of the span ends up reversed the inflow is no longer under control
        and the boundary wants moving upstream, rather than this value tuning.

        Parameters
        ----------
        P : float or array
            Static pressure :math:`p` [Pa]; must be positive and finite. A
            scalar or an array that broadcasts to
            :attr:`~ember.patch.Patch.shape`, of which only the pitchwise
            mean at each span station is imposed.

        See Also
        --------
        ember.patch.OutletPatch.set_backflow_ho_s : The mirror of this,
            prescribing the inflow state an outflow face falls back on
        """
        arr = np.asarray(P)
        if not np.isfinite(arr).all():
            raise ValueError("P must be finite")
        if not (arr > 0.0).all():
            raise ValueError("P must be positive")
        self._set_target_row(4, "P", arr / self.block.fluid.P_ref)

    @replayable
    def set_Beta(self, Beta):
        r"""Prescribe the inflow pitch angle.

        The angle is measured from the machine axis, :math:`\tan\beta = V_r /
        V_x`, whatever the orientation of the face it is prescribed on. The
        condition itself works in the interface frame, where the same flow
        makes the angle :math:`\beta - \chi` with the frame axis, so what is
        stored is :math:`\sin(\beta - \chi)` for the face angle
        :attr:`~ember.patch.RevolutionPatch.chi_node`. On a face of constant
        :math:`x` the frame axis is :math:`x` and the two coincide.

        What it must satisfy is that flow actually comes in through the face:
        :math:`|\beta - \chi| \leq 90`. On a face of constant :math:`x` that is
        the familiar :math:`|\beta| \leq 90`, and on a radial or reversed face
        it is the same condition said properly -- a duct running along
        :math:`-x` takes :math:`\beta = 180`, one flowing inward radially
        :math:`\beta = -90`.

        Parameters
        ----------
        Beta : float or array
            Prescribed inflow pitch angle :math:`\beta` [deg], measured from
            the machine axis as :math:`\arctan(V_r/V_x)` over the full turn. A
            scalar or an array that broadcasts to
            :attr:`~ember.patch.Patch.shape`, of which only the pitchwise mean
            at each span station is imposed.
        """
        Beta_face = np.radians(np.asarray(Beta, dtype=np.float32)) - self.chi_node
        # Onto (-180, 180] before testing, so an angle that reaches the face
        # the long way round is judged on where it points and not on how it was
        # spelled. The sine stored below is periodic and does not care.
        Beta_face = np.arctan2(np.sin(Beta_face), np.cos(Beta_face))
        # The stored sine cannot tell an inflow from an outflow, so a pitch
        # angle that leaves no velocity along the frame axis to come in on is
        # refused here rather than silently read back as its reflection.
        if not (np.abs(Beta_face) <= 0.5 * np.pi + 1e-5).all():
            worst = float(np.degrees(Beta_face).flat[np.argmax(np.abs(Beta_face))])
            raise ValueError(
                "Beta must be within +/-90 degrees of the face normal, which "
                f"lies at {float(np.degrees(self.chi_node).mean()):.4g} degrees "
                f"to the axis; the prescribed angle is {worst:.4g} degrees from "
                "it, so no flow would enter through the face."
            )
        self._set_target_row(3, "Beta", np.sin(Beta_face))

    @replayable
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
            array that broadcasts to :attr:`~ember.patch.Patch.shape`, of
            which only the pitchwise mean at each span station is imposed.
        s : float or array
            Prescribed entropy :math:`s` [J/kg/K].
        """
        fluid = self.block.fluid
        self._set_target_row(0, "ho", np.asarray(ho) / fluid.u_ref)
        self._set_target_row(1, "s", np.asarray(s) / fluid.Rgas_ref)

    @replayable
    def set_Po_To(self, Po, To):
        r"""Prescribe the inflow stagnation pressure and temperature.

        Converted to the stagnation enthalpy and entropy of :meth:`set_ho_s`
        using the fluid of the block this patch is attached to. The prescription
        is what survives, not the conversion: the pressure and temperature given
        here are kept, and a later change of fluid re-converts them against the
        new one, so this stays the stagnation state asked for rather than
        whatever number the old reference scales and datum made of it.

        Parameters
        ----------
        Po : float or array
            Prescribed stagnation pressure :math:`p_0` [Pa]; must be positive.
            A scalar or an array that broadcasts to
            :attr:`~ember.patch.Patch.shape`, of which only the pitchwise
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

    @property
    def Alpha(self):
        r"""Prescribed inflow yaw angle :math:`\alpha` [deg]. Inverse of :meth:`set_Alpha`."""
        return np.degrees(np.arctan(self.tanAlpha))

    @property
    def Beta(self):
        r"""Prescribed inflow pitch angle :math:`\beta` [deg], measured from the machine axis.

        Inverse of :meth:`set_Beta`: the stored sine is that of the face-frame
        angle :math:`\beta - \chi`, which :meth:`set_Beta` keeps within
        :math:`\pm 90` degrees, exactly :func:`numpy.arcsin`'s range, so the
        recovery is exact up to that wrap.
        """
        Beta_face = np.arcsin(self.sinBeta)
        return np.degrees(Beta_face) + np.degrees(self.chi_node)

    @property
    def Po(self):
        """Prescribed inflow stagnation pressure [Pa].

        Recovered from the currently stored :attr:`ho_nd` and :attr:`s_nd`
        through the block's fluid, so it reads back the prescribed state
        however it was set -- by :meth:`set_Po_To` or by :meth:`set_ho_s` --
        rather than undoing one setter in particular.

        See Also
        --------
        To : The stagnation temperature this state also implies
        """
        rhoo_nd, uo_nd = self._stagnation_state()
        return self.block.fluid.get_P(rhoo_nd, uo_nd) * self.block.fluid.P_ref

    @property
    def To(self):
        """Prescribed inflow stagnation temperature [K]. See :attr:`Po`."""
        rhoo_nd, uo_nd = self._stagnation_state()
        return self.block.fluid.get_T(rhoo_nd, uo_nd) * self.block.fluid.T_ref
