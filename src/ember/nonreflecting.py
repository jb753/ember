r"""Shared machinery for the steady non-reflecting boundary conditions of EMBER CFD.

:class:`NonReflectingPatch` implements the steady non-reflecting inflow and
outflow conditions of :cite:t:`Giles1988` (his Chapter 5), as extended to three
dimensions by :cite:t:`Saxer1993`, as one condition. Its subclasses supply a
side of the face, a set of variables to prescribe and the setters that fill
them; the characteristic treatment itself lives here.

The interface frame
-------------------

The condition is written against the velocity through the boundary, and the
boundary may be any surface of revolution -- a plane of constant :math:`x`, a
cone, an annular bend, or a curved surface whose normal turns from hub to tip.
What makes that possible is a change of frame and nothing else. Write
:math:`\chi` for the angle of the frame axis :math:`n` from :math:`+x` in the
meridional plane
(:attr:`~ember.patch.RevolutionPatch.chi_node`, one value per span node):

.. math::

    V_n &= \cos\chi\, V_x + \sin\chi\, V_r \\
    V_s &= -\sin\chi\, V_x + \cos\chi\, V_r

:meth:`~ember.patch.RevolutionPatch.resolve_to_interface` applies that to the
momentum of the boundary nodes on the way in and
:meth:`~ember.patch.RevolutionPatch.resolve_from_interface` undoes it on the way
out, so everything below reads :math:`V_x` as the velocity through the face and
:math:`V_r` as the one in it. Three things make the substitution exact rather
than approximate: the rotation is orthogonal, so it leaves the Euler equations'
form and the meridional speed :math:`V_m = \sqrt{V_n^2 + V_s^2}` alone; the
pitch direction of a surface of revolution is pure :math:`\theta`, so the
Hilbert transform is untouched; and Saxer's quasi-3D theory already treats each
span station as a two-dimensional cascade in :math:`(n, \theta)` and neglects
derivatives along the third direction, which is what the frame renames. On a
plane of constant :math:`x` the rotation is the identity and is skipped.

**The frame axis points along the mean through-flow**, which is not a free
choice: see the harmonic relations below. An inflow or outflow condition knows
which way that is from its own class, and orientation is then the user's
declaration -- a patch works at any angle and the geometry is not consulted to
second-guess it. A mixing plane cannot know, since both its sides are the same
class and which is upstream is a property of the machine rather than the mesh,
so it settles its frame against the flow on the first step; see
:class:`~ember.mixing.MixingPatch`.

The characteristic split
------------------------

Of the five characteristics at a subsonic boundary, four propagate downstream
(entropy, both vorticity waves, the downstream-running pressure wave) and one,
the upstream-running pressure wave, propagates upstream. A characteristic is
*outgoing* -- owned by the interior march, read from the boundary node exactly
as the scheme left it and never overwritten, so a wave reaching the boundary
passes through -- when its wave speed carries it out of the domain,
:math:`\lambda\,n < 0` for the inward face normal. The rest are *incoming*:
discarded rather than taken from the march, and rebuilt once a timestep from the
prescribed mean state and the non-reflecting relations, under-relaxed by
:attr:`~NonReflectingPatch.sigma`.

Working that out for the wave speeds
:math:`[V_n - a,\, V_n + a,\, V_n,\, V_n,\, V_n]` of
:math:`[c_\mathrm{up}, c_\mathrm{down}, c_r, c_t, c_s]` gives four splits, not
two, and they have a simple structure: **the acoustic split is fixed by the
geometry** -- :math:`c_\mathrm{up}` is always outgoing when the interior lies on
the :math:`+n` side, :math:`c_\mathrm{down}` always outgoing when it lies on the
:math:`-n` side -- **and the three convective characteristics follow the flow**,
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
per span station, in the space the patch maps characteristics into. Rows 0, 1
and 4 are stagnation enthalpy, entropy and static pressure in every such space;
only rows 2 and 3 distinguish them, the mix variables
:math:`(V_s, V_\theta)` of :func:`~ember.perturbation.chic_to_mix` from the
angles :math:`(\tan\alpha, \sin\beta)` of
:func:`~ember.perturbation.chic_to_bcond`. Both live in the interface frame,
where :math:`V_s` is the velocity in the surface and :math:`\beta` is measured
from the frame axis; the setters that fill them take machine-frame quantities
and convert, so :math:`\tan\beta = V_r/V_x` still means what it says
(:meth:`~ember.inlet.InletPatch.set_Beta`), and the one quantity that cannot be
converted -- a meridional velocity, which needs the normal component to resolve
and that is what the solve derives -- is not offered as a setter at all
(:class:`~ember.outlet.OutletPatch`). One span station rather than one node
loses nothing: every target is read only through the pitch mean of its own
residual, which is linear, so the pitch mean of a prescribed profile is all
that was ever imposed.

The harmonic relations are the one place the two directions genuinely differ,
and each patch needs only one of them. Giles and Saxer derive them for mean flow
along the frame axis, so a relation applies only where :math:`V_n > 0` -- and
there, entering implies an inward normal of :math:`+1` and leaving one of
:math:`-1`. A patch's normal is fixed, so exactly one relation is ever live on
it and the other kind of station takes zeroed harmonics, which is the honest
thing to do where the derivation does not hold.

The two rows of the table that zero them are exactly the two with
:math:`V_n < 0`, so this is a statement about reversal and not about
orientation: a station running with the frame absorbs, one running against it
does not. That is why the frame axis has to point along the mean through-flow.
Pointed the other way it would still balance the mean -- the mean-mode Newton
step is indifferent to it -- while zeroing the harmonics at *every* station, so
the boundary would go quietly reflective rather than fail.

See Also
--------
ember.patch.RevolutionPatch : Base class providing the pitchwise geometry
ember.patch.InletPatch : Subsonic inflow
ember.patch.OutletPatch : Subsonic outflow
ember.patch.MixingPatch : Either side of an interface
ember.perturbation.chic_to_mix : Jacobian the characteristic solves are built on
"""

import contextlib
import functools
import warnings

import numpy as np

import ember.fortran
from ember import perturbation, util
from ember.basepatch import RevolutionPatch


# Numpy error state for a mean state already reported outside the implemented
# envelope, and the do-nothing stand-in for the ordinary path; see
# NonReflectingPatch._calc_reference.
_INVALID_IGNORED = {"invalid": "ignore", "divide": "ignore", "over": "ignore"}
_NULL_CONTEXT = contextlib.nullcontext()


class UnsupportedMeanStateWarning(UserWarning):
    """A characteristic boundary's mean state has left the implemented envelope.

    Issued when the frozen mean state at the boundary goes supersonic, where
    the steady non-reflecting theory these conditions implement does not hold
    and what they compute is meaningless.

    A warning rather than an error because the usual way to get here is a march
    already diverging, which the solver detects and reports for itself through
    :meth:`ember.grid.Grid.check_nan`; raising from the boundary condition
    pre-empted that with an exception and lost the trimmed convergence history.
    Given its own class so a run that expects the excursion can filter it, and
    a run that does not can turn it into an error.
    """


def calc_backflow_rho(fluid, snapshot, rho_soln_nd, rho_nd, Max, rf):
    r"""Relaxed boundary density for reversed-flow nodes, capped to keep :math:`V_x` real.

    A reversed node takes its stagnation enthalpy, entropy and transverse
    velocities from the prescribed backflow state, which leaves density as the
    one quantity still free to come from the interior. It
    is relaxed from the start-of-step value toward the current one at a rate
    that falls away with the local axial Mach number,

    .. math::

        \rho^\mathrm{new} = \rho^n
            + \min\left(\mathit{rf}\,\left|M_x\right|,\, 0.8\right)
              \left(\rho - \rho^n\right),

    then capped just below the density at which the static enthalpy would reach
    :math:`h_0 - \tfrac{1}{2}(V_r^2 + V_\theta^2)`. Static enthalpy rises with
    density at fixed entropy, so under that cap the radicand of the axial
    velocity the caller recovers from the energy equation,
    :math:`V_x = \sqrt{2(h_0 - h) - V_r^2 - V_\theta^2}`, is non-negative.

    Parameters
    ----------
    fluid : ember.fluid.Fluid
        Fluid the equation of state is evaluated on.
    snapshot : sequence of array
        Nondimensional ``[ho, s, Vr, Vt]`` to impose, taken from the target
        rows; they need only broadcast against ``rho_nd``.
    rho_soln_nd : array
        Nondimensional density at the start of the step, the anchor the
        relaxation runs from.
    rho_nd : array
        Nondimensional density the relaxation runs toward.
    Max : array
        Axial Mach number, setting the local relaxation rate.
    rf : float
        Relaxation factor.

    Returns
    -------
    array
        Nondimensional density, of the shape the inputs broadcast to. Computed
        over the whole array; the caller selects the reversed nodes.

    See Also
    --------
    ember.patch.OutletPatch.set_backflow_ho_s : Prescribes the state this relaxes against
    """
    ho_snap, s_snap, Vr_snap, Vt_snap = snapshot
    h_max_nd = ho_snap - 0.5 * (Vr_snap**2 + Vt_snap**2)
    rho_cap_nd = fluid.set_h_s(h_max_nd, s_snap)[0]
    rho_new_nd = rho_soln_nd + np.minimum(rf * np.abs(Max), 0.8) * (
        rho_nd - rho_soln_nd
    )
    return np.minimum(rho_new_nd, 0.9999 * rho_cap_nd)


def replayable(setter):
    """Record a target setter's raw arguments so it can be replayed.

    Decorates the public setters that convert a dimensional quantity into a
    target row. The arguments are kept exactly as the caller passed them --
    physical units, unconverted -- so that
    :meth:`NonReflectingPatch.update_ref_scales` can re-run the setter against a
    new fluid and get the target the caller asked for, rather than the old
    number rescaled. Only reconversion is correct in general: a change of datum
    shifts enthalpy and entropy affinely, which no scale factor can express.

    Recording the call rather than the value keeps every conversion written once
    in its own setter, including the coupled two-step one of
    :meth:`~ember.patch.InletPatch.set_Po_To`, whose
    two rows do not decompose into independent per-row conversions.

    The record is taken only once the setter returns, so a rejected value leaves
    nothing to replay, and re-recording moves a setter to the end of the replay
    order, so replay reproduces the order the caller actually set things in.
    """

    @functools.wraps(setter)
    def wrapper(self, *args, **kwargs):
        setter(self, *args, **kwargs)
        self._target_calls.pop(setter.__name__, None)
        self._target_calls[setter.__name__] = (args, kwargs)

    return wrapper


class _TargetRow:
    """Read-only view of one row of a patch's prescribed target vector.

    A descriptor rather than a plain attribute, so that the named rows stay
    views on ``NonReflectingPatch._target`` with nothing to re-link when a
    patch is copied or unpickled, and so that a name the patch's target space
    does not carry raises rather than quietly returning whatever that row holds:
    an inflow condition working in angles has no ``Vr_nd``, and one working in
    mix variables has no ``tanAlpha``.

    Resolution is by name against ``NonReflectingPatch._target_names`` of
    the instance, not by a fixed index, because the row order is a property of
    the target space and the classes do not share one.
    """

    def __set_name__(self, _owner, name):
        self._name = name

    def __get__(self, obj, _objtype=None):
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

    Subclasses declare a description for error messages, the inward face
    normal (or leave the geometry to decide it), the target space's name and
    Jacobian, a mapping of each required target row to the setter that fills
    it, and which rows are seeded from the flow when nothing prescribes them.
    They add the setters themselves and nothing else: the characteristic
    treatment, both harmonic relations and the reversed-flow handling are all
    here.

    :meth:`update_soln` and :meth:`advance` are called once per timestep, the
    first refreshing the reference state to match Giles' definition of the
    characteristic variables as perturbations about the time-level-:math:`n`
    average, the second taking the condition's one under-relaxed step on it.
    :meth:`apply` is called once per Runge-Kutta stage and only imposes what
    those two settled, so the rate of the condition does not scale with the
    stage count.

    The face may be any surface of revolution; the condition works in the
    interface frame and the module docstring says how. What it is restricted to
    is a mean state subsonic both normal to the face and absolutely, which is
    checked and warned about. That restriction does not concern the *direction*
    of the flow through the face: a span station whose mean has reversed simply
    takes the other characteristic split, and drives the quantities that split
    prescribes toward rows of the same target.
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

    sigma = 0.05
    """Under-relaxation of the characteristic correction, Giles Eq. 5.25,
    needed for wellposedness. He suggests 1/N for N pitchwise nodes, applied
    once per timestep, and :meth:`advance` takes it exactly once per timestep,
    so the two are in the same units: set it to 1/N and it is 1/N. The bound
    is not about the transform amplifying -- it cannot, its norm grows only
    logarithmically -- but about how far the pitchwise-nonlocal harmonic
    relations may spread information in one application while the explicit
    interior march moves it one cell. Overridden by
    :attr:`~ember.solver.Solver.rf_inlet`/:attr:`~ember.solver.Solver.rf_outlet`
    at the start of a run."""

    # Inward face normal: +1 if the interior lies on the +n side of the face,
    # -1 if on the -n side, for the interface frame axis n. Equivalently, +1
    # where the flow nominally enters through the face and -1 where it leaves,
    # since the frame axis runs downstream. A value here is the authority and
    # the frame is built to match it; None means the class cannot know, and
    # leaves _check_face to take a provisional answer from the geometry for a
    # subclass to settle against the flow.
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
        :func:`~ember.nonreflecting.calc_backflow_rho` and :meth:`_calc_override` read
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
            self._pitch_mean(self._normal_momentum_offset_1())
            / self._pitch_mean(cons[..., 0])
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
        weights :attr:`~ember.patch.RevolutionPatch.weight_pitch` gives

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
        return self._span_bcast(np.linalg.inv(jac))

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

    @util.profile
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
        inflow = self._normal_momentum_offset_1() * self._sign_interior > 0.0
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
        # Filled in place (not reallocated) so the fused kernel in _recombine
        # reads a contiguous float32 array every stage without a repeated
        # implicit copy; refreshed once per timestep, not per stage. Plain
        # broadcasting assignment, not np.broadcast_to: the buffer is a real
        # owned array (see attach_to_block), so this is a write, not a view.
        self._mask_out_bcast[...] = self._mask_out

        # Tested on the magnitude, so a station running backwards fast enough
        # to be supersonic normal to the face is caught too: there one of the two acoustic
        # characteristics changes direction and even the reversed split is
        # wrong.
        #
        # Warned rather than raised, and the step taken anyway. The condition
        # is genuinely not implemented above Mach 1 and what it computes there
        # is meaningless -- the wave parameter goes imaginary and the state
        # turns to NaN within a step or two -- but the common way to arrive
        # here is a march on its way to blowing up, and that is the solver's
        # divergence to report, through Grid.check_nan, not the boundary
        # condition's to pre-empt. Raising took a run that would have exited
        # cleanly with a trimmed history and killed it with an exception
        # instead. A case that is supersonic by design gets the same warning on
        # its first step, which says plainly what is wrong.
        unsupported = True
        if np.any(np.abs(Mn) >= 1.0):
            self._warn_unsupported(
                f"is supersonic normal to the face (max normal Mach "
                f"{float(np.max(np.abs(Mn))):.4g}); only a normally subsonic "
                "mean state is implemented"
            )
        elif np.any(Msq >= 1.0):
            self._warn_unsupported(
                f"has a supersonic mean state (max Mach "
                f"{float(np.sqrt(np.max(Msq))):.4g}); the supersonic branch of "
                "the wave parameter is not implemented"
            )
        else:
            # Back inside the envelope, so a later excursion is news again.
            self._warned_unsupported = False
            unsupported = False

        # Past Mach 1 the wave parameter below takes the square root of a
        # negative number and the Jacobians go singular, so the rest of this
        # runs on invalid values by construction. The warning above is the
        # report; numpy's per-operation RuntimeWarnings on top of it are noise,
        # and a diverging march would emit them every step from deep inside the
        # linear algebra. Suppressed only on the branch that has already warned,
        # so the ordinary path still surfaces an unexpected invalid value.
        with np.errstate(**_INVALID_IGNORED) if unsupported else _NULL_CONTEXT:
            self._calc_reference_tail(avg, Mn, Mt, Msq)

    def _calc_reference_tail(self, avg, Mn, Mt, Msq):
        """Build the frozen Jacobians and wave parameter of :meth:`_calc_reference`.

        Split out so the caller can wrap it in the error state a mean state
        outside the implemented envelope needs, without indenting the whole
        body behind a conditional context manager.
        """
        c2t = self._chic_to_target(avg)
        # Filled in place into buffers sized once in attach_to_block, instead
        # of allocating prim/p2c/c2p fresh every timestep; _span_bcast's
        # reshape is a view of the same buffer, not a copy, so this remains
        # zero-allocation on repeat calls.
        np.stack(
            (avg.rho_nd, avg.Vx_nd, avg.Vr_nd, avg.Vt_nd, avg.P_nd),
            axis=-1,
            out=self._ref_prim_buf,
        )
        perturbation.primitive_to_chic(avg, out=self._ref_p2c_buf)
        perturbation.chic_to_primitive(avg, out=self._ref_c2p_buf)
        self._ref = {
            "prim": self._span_bcast(self._ref_prim_buf),
            "p2c": self._span_bcast(self._ref_p2c_buf),
            "c2p": self._span_bcast(self._ref_c2p_buf),
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
            "inv_local": self._span_bcast(np.linalg.inv(jac_local)),
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
        # conjugate; it is bounded away from zero by the normally subsonic check
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

    def _check_face(self):
        """Validate the boundary surface and settle the interface frame on it.

        The face may be any surface of revolution, so the frame axis
        :math:`n` is derived from the geometry rather than assumed to be
        :math:`x`; see the module docstring for why it has to point along the
        mean through-flow. Which side of the face the interior lies on decides
        which way round that is, and a patch whose class fixes
        :attr:`_sign_interior` -- an inflow or an outflow condition, which knows
        by construction whether the flow comes in or goes out -- says so
        directly. The geometry is not consulted to second-guess it: on a face
        of any orientation the patch a user put there is the declaration of
        what that face is.
        """
        # The fused kernel of _recombine broadcasts the reference state along
        # the span, and only the j and k variants of it are built. A face of
        # constant i can never span along i, so this was unreachable while the
        # condition was restricted to planes of constant x; a face of constant
        # j or k on a general surface of revolution can.
        if self.span_dim == 0:
            raise NotImplementedError(
                f"{self._desc.capitalize()} {self.label!r} spans along i, which "
                "the characteristic reconstruction has no kernel for; put the "
                "patch on a face of constant i, or add "
                "nonreflecting_recombine_bcast_i."
            )

        block = self.block
        Lref = max(np.ptp(block.x), np.ptp(block.r))
        inward = self._inward_meridional()
        if np.linalg.norm(inward, axis=-1).max() <= self._rtol_geom * Lref:
            raise ValueError(
                f"{self._desc.capitalize()} {self.label!r} cannot tell which "
                "side its interior lies on: the first interior layer lies in "
                "the face."
            )

        # Read off the class, not the instance, so that a patch whose class
        # leaves the side to the geometry can be re-attached to the other side
        # of a face rather than pinned to its own previous answer.
        fixed = type(self)._sign_interior
        if fixed is not None:
            self._sign_interior = fixed
            self._build_rot_matrices(inward=fixed > 0)
            return

        # Nothing on the class, so take a provisional frame from the geometry;
        # a subclass that can settle it properly does so from the flow, once
        # there is one. Built inward first so the normal that decides is the
        # one _build_rot_matrices derives and flips, not the raw offset.
        self._build_rot_matrices(inward=True)
        self._sign_interior = self._provisional_sign()
        if self._sign_interior < 0:
            self._build_rot_matrices(inward=False)

    def _provisional_sign(self):
        """Which side of the face the interior is on, from the geometry alone.

        Used only by a patch whose class leaves :attr:`_sign_interior` open and
        which has no flow yet to settle it against. The rule is to make the
        frame axis the face normal that points along :math:`+x`, falling back
        to :math:`+r` where the face has no axial normal component at all --
        deterministic, and on a plane of constant :math:`x` exactly the rule
        the condition has always used.

        What it guarantees is what the mixing plane needs of it before there is
        a flow: the two sides of one interface have antiparallel normals, so
        they are given opposite signs whatever their orientation.

        Returns
        -------
        int
            ``+1`` or ``-1``.
        """
        normal = self._rot_to[..., 0, :].reshape(-1, 2)
        n_x = float(normal[:, 0].mean())
        if abs(n_x) > self._rtol_geom:
            return 1 if n_x > 0.0 else -1
        return 1 if float(normal[:, 1].mean()) > 0.0 else -1

    def _copy(self, c):
        c._target = None if self._target is None else np.copy(self._target)
        c._target_set = self._target_set.copy()
        c._target_calls = dict(self._target_calls)
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

    def _normal_momentum_offset_1(self):
        r"""Momentum along the frame axis in the first interior layer.

        The two tests that ask which way the interior is pushing flow --
        :meth:`_calc_entering` and :meth:`_calc_override` -- read the layer one
        node in from the face, and that layer is *not* part of
        :attr:`~ember.patch.RevolutionPatch.block_view`, so it is still in
        :math:`(x, r)` coordinates while the face around them has been rotated
        into the interface frame. Projecting it here is what keeps the two
        comparable:

        .. math::

            \rho V_n = \cos\chi\, \rho V_x + \sin\chi\, \rho V_r

        Reads the projection off :attr:`_rot_to` rather than recomputing a
        cosine, and off the raw conserved array rather than
        :attr:`~ember.block.Block.Vx_nd`, so nothing here depends on the
        interior layer's primitives being current.

        Returns
        -------
        array
            Shape ``block_view.shape``.
        """
        cons = self.block_view_offset_1.conserved_nd
        if self._rot_identity:
            return cons[..., 1]
        rot = self._rot_to
        return rot[..., 0, 0] * cons[..., 1] + rot[..., 0, 1] * cons[..., 2]

    @staticmethod
    def _mask_from_split(split):
        """Length-5 boolean mask of the characteristics a split leaves outgoing."""
        mask = np.ones(5, dtype=bool)
        mask[list(split[0])] = False
        return mask

    def _interp_profile(self, value, src, spf_src):
        """Re-express one recorded setter argument on this patch's span stations.

        Returns ``value`` unchanged unless it is a spanwise profile on ``src``:
        either bare, of shape ``(nspan_src,)``, or on ``src``'s own patch axes
        with its span axis that long -- the two shapes
        :meth:`_set_target_row` accepts. Anything else is a scalar, or a value
        that setter will reject on its own terms, and is not this method's to
        reinterpret.
        """
        arr = np.asarray(value)
        if arr.size == 1 or not np.issubdtype(arr.dtype, np.number):
            return value
        nspan_src = len(spf_src)
        if arr.shape == (nspan_src,):
            profile = arr
        elif arr.ndim == 3 and arr.shape[src.span_dim] == nspan_src:
            # Patch-axes form, so the other two axes are length 1 (a
            # pitchwise-varying value never made it past the setter).
            profile = np.moveaxis(arr, src.span_dim, 0).reshape(nspan_src)
        else:
            return value
        interp = np.interp(self.spf, spf_src, profile)
        # Back onto this patch's own axes rather than left bare: the two spell
        # the same prescription, and this way a patch whose span axis moved in
        # the resample still reads as spanwise.
        shape = [1, 1, 1]
        shape[self.span_dim] = len(interp)
        return interp.reshape(shape)

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

    def _replay_target_calls(self):
        """Re-run every recorded :func:`replayable` setter, in call order.

        The record is rebuilt as the setters re-record themselves, so the
        replay order survives any number of replays.
        """
        calls, self._target_calls = self._target_calls, {}
        for name, (args, kwargs) in calls.items():
            getattr(self, name)(*args, **kwargs)

    def _seed_target(self):
        """Fill any seeded target row nothing has prescribed, once.

        The rows of :attr:`_target_seeded` are taken from the pitchwise mean of
        the face as it stands the first time a solve needs them -- the initial
        condition, in a run that starts from one -- and frozen there. Freezing
        matters: a row re-derived from the face every step would drive its
        residual to zero and leave the quantity with no condition on it at all.

        Read off :attr:`~ember.patch.Patch.block_view` and pitch-averaged
        here rather than taken from
        :attr:`~ember.patch.RevolutionPatch.block_avg`, which a mixing
        exchange overwrites with the symmetrised cross-plane average, or from
        ``block_view.mean()``, whose derived properties read as zero before the
        conserved cache is primed.

        Takes its own rotation window rather than relying on a caller's. The
        target lives in interface coordinates, and this is reached both from
        inside the condition, where the window is already open, and from
        outside it -- a mixing plane seeds itself from
        :meth:`~ember.mixing.MixingPatch.get_target`, which the communicator
        calls before any boundary condition has run. Seeding from outside
        without one filled the in-surface velocity row with the radial velocity
        instead, which on a canted face is a different quantity.
        """
        rows = [row for row in self._target_seeded if not self._target_set[row]]
        if not rows:
            return
        with self._resolved():
            b = self.block_view
            target = self._target_from_prim(
                np.stack((b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1)
            )
            for row in rows:
                self._target[..., row] = self._pitch_mean(target[row])
                self._target_set[row] = True

    def _set_target_row(self, row, name, value):
        r"""Check a prescribed value against the patch shape and store it in a target row.

        A target row is one number per span station: these conditions impose
        pitchwise means and nothing finer. So the only values that mean anything
        are a scalar, uniform over the whole face, and a spanwise profile, one
        value per span station.

        A pitchwise profile is rejected rather than averaged. Averaging it would
        take the value silently, impose its mean, and discard the variation the
        caller asked for -- the prescription would read as node-by-node and
        behave as a mean. The narrower shapes are unambiguous, so the caller is
        made to pick one.

        Accepted shapes, given a patch shape with the constant dimension of
        length 1:

        - a scalar, or any array of size 1;
        - a bare 1-D array of length ``nspan``, the plain way to write a
          spanwise profile;
        - the same profile with the patch's own axes,
          e.g. ``(1, nspan, 1)`` for a patch whose ``span_dim`` is 1.

        The 1-D form is unambiguous *because* pitchwise variation is refused: a
        one-dimensional prescription has nothing else it could mean. The one
        mesh where a caller could still be surprised is a patch with as many
        pitchwise nodes as span stations, where an array meant pitchwise has the
        right length to be read as spanwise; there is no shape-based way to tell
        those apart, and the alternative -- refusing the natural spelling on
        every mesh to guard the one -- costs more than it saves.
        """
        arr = np.asarray(value)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")

        want = list(self.block_view.shape)
        want[self.pitch_dim] = 1
        want = tuple(want)
        nspan = self.block_view.shape[self.span_dim]

        if arr.size == 1:
            pass
        elif arr.shape == (nspan,):
            # Onto the patch's own axes, so the broadcast below puts it along
            # the span rather than wherever trailing-axis alignment lands it.
            arr = arr.reshape(want)
        elif arr.shape != want:
            raise ValueError(
                f"{name} of shape {arr.shape} is not a valid prescription for "
                f"{self._desc} {self.label!r}: give a scalar, a spanwise "
                f"profile of shape ({nspan},), or the same with the patch's own "
                f"axes, {want} (span_dim={self.span_dim}). Only the pitchwise "
                "mean at each span station is imposed, so a pitchwise-varying "
                "value is rejected rather than averaged."
            )

        # Broadcast rather than assign: a scalar has to reach every span
        # station, and _pitch_mean expects a full patch-shaped field.
        bcast = np.broadcast_to(arr, self.block_view.shape)
        self._target[..., row] = self._pitch_mean(bcast)
        self._target_set[row] = True

    def _setup(self):
        super()._setup()
        # Prescribed boundary state, one nondimensional five-vector per span
        # station in the space _chic_to_target maps into, allocated on attach;
        # and which of its rows have been filled.
        self._target = None
        self._target_set = np.zeros(5, dtype=bool)
        # The @replayable setter calls that filled those rows, keyed by setter
        # name and held in the order they were made, with the arguments as the
        # caller gave them: dimensional, unconverted. Replayed by
        # update_ref_scales when the reference scales move under them.
        self._target_calls = {}
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
        # Output scratch buffers for _recombine's fused Fortran kernel, sized
        # on attach so the per-stage hot path allocates nothing; see
        # attach_to_block.
        self._recombine_dchic = None
        self._recombine_prim = None
        # mask_out broadcast to the full span_dim-broadcast shape (matching
        # _ref["p2c"]/["prim"]) and cast to float32, so the fused kernel can
        # read it directly without a per-stage broadcast/copy. Refilled
        # in-place once per timestep in _calc_reference, alongside
        # _mask_out itself; sized in attach_to_block.
        self._mask_out_bcast = None
        # Unbroadcast (nspan, 5)/(nspan, 5, 5) buffers _calc_reference fills
        # in-place every timestep for _ref["prim"]/["p2c"]/["c2p"], instead of
        # allocating fresh arrays each time. _span_bcast reshapes these (a
        # view, not a copy) into the broadcast shape _ref actually stores.
        # Sized in attach_to_block from block_avg's span count.
        self._ref_prim_buf = None
        self._ref_p2c_buf = None
        self._ref_c2p_buf = None
        # Start-of-step density the reversed-node relaxation runs from, taken
        # by update_soln.
        self._rho_nd_soln = None
        # Whether the mean state has already been reported outside the
        # implemented envelope, so the warning is one per excursion rather than
        # one per timestep; cleared when it comes back inside.
        self._warned_unsupported = False

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

    def _warn_unsupported(self, clause):
        """Report a mean state outside the implemented envelope, once per excursion.

        See :class:`UnsupportedMeanStateWarning` for why this warns rather than
        raises. ``stacklevel`` is left at the default: the useful frame is the
        solver loop, which is many frames up and varies, so the patch's own
        line is as good a place to point as any.
        """
        if self._warned_unsupported:
            return
        self._warned_unsupported = True
        warnings.warn(
            f"{self._desc.capitalize()} {self.label!r} {clause}. Continuing; "
            "the boundary state from here is not meaningful and the march will "
            "most likely diverge.",
            UnsupportedMeanStateWarning,
        )

    @util.profile
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

        if self._prim_prev is None:
            self._prim_prev = np.stack(
                (b.rho_nd, b.Vx_nd, b.Vr_nd, b.Vt_nd, b.P_nd), axis=-1
            ).copy()

        # Resolved by span_dim on every call rather than cached on attach: a
        # cached bound reference to an f2py-wrapped subroutine isn't
        # picklable (Grid.write_emb pickles the whole grid, patches
        # included), and the lookup itself -- a module attribute access -- is
        # negligible next to the kernel call it selects.
        kernel = (
            ember.fortran.nonreflecting_recombine_bcast_j
            if self.span_dim == 1
            else ember.fortran.nonreflecting_recombine_bcast_k
        )
        kernel(
            b.rho_nd,
            b.Vx_nd,
            b.Vr_nd,
            b.Vt_nd,
            b.P_nd,
            self._prim_prev,
            ref["prim"],
            ref["p2c"],
            ref["c2p"],
            self._mask_out_bcast,
            self._recombine_dchic,
            self._recombine_prim,
        )
        return self._recombine_dchic, self._recombine_prim

    def advance(self):
        r"""Take the boundary condition's one step; call once per timestep.

        The change in the incoming characteristics is scaled by :attr:`sigma`,
        which is exactly Giles' Eq. 5.25 correction. This is the whole of a
        timestep's boundary-condition change: :meth:`apply` only imposes the
        result, once per stage.

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
        with self._resolved():
            if self._ref is None:
                self._calc_reference()

            dchic, prim = self._recombine()
            self._prim_prev = prim + self.sigma * util.matvec(
                self._ref["c2p"], self._calc_dchic(dchic, prim)
            )

    @util.profile
    def apply(self):
        r"""Impose the non-reflecting condition on the patch.

        Called once per Runge-Kutta stage, and imposes only: the outgoing
        characteristics are re-read from the marched face every stage so a wave
        reaching the boundary still passes through within the step, while the
        incoming ones are the state :meth:`update_soln` last authored. The
        :attr:`sigma`-relaxed correction that advances that state is taken there,
        once per timestep, not here.

        A node-level override is then given the chance to change what actually
        reaches the block, and its result is deliberately not carried back into
        the state the solve is still working from, so a condition that has to
        depart from its own linear theory somewhere does not thereby corrupt
        the characteristic state it is still solving on.
        """
        if not self._target_set[list(self._target_setters)].all():
            self._raise_unset()
        with self._resolved():
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
        re-attachment, and one of the wrong shape is rebuilt at the new shape
        rather than silently misread -- every prescribed row by re-running the
        setter that filled it, the rest by re-seeding. Replay is at the
        original arguments, so a prescribed spanwise profile reaches the setter
        at the length it was set on and is refused if the span station count
        has moved. Going onto a coarser grid, as the multigrid hierarchy and
        :func:`~ember.block_util.resample` do it, therefore goes through
        :meth:`attach_to_block_resampled`, which interpolates those profiles
        onto the new stations first.
        """
        super().attach_to_block(block)

        if self._block_ref is None:
            return

        self._check_face()
        self._split_entering = self._calc_split(True)
        self._split_leaving = self._calc_split(False)
        self._calc_hilbert()

        shape = self._target_shape()
        if self._target is None or self._target.shape != shape:
            self._target = util.zeros(shape)
            self._target_set = np.zeros(5, dtype=bool)
            self._replay_target_calls()

        if self._mask_out_bcast is None or self._mask_out_bcast.shape != shape:
            self._mask_out_bcast = util.zeros(shape)

        nspan = self._block_view.shape[self.span_dim]
        if self._ref_prim_buf is None or self._ref_prim_buf.shape[0] != nspan:
            self._ref_prim_buf = util.zeros((nspan, 5))
            self._ref_p2c_buf = util.zeros((nspan, 5, 5))
            self._ref_c2p_buf = util.zeros((nspan, 5, 5))

        recombine_shape = self._block_view.shape + (5,)
        if (
            self._recombine_dchic is None
            or self._recombine_dchic.shape != recombine_shape
        ):
            self._recombine_dchic = util.zeros(recombine_shape)
            self._recombine_prim = util.zeros(recombine_shape)

    def attach_to_block_resampled(self, block, src):
        """Attach to a resampled ``block``, interpolating prescribed profiles.

        A prescribed row is recorded as the setter call that filled it (see
        :func:`replayable`), arguments as the caller gave them -- which for a
        spanwise profile is one number per span station of the grid the patch
        was configured on. Replaying that call against a block with a different
        number of stations, as :meth:`attach_to_block` does on its own, hands
        the setter a profile of the wrong length and it refuses it. So the
        replay is deferred and every profile argument re-expressed on this
        patch's own stations first, interpolated against span fraction
        (:attr:`~ember.patch.RevolutionPatch.spf`, meridional arc-length)
        rather than node index, so a prescription follows the geometry it was
        written against and not the mesh spacing.

        Scalar arguments pass through untouched, which is what makes this
        agree with plain re-attachment wherever plain re-attachment worked.
        """
        # Cleared before attaching so the replay inside attach_to_block finds
        # nothing to do; the interpolated calls below refill both the target
        # rows and the record, in the order the caller originally set them.
        calls, self._target_calls = self._target_calls, {}
        self.attach_to_block(block)
        if not calls:
            return
        spf_src = src.spf
        for name, (args, kwargs) in calls.items():
            args = tuple(self._interp_profile(arg, src, spf_src) for arg in args)
            kwargs = {
                key: self._interp_profile(arg, src, spf_src)
                for key, arg in kwargs.items()
            }
            getattr(self, name)(*args, **kwargs)

    def update_ref_scales(self):
        """Re-derive the prescribed target against the block's current fluid.

        Every :func:`replayable` setter that filled a row is re-run with the
        dimensional arguments it was given, in the order it was given them, so a
        prescribed condition keeps meaning what it says: ``set_Po_To(4e5, 300)``
        is four bar and three hundred kelvin whatever reference scales and datum
        come to be in force. That makes
        :meth:`~ember.grid.Grid.set_fluid` safe to call at any point, rather
        than only before the patches are configured.

        Rows nothing prescribed are cleared instead, to be taken afresh from the
        rescaled face -- they are a frozen picture of the flow, and the only
        honest way to re-express one is to look again. They are cleared before
        the replay so that a row which *is* prescribed, and merely happens to
        be seedable, is refilled by its own setter.

        The characteristic state is nondimensional with no dimensional
        original to return to, so it is dropped and rebuilt from the face. A
        fluid changed mid-march therefore restarts the condition from the
        marched state rather than continuing on the one it was solving: a
        small perturbation, and the alternative is carrying numbers that mean
        nothing under the new scales.
        """
        super().update_ref_scales()

        self._ref = None
        self._prim_prev = None
        self._rho_nd_soln = None

        if self._target is None:
            return

        for row in self._target_seeded:
            self._target_set[row] = False

        self._replay_target_calls()

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
        with self._resolved():
            self._rho_nd_soln = self.block_view.rho_nd.copy()
            self._calc_reference()
