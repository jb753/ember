r"""Non-reflecting mixing plane for EMBER CFD.

:class:`NonReflectingMixingInletPatch` and
:class:`NonReflectingMixingOutletPatch` are the two sides of a steady
stator/rotor interface after :cite:t:`Saxer1993` (his Section 5.5, Eqs.
5.60-5.66), which lets the pitchwise mean cross the plane while absorbing the
circumferential harmonics that reach it from either row.

Saxer's interface is deliberately not a new boundary condition. He flux-averages
each side to its mixed-out state (his Eqs. 5.60-5.64), takes the jump in that
state across the plane (Eq. 5.65), converts the jump to a jump in characteristic
variables and splits it by direction of propagation (Eq. 5.66) -- the upstream
side owns the upstream-running pressure characteristic and the downstream side
the other four -- and then, in his words, "the remainder of the boundary
condition treatment is exactly the same as for a standard inflow and outflow
boundary".

So a mixing plane replaces only the **mean mode**. Everything else is inherited
from :class:`~ember.inlet_nonreflecting.NonReflectingInletPatch` and
:class:`~ember.outlet_nonreflecting.NonReflectingOutletPatch` untouched: the
pitchwise Hilbert transform, the frozen pitchwise-mean reference state, the
split of the boundary node into the characteristics the interior march owns and
those the boundary condition owns, the under-relaxation
:attr:`~ember.nonreflecting.NonReflectingPatch.sigma`, and the non-reflecting
harmonic relations themselves. These two classes add no numerics at all; they
add pairing across the plane, flux averaging, and acceptance of the exchanged
target.

The exchange is carried out by
:class:`~ember.mixing_communicator.NonReflectingMixingCommunicator`, which
writes the target in the mix variables :math:`[h_0, s, V_r, V_\theta, p]` of
:func:`~ember.perturbation.chic_to_mix` -- exactly the space
:class:`~ember.nonreflecting.NonReflectingPatch` stores its prescribed target
in. Rows 0-3 are what the inflow side takes its pitchwise-mean residuals
against, row 4 what the outflow side does, which is Saxer's split by direction
of propagation expressed in those variables. So the exchange writes the patches'
own target array and there is nothing to translate.

:class:`~ember.mixing.MixingPatch` exchanges conserved variables instead and
imposes them outright, so the two mixing planes now share only their pairing
and their relaxation factor. The inflow side takes its residuals in mix
variables rather than in the :math:`[h_0, s, \tan\alpha, \sin\beta]` its parent
:class:`~ember.inlet_nonreflecting.NonReflectingInletPatch` uses; see that
class for why the angles suit a physical inlet and not an interface. The
outflow side is unaffected either way, because the static-pressure row is
identical in the two variable sets.

Unlike :class:`~ember.mixing.MixingPatch` the two sides are distinct classes
rather than one class that infers its side from the geometry, because the
characteristic split, and so every hook the base class dispatches through,
differs between them.

Restrictions beyond those of the reflecting plane, all inherited:

* the plane must be one of constant :math:`x` with the flow running along
  :math:`+x`, and the mean state axially and absolutely subsonic;
* pitchwise-mean backflow raises on the inflow side, where the reflecting plane
  is indifferent to the sign of the flow through it -- it imposes a conserved
  state and never asks which way it points -- because the characteristic split
  that side is built on is invalid there. The outflow side carries it, and
  needs nothing configured to: the four quantities a reversed station has to be
  given are rows 0-3 of the exchanged target, so such a station is driven
  toward the flow actually standing on the other side of the plane.

Both sides build their own Hilbert transform on their own pitch and no harmonic
crosses the plane, so the two sides may have different pitchwise node counts and
different blade counts. The spanwise node counts must match, as for
:class:`~ember.mixing.MixingPatch`.

See Also
--------
ember.mixing.MixingPatch : The reflecting mixing plane
ember.mixing_communicator.NonReflectingMixingCommunicator : The exchange
ember.nonreflecting.NonReflectingPatch : Base class holding the shared machinery
"""

from ember import perturbation, util
from ember.inlet_nonreflecting import NonReflectingInletPatch
from ember.nonreflecting import NonReflectingPatch
from ember.outlet_nonreflecting import NonReflectingOutletPatch


class NonReflectingMixingPatch:
    r"""Mixin holding what the two sides of a non-reflecting mixing plane share.

    Not a patch on its own: it carries no geometry and must be mixed in ahead of
    a :class:`~ember.nonreflecting.NonReflectingPatch` subclass, which supplies
    everything else. It exists as a class in its own right so that
    :class:`~ember.grid.GridConnectivity` and
    :class:`~ember.collections.BlockPatchCollection` have a single type to
    filter the two sides on; their only other common base is
    :class:`~ember.nonreflecting.NonReflectingPatch`, which would also sweep up
    ordinary non-reflecting inlets and outlets.

    Both sides take their whole target from the exchange, so neither requires a
    setter, and both seed every row from their own pitchwise mean before the
    first exchange has happened.

    A note on which average is which. The communicator evaluates its Jacobians
    on the *symmetrised cross-plane* average, so both sides linearise the
    interface jump about the same state; each patch's own frozen reference
    state stays its *local* pitchwise mean, because
    :meth:`~ember.nonreflecting.NonReflectingPatch._calc_reference` calls
    :meth:`~ember.basepatch.RevolutionPatch.set_block_avg` itself and so
    re-derives it after the exchange has overwritten
    :attr:`~ember.basepatch.RevolutionPatch.block_avg`. The split is deliberate
    and follows Saxer: the interface jump belongs to the interface, the boundary
    condition to the boundary.
    """

    _collection_name = "mixing_nonreflecting"

    # The exchange fills every row, so nothing is required of the user, and
    # every row is seeded from this side's own mean until it has run once.
    _target_setters = {}

    _target_seeded = (0, 1, 2, 3, 4)

    def _setup(self):
        super()._setup()
        self._flux_avg = None

    def attach_to_block(self, block):
        """Attach to a block, validate the plane, and allocate the flux average."""
        super().attach_to_block(block)

        if self._block_ref is None:
            return

        nspan = self._block_view.shape[self.span_dim]
        self._flux_avg = util.zeros((nspan, 5))

    def check_match(self, other, rtol=1e-5):
        """Check whether this patch pairs with another across a mixing plane.

        Pairs only with the opposite side of a non-reflecting mixing plane,
        which :attr:`~ember.nonreflecting.NonReflectingPatch._sign_interior`
        identifies: the two sides of one plane face each other, so their
        interiors lie on opposite sides of it. Matching is then on meridional
        geometry alone, so the two sides may differ in pitchwise resolution and
        blade count but not in spanwise resolution.

        Parameters
        ----------
        other : Patch
            The other patch to compare with.
        rtol : float, optional
            Relative tolerance for matching.

        Returns
        -------
        bool or None
            None if the patches do not match. False if they match with no
            spanwise flip needed. True if they match but ``other``'s span must
            be reversed. Always test with ``is not None``; do not use as a bare
            truthiness check since False is a valid match result.
        """
        if not isinstance(other, NonReflectingMixingPatch):
            return None

        if other._sign_interior == self._sign_interior:
            return None

        if self.shape[self.span_dim] != other.shape[other.span_dim]:
            return None

        return self._check_match_xr(other, rtol)

    def set_flux_avg(self):
        """Compute pitch-averaged node fluxes and store in :attr:`flux_avg_nd`.

        Called by
        :class:`~ember.mixing_communicator.NonReflectingMixingCommunicator`
        before reading :attr:`flux_avg_nd` to form the cross-plane flux
        difference of Saxer Eq. 5.65. No rotation into interface coordinates is
        needed: the base class restricts this patch to a plane of constant
        ``x``, so ``Vx`` is already the face-normal velocity and ``Vr`` the
        spanwise one.
        """
        import ember.fortran as ft

        b = self.block_view
        cons = b.conserved_nd
        w = self.weight_pitch.ravel()
        ni, nj, nk = b.shape
        if self.pitch_dim == 0:
            dest = self._flux_avg.reshape(nj, nk, 5)
            ft.flux_avg_i(cons, b.P_nd, b.ho_nd, w, dest)
        elif self.pitch_dim == 1:
            dest = self._flux_avg.reshape(ni, nk, 5)
            ft.flux_avg_j(cons, b.P_nd, b.ho_nd, w, dest)
        else:
            dest = self._flux_avg.reshape(ni, nj, 5)
            ft.flux_avg_k(cons, b.P_nd, b.ho_nd, w, dest)

    def get_target(self):
        """Return the exchanged target, a nondimensional ``(nspan, 5)`` array.

        Rows are ``[ho, s, Vr, Vt, P]``. Read by
        :class:`~ember.mixing_communicator.NonReflectingMixingCommunicator` to
        form the symmetrised baseline the cross-plane mismatch is relaxed onto,
        which is why a patch that has never been exchanged is seeded here rather
        than left at zero.
        """
        self._check_attached()
        self._seed_target()
        return self._target.squeeze()

    def set_target(self, target=None):
        """Set the exchanged target, from an explicit array or this side's own mean.

        Called by
        :class:`~ember.mixing_communicator.NonReflectingMixingCommunicator`
        after each exchange. Omitting ``target`` re-seeds from the pitchwise
        mean of the current face state instead, which is how a patch that has
        not yet been exchanged gets a consistent starting point.

        Parameters
        ----------
        target : array of shape ``(nspan, 5)``, optional
            Nondimensional ``[ho, s, Vr, Vt, P]`` target values.
        """
        self._check_attached()
        if target is None:
            self._target_set[:] = False
            self._seed_target()
        else:
            self._target[...] = target.reshape(self._target_shape())
            self._target_set[:] = True

    @property
    def flux_avg_nd(self):
        """Pitch-averaged flux array, shape ``(nspan, 5)``; populated by :meth:`set_flux_avg` and read by :class:`~ember.mixing_communicator.NonReflectingMixingCommunicator` to form the cross-plane flux difference."""
        self._check_attached()
        return self._flux_avg


class NonReflectingMixingInletPatch(NonReflectingMixingPatch, NonReflectingInletPatch):
    r"""Downstream side of a non-reflecting mixing plane.

    An inflow face, so four of its five characteristics are incoming. Their
    pitchwise means are driven to the exchanged stagnation enthalpy, entropy and
    two transverse velocities by the same Newton step that
    :class:`~ember.inlet_nonreflecting.NonReflectingInletPatch` uses, and their
    harmonics by the same non-reflecting relations.

    Two things differ from that condition. The target comes from the cross-plane
    exchange each timestep rather than from a user calling
    :meth:`~ember.inlet_nonreflecting.NonReflectingInletPatch.set_ho_s` and
    friends; :meth:`set_ho_s` and :meth:`set_Po_To` still work but are
    overwritten by the next exchange, so there is no point calling them.

    And the prescribed set is the *mix* set :math:`[h_0, s, V_r, V_\theta]`
    rather than that condition's :math:`[h_0, s, \tan\alpha, \sin\beta]`, so the
    Newton step is taken against ``ember.perturbation.chic_to_mix``. A physical
    inlet knows its flow angles and not its velocity magnitude, so the angles
    are the right variables there; across a mixing plane both sides know the
    velocities, and the angles are actively worse. They are measured against the unsigned meridional speed
    :math:`V_m = \sqrt{V_x^2 + V_r^2}`, so they cannot tell :math:`V_x` from
    :math:`-V_x`, and their rows of the mean-mode Jacobian carry a factor of
    :math:`V_x` that takes the system towards singular as the axial velocity
    falls. The mix rows carry no :math:`V_x` at all, so the solve stays
    conditioned down to and through zero axial velocity. Only rows 2 and 3 of
    the two Jacobians differ; rows 0, 1 and 4 are identical, which is why the
    outflow side and the harmonic relations are untouched by the choice.

    :meth:`set_Alpha` and :meth:`set_Beta` therefore raise rather than setting
    a target row nothing reads.
    """

    _desc = "non-reflecting mixing plane inflow side"

    # An inflow face, but not a physical inlet, so the angle specialisation of
    # NonReflectingInletPatch is reverted to the mix target space the base
    # class and the exchange both work in; see the class docstring.
    _chic_to_target = staticmethod(perturbation.chic_to_mix)

    _target_names = NonReflectingPatch._target_names

    _target_from_prim = NonReflectingPatch._target_from_prim

    def _raise_angle_setter(self, name):
        """Report that an angle setter does not apply to the mix target set."""
        raise ValueError(
            f"{self._desc.capitalize()} {self.label!r} takes its inflow state "
            f"from the mixing exchange as (ho, s, Vr, Vt), so {name} has "
            "nothing to set; the flow angles cross the plane as the velocity "
            "components that carry them."
        )

    def set_Alpha(self, Alpha):
        """Not available: the exchanged target prescribes velocities, not angles.

        Raises
        ------
        ValueError
            Always; see :meth:`set_Beta`.
        """
        self._raise_angle_setter("set_Alpha")

    def set_Beta(self, Beta):
        r"""Not available: the exchanged target prescribes velocities, not angles.

        Raises
        ------
        ValueError
            Always. This side prescribes :math:`(h_0, s, V_r, V_\theta)` from
            the cross-plane exchange, so there is no :attr:`tanAlpha` or
            :attr:`sinBeta` row for these to fill and calling them would set
            something nothing reads. Yaw and pitch still cross the plane, as
            the velocity components that carry them.
        """
        self._raise_angle_setter("set_Beta")


class NonReflectingMixingOutletPatch(
    NonReflectingMixingPatch, NonReflectingOutletPatch
):
    r"""Upstream side of a non-reflecting mixing plane.

    An outflow face, so only the upstream-running pressure characteristic is
    incoming. Its pitchwise mean is driven to the exchanged static pressure and
    its harmonics follow the same non-reflecting relation
    :class:`~ember.outlet_nonreflecting.NonReflectingOutletPatch` uses; the four
    outgoing characteristics are carried through untouched.

    The exchanged pressure is a spanwise profile taken from the flow on the far
    side of the plane, so it already carries that flow's radial equilibrium.
    :meth:`set_adjustment` therefore raises rather than adding a second one.

    A span station whose mean has reversed needs no configuring here.
    :meth:`~ember.outlet_nonreflecting.NonReflectingOutletPatch.set_backflow`
    exists on this class and still works, but the four rows it fills are the
    four the exchange writes anyway, so an unconfigured reversed station is
    driven toward the flow standing on the other side of the plane -- which is
    where the flow entering through that station is in fact coming from.
    """

    _desc = "non-reflecting mixing plane outflow side"

    def set_adjustment(self, radial_equilibrium=True, rf=0.1):
        """Not available: the exchanged pressure profile already carries radial equilibrium.

        Raises
        ------
        ValueError
            Always. The reflecting outlet needs the adjustment because it
            prescribes one pressure level; here the spanwise profile comes from
            the flow across the plane, and adding an adjustment on top of it
            would count the centrifugal gradient twice.
        """
        raise ValueError(
            f"{self._desc.capitalize()} {self.label!r} takes its spanwise "
            "pressure profile from the mixing exchange, which already carries "
            "the radial equilibrium of the flow across the plane; a further "
            "adjustment would double count it."
        )
