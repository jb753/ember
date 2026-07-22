r"""Non-reflecting mixing plane for EMBER CFD.

:class:`NonReflectingMixingPatch` is either side of a steady stator/rotor
interface after :cite:t:`Saxer1993` (his Section 5.5, Eqs. 5.60-5.66), which
lets the pitchwise mean cross the plane while absorbing the circumferential
harmonics that reach it from either row.

Saxer's interface is deliberately not a new boundary condition. He flux-averages
each side to its mixed-out state (his Eqs. 5.60-5.64), takes the jump in that
state across the plane (Eq. 5.65), converts the jump to a jump in characteristic
variables and splits it by direction of propagation (Eq. 5.66) -- the upstream
side owns the upstream-running pressure characteristic and the downstream side
the other four -- and then, in his words, "the remainder of the boundary
condition treatment is exactly the same as for a standard inflow and outflow
boundary".

So a mixing plane replaces only the **mean-mode target**. Everything else is
inherited from :class:`~ember.nonreflecting.NonReflectingPatch` untouched: the
pitchwise Hilbert transform, the frozen pitchwise-mean reference state, the
characteristic split, the under-relaxation
:attr:`~ember.nonreflecting.NonReflectingPatch.sigma`, and the non-reflecting
harmonic relations themselves. This class adds no numerics at all; it adds
pairing across the plane, flux averaging, and acceptance of the exchanged
target.

One class serves both sides. Which side a patch is on is the inward face normal
:attr:`~ember.nonreflecting.NonReflectingPatch._sign_interior`, which the base
class reads off the geometry at attach time rather than taking from the class,
so there is nothing for the caller to get right that the mesh does not already
say. The upstream side then prescribes static pressure and the downstream side
the four inflow quantities -- not because they are different classes, but
because that is what their splits work out to.

The exchange is carried out by
:class:`~ember.mixing_communicator.NonReflectingMixingCommunicator`, which
writes the target in the mix variables :math:`[h_0, s, V_r, V_\theta, p]` of
:func:`~ember.perturbation.chic_to_mix` -- exactly the space
:class:`~ember.nonreflecting.NonReflectingPatch` stores its prescribed target
in, so the exchange writes the patches' own target array and there is nothing to
translate. Rows 0-3 are what a side the flow enters takes its pitchwise-mean
residuals against, row 4 what a side the flow leaves does, which is Saxer's
split by direction of propagation expressed in those variables.

That correspondence is exact only while each side's flow runs the way its
geometry expects. Where a span station has reversed, the side reads the other
rows of the same target -- so a reversed station on the upstream side is driven
toward the flow standing on the other side of the plane, which is where the flow
entering through it is in fact coming from -- but the *increment* the
communicator formed for those rows still came from the downstream-running part
of the interface jump. At such a station this is no longer Saxer's Eq. 5.66
direction split; it is a relaxation toward a matched state, which is enough.

:class:`~ember.mixing.MixingPatch` exchanges conserved variables instead and
imposes them outright, so the two mixing planes now share only their pairing and
their relaxation factor.

Restrictions beyond those of the reflecting plane, both inherited: the plane
must be one of constant :math:`x`, and the mean state axially and absolutely
subsonic. Backflow is not among them -- the reflecting plane is indifferent to
the sign of the flow through it because it imposes a conserved state and never
asks which way it points, and this one is indifferent because it changes split.

Both sides build their own Hilbert transform on their own pitch and no harmonic
crosses the plane, so the two sides may have different pitchwise node counts and
different blade counts. The spanwise node counts must match, as for
:class:`~ember.mixing.MixingPatch`.

See Also
--------
ember.mixing.MixingPatch : The reflecting mixing plane
ember.mixing_communicator.NonReflectingMixingCommunicator : The exchange
ember.nonreflecting.NonReflectingPatch : The condition itself
"""

from ember import util
from ember.nonreflecting import NonReflectingPatch


class NonReflectingMixingPatch(NonReflectingPatch):
    r"""One side of a non-reflecting mixing plane.

    Takes its whole prescribed target from the cross-plane exchange, so it needs
    no setter, and seeds every row from its own pitchwise mean before the first
    exchange has happened. Which side of the plane it is on, and so which rows
    it actually imposes, follows from the geometry; see the module docstring.

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

    _desc = "non-reflecting mixing plane"

    # Either side of the plane; the geometry says which at attach time.
    _sign_interior = None

    # The exchange fills every row, so nothing is required of the user, and
    # every row is seeded from this side's own mean until it has run once.
    _target_setters = {}

    _target_seeded = (0, 1, 2, 3, 4)

    def _setup(self):
        super()._setup()
        self._flux_avg = None

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

    @property
    def flux_avg_nd(self):
        """Pitch-averaged flux array, shape ``(nspan, 5)``; populated by :meth:`set_flux_avg` and read by :class:`~ember.mixing_communicator.NonReflectingMixingCommunicator` to form the cross-plane flux difference."""
        self._check_attached()
        return self._flux_avg
