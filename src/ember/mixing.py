"""Mixing plane boundary condition patch for EMBER CFD.

MixingPatch provides circumferentially-averaged connections between blade rows
in multi-stage turbomachinery. Both sides of the plane are driven to carry the
same pitch-uniform conserved state -- the cross-plane average of the two sides'
pitch means -- so the mean flow passes between rows while every circumferential
variation that reaches the plane is absorbed into that mean.

See Also
--------
ember.mixing_communicator.MixingCommunicator : Forms the cross-plane average
ember.mixing_nonreflecting : Non-reflecting mixing plane, after Saxer
ember.patch.Patch : Base class for all patches
ember.patch.PeriodicPatch : For circumferentially periodic boundaries
"""

from ember import util
from ember.basepatch import RevolutionPatch


class MixingPatch(RevolutionPatch):
    r"""Circumferentially averaged connection between blade rows.

    The patch holds one pitch-uniform conserved state per span station, its
    target, and writes that state onto every node of its face.
    :class:`~ember.mixing_communicator.MixingCommunicator` relaxes the target
    towards the average of the two sides' pitch-mean conserved states, so at
    convergence both blocks carry the same state at the plane, both evaluate
    the same face fluxes, and no circumferential variation crosses.

    Imposing all five conserved variables over-specifies the boundary: four of
    the five characteristics run out of the domain on the upstream side, one on
    the downstream side, and those are overwritten along with the rest. The
    fixed point is still consistent -- each side's own pitch mean equals the
    state imposed on it -- but nothing makes the iteration towards it
    contracting except the relaxation factor :attr:`rf_exchange` the exchange
    reads off this patch, which is therefore the only stability knob the plane
    has. Waves reaching the
    plane reflect off it in full; :mod:`ember.mixing_nonreflecting` is the
    alternative that absorbs them.

    Averaging conserved variables rather than fluxes conserves mass exactly --
    the axial mass flux :math:`\rho V_x` *is* the second conserved variable, so
    a pitch average preserves it by construction -- but not momentum or energy
    flux, which are nonlinear in the conserved variables. That mismatch is
    second order in the pitchwise nonuniformity arriving at the plane and does
    not vanish as the solution converges: expect a step across the plane of
    order the square of the wake velocity deficit in the stagnation quantities.

    Both sides must have the same spanwise resolution. Pitchwise resolution,
    blade count, and reference frame may all differ: the conserved variables
    are stored in the absolute frame with angular momentum
    :math:`\rho r V_\theta`, so a stator/rotor pair needs no frame conversion,
    and the geometric match :meth:`check_match` enforces is what makes a given
    span station mean the same thing on both sides.
    """

    _collection_name = "mixing"

    def _setup(self):
        super()._setup()
        self._target = None
        # Relaxation of the cross-plane mismatch, read by the communicator at
        # every exchange. Held here rather than on the communicator so it
        # survives the pickle that drops the communicator, and so the two
        # planes of a multi-row grid can damp at different rates; both sides of
        # a plane must agree on it.
        self.rf_exchange = 0.05

    def _copy(self, c):
        # The target is deliberately not carried: it is seeded lazily from the
        # block state a copy is attached to. rf_exchange is configuration, not
        # solver state, so it travels with the patch.
        super()._copy(c)
        c.rf_exchange = self.rf_exchange

    def set_target(self, target=None):
        """Set the pitch-uniform target from the current block state or an explicit array.

        :class:`~ember.mixing_communicator.MixingCommunicator` calls this to
        write back the updated cross-plane target after each exchange. If
        ``target`` is omitted, the pitch mean of the current block state is
        used to initialise the target before the first exchange -- the same
        pitch average, from the same weights, that the exchange itself takes.

        Parameters
        ----------
        target : array of shape ``(nspan, 5)``, optional
            Nondimensional conserved variables
            :math:`[\\rho, \\rho V_x, \\rho V_r, \\rho r V_\\theta, \\rho e]`.
        """
        self._check_attached()
        shape = [1, 1, 1]
        shape[self.span_dim] = self._block_view.shape[self.span_dim]
        if self._target is None:
            self._target = util.zeros((*shape, 5))
        if target is None:
            self.set_block_avg()
            self._target[...] = self.block_avg.conserved_nd.reshape((*shape, 5))
        else:
            self._target[...] = target.reshape((*shape, 5))

    def get_target(self):
        r"""Return the target as a nondimensional ``(nspan, 5)`` array of conserved variables.

        Stack along the last axis is
        :math:`[\rho, \rho V_x, \rho V_r, \rho r V_\theta, \rho e]`. Read by
        :class:`~ember.mixing_communicator.MixingCommunicator` during the
        exchange as the baseline its relaxation increment is added to.
        """
        self._check_attached()
        if self._target is None:
            self.set_target()
        return self._target.squeeze()

    def apply(self):
        """Impose the pitch-uniform target on every node of the patch face.

        Called each Runge-Kutta stage, after
        :class:`~ember.mixing_communicator.MixingCommunicator` has refreshed
        the target via :meth:`set_target`. Every node at a span station takes
        that station's target, so the face comes out uniform in pitch and
        identical to the face on the far side of the plane. Nothing here
        depends on which side of the plane the patch is on, or on the sign of
        the flow through it; reversed flow at the plane is imposed like any
        other state.
        """
        b = self.block_view
        if self._target is None:
            self.set_target()
        # Writing conserved_nd directly bypasses cache invalidation, so flush
        # the derived-property caches explicitly.
        b.conserved_nd[...] = self._target
        b.update_cached_conserved()

    def attach_to_block(self, block):
        """Attach to block and validate any target carried over from a previous attach.

        Safe to call repeatedly; an existing ``_target`` of the correct shape is
        preserved so targets set before re-attachment are not lost. ``_target``
        itself is left unallocated (``None``) until first used --
        :meth:`set_target` and :meth:`get_target` seed it lazily from the
        current block state -- so a fresh attach never masks the "nothing has
        set a real target yet" state behind a zero-filled array.
        """
        super().attach_to_block(block)

        shape = [1, 1, 1]
        shape[self.span_dim] = self._block_view.shape[self.span_dim]
        target_shape = (*shape, 5)
        if self._target is not None and self._target.shape != target_shape:
            self._target = None

    def check_match(self, other, rtol=1e-5):
        """Check if this MixingPatch matches another for pairing purposes.

        MixingPatch matching requires x,r coordinates to match at all spanwise
        nodes (pitch-averaged), ignoring theta. Patches must have the same
        spanwise resolution but can have different pitchwise resolutions.

        Parameters
        ----------
        other : Patch
            The other patch to compare with
        rtol : float, optional
            Relative tolerance for matching

        Returns
        -------
        bool or None
            None if patches do not match. False if they match with no spanwise
            flip needed. True if they match but other's span must be reversed.
            Always test with ``is not None``; do not use as a bare truthiness
            check since False is a valid match result.
        """
        if not isinstance(other, MixingPatch):
            return None

        if self.shape[self.span_dim] != other.shape[other.span_dim]:
            return None

        return self._check_match_xr(other, rtol)
