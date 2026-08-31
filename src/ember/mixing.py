r"""Mixing plane joining bladerows with circumferentially averaged flow.

:class:`MixingPatch` is either side of a steady stator/rotor
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
inherited from :class:`~ember.patch.NonReflectingPatch` untouched: the
pitchwise Hilbert transform, the frozen pitchwise-mean reference state, the
characteristic split, the under-relaxation
:attr:`~ember.patch.NonReflectingPatch.sigma`, and the non-reflecting
harmonic relations themselves. This class adds no numerics at all; it adds
pairing across the plane, flux averaging, and acceptance of the exchanged
target.

One class serves both sides. Which side a patch is on is the inward face normal
the inward face normal, which is not taken from
the class -- both sides are this class -- but settled by the patch itself, so
there is nothing for the caller to get right. The upstream side then prescribes
static pressure and the downstream side the four inflow quantities -- not
because they are different classes, but because that is what their splits work
out to.

Settling it takes two stages, because the answer is not in the mesh. The
geometry gives a provisional side at attach time, which is all the pairing in
:meth:`MixingPatch.check_match` needs and is enough for an axial plane; then
the patch orients the interface frame along the mean
mass flux the first time there is one, and freezes it. Which way the frame
points is what decides whether the plane absorbs the harmonics reaching it or
zeroes them (see :class:`~ember.patch.NonReflectingPatch`), and on a
radial plane the geometry alone would point it upstream half the time -- a
centrifugal stage and a radial-inflow one have the same mesh and opposite
through-flow.

The exchange is carried out by
:class:`~ember.mixing_communicator.MixingCommunicator`, which
writes the target in the mix variables :math:`[h_0, s, V_s, V_\theta, p]` of
:func:`~ember.perturbation.chic_to_mix` -- exactly the space
:class:`~ember.patch.NonReflectingPatch` stores its prescribed target
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

One restriction, inherited from
:class:`~ember.patch.NonReflectingPatch`: the mean state must be subsonic both
normal to the plane and absolutely. The plane itself may be any surface of
revolution. Backflow is not a restriction either -- a reversed station is
carried by changing which side of the split it is on, and does not re-settle
the frame.

Both sides build their own Hilbert transform on their own pitch and no harmonic
crosses the plane, so the two sides may have different pitchwise node counts and
different blade counts. The spanwise node counts must match.

See Also
--------
ember.mixing_communicator.MixingCommunicator : The exchange
ember.patch.NonReflectingPatch : The condition itself
"""

import numpy as np

import ember.fortran as ft
from ember import util
from ember.nonreflecting import NonReflectingPatch


class MixingPatch(NonReflectingPatch):
    r"""One side of a non-reflecting mixing plane.

    Takes its whole prescribed target from the cross-plane exchange, so it needs
    no setter, and seeds every row from its own pitchwise mean before the first
    exchange has happened. Which side of the plane it is on, and so which rows
    it actually imposes, is settled from the flow rather than declared; see the
    module docstring.

    A note on which average is which. The communicator evaluates its Jacobians
    on the *symmetrised cross-plane* average, so both sides linearise the
    interface jump about the same state; each patch's own frozen reference
    state stays its *local* pitchwise mean, because
    the reference-state calculation calls
    :meth:`~ember.patch.RevolutionPatch.set_block_avg` itself and so
    re-derives it after the exchange has overwritten
    :attr:`~ember.patch.RevolutionPatch.block_avg`. The split is deliberate
    and follows Saxer: the interface jump belongs to the interface, the boundary
    condition to the boundary.
    """

    _desc = "non-reflecting mixing plane"

    # Impose the mixed-out state directly instead of the characteristic
    # exchange. The mechanism of :attr:`ember.solver.Solver.mix_reflective`,
    # which is where what this does -- and what it gives up -- is written down,
    # and which is the only supported way to set it: a march stamps every
    # mixing patch of every level from its configuration, so a plane cannot
    # differ from its run, and the two sides of a plane cannot differ from each
    # other. Read in five places here (set_block_avg, step, apply,
    # _calc_reference, check_match) and by
    # :class:`~ember.mixing_communicator.MixingCommunicator`, none of which can
    # see a Solver, which is why it lives on the patch at all.
    #
    # Both sides of a plane must agree: :meth:`check_match` will not pair a
    # reflective face with a Saxer one, since the exchange has to know which of
    # the two it is carrying out.
    _reflective = False

    # Either side of the plane. The geometry gives a provisional answer at
    # attach time and the flow settles it on the first step; see
    # _enter_resolved.
    _sign_interior = None

    # A face whose mean normal Mach number is below this is treated as having
    # no direction to settle against, and the frame is left provisional for
    # another step. Measured against the speed of sound rather than against the
    # face's own flux, which is a scale that vanishes with the thing it is
    # scaling and leaves a grid at rest settling on round-off. Loose: the point
    # is only to reject a grid that is not yet moving, not to wait for a
    # converged flow.
    _Ma_settle = 1e-3

    # The exchange fills every row, so nothing is required of the user, and
    # every row is seeded from this side's own mean until it has run once.
    _target_setters = {}

    _target_seeded = (0, 1, 2, 3, 4)

    # No nodal backflow limiter. The mix variables can express the state it
    # imposes, unlike the inflow condition's angles, but what it would impose at
    # a plane is the other row's pitch-uniform mixed-out state, and the axial
    # velocity it derives from that comes out of an energy balance with no
    # bearing on how hard the node was actually reversed. On the LISA rotor exit
    # that turned a wake core reversed at -5 m/s into -31 m/s in one
    # application, and the correction feeds its own rate through the Mach number
    # it drives, so the node ran away with the interior held frozen. A station
    # whose mean reverses is still carried by the characteristic split.
    _nodal_backflow = False

    def _setup(self):
        super()._setup()
        self._flux_avg = None
        # Relaxation of the cross-plane mismatch, read by the communicator at
        # every exchange. Held here rather than on the communicator so it
        # survives the pickle that drops the communicator, and so the two
        # planes of a multi-row grid can damp at different rates; both sides of
        # a plane must agree on it. Distinct from
        # :attr:`~ember.patch.NonReflectingPatch.sigma`, which relaxes
        # this side's own characteristic correction. Kept low: the
        # direction-switched split (see
        # :class:`~ember.mixing_communicator.MixingCommunicator`) is stiff
        # feedback, and the integrating form of the relaxation has a tighter
        # stability limit than a proportional one would.
        self.rf_exchange = 0.02
        # Per-station entering flag the communicator computed from the shared
        # symmetrised interface state, stamped here by
        # :class:`~ember.mixing_communicator.MixingCommunicator`
        # once per exchange; see :meth:`_calc_entering`. ``None`` until the
        # first exchange has run, so attach-time and pre-exchange solves fall
        # back to this patch's own local reading.
        self._entering_shared = None
        # Reflective mode: see the class attribute of the same name, and
        # :attr:`ember.solver.Solver.mix_reflective`, which is what sets it.
        # Set in _setup as well as on the class so that a patch pickled before
        # the flag existed unpickles with it, rather than falling through to a
        # class attribute a future edit might move.
        self._reflective = False
        # The pitch-uniform conserved state a reflective plane imposes, one
        # five-vector per span station in the target's broadcast shape, and
        # whether anything has filled it yet. Allocated on attach; see
        # set_uniform. Unused, and left at zero, on the default plane.
        self._uniform = None
        self._uniform_set = False
        # Whether the frame axis has been settled against a flow; see
        # _enter_resolved. False means the provisional geometric frame the base
        # class built at attach is still in force.
        self._sign_settled = False

    def __setstate__(self, state):
        """Restore pickled state, carrying a public ``reflective`` over.

        The flag was briefly a public attribute (commits 46d9b79..cb98404)
        before :attr:`ember.solver.Solver.mix_reflective` took ownership of
        it. An EMB written in that window carries ``reflective`` in its state
        and nothing else would read it, so a reflective plane would come back
        as a Saxer one -- quietly, and only visibly in the answer. Map it onto
        the private name instead. A patch from before the flag existed carries
        neither, and takes the default applied at construction.
        """
        state = dict(state)
        legacy = state.pop("reflective", None)
        if legacy is not None:
            state.setdefault("_reflective", legacy)
        super().__setstate__(state)

    def _calc_entering(self, avg):
        """Span stations the mean flow enters through, from the shared direction.

        Both sides of a mixed-sign station must agree on which characteristic
        split they are on -- the communicator's own split of the interface jump
        (see :meth:`~ember.mixing_communicator.MixingCommunicator._write_targets`)
        already assumes one shared direction, so if each side instead read its
        own local interior to decide entering/leaving, a station straddling the
        two could disagree with the split the exchange built for it. The
        communicator computes one direction from the symmetrised interface
        state and stamps it here every exchange; only before the first
        exchange, or on a patch the communicator has not touched (there is
        none such for this class, but a resized target falls back safely too),
        is the local computation of the base class used.
        """
        shared = self._entering_shared
        if shared is not None and shared.shape == avg.Vx_nd.reshape(-1).shape:
            return shared
        return super()._calc_entering(avg)

    def _copy(self, c):
        # NonReflectingPatch._copy is shared with the inlet and outlet, neither
        # of which has an exchange to relax, so extend it here rather than
        # there.
        super()._copy(c)
        c.rf_exchange = self.rf_exchange
        c._reflective = self._reflective
        # _sign_settled is deliberately not carried: the copy re-settles
        # against its own block's flow, as _ref and the splits are rebuilt.

    def _enter_resolved(self):
        r"""Settle the frame axis against the flow, once, before the first rotation.

        A mixing plane is the one condition in the family that cannot know from
        its class which way its flow runs -- both sides are the same class, and
        which is upstream is a property of the machine, not of the mesh. So the
        provisional frame the base class took from the geometry is replaced
        here, the first time there is a flow to take it from, by the one whose
        axis runs along the mean mass flux through the face.

        That orientation is what keeps the plane non-reflecting. The Giles and
        Saxer harmonic relations are derived for mean flow along the frame axis
        and the condition zeroes the harmonics where the flow opposes it (see
        :class:`~ember.patch.NonReflectingPatch`), so an axis pointing
        the wrong way would leave a plane that still balances the mean but
        reflects every harmonic reaching it -- silently, and at every span
        station rather than only the reversed ones. On an axial machine the
        provisional frame is already right and this changes nothing; on a
        radial one it is what stops the plane being pointed upstream.

        Both sides of a plane see the same flow direction and start from
        antiparallel normals, so they settle to opposite signs, which is what
        :meth:`check_match` needs of them.

        Deferred rather than forced on the first call: a grid initialised at
        rest has no direction to offer, and freezing an arbitrary one would be
        worse than waiting a step for the march to develop one. A later
        *reversal* does not re-settle -- that is the per-station split's job,
        and re-settling would move the frame under a target already written in
        it.
        """
        if self._sign_settled or self._rot_to is None:
            return

        # Read in (x, r) coordinates: this runs before the rotation goes on.
        # Area- and pitch-weighted, so a face whose flux changes sign along the
        # span settles on the direction that carries the net flow.
        b = self.block_view
        cons = b.conserved_nd
        rot = self._rot_to
        rhoVn = rot[..., 0, 0] * cons[..., 1] + rot[..., 0, 1] * cons[..., 2]
        dA = self._dA_node
        mass = float(np.sum(self._pitch_mean(rhoVn).reshape(-1) * dA))
        rho = float(np.sum(self._pitch_mean(cons[..., 0]).reshape(-1) * dA))
        a_nd = float(np.sum(self._pitch_mean(b.a_nd).reshape(-1) * dA) / np.sum(dA))
        if rho <= 0.0 or abs(mass / rho) <= self._Ma_settle * a_nd:
            return

        self._sign_settled = True
        if mass > 0.0:
            # The provisional axis already runs with the flow.
            return

        # Turn the frame through pi to face the other way. The velocity in the
        # surface turns with it, so anything already written in the old frame
        # -- a target from an exchange that ran before the flow developed, a
        # face state this patch authored -- has to turn too.
        self._sign_interior = -self._sign_interior
        self._build_rot_matrices(inward=self._sign_interior > 0)
        self._split_entering = self._calc_split(True)
        self._split_leaving = self._calc_split(False)
        if self._target is not None:
            self._target[..., 2] = -self._target[..., 2]
        if self._prim_prev is not None:
            self._prim_prev[..., 1:3] = -self._prim_prev[..., 1:3]

    def set_block_avg(self):
        r"""Pitch-average the face, in interface coordinates unless the plane is reflective.

        Overridden only to hold the rotation: the communicator calls this from
        outside any of the boundary condition's own entry points, and the
        cross-plane average it builds has to be in the same frame on both
        sides. See :meth:`set_flux_avg`.

        A reflective plane takes the average in ``(x, r)`` instead, and wants
        to. Its conserved variables are the absolute-frame
        :math:`[\rho, \rho V_x, \rho V_r, \rho r V_\theta, \rho e]`, which
        the two sides can compare directly: they share a meridional geometry and
        a radius, so nothing has to be resolved into a common frame first. The
        rotation would in fact be actively wrong before the frame has settled,
        since until then the two sides' provisional axes are antiparallel and
        their normal components carry opposite signs; and the frame never does
        settle on a reflective plane, because settling happens inside the very
        window this skips.
        """
        if self._reflective:
            super().set_block_avg()
            return
        with self._resolved():
            super().set_block_avg()

    def set_flux_avg(self):
        """Compute pitch-averaged node fluxes and store in :attr:`flux_avg_nd`.

        Called by
        :class:`~ember.mixing_communicator.MixingCommunicator`
        before reading :attr:`flux_avg_nd` to form the cross-plane flux
        difference of Saxer Eq. 5.65.

        Taken in interface coordinates, so what the kernels below compute as
        the ``x``-direction flux is the flux through the face whatever the
        face's orientation. The two sides of a plane share a meridional
        geometry and settle to opposite signs, so their frame axes coincide and
        the difference the communicator takes is between fluxes resolved the
        same way.
        """
        with self._resolved():
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
        :class:`~ember.mixing_communicator.MixingCommunicator`
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

    def set_uniform(self, cons=None):
        """Set the pitch-uniform conserved state a reflective plane imposes.

        Called by
        :class:`~ember.mixing_communicator.MixingCommunicator`
        after each exchange, with the average of the two sides' circumferential
        means. Omitting ``cons`` seeds from this side's own circumferential mean
        instead, which is how a face that has not yet been exchanged has
        something physical to impose; :meth:`apply` does that for itself rather
        than imposing zeros.

        Parameters
        ----------
        cons : array of shape ``(nspan, 5)``, optional
            Nondimensional conserved variables, in ``(x, r)`` components and
            in this patch's own span order.
        """
        self._check_attached()
        if cons is None:
            self.set_block_avg()
            cons = self.block_avg.conserved_nd
        self._uniform[...] = np.asarray(cons).reshape(self._target_shape())
        self._uniform_set = True

    def get_target(self):
        """Return the exchanged target, a nondimensional ``(nspan, 5)`` array.

        Rows are ``[ho, s, Vr, Vt, P]``. Read by
        :class:`~ember.mixing_communicator.MixingCommunicator` to
        form the symmetrised baseline the cross-plane mismatch is relaxed onto,
        which is why a patch that has never been exchanged is seeded here rather
        than left at zero.
        """
        self._check_attached()
        self._seed_target()
        return self._target.squeeze()

    def get_uniform(self):
        """Return the reflective plane's imposed state, nondimensional ``(nspan, 5)``.

        Seeded from this side's own circumferential mean if nothing has been
        exchanged onto it yet, for the same reason :meth:`get_target` seeds.
        """
        self._check_attached()
        if not self._uniform_set:
            self.set_uniform()
        return self._uniform.squeeze()

    def advance(self):
        """Take the boundary condition's step; a no-op on a reflective plane.

        A reflective plane has no state of its own between exchanges: what it
        imposes is settled entirely by the communicator, and :meth:`apply`
        imposes it outright.
        """
        if self._reflective:
            return
        super().advance()

    def apply(self):
        """Impose the condition on the face.

        The plain plane runs the non-reflecting condition of
        :meth:`~ember.patch.NonReflectingPatch.apply`. A reflective one
        overwrites the whole face with the pitch-uniform state the last exchange
        left, span station by span station and every stage, hub and casing
        nodes included -- there is no characteristic content to preserve and no
        relaxation to take, so there is nothing here to be gradual about.
        """
        if not self._reflective:
            super().apply()
            return
        if not self._uniform_set:
            self.set_uniform()
        b = self.block_view
        b.conserved_nd[...] = self._uniform
        b.update_cached_conserved()

    def attach_to_block(self, block):
        """Attach to a block, validate the plane, and allocate the flux average.

        Drops any settled frame: the base class has just rebuilt the
        provisional one from the new block's geometry, so a stale ``settled``
        flag would pin the patch to that provisional frame for the rest of the
        run and never let the flow correct it.
        """
        self._sign_settled = False
        super().attach_to_block(block)

        if self._block_ref is None:
            return

        nspan = self._block_view.shape[self.span_dim]
        self._flux_avg = util.zeros((nspan, 5))

        # Re-seeded rather than carried across a re-attach: the new block's
        # span count may differ -- this is how a patch follows its block onto a
        # coarser multigrid level -- and a stale state of the old length would
        # be imposed on the wrong stations.
        self._uniform = util.zeros(self._target_shape())
        self._uniform_set = False

    def check_match(self, other, rtol=1e-5):
        """Check whether this patch pairs with another across a mixing plane.

        Pairs only with the opposite side of a mixing plane running the same
        treatment, which the inward face normal and the
        reflective flag identify between them: the two sides of one plane
        face each other, so their interiors lie on opposite sides of it.
        Matching is then on meridional geometry alone, so the two sides may
        differ in pitchwise resolution and blade count but not in spanwise
        resolution.

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
        if not isinstance(other, MixingPatch):
            return None

        # The exchange is one thing or the other, so the pair has to be too:
        # a reflective face opposite a Saxer one leaves the communicator with
        # no answer to which of the two it is carrying out.
        if other._reflective != self._reflective:
            return None

        if other._sign_interior == self._sign_interior:
            return None

        if self.shape[self.span_dim] != other.shape[other.span_dim]:
            return None

        return self._check_match_xr(other, rtol)

    def update_soln(self):
        """Refresh the frozen reference state; a no-op on a reflective plane.

        Nothing on a reflective plane is linearised about a mean state, so
        there is no reference to freeze.
        """
        if self._reflective:
            return
        super().update_soln()

    @property
    def flux_avg_nd(self):
        """Pitch-averaged flux array, shape ``(nspan, 5)``; populated by :meth:`set_flux_avg` and read by :class:`~ember.mixing_communicator.MixingCommunicator` to form the cross-plane flux difference."""
        self._check_attached()
        return self._flux_avg
