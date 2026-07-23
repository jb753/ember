"""Mixing plane boundary condition communication.

:class:`MixingCommunicator` serves the reflecting mixing plane of
:mod:`ember.mixing` and :class:`NonReflectingMixingCommunicator` the
non-reflecting one of :mod:`ember.mixing_nonreflecting`. They share the pairing,
the relaxation factor, and the per-pair diagnostics, but not the exchange
itself: the reflecting plane averages the two sides' pitch-mean conserved
states, while the non-reflecting plane takes the cross-plane *flux* mismatch,
splits it by direction of propagation after :cite:t:`Saxer1993`, and writes the
result in the mix variables :math:`[h_0, s, V_r, V_\\theta, p]` its patches
take their pitchwise-mean residuals against.
"""

from ember import perturbation, util
from ember.util import profile
import numpy as np


class MixingCommunicator:
    r"""Manages data communication between mixing patch pairs.

    Each side's face is pitch-averaged in conserved variables, the two averages
    are averaged across the plane, and the shared target is relaxed towards
    that with the plane's relaxation factor :math:`\mathrm{rf\_exchange}`,

    .. math::

        \mathrm{target}_n = \mathrm{target}_{n-1}
            + \mathrm{rf\_exchange}\,\bigl(\tfrac{1}{2}(U_1 + U_2)
            - \mathrm{target}_{n-1}\bigr).

    The factor is read from the patches at every exchange, so it is per plane
    rather than per grid, and a solver run can retune it on a communicator that
    already exists. Both sides of a plane must agree on it. It is the same on
    every multigrid level. Because :class:`~ember.mixing.MixingPatch` imposes
    all five conserved variables at its face, this relaxation is the only thing
    damping the resulting Dirichlet-Dirichlet coupling between the two blocks --
    it carries the whole stability margin of the plane. See that class for what
    the choice of conserved variables costs in conservation.
    """

    def __init__(
        self,
        grid,
        mixing_pairs,
    ):
        """Initialize with grid and mixing patch pairs.

        Parameters
        ----------
        grid : Grid
            The grid instance.
        mixing_pairs : dict
            Mapping of mixing patch pair information.

        Raises
        ------
        ValueError
            If the two sides of a plane disagree on ``rf_exchange``.
        """
        self._grid = grid
        self.pairs = {}
        self._prune_pairs(mixing_pairs)
        self._check_rf_exchange()

        # Per-pair diagnostic snapshots, lazily allocated on first exchange.
        # Keys: (bid, pid). Values: dict with 'du' (the relaxation increment in
        # the exchanged target's own variables, shape (nspan, 5)).
        self._pair_state = {}

    def _check_rf_exchange(self):
        """Raise if either side of a plane would relax the exchange differently.

        The exchange writes one shared target, so a pair holds one relaxation
        factor; the exchange reads it from the first side. Checked once here
        rather than per exchange, so a value changed on one side alone
        afterwards -- which the solver's push cannot do, since it writes the
        same value to every patch -- goes unnoticed.
        """
        for bid, pid in self.pairs:
            patch1, patch2 = self._get_pair(bid, pid)
            if patch1.rf_exchange != patch2.rf_exchange:
                raise ValueError(
                    f"Mixing plane sides disagree on rf_exchange: "
                    f"{patch1.label!r} has {patch1.rf_exchange}, "
                    f"{patch2.label!r} has {patch2.rf_exchange}"
                )

    def _prune_pairs(self, mixing_pairs):
        """Prune bidirectional pairs to unidirectional mapping."""
        seen_pairs = set()

        for (bid, pid), ((nxbid, nxpid), transform) in mixing_pairs.items():
            pair_key = tuple(sorted([(bid, pid), (nxbid, nxpid)]))

            if pair_key not in seen_pairs:
                if (bid, pid) < (nxbid, nxpid):
                    self.pairs[(bid, pid)] = ((nxbid, nxpid), transform)
                else:
                    reverse_transform = mixing_pairs.get((nxbid, nxpid))
                    if reverse_transform is not None:
                        self.pairs[(nxbid, nxpid)] = reverse_transform

                seen_pairs.add(pair_key)

    def _ensure_pair_state(self, key, nspan):
        """Allocate or resize the per-pair diagnostic state for the given pair."""
        state = self._pair_state.get(key)
        if state is None or state["du"].shape[0] != nspan:
            self._pair_state[key] = {
                # Relaxation increment in target space, kept for get_stats.
                "du": np.zeros((nspan, 5), dtype=np.float32),
            }
        return self._pair_state[key]

    def _get_pair(self, bid, pid):
        """Return the two patch objects either side of one mixing plane."""
        patch1 = self._grid[bid].patches[pid]
        (nxbid, nxpid), _ = self.pairs[(bid, pid)]
        patch2 = self._grid[nxbid].patches[nxpid]
        return patch1, patch2

    @profile
    def _exchange_pair(self, bid, pid, flip):
        """Average the two sides' pitch-mean conserved states into a shared target.

        Performs inter-patch communication only; does not apply the target to
        block_view.conserved. Call :meth:`~ember.mixing.MixingPatch.apply` on
        each patch afterwards.
        """
        patch1, patch2 = self._get_pair(bid, pid)

        # Pitch-average each side's face onto its own block_avg.
        patch1.set_block_avg()
        patch2.set_block_avg()
        cons1 = patch1.block_avg.conserved_nd
        cons2 = patch2.block_avg.conserved_nd
        if flip:
            cons2 = cons2[::-1]

        # The baseline the increment is added to. Both sides hold the same
        # target from the exchange onwards, but before the first one each has
        # seeded itself from its own interior, so symmetrise here too.
        target1 = patch1.get_target()
        target2 = patch2.get_target()
        if flip:
            target2 = target2[::-1]
        target = 0.5 * (target1 + target2)

        # du = rf_exchange * (cross-plane average - baseline).
        du = 0.5 * (cons1 + cons2)
        du -= target
        du *= patch1.rf_exchange

        state = self._ensure_pair_state((bid, pid), target.shape[0])
        state["du"][:] = du

        target += du
        patch1.set_target(target)
        patch2.set_target(target[::-1] if flip else target)

    def get_stats(self, bid, pid):
        """Return last-step relaxation increment for one pair.

        Returns
        -------
        dict or None
            Key ``du`` (last relaxation increment in the exchanged target's own
            variables, shape ``(nspan, 5)``). Returns ``None`` if the pair has
            not been exchanged yet.
        """
        state = self._pair_state.get((bid, pid))
        if state is None:
            return None
        return {"du": state["du"].copy()}

    def exchange(self):
        """Compute and write targets for all pairs (no apply step)."""
        for bid, pid in self.pairs.keys():
            _, flip = self.pairs[(bid, pid)]
            self._exchange_pair(bid, pid, flip)


class NonReflectingMixingCommunicator(MixingCommunicator):
    r"""Cross-plane exchange for the non-reflecting mixing plane.

    Where the reflecting plane averages conserved variables outright, this one
    follows :cite:t:`Saxer1993` (his Section 5.5): it flux-averages each side,
    takes the jump in that state across the plane, converts the jump to
    characteristic variables and splits it by direction of propagation -- the
    upstream side owning the upstream-running pressure characteristic and the
    downstream side the other four -- and relaxes the result onto the
    symmetrised baseline of the two sides' targets,

    .. math::

        \mathrm{target}_n = \tfrac{1}{2}(\mathrm{mix}_1 + \mathrm{mix}_2)
            + \mathrm{rf\_exchange}\,\varepsilon_n.

    The target is written in the mix variables :math:`[h_0, s, V_r, V_\theta,
    p]`, which are exactly the quantities the two patches take their
    pitchwise-mean residuals against; they drive only the mean mode of each
    side's boundary condition and leave the harmonics to the non-reflecting
    relations of the patches themselves.

    See Also
    --------
    ember.mixing_nonreflecting : The two patch classes this pairs
    """

    # Jacobian mapping characteristic variables to the space the exchanged
    # target is written in. Its last row must be the static pressure and its
    # first four the quantities an inflow prescribes, because _write_targets
    # expresses Saxer's split of the interface jump by direction of propagation
    # as a pair of row masks on the target vector.
    _chic_to_target = staticmethod(perturbation.chic_to_mix)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scratch buffers, lazily allocated on first exchange
        self._vec1 = None
        self._vec2 = None
        self._jac_buf = None

    def _ensure_buffers(self, nspan):
        """Allocate or resize scratch buffers for the given spanwise size."""
        if self._vec1 is None or self._vec1.shape[0] < nspan:
            self._vec1 = util.empty((nspan, 5))
            self._vec2 = util.empty((nspan, 5))
            self._jac_buf = util.empty((nspan, 5, 5))

    @profile
    def _exchange_pair(self, bid, pid, flip):
        """Compute cross-plane targets and write absolute values into each patch.

        Performs inter-patch communication only; does not apply the targets to
        block_view.conserved.  Call the patches' apply() on each side afterwards.
        """
        patch1, patch2 = self._get_pair(bid, pid)

        b_avg, nspan = self._prepare_pair(patch1, patch2, flip)
        self._write_targets(patch1, patch2, flip, b_avg, nspan, (bid, pid))

    def _prepare_pair(self, patch1, patch2, flip):
        """Symmetrise the cross-plane average and reduce the flux mismatch to chic space.

        Leaves the characteristic mismatch ``dchic`` in the shared scratch
        buffer ``self._vec1[:nspan]``, which :meth:`_write_targets` consumes.

        Returns
        -------
        b_avg : Block
            The symmetrised pitch-averaged state both sides now share, with its
            axial Mach number clipped away from zero. Every Jacobian downstream
            is evaluated on it, so both sides see the same linearisation.
        nspan : int
            Number of span stations; the length the scratch buffers are sliced to.
        """
        # The patches have to agree on a common pitch-avg state
        # before exchanging, before resolving to interface coordinates

        # First area average the flow field into respective block_avg
        patch1.set_block_avg()
        patch2.set_block_avg()

        # Extract conserved variables
        cons1 = patch1.block_avg.conserved_nd
        cons2 = patch2.block_avg.conserved_nd

        # Compute pitch-averages
        patch1.set_flux_avg()
        patch2.set_flux_avg()
        flux1 = patch1.flux_avg_nd
        flux2 = patch2.flux_avg_nd

        if flip:
            cons2 = cons2[::-1]
            flux2 = flux2[::-1]

        # Ensure scratch buffers are allocated
        # One shared for all patches which may have different size
        # so we have to slice to nspan
        nspan = cons1.shape[0]
        self._ensure_buffers(nspan)
        v1 = self._vec1[:nspan]
        v2 = self._vec2[:nspan]
        J = self._jac_buf[:nspan]

        # Take arithmetic average of conserved variables
        v2[:] = cons1
        v2 += cons2
        v2 *= 0.5  # v2 = cons_avg

        # Put back into block_avg for use in perturbation Jacobians
        patch1.block_avg.conserved_nd[...] = v2
        patch2.block_avg.conserved_nd[...] = v2[::-1] if flip else v2

        patch1.block_avg.update_cached_conserved()
        patch2.block_avg.update_cached_conserved()

        # Store the flux difference in v1
        v1[:] = flux2
        v1 -= flux1

        # Clip b_avg axial Mach to Ma_clip before evaluating Jacobians.
        # np.sign is not usable for the direction: it returns 0 at Max == 0, so
        # a stalled station would be clipped to exactly zero axial momentum --
        # the one value the clip exists to keep out of flux_to_primitive, which
        # divides by it two lines below. A station with no direction of its own
        # takes the downstream one.
        Ma_clip = 0.01
        b_avg = patch1.block_avg
        Max = b_avg.Max
        too_low = np.abs(Max) < Ma_clip
        if too_low.any():
            sign = np.where(Max >= 0.0, 1.0, -1.0)
            rhoVx_clip = sign * Ma_clip * b_avg.rho_nd * b_avg.a_nd
            b_avg.conserved_nd[..., 1] = np.where(
                too_low, rhoVx_clip, b_avg.conserved_nd[..., 1]
            )
            b_avg.update_cached_conserved()

        # Convert flux difference to chic difference using sequential Jacobians
        perturbation.flux_to_primitive(b_avg, out=J)
        util.matvec(J, v1, out=v1)  # v1 = dprim
        perturbation.primitive_to_chic(b_avg, out=J)
        util.matvec(J, v1, out=v1)  # v1 = dchic

        return b_avg, nspan

    def _write_targets(self, patch1, patch2, flip, b_avg, nspan, key):
        """Project the characteristic mismatch into target space and write both sides.

        Reads ``dchic`` from the scratch buffer :meth:`_prepare_pair` left it
        in, splits it by direction of propagation -- the upstream-running
        pressure characteristic against the four downstream-running ones, which
        is :cite:t:`Saxer1993` Eq. 5.66 -- and relaxes the resulting mismatch
        onto the symmetrised baseline of the two sides' current targets.

        The split is expressed as a pair of row masks on the target vector, so
        :attr:`_chic_to_target` has to map characteristics into a space whose
        last row is the static pressure and whose first four rows are the
        quantities an inflow prescribes. Mix space satisfies that.
        """
        v1 = self._vec1[:nspan]
        v2 = self._vec2[:nspan]
        J = self._jac_buf[:nspan]

        # Each side's current target, the baseline the mismatch is relaxed onto.
        target1 = patch1.get_target()
        target2 = patch2.get_target()
        if flip:
            target2 = target2[::-1]

        # Split into upstream/downstream contributions in chic space
        v2[:] = v1  # copy dchic into v2
        v1[..., 1:] = 0.0  # v1 = dchic_up (keep upstream acoustic)
        v2[..., 0] = 0.0  # v2 = dchic_dn (keep downstream acoustic and convective)

        # Convert both to target space with the single fused Jacobian
        self._chic_to_target(b_avg, out=J)
        util.matvec(J, v1, out=v1)  # v1 = dtarget_up
        v1[..., :-1] = 0.0  # zero non-P contribution

        util.matvec(J, v2, out=v2)  # v2 = dtarget_dn
        v2[..., -1] = 0.0  # zero P contribution

        # Combine: v1 = dtarget = dtarget_up - dtarget_dn
        # Change to the first four rows comes from downstream chics
        # Change to P comes from upstream chics
        # For some reason need a -ve sign on dtarget_dn here!
        v1 -= v2

        # Relax the target-space mismatch onto the symmetrised baseline. v1
        # holds the error e_n = dtarget; the increment is du = rf_exchange * e_n
        # and the updated target is target_n = 0.5*(target1 + target2) + du.
        state = self._ensure_pair_state(key, nspan)

        v1 *= patch1.rf_exchange  # v1 = du
        state["du"][:] = v1

        # target_n = baseline + du.
        v2[:] = target1
        v2 += target2
        v2 *= 0.5
        v2 += v1

        # 0th-order extrapolation of targets at hub/casing walls
        v2[0] = v2[1]
        v2[-1] = v2[-2]

        # Assign targets back to patches with correct flip
        patch1.set_target(v2)
        patch2.set_target(v2[::-1] if flip else v2)
