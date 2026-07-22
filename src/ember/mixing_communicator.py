"""Mixing plane boundary condition communication.

:class:`MixingCommunicator` serves the reflecting mixing plane of
:mod:`ember.mixing` and :class:`NonReflectingMixingCommunicator` the
non-reflecting one of :mod:`ember.mixing_nonreflecting`. They differ only in the
variables the exchanged target is expressed in; everything else is shared.
"""

from ember import perturbation, util
from ember.util import profile
import numpy as np


class MixingCommunicator:
    r"""Manages data communication between mixing patch pairs.

    The cross-plane flux mismatch :math:`\varepsilon` (projected into mix
    space) is relaxed onto the symmetrised baseline
    :math:`\tfrac{1}{2}(\mathrm{mix}_1 + \mathrm{mix}_2)` with a single
    constant relaxation factor :math:`\mathrm{rf\_mix}`,

    .. math::

        \mathrm{target}_n = \tfrac{1}{2}(\mathrm{mix}_1 + \mathrm{mix}_2)
            + \mathrm{rf\_mix}\,\varepsilon_n.

    The relaxation factor is the same on every multigrid level.
    """

    # Jacobian mapping characteristic variables to the space the exchanged
    # target is written in. Its last row must be the static pressure and its
    # first four the quantities an inflow prescribes, because _write_targets
    # expresses Saxer's split of the interface jump by direction of propagation
    # as a pair of row masks on the target vector.
    _chic_to_target = staticmethod(perturbation.chic_to_mix)

    def __init__(
        self,
        grid,
        mixing_pairs,
        rf_mix=0.01,
    ):
        """Initialize with grid and mixing patch pairs.

        Parameters
        ----------
        grid : Grid
            The grid instance.
        mixing_pairs : dict
            Mapping of mixing patch pair information.
        rf_mix : float, optional
            Constant relaxation factor applied to the cross-plane mix-space
            mismatch. The same value is used on all grid levels. Defaults to
            ``0.01``.
        """
        self._grid = grid
        self.pairs = {}
        self._prune_pairs(mixing_pairs)

        # Scratch buffers, lazily allocated on first exchange
        self._vec1 = None
        self._vec2 = None
        self._jac_buf = None

        # Per-pair diagnostic snapshots, lazily allocated on first exchange.
        # Keys: (bid, pid). Values: dict with 'du' (the relaxation increment in
        # mix space, shape (nspan, 5)).
        self._pair_state = {}

        self._rf_mix = np.float32(rf_mix)

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

    def _ensure_buffers(self, nspan):
        """Allocate or resize scratch buffers for the given spanwise size."""
        if self._vec1 is None or self._vec1.shape[0] < nspan:
            self._vec1 = util.empty((nspan, 5))
            self._vec2 = util.empty((nspan, 5))
            self._jac_buf = util.empty((nspan, 5, 5))

    def _ensure_pair_state(self, key, nspan):
        """Allocate or resize the per-pair diagnostic state for the given pair."""
        state = self._pair_state.get(key)
        if state is None or state["du"].shape[0] != nspan:
            self._pair_state[key] = {
                # Relaxation increment in mix space, kept for get_stats.
                "du": np.zeros((nspan, 5), dtype=np.float32),
            }
        return self._pair_state[key]

    @profile
    def _exchange_pair(self, bid, pid, flip):
        """Compute cross-plane targets and write absolute values into each patch.

        Performs inter-patch communication only; does not apply the targets to
        block_view.conserved.  Call MixingPatch.apply(rf) on each patch afterwards.
        """

        # Get patch object either side of mixing plane
        patch1 = self._grid[bid].patches[pid]
        (nxbid, nxpid), _ = self.pairs[(bid, pid)]
        patch2 = self._grid[nxbid].patches[nxpid]

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
        # print("cons_avg", v2[nspan // 2])

        # Put back into block_avg for use in perturbation Jacobians
        patch1.block_avg.conserved_nd[...] = v2
        patch2.block_avg.conserved_nd[...] = v2[::-1] if flip else v2

        patch1.block_avg.update_cached_conserved()
        patch2.block_avg.update_cached_conserved()

        # Now the average state is consistent across the plane
        # resolve to interface coordinates
        # patch1.resolve_to_interface()
        # patch2.resolve_to_interface()

        # Store the flux difference in v1
        v1[:] = flux2
        v1 -= flux1

        # Clip b_avg axial Mach to Ma_clip before evaluating Jacobians
        Ma_clip = 0.01
        b_avg = patch1.block_avg
        Max = b_avg.Max
        too_low = np.abs(Max) < Ma_clip
        if too_low.any():
            rhoVx_clip = np.sign(Max) * Ma_clip * b_avg.rho_nd * b_avg.a_nd
            b_avg.conserved_nd[..., 1] = np.where(
                too_low, rhoVx_clip, b_avg.conserved_nd[..., 1]
            )
            b_avg.update_cached_conserved()

        # Convert flux difference to chic difference using sequential Jacobians
        # print("cons", b_avg.conserved.mean(axis=0))
        perturbation.flux_to_primitive(b_avg, out=J)
        util.matvec(J, v1, out=v1)  # v1 = dprim
        # print("dprim", v1[nspan // 2])
        perturbation.primitive_to_chic(b_avg, out=J)
        # print("J prim to chic", J[nspan // 2])

        # assert not np.isnan(J[nspan // 2]).any()
        util.matvec(J, v1, out=v1)  # v1 = dchic
        # print("dchic", v1[nspan // 2])

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
        quantities an inflow prescribes. Both mix space and bcond space
        satisfy that, which is what lets the non-reflecting mixing plane reuse
        this method unchanged.
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
        # holds the error e_n = dtarget; the increment is du = rf_mix * e_n and
        # the updated target is target_n = 0.5*(target1 + target2) + du.
        state = self._ensure_pair_state(key, nspan)

        v1 *= self._rf_mix  # v1 = du
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

        # Back to axial/radial coordinates
        # patch1.resolve_from_interface()
        # patch2.resolve_from_interface()

    def get_stats(self, bid, pid):
        """Return last-step relaxation increment for one pair.

        Returns
        -------
        dict or None
            Key ``du`` (last relaxation increment in target space, shape
            ``(nspan, 5)``). Returns ``None`` if the pair has not been
            exchanged yet.
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

    Identical to :class:`MixingCommunicator` but for the space the exchanged
    target is written in. The reflecting plane exchanges mix variables
    :math:`[h_0, s, V_r, V_\theta, p]`, which its patches impose node by node;
    the non-reflecting plane exchanges boundary-condition variables
    :math:`[h_0, s, \tan\alpha, \sin\beta, p]`, which are exactly the targets
    :class:`~ember.inlet_nonreflecting.NonReflectingInletPatch` and
    :class:`~ember.outlet_nonreflecting.NonReflectingOutletPatch` already take
    their pitchwise-mean residuals against.

    So the whole exchange -- the symmetrised cross-plane average, the flux
    mismatch, the split by direction of propagation, the relaxation onto the
    symmetrised baseline -- is inherited unchanged, and only the mean mode of
    each side's boundary condition is driven by it. The harmonics are left to
    the non-reflecting relations of the patches themselves, which is exactly
    how :cite:t:`Saxer1993` (his Section 5.5) specifies the interface.

    See Also
    --------
    ember.mixing_nonreflecting : The two patch classes this pairs
    """

    _chic_to_target = staticmethod(perturbation.chic_to_bcond)
