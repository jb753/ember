"""Mixing plane boundary condition communication."""

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

    def __init__(
        self,
        grid,
        mixing_pairs,
        rf_mix=0.1,
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
        """Compute cross-plane mix targets and write absolute values into each patch.

        Performs inter-patch communication only; does not apply the targets to
        block_view.conserved.  Call MixingPatch.apply(rf) on each patch afterwards.
        """

        # Get patch object either side of mixing plane
        patch1 = self._grid[bid].patches[pid]
        (nxbid, nxpid), _ = self.pairs[(bid, pid)]
        patch2 = self._grid[nxbid].patches[nxpid]

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

        # Extract pitch-averaged conserved variables and fluxes
        mix1 = patch1.get_target()
        mix2 = patch2.get_target()
        if flip:
            mix2 = mix2[::-1]

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

        # Split into upstream/downstream contributions in chic space
        v2[:] = v1  # copy dchic into v2
        v1[..., 1:] = 0.0  # v1 = dchic_up (keep upstream acoustic)
        v2[..., 0] = 0.0  # v2 = dchic_dn (keep downstream acoustic and convective)

        # Convert upstream/downstream chic to mix space via chic->prim->mix
        perturbation.chic_to_primitive(b_avg, out=J)
        util.matvec(J, v1, out=v1)  # v1 = dprim_up
        # print("dprim_up", v1[nspan // 2])
        util.matvec(J, v2, out=v2)  # v2 = dprim_dn
        # print("dprim_dn", v2[nspan // 2])
        perturbation.primitive_to_mix(b_avg, out=J)
        util.matvec(J, v1, out=v1)  # v1 = dmix_up
        # print("dmix_up", v1[nspan // 2])
        v1[..., :-1] = 0.0  # zero non-P contribution

        # Convert downstream prim to mix space -> v2 = dmix_dn
        util.matvec(J, v2, out=v2)  # reuse same J
        v2[..., -1] = 0.0  # zero P contribution
        # print("dmix_dn", v2[nspan // 2])

        # Combine: v1 = dmix = dmix_up - dmix_dn
        # Change to [ho, s, Vr, Vt] comes from downstream chics
        # Change to P comes from upstream chics
        # For some reason need a -ve sign on dmix_dn here!
        v1 -= v2

        # Relax the mix-space mismatch onto the symmetrised baseline. v1 holds
        # the error e_n = dmix; the increment is du = rf_mix * e_n and the
        # updated target is target_n = 0.5*(mix1 + mix2) + du.
        state = self._ensure_pair_state((bid, pid), nspan)

        v1 *= self._rf_mix  # v1 = du
        state["du"][:] = v1

        # target_n = baseline + du.
        v2[:] = mix1
        v2 += mix2
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
            Key ``du`` (last relaxation increment in mix space, shape
            ``(nspan, 5)``). Returns ``None`` if the pair has not been
            exchanged yet.
        """
        state = self._pair_state.get((bid, pid))
        if state is None:
            return None
        return {"du": state["du"].copy()}

    def exchange(self):
        """Compute and write mix targets for all pairs (no apply step)."""
        for bid, pid in self.pairs.keys():
            _, flip = self.pairs[(bid, pid)]
            self._exchange_pair(bid, pid, flip)
