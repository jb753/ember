"""Periodic boundary condition communication and data averaging utilities.

This module provides the PeriodicCommunicator class which manages data exchange and averaging
between periodic patch pairs in multi-block grids. The communicator takes connectivity
information from the grid and sets up efficient index mappings that account for arbitrary
coordinate transformations (permutations and flips) between matching periodic patches. The
class prunes bidirectional connectivity to create unidirectional pairs, precomputes transformed
index arrays for fast access, and applies periodic boundary conditions by averaging conserved
flow variables at corresponding spatial locations using Fortran-accelerated routines. This
ensures consistency across periodic boundaries in turbomachinery simulations with blade row
periodicity or other rotationally symmetric configurations.
"""

import numpy as np

import ember.fortran

from ember.util import apply_perm_flip


class PeriodicCommunicator:
    """Manages data communication between periodic patches.

    Takes the output of grid.connectivity.periodic.pair() and sets up matching
    ijk indices for efficient periodic boundary condition communication.

    Parameters
    ----------
    grid : Grid
        The grid containing blocks with periodic patches
    periodic_pairs : dict
        Dictionary from grid.connectivity.periodic.pair() with format:
        {(bid, pid): ((nxbid, nxpid), (perm, flip))}

    Attributes
    ----------
    pairs : dict
        Pruned unidirectional pairs: {(bid, pid): ((nxbid, nxpid), (perm, flip))}
    ijk_node_flat : dict
        Flattened ijk indices: {(bid, pid): indices.reshape(-1, 3)} for both
        source patches (untransformed) and target patches (transformed)
    """

    def __init__(self, grid, periodic_pairs):
        self._grid = grid
        self.pairs = {}
        self.ijk_node_flat = {}
        self.ijk_halo_flat = {}
        self.ijk_ec_flat = {}  # owned edge-cell indices, Fortran 1-based (precomputed)
        # ijk_halo_flat is overwritten to Fortran 1-based ints during setup

        self._prune_pairs(periodic_pairs)
        self._setup_matching_indices()

    def _prune_pairs(self, periodic_pairs):
        """Prune bidirectional pairs to create unidirectional mapping.

        Keeps only one direction for each patch pair by lexicographic ordering.
        """
        seen_pairs = set()

        for (bid, pid), ((nxbid, nxpid), transform) in periodic_pairs.items():
            # Create canonical pair ordering
            pair_key = tuple(sorted([(bid, pid), (nxbid, nxpid)]))

            if pair_key not in seen_pairs:
                # Keep the lexicographically smaller key as the source
                if (bid, pid) < (nxbid, nxpid):
                    self.pairs[(bid, pid)] = ((nxbid, nxpid), transform)
                else:
                    # Get the reverse transform from the other direction
                    reverse_transform = periodic_pairs.get((nxbid, nxpid))
                    if reverse_transform is not None:
                        self.pairs[(nxbid, nxpid)] = reverse_transform

                seen_pairs.add(pair_key)

    @staticmethod
    def _face_to_cell_indices(patch, block, ijk_face_flat):
        """Convert face indices to cell indices along the constant dimension.

        On the low face (const index == 0), the adjacent cell has the same
        index. On the high face (const index == max node index), subtract 1.
        """
        const_dim = patch.const_dim
        const_idx = patch.ijk_lim_abs[const_dim, 0]
        is_high_face = const_idx == block.shape[const_dim] - 1
        if is_high_face:
            ijk_cell_flat = ijk_face_flat.copy()
            ijk_cell_flat[:, const_dim] -= 1
        else:
            ijk_cell_flat = ijk_face_flat
        return ijk_cell_flat

    @staticmethod
    def _cell_to_halo_indices(patch, block, ijk_cell_flat):
        """Convert cell indices to tau_q_halo indices for the boundary halo slot.

        tau_q_halo is (ni+1, nj+1, nk+1, 9) with owned cell c at halo index c+1.
        The halo slot for a boundary face is one step outside the owned range:
          low face:  halo index = cell index      (i.e. cell+1 - 1 = cell)
          high face: halo index = cell index + 2  (i.e. cell+1 + 1 = cell+2)
        Non-const dims are offset by +1 to account for the halo padding.
        """
        const_dim = patch.const_dim
        const_idx = patch.ijk_lim_abs[const_dim, 0]
        is_high_face = const_idx == block.shape[const_dim] - 1
        ijk_halo = ijk_cell_flat.copy()
        ijk_halo += 1  # offset all dims for halo padding
        if is_high_face:
            ijk_halo[:, const_dim] += 1  # one further out on high side
        else:
            ijk_halo[:, const_dim] -= 1  # one further out on low side
        return ijk_halo

    def _setup_matching_indices(self):
        """Setup matching ijk indices for each patch pair.

        Stores node indices (for conserved variable averaging) and halo-slot
        indices (for halo exchange) for both patches: source patch untransformed,
        and target patch with transformation applied.
        """
        for (bid, pid), ((nxbid, nxpid), (perm, flip)) in self.pairs.items():
            # Store original source patch indices as int16 F-contiguous
            # (Fortran average_by_ijk declares integer*2, so matching dtype
            # avoids f2py copying and converting on every call)
            source_patch = self._grid[bid].patches[pid]
            self.ijk_node_flat[(bid, pid)] = np.asfortranarray(
                (source_patch.get_ijk_node().reshape(-1, 3) + 1).astype(np.int16)
            )

            # Store transformed target patch indices as int16 F-contiguous
            target_patch = self._grid[nxbid].patches[nxpid]
            target_ijk = apply_perm_flip(target_patch.get_ijk_node(), perm, flip)
            self.ijk_node_flat[(nxbid, nxpid)] = np.asfortranarray(
                (target_ijk.reshape(-1, 3) + 1).astype(np.int16)
            )

            # Cell indices (0-based) feeding the halo-slot index computation
            source_face = source_patch.get_ijk_face().reshape(-1, 3)
            src_cells = self._face_to_cell_indices(
                source_patch, self._grid[bid], source_face
            )

            target_face = apply_perm_flip(
                target_patch.get_ijk_face(), perm, flip
            ).reshape(-1, 3)
            tgt_cells = self._face_to_cell_indices(
                target_patch, self._grid[nxbid], target_face
            )

            # Halo slot indices into tau_q_halo for exchange_halos()
            self.ijk_halo_flat[(bid, pid)] = self._cell_to_halo_indices(
                source_patch, self._grid[bid], src_cells
            )
            self.ijk_halo_flat[(nxbid, nxpid)] = self._cell_to_halo_indices(
                target_patch, self._grid[nxbid], tgt_cells
            )

            # Precompute Fortran (1-based) indices for exchange_halos.
            # cells is 0-based; owned cell c sits at tau_q_halo[c+1] (0-based)
            # = tau_q_halo[c+2] in Fortran 1-based indexing.
            # ijk_halo_flat is already 0-based halo slot; add 1 for Fortran.
            # Use int32 C-contiguous arrays so f2py passes them without copying.
            for key, cells in (
                ((bid, pid), src_cells),
                ((nxbid, nxpid), tgt_cells),
            ):
                # cells is 0-based; owned cell c is at tau_q_halo[c+1] (0-based)
                # = tau_q_halo[c+2] in Fortran 1-based indexing
                self.ijk_ec_flat[key] = np.asfortranarray((cells + 2).astype(np.int16))
                # ijk_halo_flat was computed 0-based; convert to Fortran 1-based
                self.ijk_halo_flat[key] = np.asfortranarray(
                    (self.ijk_halo_flat[key] + 1).astype(np.int16)
                )

    def apply(self):
        """Apply periodic boundary conditions by averaging conserved variables.

        Loops over all patch pairs and calls Fortran average_by_ijk to average
        the conserved variables at corresponding ijk locations between patches.
        """
        for (bid, pid), ((nxbid, nxpid), _) in self.pairs.items():
            #
            # Get conserved variables (writeable views)
            cons1 = self._grid[bid].conserved_nd
            cons2 = self._grid[nxbid].conserved_nd

            # Get precomputed ijk indices (already int16, F-contiguous, 1-based)
            ijk1 = self.ijk_node_flat[(bid, pid)]
            ijk2 = self.ijk_node_flat[(nxbid, nxpid)]

            # Call Fortran averaging function
            ember.fortran.average_by_ijk(cons1, cons2, ijk1, ijk2, 1.0)

    def exchange_halos(self):
        """Copy periodic neighbour tau/q into the local halo slot of tau_q_halo.

        Called between set_tau_q_soa and set_visc_force.  Each block's halo slot
        at its periodic face is filled with the adjacent block's owned edge-cell
        tau/q, so that set_visc_force averages two real cell values there instead
        of the -edge ghost written by eval_tau_q.
        """
        for (bid, pid), ((nxbid, nxpid), _) in self.pairs.items():
            h1 = self._grid[bid].tau_q_halo  # (ni+1, nj+1, nk+1, 9)
            h2 = self._grid[nxbid].tau_q_halo
            key1 = (bid, pid)
            key2 = (nxbid, nxpid)
            ember.fortran.swap_by_ijk(
                h1,
                h2,
                self.ijk_halo_flat[key1],
                self.ijk_ec_flat[key1],
                self.ijk_halo_flat[key2],
                self.ijk_ec_flat[key2],
            )
