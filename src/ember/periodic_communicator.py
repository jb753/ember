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
        self.ij_face_flat = {}  # (npt, 2) face-buffer indices, Fortran 1-based
        self.face_of = {}  # which Block.tau_q_faces entry each patch sits on

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
    def _cell_to_face_indices(patch, block, ijk_cell_flat):
        """Convert cell indices to ``(face, (a, b))`` for ``Block.tau_q_faces``.

        The face buffers index a boundary face by its two free cell
        coordinates -- ``(j, k)`` on an i face, ``(i, k)`` on a j face,
        ``(i, j)`` on a k face -- so a patch's points need only two indices,
        not three. Which face is settled once here rather than carried per
        point: a patch has exactly one constant dimension and sits at one end
        of it, so every point of a given patch lands on the same face.

        Returns
        -------
        tuple[int, Array]
            The face's position in the ``Block.tau_q_faces`` tuple
            (0=i1, 1=ini, 2=j1, 3=jnj, 4=k1, 5=knk), and an ``(npt, 2)``
            array of Fortran 1-based ``(a, b)`` indices into it.
        """
        const_dim = patch.const_dim
        const_idx = patch.ijk_lim_abs[const_dim, 0]
        is_high_face = const_idx == block.shape[const_dim] - 1
        face = 2 * const_dim + (1 if is_high_face else 0)
        free_dims = [d for d in range(3) if d != const_dim]
        ab = ijk_cell_flat[:, free_dims] + 1  # 0-based cell -> Fortran 1-based
        return face, ab

    def _setup_matching_indices(self):
        """Setup matching ijk indices for each patch pair.

        Stores node indices (for conserved variable averaging) and face-buffer
        coordinates (for the boundary tau/q exchange) for both patches: source
        patch untransformed, and target patch with transformation applied.
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

            # Cell indices (0-based) feeding the face-buffer index computation
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

            # Face-buffer coordinates for exchange_faces(): the same points, in
            # the same order, with the same already-applied transform as the
            # node lists above.
            for key, patch, blk, cells in (
                ((bid, pid), source_patch, self._grid[bid], src_cells),
                ((nxbid, nxpid), target_patch, self._grid[nxbid], tgt_cells),
            ):
                face, ab = self._cell_to_face_indices(patch, blk, cells)
                self.face_of[key] = face
                self.ij_face_flat[key] = np.asfortranarray(ab.astype(np.int16))


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

    def exchange_faces(self):
        """Fill each block's face-buffer halo layer from its periodic partner.

        The one seam exchange the viscous pass needs, run between
        ``set_tau_q_faces`` and ``set_visc_force``: it fills the halo layer of
        each block's :attr:`~ember.block.Block.tau_q_faces` with the adjacent
        block's owned edge-cell tau/q, so the face flux there averages two real
        cell values instead of the -edge ghost the producer seeded. Reads layer
        0 (the partner's own edge cells) and writes layer 1 (this side's halo),
        so it never reads and writes the same storage and needs no temporary.

        TWO CALLS PER PAIR, one each way, because ``self.pairs`` is pruned to a
        single key per pair, and a copy cannot move both
        directions at once. A block periodic to itself cannot catch a
        regression here -- both ends are the same block -- which is why the
        gate for this is a two-block case.
        """
        for (bid, pid), ((nxbid, nxpid), _) in self.pairs.items():
            key1, key2 = (bid, pid), (nxbid, nxpid)
            faces1 = self._grid[bid].tau_q_faces
            faces2 = self._grid[nxbid].tau_q_faces
            f1 = faces1[self.face_of[key1]]
            f2 = faces2[self.face_of[key2]]
            idx1 = self.ij_face_flat[key1]
            idx2 = self.ij_face_flat[key2]
            ember.fortran.copy_faces_by_ij(f1, f2, idx1, idx2)
            ember.fortran.copy_faces_by_ij(f2, f1, idx2, idx1)

