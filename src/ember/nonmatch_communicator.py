"""Non-matching patch boundary condition communication.

This module provides the NonMatchCommunicator class which manages data exchange
between non-matching patch pairs in multi-block grids. Unlike periodic patches that
require identical node distributions, non-matching patches allow arbitrary mesh
refinement changes across block boundaries.

The communicator uses parametric interpolation to transfer conserved variables
between patches with different nodal distributions but identical physical locations.
It precomputes parametric (u,v) coordinates during initialization and uses bilinear
interpolation to transfer data at each timestep.
"""

import numpy as np

from ember.geometry import compute_parametric_coords
from ember.util import apply_perm_flip


class NonMatchCommunicator:
    """Manages data communication between non-matching patches.

    Takes the output of grid.connectivity.nonmatch.pair() and sets up parametric
    interpolation for efficient non-matching boundary condition communication.

    Unlike PeriodicCommunicator which averages at matching ijk indices, this
    communicator interpolates between patches with different node distributions.

    Parameters
    ----------
    grid : Grid
        The grid containing blocks with non-matching patches
    nonmatch_pairs : dict
        Dictionary from grid.connectivity.nonmatch.pair() with format:
        {(bid, pid): ((nxbid, nxpid), (perm, flip))}

    Attributes
    ----------
    pairs : dict
        Pruned unidirectional pairs: {(bid, pid): ((nxbid, nxpid), (perm, flip))}
    uv_coords : dict
        Cached parametric coordinates: {(bid, pid): uv_array}
        where uv_array has shape (..., ..., 2) with [u, v] ∈ [0,1]^2
    """

    def __init__(self, grid, nonmatch_pairs):
        self._grid = grid
        self.pairs = {}
        self.uv_coords = {}

        self._tmp = {}

        self._prune_pairs(nonmatch_pairs)
        self._setup_parametric_coords()

    def _prune_pairs(self, nonmatch_pairs):
        """Prune bidirectional pairs to create unidirectional mapping.

        Keeps only one direction for each patch pair by lexicographic ordering.
        Identical to PeriodicCommunicator pruning logic.
        """
        seen_pairs = set()

        for (bid, pid), ((nxbid, nxpid), transform) in nonmatch_pairs.items():
            # Create canonical pair ordering
            pair_key = tuple(sorted([(bid, pid), (nxbid, nxpid)]))

            if pair_key not in seen_pairs:
                # Keep the lexicographically smaller key as the source
                if (bid, pid) < (nxbid, nxpid):
                    self.pairs[(bid, pid)] = ((nxbid, nxpid), transform)
                else:
                    # Get the reverse transform from the other direction
                    reverse_transform = nonmatch_pairs.get((nxbid, nxpid))
                    if reverse_transform is not None:
                        self.pairs[(nxbid, nxpid)] = reverse_transform

                seen_pairs.add(pair_key)

    def _setup_parametric_coords(self):
        """Compute and cache parametric coordinates for all patches.

        Computes (u,v) ∈ [0,1]^2 parametric coordinates for each patch using
        arc length along grid lines. These are computed once during initialization
        and reused for all timesteps.
        """
        for (bid, pid), ((nxbid, nxpid), (perm, flip)) in self.pairs.items():
            # Source patch
            if (bid, pid) not in self.uv_coords:
                source_patch = self._grid[bid].patches[pid]
                source_xrt = self._grid[bid][source_patch.slice].xrt
                self.uv_coords[(bid, pid)] = compute_parametric_coords(
                    source_xrt, source_patch.const_dim
                )

            # Target patch (with transformation applied)
            if (nxbid, nxpid) not in self.uv_coords:
                target_patch = self._grid[nxbid].patches[nxpid]
                target_xrt = self._grid[nxbid][target_patch.slice].xrt
                target_xrt_transformed = apply_perm_flip(target_xrt, perm, flip)
                self.uv_coords[(nxbid, nxpid)] = compute_parametric_coords(
                    target_xrt_transformed, target_patch.const_dim
                )

    def apply(self):
        """Apply non-matching boundary conditions via interpolation.

        Loops over all patch pairs and interpolates conserved variables between
        patches with different node distributions. Performs bidirectional
        interpolation and averaging for consistency at the interface.
        """
        import ember.fortran

        for (bid, pid), ((nxbid, nxpid), _) in self.pairs.items():
            source_patch = self._grid[bid].patches[pid]
            target_patch = self._grid[nxbid].patches[nxpid]

            Q_src = np.squeeze(self._grid[bid][source_patch.slice].conserved_nd)
            Q_tgt = np.squeeze(self._grid[nxbid][target_patch.slice].conserved_nd)

            uv_src = np.squeeze(self.uv_coords[(bid, pid)])
            uv_tgt = np.squeeze(self.uv_coords[(nxbid, nxpid)])

            u_src = uv_src[:, 0, 0]
            v_src = uv_src[0, :, 1]
            u_tgt = uv_tgt[:, 0, 0]
            v_tgt = uv_tgt[0, :, 1]

            # Lazy-allocate cached temp buffers (zero heap alloc after first call)
            if (bid, pid) not in self._tmp:
                self._tmp[(bid, pid)] = np.empty(
                    Q_src.shape, dtype=np.float32, order="F"
                )
                self._tmp[(nxbid, nxpid)] = np.empty(
                    Q_tgt.shape, dtype=np.float32, order="F"
                )
            buf_src = self._tmp[(bid, pid)]
            buf_tgt = self._tmp[(nxbid, nxpid)]

            # Interpolate tgt->src and src->tgt (each reads original, writes to buf)
            ember.fortran.bilinear_scattered(Q_tgt, u_tgt, v_tgt, uv_src, buf_src)
            ember.fortran.bilinear_scattered(Q_src, u_src, v_src, uv_tgt, buf_tgt)

            # Blend in-place (writes through squeeze views into block _data)
            Q_src[:] = 0.5 * (Q_src + buf_src)
            Q_tgt[:] = 0.5 * (Q_tgt + buf_tgt)
