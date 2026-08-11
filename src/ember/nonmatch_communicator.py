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

from ember.util import apply_perm_flip, pol_to_pseudocart

f32 = np.float32


def _compute_parametric_coords(xrt, const_dim):
    """Compute parametric coordinates for a structured patch face.

    Maps a 3D patch to 2D parametric space (u,v) ∈ [0,1]^2 using
    arc length along grid lines. Both u and v span [0,1] with
    u=0,v=0 at one corner and u=1,v=1 at opposite corner.

    Used to interpolate conserved variables between a non-matching patch
    pair, where two block faces occupy the same physical space but have
    different nodal distributions. The parametric coordinates provide a
    common reference frame for transferring data between patches.

    Parameters
    ----------
    xrt : Array, shape (..., ..., 3)
        Patch coordinates in (x, r, theta). One dimension should be size 1
        (the constant dimension indicating this is a 2D patch face).
    const_dim : int
        Constant dimension (0=i, 1=j, 2=k) that defines the face orientation.

    Returns
    -------
    uv : Array, shape (..., ..., 2)
        Parametric coordinates normalized to [0,1] x [0,1]. Last dimension
        contains [u, v] coordinates. For a patch with varying dimensions
        (i1, i2), the parametric coords have shape (i1, i2, 2).

    Examples
    --------
    >>> # Patch on i=0 face with shape (1, 10, 20, 3)
    >>> xrt = block[patch.slice].xrt
    >>> uv = _compute_parametric_coords(xrt, const_dim=0)
    >>> # Result has shape (1, 10, 20, 2) with u,v ∈ [0,1]
    >>> assert uv[0, 0, 0, :] == [0.0, 0.0]  # Corner
    >>> assert uv[0, -1, -1, :] == [1.0, 1.0]  # Opposite corner
    """
    # Squeeze out the constant dimension to get 2D patch
    xrt_2d = np.squeeze(xrt, axis=const_dim)

    if xrt_2d.ndim != 3 or xrt_2d.shape[-1] != 3:
        raise ValueError(
            f"Expected 2D patch after squeezing const_dim={const_dim}, "
            f"got shape {xrt_2d.shape}"
        )

    ni, nj, _ = xrt_2d.shape

    # Convert to pseudo-Cartesian for distance calculations
    # This handles the polar coordinate metric properly
    xyz = pol_to_pseudocart(xrt_2d)

    # Compute parametric coordinate u along first dimension (i-direction)
    # Arc length between consecutive nodes
    u = np.zeros((ni, nj), dtype=f32, order="F")
    for j in range(nj):
        # Distance between consecutive nodes along i-direction at constant j
        dx = np.diff(xyz[:, j, :], axis=0)
        ds = np.linalg.norm(dx, axis=-1)
        # Cumulative distance
        u[1:, j] = np.cumsum(ds)
        # Normalize to [0, 1]
        total_length = u[-1, j]
        if total_length > 0:
            u[:, j] /= total_length

    # Compute parametric coordinate v along second dimension (j-direction)
    v = np.zeros((ni, nj), dtype=f32, order="F")
    for i in range(ni):
        # Distance between consecutive nodes along j-direction at constant i
        dx = np.diff(xyz[i, :, :], axis=0)
        ds = np.linalg.norm(dx, axis=-1)
        # Cumulative distance
        v[i, 1:] = np.cumsum(ds)
        # Normalize to [0, 1]
        total_length = v[i, -1]
        if total_length > 0:
            v[i, :] /= total_length

    # Stack u and v into (ni, nj, 2) array
    uv_2d = np.stack([u, v], axis=-1)

    # Expand back to original dimensionality by adding the constant dimension
    uv = np.expand_dims(uv_2d, axis=const_dim)

    return uv


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
                self.uv_coords[(bid, pid)] = _compute_parametric_coords(
                    source_xrt, source_patch.const_dim
                )

            # Target patch (with transformation applied)
            if (nxbid, nxpid) not in self.uv_coords:
                target_patch = self._grid[nxbid].patches[nxpid]
                target_xrt = self._grid[nxbid][target_patch.slice].xrt
                target_xrt_transformed = apply_perm_flip(target_xrt, perm, flip)
                self.uv_coords[(nxbid, nxpid)] = _compute_parametric_coords(
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
