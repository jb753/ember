"""Cusp boundary condition patch for EMBER CFD.

CuspPatch handles cusp boundaries across a modelled trailing edge.

See Also
--------
ember.patch.Patch : Base class for all patches
"""

import itertools

import numpy as np

from ember.basepatch import Patch


class CuspPatch(Patch):
    """Cusp boundary at a modelled trailing edge.

    Marks the faces either side of a zero-thickness trailing edge where two
    block faces meet at a cusp. Each solver step the conserved variables at
    the cusp nodes (the axial range covered by the patch) are averaged across
    the two faces so that the solution remains continuous at the trailing edge.

    Must be on a constant-k face and must span the full j extent of the block.
    Paired with the corresponding face on the other side of the trailing edge
    via :py:meth:`~ember.basepatch.Patch.check_match`.
    """

    _collection_name = "cusp"

    def _get_viable_transforms(self, other_shape):
        """Get viable permutation/flip combinations based on patch shapes.

        Parameters
        ----------
        other_shape : tuple
            Shape of the other patch to match

        Returns
        -------
        list
            List of (perm, flip) tuples that could map self.shape to other_shape
        """
        self_shape = np.array(self.shape)
        other_shape = np.array(other_shape)

        transforms = []

        # Generate all permutations
        for perm in itertools.permutations([0, 1, 2]):
            for r in range(4):
                for flip in itertools.combinations([0, 1, 2], r=r):
                    # Must map other shape onto this shape
                    if tuple(self_shape) == tuple(other_shape[list(perm)]):
                        # Must not flip any unit dimensions
                        if not any(self_shape[i] == 1 for i in flip):
                            transforms.append((perm, flip))

        return transforms

    def attach_to_block(self, block):
        """Attach to block and validate cusp patch constraints."""
        super().attach_to_block(block)

        lim = self.ijk_lim_abs
        block_shape = block.shape

        if self.const_dim != 2:
            raise ValueError(
                f"CuspPatch must be on a constant-k face, but const_dim={self.const_dim}."
            )

        if lim[1, 0] != 0 or lim[1, 1] != block_shape[1] - 1:
            raise ValueError(
                f"CuspPatch must span the entire j extent, but j limits are "
                f"{lim[1, 0]}:{lim[1, 1]} for block j size {block_shape[1]}."
            )

        return self

    def check_match(self, other, rtol=1e-6):
        """Check if this CuspPatch matches another for pairing purposes.

        CuspPatch matching requires x and r coordinates to match within
        tolerance, but allows theta to differ.

        Parameters
        ----------
        other : Patch
            The other patch to compare with
        rtol : float, optional
            Relative tolerance for matching

        Returns
        -------
        Optional[Tuple[tuple, tuple]]
            (perm, flip) if patches match, None otherwise
        """
        if not isinstance(other, CuspPatch):
            return None

        if self.size != other.size:
            return None

        if self.block.Nb != other.block.Nb:
            return None

        transforms = self._get_viable_transforms(other.shape)
        for transform in transforms:
            if not self._compare_coords(
                other, transform, corners_only=True, xr_only=True, rtol=rtol
            ):
                continue
            if self._compare_coords(
                other, transform, corners_only=False, xr_only=True, rtol=rtol
            ):
                return transform

        return None
