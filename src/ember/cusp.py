"""Cusp boundary condition patch.

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

    Must be on a constant-k face. The i and j extents are both arbitrary: a
    blade that does not run the full span leaves a hub or tip gap where there
    is no trailing edge to cut, and the k face there carries a different patch
    (a periodic seam, whose two sides are nodally coincident and so need no
    correction at all). Paired with the corresponding face on the other side of
    the trailing edge via :py:meth:`~ember.patch.Patch.check_match`.

    Every CuspPatch on a block must cover the SAME i and j range; see
    :py:meth:`attach_to_block`.
    """

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
        """Attach to block and validate cusp patch constraints.

        Beyond the constant-k face requirement, every CuspPatch already on the
        block must cover the same i and j range as this one. The seam
        correction in the kernels pairs cell ``(i, j)`` on the ``k=1`` face
        with cell ``(i, j)`` on the ``k=nk`` face directly, with no transform,
        and reads the span from whichever patch
        :py:attr:`~ember.block.Block.i_cusp` and
        :py:attr:`~ember.block.Block.j_cusp` find first --- so two cusps
        disagreeing about their extent would silently correct the wrong cells.

        This also forbids a cusp interrupted mid-span: a second patch on the
        same k face would have to carry the same limits, which makes it overlap
        the first, and the collection rejects overlaps. Each cusp is therefore
        a single contiguous i,j interval and a block holds at most the
        ``k=1``/``k=nk`` pair.
        """
        super().attach_to_block(block)

        if self.const_dim != 2:
            raise ValueError(
                f"CuspPatch must be on a constant-k face, but const_dim={self.const_dim}."
            )

        lim = self.ijk_lim_abs
        for other in block.patches.cusp:
            if other is self:
                continue
            olim = other.ijk_lim_abs
            if not np.array_equal(lim[:2], olim[:2]):
                raise ValueError(
                    f"All CuspPatches on a block must cover the same i and j "
                    f"range, but this patch spans i {lim[0, 0]}:{lim[0, 1]}, "
                    f"j {lim[1, 0]}:{lim[1, 1]} and an existing one spans "
                    f"i {olim[0, 0]}:{olim[0, 1]}, j {olim[1, 0]}:{olim[1, 1]}."
                )

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
