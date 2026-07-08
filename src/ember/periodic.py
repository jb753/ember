"""Periodic boundary condition patch for EMBER CFD.

PeriodicPatch connects coincident or circumferentially periodic block faces.
Connectivity is detected automatically and conserved variables are averaged
at each time step to enforce periodicity.

See Also
--------
ember.patch.Patch : Base class for all patches
"""

import itertools

import numpy as np

from ember.basepatch import Patch


class PeriodicPatch(Patch):
    """Connected boundary to another block.

    This applies to patches which are exactly coincident, and patches which are periodic in the circumferential direction of an annular domain.
    The patches should come in pairs, and connectivity between the two patches is detected automatically. The conserved variables on the two patches are averaged at each time step to enforce periodicity.

    """

    _collection_name = "periodic"

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

    def check_match(self, other, rtol=1e-6):
        """Check if this PeriodicPatch matches another for pairing purposes.

        PeriodicPatch matching requires all x, r, t coordinates to match
        within tolerance, accounting for periodicity in theta and allowing
        for permutations and flips.

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
        if not isinstance(other, PeriodicPatch):
            return None

        if self.size != other.size:
            return None

        if self.block.Nb != other.block.Nb:
            return None

        transforms = self._get_viable_transforms(other.shape)
        for transform in transforms:
            if not self._compare_coords(other, transform, corners_only=True, rtol=rtol):
                continue
            if self._compare_coords(other, transform, corners_only=False, rtol=rtol):
                return transform

        return None
