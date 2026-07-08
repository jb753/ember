"""Non-matching interface patch for EMBER CFD.

NonMatchPatch connects two block faces that occupy the same physical space
but have different nodal distributions, using parametric interpolation.

See Also
--------
ember.patch.Patch : Base class for all patches
ember.periodic.PeriodicPatch : Matching periodic boundaries
"""

import itertools

import numpy as np

from ember.basepatch import Patch


class NonMatchPatch(Patch):
    """Non-matching interface between block faces with different node counts.

    Connects two block faces that occupy the same physical space but have
    different nodal distributions. Conserved variables are transferred via
    bilinear parametric interpolation by
    :class:`~ember.nonmatch_communicator.NonMatchCommunicator`, which
    precomputes parametric coordinates at initialisation and interpolates
    each step.

    Unlike :class:`~ember.periodic.PeriodicPatch`, which requires identical
    node distributions, and :class:`~ember.mixing.MixingPatch`, which
    exchanges only the pitch average, ``NonMatchPatch`` transfers the full
    pointwise solution across the interface.

    Corner x-r coordinates must match between paired faces. Pitchwise and
    spanwise node counts may differ freely.
    """

    _collection_name = "nonmatch"

    def _get_viable_transforms_nonmatch(self, other_shape):
        """Get viable permutation/flip combinations for non-matching patches.

        Non-matching patches allow arbitrary dimension mismatches in the two
        varying dimensions. Only requires that the constant dimension matches
        after transformation.

        Parameters
        ----------
        other_shape : tuple
            Shape of the other patch to match

        Returns
        -------
        list
            List of (perm, flip) tuples that could map patches together
        """
        self_shape = np.array(self.shape)
        other_shape = np.array(other_shape)

        transforms = []

        # Generate all permutations
        for perm in itertools.permutations([0, 1, 2]):
            # For non-matching patches, we don't require dimension sizes to match
            # We only require that the constant dimension is maintained
            # (both patches must have same const_dim after transformation)

            # Get which dimension is constant in self
            self_const_idx = np.where(self_shape == 1)[0]
            if len(self_const_idx) != 1:
                continue  # Skip if ambiguous

            # After permutation, where does self's constant dimension map?
            const_dim_after_perm = perm.index(self_const_idx[0])

            # Check if other also has constant dimension at this position
            if other_shape[const_dim_after_perm] == 1:
                # Valid permutation - constant dimensions align
                # Now generate flipping combinations for non-constant dimensions
                # Only flip non-unit dimensions
                flip_dims = [i for i in range(3) if self_shape[i] > 1]

                # No flipping
                transforms.append((perm, ()))

                # Generate all combinations of flipping non-constant dimensions
                for r in range(1, len(flip_dims) + 1):
                    for flip_subset in itertools.combinations(flip_dims, r=r):
                        transforms.append((perm, flip_subset))

        return transforms

    def check_match(self, other, rtol=1e-6):
        """Check if this NonMatchPatch matches another for pairing purposes.

        NonMatchPatch matching requires only x,r coordinates to match at corners,
        allowing for different nodal distributions in the varying dimensions.
        Theta coordinates are ignored to allow for circumferential mismatch.

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
        if not isinstance(other, NonMatchPatch):
            return None

        transforms = self._get_viable_transforms_nonmatch(other.shape)

        for transform in transforms:
            if self._compare_coords(
                other, transform, corners_only=True, xr_only=True, rtol=rtol
            ):
                return transform

        return None
