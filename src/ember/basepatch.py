"""Base patch classes for boundary condition specification.

Defines the abstract Patch base class, the RevolutionPatch intermediate base
class, and supporting utilities. Concrete patch types live in their own modules
(ember.inlet, ember.outlet, ember.mixing, etc.) and are re-exported from
ember.patch for convenience.

## Limiting index rules

Patches are defined by specifying which block face or part of a face they are
on. A `Patch` constructor takes one argument for each of the three indexing
directions `i`, `j`, and `k` subject to the following rules:

* The first point in a direction is indexed 0; negative indices wrap around
such that -1 is the last point.
* Indices are inclusive, so `i=(0,-1)` spans the entire range of `i`
coordinate.
* Integer arguments are interpreted as a constant value of that index: `i=0`
means the patch spans the first i face; `j=-1` means the patch spans the last j
face. Integer arguments are shorthand for, e.g. `i=(0,0)`.
* Patches must be 2D subsets of an external face of the block. This implies
that at least one constant dimensions must be specified with a value 0 or -1.
* Omitting a direction argument implies the patch should include every point in
that direction, and is shorthand for e.g. `j=(0, -1)`.
* The elements of a direction tuple should be in ascending order after negative
indices are wrapped. `k=(6,4)` is not valid, and neither is `k=(-1, -2)`.
"""

import itertools
import logging
import weakref
from abc import ABC, abstractmethod

import numpy as np

from ember.util import pol_to_pseudocart
from ember import util

logger = logging.getLogger(__name__)


class Patch(ABC):
    # Passive-overlay opt-outs, both False for ordinary patches. Subclasses
    # (e.g. ProbePatch) may set these True to relax the rules below.
    _allow_interior_const = False  # allow a region patch on an interior plane
    _allow_overlap = False  # allow coinciding with another patch

    @property
    @abstractmethod
    def _collection_name(self):
        pass

    @staticmethod
    def _cast_lim(i):
        """Cast a limit to a tuple of two integers."""
        if isinstance(i, int):
            return (i, i)
        elif isinstance(i, (tuple, np.ndarray)) and len(i) == 2:
            if not all(isinstance(x, (int, np.integer)) for x in i):
                raise ValueError(
                    f"i must be an int, a tuple of two ints, or a length-2 numpy array, got {type(i)} with non-integer elements"
                )
            return tuple(int(x) for x in i)
        else:
            raise ValueError(
                f"i must be an int, a tuple of two ints, or a length-2 numpy array, got {type(i)}"
            )

    def __init__(
        self,
        i=(0, -1),
        j=(0, -1),
        k=(0, -1),
        label=None,
    ):
        """Initialize with start and end indices for each dimension.

        Indices are inclusive and a single integer sets a constant value in
        that dimension. See :mod:`ember.patch` for the full index rules.

        Parameters
        ----------
        i : int or tuple
            Start and end indices along the 1st axis.
        j : int or tuple
            Start and end indices along the 2nd axis.
        k : int or tuple
            Start and end indices along the 3rd axis.
        label : str, optional
            String identifier for the patch.
        """

        # Allocate storage for limits
        # Indexed [dimension, start/end]
        self._ijk_lim = np.zeros((3, 2), dtype=int)

        # Set the limits for each dimension
        self.set_i_lim(i)
        self.set_j_lim(j)
        self.set_k_lim(k)

        # Need one of the dimensions to be constant, Error if this patch is not 2D
        if np.sum(np.diff(self._ijk_lim, axis=1) == 0) < 1:
            raise ValueError(
                "Patch must have at least one constant dimension (2D patch required)"
            )

        self._label = label

        # Store weak reference to parent block
        self._block_ref = None

        # Store block view
        self._block_view = None
        self._block_view_offset_1 = None
        self._block_view_offset_2 = None

        self._setup()

    def _setup(self):
        """Hook for subclass attribute initialisation. Called at end of ``__init__``."""

    def __setstate__(self, state):
        """Restore pickled state, defaulting any attributes added since the pickle was made.

        ``_setup()`` seeds today's default instance attributes first (patch
        subclasses gain new cache/relaxation state over time, e.g. a solver
        tuning change), then the pickled state is applied on top. Without
        this, an EMB file written by an older ember version would unpickle
        objects missing whatever attributes were added since, raising
        AttributeError the first time solver code reads one of them.
        """
        self._setup()
        self.__dict__.update(state)

    def _set_lim(self, dim, value):
        """Set limits for specified dimension."""
        self._ijk_lim[dim] = self._cast_lim(value)

    def _compare_coords(
        self, other, transform, corners_only=False, xr_only=False, rtol=1e-6
    ):
        """Compare coordinates of this patch against another after a transform.

        Extracts coordinates, applies pitch-wrapping on theta, computes an
        absolute tolerance, applies the transform to the other patch's coords,
        and returns whether all points agree within tolerance.

        Parameters
        ----------
        other : Patch
            The other patch to compare with
        transform : tuple
            (perm, flip) to apply to the other patch's coordinates
        corners_only : bool, optional
            If True, compare only the corner points of each patch.
        xr_only : bool, optional
            If True, compare only x and r coordinates (ignore theta).
            If False, convert to pseudo-Cartesian space before comparing.
        rtol : float, optional
            Relative tolerance for pitch-wrapping and distance comparison.

        Returns
        -------
        bool
            True if all compared points are within tolerance.
        """
        perm, flip = transform

        xrt_self = self.block[self.slice].xrt.copy()
        xrt_other = other.block[other.slice].xrt.copy()

        # Pitch-wrap theta on both
        pitch = self.block[self.slice].pitch
        for xrt in (xrt_self, xrt_other):
            t = np.mod(xrt[..., 2], pitch)
            xrt[..., 2] = np.where(t / pitch > (1.0 - rtol), 0.0, t)

        atol = rtol * max(np.ptp(xrt_self[..., 0]), np.ptp(xrt_self[..., 1]))

        xrt_other_t = util.apply_perm_flip(xrt_other, perm, flip)

        if corners_only:
            xrt_self = util.corners(xrt_self, axis_exclude=-1)
            xrt_other_t = util.corners(xrt_other_t, axis_exclude=-1)

        if xr_only:
            a = xrt_self[..., :2]
            b = xrt_other_t[..., :2]
        else:
            a = pol_to_pseudocart(xrt_self)
            b = pol_to_pseudocart(xrt_other_t)

        distances = np.linalg.norm(a - b, axis=-1)

        return np.all(distances <= atol)

    def _validate_and_resolve_limits(self):
        """Validate patch limits and return absolute indices."""

        if self._block_ref is None:
            # Do not need block if all indices are positive
            # But cannot validate in bounds
            if (self._ijk_lim >= 0).all():
                ijk_lim_abs = self._ijk_lim
            else:
                raise ValueError(
                    "Patch limits contain negative indices but patch is not attached to a block. Call attach_to_block() first."
                )
            block_shape = None
        else:
            block_shape = np.array(self.block.shape).reshape((3, 1))
            # Convert negative indices to positive using block shape
            ijk_lim_abs = np.where(
                self._ijk_lim < 0, block_shape + self._ijk_lim, self._ijk_lim
            )
            # Check that limits are within bounds
            if np.any(ijk_lim_abs >= block_shape):
                raise ValueError(
                    f"Patch limits {self._ijk_lim} are out of bounds for block shape {block_shape.flatten()}."
                )

        # Should not have negative indices after conversion
        if np.any(ijk_lim_abs < 0):
            raise ValueError(
                f"Patch limits out of bounds {self._ijk_lim} contain negative indices after absolute conversion {ijk_lim_abs}."
            )
        # Should not have start > end after conversion
        if np.any(np.diff(ijk_lim_abs, axis=1) < 0):
            raise ValueError(
                f"Patch limits {ijk_lim_abs.tolist()} out of bounds: start index greater than end index."
            )

        # Should either be on start or end of the constant dimension
        # If not a point probe
        npts = np.prod(np.diff(ijk_lim_abs, axis=1) + 1)
        if npts > 1 and block_shape is not None and not self._allow_interior_const:
            const_ind = ijk_lim_abs[self.const_dim, 0]
            if const_ind != 0 and const_ind != block_shape[self.const_dim] - 1:
                raise ValueError(
                    f"Patch limits {ijk_lim_abs.tolist()} out of bounds: constant dimension is not at start or end."
                )

        return ijk_lim_abs

    def _get_offset_slice(self, offset):
        """Slice object offset along constant dimension.

        Parameters
        ----------
        offset : int
            Index offset along the constant dimension. Positive if patch at
            const_dim == 0, negative if patch at const_dim > 0.
        """
        slices = []
        for lim in self._ijk_lim:
            # Apply offset to constant dimension
            if lim[0] == lim[1]:  # Constant dimension
                adjusted_lim = lim + offset if lim[0] == 0 else lim - offset
            else:
                adjusted_lim = lim

            # Create slice
            if adjusted_lim[1] == -1:
                slices.append(slice(int(adjusted_lim[0]), None))
            else:
                slices.append(slice(int(adjusted_lim[0]), int(adjusted_lim[1] + 1)))

        return tuple(slices)

    def _copy(self, c):
        """Copy subclass-specific state onto a freshly constructed patch ``c``."""

    def __repr__(self):
        """String representation of the patch."""
        # Convert numpy arrays to Python tuples for cleaner display
        i_lim = tuple(int(x) for x in self._ijk_lim[0])
        j_lim = tuple(int(x) for x in self._ijk_lim[1])
        k_lim = tuple(int(x) for x in self._ijk_lim[2])
        return f"{self.__class__.__name__}(i={i_lim}, j={j_lim}, k={k_lim}, label={self.label!r})"

    def set_i_lim(self, i):
        """Set the start and end indices on the i dimension.

        Indices are inclusive and a single integer sets a constant value in
        that dimension. See :mod:`ember.patch` for the full index rules.

        Parameters
        ----------
        i : int or tuple
            Start and end indices along the 1st axis.
        """
        self._set_lim(0, i)

    def set_j_lim(self, j):
        """Set the start and end indices on the j dimension.

        Indices are inclusive and a single integer is shorthand for a constant
        face, e.g. ``j=0`` is equivalent to ``j=(0, 0)``. See
        :mod:`ember.patch` for the full index rules.

        Parameters
        ----------
        j : int or tuple
            Start and end indices along the 2nd axis.
        """
        self._set_lim(1, j)

    def set_k_lim(self, k):
        """Set the start and end indices on the k dimension.

        Indices are inclusive and a single integer is shorthand for a constant
        face, e.g. ``k=0`` is equivalent to ``k=(0, 0)``. See
        :mod:`ember.patch` for the full index rules.

        Parameters
        ----------
        k : int or tuple
            Start and end indices along the 3rd axis.
        """
        self._set_lim(2, k)

    def set_label(self, label):
        """Set patch label."""
        self._label = label

    def get_ijk_face(self, perm=(0, 1, 2), flip=()):
        """Block indices for faces on the patch.

        For example the constant k face bounded by (i -> i+1) and (j -> j+1) has
        indices (i, j, k).

        Parameters
        ----------
        perm : tuple of int, optional
            Permutation of the dimensions for the output. Default is (0, 1, 2) which
            corresponds to (i, j, k).
        flip : tuple of int, optional
            Dimensions to flip in the output. Default is () which means no flipping.
        """

        ijk_node = self.get_ijk_node().copy()

        # We need to exclude indices j==jmax and k==kmax if on const i face, etc
        match self.const_dim:
            case 0:
                # Constant i face, exclude jmax and kmax
                ijk_face = ijk_node[:, :-1, :-1, :]
            case 1:
                # Constant j face, exclude imax and kmax
                ijk_face = ijk_node[:-1, :, :-1, :]
            case 2:
                # Constant k face, exclude imax and jmax
                ijk_face = ijk_node[:-1, :-1, :, :]
            case _:
                raise ValueError("Invalid constant dimension")

        # Apply permutation and flipping
        ijk_face = util.apply_perm_flip(ijk_face, perm, flip)

        return ijk_face

    def get_ijk_node(self, perm=(0, 1, 2), flip=()):
        """Block indices for nodes on the patch.

        Parameters
        ----------
        perm : tuple of int, optional
            Permutation of the dimensions for the output. Default is (0, 1, 2) which
            corresponds to (i, j, k).

        flip : tuple of int, optional
            Dimensions to flip in the output. Default is () which means no flipping.
        """

        # Get limits for each dimension compatible with range
        ijk_lim = self.ijk_lim_abs.copy()
        ijk_lim[:, 1] += 1

        # Generate the ijk vectors (these can have different lengths)
        i_vec = np.arange(ijk_lim[0, 0], ijk_lim[0, 1])
        j_vec = np.arange(ijk_lim[1, 0], ijk_lim[1, 1])
        k_vec = np.arange(ijk_lim[2, 0], ijk_lim[2, 1])

        # Meshgrid to get nodal indices
        ijk_node = np.stack(np.meshgrid(i_vec, j_vec, k_vec, indexing="ij"), axis=-1)

        # Permutation and flipping
        # Apply permutation to spatial dimensions, keep coordinate index (last dim)
        ijk_node = util.apply_perm_flip(ijk_node, perm, flip)

        return ijk_node

    def attach_to_block(self, block):
        """Attach this patch to a block and validate limits against block shape.

        Do not call directly; attachment is handled automatically when a patch
        is added to :py:attr:`~ember.block.Block.patches` via
        :py:class:`~ember.collections.BlockPatchCollection`.

        Parameters
        ----------
        block : :py:class:`~ember.block.Block`
            The block this patch belongs to.
            A weak reference is stored.
        """
        if block is None:
            raise ValueError("Cannot attach patch to None block")

        block_shape = np.array(block.shape)
        self._block_ref = weakref.ref(block)

        # Check that block is 3D
        if block_shape.size != 3:
            raise ValueError(
                f"Patches require 3D blocks (ndim=3), but block has {block_shape.size} dimensions. "
                f"Got block shape={tuple(block_shape.flatten())}."
            )

        # Validate patch limits against block shape
        self._validate_and_resolve_limits()

        # Cache block_view for real Block objects (after validation resolves limits)
        if self._block_ref is not None:
            self._block_view = block[self.slice]
            self._block_view_offset_1 = block[self._get_offset_slice(1)]
            self._block_view_offset_2 = block[self._get_offset_slice(2)]

    def check_match(self, other, rtol=1e-6):
        """Check if this patch matches another patch for pairing purposes.

        Base implementation always returns None. Subclasses should override
        this method to implement their specific matching criteria.

        Parameters
        ----------
        other : Patch
            The other patch to compare with
        rtol : float, optional
            Relative tolerance for matching
        """
        return None

    def copy(self):
        """Return a new unattached patch of the same type with the same limits, label, and boundary condition state.

        The returned patch is fully independent: it shares no mutable state with
        the original and is not attached to any block. Attach it to a block via
        ``block.patches.append(copy)`` before using geometry-dependent properties.

        Boundary condition parameters (e.g. stagnation conditions on an inlet,
        static pressure on an outlet) are copied; any cached solver state that
        depends on block geometry is not.
        """
        c = self.__class__(
            i=self._ijk_lim[0],
            j=self._ijk_lim[1],
            k=self._ijk_lim[2],
            label=self.label,
        )
        self._copy(c)
        return c

    @property
    def block(self):
        """Access the parent block this patch is attached to."""
        if self._block_ref is None:
            raise ValueError(
                "Patch is not attached to any block. Call attach_to_block() first."
            )

        block = self._block_ref()
        if block is None:
            raise ValueError("Block has been garbage collected")

        return block

    @property
    def block_view(self):
        """Sliced view of the parent block at this patch location; :class:`~ember.block.Block` with shape :attr:`shape`.

        Equivalent to ``block[patch.slice]``. Cached at :meth:`attach_to_block`
        to avoid repeated sliced Block creation overhead.
        """
        if not self._block_view:
            raise ValueError(
                "Patch is not attached to any block. Call attach_to_block() first."
            )
        return self._block_view

    @property
    def block_view_offset_1(self):
        """Sliced view one layer interior to the patch face; :class:`~ember.block.Block` with shape :attr:`shape`.

        Used to read the outgoing characteristic state (e.g. entropy at a
        subsonic outlet) from the first interior layer. Equivalent to
        ``block[patch.slice]`` offset by one along the constant dimension.
        Cached at :meth:`attach_to_block` to avoid repeated sliced Block
        creation overhead.
        """
        if not self._block_view_offset_1:
            raise ValueError(
                "Patch is not attached to any block. Call attach_to_block() first."
            )
        return self._block_view_offset_1

    @property
    def block_view_offset_2(self):
        """Sliced view two layers interior to the patch face; :class:`~ember.block.Block` with shape :attr:`shape`.

        Used together with :attr:`block_view_offset_1` to linearly (two-point)
        extrapolate the outgoing characteristic state to the boundary face,
        ``X_face = 2 * X_1 - X_2``. Equivalent to ``block[patch.slice]`` offset
        by two along the constant dimension. Cached at :meth:`attach_to_block`
        to avoid repeated sliced Block creation overhead.
        """
        if not self._block_view_offset_2:
            raise ValueError(
                "Patch is not attached to any block. Call attach_to_block() first."
            )
        return self._block_view_offset_2

    @property
    def const_dim(self):
        """Axis of the constant dimension; ``int`` in ``{0, 1, 2}``."""
        cdim = np.where(np.diff(self._ijk_lim, axis=1) == 0)[0]
        if cdim.size > 1:
            raise ValueError("Patch has ambigous constant dimension")
        return cdim[0]

    @property
    def ien(self):
        """End index in the i dimension; ``int``."""
        return self._ijk_lim[0, 1]

    @property
    def ijk_lim_abs(self):
        """Limits with negative indices resolved to positive; ``ndarray`` of shape ``(3, 2)``."""
        return self._validate_and_resolve_limits()

    @property
    def ist(self):
        """Start index in the i dimension; ``int``."""
        return self._ijk_lim[0, 0]

    @property
    def jen(self):
        """End index in the j dimension; ``int``."""
        return self._ijk_lim[1, 1]

    @property
    def jst(self):
        """Start index in the j dimension; ``int``."""
        return self._ijk_lim[1, 0]

    @property
    def ken(self):
        """End index in the k dimension; ``int``."""
        return self._ijk_lim[2, 1]

    @property
    def kst(self):
        """Start index in the k dimension; ``int``."""
        return self._ijk_lim[2, 0]

    @property
    def label(self):
        """String identifier for the patch; ``str`` or ``None``."""
        return self._label

    @property
    def shape(self):
        """Extent of the patch in each dimension as ``(ni, nj, nk)``; the constant dimension is always 1."""
        return tuple(int(x) for x in (np.diff(self.ijk_lim_abs, axis=1).flatten() + 1))

    @property
    def size(self):
        """Number of nodes on the patch; ``int``, equal to the product of :attr:`shape`."""
        return np.prod(self.shape)

    @property
    def slice(self):
        """``tuple`` of ``slice`` objects for indexing the parent block array."""
        return self._get_offset_slice(offset=0)

    @property
    def xrt_centre(self):
        """Centre coordinates of the patch as ``(x, r, t)``; ``ndarray`` of shape ``(3,)``."""
        block = self.block  # Will raise if not attached
        xrt_corner = util.corners(block[self.slice].xrt, axis_exclude=-1)
        return np.mean(xrt_corner, axis=0)

    # begin property
    # end property


class RevolutionPatch(Patch):
    """Patch on a surface of revolution.

    Intermediate base class for patches that require surface-of-revolution
    geometry (inlet, outlet, mixing). A surface of revolution is an annular or
    axisymmetric surface where one patch axis is purely circumferential (the
    pitch direction, along which only theta varies) and the other runs
    meridionally from hub to tip (the span direction, along which both x and r
    vary).

    When a patch is added to a block via :py:attr:`~ember.block.Block.patches`,
    :meth:`attach_to_block` automatically identifies the span and pitch axes from
    the block geometry. It raises :py:exc:`ValueError` if the geometry is not a
    surface of revolution (i.e. a pitch axis with constant x and r cannot be
    found).

    The key operation provided by this class is pitch-averaging: computing a
    circumferentially averaged flow state that subclasses (inlet, outlet, mixing)
    use to apply boundary conditions.
    """

    def _setup(self):
        super()._setup()
        # Cached span and pitch axes
        self._dim_span = None
        self._dim_pitch = None
        self._spf = None

        self._rot_to = None
        self._rot_from = None
        self._rot_buf = None
        self._block_avg = None
        self._weight_pitch = None
        self._dA_node = None
        self._dA_full = None

    def _check_attached(self):
        """Raise ValueError if this patch is not attached to a block."""
        if self._block_ref is None:
            raise ValueError("Patch is not attached to a block.")
        if self._block_avg is not None:
            try:
                self._block_avg.fluid
            except ValueError:
                try:
                    self._block_avg.set_fluid(self.block.fluid)
                except ValueError:
                    pass

    def _check_match_xr(self, other, rtol=1e-5):
        """Match another patch on meridional geometry alone, ignoring theta.

        The geometric half of the matching test used by the patch types that
        exchange only pitch-averaged data across a plane -- the mixing planes,
        whose two sides need not share a pitchwise node count or even a blade
        count, so only x and r can be compared. Callers supply their own type
        and spanwise-size guards first; this method assumes the two patches are
        already candidates for pairing.

        Every combination of spanwise and pitchwise flips is tried against the
        corner coordinates, and the surviving candidate is confirmed by
        comparing the span fractions of :attr:`spf`, which detects a spanwise
        reversal that the corners alone cannot when the two sides are
        symmetric about midspan.

        Parameters
        ----------
        other : Patch
            The other patch to compare with.
        rtol : float, optional
            Relative tolerance for matching.

        Returns
        -------
        bool or None
            None if the patches do not match. False if they match with no
            spanwise flip needed. True if they match but ``other``'s span must
            be reversed. Always test with ``is not None``; do not use as a bare
            truthiness check since False is a valid match result.
        """
        perm = [0, 0, 0]
        perm[self.const_dim] = other.const_dim
        perm[self.span_dim] = other.span_dim
        perm[self.pitch_dim] = other.pitch_dim
        perm = tuple(perm)

        flip_axes = [ax for ax in (self.span_dim, self.pitch_dim) if self.shape[ax] > 1]
        flip_candidates = [
            combo
            for r in range(len(flip_axes) + 1)
            for combo in itertools.combinations(flip_axes, r)
        ]

        for flip in flip_candidates:
            if not self._compare_coords(
                other, (perm, flip), corners_only=True, xr_only=True, rtol=rtol
            ):
                continue

            span_flipped = self.span_dim in flip
            other_spf = 1.0 - other.spf[::-1] if span_flipped else other.spf
            if np.allclose(self.spf, other_spf, atol=1e-4, rtol=0):
                return span_flipped

            err = np.abs(self.spf - other_spf)
            logger.debug(
                f"spf mismatch with flip {flip}: "
                f"self ends {self.spf[(0, -1),]}, other ends {other_spf[(0, -1),]}, "
                f"max abs error {err.max()}, mean abs error {err.mean()}"
            )

        return None

    @property
    def _std_perm(self):
        """Permutation to standard (const, span, pitch) axis order."""
        return (self.const_dim, self.span_dim, self.pitch_dim)

    def _build_rot_matrices(self, inward=True):
        """Compute xi, cosxi/sinxi and build rotation matrix pairs.

        Derives the meridional face-normal angle from block_view geometry,
        flips to point inward, averages to nodes, then builds rotation matrices.

        Parameters
        ----------
        inward : bool
            If True, rotation aligns with the inward-pointing face normal.
            If False, shifts xi by pi so rotation aligns with the outward normal.
        """
        x = self._block_view.x
        r = self._block_view.r
        xm = x.mean(axis=self.pitch_dim).squeeze()
        rm = r.mean(axis=self.pitch_dim).squeeze()
        dx_face = np.diff(xm)
        dr_face = np.diff(rm)
        xi = np.arctan2(dx_face, -dr_face)

        # Flip xi so it always points inward
        block = self.block
        xr_patch = block.xrt[self.slice][..., :2].mean(axis=self.pitch_dim).squeeze()
        xr_offset = (
            block.xrt[self._get_offset_slice(1)][..., :2]
            .mean(axis=self.pitch_dim)
            .squeeze()
        )
        inward_vec = xr_offset - xr_patch  # (nspan, 2)
        inward_face = 0.5 * (inward_vec[:-1] + inward_vec[1:])  # (nspan-1, 2)
        dot = inward_face[:, 0] * np.cos(xi) + inward_face[:, 1] * np.sin(xi)
        xi = np.where(dot < 0, xi + np.pi, xi)

        # Average xi to nodes
        xi_node = np.empty(len(xi) + 1, dtype=xi.dtype)
        xi_node[0] = xi[0]
        xi_node[1:-1] = 0.5 * (xi[:-1] + xi[1:])
        xi_node[-1] = xi[-1]

        angle = xi_node if inward else xi_node + np.pi
        c = np.cos(angle).astype(np.float32)
        s = np.sin(angle).astype(np.float32)
        n = len(c)
        rot_to = np.empty((n, 2, 2), dtype=np.float32, order="F")
        rot_to[:, 0, 0] = c
        rot_to[:, 0, 1] = s
        rot_to[:, 1, 0] = -s
        rot_to[:, 1, 1] = c
        rot_from = np.empty((n, 2, 2), dtype=np.float32, order="F")
        rot_from[:, 0, 0] = c
        rot_from[:, 0, 1] = -s
        rot_from[:, 1, 0] = s
        rot_from[:, 1, 1] = c
        bcast_shape = [1, 1, 1]
        bcast_shape[self._dim_span] = -1
        self._rot_to = rot_to.reshape(bcast_shape + [2, 2])
        self._rot_from = rot_from.reshape(bcast_shape + [2, 2])

    def set_block_avg(self):
        """Compute pitch-averaged conserved variables and store in block_avg.

        Uses node-based pitch weights to compute a weighted sum of
        ``block_view.conserved_nd`` over the pitch dimension, writing the result
        directly into ``self.block_avg.conserved_nd``.
        """
        import ember.fortran as ft

        cons = self.block_view.conserved_nd
        w = self.weight_pitch.ravel()
        dest = self.block_avg.conserved_nd
        ni, nj, nk = self.block_view.shape
        if self.pitch_dim == 0:
            ft.pitch_avg_i(cons, w, dest.reshape(nj, nk, 5))
        elif self.pitch_dim == 1:
            ft.pitch_avg_j(cons, w, dest.reshape(ni, nk, 5))
        else:
            ft.pitch_avg_k(cons, w, dest.reshape(ni, nj, 5))
        self.block_avg.update_cached_conserved()

    def attach_to_block(self, block):
        """Attach to block and detect surface-of-revolution geometry.

        Calls the base Patch attach, then determines span/pitch dimensions
        and computes meridional properties. Raises ValueError if the patch
        is not a surface of revolution.
        """
        super().attach_to_block(block)

        if self._block_ref is None:
            return

        # Determine if we are a surface of revolution
        # and set span and pitch dimensions accordingly
        x = self._block_view.x
        r = self._block_view.r
        Lref = max(np.ptp(x), np.ptp(r))
        rtol = 1e-4
        atol = rtol * Lref
        # Loop over dimensions to find span and pitch
        self._dim_pitch = None
        self._dim_span = None
        for dim in range(3):
            if dim == self.const_dim:
                continue
            dx = np.diff(x, axis=dim)
            dr = np.diff(r, axis=dim)
            # If no variation in x or r along this axis, it is pitch
            if (np.abs(dx) <= atol).all() and (np.abs(dr) <= atol).all():
                self._dim_pitch = dim
            else:
                self._dim_span = dim

        # If we didn't find both span and pitch, raise
        if self._dim_pitch is None or self._dim_span is None:
            self._dim_pitch = None
            self._dim_span = None
            raise ValueError(
                "Patch is not a surface of revolution: "
                "could not identify both span and pitch dimensions."
            )

        # Compute node-based pitch weights: fraction of block.pitch at each node
        # Permute to (const, span, pitch) then squeeze const -> (nspan, npitch)
        t_sp = self._block_view.t.transpose(self._std_perm).squeeze(
            axis=0
        )  # (nspan, npitch)
        t1d = t_sp[0]  # theta values along pitch, shape (npitch,)
        # Midpoint intervals: dt[k] = t_mid[k] - t_mid[k-1]
        t_mid = 0.5 * (t1d[:-1] + t1d[1:])
        dt = np.empty_like(t1d)
        dt[0] = t_mid[0] - t1d[0]
        dt[1:-1] = t_mid[1:] - t_mid[:-1]
        dt[-1] = t1d[-1] - t_mid[-1]
        # Shape to broadcast against block_view: place weights at pitch_dim
        w = dt / block.pitch
        shape = [1, 1, 1]
        shape[self._dim_pitch] = -1
        self._weight_pitch = w.reshape(shape)

        # If we found a span direction, set span fraction vector
        xm = x.mean(axis=self.pitch_dim).squeeze()
        rm = r.mean(axis=self.pitch_dim).squeeze()
        ds = np.sqrt(np.diff(xm) ** 2 + np.diff(rm) ** 2)
        spf_raw = np.cumsum(np.concatenate(([0.0], ds)))
        self._spf = spf_raw / spf_raw[-1]

        # Compute pitch-normalised face area fractions
        # Note: use if/elif rather than tuple indexing to avoid eagerly evaluating
        # all three dA properties, which would error if xrt is not yet set.
        if self.const_dim == 0:
            dA_raw = np.linalg.norm(self._block_view.dAi, axis=0)
        elif self.const_dim == 1:
            dA_raw = np.linalg.norm(self._block_view.dAj, axis=0)
        else:
            dA_raw = np.linalg.norm(self._block_view.dAk, axis=0)
        A = np.sum(dA_raw, axis=self.pitch_dim)

        # Allocate pitch-averaged block with mean coordinates along pitch dim
        import ember.block as _block_mod

        nspan = self._block_view.shape[self.span_dim]

        # Compute node-centred span area weights via trapezoid face-to-node mapping
        const_dim_reduced = (
            self.const_dim if self.const_dim < self.pitch_dim else self.const_dim - 1
        )
        A_face = A.squeeze(axis=const_dim_reduced)  # shape (nspan-1,)
        ws = np.empty(nspan)
        ws[0] = A_face[0] / 2
        ws[1:-1] = (A_face[:-1] + A_face[1:]) / 2
        ws[-1] = A_face[-1] / 2
        self._dA_node = ws

        def _face_to_node_1d(arr, axis):
            a = np.moveaxis(arr, axis, 0)
            out = np.empty((a.shape[0] + 1,) + a.shape[1:], dtype=arr.dtype)
            out[0] = a[0] / 2
            out[1:-1] = (a[:-1] + a[1:]) / 2
            out[-1] = a[-1] / 2
            return np.moveaxis(out, 0, axis)

        dA_nodes = _face_to_node_1d(dA_raw, self.span_dim)
        dA_nodes = _face_to_node_1d(dA_nodes, self.pitch_dim)
        self._dA_full = dA_nodes

        x_avg = self._block_view.x.mean(axis=self.pitch_dim).squeeze()
        r_avg = self._block_view.r.mean(axis=self.pitch_dim).squeeze()
        t_avg = self._block_view.t.mean(axis=self.pitch_dim).squeeze()
        self._block_avg = _block_mod.Block(shape=(nspan,))
        self._block_avg.set_L_ref(block.L_ref)
        try:
            self._block_avg.set_fluid(block.fluid)
        except ValueError:
            pass
        self._block_avg.set_x(x_avg)
        self._block_avg.set_r(r_avg)
        self._block_avg.set_t(t_avg)
        # self._block_avg.set_conserved(util.zeros((nspan, 5)))

        # Scratch buffer for 2x2 rotation matvec output
        self._rot_buf = util.empty(self._block_view.shape + (2,))

    def resolve_from_interface(self):
        """Rotate block_view momentum in-place from (norm, span) to (x, r) coordinates.

        Inverse of ``resolve_to_interface``::

            rhoV_norm -> rhoVx = cosxi * rhoV_norm - sinxi * rhoV_span
            rhoV_span -> rhoVr = sinxi * rhoV_norm + cosxi * rhoV_span
        """
        cons = self.block_view.conserved_nd
        util.matvec(self._rot_from, cons[..., 1:3], out=self._rot_buf)
        cons[..., 1:3] = self._rot_buf
        self.block_view.update_cached_conserved()

    def resolve_to_interface(self):
        """Rotate block_view momentum in-place from (x, r) to (norm, span) coordinates.

        Modifies ``block_view.conserved`` so that the axial and radial momentum
        components become the interface-normal and interface-span components::

            rhoVx -> rhoV_norm =  cosxi * rhoVx + sinxi * rhoVr
            rhoVr -> rhoV_span = -sinxi * rhoVx + cosxi * rhoVr

        Uses the pre-computed rotation matrix ``_rot_to`` broadcast along
        ``span_dim`` to match the full block shape.
        """
        cons = self.block_view.conserved_nd
        util.matvec(self._rot_to, cons[..., 1:3], out=self._rot_buf)
        cons[..., 1:3] = self._rot_buf
        self.block_view.update_cached_conserved()

    def smooth_pitch_121(self, field, alpha):
        r"""Apply a periodic 1-2-1 smoothing pass along the pitch axis.

        Returns ``alpha * smoothed + (1 - alpha) * field`` where ``smoothed``
        is one pass of the discrete 1-2-1 filter
        ``f[i] = (f[i-1] + 2*f[i] + f[i+1]) / 4`` with periodic wrap along
        :attr:`pitch_dim`. The pitch direction is circumferential, so periodic
        wrap is exact for an annular passage.

        The 1-2-1 filter has amplification :math:`\cos^2(k\Delta/2)`: it
        preserves the pitch mean and smooth variation, and annihilates the
        Nyquist (sawtooth) mode. Blending with the unsmoothed field by
        ``alpha`` tunes the strength: ``alpha=1`` is a full 1-2-1 pass,
        ``alpha=0`` leaves the field unchanged.

        Parameters
        ----------
        field : ndarray
            Field to smooth; any shape with axis :attr:`pitch_dim`.
        alpha : float
            Blend factor in ``[0, 1]``. ``0`` disables, ``1`` is a full pass.

        Returns
        -------
        ndarray
            Smoothed field, same shape and dtype as ``field``.
        """
        if alpha == 0.0:
            return field
        axis = self.pitch_dim
        smoothed = 0.25 * (
            np.roll(field, 1, axis=axis) + 2.0 * field + np.roll(field, -1, axis=axis)
        )
        if alpha == 1.0:
            return smoothed
        return alpha * smoothed + (1.0 - alpha) * field

    @property
    def block_avg(self):
        """Pitch-averaged flow field; :class:`~ember.block.Block` of shape ``(nspan,)``.

        Coordinates are the pitch-mean x, r, t at each span station. The
        conserved variables are populated by calling :meth:`set_block_avg`;
        before that call the flow-field arrays contain uninitialised values.
        """
        self._check_attached()
        return self._block_avg

    @property
    def pitch_dim(self):
        """Axis of the pitchwise (circumferential) dimension; ``int`` in ``{0, 1, 2}``.

        Detected automatically from block geometry on :meth:`attach_to_block`:
        the axis along which only theta varies while x and r remain constant.
        """
        self._check_attached()
        return self._dim_pitch

    @property
    def span_dim(self):
        """Axis of the spanwise (meridional) dimension; ``int`` in ``{0, 1, 2}``.

        Detected automatically from block geometry on :meth:`attach_to_block`:
        the axis along which x and r vary (hub to tip).
        """
        self._check_attached()
        return self._dim_span

    @property
    def spf(self):
        """Span fraction at each node, normalised to ``[0, 1]`` by meridional arc-length; ``ndarray`` of shape ``(nspan,)``.

        ``spf[0] == 0.0`` at the hub/start corner and ``spf[-1] == 1.0`` at the
        tip/end corner. Spacing reflects the actual meridional distances between
        nodes, not their indices.
        """
        self._check_attached()
        return self._spf

    @property
    def weight_pitch(self):
        """Pitch weights per node as a fraction of ``block.pitch``; ``ndarray`` broadcastable against :attr:`block_view`.

        Weights sum to 1 along :attr:`pitch_dim`, so a pitch-averaged scalar
        field is ``(field * patch.weight_pitch).sum(axis=patch.pitch_dim)``.
        """
        self._check_attached()
        return self._weight_pitch

    # begin property
    # end property
