"""Structured data framework with version tracking and cached property management.

This module provides the foundational infrastructure for managing multi-dimensional structured
CFD data through the StructuredData base class. The class implements an intelligent caching
system using version-tracked data keys, where derived properties are automatically invalidated
and recomputed when underlying data changes. The cached_array decorator enables efficient
lazy evaluation of expensive computations like gradients and geometric quantities while
supporting in-place updates to minimize memory allocations. The framework handles both primary
data storage (coordinates, conserved variables) and derived quantities (primitives, fluxes)
with automatic dependency tracking. This architecture is essential for maintaining consistency
across the codebase while achieving high performance in iterative solvers that repeatedly
access the same computed properties.

Indexing and slicing over the spatial axes is supported numpy-style; see the
"Indexing and slicing" section in :mod:`ember.block` for the full rules.
"""

import logging
import numpy as np
from collections import defaultdict
from functools import wraps
import warnings
from ember import util

logger = logging.getLogger(__name__)

f32 = np.float32


def cached_array(*data_keys):
    """Cached property decorator that invalidates when specified data keys change.

    The decorator passes any existing cached array(s) to the function via an 'out'
    parameter for in-place operations to avoid memory reallocation.

    The result is always locked read-only. Called with one or more ``data_keys``
    the cache is invalidated (and recomputed) whenever a tracked key changes --
    the usual cached-field case. Called with no keys (``@cached_array()``) the
    buffer is allocated once and never invalidated, so it serves as a reusable
    scratch buffer; code that owns such a buffer must toggle
    ``arr.flags.writeable`` around its in-place writes (see ``Grid.update_sources``).

    Parameters
    ----------
    *data_keys : str
        Data keys to track for cache invalidation.

    Returns
    -------
    callable
        Property decorator that caches results based on data key versions.
    """

    def decorator(func):
        func_key = func.__name__

        @wraps(func)
        def wrapper(self):
            # Get current version tuple for tracked data keys
            current_versions = self._get_version(data_keys)

            # Check if we have cached value and versions match
            if func_key in self._store and self._store[func_key][0] == current_versions:
                return self._store[func_key][1]

            # Get existing arrays for potential reuse
            if func_key in self._store:
                out_param = self._store[func_key][1]
                out_param.flags.writeable = True
            else:
                out_param = None

            # Call function with out parameter
            result = func(self, out=out_param)

            result.flags.writeable = False
            self._store[func_key] = (current_versions, result)
            return result

        return property(wrapper)

    return decorator


def cached_object(func):
    """Cached property for a non-array result (dict, tuple, scalar).

    The object counterpart to :func:`cached_array`: use it when the cached
    value is not a lockable numpy array. The value is computed once on first
    access and stored in ``self._store`` using the same ``(versions, value)``
    entry shape as ``cached_array`` (with empty versions), so the store stays
    homogeneous for consumers that walk it. It is never invalidated -- there is
    no key tracking and no read-only lock -- so the result must not depend on
    mutable flow state. For array results that should be locked and invalidated
    when their data keys change, use :func:`cached_array` instead.
    """
    func_key = func.__name__

    @wraps(func)
    def wrapper(self):
        if func_key not in self._store:
            self._store[func_key] = ((), func(self))
        return self._store[func_key][1]

    return property(wrapper)


def scratch_array(func):
    """Property for an owned, never-invalidated, writeable scratch buffer.

    Like a zero-arg :func:`cached_array` -- allocated once via the ``out``
    parameter, never invalidated, stored in ``self._store`` with the same
    ``(versions, value)`` entry shape so the store stays homogeneous -- but the
    buffer is left **writeable**. It is transient kernel workspace, not a cached
    value, so owners write into it (or pass it to an ``intent(inout)`` kernel)
    directly, without toggling ``flags.writeable``. Do not read it expecting a
    consistent value. Use :func:`cached_array` instead for buffers that are read
    as values between writes and so must be locked read-only.
    """
    func_key = func.__name__

    @wraps(func)
    def wrapper(self):
        if func_key not in self._store:
            self._store[func_key] = ((), func(self, out=None))
        return self._store[func_key][1]

    return property(wrapper)


def derived_array(func):
    """Decorator for read-only derived array properties.

    Wraps a plain ``@property`` so that the returned array cannot be written
    to in-place.  Any attempt to write into it (e.g. ``block.conserved[...] = 1``)
    raises a ``ValueError`` rather than silently discarding the write.
    Use the block set_* methods to modify the flow state.
    """

    @wraps(func)
    def wrapper(self):
        result = func(self)
        arr = np.asarray(result)
        arr.flags.writeable = False
        return arr

    return property(wrapper)


class StructuredData:
    """General storage of arrays with metadata in one sliceable object.

    All raw variables are stored contiguously in a single ``float32`` array of
    shape ``(*shape, nvar)``, in Fortran (column-major) order. Derived
    properties are computed lazily on demand and cached separately per
    instance; they do not share the backing array.
    """

    _defaults = {}

    _data_keys = ()
    """String keys for each subarray in the data array."""

    def __init__(self, shape=()):
        """Allocate the data array.

        Parameters
        ----------
        shape : tuple
            Shape of a single property array.

        """

        assert self.nvar > 0, "StructuredData must have at least one variable."

        self._data = util.full(tuple(shape) + (self.nvar,), np.nan)

        self._metadata = {}

        # Setup inverse mapping of keys to indices from _data_rows
        self._data_inds = {k: i for i, k in enumerate(self._data_keys)}

        # Single version counter for all data and metadata keys
        self._versions = defaultdict(int)

        self._metadata["triangulated"] = False

        # Initialize cache storage for cached properties
        self._store = {}

        self.__post_init__()  # end method

    def __post_init__(self):
        """Post-initialisation function to be called after all variables are set."""
        # Default no-op
        pass  # end method

    def __getitem__(self, key):
        """Index or slice the spatial axes, returning a view.

        Accepts the same index expressions as numpy (integer, slice, or a
        tuple thereof). Returns a new instance sharing the backing array with
        the original; writes to raw variables in the result are visible in the
        original and vice versa. A scalar index on an axis removes that axis
        (``ndim`` decreases by one); a slice preserves it.

        Parameters
        ----------
        key : int, slice, or tuple of int/slice
            Index expression applied to the spatial axes.

        Returns
        -------
        out : same type as ``self``
            A view with the selected spatial region.

        """
        if not isinstance(key, tuple):
            key = (key,)
        # Append a trailing colon to the key if it is a scalar
        key = key + (slice(None),)
        out = self.view()
        out._data = self._data[key]
        return out  # end method

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_store"] = {}
        return state  # end method

    def __setstate__(self, state):
        state["_store"] = {}
        self.__dict__.update(state)  # end method

    def _bare_copy(self):
        """Create a new instance sharing metadata/init dicts, bypassing __init__."""
        out = object.__new__(self.__class__)
        out._metadata = self._metadata
        out._versions = self._versions
        out._data = self._data
        out._data_inds = self._data_inds
        out._store = {}
        return out  # end method

    def _get_data_by_keys(self, keys, raise_uninit=True, writeable=False):
        """Extract data by variable name(s).

        Parameters
        ----------
        keys : tuple of str
            Variable names. Must be consecutive in self._data_rows. If a
            single-element tuple, the trailing dimension is squeezed out so
            the return shape matches self.shape.
        raise_uninit : bool
            Whether to raise an error if any variable is not initialised.
        writeable : bool
            Whether to return a writeable view. Default is False (read-only).

        Returns
        -------
        out: ndarray
            Data array for the specified variables. Shape is (*self.shape,)
            for a single key, (*self.shape, len(keys)) for multiple keys.

        """

        # Check that keys are initialised
        if raise_uninit:
            for k in keys:
                if not self._versions[k]:
                    raise ValueError(f"Data for variable {k} has not been initialised.")

        # Build a slice object
        inds = tuple(self._data_inds[k] for k in keys)
        slice_obj = slice(inds[0], inds[-1] + 1)
        out = self._data[..., slice_obj]

        # Make the output read-only only if writeable=False
        # (scalars are already read-only, but arrays are not)
        out = out.view()
        if out.ndim != 0 and not writeable:
            out.flags.writeable = False

        if len(keys) == 1:
            return out[..., 0]
        return out  # end method

    def _get_metadata_by_key(self, key):
        """Extract metadata by variable name."""
        if key in self._metadata:
            return self._metadata[key]
        elif key in self._defaults:
            return self._defaults[key]
        else:
            raise KeyError(f"Metadata key '{key}' not found.")  # end method

    def _increment_data(self, keys, delta, slice_obj=None):
        """Increment data in-place for specified keys at given slice."""
        # Get indices for the keys
        inds = [self._data_inds[k] for k in keys]

        # Check that indices are consecutive
        if len(inds) > 1 and sorted(inds) != list(range(min(inds), max(inds) + 1)):
            raise ValueError(
                f"Variable indices must be consecutive. Got indices {inds} for keys {keys}"
            )

        # Apply increment for each variable individually
        for i, var_ind in enumerate(inds):
            if slice_obj is None:
                self._data[..., var_ind] += delta[..., i]
            else:
                self._data[slice_obj + (var_ind,)] += delta[..., i]  # end method

    def _set_data_by_keys(self, keys, val, store_init=True):
        """Set data variables at once.

        Parameters
        ----------
        keys : tuple of str
            Variable names. Must be consecutive in self._data_keys. If a
            single-element tuple, val should have shape broadcastable to
            self.shape (no trailing dimension needed).
        val : Array
            For a single key: shape broadcastable to self.shape.
            For multiple keys: shape (..., len(keys)).
        store_init : bool
            Whether to mark the variables as initialised. Defaults to True.
        """
        # For a single key, add the trailing variable dimension that the rest
        # of this method expects, mirroring the old _set_data_by_key behaviour.
        if len(keys) == 1:
            val = np.asarray(val)[..., np.newaxis]

        # Build indices for the keys
        inds = tuple(self._data_inds[k] for k in keys)

        # Check they're consecutive
        if not all(inds[i + 1] - inds[i] == 1 for i in range(len(inds) - 1)):
            raise ValueError(f"Keys {keys} are not consecutive in data array")

        # Check the last dimension matches number of keys
        if val.shape[-1] != len(keys):
            raise ValueError(
                f"Last dimension of val ({val.shape[-1]}) must match number of keys ({len(keys)})"
            )

        # Only squeeze if target is scalar - allows (1, N) -> (N,) assignments
        # while preserving legitimate singleton-dimension broadcasting patterns.
        # Squeeze spatial axes only, not the variable axis (last axis).
        if self.shape == ():
            # For scalar target with shape (1, ..., 1, n_vars), squeeze spatial dims
            # Squeeze all axes except the last (which is the variable axis)
            axes_to_squeeze = tuple(i for i in range(val.ndim - 1) if val.shape[i] == 1)
            if axes_to_squeeze:
                val_squeezed = np.squeeze(val, axis=axes_to_squeeze)
            else:
                val_squeezed = val
        else:
            val_squeezed = val

        # Check broadcasting compatibility
        try:
            np.broadcast_to(val_squeezed[..., 0], self.shape)
        except ValueError as e:
            raise ValueError(
                f"Cannot broadcast value with shape {val.shape[:-1]} to data shape {self.shape}"
            ) from e

        # Create slice object and set the data
        slice_obj = slice(inds[0], inds[-1] + 1)
        self._data[..., slice_obj] = val_squeezed

        for k in keys:
            if store_init:
                self._versions[k] += 1  # end method

    def _set_metadata_by_key(self, key, val):
        """Set metadata by variable name.

        Parameters
        ----------
        key : str
            Variable name.
        val : object
            Value to set for the metadata variable.

        """
        if key in self._data_keys:
            raise ValueError(
                f"Cannot set metadata key '{key}' - conflicts with data key"
            )

        if isinstance(val, (float, np.floating)):
            val = f32(val)
        self._metadata[key] = val

        # Increment version counter for this metadata key
        self._versions[key] += 1  # end method

    def _get_version(self, key):
        """Get the version number for data or metadata key(s).

        Parameters
        ----------
        key : str or tuple of str
            Variable name(s). Can be data keys or metadata keys.

        Returns
        -------
        int or tuple of int
            Version number(s) (0 if not initialised, increments with each update).
        """
        if isinstance(key, (tuple, list)):
            return tuple(self._versions[k] for k in key)
        else:
            return self._versions[key]  # end method

    def set_triangulated(self, value):
        """Set whether the data represents a triangulated mesh.

        When ``True``, the data must have shape ``(ntri, 3)``: axis-0 indexes
        triangles and axis-1 indexes the three nodes of each triangle. Points
        may be duplicated across triangles.
        """
        if value and len(self.shape) >= 2 and self.shape[1] != 3:
            raise ValueError(
                f"Triangulated data requires shape[1] == 3, got {self.shape[1]}"
            )
        self._set_metadata_by_key("triangulated", value)
        return self  # end method

    def clear_cache(self):
        """Clear all cached property values.

        This forces all cached properties to recalculate on next access.
        """
        import traceback

        caller = traceback.extract_stack()[-2]
        keys = list(self._store)
        logger.debug(
            "clear_cache: purging %d key(s) %s, called from %s:%d",
            len(keys),
            keys,
            caller.filename,
            caller.lineno,
        )
        self._store.clear()  # end method

    def copy(self):
        """Take an independent copy of the data.

        Data and metadata dicts are shallow-copied. Beware: mutable metadata
        values (e.g. arrays) edited in place are shared with the original.

        Returns
        -------
        out : same type as ``self``
            A new instance with copied data.

        """
        out = self.view()
        out._data = self._data.copy(order="F").astype(f32, copy=False)
        out._metadata = self._metadata.copy()
        out._versions = self._versions.copy()
        return out  # end method

    def empty(self, shape=()):
        """Create a new uninitialised instance with the same metadata.

        Returns a fresh object of the same class with all data set to NaN.
        Metadata is shallow-copied from the original; beware that mutable
        metadata values are shared.

        Parameters
        ----------
        shape : tuple, optional
            Shape of the new array. Defaults to scalar.

        Returns
        -------
        out : same type as ``self``
            A new uninitialised instance with copied metadata.

        """
        out = self.__class__(shape=shape)
        out._metadata = {**self._metadata, "triangulated": False}
        out._versions = self._versions.copy()
        return out  # end method

    def flat(self):
        """Flatten all axes into a single axis, not a copy.

        Returns a new instance sharing metadata and data with the original via
        a numpy reshaped view. Writes through either object are visible in the
        other.

        Returns
        -------
        out : same type as ``self``, shape (npoints,)
            A new instance with all points in a single dimension.

        """
        out = self.view()
        out._data = self._data.reshape(-1, self.nvar)
        return out  # end method

    def flip(self, axis):
        """Reverse indexing along the specified axis, not a copy.

        Returns a new instance sharing metadata and data with the original via
        numpy reversed-stride views. Writes through either object are visible
        in the other.

        Parameters
        ----------
        axis : int
            Axis along which to reverse indexing.

        Returns
        -------
        out : same type as ``self``
            A new instance with reversed indexing along the given axis.

        """

        if not (-self.ndim <= axis < self.ndim):
            raise ValueError(
                f"Invalid axis {axis} for data with {self.ndim} dimensions."
            )
        axis = axis % self.ndim  # handle negative indices
        out = self.view()
        out._data = np.flip(self._data, axis=axis)
        return out  # end method

    def mean(self, axis=0, keepdims=False):
        """Calculate mean along specified axis, creating a new object.

        Parameters
        ----------
        axis : int
            Axis along which to calculate mean. Must be less than ndim.
            Default is 0 (first spatial dimension).

        Returns
        -------
        out : same type as ``self``
            New object with averaged data along specified axis.

        The mean is taken over the underlying raw data variables, not derived
        properties. Which variables are present depends on the subclass.

        """
        # Validate axis
        if not (-self.ndim <= axis < self.ndim):
            raise ValueError(
                f"Invalid axis {axis} for data with {self.ndim} dimensions."
            )

        # Handle negative axis
        axis = axis % self.ndim

        # Forbid averaging over last axis (variable axis)
        if axis == self.ndim:  # This would be the variable axis in _data
            raise ValueError("Cannot average over variable axis (last axis).")

        # Create new object and compute mean
        out = self.view()
        out._data = np.mean(self._data, axis=axis, keepdims=keepdims)

        return out  # end method

    def nanmean(self, axis=0):
        """Calculate nanmean along specified axis, ignoring NaN values.

        Parameters
        ----------
        axis : int
            Axis along which to calculate nanmean. Must be less than ndim.
            Default is 0 (first spatial dimension).

        Returns
        -------
        out : same type as ``self``
            New object with averaged data along specified axis, ignoring NaN values.

        The mean is taken over the underlying raw data variables, not derived
        properties. Which variables are present depends on the subclass.

        """
        # Validate axis
        if not (-self.ndim <= axis < self.ndim):
            raise ValueError(
                f"Invalid axis {axis} for data with {self.ndim} dimensions."
            )

        # Handle negative axis
        axis = axis % self.ndim

        # Forbid averaging over last axis (variable axis)
        if axis == self.ndim:  # This would be the variable axis in _data
            raise ValueError("Cannot average over variable axis (last axis).")

        # Create new object and compute nanmean
        out = self.view()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            out._data = np.nanmean(self._data, axis=axis)

        return out  # end method

    def reshape(self, shape):
        """Reshape the data axes, not a copy.

        Returns a new instance sharing metadata with the original. The output
        data is a zero-copy view where possible (contiguous input); otherwise
        numpy makes a copy. The total number of spatial points must be unchanged.

        Parameters
        ----------
        shape : tuple
            New shape. Must contain the same number of elements as the original.

        Returns
        -------
        out : same type as ``self``
            A new instance with the specified shape.

        """
        out = self.view()
        out._data = self._data.reshape(shape + (self.nvar,))
        return out  # end method

    def squeeze(self):
        """Remove singleton axes.

        Returns a new instance sharing metadata and data with the original;
        this is a zero-copy view. Writes through either object are visible
        in the other.

        Returns
        -------
        out : same type as ``self``
            A new instance with all length-1 axes removed.

        """

        out = self.view()
        axes = tuple(i for i, s in enumerate(self.shape) if s == 1)
        if not axes:
            return out
        out._data = np.squeeze(self._data, axis=axes)
        return out  # end method

    def transpose(self, axes=None):
        """Reorder the data axes, defaulting to reversal.

        Returns a new instance sharing metadata with the original. The data is
        a zero-copy view where possible (contiguous input); otherwise numpy
        makes a copy. Writes through a view are visible in the original.

        Parameters
        ----------
        axes : tuple, optional
            New order of the axes. If None, the axes order is reversed.

        Returns
        -------
        out : same type as ``self``
            A new instance with axes reordered as specified.

        """

        # Default to reverse
        if axes is None:
            axes = tuple(reversed(range(self.ndim)))

        # Check that axes are valid
        for a in axes:
            if not (-self.ndim <= a < self.ndim):
                raise ValueError(
                    f"Invalid axis {axes} for data with {self.ndim} dimensions."
                )
        # Handle negative axes
        axes = tuple(a % self.ndim for a in axes)

        # Check that axes are valid
        if len(axes) != self.ndim:
            raise ValueError(
                f"Invalid transpose {axes} for data with {self.ndim} dimensions."
            )

        # Add a trailing axis for the variable
        axes1 = tuple(a for a in axes) + (self.ndim,)

        # Create a new view with the transposed data
        out = self.view()
        out._data = np.transpose(self._data, axes1)

        return out  # end method

    def view(self):
        """Create a new view onto the original data, not a copy.

        Returns a new instance of the same class sharing the underlying data
        array, metadata dict, and version counters with the original. Mutations
        to data (e.g. writing to a variable array) are visible through both
        objects. Derived properties are held in a separate per-instance cache,
        so each view starts cold.

        Returns
        -------
        out : same type as ``self``
            A new instance sharing all data and metadata with the original.

        """
        return self._bare_copy()  # end method

    @property
    def ndim(self):
        """Number of spatial dimensions."""
        # Exclude the last axis which indexes the variables
        return len(self.shape)  # end property

    @property
    def ni(self):
        """Number of points along first axis."""
        if self.ndim < 1:
            raise AttributeError("ni is only valid for 1D data (ndim >= 1)")
        return self.shape[0]  # type: ignore  # end property

    @property
    def nj(self):
        """Number of points along second axis."""
        if self.ndim < 2:
            raise AttributeError("nj is only valid for 2D data (ndim >= 2)")
        return self.shape[1]  # type: ignore  # end property

    @property
    def nk(self):
        """Number of points along third axis."""
        if self.ndim < 3:
            raise AttributeError("nk is only valid for 3D data (ndim >= 3)")
        return self.shape[2]  # type: ignore  # end property

    @property
    def nvar(self):
        """Number of variables stored at each spatial point."""
        return len(self._data_keys)  # end property

    @property
    def shape(self):
        """Shape of the grid points."""
        return self._data.shape[:-1]  # end property

    @property
    def shape_cell(self):
        """Shape of cell-centred arrays `(ni-1, nj-1, nk-1)`."""
        if self.ndim < 3:
            raise AttributeError("shape_cell is only valid for 3D data (ndim == 3)")
        return (self.shape[0] - 1, self.shape[1] - 1, self.shape[2] - 1)  # end property

    @property
    def shape_iface(self):
        """Shape of i-face arrays `(ni, nj-1, nk-1)`."""
        if self.ndim < 3:
            raise AttributeError("shape_iface is only valid for 3D data (ndim == 3)")
        return (self.shape[0], self.shape[1] - 1, self.shape[2] - 1)  # end property

    @property
    def shape_jface(self):
        """Shape of j-face arrays `(ni-1, nj, nk-1)`."""
        if self.ndim < 3:
            raise AttributeError("shape_jface is only valid for 3D data (ndim == 3)")
        return (self.shape[0] - 1, self.shape[1], self.shape[2] - 1)  # end property

    @property
    def shape_kface(self):
        """Shape of k-face arrays `(ni-1, nj-1, nk)`."""
        if self.ndim < 3:
            raise AttributeError("shape_kface is only valid for 3D data (ndim == 3)")
        return (self.shape[0] - 1, self.shape[1] - 1, self.shape[2])  # end property

    @property
    def size(self):
        """Total number of spatial points."""
        return int(np.prod(self.shape))  # end property

    @property
    def triangulated(self):
        """Whether the data represents a triangulated mesh."""
        return self._get_metadata_by_key("triangulated")  # end property

    #
    # numpy ndarray style functions
    #

    #
    # Methods for accessing data and metadata
    #

    #
    # Size and shape properties
    #
