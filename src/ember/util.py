r"""Array functions shared across the codebase.

Each function below is a standalone operation on plain NumPy arrays. This
module acts a a place to collect common conventions, patterns, and conversions
so that they can be used consistently across the code.

Flow angle and coordinate conversions
======================================

Paired functions convert between the native ember data representation
and and other conventions, each the inverse of its partner. There are utilities
for conversion between flow angles and velocity components, between Cartesian
and polar coordinates, and between polar and pseudo-Cartesian coordinates.

.. autosummary::

   angles_to_components
   components_to_angles
   cart_to_pol
   pol_to_cart
   pol_to_pseudocart

Vector and matrix operations
=============================

These functions are intended for use batches of vectors or matrices stacked
along leading axes, with the last one or two axes holding the vector or matrix
components. Some are thin wrappers around :obj:`numpy.einsum`, but for
operations on the hot path of a solver loop they are implemented in Fortran for
better performance.

.. autosummary::

   dot
   vecnorm
   matmat
   matvec

Grid construction
=================

The below functions are used to build coordinate arrays for the solver. For
simple duct geometries, they allow convenient construction of uniform or
clustered :attr:`~ember.block.Block.xrt` arrays. The :func:`resample` is
useful for coarsening or refining the distribution of an existing grid.

.. autosummary::

   meshgrid3
   linmesh3
   cosine_cluster
   cluster
   cluster_symmetric
   resample

Array allocation and buffers
============================

Many of these functions are are thin wrappers around their :mod:`numpy`
namesakes that fix the dtype and memory layout that weember standardises on:
Fortran order and single-precision ``float32``. Calling these rather than the
bare NumPy functions signals intent and keeps that convention in one place --
use them for any new array creation. There are also functions for memory
management: allocation, or buffer creation and reuse.

.. autosummary::

   zeros
   array
   empty
   full
   allocate_or_reuse
   bcast_if_needed
   carve_view

Miscellaneous geometry
========================

Assorted utilities for working with coordinates and bounding boxes.

.. autosummary::

   extent
   bounding_box
   apply_perm_flip
   unwrap_meridional
"""

import numpy as np
import ember.fortran

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func


f32 = np.float32


def dot(a, b):
    """Dot product of two arrays along the last axis.

    Parameters
    ----------
    a : Array, shape (..., n)
        First input array.
    b : Array, shape (..., n)
        Second input array.

    Returns
    -------
    prod : Array, shape (...)
        Dot product of the input arrays, components summed along the last axis.

    """
    return np.einsum("...i,...i", a, b)


def angles_to_components(V, Alpha, Beta):
    """Resolve velocity magnitude into polar components.

    Uses numerically stable trigonometry for all angles including 90 degrees.

    Flow angle conventions:

    - Alpha = 0°: No swirl (Vt = 0)
    - Beta = 0°: Pure axial flow (Vr = 0)
    - Beta = ±90°: Pure radial flow (Vx = 0)

    Parameters
    ----------
    V_rel : Array
        Relative velocity magnitude [m/s]
    Alpha : Array
        Relative yaw angle (tangential flow direction) [degrees]
    Beta : Array
        Pitch angle (radial flow direction) [degrees]

    Returns
    -------
    Vx : Array
        Axial velocity component [m/s]
    Vr : Array
        Radial velocity component [m/s]
    Vt : Array
        Tangential velocity component [m/s]

    """
    # Convert to consistent dtype and radians
    V = np.asarray(V, dtype=f32)
    alpha_rad = np.radians(np.asarray(Alpha, dtype=f32))
    beta_rad = np.radians(np.asarray(Beta, dtype=f32))

    # Use pure trigonometry for numerical stability
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)
    cos_beta = np.cos(beta_rad)
    sin_beta = np.sin(beta_rad)

    # Alpha decomposition: meridional and tangential components
    Vm = V * cos_alpha  # Meridional velocity magnitude
    Vt = V * sin_alpha  # Tangential relative velocity

    # Beta decomposition: axial and radial components
    Vx = Vm * cos_beta  # Axial velocity
    Vr = Vm * sin_beta  # Radial velocity

    return Vx, Vr, Vt


def components_to_angles(Vx, Vr, Vt):
    """Convert velocity components to velocity magnitude and flow angles.

    This is the inverse of :func:`angles_to_components`, obeying the conventions:

    - Alpha = 0°: No swirl (Vt_rel = 0)
    - Beta = 0°: Pure axial flow (Vr = 0)
    - Beta = ±90°: Pure radial flow (Vx = 0)

    For velocity components with very small magnitudes, angles may be
    numerically unstable. Zero velocity returns (0, 0, 0).

    Parameters
    ----------
    Vx : Array
        Axial velocity component [m/s]
    Vr : Array
        Radial velocity component [m/s]
    Vt : Array
        Tangential velocity component [m/s]

    Returns
    -------
    V : Array
        Velocity magnitude [m/s]
    Alpha : Array
        Yaw angle (tangential flow direction) [degrees]
    Beta : Array
        Pitch angle (radial flow direction) [degrees]

    """
    # Convert to consistent dtype
    Vx = np.asarray(Vx, dtype=f32)
    Vr = np.asarray(Vr, dtype=f32)
    Vt = np.asarray(Vt, dtype=f32)

    # Calculate meridional velocity magnitude
    Vm = np.sqrt(Vx**2 + Vr**2)

    # Calculate total relative velocity magnitude
    V = np.sqrt(Vm**2 + Vt**2)

    # Handle zero velocity case
    zero_velocity = V < 1e-12

    # Calculate Alpha_rel (yaw angle from meridional vs tangential)
    # atan2(Vt, V_rel_m) but handle zero meridional velocity
    Alpha_rad = np.where(zero_velocity, 0.0, np.arctan2(Vt, Vm))

    # Calculate Beta (pitch angle from axial vs radial)
    # atan2(Vr, Vx) but handle zero meridional velocity
    Beta_rad = np.where(
        Vm < 1e-12,  # Pure tangential flow or zero velocity
        0.0,
        np.arctan2(Vr, Vx),
    )

    # Convert to degrees
    Alpha_deg = np.degrees(Alpha_rad)
    Beta_deg = np.degrees(Beta_rad)

    return V, Alpha_deg, Beta_deg


def vecnorm(x):
    """Calculate the norm of a vector array along the last axis.

    Parameters
    ----------
    x : Array, shape (..., n)
        Input array with last dimension as vector components.

    Returns
    -------
    norm : Array, shape (...)
        Norm of the input vector array.

    """
    return np.sqrt(np.einsum("...i,...i", x, x))


def zeros(shape, dtype=np.float32):
    """Zero-filled array in standard layout and dtype.

    Equivalent to :obj:`numpy.zeros`, but defaulting to ``dtype=float32``
    and always ``order="F"``.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the new array.
    dtype : numpy dtype, optional
        Data type for the new array. Default is ``np.float32``.

    Returns
    -------
    Array
        Fortran-ordered array of the given shape and dtype, filled with 0.
    """
    return np.zeros(shape, dtype=dtype, order="F")


def array(x, dtype=np.float32):
    """Copy array data into standard layout and dtype.

    Equivalent to :obj:`numpy.array`, but defaulting to ``dtype=float32``
    and always ``order="F"``.

    Parameters
    ----------
    x : array_like
        Data to copy into the new array.
    dtype : numpy dtype, optional
        Data type for the new array. Default is ``np.float32``.

    Returns
    -------
    Array
        Fortran-ordered array of the given dtype, holding a copy of ``x``.
    """
    return np.array(x, dtype=dtype, order="F")


def empty(shape):
    """Uninitialised array in standard layout and dtype.

    Equivalent to :obj:`numpy.empty`, but always ``dtype=float32`` and
    ``order="F"``. As with :obj:`numpy.empty`, contents are arbitrary until
    written -- use this only where every element will be set before being
    read.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the new array.

    Returns
    -------
    Array
        Fortran-ordered, float32 array of the given shape with
        uninitialised contents.
    """
    return np.empty(shape, dtype=np.float32, order="F")


def full(shape, fill_value):
    """Constant-filled array in standard layout and dtype.

    Equivalent to :obj:`numpy.full`, but always ``dtype=float32`` and
    ``order="F"``.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the new array.
    fill_value : scalar
        Value to fill every element with.

    Returns
    -------
    Array
        Fortran-ordered, float32 array of the given shape, filled with
        ``fill_value``.
    """
    return np.full(shape, fill_value, dtype=np.float32, order="F")


def allocate_or_reuse(out, shape, dtype=np.float32):
    """Allocate output array if not provided, otherwise reuse existing array.

    Helper function for functions that accept an optional `out` parameter.
    If ``out is None``, allocates a new F-contiguous zero array with the
    specified shape and dtype. Otherwise returns the provided array.

    Parameters
    ----------
    out : Array or None
        Pre-allocated output array, or None to allocate new array.
    shape : tuple
        Shape of array to allocate if out is None.
    dtype : numpy dtype, optional
        Data type for new array. Default is np.float32.

    Returns
    -------
    Array
        Either the provided `out` array or a newly allocated zero array.

    Examples
    --------
    >>> def my_function(x, out=None):
    ...     out = allocate_or_reuse(out, x.shape)
    ...     # ... compute results into out ...
    ...     return out
    """
    if out is None:
        return np.zeros(shape, dtype=dtype, order="F")
    else:
        return out


def bcast_if_needed(a, shape):
    """Broadcast ``a`` to ``shape`` only if it doesn't already have it.

    A shape-tuple comparison is far cheaper than :func:`numpy.broadcast_to`
    itself, so this is a near-zero-cost no-op on the common path where
    ``a`` is already ``shape`` -- e.g. a hot-path setter that documents its
    inputs as "must broadcast to X" for callers that need it, but whose
    actual callers already pass exactly ``X``-shaped arrays.

    Parameters
    ----------
    a : array_like
        Candidate array; may already have ``shape`` or be broadcastable to it.
    shape : tuple
        Target shape.

    Returns
    -------
    Array
        ``a`` unchanged if ``a.shape == shape``, otherwise
        ``np.broadcast_to(a, shape)``.
    """
    if getattr(a, "shape", None) == shape:
        return a
    return np.broadcast_to(a, shape)


def carve_view(buf, *shapes):
    """Carve one or more zero-copy Fortran-order views from a buffer of any shape.

    Reinterprets successive spans of a Fortran-contiguous `buf` as arrays of the
    requested `shapes`, packing them end-to-end (view ``k`` starts where view
    ``k-1`` ends) so every returned view aliases distinct storage and all may be
    held live simultaneously. Used to borrow differently-shaped scratch arrays
    out of one oversized solver buffer without allocating.

    The offsets are computed internally, so a caller carving several coexisting
    slots cannot accidentally overlap or gap them. ``buf.reshape(-1,
    order="F")`` on such a buffer is itself a free view, so repeated calls stay
    zero-copy and there is no need to hoist a shared flat view at the call
    site.

    Parameters
    ----------
    buf : Array
        Fortran-contiguous source buffer to reinterpret. Must be large enough to
        hold the concatenation of all `shapes` (``sum(prod(shape))`` elements).
    *shapes : tuple
        One shape per view, in packing order. Passing a single shape returns a
        single view; passing several returns a list, one per shape.

    Returns
    -------
    Array or list of Array
        A single F-order view when one shape is given, else a list of them, each
        aliasing a disjoint span of `buf`.

    Raises
    ------
    ValueError
        If the packed views would not fit within `buf`.
    """
    flat = buf.reshape(-1, order="F")
    counts = [int(np.prod(shape)) for shape in shapes]
    if sum(counts) > flat.size:
        raise ValueError(
            f"carve_view: {sum(counts)} elements requested "
            f"({shapes}) exceed buffer capacity {flat.size}"
        )
    views = []
    start = 0
    for shape, count in zip(shapes, counts):
        views.append(flat[start : start + count].reshape(shape, order="F"))
        start += count
    return views[0] if len(shapes) == 1 else views


def meshgrid3(xv, rv, tv):
    """Create 3D coordinate rectangular meshgrid.

    This function combines the common pattern of creating a 3D meshgrid
    and stacking the results into a single coordinate array. It preserves
    the input dtype unless the inputs have mixed dtypes.

    Parameters
    ----------
    xv : array_like
        X-coordinate vector.
    rv : array_like
        R-coordinate vector.
    tv : array_like
        Theta-coordinate vector.

    Returns
    -------
    Array, shape (len(xv), len(rv), len(tv), 3)
        Coordinate array with [x, r, t] components.

    Examples
    --------
    >>> xv = np.linspace(0, 1, 3)
    >>> rv = np.linspace(1, 2, 4)
    >>> tv = np.linspace(0, np.pi, 5)
    >>> xrt = meshgrid3(xv, rv, tv)
    >>> xrt.shape
    (3, 4, 5, 3)
    """
    xm, rm, tm = np.meshgrid(xv, rv, tv, indexing="ij")
    return np.stack([xm, rm, tm], axis=-1)


def linmesh3(x, r, t, shape):
    """Create 3D coordinate meshgrid from ranges and shape.

    This function creates linearly spaced vectors from coordinate ranges and
    then generates a 3D meshgrid using :func:`meshgrid3`. It combines the
    common pattern of creating linspace vectors and then meshing them.

    Parameters
    ----------
    x : tuple or array_like
        Axial coordinate range [x_min, x_max].
    r : tuple or array_like
        Radial coordinate range [r_min, r_max].
    t : tuple or array_like
        Angular coordinate range [t_min, t_max].
    shape : tuple
        Shape of the grid (ni, nj, nk).

    Returns
    -------
    Array, shape (ni, nj, nk, 3)
        Coordinate array with (x, r, t) components, dtype=float32.

    Examples
    --------
    >>> xrt = linmesh3([0, 1], [1, 2], [0, np.pi], (3, 4, 5))
    >>> xrt.shape
    (3, 4, 5, 3)
    >>> xrt.dtype
    dtype('float32')
    """
    ni, nj, nk = shape
    xv = np.linspace(x[0], x[1], ni, dtype=np.float32)
    rv = np.linspace(r[0], r[1], nj, dtype=np.float32)
    tv = np.linspace(t[0], t[1], nk, dtype=np.float32)
    return meshgrid3(xv, rv, tv)


def pol_to_pseudocart(xrt, inplace=False):
    """Convert (x, r, θ) coordinates to pseudo-Cartesian (x, r, rθ).

    In the pseudo-Cartesian system, the theta coordinate is multiplied by
    the radial coordinate to give rθ, which behaves like a Cartesian
    coordinate for distance calculations in cylindrical coordinates.

    Parameters
    ----------
    xrt : array_like
        Input coordinates with shape (..., 3) where the last dimension
        contains [x, r, θ] coordinates.
    inplace : bool, optional
        If True, modify the input array in-place. If False (default),
        return a copy with the conversion applied.

    Returns
    -------
    Array
        Coordinates with shape (..., 3) containing [x, r, rθ].
        If inplace=True, returns the modified input array.
        If inplace=False, returns a new array.
    """
    if inplace:
        xrt[..., 2] *= xrt[..., 1]
        return xrt
    else:
        result = xrt.copy()
        result[..., 2] *= result[..., 1]
        return result


def extent(*args):
    """Calculate per-component min and max values.

    Parameters
    ----------
    *args : array_like
        Arrays to compute the extent of, must have same number of
        dimensions, and same trailing dimension if ndim > 1.

    Returns
    -------
    Array, shape (2, ...)
        Extent array where extent[0, ...] contains minimum values and
        extent[1, ...] contains maximum values.
    """

    # Check all args have same ndim
    arrays = [np.asarray(arg) for arg in args]
    ndims = [arr.ndim for arr in arrays]
    ndim = ndims[0]
    assert all(nd == ndim for nd in ndims), "All arguments must have same ndim"

    # Determine axes to take min/max over
    axes = tuple(range(ndim - 1)) if ndim > 1 else (0,)

    # Find extent over axes of all arrays
    min_vals = np.stack([np.min(arr, axis=axes) for arr in arrays]).min(axis=0)
    max_vals = np.stack([np.max(arr, axis=axes) for arr in arrays]).max(axis=0)

    return np.stack([min_vals, max_vals])


def bounding_box(xyz):
    """Calculate bounding box vertices from Cartesian coordinates.

    Parameters
    ----------
    xyz : array_like, shape (N, 3)
        Cartesian coordinates [x, y, z] with components on last axis

    Returns
    -------
    Array, shape (8, 3)
        Eight vertices of the bounding box representing all combinations
        of (min/max x, min/max y, min/max z)
    """
    xyz = np.asarray(xyz)
    assert xyz.shape[-1] == 3, "xyz must have 3 components on last axis"

    # Get the per-component extent (min/max) of the coordinates
    xyz_extent = extent(xyz)  # Shape (2, 3)

    # Generate all 8 combinations using meshgrid
    meshes = np.meshgrid(*xyz_extent.T, indexing="ij")
    vertices = np.stack([mesh.ravel() for mesh in meshes], axis=1)

    return vertices


def _segment_conformal(ds, r0, r1):
    r"""Conformal distance along straight meridional segments.

    With :math:`r` varying linearly from `r0` to `r1` over a segment of
    meridional length `ds`, the integral has a closed form,

    .. math::

        \Delta m' = \int_0^1 \frac{\mathrm{d}s\,\mathrm{d}t}
                              {r_0 + t(r_1 - r_0)}
                  = \frac{\mathrm{d}s}{r_1 - r_0} \ln \frac{r_1}{r_0},

    tending to :math:`\mathrm{d}s / r_0` as :math:`r_1 \to r_0`. Evaluating it
    rather than quadrating leaves no discretisation error of its own: a
    polyline's conformal distance is exact for that polyline.
    """
    dr = r1 - r0

    # The logarithmic form loses all its significant figures as dr -> 0, where
    # it is a ratio of two vanishing quantities, so the cylindrical limit is
    # taken directly. The switch is on a relative tolerance because dr and r
    # are both lengths and only their ratio says whether the segment is
    # conical or cylindrical.
    conical = np.abs(dr) > 1e-6 * np.abs(r0 + r1)
    dr_safe = np.where(conical, dr, 1.0)
    r_mid = 0.5 * (r0 + r1)

    return np.where(
        conical,
        ds / dr_safe * np.log(np.where(conical, r1 / r0, 1.0)),
        ds / r_mid,
    )


def unwrap_meridional(xr_curve, xr_query):
    r"""Unwrap meridional coordinates onto conformal distance along a curve.

    Returns the conformal (or "blade-to-blade") meridional coordinate

    .. math::

        m' = \int \frac{\mathrm{d}m}{r},
        \qquad \mathrm{d}m = \sqrt{\mathrm{d}x^2 + \mathrm{d}r^2},

    integrated along `xr_curve` from its first point. Paired with
    :math:`\theta` in radians, :math:`m'` spans a conformal plane: angles and
    aspect ratios are preserved, so an aerofoil section drawn on
    :math:`(m', \theta)` axes keeps its shape at any radius.

    The coordinate is a property of the *curve*, not of any grid: every query
    point is referred to one datum by one integration, so points from
    different blocks --- or from different cuts of the same machine --- come
    back on a common scale with nothing to match up afterwards. Where the
    curve is the same one passed to :func:`ember.cut.structured_meridional`,
    the coordinate and the surface it describes are exactly consistent.

    Query points are projected onto the curve and their perpendicular offset
    discarded, so points lying near it rather than exactly on it --- the
    triangulated output of :func:`ember.cut.unstructured`, say, or a blade
    surface --- are handled without special-casing. A point beyond either end
    clamps to that end.

    Parameters
    ----------
    xr_curve : array_like, shape (n_point, 2)
        Meridional :math:`(x, r)` polyline defining the curve, in order.
        Must not touch the axis, where :math:`m'` diverges.
    xr_query : array_like, shape (..., 2)
        Meridional :math:`(x, r)` coordinates to evaluate.

    Returns
    -------
    Array, shape (...)
        Conformal distance :math:`m'` [-] at each query point, zero at the
        first point of the curve.

    Raises
    ------
    ValueError
        If the curve has fewer than two points, or reaches the axis.

    Examples
    --------
    >>> curve = np.array([[0.0, 2.0], [1.0, 2.0]])  # cylindrical, r = 2
    >>> float(unwrap_meridional(curve, np.array([1.0, 2.0])))  # m' = m / r
    0.5
    """
    xr_curve = np.asarray(xr_curve, dtype=np.float64)
    xr_query = np.asarray(xr_query, dtype=np.float64)

    if xr_curve.ndim != 2 or xr_curve.shape[-1] != 2:
        raise ValueError(f"Expected xr_curve shape (n_point, 2), got {xr_curve.shape}")
    if len(xr_curve) < 2:
        raise ValueError("A meridional curve needs at least two points")
    if xr_query.shape[-1] != 2:
        raise ValueError(f"Expected xr_query shape (..., 2), got {xr_query.shape}")
    if np.any(xr_curve[:, 1] <= 0.0):
        raise ValueError("Conformal distance diverges on the axis, so r must be > 0")

    start, end = xr_curve[:-1], xr_curve[1:]
    delta = end - start
    ds = np.sqrt(np.sum(delta**2, axis=-1))

    # Cumulative conformal distance at each vertex, zero at the first.
    mp_start = np.concatenate(
        [[0.0], np.cumsum(_segment_conformal(ds, start[:, 1], end[:, 1]))]
    )[:-1]

    query = xr_query.reshape(-1, 2)
    mp = np.zeros(len(query))
    nearest = np.full(len(query), np.inf)

    # Looped over segments rather than broadcast against them: a curve is
    # short and a grid is not, so this costs one pass per segment but never
    # materialises an (n_query, n_segment) array.
    for i, (p0, d, length) in enumerate(zip(start, delta, ds)):
        if length == 0.0:
            continue

        # Position along this segment of the closest point on it, clamped to
        # the segment so the ends extrapolate flat rather than off the curve.
        offset = query - p0
        t = np.clip(offset @ d / (length**2), 0.0, 1.0)

        distance = np.sqrt(np.sum((offset - t[:, None] * d) ** 2, axis=-1))
        closer = distance < nearest

        # The same closed form as the whole segment, stopped at the projected
        # point, so a query at t = 1 gives exactly the next vertex's value.
        r_projected = p0[1] + t * d[1]
        mp_partial = _segment_conformal(t * length, p0[1], r_projected)

        nearest = np.where(closer, distance, nearest)
        mp = np.where(closer, mp_start[i] + mp_partial, mp)

    return array(mp.reshape(xr_query.shape[:-1]))


def cart_to_pol(xyz, Vxyz, perm=(0, 1, 2), signs=(1, 1, 1)):
    r"""Convert Cartesian coordinates and velocities to polar form.

    Inverts the ember :math:`(x, r, \theta)` convention described at
    :ref:`coordinate-system`, i.e. the map implemented by :func:`pol_to_cart`:

    .. math::

        y = r \cos\theta, \qquad z = -r \sin\theta

    This function takes Cartesian data stacked on the last axis, and returns
    polar data of the same shape, e.g. for setting a block's raw coordinates
    and velocities :attr:`~ember.block.Block.xrt` and
    :attr:`~ember.block.Block.Vxrt` after reading a Cartesian solution.

    ``perm`` and ``signs`` arguments, when supplied, reorient the input before
    conversion, for source data that doesn't already follow our
    Cartesian axis order or sense. ``perm=(1, 0, 2)`` swaps :math:`x` and :math:`y`
    values at every point. ``signs[i] = -1`` negates the (already permuted)
    component ``i``. A mesh authored with :math:`z` pointing the opposite
    way to our convention should be read with ``signs=(1, 1, -1)``, for example.

    Parameters
    ----------
    xyz : array_like, shape (..., 3)
        Cartesian coordinates (x, y, z) with components on last axis
    Vxyz : array_like, shape (..., 3)
        Cartesian velocity components (Vx, Vy, Vz) with components on last axis
    perm : tuple of int, optional
        Coordinate permutation (0, 1, 2) -> reordered indices. Default: (0, 1, 2)
    signs : tuple of int, optional
        Coordinate signs (-1 or 1 for each axis). Default: (1, 1, 1)

    Returns
    -------
    xrt : Array, shape (..., 3)
        Polar coordinates (x, r, t) with components on last axis
    Vxrt : Array, shape (..., 3)
        Polar velocity components (Vx, Vr, Vt) with components on last axis
    """
    # Use double precision for intermediate calculations
    xyz = np.asarray(xyz, dtype=np.float64)
    Vxyz = np.asarray(Vxyz, dtype=np.float64)
    signs = np.array(signs, dtype=np.float64)

    # Apply coordinate transformation
    xyz_transformed = xyz[..., perm] * signs
    Vxyz_transformed = Vxyz[..., perm] * signs

    # Extract transformed coordinates
    x = xyz_transformed[..., 0]
    y = xyz_transformed[..., 1]
    z = xyz_transformed[..., 2]

    # Convert to polar coordinates
    r = np.sqrt(y**2 + z**2)
    t = np.arctan2(-z, y)

    # Handle angle wrapping: ensure t is in [0, 2π] to match typical CFD convention
    t = np.where(t < 0, t + 2 * np.pi, t)

    # Compute trigonometric functions from coordinates (more accurate than cos/sin of arctan2)
    cos_t = y / r
    sin_t = -z / r

    # Extract transformed velocities
    Vx = Vxyz_transformed[..., 0]
    Vy = Vxyz_transformed[..., 1]
    Vz = Vxyz_transformed[..., 2]

    # Convert to polar velocities (reuse cos_t, sin_t)
    Vr = Vy * cos_t - Vz * sin_t
    Vt = -Vy * sin_t - Vz * cos_t

    # Assemble results
    xrt = np.stack([x, r, t], axis=-1)
    Vxrt = np.stack([Vx, Vr, Vt], axis=-1)

    # Convert back to float32
    xrt = xrt.astype(np.float32)
    Vxrt = Vxrt.astype(np.float32)

    return xrt, Vxrt


def pol_to_cart(xrt, Vxrt, perm=(0, 1, 2), signs=(1, 1, 1)):
    r"""Convert polar coordinates and velocities to Cartesian form.

    The reverse of :func:`cart_to_pol`. Implements the ember :math:`(x, r,
    \theta)` convention described at :ref:`coordinate-system`:

    .. math::

        y = r \cos\theta, \qquad z = -r \sin\theta

    The inputs are typically read from :attr:`~ember.block.Block.xrt` and
    :attr:`~ember.block.Block.Vxrt`; the returned arrays are
    Cartesian data of the same shape.

    ``perm`` and ``signs`` reorient the output after conversion, for a target
    convention that isn't ember's own axis order or sense.
    ``perm[i]`` says which converted component
    supplies output component ``i``, and then ``signs[i] = -1`` negates the
    (already permuted) component ``i``.

    Note that reorientation is applied on the way in by :func:`cart_to_pol` but
    on the way out here, so the two are not simply symmetric for general
    ``perm`` and ``signs``. :meth:`ember.grid.Grid.align_cart_unstr` detects
    the correct orientation to align with a destination grid automatically when
    given Cartesian data.

    Parameters
    ----------
    xrt : array_like, shape (..., 3)
        Polar coordinates (x, r, t) with components on last axis
    Vxrt : array_like, shape (..., 3)
        Polar velocity components (Vx, Vr, Vt) with components on last axis
    perm : tuple of int, optional
        Coordinate permutation (0, 1, 2) -> reordered indices. Default: (0, 1, 2)
    signs : tuple of int, optional
        Coordinate signs (-1 or 1 for each axis). Default: (1, 1, 1)

    Returns
    -------
    xyz : Array, shape (..., 3)
        Cartesian coordinates (x, y, z) with components on last axis
    Vxyz : Array, shape (..., 3)
        Cartesian velocity components (Vx, Vy, Vz) with components on last axis
    """
    # Use double precision for intermediate calculations
    xrt = np.asarray(xrt, dtype=np.float64)
    Vxrt = np.asarray(Vxrt, dtype=np.float64)
    signs = np.array(signs, dtype=np.float64)

    # Extract polar coordinates
    x = xrt[..., 0]
    r = xrt[..., 1]
    t = xrt[..., 2]

    # Compute trigonometric functions once
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    # Convert to Cartesian coordinates
    y = r * cos_t
    z = -r * sin_t

    # Extract polar velocities
    Vx = Vxrt[..., 0]
    Vr = Vxrt[..., 1]
    Vt = Vxrt[..., 2]

    # Convert to Cartesian velocities (reuse cos_t, sin_t)
    Vy = Vr * cos_t - Vt * sin_t
    Vz = -Vr * sin_t - Vt * cos_t

    # Assemble Cartesian coordinates and velocities
    xyz = np.stack([x, y, z], axis=-1)
    Vxyz = np.stack([Vx, Vy, Vz], axis=-1)

    # Apply permutation and signs
    xyz = xyz[..., perm] * signs
    Vxyz = Vxyz[..., perm] * signs

    # Convert back to float32
    xyz = xyz.astype(np.float32)
    Vxyz = Vxyz.astype(np.float32)

    return xyz, Vxyz


def matmat(A, B):
    """Matrix-matrix multiplication over trailing dimensions.

    Performs matrix multiplication on stacks of matrices where the matrices
    are stored in the trailing dimensions. This is optimized for arrays with
    matrix data in the last two dimensions and arbitrary leading dimensions.

    Parameters
    ----------
    A : Array, shape (..., m, k)
        First input array with matrices in trailing dimensions.
    B : Array, shape (..., k, n)
        Second input array with matrices in trailing dimensions.

    Returns
    -------
    Array, shape (..., m, n)
        Result of matrix multiplication A @ B for each corresponding pair
        of matrices in the trailing dimensions. Uses f32 precision and
        Fortran ordering for optimal performance.

    Examples
    --------
    >>> # Stack of 2x2 matrices
    >>> A = np.random.randn(10, 5, 2, 2).astype(np.float32, order='F')
    >>> B = np.random.randn(10, 5, 2, 2).astype(np.float32, order='F')
    >>> C = matmat(A, B)  # Shape: (10, 5, 2, 2)

    >>> # Single matrix multiplication
    >>> A = np.eye(3, dtype=np.float32, order='F')
    >>> B = np.ones((3, 3), dtype=np.float32, order='F')
    >>> C = matmat(A, B)  # C = B
    """
    result = np.einsum("...ik,...kj->...ij", A, B)
    return np.asfortranarray(result.astype(f32))


def rotation_matrices(chi):
    r"""Build a paired 2x2 rotation matrix and its inverse from a meridional-plane angle.

    Both :class:`~ember.patch.RevolutionPatch`'s interface frame and
    :func:`~ember.block_util.resolve_to_interface` rotate a velocity pair
    :math:`(V_x, V_r)` by the same convention, so this is the one place that
    convention is written down:

    .. math::

        V_n &= \cos\chi\, V_x + \sin\chi\, V_r \\
        V_s &= -\sin\chi\, V_x + \cos\chi\, V_r

    ``rot_from`` is the transpose of ``rot_to`` -- the rotation is
    orthogonal -- so it undoes exactly this and nothing has to be re-derived
    to invert it.

    Parameters
    ----------
    chi : float or Array
        Angle [rad] of the frame axis from :math:`+x`, any shape.

    Returns
    -------
    rot_to, rot_from : Array, shape ``chi.shape + (2, 2)``
        ``rot_to`` turns :math:`(V_x, V_r)` into :math:`(V_n, V_s)`;
        ``rot_from`` turns :math:`(V_n, V_s)` back into :math:`(V_x, V_r)`.
    """
    chi = np.asarray(chi)
    c = np.cos(chi).astype(f32)
    s = np.sin(chi).astype(f32)
    rot_to = np.empty(c.shape + (2, 2), dtype=f32, order="F")
    rot_to[..., 0, 0] = c
    rot_to[..., 0, 1] = s
    rot_to[..., 1, 0] = -s
    rot_to[..., 1, 1] = c
    rot_from = np.empty_like(rot_to)
    rot_from[..., 0, 0] = c
    rot_from[..., 0, 1] = -s
    rot_from[..., 1, 0] = s
    rot_from[..., 1, 1] = c
    return rot_to, rot_from


def matvec(A, b, out=None):
    """Matrix-vector multiplication using einsum over trailing dimensions.

    Parameters
    ----------
    A : Array, shape (..., n, m)
        Input matrices with matrix dimensions in the last two axes.
    b : Array, shape (..., m)
        Input vectors with vector dimension in the last axis.
    out : Array, shape (..., n), optional
        Preallocated array to write the result into, sparing a new
        allocation.

    Returns
    -------
    Array, shape (..., n)
        Result of matrix-vector multiplication A @ b for each corresponding
        matrix and vector in the trailing dimensions. Uses f32 precision and
        Fortran ordering for optimal performance.

    Notes
    -----
    When ``out`` is given, ``A`` and ``b`` are checked against two specific
    hot-path shapes -- the per-grid-point 5x5 Jacobian contractions the
    solver runs every iteration -- and dispatched to a specialised Fortran
    kernel if they match. Any other shape combination falls back to
    ``numpy.matmul(..., out=...)``, which is still correct but without the
    specialised kernel. Either way ``out`` is updated in place and returned
    by identity.

    The two fast-path shapes, with ``b.shape[-1]`` fixed at 5 in every case:

    - **Single batch axis**: ``b.shape == (N, 5)`` and ``A.shape == (N, 5, 5)`` -- one matrix per vector, no broadcasting.
    - **Three-dimensional grid with one broadcast**: ``b.shape === (ni, nj, nk, 5)``,
      and ``A.shape`` one of:

      - ``(ni, nj, 1, 5, 5)`` -- one matrix per :math:`(i,j)`, shared
        across :math:`k`
      - ``(1, 1, nk, 5, 5)`` -- one matrix per :math:`k`, shared across
        :math:`(i,j)`
      - ``(ni, 1, 1, 5, 5)`` -- one matrix per :math:`i`, shared across
        :math:`(j,k)`

      A parallel set of kernels handles the same three broadcast patterns
      for ``b.shape[-1] == 2`` instead of 5.
      Any other broadcast pattern within the 3-grid-axis case,  including
      a full ``A`` shape ``(ni, nj, nk, n, m)`` with no broadcast axis at
      all, falls back to :obj:`numpy.matmul`.

    Examples
    --------
    >>> # Stack of 3x3 matrices with 3-element vectors
    >>> A = np.random.randn(10, 5, 3, 3).astype(np.float32, order='F')
    >>> b = np.random.randn(10, 5, 3).astype(np.float32, order='F')
    >>> y = matvec(A, b)  # Shape: (10, 5, 3)

    >>> # Single matrix-vector multiplication
    >>> A = np.eye(3, dtype=np.float32, order='F')
    >>> b = np.array([1, 2, 3], dtype=np.float32, order='F')
    >>> y = matvec(A, b)  # y = b

    >>> # Write into a preallocated buffer instead of allocating
    >>> A = np.random.randn(4, 5, 5).astype(np.float32, order='F')
    >>> b = np.random.randn(4, 5).astype(np.float32, order='F')
    >>> out = np.empty((4, 5), dtype=np.float32, order='F')
    >>> matvec(A, b, out=out) is out
    True
    """
    if out is not None:
        ndim = b.ndim - 1
        if ndim == 1:
            ember.fortran.matvec5(A, b, out)
        elif ndim == 3:
            # The two-axis broadcasts are tested before the one-axis one: an
            # (ni, 1, 1) matrix also satisfies the A.shape[2] == 1 test that
            # selects _bcast_j, so checking _bcast_j first would make _bcast_i
            # unreachable and hand the j kernel an nj it cannot use.
            if A.shape[-1] == 2:
                if A.shape[0] == 1 and A.shape[1] == 1:
                    ember.fortran.matvec2_bcast_k(A, b, out)
                elif A.shape[1] == 1 and A.shape[2] == 1:
                    ember.fortran.matvec2_bcast_i(A, b, out)
                elif A.shape[2] == 1:
                    ember.fortran.matvec2_bcast_j(A, b, out)
                else:
                    np.matmul(A, b[..., np.newaxis], out=out[..., np.newaxis])
            elif A.shape[0] == 1 and A.shape[1] == 1:
                ember.fortran.matvec5_bcast_k(A, b, out)
            elif A.shape[1] == 1 and A.shape[2] == 1:
                ember.fortran.matvec5_bcast_i(A, b, out)
            elif A.shape[2] == 1:
                ember.fortran.matvec5_bcast_j(A, b, out)
            else:
                np.matmul(A, b[..., np.newaxis], out=out[..., np.newaxis])
        else:
            np.matmul(A, b[..., np.newaxis], out=out[..., np.newaxis])
        return out
    result = np.matmul(A, b[..., np.newaxis]).squeeze(-1)
    return np.asfortranarray(result.astype(f32))


def resample(factor, vector, i_crit=None):
    """Resampled a vector with specified factor, optionally preserving critical points.

    Creates a new vector by resampling with a given factor. The new length equals
    len(vector) * factor (approximately). Critical indices are preserved, and for
    upsampling, fractional indices are linearly interpolated.

    Parameters
    ----------
    factor : float
        Resampling factor. Values > 1 increase resolution, < 1 decrease resolution.
        Must be > 0.
    vector : array_like
        Input vector to resample.
    i_crit : array_like, optional
        Critical indices that must be preserved in the output.
        Must be sorted and within [0, len(vector)-1]. If None, uses [0, len(vector)-1].

    Returns
    -------
    resampled_values : Array
        Resampled vector with length approximately len(vector) * factor,
        with values at critical indices preserved and interpolated elsewhere.
    crit_mapping : dict
        Dictionary mapping old critical indices to their positions in the new vector.
        Format: {old_index: new_index}

    Examples
    --------
    >>> # Increase resolution by factor of 2 with linear interpolation
    >>> x = np.array([0, 1, 4, 9, 16])
    >>> x_resampled, mapping = resample(2.0, x)
    >>> len(x_resampled)  # approximately 10
    >>> mapping[0], mapping[4]  # endpoints preserved at new positions

    >>> # Decrease resolution preserving critical points
    >>> x_resampled, mapping = resample(0.5, x, [0, 2, 4])
    >>> len(x_resampled)  # approximately 2-3
    >>> mapping  # {0: 0, 2: 1, 4: 2} - critical indices mapped to new positions
    """
    # Input validation
    if factor <= 0:
        raise ValueError("factor must be positive")

    vector = np.asarray(vector, dtype=np.float32)
    ni = len(vector)
    if ni == 0:
        return vector.copy(), {}

    # Set default critical indices to endpoints
    if i_crit is None:
        if ni == 1:
            # Single point vector - no resampling possible
            return vector.copy(), {0: 0}
        i_crit = [0, ni - 1]

    i_crit = np.asarray(i_crit, dtype=int)
    if len(i_crit) < 2:
        raise ValueError("i_crit must have at least 2 elements")
    if not np.all(np.diff(i_crit) > 0):
        raise ValueError("i_crit must be strictly increasing")
    if i_crit[0] < 0 or i_crit[-1] >= ni:
        raise ValueError(f"i_crit must be in range [0, {ni - 1}]")

    # Calculate spans between consecutive critical indices
    spans = np.diff(i_crit)

    # Allocate per-segment cell counts so they sum exactly to the target
    # total, instead of rounding each segment independently. Independent
    # rounding lets multiple segments each accrue a half-cell overshoot,
    # producing an output one node larger than the global target -- which
    # then violates the multigrid coarsening invariant
    # n_coarse == (n_fine - 1) // 2 + 1.
    #
    # Largest-remainder (Hare quota): floor each segment's ideal cell count,
    # then distribute the leftover cells to the segments with the largest
    # fractional remainder. Segments are clamped to >= 1 cell to keep the
    # critical-index mapping strictly increasing (two adjacent criticals
    # cannot collapse to the same output node).
    total_cells = int(np.round((ni - 1) * factor))
    seg_float = spans * factor
    seg_cells = np.floor(seg_float).astype(int)
    seg_cells = np.maximum(seg_cells, 1)
    remainder = seg_float - np.floor(seg_float)
    deficit = total_cells - seg_cells.sum()
    if deficit > 0:
        # Award extras to segments with largest fractional remainder.
        order = np.argsort(-remainder, kind="stable")
        seg_cells[order[:deficit]] += 1
    elif deficit < 0:
        # Trim from segments with smallest remainder, but never below 1.
        # Sweep repeatedly: a single pass removes at most one cell per
        # segment, so when the surplus exceeds the number of trimmable
        # segments it must keep sweeping the large segments down. A sweep
        # that trims nothing means every segment is already at 1 -- the
        # target genuinely cannot be met without collapsing a segment.
        order = np.argsort(remainder, kind="stable")
        while deficit != 0:
            trimmed = False
            for idx in order:
                if deficit == 0:
                    break
                if seg_cells[idx] > 1:
                    seg_cells[idx] -= 1
                    deficit += 1
                    trimmed = True
            if not trimmed:
                raise ValueError(
                    f"resample: cannot allocate {total_cells} cells across "
                    f"{len(spans)} segments without collapsing a segment "
                    f"(spans={spans.tolist()}, factor={factor})"
                )

    # Generate fractional indices for each segment
    fractional_indices = []
    for i, span in enumerate(spans):
        # Number of points in this segment (including both endpoints)
        n_points_in_segment = int(seg_cells[i]) + 1

        # Create linearly spaced fractional indices between critical points
        start_idx = i_crit[i]
        end_idx = i_crit[i + 1]
        segment_indices = np.linspace(start_idx, end_idx, n_points_in_segment)
        fractional_indices.append(segment_indices)

    # Combine all segments and remove duplicates while preserving order
    all_indices = np.concatenate(fractional_indices)
    # Remove duplicates by rounding and using unique, then sort
    unique_indices = np.unique(all_indices)

    # Interpolate vector values at the fractional indices
    # For integer indices, use exact values; for fractional, interpolate
    resampled_values = np.interp(
        unique_indices, np.arange(ni), vector.astype(np.float64)
    )
    # Convert back to float32 for consistency
    resampled_values = resampled_values.astype(np.float32)

    # Verify all critical indices are preserved (within tolerance for floating point)
    # and create mapping dictionary
    crit_mapping = {}
    for ic in i_crit:
        # Find the position of this critical index in the new vector
        closest_idx = np.argmin(np.abs(unique_indices - ic))
        assert np.abs(unique_indices[closest_idx] - ic) < 1e-10, (
            f"Critical index not preserved, want {ic}\n got {unique_indices}"
        )
        crit_mapping[int(ic)] = int(closest_idx)

    return resampled_values, crit_mapping


def apply_perm_flip(array, perm, flip=()):
    """Apply permutation and flipping to array while preserving coordinate dimension.

    This function applies a permutation and optional flipping to the spatial dimensions
    of an array, while preserving the last dimension (typically coordinates). This is
    commonly used in patch operations for coordinate transformations.

    Parameters
    ----------
    array : Array
        Input array to transform, typically with shape (..., 3) where the last
        dimension contains coordinate components.
    perm : tuple
        Permutation to apply to spatial dimensions (0, 1, 2).
    flip : tuple, optional
        Dimensions to flip after permutation. Default is () (no flipping).

    Returns
    -------
    Array
        Transformed array with permutation and flipping applied to spatial
        dimensions, last dimension preserved.

    Examples
    --------
    >>> coords = np.random.randn(5, 4, 3).astype(np.float32, order='F')
    >>> # Swap i and j dimensions
    >>> transformed = apply_perm_flip(coords, perm=(1, 0, 2))
    >>> transformed.shape == (4, 5, 3)  # i,j swapped, coordinates preserved
    True

    >>> # Swap dimensions and flip along first dimension
    >>> transformed = apply_perm_flip(coords, perm=(1, 0, 2), flip=(0,))
    """
    # Apply permutation to spatial dimensions, keep coordinate index (last dim)
    array_transformed = array.transpose(tuple(perm) + (3,))
    if flip:
        array_transformed = np.flip(array_transformed, axis=flip)
    return array_transformed


def cosine_cluster(n):
    r"""Generate cosine-clustered points from 0 to 1, dense at both ends.

    Produces ``n`` points on the unit interval following a half-cosine
    distribution, which clusters nodes near both endpoints (``zeta = 0`` and
    ``zeta = 1``) and spaces them most coarsely in the middle:

    .. math::

        \zeta_k = \tfrac{1}{2}\left(1 - \cos\frac{\pi k}{n - 1}\right),
        \quad k = 0, \ldots, n - 1.

    The result is symmetric about 0.5 with exact endpoints 0 and 1.

    Parameters
    ----------
    n : int
        Number of points to generate (must be >= 2).

    Returns
    -------
    Array, shape (n,)
        Cosine-clustered vector from 0 to 1, dtype=float32.

    Examples
    --------
    >>> z = cosine_cluster(5)
    >>> z[0], z[-1]  # exact endpoints
    (0.0, 1.0)
    >>> np.allclose(z + z[::-1], 1.0)  # symmetric
    True
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    return (0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n)))).astype(f32)


def cluster(ni, ER, dmax):
    """Generate geometrically spaced points from 0 to 1 with expansion ratio and max spacing.

    Creates a vector from 0 to 1 with geometrically spaced points, where the spacing
    expands at ratio ER but is capped to maximum size dmax. The algorithm creates an
    initial geometric dx vector, then iteratively adjusts the scaling to achieve
    unit total length while respecting dmax after capping.

    Parameters
    ----------
    ni : int
        Number of points to generate (must be >= 2)
    ER : float
        Expansion ratio for geometric spacing (must be > 0)
    dmax : float
        Maximum allowed spacing between consecutive points (must be > 0)

    Returns
    -------
    Array, shape (ni,)
        Vector from 0 to 1 with clustered spacing, dtype=float32

    Examples
    --------
    >>> # Basic clustering with expansion ratio 1.2
    >>> x = cluster(10, 1.2, 0.5)
    >>> x[0], x[-1]  # Should be (0.0, 1.0)

    >>> # Uniform spacing when ER=1.0
    >>> x = cluster(5, 1.0, 1.0)
    >>> np.allclose(x, np.linspace(0, 1, 5))
    True
    """
    if ni < 2:
        raise ValueError("ni must be >= 2")
    if ER <= 0:
        raise ValueError("ER must be > 0")
    if dmax <= 0:
        raise ValueError("dmax must be > 0")

    # Handle uniform spacing case
    if ER == 1.0:
        return np.linspace(0, 1, ni, dtype=f32)

    # Create initial geometric spacing
    dx_geom = np.ones(ni - 1, dtype=f32)
    for i in range(1, ni - 1):
        dx_geom[i] = dx_geom[i - 1] * ER

    # Check if dmax is too restrictive
    min_uniform_spacing = 1.0 / (ni - 1)
    if dmax < min_uniform_spacing:
        # dmax is too restrictive, fall back to uniform spacing
        return np.linspace(0, 1, ni, dtype=f32)

    # Scale the geometric spacing to achieve unit length after capping
    max_iterations = 100
    tolerance = 1e-8
    scale = 1.0

    for _ in range(max_iterations):
        # Apply scaling and cap
        dx_scaled = dx_geom * scale
        dx_capped = np.minimum(dx_scaled, dmax)
        current_length = np.sum(dx_capped)

        if abs(current_length - 1.0) < tolerance:
            break

        # Check if we've hit the cap limit
        if np.allclose(dx_capped, dmax, rtol=1e-6) and current_length < 1.0:
            # All spacings are at dmax but total length < 1, impossible to satisfy
            # Fall back to uniform spacing
            return np.linspace(0, 1, ni, dtype=f32)

        # Adjust scaling to get closer to unit length
        scale *= 1.0 / current_length

    # Final spacing with scaling and capping
    dx = np.minimum(dx_geom * scale, dmax)

    # Rescale dx to ensure exact unit total length
    dx_sum = np.sum(dx)
    if dx_sum > 0:
        dx *= 1.0 / dx_sum

    # Construct coordinate vector by cumulative sum
    x = np.zeros(ni, dtype=f32)
    x[1:] = np.cumsum(dx)

    # Ensure exact endpoints (should already be correct now)
    x[0] = 0.0
    x[-1] = 1.0

    return x


def cluster_symmetric(n, ER, dmax=1.0):
    """Generate geometrically spaced points from 0 to 1, dense at both ends.

    Where :func:`cluster` expands away from a single end, this mirrors a
    half-width :func:`cluster` vector about the centreline, so the spacing
    grows at expansion ratio ``ER`` away from *both* endpoints and is coarsest
    in the middle. Unlike :func:`cosine_cluster`, which is also symmetric, the
    growth rate is controlled rather than fixed by the distribution.

    Parameters
    ----------
    n : int
        Number of points to generate. Must be odd and >= 3, so that the two
        mirrored halves share their midpoint.
    ER : float
        Expansion ratio for geometric spacing (must be > 0).
    dmax : float
        Maximum allowed spacing in the returned vector (must be > 0).

    Returns
    -------
    Array, shape (n,)
        Vector from 0 to 1 with spacing clustered at both ends, dtype=float32.

    Examples
    --------
    >>> z = cluster_symmetric(9, 1.2, 1.0)
    >>> z[0], z[-1]  # exact endpoints
    (0.0, 1.0)
    >>> np.allclose(z + z[::-1], 1.0)  # symmetric about 0.5
    True

    >>> # Uniform spacing when ER=1.0
    >>> np.allclose(cluster_symmetric(5, 1.0, 1.0), np.linspace(0, 1, 5))
    True
    """
    if n < 3 or n % 2 == 0:
        raise ValueError(f"n must be odd and >= 3 to mirror a half-width, got {n}")

    # The half-vector spans 0 to 1 and is then scaled onto 0 to 0.5, so a cap
    # of dmax on the result corresponds to 2*dmax on the half.
    half = cluster((n + 1) // 2, ER, 2.0 * dmax)
    x = np.concatenate([0.5 * half, 1.0 - 0.5 * half[-2::-1]]).astype(f32)

    # Guard the endpoints and midpoint against round-off in the mirror
    x[0] = 0.0
    x[n // 2] = 0.5
    x[-1] = 1.0

    return x
