"""Utility functions for EMBER CFD computations.

This module provides a comprehensive collection of utility functions for computational
fluid dynamics simulations in turbomachinery applications. Functions are organized
into several categories:

**Flow Angle and Velocity Conversions**
- `angles_to_components()` - Convert velocity magnitude and flow angles to components
- `components_to_angles()` - Convert velocity components to magnitude and flow angles

**Vector and Matrix Operations**
- `dot()` - Dot product along last axis using einsum
- `vecnorm()` - Vector norm along last axis
- `matmat()` - Matrix-matrix multiplication over trailing dimensions
- `matvec()` - Matrix-vector multiplication over trailing dimensions
- `inv()` - Matrix inversion for stacks of matrices

**Array Creation and Broadcasting**
- `zeros()` - Create zero array with F-order and float32
- `empty()` - Create empty array with F-order and float32
- `full()` - Create filled array with F-order and float32
- `allocate_or_reuse()` - Allocate output array if not provided (for out= parameters)
- `full_bcast()` - Create filled array with broadcast shape
- `stack_matrix()` - Stack nested iterables into matrix with trailing dimensions

**Coordinate System Utilities**
- `meshgrid3()` - Create 3D coordinate meshgrid for EMBER (x,r,θ)
- `linmesh3()` - Create 3D meshgrid from coordinate ranges and shape
- `cart_to_pol()` - Convert Cartesian coordinates/velocities to polar form
- `pol_to_cart()` - Convert polar coordinates/velocities to Cartesian form
- `pol_to_pseudocart()` - Convert (x,r,θ) to pseudo-Cartesian (x,r,rθ)

**CFD-Specific Utilities**
- `get_atol()` - Calculate absolute tolerances for flux conservation tests
- `resolve_to_interface()` - Convert meridional velocities to interface-aligned
- `resolve_from_interface()` - Convert interface-aligned velocities to meridional
- `signed_distance()` - Distance above/below piecewise line in meridional plane
- `dot_conserved()` - Specialized dot product for conserved variable fluxes
- `rms()` - Root mean square calculation

**Turbomachinery Operations**
- `pitchwise_repeat()` - Repeat blocks pitchwise in theta direction
- `bounds()` - Calculate min/max values of arrays
- `bounding_box()` - Calculate bounding box vertices from Cartesian coordinates

**Grid Spacing**
- `cosine_cluster()` - Cosine-clustered points on [0, 1], dense at both ends
- `cluster()` - Geometrically spaced points on [0, 1] with expansion ratio and max spacing

**Array Manipulation**
- `corners()` - Extract corner elements from N-D arrays
- `resample()` - Resample vector with specified factor preserving critical points
- `apply_perm_flip()` - Apply permutation and flipping while preserving coordinates
- `reverse_perm_flip()` - Reverse transformations from apply_perm_flip

**Patch and Grid Utilities**
- `perm_flip_to_dirs()` - Convert permutation/flip to TS3-style direction encoding

All functions are optimized for performance with Fortran-order arrays and float32
precision where appropriate. The module supports both scalar and array inputs with
proper broadcasting behavior.
"""

import numpy as np
import itertools

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


def angles_to_components(V_rel, Alpha_rel_deg, Beta_deg):
    """Resolve relative velocity into axial, radial, and tangential components.

    Parameters
    ----------
    V_rel : Array
        Relative velocity magnitude [m/s]
    Alpha_rel_deg : Array
        Relative yaw angle (tangential flow direction) [degrees]
    Beta_deg : Array
        Pitch angle (radial flow direction) [degrees]

    Returns
    -------
    Vx : Array
        Axial velocity component [m/s]
    Vr : Array
        Radial velocity component [m/s]
    Vt_rel : Array
        Relative tangential velocity component [m/s]

    Notes
    -----
    Uses numerically stable trigonometry for all angles including ±90°.

    Flow angle conventions:
    - Alpha_rel = 0°: No relative swirl (Vt_rel = 0)
    - Beta = 0°: Pure axial flow (Vr = 0)
    - Beta = ±90°: Pure radial flow (Vx = 0)

    The velocity decomposition follows turbomachinery conventions:
    1. Alpha decomposition: V_rel → V_meridional, Vt_rel
    2. Beta decomposition: V_meridional → Vx, Vr
    """
    # Convert to consistent dtype and radians
    V_rel = np.asarray(V_rel, dtype=f32)
    alpha_rad = np.radians(np.asarray(Alpha_rel_deg, dtype=f32))
    beta_rad = np.radians(np.asarray(Beta_deg, dtype=f32))

    # Use pure trigonometry for numerical stability
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)
    cos_beta = np.cos(beta_rad)
    sin_beta = np.sin(beta_rad)

    # Alpha decomposition: meridional and tangential components
    V_rel_m = V_rel * cos_alpha  # Meridional velocity magnitude
    Vt_rel = V_rel * sin_alpha  # Tangential relative velocity

    # Beta decomposition: axial and radial components
    Vx = V_rel_m * cos_beta  # Axial velocity
    Vr = V_rel_m * sin_beta  # Radial velocity

    return Vx, Vr, Vt_rel


def components_to_angles(Vx, Vr, Vt_rel):
    """Convert velocity components to relative velocity magnitude and flow angles.

    Parameters
    ----------
    Vx : Array
        Axial velocity component [m/s]
    Vr : Array
        Radial velocity component [m/s]
    Vt_rel : Array
        Relative tangential velocity component [m/s]

    Returns
    -------
    V_rel : Array
        Relative velocity magnitude [m/s]
    Alpha_rel_deg : Array
        Relative yaw angle (tangential flow direction) [degrees]
    Beta_deg : Array
        Pitch angle (radial flow direction) [degrees]

    Notes
    -----
    This is the inverse of angles_to_components. Flow angle conventions:
    - Alpha_rel = 0°: No relative swirl (Vt_rel = 0)
    - Beta = 0°: Pure axial flow (Vr = 0)
    - Beta = ±90°: Pure radial flow (Vx = 0)

    For velocity components with very small magnitudes, angles may be
    numerically unstable. Zero velocity returns (0, 0, 0).
    """
    # Convert to consistent dtype
    Vx = np.asarray(Vx, dtype=f32)
    Vr = np.asarray(Vr, dtype=f32)
    Vt_rel = np.asarray(Vt_rel, dtype=f32)

    # Calculate meridional velocity magnitude
    V_rel_m = np.sqrt(Vx**2 + Vr**2)

    # Calculate total relative velocity magnitude
    V_rel = np.sqrt(V_rel_m**2 + Vt_rel**2)

    # Handle zero velocity case
    zero_velocity = V_rel < 1e-12

    # Calculate Alpha_rel (yaw angle from meridional vs tangential)
    # atan2(Vt_rel, V_rel_m) but handle zero meridional velocity
    Alpha_rel_rad = np.where(zero_velocity, 0.0, np.arctan2(Vt_rel, V_rel_m))

    # Calculate Beta (pitch angle from axial vs radial)
    # atan2(Vr, Vx) but handle zero meridional velocity
    Beta_rad = np.where(
        V_rel_m < 1e-12,  # Pure tangential flow or zero velocity
        0.0,
        np.arctan2(Vr, Vx),
    )

    # Convert to degrees
    Alpha_rel_deg = np.degrees(Alpha_rel_rad)
    Beta_deg = np.degrees(Beta_rad)

    return V_rel, Alpha_rel_deg, Beta_deg


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
    return np.zeros(shape, dtype=dtype, order="F")


def array(x, dtype=np.float32):
    return np.array(x, dtype=dtype, order="F")


def empty(shape):
    return np.empty(shape, dtype=np.float32, order="F")


def full(shape, fill_value):
    return np.full(shape, fill_value, dtype=np.float32, order="F")


def allocate_or_reuse(out, shape, dtype=np.float32):
    """Allocate output array if not provided, otherwise reuse existing array.

    Helper function for functions that accept an optional `out` parameter.
    If `out` is None, allocates a new F-contiguous zero array with the
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
    ...     out = allocate_or_reuse(out, x.shape, dtype=np.float64)
    ...     # ... compute results into out ...
    ...     return out
    """
    if out is None:
        return np.zeros(shape, dtype=dtype, order="F")
    else:
        return out


def carve_view(buf, *shapes):
    """Carve one or more non-overlapping zero-copy F-order views from a buffer.

    Reinterprets successive spans of a Fortran-contiguous `buf` as arrays of the
    requested `shapes`, packing them end-to-end (view ``k`` starts where view
    ``k-1`` ends) so every returned view aliases distinct storage and all may be
    held live simultaneously. Used to borrow differently-shaped scratch arrays
    out of one oversized solver buffer (e.g. a block's ``scratch``, ``store`` or
    ``tau_q_halo``) without allocating.

    The offsets are computed internally, so a caller carving several coexisting
    slots (e.g. the ``aplane``/``bb``/``corr`` MG scratch out of ``tau_q_halo``)
    cannot accidentally overlap or gap them. ``buf.reshape(-1, order="F")`` on
    such a buffer is itself a free view, so repeated calls stay zero-copy and
    there is no need to hoist a shared flat view at the call site.

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


def stack_matrix(*args, shape, out=None):
    """Stack nested iterables into a matrix with trailing matrix dimensions.

    Parameters
    ----------
    args : nested iterables length [nrow][ncol]
        Variables to stack, where args[i][j] contains the (i,j) matrix element.
        Use None for zero entries to skip the copy.
    shape : tuple
        Grid shape for the batch dimensions.
    out : array, optional
        Preallocated output array of shape (*shape, nrow, ncol). If None, a new
        array is allocated.

    Returns
    -------
    out : Array, shape (*shape, nrow, ncol)
        A composite matrix variable with matrix dimensions in trailing axes.
        Uses f32 precision and Fortran ordering for optimal performance.
    """
    nrow = len(args)
    ncol = len(args[0])

    if out is None:
        out = np.empty(shape + (nrow, ncol), dtype=f32, order="F")
    out.fill(0.0)
    for i in range(nrow):
        for j in range(ncol):
            v = args[i][j]
            if v is not None:
                out[..., i, j] = v
    return out


def full_bcast(a, b, fill_value):
    """Create a filled array with broadcast shape of two input arrays.

    This function determines the broadcast shape of two arrays and creates
    a new array filled with the specified value using that shape. The resulting
    array has optimal memory layout (Fortran order) and consistent dtype.

    Parameters
    ----------
    a : array_like
        First input array for shape broadcasting.
    b : array_like
        Second input array for shape broadcasting.
    fill_value : scalar
        Value to fill the array with.

    Returns
    -------
    Array
        New array with broadcast shape filled with fill_value, using
        dtype=np.float32 and order="F" for optimal performance.

    Examples
    --------
    >>> a = np.array([[1, 2]])  # shape (1, 2)
    >>> b = np.array([[1], [2], [3]])  # shape (3, 1)
    >>> result = full_bcast(a, b, 5.0)
    >>> result.shape
    (3, 2)
    >>> np.all(result == 5.0)
    True
    """
    return full(np.broadcast(a, b).shape, fill_value)


def meshgrid3(xv, rv, tv):
    """Create 3D coordinate meshgrid for EMBER.

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

    This function creates linspace vectors from coordinate ranges and then
    generates a 3D meshgrid. It combines the common pattern of creating
    linspace vectors and then meshing them.

    Parameters
    ----------
    x : tuple or array_like
        X-coordinate range [x_min, x_max].
    r : tuple or array_like
        R-coordinate range [r_min, r_max].
    t : tuple or array_like
        Theta-coordinate range [t_min, t_max].
    shape : tuple
        Shape of the grid (ni, nj, nk).

    Returns
    -------
    Array, shape (ni, nj, nk, 3)
        Coordinate array with [x, r, t] components, dtype=float32.

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


def get_atol(conserved, r_av, rtol):
    """Calculate absolute tolerances for flux conservation tests.

    This function computes physically-based absolute tolerances for testing
    flux conservation in CFD simulations. The tolerances are based on
    mean flow properties and account for the different physical scales
    of each conserved variable.

    Parameters
    ----------
    conserved : Array, shape (..., 5)
        Conserved variables [rho, rho*Vx, rho*Vr, rho*r*Vt, rho*e].
    r : Array
        Radial coordinate.
    rtol : float
        Relative tolerance multiplier.

    Returns
    -------
    Array, shape (5,)
        Absolute tolerances for [mass, x-momentum, r-momentum, θ-momentum, energy] fluxes.

    Examples
    --------
    >>> atol = get_atol(block.conserved, block.r.mean(), 1e-12)
    >>> assert np.all(np.abs(net_flow) <= atol)
    """

    r_av = np.mean(r_av)

    # Mean conserved quantities for physical scales
    rho_av = conserved[..., 0].mean()  # Density scale
    Vx_av = np.abs(conserved[..., 1] / conserved[..., 0]).mean()
    Vr_av = np.abs(conserved[..., 2] / conserved[..., 0]).mean()
    Vt_av = np.abs(conserved[..., 3] / conserved[..., 0] / r_av).mean()
    V_av = np.sqrt(Vx_av**2 + Vr_av**2 + Vt_av**2)  # Representative velocity scale

    # Assemble absolute tolerances for each conserved variable
    atol = (
        np.array(
            [
                rho_av,
                rho_av * V_av,
                rho_av * V_av,
                rho_av * r_av * V_av,
                rho_av * V_av**2,
            ]
        )
        * rtol
    )

    return atol


def pitchwise_repeat(blocks, n, symmetric=False):
    """Repeat blocks pitchwise in theta direction.

    This function creates copies of blocks with shifted theta coordinates to
    simulate pitchwise repetition in turbomachinery applications. Each repeated
    block has its Nb updated to match the new periodicity.

    Parameters
    ----------
    blocks : Grid, list of Block, or Block
        Original blocks to repeat. Can be:
        - A Grid object (returns Grid)
        - A list of Block objects (returns list)
        - A single Block (returns list)
    n : int
        Number of repetitions:
        - n > 0: repeat n times
        - n = 0: return empty list/Grid
        - If symmetric=False: range is [0, 1, ..., n-1] for positive n
        - If symmetric=True: range is [-n, -n+1, ..., -1, 1, 2, ..., n] (excludes 0)
    symmetric : bool, optional
        If True, create symmetric repetitions around the original position.
        If False, create repetitions in one direction only. Default: False.

    Returns
    -------
    Grid or list of Block
        Returns same type as input (Grid→Grid, list→list, Block→list).
        Contains repeated blocks with shifted theta coordinates.
        Each block has Nb = original_Nb * total_repetitions to match the new periodicity.

    Examples
    --------
    >>> # Repeat single block 3 times in positive direction
    >>> repeated = pitchwise_repeat(block, 3)
    >>> len(repeated)  # 3 blocks: [0, 1, 2]
    3

    >>> # Repeat symmetrically: 3 negative + 3 positive (excludes original)
    >>> repeated = pitchwise_repeat(block, 3, symmetric=True)
    >>> len(repeated)  # 6 blocks: [-3, -2, -1, 1, 2, 3]
    6

    >>> # Single repetition symmetrically
    >>> repeated = pitchwise_repeat(block, 1, symmetric=True)
    >>> len(repeated)  # 2 blocks: [-1, 1]
    2

    >>> # Repeat Grid object
    >>> grid = Grid([block1, block2])
    >>> repeated_grid = pitchwise_repeat(grid, 3)
    >>> isinstance(repeated_grid, Grid)  # Returns Grid
    True
    """
    # Track input type to return matching output type
    input_was_grid = False

    # Handle Grid input
    if hasattr(blocks, "__class__") and blocks.__class__.__name__ == "Grid":
        input_was_grid = True
        # Grid is iterable, no need to convert to list
    # Convert single block to list
    elif not isinstance(blocks, list):
        blocks = [blocks]

    if n == 0:
        if input_was_grid:
            from ember.grid import Grid

            return Grid([])
        return []

    if n < 0:
        raise ValueError("n must be >= 0")

    if symmetric:
        # Create symmetric range: [-n, -n+1, ..., -1, 1, 2, ..., n]
        # Excludes the original position at i=0
        i_range = list(range(-n, 0)) + list(range(1, n + 1))
        total_repetitions = 2 * n
    else:
        # Create asymmetric range: [0, 1, ..., n-1]
        i_range = list(range(0, n))
        total_repetitions = n

    result = []
    for i in i_range:
        for block in blocks:
            new_block = block.copy()
            new_block.set_t(new_block.t + new_block.pitch * i)
            # Update Nb to match the new periodicity
            new_block.set_Nb(int(new_block.Nb * total_repetitions))
            result.append(new_block)

    # Return same type as input
    if input_was_grid:
        from ember.grid import Grid

        return Grid(result)
    return result


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


def bounds(*args):
    """Calculate the minimum and maximum values of arrays.

    Parameters
    ----------
    *args : array_like
        Arrays to compute bounds for, must have same number of dimensions, and
        same trailing dimension if ndim > 1.

    Returns
    -------
    Array, shape (2, ...)
        Bounds array where bounds[0, ...] contains minimum values and
        bounds[1, ...] contains maximum values.
    """

    # Check all args have same ndim
    arrays = [np.asarray(arg) for arg in args]
    ndims = [arr.ndim for arr in arrays]
    ndim = ndims[0]
    assert all(nd == ndim for nd in ndims), "All arguments must have same ndim"

    # Determine axes to take min/max over
    axes = tuple(range(ndim - 1)) if ndim > 1 else (0,)

    # Find bounds over axes of all arrays
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

    # Get min/max bounds for each coordinate
    xyz_bounds = bounds(xyz)  # Shape (2, 3)

    # Generate all 8 combinations using meshgrid
    meshes = np.meshgrid(*xyz_bounds.T, indexing="ij")
    vertices = np.stack([mesh.ravel() for mesh in meshes], axis=1)

    return vertices


def cart_to_pol(xyz, Vxyz, perm=(0, 1, 2), signs=(1, 1, 1)):
    """Convert Cartesian coordinates and velocities to polar form.

    Parameters
    ----------
    xyz : array_like, shape (..., 3)
        Cartesian coordinates [x, y, z] with components on last axis
    Vxyz : array_like, shape (..., 3)
        Cartesian velocity components [Vx, Vy, Vz] with components on last axis
    perm : tuple of int, optional
        Coordinate permutation (0, 1, 2) -> reordered indices. Default: (0, 1, 2)
    signs : tuple of int, optional
        Coordinate signs (-1 or 1 for each axis). Default: (1, 1, 1)

    Returns
    -------
    xrt : Array, shape (..., 3)
        Polar coordinates [x, r, t] with components on last axis
    Vxrt : Array, shape (..., 3)
        Polar velocity components [Vx, Vr, Vt] with components on last axis
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
    """Convert polar coordinates and velocities to Cartesian form.

    Parameters
    ----------
    xrt : array_like, shape (..., 3)
        Polar coordinates [x, r, t] with components on last axis
    Vxrt : array_like, shape (..., 3)
        Polar velocity components [Vx, Vr, Vt] with components on last axis
    perm : tuple of int, optional
        Coordinate permutation (0, 1, 2) -> reordered indices. Default: (0, 1, 2)
    signs : tuple of int, optional
        Coordinate signs (-1 or 1 for each axis). Default: (1, 1, 1)

    Returns
    -------
    xyz : Array, shape (..., 3)
        Cartesian coordinates [x, y, z] with components on last axis
    Vxyz : Array, shape (..., 3)
        Cartesian velocity components [Vx, Vy, Vz] with components on last axis
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
    """Matrix-matrix multiplication using einsum over trailing dimensions.

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


def matvec(A, b, out=None):
    """Matrix-vector multiplication using einsum over trailing dimensions.

    Performs matrix-vector multiplication on stacks of matrices and vectors
    where the matrices are in trailing dimensions (..., n, m) and vectors
    are in trailing dimensions (..., m).

    Parameters
    ----------
    A : Array, shape (..., n, m)
        Input matrices with matrix dimensions in the last two axes.
    b : Array, shape (..., m)
        Input vectors with vector dimension in the last axis.

    Returns
    -------
    Array, shape (..., n)
        Result of matrix-vector multiplication A @ b for each corresponding
        matrix and vector in the trailing dimensions. Uses f32 precision and
        Fortran ordering for optimal performance.

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
    """
    if out is not None:
        import ember.fortran

        ndim = b.ndim - 1
        if ndim == 1:
            ember.fortran.matvec5(A, b, out)
        elif ndim == 3:
            if A.shape[-1] == 2:
                if A.shape[2] == 1:
                    ember.fortran.matvec2_bcast_j(A, b, out)
                elif A.shape[0] == 1 and A.shape[1] == 1:
                    ember.fortran.matvec2_bcast_k(A, b, out)
                elif A.shape[1] == 1 and A.shape[2] == 1:
                    ember.fortran.matvec2_bcast_i(A, b, out)
                else:
                    np.matmul(A, b[..., np.newaxis], out=out[..., np.newaxis])
            elif A.shape[2] == 1:
                ember.fortran.matvec5_bcast_j(A, b, out)
            elif A.shape[0] == 1 and A.shape[1] == 1:
                ember.fortran.matvec5_bcast_k(A, b, out)
            elif A.shape[1] == 1 and A.shape[2] == 1:
                ember.fortran.matvec5_bcast_i(A, b, out)
            else:
                return np.matmul(
                    A, b[..., np.newaxis], out=out[..., np.newaxis]
                ).squeeze(-1)
        else:
            return np.matmul(A, b[..., np.newaxis], out=out[..., np.newaxis]).squeeze(
                -1
            )
        return out
    result = np.matmul(A, b[..., np.newaxis]).squeeze(-1)
    return np.asfortranarray(result.astype(f32))


def inv(A):
    """Matrix inverse for arrays where matrix dims are last two axes.

    Parameters
    ----------
    A : array_like, shape (..., m, m)
        Input matrices with matrix dimensions as trailing axes

    Returns
    -------
    A_inv : array_like, shape (..., m, m)
        Inverse matrices with same shape as input
    """
    # For trailing dimensions, numpy.linalg.inv already expects (..., m, m)
    return np.linalg.inv(A)


def signed_distance(xr, xr_query):
    """Distance above or below a piecewise line in meridional plane.

    Note that this becomes increasingly inaccurate far away from the
    curve but the zero level is correct (which is sufficient for cutting).

    Parameters
    ----------
    xr : Array, shape (nseg, 2)
        Coordinates of the cut plane segments following ember convention
        with components in last axis.
    xr_query : Array, shape (..., 2)
        Meridional coordinates to evaluate distance at.

    Returns
    -------
    Array, shape (...)
        Signed distance above or below the cut.

    """

    assert xr.shape[-1] == 2, "Segments must have shape (..., 2)"
    assert xr_query.shape[-1] == 2, "Points must have shape (..., 2)"
    assert xr.ndim >= 2, "Segments must be at least 2D"

    # Preallocate the signed distance
    d = np.full(xr_query.shape[:-1], np.inf)

    # Number of segments
    nseg = xr.shape[0]

    # Loop over line segments
    for i in range(nseg - 1):
        # Get segment endpoints: current and next points
        seg_start = xr[i]  # Shape (..., 2)
        seg_end = xr[i + 1]  # Shape (..., 2)

        # Calculate vectors from segment start to points and along segment
        a = xr_query - seg_start  # Point vector from segment start
        b = seg_end - seg_start  # Segment direction vector

        # Project point onto segment and clamp to [0,1]
        L = np.maximum(dot(b, b), 1e-9)  # Segment length squared
        h = np.clip(dot(a, b) / L, 0.0, 1.0)  # Normalized distance along segment

        # Get perpendicular component (shortest distance to segment)
        parallel_component = b * h[..., np.newaxis]  # Add axis for broadcasting
        perpendicular = a - parallel_component  # Perpendicular vector
        di = np.sqrt(dot(perpendicular, perpendicular))  # Distance magnitude

        # Find points where this segment gives the closest distance
        ind = np.where(di < np.abs(d))

        # Make the distance signed using perpendicular vector to segment
        # For 2D, perpendicular to [bx, br] is [-br, bx]
        normal = np.stack([-b[1], b[0]], axis=-1)  # Normal vector to segment
        di *= np.sign(dot(perpendicular, normal))

        # Update minimum distance where this segment is closest
        d[ind] = di[ind]

    return d


def rms(*args):
    """Compute root mean square."""
    if len(args) == 1:
        arrays = np.asarray(args[0])
    else:
        arrays = np.concatenate(args)
    return np.sqrt(np.mean(arrays**2))


def dot_conserved(flux, dA, axes):
    return np.sum(
        np.einsum(
            "...ij,...i->...j",
            flux,
            dA,
        ),
        axis=axes,
    )


def resolve_to_interface(block, chi):
    """Convert meridional velocity to interface-aligned velocities.

    Resolves the meridional velocity components (Vx, Vr) to velocities
    aligned with an interface at angle chi: velocity through the interface (Vm)
    and velocity normal to the interface (Vn).

    Parameters
    ----------
    block : Block
        Block containing velocity data to be resolved.
    chi : float or Array
        Interface angle in degrees. When chi=0, Vm=Vx and Vn=Vr.
        When chi=90, Vm=Vr and Vn=-Vx.

    Returns
    -------
    Block
        The input block with velocities updated to interface-aligned form.
        Vm becomes the new Vx, Vn becomes the new Vr, Vt unchanged.
    """
    chi_rad = np.radians(chi)
    cos_chi = np.cos(chi_rad)
    sin_chi = np.sin(chi_rad)

    # Current meridional velocities
    Vx_old = block.Vx
    Vr_old = block.Vr
    Vt = block.Vt

    # Transform to interface coordinates
    Vm = cos_chi * Vx_old + sin_chi * Vr_old
    Vn = -sin_chi * Vx_old + cos_chi * Vr_old

    # Update block velocities: Vm->Vx, Vn->Vr, Vt unchanged
    block.set_Vx(Vm)
    block.set_Vr(Vn)
    block.set_Vt(Vt)
    return block


def resolve_from_interface(block, chi):
    """Convert interface-aligned velocities back to meridional components.

    Converts interface-aligned velocities (Vm=block.Vx through interface,
    Vn=block.Vr normal to interface) back to meridional components (Vx, Vr)
    using interface angle chi.

    Parameters
    ----------
    block : Block
        Block containing interface-aligned velocities (Vm=block.Vx, Vn=block.Vr).
    chi : float or Array
        Interface angle in degrees.

    Returns
    -------
    Block
        The input block with velocities updated to meridional form.
    """
    chi_rad = np.radians(chi)
    cos_chi = np.cos(chi_rad)
    sin_chi = np.sin(chi_rad)

    # Current interface-aligned velocities
    Vm = block.Vx
    Vn = block.Vr
    Vt = block.Vt

    # Transform from interface coordinates to meridional
    Vx = cos_chi * Vm - sin_chi * Vn
    Vr = sin_chi * Vm + cos_chi * Vn

    # Update block velocities
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)
    return block


def perm_flip_to_dirs(perm, flip, const_dim):
    """Convert permutation/flip to (idir, jdir, kdir) encoding.

    This function converts the permutation and flip information from patch
    matching into the TS3-style direction indices used to describe how patch
    coordinates align between blocks.

    Parameters
    ----------
    perm : tuple of int
        Permutation tuple where perm[self_axis] = other_axis.
        For example, perm[0] = 1 means self's i-axis aligns with other's j-axis.
    flip : tuple of int
        Axes to flip after permutation (applied to self coordinate system).
        Flip is applied AFTER the permutation to align permuted other with self.
    const_dim : int
        Which dimension (0=i, 1=j, 2=k) is constant for this patch.

    Returns
    -------
    tuple of int
        (idir, jdir, kdir) where each value encodes:
        0: aligns with i-axis of other (positive)
        1: aligns with j-axis of other (positive)
        2: aligns with k-axis of other (positive)
        3: aligns with i-axis of other (negative)
        4: aligns with j-axis of other (negative)
        5: aligns with k-axis of other (negative)
        6: patch lies on constant face in that dimension

    Examples
    --------
    >>> # Identity transformation on j-constant patch
    >>> perm_flip_to_dirs((0, 1, 2), (), 1)
    (0, 6, 2)

    >>> # Permuted with flip on k-constant patch
    >>> perm_flip_to_dirs((1, 0, 2), (0,), 2)
    (4, 3, 6)
    """
    dirs = []

    for self_axis in range(3):
        if self_axis == const_dim:
            # This dimension is constant, so direction is 6
            dirs.append(6)
        else:
            # Find which other axis this self axis aligns with
            other_axis = perm[self_axis]

            # Check if this self axis is flipped
            is_flipped = self_axis in flip

            # Encode as 0-2 for positive, 3-5 for negative alignment
            dir_value = other_axis + (3 if is_flipped else 0)
            dirs.append(dir_value)

    return tuple(dirs)


def corners(x, axis_exclude=None):
    """Extract corner elements from an ND numpy array.

    This function extracts all corner elements from an N-dimensional array
    by taking the first and last indices (0, -1) along each dimension,
    except for dimensions specified in axis_exclude. The corners are
    stacked along axis 0 in the returned array.

    Parameters
    ----------
    x : array_like
        Input N-dimensional array from which to extract corners.
    axis_exclude : int or tuple of int, optional
        Axis or axes to exclude from corner extraction. These axes
        will be preserved in full (using slice(None)). Default: None.

    Returns
    -------
    Array
        Array containing all corner elements stacked along axis 0.
        Shape is (2^n_varying_dims, ...) where n_varying_dims is the
        number of dimensions not excluded.

    Examples
    --------
    >>> # 2D array corners
    >>> x = np.arange(20).reshape(4, 5)
    >>> corners(x).shape
    (4, ...)
    >>> # Returns x[0,0], x[0,-1], x[-1,0], x[-1,-1] stacked along axis 0

    >>> # 3D array with last axis excluded
    >>> x = np.arange(60).reshape(3, 4, 5)
    >>> corners(x, axis_exclude=-1).shape
    (4, 5)
    >>> # Returns x[0,0,:], x[0,-1,:], x[-1,0,:], x[-1,-1,:] stacked along axis 0

    >>> # 3D array, all corners
    >>> corners(x).shape
    (8, ...)
    >>> # Returns all 8 corners: x[0,0,0], x[0,0,-1], x[0,-1,0], etc.
    """
    x = np.asarray(x)

    # Handle axis_exclude parameter
    if axis_exclude is None:
        exclude_set = set()
    elif isinstance(axis_exclude, int):
        exclude_set = {axis_exclude % x.ndim}
    else:
        exclude_set = {ax % x.ndim for ax in axis_exclude}

    # Build corner indices for each dimension
    corner_indices = []
    for dim in range(x.ndim):
        if dim in exclude_set:
            corner_indices.append([slice(None)])
        else:
            corner_indices.append([0, -1])

    # Generate all combinations of corner indices
    index_combinations = list(itertools.product(*corner_indices))

    # Extract each corner and collect results
    corner_arrays = []
    for indices in index_combinations:
        corner_arrays.append(x[indices])

    # Stack all corners along axis 0
    return np.stack(corner_arrays, axis=0)


def resample(factor, vector, i_crit=None):
    """Return resampled vector with specified factor, optionally preserving critical points.

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


def upsample_1d(zeta_fine, zeta_coarse, y_coarse):
    """Upsample coarse data to fine grid using linear interpolation.

    Interpolates coarse grid data onto a fine grid using linear interpolation.
    Assumes coarse grid points correspond to every other fine grid point
    (i.e., coarse points are at even indices 0, 2, 4, ... of the fine grid).

    Parameters
    ----------
    zeta_fine : array_like, shape (n_fine,)
        Fine grid normalized arc length values [0 to 1].
    zeta_coarse : array_like, shape (n_coarse,)
        Coarse grid normalized arc length values [0 to 1].
        Must be a subset of zeta_fine at even indices.
    y_coarse : array_like, shape (n_coarse,)
        Data values at coarse grid points.

    Returns
    -------
    y_fine : Array, shape (n_fine,)
        Linearly interpolated data at fine grid points.

    Examples
    --------
    >>> zeta_fine = np.linspace(0, 1, 9)
    >>> zeta_coarse = zeta_fine[::2]  # [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> y_coarse = zeta_coarse**2
    >>> y_fine = upsample_1d(zeta_fine, zeta_coarse, y_coarse)
    >>> # y_fine contains interpolated values at all 9 fine grid points
    """

    zeta_fine = np.asarray(zeta_fine, dtype=f32, order="F")
    zeta_coarse = np.asarray(zeta_coarse, dtype=f32, order="F")
    y_coarse = np.asarray(y_coarse, dtype=f32, order="F")

    n_fine = len(zeta_fine)
    n_coarse = len(zeta_coarse)

    # Validate inputs
    if n_coarse != (n_fine + 1) // 2:
        raise ValueError(
            f"Expected n_coarse={(n_fine + 1) // 2} for n_fine={n_fine}, got {n_coarse}"
        )

    # Allocate output array
    y_fine = zeros(n_fine)

    # Call Fortran implementation
    import ember.fortran

    ember.fortran.upsample_1d(zeta_fine, zeta_coarse, y_coarse, y_fine)

    return y_fine


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


def reverse_perm_flip(array, perm, flip=()):
    """Reverse the transformations applied by apply_perm_flip.

    This function reverses the permutation and flipping operations applied by
    apply_perm_flip, restoring the array to its original orientation. The operations
    are applied in reverse order: first reverse the flip, then reverse the permutation.

    Parameters
    ----------
    array : Array
        Input array that was transformed by apply_perm_flip.
    perm : tuple
        The same permutation that was used in apply_perm_flip.
    flip : tuple, optional
        The same flip dimensions that were used in apply_perm_flip. Default is () (no flipping).

    Returns
    -------
    Array
        Array restored to its original orientation before apply_perm_flip was applied.

    Examples
    --------
    >>> coords = np.random.randn(5, 4, 3).astype(np.float32, order='F')
    >>> # Apply transformation
    >>> transformed = apply_perm_flip(coords, perm=(1, 0, 2), flip=(0,))
    >>> # Reverse transformation
    >>> restored = reverse_perm_flip(transformed, perm=(1, 0, 2), flip=(0,))
    >>> np.allclose(coords, restored)
    True
    """
    # Reverse the flip first (flipping is its own inverse)
    array_unflipped = array
    if flip:
        array_unflipped = np.flip(array, axis=flip)

    # Reverse the permutation by computing the inverse permutation
    # If perm = (1, 0, 2), then inverse_perm[perm[i]] = i
    inverse_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse_perm[p] = i

    # Apply inverse permutation to spatial dimensions, keep coordinate index (last dim)
    array_restored = array_unflipped.transpose(tuple(inverse_perm) + (3,))
    return array_restored


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
