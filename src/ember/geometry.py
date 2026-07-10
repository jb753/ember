"""Functions for calculating face areas and cell volumes on structured grids."""

import numpy as np

import ember.util as util

try:
    import ember.fortran
except ImportError as e:
    raise ImportError(f"Failed to import Fortran module: {e}") from e

f32 = np.float32


def _handle_output(result, out=None):
    """Helper function to handle optional output array parameter.

    Parameters
    ----------
    result : Array
        The computed result array.
    out : Array, optional
        Output array to store results. Must have compatible shape with result.

    Returns
    -------
    Array
        Either the original result or the out array with result copied into it.
    """
    if out is not None:
        out[...] = result
        return out
    return result


def node_to_face_2d(nodal_data):
    r"""Average nodal values to face centres for 2D data.

    For a :math:`(n_i, n_j)` array of nodal values, the face-centred value is

    .. math::

        \bar{q}_{i,j} = \tfrac{1}{4}\bigl(
            q_{i,j} + q_{i+1,j} + q_{i+1,j+1} + q_{i,j+1}\bigr)

    Parameters
    ----------
    nodal_data : Array, shape (ni, nj, ...)
        Values at grid nodes.

    Returns
    -------
    Array, shape (ni-1, nj-1, ...)
        Values averaged to face centres.
    """
    return 0.25 * (
        nodal_data[:-1, :-1, ...]
        + nodal_data[1:, :-1, ...]
        + nodal_data[1:, 1:, ...]
        + nodal_data[:-1, 1:, ...]
    )


def node_to_cell(x):
    r"""Average the eight corner nodes of each cell to get volume-averaged values.

    .. math::

        \bar{q}_{i,j,k} = \tfrac{1}{8} \sum_{\delta i \in \{0,1\}}
            \sum_{\delta j \in \{0,1\}} \sum_{\delta k \in \{0,1\}}
            q_{i+\delta i,\, j+\delta j,\, k+\delta k}

    For a :math:`(n_i, n_j, n_k, \ldots)` nodal array, returns a
    :math:`(n_i-1, n_j-1, n_k-1, \ldots)` cell-centred array."""
    if x.shape[1] == 2 and x.shape[2] == 2:
        # Special case for 1D arrays with dummy j and k dimensions of length 2
        xout = np.stack(
            (
                x[:-1, 0, 0, ...],
                x[1:, 0, 0, ...],
                x[:-1, 1, 0, ...],
                x[1:, 1, 0, ...],
            )
        ).mean(axis=0)[:, None, None, ...]
        return xout

    return np.mean(
        np.stack(
            (
                x[:-1, :-1, :-1, ...],  # i, j, k
                x[1:, :-1, :-1, ...],  # i+1, j, k
                x[:-1, 1:, :-1, ...],  # i, j+1, k
                x[1:, 1:, :-1, ...],  # i+1, j+1, k
                x[:-1, :-1, 1:, ...],  # i, j, k+1
                x[1:, :-1, 1:, ...],  # i+1, j, k+1
                x[:-1, 1:, 1:, ...],  # i, j+1, k+1
                x[1:, 1:, 1:, ...],  # i+1, j+1, k+1
            ),
        ),
        axis=0,
    )


def cell_to_node(cell_data):
    """Interpolate cell-centered data to nodes using Fortran implementation.

    For a (ni-1, nj-1, nk-1, ...) matrix of cell-centered properties,
    interpolate to produce an (ni, nj, nk, ...) matrix of node-centered properties.

    Parameters
    ----------
    cell_data : Array, shape (ni-1, nj-1, nk-1, ...)
        Cell-centered data to interpolate to nodes.

    Returns
    -------
    Array, shape (ni, nj, nk, ...)
        Node-centered data interpolated from cells.

    Notes
    -----
    This function calls the optimized Fortran routine `ember.fortran.cell_to_node`.
    The interpolation uses averaging of neighboring cell values to compute node values.
    """
    if cell_data.ndim < 3:
        raise ValueError(
            "cell_to_node requires at least 3D cell data (ni-1, nj-1, nk-1, ...)"
        )

    # Calculate output shape: add 1 to first three dimensions
    output_shape = list(cell_data.shape)
    output_shape[0] += 1  # ni-1 -> ni
    output_shape[1] += 1  # nj-1 -> nj
    output_shape[2] += 1  # nk-1 -> nk

    # Ensure input data is Fortran-ordered and float32 for optimal performance
    cell_data_f = np.asarray(cell_data, dtype=np.float32, order="F")

    # Allocate output array with Fortran ordering and matching dtype for compatibility
    node_data = np.zeros(tuple(output_shape), dtype=np.float32, order="F")

    # Call Fortran routine to perform interpolation
    ember.fortran.cell_to_node(cell_data_f, node_data)

    return node_data


def node_to_face(x, out=None):
    r"""Average nodal values to the centres of all three families of cell faces.

    The four-node average for each face family is

    .. math::

        \bar{q}^{(i)}_{i,j,k} &= \tfrac{1}{4}\bigl(
            q_{i,j,k} + q_{i,j+1,k} + q_{i,j+1,k+1} + q_{i,j,k+1}\bigr) \\
        \bar{q}^{(j)}_{i,j,k} &= \tfrac{1}{4}\bigl(
            q_{i,j,k} + q_{i+1,j,k} + q_{i+1,j,k+1} + q_{i,j,k+1}\bigr) \\
        \bar{q}^{(k)}_{i,j,k} &= \tfrac{1}{4}\bigl(
            q_{i,j,k} + q_{i+1,j,k} + q_{i+1,j+1,k} + q_{i,j+1,k}\bigr)

    Parameters
    ----------
    x : Array, shape (ni, nj, nk, ...) or (ni, nj, ...)
        Values at grid nodes.  2D arrays are padded with a dummy k dimension.
    out : tuple of Arrays, optional
        Tuple of 3 pre-allocated output arrays ``(xi, xj, xk)``.

    Returns
    -------
    xi : Array, shape (ni, nj-1, nk-1, ...)
        Averaged values on constant-i faces.
    xj : Array, shape (ni-1, nj, nk-1, ...)
        Averaged values on constant-j faces.
    xk : Array, shape (ni-1, nj-1, nk, ...)
        Averaged values on constant-k faces.

    """

    if x.ndim == 0:
        # For points arrays, duplicate thrice to create 3D shape of (2, 2, 2)
        x = np.stack([x, x], axis=0)
        x = np.stack([x, x], axis=1)
        x = np.stack([x, x], axis=2)

    # Handle 1D arrays by adding dummy j and k dimensions
    elif x.ndim == 1:
        # For 1D arrays, duplicate to create j and k dimensions of length 2
        x = np.stack([x, x], axis=1)
        x = np.stack([x, x], axis=2)

    # Handle 2D arrays by adding dummy k dimension
    elif x.ndim == 2:
        # For 2D arrays, duplicate to create k dimension of length 2
        x = np.stack([x, x], axis=2)

    # Compute all faces with optional out parameter support
    if out is not None and len(out) == 3:
        out_xi, out_xj, out_xk = out
    else:
        out_xi = out_xj = out_xk = None

    # Compute constant-i faces
    xi_computed = 0.25 * (
        x[:, :-1, :-1, ...]
        + x[:, 1:, :-1, ...]
        + x[:, 1:, 1:, ...]
        + x[:, :-1, 1:, ...]
    )
    xi = _handle_output(xi_computed, out_xi)

    # Compute constant-j faces
    xj_computed = 0.25 * (
        x[:-1, :, :-1, ...]
        + x[1:, :, :-1, ...]
        + x[1:, :, 1:, ...]
        + x[:-1, :, 1:, ...]
    )
    xj = _handle_output(xj_computed, out_xj)

    # Compute constant-k faces
    xk_computed = 0.25 * (
        x[:-1, :-1, :, ...]
        + x[1:, :-1, :, ...]
        + x[1:, 1:, :, ...]
        + x[:-1, 1:, :, ...]
    )
    xk = _handle_output(xk_computed, out_xk)

    return xi, xj, xk


def get_dA_tri(xrt, out=None):
    r"""Area vectors of triangular faces.

    For a triangle with vertices :math:`A, B, C` in pseudo-Cartesian space
    :math:`(x, r, r\theta)`:

    .. math::

        \delta\!\mathbf{A} = \tfrac{1}{2}\,\overrightarrow{AC} \times \overrightarrow{AB}

    Parameters
    ----------
    xrt : Array, shape (ntri, 3, 3)
        Polar coordinates :math:`(x, r, \theta)` at the three vertices of each triangle.
    out : Array, optional
        Output array to store results. Must have shape (ntri, 3).

    Returns
    -------
    dA : Array, shape (ntri, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.

    """

    xrrt = util.pol_to_pseudocart(xrt)
    qAB = xrrt[:, 1, :] - xrrt[:, 0, :]
    qAC = xrrt[:, 2, :] - xrrt[:, 0, :]
    # Swap order to match structured area orientation
    return _handle_output(0.5 * np.cross(qAC, qAB, axis=-1), out)


def get_dAi(xrt, out=None):
    r"""Area vectors of constant-i faces.

    Each face is bounded by the four nodes
    :math:`(i,j,k),\,(i,j,k{+}1),\,(i,j{+}1,k{+}1),\,(i,j{+}1,k)`,
    circulating so that the area vector points along increasing i.
    Evaluated as half the cross product of the face diagonals, which is
    exact for a warped face; see the Geometry and indexing documentation.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, nk, 3)
        Polar coordinates :math:`(x, r, \theta)` at grid nodes.
    out : Array, optional
        Output array. Must have shape (ni, nj-1, nk-1, 3).

    Returns
    -------
    dAi : Array, shape (ni, nj-1, nk-1, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.

    """

    # Validate input
    ndim = xrt.ndim - 1  # Spatial dimensions only
    if ndim != 3:
        raise ValueError(f"dAi is not defined for ndim={ndim}.")

    # Preserve input dtype for precision
    input_dtype = xrt.dtype

    # Ensure inputs are Fortran-ordered and float64 for Fortran compatibility
    xrt_f = np.asarray(xrt, dtype=np.float64, order="F")

    # Allocate output array if not provided
    ni, nj, nk = xrt.shape[:3]
    dAi_temp = util.allocate_or_reuse(None, (ni, nj - 1, nk - 1, 3), dtype=np.float64)

    # Call Fortran routine to perform face area calculation
    ember.fortran.get_dai(xrt_f, dAi_temp)

    # Convert back to input dtype to preserve precision
    dAi = dAi_temp.astype(input_dtype, copy=False)

    return _handle_output(dAi, out)


def get_dAj(xrt, out=None):
    r"""Area vectors of constant-j faces.

    Each face is bounded by the four nodes
    :math:`(i,j,k),\,(i{+}1,j,k),\,(i{+}1,j,k{+}1),\,(i,j,k{+}1)`,
    circulating so that the area vector points along increasing j.
    Evaluated as half the cross product of the face diagonals, which is
    exact for a warped face; see the Geometry and indexing documentation.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, nk, 3)
        Polar coordinates :math:`(x, r, \theta)` at grid nodes.
    out : Array, optional
        Output array. Must have shape (ni-1, nj, nk-1, 3).

    Returns
    -------
    dAj : Array, shape (ni-1, nj, nk-1, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.

    """

    # Validate input
    ndim = xrt.ndim - 1  # Spatial dimensions only
    if ndim != 3:
        raise ValueError(f"dAj is not defined for ndim={ndim}.")

    # Preserve input dtype for precision
    input_dtype = xrt.dtype

    # Ensure inputs are Fortran-ordered and float64 for Fortran compatibility
    xrt_f = np.asarray(xrt, dtype=np.float64, order="F")

    # Allocate output array if not provided
    ni, nj, nk = xrt.shape[:3]
    dAj_temp = util.allocate_or_reuse(None, (ni - 1, nj, nk - 1, 3), dtype=np.float64)

    # Call Fortran routine to perform face area calculation
    ember.fortran.get_daj(xrt_f, dAj_temp)

    # Convert back to input dtype to preserve precision
    dAj = dAj_temp.astype(input_dtype, copy=False)

    return _handle_output(dAj, out)


def get_dAk(xrt, out=None):
    r"""Area vectors of constant-k faces.

    Each face is bounded by the four nodes
    :math:`(i,j,k),\,(i,j{+}1,k),\,(i{+}1,j{+}1,k),\,(i{+}1,j,k)`,
    circulating so that the area vector points along increasing k.
    Evaluated as half the cross product of the face diagonals, which is
    exact for a warped face; see the Geometry and indexing documentation.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, nk, 3)
        Polar coordinates :math:`(x, r, \theta)` at grid nodes.
    out : Array, optional
        Output array. Must have shape (ni-1, nj-1, nk, 3).

    Returns
    -------
    dAk : Array, shape (ni-1, nj-1, nk, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.

    """

    # Validate input
    ndim = xrt.ndim - 1  # Spatial dimensions only
    if ndim != 3:
        raise ValueError(f"dAk is not defined for ndim={ndim}.")

    # Preserve input dtype for precision
    input_dtype = xrt.dtype

    # Ensure inputs are Fortran-ordered and float64 for Fortran compatibility
    xrt_f = np.asarray(xrt, dtype=np.float64, order="F")

    # Allocate output array if not provided
    ni, nj, nk = xrt.shape[:3]
    dAk_temp = util.allocate_or_reuse(None, (ni - 1, nj - 1, nk, 3), dtype=np.float64)

    # Call Fortran routine to perform face area calculation
    ember.fortran.get_dak(xrt_f, dAk_temp)

    # Convert back to input dtype to preserve precision
    dAk = dAk_temp.astype(input_dtype, copy=False)

    return _handle_output(dAk, out)


def get_dA_quad(xrt, out=None):
    r"""Area vectors of quadrilateral faces on a two-dimensional cut.

    Delegates to :func:`get_dAk` with a dummy third dimension added and then
    removed.  The four nodes of each face are
    :math:`(i,j),\,(i,j{+}1),\,(i{+}1,j{+}1),\,(i{+}1,j)`.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, 3)
        Polar coordinates :math:`(x, r, \theta)` at cut nodes.
    out : Array, optional
        Output array. Must have shape (ni-1, nj-1, 3).

    Returns
    -------
    dA : Array, shape (ni-1, nj-1, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.

    """

    ndim = xrt.ndim - 1  # Exclude the coordinate index
    assert ndim == 2, "Face area is only defined for 2D grids"

    # Add a dummy third dimension for compatibility and calculate dAk
    xrt = xrt[:, :, None, :]
    dA = get_dAk(xrt)
    dA = dA[:, :, 0, :]

    return _handle_output(dA, out)


def get_vol(xrt, dAi, dAj, dAk, out=None):
    r"""Cell volumes via the divergence theorem.

    With the vector field :math:`\mathbf{F} = (x,\, r/2,\, r\theta)`,
    :math:`\nabla\cdot\mathbf{F} = 3` in cylindrical coordinates, so

    .. math::

        \delta\mathcal{V} = \frac{1}{3}
            \sum_{\text{faces}} \mathbf{F}_f \cdot \delta\mathbf{A}_f

    where :math:`\mathbf{F}_f` is the average of the four corner nodes on
    each face.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, nk, 3)
        Polar coordinates :math:`(x, r, \theta)` at grid nodes.
    dAi : Array, shape (3, ni, nj-1, nk-1) or (ni, nj-1, nk-1, 3)
        Constant-i face area vectors.
    dAj : Array, shape (3, ni-1, nj, nk-1) or (ni-1, nj, nk-1, 3)
        Constant-j face area vectors.
    dAk : Array, shape (3, ni-1, nj-1, nk) or (ni-1, nj-1, nk, 3)
        Constant-k face area vectors.
    out : Array, optional
        Output array. Must have shape (ni-1, nj-1, nk-1).

    Returns
    -------
    vol : Array, shape (ni-1, nj-1, nk-1)
        Cell volumes.

    """

    # Check number of spatial dimensions
    ndim = xrt.ndim - 1  # Exclude the coordinate index
    assert ndim == 3, "Volume is only defined for 3D grids"

    ni, nj, nk = xrt.shape[:3]

    # Accept both (3, ...) components-first and (..., 3) components-last layouts
    if dAi.shape == (ni, nj - 1, nk - 1, 3):
        dAi = np.moveaxis(dAi, -1, 0)
    if dAj.shape == (ni - 1, nj, nk - 1, 3):
        dAj = np.moveaxis(dAj, -1, 0)
    if dAk.shape == (ni - 1, nj - 1, nk, 3):
        dAk = np.moveaxis(dAk, -1, 0)

    if dAi.shape != (3, ni, nj - 1, nk - 1):
        raise ValueError(f"Invalid shape for dAi: {dAi.shape}")
    if dAj.shape != (3, ni - 1, nj, nk - 1):
        raise ValueError(f"Invalid shape for dAj: {dAj.shape}")
    if dAk.shape != (3, ni - 1, nj - 1, nk):
        raise ValueError(f"Invalid shape for dAk: {dAk.shape}")

    # Preserve input dtype for precision (use xrt as reference)
    input_dtype = xrt.dtype

    # Ensure inputs are Fortran-ordered and float64 for Fortran compatibility
    xrt_f = np.asarray(xrt, dtype=np.float64, order="F")
    dAi_f = np.asarray(dAi, dtype=np.float64, order="F")
    dAj_f = np.asarray(dAj, dtype=np.float64, order="F")
    dAk_f = np.asarray(dAk, dtype=np.float64, order="F")

    # Allocate output array if not provided
    vol_temp = util.allocate_or_reuse(None, (ni - 1, nj - 1, nk - 1), dtype=np.float64)

    # Call Fortran routine to perform volume calculation
    ember.fortran.get_vol(xrt_f, dAi_f, dAj_f, dAk_f, vol_temp)

    # Convert back to input dtype to preserve precision
    vol = vol_temp.astype(input_dtype, copy=False)

    return _handle_output(vol, out)


def compute_parametric_coords(xrt, const_dim):
    """Compute parametric coordinates for a structured patch face.

    Maps a 3D patch to 2D parametric space (u,v) ∈ [0,1]^2 using
    arc length along grid lines. Both u and v span [0,1] with
    u=0,v=0 at one corner and u=1,v=1 at opposite corner.

    This function is used for non-matching patch interpolation where two
    block faces occupy the same physical space but have different nodal
    distributions. The parametric coordinates provide a common reference
    frame for transferring data between patches.

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
    >>> uv = compute_parametric_coords(xrt, const_dim=0)
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
    xyz = util.pol_to_pseudocart(xrt_2d)

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
