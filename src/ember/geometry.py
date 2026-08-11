"""Interpolation between nodes, faces and cells on structured grids.

.. warning::

    Neither function in this module currently has a production caller --
    :func:`cell_to_node` and :func:`node_to_face` are exercised only by
    ``tests/test_geometry.py``. :mod:`ember.solver`'s docstring narrates a
    multigrid restrict/prolong step that would use :func:`cell_to_node`, but
    nothing wires it up yet. Kept for now as apparently-intentional
    groundwork; worth revisiting for deletion if no caller appears.

The face-area and cell-volume functions that used to live here have moved to
their single respective callers as private helpers: see ``Block._get_dai``,
``_get_daj``, ``_get_dak``, ``_get_da_quad``, ``_get_da_tri`` and ``_get_vol``
in :mod:`ember.block`. Likewise ``node_to_face_2d`` moved into
:mod:`ember.average` (as ``_node_to_face_2d``) and ``compute_parametric_coords``
into :mod:`ember.nonmatch_communicator` (as ``_compute_parametric_coords``).
"""

import numpy as np

try:
    import ember.fortran
except ImportError as e:
    raise ImportError(f"Failed to import Fortran module: {e}") from e


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
