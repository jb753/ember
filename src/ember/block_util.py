r"""Operations on :class:`~ember.block.Block` instances that don't belong on the class itself.

Each function below takes one or more :class:`~ember.block.Block`\ s and
returns a new one (or mutates in place, per its docstring). Grid-level
counterparts such as :meth:`~ember.grid.Grid.resample` and
:meth:`~ember.grid.Grid.interp_from_grid` are thin loops over these
block-granularity functions.

Combining and reshaping blocks
===============================

.. autosummary::

   concatenate
   resample

Interface-aligned velocities
=============================

Paired functions that rotate the meridional velocity components (Vx, Vr) by a
precomputed 2x2 matrix from :func:`~ember.util.rotation_matrices`, each the
inverse of its partner.

.. autosummary::

   resolve_to_interface
   resolve_from_interface

Solution transfer
==================

.. autosummary::

   interp_from_arrays
   interp_from_grid

Post-processing and I/O
=========================

.. autosummary::

   wall_yplus
   to_tm3
"""

import logging

import numpy as np

import ember.collections
import ember.fortran
from ember import util
from ember.block import Block

logger = logging.getLogger(__name__)


def concatenate(*blocks, axis=0):
    """Concatenate multiple blocks along a specified axis.

    Parameters
    ----------
    *blocks : Block
        Blocks to concatenate in order
    axis : int, optional
        Axis along which to concatenate (0, 1, or 2), default 0

    Returns
    -------
    Block
        New block containing concatenated data

    Raises
    ------
    ValueError
        If no blocks provided, blocks have incompatible working fluids,
        incompatible shapes, or patches on concatenation interfaces
    """
    if len(blocks) == 0:
        raise ValueError("At least 1 block required for concatenation")
    elif len(blocks) == 1:
        return blocks[0].copy()

    # Concatenate sequentially using the private helper
    result = blocks[0]
    for block in blocks[1:]:
        result = _concatenate_two_blocks(result, block, axis)

    return result


def _concatenate_two_blocks(block1, block2, axis=0):
    """Private helper to concatenate exactly two blocks along a specified axis.

    Parameters
    ----------
    block1 : Block
        First block to concatenate
    block2 : Block
        Second block to concatenate
    axis : int, optional
        Axis along which to concatenate (0, 1, or 2), default 0

    Returns
    -------
    Block
        New block containing concatenated data

    Raises
    ------
    ValueError
        If blocks have incompatible working fluids, incompatible shapes,
        or patches on concatenation interfaces
    """
    # Check that shapes are compatible for concatenation
    shape1 = block1.shape
    shape2 = block2.shape

    if axis < 0 or axis >= len(shape1):
        raise ValueError(f"Invalid axis {axis} for shapes {shape1}")

    # Check all dimensions except concatenation axis match
    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            raise ValueError(
                f"Incompatible shapes for concatenation: {shape1} and {shape2} along axis {axis}"
            )

    # Calculate new shape
    new_shape = list(shape1)
    new_shape[axis] = shape1[axis] + shape2[axis]
    new_shape = tuple(new_shape)

    # Create new block with concatenated shape
    result = Block(shape=new_shape)

    # Copy metadata from first block (prioritize block1 properties)
    for key, value in block1._metadata.items():
        if key != "patches":  # Handle patches separately
            result._set_metadata_by_key(key, value)

    # Check for metadata mismatches and error
    for key in block1._metadata:
        if key in ["patches", "label", "Omega", "Nb"]:
            continue
        if key in block2._metadata:
            value1 = block1._metadata[key]
            value2 = block2._metadata[key]

            # Handle different types of comparisons
            if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                if not np.array_equal(value1, value2):
                    raise ValueError(
                        f"Metadata mismatch for '{key}': block1={value1}, block2={value2}"
                    )
            elif isinstance(value1, (int, float, np.number)) and isinstance(
                value2, (int, float, np.number)
            ):
                if not np.allclose(value1, value2, rtol=1e-10):
                    raise ValueError(
                        f"Metadata mismatch for '{key}': block1={value1}, block2={value2}"
                    )
            elif value1 != value2:
                raise ValueError(
                    f"Metadata mismatch for '{key}': block1={value1}, block2={value2}"
                )

    # Error about metadata keys that exist in only one block
    keys1 = set(block1._metadata.keys()) - {"patches"}
    keys2 = set(block2._metadata.keys()) - {"patches"}
    only_in_block1 = keys1 - keys2
    only_in_block2 = keys2 - keys1

    if only_in_block1:
        raise ValueError(f"Metadata keys {only_in_block1} exist only in block1")
    if only_in_block2:
        raise ValueError(f"Metadata keys {only_in_block2} exist only in block2")

    # Concatenate data arrays
    data1 = block1._data
    data2 = block2._data
    result._data = np.concatenate([data1, data2], axis=axis)

    # Mark all data keys as initialized if they were in both blocks
    for key in block1._data_keys:
        if block1._versions[key] and block2._versions[key]:
            result._versions[key] = 1

    # Handle patches with proper index adjustment
    result._metadata["patches"] = ember.collections.BlockPatchCollection(result)
    Omega = np.array(
        [block1._metadata.get("Omega", 0), block2._metadata.get("Omega", 0)]
    )
    result._metadata["Omega"] = Omega[np.argmax(np.abs(Omega))]

    axis_offset = block1.shape[axis]

    def is_interface_patch(
        patch, block_shape, concatenation_axis, is_second_block=False
    ):
        """Check if patch lies on the concatenation interface."""
        const_dim = patch.const_dim

        if const_dim != concatenation_axis:
            return False  # Not on concatenation face

        # Get absolute limits for the constant dimension
        const_index = patch.ijk_lim_abs[const_dim, 0]  # Since it's constant, start==end

        if is_second_block:
            # For block2: interface is at index 0 of concatenation axis
            return const_index == 0
        else:
            # For block1: interface is at last index of concatenation axis
            return const_index == block_shape[const_dim] - 1

    def adjust_patch_indices(patch, axis_offset, concatenation_axis, block2_shape):
        """Create new patch with adjusted indices for block2."""
        # Get the original limits
        i_lim = patch._ijk_lim[0].copy()
        j_lim = patch._ijk_lim[1].copy()
        k_lim = patch._ijk_lim[2].copy()

        # Convert negative indices to positive for the concatenation axis
        # before applying the offset
        if concatenation_axis == 0:
            # Convert negative i indices to positive using block2 shape
            i_lim = np.where(i_lim < 0, block2_shape[0] + i_lim, i_lim)
            # Then add the axis offset
            i_lim = i_lim + axis_offset
        elif concatenation_axis == 1:
            # Convert negative j indices to positive using block2 shape
            j_lim = np.where(j_lim < 0, block2_shape[1] + j_lim, j_lim)
            # Then add the axis offset
            j_lim = j_lim + axis_offset
        elif concatenation_axis == 2:
            # Convert negative k indices to positive using block2 shape
            k_lim = np.where(k_lim < 0, block2_shape[2] + k_lim, k_lim)
            # Then add the axis offset
            k_lim = k_lim + axis_offset

        new_patch = type(patch)(i=i_lim, j=j_lim, k=k_lim, label=patch.label)

        return new_patch

    # Check for interface patches and raise error if found
    interface_patches_block1 = []
    interface_patches_block2 = []

    for patch in block1.patches:
        if is_interface_patch(patch, block1.shape, axis, is_second_block=False):
            interface_patches_block1.append(patch)

    for patch in block2.patches:
        if is_interface_patch(patch, block2.shape, axis, is_second_block=True):
            interface_patches_block2.append(patch)

    if interface_patches_block1 or interface_patches_block2:
        error_msg = "Cannot concatenate blocks with patches on concatenation interface:"
        if interface_patches_block1:
            error_msg += f"\n  Block1 interface patches: {[str(p) for p in interface_patches_block1]}"
        if interface_patches_block2:
            error_msg += f"\n  Block2 interface patches: {[str(p) for p in interface_patches_block2]}"
        raise ValueError(error_msg)

    # Add patches from block1 (no interface patches present)
    for patch in block1.patches:
        result.patches.append(patch)

    # Add adjusted patches from block2 (no interface patches present)
    for patch in block2.patches:
        adjusted_patch = adjust_patch_indices(patch, axis_offset, axis, block2.shape)
        adjusted_patch.attach_to_block(result)  # Update to new block
        result.patches.append(adjusted_patch)

    return result


def _rotate_meridional(block, rot):
    r"""Rotate a block's meridional momentum :math:`(\rho V_x, \rho V_r)` in place by ``rot``.

    Shared by :func:`resolve_to_interface` and :func:`resolve_from_interface`,
    which differ only in which of the paired matrices from
    :func:`~ember.util.rotation_matrices` they are handed -- the two are each
    other's inverse, so there is nothing else to tell them apart. ``Vt`` is
    untouched.

    Operates directly on the nondimensional momentum components of
    :attr:`~ember.block.Block.conserved_nd`, mirroring
    :meth:`~ember.patch.RevolutionPatch.resolve_to_interface`. A proper
    rotation leaves :math:`V_x^2 + V_r^2` exactly invariant, so density and
    energy need no update -- going via :meth:`~ember.block.Block.set_Vx` and
    :meth:`~ember.block.Block.set_Vr` instead would rebuild energy from
    internal energy plus new kinetic energy on every call, subtracting and
    re-adding terms of very different magnitude for no mathematical reason.

    Parameters
    ----------
    block : Block
        Block whose momentum is rotated.
    rot : Array, shape ``chi.shape + (2, 2)``
        Rotation matrix, from :func:`~ember.util.rotation_matrices`.

    Returns
    -------
    Block
        ``block``, with the momentum components of ``conserved_nd`` updated
        in place.
    """
    cons = block.conserved_nd
    cons[..., 1:3] = util.matvec(rot, cons[..., 1:3])
    block.update_cached_conserved()
    return block


def resolve_to_interface(block, rot_to):
    r"""Convert meridional velocity to interface-aligned velocities.

    Resolves the meridional velocity components :math:`(V_x, V_r)` to
    velocities aligned with an interface: velocity through the interface
    :math:`V_n` and velocity in it :math:`V_s`,

    .. math::

        V_n &= \cos\chi\, V_x + \sin\chi\, V_r \\
        V_s &= -\sin\chi\, V_x + \cos\chi\, V_r

    for the interface angle :math:`\chi` that ``rot_to`` was built from by
    :func:`~ember.util.rotation_matrices`. Inverse of
    :func:`resolve_from_interface`. Applies the rotation directly to the
    nondimensional momentum in :attr:`~ember.block.Block.conserved_nd`,
    the same approach as
    :meth:`~ember.patch.RevolutionPatch.resolve_to_interface` on a
    patch's averaging plane -- see :func:`_rotate_meridional`.

    Parameters
    ----------
    block : Block
        Block containing velocity data to be resolved.
    rot_to : Array, shape ``chi.shape + (2, 2)``
        Rotation matrix, the first of the pair returned by
        :func:`~ember.util.rotation_matrices`. Building it once and reusing
        it across repeated to/from calls at the same angle -- as
        :func:`~ember.average.mix_out` does each Newton iteration -- avoids
        re-deriving the same sine and cosine on every call.

    Returns
    -------
    Block
        The input block with momentum updated to interface-aligned form.
        :math:`V_n` becomes the new Vx, :math:`V_s` becomes the new Vr, Vt
        unchanged.
    """
    return _rotate_meridional(block, rot_to)


def resolve_from_interface(block, rot_from):
    r"""Convert interface-aligned velocities back to meridional components.

    Converts interface-aligned velocities (:math:`V_n` = ``block.Vx`` through
    the interface, :math:`V_s` = ``block.Vr`` in it) back to meridional
    components :math:`(V_x, V_r)`,

    .. math::

        V_x &= \cos\chi\, V_n - \sin\chi\, V_s \\
        V_r &= \sin\chi\, V_n + \cos\chi\, V_s

    for the interface angle :math:`\chi` that ``rot_from`` was built from by
    :func:`~ember.util.rotation_matrices`. Inverse of
    :func:`resolve_to_interface`. Applies the rotation directly to the
    nondimensional momentum in :attr:`~ember.block.Block.conserved_nd`,
    the same approach as
    :meth:`~ember.patch.RevolutionPatch.resolve_from_interface` on a
    patch's averaging plane -- see :func:`_rotate_meridional`.

    Parameters
    ----------
    block : Block
        Block containing interface-aligned velocities (Vn=block.Vx, Vs=block.Vr).
    rot_from : Array, shape ``chi.shape + (2, 2)``
        Rotation matrix, the second of the pair returned by
        :func:`~ember.util.rotation_matrices`.

    Returns
    -------
    Block
        The input block with momentum updated to meridional form.
    """
    return _rotate_meridional(block, rot_from)


def resample(block, factors):
    """Resample 3D block with vectorized interpolation while preserving patch connectivity.

    Creates a new block by resampling with given factor(s). Critical indices from
    patch boundaries are preserved to maintain connectivity. Uses scipy.interpn
    for efficient multi-dimensional interpolation.

    Parameters
    ----------
    block : Block
        Block to resample
    factors : float or tuple of 3 floats
        Resampling factor(s). Values > 1 increase resolution, < 1 decrease resolution.
        If scalar, same factor applied to all three dimensions.

    Returns
    -------
    Block
        New resampled block with updated patch indices and preserved metadata
    """
    # Handle scalar factor
    if not isinstance(factors, (list, tuple)):
        factors = (factors,) * 3

    old_shape = block.shape

    # 1. Collect critical indices using list comprehension
    ijk_crit = [
        np.unique(
            [0, old_shape[dim] - 1]
            + [idx for patch in block.patches for idx in patch.ijk_lim_abs[dim]]
        )
        for dim in range(3)
    ]

    # 2. Resample each direction with mappings
    ijk_new, ijk_mappings = zip(
        *[
            util.resample(factors[dim], np.arange(old_shape[dim]), ijk_crit[dim])
            for dim in range(3)
        ]
    )

    new_shape = tuple(len(coords) for coords in ijk_new)

    # 3. Interpolate all variables in a single Fortran call
    data_new = ember.fortran.map_coordinates_3d(
        block._data,
        ijk_new[0].astype(np.float32),
        ijk_new[1].astype(np.float32),
        ijk_new[2].astype(np.float32),
    )

    # Get all old patch limits before creating new block
    old_patch_limits = []
    for patch in block.patches:
        old_patch_limits.append([patch.ijk_lim_abs[dim].copy() for dim in range(3)])

    # 4. Create new block with independent patch collection
    new_block = block.empty(shape=new_shape)
    new_block._data = data_new

    # empty() shallow-copies _metadata, so new_block.patches is still the original
    # BlockPatchCollection. Replace it with unattached copies; indices will be
    # remapped and patches re-attached in step 5 below.
    copied_patches = [p.copy() for p in block.patches]
    new_patch_collection = ember.collections.BlockPatchCollection(new_block)
    new_patch_collection._items = copied_patches
    new_block._set_metadata_by_key("patches", new_patch_collection)

    # 5. Update all patch indices using absolute limits and mapping
    for old_limits, new_patch in zip(old_patch_limits, new_block.patches):
        new_limits = [
            tuple(
                [
                    ijk_mappings[dim][old_limits[dim][0]],
                    ijk_mappings[dim][old_limits[dim][1]],
                ]
            )
            for dim in range(3)
        ]
        new_patch.set_i_lim(new_limits[0])
        new_patch.set_j_lim(new_limits[1])
        new_patch.set_k_lim(new_limits[2])
        new_patch.attach_to_block(new_block)  # Re-validate with new block

    return new_block


def _patch_crit(block, src):
    """Critical indices per dimension for a block-to-block interpolation.

    A critical index is a location that must land exactly where it started:
    the ends of the block, and every patch boundary. Returned as one
    ``(src, block)`` pair per dimension, which is all
    :func:`_interp_coords` needs -- so the notion of a patch stays here, on
    the side of the interface that has blocks, and never reaches the array
    path.
    """
    crit = []
    for d in range(3):
        crit.append(
            (
                np.unique(
                    [0, src.shape[d] - 1]
                    + [int(idx) for p in src.patches for idx in p.ijk_lim_abs[d]]
                ),
                np.unique(
                    [0, block.shape[d] - 1]
                    + [int(idx) for p in block.patches for idx in p.ijk_lim_abs[d]]
                ),
            )
        )
    return crit


def _interp_coords(block_shape, src_shape, crit=None):
    """Build per-dimension float32 query coordinate arrays for interpolation.

    Between each pair of consecutive critical indices a linspace maps block
    index space into source index space, so those locations land exactly
    where they started and only the spans between them are stretched.

    With no critical indices given, the ends are the only ones: they line up
    and everything between is stretched evenly. That is the right behaviour
    for a bare field, which has no patch layout to align to -- and for a flow
    field, unlike for coordinates, a boundary landing a fraction of a cell out
    is immaterial.

    Parameters
    ----------
    block_shape : tuple
        Shape of the target, whose index space the coordinates are defined over.
    src_shape : tuple
        Shape of the source, in whose index space they are expressed.
    crit : list of tuple, optional
        One ``(src, block)`` pair of critical index arrays per dimension.

    Returns
    -------
    list of Array
        Three float32 arrays, one per dimension, each of length
        ``block_shape[d]``, containing source-index-space coordinates.

    Raises
    ------
    ValueError
        If a dimension's critical-index count differs between source and block.
    """
    coords = []
    for d in range(3):
        if crit is None:
            src_crit = np.array([0, src_shape[d] - 1])
            blk_crit = np.array([0, block_shape[d] - 1])
        else:
            src_crit, blk_crit = crit[d]

        if len(src_crit) != len(blk_crit):
            raise ValueError(
                f"Dimension {d}: src has {len(src_crit)} critical indices "
                f"but block has {len(blk_crit)}"
            )
        segments = []
        for i in range(len(src_crit) - 1):
            n = blk_crit[i + 1] - blk_crit[i] + 1
            seg = np.linspace(
                float(src_crit[i]), float(src_crit[i + 1]), n, dtype=np.float32
            )
            segments.append(seg[:-1])
        segments.append(np.array([src_crit[-1]], dtype=np.float32))
        coords.append(np.concatenate(segments))
    return coords


STATE = ("P", "T", "Vx", "Vr", "Vt", "mu_turb")
"""Quantities transferred by the interpolation functions below.

Primitives, not the conserved variables. Conserved energy is measured from its
fluid's datum, so copying it between blocks whose fluids differ silently
reinterprets it -- a datum 600 K apart turns 400 K into 1000 K, with nothing
raised. Pressure, temperature and velocity are datum-free and cross unchanged,
which is also why interpolating them cannot produce a negative temperature the
way interpolating ``rhoe`` can.
"""


def interp_from_arrays(block, arrays, crit=None):
    """Interpolate a flow field onto ``block`` by index-space trilinear interpolation.

    The caller must have already set the fluid on block.

    Parameters
    ----------
    block : Block
        Target block to receive the field.
    arrays : sequence of Array
        One array per entry of :data:`STATE`, in that order, all of the same
        shape and all dimensional. They need not match ``block``'s shape.
    crit : list of tuple, optional
        One ``(src, block)`` pair of critical index arrays per dimension,
        locations to be held fixed through the mapping. None for a bare
        field, which maps end to end instead. Built from patches by
        :func:`interp_from_grid`; this function has no notion of a patch.

    Raises
    ------
    AssertionError
        If a different-shape interpolation produces values outside the
        source's range -- trilinear interpolation must not create new extrema.
    """
    data_in = np.stack([np.asarray(a, dtype=np.float32) for a in arrays], axis=-1)
    src_shape = data_in.shape[:3]

    logger.debug("interp: src %s -> block %s", src_shape, block.shape)

    if src_shape == tuple(block.shape):
        data_out = data_in
    else:
        coords = _interp_coords(block.shape, src_shape, crit)

        data_out = ember.fortran.map_coordinates_3d(
            data_in, coords[0], coords[1], coords[2]
        )

        # Trilinear interpolation must not create new extrema; allow a small
        # tolerance for float32 rounding relative to the range of each variable.
        lo = data_in.reshape(-1, data_in.shape[-1]).min(axis=0)
        hi = data_in.reshape(-1, data_in.shape[-1]).max(axis=0)
        tol = np.maximum(np.float32(1e-4) * (hi - lo), np.float32(1e-4) * np.abs(hi))
        assert np.all(data_out >= lo - tol) and np.all(data_out <= hi + tol), (
            "Interpolated values exceed source bounds"
        )

    block.set_P_T(data_out[..., 0], data_out[..., 1])
    block.set_Vx(data_out[..., 2])
    block.set_Vr(data_out[..., 3])
    block.set_Vt(data_out[..., 4])
    block.set_mu_turb(data_out[..., 5])

    assert np.all(np.isfinite(block.T)) and np.all(block.T > 0), (
        "Target block has non-finite or non-positive temperatures after interpolation"
    )


def interp_from_grid(block, src):
    """Interpolate the solution on ``src`` onto ``block``.

    A thin unpacking of :func:`interp_from_arrays`: the state is read off
    ``src`` dimensionally, so the two blocks may carry different fluids --
    different reference scales, and different entropy and energy datums --
    without any conversion being needed.

    Parameters
    ----------
    block : Block
        Target block to receive the interpolated solution.
    src : Block
        Source block providing the solution.
    """
    interp_from_arrays(
        block,
        [getattr(src, name) for name in STATE],
        crit=_patch_crit(block, src),
    )


def wall_yplus(block):
    """y+ on all six wall-adjacent boundary faces of ``block``.

    Post-processing only -- NOT part of the per-step viscous kernel.
    ``wall_yplus_field`` (``_fortran/viscous.f90``) reuses the exact Re/cf/d
    that ``set_visc_force``'s own wall function uses (both call the shared
    ``wall_core``), so this cannot silently drift from what the solver
    actually modeled at a face; it carries none of ``set_visc_force``'s
    k-slab/rolling-buffer machinery since it costs O(surface) per call, not
    O(volume) per step.

    The height is the wall-adjacent cell thickness ``vol/|dA|``, i.e. the
    distance to the first off-wall *node* -- the point whose velocity the wall
    function actually samples in this cell-vertex scheme, so this is y+ where
    the closure is evaluated. Cell-centred codes report y+ at the first cell
    centroid instead, which is the usual "first cell y+" of mesh-sizing
    guidance; **halve** these values for that convention, since y+ is linear
    in wall distance at fixed friction velocity. That is not a conversion for
    comparing against another code's number on the same mesh, though: a
    cell-centred solver samples its velocity at the half height too, so it
    infers its own friction velocity and does not simply see half of this.

    Parameters
    ----------
    block : Block

    Returns
    -------
    dict[str, numpy.ndarray]
        Keys ``yplus_i1``, ``yplus_j1``, ``yplus_k1``, ``yplus_ni``,
        ``yplus_nj``, ``yplus_nk``, each shaped like the corresponding
        :attr:`~ember.block.Block.ijk_wall_visc` face array, zero on
        non-wall cells.
    """
    keys = ("yplus_i1", "yplus_j1", "yplus_k1", "yplus_ni", "yplus_nj", "yplus_nk")
    result = ember.fortran.wall_yplus_field(
        cons=block.conserved_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        omega_block=block.Omega_nd,
        r=block.r_nd,
        mu=block.mu_nd,
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        **block.ijk_wall_visc,
        **block.Omega_wall_nd,
    )
    return dict(zip(keys, result))


def to_tm3(block, filename, clip_quantile=0.01, **kwargs):
    """Write a triangulated cut to a tm3 binary file.

    Parameters
    ----------
    block : Block
        Triangulated block with shape ``(ntri, 3)``.
    filename : str or path-like
        Output file path.
    clip_quantile : float, optional
        Quantile used to clip the property colour range. The range is set
        to [q, 1-q] percentiles so that extreme outliers do not dominate
        the colour scale. Default is 0.01 (1%). Set to 0 to use the
        full min/max range.
    **kwargs : array_like, shape (ntri, 3)
        Exactly one keyword argument: the key is used as the property name
        in the file, the value is the per-vertex scalar array.

    Raises
    ------
    ValueError
        If the block is not triangulated, kwargs count != 1, or the value
        shape does not match the block shape.
    """
    if not block.triangulated:
        raise ValueError("to_tm3 requires a triangulated block")
    if len(block.shape) != 2 or block.shape[1] != 3:
        raise ValueError(f"to_tm3 requires shape (ntri, 3), got {block.shape}")
    if len(kwargs) != 1:
        raise ValueError(
            f"to_tm3 requires exactly one keyword argument (property name=array), "
            f"got {len(kwargs)}"
        )

    prop_name, values = next(iter(kwargs.items()))
    values = np.asarray(values)
    if values.shape != block.shape:
        raise ValueError(
            f"Property array shape {values.shape} does not match block shape {block.shape}"
        )

    ntri = block.shape[0]
    nverts = ntri * 3

    # Cartesian vertex coordinates, shape (ntri, 3, 3): [tri, node, xyz]
    xyz = np.stack([block.x, block.y, block.z], axis=-1)  # (ntri, 3, 3)
    nodes = xyz.reshape(-1, 3).astype(np.float32)  # (nverts, 3)

    # Each triangle's vertices are consecutive — no shared vertices
    indices = np.arange(nverts, dtype=np.int32).reshape(ntri, 3)

    # Bounding geometry
    rmax = np.sqrt((nodes**2).sum(axis=1)).max()
    xrange_ = np.array([nodes[:, 0].min(), nodes[:, 0].max()], dtype=np.float32)
    yrange_ = np.array([nodes[:, 1].min(), nodes[:, 1].max()], dtype=np.float32)
    zrange_ = np.array([nodes[:, 2].min(), nodes[:, 2].max()], dtype=np.float32)

    values_flat = values.ravel().astype(np.float32)
    q_lo = float(np.percentile(values_flat, 100.0 * clip_quantile))
    q_hi = float(np.percentile(values_flat, 100.0 * (1.0 - clip_quantile)))
    prange_ = np.array([q_lo, q_hi], dtype=np.float32)

    surface_name = block.label if block.label is not None else "surface"

    with open(filename, "wb") as f:
        f.write(np.int32(1).tobytes())  # nsteps
        f.write(np.int32(1).tobytes())  # nsurfaces
        f.write(np.array(surface_name, dtype="S96").tobytes())  # surface name
        f.write(np.int32(nverts).tobytes())  # nverts
        f.write(np.int32(ntri).tobytes())  # ntris
        f.write(np.int32(1).tobytes())  # nprops
        f.write(np.float32(rmax).tobytes())  # rmax
        f.write(xrange_.tobytes())  # xrange
        f.write(yrange_.tobytes())  # yrange
        f.write(zrange_.tobytes())  # zrange
        f.write(nodes.ravel().tobytes())  # vertices
        f.write(indices.ravel().tobytes())  # triangle indices
        f.write(np.array(prop_name, dtype="S96").tobytes())  # prop name
        f.write(prange_.tobytes())  # prange
        f.write(values_flat.tobytes())  # property values
