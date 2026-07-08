"""Utility functions for Block operations including concatenation, resampling, and I/O."""

import copy as _copy_module
import logging
import sys

import numpy as np

import ember.collections
from ember import util

logger = logging.getLogger(__name__)


def _copy_patch(p):
    if hasattr(p, "copy"):
        return p.copy()
    return _copy_module.deepcopy(p)


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
    from ember.block import Block

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
    import ember.fortran

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
    copied_patches = [_copy_patch(p) for p in block.patches]
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


def _interp_coords(block, src):
    """Build per-dimension float32 query coordinate arrays for interp_from.

    For each dimension, critical indices (patch boundaries + endpoints) are
    collected from both src and block. The number of critical indices must
    match. Between each pair of consecutive critical indices a linspace maps
    block index space into src index space, preserving the critical locations
    exactly.

    Returns a list of three float32 arrays, one per dimension, each of length
    block.shape[d], containing src-index-space coordinates.
    """
    coords = []
    for d in range(3):
        src_crit = np.unique(
            [0, src.shape[d] - 1]
            + [int(idx) for p in src.patches for idx in p.ijk_lim_abs[d]]
        )
        blk_crit = np.unique(
            [0, block.shape[d] - 1]
            + [int(idx) for p in block.patches for idx in p.ijk_lim_abs[d]]
        )
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


def interp_from_conserved(block, conserved):
    if block.shape == conserved.shape[:-1]:
        block.set_conserved(conserved)
    else:
        import ember.fortran

        coords = [
            np.linspace(0, conserved.shape[d] - 1, block.shape[d], dtype=np.float32)
            for d in range(3)
        ]
        data_out = ember.fortran.map_coordinates_3d(conserved, *coords)
        block.set_conserved(data_out)


def interp_from(block, src):
    """Interpolate solution from src onto block by index-space trilinear interpolation.

    The caller must have already set the fluid on block. All quantities are
    handled in dimensional form so that differing reference scales between
    block and src are handled correctly.

    Parameters
    ----------
    block : Block
        Target block to receive the interpolated solution.
    src : Block
        Source block providing the solution.
    """

    src_has_cfl = "cfl" in src.working
    logger.debug(
        "interp_from: src %s -> block %s; src has cfl: %s",
        src.shape,
        block.shape,
        src_has_cfl,
    )

    if block.shape == src.shape:
        block.set_conserved(src.conserved)
        block.set_mu_turb(src.mu_turb)
        if src_has_cfl:
            logger.debug("interp_from: shapes match, copying cfl directly")
            block.working.cfl[...] = src.working.cfl
        else:
            logger.debug("interp_from: shapes match, no cfl on src — skipping cfl copy")
    else:
        import ember.fortran

        data_in = np.concatenate(
            [src.conserved, src.mu_turb[..., np.newaxis]], axis=-1
        ).astype(np.float32)

        coords = _interp_coords(block, src)

        data_out = ember.fortran.map_coordinates_3d(
            data_in, coords[0], coords[1], coords[2]
        )

        # Trilinear interpolation must not create new extrema; allow a small
        # tolerance for float32 rounding relative to the range of each variable.
        lo = data_in.reshape(-1, data_in.shape[-1]).min(axis=0)
        hi = data_in.reshape(-1, data_in.shape[-1]).max(axis=0)
        tol = np.maximum(np.float32(1e-4) * (hi - lo), np.float32(1e-4) * np.abs(hi))
        assert np.all(data_out >= lo - tol) and np.all(data_out <= hi + tol), (
            "Interpolated conserved variables exceed source bounds"
        )

        block.set_conserved(data_out[..., :5])
        block.set_mu_turb(data_out[..., 5])

        if src_has_cfl:
            # cfl is cell-centred; convert node coords to cell-centre coords
            # by averaging adjacent node pairs (dropping the last node per dim).
            logger.debug(
                "interp_from: shapes differ, interpolating cfl from src using cell-centre coords"
            )
            cell_coords = [(c[:-1] + c[1:]) * np.float32(0.5) for c in coords]
            block.working.cfl[...] = ember.fortran.map_coordinates_3d(
                src.working.cfl.astype(np.float32),
                cell_coords[0],
                cell_coords[1],
                cell_coords[2],
            )
        else:
            logger.debug(
                "interp_from: shapes differ, no cfl on src — skipping cfl interpolation"
            )

    assert np.all(np.isfinite(block.T)) and np.all(block.T > 0), (
        "Target block has non-finite or non-positive temperatures after interpolation"
    )


def memory_usage(block):
    """Return memory usage of a block's data, metadata, and cached properties.

    Parameters
    ----------
    block : Block
        The block to measure.

    Returns
    -------
    data_usage : dict
        Bytes per data key (equal share of the contiguous _data array).
    metadata_usage : dict
        Bytes per metadata key (nbytes for arrays, sys.getsizeof for others).
    cache_usage : dict
        Bytes per cached property in _store (nbytes for arrays, sys.getsizeof for others).
    """
    # Data: each field occupies 1/nvar of the contiguous array
    bytes_per_field = block._data.nbytes // block.nvar
    data_usage = {key: bytes_per_field for key in block._data_keys}

    # Metadata
    metadata_usage = {}
    for key, val in block._metadata.items():
        if hasattr(val, "nbytes"):
            metadata_usage[key] = val.nbytes
        else:
            metadata_usage[key] = sys.getsizeof(val)

    # Cached properties in _store: tuple (version, result) entries from cached_array.
    # Working arrays in block.working._store: bare result objects.
    cache_usage = {}
    for key, entry in block._store.items():
        result = entry[1]
        if hasattr(result, "nbytes"):
            cache_usage[key] = result.nbytes
        else:
            cache_usage[key] = sys.getsizeof(result)
    for key, entry in block.working._store.items():
        if hasattr(entry, "nbytes"):
            cache_usage[key] = entry.nbytes
        else:
            cache_usage[key] = sys.getsizeof(entry)

    return data_usage, metadata_usage, cache_usage


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
