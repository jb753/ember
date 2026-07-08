"""Plot3D multi-block grid format I/O utilities.

This module provides read and write functionality for the Plot3D multi-block structured
grid format, a widely-used NASA standard for storing grid coordinates in CFD applications.
The module handles conversion between Ember's Grid objects and Plot3D ASCII format files,
with automatic handling of coordinate ordering transformations between Plot3D's (k,j,i)
convention and Ember's (i,j,k) convention. It also includes support for FVBND boundary
condition files which specify patch types and locations on each block. The implementation
includes optional k-axis flipping for proper volume orientation in external tools like
Pointwise, and supports reading and writing multiple blocks with arbitrary grid dimensions.
"""

import numpy as np
from collections import defaultdict
from ember.patch import InletPatch, OutletPatch, MixingPatch, PeriodicPatch


def _parse_boundary_line(label, line):
    """Parse a single boundary line into a Patch object and grid number."""
    parts = line.split()
    if len(parts) != 10:
        raise ValueError(f"Invalid boundary line format: {line}")

    # Parse line: type grid-number I-min I-max J-min J-max K-min K-max results_flag norm_dir
    grid_number = int(parts[1]) - 1  # Convert to 0-based
    i_min, i_max = int(parts[2]) - 1, int(parts[3]) - 1  # Convert to 0-based
    j_min, j_max = int(parts[4]) - 1, int(parts[5]) - 1
    k_min, k_max = int(parts[6]) - 1, int(parts[7]) - 1

    # Determine patch class from label
    if "inlet" in label:
        patch_class = InletPatch
    elif "outlet" in label:
        patch_class = OutletPatch
    elif "mixing" in label:
        patch_class = MixingPatch
    elif "periodic" in label or "interface" in label:
        patch_class = PeriodicPatch
    else:
        raise ValueError(f"Cannot determine patch type from label: {label}")

    patch = patch_class(
        i=(i_min, i_max), j=(j_min, j_max), k=(k_min, k_max), label=label
    )
    return grid_number, patch


def read_fvbnd(fname):
    """Load patches from an FVBND file.

    Parameters
    ----------
    fname : str
        Path to the FVBND file

    Returns
    -------
    Dict[int, List[Patch]]
        Mapping of block id to list of patches for that block
    """

    with open(fname, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Validate header
    if not lines[0].startswith("FVBND"):
        raise ValueError(f"Unexpected header line: {lines[0]}")

    # Find BOUNDARIES section
    try:
        boundaries_idx = lines.index("BOUNDARIES") + 1
    except ValueError:
        raise ValueError("BOUNDARIES section not found")

    # Extract labels and boundaries
    labels = lines[1 : boundaries_idx - 1]
    boundaries = lines[boundaries_idx:]

    # Validate matching counts
    if len(labels) != len(boundaries):
        raise ValueError(
            f"Mismatch: {len(labels)} labels but {len(boundaries)} boundaries"
        )

    # Parse patches and build mapping
    grid_patches = defaultdict(list)
    for label, line in zip(labels, boundaries):
        bid, patch = _parse_boundary_line(label, line)
        grid_patches[bid].append(patch)

    return dict(grid_patches)


def read_plot3d(filename, flip_k=True):
    """Read grid from Plot3D format file.

    Parameters
    ----------
    filename : str
        Input Plot3D grid file to read
    flip_k : bool, optional
        Whether to flip the k-axis to match write_plot3d behavior (default True)

    Returns
    -------
    Grid
        New grid containing blocks with coordinates from Plot3D file

    Raises
    ------
    FileNotFoundError
        If input file does not exist
    ValueError
        If file format is invalid
    """
    from ember.grid import Grid
    from ember.block import Block

    # Read all file content as numpy array
    with open(filename, "r") as f:
        content = f.read()

    lines = content.strip().split("\n")
    if len(lines) < 1:
        raise ValueError("Plot3D file is empty")

    # Read number of blocks
    try:
        nb = int(lines[0])
    except ValueError:
        raise ValueError("Cannot read number of blocks from Plot3D file")

    if nb <= 0:
        raise ValueError(f"Invalid number of blocks: {nb}")

    if len(lines) < nb + 1:
        raise ValueError("File too short for specified number of blocks")

    # Read block dimensions using numpy
    block_shapes = []
    for i in range(1, nb + 1):
        try:
            dims = np.fromstring(lines[i], sep=" ", dtype=int)
            if len(dims) != 3:
                raise ValueError(
                    f"Block {i} dimensions must have 3 values, got {len(dims)}"
                )
            block_shapes.append(tuple(dims))
        except ValueError as e:
            raise ValueError(f"Cannot read dimensions for block {i}: {e}")

    # Parse coordinate data using numpy
    coord_text = " ".join(lines[nb + 1 :])
    coord_data = np.fromstring(coord_text, sep=" ", dtype=np.float64)

    # Verify coordinate count
    expected_count = sum(ni * nj * nk * 3 for ni, nj, nk in block_shapes)
    if len(coord_data) != expected_count:
        raise ValueError(
            f"Expected {expected_count} coordinates, got {len(coord_data)}"
        )

    # Create blocks and reconstruct coordinates
    blocks = []
    coord_idx = 0

    for ni, nj, nk in block_shapes:
        n_nodes = ni * nj * nk

        # Extract coordinates for this block in Plot3D order: all X, then all Y, then all Z
        block_coords = coord_data[coord_idx : coord_idx + 3 * n_nodes].reshape(
            3, n_nodes
        )
        coord_idx += 3 * n_nodes

        # Reshape each coordinate array from Plot3D order (k,j,i) to ember order (i,j,k)
        x = block_coords[0].reshape(nk, nj, ni).transpose(2, 1, 0)
        y = block_coords[1].reshape(nk, nj, ni).transpose(2, 1, 0)
        z = block_coords[2].reshape(nk, nj, ni).transpose(2, 1, 0)

        # Stack into xyz array
        xyz = np.stack((x, y, z), axis=-1)

        # Apply k-flip to match write_plot3d behavior
        if flip_k:
            xyz = np.flip(xyz, axis=2)

        # Create block and set coordinates
        block = Block(shape=(ni, nj, nk))
        block.set_xyz(xyz)
        blocks.append(block)

    return Grid(blocks)


def write_plot3d(grid, filename, flip_k=True):
    """Write grid in Plot3D format.

    Parameters
    ----------
    grid : Grid
        Grid object to write
    filename : str
        Output filename for Plot3D grid file
    flip_k : bool, optional
        Whether to flip the k-axis for proper volume orientation in Pointwise (default True)

    Raises
    ------
    ValueError
        If grid is empty (contains no blocks)
    """
    if len(grid) == 0:
        raise ValueError("Cannot write Plot3D file: grid contains no blocks")

    with open(filename, "w") as f:
        # Number of blocks
        nb = len(grid)
        f.write(f"{nb}\n")

        # Size of all blocks
        for block in grid:
            ni, nj, nk = block.shape
            f.write(f"{ni} {nj} {nk}\n")

        # Write block coordinates
        for block in grid:
            xyz = np.stack([block.x, block.y, block.z], axis=-1)

            # Flip k-axis if requested for proper volume orientation
            if flip_k:
                xyz = np.flip(xyz, axis=2)

            # Reshape to Plot3D format following NASA specification:
            # Plot3D expects ALL x-coords, then ALL y-coords, then ALL z-coords
            # with loop order: for k (for j (for i))
            xyz_transposed = xyz.transpose(2, 1, 0, 3)  # (nk, nj, ni, 3)

            # Flatten each coordinate component separately
            x_flat = xyz_transposed[..., 0].flatten()
            y_flat = xyz_transposed[..., 1].flatten()
            z_flat = xyz_transposed[..., 2].flatten()

            # Write coordinates in Plot3D order: all X, then all Y, then all Z
            for coords in [x_flat, y_flat, z_flat]:
                np.savetxt(f, coords, newline=" ", fmt="%.12f")


def _patch_to_line(patch, block_id, patch_index):
    """Convert a patch to FVBND boundary line format."""
    # Get patch limits (ember uses 0-based, FVBND uses 1-based)
    i_min, i_max = patch.ist + 1, patch.ien + 1
    j_min, j_max = patch.jst + 1, patch.jen + 1
    k_min, k_max = patch.kst + 1, patch.ken + 1

    results_flag = "F"

    # Determine normal direction based on constant dimension
    const_dim = patch.const_dim
    if const_dim == 0:  # i-constant
        norm_dir = 0 if patch.ist == 0 else -1
    elif const_dim == 1:  # j-constant
        norm_dir = 0 if patch.jst == 0 else -1
    else:  # k-constant
        norm_dir = 0 if patch.kst == 0 else -1

    return f"{patch_index + 1} {block_id + 1} {i_min} {i_max} {j_min} {j_max} {k_min} {k_max} {results_flag} {norm_dir}"


def write_fvbnd(grid, filename, iregion=0):
    """Write patches to FVBND format file.

    Parameters
    ----------
    grid : Grid
        Grid object containing blocks with patches
    filename : str
        Output filename for FVBND file
    iregion : int, optional
        Region number for labeling (default 0)
    """
    patches = {}

    # Counters for different patch types
    inlet_count = 0
    outlet_count = 0
    mixing_count = 0
    periodic_count = 0

    # Loop over blocks to get block indexes
    for block_id, block in enumerate(grid):
        # Loop over patches on each block
        for patch in block.patches:
            # Determine patch type and create label
            if isinstance(patch, InletPatch):
                label = f"region_{iregion}_inlet_{inlet_count}"
                inlet_count += 1
            elif isinstance(patch, OutletPatch):
                label = f"region_{iregion}_outlet_{outlet_count}"
                outlet_count += 1
            elif isinstance(patch, MixingPatch):
                label = f"region_{iregion}_mixing_{mixing_count}"
                mixing_count += 1
            elif isinstance(patch, PeriodicPatch):
                label = f"region_{iregion}_periodic_{periodic_count}"
                periodic_count += 1
            else:
                continue  # Skip unknown patch types

            patches[label] = (patch, block_id)

    # Write file
    with open(filename, "w") as f:
        # Header
        f.write("FVBND 1 4\n")

        # Labels
        for label in patches.keys():
            f.write(f"{label}\n")

        # BOUNDARIES section
        f.write("BOUNDARIES\n")

        # Write patch lines
        for patch_index, (label, (patch, block_id)) in enumerate(patches.items()):
            f.write(_patch_to_line(patch, block_id, patch_index) + "\n")
