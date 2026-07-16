"""Collection of connected blocks forming a complete flow domain.

This module defines the :class:`Grid`, the top-level container for a
multi-block structured simulation. A grid is an ordered collection of
:class:`~ember.block.Block` objects together with the topology that connects
them. A :class:`Grid` stores no flow field of its own: every coordinate and
conserved quantity lives on the constituent blocks.
Therefore the solution is read from one block at a time as in ``grid[0].P``.


Construction and labelled access
================================

Like a list, blocks can be added to a grid at construction or a later time::

    from ember.block import Block
    from ember.grid import Grid

    grid = Grid([Block(shape=(10, 11, 10))])
    rotor = Block(shape=(20, 20, 20))
    rotor.set_label("rotor")
    grid.append(rotor)

The grid then behaves as a standard Python collection: it supports iteration,
:func:`len`, membership testing, and the usual mutating operations
(:meth:`Grid.append`, :meth:`Grid.extend`, :meth:`Grid.insert`,
:meth:`Grid.remove`, :meth:`Grid.pop`, :meth:`Grid.clear`). Indexing accepts
either an integer position or a label string, and membership testing accepts
either a block or a label. :attr:`Grid.labels` lists the labels in order, with
``None`` for any unlabelled block::

    len(grid)                 # 2
    grid[1] is grid["rotor"]  # True -- refers to same block
    "rotor" in grid           # True -- membership by label
    rotor in grid             # True -- membership by block
    grid.labels               # [None, 'rotor'] -- None if unlabelled


Connectivity
============

What distinguishes a :class:`Grid` from a plain list of blocks is the topology it
derives from the blocks' boundary patches, as found in :attr:`~ember.block.Block.patches`.

:attr:`Grid.patches` presents every patch on every block as one flat, read-only
sequence, filterable by patch type (``grid.patches.inlet``,
``grid.patches.periodic``, and so on). It is a view: patches are still owned by
the block they sit on, and are added and removed there.

:attr:`Grid.connectivity` manages communicators that exchange data across
the seams between blocks, one per patch type, reached as
``grid.connectivity.periodic`` and likewise ``mixing``, ``nonmatch``, ``cusp``.

A communicator pairs its patches -- matching each to its partner on a
neighbouring block -- the first time it is used, and caches the result for subsequent usages. Pairing
is therefore automatic, and driven by the exchange itself, e.g. a call to
:meth:`Grid.apply_bconds`.

Changing grid topology -- adding or removing a block or a patch -- may break the
indexing describing pairing, and unfortunately the cache does not detect this.
In these situations, the cache must be flushed by hand::

    grid.append(another_block)
    grid.connectivity.clear()  # drop the stale pairs

The next communicator exchange will then pair the new topology from scratch.

The pairings can also be inspected directly, by calling ``pair()`` on the whole
manager or on one patch type. It returns a dict keyed by the ``(bid, pid)``
identifier of each patch, indexing like ``grid[bid].patches[pid]``.
The  values are the corresponding ``(bid, pid)`` of the patch it matches
and the geometric transform between the two. Both halves of a pair appear as
keys, so the mapping can be followed from either side::

    pairs = grid.connectivity.periodic.pair()
    # block 0 patch 0 is paired with block 1 patch 0, and vice versa
    pairs[(0, 0)]  # ((1, 0), transform)
    pairs[(1, 0)]  # ((0, 0), transform)

Blocks joined to one another by periodic patches make up a single blade row.
Rows are separated from one another by mixing patches, and
:attr:`Grid.rows` groups the blocks accordingly, ordering the rows from inlet to
outlet; :attr:`Grid.n_row` is their count. The first row's upstream face is the
domain inlet and the last row's downstream face is the domain outlet. Both
properties pair the periodic patches for themselves.

Driving a solver
================

Many of the grid methods, such as :meth:`Grid.update_residual` and
:meth:`Grid.apply_bconds`, form the inner loop of a time-marching solver. They
are documented in :mod:`ember.solver`, and should be used with care.

During time marching,
:meth:`Grid.get_convergence` returns a :class:`ConvergenceStep` of
residual and station monitors for the current step, which
:class:`~ember.convergence_history.ConvergenceHistory` accumulates into a time
series. :meth:`Grid.check_nan` raises :class:`DivergenceError` if the flow field
has blown up.

File formats
============

A grid can be read from and written to three formats. The reading methods are
constructors, returning a new :class:`Grid`.

* EMB -- :meth:`Grid.read_emb`, :meth:`Grid.write_emb`. Our native format: a pickle of the grid with its blocks, patches, and  labels, optionally gzip-compressed. Being a pickle of the objects themselves, it is the format   that preserves a grid most completely.

* Plot3D -- :meth:`Grid.read_plot3d`, :meth:`Grid.write_plot3d`. The standard multi-block structured interchange format, carrying coordinates only. Boundary patches are stored alongside it in a separate FieldView  boundary file, which may be read and written with the Plot3D file  or on its own via :meth:`Grid.write_fvbnd`.

* TS3 -- :meth:`Grid.read_ts3`, :meth:`Grid.write_ts3`. The HDF5-based format of the Turbostream 3 solver, carrying geometry  and flow field.
"""

from ember.collections import _LabelledList, GridPatchCollection
from pykdtree.kdtree import KDTree
import ember.block_util
import ember.fortran
from ember.block_restart import apply_restart
import numpy as np
import itertools
import pickle
import gzip
from dataclasses import dataclass
from ember import util
import ember.block
from ember.patch import RotatingPatch
import ember.periodic_communicator
import ember.mixing_communicator
import ember.nonmatch_communicator


# k-slab depth for the tiled kernels (set_visc_force, set_residual): cell
# planes per slab, so that a slab's input planes stay cache-resident across
# all three face directions. Clamped per block to nk-1. Value chosen by
# benchmark sweep (see docs/dev/viscous_kernels.md).
_KB_SLAB = 8


class Grid(_LabelledList):
    """An ordered, labelled collection of connected blocks.

    See the :mod:`ember.grid` module documentation for the collection
    interface, the topology derived from block patches, and the file formats.
    """

    def __init__(self, blocks=None):
        """Initialize grid with optional list of blocks.

        Parameters
        ----------
        blocks : list, optional
            Initial list of blocks to add to the grid.
        """
        # Import here to avoid circular imports
        from ember import block

        super().__init__(blocks, item_class=block.Block)

        self.config = None
        self._connectivity = None

    def __getstate__(self):
        """Drop the cached connectivity manager before pickling.

        The manager holds a back-reference to this grid and large precomputed
        communicator index arrays. Excluding it keeps EMB files lean; it is
        rebuilt lazily on first access after unpickling.
        """
        state = self.__dict__.copy()
        state["_connectivity"] = None
        return state

    def __setstate__(self, state):
        """Restore state, leaving connectivity to rebuild lazily."""
        self.__dict__.update(state)
        self._connectivity = None

    def __repr__(self):
        """String representation of the grid."""
        labels = self.labels
        return f"Grid(blocks={len(self)}, labels={labels})"

    def _find_periodic_connected_components(self):
        """Find connected components of blocks linked by periodic patches.

        Blocks connected by periodic patches are in the same blade row.
        Uses BFS to find connected components in the periodic connectivity graph.

        Returns
        -------
        List[List[int]]
            List of row groups, where each group is a list of block indices.
            Blocks within a group are connected by periodic patches.
        """
        if len(self) == 0:
            return []

        # Get periodic connectivity
        try:
            periodic_conn = self.connectivity.periodic.pair()
        except ValueError:
            # If periodic pairing fails (unmatched patches), treat each block as separate
            periodic_conn = {}

        # Build adjacency list for block-to-block connections via periodic patches
        adjacency = {bid: set() for bid in range(len(self))}
        for (bid, pid), ((match_bid, _), transform) in periodic_conn.items():
            adjacency[bid].add(match_bid)
            adjacency[match_bid].add(bid)

        # Find connected components using BFS
        visited = set()
        components = []

        for start_bid in range(len(self)):
            if start_bid in visited:
                continue

            # BFS to find all blocks in this component
            component = []
            queue = [start_bid]
            visited.add(start_bid)

            while queue:
                bid = queue.pop(0)
                component.append(bid)

                # Add neighbors
                for neighbor_bid in adjacency[bid]:
                    if neighbor_bid not in visited:
                        visited.add(neighbor_bid)
                        queue.append(neighbor_bid)

            components.append(component)

        return components

    def _order_row_groups(self, row_groups):
        """Order row groups from inlet to outlet using mixing patch connectivity.

        Parameters
        ----------
        row_groups : List[List[int]]
            List of row groups (each group is list of block indices)

        Returns
        -------
        List[List[int]]
            Ordered list of row groups from inlet to outlet
        """
        if len(row_groups) <= 1:
            return row_groups

        # Get mixing connectivity
        try:
            mixing_conn = self.connectivity.mixing.pair()
        except ValueError:
            # No mixing patches or unmatched - return groups as-is
            mixing_conn = {}

        # Build mapping from block ID to row group index
        bid_to_group = {}
        for group_idx, group in enumerate(row_groups):
            for bid in group:
                bid_to_group[bid] = group_idx

        # Build adjacency for row groups (which groups are connected by mixing patches)
        group_adjacency = {i: set() for i in range(len(row_groups))}
        for (bid, pid), ((match_bid, _), transform) in mixing_conn.items():
            group1 = bid_to_group[bid]
            group2 = bid_to_group[match_bid]
            if group1 != group2:
                group_adjacency[group1].add(group2)
                group_adjacency[group2].add(group1)

        # Find inlet row (contains inlet patches)
        inlet_row_idx = None
        for group_idx, group in enumerate(row_groups):
            for bid in group:
                if len(self[bid].patches.inlet) > 0:
                    inlet_row_idx = group_idx
                    break
            if inlet_row_idx is not None:
                break

        # Find outlet row (contains outlet patches)
        outlet_row_idx = None
        for group_idx, group in enumerate(row_groups):
            for bid in group:
                if len(self[bid].patches.outlet) > 0:
                    outlet_row_idx = group_idx
                    break
            if outlet_row_idx is not None:
                break

        # If no inlet or outlet found, return groups as-is
        if inlet_row_idx is None or outlet_row_idx is None:
            return row_groups

        # BFS from inlet to outlet to order rows
        ordered = []
        visited = set()
        queue = [inlet_row_idx]
        visited.add(inlet_row_idx)

        while queue:
            group_idx = queue.pop(0)
            ordered.append(row_groups[group_idx])

            # Add neighbors that haven't been visited
            for neighbor_idx in sorted(group_adjacency[group_idx]):
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

        # Add any unvisited groups at the end (orphan blocks)
        for group_idx, group in enumerate(row_groups):
            if group_idx not in visited:
                ordered.append(group)

        return ordered

    def _align_cartesian(self, xyz):
        """Align Cartesian coordinates with grid by detecting coordinate transformation.

        Parameters
        ----------
        xyz : array_like, shape (N, 3)
            Cartesian coordinates [x, y, z] where N == Grid.size

        Returns
        -------
        - perm: coordinate permutation tuple
        - signs: coordinate signs tuple
        - block_indices: list of index arrays, one per block with shape block.shape
        """
        xyz = np.asarray(xyz)
        assert xyz.shape[0] == self.size, (
            f"Input has {xyz.shape[0]} points, grid has {self.size}"
        )
        assert xyz.shape[1] == 3, "xyz must have 3 columns"

        # Phase 1: Coarse alignment using bounding boxes
        bbox_input = util.bounding_box(xyz)
        # Reduce each block to its own eight corners before taking the bounding
        # box of those, rather than materialising every node of the grid at
        # once: the corners already carry the per-axis extremes, so the box of
        # the corners is the box of the nodes.
        bbox_grid = util.bounding_box(
            np.concatenate(
                [
                    util.bounding_box(
                        np.stack([b.x.ravel(), b.y.ravel(), b.z.ravel()], axis=-1)
                    )
                    for b in self
                ]
            )
        )

        # Test all combinations on bounding box vertices
        transform_errors = []
        for perm in itertools.permutations([0, 1, 2]):
            for sign in itertools.product([-1, 1], repeat=3):
                signs = np.array(sign).reshape(3, 1)

                # Apply transformation to input bbox
                bbox_transformed = bbox_input[:, perm] * signs.T

                # Calculate total squared distance
                total_error = sum(
                    np.min(np.sum((bbox_grid - vertex) ** 2, axis=1))
                    for vertex in bbox_transformed
                )

                transform_errors.append((total_error, perm, signs))

        # Sort by total error
        transform_errors.sort(key=lambda x: x[0])

        # Phase 2: Precise alignment with fallback
        atol = 1e-6 * np.max(np.ptp(bbox_grid, axis=0))

        for _, perm, signs in transform_errors:
            # Build KDTree from transformed input coordinates
            kdtree = KDTree(xyz[:, perm] * signs.T)

            # Try to map all blocks
            block_indices = []
            mapping_failed = False

            for block in self:
                flat = block.flat()
                distances, indices = kdtree.query(
                    np.stack([flat.x, flat.y, flat.z], axis=-1),
                    distance_upper_bound=atol,
                )

                if not (distances <= atol).all():
                    mapping_failed = True
                    break

                block_indices.append(indices.reshape(block.shape))

            if not mapping_failed:
                return tuple(perm), tuple(signs.ravel()), block_indices

        raise ValueError("No valid coordinate transformation found within tolerance")

    def _station_stats(self, indices):
        """Mass flow and mass-averaged ho/s over one through-flow station.

        ``indices`` is a list of ``(bid, pid)`` patch identifiers naming the
        faces that make up the station (e.g. an inlet split across blocks, or
        the two-or-more mixing patches of one side of a mixing plane). All
        three returns are non-dimensionalised by the fluid reference scales, the
        same convention as ``Block.residual_nd``: mdot by the mass-flux scale
        ``rho_ref * V_ref * L_ref**2``, ho by ``u_ref``, s by ``Rgas_ref``.
        """
        import ember.average

        mdot = ho_num = s_num = 0.0
        for bid, pid in indices:
            p = self[bid].patches[pid]
            cut = p.block_view.squeeze()
            m = ember.average.flow_mass(cut) * p.block.Nb
            mdot += m
            ho_num += m * ember.average.mass_average(cut.ho, cut)
            s_num += m * ember.average.mass_average(cut.s, cut)
        blk = self[indices[0][0]]
        fl = blk.fluid
        mdot_ref = fl.rhoV_ref * blk.L_ref**2
        return mdot / mdot_ref, ho_num / mdot / fl.u_ref, s_num / mdot / fl.Rgas_ref

    @classmethod
    def read_emb(cls, filename):
        """Read grid from EMB binary format file.

        Automatically detects and handles both uncompressed and gzip-compressed EMB files.

        Parameters
        ----------
        filename : str
            Input EMB file to read

        Returns
        -------
        Grid
            New grid containing all blocks, patches, flow data, and metadata from EMB file

        Raises
        ------
        FileNotFoundError
            If file does not exist
        """
        # Try gzip first, fall back to uncompressed
        try:
            with gzip.open(filename, "rb") as f:
                grid = pickle.load(f)
        except gzip.BadGzipFile:
            with open(filename, "rb") as f:
                grid = pickle.load(f)

        # Re-establish weak block references in all patches
        for block in grid:
            for patch in block.patches:
                patch.attach_to_block(block)

        return grid

    @classmethod
    def read_plot3d(
        cls, p3d_file: str, fvbnd_file: str = None, flip_k: bool = True
    ) -> "Grid":
        """Read grid from Plot3D format file with optional boundary patches.

        Parameters
        ----------
        p3d_file : str
            Input Plot3D grid file to read
        fvbnd_file : str, optional
            Input FVBND boundary file to read patches from
        flip_k : bool, optional
            Whether to flip the k-axis to match write_plot3d behavior (default True)

        Returns
        -------
        Grid
            New grid containing blocks with coordinates and optional patches from files
        """
        from ember.plot3d import read_plot3d, read_fvbnd

        # Read the grid
        grid = read_plot3d(p3d_file, flip_k=flip_k)

        # Optionally read and assign patches
        if fvbnd_file is not None:
            patches_by_block = read_fvbnd(fvbnd_file)

            # Assign patches to blocks
            for block_id, patch_list in patches_by_block.items():
                if block_id >= len(grid):
                    raise ValueError(
                        f"Block ID {block_id} in FVBND file exceeds grid size {len(grid)}"
                    )

                for patch in patch_list:
                    grid[block_id].patches.append(patch)

        return grid

    @classmethod
    def read_ts3(cls, filename):
        """Read grid from TS3 format file.

        Parameters
        ----------
        filename : str
            Input TS3 file to read

        Returns
        -------
        Grid
            New grid containing blocks with coordinates and flow data from TS3 file
        """
        from ember.ts3 import read_ts3

        return read_ts3(filename)

    def set_conserved_cart_unstr(self, xyz, conserved_cart):
        r"""Set conserved variables from Cartesian unstructured data.

        Useful for importing flow solutions from unstructured CFD solvers that
        store data on arbitrary point clouds in Cartesian coordinates.

        Automatically detects the coordinate permutation and sign mapping that
        aligns the Cartesian data with the structured grid, converts coordinates
        to polar, and transforms the momentum vector accordingly.

        The input conserved state vector is:

        .. math::

            \mathcal{U}_{\mathrm{cart}} =
            \begin{pmatrix} \rho,\ \rho V_x,\ \rho V_y,\ \rho V_z,\ \rho e \end{pmatrix}

        The Cartesian momentum components are first converted to velocities,
        rotated into the polar frame :math:`(x, r, \theta)`, then reassembled
        as polar conserved variables:

        .. math::

            \mathcal{U} =
            \begin{pmatrix} \rho,\ \rho V_x,\ \rho V_r,\ \rho r V_\theta,\ \rho e \end{pmatrix}

        where the polar velocity components are:

        .. math::

            V_r &= V_y \cos\theta - V_z \sin\theta \\
            V_\theta &= -V_y \sin\theta - V_z \cos\theta

        with :math:`\theta = \mathrm{atan2}(-z,\, y)`.

        Parameters
        ----------
        xyz : array_like, shape (N, 3)
            Cartesian coordinates [x, y, z] with components on last axis
        conserved_cart : array_like, shape (N, 5)
            Cartesian conserved flow variables [rho, rhoVx, rhoVy, rhoVz, rhoe] with components on last axis
        """
        # Validation
        assert xyz.shape[1] == 3, "xyz must have 3 columns"
        assert conserved_cart.shape[1] == 5, "Conserved data must have 5 columns"
        assert conserved_cart.ndim == 2, "Conserved data must be 2D (N, 5)"
        assert xyz.shape[0] == conserved_cart.shape[0], (
            "xyz and conserved_cart must have same number of points"
        )

        # Step 1: Detect coordinate transformation and get block indices
        perm, signs, block_indices = self._align_cartesian(xyz)

        # Step 2: Convert Cartesian conserved variables to polar form
        rho = conserved_cart[..., 0]
        rhoVx_cart = conserved_cart[..., 1]
        rhoVy_cart = conserved_cart[..., 2]
        rhoVz_cart = conserved_cart[..., 3]
        rhoe = conserved_cart[..., 4]

        # Convert momentum to velocity, apply transformation, convert back
        Vxyz_cart = np.stack(
            [rhoVx_cart / rho, rhoVy_cart / rho, rhoVz_cart / rho], axis=-1
        )
        xrt, Vxrt = util.cart_to_pol(xyz, Vxyz_cart, perm, signs)

        # Extract polar components and convert back to momentum
        r = xrt[..., 1]
        rhoVx = rho * Vxrt[..., 0]
        rhoVr = rho * Vxrt[..., 1]
        rhoVt = rho * Vxrt[..., 2] * r

        # Assemble polar conserved variables
        conserved_pol = np.stack([rho, rhoVx, rhoVr, rhoVt, rhoe], axis=-1)

        # Step 3: Set conserved variables on blocks using block indices
        for ib, block in enumerate(self):
            # Use indices from _align_cartesian
            ind = block_indices[ib].flatten()

            # Set conserved variables
            conserved_block = conserved_pol[ind, :].reshape(block.shape + (5,))
            block.set_conserved(conserved_block)

    def set_fluid(self, fluid_obj):
        """Set equation of state on all blocks, preserving any existing flow field.

        The fluid object specifies the reference scales used to normalise stored
        thermodynamic quantities. This method rescales the stored nondimensional
        flow field so that the underlying dimensional state is unchanged. See
        :py:meth:`ember.block.Block.set_fluid` for full details.

        Parameters
        ----------
        fluid_obj : Fluid
            New fluid / equation of state object.
        """
        for block in self:
            block.set_fluid(fluid_obj)

    def set_L_ref(self, L_ref):
        """Set reference length scale on all blocks, preserving dimensional geometry and flow field.

        The reference length scale is used to normalise stored coordinates. This
        method rescales the stored nondimensional values so that the underlying
        dimensional geometry and flow field are unchanged. See
        :py:meth:`ember.block.Block.set_L_ref` for full details of the
        nondimensionalisation.

        Parameters
        ----------
        L_ref : float
            Reference length scale [m].
        """
        for block in self:
            block.set_L_ref(L_ref)

    def set_primitive_cart_unstr(self, xyz, primitive_cart):
        r"""Set primitive variables from Cartesian unstructured data.

        Useful for importing flow solutions from unstructured CFD solvers that
        store data on arbitrary point clouds in Cartesian coordinates.

        Automatically detects the coordinate permutation and sign mapping that
        aligns the Cartesian data with the structured grid, converts coordinates
        to polar, and rotates the velocity vector accordingly.

        The input primitive state vector is:

        .. math::

            \mathcal{P}_{\mathrm{cart}} =
            \begin{pmatrix} \rho,\ V_x,\ V_y,\ V_z,\ p \end{pmatrix}

        The Cartesian velocity components are rotated into the polar frame
        :math:`(x, r, \theta)` to give the polar primitive state:

        .. math::

            \mathcal{P} =
            \begin{pmatrix} \rho,\ V_x,\ V_r,\ V_\theta,\ p \end{pmatrix}

        where:

        .. math::

            V_r &= V_y \cos\theta - V_z \sin\theta \\
            V_\theta &= -V_y \sin\theta - V_z \cos\theta

        with :math:`\theta = \mathrm{atan2}(-z,\, y)`.

        Parameters
        ----------
        xyz : array_like, shape (N, 3)
            Cartesian coordinates [x, y, z] with components on last axis
        primitive_cart : array_like, shape (N, 5)
            Cartesian primitive flow variables [rho, Vx, Vy, Vz, P] with components on last axis
        """
        # Validation
        assert xyz.shape[1] == 3, "xyz must have 3 columns"
        assert primitive_cart.shape[1] == 5, "Primitive data must have 5 columns"
        assert primitive_cart.ndim == 2, "Primitive data must be 2D (N, 5)"
        assert xyz.shape[0] == primitive_cart.shape[0], (
            "xyz and primitive_cart must have same number of points"
        )

        # Convert to double precision for internal calculations
        xyz_fp64 = np.asarray(xyz, dtype=np.float64)
        primitive_cart_fp64 = np.asarray(primitive_cart, dtype=np.float64)

        # Step 1: Detect coordinate transformation and get block indices
        perm, signs, block_indices = self._align_cartesian(xyz_fp64)

        # Step 2: Convert Cartesian primitive variables to polar form
        rho = primitive_cart_fp64[..., 0]
        Vxyz_cart = primitive_cart_fp64[..., 1:4]  # Extract velocity components
        P = primitive_cart_fp64[..., 4]

        # Convert coordinates and velocities to polar
        xrt, Vxrt = util.cart_to_pol(xyz_fp64, Vxyz_cart, perm, signs)

        # Assemble polar primitive variables
        primitive_pol = np.stack(
            [rho, Vxrt[..., 0], Vxrt[..., 1], Vxrt[..., 2], P], axis=-1
        )

        # Step 3: Set primitive variables on blocks using block indices
        for ib, block in enumerate(self):
            # Use indices from _align_cartesian
            ind = block_indices[ib].flatten()

            # Set primitive variables (convert back to float32)
            primitive_block = (
                primitive_pol[ind, :].reshape(block.shape + (5,)).astype(np.float32)
            )
            block.set_P_rho(primitive_block[..., 4], primitive_block[..., 0])
            block.set_Vx(primitive_block[..., 1])
            block.set_Vr(primitive_block[..., 2])
            block.set_Vt(primitive_block[..., 3])

    def get_convergence(self):
        """Grid-representative convergence monitors at one step, non-dimensional.

        Stations are taken from :attr:`row_station_bid_pid`, derived from the
        grid's own topology.

        Returns
        -------
        ConvergenceStep
            Residual and station monitors for this step, with the outlet PID
            throttle state taken from
            :meth:`ember.outlet.OutletPatch.get_throttle_stats`. See
            :class:`ConvergenceStep` for the meaning of each field.

        Notes
        -----
        ``residual_nd`` itself (which drives the RK sweep) is untouched by the
        ``rhorVt`` rescaling applied to the reported residual.
        """

        def _block_residual(b):
            # abs into a cell-shaped view of block.scratch (free during logging,
            # residual_nd already built) to avoid a full-field |residual| temp.
            absres = util.carve_view(b.scratch, b.residual_nd.shape)
            res = np.abs(b.residual_nd, out=absres).mean(axis=(0, 1, 2))
            res[3] /= b.r_mid_nd
            return res

        residual = sum(_block_residual(b) for b in self) / len(self)

        mdot, ho, s = [], [], []
        for up_idx, dn_idx in self.row_station_bid_pid:
            for indices in (up_idx, dn_idx):
                m, h, se = self._station_stats(indices)
                mdot.append(m)
                ho.append(h)
                s.append(se)
        return ConvergenceStep(
            residual=residual,
            mdot=np.array(mdot),
            ho=np.array(ho),
            s=np.array(s),
            **self.patches.outlet[0].get_throttle_stats(),
        )

    def get_r_ref(self):
        """Calculate reference radial coordinates for each row of blocks.

        For each element in Grid.rows, calculates the maximum and minimum r
        coordinates across all blocks, then stores the mean value in Grid.r_ref
        as a numpy array with the same length as rows.
        """
        if len(self) == 0:
            self.r_ref = np.array([], dtype=np.float32)
            return

        r_mins = [np.min(b.r) for row in self.rows for b in row]
        r_maxs = [np.max(b.r) for row in self.rows for b in row]
        return np.mean([r_mins, r_maxs], axis=0).astype(np.float32)

    def accumulate_avg(self, n_step_avg):
        """Add the current conserved field into the running time-average.

        Accumulates each block's ``conserved_nd / n_step_avg`` into its
        :attr:`~ember.block.Block.conserved_avg_nd` buffer via the Fortran kernel. Called
        once per step over the final ``n_step_avg`` steps of a march, this
        builds the mean of the converged limit cycle. The buffer is a read-only
        cached array, so its ``flags.writeable`` is toggled around the in-place
        kernel write (mirrors :meth:`update_sources`).

        """
        for block in self:
            avg = block.conserved_avg_nd
            avg.flags.writeable = True
            ember.fortran.accumulate_avg(block.conserved_nd, avg, n_step_avg)
            avg.flags.writeable = False

    def apply_bconds(self):
        """Apply all boundary conditions across the grid once.

        Refreshes the mixing-plane targets, imposes the physical inlet, outlet,
        and mixing patch conditions on every block (each using its own
        relaxation factor ``rf``), then closes the point-matched periodic seams.

        Unlike the time-marching stepper this always applies the full set with
        no freeze or multigrid-level gating. The first call builds the periodic
        and mixing communicators lazily via :attr:`connectivity`; subsequent
        calls reuse the cached communicators until ``connectivity.clear()``.

        """
        # Refresh mixing-plane targets from the current cross-plane state before
        # the mixing patches read them in their apply step below.
        self.connectivity.mixing.exchange()

        for block in self:
            for patch in block.patches.inlet:
                patch.apply()
            for patch in block.patches.outlet:
                patch.apply()
            for patch in block.patches.mixing:
                patch.apply()

        # Close the point-matched periodic seams last.
        self.connectivity.periodic.apply()

    def apply_guess_meridional(self, block_guess, refine_factor=1):
        """Apply meridional flow field guess using curvilinear interpolation.

        Uses a 1D meridional block as initial guess, interpolating flow
        properties to all blocks in the grid using nearest-neighbor search in
        the (x, r) meridional plane. Optionally refines the guess using
        curvilinear arc-length coordinates for improved interpolation quality.

        Parameters
        ----------
        block_guess : Block
            Block containing the 1D guess flow field with shape (ni,).
            Use block_guess.squeeze() first if your block has singleton dimensions.
        refine_factor : int, optional
            Refinement factor for the guess block (default=1, no refinement).
            If > 1, the guess is interpolated along curvilinear arc length,
            providing a denser point cloud and smoother results.

        Raises
        ------
        ValueError
            If block_guess does not have shape (ni,)
        """
        # Validate that block_guess is 1D
        if len(block_guess.shape) != 1:
            raise ValueError(
                f"block_guess must be 1D with shape (ni,), got shape={block_guess.shape}. "
                f"Use block_guess.squeeze() to remove singleton dimensions."
            )

        # Extract 1D coordinates and conserved variables
        x_guess = block_guess.x
        r_guess = block_guess.r
        conserved_guess = block_guess.conserved  # Shape (ni, 5)

        # Compute curvilinear arc-length coordinate
        dx = np.diff(x_guess)
        dr = np.diff(r_guess)
        ds = np.sqrt(dx**2 + dr**2)
        s_guess = np.concatenate([[0.0], np.cumsum(ds)])

        # Apply refinement if requested
        if refine_factor > 1:
            # Resample arc length using util.resample
            s_refined, _ = util.resample(refine_factor, s_guess)

            # Linearly interpolate coordinates as functions of s
            x_refined = np.interp(s_refined, s_guess, x_guess)
            r_refined = np.interp(s_refined, s_guess, r_guess)

            # Linearly interpolate each conserved variable as function of s
            conserved_refined = np.stack(
                [
                    np.interp(s_refined, s_guess, conserved_guess[:, i])
                    for i in range(5)
                ],
                axis=-1,
            )  # Shape (ni*refine_factor, 5)
        else:
            x_refined = x_guess
            r_refined = r_guess
            conserved_refined = conserved_guess

        # Build KDTree from refined meridional coordinates
        xr_refined = np.stack([x_refined, r_refined], axis=-1)
        kdtree = KDTree(xr_refined)

        # Scalar mean pressure over the guess, used as normalisation for blade loading check
        P_guess_mean = np.mean(block_guess.P)

        # Apply to each block
        for block in self:
            # Copy metadata
            block.set_fluid(block_guess.fluid)

            # Find nearest neighbors
            shape_block = block.shape
            x_block = block.x.reshape(-1)
            r_block = block.r.reshape(-1)
            xr_block = np.stack([x_block, r_block], axis=-1)
            _, ind_nearest = kdtree.query(xr_block)

            # Interpolate and set conserved variables
            conserved_interp = conserved_refined[ind_nearest].reshape(*shape_block, 5)
            block.set_conserved(conserved_interp)

            # Snap velocities to i-direction gridlines in blade passages.
            # A blade passage is detected pointwise in (i,j) by a pressure
            # difference across the k-faces exceeding 1e-4 of the guess mean.
            P_k0 = block.P[:, :, 0]
            P_km1 = block.P[:, :, -1]
            loaded = np.abs(P_k0 - P_km1) / P_guess_mean > 1e-4  # (ni, nj) bool

            if np.any(loaded):
                # i-direction unit vector in (x, r) space — centred diffs,
                # one-sided at boundaries.  Shape: (ni, nj, nk).
                x3 = block.x
                r3 = block.r
                dx = np.gradient(x3, axis=0)
                dr = np.gradient(r3, axis=0)
                ds = np.sqrt(dx**2 + dr**2)
                lx = dx / ds  # (ni, nj, nk)
                lr = dr / ds  # (ni, nj, nk)

                # i-direction unit vector in (x, rt) space
                rt3 = block.r * block.t
                drt = np.gradient(rt3, axis=0)
                ds_xrt = np.sqrt(dx**2 + drt**2)
                lrt = drt / ds_xrt  # (ni, nj, nk)

                # Broadcast loaded mask over k
                mask = loaded[
                    :, :, np.newaxis
                ]  # (ni, nj, 1) -> broadcasts to (ni, nj, nk)

                # --- Meridional snap: preserve Vm, align (Vx, Vr) with (lx, lr) ---
                Vx = block.Vx.copy()
                Vr = block.Vr.copy()
                Vm = Vx * lx + Vr * lr  # signed projection onto i-gridline
                Vx_snapped = np.where(mask, Vm * lx, Vx)
                Vr_snapped = np.where(mask, Vm * lr, Vr)
                block.set_Vx(Vx_snapped)
                block.set_Vr(Vr_snapped)
                block.set_Vt(block.Vt)

                # --- Tangential snap: preserve |Vt|, sign from lrt ---
                Vt = block.Vt.copy()
                Vt_snapped = np.where(mask, np.abs(Vt) * np.sign(lrt), Vt)
                block.set_Vt(Vt_snapped)

            # Set turbulent viscosity to mean value
            block.set_mu_turb(np.full(shape_block, np.mean(block_guess.mu)))

    def apply_guess_quasi3d(self, block_guess):
        """Apply quasi-3D flow field guess by interpolating between two meridional planes.

        Uses two 2D meridional faces as boundary conditions on the low- and
        high-theta k-faces of each block, then linearly interpolates conserved
        variables across k as a function of the circumferential coordinate theta.

        Parameters
        ----------
        block_guess : Block, shape (ni, nj, 2)
            A single block whose k=0 face is the low-theta meridional plane and
            k=1 face is the high-theta meridional plane. i is streamwise, j is
            radial.

        Raises
        ------
        ValueError
            If block_guess does not have shape (ni, nj, 2).
        """
        if block_guess.ndim != 3 or block_guess.shape[2] != 2:
            raise ValueError(
                f"block_guess must have shape (ni, nj, 2), got shape={block_guess.shape}."
            )

        # Build one KDTree per boundary face from flattened (x, r) coordinates
        def _build_kdtree(face):
            # face: Block slice of shape (ni, nj)
            xr = np.stack([face.x.ravel(), face.r.ravel()], axis=-1)
            return KDTree(xr), face.conserved.reshape(-1, 5)

        kdtree_lo, conserved_lo_pts = _build_kdtree(block_guess[:, :, 0])
        kdtree_hi, conserved_hi_pts = _build_kdtree(block_guess[:, :, 1])

        mu_mean = 0.5 * (
            np.mean(block_guess[:, :, 0].mu) + np.mean(block_guess[:, :, 1].mu)
        )
        P_guess_mean = 0.5 * (
            np.mean(block_guess[:, :, 0].P) + np.mean(block_guess[:, :, 1].P)
        )

        for block in self:
            block.set_fluid(block_guess.fluid)

            ni, nj, nk = block.shape
            t_k0 = block.t[:, :, 0].mean()
            t_km1 = block.t[:, :, -1].mean()

            if t_k0 <= t_km1:
                k_lo, k_hi = 0, -1
            else:
                k_lo, k_hi = -1, 0

            # Interpolate each guess line onto the corresponding k-face using
            # nearest-neighbour lookup in (x, r), matching apply_guess_meridional.
            def _interp_face(kdtree, conserved_pts, k_idx):
                x_face = block.x[:, :, k_idx].reshape(-1)
                r_face = block.r[:, :, k_idx].reshape(-1)
                xr_face = np.stack([x_face, r_face], axis=-1)
                _, ind = kdtree.query(xr_face)
                return conserved_pts[ind].reshape(ni, nj, 5)

            conserved_face_lo = _interp_face(kdtree_lo, conserved_lo_pts, k_lo)
            conserved_face_hi = _interp_face(kdtree_hi, conserved_hi_pts, k_hi)

            # Theta at the two k-boundary faces, shape (ni, nj)
            t_lo = block.t[:, :, k_lo]  # low-theta face
            t_hi = block.t[:, :, k_hi]  # high-theta face

            # Linearly interpolate across all k at once.
            # block.t shape: (ni, nj, nk); w shape: (ni, nj, nk)
            t_all = block.t  # (ni, nj, nk)
            denom = (t_hi - t_lo)[:, :, np.newaxis]  # (ni, nj, 1)
            w = (t_all - t_lo[:, :, np.newaxis]) / denom  # (ni, nj, nk)

            # conserved_face_lo/hi: (ni, nj, 5) -> (ni, nj, 1, 5) for broadcast
            c_lo = conserved_face_lo[:, :, np.newaxis, :]  # (ni, nj, 1, 5)
            c_hi = conserved_face_hi[:, :, np.newaxis, :]  # (ni, nj, 1, 5)
            w4 = w[:, :, :, np.newaxis]  # (ni, nj, nk, 1)

            conserved_block = (1.0 - w4) * c_lo + w4 * c_hi  # (ni, nj, nk, 5)

            block.set_conserved(conserved_block)

            P_k0 = block.P[:, :, 0]
            P_km1 = block.P[:, :, -1]
            loaded = np.abs(P_k0 - P_km1) / P_guess_mean > 1e-4

            if np.any(loaded):
                dx = np.gradient(block.x, axis=0)
                dr = np.gradient(block.r, axis=0)
                drt = np.gradient(block.r * block.t, axis=0)
                ds = np.sqrt(dx**2 + dr**2 + drt**2)
                lx = dx / ds
                lr = dr / ds
                lrt = drt / ds

                mask = loaded[:, :, np.newaxis]

                Vx = block.Vx.copy()
                Vr = block.Vr.copy()
                Vt = block.Vt.copy()
                V_i = Vx * lx + Vr * lr + Vt * lrt  # speed along i-direction
                Vx_snapped = np.where(mask, V_i * lx, Vx)
                Vr_snapped = np.where(mask, V_i * lr, Vr)
                Vt_snapped = np.where(mask, V_i * lrt, Vt)
                block.set_Vx(Vx_snapped)
                block.set_Vr(Vr_snapped)
                block.set_Vt(Vt_snapped)

            block.set_mu_turb(np.full((ni, nj, nk), mu_mean))

    def apply_guess_restart(self, restarts):
        """Apply a list of BlockRestart objects to this Grid, block by block.

        Use this to initialize a fresh grid from a previously-saved solution.
        Same-shape blocks are set directly; differing shapes are trilinearly
        interpolated in index space. Only conserved variables are transferred;
        mu_turb is untouched.

        Parameters
        ----------
        restarts : list of BlockRestart
            One BlockRestart per block in this Grid.

        """
        for block, restart in zip(self, restarts):
            apply_restart(block, restart)

    def apply_rotation(self, row_types, Omega):
        """Apply rotation settings to blocks based on row types.

        Sets angular velocity and adds appropriate rotating wall patches to blocks
        based on the specified row type configuration. Supports stationary, tip_gap,
        and shroud configurations for turbomachinery applications.

        Parameters
        ----------
        row_types : list of str
            List of row type strings, one per row in the grid. Valid values:
            - "stationary": No rotating patches (fixed frame)
            - "tip_gap": Rotating patches on i=0, i=-1, j=0, k=0, k=-1
            - "shroud": Rotating patches on all boundaries (i=0, i=-1, j=0, j=-1, k=0, k=-1)
        Omega : list of float
            List of angular velocities [rad/s], one per row in the grid.
            Positive values indicate rotation direction.

        Raises
        ------
        AssertionError
            If length of row_types and Omega don't match
        ValueError
            If unknown row_type is specified

        Examples
        --------
        >>> grid = Grid([block1, block2])
        >>> grid.apply_rotation(["tip_gap"], [1000.0])  # Single rotating row
        """
        assert len(row_types) == len(Omega), (
            f"row_types length ({len(row_types)}) must match Omega length ({len(Omega)})"
        )

        for row_block, row_type, Omegai in zip(self.rows, row_types, Omega):
            for block in row_block:
                block.set_Omega(Omegai)
                if row_type == "stationary":
                    patches = []
                elif row_type == "tip_gap":
                    patches = [
                        RotatingPatch(i=0),
                        RotatingPatch(i=-1),
                        RotatingPatch(j=0),
                        RotatingPatch(k=0),
                        RotatingPatch(k=-1),
                    ]
                elif row_type == "shroud":
                    patches = [
                        RotatingPatch(i=0),
                        RotatingPatch(i=-1),
                        RotatingPatch(j=0),
                        RotatingPatch(j=-1),
                        RotatingPatch(k=0),
                        RotatingPatch(k=-1),
                    ]
                else:
                    raise ValueError(f"Unknown row type '{row_type}'")

                for patch in patches:
                    patch.set_Omega(Omegai)

                block.patches.extend(patches)

    def calculate_wdist(self, limit_pitch=np.inf):
        """
        This method creates a pitchwise-repeated grid to include neighboring passages,
        extracts all wall nodes from the repeated blocks, builds a KDTree for efficient
        nearest neighbor search, and calculates the distance from each node to the
        nearest wall surface. The results are stored in each block using set_wdist().

        The method uses real Cartesian coordinates (xyz) for accurate 3D distance
        calculations in turbomachinery applications.

        Parameters
        ----------
        limit_pitch : float, optional
            Cap the wall distance at ``limit_pitch`` times each row's reference
            blade pitch (``2*pi*r_ref/Nb``). This bakes the mixing-length limit
            directly into the stored wall distance, so downstream turbulence
            models need no separate cap. Defaults to ``np.inf`` (no limit).

        Examples
        --------
        >>> grid = Grid([block1, block2])
        >>> # Wall distances are now available as block.wdist for each block
        """
        if len(self) == 0:
            return  # No blocks to process

        for row in self.rows:
            _calculate_wdist_row(row, limit_pitch)

    def check_nan(self):
        """Scan every block's density field for NaN; report the first bad block.

        Cheap enough to call each solver step: only the density component
        ``conserved_nd[..., 0]`` is inspected, since a NaN in any conserved
        variable propagates into density within a step through the
        pressure/flux coupling. On the duct smoke test this costs ~0.5% of a
        single full-field residual evaluation.

        Raises
        ------
        DivergenceError
            If any block contains a NaN. The message names the first such block
            (index and label), the ``(i, j, k)`` node bounding box of the NaN
            region, and which of the six boundary faces it touches -- enough to
            tell a boundary-seeded blow-up from an interior one. The grid is left
            untouched so the invalid field can be inspected.
        """
        for iblock, block in enumerate(self):
            nan_mask = np.isnan(block.conserved_nd[..., 0])
            if not nan_mask.any():
                continue
            ni, nj, nk = block.ni, block.nj, block.nk
            ii, jj, kk = np.nonzero(nan_mask)
            box = (
                f"i[{ii.min()}:{ii.max()}]/{ni - 1} "
                f"j[{jj.min()}:{jj.max()}]/{nj - 1} "
                f"k[{kk.min()}:{kk.max()}]/{nk - 1}"
            )
            faces = []
            if ii.min() == 0:
                faces.append("i-lo")
            if ii.max() == ni - 1:
                faces.append("i-hi")
            if jj.min() == 0:
                faces.append("j-lo")
            if jj.max() == nj - 1:
                faces.append("j-hi")
            if kk.min() == 0:
                faces.append("k-lo")
            if kk.max() == nk - 1:
                faces.append("k-hi")
            touch = ", ".join(faces) if faces else "interior only"
            raise DivergenceError(
                f"NaN in conserved_nd density of block {iblock} ({block.label!r}): "
                f"{nan_mask.sum()} node(s), bbox {box}, touches [{touch}]"
            )

    def copy(self, keep_patches=True):
        """Create a deep copy of the grid with copied blocks.

        Returns
        -------
        Grid
            New Grid instance containing copies of all blocks.

        Example
        -------
        >>> grid_copy = grid.copy()
        >>> grid_copy[0].conserved[...] = 0  # Does not affect original grid
        """
        return Grid([b.copy(keep_patches) for b in self])

    def finalise_average(self):
        """Commit the accumulated time-average as the solution.

        Copies each block's :attr:`~ember.block.Block.conserved_avg_nd` into ``conserved_nd``,
        refreshes the conserved-dependent caches, then re-zeros the accumulator
        so any subsequent averaging window starts clean. Owns the
        ``flags.writeable`` toggle on the read-only average buffer.

        """
        for block in self:
            block.conserved_nd[...] = block.conserved_avg_nd
            block.update_cached_conserved()
            avg = block.conserved_avg_nd
            avg.flags.writeable = True
            avg.fill(0.0)
            avg.flags.writeable = False

    def interp_from(self, src):
        """Interpolate solution from src Grid onto this one, block by block.

        Parameters
        ----------
        src : Grid
            Source Grid providing the solution.

        """
        for tgt_block, src_block in zip(self, src):
            ember.block_util.interp_from(tgt_block, src_block)

    def resample(self, factors):
        """Resample all blocks, returning a new Grid at the new resolution."""
        return Grid([ember.block_util.resample(b, factors) for b in self])

    def smooth(self, sf4, sf2):
        """Apply constant-coefficient artificial dissipation to every block.

        Fans the fixed-``sf2``/``sf4`` :func:`ember.fortran.smooth3d_const`
        kernel across the grid, filtering each block's ``conserved_nd`` in place.
        Unlike the adaptive (P/T-weighted) smoother it has no thermodynamic input
        at all and never touches the pressure cache, so it is safe to run on the
        post-march solution while P/T are frozen (Denton-style "smooth with the
        old pressure"). Purely per-block; the nodal work array is borrowed from
        each block's transient scratch buffer.

        ``sf4``/``sf2`` are the final coefficients -- any CFL scaling is the
        caller's responsibility.
        """
        for block in self:
            ember.fortran.smooth3d_const(
                x=block.conserved_nd,
                sf4=sf4,
                sf2=sf2,
                xs=block.scratch[..., 0],
            )

    def update_bconds(self, freeze=False):
        """Refresh boundary-condition targets across the grid once.

        Advances the slowly-varying BC state that the per-substep
        :meth:`apply_bconds` then imposes: exchanges mixing-plane data,
        snapshots the inlet pressure datum, and re-derives the outlet
        PID/spanwise pressure target. Should be called once per outer
        timestep, before the Runge-Kutta stages.

        When ``freeze`` is True the targets are held stationary -- the mixing
        exchange and the outlet target re-derivation are skipped -- so an
        averaging window sees a fixed boundary. The ``update_soln`` snapshots
        still run so backflow density relaxation stays anchored to the current
        step.

        """
        if not freeze:
            self.connectivity.mixing.exchange()

        for block in self:
            for patch in block.patches.inlet:
                patch.update_soln()
            for patch in block.patches.mixing:
                patch.update_soln()
            for patch in block.patches.outlet:
                patch.update_soln()
                if not freeze:
                    patch.update_target()

    def update_cached_conserved(self):
        """Refresh conserved-dependent caches on every block.

        Fans :meth:`~ember.block.Block.update_cached_conserved` out across the grid, forcing
        each block's cached properties keyed on the conserved variables to
        recompute on next access. Needed after writing ``conserved_nd`` directly
        (bypassing the setters), e.g. the explicit time march.

        """
        for block in self:
            block.update_cached_conserved()

    def update_filter(self, cfl, delta_filt):
        """Evolve the SFD low-pass filter one step on every block.

        First-order exponential moving average of each block's cell-centred
        conserved state toward its current cell state, with per-cell timestep
        ``dt = cfl * dt_vol * vol``. ``cfl`` may be a per-cell/per-equation
        array of shape ``(ni-1, nj-1, nk-1, 5)`` or a single scalar; the rank
        selects the matching kernel. ``delta_filt`` is the filter time constant.

        Must run after the CFL and ``dt_vol`` for the step are current. This is
        the lone per-step writer of the read-only ``conserved_filt_nd`` buffer
        (the restart apply is the only other writer), so it owns the
        ``flags.writeable`` toggle (mirrors the timestep writers).

        """
        kernel = (
            ember.fortran.update_filter_scalar
            if np.ndim(cfl) == 0
            else ember.fortran.update_filter_array
        )
        for block in self:
            cons_filt = block.conserved_filt_nd
            cons_filt.flags.writeable = True
            kernel(
                cons_filt=cons_filt,
                cons_cell=block.conserved_cell_nd,
                cfl=cfl,
                dt_vol=block.dt_vol_nd,
                vol=block.vol_nd,
                delta_filt=delta_filt,
            )
            cons_filt.flags.writeable = False

    def update_residual(self, dampin=None, sf=0.0):
        """Rebuild the unintegrated net-flow residual on every block.

        Fans the fused ``set_residual`` kernel across the grid, writing each
        block's ``residual_nd`` from its frozen P/T cache, face areas, and body
        force. Purely per-block (no inter-block exchange), so it simply loops.

        Optional post-processing runs in place on each block's residual, in
        order: implicit residual smoothing (``sf``), then the change limiter
        (``dampin``).

        Parameters
        ----------
        dampin : float, optional
            Negative-feedback change limiter (multall's ``DAMP``). When given,
            each block's residual is passed through ``damp_residual``, which
            soft-clips cells whose per-step change ``residual * dt_vol`` is a
            large outlier relative to the per-variable block mean, shrinking
            them by ``1/(1 + fdamp/dampin)`` with ``fdamp = |change|/mean``.
            Large outliers saturate towards ``dampin * mean``. ``None`` (the
            default) disables it. multall recommends ``2..100``. Requires
            ``dt_vol_nd`` to have been populated by :meth:`update_timestep`.
        sf : float, optional
            Implicit residual-smoothing (IRS) coefficient (epsilon). ``0`` (the
            default) disables IRS; ``> 0`` applies the exact factored-tridiagonal
            smoother to each block via ``smooth_residual_tri_tiled``. IRS damps
            high-frequency residual content so the explicit march tolerates a
            higher CFL; because it acts only on the residual (which vanishes at
            convergence) it does not change the steady-state solution. Per-block
            only: block/periodic interfaces are treated as zero-gradient. Borrows
            ``block.scratch`` as its work buffer -- free at this point, since
            ``set_residual`` stages its face flows in ``tau_q_halo`` and the
            march reuses ``scratch`` only afterwards.

        """
        for block in self:
            i_cusp_start, i_cusp_end = block.i_cusp
            ni, nj, nk = block.shape
            # Rolling face-flow buffers for the fused k-tiled residual: a
            # k-face plane pair and three rows (one i, two alternating j),
            # borrowed zero-copy from the leading block.tau_q_halo storage.
            # planes takes one padding j-row exactly when its component
            # stride ni*nj*4 bytes would be a whole page multiple, so the
            # k-accumulate's component streams never 4K-alias (see
            # set_residual; the pad measurably hurts blocks it cannot help).
            kb = min(_KB_SLAB, nk - 1)
            njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
            planes, rows = util.carve_view(
                block.tau_q_halo, (ni, njp, 5, 2), (ni, 5, 3)
            )
            block.residual_nd.flags.writeable = True
            ember.fortran.set_residual(
                cons=block.conserved_nd,
                p=block.P_nd,
                p_offset=block.P_offset_nd,
                r=block.r_nd,
                omega=block.Omega_nd,
                dai=block.dAi_nd,
                daj=block.dAj_nd,
                dak=block.dAk_nd,
                du=block.residual_nd,
                f_body=block.F_body_nd,
                vx=block.Vx_nd,
                vr=block.Vr_nd,
                vt=block.Vt_nd,
                vt_rel=block.Vt_rel_nd,
                ho=block.ho_nd,
                planes=planes,
                rows=rows,
                **block.ijk_wall_conv,
                i_cusp_start=i_cusp_start,
                i_cusp_end=i_cusp_end,
                kb=kb,
                njp=njp,
                ni=ni,
                nj=nj,
                nk=nk,
            )
            if sf > 0.0:
                # Exact factored-tridiagonal IRS (Jameson ADI): a direct solve.
                # Scratch is just the Thomas coefficients, 2*(nci+ncj+nck) floats;
                # carve a 1D leading view of block.scratch (nodal (ni,nj,nk,5),
                # vastly oversized). Free here: set_residual does not touch it
                # and the march reuses it only after this returns.
                nwork = 2 * ((ni - 1) + (nj - 1) + (nk - 1))
                ember.fortran.smooth_residual_tri_tiled(
                    du=block.residual_nd,
                    sf=sf,
                    work=util.carve_view(block.scratch, (nwork,)),
                    ni=ni,
                    nj=nj,
                    nk=nk,
                )
            if dampin is not None:
                ember.fortran.damp_residual(
                    du=block.residual_nd,
                    dt_vol=block.dt_vol_nd,
                    dampin=dampin,
                    ni=ni,
                    nj=nj,
                    nk=nk,
                )
            block.residual_nd.flags.writeable = False

    def update_sources(self, inviscid, gain_filt):
        """Zero and rebuild the body force on every block of this grid level.

        Assembles, into each ``block.F_body_nd``, the viscous shear stresses
        (unless ``inviscid``), the polar source, and the optional SFD force
        (when ``gain_filt`` is nonzero) -- in that order, so the viscous
        momentum/energy negation does not flip the polar source added
        afterwards.

        The viscous calculation is phased across the whole grid: every block's
        tau/q is computed first, then a single periodic seam halo exchange runs,
        then the face fluxes are accumulated. This keeps the seam consistent for
        block-to-block periodic interfaces, where a per-block exchange would read
        a stale neighbour halo.

        Parameters
        ----------
        inviscid : bool
            Skip the viscous shear-stress/heat-flux terms when True.
        gain_filt : float
            Selective-frequency-damping gain; the SFD force is added only when
            nonzero.

        """
        # F_body_nd is a read-only cached buffer. Unlock it for the assembly below
        # and re-lock at the end, so consumers (the residual kernels) only ever
        # see an immutable array.
        for block in self:
            block.F_body_nd.flags.writeable = True
            block.F_body_nd.fill(0.0)

        if not inviscid:
            # tau_q_halo is pure scratch (always writeable); no locking needed.
            # First viscous phase: tau/q per cell (Pr_turb fixed at 1.0 for the
            # grid march; mixing-length vorticity always evaluated absolute-frame).
            for block in self:
                halo = block.tau_q_halo
                # tau_cell/q_cell are comp-last views sharing storage with the
                # halo; an order="F" reshape would alias a different cell and
                # silently drop the periodic seam exchange below.
                tau_cell = halo[..., 0:6]
                q_cell = halo[..., 6:9]
                # mu_turb is a data-row field the public property serves
                # read-only; grab a writeable view so the kernel can leave the
                # cell-centred mixing-length viscosity in place. Tolerate an
                # uninitialised field on entry since this pass is its producer.
                mu_turb = block._get_data_by_keys(
                    ("mu_turb",), raise_uninit=False, writeable=True
                )
                ember.fortran.set_tau_q_soa(
                    cons=block.conserved_nd,
                    t=block.T_nd,
                    mu=block.mu_nd,
                    cp=block.cp_nd,
                    pr_lam=block.fluid._Pr,
                    pr_turb=1.0,
                    xlength=block.xlen_sq_nd,
                    vol=block.vol_nd,
                    dai=block.dAi_nd,
                    daj=block.dAj_nd,
                    dak=block.dAk_nd,
                    r=block.r_nd,
                    vx=block.Vx_nd,
                    vr=block.Vr_nd,
                    vt=block.Vt_rel_nd,
                    tau_cell=tau_cell,
                    q_cell=q_cell,
                    mu_turb=mu_turb,
                )
                # The kernel has now populated mu_turb; mark it initialised so
                # later reads through the public property succeed.
                block._versions["mu_turb"] += 1

            # One seam exchange after all tau/q are fresh (see method docstring).
            self.connectivity.periodic.exchange_halos()

            # Second viscous phase: face fluxes from tau/q, accumulated into
            # F_body_nd. Viscous terms are negated in-kernel so the polar/SFD
            # forces added below are not flipped. No separate halo exchange is
            # needed for the cusp seam -- the kernel couples the seam flux
            # internally by averaging the two one-sided fluxes there.
            for block in self:
                halo = block.tau_q_halo
                tau_cell = halo[..., 0:6]
                q_cell = halo[..., 6:9]
                i_cusp_start, i_cusp_end = block.i_cusp
                # Rolling face-flow buffers for the fused k-tiled kernel: a
                # plane pair for the k-direction and three rows (one i, two
                # alternating j), borrowed zero-copy from the leading
                # block.scratch storage (5 nodal slots: fits for nk >= 3, or
                # nk == 2 with nj >= 6; carve_view raises otherwise).
                ni, nj, nk = block.shape
                kb = min(_KB_SLAB, nk - 1)
                planes, rows = util.carve_view(
                    block.scratch, (ni, nj, 4, 2), (ni, 4, 3)
                )
                ember.fortran.set_visc_force(
                    cons=block.conserved_nd,
                    vol=block.vol_nd,
                    dai=block.dAi_nd,
                    daj=block.dAj_nd,
                    dak=block.dAk_nd,
                    omega_block=block.Omega_nd,
                    r=block.r_nd,
                    mu=block.mu_nd,
                    fvisc=block.F_body_nd[..., 1:],
                    vx=block.Vx_nd,
                    vr=block.Vr_nd,
                    vt=block.Vt_rel_nd,
                    tau_cell=tau_cell,
                    q_cell=q_cell,
                    planes=planes,
                    rows=rows,
                    kb=kb,
                    **block.ijk_wall_visc,
                    **block.Omega_wall_nd,
                    i_cusp_start=i_cusp_start,
                    i_cusp_end=i_cusp_end,
                )

        for block in self:
            # Polar source accumulates into F_body_nd[..., 2] (radial momentum).
            ember.fortran.set_polar_source(
                cons_cell=block.conserved_cell_nd,
                r=block.r_nd,
                p=block.P_nd,
                p_offset=block.P_offset_nd,
                vol=block.vol_nd,
                net_flow=block.F_body_nd,
            )
            if gain_filt != 0.0:
                # SFD body force runs pre-step so it drives the RK integration,
                # not just the post-step residual.
                ember.fortran.apply_sfd_force(
                    f_body=block.F_body_nd,
                    cons_filt=block.conserved_filt_nd,
                    cons_cell=block.conserved_cell_nd,
                    vol=block.vol_nd,
                    gain_filt=gain_filt,
                )

        for block in self:
            block.F_body_nd.flags.writeable = False

    def update_timestep(self, rf, fac_visc=1.0):
        """Recompute the volumetric time step on every block.

        Uses a max-of-directional-radii variant of the JST/Blazek definition
        ``dt_vol = 1 / max(lam_conv, lam_diff)``, where ``lam_conv`` is the
        largest of the convective spectral radii
        ``Lambda_d = |V_rel . dA_d| + a*||dA_d||`` over the three directions and
        ``lam_diff = fac_visc * (mu_turb/rho)*max_d||dA_d||^2/vol`` is the
        turbulent-diffusion radius over the same faces
        (:func:`set_timestep_spectral`). Taking the max of the directional radii
        (rather than Blazek's sum) makes the CFL number the true 1D Courant limit
        (~``2*sqrt(2)`` for the 4-stage RK march) while staying
        aspect-ratio-independent for the viscous limit too.

        ``rf`` is the relaxation factor blending the new ``dt_vol`` with the
        existing buffer as ``rf*new + (1-rf)*old`` (pass ``rf=1.0`` for a fresh
        recompute). This is the lone writer of each block's read-only
        ``dt_vol_nd`` buffer, so it owns the ``flags.writeable`` toggle.

        ``fac_visc`` (>= 1) multiplies the diffusion radius so the viscous march
        tolerates the same cfl as the inviscid one; ``1.0`` leaves the bare
        directional radius untouched.

        """
        for block in self:
            block.dt_vol_nd.flags.writeable = True
            ember.fortran.set_timestep_spectral(
                dt_vol=block.dt_vol_nd,
                a=block.a_nd,
                cons_cell=block.conserved_cell_nd,
                r=block.r_nd,
                omega=block.Omega_nd,
                dai=block.dAi_nd,
                daj=block.dAj_nd,
                dak=block.dAk_nd,
                # Produced by the viscous pass (set_tau_q_soa, via
                # update_sources), so it is still zero on a grid that has not
                # yet marched -- tolerate it uninitialised. Zero mu_turb makes
                # lam_diff vanish and dt_vol fall back to the convective limit.
                mu_turb=block._get_data_by_keys(("mu_turb",), raise_uninit=False),
                vol=block.vol_nd,
                rf=rf,
                fac_visc=fac_visc,
            )
            block.dt_vol_nd.flags.writeable = False

    def write_emb(self, filename, compress=False):
        """Write grid to EMB binary format file.

        Parameters
        ----------
        filename : str
            Output filename for EMB file
        compress : bool, optional
            If True, compress the file using gzip (default False)

        Raises
        ------
        ValueError
            If grid is empty (contains no blocks)
        """
        if len(self) == 0:
            raise ValueError("Cannot write EMB file: grid contains no blocks")

        # Clear weak block references in all patches before pickling
        # They will be re-established after unpickling
        for block in self:
            for patch in block.patches:
                patch._block_ref = None

        try:
            opener = gzip.open if compress else open
            with opener(filename, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            # Restore weak references after pickling
            for block in self:
                for patch in block.patches:
                    patch.attach_to_block(block)

    def write_fvbnd(self, filename, region_id=0):
        """Write boundary conditions in FieldView boundary (.fvbnd) format.

        The FVBND format is used by FieldView to specify boundary regions for
        visualization purposes. See FieldView Reference Manual page 520.

        Parameters
        ----------
        filename : str
            Output filename for FieldView boundary file
        region_id : int, optional
            Region identifier for labeling patches (default 0)

        Raises
        ------
        ValueError
            If grid is empty (contains no blocks)
        """
        from ember.plot3d import write_fvbnd

        write_fvbnd(self, filename, iregion=region_id)

    def write_plot3d(
        self,
        p3d_filename: str,
        fvbnd_filename: str = None,
        flip_k: bool = True,
        iregion: int = 0,
    ) -> None:
        """Write grid to Plot3D format with optional FVBND boundary file.

        Parameters
        ----------
        p3d_filename : str
            Output filename for Plot3D grid file
        fvbnd_filename : str, optional
            Output filename for FVBND boundary file. If None, no boundary file is written
        flip_k : bool, optional
            Whether to flip the k-axis for proper volume orientation in Pointwise (default True)
        iregion : int, optional
            Region number for labeling in FVBND file (default 0)

        Raises
        ------
        ValueError
            If grid is empty (contains no blocks)
        """
        from ember.plot3d import write_plot3d, write_fvbnd

        # Write coordinate file
        write_plot3d(self, p3d_filename, flip_k=flip_k)

        # Write boundary file if requested
        if fvbnd_filename is not None:
            write_fvbnd(self, fvbnd_filename, iregion=iregion)

    def write_ts3(self, filename, strict=False):
        """Write grid to TS3 format file.

        Parameters
        ----------
        filename : str
            Output filename for TS3 file
        strict : bool, optional
            Whether to strictly validate all variables (default False)

        Raises
        ------
        ValueError
            If grid is empty (contains no blocks)
        """
        from ember.ts3 import write_ts3

        write_ts3(self, filename, strict=strict)

    @property
    def connectivity(self):
        """Get the cached connectivity manager for this grid.

        The manager is built once and cached on the grid, so the pairings and
        communicators it owns are computed lazily and reused. Call
        ``grid.connectivity.clear()`` to invalidate the cache after changing the
        grid topology (adding/removing blocks or patches).

        Returns
        -------
        GridConnectivityManager
            Connectivity manager providing access to patch connections
            via patch-type-specific properties and methods.
        """
        if self._connectivity is None:
            self._connectivity = GridConnectivityManager(self)
        return self._connectivity

    @property
    def n_row(self):
        """Get the number of blade rows in the grid.

        Returns
        -------
        int
            Number of blade rows (groups of blocks connected by periodic patches).

        Examples
        --------
        >>> grid = Grid([block1, block2])  # Single row
        >>> grid.n_row  # Returns 1
        >>>
        >>> grid_multi = Grid([stator_block, rotor_block])  # Two rows
        >>> grid_multi.n_row  # Returns 2
        """
        return len(self.rows)

    @property
    def patches(self):
        """Get read-only view of all patches across all blocks.

        Returns
        -------
        GridPatchCollection
            Read-only collection providing access to all patches from all blocks
            with patch type filtering (periodic, mixing, inlet, outlet, wall).
        """
        return GridPatchCollection(self)

    @property
    def row_station_bid_pid(self):
        """Per-row upstream/downstream measurement-station patch identifiers.

        Returns
        -------
        list of (list of (int, int), list of (int, int))
            ``[(up_idx, dn_idx), ...]``, one pair per blade row ordered inlet to
            outlet. Each entry is a list of ``(bid, pid)`` patch identifiers: the
            row's upstream face (inlet for the first row, otherwise mixing) then
            its downstream face (outlet for the last row, otherwise mixing).
            Consumed by :meth:`get_convergence` and
            :meth:`ember.convergence_history.ConvergenceHistory.from_grid`.
        """
        from ember.patch import InletPatch, OutletPatch, MixingPatch

        rows = self.rows
        n_row = len(rows)
        result = []
        for i, row_blocks in enumerate(rows):
            up_cls = InletPatch if i == 0 else MixingPatch
            dn_cls = OutletPatch if i == n_row - 1 else MixingPatch
            up_idx, dn_idx = [], []
            for b in row_blocks:
                bid = self.index(b)
                for pid, p in enumerate(b.patches):
                    if isinstance(p, up_cls):
                        up_idx.append((bid, pid))
                    if isinstance(p, dn_cls):
                        dn_idx.append((bid, pid))
            result.append((up_idx, dn_idx))
        return result

    @property
    def rows(self):
        """Get blade rows by grouping blocks connected by periodic patches.

        Blocks connected by periodic patches are in the same blade row.
        Multiple rows are separated by mixing patches. Rows are ordered
        from inlet to outlet based on mixing patch connectivity.

        Returns
        -------
        List[List[Block]]
            List of blade rows, where each row is a list of Block objects.
            Blocks within a row are connected by periodic patches.
            Rows are ordered from inlet (upstream) to outlet (downstream).

        Examples
        --------
        >>> # Single-row grid
        >>> grid = Grid([block1, block2])
        >>> rows = grid.rows  # Returns [[block1, block2]]
        >>>
        >>> # Multi-row grid with mixing patches
        >>> grid = Grid([stator_blocks, rotor_blocks])
        >>> rows = grid.rows  # Returns [[stator_blocks...], [rotor_blocks...]]
        """
        if len(self) == 0:
            return []

        # Find groups of blocks connected by periodic patches
        row_groups = self._find_periodic_connected_components()

        # Order the groups from inlet to outlet
        ordered_groups = self._order_row_groups(row_groups)

        # Convert block indices to Block objects
        return [[self[bid] for bid in group] for group in ordered_groups]

    @property
    def size(self):
        """Total number of grid points across all blocks.

        Returns
        -------
        int
            Sum of all block sizes in the grid
        """
        return sum(block.size for block in self)

    # Abstract methods implementation for _LabelledList
    # _get_item_label, _set_item_label now provided by base class

    # Grid-specific methods

    # end connectivity

    # end n_row

    # end patches

    # end rows

    # end size


class GridConnectivity:
    """Manages connectivity between patches of a specific type across all blocks in a grid.

    Takes a Patch subclass and operates only on patches matching that type.
    Provides a simplified interface with a single pair() method.

    Examples
    --------
    >>> from ember.patch import PeriodicPatch
    >>> grid = Grid([block1, block2, block3])
    >>> connectivity = GridConnectivity(grid, PeriodicPatch)
    >>> pairs = connectivity.pair()
    >>> # Returns: [((0, 2), (1, 0), transform)] where transform is check_match output
    """

    def __init__(self, grid, patch_class):
        """Initialize connectivity manager for a specific patch type.

        Parameters
        ----------
        grid : Grid
            The grid containing blocks with patches to analyze for connectivity
        patch_class : type
            Subclass of Patch to filter and operate on
        """
        self.grid = grid
        self.patch_class = patch_class
        self._pairs = {}
        self._pairs_computed = False
        self._communicator = None

    def _compute_pairs(self, rtol=1e-6):
        # Collect patches of the specified type from all blocks
        patches = []
        blocks = []
        patch_indices = []

        for block_idx, block in enumerate(self.grid):
            for patch_idx, patch in enumerate(block.patches):
                if isinstance(patch, self.patch_class):
                    patches.append(patch)
                    blocks.append(block)
                    patch_indices.append((block_idx, patch_idx))

        # If no patches or only one, no connectivity possible
        if len(patches) <= 1:
            return {}

        # Build KDTree using only x,r coordinates for spatial search
        xr_coords = np.stack([patch.xrt_centre[:2] for patch in patches])
        tree = KDTree(xr_coords)

        # Track which patches have been paired
        paired = set()
        matches = {}

        atol = 1e-4 * xr_coords[:, 1].mean()

        # Loop over patches and find all matches
        for idx_self, patch in enumerate(patches):
            if idx_self in paired:
                continue

            # Query all patches (we'll skip self in the loop)
            dist, indices = tree.query(
                xr_coords[idx_self : idx_self + 1],
                k=len(patches),
                distance_upper_bound=atol,
            )
            dist = dist[0]
            indices = indices[0]
            imax = len(indices)

            # Loop over all matches (apart from self) and check for pairing
            found_match = False
            for _, idx_other in zip(dist, indices):
                # Skip self and already paired patches
                if idx_other == idx_self or idx_other in paired:
                    continue

                # Skip infinite distances (above distance_upper_bound)
                if idx_other >= imax:
                    continue

                # Get the other patch and block
                patch_other = patches[idx_other]
                block_self = blocks[idx_self]
                block_other = blocks[idx_other]

                # Use the patch's check_match method to determine if they should be paired
                # Patches must be attached to blocks for check_match to work
                patch.attach_to_block(block_self)
                patch_other.attach_to_block(block_other)

                transform = patch.check_match(patch_other, rtol)
                if transform is not None:
                    # Get reverse transform
                    reverse_transform = patch_other.check_match(patch, rtol)

                    # Store both directions in the dictionary
                    matches[patch_indices[idx_self]] = (
                        patch_indices[idx_other],
                        transform,
                    )
                    matches[patch_indices[idx_other]] = (
                        patch_indices[idx_self],
                        reverse_transform,
                    )

                    # Mark both patches as paired and break
                    paired.add(idx_self)
                    paired.add(idx_other)
                    found_match = True
                    break

            if not found_match:
                raise ValueError(
                    f"Unmatched: {patch} bid, pid={patch_indices[idx_self]})\n"
                    "Check:\n"
                    "- Are coordinates intialised?\n"
                    "- Is Nb set?\n"
                    "- Are the patch indices correct?"
                )

        return matches

    def _get_communicator(self):
        """Build (and cache) the communicator for this patch type.

        Private to the connectivity machinery: prefer the delegating methods
        (:meth:`apply`, :meth:`exchange_halos`, :meth:`exchange`) at call sites.

        Returns
        -------
        object
            Communicator instance appropriate for ``self.patch_class``.

        Raises
        ------
        TypeError
            If no communicator is defined for ``self.patch_class``.
        """
        if self._communicator is None:
            from .patch import MixingPatch, NonMatchPatch, PeriodicPatch

            if self.patch_class is PeriodicPatch:
                self._communicator = ember.periodic_communicator.PeriodicCommunicator(
                    self.grid, self.pair()
                )
            elif self.patch_class is MixingPatch:
                # rf_mix hardcoded to the config default (0.1).
                self._communicator = ember.mixing_communicator.MixingCommunicator(
                    self.grid, self.pair()
                )
            elif self.patch_class is NonMatchPatch:
                self._communicator = ember.nonmatch_communicator.NonMatchCommunicator(
                    self.grid, self.pair()
                )
            else:
                raise TypeError(
                    f"No communicator defined for patch class {self.patch_class!r}"
                )
        return self._communicator

    def apply(self):
        """Apply the communicator's averaging (periodic and non-matching patches)."""
        return self._get_communicator().apply()

    def clear(self):
        """Drop cached pairings and communicator so they rebuild on next use."""
        self._pairs = {}
        self._pairs_computed = False
        self._communicator = None

    def exchange(self):
        """Exchange in/out flux state across mixing-plane patches."""
        return self._get_communicator().exchange()

    def exchange_halos(self):
        """Exchange halo data across periodic patches."""
        return self._get_communicator().exchange_halos()

    def pair(self, rtol=1e-6):
        """Pair patches of the specified type, caching the result.

        Filters patches to only include instances of self.patch_class, then uses
        spatial proximity (KDTree in x,r coordinates) to find potential matches
        and delegates to each patch's check_match method for validation. The
        result is cached; call :meth:`clear` to recompute after topology changes.

        Parameters
        ----------
        rtol : float, optional
            Relative tolerance for matching (passed to patch check_match methods)

        Returns
        -------
        dict
            Dictionary where keys are (bid, pid) tuples and values are
            ((matching_bid, matching_pid), transform) tuples. Both patches
            in each pair are included as separate keys.

        Raises
        ------
        ValueError
            If any patch does not have a matching pair
        """
        if not self._pairs_computed:
            self._pairs = self._compute_pairs(rtol)
            self._pairs_computed = True
        return self._pairs


def _calculate_wdist_row(grid, limit_pitch=np.inf):
    """Perform wall distance calc for a single blade row grid.

    If ``limit_pitch`` is finite, the (nondimensional) wall distance is capped
    at ``limit_pitch`` times the row's reference blade pitch.
    """

    # Reference pitch for this row, nondimensionalised by L_ref (shared per row)
    r_ref_nd = np.mean(
        [np.min(block.r_nd) for block in grid] + [np.max(block.r_nd) for block in grid]
    )
    pitch_ref_nd = 2.0 * np.pi * r_ref_nd / grid[0].Nb
    wdist_cap = limit_pitch * pitch_ref_nd

    # Collect nondim polar coordinates
    xrt_wall = []
    for block in grid:
        xrt_wall.append(block.xrt_nd[block._wall_nodes, :])

    n_wall = sum(x.shape[0] for x in xrt_wall)
    if n_wall == 0:
        raise ValueError("No wall nodes found; all faces are permeable.")

    # Repeat +/- a pitch for periodicity
    xrt_wall = np.tile(np.concatenate(xrt_wall, axis=0), (3, 1))
    xrt_wall[:n_wall, 2] -= grid[0].pitch
    xrt_wall[n_wall : 2 * n_wall, 2] += grid[0].pitch

    # Scale theta by r to get pseudo-Cartesian coordinates for KDTree
    xrt_wall[:, 2] *= xrt_wall[:, 1]
    kdtree = KDTree(xrt_wall)

    # Calculate distances for each block in the original grid
    for block in grid:
        #
        # Get pseudo-Cartesian coordinates for block nodes
        xrt_block = block.xrt_nd.reshape(-1, 3).copy()
        xrt_block[:, 2] *= xrt_block[:, 1]

        # Query KDTree for nearest wall distances
        distances, _ = kdtree.query(xrt_block)

        # Cap at the mixing-length limit (no-op when limit_pitch is inf)
        np.minimum(distances, wdist_cap, out=distances)

        # Reshape distances back to block shape and store
        block._set_data_by_keys(("wdist",), distances.reshape(block.shape))


class GridConnectivityManager:
    """Provides connectivity access for different patch types in a grid.

    Examples
    --------
    >>> grid = Grid([block1, block2])
    >>> grid.connectivity.periodic.pair()  # Get periodic patch pairs
    >>> grid.connectivity.mixing.pair()    # Get mixing patch pairs
    >>> grid.connectivity.pair()           # Get all patch pairs
    """

    def __init__(self, grid):
        """Initialize connectivity manager for a grid.

        Parameters
        ----------
        grid : Grid
            The grid containing blocks with patches
        """
        self._grid = grid
        # Cache one GridConnectivity per patch class, keyed by the class itself.
        self._by_class = {}

    def _connectivity(self, patch_class):
        """Return the cached GridConnectivity for ``patch_class``, building once."""
        conn = self._by_class.get(patch_class)
        if conn is None:
            conn = GridConnectivity(self._grid, patch_class)
            self._by_class[patch_class] = conn
        return conn

    def clear(self):
        """Drop all cached pairings and communicators across every patch type."""
        self._by_class = {}

    def pair(self, rtol=1e-6):
        """Pair all patch types and return combined connectivity dictionary.

        Parameters
        ----------
        rtol : float, optional
            Relative tolerance for matching (passed to patch check_match methods)

        Returns
        -------
        dict
            Combined dictionary where keys are (bid, pid) tuples and values are
            ((matching_bid, matching_pid), transform) tuples for all patch types.
        """
        # Combine connectivity for all supported patch types
        all_matches = {}
        all_matches.update(self.periodic.pair(rtol))
        all_matches.update(self.mixing.pair(rtol))
        all_matches.update(self.nonmatch.pair(rtol))
        return all_matches

    @property
    def cusp(self):
        """Get connectivity manager for cusp patches."""
        from .patch import CuspPatch

        return self._connectivity(CuspPatch)

    # end cusp

    @property
    def mixing(self):
        """Get connectivity manager for mixing patches."""
        from .patch import MixingPatch

        return self._connectivity(MixingPatch)

    # end mixing

    @property
    def nonmatch(self):
        """Get connectivity manager for non-matching patches."""
        from .patch import NonMatchPatch

        return self._connectivity(NonMatchPatch)

    # end nonmatch

    @property
    def periodic(self):
        """Get connectivity manager for periodic patches."""
        from .patch import PeriodicPatch

        return self._connectivity(PeriodicPatch)

    # end periodic


# eq=False: the array fields would make a generated __eq__ return an array,
# raising on truth-testing. Identity comparison is all this type needs.
@dataclass(frozen=True, eq=False)
class ConvergenceStep:
    """Grid-wide convergence monitors at a single time step, non-dimensional.

    Produced by :meth:`Grid.get_convergence` and consumed by
    :meth:`ember.convergence_history.ConvergenceHistory.record_convergence`,
    which unpacks the station vectors into one scalar column per station.

    Station vectors are ordered inlet to outlet, each blade row contributing
    its upstream then downstream face
    (``[row0_up, row0_dn, row1_up, row1_dn, ...]``), so they have length
    ``2 * n_row``.
    """

    residual: np.ndarray
    """Block-mean ``|residual_nd|`` per conserved variable, shape ``(5,)``,
    ordered ``(rho, rhoVx, rhoVr, rhorVt, rhoe)``. The ``rhorVt`` entry is
    divided per block by ``block.r_mid_nd`` so its magnitude is comparable to
    the ``rhoVx``/``rhoVr`` residuals; this rescaling is for monitoring only."""

    mdot: np.ndarray
    """Station mass flow rates, shape ``(2 * n_row,)``, non-dimensionalised by
    the fluid mass-flux scale."""

    ho: np.ndarray
    """Station stagnation enthalpies, shape ``(2 * n_row,)``,
    non-dimensionalised by ``u_ref``."""

    s: np.ndarray
    """Station specific entropies, shape ``(2 * n_row,)``,
    non-dimensionalised by ``Rgas_ref``."""

    mdot_target: float = 0.0
    """Outlet throttle mass flow setpoint [kg/s]; zero when throttle inactive."""

    mdot_throttle: float = 0.0
    """Mass flow measured at the outlet patch on its last target update [kg/s]."""

    P_throttle: float = 0.0
    """Total PID pressure correction applied at the outlet [Pa]."""

    dP_P: float = 0.0
    """Proportional contribution to :attr:`P_throttle` [Pa]."""

    dP_I: float = 0.0
    """Integral contribution to :attr:`P_throttle` [Pa]."""

    dP_D: float = 0.0
    """Derivative contribution to :attr:`P_throttle` [Pa]."""


class DivergenceError(RuntimeError):
    """Raised when a block's conserved field contains a NaN.

    A dedicated type lets a solver loop catch divergence precisely and exit
    cleanly (leaving the invalid field in place for debugging) while genuinely
    unexpected errors still propagate. See :meth:`Grid.check_nan`.
    """
