"""Collection classes for managing groups of objects with labels and type-based access.

This module provides specialized collection classes that extend Python's standard list
functionality with support for label-based indexing and type-filtered access. The internal
``_LabelledList`` abstract base class enables collections where items can be accessed both by numeric index and
by unique string labels, with automatic validation of label uniqueness. The PatchList subclass
adds type-based filtering allowing easy access to subsets of patches by their class (e.g., all
InletPatch objects or all PeriodicPatch objects). These collections provide convenient,
intuitive syntax for managing structured CFD data such as blocks, patches, and boundary
conditions while maintaining strict validation to prevent configuration errors in multi-block
grid setups.
"""

import logging
from abc import ABC

logger = logging.getLogger(__name__)


class _LabelledList(ABC):
    """Abstract base class for collections with label-based access.

    Provides a standard interface for collections that support both numeric
    indexing and string-based label access. Items in the collection can have
    optional labels that must be unique within the collection.

    Items must have a 'label' attribute and a 'set_label(label)' method.

    """

    def __init__(self, items=None, item_class=None):
        """Initialize with optional list of items.

        Parameters
        ----------
        items : list, optional
            Initial list of items to add to the collection.
        item_class : class, optional
            Class type for items in the collection.
            If not provided and items exist, will be inferred from first item.
        """
        self._items = items or []

        # Save the item class type
        if item_class is not None:
            self._item_class = item_class
        elif self._items:
            self._item_class = type(self._items[0])
        else:
            self._item_class = None

        # If items were provided, validate labels for uniqueness
        if items:
            self._validate_initial_labels()

    def _validate_initial_labels(self):
        """Validate that all initial item labels are unique."""
        labels_seen = set()
        for item in self._items:
            label = item.label
            if label is not None:
                if label in labels_seen:
                    raise ValueError(
                        f"Duplicate label '{label}' found in initial items"
                    )
                labels_seen.add(label)

    def _validate_unique_label(self, label, exclude_index=None):
        """Validate that label is unique in collection."""
        if label is None:
            return  # None labels are allowed and don't need to be unique

        for i, item in enumerate(self._items):
            if exclude_index is not None and i == exclude_index:
                continue  # Skip the item being replaced
            if item.label == label:
                raise ValueError(
                    f"Item with label '{label}' already exists in collection"
                )

    def __len__(self):
        """Return number of items in collection."""
        return len(self._items)

    def __iter__(self):
        """Iterate over items in collection."""
        return iter(self._items)

    def __getitem__(self, key):
        """Get item by index, label, slice, or index tuple.

        Parameters
        ----------
        key : int, str, slice, list, or tuple
            Key to access items:
            - int: single item by index
            - str: single item by label
            - slice: new collection with sliced items
            - list/tuple of int: new collection with selected items

        Returns
        -------
        Any or _LabelledList
            Single item (for int/str key) or new collection instance (for slice/tuple).
        """
        if isinstance(key, slice):
            # Return new instance with sliced items
            sliced_items = self._items[key]
            return self.__class__(sliced_items)

        elif isinstance(key, (list, tuple)) and all(isinstance(k, int) for k in key):
            # Handle index tuple: collection[[0, 2, 4]]
            selected_items = [self._items[i] for i in key]
            return self.__class__(selected_items)

        elif isinstance(key, int):
            # Single item by numeric index
            return self._items[key]

        elif isinstance(key, str):
            # Single item by label
            for item in self._items:
                if item.label == key:
                    return item
            raise KeyError(f"No item found with label '{key}'")

        else:
            raise TypeError(
                f"Invalid key type: {type(key)}. Expected int, str, slice, or sequence of int."
            )

    def __setitem__(self, index, item):
        """Set item at index or by label, or insert new item if label doesn't exist.

        Parameters
        ----------
        index : int or str
            Index or label to set. For string labels, if no existing item has this
            label, the item will be appended to the collection.
        item : Any
            Item to set at the given index/label. For string indexing with new labels,
            the item's label must be None, empty, or match the key.

        Notes
        -----
        When using string indexing:
        - If an item with the given label exists, it will be replaced
        - If no item with the given label exists, the item will be appended
        - The item's label will be updated to match the key if it's None or empty
        - If the item has a different non-empty label, a ValueError is raised
        """
        if isinstance(index, str):
            # String indexing by label - find the item and replace it, or append if not found
            for i, existing_item in enumerate(self._items):
                if existing_item.label == index:
                    # Validate new item label compatibility
                    item_label = item.label
                    if item_label and item_label != index:
                        raise ValueError(
                            f"Cannot assign item with label '{item_label}' to key '{index}'. "
                            f"Item label must be None, empty, or match the key '{index}'."
                        )

                    # Set the item label to match the key if it was None or empty
                    if not item_label:
                        item.set_label(index)

                    self._validate_unique_label(item.label, exclude_index=i)
                    self._items[i] = item
                    return

            # No existing item found with this label - insert new item
            item_label = item.label
            if item_label and item_label != index:
                raise ValueError(
                    f"Cannot assign item with label '{item_label}' to key '{index}'. "
                    f"Item label must be None, empty, or match the key '{index}'."
                )

            # Set the item label to match the key if it was None or empty
            if not item_label:
                item.set_label(index)

            # Use append to leverage existing validation and overlap checking
            self.append(item)
        else:
            # Numeric indexing
            if not isinstance(index, int):
                raise TypeError("Index must be int or str")
            if not (-len(self._items) <= index < len(self._items)):
                raise IndexError("Collection index out of range")

            # Validate label uniqueness if item has a label
            if (item_label := item.label) is not None:
                self._validate_unique_label(item_label, exclude_index=index)

            self._items[index] = item

    def __delitem__(self, index):
        """Delete item at index or by label.

        Parameters
        ----------
        index : int or str
            Index or label of item to delete.
        """
        if isinstance(index, str):
            # String indexing by label
            for i, item in enumerate(self._items):
                if item.label == index:
                    del self._items[i]
                    return
            raise KeyError(f"No item found with label '{index}'")
        else:
            # Numeric indexing
            if not isinstance(index, int):
                raise TypeError("Index must be int or str")
            if not (-len(self._items) <= index < len(self._items)):
                raise IndexError("Collection index out of range")

            del self._items[index]

    def __contains__(self, item):
        """Check if item or label is in collection.

        Parameters
        ----------
        item : Any or str
            Item object or label string to check for.

        Returns
        -------
        bool
            True if item is in the collection.
        """
        if isinstance(item, str):
            # String check by label
            return any(existing_item.label == item for existing_item in self._items)
        else:
            # Object check
            return item in self._items

    def append(self, item):
        """Add item to collection.

        Parameters
        ----------
        item : Any
            Item to add to the collection. The item manages its own label.

        Raises
        ------
        ValueError
            If item's label already exists in the collection.
        """
        if (item_label := item.label) is not None:
            self._validate_unique_label(item_label)

        self._items.append(item)

    def clear(self):
        """Remove all items from collection."""
        self._items.clear()

    def extend(self, items):
        """Add multiple items to collection.

        Parameters
        ----------
        items : list
            List of items to add. Each item manages its own label.

        Raises
        ------
        ValueError
            If any item's label already exists in the collection.
        """
        # Validate all item labels first
        existing_labels = set(self.labels)
        new_labels = []

        for item in items:
            item_label = item.label
            if item_label is not None:
                if item_label in existing_labels:
                    raise ValueError(
                        f"Item with label '{item_label}' already exists in collection"
                    )
                if item_label in new_labels:
                    raise ValueError(
                        f"Duplicate label '{item_label}' found in new items"
                    )
                new_labels.append(item_label)

        # Add all items
        for item in items:
            self.append(item)

    def index(self, item, start=0, stop=None):
        """Return index of first occurrence of item.

        Parameters
        ----------
        item : Any
            Item to find.
        start : int, optional
            Start index for search.
        stop : int, optional
            Stop index for search.

        Returns
        -------
        int
            Index of the item.
        """
        if stop is None:
            return self._items.index(item, start)
        return self._items.index(item, start, stop)

    def insert(self, index, item):
        """Insert item at specific index.

        Parameters
        ----------
        index : int
            Index at which to insert the item.
        item : Any
            Item to insert. The item manages its own label.
        """
        if (item_label := item.label) is not None:
            self._validate_unique_label(item_label)

        self._items.insert(index, item)

    def pop(self, index=-1):
        """Remove and return item at index (default last).

        Parameters
        ----------
        index : int, optional
            Index of item to remove and return. Default is -1 (last item).

        Returns
        -------
        Any
            The removed item.
        """
        if not self._items:
            raise IndexError("pop from empty collection")

        return self._items.pop(index)

    def remove(self, item):
        """Remove first occurrence of item.

        Parameters
        ----------
        item : Any
            Item to remove.
        """
        try:
            self._items.remove(item)
        except ValueError:
            raise ValueError("Item not in collection")

    @property
    def labels(self):
        """List of item labels, in order, including ``None`` for unlabelled items.

        Returns
        -------
        list
            The ``label`` of each item in the collection. Entries are ``None``
            where an item has no label.
        """
        return [item.label for item in self._items]


class BlockPatchCollection(_LabelledList):
    """Stores the boundary condition patches for a single :py:class:`~ember.block.Block`.

    Holds a reference to the parent block and provides the standard collection
    interface (``append``, ``extend``, ``insert``, ``del``, integer and string
    indexing, iteration, ``len``) for managing :py:class:`~ember.patch.Patch`
    objects. Adding a patch automatically attaches it to the block, validates
    its limits against the block shape, and checks for spatial overlaps with
    existing patches of the same type. Type-grouped properties (``inlet``,
    ``outlet``, ``periodic``, etc.) return plain lists. Normally accessed via
    :py:attr:`Block.patches <ember.block.Block.patches>` rather than
    instantiated directly.
    """

    def __init__(self, block):
        """Initialise with a reference to the parent block.

        Parameters
        ----------
        block : :py:class:`~ember.block.Block`
            The block whose boundary conditions this collection manages.
        """
        # Import here to avoid circular import
        from ember.patch import Patch

        super().__init__(item_class=Patch)
        self._block = block

    def __getitem__(self, key):
        """Get patch by index, label, or slice, preserving block reference on slices."""
        if isinstance(key, (slice, list, tuple)) and not isinstance(key, str):
            items = (
                self._items[key]
                if isinstance(key, slice)
                else [self._items[i] for i in key]
            )
            result = BlockPatchCollection(self._block)
            result._items = items
            return result
        return super().__getitem__(key)

    def _patches_overlap(self, patch1, patch2):
        """Check if two patches of the same type overlap in 3D space."""
        # Only check overlap for patches of the same type
        logger.debug(f"Checking overlap between {patch1} and {patch2}")
        if type(patch1) is not type(patch2):
            logger.debug("Different patch types, no overlap.")
            return False

        # Passive overlay patches (e.g. ProbePatch) may coincide with anything
        if patch1._allow_overlap or patch2._allow_overlap:
            logger.debug("Overlap-exempt patch, no overlap.")
            return False

        # Both patches must be attached to blocks for meaningful comparison
        if patch1._block_ref is None or patch2._block_ref is None:
            logger.debug(
                "One or both patches are not attached to blocks, cannot check overlap."
            )
            return False

        # Early return if either patch is a point (size == 1)
        if patch1.size == 1 or patch2.size == 1:
            logger.debug("One or both patches are points, no overlap.")
            return False

        if patch1.const_dim != patch2.const_dim:
            logger.debug("Different constant dimensions, no overlap.")
            return False

        # Get resolved absolute limits for both patches
        limits1 = patch1.ijk_lim_abs
        limits2 = patch2.ijk_lim_abs
        logger.debug(f"Patch 1 limits (abs): {patch1.ijk_lim_abs.flatten()}")
        logger.debug(f"Patch 2 limits (abs): {patch2.ijk_lim_abs.flatten()}")

        # Different const_dim means no overlap
        if limits1[patch1.const_dim, 0] != limits2[patch2.const_dim, 0]:
            logger.debug(
                f"Constant dimension {patch2.const_dim} values differ: {limits1[patch1.const_dim, 0]} vs {limits2[patch2.const_dim, 0]}, no overlap."
            )
            return False

        # Check overlap in each dimension using simple range overlap logic
        for dim in range(3):
            if dim == patch1.const_dim:
                continue

            start1, end1 = limits1[dim, 0], limits1[dim, 1]
            start2, end2 = limits2[dim, 0], limits2[dim, 1]

            # Two ranges overlap if: max(start1, start2) < min(end1, end2)
            # This excludes touching at boundaries (one-node contact)
            if max(start1, start2) >= min(end1, end2):
                # No overlap in this dimension, so patches don't overlap
                logger.debug(
                    f"No overlap in dimension {dim}: ranges ({start1}, {end1}) and ({start2}, {end2})"
                )
                return False

        logger.debug("Patches overlap.")
        return True

    def append(self, patch):
        """Attach patch to the block, validate limits and overlaps, then add."""
        patch.attach_to_block(self._block)

        for existing_patch in self._items:
            if self._patches_overlap(patch, existing_patch):
                raise ValueError(
                    f"Patch of type {type(patch).__name__} overlaps with existing patch "
                    f"(new: {patch._ijk_lim.flatten()}, existing: {existing_patch._ijk_lim.flatten()})"
                )

        super().append(patch)

    def extend(self, patches):
        """Attach each patch to the block, validate limits and overlaps, then add all."""
        for patch in patches:
            patch.attach_to_block(self._block)

        for i, patch in enumerate(patches):
            for existing_patch in self._items:
                if self._patches_overlap(patch, existing_patch):
                    raise ValueError(
                        f"Patch of type {type(patch).__name__} overlaps with existing patch "
                        f"(new: {patch._ijk_lim.flatten()}, existing: {existing_patch._ijk_lim.flatten()})"
                    )

            for j in range(i):
                if self._patches_overlap(patch, patches[j]):
                    raise ValueError(
                        f"Patch of type {type(patch).__name__} overlaps with another patch in the same batch "
                        f"(patch {i}: {patch._ijk_lim.flatten()}, patch {j}: {patches[j]._ijk_lim.flatten()})"
                    )

        super().extend(patches)

    def insert(self, index, patch):
        """Attach patch to the block, validate limits and overlaps, then insert at index."""
        patch.attach_to_block(self._block)

        for existing_patch in self._items:
            if self._patches_overlap(patch, existing_patch):
                raise ValueError(
                    f"Patch of type {type(patch).__name__} overlaps with existing patch "
                    f"(new: {patch._ijk_lim.flatten()}, existing: {existing_patch._ijk_lim.flatten()})"
                )

        super().insert(index, patch)

    @property
    def cooling(self):
        """All :py:class:`~ember.cooling.CoolingPatch` objects."""
        from ember.patch import CoolingPatch

        return [p for p in self._items if isinstance(p, CoolingPatch)]

    @property
    def cusp(self):
        """All :py:class:`~ember.cusp.CuspPatch` objects."""
        import ember.patch

        return [p for p in self._items if isinstance(p, ember.cusp.CuspPatch)]

    @property
    def inlet(self):
        """All :py:class:`~ember.inlet.InletPatch` objects."""
        from ember.patch import InletPatch

        return [p for p in self._items if isinstance(p, InletPatch)]

    @property
    def inlet_nonreflecting(self):
        """All :py:class:`~ember.inlet_nonreflecting.NonReflectingInletPatch` objects."""
        from ember.patch import NonReflectingInletPatch

        return [p for p in self._items if isinstance(p, NonReflectingInletPatch)]

    @property
    def inviscid(self):
        """All :py:class:`~ember.inviscid.InviscidPatch` objects."""
        from ember.patch import InviscidPatch

        return [p for p in self._items if isinstance(p, InviscidPatch)]

    @property
    def mixing(self):
        """All :py:class:`~ember.mixing.MixingPatch` objects."""
        from ember.patch import MixingPatch

        return [p for p in self._items if isinstance(p, MixingPatch)]

    @property
    def outlet(self):
        """All :py:class:`~ember.outlet.OutletPatch` objects."""
        from ember.patch import OutletPatch

        return [p for p in self._items if isinstance(p, OutletPatch)]

    @property
    def periodic(self):
        """All :py:class:`~ember.periodic.PeriodicPatch` objects."""
        from ember.patch import PeriodicPatch

        return [p for p in self._items if isinstance(p, PeriodicPatch)]

    @property
    def permeable(self):
        """Patches through which flow passes (non-wall faces).

        Includes :py:class:`~ember.inlet.InletPatch`,
        :py:class:`~ember.outlet.OutletPatch`,
        :py:class:`~ember.periodic.PeriodicPatch`,
        :py:class:`~ember.mixing.MixingPatch`,
        :py:class:`~ember.nonmatch.NonMatchPatch`, and
        :py:class:`~ember.cusp.CuspPatch`.
        Used to identify which boundary faces are not solid walls when
        computing block boundary fluxes.
        """
        import ember.patch

        return [p for p in self._items if isinstance(p, ember.patch.PERMEABLE_TYPES)]

    @property
    def probe(self):
        """All :py:class:`~ember.probe.ProbePatch` objects."""
        from ember.patch import ProbePatch

        return [p for p in self._items if isinstance(p, ProbePatch)]

    @property
    def rotating(self):
        """All :py:class:`~ember.rotating.RotatingPatch` objects."""
        from ember.patch import RotatingPatch

        return [p for p in self._items if isinstance(p, RotatingPatch)]

    @property
    def slip(self):
        """Patches that impose no friction (permeable faces and inviscid walls).

        Union of :py:attr:`permeable` and
        :py:class:`~ember.inviscid.InviscidPatch`.
        Used when applying viscous wall functions: faces in this set are
        treated as frictionless so no friction is applied at those boundaries.
        """
        import ember.patch

        return [p for p in self._items if isinstance(p, ember.patch.SLIP_TYPES)]

    # BlockPatchCollection-specific methods


class GridPatchCollection:
    """Read-only aggregate view of all patches across every block in a :py:class:`~ember.grid.Grid`.

    Holds a reference to the parent grid and presents all per-block patches as
    a single flat sequence. Supports integer indexing, slicing, iteration, and
    ``len``, as well as the same type-grouped properties as
    :py:class:`BlockPatchCollection` (``inlet``, ``outlet``, ``periodic``,
    etc.). String-key access and mutation methods (``append``, ``extend``,
    ``insert``, ``del``) are not available. Normally accessed via
    :py:attr:`Grid.patches <ember.grid.Grid.patches>` rather than instantiated
    directly.
    """

    def __init__(self, grid):
        """Initialise with a reference to the parent grid.

        Parameters
        ----------
        grid : :py:class:`~ember.grid.Grid`
            The grid whose block patch collections this view aggregates.
        """
        self._grid = grid

    def _get_all_patches(self):
        """Get flattened list of all patches from all blocks."""
        all_patches = []
        for block in self._grid:
            all_patches.extend(block.patches)
        return all_patches

    def __len__(self):
        """Return total number of patches across all blocks."""
        return sum(len(block.patches) for block in self._grid)

    def __iter__(self):
        """Iterate over all patches from all blocks."""
        for block in self._grid:
            for patch in block.patches:
                yield patch

    def __getitem__(self, index):
        """Get patch by numeric index or slice (no string access)."""
        if isinstance(index, slice):
            # Handle slice access by getting all patches first
            all_patches = self._get_all_patches()
            return all_patches[index]
        elif isinstance(index, int):
            # Handle negative indices by converting to positive
            total_patches = len(self)
            if index < 0:
                index = total_patches + index

            # Check bounds after conversion
            if index < 0 or index >= total_patches:
                raise IndexError("patch index out of range")

            # Handle single index access efficiently
            current_index = 0
            for block in self._grid:
                block_patch_count = len(block.patches)
                if current_index <= index < current_index + block_patch_count:
                    # Index is in this block
                    local_index = index - current_index
                    return block.patches[local_index]
                current_index += block_patch_count
            # Should not reach here after bounds check
            raise IndexError("patch index out of range")
        else:
            raise TypeError("Index must be int or slice (string access not supported)")

    def __contains__(self, patch):
        """Check if patch object is in any block."""
        for block in self._grid:
            if patch in block.patches:
                return True
        return False

    def __repr__(self):
        """String representation of grid patch collection."""
        total = len(self)
        periodic_count = len(self.periodic)
        inlet_count = len(self.inlet)
        outlet_count = len(self.outlet)

        return (
            f"GridPatchCollection(total={total}, "
            f"periodic={periodic_count}, inlet={inlet_count}, "
            f"outlet={outlet_count})"
        )

    @property
    def cooling(self):
        """Return all cooling patches from all blocks."""
        # Import here to avoid circular import
        cooling_patches = []
        for block in self._grid:
            cooling_patches.extend(block.patches.cooling)
        return cooling_patches

    @property
    def inlet(self):
        """Return all inlet patches from all blocks."""
        # Import here to avoid circular import
        inlet_patches = []
        for block in self._grid:
            inlet_patches.extend(block.patches.inlet)
        return inlet_patches

    @property
    def inlet_nonreflecting(self):
        """Return all non-reflecting inlet patches from all blocks."""
        inlet_patches = []
        for block in self._grid:
            inlet_patches.extend(block.patches.inlet_nonreflecting)
        return inlet_patches

    @property
    def mixing(self):
        """Return all mixing patches from all blocks."""
        # Import here to avoid circular import
        mixing_patches = []
        for block in self._grid:
            mixing_patches.extend(block.patches.mixing)
        return mixing_patches

    @property
    def outlet(self):
        """Return all outlet patches from all blocks."""
        # Import here to avoid circular import
        outlet_patches = []
        for block in self._grid:
            outlet_patches.extend(block.patches.outlet)
        return outlet_patches

    @property
    def periodic(self):
        """Return all periodic patches from all blocks."""
        # Import here to avoid circular import
        periodic_patches = []
        for block in self._grid:
            periodic_patches.extend(block.patches.periodic)
        return periodic_patches

    @property
    def probe(self):
        """Return all probe patches from all blocks."""
        probe_patches = []
        for block in self._grid:
            probe_patches.extend(block.patches.probe)
        return probe_patches

    @property
    def rotating(self):
        """Return all rotating patches from all blocks."""
        # Import here to avoid circular import
        rotating_patches = []
        for block in self._grid:
            rotating_patches.extend(block.patches.rotating)
        return rotating_patches

    # Read-only collection interface
