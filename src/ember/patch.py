"""Patch classes for specifying boundary conditions on structured blocks.

Limiting index rules
--------------------

Patches are defined by specifying which block face or part of a face they are
on. Every :py:class:`~ember.basepatch.Patch` subclass is constructed as::

    PatchType(i=..., j=..., k=..., label=...)

taking one argument for each of the three indexing directions ``i``, ``j``,
and ``k``, subject to the following rules:

* The first point in a direction is indexed 0; negative indices wrap around
  such that -1 is the last point.
* Indices are inclusive, so ``i=(0,-1)`` spans the entire range of ``i``
  coordinate.
* Integer arguments are interpreted as a constant value of that index: ``i=0``
  means the patch spans the first i face; ``j=-1`` means the patch spans the
  last j face. Integer arguments are shorthand for, e.g. ``i=(0,0)``.
* Patches must be 2D subsets of an external face of the block. This implies
  that at least one constant dimension must be specified with a value 0 or -1.
* Omitting a direction argument implies the patch should include every point in
  that direction, and is shorthand for e.g. ``j=(0, -1)``.
* The elements of a direction tuple should be in ascending order after negative
  indices are wrapped. ``k=(6,4)`` is not valid, and neither is ``k=(-1, -2)``.

Types of patches
----------------

The different types of patches (e.g. periodic, inlet, outlet) are represented
by subclasses of the abstract base class :py:class:`~ember.basepatch.Patch`,
but are all initialised by passing in the limiting indices and an optional
label for later debugging. Storing boundary condition information or matching
connections between patches is handled by methods on the subclasses.

Storage and attachment to blocks
---------------------------------

A patch can be constructed in isolation without a :py:class:`~ember.block.Block`.
However, any property that depends on block geometry -- resolving negative
indices to absolute coordinates, computing patch size, or accessing block
coordinate views -- will raise an error until the patch is attached to a block.

Patches are stored in a :py:class:`~ember.collections.BlockPatchCollection` accessible at ``block.patches``.
Adding a patch to this collection automatically attaches it to the block,
validates its limits against the block shape, and checks that it does not
spatially overlap with any existing patch of the same type on the same face.
The collection supports retrieval by integer index, by string label, and by
patch type::

    p = block.patches[0]           # by index
    p = block.patches["inlet_hub"] # by label
    ps = block.patches.inlet       # list of all InletPatch objects

The type-grouped properties (``inlet``, ``outlet``, ``periodic``,
``rotating``, etc.) each return a plain :py:class:`list`.

Mutation is through the standard collection interface: ``append``, ``extend``,
``insert``, and ``del``::

    block.patches.append(InletPatch(i=0, label="inlet_hub"))
    del block.patches["inlet_hub"]

:py:func:`len` returns the number of patches on the block.

A :py:class:`~ember.collections.GridPatchCollection` at ``grid.patches`` provides a corresponding
read-only aggregate view across all blocks in a
:py:class:`~ember.grid.Grid`. It supports integer indexing, slicing,
iteration, :py:func:`len`, and the same type-grouped properties (e.g.
``grid.patches.periodic``), but does not support string-key access or any
mutation methods.

Examples
--------

::

    # example: patch_examples
    import numpy as np
    import ember.patch
    from ember.block import Block
    from ember.fluid import PerfectFluid

    # Build a block with axial (x), radial (r), and circumferential (t) coordinates.
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    block = Block(shape=(4, 5, 6))
    block.set_fluid(fluid)
    block.set_x(np.linspace(0.0, 1.0, 4).reshape(-1, 1, 1) * np.ones((4, 5, 6)))
    block.set_r(np.linspace(0.5, 1.0, 5).reshape(1, -1, 1) * np.ones((4, 5, 6)))
    block.set_t(np.linspace(0.0, 0.5, 6).reshape(1, 1, -1) * np.ones((4, 5, 6)))

    # Add inlet, outlet, and periodic patches; appending attaches each to the block.
    inlet  = ember.patch.InletPatch(i=0,  label="inflow")
    outlet = ember.patch.OutletPatch(i=-1, label="outflow")
    block.patches.extend([inlet, outlet])
    block.patches.extend([
        ember.patch.PeriodicPatch(k=0,  label="lower"),
        ember.patch.PeriodicPatch(k=-1, label="upper"),
    ])

    print(len(block.patches))           # 4
    print(len(block.patches.periodic))  # 2

    # Patch shape and size are resolved against the block once attached.
    # shape is (1, 5, 6): one i-plane, full j and k extent.
    print(inlet.size)                        # 30
    print(inlet.shape)                       # (1, 5, 6)
    # const_dim is the index of the constant face (0=i, 1=j, 2=k).
    print(inlet.const_dim)                   # 0
    print(block.patches["lower"].const_dim)  # 2

    # Negative indices are stored as-is and resolved to absolute coordinates on demand.
    # i=(1,-2), k=(1,-2) on shape (4,5,6): i=-2 resolves to 2, k=-2 resolves to 4.
    partial = ember.patch.InviscidPatch(j=-1, i=(1, -2), k=(1, -2), label="tip_partial")
    block.patches.append(partial)
    print(partial.ijk_lim_abs[0, 1])  # 2
    print(partial.ijk_lim_abs[2, 1])  # 4
    print(partial.shape)              # (2, 1, 4)

    # Use the patch slice to index block coordinate arrays directly.
    print(block[inlet.slice].x.shape)   # (1, 5, 6)
    print(block.xrt[inlet.slice].shape) # (1, 5, 6, 3)

    # Set inlet stagnation conditions; values can be scalars or arrays
    # that broadcast to the patch shape.
    inlet.set_Po_To_Alpha_Beta(Po=2e5, To=1200.0, Alpha=0.0, Beta=0.0)
    print(inlet.Po)  # 200000.0
    print(inlet.To)  # 1200.0

    nj, nk = inlet.shape[1], inlet.shape[2]
    Po_array = np.linspace(1.8e5, 2.2e5, nj * nk).reshape(1, nj, nk)
    inlet.set_Po_To_Alpha_Beta(Po=Po_array)
    print(inlet.Po.shape)  # (1, 5, 6)

    # Set static pressure on the outlet, or attach a mass-flow PID throttle.
    outlet.set_P(1e5)
    print(outlet.P)  # 100000.0

    outlet.set_throttle(mdot_target=3.0, K_pid=(1.0, 0.1, 0.0))

    # Set angular velocity on a rotating wall patch.
    rot_patch = ember.patch.RotatingPatch(j=0, label="hub")
    block.patches.append(rot_patch)
    rot_patch.set_Omega(500.0)
    print(rot_patch.rpm)  # 4774.648

"""

from ember.basepatch import Patch, RevolutionPatch
from ember.collections import BlockPatchCollection, GridPatchCollection
from ember.cooling import CoolingPatch
from ember.cusp import CuspPatch
from ember.inlet import InletPatch
from ember.inviscid import InviscidPatch
from ember.mixing import MixingPatch
from ember.nonmatch import NonMatchPatch
from ember.outlet import OutletPatch
from ember.periodic import PeriodicPatch
from ember.probe import ProbePatch
from ember.rotating import RotatingPatch

__all__ = [
    "Patch",
    "RevolutionPatch",
    "PeriodicPatch",
    "InletPatch",
    "OutletPatch",
    "MixingPatch",
    "NonMatchPatch",
    "RotatingPatch",
    "CoolingPatch",
    "InviscidPatch",
    "CuspPatch",
    "ProbePatch",
    "BlockPatchCollection",
    "GridPatchCollection",
    "PERMEABLE_TYPES",
    "SLIP_TYPES",
]

PERMEABLE_TYPES = (
    InletPatch,
    OutletPatch,
    PeriodicPatch,
    MixingPatch,
    NonMatchPatch,
    CuspPatch,
)


SLIP_TYPES = PERMEABLE_TYPES + (InviscidPatch,)
