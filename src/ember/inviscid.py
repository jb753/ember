"""Inviscid wall boundary condition patch for EMBER CFD.

InviscidPatch represents an impermeable but frictionless boundary condition,
typically used for slip walls in inviscid flow calculations.

See Also
--------
ember.patch.Patch : Base class for all patches
"""

from ember.basepatch import Patch


class InviscidPatch(Patch):
    """Frictionless slip-wall boundary.

    Marks a face as impermeable but frictionless. The patch is included in the
    ``slip`` collection, which causes the Fortran flux routines to apply a
    slip (zero normal-velocity) condition without viscous stress. Use this
    instead of a no-slip wall when viscous effects on that face should be
    suppressed, for example on a symmetry plane or an inviscid endwall.
    """

    _collection_name = "inviscid"
