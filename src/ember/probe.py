"""Flow-sampling probe patch for EMBER CFD.

ProbePatch marks a point, face, or interior plane where flow history is
recorded. It is a passive overlay: it does not change the boundary condition of
the face it samples, may sit on an interior plane, and is exempt from overlap
checks so it can coincide with another patch. It is not currently wired into
ember's solver and is used only for Turbostream 3 file round-tripping.

See Also
--------
ember.patch.Patch : Base class for all patches
"""

from ember.basepatch import Patch


class ProbePatch(Patch):
    """Passive flow-sampling probe (Turbostream 3 kind 8).

    Records flow history at a point, face, or interior plane. Unlike boundary
    patches it does not affect wall/slip treatment of the sampled face -- it is
    deliberately excluded from ``PERMEABLE_TYPES``/``SLIP_TYPES``. The two class
    flags below relax the generic patch rules so a probe can be placed on an
    interior constant plane and may coincide with another patch.
    """

    _collection_name = "probe"
    _allow_interior_const = True  # permit interior const-plane region probes
    _allow_overlap = True  # may coincide with another patch
