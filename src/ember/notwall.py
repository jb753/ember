"""Collapsed-face marker.

NotWallPatch marks a block face that is not a boundary at all, because it has
no area: the grid lines meeting it have converged to a line or a point.

See Also
--------
ember.patch.Patch : Base class for all patches

"""

from ember.basepatch import Patch


class NotWallPatch(Patch):
    """A face with no area, and so no boundary condition.

    A structured block can be degenerate at one end of a direction: the grid
    lines converge and the face collapses onto a line. The prism filling a
    blade tip clearance is the case this exists for --- its streamwise ends sit
    at the leading edge and at the trailing edge or cusp tip, where the two
    surfaces of the section meet.

    Such a face carries no flux, the flux being area-weighted and the area
    zero, so it needs no boundary condition and takes no pair. What it does
    need is to be told apart from a wall. Every face without a patch is a wall,
    which is what makes a mesher's job small, but a collapsed face is the one
    kind that is neither a wall nor anywhere flow passes. Left unmarked it
    would seed the wall-distance search, putting a line of zero wall distance
    through the middle of the flow, and be handed to a wall function that
    divides the cell volume by the face area.

    So this is a marker rather than a condition. It is in ``PERMEABLE_TYPES``,
    which is the one place a face is classified, and has no communicator and no
    data of its own.
    """
