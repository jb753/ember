Geometry and indexing
=====================

Indexing
--------

For a block whose nodes have shape ``(ni, nj, nk)``, all arrays take
one of five shapes depending on where the quantity is located.

.. list-table::
   :header-rows: 1
   :widths: 16 30 34 20

   * - Location
     - Shape
     - Description
     - Examples
   * - Node
     - ``(ni, nj, nk)``
     - Vertices at corners of cell control volumes
     - | Coordinates,
       | conserved variables
   * - Cell
     - ``(ni-1, nj-1, nk-1)``
     - Hexahedral cell ``(i,j,k)`` is enclosed by the nodes ``i, i+1, j, j+1, k, k+1``
     - | Volumes,
       | residual
   * - :math:`i`-face
     - ``(ni, nj-1, nk-1)``
     - Constant-:math:`i` boundaries of each cell
     - | Face areas,
       | fluxes
   * - :math:`j`-face
     - ``(ni-1, nj, nk-1)``
     - Constant-:math:`j` boundaries of each cell
     - | Face areas,
       | fluxes
   * - :math:`k`-face
     - ``(ni-1, nj-1, nk)``
     - Constant-:math:`k` boundaries of each cell
     - | Face areas,
       | fluxes

The indexing conventions are illustrated graphically in the following diagram,
showing node, cell, and face indices for a single hexahedral cell.

.. tikz:: _tikz/cell_indexing.tikz
   :alt: Diagram of a single hexahedral cell showing node indices at each corner and the three face types (i-face, j-face, k-face) with their outward normals.
   :align: center
   :width: 90%

Multi-component quantities append a trailing dimension so that the
each component is contiguous in memory with Fortran column-major ordering.
For example, nodal velocity vector
has shape ``(ni, nj, nk, 3)`` with the coordinate direction on
the last axis, and cell residuals have shape
``(ni-1, nj-1, nk-1, 5)`` with the equation index on the last
axis.

.. _coordinate-system:

Coordinate system
-----------------

Grids are described in cylindrical polar coordinates :math:`(x, r, \theta)`
where :math:`x` is the axial direction, :math:`r` the radial direction, and
:math:`\theta` the circumferential angle measured clockwise when looking
upstream, giving a *left-handed* system opposite to the right-handed convention
common in general CFD codes.  A consequence is that cell volumes computed from
the divergence theorem are positive when the index triple :math:`(i, j, k)` is
a left-handed set, i.e.\ when :math:`i`, :math:`j`, and :math:`k` increase in
the :math:`x`, :math:`r`, and :math:`\theta` directions respectively. The polar
system is related to a standard right-handed Cartesian frame by

.. math::

    y = r \cos\theta, \qquad z = -r \sin\theta

so that the :math:`\theta = 0` datum lies along :math:`+y` and the minus sign
on :math:`z` produces the clockwise sense of increasing :math:`\theta`, as
illustrated in the diagram.

.. tikz:: _tikz/coordinate_system.tikz
   :alt: Left-handed polar coordinate system viewed looking upstream, showing the axial x, radial r, and circumferential r-theta directions at a point, with a circular arrow marking the clockwise sense of increasing theta.
   :align: center
   :width: 70%


.. _face-areas:

Face areas
----------

A face of a hexahedral cell is a quadrilateral with four corner nodes :math:`A,
B, C, D`, which in general are *not* coplanar.  Its area vector
:math:`\delta\mathbf{A}` is defined so that each component is the signed area
of the projection of the quadrilateral perpendicular to that component's
direction.

First, we convert to pseudo-Cartesian coordinates. Subtract the mean angle of all
four nodes so that :math:`\theta` is measured from the centre of the face. This
allows us to locally linearise the circumferential direction by
replacing :math:`\theta` with :math:`r\theta` to give a pseudo-Cartesian
coordinate system and use standard vector operations. Then, we cross the diagonals of the quadrilateral to get the area vector.

.. math::

    \delta\mathbf{A} = \tfrac{1}{2}\,
        \overrightarrow{BD} \times \overrightarrow{AC}

Note that the sign convention is index-aligned. For example,
:math:`\delta\mathbf{A}_i` is positive when the face normal points in the
increasing :math:`i` direction.  The area vector is shared between cells and so
is not outward with respect to any particular cell.

.. _cell-volumes:

Cell volumes
------------

Cell volumes are obtained from the divergence theorem applied to the vector
field :math:`\mathbf{F} = (x,\, r/2,\, r\theta)`, for which
:math:`\nabla \cdot \mathbf{F} = 3` in cylindrical coordinates.
The angle :math:`\theta` in :math:`\mathbf{F}` is measured from the mean angle
of the eight corner nodes of the cell. Although the origin of :math:`\theta` is
arbitrary, this choice reduces round-off error and guarantees that the volume
is independent of circumferential shifts of the cell.

The volume of a cell is

.. math::

    \delta\mathcal{V} = \frac{1}{3}
        \oint \mathbf{F} \cdot \mathrm{d}\mathbf{A}
      = \frac{1}{3} \sum_{d \,\in\, \{i,\,j,\,k\}}
        \bigl( \mathbf{F}_{d}^{+} \cdot \delta\mathbf{A}_{d}^{+}
             - \mathbf{F}_{d}^{-} \cdot \delta\mathbf{A}_{d}^{-} \bigr)

where :math:`\mathbf{F}` is evaluated at the face centre, taken as the
average of the four corner nodes of each face, and the superscripts
:math:`\mp` denote the lower and upper face of the cell in each
index direction.  The lower face contributions are subtracted because, by the
convention of the previous section, every :math:`\delta\mathbf{A}` points along
the increasing index direction and so the lower face vectors point into the
cell rather than out of it.
