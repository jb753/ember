Geometry and indexing
=====================

Indexing
--------

For a block whose nodes have shape :math:`(n_i, n_j, n_k)`, all arrays take
one of three shapes depending on where the quantity is located.

.. list-table::
   :header-rows: 1
   :widths: 18 28 34 20

   * - Location
     - Shape
     - Description
     - Examples
   * - Node
     - :math:`(n_i,\; n_j,\; n_k)`
     - Vertices at corners of cell control volumes
     - | Coordinates
       | Conserved variables
   * - Cell
     - :math:`(n_i-1,\; n_j-1,\; n_k-1)`
     - Hexahedral cell :math:`(i,j,k)` is enclosed by nodes :math:`i, i{+}1`,  :math:`j, j{+}1`,  :math:`k, k{+}1`
     - | Volumes
       | Residual
   * - :math:`i`-face
     - :math:`(n_i,\; n_j-1,\; n_k-1)`
     - Constant-:math:`i` boundaries of each cell
     - | Face areas
       | Fluxes
   * - :math:`j`-face
     - :math:`(n_i-1,\; n_j,\; n_k-1)`
     - Constant-:math:`j` boundaries of each cell
     - | Face areas
       | Fluxes
   * - :math:`k`-face
     - :math:`(n_i-1,\; n_j-1,\; n_k)`
     - Constant-:math:`k` boundaries of each cell
     - | Face areas
       | Fluxes

.. tikz:: _tikz/cell_indexing.tikz
   :alt: Diagram of a single hexahedral cell showing node indices at each corner and the three face types (i-face, j-face, k-face) with their outward normals.
   :align: center
   :width: 90%

Multi-component quantities append a trailing dimension :math:`m` so that the
component index varies fastest, matching the column-major (contiguous last
index) layout of the underlying Fortran arrays.  For example, nodal velocity
has shape :math:`(n_i,\; n_j,\; n_k,\; 3)` with the coordinate direction on
the last axis, and cell residuals have shape
:math:`(n_i-1,\; n_j-1,\; n_k-1,\; 5)` with the equation index on the last
axis.

Coordinate system
-----------------

Grids are described in polar coordinates :math:`(x, r, \theta)` where :math:`x`
is the axial direction, :math:`r` the radial direction, and :math:`\theta` the
circumferential angle measured clockwise when looking upstream, giving a
*left-handed* system opposite to the right-handed convention common in general
CFD codes.  A consequence is that cell volumes computed from the divergence
theorem are positive when the index triple :math:`(i, j, k)` is a left-handed
set, i.e.\ when :math:`i`, :math:`j`, and :math:`k` increase in the
:math:`x`, :math:`r`, and :math:`\theta` directions respectively.


Area and volume calculations are done in pseudo-Cartesian space :math:`(x, r,
r\theta)` which locally linearises the circumferential metric and allows
standard vector identities to be applied.

.. _face-areas:

Face areas
~~~~~~~~~~

A face of a hexahedral cell is a quadrilateral with four corner nodes
:math:`A, B, C, D`, which in general are *not* coplanar.  Its area vector
:math:`\delta\mathbf{A}` is defined so that each component is
the signed area of the projection of the quadrilateral on to the plane
perpendicular to that component's direction. We calculate the area vector using
Gauss's theorem on a vector field with known divergence as explained below.

The four nodes are taken in the order below, circulating around the face so
that the resulting area vector points along the increasing index direction.

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 22 22

   * - Face
     - :math:`A`
     - :math:`B`
     - :math:`C`
     - :math:`D`
   * - :math:`i`
     - :math:`(i,\,j,\,k)`
     - :math:`(i,\,j,\,k{+}1)`
     - :math:`(i,\,j{+}1,\,k{+}1)`
     - :math:`(i,\,j{+}1,\,k)`
   * - :math:`j`
     - :math:`(i,\,j,\,k)`
     - :math:`(i{+}1,\,j,\,k)`
     - :math:`(i{+}1,\,j,\,k{+}1)`
     - :math:`(i,\,j,\,k{+}1)`
   * - :math:`k`
     - :math:`(i,\,j,\,k)`
     - :math:`(i,\,j{+}1,\,k)`
     - :math:`(i{+}1,\,j{+}1,\,k)`
     - :math:`(i{+}1,\,j,\,k)`

The pattern is the same in each row: going :math:`A \to B` advances the index
that cyclically precedes the face's own index, and :math:`B \to C` advances the
one that follows it.  All three circulations therefore turn the same way, and
the sum below gives an area vector along the increasing index direction
directly, with no sign correction.

Note that this is a direction convention, not an orientation convention: a face
area vector is not outward with respect to any particular cell.  For cell
:math:`(i,j,k)` the vectors of its three lower faces point inwards and those of
its three upper faces point outwards.

The steps are as follows.

**1. Convert to pseudo-Cartesian coordinates.**  Each node
:math:`(x, r, \theta)` becomes :math:`(x,\, r,\, r\theta)`.  The
circumferential angles are first shifted by the mean angle of the four nodes,
so that :math:`\theta` is measured from the centre of the face.  This keeps
:math:`r\theta` small.

**2. Shift to the face centre.**  The mean of the four pseudo-Cartesian node
positions is subtracted from each, so that the face centre is the origin
Working with small coordinates relative to the face,
rather than large coordinates relative to the machine axis, avoids
catastrophic cancellation when differencing nearby nodes.

**3. Form edge vectors and midpoints.**  For each of the four edges
:math:`e` of the perimeter :math:`A \to B \to C \to D \to A`, take the vector
between its endpoints :math:`\delta\mathbf{x}_e` and the midpoint
:math:`\mathbf{x}_e`.

**4. Sum around the perimeter.**  Each component :math:`d` of the area vector
is

.. math::

    \delta A_d = \tfrac{1}{2}
        \sum_{e=1}^{4} \mathbf{F}_d(\mathbf{x}_e) \cdot \delta\mathbf{n}_{ed}

The vector field :math:`\mathbf{F}_d` has its :math:`d`-th component set to
zero and the other two equal to the corresponding coordinates, so that within
the projection plane :math:`\nabla \cdot \mathbf{F}_d = 2`.  The edge normal
:math:`\delta\mathbf{n}_{ed}` is the edge vector projected on to that same
plane and rotated by a quarter turn to point out of the polygon, with length
equal to that of the projected edge.  It therefore depends on :math:`d` as well
as on :math:`e`, being zero along :math:`d` and in-plane otherwise.

Suppressing the edge index, and writing :math:`\delta x`, :math:`\delta r` and
:math:`\delta r\theta` for the components of the edge vector :math:`\delta\mathbf{x}_e`,

.. math::

    \begin{aligned}
    \mathbf{F}_x         &= (0,\; r,\; r\theta)
        & \delta\mathbf{n}_x &= (0,\; -\delta r\theta,\; \delta r) \\
    \mathbf{F}_r         &= (x,\; 0,\; r\theta)
        & \delta\mathbf{n}_r &= (\delta r\theta,\; 0,\; -\delta x) \\
    \mathbf{F}_{r\theta} &= (x,\; r,\; 0)
        & \delta\mathbf{n}_{r\theta} &= (-\delta r,\; \delta x,\; 0)
    \end{aligned}

So :math:`\mathbf{F}_d \cdot \delta\mathbf{n}_{ed}` reduces to the cross product
of the two in-plane components of :math:`\mathbf{x}_e` and
:math:`\delta\mathbf{x}_e`.  Restoring the edge index, and writing :math:`x_e`,
:math:`r_e` and :math:`r\theta_e` for the coordinates of the midpoint of edge
:math:`e`, the axial component is, for example,

.. math::

    \delta A_x = \tfrac{1}{2} \sum_{e=1}^{4}
        \bigl( r\theta_e\, \delta r_e
             - r_e\, \delta r\theta_e \bigr)

Because the nodes were shifted so that the face centre is the origin, each
term of this sum is the signed area of the triangle joining the face centre to
edge :math:`e`.  The four triangles tile the projected quadrilateral, and the
signed areas cancel correctly for a non-convex projection.  The figure below
shows this for the axial component of an :math:`i` face; the radial and
circumferential components follow by projecting on to the other two planes.

.. tikz:: _tikz/face_area.tikz
   :alt: Two-panel diagram. Left, a warped quadrilateral i face with corner nodes A, B, C, D, edge vectors, and its centre marked as the origin. Right, the same face projected on to the r-rtheta plane, divided into four triangles radiating from the centre, with one triangle highlighted to show the edge midpoint, edge vector, and outward edge normal that make up its contribution to the area.
   :align: center
   :width: 95%

Nothing in this construction assumes the four nodes are coplanar.  Carrying the
sum through algebraically collapses it to a cross product of the two diagonals,

.. math::

    \delta\!\mathbf{A} = \tfrac{1}{2}\,
        \overrightarrow{BD} \times \overrightarrow{AC}

with the diagonals taken in the pseudo-Cartesian coordinates of step 1, so a
warped face is treated as if the corners were joined by straight diagonals.
Both sides are built only from differences of node positions, so the shift in
step 2 leaves the result unchanged; it is a round-off measure, not part of the
definition.  It does, however, fix the decomposition into triangles above, which
is only a sum of areas about the face centre once that centre is the origin.
The code evaluates the perimeter sum rather than this identity
because the shift to the face centre in step 2 keeps the round-off error small.

Collapsing the quadrilateral to a triangle by letting :math:`D \to A` recovers
the area vector of a triangular face with vertices :math:`A, B, C`,

.. math::

    \delta\!\mathbf{A} = \tfrac{1}{2}\,
        \overrightarrow{AC} \times \overrightarrow{AB}

.. _cell-volumes:

Cell volumes
~~~~~~~~~~~~

Cell volumes are obtained from the divergence theorem.  Using the vector
field :math:`\mathbf{F} = (x,\, r/2,\, r\theta)`, for which
:math:`\nabla \cdot \mathbf{F} = 3` in cylindrical coordinates, the
volume of a cell is

.. math::

    \delta\mathcal{V} = \frac{1}{3}
        \oint \mathbf{F} \cdot \mathrm{d}\mathbf{A}
      = \frac{1}{3} \sum_{d \,\in\, \{i,\,j,\,k\}}
        \bigl( \mathbf{F}_{d}^{+} \cdot \delta\mathbf{A}_{d}^{+}
             - \mathbf{F}_{d}^{-} \cdot \delta\mathbf{A}_{d}^{-} \bigr)

where :math:`\mathbf{F}_f` is evaluated at the face centre, taken as the
average of the four corner nodes of each face, and the superscripts
:math:`-` and :math:`+` denote the lower and upper face of the cell in each
index direction.  The lower face contributions are subtracted because, by the
convention of the previous section, every :math:`\delta\mathbf{A}` points along
the increasing index direction and so the lower face vectors point into the
cell rather than out of it.

