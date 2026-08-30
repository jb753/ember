r"""Storage and manipulation of flow field data for a single structured grid block.

This module defines the :class:`Block`, our fundamental data structure for representing flow fields on structured grids with any number of dimensions. The class stores coordinates and conserved quantities, and provides properties for derived quantities such as velocity, stagnation pressure, and Mach number. There is also a store for scalar metadata related to
the entire field, such as reference frame angular velocity. All data flows are managed through setter methods that ensure validity and consistency of the flow field. The class also stores boundary patches to specify simulation boundary conditions in :py:attr:`Block.patches`.

Initialisation
==============

The only required argument to the :class:`Block` constructor is the shape of the structured grid, which may be any number of dimensions::

    from ember.block import Block
    block = Block((ni, nj, nk, ...))  # an ND block

To begin with, an array is allocated to store the raw data only. Storage
for derived quantities is then allocated lazily on first access, and cached for
subsequent calls to save memory. Data and metadata are stored after initialisation using :ref:`block-setters`, and the raw and derived quantities are accessed via attributes such as :attr:`Block.x`, :attr:`Block.P`, and :attr:`Block.Ma`.

Indexing and slicing
====================

A :class:`Block` supports numpy-style indexing and slicing over the spatial axes::

    block[i]          # scalar index -- reduces ndim by one
    block[ist:ien]        # slice -- preserves ndim
    block[i, jst:jen, :]     # mixed index tuple for 3D data

Indexing returns a new :class:`Block` instance that shares the same underlying
backing array as the original (a zero-copy view). Writes to the indexing result
are visible in the original and vice versa.

.. _block-equations-of-state:

Equations of state
==================

:class:`Block` does not implement an equation of state itself.  It stores only
the conserved quantities at grid nodes and delegates every thermodynamic
relation to a :mod:`ember.fluid` equation of state attached by
:meth:`Block.set_fluid()`. The block works in terms of density and internal energy, and the fluid performs calculations to convert from other thermodynamic properties as needed.

Reading a thermodynamic property such as static pressure :attr:`Block.P` first extracts internal energy :attr:`Block.u` from the conserved quantities :attr:`Block.conserved` by subtracting kinetic energy.
Then, density and internal energy are passed to :py:meth:`ember.fluid.PerfectFluid.get_P` which evaluates the equation of state to calculate pressure. The result is stored in a cache array for repeated use, that is cleared if the underlying conserved data changes. Temperature, entropy, and so on follow this same pattern.

Writing a thermodynamic state is the reverse of reading out a derived property, although by the two-property rule the set methods must take two arguments.
:py:meth:`Block.set_P_T` passes pressure
and temperature to :py:meth:`ember.fluid.PerfectFluid.set_P_T`, which inverts the equation of state to find the corresponding density and internal energy.
:class:`Block` then saves density directly, and updates total energy to reflect the new thermodynamic state while preserving the velocity field.

This works even before any velocity has been set, because a new block starts
with dummy initial values for density, radius, momenta, and energy.
The kinetic energy therefore evaluates to zero on an uninitialised block, and
the thermodynamic round-trip stays consistent once velocities are later supplied.

.. _block-reference-scales:

Reference scales
================

Block non-dimensionalisation follows the scheme described in
:mod:`ember.fluid` with an additional length scale; see :ref:`reference-scales`.
Three base scales are chosen by the user and passed to the working fluid constructor:
:math:`\rho_\mathrm{ref}`, :math:`V_\mathrm{ref}`, and :math:`R_\mathrm{ref}`.
Three derived thermodynamic scales are then formed:
:math:`p_\mathrm{ref} = \rho_\mathrm{ref} V_\mathrm{ref}^2`,
:math:`u_\mathrm{ref} = V_\mathrm{ref}^2`, and
:math:`T_\mathrm{ref} = V_\mathrm{ref}^2 / R_\mathrm{ref}`.
All six are accessible via the attached fluid at :py:attr:`Block.fluid`.

Spatial coordinates are normalised by a separate reference length
:math:`L_\mathrm{ref}` [m], set via :py:meth:`Block.set_L_ref` and accessible
as :py:attr:`Block.L_ref`.  It defaults to 1.0, leaving supposedly
non-dimensional coordinates in SI units, and is independent of the fluid.

At rest, a :class:`Block` stores the raw data in non-dimensional form. Calls to, for example, :meth:`Block.set_P_T` and :meth:`Block.set_Vx` divide their dimensional input by the appropriate reference scale before storage. :meth:`Block.set_rho_u_Vxrt_nd` is the one exception to this rule as indicated by its `_nd` suffix: it takes non-dimensional inputs and stores them directly without rescaling.
Calls to
:meth:`Block.set_L_ref` and :meth:`Block.set_fluid` rescale the raw data in
place to maintain the same dimensional values if the reference scales change.
This keeps the non-dimensional storage completely transparent to the
user.

Non-dimensional versions of dimensional properties such as :attr:`Block.P_nd` and :attr:`Block.Vx_nd` have an `_nd` suffix to distinguish them from the dimensional versions. The same suffix also applies to setters which take non-dimensional inputs like :meth:`Block.set_P_rho_nd`.

Array methods
=============

A :class:`Block` provides a family of numpy-style array methods that reshape,
reorder, reduce or copy the block. They all act on the underlying *raw*
variables -- the coordinates and conserved quantities -- and not on derived
thermodynamic properties, which are recomputed from the transformed raw
data on the returned instance.

Views and copies:

.. autosummary::

   Block.copy
   Block.empty
   Block.freeze
   Block.masked
   Block.view

A frozen block is read-only: its setters raise and its backing array is marked
read-only, so views of it are read-only too. :attr:`Block.frozen` reports
whether a block is in that state, and :meth:`Block.copy` returns a writeable
one again.

Reshaping and reordering (a zero-copy view where the layout allows, otherwise a copy):

.. autosummary::

   Block.flip
   Block.reshape
   Block.squeeze
   Block.transpose

Reduction over a spatial axis:

.. autosummary::

   Block.mean
   Block.nanmean

Cache:

Methods that bypass the usual lazy, per-property cache invalidation -- see
:meth:`Block.update_cached_conserved` and :meth:`Block.update_primitive` for
when each is needed.

.. autosummary::

   Block.clear_cache
   Block.update_cached_conserved
   Block.update_primitive

Diagnostics:

.. autosummary::

   Block.memory_usage

.. _block-setters:

Setter methods
==============

All writes to a :class:`Block` go through a setter method, which validates the
input, non-dimensionalises it (see :ref:`block-reference-scales`), and
invalidates any cached derived quantities that depend on it. The setters are:

Geometry:

.. autosummary::

   Block.set_r
   Block.set_t
   Block.set_wdist
   Block.set_x
   Block.set_xrt
   Block.set_xyz

Kinematics:

.. autosummary::

   Block.set_V_Alpha_Beta
   Block.set_Vr
   Block.set_Vt
   Block.set_Vx
   Block.set_Vxrt

Thermodynamic state:

By the two-property rule, each of these takes two independent properties and
inverts the equation of state to recover density and internal energy, leaving
the velocity field untouched. See :ref:`block-equations-of-state` for details.

.. autosummary::

   Block.set_h_s
   Block.set_P_h
   Block.set_P_rho
   Block.set_P_rho_nd
   Block.set_P_s
   Block.set_P_T
   Block.set_rho_s
   Block.set_rho_u
   Block.set_T_s

Combined:

Five independent properties are enough to fully specify the flow field.

.. autosummary::

   Block.set_conserved
   Block.set_rho_u_Vxrt_nd

Metadata:

Scalar properties of the field as a whole, rather than per-node data. The first
two are exceptions: they rescale the raw data in place so that dimensional
values are preserved when the reference scales change.

.. autosummary::

   Block.set_fluid
   Block.set_L_ref
   Block.set_label
   Block.set_Nb
   Block.set_Omega
   Block.set_rpm
   Block.set_triangulated

Miscellaneous:

.. autosummary::

   Block.set_mu_turb

.. _block-properties:

Properties
==========

Raw and derived quantities are read back via properties. Nodal arrays have
shape matching :attr:`Block.shape`; cell and face quantities are one node
shorter along the relevant axis or axes.

Geometry:

.. autosummary::

   Block.dA_quad
   Block.dA_tri
   Block.dAi
   Block.dAj
   Block.dAk
   Block.r
   Block.rt
   Block.t
   Block.vol
   Block.wdist
   Block.x
   Block.xr
   Block.xrrt
   Block.xrt
   Block.y
   Block.z

Kinematics:

.. autosummary::

   Block.Alpha
   Block.Alpha_rel
   Block.Beta
   Block.sinBeta
   Block.tanAlpha
   Block.tanAlpha_rel
   Block.tanBeta
   Block.U
   Block.V
   Block.V_rel
   Block.Vm
   Block.Vr
   Block.Vt
   Block.Vt_rel
   Block.Vx
   Block.Vxrt
   Block.Vxrt_rel
   Block.Vy
   Block.Vz

Thermodynamic state:

Pure equation-of-state outputs and transport properties, evaluated from
:attr:`Block.rho` and :attr:`Block.u` alone -- see :ref:`block-equations-of-state`.

.. autosummary::

   Block.a
   Block.cp
   Block.cv
   Block.gamma
   Block.h
   Block.kappa
   Block.mu
   Block.P
   Block.Pr
   Block.Rgas
   Block.rho
   Block.s
   Block.T
   Block.u

Combined:

Quantities that mix thermodynamic state with velocity or rotation --
stagnation properties, Mach numbers, rothalpy, mass flux -- and the
conserved variables themselves.

.. autosummary::

   Block.ao
   Block.conserved
   Block.ho
   Block.ho_rel
   Block.I
   Block.Ma
   Block.Ma_rel
   Block.Mam
   Block.Max
   Block.mu_turb
   Block.P_rot
   Block.Po
   Block.Po_rel
   Block.rhoe
   Block.rhoo
   Block.rhoo_rel
   Block.rhorVt
   Block.rhoVm
   Block.rhoVr
   Block.rhoVx
   Block.To
   Block.To_rel
   Block.uo
   Block.uo_rel

Grid shape and array metadata:

.. autosummary::

   Block.flat
   Block.frozen
   Block.ndim
   Block.ni
   Block.nj
   Block.nk
   Block.nvar
   Block.shape
   Block.shape_cell
   Block.shape_iface
   Block.shape_jface
   Block.shape_kface
   Block.size

Metadata:

.. autosummary::

   Block.fluid
   Block.L_ref
   Block.label
   Block.Nb
   Block.Omega
   Block.patches
   Block.pitch
   Block.rpm
   Block.triangulated

Miscellaneous:

.. autosummary::

   Block.i_cusp
   Block.i_perk
   Block.ijk_wall_conv
   Block.ijk_wall_visc
   Block.scratch
   Block.store
   Block.tau_q_faces

Nondimensional:

Every dimensional quantity above (plus a handful of solver-only quantities)
has a nondimensional counterpart with an `_nd` suffix; see
:ref:`block-reference-scales`. These back the dimensional properties
directly and are not usually needed by end users.

.. autosummary::

   Block.a_nd
   Block.conserved_avg_nd
   Block.conserved_filt_nd
   Block.conserved_nd
   Block.cp_nd
   Block.dA_quad_nd
   Block.dA_tri_nd
   Block.dAi_nd
   Block.dAj_nd
   Block.dAk_nd
   Block.dhdP_rho_nd
   Block.dhdrho_P_nd
   Block.dsdP_rho_nd
   Block.dsdrho_P_nd
   Block.dt_vol_nd
   Block.dudP_rho_nd
   Block.dudrho_P_nd
   Block.F_body_nd
   Block.ho_nd
   Block.kappa_nd
   Block.mu_nd
   Block.Omega_nd
   Block.Omega_wall_nd
   Block.P_nd
   Block.P_offset_nd
   Block.r_mid_nd
   Block.r_nd
   Block.residual_nd
   Block.rho_nd
   Block.s_nd
   Block.T_nd
   Block.u_nd
   Block.V_nd
   Block.vol_nd
   Block.Vr_nd
   Block.Vt_nd
   Block.Vt_rel_nd
   Block.Vx_nd
   Block.Vxrt_nd
   Block.wdist_nd
   Block.xrt_nd



Example usage
=============

Construct a scalar block, set coordinates, fluid, thermodynamic state,
and velocity::

    # example: construct
    from ember.block import Block
    from ember.fluid import PerfectFluid
    import numpy as np

    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
    b = Block()
    b.set_fluid(fluid)
    b.set_x(0.0)
    b.set_r(0.75)
    b.set_t(0.0)
    b.set_P_T(1e5, 300.0)
    b.set_Vx(100.0)
    b.set_Vr(0.0)
    b.set_Vt(0.0)
    print(b.P)   # 100000.0
    print(b.T)   # 300.0
    print(b.Ma)  # 0.28795615
    print(b.ho)  # 91142.84

Indexing and slicing return a view over a sub-region::

    # example: indexing
    from ember.block import Block
    import numpy as np

    b = Block((6,))
    b.set_x(np.linspace(0.0, 0.5, 6))
    print(b[2].x)    # 0.2
    print(b[-1].x)   # 0.5
    print(b[1:4].x)  # [0.1 0.2 0.3]

    b2 = Block((3, 2))
    b2.set_x(np.arange(6, dtype=float).reshape(3, 2) * 0.1)
    print(b2[0, :].x)  # [0.  0.1]
    print(b2[:, 1].x)  # [0.1 0.3 0.5]

:py:meth:`Block.copy` decouples the backing array so mutations do not propagate::

    # example: copy
    from ember.block import Block

    b1 = Block()
    b1.set_x(2.0)
    b2 = b1.copy()
    b2.set_x(-6.0)
    print(b1.x)  # 2.0
    print(b2.x)  # -6.0
"""

import logging

import ember._struct
import ember.perturbation
import ember.collections
import numpy as np
from ember.util import pol_to_pseudocart
from ember import util
from ember._struct import cached_array, cached_object, derived_array, scratch_array
from functools import wraps
import ember.fortran

logger = logging.getLogger(__name__)

__all__ = [
    "Block",
]


_GEOM_KCHUNK = 8
"""k-planes of nodes the geometry helpers promote at a time.

The face-area and volume kernels are double precision -- the cross products
differencing nearly-equal node coordinates need it, and
tests/test_geometry.py's theta-origin invariance pins that -- while a block's
coordinates are float32. A whole-block call therefore had to promote the
entire coordinate stack, hold a double-precision result beside it and then
cast that back down: about 59 MB of transient per face array at 273x65x57,
and with the four of them it was the process's peak RSS, reached before the
march had taken a step.

Every face's and every cell's stencil is contained within its own k-slab, so
walking in slabs bounds the promotion to a few MB and changes not one
returned value. The chunk is in PLANES rather than bytes because that is what
the stencil overlap is expressed in; 8 is a few MB at any block this solver
marches, and the per-call overhead is a handful of calls per array.
"""


def _slab_ranges(n_face_k, n_overlap):
    """Walk `n_face_k` face planes in slabs, yielding (k0, n_face, n_node).

    `n_overlap` is how many extra node planes a face plane needs beyond its
    own: 1 where the face spans k..k+1 (i- and j-faces, and cells), 0 where it
    sits in a single plane (k-faces).
    """
    for k0 in range(0, n_face_k, _GEOM_KCHUNK):
        n_face = min(_GEOM_KCHUNK, n_face_k - k0)
        yield k0, n_face, n_face + n_overlap


def _da_dest(out, shape, dtype):
    """Destination for a face-area walk, and which layout it is in.

    Returns ``(dest, comp_first)``. `shape` is the components-LAST shape the
    helper documents; the cached ``dA*_nd`` buffers are the transpose of it,
    components first, and are passed in as `out` so the walk can write them
    without a whole-block temporary in between. Anything else is rejected here
    rather than guessed at from the shape, which for a small enough block is
    genuinely ambiguous.
    """
    if out is None:
        return np.empty(shape, dtype=dtype, order="F"), False
    if out.shape == shape:
        return out, False
    if out.shape == (3,) + shape[:-1]:
        return out, True
    raise ValueError(
        f"out has shape {out.shape}, wanted {shape} (components last) "
        f"or {(3,) + shape[:-1]} (components first)"
    )


def _store_slab(dest, slab, k0, n_face, comp_first):
    """Write one double-precision slab into `dest`, in `dest`'s own layout.

    Assigning through the slice casts to `dest`'s dtype, so the rounding to
    float32 is the same single rounding the whole-block path applied and the
    stored values are unchanged by the walk.
    """
    if comp_first:
        dest[:, :, :, k0:k0 + n_face] = np.moveaxis(slab, -1, 0)
    else:
        dest[:, :, k0:k0 + n_face, :] = slab


def _handle_output(result, out=None):
    """Copy `result` into `out` if given, otherwise return `result` unchanged.

    Parameters
    ----------
    result : Array
        The computed result array.
    out : Array, optional
        Output array to store results. Must have compatible shape with result.

    Returns
    -------
    Array
        Either `result`, or `out` with `result` copied into it.
    """
    if out is not None:
        out[...] = result
        return out
    return result


def _get_da_tri(xrt, out=None):
    r"""Area vectors of triangular faces, backing :attr:`Block.dA_tri`.

    For a triangle with vertices :math:`A, B, C` in pseudo-Cartesian space
    :math:`(x, r, r\theta)`:

    .. math::

        \delta\!\mathbf{A} = \tfrac{1}{2}\,\overrightarrow{AC} \times \overrightarrow{AB}

    Parameters
    ----------
    xrt : Array, shape (ntri, 3, 3)
        Polar coordinates :math:`(x, r, \theta)` at the three vertices of each triangle.
    out : Array, optional
        Output array to store results. Must have shape (ntri, 3).

    Returns
    -------
    dA : Array, shape (ntri, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.
    """
    xrrt = pol_to_pseudocart(xrt)
    qAB = xrrt[:, 1, :] - xrrt[:, 0, :]
    qAC = xrrt[:, 2, :] - xrrt[:, 0, :]
    # Swap order to match structured area orientation
    return _handle_output(0.5 * np.cross(qAC, qAB, axis=-1), out)


def _get_dai(xrt, out=None):
    r"""Area vectors of constant-i faces, backing :attr:`Block.dAi_nd`.

    Each face is bounded by the four nodes
    :math:`(i,j,k),\,(i,j,k{+}1),\,(i,j{+}1,k{+}1),\,(i,j{+}1,k)`,
    circulating so that the area vector points along increasing i.
    Evaluated as half the cross product of the face diagonals, which is
    exact for a warped face; see :ref:`face-areas`.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, nk, 3)
        Polar coordinates :math:`(x, r, \theta)` at grid nodes.
    out : Array, optional
        Output array. Must have shape (ni, nj-1, nk-1, 3).

    Returns
    -------
    dAi : Array, shape (ni, nj-1, nk-1, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.
    """
    # Validate input
    ndim = xrt.ndim - 1  # Spatial dimensions only
    if ndim != 3:
        raise ValueError(f"dAi is not defined for ndim={ndim}.")

    # Preserve input dtype for precision
    input_dtype = xrt.dtype
    ni, nj, nk = xrt.shape[:3]
    dest, comp_first = _da_dest(out, (ni, nj - 1, nk - 1, 3), input_dtype)

    # A slab at a time: each i-face spans k..k+1, so a slab of n+1 node planes
    # carries every stencil of its n face planes (see _GEOM_KCHUNK).
    for k0, n_face, n_node in _slab_ranges(nk - 1, 1):
        node = np.asarray(xrt[:, :, k0:k0 + n_node, :], dtype=np.float64, order="F")
        slab = util.allocate_or_reuse(
            None, (ni, nj - 1, n_face, 3), dtype=np.float64
        )
        ember.fortran.get_dai(node, slab)
        _store_slab(dest, slab, k0, n_face, comp_first)

    return dest


def _get_daj(xrt, out=None):
    r"""Area vectors of constant-j faces, backing :attr:`Block.dAj_nd`.

    Each face is bounded by the four nodes
    :math:`(i,j,k),\,(i{+}1,j,k),\,(i{+}1,j,k{+}1),\,(i,j,k{+}1)`,
    circulating so that the area vector points along increasing j.
    Evaluated as half the cross product of the face diagonals, which is
    exact for a warped face; see :ref:`face-areas`.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, nk, 3)
        Polar coordinates :math:`(x, r, \theta)` at grid nodes.
    out : Array, optional
        Output array. Must have shape (ni-1, nj, nk-1, 3).

    Returns
    -------
    dAj : Array, shape (ni-1, nj, nk-1, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.
    """
    # Validate input
    ndim = xrt.ndim - 1  # Spatial dimensions only
    if ndim != 3:
        raise ValueError(f"dAj is not defined for ndim={ndim}.")

    # Preserve input dtype for precision
    input_dtype = xrt.dtype
    ni, nj, nk = xrt.shape[:3]
    dest, comp_first = _da_dest(out, (ni - 1, nj, nk - 1, 3), input_dtype)

    # As _get_dai: a j-face spans k..k+1, so the slab carries one extra plane.
    for k0, n_face, n_node in _slab_ranges(nk - 1, 1):
        node = np.asarray(xrt[:, :, k0:k0 + n_node, :], dtype=np.float64, order="F")
        slab = util.allocate_or_reuse(
            None, (ni - 1, nj, n_face, 3), dtype=np.float64
        )
        ember.fortran.get_daj(node, slab)
        _store_slab(dest, slab, k0, n_face, comp_first)

    return dest


def _get_dak(xrt, out=None):
    r"""Area vectors of constant-k faces, backing :attr:`Block.dAk_nd`.

    Each face is bounded by the four nodes
    :math:`(i,j,k),\,(i,j{+}1,k),\,(i{+}1,j{+}1,k),\,(i{+}1,j,k)`,
    circulating so that the area vector points along increasing k.
    Evaluated as half the cross product of the face diagonals, which is
    exact for a warped face; see :ref:`face-areas`.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, nk, 3)
        Polar coordinates :math:`(x, r, \theta)` at grid nodes.
    out : Array, optional
        Output array. Must have shape (ni-1, nj-1, nk, 3).

    Returns
    -------
    dAk : Array, shape (ni-1, nj-1, nk, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.
    """
    # Validate input
    ndim = xrt.ndim - 1  # Spatial dimensions only
    if ndim != 3:
        raise ValueError(f"dAk is not defined for ndim={ndim}.")

    # Preserve input dtype for precision
    input_dtype = xrt.dtype
    ni, nj, nk = xrt.shape[:3]
    dest, comp_first = _da_dest(out, (ni - 1, nj - 1, nk, 3), input_dtype)

    # A k-face lies IN a node plane rather than spanning two, so here the slab
    # needs no extra plane and there are nk of them, not nk-1.
    for k0, n_face, n_node in _slab_ranges(nk, 0):
        node = np.asarray(xrt[:, :, k0:k0 + n_node, :], dtype=np.float64, order="F")
        slab = util.allocate_or_reuse(
            None, (ni - 1, nj - 1, n_face, 3), dtype=np.float64
        )
        ember.fortran.get_dak(node, slab)
        _store_slab(dest, slab, k0, n_face, comp_first)

    return dest


def _get_da_quad(xrt, out=None):
    r"""Area vectors of quadrilateral faces on a 2D cut, backing :attr:`Block.dA_quad`.

    Delegates to :func:`_get_dak` with a dummy third dimension added and then
    removed.  The four nodes of each face are
    :math:`(i,j),\,(i,j{+}1),\,(i{+}1,j{+}1),\,(i{+}1,j)`.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, 3)
        Polar coordinates :math:`(x, r, \theta)` at cut nodes.
    out : Array, optional
        Output array. Must have shape (ni-1, nj-1, 3).

    Returns
    -------
    dA : Array, shape (ni-1, nj-1, 3)
        Face area vectors in pseudo-Cartesian components :math:`(x, r, r\theta)`.
    """
    ndim = xrt.ndim - 1  # Exclude the coordinate index
    assert ndim == 2, "Face area is only defined for 2D grids"

    # Add a dummy third dimension for compatibility and calculate dAk
    xrt = xrt[:, :, None, :]
    dA = _get_dak(xrt)
    dA = dA[:, :, 0, :]

    return _handle_output(dA, out)


def _get_vol(xrt, dAi, dAj, dAk, out=None):
    r"""Cell volumes via the divergence theorem, backing :attr:`Block.vol_nd`.

    With the vector field :math:`\mathbf{F} = (x,\, r/2,\, r\theta)`,
    :math:`\nabla\cdot\mathbf{F} = 3` in cylindrical coordinates, so

    .. math::

        \delta\mathcal{V} = \frac{1}{3}
            \sum_{\text{faces}} \mathbf{F}_f \cdot \delta\mathbf{A}_f

    where :math:`\mathbf{F}_f` is the average of the four corner nodes on
    each face.

    Parameters
    ----------
    xrt : Array, shape (ni, nj, nk, 3)
        Polar coordinates :math:`(x, r, \theta)` at grid nodes.
    dAi : Array, shape (3, ni, nj-1, nk-1) or (ni, nj-1, nk-1, 3)
        Constant-i face area vectors.
    dAj : Array, shape (3, ni-1, nj, nk-1) or (ni-1, nj, nk-1, 3)
        Constant-j face area vectors.
    dAk : Array, shape (3, ni-1, nj-1, nk) or (ni-1, nj-1, nk, 3)
        Constant-k face area vectors.
    out : Array, optional
        Output array. Must have shape (ni-1, nj-1, nk-1).

    Returns
    -------
    vol : Array, shape (ni-1, nj-1, nk-1)
        Cell volumes.
    """
    # Check number of spatial dimensions
    ndim = xrt.ndim - 1  # Exclude the coordinate index
    assert ndim == 3, "Volume is only defined for 3D grids"

    ni, nj, nk = xrt.shape[:3]

    # Accept both (3, ...) components-first and (..., 3) components-last layouts
    if dAi.shape == (ni, nj - 1, nk - 1, 3):
        dAi = np.moveaxis(dAi, -1, 0)
    if dAj.shape == (ni - 1, nj, nk - 1, 3):
        dAj = np.moveaxis(dAj, -1, 0)
    if dAk.shape == (ni - 1, nj - 1, nk, 3):
        dAk = np.moveaxis(dAk, -1, 0)

    if dAi.shape != (3, ni, nj - 1, nk - 1):
        raise ValueError(f"Invalid shape for dAi: {dAi.shape}")
    if dAj.shape != (3, ni - 1, nj, nk - 1):
        raise ValueError(f"Invalid shape for dAj: {dAj.shape}")
    if dAk.shape != (3, ni - 1, nj - 1, nk):
        raise ValueError(f"Invalid shape for dAk: {dAk.shape}")

    # Preserve input dtype for precision (use xrt as reference)
    input_dtype = xrt.dtype
    if out is None:
        out = np.empty((ni - 1, nj - 1, nk - 1), dtype=input_dtype, order="F")

    # A slab at a time, as the face-area helpers do, and for the same reason:
    # this one would otherwise promote the coordinate stack AND all three face
    # arrays at once, the largest transient of the four. A cell spans k..k+1,
    # so it needs n+1 node planes, n planes of the i- and j-face arrays (which
    # sit between nodes in k) and n+1 of the k-face array (which does not).
    for k0, n_cell, n_node in _slab_ranges(nk - 1, 1):
        ks = slice(k0, k0 + n_cell)
        xrt_f = np.asarray(
            xrt[:, :, k0:k0 + n_node, :], dtype=np.float64, order="F"
        )
        dAi_f = np.asarray(dAi[:, :, :, ks], dtype=np.float64, order="F")
        dAj_f = np.asarray(dAj[:, :, :, ks], dtype=np.float64, order="F")
        dAk_f = np.asarray(
            dAk[:, :, :, k0:k0 + n_node], dtype=np.float64, order="F"
        )
        slab = util.allocate_or_reuse(
            None, (ni - 1, nj - 1, n_cell), dtype=np.float64
        )
        ember.fortran.get_vol(xrt_f, dAi_f, dAj_f, dAk_f, slab)
        out[:, :, ks] = slab

    return out


class _MaskedBlock:
    """Proxy that confines any :class:`Block` setter to the masked nodes.

    Returned by :meth:`Block.masked`. Forwarding a ``set_*`` call runs the
    underlying setter on the *whole* block, then restores every node outside the
    mask from a snapshot, so only nodes where the mask is True are changed.
    Non-setter attribute access is forwarded to the wrapped block unchanged.

    Because the setter runs over the full field before the rollback, a single
    snapshot/restore makes every setter work without per-setter special casing.
    The snapshot copies the wrapped block's backing array, so pre-slicing with
    basic indexing (``block[i].masked(mask)``) keeps the copy cost proportional
    to the slice rather than the whole block while still writing through to the
    parent, since a basic-index slice is a view.
    """

    __slots__ = ("_block", "_mask")

    def __init__(self, block, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != block.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match block shape {block.shape}"
            )
        self._block = block
        self._mask = mask

    def __getattr__(self, name):
        attr = getattr(self._block, name)
        if not (name.startswith("set_") and callable(attr)):
            return attr

        @wraps(attr)
        def wrapper(*args, **kwargs):
            block = self._block
            keep = ~self._mask
            saved = block._data.copy()  # snapshot whole backing array
            attr(*args, **kwargs)  # mutate full field
            block._data[keep] = saved[keep]  # roll back untouched nodes
            # The rollback writes raw data without bumping versions, so any
            # cache populated during the setter is now stale; drop it.
            block.clear_cache()

        return wrapper



# The scratch arena is sized for at most this many multigrid levels. Solver
# configuration is validated against it (ember.solver._validate_mg), so a run
# cannot ask for a coarser hierarchy than the arena was built to hold.
MAX_MG_LEVELS = 3


def _viscous_face_shapes(ni, nj, nk):
    """Shapes of the six boundary tau/q face buffers, in tau_q_faces order."""
    shp_i = (nj - 1, 9, nk - 1, 2)
    shp_j = (ni - 1, 9, nk - 1, 2)
    shp_k = (ni - 1, 9, nj - 1, 2)
    return (shp_i, shp_i, shp_j, shp_j, shp_k, shp_k)


def _scratch_len(shape, n_levels=MAX_MG_LEVELS):
    """Elements the shared scratch arena needs, sized by its worst phase.

    :attr:`Block.scratch` backs every throwaway buffer in the step. Its
    consumers fall into phases that never overlap, so the arena holds the SUM
    of what is live within a phase and the MAX across phases:

      update_sources   the six boundary tau/q face buffers + set_visc_force's
                       rolling tau/q cell-plane pair, planes and rows + the
                       nodal transport trio (mu, kappa, cp) both kernels read
      update_primitive the nodal kinetic energy the kinematic kernel writes
                       and `ho` absorbs two lines later, live for that window
                       only. Every caller fills the primitive cache before
                       carving anything of its own, so this never coexists
                       with another phase's buffers
      update_timestep  the nodal acoustic speed set_timestep_spectral reads
      filter / SFD     one cell-shaped conserved volume, materialised for
                       apply_sfd_force and update_filter (both off by default;
                       every other cell-conserved reader averages the nodal
                       state as it walks). A sub-phase of update_sources for
                       the first and a whole method for the second, never live
                       alongside either's other buffers
      update_residual  set_residual's rolling planes and rows + the IRS work
                       vector
      scree / RK, MG   the seven multigrid coarse buffers + the caller's
                       rolling two-plane increment (the prolongation is
                       injection, collapsed in place inside ``corr_all``, and
                       is fused with the fine term's cell->node scatter)
      scree / RK, no MG  the caller's full-volume cell-shaped increment, which
                       the multigrid-off kernels still materialise

    WHICH PHASE BINDS depends on the shape, and that is new. The multigrid
    phase used to bind at every shape tried, on twelve coarse buffers of which
    the separable-prolong scratch alone was 2.5M elements at 273x65x57.
    Replacing the trilinear cascade with injection dropped five of those
    buffers and took the phase from 25.02 MB to 9.82 MB there, so it no longer
    binds anywhere: ``update_residual`` binds at 273x65x57 (20.96 MB) and
    ``update_sources`` on a cube (2.67 MB at 49x49x49). The arena itself fell
    only 25.02 -> 20.96 MB and 2.89 -> 2.67 MB, because the next phase down
    takes over.

    That matters for a decision recorded here, which the change has quietly
    undermined: the transport trio -- three nodal fields that used to be cached
    arrays outliving the phase that reads them -- was borrowed into this arena
    on the argument that it cost nothing, being space the multigrid phase was
    already sizing. It is no longer free on a cube, where ``update_sources`` is
    now the binding phase and the trio is what puts it there. The trade is
    still favourable (2.67 against the 2.89 the old multigrid phase demanded)
    but it is a trade again, and shrinking the viscous phase is now the way to
    shrink the arena.

    Sizes are computed, never written as literals, so a buffer added to a phase
    shows up here rather than silently overrunning its neighbour -- which is
    what tests/test_scratch_arena exists to enforce.
    """
    from ember.solver import mg_coarse_shapes  # noqa: PLC0415 - circular import

    ni, nj, nk = shape
    # grid.py pads the j extent when the component stride would be a whole page
    # multiple, so the k-accumulate's streams never 4K-alias. Size for the pad.
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
    visc_pr = ni * nj * 4 * 2 + ni * 4 * 3
    faces = sum(int(np.prod(sh)) for sh in _viscous_face_shapes(ni, nj, nk))
    tq = (ni + 1) * (nj + 1) * 9 * 2
    transport = ni * nj * nk * 3
    mg = sum(int(np.prod(sh)) for sh in mg_coarse_shapes(ni, nj, nk, n_levels))
    return max(
        faces + tq + visc_pr + transport,                      # update_sources
        ni * nj * nk,                                          # update_primitive
        ni * nj * nk,                                          # update_timestep
        (ni - 1) * (nj - 1) * (nk - 1) * 5,                    # filter / SFD
        ni * njp * 5 * 2 + ni * 5 * 3 + ni * nj * nk * 5,      # update_residual
        mg + (ni - 1) * (nj - 1) * 5 * 2,                      # scree/RK + multigrid
        (ni - 1) * (nj - 1) * (nk - 1) * 5,                    # scree/RK, no multigrid
    )


def _carve_viscous(block):
    """Everything ``update_sources`` needs from the arena, from one carve.

    Returns ``(faces, tq, planes, rows, transport)``: the six boundary tau/q
    face buffers, the rolling tau/q cell-plane pair, ``set_visc_force``'s
    rolling face-flow planes and rows, and the nodal transport trio
    ``(mu, kappa, cp)`` both viscous kernels read. This is the only place the
    viscous phase's arena layout is written down, and every caller that needs
    any part of it comes through here -- ``grid.update_sources``, the
    :attr:`Block.tau_q_faces` accessor, and the tests and bench arms that drive
    ``set_visc_force`` directly. Being deterministic, separate calls agree.

    All of them reach the same ``set_visc_force`` call, so they must not
    overlap. One carve is what guarantees that -- ``util.carve_view`` packs the
    shapes end to end -- and it is why the accessor and ``grid.update_sources``
    both come through here rather than each carving what it happens to want.

    The trio is the one part with a lifetime longer than a single kernel call:
    ``grid.update_sources`` fills it in the boundary phase and reads it back in
    the face-flux phase, across the seam exchange in between, exactly as the
    face buffers do. That is safe because each block owns its arena and nothing
    else carves it during the pass -- the same contract the rest of the arena
    runs on, and not something the code can check.
    """
    ni, nj, nk = block.shape
    bufs = util.carve_view(
        block.scratch,
        *_viscous_face_shapes(ni, nj, nk),
        (ni + 1, nj + 1, 9, 2),
        (ni, nj, 4, 2),
        (ni, 4, 3),
        *((ni, nj, nk),) * 3,
    )
    return tuple(bufs[:6]), bufs[6], bufs[7], bufs[8], tuple(bufs[9:])


class Block(ember._struct.StructuredData):
    def __init__(self, shape=()):
        """Allocate a structured grid block.

        This is the primary data container for flow fields. It stores coordinates and conserved variables, and provides properties for derived variables such as velocity, pressure and Mach number. All data flows are managed through setter methods that ensure validity and consistency of the flow field. The class also stores boundary patches to specify simulation boundary conditions in :py:attr:`Block.patches`.

        The setters fall into two complementary families: thermodynamic setters
        such as :py:meth:`set_P_T` store pressure and temperature
        while preserving the velocity field, and kinematic
        setters like :py:meth:`set_Vx` store
        the velocity while preserving thermodynamic
        state. The setters may be
        called in either order to build up a complete flow field.

        Parameters
        ----------
        shape : tuple of int, optional
            Number of nodes in each dimension `(ni, nj, nk, ...)`. Any number of
            dimensions is supported. Defaults to `()`, giving a scalar block with
            no grid dimensions.

        """

        super().__init__(shape)

    def __post_init__(self):
        """Initialize the block with dummy values."""

        # Set unity radius to avoid division by zero
        self._set_data_by_keys(("r",), -1.0, store_init=False)

        # Set unity density to avoid division by zero in velocity calculations
        self._set_data_by_keys(("rho",), -1.0, store_init=False)

        # Set zero velocities directly (before thermodynamic state is set)
        self._set_data_by_keys(("rhoVx",), 0.0, store_init=False)
        self._set_data_by_keys(("rhoVr",), 0.0, store_init=False)
        self._set_data_by_keys(("rhorVt",), 0.0, store_init=False)

        # Set zero total energy (datum is arbitrary, zero velocity means no KE)
        self._set_data_by_keys(("rhoe",), 0.0, store_init=False)

        # Turbulent viscosity: 0 until a viscous pass writes it. Stored (not
        # version-marked) so it reads as a benign zero for the always-on
        # diffusion timestep, while still counting as "unset" for the TS3 writer.
        self._set_data_by_keys(("mu_turb",), 0.0, store_init=False)

        # Initialize patch collection (only if not already present from deserialization)
        if "patches" not in self._metadata:
            patch_collection = ember.collections.BlockPatchCollection(self)
            self._set_metadata_by_key("patches", patch_collection)

        # Initialize cache storage for cached properties
        self._store = {}

        # If we are a single point, unset triangulated flag
        if self.ndim == 0:
            self.set_triangulated(False)

    def _update_rhoe_nd(self, rho_nd, u_nd):
        """Update rhoe from nondim rho and u without touching rho or momentum."""
        e_new = u_nd + self._halfVsq_nd_uninit
        self._set_data_by_keys(("rhoe",), rho_nd * e_new, store_init=False)

    def _set_rho_u_nd(self, rho_nd, u_nd):
        """Set nondimensional density and internal energy, preserving velocities."""
        Vxrt_nd = self._Vxrt_nd_uninit
        Vx_nd, Vr_nd, Vt_nd = Vxrt_nd[..., 0], Vxrt_nd[..., 1], Vxrt_nd[..., 2]
        r_nd = self._get_data_by_keys(("r",), raise_uninit=False)
        self._set_data_by_keys(("rho",), rho_nd)
        self._set_data_by_keys(("rhoVx",), rho_nd * Vx_nd, store_init=False)
        self._set_data_by_keys(("rhoVr",), rho_nd * Vr_nd, store_init=False)
        self._set_data_by_keys(("rhorVt",), rho_nd * r_nd * Vt_nd, store_init=False)
        halfVsq_nd = 0.5 * (Vx_nd**2 + Vr_nd**2 + Vt_nd**2)
        e_nd = u_nd + halfVsq_nd
        self._set_data_by_keys(("rhoe",), rho_nd * e_nd, store_init=True)

    def _get_face_wall_arrays(self, non_wall_patches=None):
        """Get face wall indicator arrays (iwall, jwall, kwall).

        Parameters
        ----------
        non_wall_patches : list, optional
            Patches to treat as non-wall. Defaults to self.patches.permeable.

        Returns
        -------
        tuple[Array, Array, Array]
            - iwall: shape self.shape_iface, 0=wall, >0=non-wall
            - jwall: shape self.shape_jface, 0=wall, >0=non-wall
            - kwall: shape self.shape_kface, 0=wall, >0=non-wall
        """
        if self.ndim != 3:
            raise ValueError(
                f"Wall indicator requires 3D block (ndim=3), got ndim={self.ndim}"
            )

        if non_wall_patches is None:
            non_wall_patches = self.patches.permeable

        iwall = np.zeros(self.shape_iface, dtype=np.uint8)
        jwall = np.zeros(self.shape_jface, dtype=np.uint8)
        kwall = np.zeros(self.shape_kface, dtype=np.uint8)

        # Set interior faces to non-wall
        iwall[1:-1, :, :] = 1
        jwall[:, 1:-1, :] = 1
        kwall[:, :, 1:-1] = 1

        # Process non-wall patches
        for patch in non_wall_patches:
            ijk_face = patch.get_ijk_face()
            if patch.const_dim == 0:  # i-face patch
                iwall[*ijk_face.T] += 1
            elif patch.const_dim == 1:  # j-face patch
                jwall[*ijk_face.T] += 1
            elif patch.const_dim == 2:  # k-face patch
                kwall[*ijk_face.T] += 1

        return iwall, jwall, kwall

    def _make_fluid_property(prop_name, doc, ref=None):
        """Factory for creating fluid property getters.

        Calls ``fluid.get_<prop_name>(rho_nd, u_nd)`` and rescales the
        nondimensional result to dimensional units.

        Parameters
        ----------
        prop_name : str
            Suffix for the fluid method, e.g. ``"cp"`` -> ``fluid.get_cp``.
        doc : str
            Property docstring.
        ref : {None, "Rgas", "P", "V", "u", "T", "rhoV", "kappa"}
            Reference scale to multiply the nondim output by.
            ``None`` for dimensionless quantities. ``"kappa"`` is the
            conductivity scale, the mass flux times the gas constant --- see
            :ref:`reference-scales`.
        """

        def getter(self):
            val = getattr(self.fluid, f"get_{prop_name}")(
                self._rho_nd_uninit, self.u_nd
            )
            if ref == "Rgas":
                val *= self.fluid.Rgas_ref
            elif ref == "P":
                val *= self.fluid.P_ref
            elif ref == "V":
                val *= self.fluid.V_ref
            elif ref == "u":
                val *= self.fluid.u_ref
            elif ref == "T":
                val *= self.fluid.T_ref
            elif ref == "rhoV":
                val *= self.fluid.rhoV_ref
            elif ref == "kappa":
                val *= self.fluid.rhoV_ref * self.fluid.Rgas_ref
            return val

        getter.__doc__ = doc
        return property(getter)

    @property
    def _face_wall_arrays(self):
        """Permeable-variant face wall arrays (iwall, jwall, kwall)."""
        return self._get_face_wall_arrays()

    def _fill_transport_nd(self, mu=None, kappa=None, cp=None):
        r"""Write the nodal transport properties into caller buffers.

        Fills whichever of `mu`, `kappa` and `cp` are given, each a nodal-shaped
        buffer the caller owns, with :attr:`mu_nd`, :attr:`kappa_nd` and
        :attr:`cp_nd` -- see those for what the nondimensionalisations mean.
        This is the one place they are applied; the three properties and
        :meth:`ember.grid.Grid.update_sources`, which borrows the trio from the
        scratch arena rather than keeping it, both come through here.

        Not cached and not stored: the viscous pass is the only consumer that
        wants all three at once, and it supplies its own storage.
        """
        rho, u = self._rho_nd_uninit, self.u_nd
        if mu is not None:
            self.fluid.get_mu(rho, u, out=mu)
            mu /= self.L_ref
        if kappa is not None:
            self.fluid.get_kappa(rho, u, out=kappa)
            kappa /= self.L_ref
        if cp is not None:
            self.fluid.get_cp(rho, u, out=cp)

    @cached_object
    def _face_wall_arrays_slip(self):
        """Slip-variant face wall arrays (iwall, jwall, kwall).

        Treats slip (frictionless) patches as non-wall, in addition to the
        permeable types handled by :attr:`_face_wall_arrays`.
        """
        return self._get_face_wall_arrays(self.patches.slip)

    @property
    def _halfVsq_nd_uninit(self):
        """Nondimensional half velocity squared [-], tolerating uninitialised data.

        Derived, not cached, and each access allocates. The one consumer that
        reads it every step is :meth:`update_primitive`, which needs it only
        between the kinematic kernel that makes it and the ``ho`` it is added
        into -- a lifetime inside a single method -- so it is carved from
        :attr:`scratch` there rather than kept on the block. That is 4.05 MB
        per block at 273x65x57 not held for the whole run.

        What is left here serves the readers outside that window: the
        :attr:`_u_nd_uninit` fallback on patch views (a surface, not a volume),
        :attr:`V_nd` and the relative-frame stagnation properties in
        post-processing, and :meth:`_update_rhoe_nd` in the state setters,
        which a march never reaches. Do not put an access in a per-node loop
        over a full block, and do not add a per-step reader without giving it
        the scratch treatment instead.
        """
        Vxrt_nd = self._Vxrt_nd_uninit
        out = util.empty(self.shape)
        np.einsum("...i,...i->...", Vxrt_nd, Vxrt_nd, out=out)
        out *= 0.5
        return out

    @property
    def _halfVsq_rel_nd(self):
        """Nondimensional half relative velocity squared [-]."""
        Vxrt_nd = self._Vxrt_nd_uninit
        return 0.5 * (Vxrt_nd[..., 0] ** 2 + Vxrt_nd[..., 1] ** 2 + self._Vt_rel_nd**2)

    @property
    def _ho_rel_nd(self):
        """Nondimensional relative frame stagnation enthalpy [-]."""
        # Stagnation quantities are undefined without a velocity; require the
        # momenta (r is tolerated, matching the velocity getters).
        self._get_data_by_keys(("rhoVx", "rhoVr", "rhorVt"))
        return self.fluid.get_h(self._rho_nd_uninit, self.u_nd) + self._halfVsq_rel_nd

    @property
    def _rho_nd_uninit(self):
        """Nondimensional mass density, tolerating uninitialised data.

        For the raising counterpart, use the public :attr:`rho_nd`.
        """
        return self._get_data_by_keys(("rho",), raise_uninit=False)

    @property
    def _rho_ref(self):
        """Density reference scale [kg/m^3], falling back to 1.0 if no fluid is set.

        Same rationale as _V_ref: allows set_conserved and related setters to
        store nondimensional density before a fluid is assigned, with the
        convention that nondimensional == dimensional when rho_ref=1.
        """
        return self.fluid.rho_ref if "fluid" in self._metadata else 1.0

    @property
    def _rhoo_nd(self):
        """Nondimensional stagnation density [-]."""
        return self.fluid.set_h_s(self.ho_nd, self.s_nd)[0]

    @property
    def _rhoo_rel_nd(self):
        """Nondimensional relative frame stagnation density [-]."""
        return self.fluid.set_h_s(self._ho_rel_nd, self.s_nd)[0]

    @property
    def _rhoV_ref(self):
        """rho_ref * V_ref: reference scale for momentum [kg/m^2/s]."""
        return self.fluid.rhoV_ref

    @property
    def _rhoVsq_ref(self):
        """rho_ref * V_ref^2: reference scale for energy [J/m^3]."""
        return self.fluid.P_ref

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def _u_nd_uninit(self, out):
        """Nondimensional specific internal energy [-], tolerating uninitialised data.

        For the raising counterpart, use the public :attr:`u_nd`.
        """
        rhoe_nd = self._get_data_by_keys(("rhoe",), raise_uninit=False)
        out = util.allocate_or_reuse(out, self.shape)
        np.divide(rhoe_nd, self._rho_nd_uninit, out=out)
        out -= self._halfVsq_nd_uninit
        return out

    @property
    def _uo_nd(self):
        """Nondimensional stagnation internal energy [-]."""
        return self.fluid.set_h_s(self.ho_nd, self.s_nd)[1]

    @property
    def _uo_rel_nd(self):
        """Nondimensional relative frame stagnation internal energy [-]."""
        return self.fluid.set_h_s(self._ho_rel_nd, self.s_nd)[1]

    @property
    def _V_ref(self):
        """Velocity reference scale [m/s], falling back to 1.0 if no fluid is set.

        Allows set_Vx/set_Vr/set_Vt to store nondimensional momenta before a
        fluid is assigned. With V_ref=1 the stored value equals the dimensional
        velocity, which is consistent once a fluid is later set via set_fluid.
        """
        return self.fluid.V_ref if "fluid" in self._metadata else 1.0

    @property
    def _Vsq_ref(self):
        """V_ref^2: reference scale for specific energy [J/kg]."""
        return self.fluid.u_ref

    @property
    def _Vt_rel_nd(self):
        """Nondimensional relative tangential velocity [-]."""
        return self._Vt_rel_nd_uninit

    @property
    def _Vt_rel_nd_uninit(self):
        r"""Nondimensional relative tangential velocity [-], own contiguous buffer.

        :math:`V_\theta^\mathrm{rel} = V_\theta - \Omega r`, formed in the
        relative frame so the viscous kernels can differentiate it directly
        (see :meth:`ember.grid.Grid.update_sources`). Tolerates uninitialised
        momenta like :attr:`_Vxrt_nd_uninit`; requires r/Omega via
        :attr:`r_nd` and :attr:`Omega_nd` (blade speed ``U* = r* Omega*``).
        Public access is via :attr:`Vt_rel_nd`, which guards the momenta first.
        """
        out = util.empty(self.shape)
        # Form blade speed U* = r* Omega* in-place in the fresh buffer, then
        # Vt_rel = Vt - U* (subtract aliases its second input safely), avoiding
        # a separate r_nd * Omega_nd temporary.
        np.multiply(self.r_nd, self.Omega_nd, out=out)
        np.subtract(self._vel_nd_uninit("rhorVt"), out, out=out)
        return out

    def _vel_nd_uninit(self, key):
        """One nondimensional velocity component, derived from its momentum.

        The per-component counterpart to :attr:`_Vxrt_nd_uninit`, so an
        accessor that wants one component allocates one array rather than a
        three-component stack it immediately slices. Same tolerance of
        uninitialised data, and the same division order.
        """
        rho = self._get_data_by_keys(("rho",), raise_uninit=False)
        mom = self._get_data_by_keys((key,), raise_uninit=False)
        out = np.divide(mom, rho)
        if key == "rhorVt":
            out /= self._get_data_by_keys(("r",), raise_uninit=False)
        return out

    @property
    def _Vxrt_nd_uninit(self):
        """Nondimensional polar velocity (Vx, Vr, Vt) stacked on the last axis.

        Single source of truth for velocity on the PYTHON side, and derived
        rather than cached: every solver kernel that wanted a nodal velocity
        volume now forms it from ``cons`` at the corners it walks (see
        ``vel_at`` in _fortran/viscous.f90), so caching this held 12.14 MB per
        block at 273x65x57 -- and ``Vt_rel`` another 4.05 MB -- for consumers
        that are all O(surface): the boundary conditions, the mixing planes and
        post-processing. Each access allocates; do not put one in a per-node
        loop over a full block.

        Tolerates uninitialised data (does not raise), so it is safe to call on
        a partially built block. Public access is via :attr:`Vxrt_nd` and the
        per-component :attr:`Vx_nd`, :attr:`Vr_nd`, :attr:`Vt_nd`, which guard
        against uninitialised momenta first.

        Vt = rhorVt / (r * rho) is split into two sequential divisions to
        avoid allocating the r*rho temporary array.
        """
        rho = self._get_data_by_keys(("rho",), raise_uninit=False)
        rhoVx = self._get_data_by_keys(("rhoVx",), raise_uninit=False)
        rhoVr = self._get_data_by_keys(("rhoVr",), raise_uninit=False)
        rhorVt = self._get_data_by_keys(("rhorVt",), raise_uninit=False)
        r = self._get_data_by_keys(("r",), raise_uninit=False)
        out = util.empty(self.shape + (3,))
        np.divide(rhoVx, rho, out=out[..., 0])
        np.divide(rhoVr, rho, out=out[..., 1])
        np.divide(rhorVt, rho, out=out[..., 2])  # Vt = rhorVt/rho ...
        out[..., 2] /= r  # ... then /r, avoiding the r*rho temp
        return out

    @derived_array
    def _Vxyz(self):
        """Cartesian velocity components [m/s, m/s, m/s]"""
        _, Vxyz = util.pol_to_cart(self.xrt, self.Vxrt)
        return Vxyz

    @property
    def _wall_nodes(self):
        """Boolean node array: True where the node lies on a wall surface.

        A node is a wall if every face touching it is a wall. The threshold
        varies by topological position: interior=0, face=8, edge=4, corner=3.

        Not cached: safe to call after modifying patches.
        """
        iwall, jwall, kwall = self._face_wall_arrays

        wall_node = np.zeros(self.shape, dtype=np.uint8)

        wall_node[:, :-1, :-1] += iwall
        wall_node[:, 1:, :-1] += iwall
        wall_node[:, :-1, 1:] += iwall
        wall_node[:, 1:, 1:] += iwall

        wall_node[:-1, :, :-1] += jwall
        wall_node[1:, :, :-1] += jwall
        wall_node[:-1, :, 1:] += jwall
        wall_node[1:, :, 1:] += jwall

        wall_node[:-1, :-1, :] += kwall
        wall_node[1:, :-1, :] += kwall
        wall_node[:-1, 1:, :] += kwall
        wall_node[1:, 1:, :] += kwall

        thresh = np.zeros(self.shape, dtype=np.uint8)

        thresh[0, :, :] = 8
        thresh[-1, :, :] = 8
        thresh[:, 0, :] = 8
        thresh[:, -1, :] = 8
        thresh[:, :, 0] = 8
        thresh[:, :, -1] = 8

        thresh[:, 0, 0] = 4
        thresh[:, 0, -1] = 4
        thresh[:, -1, 0] = 4
        thresh[:, -1, -1] = 4
        thresh[0, :, 0] = 4
        thresh[0, :, -1] = 4
        thresh[-1, :, 0] = 4
        thresh[-1, :, -1] = 4
        thresh[0, 0, :] = 4
        thresh[0, -1, :] = 4
        thresh[-1, 0, :] = 4
        thresh[-1, -1, :] = 4

        thresh[0, 0, 0] = 3
        thresh[-1, 0, 0] = 3
        thresh[0, -1, 0] = 3
        thresh[-1, -1, 0] = 3
        thresh[0, 0, -1] = 3
        thresh[-1, 0, -1] = 3
        thresh[0, -1, -1] = 3
        thresh[-1, -1, -1] = 3

        return wall_node < thresh

    @property
    def _xrt_nd(self):
        """Raw nondimensional polar coordinates (x/L_ref, r/L_ref, t)."""
        return self._get_data_by_keys(("x", "r", "t"))

    def _primitive_buffer(self, cache_key, shape):
        """Existing cached buffer for `cache_key`, unlocked for writing.

        The buffers are the very ones :func:`ember._struct.cached_array` hands to
        its wrapped function as ``out``, so reusing them keeps every pointer
        stable and allocates nothing after the first step.
        """
        entry = self._store.get(cache_key)
        out = None
        if entry is not None:
            out = entry[1]
            out.flags.writeable = True
        return util.allocate_or_reuse(out, shape)

    def set_conserved(self, conserved):
        r"""Store conserved variables.

        The conserved variables are density, axial momentum, radial momentum, angular momentum, and total energy:

        .. math::

            \mathcal{U} = \begin{bmatrix} \rho \\ \rho V_x \\ \rho V_r \\ \rho r V_\theta \\ \rho e \end{bmatrix}

        where :math:`e = u + \frac{1}{2}(V_x^2 + V_r^2 + V_\theta^2)` is the total specific energy.

        Together, the five conserved variables uniquely determine the
        thermodynamic state and velocity field, and being most convenient for
        computational fluid dynamics calculations, are the primary data stored
        in the block. Other variables like pressure and temperature are
        computed from the conserved variables via the equation of state in
        :py:attr:`Block.fluid`.

        Parameters
        ----------
        conserved : array-like, shape (..., 5)
            Dimensional conserved variables with components along the last axis. Each component must broadcast to block shape and be finite. Density must be >0.

        """

        if conserved.shape[-1] != 5:
            raise ValueError(
                f"Expected conserved shape (..., 5), but got {conserved.shape}"
            )
        if np.any(conserved[..., 0] <= 0):
            raise ValueError("Density (conserved[..., 0]) must be positive.")
        if np.any(~np.isfinite(conserved)):
            raise ValueError("Conserved variables must be finite.")

        conserved = np.array(conserved, dtype=float)
        keys = ("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
        conserved[..., 0] /= self.fluid.rho_ref
        conserved[..., 1] /= self._rhoV_ref
        conserved[..., 2] /= self._rhoV_ref
        conserved[..., 3] /= self._rhoV_ref * self.L_ref
        conserved[..., 4] /= self._rhoVsq_ref
        self._set_data_by_keys(keys, conserved)

    def set_fluid(self, fluid_new):
        """Set equation of state preserving any existing flow field.

        An equation of state, encapsulated in a :class:`~ember.fluid.PerfectFluid` instance, must
        be set before any thermodynamic properties can be computed.

        If an old fluid is already set, dimensional density, temperature, and
        velocities are read out, the fluid instance is swapped, and the
        stored flow field is rewritten using the new fluid's reference scales and
        datum levels.

        Parameters
        ----------
        fluid_new : Fluid
            New fluid / equation of state object.

        See Also
        --------
        ember.grid.Grid.set_fluid : Apply to every block in a Grid at once.
            Prefer this when the block is part of a Grid, rather than looping
            over blocks and calling this method individually.

        """
        # Re-expressing the stored field is only meaningful when there is a
        # field to re-express. A block whose storage was allocated but never
        # written -- a boundary patch's average block, say -- holds arbitrary
        # values, and pushing those through an equation of state produces
        # nonsense such as negative pressure. The result was discarded anyway
        # (the writes below pass store_init=False, so the data stays marked
        # uninitialised), but an equation of state that has to invert
        # numerically cannot be asked to do it and rightly refuses.
        #
        # Radius is deliberately absent from this list. It is read tolerantly
        # below and takes no part in the thermodynamic state, so a block
        # holding a flow field but no coordinates yet still has to be
        # re-expressed -- leaving it alone would strand the field on the
        # reference scales of a fluid that no longer applies.
        has_old = "fluid" in self._metadata and all(
            self._versions[key] for key in ("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
        )
        if has_old:
            old = self.fluid
            # Read dimensional thermodynamic state
            rho_nd, u_nd = self._rho_nd_uninit, self._u_nd_uninit
            T_nd = old.get_T(rho_nd, u_nd)
            P_nd = old.get_P(rho_nd, u_nd)
            T = T_nd * old.T_ref
            P = P_nd * old.P_ref
            # Read dimensional velocities
            Vxrt_nd = self._Vxrt_nd_uninit
            Vx_nd, Vr_nd, Vt_nd = Vxrt_nd[..., 0], Vxrt_nd[..., 1], Vxrt_nd[..., 2]
            Vx = Vx_nd * old.V_ref
            Vr = Vr_nd * old.V_ref
            Vt = Vt_nd * old.V_ref

        self._set_metadata_by_key("fluid", fluid_new)

        if has_old:
            # Recompute nondimensional state with new fluid
            rho_nd_new, u_nd_new = fluid_new.set_P_T(
                P / fluid_new.P_ref, T / fluid_new.T_ref
            )
            Vx_nd_new = Vx / fluid_new.V_ref
            Vr_nd_new = Vr / fluid_new.V_ref
            Vt_nd_new = Vt / fluid_new.V_ref
            r_nd = self._get_data_by_keys(("r",), raise_uninit=False)
            halfVsq = 0.5 * (Vx_nd_new**2 + Vr_nd_new**2 + Vt_nd_new**2)

            self._set_data_by_keys(("rho",), rho_nd_new, store_init=False)
            self._set_data_by_keys(("rhoVx",), rho_nd_new * Vx_nd_new, store_init=False)
            self._set_data_by_keys(("rhoVr",), rho_nd_new * Vr_nd_new, store_init=False)
            self._set_data_by_keys(
                ("rhorVt",), rho_nd_new * r_nd * Vt_nd_new, store_init=False
            )
            self._set_data_by_keys(
                ("rhoe",),
                rho_nd_new * (u_nd_new + halfVsq),
                store_init=False,
            )

            self.clear_cache()

        # The stored field now reads against the new scales; the patches still
        # hold values nondimensionalised against the old ones. Re-derive them
        # before anything can impose a stale target, which raises nothing and
        # shows up several steps into a march as a diverged boundary.
        for p in self.patches:
            p.update_ref_scales()

    def set_h_s(self, h, s):
        """Store enthalpy and entropy.

        Set the thermodynamic state by specifying static enthalpy and entropy per unit mass. The velocity field, if present, is preserved.

        Parameters
        ----------
        h : array-like
            Specific static enthalpy [J/kg]. Must be finite and broadcast to block shape.
        s : array-like
            Specific entropy [J/kg/K]. Must be finite and broadcast to block shape.

        """

        if np.any(~np.isfinite(h)):
            raise ValueError("Enthalpy must be finite.")
        if np.any(~np.isfinite(s)):
            raise ValueError("Entropy must be finite.")

        self._set_rho_u_nd(
            *self.fluid.set_h_s(h / self.fluid.u_ref, s / self.fluid.Rgas_ref)
        )

    def set_L_ref(self, L_ref):
        """Set reference length scale preserving existing dimensional values.

        The underlying block data is stored in a nondimensional form for
        reasons of numerical precision. For example, :py:attr:`Block.r` is
        actually stored as radius normalised by the reference length scale
        with the raw value accessible as :py:attr:`Block.r_nd`.

        Note that the :py:attr:`Block.fluid` instance specifies additional reference scales needed to make thermodynamic properties non-dimensional.

        This method sets a new reference length, rescaling stored
        nondimensional coordinates and angular momentum so that dimensional
        values are preserved.

        Parameters
        ----------
        L_ref : float
            Reference length scale [m]. Should be scalar, positive, and finite.

        """

        if not np.isscalar(L_ref) or L_ref <= 0 or not np.isfinite(L_ref):
            raise ValueError("L_ref must be a positive finite scalar.")

        old_L_ref = self.L_ref
        self._set_metadata_by_key("L_ref", np.float32(L_ref))
        L_fac = np.float32(old_L_ref / L_ref)

        q = self.conserved_nd
        q[..., 3] *= L_fac

        self._set_data_by_keys(
            ("x",),
            self._get_data_by_keys(("x",), raise_uninit=False) * L_fac,
            store_init=False,
        )
        self._set_data_by_keys(
            ("r",),
            self._get_data_by_keys(("r",), raise_uninit=False) * L_fac,
            store_init=False,
        )
        self._set_data_by_keys(
            ("wdist",),
            self._get_data_by_keys(("wdist",), raise_uninit=False) * L_fac,
            store_init=False,
        )

        self.clear_cache()

        # As in set_fluid: the patches hold values nondimensionalised against
        # the old length scale, including the coordinates of the averaged block
        # a surface-of-revolution patch carries. Only reachable by setting the
        # length scale after attaching the patches; attach_to_block builds those
        # at the right scale to begin with.
        for p in self.patches:
            p.update_ref_scales()

    def set_label(self, label):
        """Set a string label describing the block.

        Parameters
        ----------
        label : str
            Descriptive label for the block.

        """
        self._set_metadata_by_key("label", label)

    def set_mu_turb(self, mu_turb):
        """Store turbulent viscosity.

        See :py:attr:`Block.mu_turb` for more details.

        Parameters
        ----------
        mu_turb : array-like
            Turbulent viscosity [kg/m/s]. Must be >=0 and finite, and broadcast to block shape.

        """
        if np.any(mu_turb < 0) or np.any(~np.isfinite(mu_turb)):
            raise ValueError("mu_turb must be positive and finite.")
        self._set_data_by_keys(("mu_turb",), mu_turb)

    def set_Nb(self, Nb):
        """Set number of blades in the row containing this block.

        Used to determine circumferential periodicity.

        Parameters
        ----------
        Nb : int
            Number of blades in the row containing this block [-].

        """
        self._set_metadata_by_key("Nb", int(Nb))

    def set_Omega(self, Omega):
        """Set reference frame angular velocity.

        Properties suffixed ``_rel`` are defined in the rotating reference
        frame spinning at this angular velocity.

        Parameters
        ----------
        Omega : float
            Angular velocity of the rotating reference frame [rad/s].

        """
        self._set_metadata_by_key("Omega", np.float32(Omega))

    def set_P_h(self, P, h):
        """Store static pressure and enthalpy.

        Set the thermodynamic state by specifying static pressure and specific static
        enthalpy. The velocity field, if present, is preserved.

        Parameters
        ----------
        P : array-like
            Static pressure [Pa]. Must be positive, finite, and broadcast to block shape.
        h : array-like
            Specific static enthalpy [J/kg]. Must be finite and broadcast to block shape.

        """
        if np.any(P <= 0) or np.any(~np.isfinite(P)):
            raise ValueError("Pressure must be positive and finite.")
        if np.any(~np.isfinite(h)):
            raise ValueError("Enthalpy must be finite.")

        self._set_rho_u_nd(
            *self.fluid.set_P_h(P / self.fluid.P_ref, h / self.fluid.u_ref)
        )

    def set_P_rho(self, P, rho):
        """Store static pressure and density.

        Set the thermodynamic state by specifying static pressure and density. The
        velocity field, if present, is preserved.

        Parameters
        ----------
        P : array-like
            Static pressure [Pa]. Must be positive, finite, and broadcast to block shape.
        rho : array-like
            Density [kg/m^3]. Must be positive, finite, and broadcast to block shape.

        """
        if np.any(P <= 0) or np.any(~np.isfinite(P)):
            raise ValueError("Pressure must be positive and finite.")
        if np.any(rho <= 0) or np.any(~np.isfinite(rho)):
            raise ValueError("Density must be positive and finite.")

        self.set_P_rho_nd(P / self.fluid.P_ref, rho / self.fluid.rho_ref)

    def set_P_rho_nd(self, P_nd, rho_nd):
        """Store static pressure and density, nondimensional inputs.

        Set the thermodynamic state by specifying nondimensional static pressure and
        density. The velocity field, if present, is preserved.

        Parameters
        ----------
        P_nd : array-like
            Static pressure normalised by ``fluid.P_ref`` [--]. Should be positive and
            finite; no validation is performed as this setter is on the hot path for
            boundary condition application.
        rho_nd : array-like
            Density normalised by ``fluid.rho_ref`` [--]. Should be positive and
            finite; no validation is performed.

        """
        self._set_rho_u_nd(*self.fluid.set_P_rho(P_nd, rho_nd))

    def set_P_s(self, P, s):
        """Store static pressure and entropy.

        Set the thermodynamic state by specifying static pressure and entropy per
        unit mass. The velocity field, if present, is preserved.

        Parameters
        ----------
        P : array-like
            Static pressure [Pa]. Must be positive, finite, and broadcast to block shape.
        s : array-like
            Specific entropy [J/kg/K]. Must be finite and broadcast to block shape.

        """
        if np.any(P <= 0) or np.any(~np.isfinite(P)):
            raise ValueError("Pressure must be positive and finite.")
        if np.any(~np.isfinite(s)):
            raise ValueError("Entropy must be finite.")

        rho_nd, u_nd = self.fluid.set_P_s(P / self.fluid.P_ref, s / self.fluid.Rgas_ref)
        self._set_rho_u_nd(rho_nd, u_nd)

    def set_P_T(self, P, T):
        """Store static pressure and temperature.

        Set the thermodynamic state by specifying static pressure and temperature. The velocity field, if present, is preserved.

        Parameters
        ----------
        P : array-like
            Static pressure [Pa]. Must be positive, finite, and broadcast to block shape.
        T : array-like
            Temperature [K]. Must be positive, finite, and broadcast to block shape.

        """

        if np.any(P <= 0) or np.any(~np.isfinite(P)):
            raise ValueError("Pressure must be positive and finite.")
        if np.any(T <= 0) or np.any(~np.isfinite(T)):
            raise ValueError("Temperature must be positive and finite.")

        self._set_rho_u_nd(
            *self.fluid.set_P_T(P / self.fluid.P_ref, T / self.fluid.T_ref)
        )

    def set_r(self, r):
        """Store radial coordinates.

        Parameters
        ----------
        r : array-like
            Radial coordinates [m]. Must be >0 and finite, and broadcast to block shape.

        """

        if np.any(r == 0):
            raise ValueError("Radial coordinate cannot be zero.")

        # Preserve angular momentum by scaling rhorVt
        # Both old and new r are nondim, so the ratio is L_ref-independent
        r_nd = r / self.L_ref
        r_old = self._get_data_by_keys(("r",), raise_uninit=False)
        rhorVt_old = self._get_data_by_keys(("rhorVt",), raise_uninit=False)
        rhorVt_new = rhorVt_old * r_nd / r_old

        self._set_data_by_keys(("rhorVt",), rhorVt_new, store_init=False)
        self._set_data_by_keys(("r",), r_nd)

    def set_rho_s(self, rho, s):
        """Store density and entropy.

        Set the thermodynamic state by specifying density and entropy per unit mass.
        The velocity field, if present, is preserved.

        Parameters
        ----------
        rho : array-like
            Density [kg/m^3]. Must be positive, finite, and broadcast to block shape.
        s : array-like
            Specific entropy [J/kg/K]. Must be finite and broadcast to block shape.

        """
        if np.any(rho <= 0) or np.any(~np.isfinite(rho)):
            raise ValueError("Density must be positive and finite.")
        if np.any(~np.isfinite(s)):
            raise ValueError("Entropy must be finite.")

        self._set_rho_u_nd(
            *self.fluid.set_rho_s(rho / self.fluid.rho_ref, s / self.fluid.Rgas_ref)
        )

    def set_rho_u(self, rho, u):
        """Store density and internal energy.

        Set the thermodynamic state by specifying density and internal energy
        per unit mass. The velocity field, if present, is preserved.

        Parameters
        ----------
        rho : array-like
            Density [kg/m^3]. Must be positive, finite, and broadcast to block shape.
        u : array-like
            Specific internal energy [J/kg]. Must be finite and broadcast to block shape.

        """

        if np.any(rho <= 0) or np.any(~np.isfinite(rho)):
            raise ValueError("Density must be positive and finite.")

        if np.any(~np.isfinite(u)):
            raise ValueError("Internal energy must be finite.")

        self._set_rho_u_nd(rho / self.fluid.rho_ref, u / self._Vsq_ref)

    def set_rho_u_Vxrt_nd(self, rho_nd, u_nd, Vx_nd, Vr_nd, Vt_nd):
        r"""Write conserved variables from non-dimensional density, internal energy, and velocity components.

        Low-level, no-validation setter on the boundary-condition hot path: all
        inputs are non-dimensionalised by the fluid reference scales. The
        velocity components are supplied explicitly, so the internal energy
        follows from

        .. math::

            e = u + \tfrac{1}{2}(V_x^2 + V_r^2 + V_\theta^2).

        Boundary conditions own the physics that produces ``(rho, u)`` and the
        velocity vector (e.g. ``fluid.set_P_s`` or ``fluid.set_rho_s`` followed
        by a flow-angle or energy-equation reconstruction) and then call this
        primitive to store the result.

        Parameters
        ----------
        rho_nd : array-like
            Non-dimensional density. Must broadcast to block shape.
        u_nd : array-like
            Non-dimensional specific internal energy. Must broadcast to block shape.
        Vx_nd : array-like
            Non-dimensional axial velocity. Must broadcast to block shape.
        Vr_nd : array-like
            Non-dimensional radial velocity. Must broadcast to block shape.
        Vt_nd : array-like
            Non-dimensional tangential velocity. Must broadcast to block shape.

        """
        r_nd = self._get_data_by_keys(("r",), raise_uninit=False)

        keys = ("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
        i0 = self._data_inds["rho"]  # rho..rhoe are consecutive at i0..i0+4

        # Fused Fortran pass into Block.scratch (see its docstring's consumer
        # list) instead of the chain of np.multiply(out=...) calls this used
        # to be: on the small, patch-face-sized arrays this runs on, the
        # per-call numpy dispatch overhead dominated over the actual
        # arithmetic. scratch can't be written to directly by the kernel
        # from here -- _data is a non-contiguous slice for a patch's
        # block_view, which f2py refuses as intent(inout) -- so the kernel
        # lands in scratch and this does one explicit copy into _data.
        # util.bcast_if_needed preserves the "inputs may broadcast to block
        # shape" contract the docstring promises (the kernel itself needs
        # exact shapes), but skips broadcast_to's own overhead on the common
        # path where the caller already passed exactly self.shape -- true of
        # every current caller.
        shape = self.shape
        # The arena is flat, so carve the nodal view this kernel writes.
        nodal = util.carve_view(self.scratch, shape + (5,))
        ember.fortran.set_rho_u_vxrt_write(
            util.bcast_if_needed(rho_nd, shape),
            util.bcast_if_needed(u_nd, shape),
            util.bcast_if_needed(Vx_nd, shape),
            util.bcast_if_needed(Vr_nd, shape),
            util.bcast_if_needed(Vt_nd, shape),
            util.bcast_if_needed(r_nd, shape),
            nodal,
        )
        self._data[..., i0 : i0 + 5] = nodal

        for k in keys:
            self._versions[k] += 1

    def set_rpm(self, rpm):
        """Set reference frame angular velocity in revolutions per minute.

        Converts to rad/s and calls :meth:`set_Omega`.

        Parameters
        ----------
        rpm : float
            Angular velocity of the rotating reference frame [rpm].

        """
        self.set_Omega(rpm * np.pi / 30.0)

    def set_t(self, t):
        """Store circumferential coordinates.

        Parameters
        ----------
        t : array-like
            Circumferential coordinates [rad]. Must be finite and broadcast to block shape.

        """

        if np.any(~np.isfinite(t)):
            raise ValueError("Circumferential coordinates must be finite.")

        self._set_data_by_keys(("t",), t)

    def set_T_s(self, T, s):
        """Store temperature and entropy.

        Set the thermodynamic state by specifying static temperature and entropy per
        unit mass. The velocity field, if present, is preserved.

        Parameters
        ----------
        T : array-like
            Temperature [K]. Must be positive, finite, and broadcast to block shape.
        s : array-like
            Specific entropy [J/kg/K]. Must be finite and broadcast to block shape.

        """
        if np.any(T <= 0) or np.any(~np.isfinite(T)):
            raise ValueError("Temperature must be positive and finite.")
        if np.any(~np.isfinite(s)):
            raise ValueError("Entropy must be finite.")

        self._set_rho_u_nd(
            *self.fluid.set_T_s(T / self.fluid.T_ref, s / self.fluid.Rgas_ref)
        )

    def set_triangulated(self, value):
        """Set whether the data represents triangulated (unstructured) cut data.

        Parameters
        ----------
        value : bool
            True if the block holds triangulated (unstructured) data with shape
            ``(ntri, 3)``; False for a structured quadrilateral mesh.

        """
        super().set_triangulated(value)

    def set_V_Alpha_Beta(self, V, Alpha, Beta):
        r"""Set the velocity vector from speed, yaw angle, and pitch angle.

        The velocity components are

        .. math::

            \begin{aligned}
            V_x      &= V \cos\beta\cos\alpha \\
            V_r      &= V \sin\beta\cos\alpha \\
            V_\theta &= V \sin\alpha
            \end{aligned}

        where :math:`\alpha` is the yaw angle and :math:`\beta` is the pitch
        angle. Trigonometric identities are used to avoid the
        :math:`\tan 90^\circ` singularity.

        Parameters
        ----------
        V : array-like
            Velocity magnitude [m/s]. Must broadcast to block shape.
        Alpha : array-like
            Yaw angle :math:`\alpha` [deg]. Must broadcast to block shape.
        Beta : array-like
            Pitch angle :math:`\beta` [deg]. Must broadcast to block shape.

        """
        # Use trigonometric identities to avoid tan(90 deg) singularity
        cosAlpha = np.cos(np.radians(Alpha))
        sinAlpha = np.sin(np.radians(Alpha))
        cosBeta = np.cos(np.radians(Beta))
        sinBeta = np.sin(np.radians(Beta))

        Vxrt = (
            np.stack(
                (
                    cosBeta * cosAlpha,
                    sinBeta * cosAlpha,
                    sinAlpha,
                ),
                axis=-1,
            )
            * V[..., None]
        )

        self.set_Vx(Vxrt[..., 0])
        self.set_Vr(Vxrt[..., 1])
        self.set_Vt(Vxrt[..., 2])

    def set_Vr(self, Vr):
        """Store radial velocity.

        The thermodynamic state (density and internal energy) is preserved, so
        this may be called before or after a thermodynamic setter such as
        :py:meth:`set_P_T` when building up a flow field.

        If you are setting all three velocity components, prefer
        :meth:`set_Vxrt`, which updates the internal energy only once instead
        of three times for all components.

        Parameters
        ----------
        Vr : array-like
            Radial velocity [m/s]. Must be finite and broadcast to block shape.
        """

        if np.any(~np.isfinite(Vr)):
            raise ValueError("Radial velocity must be finite.")

        rho_nd, u_nd = self._rho_nd_uninit, self._u_nd_uninit
        self._set_data_by_keys(("rhoVr",), rho_nd * Vr / self._V_ref)
        self._update_rhoe_nd(rho_nd, u_nd)

    def set_Vt(self, Vt):
        """Store circumferential velocity.

        The thermodynamic state (density and internal energy) is preserved, so
        this may be called before or after a thermodynamic setter such as
        :py:meth:`set_P_T` when building up a flow field.

        If you are setting all three velocity components, prefer
        :meth:`set_Vxrt`, which updates the internal energy only once instead
        of three times for all components.

        Parameters
        ----------
        Vt : array-like
            Circumferential velocity [m/s]. Must be finite and broadcast to block shape.
        """

        if np.any(~np.isfinite(Vt)):
            raise ValueError("Circumferential velocity must be finite.")

        rho_nd, u_nd = self._rho_nd_uninit, self._u_nd_uninit
        r_nd = self._get_data_by_keys(("r",), raise_uninit=False)
        self._set_data_by_keys(("rhorVt",), rho_nd * r_nd * Vt / self._V_ref)
        self._update_rhoe_nd(rho_nd, u_nd)

    def set_Vx(self, Vx):
        """Store axial velocity.

        The thermodynamic state (density and internal energy) is preserved, so
        this may be called before or after a thermodynamic setter such as
        :py:meth:`set_P_T` when building up a flow field.

        If you are setting all three velocity components, prefer
        :meth:`set_Vxrt`, which updates the internal energy only once instead
        of three times for all components.

        Parameters
        ----------
        Vx : array-like
            Axial velocity [m/s]. Must be finite and broadcast to block shape.
        """

        if np.any(~np.isfinite(Vx)):
            raise ValueError("Axial velocity must be finite.")

        rho_nd, u_nd = self._rho_nd_uninit, self._u_nd_uninit
        self._set_data_by_keys(("rhoVx",), rho_nd * Vx / self._V_ref)
        self._update_rhoe_nd(rho_nd, u_nd)

    def set_Vxrt(self, Vxrt):
        """Store polar velocity components from a single array.

        More efficient than three separate :meth:`set_Vx`, :meth:`set_Vr`,
        :meth:`set_Vt` calls as the energy update is performed only once.

        The thermodynamic state (density and internal energy) is preserved, so
        this may be called before or after a thermodynamic setter such as
        :py:meth:`set_P_T` when building up a flow field.

        Parameters
        ----------
        Vxrt : array-like, shape (..., 3)
            Polar velocity components [m/s], with Vx, Vr, Vt along the last
            axis. Must be finite and broadcast to block shape.

        """

        if Vxrt.shape[-1] != 3:
            raise ValueError(f"Expected Vxrt shape (..., 3), but got {Vxrt.shape}")

        if np.any(~np.isfinite(Vxrt)):
            raise ValueError("Velocity components must be finite.")

        Vx, Vr, Vt = Vxrt[..., 0], Vxrt[..., 1], Vxrt[..., 2]
        rho_nd, u_nd = self._rho_nd_uninit, self._u_nd_uninit
        r_nd = self._get_data_by_keys(("r",), raise_uninit=False)
        self._set_data_by_keys(("rhoVx",), rho_nd * Vx / self._V_ref)
        self._set_data_by_keys(("rhoVr",), rho_nd * Vr / self._V_ref)
        self._set_data_by_keys(("rhorVt",), rho_nd * r_nd * Vt / self._V_ref)
        self._update_rhoe_nd(rho_nd, u_nd)

    def set_wdist(self, wdist):
        """Store distance to nearest wall.

        See :py:attr:`Block.wdist` for more details.

        Parameters
        ----------
        wdist : array-like
            Distance to nearest viscous wall [m]. Must be >=0 and finite,
            and broadcast to block shape.

        """
        if np.any(wdist < 0) or np.any(~np.isfinite(wdist)):
            raise ValueError("wdist must be positive and finite.")
        self._set_data_by_keys(("wdist",), wdist / self.L_ref)

    def set_x(self, x):
        """Store axial coordinates.

        Parameters
        ----------
        x : array-like
            Axial coordinates [m]. Must be finite and broadcast to block shape.

        """
        if np.any(~np.isfinite(x)):
            raise ValueError("Axial coordinates must be finite.")
        self._set_data_by_keys(("x",), x / self.L_ref)

    def set_xrt(self, xrt):
        """Store polar coordinates from a single array.

        Parameters
        ----------
        xrt : array-like, shape (..., 3)
            Polar coordinates, with x [m], r [m], t [rad] along the last axis.
            Must be finite and broadcast to block shape.

        """

        if xrt.shape[-1] != 3:
            raise ValueError(f"Expected xrt shape (..., 3), but got {xrt.shape}")

        x, r, t = xrt[..., 0], xrt[..., 1], xrt[..., 2]
        self.set_x(x)
        self.set_r(r)
        self.set_t(t)

    def set_xyz(self, xyz):
        """Store Cartesian coordinates, converted to polar on write.

        Converts to polar coordinates via:

        .. math::

            r = \\sqrt{y^2 + z^2}

            \\theta = \\mathrm{arctan2}(-z,\\, y)

        Parameters
        ----------
        xyz : array-like, shape (..., 3)
            Cartesian coordinates [m], with x, y, z along the last axis. Must be finite and broadcast to block shape.

        """

        if xyz.shape[-1] != 3:
            raise ValueError(f"Expected xyz shape (..., 3), but got {xyz.shape}")

        if np.any(~np.isfinite(xyz)):
            raise ValueError("Cartesian coordinates must be finite.")

        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        # ember uses z = -r * sin(t), so t = arctan2(-z, y)
        r = np.sqrt(y**2 + z**2)
        t = np.arctan2(-z, y)
        self.set_x(x)
        self.set_r(r)
        self.set_t(t)

    def copy(self, keep_patches=True):
        """Return an independent copy of this block.

        All data arrays, metadata, and derived-property caches are copied so
        that modifications to the returned block do not affect the original.
        Patches are deep-copied by default so each block owns its own patch
        objects; pass ``keep_patches=False`` to get a copy with an empty patch
        collection instead.

        Parameters
        ----------
        keep_patches : bool, optional
            If True (default), all patches are deep-copied onto the new block.
            If False, the returned block has no patches attached.

        Returns
        -------
        Block
            An independent copy of this block.

        """
        import copy as copy_module  # noqa: PLC0415 - only needed on this path

        out = super().copy()

        if not keep_patches:
            out._set_metadata_by_key(
                "patches", ember.collections.BlockPatchCollection(out)
            )
            return out

        # Deep copy all patches to ensure independence between blocks
        copied_patches = [copy_module.deepcopy(patch) for patch in self.patches]

        patch_collection = ember.collections.BlockPatchCollection(out)
        patch_collection.extend(copied_patches)

        # Set the new patch collection on the copied block
        out._set_metadata_by_key("patches", patch_collection)

        return out

    def masked(self, mask):
        r"""Confine subsequent setters to the nodes where `mask` is True.

        Boolean indexing a block (``block[mask]``) cannot be used to write back
        into the original, because numpy advanced indexing returns a copy rather
        than a view. This method works around that: it returns a proxy whose
        ``set_*`` methods apply to the whole block and then roll back every node
        outside the mask, so only masked nodes are changed and all other state
        (including the velocity field preserved by thermodynamic setters) is
        untouched.

        Any setter is supported. The proxy snapshots this block's backing array
        on each setter call, so to keep the copy cheap on a large block, narrow
        it first with a basic-index slice -- a slice is a view, so writes still
        propagate to the parent::

            block[0].masked(mask).set_P_T(1e5, 600.0)

        Parameters
        ----------
        mask : array-like of bool
            Boolean array matching the block shape. Setters modify only the
            nodes where it is True.

        Returns
        -------
        _MaskedBlock
            Proxy whose ``set_*`` methods are confined to the masked nodes.

        Examples
        --------
        Heat only the cold nodes, leaving the rest of the field alone::

            # example: masked
            from ember.block import Block
            from ember.fluid import PerfectFluid
            import numpy as np

            fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.7)
            b = Block((4,))
            b.set_fluid(fluid)
            b.set_x(0.0)
            b.set_r(1.0)
            b.set_t(0.0)
            b.set_P_T(1e5, 300.0)
            b.set_Vx(5.0)
            b.set_Vr(0.0)
            b.set_Vt(0.0)
            b.masked(np.array([True, False, True, False])).set_P_T(1e5, 600.0)
            print(b.T)   # [600. 300. 600. 300.]
            print(b.Vx)  # [5. 5. 5. 5.]

        """
        return _MaskedBlock(self, mask)

    def memory_usage(self):
        """Return memory usage of this block's data, metadata, and cached properties.

        Returns
        -------
        data_usage : dict
            Bytes per data key (equal share of the contiguous backing array).
        metadata_usage : dict
            Bytes per metadata key (nbytes for arrays, sys.getsizeof for others).
        cache_usage : dict
            Bytes per cached property (nbytes for arrays, sys.getsizeof for others).

        """
        import sys  # noqa: PLC0415 - only needed on this debug path

        # Data: each field occupies 1/nvar of the contiguous array
        bytes_per_field = self._data.nbytes // self.nvar
        data_usage = {key: bytes_per_field for key in self._data_keys}

        # Metadata
        metadata_usage = {}
        for key, val in self._metadata.items():
            if isinstance(val, np.ndarray):
                metadata_usage[key] = val.nbytes
            else:
                metadata_usage[key] = sys.getsizeof(val)

        # Cached properties in _store: tuple (version, result) entries from cached_array.
        cache_usage = {}
        for key, entry in self._store.items():
            result = entry[1]
            if isinstance(result, np.ndarray):
                cache_usage[key] = result.nbytes
            else:
                cache_usage[key] = sys.getsizeof(result)

        return data_usage, metadata_usage, cache_usage

    def update_cached_conserved(self):
        """Refresh caches that depend on the conserved variables.

        Bumps the conserved-variable versions so every cached property keyed on
        them recomputes on next access. Only needed if you modify
        :attr:`conserved_nd` directly, as that bypasses the usual cache
        invalidation that happens in the setter methods.

        Unlike :meth:`clear_cache`, this does not clear cached geometry
        such as :attr:`vol_nd`.

        """
        for k in ("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe"):
            self._versions[k] += 1

    def update_primitive(self):
        """Evaluate the primitive cache eagerly.

        Populates :attr:`P_nd`, :attr:`T_nd` and the internal energy behind
        :attr:`u_nd` together, which may save time when done in the solver hot
        loop. Lazily accessing any of those properties afterwards is a fast
        cache hit.

        Three things this forms are NOT published, each because a consumer
        derives it where it walks instead: the velocity and the kinetic energy,
        which the kernels needing them rebuild from the conserved state as they
        sweep rather than reading a stored copy, and the stagnation enthalpy
        (:attr:`ho_nd`, which ``set_residual`` builds from ``cons`` and the
        pressure at its own face corners).

        Returns early when the caches are already current, so calling it
        more often than necessary costs a handful of dict lookups.

        """
        # Raise on uninitialised state exactly as the public properties do:
        # P_nd/T_nd need rho and rhoe, and the kinematic pass the momenta.
        for key in ("rho", "rhoe", "rhoVx", "rhoVr", "rhorVt"):
            self._get_data_by_keys((key,))

        # Every cache filled here depends on the energy as well as the momenta;
        # the kinetic energy, which depended on the radius instead, is no
        # longer among them (it is borrowed below, not cached).
        ver_e = self._get_version(("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe"))
        stamps = (
            ("_u_nd_uninit", ver_e),
            ("P_nd", ver_e),
            ("T_nd", ver_e),
        )
        if all(self._store.get(k, (None,))[0] == v for k, v in stamps):
            return

        shape = self.shape
        # One borrowed nodal buffer serves both passes in turn, because their
        # two throwaways never coexist: the kinetic energy exists only to make
        # `u` inside the kernel, and the static enthalpy only because
        # get_P_h_T produces it beside the pressure and temperature that are
        # wanted. Neither is kept -- `ho_nd` is derived now, set_residual
        # forming it from the conserved state at its own corners -- so the
        # second write lands on top of the first. The arena is free at every
        # call site: each calls this before carving anything of its own (see
        # Grid.update_sources, update_residual, update_timestep).
        scrap = util.carve_view(self.scratch, shape)
        u = self._primitive_buffer("_u_nd_uninit", shape)
        P = self._primitive_buffer("P_nd", shape)
        T = self._primitive_buffer("T_nd", shape)

        # Pass 1: kinematics, fluid-agnostic (velocity is defined by the
        # conserved variables for any fluid).
        ember.fortran.set_primitive_kinematic(
            cons=self.conserved_nd, r=self.r_nd, u=u, halfvsq=scrap
        )
        # Pass 2: thermodynamics, behind the Fluid interface. The enthalpy is
        # written over the kinetic energy, which is dead by now.
        self.fluid.get_P_h_T(self._rho_nd_uninit, u, P, scrap, T)

        for (cache_key, versions), arr in zip(stamps, (u, P, T)):
            arr.flags.writeable = False
            self._store[cache_key] = (versions, arr)

    @derived_array
    def a(self):
        r"""Acoustic speed :math:`a` [m/s], nodal array.

        .. math::
            a^2 = \frac{\partial p}{\partial \rho}\Bigg|_s

        """
        return self.a_nd * self.fluid.V_ref

    @derived_array
    def a_nd(self):
        r"""Nondimensional acoustic speed :math:`a/V_\mathrm{ref}` [-], nodal array.

        Derived, not cached: each access allocates. The equation of state call
        is cheap next to a nodal buffer that would live for the whole run, and
        the solver's one whole-block consumer,
        :meth:`ember.grid.Grid.update_timestep`, does not come through here --
        it writes the same expression into :attr:`scratch` instead. What is
        left are the patch-average consumers (the mixing planes, the
        nonreflecting boundaries and the :mod:`ember.perturbation` matrices
        they drive), whose blocks are a surface rather than a volume. Do not
        put this in a per-node loop over a full block.
        """
        return self.fluid.get_a(self._rho_nd_uninit, self.u_nd)

    @derived_array
    def Alpha(self):
        r"""Absolute yaw angle :math:`\alpha` [deg], nodal array.

        Yaw is the angle between the absolute velocity and its projection onto
        the meridional (x-r) plane, i.e. the out-of-plane swirl angle.

        .. math::
            \tan\alpha = \frac{V_\theta}{V_m}

        """
        return np.degrees(np.arctan2(self.Vt, self.Vm))

    @derived_array
    def Alpha_rel(self):
        r"""Relative-frame yaw angle :math:`\alpha^\mathrm{rel}` [deg], nodal array.

        .. math::
            \tan\alpha^\mathrm{rel} = \frac{V_\theta^\mathrm{rel}}{V_m}

        """
        return np.degrees(np.arctan2(self.Vt_rel, self.Vm))

    @derived_array
    def ao(self):
        r"""Stagnation acoustic speed :math:`a_0` [m/s], nodal array."""
        return self.fluid.get_a(self._rhoo_nd, self._uo_nd) * self.fluid.V_ref

    @derived_array
    def Beta(self):
        r"""Pitch angle :math:`\beta` [deg], nodal array.

        Pitch is the angle between the meridional velocity and the axial
        direction, i.e. the inclination of the flow in the x-r plane.

        .. math::
            \tan\beta = \frac{V_r}{V_x}

        """
        return np.degrees(np.arctan2(self.Vr, self.Vx))

    @derived_array
    def conserved(self):
        r"""Stacked conserved variables :math:`\mathcal{U}`, five-component nodal array.


        .. math::
            \mathcal{U} = \begin{bmatrix}
            \rho \\
            \rho V_x \\
            \rho V_r \\
             \rho r V_\theta \\
            \rho e
            \end{bmatrix}

        Shape ``(ni, nj, nk, 5)`` with components over the last axis.

        """
        # Guard initialisation, then rescale the nondimensional view in place
        # (a stack of five component temps would be the alternative).
        self._get_data_by_keys(("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe"))
        nd = self.conserved_nd
        out = np.empty_like(nd)
        out[..., 0] = nd[..., 0] * self.fluid.rho_ref
        out[..., 1] = nd[..., 1] * self._rhoV_ref
        out[..., 2] = nd[..., 2] * self._rhoV_ref
        out[..., 3] = nd[..., 3] * self._rhoV_ref * self.L_ref
        out[..., 4] = nd[..., 4] * self._rhoV_ref * self.fluid.V_ref
        return out

    @cached_array()
    def conserved_avg_nd(self, out):
        """Time-averaged nodal nondimensional conserved variables, shape (ni, nj, nk, 5).

        Running-mean accumulator built over the final ``n_step_avg`` steps of a
        march. Like :attr:`F_body_nd` this is a no-key cached buffer: allocated
        once, never invalidated, read-only to consumers. Zero-initialised here
        so accumulation starts from a clean slate; its owners
        (:meth:`ember.grid.Grid.accumulate_avg` and
        :meth:`ember.grid.Grid.finalise_average`) toggle ``flags.writeable``
        around their in-place writes.
        """
        cons_avg = util.allocate_or_reuse(out, self.shape + (5,), dtype=np.float32)
        cons_avg.fill(0.0)
        return cons_avg

    @cached_array()
    def conserved_filt_nd(self, out):
        """Low-pass-filtered cell-centred conserved state, shape (ni-1, nj-1, nk-1, 5).

        Stateful selective-frequency-damping scratch: seeded to the current
        cell-averaged conserved state on first access, then evolved each step by
        :meth:`ember.grid.Grid.update_filter` and read by the SFD body force in
        :meth:`ember.grid.Grid.update_sources`. Only allocated when
        ``Solver.gain_filt`` is nonzero, since nothing else touches it. The
        no-key ``cached_array`` allocates it once and never invalidates it;
        read-only to consumers, and its one writer
        (:meth:`~ember.grid.Grid.update_filter`) toggles ``flags.writeable``
        around its writes.
        """
        out = util.allocate_or_reuse(out, self.shape_cell + (5,))
        ember.fortran.node_to_cell(self.conserved_nd, out)
        return out

    @property
    def conserved_nd(self):
        r"""Stacked non-dimensional conserved variables :math:`\mathcal{U}^*`, nodal array with 5 components on last axis.

        .. math::
            \mathcal{U}^* = \begin{bmatrix}
            \rho / \rho_\mathrm{ref} \\
            \rho V_x / \rho_\mathrm{ref} V_\mathrm{ref} \\
            \rho V_r / \rho_\mathrm{ref} V_\mathrm{ref} \\
            \rho r V_\theta / \rho_\mathrm{ref} L_\mathrm{ref} V_\mathrm{ref} \\
            \rho e / \rho_\mathrm{ref} u_\mathrm{ref}
            \end{bmatrix}

        Note that this property is a writable view onto the raw storage array,
        so modifying it will change the flow field without flushing the cache
        of derived properties or performing any validation. It is the low-level
        access point used by the CFD solver hot paths, so it is designed for
        speed rather than safety. Use with caution!

        """
        return self._get_data_by_keys(
            ("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe"),
            writeable=True,
            raise_uninit=False,
        )

    @derived_array
    def cp_nd(self):
        r"""Non-dimensional specific heat at constant pressure :math:`c_p / R_\mathrm{ref}` [-], nodal array.

        Derived, not cached, like :attr:`mu_nd` and :attr:`kappa_nd`; see
        :attr:`mu_nd` for why the three of them are not kept.
        """
        out = util.empty(self.shape)
        self._fill_transport_nd(cp=out)
        return out

    @derived_array
    def dA_quad(self):
        r"""Face area vectors for a 2D structured cut :math:`\delta A` [m^2], shape ``(ni-1, nj-1, 3)``.

        See :attr:`dA_quad_nd` for the nondimensional form and the geometry
        reference.
        """
        return self.dA_quad_nd * self.L_ref**2

    @derived_array
    def dA_quad_nd(self):
        r"""Face area vectors for a 2D structured cut :math:`\delta A / L_\mathrm{ref}^2` [-], shape ``(ni-1, nj-1, 3)``.

        Components on the trailing axis, as for :attr:`dA_quad`.

        See :ref:`face-areas` for the calculation.
        """
        assert self.ndim == 2, "dA_quad is only defined for a two-dimensional cut."
        assert not self.triangulated, "dA_quad requires triangulated=False"
        return _get_da_quad(self._xrt_nd)

    @derived_array
    def dA_tri(self):
        r"""Face area vectors for a 2D unstructured cut :math:`\delta\!A` [m^2], shape ``(ntri, 3)``.

        See :attr:`dA_tri_nd` for the nondimensional form and the geometry
        reference.
        """
        return self.dA_tri_nd * self.L_ref**2

    @derived_array
    def dA_tri_nd(self):
        r"""Face area vectors for a 2D unstructured cut :math:`\delta\!A / L_\mathrm{ref}^2` [-], shape ``(ntri, 3)``.

        Components on the trailing axis, as for :attr:`dA_tri`.

        See :ref:`face-areas` for the calculation.
        """
        if len(self.shape) != 2 or self.shape[1] != 3:
            raise AssertionError(
                f"dA_tri requires triangular block with shape (ntri, 3), "
                f"got shape {self.shape}"
            )
        assert self.triangulated, "dA_tri requires triangulated=True"
        return _get_da_tri(self._xrt_nd)

    @derived_array
    def dAi(self):
        r"""Constant-i face area vectors of a 3D block :math:`\delta A_i` [m^2], shape ``(ni, nj-1, nk-1, 3)``.

        See :attr:`dAi_nd` for the nondimensional form and the geometry reference.
        """
        return self.dAi_nd * self.L_ref**2

    @cached_array("x", "r", "t")
    def dAi_nd(self, out):
        r"""Constant-i face area vectors of a 3D block :math:`\delta A_i / L_\mathrm{ref}^2` [-], components on first axis.

        See :ref:`face-areas` for the calculation.
        """
        # The helper walks in slabs and writes this component-first buffer
        # directly, so neither a whole-block double array nor a transposed
        # copy of the result is ever materialised.
        out = util.allocate_or_reuse(out, (3,) + self.shape_iface)
        return _get_dai(self._xrt_nd, out)

    @derived_array
    def dAj(self):
        r"""Constant-j face area vectors of a 3D block :math:`\delta A_j` [m^2], shape ``(ni-1, nj, nk-1, 3)``.

        See :attr:`dAj_nd` for the nondimensional form and the geometry reference.
        """
        return self.dAj_nd * self.L_ref**2

    @cached_array("x", "r", "t")
    def dAj_nd(self, out):
        r"""Constant-j face area vectors of a 3D block :math:`\delta A_j / L_\mathrm{ref}^2` [-], components on first axis.

        See :ref:`face-areas` for the calculation.
        """
        # The helper walks in slabs and writes this component-first buffer
        # directly, so neither a whole-block double array nor a transposed
        # copy of the result is ever materialised.
        out = util.allocate_or_reuse(out, (3,) + self.shape_jface)
        return _get_daj(self._xrt_nd, out)

    @derived_array
    def dAk(self):
        r"""Constant-k face area vectors of a 3D block :math:`\delta A_k` [m^2], shape ``(ni-1, nj-1, nk, 3)``.

        See :attr:`dAk_nd` for the nondimensional form and the geometry reference.
        """
        return self.dAk_nd * self.L_ref**2

    @cached_array("x", "r", "t")
    def dAk_nd(self, out):
        r"""Constant-k face area vectors of a 3D block :math:`\delta A_k / L_\mathrm{ref}^2` [-], components on first axis.

        See :ref:`face-areas` for the calculation.
        """
        # The helper walks in slabs and writes this component-first buffer
        # directly, so neither a whole-block double array nor a transposed
        # copy of the result is ever materialised.
        out = util.allocate_or_reuse(out, (3,) + self.shape_kface)
        return _get_dak(self._xrt_nd, out)

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def dhdP_rho_nd(self, out):
        r"""Nondimensional derivative of enthalpy wrt. pressure at constant density :math:`(\partial h/\partial p)_\rho \, \rho_\mathrm{ref}` [-]."""
        out = util.allocate_or_reuse(out, self.shape)
        return self.fluid.get_dhdP_rho(self._rho_nd_uninit, self.u_nd, out=out)

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def dhdrho_P_nd(self, out):
        r"""Nondimensional derivative of enthalpy wrt. density at constant pressure :math:`(\partial h/\partial \rho)_p \, \rho_\mathrm{ref} / V_\mathrm{ref}^2` [-]."""
        out = util.allocate_or_reuse(out, self.shape)
        return self.fluid.get_dhdrho_P(self._rho_nd_uninit, self.u_nd, out=out)

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def dsdP_rho_nd(self, out):
        r"""Nondimensional derivative of entropy wrt. pressure at constant density :math:`(\partial s/\partial p)_\rho \, p_\mathrm{ref} / R_\mathrm{ref}` [-]."""
        out = util.allocate_or_reuse(out, self.shape)
        return self.fluid.get_dsdP_rho(self._rho_nd_uninit, self.u_nd, out=out)

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def dsdrho_P_nd(self, out):
        r"""Nondimensional derivative of entropy wrt. density at constant pressure :math:`(\partial s/\partial \rho)_p \, \rho_\mathrm{ref} / R_\mathrm{ref}` [-]."""
        out = util.allocate_or_reuse(out, self.shape)
        return self.fluid.get_dsdrho_P(self._rho_nd_uninit, self.u_nd, out=out)

    @cached_array()
    def dt_vol_nd(self, out):
        """Unscaled volumetric time step (dt/vol) per cell, shape (ni-1, nj-1, nk-1).

        Persistent scratch buffer, not a cache keyed on the conserved state: the
        no-key ``cached_array`` allocates it once and never invalidates it, so
        the lagged ``rf`` relaxation in its writer can blend the new value into
        the previous one. Like every cached property it is read-only to
        consumers; its writer (:meth:`ember.grid.Grid.update_timestep`) toggles
        ``flags.writeable`` around the write (mirrors :attr:`F_body_nd`).
        """
        return util.allocate_or_reuse(out, self.shape_cell)

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def dudP_rho_nd(self, out):
        r"""Nondimensional derivative of internal energy wrt. pressure at constant density :math:`(\partial u/\partial p)_\rho \, \rho_\mathrm{ref}` [-]."""
        out = util.allocate_or_reuse(out, self.shape)
        return self.fluid.get_dudP_rho(self._rho_nd_uninit, self.u_nd, out=out)

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def dudrho_P_nd(self, out):
        r"""Nondimensional derivative of internal energy wrt. density at constant pressure :math:`(\partial u/\partial \rho)_p \, \rho_\mathrm{ref} / V_\mathrm{ref}^2` [-]."""
        out = util.allocate_or_reuse(out, self.shape)
        return self.fluid.get_dudrho_P(self._rho_nd_uninit, self.u_nd, out=out)

    @cached_array()
    def F_body_nd(self, out):
        """Cell-volume-integrated body force, shape (ni-1, nj-1, nk-1, 5).

        Scratch accumulator, not a cached physical field: it is zeroed and
        rebuilt every pre-step (viscous + polar + prescribed + SFD). The no-key
        ``cached_array`` allocates the buffer once and never invalidates it.
        Like every cached property it is read-only to consumers; its owners
        (``Grid.update_sources`` and the FAS coarse-forcing assembly) toggle
        ``flags.writeable`` around their writes. Components are the
        cell-volume-integrated source terms
        ``(rho, rho*Vx, rho*Vr, rho*r*Vt, rho*E)``.
        """
        return util.allocate_or_reuse(out, self.shape_cell + (5,))

    @property
    def flat(self):
        """Flatten all axes into a single axis, returning a view rather than a copy.

        This copies the metadata dict and but clears patches,
        since 2D spatial patches have no meaning on a 1D flattened layout.

        Points are ordered Fortran-style, with the first axis varying fastest,
        matching the column-major layout of the backing array. That ordering is
        what makes the result a view rather than a copy, so anything reshaping
        the result back must pass ``order="F"``. Raises ``ValueError`` if this
        instance is a non-contiguous view that cannot be flattened without
        copying.

        Returns
        -------
        out : Block, shape (npoints,)
            A new instance with all points in a single dimension and no patches.

        """
        out = ember._struct.StructuredData.flat.fget(self)
        out._metadata = self._metadata.copy()
        out._metadata["patches"] = ember.collections.BlockPatchCollection(out)
        return out

    @property
    def fluid(self):
        """:class:`~ember.fluid.PerfectFluid` object for equation of state calculations."""
        if "fluid" not in self._metadata:
            raise ValueError(
                "Working fluid must be set using set_fluid() before accessing fluid properties"
            )
        return self._get_metadata_by_key("fluid")

    @derived_array
    def ho(self):
        r"""Stagnation enthalpy :math:`h_0` [J/kg], nodal array.

        .. math::
            h_0 = h + \frac{1}{2}V^2

        Carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes in :math:`h_0` are physically meaningful, so
        :math:`h_0 \neq c_p T_0`. See :ref:`datum-state`.

        """
        return self.ho_nd * self.fluid.u_ref

    @derived_array
    def ho_nd(self):
        r"""Nondimensional stagnation enthalpy :math:`h_0/u_\mathrm{ref}` [-].

        Derived, not cached: each access allocates. The solver's one whole-block
        consumer, ``set_residual``, does not come through here -- it forms the
        same quantity at the face corners it is already walking, from the
        conserved state and the pressure it is already handed:

        .. math::
            h_0 = u + p/\rho + \tfrac{1}{2}V^2 = e + p/\rho
                = (\rho e + p) / \rho

        which is exact for any fluid, :math:`h = u + p/\rho` being the
        definition of enthalpy rather than an approximation of it. What is left
        here are the patch-average and post-processing readers, whose blocks are
        a surface rather than a volume. Do not put this in a per-node loop over
        a full block.

        Carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes are physically meaningful. See :ref:`datum-state`.
        """
        # Stagnation quantities are undefined without a velocity; require the
        # momenta (r is tolerated, matching the velocity getters).
        self._get_data_by_keys(("rhoVx", "rhoVr", "rhorVt"))
        return self.fluid.get_h(self._rho_nd_uninit, self.u_nd) + (
            self._halfVsq_nd_uninit
        )

    @derived_array
    def ho_rel(self):
        r"""Relative-frame stagnation enthalpy :math:`h_0^\mathrm{rel}` [J/kg], nodal array.

        .. math::
            h_0^\mathrm{rel} = h + \frac{1}{2}{V^\mathrm{rel}}^2

        Carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes in :math:`h_0^\mathrm{rel}` are physically meaningful, so
        :math:`h_0^\mathrm{rel} \neq c_p T_0^\mathrm{rel}`. See
        :ref:`datum-state`.

        """
        return self._ho_rel_nd * self.fluid.u_ref

    @derived_array
    def I(self):  # noqa: E743
        r"""Rothalpy :math:`I` [J/kg], nodal array.

        .. math::
            I = h_0 - U V_\theta = h_0^\mathrm{rel} - \frac{1}{2}U^2

        Carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes in :math:`I` are physically meaningful. See :ref:`datum-state`.

        """
        return self.ho - self.U * self.Vt

    @cached_object
    def i_cusp(self):
        """1-based start and end node indices of the cusp patch, (start, end).

        Returns (0, 0) if the block has no cusp patches.
        """
        for patch in self.patches.cusp:
            lim = patch.ijk_lim_abs
            ist, ien = int(lim[0, 0]), int(lim[0, 1])
            return (ist + 1, ien + 1)
        return (0, 0)

    @cached_object
    def i_perk(self):
        """1-based (i_LE, i_TE) bounding the k-periodic intervals of an H-mesh.

        For a block periodic to itself in k (k=1 coincident with k=nk) over an
        upstream interval at the leading edge and a downstream interval at the
        trailing edge, returns the inclusive end i of the upstream interval and
        the inclusive start i of the downstream interval. These bound the two
        streamwise ranges (1..i_LE and i_TE..ni) over which the k=1/k=nk faces
        are periodic to self; in between (the bladed region) they are not.

        Derived from k-face PeriodicPatches (const_dim == 2): a patch starting
        at i=0 sets the upstream end, a patch ending at i=ni-1 sets the
        downstream start. Returns (0, 0) if the block has no k-face
        PeriodicPatch.

        Cached: patches must not be modified after first access.
        """
        ni = self.shape[0]
        i_le, i_te = 0, 0
        for patch in self.patches.periodic:
            if patch.const_dim != 2:
                continue
            lim = patch.ijk_lim_abs
            ist, ien = int(lim[0, 0]), int(lim[0, 1])
            if ist == 0:  # upstream interval 1..i_le
                i_le = max(i_le, ien + 1)
            if ien == ni - 1:  # downstream interval i_te..ni
                i_te = ist + 1 if i_te == 0 else min(i_te, ist + 1)
        return (i_le, i_te)

    @cached_object
    def ijk_wall_conv(self):
        """Per-face wall indicator dict for the convective (inviscid) kernel.

        Treats all PERMEABLE_TYPES as non-wall. Keys walli1, wallni, wallj1,
        wallnj, wallk1, wallnk, each a float32 array (0.0=wall, 1.0=free) for
        splatting into the inviscid Fortran kernel call (:func:`set_residual`).

        Cached: patches must not be modified after first access.
        """
        iwall, jwall, kwall = self._face_wall_arrays

        def _f(arr):
            return arr.astype(np.float32, order="F")

        return {
            "walli1": _f(~(iwall[0] == 0))[np.newaxis, :, :],
            "wallni": _f(~(iwall[-1] == 0))[np.newaxis, :, :],
            "wallj1": _f(~(jwall[:, 0, :] == 0))[:, np.newaxis, :],
            "wallnj": _f(~(jwall[:, -1, :] == 0))[:, np.newaxis, :],
            "wallk1": _f(~(kwall[:, :, 0] == 0))[:, :, np.newaxis],
            "wallnk": _f(~(kwall[:, :, -1] == 0))[:, :, np.newaxis],
        }

    @cached_object
    def ijk_wall_visc(self):
        """Per-face wall indicator dict for the viscous kernel.

        Like the inviscid :attr:`ijk_wall_conv`, but treats slip (frictionless)
        patches as non-wall in addition to all PERMEABLE_TYPES, so slip walls
        carry zero shear. Keys walli1, wallni, wallj1, wallnj, wallk1, wallnk,
        each a float32 array (0.0=wall, 1.0=free) for splatting into the viscous
        Fortran kernel call.

        Cached: patches must not be modified after first access.
        """
        iwall, jwall, kwall = self._face_wall_arrays_slip

        def _f(arr):
            return arr.astype(np.float32, order="F")

        return {
            "walli1": _f(~(iwall[0] == 0))[np.newaxis, :, :],
            "wallni": _f(~(iwall[-1] == 0))[np.newaxis, :, :],
            "wallj1": _f(~(jwall[:, 0, :] == 0))[:, np.newaxis, :],
            "wallnj": _f(~(jwall[:, -1, :] == 0))[:, np.newaxis, :],
            "wallk1": _f(~(kwall[:, :, 0] == 0))[:, :, np.newaxis],
            "wallnk": _f(~(kwall[:, :, -1] == 0))[:, :, np.newaxis],
        }

    @derived_array
    def kappa_nd(self):
        r"""Non-dimensional thermal conductivity :math:`\kappa^*` [--], nodal array.

        .. math ::
            \kappa^* = \frac{\kappa}
                {\rho_\mathrm{ref} V_\mathrm{ref} R_\mathrm{ref} L_\mathrm{ref}}

        The scaling that leaves :math:`\mathit{Pr} = \mu^* c_p^* / \kappa^*`
        dimensionless, so this is what the viscous kernel's heat flux takes in
        place of the viscosity and Prandtl number it used to be handed.

        Derived, not cached, like :attr:`mu_nd` and :attr:`cp_nd`; see
        :attr:`mu_nd` for why the three of them are not kept.
        """
        out = util.empty(self.shape)
        self._fill_transport_nd(kappa=out)
        return out

    @property
    def L_ref(self):
        r"""Reference length for non-dimensionalisation :math:`L_\mathrm{ref}` [m]."""
        return self._get_metadata_by_key("L_ref")

    @property
    def label(self):
        """String label describing the block."""
        return self._get_metadata_by_key("label")

    @property
    def Ma(self):
        r"""Absolute Mach number :math:`\mathit{M\kern-0.1ema}` [-], nodal array."""
        return self.V / self.a

    @property
    def Ma_rel(self):
        r"""Relative-frame Mach number :math:`\mathit{M\kern-0.1ema}^\mathrm{rel}` [-], nodal array."""
        return self.V_rel / self.a

    @property
    def Mam(self):
        r"""Meridional Mach number :math:`\mathit{M\kern-0.1ema}_m` [-], nodal array."""
        return self.Vm / self.a

    @property
    def Max(self):
        r"""Axial Mach number :math:`\mathit{M\kern-0.1ema}_x` [-], nodal array."""
        return self.Vx / self.a

    @derived_array
    def mu_nd(self):
        r"""Non-dimensional dynamic viscosity :math:`\mu^*` [--], nodal array.

        .. math ::
            \mu^*  = \frac{\mu}{\rho_\mathrm{ref}  V_\mathrm{ref} L_\mathrm{ref}}

        May be thought of as a reciprocal Reynolds number based on the reference scales.

        Nodal rather than one number for the block, because a real gas's
        viscosity is a surface over the field. A perfect gas fills the same
        array with one repeated constant, as it already does for
        :attr:`cp_nd`.

        Derived, not cached: each access allocates. The transport trio (this,
        :attr:`kappa_nd` and :attr:`cp_nd`) is read in one phase of the step
        and nowhere else -- the two viscous kernels of
        :meth:`ember.grid.Grid.update_sources` -- and that phase borrows all
        three from the scratch arena instead of coming through here, so
        caching them meant three nodal volumes sitting allocated for a whole
        run to serve nothing but diagnostics. What is left here are those
        diagnostics and the tests. Do not put this in a per-node loop over a
        full block.
        """
        out = util.empty(self.shape)
        self._fill_transport_nd(mu=out)
        return out

    @property
    def mu_turb(self):
        r"""Turbulent viscosity :math:`\mu_\mathrm{turb}` [kg/m/s]."""
        return self._get_data_by_keys(("mu_turb",))

    @property
    def Nb(self):
        r"""Number of blades in the row containing this block :math:`N_\mathrm{b}` [-], scalar int."""
        return self._get_metadata_by_key("Nb")

    @property
    def Omega(self):
        r"""Reference frame angular velocity :math:`\Omega` [rad/s], scalar float."""
        return self._get_metadata_by_key("Omega")

    @property
    def Omega_nd(self):
        r"""Nondimensional angular velocity :math:`\Omega^*` [--], scalar float.

        .. math::
            \Omega^* = \frac{\Omega L_\mathrm{ref}}{V_\mathrm{ref}}

        """
        return self.Omega * self.L_ref / self.fluid.V_ref

    @cached_object
    def Omega_wall_nd(self):
        """Per-face wall angular velocity dict (nondimensional).

        Keys Omega_walli1_nd, Omega_wallni_nd, etc., each a float32 array of the
        same shape as the corresponding wall array. Defaults to Omega_nd on all
        faces; overridden by RotatingPatch faces.
        """
        ni, nj, nk = self.shape
        omega_nd = self.Omega_nd

        # Initialise all six face arrays to block Omega_nd
        # Shapes: iface boundary = (nj-1, nk-1), jface = (ni-1, nk-1), kface = (ni-1, nj-1)
        oi1 = np.full((nj - 1, nk - 1), omega_nd, dtype=np.float32)
        oni = np.full((nj - 1, nk - 1), omega_nd, dtype=np.float32)
        oj1 = np.full((ni - 1, nk - 1), omega_nd, dtype=np.float32)
        onj = np.full((ni - 1, nk - 1), omega_nd, dtype=np.float32)
        ok1 = np.full((ni - 1, nj - 1), omega_nd, dtype=np.float32)
        onk = np.full((ni - 1, nj - 1), omega_nd, dtype=np.float32)

        for patch in self.patches.rotating:
            patch_omega_nd = np.float32(patch.Omega * self.L_ref / self.fluid.V_ref)
            ijk_face = patch.get_ijk_face()
            if patch.const_dim == 0:  # i-face patch
                i_val = ijk_face[..., 0].flat[0]
                if i_val == 0:
                    oi1[*ijk_face.T[1:]] = patch_omega_nd
                else:
                    oni[*ijk_face.T[1:]] = patch_omega_nd
            elif patch.const_dim == 1:  # j-face patch
                j_val = ijk_face[..., 1].flat[0]
                if j_val == 0:
                    oj1[*ijk_face.T[[0, 2]]] = patch_omega_nd
                else:
                    onj[*ijk_face.T[[0, 2]]] = patch_omega_nd
            elif patch.const_dim == 2:  # k-face patch
                k_val = ijk_face[..., 2].flat[0]
                if k_val == 0:
                    ok1[*ijk_face.T[:2]] = patch_omega_nd
                else:
                    onk[*ijk_face.T[:2]] = patch_omega_nd

        def _f(arr):
            return np.asfortranarray(arr)

        return {
            "omega_walli1_nd": _f(oi1)[np.newaxis, :, :],
            "omega_wallni_nd": _f(oni)[np.newaxis, :, :],
            "omega_wallj1_nd": _f(oj1)[:, np.newaxis, :],
            "omega_wallnj_nd": _f(onj)[:, np.newaxis, :],
            "omega_wallk1_nd": _f(ok1)[:, :, np.newaxis],
            "omega_wallnk_nd": _f(onk)[:, :, np.newaxis],
        }

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def P_nd(self, out):
        r"""Nondimensional static pressure :math:`p^*` [-], nodal array.

        .. math::
            p^* = \frac{p}{p_\mathrm{ref}}

        """
        out = util.allocate_or_reuse(out, self.shape)
        return self.fluid.get_P(self._rho_nd_uninit, self.u_nd, out=out)

    @cached_array()
    def P_offset_nd(self, out):
        """Nondimensional pressure datum for the flux/source kernels [-], scalar.

        Mean nondimensional pressure, computed once on first access and then
        frozen (no data keys, so the cache never invalidates). The flux
        divergence (residual.f90) and the polar source (polar.f90) both subtract
        this datum from the pressure, so a uniform offset cancels exactly and
        only reduces float32 round-off; the converged solution is independent of
        its value, which is why a fixed datum is fine and recomputing it every
        iteration would be needless cost. Returned as a 0-d array so it can be
        locked read-only like other cached properties.
        """
        out = util.allocate_or_reuse(out, ())
        out[...] = self.P_nd.mean()
        return out

    @derived_array
    def P_rot(self):
        r"""Rotation-corrected static pressure :math:`p_\mathrm{rot}` [Pa], nodal array.

        Accounts for the pressure changes due to centrifugal and Coriolis forces in a rotating frame. Calculated by subtracting :math:`\frac{1}{2}U^2` from static enthalpy and then using the equation of state to get the corresponding pressure in an isentropic process.

        """
        # Isentropic pressure at enthalpy reduced by the blade-speed dynamic
        # head, evaluated directly from the equation of state (no block copy).
        h_rot_nd = (
            self.fluid.get_h(self._rho_nd_uninit, self.u_nd)
            - 0.5 * (self.r_nd * self.Omega_nd) ** 2
        )
        rho_nd, u_nd = self.fluid.set_h_s(h_rot_nd, self.s_nd)
        return self.fluid.get_P(rho_nd, u_nd) * self.fluid.P_ref

    @property
    def patches(self):
        """Boundary conditions for the block."""
        return self._get_metadata_by_key("patches")

    @property
    def pitch(self):
        r"""Circumferential pitch [rad].

        .. math::

            \Delta\theta = \frac{2\pi}{N_\mathrm{b}}
        """
        return np.float32(2.0 * np.pi / self.Nb)

    @derived_array
    def Po(self):
        r"""Stagnation pressure :math:`p_0` [Pa], nodal array."""
        return self.fluid.get_P(self._rhoo_nd, self._uo_nd) * self.fluid.P_ref

    @derived_array
    def Po_rel(self):
        r"""Relative-frame stagnation pressure :math:`p_0^\mathrm{rel}` [Pa], nodal array."""
        return self.fluid.get_P(self._rhoo_rel_nd, self._uo_rel_nd) * self.fluid.P_ref

    @derived_array
    def r(self):
        r"""Radial coordinate :math:`r` [m], nodal array."""
        return self.r_nd * self.L_ref

    @cached_object
    def r_mid_nd(self):
        r"""Midspan nondimensional radius, :math:`\tfrac12(\min r_\mathrm{nd} + \max r_\mathrm{nd})` [-].

        Derived from the block's own coordinates (a representative radius for
        this block), not a user-set reference scale -- hence ``_mid`` rather than
        the ``_ref`` suffix carried by the arbitrary fluid/length scales. Used to
        rescale the angular-momentum (``rhorVt``) residual by a radius so its
        magnitude is comparable to the linear-momentum residuals.

        Cached once: the block geometry is fixed for the lifetime of a solve.
        """
        return 0.5 * (self.r_nd.min() + self.r_nd.max())

    @derived_array
    def r_nd(self):
        r"""Nondimensional radial coordinate :math:`r / L_\mathrm{ref}` [-], nodal array"""
        return self._get_data_by_keys(("r",))

    @cached_array()
    def residual_nd(self, out):
        r"""Unintegrated net-flow residual + body forces, shape (ni-1, nj-1, nk-1, 5).

        Sign convention: the residual is the net flux **into** the control volume
        (sum of face flows entering minus those leaving) plus body-force sources,
        i.e. the rate of accumulation :math:`\mathrm{d}U/\mathrm{d}t` of each
        conserved quantity within the cell. It is *unintegrated* -- not yet scaled
        by the local timestep or cell volume.

        Because it points in the direction of accumulation, the increment is
        **added** to (never subtracted from) the conserved variables to take a
        time step::

            conserved_nd += cfl * dt_vol_nd * residual_nd

        At steady state the residual tends to zero. See ``solver.scree_step`` and
        ``solver.advance_rk_stage_mg`` for the integrators that consume it.
        """
        return util.allocate_or_reuse(out, self.shape_cell + (5,))

    @derived_array
    def rho(self):
        r"""Mass density :math:`\rho` [kg/m^3], nodal array."""
        return self.rho_nd * self.fluid.rho_ref

    @derived_array
    def rho_nd(self):
        r"""Non-dimensional mass density :math:`\rho/\rho_\mathrm{ref}` [-], nodal array."""
        return self._get_data_by_keys(("rho",))

    @derived_array
    def rhoe(self):
        r"""Volumetric total energy :math:`\rho e` [J/m^3], nodal array.

        .. math::
            e = u + \tfrac{1}{2}(V_x^2 + V_r^2 + V_\theta^2)

        """
        return self._get_data_by_keys(("rhoe",)) * self._rhoVsq_ref

    @derived_array
    def rhoo(self):
        r"""Stagnation density :math:`\rho_0` [kg/m^3], nodal array."""
        return self._rhoo_nd * self.fluid.rho_ref

    @derived_array
    def rhoo_rel(self):
        r"""Relative-frame stagnation density :math:`\rho_0^\mathrm{rel}` [kg/m^3], nodal array."""
        return self._rhoo_rel_nd * self.fluid.rho_ref

    @derived_array
    def rhorVt(self):
        r"""Volumetric angular momentum :math:`\rho r V_\theta` [kg/m^2/s], nodal array."""
        return self._get_data_by_keys(("rhorVt",)) * self._rhoV_ref * self.L_ref

    @derived_array
    def rhoVm(self):
        r"""Meridional mass flux :math:`\rho V_m` [kg/m^2/s], nodal array."""
        return self.rho * self.Vm

    @derived_array
    def rhoVr(self):
        r"""Volumetric radial momentum :math:`\rho V_r` [kg/m^2/s], nodal array."""
        return self._get_data_by_keys(("rhoVr",)) * self._rhoV_ref

    @derived_array
    def rhoVx(self):
        r"""Volumetric axial momentum :math:`\rho V_x` [kg/m^2/s], nodal array."""
        return self._get_data_by_keys(("rhoVx",)) * self._rhoV_ref

    @property
    def rpm(self):
        """Reference frame revolutions per minute [rpm]"""
        return self.Omega * np.float32(30.0 / np.pi)

    @derived_array
    def rt(self):
        r"""Pseudo-Cartesian circumferential coordinate :math:`r\theta` [m], nodal array."""
        return self.r * self.t

    @derived_array
    def s_nd(self):
        r"""Nondimensional entropy :math:`s / R_\mathrm{ref}` [-].

        Defined relative to an arbitrary datum where :math:`u = s = 0` at
        :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only changes are physically
        meaningful. See :ref:`datum-state`.
        """
        return self.fluid.get_s(self._rho_nd_uninit, self.u_nd)

    @scratch_array
    def scratch(self, out):
        """Shared scratch arena, flat, sized to the most demanding phase of a step.

        Pure transient scratch. This is shared, throwaway kernel
        workspace, NOT a cached value. Its contents are meaningless between
        kernel calls: every consumer overwrites it on entry and nothing may rely
        on what it holds after a kernel returns. Do not read it expecting a
        consistent value; do not stash a reference and assume it survives.

        Owned writeable workspace for Fortran kernels that need transient
        per-node scratch, allocated once and never invalidated. Left writeable
        so callers can pass it straight to an ``intent(inout)`` kernel without
        toggling ``flags.writeable``.

        THE ONE ARENA. Every throwaway buffer in the step comes from here,
        including the six boundary tau/q face buffers (:attr:`tau_q_faces`),
        ``set_visc_force``'s rolling tau/q cell-plane pair, the nodal transport
        trio the viscous kernels read, the nodal acoustic
        speed ``set_timestep_spectral`` reads, ``set_residual``'s
        and ``set_visc_force``'s rolling planes and rows, the IRS work vector,
        and the multigrid coarse scratch. The arena is sized from whichever
        phase needs most, so every phase fits without it being resized.

        THE RULE, and it is the whole safety argument. Buffers that reach the
        same kernel call must come from ONE ``util.carve_view``, which packs
        them end to end and guarantees they alias distinct storage. Buffers in
        different phases may reuse the same span freely, because no two phases
        are live at once -- that invariant is what makes the arena small, and
        it is a contract, not something the code can check. Never carve a
        second view during a phase that is already using the arena.

        This buffer is flat: consumers reshape it through ``carve_view``, so
        needing a particular rank is not a reason to allocate separately.

        If you need storage that must survive *alongside* this one within a
        single kernel call or between calls, see :attr:`store` the persistent
        buffer.
        """
        # First touch only: scratch_array calls this once per block, and the
        # arena is one of the two big solver allocations.
        n = _scratch_len(self.shape)
        logger.debug(
            "alloc: scratch arena %d elements (%.1f MB) for block %s",
            n, n * 4 / 1024**2, self.shape,
        )
        return util.allocate_or_reuse(out, (n,))

    @derived_array
    def sinBeta(self):
        r"""Sine of pitch angle :math:`\sin\beta` [-], nodal array.

        .. math::
            \sin\beta = \frac{V_r}{V_m}

        """
        return self.Vr / self.Vm

    @scratch_array
    def store(self, out):
        """Persistent cross-step solver buffer, nodal shape (ni, nj, nk, 5).

        Counterpart to :attr:`scratch`: a buffer that does carry meaning between
        kernel calls. Unlike :attr:`scratch` its value must survive across calls,
        so no consumer may treat it as throwaway. Used by time integrators to
        carry state between stages.

        Seeded to zeros on first access.

        Returns
        -------
        Array, shape (ni, nj, nk, 5)
        """
        # First touch only, like scratch: the other big solver allocation.
        logger.debug(
            "alloc: store buffer %.1f MB for block %s",
            int(np.prod(self.shape)) * 5 * 4 / 1024**2, self.shape,
        )
        return util.zeros(self.shape + (5,))

    @derived_array
    def t(self):
        r"""Circumferential coordinate :math:`\theta` [rad], nodal array."""
        return self._get_data_by_keys(("t",))

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def T_nd(self, out):
        r"""Nondimensional temperature :math:`T / T_\mathrm{ref}` [-], nodal array."""
        out = util.allocate_or_reuse(out, self.shape)
        return self.fluid.get_T(self._rho_nd_uninit, self.u_nd, out=out)

    @derived_array
    def tanAlpha(self):
        r"""Tangent of absolute yaw angle :math:`\tan\alpha` [-], nodal array.

        .. math::
            \tan\alpha = \frac{V_\theta}{V_m}

        """
        return self.Vt / self.Vm

    @derived_array
    def tanAlpha_rel(self):
        r"""Tangent of relative-frame yaw angle :math:`\tan\alpha^\mathrm{rel}` [-], nodal array.

        .. math::
            \tan\alpha^\mathrm{rel} = \frac{V_\theta^\mathrm{rel}}{V_m}

        """
        return self.Vt_rel / self.Vm

    @derived_array
    def tanBeta(self):
        r"""Tangent of pitch angle :math:`\tan\beta` [-], nodal array.

        .. math::
            \tan\beta = \frac{V_r}{V_x}

        """
        return self.Vr / self.Vx

    @property
    def tau_q_faces(self):
        """Boundary tau/q as six surface buffers: ``(i1, ini, j1, jnj, k1, knk)``.

        WARNING -- PURE TRANSIENT SCRATCH, and a VIEW into :attr:`scratch`,
        not its own allocation. Valid only within a single viscous pass and
        only in the slots that pass refreshes: ``set_tau_q_faces`` writes them,
        ``exchange_faces`` overwrites the halo layer wherever a patch connects,
        and ``set_visc_force`` reads them back -- all sequentially, within one
        :meth:`ember.grid.Grid.update_sources`. Nothing may rely on what they
        hold after that.

        This is the ONLY tau/q that reaches memory. ``set_visc_force`` produces
        interior tau/q inside its own k walk, into a rolling cell-plane pair,
        and reads nothing but the boundary shell from outside it -- so the
        values a viscous pass has to keep are O(surface), which is what these
        buffers hold. They are also all the grid-wide periodic seam exchange
        between the two kernels has to carry, and they are what lets that
        kernel's halo source not depend on the block's topology.

        Each face carries TWO layers on its trailing axis:

        * layer 0, the block's own edge-cell tau/q, written by the boundary
          producer;
        * layer 1, the halo value the face-flux kernel reads. The producer
          seeds it to ``(2*wall - 1) * layer0`` -- ``+edge`` for a permeable or
          slip face, so the boundary face takes the single-sided stress,
          ``-edge`` for a viscous wall, so the face average is zero -- and the
          periodic exchange then overwrites it wherever a patch connects.
          Applying the sign once here is what lets the consumer read the halo
          with no wall mask at all.

        Keeping the two layers apart is what makes that exchange a
        one-directional copy: it reads layer 0 and writes layer 1, which never
        coincide, so it needs no temporary and tolerates a face pairing to
        itself.

        The component axis sits second so that, at a fixed index on the
        trailing spatial axis, the ``(edge, component)`` block is contiguous --
        the order the face-flux kernel walks it in. The cusp seam correction
        reads layer 0 as well: ``k1`` and ``knk`` between them hold cell planes
        1 and nk-1 and both their halos for the whole call, which is exactly
        what that correction needs and what a rolling pair could never give.

        Returns
        -------
        tuple of Array
            ``(i1, ini, j1, jnj, k1, knk)``, shapes ``(nj-1, 9, nk-1, 2)``,
            ``(ni-1, 9, nk-1, 2)`` and ``(ni-1, 9, nj-1, 2)`` respectively,
            all carved from one allocation and therefore mutually disjoint.
        """
        return _carve_viscous(self)[0]


    @derived_array
    def To(self):
        r"""Stagnation temperature :math:`T_0` [K], nodal array."""
        return self.fluid.get_T(self._rhoo_nd, self._uo_nd) * self.fluid.T_ref

    @derived_array
    def To_rel(self):
        r"""Relative-frame stagnation temperature :math:`T_0^\mathrm{rel}` [K], nodal array."""
        return self.fluid.get_T(self._rhoo_rel_nd, self._uo_rel_nd) * self.fluid.T_ref

    @derived_array
    def U(self):
        r"""Blade speed :math:`U` [m/s], nodal array.

        .. math::
            U = \Omega r

        """
        return self.r_nd * self.Omega_nd * self.fluid.V_ref

    @derived_array
    def u(self):
        r"""Specific internal energy :math:`u` [J/kg], nodal array.

        Defined relative to an arbitrary datum where :math:`u = s = 0` at
        :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only changes in :math:`u` are
        physically meaningful, so :math:`u \neq c_v T`. See :ref:`datum-state`.
        """
        return self.u_nd * self._Vsq_ref

    @derived_array
    def u_nd(self):
        r"""Nondimensional specific internal energy :math:`u/u_\mathrm{ref}` [-], nodal array.

        Raises if the thermodynamic state is unset.
        """
        self._get_data_by_keys(("rho",))
        self._get_data_by_keys(("rhoe",))
        return self._u_nd_uninit

    @derived_array
    def uo(self):
        r"""Stagnation internal energy :math:`u_0` [J/kg], nodal array.

        Carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes in :math:`u_0` are physically meaningful, so
        :math:`u_0 \neq c_v T_0`. See :ref:`datum-state`.
        """
        return self._uo_nd * self.fluid.u_ref

    @derived_array
    def uo_rel(self):
        r"""Relative-frame stagnation internal energy :math:`u_0^\mathrm{rel}` [J/kg], nodal array.

        Carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes in :math:`u_0^\mathrm{rel}` are physically meaningful, so
        :math:`u_0^\mathrm{rel} \neq c_v T_0^\mathrm{rel}`. See
        :ref:`datum-state`.
        """
        return self._uo_rel_nd * self.fluid.u_ref

    @derived_array
    def V(self):
        r"""Absolute velocity magnitude :math:`V` [m/s], nodal array."""
        return self.V_nd * self._V_ref

    @derived_array
    def V_nd(self):
        r"""Nondimensional absolute velocity magnitude :math:`V/V_\mathrm{ref}` [-], nodal array."""
        # V = sqrt(2 * half|V|^2), reusing the cached kinetic energy. half|V|^2
        # is tolerant, so guard the momenta here (r tolerated, as for Vx/Vr/Vt).
        self._get_data_by_keys(("rhoVx", "rhoVr", "rhorVt"))
        return np.sqrt(2.0 * self._halfVsq_nd_uninit)

    @derived_array
    def V_rel(self):
        r"""Relative velocity magnitude :math:`V^\mathrm{rel}` [m/s], nodal array.

        .. math::
            V^\mathrm{rel} = \sqrt{V_x^2 + V_r^2 + (V_\theta - \Omega r)^2}

        """
        return np.sqrt(self.Vm**2 + self.Vt_rel**2)

    @derived_array
    def Vm(self):
        r"""Meridional velocity magnitude :math:`V_m` [m/s], nodal array.

        .. math::
            V_m = \sqrt{V_x^2 + V_r^2}

        """
        return np.sqrt(self.Vx**2 + self.Vr**2)

    @derived_array
    def vol(self):
        r"""Volume elements for a 3D block :math:`\delta \mathcal{V}` [m^3], cell array.

        See :attr:`vol_nd` for the nondimensional form and the geometry reference.
        """
        return self.vol_nd * self.L_ref**3

    @cached_array("x", "r", "t")
    def vol_nd(self, out):
        r"""Nondimensional volume elements for a 3D block :math:`\delta \mathcal{V}^*` [-], cell array.

        .. math::
            \delta \mathcal{V}^* = {\delta \mathcal{V}}/{L_\mathrm{ref}^3}

        See :ref:`cell-volumes` for the calculation.
        """
        assert self.ndim == 3, "volume is only defined for a three-dimensional block."
        out = _get_vol(self._xrt_nd, self.dAi_nd, self.dAj_nd, self.dAk_nd, out)
        return out

    @derived_array
    def Vr(self):
        """Radial velocity [m/s]."""
        return self.Vr_nd * self._V_ref

    @derived_array
    def Vr_nd(self):
        r"""Non-dimensional radial velocity :math:`V_r^*` [-], nodal array.

        .. math::
            V_r^* = \frac{V_r}{V_\mathrm{ref}}

        """
        self._get_data_by_keys(("rhoVr",))  # raise if velocity uninitialised
        return self._vel_nd_uninit("rhoVr")

    @derived_array
    def Vt(self):
        r"""Tangential velocity :math:`V_\theta` [m/s], nodal array."""
        # Guard rhorVt but tolerate uninitialised r, so velocities may be read
        # before coordinates are set (matching Vx and Vr).
        self._get_data_by_keys(("rhorVt",))
        return self._vel_nd_uninit("rhorVt") * self._V_ref

    @derived_array
    def Vt_nd(self):
        r"""Non-dimensional tangential velocity :math:`V_\theta/V_\mathrm{ref}` [-], nodal array."""
        # Guard rhorVt before r so uninitialised velocity surfaces as rhorVt.
        self._get_data_by_keys(("rhorVt",))
        self._get_data_by_keys(("r",))
        return self._vel_nd_uninit("rhorVt")

    @derived_array
    def Vt_rel(self):
        r"""Relative-frame tangential velocity :math:`V_\theta^\mathrm{rel}` [m/s], nodal array.

        .. math::
            V_\theta^\mathrm{rel} = V_\theta - \Omega r

        """
        return self._Vt_rel_nd * self.fluid.V_ref

    @derived_array
    def Vt_rel_nd(self):
        r"""Non-dimensional relative tangential velocity :math:`(V_\theta - \Omega r)/V_\mathrm{ref}` [-], nodal array."""
        # Mirror Vt_nd's guards: raise on uninitialised velocity, then coords.
        self._get_data_by_keys(("rhorVt",))
        self._get_data_by_keys(("r",))
        return self._Vt_rel_nd_uninit

    @derived_array
    def Vx(self):
        """Axial velocity :math:`V_x` [m/s], nodal array."""
        return self.Vx_nd * self._V_ref

    @derived_array
    def Vx_nd(self):
        r"""Non-dimensional axial velocity :math:`V_x/V_\mathrm{ref}` [-], nodal array."""
        self._get_data_by_keys(("rhoVx",))  # raise if velocity uninitialised
        return self._vel_nd_uninit("rhoVx")

    @derived_array
    def Vxrt(self):
        r"""Stacked polar velocity vector :math:`\mathbf{V}` [m/s, m/s, m/s], three-component nodal array."""
        # Require velocity, then scale the cached nondimensional stack in a
        # single allocation (no stack of three component temps).
        self._get_data_by_keys(("rhoVx", "rhoVr", "rhorVt"))
        return self._Vxrt_nd_uninit * self._V_ref

    @derived_array
    def Vxrt_nd(self):
        r"""Stacked nondimensional polar velocity :math:`\mathbf{V}/V_\mathrm{ref}` [-], three-component nodal array."""
        # Guard the momenta and r before returning the cached values.
        self._get_data_by_keys(("rho", "rhoVx", "rhoVr", "rhorVt"))
        self._get_data_by_keys(("r",))
        return self._Vxrt_nd_uninit

    @derived_array
    def Vxrt_rel(self):
        r"""Stacked relative-frame velocity vector :math:`\mathbf{V}^\mathrm{rel}` [m/s, m/s, m/s], nodal array of three components."""
        return np.stack((self.Vx, self.Vr, self.Vt_rel), axis=-1)

    @derived_array
    def Vy(self):
        r"""Cartesian y-velocity :math:`V_y` [m/s], nodal array.

        .. math::

            V_y = V_r \cos\theta - V_\theta \sin\theta
        """
        return self._Vxyz[..., 1]

    @derived_array
    def Vz(self):
        r"""Cartesian z-velocity :math:`V_z` [m/s], nodal array.

        .. math::

            V_z = -V_r \sin\theta - V_\theta \cos\theta
        """
        return self._Vxyz[..., 2]

    @derived_array
    def wdist(self):
        """Distance to nearest wall :math:`w` [m], nodal array.

        Defined as the distance from each grid node to the nearest viscous
        wall. Used by the turbulence models to compute turbulent viscosity;
        only required for viscous runs. Usually populated automatically by
        :meth:`~ember.grid.Grid.calculate_wdist` rather than called directly.

        """

        return self.wdist_nd * self.L_ref

    @derived_array
    def wdist_nd(self):
        r"""Nondimensional distance to nearest wall :math:`w/L_\mathrm{ref}` [-], nodal array."""
        return self._get_data_by_keys(("wdist",))

    @cached_array("x")
    def x(self, out):
        """Axial coordinate :math:`x` [m], nodal array."""
        x_nd = self._get_data_by_keys(("x",))
        out = util.allocate_or_reuse(out, x_nd.shape)
        np.multiply(x_nd, self.L_ref, out=out)
        return out

    @derived_array
    def xr(self):
        """Stacked meridional coordinates :math:`(x, r)` [m, m], two-component nodal array."""
        return np.stack((self.x, self.r), axis=-1)

    @derived_array
    def xrrt(self):
        r"""Stacked pseudo-Cartesian coordinates :math:`(x, r, r\theta)` [m, m, m], three-component nodal array."""
        return pol_to_pseudocart(self.xrt)

    @derived_array
    def xrt(self):
        r"""Stacked polar coordinates :math:`(x, r, \theta)` [m, m, rad], three-component nodal array."""
        xrt = self._get_data_by_keys(("x", "r", "t")).copy()
        xrt[..., :2] *= self.L_ref
        return xrt

    @property
    def xrt_nd(self):
        r"""Stacked nondimensional polar coordinates :math:`(x/L_\mathrm{ref}, r/L_\mathrm{ref}, \theta)` [-, -, rad], nodal array of three components."""
        return self._get_data_by_keys(("x", "r", "t"))

    @derived_array
    def y(self):
        r"""Cartesian y-coordinate :math:`y` [m], nodal array.

        .. math::
            y = r \cos\theta

        """
        r = self.xrt[..., 1]
        t = self.xrt[..., 2]
        return r * np.cos(t)

    @derived_array
    def z(self):
        r"""Cartesian z-coordinate :math:`z` [m], nodal array.

        .. math::
            z = -r \sin\theta

        """
        r = self.xrt[..., 1]
        t = self.xrt[..., 2]
        return -r * np.sin(t)

    _data_keys = (
        "x",
        "r",
        "t",
        "rho",
        "rhoVx",
        "rhoVr",
        "rhorVt",
        "rhoe",
        "wdist",
        "mu_turb",
    )
    _defaults = {
        "Nb": 1,
        "Omega": np.float32(0.0),
        "label": None,
        "L_ref": np.float32(1.0),
    }

    #
    # METADATA SETTERS
    #

    #
    # SINGLE VAR SETTERS
    #

    #
    # MULTIVAR SETTERS
    #

    cp = _make_fluid_property(
        "cp",
        r"""Specific heat at constant pressure :math:`c_p` [J/kg/K], nodal array.

        .. math::
            c_p = \frac{\partial h}{\partial T}\Bigg|_p

        """,
        "Rgas",
    )

    cv = _make_fluid_property(
        "cv",
        r"""Specific heat at constant volume :math:`c_v` [J/kg/K], nodal array.

        .. math::
            c_v = \frac{\partial u}{\partial T}\Bigg|_\rho

        """,
        "Rgas",
    )

    gamma = _make_fluid_property(
        "gamma",
        r"""Ratio of specific heats :math:`\gamma` [-].

    .. math::
        \gamma = \frac{c_p}{c_v}

    """,
    )

    h = _make_fluid_property(
        "h",
        r"""Static enthalpy :math:`h` [J/kg], nodal array.

        Carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes in :math:`h` are physically meaningful, so :math:`h \neq c_p T`.
        See :ref:`datum-state`.
        """,
        "u",
    )

    kappa = _make_fluid_property(
        "kappa",
        "Thermal conductivity :math:`\\kappa` [W/m/K], nodal array.",
        "kappa",
    )

    mu = _make_fluid_property(
        "mu", "Dynamic viscosity :math:`\\mu` [kg/m/s], nodal array.", "rhoV"
    )

    P = _make_fluid_property("P", "Static pressure :math:`p` [Pa], nodal array", "P")

    Pr = _make_fluid_property(
        "Pr",
        r"""Prandtl number [-], nodal array.

    .. math ::
        \mathit{Pr} = \frac{c_p \mu}{k}

    """,
    )

    Rgas = _make_fluid_property("Rgas", "Specific gas constant [J/kg/K].", "Rgas")

    s = _make_fluid_property(
        "s",
        r"""Specific entropy :math:`s` [J/kg/K], nodal array.

        Defined relative to an arbitrary datum where :math:`u = s = 0` at
        :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only changes in :math:`s` are
        physically meaningful. See :ref:`datum-state`.
        """,
        "Rgas",
    )

    T = _make_fluid_property("T", "Temperature [K], nodal array.", "T")
