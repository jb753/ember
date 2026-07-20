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

.. _block-setters:

Setter methods
==============

All writes to a :class:`Block` go through a setter method, which validates the
input, non-dimensionalises it (see :ref:`block-reference-scales`), and
invalidates any cached derived quantities that depend on it. The setters are:

Geometry:

* :meth:`Block.set_x` -- axial coordinates
* :meth:`Block.set_r` -- radial coordinates
* :meth:`Block.set_t` -- circumferential coordinates
* :meth:`Block.set_xrt` -- all three polar coordinates from one array
* :meth:`Block.set_xyz` -- Cartesian coordinates, converted to polar on write
* :meth:`Block.set_wdist` -- distance to the nearest wall

Kinematics:

* :meth:`Block.set_Vx` -- axial velocity
* :meth:`Block.set_Vr` -- radial velocity
* :meth:`Block.set_Vt` -- circumferential velocity
* :meth:`Block.set_Vxrt` -- all three velocity components from one array
* :meth:`Block.set_V_Alpha_Beta` -- velocity from speed, yaw angle, and pitch angle

Thermodynamic state:

By the two-property rule, each of these takes two independent properties and
inverts the equation of state to recover density and internal energy, leaving
the velocity field untouched. See :ref:`block-equations-of-state` for details.

* :meth:`Block.set_P_T` -- static pressure and temperature
* :meth:`Block.set_P_h` -- static pressure and enthalpy
* :meth:`Block.set_P_s` -- static pressure and entropy
* :meth:`Block.set_P_rho` -- static pressure and density
* :meth:`Block.set_P_rho_nd` -- static pressure and density, non-dimensional
* :meth:`Block.set_rho_u` -- density and internal energy
* :meth:`Block.set_rho_s` -- density and entropy
* :meth:`Block.set_T_s` -- temperature and entropy
* :meth:`Block.set_h_s` -- enthalpy and entropy

Combined:

Five independent properties are enough to fully specify the flow field.

* :meth:`Block.set_conserved` -- the conserved variables directly
* :meth:`Block.set_rho_u_Vxrt_nd` -- density, internal energy, and velocity components, non-dimensional

Metadata:

Scalar properties of the field as a whole, rather than per-node data. The first
two are exceptions: they rescale the raw data in place so that dimensional
values are preserved when the reference scales change.

* :meth:`Block.set_fluid` -- equation of state
* :meth:`Block.set_L_ref` -- reference length scale
* :meth:`Block.set_Omega` -- reference frame angular velocity
* :meth:`Block.set_rpm` -- reference frame angular velocity, in rev/min
* :meth:`Block.set_Nb` -- number of blades in the row containing this block
* :meth:`Block.set_label` -- a string describing the block
* :meth:`Block.set_triangulated` -- flag for triangulated cut data

Miscellaneous:

* :meth:`Block.set_mu_turb` -- turbulent viscosity


Indexing and slicing
====================

A :class:`Block` supports numpy-style indexing and slicing over the spatial axes::

    block[i]          # scalar index -- reduces ndim by one
    block[ist:ien]        # slice -- preserves ndim
    block[i, jst:jen, :]     # mixed index tuple for 3D data

Indexing returns a new :class:`Block` instance that shares the same underlying
backing array as the original (a zero-copy view). Writes to the indexing result
are visible in the original and vice versa.

Array methods
=============

A :class:`Block` provides a family of numpy-style array methods that reshape,
reorder, reduce or copy the block. They all act on the underlying *raw*
variables -- the coordinates and conserved quantities -- and not on derived
thermodynamic properties, which are recomputed from the transformed raw
data on the returned instance.

Views and copies:

* :meth:`Block.view` -- new instance sharing the same data and metadata
* :meth:`Block.copy` -- independent copy of the raw data.
* :meth:`Block.empty` -- fresh uninitialised instance with the same metadata

Reshaping and reordering (a zero-copy view where the layout allows, otherwise a copy):

* :meth:`Block.flat` -- collapse all axes into one
* :meth:`Block.reshape` -- change the axes, keeping the total node count
* :meth:`Block.squeeze` -- drop singleton axes
* :meth:`Block.transpose` -- reorder the axes (reversed by default)
* :meth:`Block.flip` -- reverse indexing along an axis

Reduction over a spatial axis:

* :meth:`Block.mean` -- arithmetic mean of the raw variables
* :meth:`Block.nanmean` -- as above, ignoring NaNs


.. _block-equations-of-state:

Equations of state
==================

:class:`Block` does not implement an equation of state itself.  It stores only
the conserved quantities at grid nodes and delegates every thermodynamic
relation to a :mod:`ember.fluid` equation of state attached by
:meth:`Block.set_fluid()`. The :class:`Block` works in terms of density and internal energy, and the :mod:`fluid` performs calculations to convert from other thermodynamic properties as needed.

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
:mod:`ember.fluid`; see :ref:`reference-scales`.
Three base scales are chosen by the user and passed to the working fluid constructor:
:math:`\rho_\mathrm{ref}`, :math:`V_\mathrm{ref}`, and :math:`R_\mathrm{ref}`.
Three derived thermodynamic scales are then formed:
:math:`p_\mathrm{ref} = \rho_\mathrm{ref} V_\mathrm{ref}^2`,
:math:`u_\mathrm{ref} = V_\mathrm{ref}^2`, and
:math:`T_\mathrm{ref} = V_\mathrm{ref}^2 / R_\mathrm{ref}`.
All six are accessible via the attached fluid at :py:attr:`Block.fluid`.

Spatial coordinates are normalised by a separate reference length
:math:`L_\mathrm{ref}` [m], set via :py:meth:`Block.set_L_ref` and
accessible as :py:attr:`Block.L_ref`.  It defaults to 1.0 and is
independent of the fluid.

At rest, a :class:`Block` stores the raw data in non-dimensional form. Calls to, for example, :meth:`Block.set_P_T` and :meth:`Block.set_Vx` divide the dimensional input by the appropriate reference scale before storage.
Calls to
:meth:`Block.set_L_ref` and :meth:`Block.set_fluid` rescale the raw data in
place to maintain the same dimensional values if the reference scales change.
This keeps the non-dimensional storage completely transparent to the
user.

Non-dimensional versions of dimensional properties such as :attr:`Block.P_nd` and :attr:`Block.Vx_nd` have an `_nd` suffix to distinguish them from the dimensional versions. The same suffix also applies to setters which take non-dimensional inputs like :meth:`Block.set_P_rho_nd`.

Examples
========

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

import ember.struct
import ember.geometry
import ember.perturbation
import ember.collections
import numpy as np
from ember.util import pol_to_pseudocart
from ember import util
from ember.struct import cached_array, cached_object, derived_array, scratch_array
from functools import wraps
import ember.fortran

__all__ = [
    "Block",
]


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


class Block(ember.struct.StructuredData):
    def __init__(self, shape=()):
        """Allocate a structured grid block.

        This is the primary data container for flow fields. It stores coordinates and conserved variables, and provides properties for derived variables such as velocity, pressure and Mach number. All data flows are managed through setter methods that ensure validity and consistency of the flow field. The class also stores boundary patches to specify simulation boundary conditions in :py:attr:`Block.patches`.

        The setters fall into two complementary families: thermodynamic setters
        (e.g. :py:meth:`set_P_T`, :py:meth:`set_rho_u`) set the density and
        internal energy while preserving the velocity field, and velocity
        setters (:py:meth:`set_Vx`, :py:meth:`set_Vr`, :py:meth:`set_Vt`) set
        the velocity while preserving the thermodynamic
        state. Because each family preserves what the other sets, the two may be
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

    def _mixing_refs(self):
        """Reference scales for the [ho, s, Vr, Vt, P] _target stack."""
        return np.array(
            [
                self.fluid.u_ref,
                self.fluid.Rgas_ref,
                self.fluid.V_ref,
                self.fluid.V_ref,
                self.fluid.P_ref,
            ],
            dtype=np.float32,
        )

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
        ref : {None, "Rgas", "P", "V", "u", "T", "rhoV"}
            Reference scale to multiply the nondim output by.
            ``None`` for dimensionless quantities.
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
            return val

        getter.__doc__ = doc
        return property(getter)

    @property
    def _face_wall_arrays(self):
        """Permeable-variant face wall arrays (iwall, jwall, kwall)."""
        return self._get_face_wall_arrays()

    @cached_object
    def _face_wall_arrays_slip(self):
        """Slip-variant face wall arrays (iwall, jwall, kwall).

        Treats slip (frictionless) patches as non-wall, in addition to the
        permeable types handled by :attr:`_face_wall_arrays`.
        """
        return self._get_face_wall_arrays(self.patches.slip)

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "r")
    def _halfVsq_nd_uninit(self, out):
        """Nondimensional half velocity squared [-], tolerating uninitialised data.

        Sums the squared components of the cached nondimensional velocity
        :attr:`_Vxrt_nd_uninit` into its own cached buffer, so the returned
        array is stable across other computations (no shared scratch).
        """
        Vxrt_nd = self._Vxrt_nd_uninit
        out = util.allocate_or_reuse(out, self.shape)
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

    @cached_array("rho", "rhoVr", "rhorVt", "r", "Omega", "fluid")
    def _Vt_rel_nd_uninit(self, out):
        r"""Nondimensional relative tangential velocity [-], own contiguous buffer.

        :math:`V_\theta^\mathrm{rel} = V_\theta - \Omega r`, formed in the
        relative frame so the viscous kernels can differentiate it directly
        (see :meth:`ember.grid.Grid.update_sources`). Tolerates uninitialised
        momenta like :attr:`_Vxrt_nd_uninit`; requires r/Omega via
        :attr:`r_nd` and :attr:`Omega_nd` (blade speed ``U* = r* Omega*``).
        Public access is via :attr:`Vt_rel_nd`, which guards the momenta first.
        """
        out = util.allocate_or_reuse(out, self.shape)
        # Form blade speed U* = r* Omega* in-place in the cache buffer, then
        # Vt_rel = Vt - U* (subtract aliases its second input safely), avoiding
        # a separate r_nd * Omega_nd temporary.
        np.multiply(self.r_nd, self.Omega_nd, out=out)
        np.subtract(self._Vxrt_nd_uninit[..., 2], out, out=out)
        return out

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "r")
    def _Vxrt_nd_uninit(self, out):
        """Nondimensional polar velocity (Vx, Vr, Vt) stacked on the last axis.

        Single source of truth for velocity derived from the conserved
        momenta. Tolerates uninitialised data (does not raise), so it is safe
        to call on a partially built block. Public access is via
        :attr:`Vxrt_nd` and the per-component :attr:`Vx_nd`, :attr:`Vr_nd`,
        :attr:`Vt_nd`, which guard against uninitialised momenta first.

        Vt = rhorVt / (r * rho) is split into two sequential divisions to
        avoid allocating the r*rho temporary array.
        """
        rho = self._get_data_by_keys(("rho",), raise_uninit=False)
        rhoVx = self._get_data_by_keys(("rhoVx",), raise_uninit=False)
        rhoVr = self._get_data_by_keys(("rhoVr",), raise_uninit=False)
        rhorVt = self._get_data_by_keys(("rhorVt",), raise_uninit=False)
        r = self._get_data_by_keys(("r",), raise_uninit=False)
        out = util.allocate_or_reuse(out, self.shape + (3,))
        np.divide(rhoVx, rho, out=out[..., 0])
        np.divide(rhoVr, rho, out=out[..., 1])
        np.divide(rhorVt, rho, out=out[..., 2])  # Vt = rhorVt/rho ...
        out[..., 2] /= r  # ... then /r, avoiding the r*rho temp
        return out

    @derived_array
    def _Vxyz(self):
        """Cartesian velocity components [m/s, m/s, m/s]"""
        from ember import util

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
        has_old = "fluid" in self._metadata
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

            for p in self.patches.inlet:
                p._target_nd = None
                p._Po_nd_target = None
            for p in self.patches.outlet:
                p._P_target_nd = None

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

        for p in self.patches.inlet:
            p._target_nd = None
            p._Po_nd_target = None
        for p in self.patches.outlet:
            p._P_target_nd = None

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
        r"""Write conserved variables from non-dimensional state and velocity.

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
        d = self._data

        # Write the five conserved columns straight into the backing store with
        # np.multiply(out=...), avoiding the (..., 5) np.stack and every product/
        # energy temporary this used to allocate on the bcond hot path. Inputs
        # are freshly-computed by the caller (never views of these columns), so
        # read-as-we-write is safe; they may be face-shaped, and out= broadcasts
        # them up to each column's shape exactly as the old slice-assign did.
        #
        # Energy first: e = u + 0.5*(Vx^2 + Vr^2 + Vt^2), written into the rhoe
        # column. The not-yet-final rhoVr column (i0+2) is borrowed as scratch for
        # the square terms -- it is overwritten with its real value below, before
        # return, so the borrow is invisible to callers (incl. the masked proxy).
        e = d[..., i0 + 4]
        s = d[..., i0 + 2]
        np.multiply(Vx_nd, Vx_nd, out=e)
        np.multiply(Vr_nd, Vr_nd, out=s)
        e += s
        np.multiply(Vt_nd, Vt_nd, out=s)
        e += s
        e *= 0.5
        e += u_nd
        e *= rho_nd  # e now holds rho*e

        # Density + momentum columns (overwrites the borrowed rhoVr scratch last).
        d[..., i0] = rho_nd
        np.multiply(rho_nd, Vx_nd, out=d[..., i0 + 1])
        np.multiply(rho_nd, Vr_nd, out=d[..., i0 + 2])
        np.multiply(rho_nd, r_nd, out=d[..., i0 + 3])
        d[..., i0 + 3] *= Vt_nd

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
        """Set whether the data represents a triangulated mesh.

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
        """Store Cartesian coordinates.

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
        import copy as copy_module

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

    def flat(self):
        """Flatten all axes into a single axis, returning a view rather than a copy.

        This copies the metadata dict and but clears patches,
        since 2D spatial patches have no meaning on a 1D flattened layout.

        Returns
        -------
        out : Block, shape (npoints,)
            A new instance with all points in a single dimension and no patches.

        """
        out = super().flat()
        out._metadata = self._metadata.copy()
        out._metadata["patches"] = ember.collections.BlockPatchCollection(out)
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

    @derived_array
    def a(self):
        r"""Acoustic speed :math:`a` [m/s], nodal array.

        .. math::
            a^2 = \frac{\partial p}{\partial \rho}\Bigg|_s

        """
        return self.a_nd * self.fluid.V_ref

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def a_nd(self, out):
        r"""Nondimensional acoustic speed :math:`a/V_\mathrm{ref}` [-], nodal array."""
        out = util.allocate_or_reuse(out, self.shape)
        return self.fluid.get_a(self._rho_nd_uninit, self.u_nd, out=out)

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
        r"""Stacked conserved variables :math:`U`, nodal array with 5 components on last axis.

        .. math::
            U = \begin{bmatrix}
            \rho \\
            \rho V_x \\
            \rho V_r \\
             \rho r V_\theta \\
            \rho e
            \end{bmatrix}
        """
        # Guard initialisation, then rescale the nondimensional view in place
        # (mirrors conserved_cell, avoiding a stack of five component temps).
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

    @derived_array
    def conserved_cell(self):
        r"""Stacked cell-centered conserved variables :math:`U_\mathrm{cell}`, array with 5 components on last axis.

        .. math::
            U_\mathrm{cell} = \begin{bmatrix}
            \rho \\
            \rho V_x \\
            \rho V_r \\
            \rho r V_\theta \\
            \rho e
            \end{bmatrix}

        Each component is the 8-corner average of the corresponding nodal
        component, with shape ``(ni-1, nj-1, nk-1, 5)``.

        """
        nd = self.conserved_cell_nd
        out = np.empty_like(nd)
        out[..., 0] = nd[..., 0] * self.fluid.rho_ref
        out[..., 1] = nd[..., 1] * self._rhoV_ref
        out[..., 2] = nd[..., 2] * self._rhoV_ref
        out[..., 3] = nd[..., 3] * self._rhoV_ref * self.L_ref
        out[..., 4] = nd[..., 4] * self._rhoV_ref * self.fluid.V_ref
        return out

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def conserved_cell_nd(self, out):
        r"""Stacked non-dimensional cell-centered conserved variables :math:`U^*_\mathrm{cell}`, array with 5 components on last axis.

        .. math::
            U^*_\mathrm{cell} = \begin{bmatrix}
            \rho / \rho_\mathrm{ref} \\
            \rho V_x / \rho_\mathrm{ref} V_\mathrm{ref} \\
            \rho V_r / \rho_\mathrm{ref} V_\mathrm{ref} \\
            \rho r V_\theta / \rho_\mathrm{ref} L_\mathrm{ref} V_\mathrm{ref} \\
            \rho e / \rho_\mathrm{ref} u_\mathrm{ref}
            \end{bmatrix}

        Each component is the 8-corner average of the corresponding nodal
        component of :attr:`conserved_nd`, with shape ``(ni-1, nj-1, nk-1, 5)``.

        """
        out = util.allocate_or_reuse(out, self.shape_cell + (5,))
        ember.fortran.node_to_cell(self.conserved_nd, out)
        return out

    @cached_array()
    def conserved_filt_nd(self, out):
        """Low-pass-filtered cell-centred conserved state, shape (ni-1, nj-1, nk-1, 5).

        Stateful selective-frequency-damping scratch: seeded to the current
        cell-averaged conserved state on first access, then evolved each step by
        ``adapt_cfl`` and read by the SFD body force. The no-key
        ``cached_array`` allocates it once and never invalidates it; read-only
        to consumers, its writers (``set_cfl`` and the restart apply) toggle
        ``flags.writeable`` around their writes.
        """
        out = util.allocate_or_reuse(out, self.shape_cell + (5,))
        ember.fortran.node_to_cell(self.conserved_nd, out)
        return out

    @property
    def conserved_nd(self):
        r"""Stacked non-dimensional conserved variables :math:`U^*`, nodal array with 5 components on last axis.

        .. math::
            U^* = \begin{bmatrix}
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

    @property
    def cp_nd(self):
        r"""Non-dimensional specific heat at constant pressure :math:`c_p / R_\mathrm{ref}` [-], nodal array."""
        return self.fluid._cp_nd

    @derived_array
    def dA_quad(self):
        r"""Face area vectors for a 2D structured cut :math:`\delta A` [m^2], shape `(ni-1, nj-1, 3)`.

        See :ref:`face-areas` for the calculation.
        """
        assert self.ndim == 2, "dA_quad is only defined for a two-dimensional cut."
        assert not self.triangulated, "dA_quad requires triangulated=False"
        return ember.geometry.get_dA_quad(self._xrt_nd) * self.L_ref**2

    @derived_array
    def dA_tri(self):
        r"""Face area vectors for a 2D unstructured cut :math:`\delta\!A` [m^2], shape `(ntri, 3)`.

        See :ref:`face-areas` for the calculation.
        """
        if len(self.shape) != 2 or self.shape[1] != 3:
            raise AssertionError(
                f"dA_tri requires triangular block with shape (ntri, 3), "
                f"got shape {self.shape}"
            )
        assert self.triangulated, "dA_tri requires triangulated=True"
        return ember.geometry.get_dA_tri(self._xrt_nd) * self.L_ref**2

    @derived_array
    def dAi(self):
        r"""Constant-i face area vectors of a 3D block :math:`\delta A_i` [m^2], shape `(ni, nj-1, nk-1, 3)`.

        See :attr:`dAi_nd` for the nondimensional form and the geometry reference.
        """
        return self.dAi_nd * self.L_ref**2

    @cached_array("x", "r", "t")
    def dAi_nd(self, out):
        r"""Constant-i face area vectors of a 3D block :math:`\delta A_i / L_\mathrm{ref}^2` [-], components on first axis.

        See :ref:`face-areas` for the calculation.
        """
        dAi = ember.geometry.get_dAi(self._xrt_nd)
        out = util.allocate_or_reuse(out, (3,) + self.shape_iface)
        out[...] = np.moveaxis(dAi, -1, 0)
        return out

    @derived_array
    def dAj(self):
        r"""Constant-j face area vectors of a 3D block :math:`\delta A_j` [m^2], shape `(ni-1, nj, nk-1, 3)`.

        See :attr:`dAj_nd` for the nondimensional form and the geometry reference.
        """
        return self.dAj_nd * self.L_ref**2

    @cached_array("x", "r", "t")
    def dAj_nd(self, out):
        r"""Constant-j face area vectors of a 3D block :math:`\delta A_j / L_\mathrm{ref}^2` [-], components on first axis.

        See :ref:`face-areas` for the calculation.
        """
        dAj = ember.geometry.get_dAj(self._xrt_nd)
        out = util.allocate_or_reuse(out, (3,) + self.shape_jface)
        out[...] = np.moveaxis(dAj, -1, 0)
        return out

    @derived_array
    def dAk(self):
        r"""Constant-k face area vectors of a 3D block :math:`\delta A_k` [m^2], shape `(ni-1, nj-1, nk, 3)`.

        See :attr:`dAk_nd` for the nondimensional form and the geometry reference.
        """
        return self.dAk_nd * self.L_ref**2

    @cached_array("x", "r", "t")
    def dAk_nd(self, out):
        r"""Constant-k face area vectors of a 3D block :math:`\delta A_k / L_\mathrm{ref}^2` [-], components on first axis.

        See :ref:`face-areas` for the calculation.
        """
        dAk = ember.geometry.get_dAk(self._xrt_nd)
        out = util.allocate_or_reuse(out, (3,) + self.shape_kface)
        out[...] = np.moveaxis(dAk, -1, 0)
        return out

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

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def ho_nd(self, out):
        r"""Nondimensional stagnation enthalpy :math:`h_0/u_\mathrm{ref}` [-].

        Carries an offset dependent on the arbitrary datum state where
        :math:`u = s = 0` at :math:`(p_\mathrm{dtm}, T_\mathrm{dtm})`; only
        changes are physically meaningful. See :ref:`datum-state`.
        """
        # Stagnation quantities are undefined without a velocity; require the
        # momenta (r is tolerated, matching the velocity getters).
        self._get_data_by_keys(("rhoVx", "rhoVr", "rhorVt"))
        out = util.allocate_or_reuse(out, self.shape)
        self.fluid.get_h(self._rho_nd_uninit, self.u_nd, out=out)
        out += self._halfVsq_nd_uninit
        return out

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

    @property
    def mu_nd(self):
        r"""Non-dimensional dynamic viscosity :math:`\mu^*` [--], nodal array.

        .. math ::
            \mu^*  = \frac{\mu}{\rho_\mathrm{ref}  V_\mathrm{ref} L_\mathrm{ref}}

        May be thought of as a reciprocal Reynolds number based on the reference scales.

        """

        return self.fluid._mu_nd / self.L_ref

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
        """Reusable nodal scratch buffer, shape (ni, nj, nk, 5).

        WARNING -- PURE TRANSIENT SCRATCH. This is shared, throwaway kernel
        workspace, NOT a cached value. Its contents are meaningless between
        kernel calls: every consumer overwrites it on entry and nothing may rely
        on what it holds after a kernel returns. Do not read it expecting a
        consistent value; do not stash a reference and assume it survives.

        Owned writeable workspace for Fortran kernels that need transient
        per-node scratch, allocated once and never invalidated. Left writeable
        (see :func:`scratch_array`), so callers pass it straight to an
        ``intent(inout)`` kernel without toggling ``flags.writeable``. Current
        consumers (all sequential, none overlapping):

        - viscous face-flux scratch (slots 0-3) in
          :meth:`ember.grid.Grid.update_sources`;
        - inviscid ``flow`` buffer (all 5 slots) in :attr:`residual_nd`, and the
          per-step increment buffer in ``solver.scree_step`` /
          ``solver.advance_rk_stage_mg``.

        Because nothing persists, each consumer owns the whole buffer for the
        duration of its own call and may treat it as freshly-allocated private
        memory -- the consumer list above is NOT a claim that the slots carry
        distinct, coexisting meanings, only a record of who currently borrows it.
        Two consumers never overlap in time, so the same slots are reused freely.

        DO NOT, however, route a second array *into the same kernel call* that
        already takes this buffer as scratch, by aliasing it onto this storage.
        Within one call the kernel reads and writes its scratch slots freely, so
        any other argument (e.g. an accumulation target) sharing this memory is
        silently corrupted. Concretely: do not point a kernel's output/inout
        argument at ``scratch`` while that same call also receives ``scratch``
        as its workspace. If you need a transient buffer that must survive
        *alongside* this one within a single kernel call or assembly phase,
        allocate a separate one -- do not carve it out of ``scratch``.
        """
        return util.allocate_or_reuse(out, self.shape + (5,))

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

        Counterpart to :attr:`scratch`: a buffer that DOES carry meaning between
        kernel calls. UNLIKE :attr:`scratch` its value must survive across calls,
        so no consumer may treat it as throwaway. It is sized to the nodal shape
        and serves two mutually exclusive integrators (selected by
        ``SolverConfig.n_stage``):

        - Denton lagged march (``n_stage == 0``): holds the ``(dF/dt)_{n-1}`` term
          of the scree extrapolation (``solver.scree_step``) -- written at the end of
          one step and read at the start of the next. That term is cell-shaped, so
          ``scree_step`` takes a leading ``(ni-1, nj-1, nk-1, 5)`` F-order view of
          this buffer (zero copy) and feeds it to the scree kernels (which also
          form the extrapolated ``q = 2*residual - store`` in place in it).
        - Jameson RK march (``n_stage >= 1``): holds the nodal conserved snapshot
          ``U^(0)`` taken at the start of each step; every stage marches off it
          (``solver.advance_rk_stage_mg``).

        Uses the :func:`scratch_array` mechanism (allocated once, never
        invalidated, left writeable for the ``intent(inout)`` kernel write).

        Seeded to zeros so the first Denton step is a pure (doubled) forward step,
        matching multall's zero-initialised residual history. (The RK path
        overwrites it with the conserved snapshot before first use.)

        Returns
        -------
        Array, shape (ni, nj, nk, 5)
        """
        return util.zeros(self.shape + (5,))

    @derived_array
    def t(self):
        r"""Circumferential coordinate :math:`\theta` [rad], nodal array."""
        return self._get_data_by_keys(("t",))

    @cached_array("rho", "rhoVx", "rhoVr", "rhorVt", "rhoe")
    def T_nd(self, out):
        r"""Nondimensional temperature :math:`T / T_\mathrm{ref}` [-]."""
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

    @scratch_array
    def tau_q_halo(self, out):
        """Halo-padded viscous stress / heat-flux scratch, shape (ni+1, nj+1, nk+1, 9).

        WARNING -- PURE TRANSIENT SCRATCH. This is throwaway kernel workspace,
        NOT a cached value. Its contents are
        valid only *within* a single viscous pass and *only* in the slots that
        pass refreshes: the tau/q phase writes the owned cells, ``exchange_halos``
        fills the periodic neighbour slots, then the face-flux phase reads them
        back -- all sequentially, within one :meth:`ember.grid.Grid.update_sources`. Nothing
        may rely on what it holds after that. (Verified: the viscous force is
        bit-identical even if the entire buffer is poisoned before the pass,
        because the pass re-derives every slot it reads.)

        Unlike :attr:`scratch`, the slots DO carry coordinated meaning *within
        a single viscous pass*: the three sub-steps cooperate on the same data
        (writer -> halo fill -> reader), so the tau/q layout documented below is
        real for the duration of that pass. That coordination does not extend
        beyond the pass -- once it ends the buffer is pure private scratch again.

        Because it carries no state between passes, the flat buffer doubles as
        the coarse block-sum accumulator and separable-prolong scratch in
        ``solver.advance_rk_stage_mg`` (see that function's docstring) and,
        likewise, in ``solver.scree_step``'s multigrid path (``n_levels >= 1``,
        calling ``ember.fortran.scree_mg_irs``/``scree_mg_noirs``), where the (i,j,k) layout
        inside is irrelevant -- only the element count matters. Each borrower
        (a viscous pass, or one scree/RK-stage multigrid call) owns the whole
        buffer for its own duration and may treat it as freshly-allocated
        private memory; borrowers never overlap in time.

        DO NOT alias a second array onto this storage and pass both into the
        same kernel call that already takes this buffer (as scratch or as the
        tau/q workspace): the kernel writes these slots freely, so any other
        argument sharing the memory is silently corrupted. If you need a buffer
        that must survive *alongside* this one, allocate a separate one -- do
        not carve it out of ``tau_q_halo``.

        Left writeable (see :func:`scratch_array`), so the viscous passes and
        the periodic exchange write through it without toggling
        ``flags.writeable``.

        Returns
        -------
        Array, shape (ni+1, nj+1, nk+1, 9)
            Slots 0-5: tau_cell (6 components), slots 6-8: q_cell (3 components).
            Owned cells occupy indices [1:ni+1, 1:nj+1, 1:nk+1] (0-based),
            i.e. Fortran indices 2..ni, 2..nj, 2..nk.
            Halo slots at index 0 and ni (etc.) are reserved for periodic
            neighbour exchange.
        """
        ni, nj, nk = self.shape
        # 10 slots, not 9: the viscous pass uses slots 0-8 (tau/q); the spare
        # capacity is kept for the transient borrowers, carved from this
        # storage while it is dead outside the viscous pass -- the fused
        # inviscid residual's rolling flow buffers (set_residual, a small
        # leading span) and the multigrid coarse-correction scratch
        # (solver._mg_coarse_carve).
        return util.allocate_or_reuse(out, (ni + 1, nj + 1, nk + 1, 10))

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

        Raises if the thermodynamic state is unset; for the
        uninitialised-tolerant form see :attr:`_u_nd_uninit`.
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
        # V = sqrt(2 * half|V|^2), reusing the cached kinetic energy. half|V|^2
        # is tolerant, so guard the momenta here (r tolerated, as for Vx/Vr/Vt).
        self._get_data_by_keys(("rhoVx", "rhoVr", "rhorVt"))
        return np.sqrt(2.0 * self._halfVsq_nd_uninit) * self._V_ref

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
    def Vm_nd(self):
        r"""Nondimensional meridional velocity magnitude :math:`V_m^*` [-].

        .. math::
            V_m^* = V_m / V_\mathrm{ref}

        """
        return np.sqrt(self.Vx_nd**2 + self.Vr_nd**2)

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
        out = ember.geometry.get_vol(
            self._xrt_nd, self.dAi_nd, self.dAj_nd, self.dAk_nd, out
        )
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
        return self._Vxrt_nd_uninit[..., 1]

    @derived_array
    def Vt(self):
        r"""Tangential velocity :math:`V_\theta` [m/s], nodal array."""
        # Guard rhorVt but tolerate uninitialised r, so velocities may be read
        # before coordinates are set (matching Vx and Vr).
        self._get_data_by_keys(("rhorVt",))
        return self._Vxrt_nd_uninit[..., 2] * self._V_ref

    @derived_array
    def Vt_nd(self):
        r"""Non-dimensional tangential velocity :math:`V_\theta/V_\mathrm{ref}` [-], nodal array."""
        # Guard rhorVt before r so uninitialised velocity surfaces as rhorVt.
        self._get_data_by_keys(("rhorVt",))
        self._get_data_by_keys(("r",))
        return self._Vxrt_nd_uninit[..., 2]

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
        return self._Vxrt_nd_uninit[..., 0]

    @derived_array
    def Vxrt(self):
        r"""Stacked polar velocity vector :math:`\mathbf{V}` [m/s, m/s, m/s], nodal array of three components."""
        # Require velocity, then scale the cached nondimensional stack in a
        # single allocation (no stack of three component temps).
        self._get_data_by_keys(("rhoVx", "rhoVr", "rhorVt"))
        return self._Vxrt_nd_uninit * self._V_ref

    @derived_array
    def Vxrt_nd(self):
        r"""Stacked nondimensional polar velocity :math:`\mathbf{V}/V_\mathrm{ref}` [-], nodal array of three components."""
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

    @cached_array("wdist")
    def xlen_sq_nd(self, out):
        r"""Mixing-length squared :math:`(\kappa w)^2` [-], cell-shaped.

        The turbulent mixing length is :math:`\kappa w` with von Karman
        constant :math:`\kappa = 0.41` and :math:`w` the cell-averaged wall
        distance. Any mixing-length cap is baked into :attr:`wdist_nd` upstream
        by :meth:`~ember.grid.Grid.calculate_wdist`, so none is applied here.
        Cached against the ``wdist`` data key: recomputed only when the wall
        distance changes.
        """
        out = util.allocate_or_reuse(out, self.shape_cell)
        node = util.zeros(self.shape)
        node[...] = self.wdist_nd
        ember.fortran.node_to_cell(node, out)
        out *= 0.41
        out *= out
        return out

    @derived_array
    def xr(self):
        """Stacked meridional coordinates :math:`(x, r)` [m, m], nodal array of two components."""
        return np.stack((self.x, self.r), axis=-1)

    @derived_array
    def xrrt(self):
        r"""Stacked pseudo-Cartesian coordinates :math:`(x, r, r\theta)` [m, m, m], nodal array of three components."""
        return pol_to_pseudocart(self.xrt)

    @derived_array
    def xrt(self):
        r"""Stacked polar coordinates :math:`(x, r, \theta)` [m, m, rad], nodal array of three components."""
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

    T = _make_fluid_property("T", "Temperature [K].", "T")
