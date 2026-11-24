r"""Flow data on an unstructured triangulated surface.

An :class:`UnstructuredBlock` holds flow data at a set of vertices together
with the triangular faces that join them. It is what
:func:`ember.cut.unstructured` returns, and it is accepted by the integrals
and averages in :mod:`ember.average` alongside a structured 2D cut.

Each vertex is stored once, however many faces share it. Every flow
property of a :class:`~ember.block.Block` is available, evaluated per
vertex, with shape ``(n_vert,)``; face quantities have shape
``(n_face, ...)``.

Construction mirrors :class:`~ember.block.Block`: the constructor sizes the
vertex data, and setters fill it in, connectivity included::

    u = UnstructuredBlock(n_vert)
    u.set_faces(faces)  # int, shape (n_face, 3)
    u.set_xrt(xrt)  # shape (n_vert, 3)
    u.set_fluid(fluid)
    u.set_P_T(P, T)
    u.set_Vxrt(Vxrt)

.. autosummary::

   UnstructuredBlock
"""

import numpy as np

import ember.block
from ember._struct import derived_array

__all__ = ["UnstructuredBlock"]


def _like(block, n_vert):
    """New :class:`UnstructuredBlock` carrying the point metadata of `block`.

    The fluid, length scale, rotation, blade count and label come across;
    the patches of a structured `block` do not. The data are left NaN for
    the caller to fill, and are marked set, as :meth:`Block.empty
    <ember._struct.StructuredData.empty>` marks them.
    """
    out = UnstructuredBlock(n_vert)
    out._metadata.update(
        {k: v for k, v in block._metadata.items() if k not in ("patches", "faces")}
    )
    out._versions = block._versions.copy()
    return out


def _not_on_unstructured(name):
    """Raise for an array operation that would scramble the face indices."""
    raise TypeError(
        f"{name} is not supported on an UnstructuredBlock: it would reorder "
        "or regroup the vertices that the faces index into."
    )


class UnstructuredBlock(ember.block._PointBlock):
    def __init__(self, n_vert):
        r"""Allocate flow data at the vertices of an unstructured surface.

        The vertex data hold every variable a :class:`~ember.block.Block`
        does, and are set and read through the same methods and properties.
        The triangles joining the vertices are set with :meth:`set_faces`.

        Each face is a triple of vertex indices :math:`(A, B, C)`. Its area
        vector, :attr:`dA`, follows the right-hand rule around
        :math:`A \to B \to C` in the pseudo-Cartesian space
        :math:`(x, r, r\theta)`, the same sense as the face areas of a
        structured cut.

        The vertices form a one-dimensional array, so the inherited
        :attr:`ndim` is 1 whatever the dimension of the surface. Operations
        that reorder or regroup the vertices (indexing, :meth:`flip`,
        :meth:`reshape` and the like) raise, because they would leave the
        faces pointing at the wrong vertices.

        A volume mesh would extend this class rather than replace it: each
        face would also store the cells either side of it, and the cells,
        their volumes and their vertex lists would be derived from those.

        Parameters
        ----------
        n_vert : int
            Number of vertices.

        """
        super().__init__(shape=(n_vert,))

    def __getitem__(self, key):
        """Raise: indexing would leave the faces pointing at the wrong vertices."""
        _not_on_unstructured("Indexing")

    def __setstate__(self, state):
        """Unpickle, locking the faces again (pickle drops the read-only flag)."""
        super().__setstate__(state)
        if "faces" in self._metadata:
            self._metadata["faces"].flags.writeable = False

    def _to_faces(self, nodal):
        """Average vertex values onto faces, shape ``(n_face, ...)``."""
        return np.mean(np.asarray(nodal)[self.faces], axis=1)

    def set_faces(self, faces):
        """Set the triangles joining the vertices.

        The array is copied and stored read-only; set new faces to change
        them.

        Parameters
        ----------
        faces : array_like of int, shape (n_face, 3)
            Indices of the three vertices of each face, each in
            ``[0, n_vert)``. Their order sets the direction of the face's
            area vector by the right-hand rule.

        """
        faces = np.asarray(faces)
        if not np.issubdtype(faces.dtype, np.integer):
            raise TypeError(f"Faces must be integer indices, got {faces.dtype}.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Faces must have shape (n_face, 3), got {faces.shape}.")
        if faces.size and (faces.min() < 0 or faces.max() >= self.n_vert):
            raise ValueError(
                f"Face indices must lie in [0, {self.n_vert}), got "
                f"[{faces.min()}, {faces.max()}]."
            )
        faces = np.array(faces, dtype=np.int32)
        faces.flags.writeable = False
        self._set_metadata_by_key("faces", faces)

    def empty(self, shape=()):
        """Create an uninitialised structured block with the same metadata.

        The result is a plain :class:`~ember.block.Block` of the given
        shape, without faces: this is how a mixed-out state or a resampled
        structured cut is made from an unstructured one.

        Parameters
        ----------
        shape : tuple, optional
            Shape of the new block. Defaults to scalar.

        Returns
        -------
        out : Block
            A new uninitialised block with the fluid, length scale, rotation,
            blade count and label of this one.

        """
        out = ember.block.Block(shape=shape)
        out._metadata.update({k: v for k, v in self._metadata.items() if k != "faces"})
        out._versions = self._versions.copy()
        return out

    def flip(self, axis):
        """Raise: flipping would reorder the vertices."""
        _not_on_unstructured("flip")

    def mean(self, axis=0, keepdims=False):
        """Raise: use the face-weighted averages in :mod:`ember.average`."""
        _not_on_unstructured("mean")

    def nanmean(self, axis=0):
        """Raise: use the face-weighted averages in :mod:`ember.average`."""
        _not_on_unstructured("nanmean")

    def reshape(self, shape):
        """Raise: reshaping would regroup the vertices."""
        _not_on_unstructured("reshape")

    def squeeze(self):
        """Raise: squeezing would regroup the vertices."""
        _not_on_unstructured("squeeze")

    def transpose(self, axes=None):
        """Raise: transposing would reorder the vertices."""
        _not_on_unstructured("transpose")

    @derived_array
    def dA(self):
        r"""Face area vectors :math:`\delta\!A` [m^2], shape ``(n_face, 3)``.

        See :attr:`dA_nd` for the nondimensional form and the geometry
        reference.
        """
        return self.dA_nd * self.L_ref**2

    @derived_array
    def dA_nd(self):
        r"""Face area vectors :math:`\delta\!A / L_\mathrm{ref}^2` [-], shape ``(n_face, 3)``.

        Pseudo-Cartesian components :math:`(x, r, r\theta)` on the trailing
        axis, directed by the right-hand rule around each face's vertices.

        See :ref:`face-areas` for the calculation.
        """
        return ember.block._get_da_tri(self._xrt_nd[self.faces])

    @property
    def faces(self):
        """Vertex indices of each face, read-only int32, shape ``(n_face, 3)``."""
        if "faces" not in self._metadata:
            raise ValueError("Faces have not been set: call set_faces first.")
        return self._metadata["faces"]

    @property
    def n_face(self):
        """Number of faces."""
        return self.faces.shape[0]

    @property
    def n_vert(self):
        """Number of vertices."""
        return self.shape[0]
