Communicators
=============

A communicator exchanges data across the seam between blocks for one patch
type. :attr:`~ember.grid.Grid.connectivity` builds and caches one per patch
type on first use -- reached as ``grid.connectivity.periodic``, ``.mixing``
and ``.nonmatch`` -- pairing each patch to its partner on a neighbouring
block the first time it is used. Pairing is therefore automatic, triggered by
the first exchange, e.g. a call to :meth:`~ember.grid.Grid.apply_bconds`. See
:ref:`grid-connectivity` for how the pairing cache is invalidated on a
topology change.

The pairings can be inspected directly, by calling say
``grid.connectivity.periodic.pair()`` for periodic patches. It returns a dict
keyed by the ``(bid, pid)`` identifier of each patch, indexing like
``grid[bid].patches[pid]``. The dict values are the corresponding ``(bid,
pid)`` of the patch it matches and the geometric transform between the two.
Both halves of a pair appear as keys, so the mapping can be followed from
either side::

    pairs = grid.connectivity.periodic.pair()
    # block 0 patch 0 is paired with block 1 patch 0, and vice versa
    pairs[(0, 0)]  # ((1, 0), transform)
    pairs[(1, 0)]  # ((0, 0), transform)

Each communicator below is built lazily by
:class:`~ember.grid.GridConnectivity` from the pairing above, and exposes
``apply()`` (write the exchange straight into ``block.conserved``) and,
where the patch type needs it, ``exchange()`` (compute the cross-plane
targets without applying them).

.. automodule:: ember.periodic_communicator
   :members:
   :undoc-members:

.. automodule:: ember.mixing_communicator
   :members:
   :undoc-members:

.. automodule:: ember.nonmatch_communicator
   :members:
   :undoc-members:
