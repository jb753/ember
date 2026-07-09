Solver configuration
====================

.. note::
   The legacy solver loop and its configuration class,
   ``ember.config.SolverConfig``, have both been removed. The current solver's
   configuration is :class:`ember.solver.SolverConfig`, documented below;
   despite the shared class name, it is a different class.

The :doc:`algorithm` page describes how each parameter below influenced
the stages of the legacy time-stepping loop.

.. autoclass:: ember.solver.SolverConfig
   :members:
   :undoc-members:
