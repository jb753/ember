Solver
======

The explicit time-marching loop, its integrators, and their configuration.
See :doc:`../algorithm` for a prose description of how each stage and
parameter fits into the time-stepping loop.

.. note::
   The legacy solver loop and its configuration class,
   ``ember.config.SolverConfig``, have both been removed. The current
   solver's configuration, :class:`ember.solver.SolverConfig` documented
   below, despite the shared class name, is a different class.

.. automodule:: ember.solver
   :members:
   :undoc-members:
