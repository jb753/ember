ember
=====

An 'Enhanced Multi-Block solvER' for turbomachinery computational fluid
dynamics, written by `James Brind <https://jamesbrind.uk/>`_ at the `Whittle Laboratory <https://whittle.eng.cam.ac.uk/>`_, University of Cambridge.
Solves the compressible Reynolds-averaged Navier-Stokes equations on
multi-block structured meshes, using an evolution of the fast and robust Denton
algorithms. Pre- and post-processing is handled through a numpy-like Python
interface, while the heavy computations run through compiled Fortran for speed.




.. toctree::
   :maxdepth: 2
   :caption: Getting started

   install
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference manual

   coordinate_system
   api/fluid
   api/block
   api/grid
   api/patch
   api/average
   api/cut
   api/solver
   api/convergence_history
   references

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
