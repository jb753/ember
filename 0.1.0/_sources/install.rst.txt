Installation
============

Supported versions and platforms
---------------------------------

.. note::
   Python environments may seem `complex <https://xkcd.com/1987/>`_ to new
   users. See the official tutorial on
   `Virtual Environments and Packages <https://docs.python.org/3/tutorial/venv.html>`_
   for why they are necessary, and PyPA's
   `Tool recommendations <https://packaging.python.org/en/latest/guides/tool-recommendations/>`_.
   ember requires Python |PythonVersion| or newer, and is incompatible with
   older versions of Python that may be provided by your operating system's
   package manager.

   The author's preference is `uv <https://docs.astral.sh/uv/>`_, as it is
   the fastest solution, and manages both virtual environments and Python
   versions.

ember follows `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_ for
its Python support window, currently Python 3.12 or newer. We
only test and provide precompiled wheels on Linux. The code may possibly be
coerced to work on other operating systems, with some modifications, but you
will need a working Fortran toolchain to build it from source.

Via the Python Package Index
----------------------------

The most convenient way to install ember is via the Python Package Index (PyPI).
Ask for the distribution name ``ember-cfd``, not the import name ``ember``:

.. code-block:: bash

   pip install ember-cfd

We precompile and distribute wheels for Linux (x86_64), so if your system
is the same you should not need a Fortran compiler to install from PyPI. On other
systems, pip will automatically attempt to build from source, which requires a
Fortran toolchain. Install these packages from your distribution's package
manager, for example on Debian/Ubuntu:

.. code-block:: bash

   apt-get update
   apt-get install -y gfortran meson ninja-build

The pip command above should then work without errors.

.. note::
   On some cloud or CI images, the system ``gfortran``
   symlink points at an uninstalled ``gfortran-<N>`` package. If the build
   fails with a missing ``gfortran`` error, run:

   .. code-block:: bash

      apt-get update
      apt-get install -y --fix-broken

   to pull in the missing compiler and fix the symlink.

From source
-----------

For performance tuning, or if you want to modify the code, you can build ember
from source. You will need a Fortran compiler and build system:

.. code-block:: bash

   apt-get update
   apt-get install -y gfortran meson ninja-build

Clone the repository and make an editable install with pip:

.. code-block:: bash

   git clone git@github.com:jb753/ember.git
   cd ember
   pip install -e .

Note that edits to ``.f90`` files are *not* picked up automatically and need
``pip install -e .`` rerun; but edits to Python files are picked up immediately.
This command always recompiles the Fortran extension from scratch, as when
installing from a local directory there is no incremental build or wheel cache
to go stale.

Verifying the install
----------------------

Run this one-liner to check the Fortran extension is working:

.. code-block:: bash

   python -c "from ember.fortran import set_residual; print('OK')"

Performance tuning
-------------------

By default the Fortran extension is built for a portable x86_64 baseline
(compiler flag ``-march=haswell``), so a wheel or source build works on any
Haswell-or-newer machine. For a build tuned to the exact CPU it will run
on, replace that flag by setting the environment variable ``EMBER_MARCH``
before installing:

.. code-block:: bash

   EMBER_MARCH="-march=native -mtune=native" pip install -e .

Both GNU and Intel Fortran toolchains are supported. To build with Intel
compilers instead of gfortran, set ``EMBER_COMPILER=ifort`` before installing:

.. code-block:: bash

   EMBER_COMPILER=ifort pip install -e .

Compiler flags can be further customised by editing ``setup.py``. See the ``tools/compile_wilkes.sh`` script for an example of how to build on an HPC cluster.
