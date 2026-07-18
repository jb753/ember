"""Enhanced Multi-block solvER (EMBER) for turbomachinery computational fluid dynamics.

EMBER is a Python-based CFD solver designed for turbomachinery applications, featuring
multi-block structured grid capabilities, explicit time integration schemes, and specialized
tools for analyzing compressor and turbine flows. The package provides comprehensive support
for grid generation, flow field initialization, boundary condition specification, time-accurate
simulations, and post-processing including circumferential averaging and performance metrics
extraction. Built on NumPy with Fortran-accelerated numerics, EMBER emphasizes ease of use
for rapid prototyping while maintaining computational efficiency for production simulations.
"""

import os

# Set default OpenMP thread limit before importing Fortran extensions.
# The Fortran code uses OpenMP for parallel loops. Default to single-threaded so
# that many independent runs (e.g. a parametric sweep) can be packed onto cores
# without oversubscription -- each process pins to one thread, and wall-time
# measurements stay contention-free. On a memory-bandwidth-bound workload a
# single run peaks at only ~4-5 threads anyway.
# Users can override this default by setting OMP_NUM_THREADS in their environment
# before importing ember, e.g.: OMP_NUM_THREADS=8 python script.py
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

__all__ = [
    "fluid",
    "block",
    "grid",
    "patch",
    "cut",
    "average",
    "cases",
]

# _version.py is generated at build time by setuptools-scm (see pyproject.toml);
# it is gitignored, so fall back to a runtime metadata lookup when running from
# an uninstalled source tree that has never been built.
try:
    from ember._version import __version__
except ImportError:  # pragma: no cover - only hit in a bare, unbuilt source tree
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("ember-cfd")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__copyright__ = "2025 James Brind"
