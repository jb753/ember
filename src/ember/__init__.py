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
import importlib.metadata

# Set default OpenMP thread limit before importing Fortran extensions
# The Fortran code uses OpenMP for parallel loops. Based on profiling, performance
# peaks at 4-5 threads on typical workstations due to memory bandwidth limitations.
# Beyond this, overhead from thread management outweighs benefits from parallelization.
# Users can override this default by setting OMP_NUM_THREADS in their environment
# before importing ember, e.g.: OMP_NUM_THREADS=8 python script.py
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "5"

__all__ = [
    "fluid",
    "block",
    "grid",
    "patch",
    "cut",
    "average",
    "cases",
]

__version__ = importlib.metadata.version("ember-cfd")
__copyright__ = "2025 James Brind"
