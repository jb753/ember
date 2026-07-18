"""Enhanced Multi-block solvER (EMBER) for turbomachinery computational fluid dynamics.

EMBER is a Python-based CFD solver designed for turbomachinery applications, featuring
multi-block structured grid capabilities, explicit time integration schemes, and specialized
tools for analyzing compressor and turbine flows. The package provides comprehensive support
for grid generation, flow field initialization, boundary condition specification, time-accurate
simulations, and post-processing including circumferential averaging and performance metrics
extraction. Built on NumPy with Fortran-accelerated numerics, EMBER emphasizes ease of use
for rapid prototyping while maintaining computational efficiency for production simulations.
"""

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
