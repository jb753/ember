"""Custom setup.py for building Fortran extensions with f2py."""

import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob

# Set to True to compile with debug flags (gfortran only): no optimisation,
# strict bounds checking, backtraces. Set to False for optimised release build.
GFORTRAN_DEBUG = False

# Compiler flags - shared configuration
# -march defaults to a portable baseline (any Haswell-or-newer x86_64 CPU).
# Named "haswell" rather than the equivalent "x86-64-v3" level-name because
# the latter is only recognized by GCC >=11, and manylinux build containers
# (yum-installed gcc-gfortran) may still ship an older GCC.
# Override with EMBER_MARCH (e.g. "-march=native -mtune=native") for perf
# runs tuned to a specific machine, without having to repeat every other flag.
GFORTRAN_MARCH = os.environ.get("EMBER_MARCH", "-march=haswell")
# -fipa-pta deliberately omitted: verified a no-op on the current whole-program
# build (identical .text section with/without it, GCC 14.2), but in an isolated
# single-file compile it suppressed AVX2 vectorization of the residual face-flux
# loops (~20% slower), with no offsetting benefit found anywhere. Re-check if the
# toolchain or the _fortran/ file set changes substantially.
GFORTRAN_FLAGS = f"-Ofast {GFORTRAN_MARCH} -funroll-all-loops -finline-functions -finline-limit=10000 --param early-inlining-insns=200 -flto -fwhole-program -fno-trapping-math -freciprocal-math -floop-nest-optimize -fvect-cost-model=unlimited -Wall -Werror -Warray-temporaries -Wfatal-errors"
GFORTRAN_DEBUG_FLAGS = "-O0 -g -fcheck=all -fbounds-check -fbacktrace -Wall -Werror -Warray-temporaries -Wfatal-errors"
# Intel flags: close equivalents of gfortran flags
INTEL_FLAGS = "-O3 -xHost -ipo -no-prec-div -fp-model fast=2 -funroll-loops -inline-forceinline -inline-factor=10000 -fast-transcendentals"


class F2PyExtension(Extension):
    """Custom extension class for f2py compilation."""

    def __init__(self, name, sourcedirs):
        Extension.__init__(self, name, sources=[])
        self.sourcedirs = sourcedirs


class F2PyBuildExt(build_ext):
    """Custom build_ext command for f2py compilation."""

    def strip_unicode_from_fortran(self, source_file):
        """Strip unicode characters from Fortran source file in place.

        Replaces all non-ASCII characters with spaces to avoid compiler issues.

        Parameters
        ----------
        source_file : str
            Path to the Fortran source file to clean
        """
        # Read the source file
        with open(source_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace any non-ASCII characters with spaces
        cleaned_content = "".join(char if ord(char) < 128 else " " for char in content)

        # Only write if changes were made
        if cleaned_content != content:
            with open(source_file, "w", encoding="ascii") as f:
                f.write(cleaned_content)
            print(f"Stripped unicode characters from {source_file}")

    def build_extension(self, ext):
        if not isinstance(ext, F2PyExtension):
            return super().build_extension(ext)

        # Find only .f90 Fortran source files using absolute paths
        fortran_sources = []
        for sourcedir in ext.sourcedirs:
            abs_sourcedir = os.path.abspath(sourcedir)
            sources = glob.glob(os.path.join(abs_sourcedir, "*.f90"))
            fortran_sources.extend(sources)

        # Sort to ensure consistent build order
        fortran_sources.sort()

        if not fortran_sources:
            raise RuntimeError(f"No Fortran source files found in {ext.sourcedirs}")

        # Strip unicode characters from all Fortran sources in place
        for source in fortran_sources:
            self.strip_unicode_from_fortran(source)

        # Get the output directory and ensure module name is just 'fortran'
        output_dir = Path(self.get_ext_fullpath(ext.name)).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine which compiler flags to use. EMBER_COMPILER selects the
        # toolchain; FC/CC/CXX are set here (not required from the caller)
        # so a serial build only needs this one variable.
        ember_compiler = os.environ.get("EMBER_COMPILER", "gfortran")
        if ember_compiler == "ifort":
            flags = INTEL_FLAGS
            os.environ.setdefault("FC", "ifort")
            os.environ.setdefault("CC", "icc")
            os.environ.setdefault("CXX", "icpc")
        elif ember_compiler == "gfortran":
            flags = GFORTRAN_DEBUG_FLAGS if GFORTRAN_DEBUG else GFORTRAN_FLAGS
        else:
            raise RuntimeError(
                f"Unknown EMBER_COMPILER '{ember_compiler}', expected "
                "'gfortran' or 'ifort'"
            )

        # Build f2py command. Force the meson backend explicitly: on Python
        # versions that still ship stdlib distutils (<=3.11), f2py -c
        # otherwise defaults to the legacy distutils backend, which is
        # broken against modern setuptools (numpy.distutils' new_compiler()
        # call doesn't match setuptools' vendored Compiler.__init__ anymore).
        f2py_cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            "--backend",
            "meson",
            # "--quiet",
            "-m",
            "fortran",  # Always use 'fortran' as module name
            f"--f90flags={flags}",
        ] + fortran_sources

        # Run f2py from the output directory, but with a clean environment
        # to avoid Python finding ember's collections.py instead of stdlib collections.
        # The issue is that ember has a collections.py that shadows the stdlib module.
        # We work around this by running from the project root, not the build dir.

        # Run f2py in a temp dir so .mod files don't pollute the project root
        import tempfile

        build_tmp = tempfile.mkdtemp()

        print(f"Running f2py command: {' '.join(f2py_cmd)}")
        print(f"Working directory: {build_tmp}")
        print(f"Output directory: {output_dir}")
        print(f"Source files found: {fortran_sources}")

        result = subprocess.run(f2py_cmd, capture_output=True, text=True, cwd=build_tmp)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"f2py compilation failed with return code {result.returncode}"
            )

        # Move the compiled extension to the correct location
        import shutil

        so_pattern = "fortran*.so"
        so_files = glob.glob(os.path.join(build_tmp, so_pattern))
        if not so_files:
            raise RuntimeError(
                f"f2py compilation succeeded but no {so_pattern} found in {os.getcwd()}. "
                f"Check f2py output above."
            )
        for so_file in so_files:
            dest = output_dir / os.path.basename(so_file)
            shutil.move(so_file, dest)
            print(f"Moved {so_file} to {dest}")


def build_extensions():
    """Configure the extensions to build."""
    return [F2PyExtension("ember.fortran", sourcedirs=["src/ember/_fortran"])]


if __name__ == "__main__":
    setup(
        ext_modules=build_extensions(),
        cmdclass={"build_ext": F2PyBuildExt},
        zip_safe=False,
    )
