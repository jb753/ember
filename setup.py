"""Custom setup.py for building Fortran extensions with f2py."""

import os
import shutil
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
# -ffree-line-length-132 pinned explicitly: gfortran 14.2 silently stops
# enforcing the free-form 132-column limit under plain -Wall (a version-
# specific regression vs. gfortran 13), so an over-length line built clean
# here but failed CI's gfortran 13 with -Werror=line-truncation.
# --param=inline-unit-growth / large-function-growth: GCC's inline budgets are
# UNIT-level, and at this program's size the defaults bind hard -- set_residual's
# face helpers were simply not being inlined. Lifting them is worth -53% serial,
# -36% at 16 ranks and -17% on a full timestep, with the goldens unregenerated
# (see bench/README.md, "Adopted: pin GCC's unit-level inline budgets"). This
# exact PAIR is the minimal set:
# it produces codegen bit-identical to lifting all four related budgets, and
# inline-unit-growth is necessary even though it is worth only 2.3% alone.
# Portable (no -march dependence), hence a default rather than opt-in.
# NOTE the --param=X=Y spelling: f2py re-splits "--param X=Y" on the space.
GFORTRAN_FLAGS = f"-Ofast {GFORTRAN_MARCH} -funroll-all-loops -finline-functions -finline-limit=10000 --param early-inlining-insns=200 --param=inline-unit-growth=1000000 --param=large-function-growth=1000000 -flto -fwhole-program -fno-trapping-math -freciprocal-math -floop-nest-optimize -fvect-cost-model=unlimited -ffree-line-length-132 -Wall -Werror -Warray-temporaries -Wfatal-errors"
# Appended verbatim to the gfortran flags. Used to test whether pinning
# GCC's UNIT-level inline budgets makes production codegen invariant to
# what else is in the build -- see bench/codegen_gauge.py.
GFORTRAN_FLAGS += " " + os.environ.get("EMBER_FFLAGS_EXTRA", "")

# Profile-guided optimisation, opt-in via EMBER_PGO=generate|use.
#
# Deliberately NOT a default and never shippable: -fprofile-use needs a
# training run, which a manylinux wheel build cannot do. This is for HPC and
# local performance builds.
#
# The profile directory is absolute and OUTSIDE build/, which `make compile`
# wipes. Combined with the deterministic f2py scratch dir (see
# build_extension), that is what makes the .gcda from the generate build
# findable by the use build.
#
# Both stages need the flag: under -flto the real codegen happens in the LTO
# backend at link, so it goes into LDFLAGS as well -- same reason as
# EMBER_OPT_REPORT below.
_PGO = os.environ.get("EMBER_PGO", "").strip()
_PGO_DIR = os.path.abspath(os.environ.get("EMBER_PGO_DIR", ".pgo"))
if _PGO == "generate":
    _PGO_FLAGS = f"-fprofile-generate -fprofile-dir={_PGO_DIR} -fprofile-update=single"
elif _PGO == "use":
    # -Wno-missing-profile: -Werror is on, and any translation unit the
    # training run did not exercise would otherwise fail the build.
    _PGO_FLAGS = (
        f"-fprofile-use -fprofile-dir={_PGO_DIR} -fprofile-correction"
        " -Wno-missing-profile -Wno-coverage-mismatch"
    )
elif _PGO:
    raise RuntimeError(f"EMBER_PGO must be 'generate' or 'use', got {_PGO!r}")
else:
    _PGO_FLAGS = ""
if _PGO_FLAGS:
    os.makedirs(_PGO_DIR, exist_ok=True)
    GFORTRAN_FLAGS += " " + _PGO_FLAGS

# Set EMBER_OPT_REPORT=<path> to write the compiler's vectorization report
# there during the build. The flag is injected at LINK time (via LDFLAGS,
# which meson passes to the linker driver): under gfortran's -flto or
# ifort's -ipo, the real whole-program codegen happens in the LTO/IPO
# backend at link, so the link-stage report describes the code that
# actually runs, whereas a compile-stage report reflects discarded per-TU
# codegen and can flag spurious misses (see bench/README.md, Gate 1). The
# file is truncated at build start (gfortran's
# -fopt-info appends per LTRANS partition; ifort overwrites, but truncate
# unconditionally for consistent behaviour across compilers).
_OPT_REPORT = os.environ.get("EMBER_OPT_REPORT")
if _OPT_REPORT:
    _OPT_REPORT = os.path.abspath(_OPT_REPORT)
    open(_OPT_REPORT, "w").close()
GFORTRAN_DEBUG_FLAGS = "-O0 -g -fcheck=all -fbounds-check -fbacktrace -Wall -Werror -Warray-temporaries -Wfatal-errors"
# Intel flags: close equivalents of gfortran flags
INTEL_FLAGS = "-O3 -xHost -ipo -no-prec-div -fp-model fast=2 -funroll-loops -inline-forceinline -inline-factor=10000 -fast-transcendentals"


# ---------------------------------------------------------------------------
# Benchmark-only Fortran sources.
#
# These implement A/B variants of production kernels (see bench/README.md)
# and nothing in src/ember/ calls them. They are EXCLUDED from the build by
# default, for two reasons:
#
#   1. They are dead code in a shipped wheel.
#   2. They change production's codegen. The build is -flto -fwhole-program,
#      and GCC's inline budgets (inline-unit-growth, large-unit-insns,
#      large-function-growth) are UNIT-level: growing the program grows the
#      absolute inlining budget and silently changes decisions for functions
#      that did not change. Measured: set_residual's inlined body went from
#      9,177 to 10,759 instructions, and its scalar divide count from 50 to
#      70, purely because unrelated arms were added to this directory.
#
# EMBER_ARMS selects which to build back in:
#     (unset)              production only -- the shipped build
#     EMBER_ARMS=nodal     production plus set_residual_nodal
#     EMBER_ARMS=nodal,rinv
#     EMBER_ARMS=all       everything (the historical behaviour)
#
# Module dependencies between arms are resolved automatically, so asking for
# `multall` also pulls in residual_staged.f90 for scale_du_all.
#
# This is a DENYLIST on purpose: a benchmark file omitted from it is merely
# built unnecessarily, whereas an allowlist that missed a production file
# would ship a broken package. Regenerate with the audit in
# docs/dev/plan_nodal_primitives.md.
BENCHMARK_ONLY = {
    "perfect.f90",
    "residual_cand.f90",
    "residual_consa.f90",
    "residual_damp_fused.f90",
    "residual_irs_bj.f90",
    "residual_irs_dirs.f90",
    "residual_irs_km.f90",
    "residual_naive.f90",
    "residual_nodal.f90",
    "residual_prod_soa.f90",
    "residual_rinv.f90",
    "residual_setdamp.f90",
    "residual_setdamp_fix.f90",
    "residual_staged.f90",
    "residual_multall.f90",
    "residual_multall_aos.f90",
    "residual_tiled.f90",
}


def select_fortran_sources(sources):
    """Drop benchmark-only sources unless EMBER_ARMS asks for them."""
    import re

    wanted = os.environ.get("EMBER_ARMS", "").strip()
    keep = [s for s in sources if os.path.basename(s) not in BENCHMARK_ONLY]
    if not wanted:
        return keep
    if wanted == "all":
        return sources

    by_name = {os.path.basename(s): s for s in sources}
    extra = set()
    for tok in (t.strip() for t in wanted.split(",") if t.strip()):
        for cand in (tok, f"{tok}.f90", f"residual_{tok}.f90"):
            if cand in by_name and cand in BENCHMARK_ONLY:
                extra.add(cand)
                break
        else:
            raise RuntimeError(
                f"EMBER_ARMS: unknown arm {tok!r}. Known: "
                + ", ".join(sorted(BENCHMARK_ONLY))
            )

    # Close over `use` dependencies among the benchmark files.
    def modules(path):
        txt = open(path).read()
        return (set(m.lower() for m in re.findall(r"(?im)^\s*module\s+(\w+)\s*$", txt)),
                set(m.lower() for m in re.findall(r"(?im)^\s*use\s+(\w+)", txt)))

    owner = {}
    for name in BENCHMARK_ONLY:
        if name in by_name:
            for m in modules(by_name[name])[0]:
                owner[m] = name
    pending = list(extra)
    while pending:
        _, needs = modules(by_name[pending.pop()])
        for m in needs:
            dep = owner.get(m)
            if dep and dep not in extra:
                extra.add(dep)
                pending.append(dep)

    return sorted(keep + [by_name[n] for n in extra])


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

        # Benchmark-only arms are excluded unless EMBER_ARMS asks for them:
        # they are dead code in production AND they perturb production's
        # codegen through whole-program inline budgets. See BENCHMARK_ONLY.
        fortran_sources = select_fortran_sources(fortran_sources)

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
            if _OPT_REPORT:
                # Level 2 is deliberately pinned, not the max (5): at this
                # program's size, ifort 2022.1.0's IPO backend segfaults
                # (internal error, "multi-file optimization compilation")
                # generating the link-stage report at level 3 and above.
                # Level 2 still includes the "loop was not vectorized:
                # <reason>" remarks needed for diagnosis, just without the
                # extra dependence-chain detail level 5 would add.
                os.environ["LDFLAGS"] = (
                    os.environ.get("LDFLAGS", "")
                    + f" -qopt-report=2 -qopt-report-phase=vec,ipo"
                    f" -qopt-report-file={_OPT_REPORT}"
                ).strip()
        elif ember_compiler == "gfortran":
            flags = GFORTRAN_DEBUG_FLAGS if GFORTRAN_DEBUG else GFORTRAN_FLAGS
            if _PGO_FLAGS:
                # LTO does the real codegen at link, so the profile flags must
                # reach the linker driver as well as the compiler.
                os.environ["LDFLAGS"] = (
                    os.environ.get("LDFLAGS", "") + " " + _PGO_FLAGS
                ).strip()
            if _OPT_REPORT:
                os.environ["LDFLAGS"] = (
                    os.environ.get("LDFLAGS", "")
                    + f" -fopt-info-vec-all={_OPT_REPORT}"
                ).strip()
        else:
            raise RuntimeError(
                f"Unknown EMBER_COMPILER '{ember_compiler}', expected "
                "'gfortran' or 'ifort'"
            )

        build_tmp = os.path.abspath(os.path.join("build", "f2py-tmp"))
        if os.path.isdir(build_tmp):
            shutil.rmtree(build_tmp)
        os.makedirs(build_tmp, exist_ok=True)

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
            # DETERMINISTIC build directory, in both senses that matter.
            #
            # cwd keeps .mod files out of the project root; --build-dir is the
            # one that matters for PGO. Without it f2py's meson backend calls
            # tempfile.mkdtemp() itself, so object files land in a fresh random
            # path every build -- and GCC mangles the OBJECT path into the
            # .gcda filename, so -fprofile-use can never find what
            # -fprofile-generate wrote. Passing it also makes f2py keep the
            # directory instead of deleting it (remove_build_dir = 0).
            #
            # Found the hard way: the first PGO attempt "worked", and only
            # -Werror=missing-profile (with the -Wno- suppression removed)
            # revealed that not one profile had been read.
            "--build-dir",
            os.path.join(build_tmp, "f2py"),
        ] + fortran_sources

        # Run f2py from the output directory, but with a clean environment
        # to avoid Python finding ember's collections.py instead of stdlib collections.
        # The issue is that ember has a collections.py that shadows the stdlib module.
        # We work around this by running from the project root, not the build dir.

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
