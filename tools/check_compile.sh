#!/bin/bash
# Check if compilation prerequisites are available before attempting to build

set -e

# Check if gfortran is available
GFORTRAN_PATH=$(command -v gfortran 2>/dev/null)
if [ -z "$GFORTRAN_PATH" ]; then
    echo "Error: gfortran is not installed or not on PATH"
    echo ""
    echo "To install gfortran on Debian/Ubuntu systems, run:"
    echo "  sudo apt update"
    echo "  sudo apt install gfortran"
    echo ""
    echo "For other systems:"
    echo "  - Fedora/RHEL: sudo dnf install gcc-gfortran"
    echo "  - Arch: sudo pacman -S gcc-fortran"
    echo "  - macOS: brew install gcc"
    exit 1
fi

# command -v succeeds even for broken symlinks; check the target actually exists
if [ -L "$GFORTRAN_PATH" ] && [ ! -e "$GFORTRAN_PATH" ]; then
    TARGET=$(readlink "$GFORTRAN_PATH")
    echo "Error: gfortran symlink is broken ($GFORTRAN_PATH -> $TARGET does not exist)"
    echo ""
    echo "The package metadata is registered but the compiler binary was never installed."
    echo "Fix with:"
    echo "  apt-get update && apt-get install -y --fix-broken"
    exit 1
fi

# Check gfortran version
GFORTRAN_VERSION=$(gfortran --version | head -n 1)
echo "gfortran: $GFORTRAN_VERSION"

# Check Fortran source files syntax (run in temp dir to avoid .mod files in project root)
# Module-defining files must be compiled before their users: gfortran
# resolves `use` against a .mod produced earlier in the same invocation. This
# is a non-issue for src/ember/_fortran/ specifically: every production file
# there that `use`s a helper module (residual.f90 -> residual_helpers,
# viscous.f90 -> viscous_helpers) defines that module IN THE SAME FILE, so a
# plain alphabetical glob is always self-sufficient -- no cross-file provider
# ordering is needed here. Experimental benchmark kernels that DO split a
# kernel from its shared helpers live under bench/subroutines/ instead, are
# not globbed by this check at all, and don't need one either: they're only
# ever compiled by `make compile EMBER_BENCH_KERNELS=...`, where f2py's meson
# backend does real dependency-graph resolution and doesn't care about file
# order (see setup.py's select_bench_kernels()).
#
# Stale .mod files are a separate, still-real hazard: running
# `gfortran -fsyntax-only` from the repo root drops .mod files there, and a
# later check can resolve `use` against a stale .mod and pass (or fail)
# wrongly even after the source was reverted or changed. Delete stray *.mod
# from the repo root before builds, or run syntax checks in a temp directory
# (which is what this script does).
ALL_F90_FILES="$(ls src/ember/_fortran/*.f90 2>/dev/null)"
SYNTAX_TMP=$(mktemp -d)
# -ffree-line-length-132 pinned explicitly: gfortran >=14 stops enforcing the
# free-form 132-column limit under plain -Wall (a version-specific
# regression vs. gfortran 13, which CI still uses), so an over-length line
# would pass this check silently and only fail in CI.
if ! gfortran -fsyntax-only -J "$SYNTAX_TMP" -ffree-line-length-132 -Wall -Werror -Warray-temporaries -Wfatal-errors $ALL_F90_FILES 2>&1; then
    echo "Error: Fortran syntax/warning errors detected"
    rm -rf "$SYNTAX_TMP"
    exit 1
fi
rm -rf "$SYNTAX_TMP"

echo "Fortran syntax passed"
