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
# resolves `use` against a .mod produced earlier in the same invocation, and
# a plain alphabetical glob puts consumers (residual_cand, residual_consa,
# ...) ahead of the providers (residual.f90, viscous.f90). This used to be
# masked by stale .mod files left in the source tree by ad-hoc syntax
# checks -- the hazard docs/dev/viscous_kernels.md section 6.4 warns about.
# f2py/meson does its own dependency ordering, so this only affects this
# pre-flight check.
#
# residual_staged.f90 and residual_multall.f90 are providers too: the benchmark
# arms share helpers rather than copying them, so that the parts NOT under test
# have identical codegen (residual_multall uses staged's scale_du_all;
# residual_multall_aos uses multall's stage_primitives). Do not rely on the glob
# to order those -- `ls` collates locale-aware and ignores the underscore, so
# residual_multall_aos.f90 sorts BEFORE residual_multall.f90 here.
F90_PROVIDERS="src/ember/_fortran/residual.f90 src/ember/_fortran/viscous.f90 \
src/ember/_fortran/residual_staged.f90 src/ember/_fortran/residual_multall.f90"
ALL_F90_FILES="$F90_PROVIDERS $(ls src/ember/_fortran/*.f90 2>/dev/null | grep -vE '/(residual|viscous|residual_staged|residual_multall)\.f90$')"
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
