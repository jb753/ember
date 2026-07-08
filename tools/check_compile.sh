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
ALL_F90_FILES=$(ls src/ember/_fortran/*.f90 2>/dev/null)
SYNTAX_TMP=$(mktemp -d)
if ! gfortran -fsyntax-only -J "$SYNTAX_TMP" -Wall -Werror -Warray-temporaries -Wfatal-errors $ALL_F90_FILES 2>&1; then
    echo "Error: Fortran syntax/warning errors detected"
    rm -rf "$SYNTAX_TMP"
    exit 1
fi
rm -rf "$SYNTAX_TMP"

echo "Fortran syntax passed"
