#!/bin/bash
# Compile script that can use either Intel or GNU compilers
#
# Usage:
#   ./tools/compile.sh [compiler] [options]
#
# Arguments:
#   compiler: intel or gfortran (default: gfortran)
#
# Options:
#   --bench    Run benchmark after compilation
#
# Examples:
#   ./tools/compile.sh intel           # Compile with Intel
#   ./tools/compile.sh gfortran --bench # Compile with gfortran and run benchmark
#   ./tools/compile.sh --bench         # Compile with gfortran (default) and run benchmark

# Parse arguments
COMPILER="gfortran"
RUN_BENCH=false

for arg in "$@"; do
    case $arg in
        intel|gfortran)
            COMPILER="$arg"
            ;;
        --bench)
            RUN_BENCH=true
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [intel|gfortran] [--bench]"
            exit 1
            ;;
    esac
done

# Nuke the venv
rm -rf ./.venv/

# Load modules
module purge
#
if hostname | grep -q '\-r\-' ; then
    echo "sapphire modules"
    module load rhel8/default-sar

else
    echo "icelake modules"
    module load rhel8/default-icl
fi

# Tune for this node's exact CPU (native, not the portable default in
# setup.py) since this is a perf run, not a redistributable build.
export EMBER_MARCH="-march=native -mtune=native"
#

# Create a new venv linked against modules
UV_CACHE_DIR=.uv-cache
uv venv


# Remove any existing build artifacts
echo "Cleaning build artifacts..."
rm -rf build/ dist/ *.egg-info src/*.egg-info
rm -f src/ember/fortran*.so

# Activate venv
source .venv/bin/activate

if [ "$COMPILER" = "intel" ]; then
    echo "Configuring Intel compilers..."
    # FC/CC/CXX are set by setup.py itself from EMBER_COMPILER; MPI_FC etc
    # are for other tools built in this same environment, not ember's build.
    export MPI_FC=mpiifort
    export MPI_CC=mpicc
    export MPI_CXX=mpicxx
    export COMPILER_SHORT=intel
    export EMBER_COMPILER=ifort
elif [ "$COMPILER" = "gfortran" ]; then
    echo "Configuring GNU compilers..."
    # Clear Intel compiler environment variables if set
    unset FC
    unset CC
    unset CXX
    unset MPI_FC
    unset MPI_CC
    unset MPI_CXX
    unset COMPILER_SHORT
    export EMBER_COMPILER=gfortran
else
    echo "Error: Unknown compiler '$COMPILER'"
    echo "Usage: $0 [intel|gfortran]"
    exit 1
fi

uv pip install numpy meson matplotlib ninja
uv pip install -e .

echo ""
echo "Done! Ember compiled with $COMPILER"

if [ "$RUN_BENCH" = true ]; then
    echo ""
    echo "Running benchmark..."
    python scripts/run_duct_inviscid.py
fi
