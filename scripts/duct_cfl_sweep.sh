#!/usr/bin/env bash
# Bisects the maximum stable CFL for scripts/run_duct.py at a fixed
# n_step/n_levels/sf_resid. run_duct.py is the sole source of truth for
# stability -- it exits 1 and prints "Diverged" if the run blows up (checked
# directly against the final conserved state, not inferred from wall-clock
# time), exit 0 otherwise. This script just bisects on that exit code; it
# does not reimplement any solver or divergence-detection logic.
#
# Usage:
#   scripts/duct_cfl_sweep.sh [--n-step N] [--n-levels N] [--sf-resid F] \
#       [--cfl-lo F] [--cfl-hi F] [--tol F] [--ncell N] [-- EXTRA_ARGS...]
#
# EXTRA_ARGS (after --) are forwarded verbatim to run_duct.py (e.g. --inviscid).

set -uo pipefail

N_STEP=500
N_LEVELS=2
SF_RESID=1
CFL_LO=0.5
CFL_HI=20.0
TOL=0.05
NCELL=1000000
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-step) N_STEP=$2; shift 2 ;;
        --n-levels) N_LEVELS=$2; shift 2 ;;
        --sf-resid) SF_RESID=$2; shift 2 ;;
        --cfl-lo) CFL_LO=$2; shift 2 ;;
        --cfl-hi) CFL_HI=$2; shift 2 ;;
        --tol) TOL=$2; shift 2 ;;
        --ncell) NCELL=$2; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stable iff run_duct.py exits 0. Its own stdout/stderr is echoed to our
# stderr for a paper trail.
is_stable() {
    local cfl="$1"
    uv run "$SCRIPT_DIR/run_duct.py" \
        --n-step "$N_STEP" --n-levels "$N_LEVELS" --sf-resid "$SF_RESID" \
        --cfl "$cfl" --ncell "$NCELL" "${EXTRA_ARGS[@]}" >&2
}

echo "Bisecting max stable CFL: n_step=$N_STEP n_levels=$N_LEVELS sf_resid=$SF_RESID ncell=$NCELL" >&2

echo "--- checking CFL_LO=$CFL_LO is stable ---" >&2
if ! is_stable "$CFL_LO"; then
    echo "CFL_LO=$CFL_LO is already unstable -- lower it with --cfl-lo and retry." >&2
    exit 1
fi

echo "--- checking CFL_HI=$CFL_HI is unstable ---" >&2
if is_stable "$CFL_HI"; then
    echo "CFL_HI=$CFL_HI is still stable -- raise it with --cfl-hi and retry." >&2
    exit 1
fi

lo=$CFL_LO
hi=$CFL_HI
while (( $(echo "$hi - $lo > $TOL" | bc -l) )); do
    mid=$(echo "scale=6; ($lo + $hi) / 2" | bc -l)
    echo "--- trying CFL=$mid ---" >&2
    if is_stable "$mid"; then
        lo=$mid
    else
        hi=$mid
    fi
done

echo "Max stable CFL ~= $lo (unstable at $hi), tol=$TOL"
