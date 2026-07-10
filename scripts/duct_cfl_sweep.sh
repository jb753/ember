#!/usr/bin/env bash
# Bisects the maximum stable CFL for scripts/run_duct.py for each of the three
# schemes compared in examples/run_duct.py (scree, RK4, RK4+IRS), holding that
# scheme's other settings fixed. run_duct.py is the sole source of truth for
# stability -- it exits 1 and prints "Diverged" if the run blows up (checked
# directly against the final conserved state, not inferred from wall-clock
# time), exit 0 otherwise. This script just bisects on that exit code; it does
# not reimplement any solver or divergence-detection logic.
#
# The three cases differ only in (n_stage, sf_resid, n_step_source); every case
# runs at the shared n_step / fac_mgrid / n_levels / ncell below. fac_mgrid=0.0
# switches the coarse-level correction off, so each bisection isolates the plain
# scheme's stability limit. Bisection assumes stability is monotone in CFL (a
# bigger step is never more stable).
#
# Usage:
#   scripts/duct_cfl_sweep.sh [--n-step N] [--fac-mgrid F] [--n-levels N] \
#       [--cfl-lo F] [--cfl-hi F] [--tol F] [--ncell N] [--case LABEL] \
#       [-- EXTRA_ARGS...]
#
# --case LABEL restricts the sweep to one scheme (scree | RK4 | RK4+IRS).
#
# EXTRA_ARGS (after --) are forwarded verbatim to run_duct.py.

set -uo pipefail

N_STEP=1000
FAC_MGRID=0.0
N_LEVELS=2
CFL_LO=0.1
CFL_HI=20.0
TOL=0.05
NCELL=1000000
CASE_FILTER=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-step) N_STEP=$2; shift 2 ;;
        --fac-mgrid) FAC_MGRID=$2; shift 2 ;;
        --n-levels) N_LEVELS=$2; shift 2 ;;
        --cfl-lo) CFL_LO=$2; shift 2 ;;
        --cfl-hi) CFL_HI=$2; shift 2 ;;
        --tol) TOL=$2; shift 2 ;;
        --ncell) NCELL=$2; shift 2 ;;
        --case) CASE_FILTER=$2; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The three schemes from examples/run_duct.py, as "label n_stage sf_resid
# n_step_source". CFL is the swept variable, so it is not fixed here.
CASES=(
    "scree   0 0.0 5"
    "RK4     4 0.0 1"
    "RK4+IRS 4 1.0 1"
)

# Stable iff run_duct.py exits 0. Its own stdout/stderr is echoed to our stderr
# for a paper trail.
is_stable() {
    local cfl="$1" n_stage="$2" sf_resid="$3" n_step_source="$4"
    uv run "$SCRIPT_DIR/run_duct.py" \
        --n-step "$N_STEP" --fac-mgrid "$FAC_MGRID" --n-levels "$N_LEVELS" \
        --n-stage "$n_stage" --sf-resid "$sf_resid" \
        --n-step-source "$n_step_source" \
        --cfl "$cfl" --ncell "$NCELL" "${EXTRA_ARGS[@]}" >&2
}

# Bisect one case; prints the result line to stdout, diagnostics to stderr.
bisect_case() {
    local label="$1" n_stage="$2" sf_resid="$3" n_step_source="$4"

    echo "=== $label: n_stage=$n_stage sf_resid=$sf_resid n_step_source=$n_step_source ===" >&2

    echo "--- checking CFL_LO=$CFL_LO is stable ---" >&2
    if ! is_stable "$CFL_LO" "$n_stage" "$sf_resid" "$n_step_source"; then
        echo "$label: CFL_LO=$CFL_LO is already unstable -- lower it with --cfl-lo and retry."
        return
    fi

    echo "--- checking CFL_HI=$CFL_HI is unstable ---" >&2
    if is_stable "$CFL_HI" "$n_stage" "$sf_resid" "$n_step_source"; then
        echo "$label: CFL_HI=$CFL_HI is still stable -- raise it with --cfl-hi and retry."
        return
    fi

    local lo=$CFL_LO hi=$CFL_HI mid
    while (( $(echo "$hi - $lo > $TOL" | bc -l) )); do
        mid=$(echo "scale=6; ($lo + $hi) / 2" | bc -l)
        echo "--- $label: trying CFL=$mid ---" >&2
        if is_stable "$mid" "$n_stage" "$sf_resid" "$n_step_source"; then
            lo=$mid
        else
            hi=$mid
        fi
    done

    echo "$label: max stable CFL ~= $lo (unstable at $hi), tol=$TOL"
}

echo "Bisecting max stable CFL per scheme: n_step=$N_STEP fac_mgrid=$FAC_MGRID n_levels=$N_LEVELS ncell=$NCELL" >&2

for case in "${CASES[@]}"; do
    label=${case%% *}
    if [[ -n "$CASE_FILTER" && "$label" != "$CASE_FILTER" ]]; then
        continue
    fi
    # shellcheck disable=SC2086
    bisect_case $case
done
