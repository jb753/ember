#!/usr/bin/env bash
# Finds the maximum CFL at which scripts/run_duct.py CONVERGES -- energy
# residual down >= 1 decade -- for the RK4+IRS scheme with the coarse-level
# correction off (fac_mgrid=0.0). Only sf_resid=1.0 and fac_mgrid=0.0 are passed
# to run_duct.py; everything else (n_stage=4, n_step, n_levels=0, ncell) is left
# at run_duct.py's own defaults. run_duct.py is the sole source of truth: it
# exits 1 if the run diverges, exit 2 if it runs but does not clear the 1-decade
# convergence bar, and exit 0 only when it converges. This script keys on that
# exit code (0 vs non-zero); it reimplements no solver, divergence, or
# convergence logic.
#
# Search strategy (descent only, never bisection): hold the highest CFL known
# NOT to converge and try to step DOWN by --dcfl. If that step lands on a
# converging CFL it overshot the top of the converging band, so reject it and
# HALVE --dcfl; if it still does not converge, commit the step and keep the same
# dcfl. The current CFL therefore only ever moves downward, converging on the
# threshold from above with an ever-finer step, and we stop once dcfl <= --tol.
# There is no lo/hi bracket and no midpoint averaging. Assumes a single upper
# threshold: above it the run either diverges or stalls short of 2 decades,
# below it (down to the band's lower edge) it converges. A diverging run exits
# early, so the cost is dominated by the full-length runs near the threshold.
#
# Usage:
#   scripts/duct_cfl_descend.sh [--cfl-start F] [--dcfl F] [--tol F] \
#       [--fac-mgrid F ...] [-- EXTRA_ARGS...]
#
# --fac-mgrid may be given multiple times to override the default {0.0}.
# EXTRA_ARGS (after --) are forwarded verbatim to run_duct.py, so any run_duct
# default (n-step, ncell, n-levels, ...) can be overridden there.

set -uo pipefail

CFL_START=12.0
DCFL=1.0
TOL=0.25
FAC_MGRIDS=()
EXTRA_ARGS=()

# The one scheme setting that differs from run_duct.py's defaults: residual
# smoothing on (its default is 0.0). n_stage=4 is already the default.
SF_RESID=1.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cfl-start) CFL_START=$2; shift 2 ;;
        --dcfl) DCFL=$2; shift 2 ;;
        --tol) TOL=$2; shift 2 ;;
        --fac-mgrid) FAC_MGRIDS+=("$2"); shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#FAC_MGRIDS[@]} -eq 0 ]]; then
    FAC_MGRIDS=(0.0)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Converged iff run_duct.py exits 0 (did not diverge AND cleared the 1-decade
# bar). Its own stdout/stderr is echoed to our stderr for a paper trail. Only
# the non-default knobs are passed.
converges() {
    local cfl="$1" fac_mgrid="$2"
    uv run "$SCRIPT_DIR/run_duct.py" \
        --cfl "$cfl" --fac-mgrid "$fac_mgrid" --sf-resid "$SF_RESID" \
        "${EXTRA_ARGS[@]}" >&2
}

# Descend from CFL_START toward the threshold, halving the step on each overshoot.
find_limit() {
    local fac_mgrid="$1"

    echo "=== fac_mgrid=$fac_mgrid: sf_resid=$SF_RESID ===" >&2

    # We approach the threshold from above, so CFL_START must not converge.
    if converges "$CFL_START" "$fac_mgrid"; then
        echo "fac_mgrid=$fac_mgrid: CFL_START=$CFL_START already converges; raise --cfl-start to bracket the threshold."
        return
    fi

    local cfl=$CFL_START dcfl=$DCFL lo="" trial
    while (( $(echo "$dcfl > $TOL" | bc -l) )); do
        trial=$(echo "scale=6; $cfl - $dcfl" | bc -l)
        if (( $(echo "$trial <= 0" | bc -l) )); then
            dcfl=$(echo "scale=6; $dcfl / 2" | bc -l)  # can't step below zero
            continue
        fi
        echo "--- fac_mgrid=$fac_mgrid: trying CFL=$trial (dcfl=$dcfl) ---" >&2
        if converges "$trial" "$fac_mgrid"; then
            lo=$trial                                   # overshoot: shrink step
            dcfl=$(echo "scale=6; $dcfl / 2" | bc -l)
        else
            cfl=$trial                                  # still not converging: commit
        fi
    done

    if [[ -z "$lo" ]]; then
        echo "fac_mgrid=$fac_mgrid: no converging CFL found down to $cfl -- lower --cfl-start or --tol."
    else
        echo "fac_mgrid=$fac_mgrid: max converging CFL ~= $lo (does not converge at $cfl), tol=$TOL"
    fi
}

echo "Finding max converging CFL (RK4+IRS): cfl_start=$CFL_START dcfl=$DCFL tol=$TOL" >&2

for fac_mgrid in "${FAC_MGRIDS[@]}"; do
    find_limit "$fac_mgrid"
done
