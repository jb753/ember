#!/usr/bin/env bash
# Finds the maximum CFL at which scripts/run_duct.py CONVERGES -- energy
# residual down >= 1 decade -- across a matrix of scheme settings. Each case is
# a (n_stage, sf_resid) scheme crossed with a fac_mgrid (coarse-level correction
# fraction). Only the swept knobs and cfl are passed to run_duct.py; everything
# else (n_step, n_levels, ncell) is left at run_duct.py's own defaults.
# run_duct.py is the sole source of truth: it exits 1 if the run diverges, exit
# 2 if it runs but does not clear the 1-decade convergence bar, and exit 0 only
# when it converges. This script keys on that exit code (0 vs non-zero); it
# reimplements no solver, divergence, or convergence logic.
#
# Search strategy (descent only, never bisection): hold the highest CFL known
# NOT to converge and try to step DOWN by --dcfl. If that step lands on a
# converging CFL it overshot the top of the converging band, so reject it and
# HALVE --dcfl; if it still does not converge, commit the step and keep the same
# dcfl. The current CFL therefore only ever moves downward, converging on the
# threshold from above with an ever-finer step, and we stop once dcfl <= --tol.
# There is no lo/hi bracket and no midpoint averaging. Assumes a single upper
# threshold: above it the run either diverges or stalls short of 1 decade,
# below it (down to the band's lower edge) it converges. A diverging run exits
# early, so the cost is dominated by the full-length runs near the threshold.
#
# Usage:
#   scripts/duct_cfl_descend.sh [--cfl-start F] [--dcfl F] [--tol F] \
#       [--scheme N_STAGE:SF_RESID ...] [--fac-mgrid F ...] \
#       [--jobs N] [--dry-run] [-- EXTRA_ARGS...]
#
# --scheme may be given multiple times to override the default set of
# (n_stage, sf_resid) pairs. --fac-mgrid may be given multiple times to override
# the default {0.0, 0.2, 0.4}. Every scheme is crossed with every fac_mgrid.
# --jobs N runs up to N of the (scheme x fac_mgrid) cases concurrently (default
# 1 = serial); the cases are independent so this is a pure speed-up. The descent
# WITHIN a case stays serial -- each trial CFL depends on the previous exit code
# -- so N never exceeds the number of cases usefully. Result summary lines are
# collected and printed in matrix order after all cases finish (so parallel runs
# do not interleave them); the per-trial stderr trace still streams live and, at
# N>1, from several cases at once. --dry-run replaces the run_duct.py call with a
# random exit code so the search and cross-product mechanics can be exercised
# without compiling or solving. EXTRA_ARGS (after --) are forwarded verbatim to
# run_duct.py, so any run_duct default (n-step, ncell, n-levels, ...) can be
# overridden there.

set -uo pipefail

CFL_START=12.0
DCFL=1.0
TOL=0.25
SCHEMES=()
FAC_MGRIDS=()
EXTRA_ARGS=()
DRY_RUN=0
JOBS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cfl-start) CFL_START=$2; shift 2 ;;
        --dcfl) DCFL=$2; shift 2 ;;
        --tol) TOL=$2; shift 2 ;;
        --scheme) SCHEMES+=("$2"); shift 2 ;;
        --fac-mgrid) FAC_MGRIDS+=("$2"); shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if ! [[ $JOBS =~ ^[1-9][0-9]*$ ]]; then
    echo "--jobs must be a positive integer (got: $JOBS)" >&2; exit 1
fi

# Default matrix: the three schemes of interest, each crossed with {0,0.2,0.4}.
#   n_stage=0 sf_resid=0  (no smoothing, single-stage)
#   n_stage=4 sf_resid=0  (RK4, no smoothing)
#   n_stage=4 sf_resid=1  (RK4 + IRS)
if [[ ${#SCHEMES[@]} -eq 0 ]]; then
    SCHEMES=("0:0.0" "4:0.0" "4:1.0")
fi
if [[ ${#FAC_MGRIDS[@]} -eq 0 ]]; then
    FAC_MGRIDS=(0.0 0.2 0.4)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Converged iff run_duct.py exits 0 (did not diverge AND cleared the 1-decade
# bar). Its own stdout/stderr is echoed to our stderr for a paper trail. Only
# the swept knobs are passed. Under --dry-run the solver is not invoked at all:
# we return a random exit code (0/1/2) so the descent and cross-product logic
# can be verified without a build.
converges() {
    local cfl="$1" fac_mgrid="$2" n_stage="$3" sf_resid="$4"
    if [[ $DRY_RUN -eq 1 ]]; then
        local rc=$(( RANDOM % 3 ))
        echo "[dry-run] cfl=$cfl fac_mgrid=$fac_mgrid n_stage=$n_stage" \
             "sf_resid=$sf_resid -> exit $rc" >&2
        return $rc
    fi
    uv run "$SCRIPT_DIR/run_duct.py" \
        --cfl "$cfl" --fac-mgrid "$fac_mgrid" \
        --n-stage "$n_stage" --sf-resid "$sf_resid" \
        "${EXTRA_ARGS[@]}" >&2
}

# Descend from CFL_START toward the threshold, halving the step on each overshoot.
find_limit() {
    local fac_mgrid="$1" n_stage="$2" sf_resid="$3"
    local tag="n_stage=$n_stage sf_resid=$sf_resid fac_mgrid=$fac_mgrid"

    echo "=== $tag ===" >&2

    # We approach the threshold from above, so CFL_START must not converge.
    if converges "$CFL_START" "$fac_mgrid" "$n_stage" "$sf_resid"; then
        echo "$tag: CFL_START=$CFL_START already converges; raise --cfl-start to bracket the threshold."
        return
    fi

    local cfl=$CFL_START dcfl=$DCFL lo="" trial
    while (( $(echo "$dcfl > $TOL" | bc -l) )); do
        trial=$(echo "scale=6; $cfl - $dcfl" | bc -l)
        if (( $(echo "$trial <= 0" | bc -l) )); then
            dcfl=$(echo "scale=6; $dcfl / 2" | bc -l)  # can't step below zero
            continue
        fi
        echo "--- $tag: trying CFL=$trial (dcfl=$dcfl) ---" >&2
        if converges "$trial" "$fac_mgrid" "$n_stage" "$sf_resid"; then
            lo=$trial                                   # overshoot: shrink step
            dcfl=$(echo "scale=6; $dcfl / 2" | bc -l)
        else
            cfl=$trial                                  # still not converging: commit
        fi
    done

    if [[ -z "$lo" ]]; then
        echo "$tag: no converging CFL found down to $cfl -- lower --cfl-start or --tol."
    else
        echo "$tag: max converging CFL ~= $lo (does not converge at $cfl), tol=$TOL"
    fi
}

echo "Finding max converging CFL: cfl_start=$CFL_START dcfl=$DCFL tol=$TOL" \
     "schemes=${#SCHEMES[@]} fac_mgrids=${#FAC_MGRIDS[@]} jobs=$JOBS" \
     "dry_run=$DRY_RUN" >&2

# Each case's result summary (find_limit's stdout) is captured to its own file so
# that concurrent cases cannot interleave their summaries; the files are printed
# in matrix order once every case has finished. The live stderr trace is not
# redirected, so it streams through as cases run.
RESULT_DIR="$(mktemp -d)"
trap 'rm -rf "$RESULT_DIR"' EXIT

SECONDS=0   # bash built-in wall-clock timer for the whole matrix
idx=0
for scheme in "${SCHEMES[@]}"; do
    n_stage=${scheme%%:*}
    sf_resid=${scheme##*:}
    for fac_mgrid in "${FAC_MGRIDS[@]}"; do
        # Throttle to JOBS concurrent cases; wait -n frees a slot as soon as any
        # running case finishes.
        while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
        find_limit "$fac_mgrid" "$n_stage" "$sf_resid" >"$RESULT_DIR/$idx.out" &
        idx=$((idx + 1))
    done
done
wait

for (( j = 0; j < idx; j++ )); do
    cat "$RESULT_DIR/$j.out"
done

echo "All $idx cases finished in ${SECONDS}s (jobs=$JOBS)." >&2
