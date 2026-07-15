#!/usr/bin/env bash
# Finds the maximum CFL at which scripts/run_duct.py CONVERGES -- energy
# residual down >= 1 decade -- across a matrix of scheme settings. Each case is
# a (n_stage, sf_resid) scheme crossed with a fac_mgrid (coarse-level correction
# fraction). The swept knobs, cfl, and a fixed n_levels=2 are passed to
# run_duct.py, along with a per-scheme run length / log cadence: scree
# (n_stage=0) runs longer (n_step=4000) at n_step_log=50, while RK4 (n_stage=4)
# runs n_step=2000 but logs finer (n_step_log=25). Both better run_duct's
# default log cadence of 100 for settling-step resolution.
# Everything else (ncell, ...) is left at run_duct.py's own defaults.
# run_duct.py is the sole source of truth: it exits 1 if the run diverges, exit
# 2 if it runs but does not clear the 1-decade convergence bar, and exit 0 only
# when it converges. This script keys on that exit code (0 vs non-zero); it
# reimplements no solver, divergence, or convergence logic.
#
# After the per-case max-CFL summaries, a speedup table is printed. For each
# converging case, the wall time to settle (run_duct's "zeta settled to <1%"
# line, at that case's max converging CFL) is the cost to reach a settled
# solution; the table normalises it to the slowest converging case, so every
# speedup is >= 1. This uses wall time only as a REPORTED metric -- convergence
# and divergence are still decided solely by run_duct's exit code, never by a
# wall-clock threshold here.
#
# Search strategy (descent only, never bisection): hold the highest CFL known
# NOT to converge and try to step DOWN by --dcfl. If that step lands on a
# converging CFL it overshot the top of the converging band, so reject it,
# record it as the best-known converging CFL (lo), and HALVE --dcfl; if it
# still does not converge, commit the step and keep the same dcfl. The current
# CFL therefore only ever moves downward, converging on the threshold from
# above with an ever-finer step. We stop the FIRST time a converging trial is
# found whose step size (dcfl at that trial) is already <= --tol -- not merely
# once dcfl has been halved below --tol -- because the gap between the reported
# lo and the untested cfl above it equals exactly that trial's step size. This
# guarantees the true threshold lies in [lo, lo + tol], i.e. lo is correct to
# within +-tol. (Checking dcfl <= tol only after halving, as an earlier version
# did, under-guarantees accuracy to within +-2*tol.)
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
# 0 = unlimited, all cases at once); the cases are independent so this is a pure
# speed-up. The descent
# WITHIN a case stays serial -- each trial CFL depends on the previous exit code
# -- so N never exceeds the number of cases usefully. Result summary lines are
# collected and printed in matrix order after all cases finish (so parallel runs
# do not interleave them); the per-trial stderr trace still streams live and, at
# N>1, from several cases at once. --dry-run replaces the run_duct.py call with a
# random exit code so the search and cross-product mechanics can be exercised
# without compiling or solving. EXTRA_ARGS (after --) are forwarded verbatim to
# run_duct.py, so any run_duct default (n-step, ncell, ...) can be overridden
# there.

set -uo pipefail

CFL_START=12.0
DCFL=1.0
TOL=0.049
SCHEMES=()
FAC_MGRIDS=()
EXTRA_ARGS=()
DRY_RUN=0
JOBS=0
SETTLE_LINE=""  # set by converges(); read by find_limit after a converging run

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

if ! [[ $JOBS =~ ^[0-9]+$ ]]; then
    echo "--jobs must be a non-negative integer, or 0 for unlimited (got: $JOBS)" >&2; exit 1
fi

# Default matrix: the full n_stage={0,4} x sf_resid={0,1} cross, each crossed
# with fac_mgrid {0,0.2,0.4} below (12 cases). Both schemes now have a
# coarse-level IRS multigrid variant, so sf_resid=1 is swept for scree as well
# as RK.
#   n_stage=0 sf_resid=0  (scree, no smoothing)
#   n_stage=0 sf_resid=1  (scree + coarse IRS)
#   n_stage=4 sf_resid=0  (RK4, no smoothing)
#   n_stage=4 sf_resid=1  (RK4 + IRS)
# Scree runs at n_step=4000, RK4 at n_step=2000 (see the per-scheme note in
# converges() and the EXTRA_ARGS note above).
if [[ ${#SCHEMES[@]} -eq 0 ]]; then
    SCHEMES=("0:0.0" "0:1.0" "4:0.0" "4:1.0")
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
    SETTLE_LINE=""  # run_duct's "zeta settled ..." line, for the caller to parse
    if [[ $DRY_RUN -eq 1 ]]; then
        local rc=$(( RANDOM % 3 ))
        echo "[dry-run] cfl=$cfl fac_mgrid=$fac_mgrid n_stage=$n_stage" \
             "sf_resid=$sf_resid -> exit $rc" >&2
        return $rc
    fi
    # Tee run_duct's combined output so the trace still streams live to stderr
    # while a copy is kept to read the settling line from. pipefail (set above)
    # makes PIPESTATUS[0] -- run_duct's own exit code -- the value we key on,
    # not tee's.
    #
    # Per-scheme run length / log cadence: scree (n_stage=0) is far slower to
    # settle, so it gets a longer run (n_step=4000 vs run_duct's default 2000).
    # RK4 (n_stage=4) settles in a few hundred steps, so it gets a finer log
    # cadence (n_step_log=25) for better settling-step resolution; scree keeps
    # n_step_log=50 (both halve or better the run_duct default of 100). These
    # come before EXTRA_ARGS, so a "-- --n-step ..." override still wins.
    local sched=()
    if [[ $n_stage -eq 0 ]]; then
        sched=(--n-step 4000 --n-step-log 50)
    else
        sched=(--n-step 2000 --n-step-log 25)
    fi
    local tmpf rc
    tmpf=$(mktemp)
    uv run "$SCRIPT_DIR/run_duct.py" \
        --cfl "$cfl" --fac-mgrid "$fac_mgrid" \
        --n-stage "$n_stage" --sf-resid "$sf_resid" \
        --n-levels 2 "${sched[@]}" \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "$tmpf" >&2
    rc=${PIPESTATUS[0]}
    SETTLE_LINE=$(grep -E 'zeta settled' "$tmpf" || true)
    rm -f "$tmpf"
    return $rc
}

# Descend from CFL_START toward the threshold, halving the step on each overshoot.
# The last argument is a path to a machine-readable record file that the final
# speedup pass reads: tab-separated "n_stage sf_resid fac_mgrid cfl step sec",
# with NA in the last three fields when no converging CFL was found.
find_limit() {
    local fac_mgrid="$1" n_stage="$2" sf_resid="$3" datfile="$4"
    local tag="n_stage=$n_stage sf_resid=$sf_resid fac_mgrid=$fac_mgrid"

    # Per-scheme threshold tolerance: scree (n_stage=0) has a much lower and
    # more tightly-spaced CFL band than RK4, so resolve its threshold twice as
    # finely by halving TOL. RK4 keeps the full TOL.
    local tol=$TOL
    if [[ $n_stage -eq 0 ]]; then
        tol=$(echo "scale=6; $TOL / 2" | bc -l)
    fi

    echo "=== $tag ===" >&2

    # We approach the threshold from above, so CFL_START must not converge.
    if converges "$CFL_START" "$fac_mgrid" "$n_stage" "$sf_resid"; then
        echo "$tag: CFL_START=$CFL_START already converges; raise --cfl-start to bracket the threshold."
        printf '%s\t%s\t%s\tNA\tNA\tNA\n' "$n_stage" "$sf_resid" "$fac_mgrid" >"$datfile"
        return
    fi

    # settle holds the "zeta settled ..." line of the most recent converging run,
    # which -- because CFL only descends and lo tracks the highest converging
    # trial -- is the settling behaviour at lo once the loop ends.
    # Termination guarantee: cfl is always known NOT to converge and lo (once
    # set) is always known TO converge, so the true threshold always lies in
    # [lo, cfl). Each time a trial converges, the gap cfl-lo equals exactly
    # that trial's dcfl. We stop the instant that gap is <= TOL -- i.e. right
    # after a converging trial whose dcfl is already <= TOL, checked BEFORE
    # halving -- so lo is guaranteed correct to within +-TOL. Stopping only
    # once dcfl has already been halved below TOL (checking at the top of the
    # loop) would under-guarantee accuracy to +-2*TOL, since the gap at that
    # point reflects the pre-halving step size.
    local cfl=$CFL_START dcfl=$DCFL lo="" trial settle=""
    while true; do
        trial=$(echo "scale=6; $cfl - $dcfl" | bc -l)
        if (( $(echo "$trial <= 0" | bc -l) )); then
            dcfl=$(echo "scale=6; $dcfl / 2" | bc -l)  # can't step below zero
            if (( $(echo "$dcfl <= $tol" | bc -l) )); then
                break  # give up: no converging CFL will be found at this resolution
            fi
            continue
        fi
        echo "--- $tag: trying CFL=$trial (dcfl=$dcfl) ---" >&2
        if converges "$trial" "$fac_mgrid" "$n_stage" "$sf_resid"; then
            lo=$trial                                   # overshoot: shrink step
            settle=$SETTLE_LINE                         # settling at this (so far max) CFL
            if (( $(echo "$dcfl <= $tol" | bc -l) )); then
                break  # this trial's step size already meets +-tol: done
            fi
            dcfl=$(echo "scale=6; $dcfl / 2" | bc -l)
        else
            cfl=$trial                                  # still not converging: commit
        fi
    done

    if [[ -z "$lo" ]]; then
        echo "$tag: no converging CFL found down to $cfl -- lower --cfl-start or --tol."
        printf '%s\t%s\t%s\tNA\tNA\tNA\n' "$n_stage" "$sf_resid" "$fac_mgrid" >"$datfile"
        return
    fi

    # Pull the settling step and wall time (ms -> s) from run_duct's line, e.g.
    # "zeta settled to <1% of range by step 450 of 1000 (48200 ms)".
    local step="NA" sec="NA"
    if [[ -n $settle ]]; then
        step=$(printf '%s' "$settle" | sed -E 's/.*by step ([0-9]+) of.*/\1/')
        local ms
        ms=$(printf '%s' "$settle" | sed -E 's/.*\(([0-9]+) ms\).*/\1/')
        sec=$(echo "scale=1; $ms / 1000" | bc -l)
    fi
    echo "$tag: max converging CFL ~= $lo, settles (zeta <1%) by step $step in ${sec}s" \
         "(does not converge at $cfl), tol=$tol"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$n_stage" "$sf_resid" "$fac_mgrid" "$lo" "$step" "$sec" >"$datfile"
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
        # running case finishes. JOBS=0 means unlimited: never throttle.
        if (( JOBS > 0 )); then
            while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
        fi
        find_limit "$fac_mgrid" "$n_stage" "$sf_resid" "$RESULT_DIR/$idx.dat" \
            >"$RESULT_DIR/$idx.out" &
        idx=$((idx + 1))
    done
done
wait

for (( j = 0; j < idx; j++ )); do
    cat "$RESULT_DIR/$j.out"
done

# Speedup table: wall time to settle (zeta within 1%) at each case's max
# converging CFL is the cost to a settled solution. Normalise to the SLOWEST
# converging case so every ratio is >= 1 ("N times faster than the slowest"),
# which avoids depending on a baseline scheme that might not settle at all. The
# .dat files are read in matrix order; awk finds the max settle time in one pass
# and prints the ratio in a second.
echo ""
echo "Speedup to settled solution (baseline = slowest converging case):"
for (( j = 0; j < idx; j++ )); do cat "$RESULT_DIR/$j.dat"; done | awk -F'\t' '
    { rows[NR] = $0; if ($6 != "NA" && $6 + 0 > max) max = $6 + 0 }
    END {
        printf "  %-8s %-9s %-10s %8s %6s %9s %8s\n", \
            "n_stage", "sf_resid", "fac_mgrid", "CFL", "step", "settle_s", "speedup"
        for (i = 1; i <= NR; i++) {
            split(rows[i], f, "\t")
            sp = (f[6] == "NA") ? "NA" : sprintf("%.2f", max / (f[6] + 0))
            printf "  %-8s %-9s %-10s %8s %6s %9s %8s\n", \
                f[1], f[2], f[3], f[4], f[5], f[6], sp
        }
    }'

echo "All $idx cases finished in ${SECONDS}s (jobs=$JOBS)." >&2
