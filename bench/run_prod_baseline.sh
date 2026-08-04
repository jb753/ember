#!/usr/bin/env bash
# Launch-replicated baseline for `prod` alone. See bench_prod_baseline.py for
# why each choice is what it is.
#
# The experimental unit is the LAUNCH, not the rep: each launch is a fresh set
# of processes, so page placement, allocation alignment, core assignment and
# thermal state are re-drawn every time. Repeats at that level are the only
# thing that produces a real error bar.
#
# 16 ranks = cores 0-15 = 8 physical cores on EACH socket. SMT siblings
# (16-31) are left idle, so no two ranks share a core. Two memory
# controllers, 8 ranks per controller. This is NOT the 6-rank one-socket
# regime used earlier and must not be spliced with it.
#
# Usage: bench/run_prod_baseline.sh [launches] [nranks] [ncell] [reps]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LAUNCHES="${1:-10}"
NRANKS="${2:-16}"
NCELL="${3:-1000000}"
REPS="${4:-30}"
ARM="${5:-prod}"
# Which kernel's arm set: residual (default), irs, update, visc, tauq.
KERNEL="${KERNEL:-residual}"
RESULTS="${RESULTS:-bench/results/bench_prod_baseline.jsonl}"

export OMP_NUM_THREADS=1
# See run_all_arms.sh: `uv run` would otherwise re-sync and rebuild the
# extension without EMBER_BENCH_KERNELS, silently dropping every arm.
export UV_NO_SYNC=1
[ "${KEEP:-0}" = 1 ] || rm -f "$RESULTS"

echo "=== $(hostname): $LAUNCHES launches x $NRANKS ranks, ncell=$NCELL, $REPS reps ==="
echo "=== cores 0-$((NRANKS - 1)), SMT siblings idle, no flush, prod only ==="

for ((L = 0; L < LAUNCHES; L++)); do
    # Unique segment per launch so a crashed run cannot poison the next one.
    BARRIER="ember-baseline-$$-$L"
    pids=()
    for ((rk = 0; rk < NRANKS; rk++)); do
        EMBER_BENCH_RANK=$rk EMBER_BARRIER="$BARRIER" \
            taskset -c "$rk" uv run python bench/bench_prod_baseline.py \
            --nranks "$NRANKS" --ncell "$NCELL" --reps "$REPS" --arm "$ARM" \
            --kernel "$KERNEL" \
            --launch "$L" --json "$RESULTS" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    # Belt and braces: drop the segment if a rank died before unlinking it.
    rm -f "/dev/shm/$BARRIER" 2>/dev/null || true
done

uv run python bench/bench_prod_baseline.py --analyze "$RESULTS"
