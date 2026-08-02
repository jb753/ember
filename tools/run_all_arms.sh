#!/usr/bin/env bash
# Re-measure every arm on the pinned-budget build, with the corrected method.
#
# One build, all arms. That is only safe because pinning GCC's unit-level
# inline budgets makes each kernel's codegen independent of what else is in
# the program -- verified with tools/codegen_gauge.py, which reports an
# identical fingerprint for set_residual under EMBER_ARMS unset / nodal / all.
# Without that pinning, the arms perturb each other's compilation and no
# ranking means anything (production's set_residual went 7,818 -> 10,726
# instructions purely from adding benchmark files).
#
# LAUNCH-OUTER, ARM-INNER. Each arm is sampled across the whole time window
# rather than in one contiguous block, so slow thermal or background drift
# cannot alias onto whichever arm happened to run last. This is the drift
# cancellation the old round-robin was reaching for, applied at the launch
# level -- where, unlike per-call interleaving, it does not couple the arms
# through cache state or rank phase.
#
# Everything else per tools/bench_prod_baseline.py: one arm per process, rank
# barrier before every timed call, no flush, median per rank then across
# ranks, replication at the launch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LAUNCHES="${LAUNCHES:-10}"
NRANKS="${NRANKS:-16}"
NCELL="${NCELL:-1000000}"
REPS="${REPS:-30}"
ARMS="${ARMS:-prod staged split multall nodal tbaos prodsoa rinv}"
RESULTS="${RESULTS:-tools/bench_all_arms.jsonl}"

export OMP_NUM_THREADS=1
rm -f "$RESULTS"

echo "=== $(hostname): $LAUNCHES launches x $NRANKS ranks, ncell=$NCELL, $REPS reps"
echo "=== arms: $ARMS"
uv run python tools/codegen_gauge.py set_residual_ | sed 's/^/    /'

for ((L = 0; L < LAUNCHES; L++)); do
    for ARM in $ARMS; do
        BARRIER="ember-all-$$-$L-$ARM"
        pids=()
        for ((rk = 0; rk < NRANKS; rk++)); do
            EMBER_BENCH_RANK=$rk EMBER_BARRIER="$BARRIER" \
                taskset -c "$rk" uv run python tools/bench_prod_baseline.py \
                --nranks "$NRANKS" --ncell "$NCELL" --reps "$REPS" --arm "$ARM" \
                --launch "$L" --json "$RESULTS" >/dev/null &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p"; done
        rm -f "/dev/shm/$BARRIER" 2>/dev/null || true
    done
    echo "  launch $L done"
done

uv run python tools/bench_prod_baseline.py --analyze "$RESULTS"
