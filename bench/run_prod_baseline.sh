#!/usr/bin/env bash
# Launch-replicated baseline for `prod` alone. See bench_prod_baseline.py for
# why each choice is what it is.
#
# The experimental unit is the LAUNCH, not the rep: each launch is a fresh set
# of processes, so page placement, allocation alignment, core assignment and
# thermal state are re-drawn every time. Repeats at that level are the only
# thing that produces a real error bar.
#
# Historical default: 16 ranks = cores 0-15 = 8 physical cores on EACH
# socket. SMT siblings (16-31) are left idle, so no two ranks share a core.
# Two memory controllers, 8 ranks per controller. This is NOT the 8-rank
# one-socket regime and must not be spliced with it.
#
# CPU pinning: set CPUS explicitly ("0 1 2 ... 7") to override, exactly like
# run_all_arms.sh's own CPUS -- needed on a hybrid part or any machine where
# consecutive cpu ids are not what you think they are (see that script's
# comment). Left unset, this auto-detects NRANKS distinct physical cores on
# one NUMA node/socket (bench/socket_cpus.py, ported from duct/job_timing.py's
# detect_socket_cpus) whenever such a node exists -- correct for the 8-rank
# one-socket regime on any topology, not just the Haswell workstation this
# harness's numbers were first measured on. When no single node has NRANKS
# physical cores (e.g. NRANKS=16 spanning both sockets on a 2x8-core part),
# it falls back to the historical sequential 0..NRANKS-1.
#
# Usage: bench/run_prod_baseline.sh [launches] [nranks] [ncell] [reps]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LAUNCHES="${1:-10}"
NRANKS="${2:-16}"
NCELL="${3:-1000000}"
REPS="${4:-30}"
ARM="${5:-prod}"
# Which kernel's arm set: residual (default), irs, update, visc, tauq, viscpair.
KERNEL="${KERNEL:-residual}"
# viscpair only: make the duct's k faces periodic ("full" or "hmesh"). The
# seam-free arm needs a real seam; without one the fused arms never exercise
# the halo path they model, and callers_pair skips the seam-free arm entirely.
PERIODIC_K="${PERIODIC_K:-}"
PERIODIC_ARGS=()
[ -n "$PERIODIC_K" ] && PERIODIC_ARGS=(--periodic-k "$PERIODIC_K")
RESULTS="${RESULTS:-bench/results/bench_prod_baseline.jsonl}"

export OMP_NUM_THREADS=1
# See run_all_arms.sh: `uv run` would otherwise re-sync and rebuild the
# extension without EMBER_BENCH_KERNELS, silently dropping every arm.
export UV_NO_SYNC=1
[ "${KEEP:-0}" = 1 ] || rm -f "$RESULTS"

CPUS="${CPUS:-}"
if [ -z "$CPUS" ]; then
    CPUS="$(uv run python bench/socket_cpus.py --n "$NRANKS" 2>/dev/null || true)"
    if [ -z "$CPUS" ]; then
        for ((c = 0; c < NRANKS; c++)); do CPUS="$CPUS $c"; done
        REGIME="cores 0-$((NRANKS - 1)) (sequential fallback; no single socket has $NRANKS physical cores)"
    else
        REGIME="cpus [$CPUS] (auto-detected: $NRANKS physical cores, one socket)"
    fi
else
    REGIME="cpus [$CPUS] (explicit CPUS override)"
fi
read -r -a CPU_LIST <<< "$CPUS"
if [ "${#CPU_LIST[@]}" -lt "$NRANKS" ]; then
    echo "CPUS lists ${#CPU_LIST[@]} cpus but NRANKS=$NRANKS" >&2
    exit 1
fi

echo "=== $(hostname): $LAUNCHES launches x $NRANKS ranks, ncell=$NCELL, $REPS reps ==="
echo "=== $REGIME, SMT siblings idle, no flush, arm=$ARM kernel=$KERNEL periodic_k=${PERIODIC_K:-none} ==="

for ((L = 0; L < LAUNCHES; L++)); do
    # Unique segment per launch so a crashed run cannot poison the next one.
    BARRIER="ember-baseline-$$-$L"
    pids=()
    for ((rk = 0; rk < NRANKS; rk++)); do
        EMBER_BENCH_RANK=$rk EMBER_BARRIER="$BARRIER" \
            taskset -c "${CPU_LIST[$rk]}" uv run python bench/bench_prod_baseline.py \
            --nranks "$NRANKS" --ncell "$NCELL" --reps "$REPS" --arm "$ARM" \
            --kernel "$KERNEL" "${PERIODIC_ARGS[@]}" \
            --launch "$L" --json "$RESULTS" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    # Belt and braces: drop the segment if a rank died before unlinking it.
    rm -f "/dev/shm/$BARRIER" 2>/dev/null || true
done

uv run python bench/bench_prod_baseline.py --analyze "$RESULTS"
