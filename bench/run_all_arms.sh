#!/usr/bin/env bash
# Re-measure every arm on the pinned-budget build, with the corrected method.
#
# One build, all arms. That is only safe because pinning GCC's unit-level
# inline budgets makes each kernel's codegen independent of what else is in
# the program -- verified with bench/codegen_gauge.py, which reports an
# identical fingerprint for set_residual under EMBER_BENCH_KERNELS unset /
# nodal / all.
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
# Everything else per bench/bench_prod_baseline.py: one arm per process, rank
# barrier before every timed call, no flush, median per rank then across
# ranks, replication at the launch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LAUNCHES="${LAUNCHES:-10}"
NRANKS="${NRANKS:-16}"
NCELL="${NCELL:-1000000}"
REPS="${REPS:-30}"
ARMS="${ARMS:-prod staged split multall nodal tbaos prodsoa rinv}"
RESULTS="${RESULTS:-bench/results/bench_all_arms.jsonl}"
# Which kernel's arm set to time: `residual` (bench/residual_arms.py), `irs`
# (bench/irs_arms.py) or `visc`/`tauq` (bench/visc_arms.py). Also picks the
# symbol fingerprinted below.
KERNEL="${KERNEL:-residual}"
case "$KERNEL" in
    irs | update) GAUGE_SYM="${GAUGE_SYM:-smooth_residual_tri_tiled_}" ;;
    visc | viscpair) GAUGE_SYM="${GAUGE_SYM:-set_visc_force_}" ;;
    tauq)         GAUGE_SYM="${GAUGE_SYM:-set_tau_q_soa_}" ;;
    *)            GAUGE_SYM="${GAUGE_SYM:-set_residual_}" ;;
esac

# Which physical CPUs the ranks are pinned to, rank r -> CPUS[r]. Defaults to
# bench/socket_cpus.py's auto-detection (NRANKS distinct physical cores on
# one NUMA node/socket -- ported from duct/job_timing.py's
# detect_socket_cpus), falling back to sequential 0,1,2,... when no single
# node has NRANKS physical cores (e.g. NRANKS=16 spanning both sockets on a
# 2x8-core part).
#
# Auto-detection is still WRONG on a hybrid part: a P-core/E-core chip's
# physical cores on one node have different clocks, cache sizes and
# prefetchers, so "median across ranks" becomes the median of a multi-modal
# distribution and means nothing. Set CPUS explicitly to a list of
# same-class, non-sibling CPUs -- check with `lscpu -e` (the CORE column
# identifies siblings, MAXMHZ the class).
CPUS="${CPUS:-}"
if [ -z "$CPUS" ]; then
    CPUS="$(uv run python bench/socket_cpus.py --n "$NRANKS" 2>/dev/null || true)"
    if [ -z "$CPUS" ]; then
        for ((c = 0; c < NRANKS; c++)); do CPUS="$CPUS $c"; done
    fi
fi
read -r -a CPU_LIST <<< "$CPUS"
if [ "${#CPU_LIST[@]}" -lt "$NRANKS" ]; then
    echo "CPUS lists ${#CPU_LIST[@]} cpus but NRANKS=$NRANKS" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
# `uv run` re-syncs the project by default, which REBUILDS the extension from
# setup.py with whatever EMBER_BENCH_KERNELS is set in *that* environment --
# i.e. unset. Every bench/subroutines/ arm then vanishes from the .so between
# the build you fingerprinted and the run you are timing. Silent, and it
# invalidates the whole comparison. Build deliberately (EMBER_BENCH_KERNELS=...
# make compile) and never let the driver rebuild underneath it.
export UV_NO_SYNC=1
rm -f "$RESULTS"

echo "=== $(hostname): $LAUNCHES launches x $NRANKS ranks, ncell=$NCELL, $REPS reps"
echo "=== kernel: $KERNEL   arms: $ARMS"
echo "=== cpus:   ${CPU_LIST[*]:0:$NRANKS}"
uv run python bench/codegen_gauge.py "$GAUGE_SYM" | sed 's/^/    /'

for ((L = 0; L < LAUNCHES; L++)); do
    for ARM in $ARMS; do
        BARRIER="ember-all-$$-$L-$ARM"
        pids=()
        for ((rk = 0; rk < NRANKS; rk++)); do
            EMBER_BENCH_RANK=$rk EMBER_BARRIER="$BARRIER" \
                taskset -c "${CPU_LIST[$rk]}" uv run python bench/bench_prod_baseline.py \
                --nranks "$NRANKS" --ncell "$NCELL" --reps "$REPS" --arm "$ARM" \
                --kernel "$KERNEL" --launch "$L" --json "$RESULTS" >/dev/null &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p"; done
        rm -f "/dev/shm/$BARRIER" 2>/dev/null || true
    done
    echo "  launch $L done"
done

uv run python bench/bench_prod_baseline.py --analyze "$RESULTS"
