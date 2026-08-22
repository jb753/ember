#!/usr/bin/env bash
# A/B two or more BUILDS of the same production kernel, launch-interleaved.
#
# WHY THIS EXISTS. run_all_arms.sh compares arms that live side by side in one
# .so, which is the right instrument for a candidate kernel written as an arm.
# It is the WRONG one for a change to production itself: freezing a full copy
# of set_visc_force alongside the modified one puts two ~370-line near-
# duplicates in a single translation unit, and bench/README.md records a +56%
# timing swing from exactly that (unit-level inline budgets starving each
# other), which vanished when the arm was isolated.
#
# So each candidate gets its OWN build, and the .so is swapped between
# launches. That keeps every arm's codegen identical to what production would
# ship, and -- because the swap is launch-outer, build-inner -- each build is
# still sampled across the whole time window, so thermal or background drift
# cannot alias onto whichever build ran last. bench_prod_baseline.py's
# --label records which build a row came from, and its paired-within-launch
# analysis then differences them inside a launch.
#
# The .so is the whole build artefact (f2py builds one extension module), so
# copying it into place is a complete swap. Build each candidate with
# `make compile` and copy src/ember/fortran*.so aside BEFORE running this.
#
# Usage:
#   BUILDS="visc=/path/base.so cand=/path/new.so" \
#   KERNEL=visc ARM=visc NRANKS=8 NCELL=1000000 REPS=30 LAUNCHES=10 \
#   RESULTS=bench/results/my_ab.jsonl bench/run_so_ab.sh
#
# The label matching bench_prod_baseline.py's baseline list (visc, tauq, prod,
# irs, unfused) is the one everything else is reported against, so name the
# incumbent build with it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUILDS="${BUILDS:?set BUILDS=\"label=path label=path ...\"}"
LAUNCHES="${LAUNCHES:-10}"
NRANKS="${NRANKS:-8}"
NCELL="${NCELL:-1000000}"
REPS="${REPS:-30}"
KERNEL="${KERNEL:-visc}"
ARM="${ARM:-visc}"
PERIODIC_K="${PERIODIC_K:-}"
PERIODIC_ARGS=()
[ -n "$PERIODIC_K" ] && PERIODIC_ARGS=(--periodic-k "$PERIODIC_K")
RESULTS="${RESULTS:-bench/results/bench_so_ab.jsonl}"
# Symbol the fingerprint is taken on. Defaults to the entry point KERNEL
# times, so the printed gauge is of the kernel actually under test rather
# than of whatever the last A/B looked at.
case "${SYMBOL:-}" in
    "") case "$KERNEL" in
            residual|update) SYMBOL=set_residual_ ;;
            irs) SYMBOL=smooth_residual_tri_tiled_ ;;
            tauq) SYMBOL=set_tau_q_soa_ ;;
            *) SYMBOL=set_visc_force_ ;;
        esac ;;
esac

SO_DEST="$(ls src/ember/fortran.cpython-*.so)"
[ -f "$SO_DEST" ] || { echo "no built extension at src/ember/fortran*.so" >&2; exit 1; }
# The in-tree .so is about to be overwritten repeatedly; keep the one that was
# there so the tree is left as it was found.
ORIG="$(mktemp)"
cp "$SO_DEST" "$ORIG"
restore() { cp "$ORIG" "$SO_DEST"; rm -f "$ORIG"; }
trap restore EXIT

for spec in $BUILDS; do
    path="${spec#*=}"
    [ -f "$path" ] || { echo "no such .so: $path" >&2; exit 1; }
done

CPUS="${CPUS:-}"
if [ -z "$CPUS" ]; then
    CPUS="$(uv run python bench/socket_cpus.py --n "$NRANKS" 2>/dev/null || true)"
    if [ -z "$CPUS" ]; then
        for ((c = 0; c < NRANKS; c++)); do CPUS="$CPUS $c"; done
    fi
fi
read -r -a CPU_LIST <<< "$CPUS"

export OMP_NUM_THREADS=1
# `uv run` would otherwise re-sync and REBUILD the extension, overwriting the
# .so this script just swapped in. See run_all_arms.sh's note.
export UV_NO_SYNC=1
rm -f "$RESULTS"

echo "=== $(hostname): $LAUNCHES launches x $NRANKS ranks, ncell=$NCELL, $REPS reps"
echo "=== kernel: $KERNEL  arm: $ARM  builds: $BUILDS"
echo "=== cpus:   ${CPU_LIST[*]:0:$NRANKS}"
for spec in $BUILDS; do
    label="${spec%%=*}"
    cp "${spec#*=}" "$SO_DEST"
    echo "--- $label"
    uv run python bench/codegen_gauge.py "$SYMBOL" 2>/dev/null | sed 's/^/    /' || true
done

for ((L = 0; L < LAUNCHES; L++)); do
    for spec in $BUILDS; do
        label="${spec%%=*}"
        cp "${spec#*=}" "$SO_DEST"
        BARRIER="ember-soab-$$-$L-$label"
        pids=()
        for ((rk = 0; rk < NRANKS; rk++)); do
            EMBER_BENCH_RANK=$rk EMBER_BARRIER="$BARRIER" \
                taskset -c "${CPU_LIST[$rk]}" uv run python bench/bench_prod_baseline.py \
                --nranks "$NRANKS" --ncell "$NCELL" --reps "$REPS" --arm "$ARM" \
                --label "$label" --kernel "$KERNEL" "${PERIODIC_ARGS[@]}" \
                --launch "$L" --json "$RESULTS" >/dev/null &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p"; done
        rm -f "/dev/shm/$BARRIER" 2>/dev/null || true
    done
    echo "  launch $L done"
done

uv run python bench/bench_prod_baseline.py --analyze "$RESULTS"
