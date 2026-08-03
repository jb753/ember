#!/usr/bin/env bash
# One-factor-at-a-time compiler flag sweep, gated by the codegen fingerprint.
#
# The largest single effect ever found on set_residual was four compiler
# --params (-36%, 73.2 -> 46.7 ns/cell), and it was found by accident. Nobody
# has swept this space with a working instrument. This does that.
#
# GAUGE FIRST, TIME SECOND. Each configuration is built and fingerprinted
# before it is timed. If a flag leaves set_residual's machine code unchanged it
# is a no-op: recorded as such and NOT timed, which turns a ~5 minute point
# into a ~70 second one. This is how -fipa-pta was originally judged
# (setup.py:23) and it is what makes a wide sweep affordable.
#
# PRODUCTION-ONLY BUILDS. EMBER_ARMS is unset throughout, so what is measured
# is the kernel that actually ships -- not the benchmark build, whose extra
# files inflate the inline budget and made prod ~20% faster than reality.
#
# NOTE since section 28, the inline-budget pair is a DEFAULT in setup.py, so
# the 'base' config below is no longer the pre-section-28 baseline -- it is
# the new default, which already includes them. To reproduce the original
# comparison, add the negated params (e.g. --param=inline-unit-growth=20).
#
# Usage:
#   bench/run_flag_sweep.sh              # serial screen (fast, uses min)
#   MODE=contended bench/run_flag_sweep.sh
#   CONFIGS="base all4" bench/run_flag_sweep.sh    # subset by name
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-serial}"
NCELL="${NCELL:-1000000}"
OUT="${OUT:-bench/results/bench_flagsweep_$MODE.jsonl}"
FPDIR="${FPDIR:-bench/.flagsweep_fp}"

if [ "$MODE" = serial ]; then
    NRANKS=1; LAUNCHES="${LAUNCHES:-10}"; REPS="${REPS:-30}"; STAT=min
else
    NRANKS="${NRANKS:-16}"; LAUNCHES="${LAUNCHES:-10}"; REPS="${REPS:-30}"; STAT=median
fi

# --- the configurations -----------------------------------------------------
# Phase 1: attribute the -36% bundle, then test whether LTO partitioning
# subsumes it (both act on cross-unit inlining).
UG="--param=inline-unit-growth=1000000"
LU="--param=large-unit-insns=1000000"
LFG="--param=large-function-growth=1000000"
LFI="--param=large-function-insns=1000000"
ALL4="$UG $LU $LFG $LFI"

declare -A CFG=(
    [base]=""
    [ug]="$UG"
    [lu]="$LU"
    [lfg]="$LFG"
    [lfi]="$LFI"
    [all4]="$ALL4"
    [no_ug]="$LU $LFG $LFI"
    [no_lu]="$UG $LFG $LFI"
    [no_lfg]="$UG $LU $LFI"
    [no_lfi]="$UG $LU $LFG"
    [ltopart]="-flto-partition=none"
    [ltopart_all4]="-flto-partition=none $ALL4"
    # Phase 1b: leave-one-out was flat (any THREE reproduce all4 bit-exactly),
    # so the open question is whether a PAIR does. ug and lu are the same
    # constraint (identical codegen), so only three pairs are distinct.
    [p_ug_lfg]="$UG $LFG"
    [p_ug_lfi]="$UG $LFI"
    [p_lfg_lfi]="$LFG $LFI"
    # Phase 2: from the all4 baseline, one factor at a time.
    [a_ipapta]="$ALL4 -fipa-pta"
    [a_prefetch]="$ALL4 -fprefetch-loop-arrays"
    [a_align]="$ALL4 -falign-loops=32 -falign-functions=64"
    [a_nosemint]="$ALL4 -fno-semantic-interposition"
    [a_unroll]="$ALL4 --param=max-unroll-times=8 --param=max-completely-peel-times=16"
)
ORDER="base ug lu lfg lfi all4 no_ug no_lu no_lfg no_lfi ltopart ltopart_all4"
[ -n "${CONFIGS:-}" ] && ORDER="$CONFIGS"

mkdir -p "$FPDIR"
rm -f "$OUT"
export OMP_NUM_THREADS=1

echo "=== flag sweep: mode=$MODE ncell=$NCELL ranks=$NRANKS launches=$LAUNCHES stat=$STAT"

for NAME in $ORDER; do
    FLAGS="${CFG[$NAME]}"
    echo
    echo "########## $NAME :: ${FLAGS:-<shipped flags>}"
    if ! EMBER_ARMS="" EMBER_FFLAGS_EXTRA="$FLAGS" \
         EMBER_MARCH="-march=native -mtune=native" make compile >/dev/null 2>&1; then
        echo "  BUILD FAILED -- skipped"
        continue
    fi

    uv run python bench/codegen_gauge.py set_residual_ --json "$FPDIR/$NAME.json" \
        2>/dev/null | sed 's/^/  /'
    FP=$(uv run python -c "import json;print(json.load(open('$FPDIR/$NAME.json'))['sha256'])")

    # Gauge screen: identical machine code => nothing to measure.
    DUP=""
    for PREV in $ORDER; do
        [ "$PREV" = "$NAME" ] && break
        [ -f "$FPDIR/$PREV.json" ] || continue
        PFP=$(uv run python -c "import json;print(json.load(open('$FPDIR/$PREV.json'))['sha256'])")
        [ "$PFP" = "$FP" ] && { DUP="$PREV"; break; }
    done
    if [ -n "$DUP" ]; then
        echo "  NO-OP: identical codegen to '$DUP' -- not timed"
        continue
    fi

    # Gate 2 before any number is believed.
    if ! uv run pytest tests/test_residual_golden.py -q >/dev/null 2>&1; then
        echo "  GOLDENS FAILED -- not timed"
        continue
    fi

    RESULTS="$OUT.$NAME"
    rm -f "$RESULTS"
    for ((L = 0; L < LAUNCHES; L++)); do
        BARRIER="ember-flag-$$-$L"
        pids=()
        for ((rk = 0; rk < NRANKS; rk++)); do
            EMBER_BENCH_RANK=$rk EMBER_BARRIER="$BARRIER" \
                taskset -c "$rk" uv run python bench/bench_prod_baseline.py \
                --nranks "$NRANKS" --ncell "$NCELL" --reps "$REPS" --arm prod \
                --launch "$L" --json "$RESULTS" >/dev/null &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p"; done
        rm -f "/dev/shm/$BARRIER" 2>/dev/null || true
    done
    uv run python - "$RESULTS" "$NAME" "$FP" "$STAT" "$OUT" <<'PY'
import json, statistics, sys
res, name, fp, stat, out = sys.argv[1:6]
rows = [json.loads(l) for l in open(res)]
per = []
for L in sorted({r["launch"] for r in rows}):
    v = [r.get(stat, r["median"]) for r in rows if r["launch"] == L]
    per.append(statistics.median(v))
m = statistics.median(per)
half = (max(per) - min(per)) / 2 / m * 100
sd = statistics.stdev(per) if len(per) > 1 else 0.0
print(f"  {m:.3f} ns/cell   half-range {half:.2f}%   stdev {sd / m * 100:.2f}%")
with open(out, "a") as fh:
    fh.write(json.dumps(dict(config=name, fingerprint=fp, stat=stat,
                             value=m, half_range_pct=half,
                             stdev_pct=sd / m * 100, launches=per)) + "\n")
PY
done

echo
echo "=== summary (vs 'base' = currently shipped flags) ==="
uv run python - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
base = next((r for r in rows if r["config"] == "base"), None)
b = base["value"] if base else None
print(f"{'config':>14} {'ns/cell':>9} {'+/-':>7} {'vs base':>9}  fingerprint")
for r in rows:
    rel = f"{(r['value'] / b - 1) * 100:+8.2f}%" if b else ""
    print(f"{r['config']:>14} {r['value']:>9.3f} {r['half_range_pct']:>6.2f}% {rel:>9}"
          f"  {r['fingerprint']}")
PY
