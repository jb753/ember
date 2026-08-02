#!/usr/bin/env bash
# Does the LLC flush remove the harness's arm-set dependence?
#
# Round-robin interleaving cancels thermal/frequency drift, which is why the
# harness does it -- but it also means every arm runs with its neighbours'
# footprints in cache. Sweeping the arm set at fixed binary, size and rank
# count moved `multall` vs `prod` at 1M contended from -15.5% (four arms) to
# -2.1% (two arms). That is the harness measuring itself.
#
# bench_residual_staged.py now streams a 48 MB buffer past the LLC before
# every timed call, untimed, so each arm starts cold regardless of what ran
# before it. This sweeps the arm set WITH and WITHOUT that flush:
#
#   flush off -> the spread across arm sets should be large  (the bug)
#   flush on  -> the spread should collapse                  (the fix)
#
# Arm sets. `ndpair` also answers the outstanding question of whether
# `nodal`'s win is itself harness-dependent -- it has only ever been measured
# inside the full set, and it must be held to the standard `multall` failed.
#
#   full    every arm
#   s26     prod staged split multall   (docs section 26's exact set)
#   tbpair  prod multall
#   ndpair  prod nodal
#
# Ranks are pinned to cores 0-5, ONE socket, the regime comparable to
# section 26. Usage: tools/run_harness_isolation.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NCELL="${NCELL:-1000000}"
REPS="${REPS:-50}"
NRANKS="${NRANKS:-6}"
RESULTS="tools/bench_harness_isolation.jsonl"

export OMP_NUM_THREADS=1
rm -f "$RESULTS"
echo "=== $(hostname) ncell=$NCELL ranks=$NRANKS reps=$REPS ==="

for FLUSH in off on; do
    for SET in full s26 tbpair ndpair; do
        case "$SET" in
            full)   ARMS="staged,split,multall,nodal,tbaos,prodsoa,rinv" ;;
            s26)    ARMS="staged,split,multall" ;;
            tbpair) ARMS="multall" ;;
            ndpair) ARMS="nodal" ;;
        esac
        FLAG=""
        [ "$FLUSH" = off ] && FLAG="--no-flush"
        echo
        echo "=== flush=$FLUSH set=$SET arms=prod,$ARMS ==="
        START=$(uv run python -c "import time; print(time.time() + 180)")
        pids=()
        for ((rk = 0; rk < NRANKS; rk++)); do
            EMBER_BENCH_RANK=$rk EMBER_BENCH_START=$START \
                taskset -c "$rk" uv run python tools/bench_residual_staged.py \
                --mode contended --ncell "$NCELL" --reps "$REPS" \
                --arms "$ARMS" $FLAG --json "$RESULTS" &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p"; done
        uv run python - "$RESULTS" "$SET" <<'PY'
import json, sys
path, armset = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(path)]
for r in rows:
    r.setdefault("armset", armset)
with open(path, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
PY
    done
done

echo
echo "=== aggregate: ratio vs prod, by arm set ==="
uv run python - "$RESULTS" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
sets = ["full", "s26", "tbpair", "ndpair"]
for arm in ("multall", "nodal"):
    print(f"\n--- {arm} vs prod ---")
    print(f"{'flush':>6} {'armset':>7} {'prod':>8} {arm:>8} {'delta%':>8} {'wins':>7}")
    for flush in (False, True):
        vals = []
        for s in sets:
            sel = [r for r in rows if r["armset"] == s and r["flush"] == flush
                   and arm in r["results"]]
            if not sel:
                continue
            p = [r["results"]["prod"]["median"] for r in sel]
            t = [r["results"][arm]["median"] for r in sel]
            d = sorted((a / b - 1) * 100 for a, b in zip(t, p))
            m = statistics.median(d)
            vals.append(m)
            print(f"{str(flush):>6} {s:>7} {statistics.median(p):8.2f} "
                  f"{statistics.median(t):8.2f} {m:+8.2f} "
                  f"{sum(1 for x in d if x < 0):>3}/{len(d):<3}")
        if len(vals) > 1:
            print(f"{'':>6} {'SPREAD':>7} {'':>8} {'':>8} {max(vals)-min(vals):8.2f} "
                  f"points across arm sets")
PY
