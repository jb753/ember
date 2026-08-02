#!/usr/bin/env bash
# Rule 7 isolation: why did `multall` vs `prod` at 1M contended move from
# -17.7% (docs section 26, four arms in the .so) to -8.6% (seven arms)?
#
# The suspect is the HARNESS, not drift: arms are compared round-robin, so
# every arm runs with the other arms' footprints in cache. multall stages
# fi/fj/fk -- ~12 MB at 1M against this box's ~20 MB L3 -- so it is the arm
# most exposed to how many siblings pass through L3 between its reps. This
# sweeps the arm SET at fixed size and rank count:
#
#   full  all seven arms                     (the current ladder)
#   s26   prod staged split multall           (section 26's exact set)
#   pair  prod multall                        (harness minimised)
#
# If `s26` reproduces -17.7%, the discrepancy is the arm set and section 26's
# headline was partly L3 residency between reps. If all three agree, it was
# run-to-run drift and the arm set is innocent.
#
# TWO RANK COUNTS, and they are not interchangeable:
#   6  -- cores 0-5, ONE socket. The only regime comparable to section 26.
#   16 -- cores 0-15, i.e. 8 physical cores on EACH socket (SMT siblings are
#         16-31 and are deliberately left idle). Two memory controllers, 8
#         ranks per controller instead of 6, plus NUMA. A different regime,
#         not a bigger one -- report it separately, never spliced.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NCELL="${NCELL:-1000000}"
REPS="${REPS:-50}"
RESULTS="tools/bench_multall_isolation.jsonl"

export OMP_NUM_THREADS=1
rm -f "$RESULTS"

echo "=== $(hostname) ncell=$NCELL reps=$REPS ==="

for NRANKS in 6 16; do
    for SET in full s26 pair; do
        case "$SET" in
            full) ARMS="staged,split,multall,nodal,tbaos,prodsoa" ;;
            s26)  ARMS="staged,split,multall" ;;
            pair) ARMS="multall" ;;
        esac
        echo
        echo "=== ranks=$NRANKS set=$SET arms=prod,$ARMS ==="
        START=$(uv run python -c "import time; print(time.time() + 180)")
        pids=()
        for ((rk = 0; rk < NRANKS; rk++)); do
            EMBER_BENCH_RANK=$rk EMBER_BENCH_START=$START EMBER_ARMSET="$SET" \
                taskset -c "$rk" uv run python tools/bench_residual_staged.py \
                --mode contended --ncell "$NCELL" --reps "$REPS" \
                --arms "$ARMS" --json "$RESULTS" &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait "$p"; done
        # Tag the rows just written with their arm set and rank count.
        uv run python - "$RESULTS" "$SET" "$NRANKS" <<'PY'
import json, sys
path, armset, nranks = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(path)]
for r in rows:
    r.setdefault("armset", armset)
    r.setdefault("nranks", nranks)
with open(path, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
PY
    done
done

echo
echo "=== aggregate ==="
uv run python - "$RESULTS" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'ranks':>6} {'armset':>7} {'n':>3} {'prod':>8} {'multall':>8} "
      f"{'delta%':>8} {'wins':>6}  spread(multall)")
for nranks in sorted({r["nranks"] for r in rows}):
    for armset in ("full", "s26", "pair"):
        sel = [r for r in rows if r["nranks"] == nranks and r["armset"] == armset]
        if not sel:
            continue
        p = [r["results"]["prod"]["median"] for r in sel]
        t = [r["results"]["multall"]["median"] for r in sel]
        d = sorted((a / b - 1) * 100 for a, b in zip(t, p))
        print(f"{nranks:>6} {armset:>7} {len(sel):>3} {statistics.median(p):8.2f} "
              f"{statistics.median(t):8.2f} {statistics.median(d):+8.2f} "
              f"{sum(1 for x in d if x < 0):>3}/{len(d):<2} "
              f"{min(t):.1f}-{max(t):.1f}")
PY
