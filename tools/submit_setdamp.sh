#!/usr/bin/env bash
#SBATCH -A brind-sl3-cpu
#SBATCH -p sapphire
#SBATCH --qos=intr
#SBATCH -N 1
#SBATCH -n 112
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH -o setdamp_bench-%j.out
#SBATCH -e setdamp_bench-%j.err
# A/B production damp_residual against damp_residual_merged, in one ifort
# build, on one sapphire node. Build happens ON THE NODE because INTEL_FLAGS
# carries -xHost (login nodes are Ice Lake, sapphire is not).
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$(dirname "$SUBMIT_DIR")"
source activate_ember.sh
cd "$SUBMIT_DIR"

RESULTS="${SUBMIT_DIR}/tools/bench_setdamp.jsonl"
OPT="${SUBMIT_DIR}/tools/opt_report_setdamp.txt"

echo "=== node: $(hostname)  $(grep -m1 'model name' /proc/cpuinfo) ==="
EMBER_COMPILER=ifort EMBER_OPT_REPORT="$OPT" make compile 2>&1 | tail -2

echo
echo "=== serial ==="
rm -f "$RESULTS"
srun --qos=intr -n1 -c1 --cpu-bind=cores \
    python3 tools/bench_setdamp.py --mode serial --reps 40 --json "$RESULTS"

echo
echo "=== saturated (100 ranks) ==="
export EMBER_BENCH_START=$(python3 -c "import time; print(time.time() + 300)")
srun --qos=intr -n100 -c1 --cpu-bind=cores \
    python3 tools/bench_setdamp.py --mode saturated --reps 40 --json "$RESULTS"

echo
echo "=== aggregate ==="
python3 - "$RESULTS" <<'PY2'
import json, statistics, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
for mode in ("serial","saturated"):
    sel=[r for r in rows if r["mode"]==mode]
    if not sel: continue
    names=list(sel[0]["results"]); base="unfused"
    print(f"\n--- {mode} ({len(sel)} rank(s)) ---")
    med={n:statistics.median([r["results"][n]["median"] for r in sel]) for n in names}
    for n in names:
        pair=sorted((r["results"][n]["median"]/r["results"][base]["median"]-1)*100 for r in sel)
        print(f"  {n:12s} {med[n]:7.3f} ns/cell   vs {base}: median {statistics.median(pair):+6.2f}%"
              f"  wins {sum(1 for x in pair if x<0)}/{len(pair)}")
PY2
