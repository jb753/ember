#!/usr/bin/env bash
#SBATCH -A brind-sl3-cpu
#SBATCH -p sapphire
#SBATCH --qos=intr
#SBATCH -N 1
#SBATCH -n 112
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH -o damp_bench-%j.out
#SBATCH -e damp_bench-%j.err
# A/B production damp_residual against damp_residual_merged, in one ifort
# build, on one sapphire node. Build happens ON THE NODE because INTEL_FLAGS
# carries -xHost (login nodes are Ice Lake, sapphire is not).
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$(dirname "$SUBMIT_DIR")"
source activate_ember.sh
cd "$SUBMIT_DIR"

RESULTS="${SUBMIT_DIR}/tools/bench_damp_ab.jsonl"
OPT="${SUBMIT_DIR}/tools/opt_report_damp.txt"

echo "=== node: $(hostname)  $(grep -m1 'model name' /proc/cpuinfo) ==="
EMBER_COMPILER=ifort EMBER_OPT_REPORT="$OPT" make compile 2>&1 | tail -2

echo
echo "=== Gate 1: vectorization of damp_residual_merged ==="
python3 - "$OPT" <<'PY'
import re, sys
txt = open(sys.argv[1], errors="replace").read()
FATAL = ("15344", "15346", "15522", "15523")
for s in re.split(r"\nBegin optimization report for: ", txt):
    name = s.split("\n", 1)[0].strip()
    if not any(t in name for t in ("damp_residual", "smooth_residual_tri")):
        continue
    vec = len(re.findall(r"remark #15300: LOOP WAS VECTORIZED", s))
    main = re.findall(r"remark #15335: loop was not vectorized[^\n]*", s)
    dep = [m for m in re.findall(r"remark #(\d+)", s) if m in FATAL]
    print(f"  {name}: vec={vec} main-miss={len(main)} dependence={len(dep)}")
    for m in dep[:4]:
        print(f"      FATAL #{m}")
PY

echo
echo "=== serial ==="
rm -f "$RESULTS"
srun --qos=intr -n1 -c1 --cpu-bind=cores \
    python3 tools/bench_damp_fused.py --mode serial --reps 50 --json "$RESULTS"

echo
echo "=== saturated (100 ranks) ==="
export EMBER_BENCH_START=$(python3 -c "import time; print(time.time() + 300)")
srun --qos=intr -n100 -c1 --cpu-bind=cores \
    python3 tools/bench_damp_fused.py --mode saturated --reps 50 --json "$RESULTS"

echo
echo "=== aggregate ==="
python3 - "$RESULTS" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for mode in ("serial", "saturated"):
    sel = [r for r in rows if r["mode"] == mode]
    if not sel:
        continue
    print(f"\n--- {mode} ({len(sel)} rank(s)) ---")
    for n in sel[0]["results"]:
        v = [r["results"][n]["median"] for r in sel]
        print(f"  {n:12s} median-of-ranks {statistics.median(v):7.3f} ns/cell"
              f"   spread {min(v):.2f}-{max(v):.2f}")
    for cand, base in (("damp_merged", "damp_prod"), ("irs_km", "irs_prod")):
        pair = sorted((r["results"][cand]["median"]
                       / r["results"][base]["median"] - 1) * 100 for r in sel)
        print(f"  => {cand} vs {base}: median {statistics.median(pair):+.2f}%"
              f"  wins {sum(1 for x in pair if x < 0)}/{len(pair)}"
              f"  worst {max(pair):+.2f}%")
PY
