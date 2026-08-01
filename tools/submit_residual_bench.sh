#!/usr/bin/env bash
#SBATCH -A brind-sl3-cpu
#SBATCH -p sapphire
#SBATCH --qos=intr
#SBATCH -N 1
#SBATCH -n 112
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH -o residual_bench-%j.out
#SBATCH -e residual_bench-%j.err
# A/B the production set_residual against the idiomatic set_residual_clean
# (src/ember/_fortran/residual_cand.f90), in one build, on one sapphire node.
#
# Directives live in this plain shell file rather than in the Python harness
# because a formatter run on a .py would happily rewrite "#SBATCH" to
# "# SBATCH", which sbatch silently ignores as a comment -- the same reasoning
# as ../duct/submit_cfl_timing.sh, which this is modelled on.
#
# THE BUILD HAPPENS ON THE COMPUTE NODE, deliberately. INTEL_FLAGS carries
# -xHost, which tunes to whichever machine runs the compiler. The login nodes
# are Ice Lake (Xeon 8368Q: avx512f/bw/dq/vl/vnni, no avx512_bf16, no amx);
# sapphire has both. A login-node build would therefore silently produce an
# Ice Lake binary and every number below would describe the wrong codegen.
# Compiling here also means production and candidate come out of one compiler
# invocation on one node, which is exactly what the same-.so protocol wants.
#
# Submit with `sbatch tools/submit_residual_bench.sh [...]` from a shell that
# already has activate_ember.sh sourced (venv + modules): sbatch snapshots the
# calling environment, so a bare login shell will fail.
set -euo pipefail

# Under sbatch this script runs from a spooled copy, so BASH_SOURCE resolves
# to the spool dir; SLURM_SUBMIT_DIR is the correct anchor.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EMBER_DIR="$SUBMIT_DIR"
PAPER_ROOT="$(dirname "$EMBER_DIR")"

cd "$PAPER_ROOT"
source activate_ember.sh
cd "$EMBER_DIR"

RESULTS="${EMBER_DIR}/tools/bench_residual_ab.jsonl"
OPT_REPORT="${EMBER_DIR}/tools/opt_report_residual_ab.txt"

echo "=== node: $(hostname)  $(grep -m1 'model name' /proc/cpuinfo) ==="

echo "=== building with ifort ON THIS NODE (-xHost targets sapphire) ==="
EMBER_COMPILER=ifort EMBER_OPT_REPORT="$OPT_REPORT" make compile

echo
echo "=== Gate 1: did the candidate vectorize? ==="
# The link-stage report describes post-IPO codegen, i.e. the code that
# actually runs. Note (doc section 16.2) that "LOOP WAS VECTORIZED" also
# covers gather-based SIMD, so this is necessary, not sufficient -- a winner
# still gets perf annotate afterwards.
for sym in iface_flow_row jface_flow_row kface_flow_plane \
           node_quantities avg_along_i avg_along_j avg_along_k face_flux; do
    echo "  ${sym}: $(grep -ci "$sym" "$OPT_REPORT" 2>/dev/null || echo 0) mentions"
done
echo "  (full report: $OPT_REPORT)"
grep -i "vector dependence prevents" "$OPT_REPORT" | head -20 || true

echo
echo "=== Gate 2 + serial regime (1 rank, pinned) ==="
rm -f "$RESULTS"
srun --qos=intr -n1 -c1 --cpu-bind=cores \
    python3 tools/bench_residual_variants.py \
        --mode serial --reps 50 --sweep-tiles --json "$RESULTS"

echo
echo "=== saturated regime (100 ranks, each with its own 1M-cell grid) ==="
# Ranks rendezvous on a shared wall-clock start so the timed windows overlap
# and the DRAM contention is real rather than staggered. The delay must cover
# the slowest rank's grid build.
export EMBER_BENCH_START=$(python3 -c "import time; print(time.time() + 300)")
srun --qos=intr -n100 -c1 --cpu-bind=cores \
    python3 tools/bench_residual_variants.py \
        --mode saturated --reps 50 --sweep-tiles --json "$RESULTS"

echo
echo "=== aggregate ==="
python3 - "$RESULTS" <<'PY'
import json, statistics, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
for mode in ("serial", "saturated"):
    sel = [r for r in rows if r["mode"] == mode]
    if not sel:
        continue
    names = list(sel[0]["results"])
    print(f"\n--- {mode} ({len(sel)} rank(s)) ---")
    med = {}
    for n in names:
        vals = [r["results"][n]["median"] for r in sel]
        mins = [r["results"][n]["min"] for r in sel]
        med[n] = statistics.median(vals)
        print(
            f"  {n:8s} median-of-ranks {med[n]:7.3f} ns/cell   "
            f"min-across-ranks {min(mins):7.3f}   "
            f"spread {min(vals):.2f}-{max(vals):.2f}"
        )
    base = names[0]
    for n in names[1:]:
        print(f"  => {n} vs {base}: {(med[n]/med[base]-1)*100:+.1f}% (median of ranks)")
PY

echo
echo "done. raw rows: $RESULTS"
