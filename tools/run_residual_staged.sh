#!/usr/bin/env bash
# A/B production set_residual against the staged (multall/multall) and split
# arms, across a size ladder, serial and socket-contended.
#
# LOCAL DRIVER -- this machine is not in a SLURM cluster, so the contended arm
# is N background processes pinned with taskset rather than srun ranks. All
# ranks are pinned to cores of ONE socket on purpose: 6 Haswell cores sustain
# more than that socket's memory controller can deliver (~10-15 GB/s per core
# against a ~59 GB/s controller), so this is a genuinely bandwidth-contended
# regime. Spread across both sockets it would not be, and the staged arm --
# which is predicted to lose on bandwidth -- would be flattered.
#
# It is still NOT the 100-rank sapphire regime. Report it as
# "N-rank socket-contended", never "saturated".
#
# Usage: tools/run_residual_staged.sh [nranks] [ncell ...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NRANKS="${1:-6}"
shift || true
SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=(100000 300000 1000000 2000000)
fi

RESULTS="tools/bench_residual_staged.jsonl"
PLOT="tools/bench_residual_staged.pdf"
REPS="${REPS:-50}"

export OMP_NUM_THREADS=1

echo "=== $(hostname)  $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-) ==="
echo "=== ranks=$NRANKS  sizes=${SIZES[*]}  reps=$REPS ==="
rm -f "$RESULTS"

echo
echo "=== correctness gate (once, at 1M cells) ==="
taskset -c 0 uv run python tools/bench_residual_staged.py --mode serial \
    --ncell 1000000 --check-only

for N in "${SIZES[@]}"; do
    echo
    echo "=== ncell=$N : serial ==="
    taskset -c 0 uv run python tools/bench_residual_staged.py --mode serial \
        --ncell "$N" --reps "$REPS" --json "$RESULTS"

    echo
    echo "=== ncell=$N : ${NRANKS}-rank socket-contended ==="
    # Every rank builds its own grid first (slow), then spins until a shared
    # wall-clock start so the timed windows overlap.
    START=$(uv run python -c "import time; print(time.time() + 180)")
    pids=()
    for ((rk = 0; rk < NRANKS; rk++)); do
        EMBER_BENCH_RANK=$rk EMBER_BENCH_START=$START \
            taskset -c "$rk" uv run python tools/bench_residual_staged.py \
            --mode contended --ncell "$N" --reps "$REPS" --json "$RESULTS" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
done

echo
echo "=== aggregate ==="
uv run python - "$RESULTS" "$PLOT" <<'PY'
import json, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open(sys.argv[1])]
arms = ["prod", "staged", "split", "multall", "nodal", "tbaos", "prodsoa"]
sizes = sorted({r["ncell"] for r in rows})
curves = {m: {a: [] for a in arms} for m in ("serial", "contended")}

for mode in ("serial", "contended"):
    for n in sizes:
        sel = [r for r in rows if r["mode"] == mode and r["ncell"] == n]
        if not sel:
            for a in arms:
                curves[mode][a].append(float("nan"))
            continue
        print(f"\n--- {mode}  ncell={n}  ({len(sel)} rank(s)) ---")
        for a in arms:
            v = [r["results"][a]["median"] for r in sel]
            curves[mode][a].append(statistics.median(v))
            print(f"  {a:8s} median-of-ranks {statistics.median(v):8.3f} ns/cell"
                  f"   spread {min(v):.2f}-{max(v):.2f}")
        for a in arms[1:]:
            pair = sorted((r["results"][a]["median"]
                           / r["results"]["prod"]["median"] - 1) * 100 for r in sel)
            print(f"  => {a} vs prod: median {statistics.median(pair):+.2f}%"
                  f"  wins {sum(1 for x in pair if x < 0)}/{len(pair)}"
                  f"  worst {max(pair):+.2f}%")
        st = [r["results"]["staged"]["median"] for r in sel]
        sp = [r["results"]["split"]["median"] for r in sel]
        d = sorted((a / b - 1) * 100 for a, b in zip(st, sp))
        print(f"  => staged vs split: median {statistics.median(d):+.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, mode in zip(axes, ("serial", "contended")):
    for a in arms:
        ax.plot(sizes, curves[mode][a], marker="o", label=a)
    ax.set_xscale("log")
    ax.set_xlabel("cells per block")
    ax.set_title(mode)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("ns/cell (median)")
axes[0].legend()
fig.tight_layout()
fig.savefig(sys.argv[2])
print(f"\nwrote {sys.argv[2]}")
PY
