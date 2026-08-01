#!/usr/bin/env bash
#SBATCH -A brind-sl3-cpu
#SBATCH -p sapphire
#SBATCH --qos=intr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=00:30:00
#SBATCH -o gate1-%j.out
#SBATCH -e gate1-%j.err
# Gate 1 only: build with ifort ON THE SAPPHIRE NODE and report every
# vectorization miss in set_residual_tiled's hot loops. No timing here --
# perf is not run until this gate is clean.
set -euo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$(dirname "$SUBMIT_DIR")"
source activate_ember.sh
cd "$SUBMIT_DIR"

OPT=tools/opt_report_gate1.txt
echo "=== node: $(hostname) $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) ==="
EMBER_COMPILER=ifort EMBER_OPT_REPORT="$OPT" make compile 2>&1 | tail -3

echo
echo "=== vectorization of the tiled kernel's hot loops ==="
python3 - "$OPT" <<'PY'
import re, sys
txt = open(sys.argv[1], errors="replace").read()
secs = re.split(r"\nBegin optimization report for: ", txt)

# The HOT loops -- the ones that run once per node/face and therefore set
# the runtime. set_wall_row is deliberately excluded: it fills one
# boundary face layer per edge tile (O(surface)), and its i-boundary fill
# writes wf(1,j,k)/wf(fi,j,k), a stride-fi store that ifort is correct to
# leave scalar.
HOT = ["node_quantities", "avg_along_i", "avg_along_j", "avg_along_k",
       "face_flux", "diff_into_du",
       "iface_flows", "jface_flows", "kface_flows", "set_residual_naive",
       "iface_flow_row_ca", "jface_flow_row_ca", "kface_flow_plane_ca"]
COLD = ["set_wall_row"]

# Fatal = a real dependence or unsupported structure. #15335 is a COST
# decision ("possible but seems inefficient"), and on peel/remainder loops
# it is benign by construction -- the main loop next to it vectorized.
FATAL = ("15344", "15346", "15522", "15523")

bad = 0
for s in secs:
    name = s.split("\n", 1)[0].strip()
    short = name.replace("residual_tiled_helpers_mp_", "").rstrip("_")
    if not any(w in name for w in HOT + COLD):
        continue
    vec  = len(re.findall(r"remark #15300: LOOP WAS VECTORIZED", s))
    main = re.findall(r"remark #15335: loop was not vectorized[^\n]*", s)
    pr   = re.findall(r"remark #15335: (?:peel|remainder) loop", s)
    dep  = [m for m in re.findall(r"remark #(\d+)", s) if m in FATAL]
    hot  = any(w in name for w in HOT)
    fail = dep or (hot and main)
    if fail:
        bad += 1
    print(f"[{'MISS' if fail else 'OK  '}] {short:20s} vec={vec:2d} "
          f"main-miss={len(main)} peel/rem-miss={len(pr)} dependence={len(dep)}"
          f"{'   (cold, not gated)' if not hot else ''}")
    for m in dep[:4]:
        print(f"         FATAL remark #{m}")
print()
print("GATE 1:", "PASS -- all hot loops vectorized, no dependence misses"
      if bad == 0 else f"FAIL -- {bad} hot routine(s) with misses")
sys.exit(1 if bad else 0)
PY
