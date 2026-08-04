#!/usr/bin/env bash
# Baseline for the viscous kernels: production set_visc_force (and one
# set_tau_q_soa point for the phase split) across the size ladder, in the
# 8-rank socket-contended regime.
#
# 8 ranks = cores 0-7 = one whole socket of this 2x8-core Haswell, SMT
# siblings idle. NOT the 16-rank both-sockets regime the set_residual
# baseline used: ranks spread across two memory controllers do not actually
# contend, which flatters a bandwidth-heavy kernel (README, "Measurement
# protocol"). Results from the two regimes must not be spliced.
#
# Usage: bench/run_visc_baseline.sh [launches] [reps]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LAUNCHES="${1:-10}"
REPS="${2:-30}"
NRANKS=8
SIZES="${SIZES:-100000 300000 1000000 2000000}"

for N in $SIZES; do
    echo "########## set_visc_force, ncell=$N ##########"
    RESULTS="bench/results/bench_visc_baseline_${N}.jsonl" KERNEL=visc \
        bench/run_prod_baseline.sh "$LAUNCHES" "$NRANKS" "$N" "$REPS" visc
done

# One phase-1 point, for the split between the two viscous kernels: item 1
# only addresses set_visc_force, so its share of the pair bounds the win.
echo "########## set_tau_q_soa, ncell=1000000 ##########"
RESULTS="bench/results/bench_tauq_baseline_1000000.jsonl" KERNEL=tauq \
    bench/run_prod_baseline.sh "$LAUNCHES" "$NRANKS" 1000000 "$REPS" tauq
