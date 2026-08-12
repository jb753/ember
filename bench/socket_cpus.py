#!/usr/bin/env -S uv run
"""Auto-detect N physical cores on one NUMA node/socket.

Port of duct/job_timing.py's detect_socket_cpus/_parse_cpulist: reads
/sys/devices/system/node/node*/cpulist for each node's cpu set, dedupes
hyperthread siblings within it by keeping only the first cpu seen per unique
.../cpu<N>/topology/core_id, and picks whichever node has the most available
physical cores (ties broken by node order).

Exists because bench/run_prod_baseline.sh and bench/run_all_arms.sh both
defaulted to "rank r -> cpu r", which is only correct by accident on a
homogeneous, non-hybrid machine where cpu ids 0..N-1 happen to be N distinct
physical cores of one socket (true of the Haswell workstation this harness's
own numbers were measured on, per bench/README.md's "8-rank socket-contended"
regime, but not guaranteed elsewhere, and silently wrong -- not an error -- on
a hybrid part or a machine where core 0..7 spans two sockets or includes SMT
siblings).

Usage:
    uv run python bench/socket_cpus.py --n 8
    # -> space-separated cpu ids on stdout, one socket, N distinct physical
    #    cores, intersected with this process's own affinity mask.
"""

import argparse
import os
import sys
from pathlib import Path


def _parse_cpulist(text):
    """Expand a Linux /sys cpulist string ("0-3,8,10-11") into a list of ints."""
    cpus = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def detect_socket_cpus(n_needed):
    """N cpu ids, all distinct physical cores on one NUMA node/socket,
    intersected with this process's own affinity mask. See module docstring.
    """
    avail = os.sched_getaffinity(0)
    node_root = Path("/sys/devices/system/node")
    node_dirs = sorted(node_root.glob("node[0-9]*")) if node_root.is_dir() else []
    best = None
    for node_dir in node_dirs:
        cpulist_path = node_dir / "cpulist"
        if not cpulist_path.exists():
            continue
        node_cpus = [c for c in _parse_cpulist(cpulist_path.read_text()) if c in avail]
        seen_cores = set()
        phys = []
        for c in sorted(node_cpus):
            core_id_path = Path(f"/sys/devices/system/cpu/cpu{c}/topology/core_id")
            try:
                core_id = core_id_path.read_text().strip()
            except OSError:
                core_id = str(c)  # fallback: no topology info, treat as its own core
            if core_id in seen_cores:
                continue
            seen_cores.add(core_id)
            phys.append(c)
        if len(phys) >= n_needed and (best is None or len(phys) > len(best)):
            best = phys
    if best is None:
        sys.exit(
            f"could not find a NUMA node/socket with >= {n_needed} available "
            f"physical cores under {node_root} -- pass CPUS explicitly if this "
            "machine's topology doesn't expose /sys/devices/system/node the "
            "way this function expects"
        )
    return sorted(best)[:n_needed]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8, help="physical cores to select")
    args = ap.parse_args()
    print(" ".join(str(c) for c in detect_socket_cpus(args.n)))


if __name__ == "__main__":
    main()
