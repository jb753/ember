#!/usr/bin/env python3
"""Re-run the 12 duct_cfl_descend cases at 90% of their max stable CFL, one at
a time, to get accurate (contention-free) timing and save each convergence
history.

Reads the per-case summary lines from a duct_cfl_descend results file, backs
each case off to 0.9 * max_stable_CFL, and drives scripts/run_duct.py serially
with the SAME per-scheme run length / log cadence the sweep used (scree
n_stage=0: n_step=5000 n_step_log=20; RK4 n_stage=4: n_step=2000 n_step_log=10)
and n_levels=2. Each run writes a .cnv convergence history to the output dir
(read back with ember.solver.ConvergenceHistory.read_cnv), and the script
tabulates the reported wall time per case.

Running serially matters: the descend sweep ran all 12 cases at once, so its
wall times share the CPU. Here each case has the machine to itself
(OMP_NUM_THREADS defaults to 1, so it is single-threaded either way), giving
comparable per-case timing.

Usage:
  scripts/duct_cfl_timing.py [RESULTS_FILE] [--out-dir DIR] [--frac F]
                             [--dry-run] [-- EXTRA_ARGS...]

RESULTS_FILE defaults to duct_cfl_descend_results.txt. --frac overrides the 0.9
CFL back-off. EXTRA_ARGS after -- are forwarded verbatim to run_duct.py.
"""
import argparse
import os
import re
import subprocess
import sys
import time

# Matches the per-case summary lines from duct_cfl_descend.sh, e.g.
#   n_stage=0 sf_resid=0.0 fac_mgrid=0.0: max converging CFL ~= .578125, ...
LINE = re.compile(
    r"n_stage=(?P<n_stage>\S+)\s+sf_resid=(?P<sf_resid>\S+)\s+"
    r"fac_mgrid=(?P<fac_mgrid>\S+):\s+max converging CFL ~=\s*(?P<cfl>\S+),"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = LINE.search(line)
            if m:
                d = m.groupdict()
                rows.append({
                    "n_stage": d["n_stage"],
                    "sf_resid": d["sf_resid"],
                    "fac_mgrid": d["fac_mgrid"],
                    "cfl": float(d["cfl"]),
                })
    return rows


def schedule(n_stage):
    """Per-scheme (n_step, n_step_log), matching duct_cfl_descend.sh."""
    if int(n_stage) == 0:
        return 5000, 20
    return 2000, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?",
                    default="duct_cfl_descend_results.txt")
    ap.add_argument("--out-dir", default="duct_cfl_timing_cnv",
                    help="Directory for the per-case .cnv histories")
    ap.add_argument("--frac", type=float, default=0.9,
                    help="Fraction of max stable CFL to run at (default 0.9)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the run_duct.py commands without running them")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="Args after -- are forwarded to run_duct.py")
    args = ap.parse_args()

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    rows = parse(args.results)
    if not rows:
        sys.exit(f"no result lines found in {args.results}")

    os.makedirs(args.out_dir, exist_ok=True)
    run_duct = os.path.join(SCRIPT_DIR, "run_duct.py")

    results = []  # (row, cfl90, cnv_path, wall, rc, timing_line)
    for r in rows:
        n_step, n_step_log = schedule(r["n_stage"])
        cfl90 = args.frac * r["cfl"]
        tag = (f"s{r['n_stage']}_sf{r['sf_resid']}_fm{r['fac_mgrid']}")
        cnv_path = os.path.join(args.out_dir, f"{tag}.cnv")
        cmd = [
            "uv", "run", run_duct,
            "--cfl", f"{cfl90:.6f}",
            "--fac-mgrid", r["fac_mgrid"],
            "--n-stage", r["n_stage"],
            "--sf-resid", r["sf_resid"],
            "--n-levels", "2",
            "--n-step", str(n_step),
            "--n-step-log", str(n_step_log),
            "--write-hist", cnv_path,
            *extra,
        ]
        header = (f"=== {tag}: CFL={cfl90:.6f} "
                  f"({args.frac:g} x {r['cfl']}) n_step={n_step} ===")
        print(header, flush=True)
        if args.dry_run:
            print("  " + " ".join(cmd), flush=True)
            continue

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        wall = time.perf_counter() - t0
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
        # run_duct's own reported wall line, e.g. "12.345s  0.678 us/node/step".
        m = re.search(r"^([0-9.]+)s\s+[0-9.]+ us/node/step", proc.stdout,
                      re.M)
        run_wall = float(m.group(1)) if m else float("nan")
        results.append((r, cfl90, cnv_path, run_wall, proc.returncode))

    if args.dry_run:
        return

    print()
    print("Timing at {:g}x max stable CFL (serial, one at a time):"
          .format(args.frac))
    hdr = ("n_stage", "sf_resid", "fac_mgrid", "CFL", "wall_s", "rc", "cnv")
    print("  {:<8} {:<9} {:<10} {:>9} {:>8} {:>4}  {}".format(*hdr))
    for r, cfl90, cnv_path, run_wall, rc in results:
        print("  {:<8} {:<9} {:<10} {:>9.5f} {:>8.2f} {:>4}  {}".format(
            r["n_stage"], r["sf_resid"], r["fac_mgrid"], cfl90, run_wall,
            rc, cnv_path))
    # run_duct exits 0 only when it converged; flag any that did not.
    bad = [r for (r, *_x, rc) in results if rc != 0]
    if bad:
        print(f"\nWARNING: {len(bad)} case(s) did not converge at "
              f"{args.frac:g}x CFL (rc != 0).")


if __name__ == "__main__":
    main()
