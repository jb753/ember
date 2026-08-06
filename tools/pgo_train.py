#!/usr/bin/env -S uv run
"""Training run for the PGO build: exercise the solver, write .gcda.

Run this once between the EMBER_PGO=generate and EMBER_PGO=use builds.

WHAT IT TRAINS ON, AND WHY
--------------------------
Real `Solver.run` steps, not the benchmark loop and not the test suite. PGO
consumes branch probabilities and loop trip counts, so the training workload
has to have the ones production has:

  - the test suite is the obvious temptation and the wrong answer -- its grids
    are tiny, so it would train set_residual's face loops for trip counts an
    order of magnitude below production's;
  - the benchmark loop calls one kernel and would leave every other routine
    (viscous, IRS, multigrid, bconds) with no profile at all, which under
    -fprofile-use means GCC treats them as cold and optimises them for size;
  - a real march exercises all of it, in the proportions production sees.

SEVERAL SHAPES, DELIBERATELY. Trip counts and the jf==1 / jf==nj boundary
branches are shape-dependent, so training on one grid risks peeling and
unrolling tuned to that grid's ni. Three shapes with different aspect ratios
are marched, and the *evaluation* case later is a different size again -- if
the win only shows up on a trained shape, it is over-fit and does not count.

SINGLE PROCESS. Concurrent ranks writing the same .gcda risks corrupting it;
the build uses -fprofile-update=single to keep the counters cheap. PGO wants
branch and trip counts, which contention does not change, so there is nothing
to gain from training under load.
"""

import argparse
import sys
import time

sys.path.insert(0, "tools")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument(
        "--ncells",
        default="300000,1000000,2000000",
        help="comma-separated grid sizes to train on",
    )
    args = ap.parse_args()

    from ember.cases import build_duct_grid
    from ember.solver import Solver

    for ncell in (int(n) for n in args.ncells.split(",")):
        t0 = time.perf_counter()
        grid = build_duct_grid(ncell)
        # dampin and sf_resid non-zero so the change limiter and the IRS
        # solves are exercised too -- under -fprofile-use an unexercised
        # branch is treated as cold, so anything production runs must run here.
        solver = Solver(n_step=args.steps, dampin=25.0, sf_resid=0.5, n_step_log=10**9)
        solver.run(grid)
        b = grid[0]
        print(
            f"  trained ncell={ncell:>8}  shape={tuple(b.shape)}  "
            f"{args.steps} steps in {time.perf_counter() - t0:5.1f}s",
            flush=True,
        )
    print("training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
