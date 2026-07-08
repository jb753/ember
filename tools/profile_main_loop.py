"""Profile the main loop using a sector case for 100 steps.

Run this script with:
    uv run python -m line_profiler -o profile_output.lprof scripts/profile_main_loop.py

Or to view results directly:
    uv run kernprof -l -v scripts/profile_main_loop.py
"""

import ember.config
import ember.cases

# Generate sector case with no skew
case = ember.cases.SectorCase(
    ni=81,
    nj=72,
    nk=65,
    Po1=1e5,
    To1=300.0,
    Ma=0.3,
    skew=0.0,
)

case.grid.calculate_wdist()

# Run solver for 50 steps (n_levels=1 to disable multigrid for this grid size)
config = ember.config.SolverConfig(
    n_step=50, n_step_avg=1, order=3, n_levels=1, sf4=0.1
)

# Run the solver - profiling handled by @profile decorators and kernprof
case.run(config)

# Print summary
conv = case.conv
print(f"\n{'=' * 80}")
print(f"Sector case (81x72x65, no skew) - Results after {config.n_step} steps:")
print(f"  Performance: {conv.tpnps:.2f} microseconds per node per step")
print(f"{'=' * 80}")
