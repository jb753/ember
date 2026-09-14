"""How often a march pays for its expensive whole-grid operations.

Modules tested: ember.solver, ember.grid (Grid.check_nan, update_sources,
update_timestep, update_residual, smooth), ember.block (update_primitive)

Pressure and temperature are filled together, in the one fused
``Fluid.get_P_h_T`` call behind :meth:`ember.block.Block.update_primitive`, and
every whole-block consumer in the step fills the cache at its point of
consumption. So a step pays for one evaluation per conserved state it reads the
primitives of, and no more: the top of the step, where ``check_nan`` is the
first consumer and the sources, timestep and first residual are cache hits, and
each Runge-Kutta substep after the first, whose residual reads a state the
previous stage has just written.

A stray read of a primitive property anywhere else in the step -- ``block.P``
after the integrator, say -- recomputes it lazily through the fluid on the whole
block. Every fluid method called on a whole-block array is therefore counted,
and each count pinned exactly, so such a read shows up whichever quantity it
asks for.

Test cases:
- Equation of state: one fused evaluation per block before the loop and one per
  scree step or n_stage per Runge-Kutta step; one acoustic speed per block per
  timestep update; the transport properties once per source pass; nothing else
- Grid operations: n_stage residuals per Runge-Kutta step and one per scree
  step, one timestep per step plus the initial one, one smooth per step
- check_nan's evaluation is the one the rest of the step reuses
- All of the above whatever the multigrid depth
"""

from collections import Counter
from math import ceil

import numpy as np
import pytest

import ember.grid
import ember.solver
from test_solver_relaxation import make_grid

N_BLOCK = 2

# The scree march lags the viscous source pass to every fifth step; see
# ember.solver._run.
N_STEP_SOURCE_SCREE = 5

# Fluid methods a clean march calls on whole blocks, beyond the fused fill:
# the acoustic speed for the timestep, and the transport properties for the
# viscous sources.
TRANSPORT = ("get_mu", "get_kappa", "get_cp")

GRID_CALLS = ("update_residual", "update_timestep", "smooth")


@pytest.fixture
def count_eos(monkeypatch):
    """Count fluid ``get_*`` calls on whole-block arrays, by method name.

    Returns a function that takes a grid, patches its fluid class, and returns
    the live Counter. Patch-sized arrays -- the boundary conditions' surface
    averages -- are left out, being a surface's worth of work, not a volume's.
    """

    def install(grid):
        shapes = {tuple(block.shape) for block in grid}
        counts = Counter()
        fluid_cls = type(grid[0].fluid)
        names = [
            name
            for name in dir(fluid_cls)
            if name.startswith("get_") and callable(getattr(fluid_cls, name))
        ]
        for name in names:
            original = getattr(fluid_cls, name)

            def counted(self, *args, _original=original, _name=name, **kwargs):
                if args and tuple(np.shape(args[0])) in shapes:
                    counts[_name] += 1
                return _original(self, *args, **kwargs)

            monkeypatch.setattr(fluid_cls, name, counted)
        return counts

    return install


@pytest.fixture
def count_grid(monkeypatch):
    """Count calls to the grid's expensive whole-grid operations, by method name."""
    counts = Counter()
    for name in GRID_CALLS:
        original = getattr(ember.grid.Grid, name)

        def counted(self, *args, _original=original, _name=name, **kwargs):
            counts[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(ember.grid.Grid, name, counted)
    return counts


def march(n_step, n_stage, n_levels):
    """Run a short march on a fresh grid and return the grid."""
    grid = make_grid(nblock=N_BLOCK)
    conf = ember.solver.Solver(
        n_step=n_step,
        n_step_avg=1,
        n_step_log=n_step,
        n_stage=n_stage,
        n_levels=n_levels,
    )
    # Built before the counters are read, so the fixtures see the whole run.
    return grid, conf


def source_passes(n_step, n_stage):
    """How many viscous source passes a march of n_step takes."""
    if n_stage == 0:
        return ceil(n_step / N_STEP_SOURCE_SCREE)
    return n_step


# Scree runs to six steps, so a second source pass falls inside the march.
MARCHES = [
    (n_step, n_stage, n_levels)
    for n_levels in (0, 2)
    for n_stage in (0, 2, 4)
    for n_step in ((1, 3, 6) if n_stage == 0 else (1, 3))
]


@pytest.mark.parametrize("n_step, n_stage, n_levels", MARCHES)
def test_the_equation_of_state_is_called_only_where_the_step_needs_it(
    count_eos, n_step, n_stage, n_levels
):
    """Every whole-block fluid call is accounted for, so a stray lazy read shows."""
    grid, conf = march(n_step, n_stage, n_levels)
    counts = count_eos(grid)

    hist = conf.run(grid)

    assert not hist.diverged
    expected = {
        "get_P_h_T": N_BLOCK * (1 + n_step * max(n_stage, 1)),
        "get_a": N_BLOCK * (1 + n_step),
    }
    for name in TRANSPORT:
        expected[name] = N_BLOCK * source_passes(n_step, n_stage)
    assert dict(counts) == expected


@pytest.mark.parametrize("n_step, n_stage, n_levels", MARCHES)
def test_a_march_calls_residual_timestep_and_smooth_once_per_use(
    count_grid, n_step, n_stage, n_levels
):
    """n_stage residuals per RK step, one per scree step; one timestep and smooth per step."""
    grid, conf = march(n_step, n_stage, n_levels)

    hist = conf.run(grid)

    assert not hist.diverged
    assert dict(count_grid) == {
        "update_residual": n_step * max(n_stage, 1),
        # One before the loop seeds the timestep buffer.
        "update_timestep": 1 + n_step,
        "smooth": n_step,
    }


def test_check_nan_fills_the_cache_the_rest_of_the_step_reads(count_eos):
    """check_nan's evaluation is reused, not repeated, by the step's consumers."""
    grid = make_grid(nblock=N_BLOCK)
    counts = count_eos(grid)
    grid.update_cached_conserved()

    grid.check_nan()
    assert counts["get_P_h_T"] == N_BLOCK

    grid.update_sources(inviscid=False, gain_filt=0.0)
    grid.update_timestep(rf=1.0)
    grid.update_residual()
    assert counts["get_P_h_T"] == N_BLOCK
