"""Tests for the pseudotime averaging window and the states it must refuse.

``Solver.n_step_avg`` names the steps at the end of a march that
``Grid.accumulate_avg`` sums into the average, each divided by the window
length. Two values used to be accepted and silently ruin the solution: a
window of 0, where the accumulator never ran but ``finalise_average`` still
copied its zeros over ``conserved_nd``, and a window longer than the march,
where fewer samples were added than the divisor accounts for and the field
came back scaled by ``n_step / n_step_avg``. Neither is detectable downstream
--- zeros and a scaled field are as finite as the real thing, so the
divergence check and the convergence history both pass them.

Test cases:
- test_a_negative_window_is_refused: no meaning, caught at construction
- test_a_window_longer_than_the_march_is_refused: the silent scaling
- test_a_window_of_the_whole_march_is_allowed: the boundary is inclusive
- test_no_averaging_leaves_the_final_state: a window of 0 or 1 is not destructive
- test_one_sample_average_matches_no_average: skipping is exactly equivalent
- test_no_averaging_allocates_no_accumulator: what the skip is worth
- test_the_conserved_caches_are_fresh_after_a_scree_march: the invalidation
  finalise_average used to cover for
"""

import numpy as np
import pytest

import ember.solver
from ember.cases import build_duct_grid

NCELL = 150000
N_STEP = 4


def _grid():
    return build_duct_grid(NCELL)


def _conf(**kwargs):
    settings = dict(
        n_step=N_STEP,
        n_step_log=N_STEP,
        cfl=2.0,
        n_stage=4,
        n_levels=0,
        fac_mgrid=0.0,
    )
    settings.update(kwargs)
    return ember.solver.Solver(**settings)


def test_a_negative_window_is_refused():
    with pytest.raises(ValueError, match="n_step_avg must be >= 0"):
        ember.solver.Solver(n_step=10, n_step_avg=-1)


def test_a_window_longer_than_the_march_is_refused():
    """The march can only add n_step samples, but the divisor is n_step_avg."""
    with pytest.raises(ValueError, match="exceeds n_step"):
        ember.solver.Solver(n_step=5, n_step_avg=6)


def test_a_window_of_the_whole_march_is_allowed():
    """Averaging every step of the run is a legitimate request, not the error."""
    assert ember.solver.Solver(n_step=5, n_step_avg=5).n_step_avg == 5


@pytest.mark.parametrize("n_step_avg", [0, 1])
def test_no_averaging_leaves_the_final_state(n_step_avg):
    """A window of 0 or 1 must leave a real solution, not an accumulator's zeros."""
    grid = _grid()
    hist = _conf(n_step_avg=n_step_avg).run(grid)

    assert not hist.diverged
    cons = np.asarray(grid[0].conserved_nd)
    assert np.all(np.isfinite(cons))
    assert np.max(np.abs(cons)) > 0.0


def test_one_sample_average_matches_no_average():
    """Skipping the accumulator at n_step_avg=1 is exactly what it replaced.

    Averaging one sample divides it by one and copies it back, so the two
    paths must agree bit for bit, not merely closely.
    """
    grid_one = _grid()
    _conf(n_step_avg=1).run(grid_one)

    grid_zero = _grid()
    _conf(n_step_avg=0).run(grid_zero)

    np.testing.assert_array_equal(
        np.asarray(grid_one[0].conserved_nd), np.asarray(grid_zero[0].conserved_nd)
    )


@pytest.mark.parametrize("n_step_avg", [0, 1])
def test_no_averaging_allocates_no_accumulator(n_step_avg):
    """The point of the skip: the nodal average buffer is never materialised."""
    grid = _grid()
    _conf(n_step_avg=n_step_avg).run(grid)

    assert "conserved_avg_nd" not in grid[0]._store


def test_the_averaged_window_still_averages():
    """Guard the gate from the other side: a real window must still run."""
    grid = _grid()
    _conf(n_step_avg=2).run(grid)

    assert "conserved_avg_nd" in grid[0]._store


@pytest.mark.parametrize("n_stage", [0, 4])
def test_the_conserved_caches_are_fresh_after_a_scree_march(n_stage):
    """Reads after a march must see the marched state, whatever the integrator.

    The integrators and ``smooth`` write ``conserved_nd`` through the
    frozen-pressure path, which does not bump the conserved versions.
    ``finalise_average`` ended in ``update_cached_conserved`` and so hid that;
    with the averaging skipped, the invalidation has to happen on its own or a
    scree march hands back a pressure from a step ago. The RK path bumps its
    versions incidentally, in its per-stage ``apply_bconds``, and is here to
    keep that difference honest rather than because it was ever at risk.
    """
    grid = _grid()
    _conf(n_stage=n_stage, n_step_avg=0).run(grid)
    block = grid[0]

    P_read = np.array(block.P_nd, copy=True)
    block.update_cached_conserved()  # force a recompute from conserved_nd
    P_fresh = np.array(block.P_nd, copy=True)

    np.testing.assert_array_equal(P_read, P_fresh)
