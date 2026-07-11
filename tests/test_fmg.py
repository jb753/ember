"""Tests for ``run_fmg``, the full-multigrid (mesh-sequencing) startup loop.

Contract:

- ``n_levels == 0`` reproduces ``run`` exactly (single-element history, grid
  marched identically).
- A finest grid whose ``(n - 1)`` is not divisible by ``2**n_levels`` raises
  through the reused ``_validate_mg`` guard.
- ``n_levels == k`` builds ``k`` halved grids, returns ``k + 1`` histories
  coarsest-first, preserves the finest grid's shape, and starts the finest
  level from a markedly lower residual than a cold ``run`` -- the point of the
  feature.
"""

import logging

import numpy as np
import pytest

from ember.cases import build_duct_grid
import ember.solver

logging.disable(logging.CRITICAL)  # silence per-step convergence logging

NCELL = 120_000  # smallest that satisfies build_duct_grid's ni >= 25 floor


def _conf(n_levels, n_step=100):
    return ember.solver.SolverConfig(
        n_step=n_step,
        n_step_log=50,
        n_step_avg=1,
        cfl=0.4,
        n_stage=0,
        n_levels=n_levels,
    )


def test_n_levels_zero_matches_run():
    """n_levels == 0 is a passthrough to run() on the given grid."""
    grid_fmg = build_duct_grid(NCELL)
    hists = ember.solver.run_fmg(grid_fmg, _conf(0))
    assert len(hists) == 1

    grid_run = build_duct_grid(NCELL)
    ember.solver.run(grid_run, _conf(0))

    # Identical seeds and identical march -> bit-for-bit conserved state.
    np.testing.assert_array_equal(grid_fmg[0].conserved, grid_run[0].conserved)


def test_non_divisible_finest_raises():
    """Finest (n-1) not divisible by 2**n_levels is rejected before marching."""
    grid = build_duct_grid(NCELL)  # nk-1 == 56, not a multiple of 16
    with pytest.raises(ValueError, match="multiple"):
        ember.solver.run_fmg(grid, _conf(4))


def test_hierarchy_shapes_and_history_length():
    """Chain is coarsest-first, halves each level, and keeps the finest shape."""
    grid = build_duct_grid(NCELL)
    finest_shape = grid[0].shape  # (ni, nj, nk)

    hists = ember.solver.run_fmg(grid, _conf(2))

    assert len(hists) == 3  # n_levels + 1
    assert not any(h.diverged for h in hists)
    # Finest grid object is mutated in place and keeps its resolution.
    assert grid[0].shape == finest_shape

    # Coarsest holds N/4 + 1 nodes per dim, matching the coincident-subset rule.
    expected_coarsest = tuple((n - 1) // 4 + 1 for n in finest_shape)
    # Rebuild the chain deterministically to check the coarsening geometry.
    ref = build_duct_grid(NCELL)
    coarser = ref.resample(0.5).resample(0.5)
    assert coarser[0].shape == expected_coarsest


def test_fmg_starts_finest_below_cold():
    """The finest level begins from a much lower residual than a cold run."""
    grid_fmg = build_duct_grid(NCELL)
    hists = ember.solver.run_fmg(grid_fmg, _conf(2))

    grid_cold = build_duct_grid(NCELL)
    cold = ember.solver.run(grid_cold, _conf(2))

    # Energy residual (column 4) at the first recorded finest-level step.
    fmg_start = hists[-1].residual[0, 4]
    cold_start = cold.residual[0, 4]
    assert fmg_start < 0.5 * cold_start
