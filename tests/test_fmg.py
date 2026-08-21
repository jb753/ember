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

import ember.block

from ember.cases import build_duct_grid
import ember.solver

logging.disable(logging.CRITICAL)  # silence per-step convergence logging

NCELL = 120_000  # smallest that satisfies build_duct_grid's ni >= 25 floor

# Halved cross-stream resolution for the solver-heavy tests below: cuts the
# cell count roughly 4x while keeping ni, and the nj-1/nk-1 divisibility by
# 2**n_levels, the same as the full-resolution grid.
NJ_SMALL, NK_SMALL = (65 + 1) // 2, (57 + 1) // 2
NCELL_SMALL = 30_000


def _conf(n_levels, n_step=50):
    return ember.solver.Solver(
        n_step=n_step,
        n_step_log=50,
        n_step_avg=1,
        cfl=0.4,
        n_stage=0,
        n_levels=n_levels,
    )


def test_n_levels_zero_matches_run():
    """n_levels == 0 is a passthrough to run() on the given grid."""
    grid_fmg = build_duct_grid(NCELL_SMALL, nj=NJ_SMALL, nk=NK_SMALL)
    hists = _conf(0).run_fmg(grid_fmg)
    assert len(hists) == 1

    grid_run = build_duct_grid(NCELL_SMALL, nj=NJ_SMALL, nk=NK_SMALL)
    _conf(0).run(grid_run)

    # Identical seeds and identical march -> conserved state matches up to
    # float error. NOTE: passes in isolation and as this file alone, but
    # has been observed to diverge by O(0.1) relative (not just rounding)
    # when run as part of the full `ember/tests/` suite -- looks like
    # order-dependent state leakage between test files rather than a
    # steady precision issue in this test itself. Needs follow-up; not
    # yet understood. (Also see test_perturbation.py::
    # test_chic_to_bcond_linearization, which fails only in the full-suite
    # run too, on an unrelated exact-zero comparison.)
    np.testing.assert_allclose(
        grid_fmg[0].conserved, grid_run[0].conserved, rtol=1e-5, atol=1e-8
    )


def test_non_divisible_finest_raises():
    """Finest (n-1) not divisible by 2**n_levels is rejected before marching."""
    grid = build_duct_grid(NCELL, nk=53)  # nk-1 == 52, not a multiple of 8
    with pytest.raises(ValueError, match="multiple"):
        _conf(3).run_fmg(grid)


def test_n_levels_above_the_arena_maximum_raises():
    """n_levels beyond what Block.scratch is sized for is refused up front.

    The scratch arena is one allocation sized for MAX_MG_LEVELS coarse levels
    (ember.block._scratch_len), so a deeper hierarchy would carve past its end
    -- silently, since carve_view only sees the buffer it is handed. Refused
    alongside the divisibility rule, and before it, because "4 is never
    allowed" is more useful than "resize your grid".
    """
    grid = build_duct_grid(NCELL)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        _conf(ember.block.MAX_MG_LEVELS + 1).run_fmg(grid)


def test_hierarchy_shapes_and_history_length():
    """Chain is coarsest-first, halves each level, and keeps the finest shape."""
    grid = build_duct_grid(NCELL_SMALL, nj=NJ_SMALL, nk=NK_SMALL)
    finest_shape = grid[0].shape  # (ni, nj, nk)

    hists = _conf(2).run_fmg(grid)

    assert len(hists) == 3  # n_levels + 1
    assert not any(h.diverged for h in hists)
    # Finest grid object is mutated in place and keeps its resolution.
    assert grid[0].shape == finest_shape

    # Coarsest holds N/4 + 1 nodes per dim, matching the coincident-subset rule.
    expected_coarsest = tuple((n - 1) // 4 + 1 for n in finest_shape)
    # Rebuild the chain deterministically to check the coarsening geometry.
    ref = build_duct_grid(NCELL_SMALL, nj=NJ_SMALL, nk=NK_SMALL)
    coarser = ref.resample(0.5).resample(0.5)
    assert coarser[0].shape == expected_coarsest


def test_fmg_starts_finest_below_cold():
    """The finest level begins from a much lower residual than a cold run."""
    grid_fmg = build_duct_grid(NCELL_SMALL, nj=NJ_SMALL, nk=NK_SMALL)
    hists = _conf(2).run_fmg(grid_fmg)

    grid_cold = build_duct_grid(NCELL_SMALL, nj=NJ_SMALL, nk=NK_SMALL)
    cold = _conf(2).run(grid_cold)

    # Energy residual (column 4) at the first recorded finest-level step.
    fmg_start = hists[-1].residual[0, 4]
    cold_start = cold.residual[0, 4]
    assert fmg_start < 0.5 * cold_start
