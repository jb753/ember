"""Tests for the soft-start configuration transform.

`BaseSolver.soft` returns a detuned copy of a solver, robust enough to march
from an initial guess the production settings would not survive. It is a
configuration transform and nothing more: it takes no grid, marches nothing,
and leaves the solver it was called on alone.

Test cases:
- test_soft_is_compulsory_on_the_interface: a solver that skips it cannot exist
- test_soft_detunes_the_aggressive_settings: what a soft march backs away from
- test_soft_is_the_same_configuration_whatever_it_started_from: absolute, not
  scaled -- a soft start is one known-robust setting, not a relative detuning
- test_soft_leaves_the_model_choices_alone: what it must not change, and why
- test_soft_does_not_change_the_original: replace(), not assignment
- test_soft_is_a_usable_configuration: __post_init__ accepts what it builds
- test_soft_then_run_marches_in_place: the two-call schedule works end to end
"""

import dataclasses

import pytest

import ember.solver
from ember.cases import build_duct_grid


def test_soft_is_compulsory_on_the_interface():
    """A BaseSolver without soft() is abstract, so it cannot be instantiated.

    This is the point of the method being abstract rather than defaulted: a
    plugin author is told at construction, not at the call that needed it.
    """

    class NoSoft(ember.solver.BaseSolver):
        def run(self, grid):
            return None

    with pytest.raises(TypeError, match="soft"):
        NoSoft()


def test_soft_detunes_the_aggressive_settings():
    solver = ember.solver.Solver(n_step=100)
    soft = solver.soft()

    # The CFL limit is what an initial guess violates first, and smoothing is
    # the dissipation that stops a startup transient steepening faster than the
    # march can carry it.
    assert soft.cfl < solver.cfl
    assert soft.sf2 > solver.sf2
    assert soft.sf4 > solver.sf4
    # fac_mgrid is the multigrid off-switch both integrators honour, so zeroing
    # it disables the coarse correction whatever n_levels says: the coarse
    # correction is an accelerator built on a residual that means something.
    assert soft.fac_mgrid == 0.0
    # The increment limiter is ON for a soft start, where the default march
    # leaves it off.
    assert soft.dampin > 0.0


def test_soft_leaves_the_model_choices_alone():
    """A soft run must seed the production run with the same problem's answer.

    n_levels included: it names the grid hierarchy for run_fmg and sets what
    _validate_mg demands of the block shapes, neither of which is aggression.
    """
    solver = ember.solver.Solver(
        n_step=100,
        inviscid=True,
        gain_filt=0.5,
        mix_reflective=True,
        n_levels=2,
    )
    soft = solver.soft()

    assert soft.inviscid == solver.inviscid
    assert soft.gain_filt == solver.gain_filt
    assert soft.mix_reflective == solver.mix_reflective
    assert soft.n_levels == solver.n_levels


def test_soft_is_the_same_configuration_whatever_it_started_from():
    """The detuned settings are absolute, not a factor on what came in.

    A soft start is one known-robust configuration rather than a relative
    step down, so it does not matter how aggressive the production solver was
    and softening twice is softening once. The march length is part of that
    fixed schedule: a soft start runs for as long as it takes to produce a
    usable guess, not for a fraction of the run it is seeding.
    """
    aggressive = ember.solver.Solver(n_step=100, cfl=8.0)
    timid = ember.solver.Solver(n_step=100, cfl=0.5)

    assert aggressive.soft().cfl == timid.soft().cfl
    assert aggressive.soft().soft() == aggressive.soft()


def test_soft_does_not_change_the_original():
    solver = ember.solver.Solver(n_step=100)
    before = dataclasses.asdict(solver)

    soft = solver.soft()

    assert soft is not solver
    assert dataclasses.asdict(solver) == before


def test_soft_is_a_usable_configuration():
    """__post_init__ runs on the copy, so a soft solver cannot be invalid.

    The field that would break it is n_step_avg: an averaging window longer
    than the march trips the check that stops a solution coming back scaled by
    n_step / n_step_avg.
    """
    solver = ember.solver.Solver(n_step=10, n_step_avg=10)

    soft = solver.soft()

    assert soft.n_step >= 1
    assert soft.n_step_avg <= soft.n_step


def test_soft_then_run_marches_in_place():
    """The schedule the method exists for, on a real grid."""
    grid = build_duct_grid(30_000, nj=33, nk=29)
    solver = ember.solver.Solver(
        n_step=2, n_step_log=2, n_step_avg=1, n_stage=4, n_levels=0
    )

    soft_hist = solver.soft().run(grid)
    hist = solver.run(grid)

    assert soft_hist.i_step.size > 0
    assert hist.i_step.size > 0
    assert not hist.diverged
