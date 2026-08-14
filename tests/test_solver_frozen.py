"""Tests that a Solver is a value rather than a thing with state.

A solver holds the parameters of a march, not the march itself: the working
state lives on the grid, and nothing writes back to the configuration once it
has been built. Freezing says so, and makes the settings safe to share between
runs, to hash, and to hand to a caller who should not be able to retune them.

Test cases:
- test_a_solver_cannot_be_mutated: the fields refuse assignment
- test_replace_makes_a_new_solver: how a changed setting is expressed instead
- test_run_fmg_derives_its_levels_by_replacement: the idiom was already in use
- test_solvers_compare_by_value: what freezing gives beyond immutability
- test_a_solver_is_hashable: so one can key a cache or sit in a set
- test_running_does_not_change_the_solver: the march writes to the grid only
"""

import dataclasses

import pytest

import ember.solver
from ember.cases import build_duct_grid


def test_a_solver_cannot_be_mutated():
    solver = ember.solver.Solver(n_step=10)

    with pytest.raises(dataclasses.FrozenInstanceError):
        solver.n_step = 20

    with pytest.raises(dataclasses.FrozenInstanceError):
        solver.cfl = 0.9


def test_replace_makes_a_new_solver():
    """Retuning is expressed by building another one, not by editing this one."""
    solver = ember.solver.Solver(n_step=10, cfl=0.4)

    coarser = dataclasses.replace(solver, n_levels=2)

    assert coarser is not solver
    assert coarser.n_levels == 2
    assert solver.n_levels == 0
    assert coarser.cfl == solver.cfl


def test_run_fmg_derives_its_levels_by_replacement():
    """The idiom freezing formalises was already the one in use.

    `_run_fmg` builds each level's configuration with `replace(conf,
    n_levels=i)` rather than assigning to the one it was given, so no caller
    ever sees a solver whose settings changed under it.
    """
    import inspect  # noqa: PLC0415

    source = inspect.getsource(ember.solver._run_fmg)

    assert "replace(conf" in source


def test_solvers_compare_by_value():
    """Two solvers with the same settings are the same configuration."""
    assert ember.solver.Solver(n_step=10) == ember.solver.Solver(n_step=10)
    assert ember.solver.Solver(n_step=10) != ember.solver.Solver(n_step=20)


def test_a_solver_is_hashable():
    """Frozen buys this, and it is what lets one key a cache or join a set."""
    a = ember.solver.Solver(n_step=10)
    b = ember.solver.Solver(n_step=10)

    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_running_does_not_change_the_solver():
    """A march writes to the grid; the configuration it was given is untouched."""
    grid = build_duct_grid(30_000, nj=33, nk=29)
    solver = ember.solver.Solver(n_step=2, n_step_log=2, n_step_avg=1, n_stage=4)
    before = dataclasses.asdict(solver)

    solver.run(grid)

    assert dataclasses.asdict(solver) == before
