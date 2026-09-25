"""Tests that selective frequency damping is wired into the march.

``tests/test_update_filter.py`` covers the filter kernel in isolation and
``tests/test_set_F_body_golden.py`` covers the body force it feeds. Neither
covers the join: for a long time ``Solver.run`` never advanced the filter at
all (the call sat commented out, with a signature that predated the
``adapt_cfl`` split), so ``conserved_filt_nd`` stayed at the seed it takes on
first access. The update now rides in :meth:`ember.grid.Grid.update_timestep`,
which makes the join easier to get wrong in a new way: the timestep runs every
step, the body force only on the source cadence, and the two gates must not be
confused. A nonzero
:attr:`~ember.solver.Solver.gain_filt` then did not low-pass anything -- it
pulled the solution back towards its own initial condition for the whole run.
These tests pin the wiring rather than the arithmetic.

Test cases:
- test_the_filter_advances_over_a_march: the filter state actually moves, and
  moves towards the flow, rather than sitting at its seed
- test_the_filter_advances_every_step: once per step, not on the scree march's
  every-fifth-step source cadence, which would stretch delta_filt fivefold
- test_no_filtering_never_touches_the_filter: the default zero gain skips the
  update and never allocates the buffer
- test_scree_body_force_is_held_between_refreshes: on the scree march the body
  force (polar + SFD) is rebuilt every fifth step and held, not re-added, in
  between
- test_the_damping_changes_the_solution: end to end, a nonzero gain reaches
  conserved_nd
"""

import numpy as np
import pytest

import ember.fortran
import ember.grid
import ember.solver
from conftest import cell_conserved
from ember.cases import build_duct_grid

NCELL = 150000
N_STEP = 4

# Filter seeded this far from the cell state, as a fraction of the RMS over
# every cell and equation. One scalar offset, not a scaling of the state and
# not a per-equation RMS: either would vanish on the duct's near-zero radial
# and swirl momenta, and it is the offset being nonzero everywhere that makes
# the direction of the filter's motion assertable.
OFFSET_FRAC = 0.1

GAIN_FILT = 0.5


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
        inviscid=True,
        gain_filt=GAIN_FILT,
    )
    settings.update(kwargs)
    return ember.solver.Solver(**settings)


def _seed_filter(block):
    """Offset the filter state from the flow, and return the seed.

    Mirrors the seeding in ``tests/test_set_F_body_golden.py``: the buffer is
    read-only to consumers, so unlock it around the write. The offset is
    positive on every component, so the filter's only way towards the flow is
    down -- which is what makes the direction of one update assertable.
    """
    cons_cell = cell_conserved(block)
    offset = np.float32(OFFSET_FRAC * np.sqrt(np.mean(cons_cell**2)))

    cons_filt = block.conserved_filt_nd
    cons_filt.flags.writeable = True
    cons_filt[...] = cons_cell + offset
    cons_filt.flags.writeable = False
    return np.array(cons_filt), offset


def _count_filter_calls(monkeypatch):
    """Count filter updates -- timestep kernel calls with the SFD gate on.

    The filter advances inside ``set_timestep_sources`` whenever it is handed
    a nonzero gain, so that is what is counted, per block (the duct is one).
    The calls are passed through, not suppressed.
    """
    calls = []
    original = ember.fortran.set_timestep_sources

    def spy(**kwargs):
        if kwargs["gain_filt"] != 0.0:
            calls.append((kwargs["cfl"], kwargs["delta_filt"]))
        return original(**kwargs)

    monkeypatch.setattr(ember.fortran, "set_timestep_sources", spy)
    return calls


def test_the_filter_advances_over_a_march():
    """The filter tracks the flow instead of staying at its seed."""
    grid = _grid()
    block = grid[0]
    seed, offset = _seed_filter(block)

    hist = _conf().run(grid)
    assert not hist.diverged

    cons_filt = np.asarray(block.conserved_filt_nd)
    assert np.all(np.isfinite(cons_filt))
    # Seeded above the flow on every component, so every cell must have
    # filtered downwards -- and by less than the offset it started with, an
    # exponential moving average being unable to pass its target.
    assert np.all(cons_filt < seed)
    assert np.all(seed - cons_filt < offset)


def test_the_filter_advances_every_step(monkeypatch):
    """Once per step on the scree march, not on its source-refresh cadence.

    ``update_sources`` runs every fifth step when ``n_stage == 0``; the filter
    increment is a per-step ``dt``, so sharing that cadence would quietly
    lengthen the time constant by a factor of five. ``n_step`` is deliberately
    not a multiple of five, so the two counts cannot coincide.
    """
    n_step = 7
    calls = _count_filter_calls(monkeypatch)

    grid = _grid()
    _seed_filter(grid[0])
    _conf(n_step=n_step, n_step_log=n_step, n_stage=0).run(grid)

    assert len(calls) == n_step


def test_no_filtering_never_touches_the_filter(monkeypatch):
    """The default zero gain costs nothing: no update, no buffer."""
    calls = _count_filter_calls(monkeypatch)

    grid = _grid()
    hist = _conf(gain_filt=0.0).run(grid)

    assert not hist.diverged
    assert calls == []
    assert "conserved_filt_nd" not in grid[0]._store


@pytest.mark.parametrize("n_stage", [0, 4])
def test_the_damping_changes_the_solution(n_stage):
    """End to end: the force built from the filter reaches conserved_nd."""
    grid_off = _grid()
    _seed_filter(grid_off[0])
    _conf(n_stage=n_stage, gain_filt=0.0).run(grid_off)

    grid_on = _grid()
    _seed_filter(grid_on[0])
    _conf(n_stage=n_stage).run(grid_on)

    cons_off = np.asarray(grid_off[0].conserved_nd)
    cons_on = np.asarray(grid_on[0].conserved_nd)
    assert np.all(np.isfinite(cons_on))
    assert not np.array_equal(cons_on, cons_off)


@pytest.mark.parametrize("gain_filt", [0.0, GAIN_FILT])
def test_scree_body_force_is_held_between_refreshes(monkeypatch, gain_filt):
    """The scree march rebuilds F_body every fifth step and holds it between.

    The sources lag on the scree march (``n_step_source = 5``), so on steps
    1-4 of each window ``F_body_nd`` must be BITWISE the one the refresh step
    built: an accumulating term (polar or SFD) added on every step rather than
    every refresh would grow it instead. Snapshots are taken at each
    ``update_residual`` call and keyed by step, the step being counted off
    ``update_timestep``, which runs once before the march and once per step.
    Zero gain leaves the polar source alone, so both are checked.
    """
    n_step = 11
    steps = []
    snaps = {}
    orig_timestep = ember.grid.Grid.update_timestep
    orig_residual = ember.grid.Grid.update_residual

    def timestep_spy(self, *args, **kwargs):
        steps.append(None)
        return orig_timestep(self, *args, **kwargs)

    def residual_spy(self, *args, **kwargs):
        snaps[len(steps) - 2] = np.array(self[0].F_body_nd)
        return orig_residual(self, *args, **kwargs)

    monkeypatch.setattr(ember.grid.Grid, "update_timestep", timestep_spy)
    monkeypatch.setattr(ember.grid.Grid, "update_residual", residual_spy)

    grid = _grid()
    _seed_filter(grid[0])
    hist = _conf(n_step=n_step, n_step_log=n_step, n_stage=0, gain_filt=gain_filt).run(
        grid
    )
    assert not hist.diverged
    assert sorted(snaps) == list(range(n_step))

    for i_step in range(n_step):
        refresh = i_step - i_step % 5
        np.testing.assert_array_equal(snaps[i_step], snaps[refresh])
    # And the refresh does rebuild it: the flow has moved in five steps.
    assert not np.array_equal(snaps[5], snaps[0])
    assert not np.array_equal(snaps[10], snaps[5])
