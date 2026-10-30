"""Tests that Solver.n_step_dt holds the time step between updates.

The time step pass, :meth:`ember.grid.Grid.update_timestep`, also adds the
polar source and advances the SFD filter, so skipping it must not skip those:
a source refresh step off the cadence adds the polar source alone, and SFD is
refused.
These tests pin the cadence rather than the arithmetic.

Test cases:
- test_default_updates_on_the_cadence: the default is 5
- test_one_updates_every_step: n_step_dt=1 runs the pass on every step
- test_scree_updates_on_the_cadence: on the scree march, n_step_dt=5 runs the
  pass on the source cadence only, and dt_vol_nd is bitwise held between
- test_rk_adds_polar_source_off_the_cadence: the RK march refreshes the
  sources every step, so with n_step_dt=5 the full pass runs on the cadence and
  the polar source alone on the other steps, with dt_vol_nd held
- test_rk_one_never_adds_polar_source_alone: n_step_dt=1 always takes the
  full pass
- test_rejects_bad_values: n_step_dt < 1, and n_step_dt > 1 with SFD on
"""

import numpy as np
import pytest

import ember.grid
import ember.solver
from ember.cases import build_duct_grid

NCELL = 150000


def _conf(**kwargs):
    settings = dict(
        n_step=11,
        n_step_log=11,
        cfl=0.4,
        n_stage=0,
        n_levels=0,
        fac_mgrid=0.0,
        inviscid=True,
    )
    settings.update(kwargs)
    return ember.solver.Solver(**settings)


def _record(monkeypatch):
    """Record dt_vol_nd at each update_residual call, and count timestep calls.

    update_residual runs once per step, after the time step pass, so its
    snapshots are the dt_vol_nd each step marches on.
    """
    n_timestep = []
    snaps = []
    orig_timestep = ember.grid.Grid.update_timestep
    orig_residual = ember.grid.Grid.update_residual

    def timestep_spy(self, *args, **kwargs):
        n_timestep.append(None)
        return orig_timestep(self, *args, **kwargs)

    def residual_spy(self, *args, **kwargs):
        snaps.append(np.array(self[0].dt_vol_nd))
        return orig_residual(self, *args, **kwargs)

    monkeypatch.setattr(ember.grid.Grid, "update_timestep", timestep_spy)
    monkeypatch.setattr(ember.grid.Grid, "update_residual", residual_spy)
    return n_timestep, snaps


def test_default_updates_on_the_cadence():
    assert _conf().n_step_dt == 5


def test_one_updates_every_step(monkeypatch):
    n_timestep, _ = _record(monkeypatch)
    conf = _conf(n_step_dt=1)
    hist = conf.run(build_duct_grid(NCELL))
    assert not hist.diverged
    # One seeding call before the march, then one per step.
    assert len(n_timestep) == 1 + conf.n_step


def test_scree_updates_on_the_cadence(monkeypatch):
    n_timestep, snaps = _record(monkeypatch)
    conf = _conf(n_step_dt=5)
    hist = conf.run(build_duct_grid(NCELL))
    assert not hist.diverged

    # Steps 0, 5 and 10, plus the seeding call.
    assert len(n_timestep) == 1 + 3
    assert len(snaps) == conf.n_step
    for i_step in range(conf.n_step):
        np.testing.assert_array_equal(snaps[i_step], snaps[i_step - i_step % 5])
    # And an update does change it: the flow has moved in five steps.
    assert not np.array_equal(snaps[5], snaps[0])


def test_rk_adds_polar_source_off_the_cadence(monkeypatch):
    n_timestep, snaps = _record(monkeypatch)
    n_polar = []
    orig_polar = ember.grid.Grid.add_polar_source

    def polar_spy(self, *args, **kwargs):
        n_polar.append(None)
        return orig_polar(self, *args, **kwargs)

    monkeypatch.setattr(ember.grid.Grid, "add_polar_source", polar_spy)

    conf = _conf(n_stage=4, cfl=2.0, n_step_dt=5)
    hist = conf.run(build_duct_grid(NCELL))
    assert not hist.diverged

    # The RK march refreshes the sources every step: the full pass on steps 0,
    # 5 and 10, and the polar source alone on the other eight.
    assert len(n_timestep) == 1 + 3
    assert len(n_polar) == conf.n_step - 3
    for i_step in range(conf.n_step):
        np.testing.assert_array_equal(snaps[i_step], snaps[i_step - i_step % 5])


def test_rk_one_never_adds_polar_source_alone(monkeypatch):
    def fail(self, *args, **kwargs):
        raise AssertionError("add_polar_source called at n_step_dt=1")

    monkeypatch.setattr(ember.grid.Grid, "add_polar_source", fail)
    hist = _conf(n_stage=4, cfl=2.0, n_step_dt=1).run(build_duct_grid(NCELL))
    assert not hist.diverged


@pytest.mark.parametrize(
    "kwargs", [dict(n_step_dt=0), dict(n_step_dt=5, gain_filt=0.5)]
)
def test_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        _conf(**kwargs)
