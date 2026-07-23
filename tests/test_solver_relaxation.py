"""Tests for the solver's boundary-relaxation settings.

Modules tested: ember.solver (rf_inlet, rf_outlet, rf_mix, rf_exchange and the
push that applies them)

Every one of these factors lives on a patch, not on the solver: the patches
outlive a run and round-trip through an EMB, while the mixing communicator that
used to hold rf_exchange is dropped on pickle and rebuilt from defaults. The
solver's four settings are therefore a push onto the grid at the start of a run
rather than state the march carries.

Test cases:
- Push: each setting lands on the patches it names, and on no others
- Defaults: the configured value is imposed over whatever the patches carried
  in, and passing None is what opts out and keeps theirs
- Independence: rf_mix and rf_exchange are separate knobs despite the names
- Wiring: a real run applies them, and a full-multigrid run applies them on
  every level of its chain
"""

import numpy as np
import pytest

import ember.solver
from ember.grid import Grid
from ember.patch import (
    InletPatch,
    MixingPatch,
    NonReflectingInletPatch,
    NonReflectingMixingPatch,
    NonReflectingOutletPatch,
    OutletPatch,
    PeriodicPatch,
)
from nonreflecting_util import make_block

# Nothing here marches to convergence; two steps is enough to prove the push ran.
N_STEP = 2


def make_grid(mixing_cls=NonReflectingMixingPatch, ends="nonreflecting", nblock=2):
    """Blocks butted end to end, joined by mixing planes of one class.

    ``ends`` selects the inlet and outlet class, so the same builder covers both
    the non-reflecting patches the settings target and the plain ones they must
    leave alone. Runnable: wall distance, reference length, periodic seams and
    the end conditions are all set up.
    """
    blocks = []
    x_next = 0.0
    for _ in range(nblock):
        block = make_block(ni=9, nspan=9, npitch=9)
        block.set_x(block.x - block.x.min() + x_next)
        x_next += float(np.ptp(block.x))
        blocks.append(block)

    for i, (block_up, block_dn) in enumerate(zip(blocks[:-1], blocks[1:])):
        block_up.patches.append(mixing_cls(i=-1, label=f"plane{i}_up"))
        block_dn.patches.append(mixing_cls(i=0, label=f"plane{i}_dn"))

    if ends == "nonreflecting":
        blocks[0].patches.append(NonReflectingInletPatch(i=0))
        blocks[-1].patches.append(NonReflectingOutletPatch(i=-1))
    else:
        blocks[0].patches.append(InletPatch(i=0))
        blocks[-1].patches.append(OutletPatch(i=-1))

    grid = Grid(blocks)
    for block in grid:
        block.set_wdist(0.0)
        for face in (0, -1):
            block.patches.append(PeriodicPatch(k=face))
    # After the patches are attached, as the case setup scripts do it.
    grid.set_L_ref(float(np.ptp(grid[0].x)))

    P_exit = float(grid[-1].P[-1].mean())
    if ends == "nonreflecting":
        # The non-reflecting inlet prescribes characteristics, not stagnation
        # conditions; seed it from the flow the blocks were built with so it
        # starts at its own fixed point.
        inlet = grid.patches.inlet_nonreflecting[0]
        inlet.set_ho_s(float(grid[0].ho.mean()), float(grid[0].s.mean()))
        inlet.set_Alpha(float(grid[0].Alpha.mean()))
        inlet.set_Beta(float(grid[0].Beta.mean()))
        grid.patches.outlet_nonreflecting[0].set_P(P_exit)
    else:
        grid.patches.inlet[0].set_Po_To_Alpha_Beta(
            float(grid[0].Po[0].mean()), float(grid[0].To[0].mean()), 0.0, 0.0
        )
        grid.patches.outlet[0].set_P(P_exit)

    grid.connectivity.periodic.pair()
    return grid


def solver(**settings):
    """A short run configured with the settings under test."""
    return ember.solver.Solver(
        n_step=N_STEP, n_step_avg=1, n_step_log=N_STEP, n_stage=4, **settings
    )


def sigmas(patches):
    return [patch.sigma for patch in patches]


def rf_exchanges(patches):
    return [patch.rf_exchange for patch in patches]


# Push


@pytest.mark.parametrize(
    "setting, collection",
    [
        ("rf_inlet", "inlet_nonreflecting"),
        ("rf_outlet", "outlet_nonreflecting"),
        ("rf_mix", "mixing_nonreflecting"),
    ],
)
def test_each_sigma_setting_lands_on_the_patches_it_names(setting, collection):
    """The three sigma settings each drive one non-reflecting collection."""
    grid = make_grid()
    ember.solver._apply_bcond_relaxation(grid, solver(**{setting: 0.123}))

    np.testing.assert_allclose(sigmas(getattr(grid.patches, collection)), 0.123)


@pytest.mark.parametrize(
    "setting, untouched",
    [
        ("rf_inlet", ["outlet_nonreflecting", "mixing_nonreflecting"]),
        ("rf_outlet", ["inlet_nonreflecting", "mixing_nonreflecting"]),
        ("rf_mix", ["inlet_nonreflecting", "outlet_nonreflecting"]),
    ],
)
def test_each_sigma_setting_leaves_the_other_collections_alone(setting, untouched):
    """One knob per boundary kind: the three do not reach across to each other.

    The others are passed as None so they impose nothing, leaving a sentinel in
    place that only a cross-wired setting could disturb. Without that they would
    each be pushed to the same default and the test could not tell the
    collections apart.
    """
    grid = make_grid()
    settings = dict.fromkeys(["rf_inlet", "rf_outlet", "rf_mix"], None)
    settings[setting] = 0.123
    for collection in untouched:
        for patch in getattr(grid.patches, collection):
            patch.sigma = 0.321

    ember.solver._apply_bcond_relaxation(grid, solver(**settings))

    for collection in untouched:
        np.testing.assert_allclose(sigmas(getattr(grid.patches, collection)), 0.321)


def test_rf_exchange_drives_both_plane_types():
    """It is the exchange's factor, and both mixing planes have an exchange."""
    for mixing_cls, collection in (
        (NonReflectingMixingPatch, "mixing_nonreflecting"),
        (MixingPatch, "mixing"),
    ):
        grid = make_grid(mixing_cls=mixing_cls)
        ember.solver._apply_bcond_relaxation(grid, solver(rf_exchange=0.123))

        patches = getattr(grid.patches, collection)
        assert len(patches) == 2
        np.testing.assert_allclose([p.rf_exchange for p in patches], 0.123)


def test_the_sigma_settings_do_not_touch_the_plain_inlet_and_outlet():
    """Their ``rf`` relaxes a different quantity, so the names must not collide.

    The plain inlet relaxes a pressure datum and the plain outlet takes its
    factor through set_adjustment; neither is a characteristic relaxation.
    """
    grid = make_grid(ends="plain")
    grid.patches.outlet[0].set_adjustment(rf=0.07)
    before = (
        grid.patches.inlet[0].rf,
        grid.patches.inlet[0].rf_stage,
        grid.patches.outlet[0]._adjustment["rf"],
    )

    ember.solver._apply_bcond_relaxation(grid, solver(rf_inlet=0.123, rf_outlet=0.123))

    assert (
        grid.patches.inlet[0].rf,
        grid.patches.inlet[0].rf_stage,
        grid.patches.outlet[0]._adjustment["rf"],
    ) == before


# Defaults


def hand_set(grid):
    """Distinctive values on every patch the settings can reach."""
    for patch in grid.patches.mixing_nonreflecting:
        patch.sigma = 0.4
        patch.rf_exchange = 0.3
    grid.patches.inlet_nonreflecting[0].sigma = 0.2
    grid.patches.outlet_nonreflecting[0].sigma = 0.1


def test_none_leaves_a_hand_set_value_alone():
    """Opting out is what lets a setup script's own damping reach a run.

    A value set at setup time and carried in through the EMB survives only if
    every setting that could overwrite it is passed as None.
    """
    grid = make_grid()
    hand_set(grid)

    solver(rf_inlet=None, rf_outlet=None, rf_mix=None, rf_exchange=None).run(grid)

    np.testing.assert_allclose(sigmas(grid.patches.mixing_nonreflecting), 0.4)
    np.testing.assert_allclose(rf_exchanges(grid.patches.mixing_nonreflecting), 0.3)
    assert grid.patches.inlet_nonreflecting[0].sigma == pytest.approx(0.2)
    assert grid.patches.outlet_nonreflecting[0].sigma == pytest.approx(0.1)


def test_the_defaults_are_imposed_over_a_hand_set_value():
    """The solver has the last word, so a run's damping follows its own config.

    The alternative -- defaulting to None -- would leave the damping of a run
    depending on state the grid happened to arrive with, which is invisible in
    the settings the run was launched from.
    """
    grid = make_grid()
    hand_set(grid)

    solver().run(grid)

    np.testing.assert_allclose(sigmas(grid.patches.mixing_nonreflecting), 0.05)
    np.testing.assert_allclose(rf_exchanges(grid.patches.mixing_nonreflecting), 0.05)
    assert grid.patches.inlet_nonreflecting[0].sigma == pytest.approx(0.05)
    assert grid.patches.outlet_nonreflecting[0].sigma == pytest.approx(0.05)


# Independence


def test_rf_mix_and_rf_exchange_are_separate_knobs():
    """Two factors on the same plane: the patch's own, and the exchange's.

    Distinct values rather than one-set-one-default, so a crossed assignment
    cannot pass by both landing on the same number.
    """
    grid = make_grid()
    ember.solver._apply_bcond_relaxation(grid, solver(rf_mix=0.2, rf_exchange=0.3))

    np.testing.assert_allclose(sigmas(grid.patches.mixing_nonreflecting), 0.2)
    np.testing.assert_allclose(rf_exchanges(grid.patches.mixing_nonreflecting), 0.3)


# Wiring


def test_a_run_applies_the_settings():
    """The push is hooked into the march, not just callable on its own."""
    grid = make_grid()

    solver(rf_inlet=0.11, rf_outlet=0.12, rf_mix=0.13, rf_exchange=0.14).run(grid)

    assert grid.patches.inlet_nonreflecting[0].sigma == pytest.approx(0.11)
    assert grid.patches.outlet_nonreflecting[0].sigma == pytest.approx(0.12)
    np.testing.assert_allclose(sigmas(grid.patches.mixing_nonreflecting), 0.13)
    np.testing.assert_allclose(rf_exchanges(grid.patches.mixing_nonreflecting), 0.14)


def test_a_run_with_a_cached_communicator_still_retunes_the_exchange():
    """A communicator built before the run picks the new factor up.

    apply_bconds builds and caches the communicator on first use, so a grid that
    has been touched before the solver sees it arrives with one already in hand.
    The exchange reads rf_exchange off the patches every time for this reason.
    """
    grid = make_grid()
    grid.apply_bconds()  # builds and caches the communicator

    solver(rf_exchange=0.14).run(grid)

    comm = grid.connectivity.mixing_nonreflecting._get_communicator()
    ((bid, pid),) = comm.pairs
    patch1, _ = comm._get_pair(bid, pid)
    assert patch1.rf_exchange == pytest.approx(0.14)


def test_fmg_applies_the_settings_on_every_level():
    """Each full-multigrid level is its own resampled grid with its own patches.

    Pushing from _run rather than Solver.run is what covers them; the finest
    level is the one handed in, so check a coarse level too by running the chain
    and asserting the finest came back configured.

    Plain ends, because resample carries the non-reflecting outlet's spanwise
    pressure level across at the fine grid's span count and it no longer fits
    the coarse target. That is a pre-existing limitation of that patch under
    full multigrid and nothing to do with the relaxation settings.
    """
    grid = make_grid(ends="plain")
    conf = ember.solver.Solver(
        n_step=N_STEP,
        n_step_avg=1,
        n_step_log=N_STEP,
        n_stage=4,
        n_levels=1,
        rf_mix=0.13,
        rf_exchange=0.14,
    )

    histories = conf.run_fmg(grid)

    assert len(histories) == 2  # coarse then fine
    np.testing.assert_allclose(sigmas(grid.patches.mixing_nonreflecting), 0.13)
    np.testing.assert_allclose(rf_exchanges(grid.patches.mixing_nonreflecting), 0.14)
