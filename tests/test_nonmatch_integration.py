"""Integration test for the non-matching interface boundary condition.

Modules tested: ember.nonmatch, ember.nonmatch_communicator, ember.grid

Test cases:
- test_nonmatch_no_nan: Two-block non-matching run completes without divergence
- test_nonmatch_closes_interface_jump: an initial pressure jump across the
  interface is pulled together by the march, which it is not if the
  interface is left unconnected
"""

import numpy as np
import pytest

import ember.block
import ember.fluid
import ember.grid
import ember.nonmatch_communicator
import ember.patch
import ember.solver
from ember import util

P = 1.0e5
dP_init = 0.02 * P


def _make_grid():
    """Two blocks in series, joined by a non-matching interface.

    The downstream block has a finer pitchwise and coarser spanwise
    distribution than the upstream one, so no interface node is shared by
    index, and starts at a higher static pressure, so the two faces of the
    interface disagree before the first step.
    """
    L = 0.1
    r1 = 2.0
    Nb = int(np.round(2 * np.pi * r1 / L))
    pitch = 2.0 * np.pi / Nb
    shape_up = (17, 13, 9)
    shape_dn = (17, 9, 17)

    fluid = ember.fluid.PerfectFluid(
        cp=1005.0,
        gamma=1.4,
        mu=1.8e-4,
        Pr=1.0,
        T_dtm=500.0,
        Rgas_ref=287.0,
        rho_ref=1.1,
        V_ref=100.0,
    )

    def _block(x, shape, Pb):
        xrt = util.linmesh3(x, [r1, r1 + L], [-pitch / 2, pitch / 2], shape)
        block = ember.block.Block(shape=shape)
        block.set_x(xrt[..., 0])
        block.set_r(xrt[..., 1])
        block.set_t(xrt[..., 2])
        block.set_Nb(Nb)
        block.set_fluid(fluid)
        block.set_P_T(Pb, 300.0)
        block.set_Vx(100.0)
        block.set_Vr(0.0)
        block.set_Vt(0.0)
        block.set_wdist(0.0)
        return block

    block_up = _block([0.0, L], shape_up, P)
    block_dn = _block([L, 2.0 * L], shape_dn, P + dP_init)

    block_up.patches.extend(
        [
            ember.patch.InletPatch(i=0),
            ember.patch.NonMatchPatch(i=-1),
            ember.patch.PeriodicPatch(k=0),
            ember.patch.PeriodicPatch(k=-1),
        ]
    )
    block_dn.patches.extend(
        [
            ember.patch.NonMatchPatch(i=0),
            ember.patch.OutletPatch(i=-1),
            ember.patch.PeriodicPatch(k=0),
            ember.patch.PeriodicPatch(k=-1),
        ]
    )

    grid = ember.grid.Grid((block_up, block_dn))
    grid.set_L_ref(L)

    inlet = grid.patches.inlet[0]
    inlet.set_Po_To(block_up.Po[0].mean(), block_up.To[0].mean())
    inlet.set_Alpha(0.0)
    inlet.set_Beta(0.0)
    grid.patches.outlet[0].set_P(P)

    grid.calculate_wdist()
    grid.connectivity.periodic.pair()
    assert len(grid.connectivity.nonmatch.pair()) == 2

    return grid


def _run(grid):
    ember.solver.Solver(
        n_step=5,
        n_step_avg=1,
        n_step_log=5,
        n_stage=4,
        n_levels=0,
    ).run(grid)


def _interface_jump(grid):
    """Difference in face-mean static pressure across the interface."""
    block_up, block_dn = grid
    return block_dn.P[0].mean() - block_up.P[-1].mean()


@pytest.fixture(scope="module")
def nonmatch_grid():
    grid = _make_grid()
    _run(grid)
    return grid


def test_nonmatch_no_nan(nonmatch_grid):
    """Non-matching run completes 5 steps without NaN or non-physical values."""
    for block in nonmatch_grid:
        assert np.all(np.isfinite(block.conserved)), "Non-finite conserved variables"
        assert np.all(block.rho > 0), "Non-positive density"
        assert np.all(block.P > 0), "Non-positive pressure"
        assert np.all(block.T > 0), "Non-positive temperature"


def test_nonmatch_closes_interface_jump(nonmatch_grid, monkeypatch):
    """The march averages the two faces of the interface together.

    Run once more with the communicator stubbed out to show that the jump
    survives on its own, so the closure is the interface's doing.
    """
    assert np.isclose(_interface_jump(_make_grid()), dP_init, rtol=1e-3)

    jump = _interface_jump(nonmatch_grid)
    assert abs(jump) < 0.01 * dP_init

    grid_off = _make_grid()
    monkeypatch.setattr(
        ember.nonmatch_communicator.NonMatchCommunicator, "apply", lambda self: None
    )
    _run(grid_off)
    assert abs(_interface_jump(grid_off)) > 0.5 * dP_init
