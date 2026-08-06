"""Integration test for the mixing plane boundary condition.

Test cases:
- test_mixing_plane_no_nan: Two-block mixing-plane run completes without divergence
- test_target_re_seeds_when_the_reference_scales_move: the exchanged target is
  taken again rather than carried across a change of fluid
"""

import numpy as np
import pytest

import ember.block
import ember.fluid
import ember.grid
import ember.mixing_communicator
import ember.patch
import ember.solver
from ember import util


@pytest.fixture(scope="module")
def mixing_grid():
    """Two-block grid connected by a skewed mixing plane."""
    L = 0.1
    r1 = 2.0
    Nb = int(np.round(2 * np.pi * r1 / L))
    pitch = 2.0 * np.pi / Nb
    shape = (17, 13, 9)

    xrt = util.linmesh3([0.0, L], [r1, r1 + L], [-pitch / 2, pitch / 2], shape)

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

    Vx = 100.0
    T = 300.0
    P = 1.0e5

    # Skew grid 30 degrees to exercise non-trivial mixing geometry
    skew = 30.0
    tanAlpha = np.tan(np.radians(skew))
    dx = xrt[:, :, :, 0] - xrt[0, :, :, 0]
    dt = tanAlpha * dx / xrt[:, :, :, 1]
    xrt[:, :, :, 2] += dt

    block_up = ember.block.Block(shape=shape)
    block_up.set_x(xrt[..., 0])
    block_up.set_r(xrt[..., 1])
    block_up.set_t(xrt[..., 2])
    block_up.set_Nb(Nb)
    block_up.set_fluid(fluid)
    block_up.set_P_T(P, T)
    block_up.set_Vx(Vx * 0.99)
    block_up.set_Vr(0.0)
    block_up.set_Vt(0.0)
    block_up.set_wdist(0.0)

    block_dn = block_up.copy()
    block_dn.set_x(block_up.x + L)
    block_dn.set_t(block_up.t + dt[-1, :, :])

    block_up.patches.extend(
        [
            ember.patch.MixingPatch(i=-1),
            ember.patch.PeriodicPatch(k=0),
            ember.patch.PeriodicPatch(k=-1),
            ember.patch.InletPatch(i=0),
        ]
    )
    block_dn.patches.extend(
        [
            ember.patch.MixingPatch(i=0),
            ember.patch.PeriodicPatch(k=0),
            ember.patch.PeriodicPatch(k=-1),
            ember.patch.OutletPatch(i=-1),
        ]
    )

    grid = ember.grid.Grid((block_up, block_dn))
    grid.set_L_ref(L)

    Po1 = block_up.Po[0].mean()
    To1 = block_up.To[0].mean()
    inlet = grid.patches.inlet[0]
    inlet.set_Po_To(Po1, To1)
    inlet.set_Alpha(0.0)
    inlet.set_Beta(0.0)
    grid.patches.outlet[0].set_P(P)

    grid.calculate_wdist()
    grid.connectivity.periodic.pair()
    grid.connectivity.mixing.pair()

    return grid


def test_mixing_plane_no_nan(mixing_grid):
    """Mixing-plane run completes 5 steps without NaN or non-physical values."""
    config = ember.solver.Solver(
        n_step=5,
        n_step_avg=1,
        n_step_log=5,
        n_stage=4,
    )

    config.run(mixing_grid)

    for block in mixing_grid:
        assert np.all(np.isfinite(block.conserved)), "Non-finite conserved variables"
        assert np.all(block.rho > 0), "Non-positive density"
        assert np.all(block.P > 0), "Non-positive pressure"
        assert np.all(block.T > 0), "Non-positive temperature"


def test_mixing_communicator_get_stats(mixing_grid):
    """get_stats returns None before exchange and the relaxation increment after."""
    comm = ember.mixing_communicator.MixingCommunicator(
        mixing_grid, mixing_grid.connectivity.mixing.pair()
    )
    (bid, pid) = next(iter(comm.pairs))

    assert comm.get_stats(bid, pid) is None

    comm.exchange()

    stats = comm.get_stats(bid, pid)
    assert stats is not None
    assert set(stats) == {"du"}
    assert stats["du"].shape[1] == 5


def test_target_re_seeds_when_the_reference_scales_move():
    """The exchanged target is dropped and taken again on a change of fluid.

    It is a nondimensional conserved state with no dimensional original to
    reconvert, so carrying it across would impose the old scales' numbers on
    the new ones -- an inconsistency at the plane that nothing reports.
    """
    fluid = ember.fluid.PerfectFluid(
        cp=1005.0, gamma=1.4, mu=1.8e-4, Pr=1.0, rho_ref=1.1, V_ref=100.0
    )
    shape = (5, 5, 5)
    xrt = util.linmesh3([0.0, 0.1], [1.0, 1.1], [0.0, 0.05], shape)
    block = ember.block.Block(shape=shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_Nb(20)
    block.set_fluid(fluid)
    block.set_P_T(1.0e5, 300.0)
    block.set_Vx(100.0)
    block.set_Vr(0.0)
    block.set_Vt(0.0)

    patch = ember.patch.MixingPatch(i=-1)
    block.patches.append(patch)
    before = patch.get_target().copy()
    assert patch._target is not None

    block.set_fluid(fluid.change_ref(rho_ref=0.7, V_ref=250.0))
    assert patch._target is None

    # Re-seeded from the rescaled field, so the plane still carries the state
    # the block is actually in: the same density and axial mass flux, read
    # against the new scales.
    after = patch.get_target()
    scale_old = np.array([1.1, 1.1 * 100.0])
    scale_new = np.array([0.7, 0.7 * 250.0])
    np.testing.assert_allclose(
        after[:, :2] * scale_new, before[:, :2] * scale_old, rtol=1e-5
    )
