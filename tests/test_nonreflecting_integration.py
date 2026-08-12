"""Solver runs with non-reflecting boundaries at every meridional orientation.

Modules tested: ember.nonreflecting, ember.inlet, ember.outlet, ember.mixing

The unit tests in test_nonreflecting_orientation.py and test_mixing.py drive
the conditions directly. These drive them the way a run does -- through
``Grid.update_bconds`` and ``Grid.apply_bconds``, once per step and once per
Runge-Kutta stage, alongside the interior march, the periodic seams and the
residual -- which is where a face left in interface coordinates, or a frame
settled the wrong way round, would show up as something other than a wrong
number on a patch.

The same duct is built at each orientation: axial, conical, radially outward,
running backwards, radially inward. Turning it is a rigid rotation in the
meridional plane, so every case is the same passage seen from a different
angle, except that a radial one changes area along its length the way a real
one does.

Test cases:
- test_duct_run_stays_finite: an inlet and outlet pair marches without diverging
- test_duct_converges_to_the_prescribed_state: the residual falls and the
  boundaries hold what they were told to hold
- test_mixing_plane_run_stays_finite: two blocks and a plane between them
- test_mixing_plane_conserves_mass: mass flow matches across the plane and
  through both ends
"""

import numpy as np
import pytest

import ember.block
import ember.fluid
import ember.grid
import ember.patch
import ember.solver
from ember import average, util

# Axial, conical, radially outward, running backwards, radially inward, plus
# one bowed case so a face whose normal turns along the span is marched too.
ORIENTATIONS = [
    (0.0, 0.0),
    (30.0, 0.0),
    (90.0, 0.0),
    (180.0, 0.0),
    (270.0, 0.0),
    (30.0, 0.15),
]

L = 0.1
R_MID = 2.0
SHAPE = (17, 13, 9)
NB = int(np.round(2 * np.pi * R_MID / L))
PITCH = 2.0 * np.pi / NB

VX = 100.0
P_MEAN = 1.0e5
T_MEAN = 300.0

FLUID = ember.fluid.PerfectFluid(
    cp=1005.0,
    gamma=1.4,
    mu=1.8e-4,
    Pr=1.0,
    T_dtm=500.0,
    Rgas_ref=287.0,
    rho_ref=1.1,
    V_ref=100.0,
)


def _turn(Vm, Vn, chi):
    """A meridional vector given along and across the duct, resolved onto (x, r)."""
    c, s = np.cos(np.radians(chi)), np.sin(np.radians(chi))
    return Vm * c - Vn * s, Vm * s + Vn * c


def _make_block(chi, bow, i_block=0):
    """One duct block, turned through chi and offset i_block lengths downstream.

    Built in duct coordinates -- m along the flow, n across it -- and turned
    into the meridional plane, so at ``chi = 0`` it is the straight annular
    duct the axial tests use and at 90 or 270 a radial one. ``bow`` curves the
    constant-m surfaces, which is what makes the end faces' normals turn from
    hub to tip rather than being one direction for the whole face.
    """
    xrt = util.linmesh3([0.0, L], [0.0, L], [-PITCH / 2, PITCH / 2], SHAPE)
    m = xrt[..., 0] + i_block * L
    n = xrt[..., 1]
    if bow:
        m = m + bow * L * np.sin(np.pi * n / L)

    x, r = _turn(m, n, chi)
    r = r + R_MID

    # Swirl building along the duct, as the axial fixture skews it: the same
    # 30 degrees of turning, measured along the duct rather than along x so it
    # means the same thing at every orientation.
    t = xrt[..., 2] + np.tan(np.radians(30.0)) * m / r

    block = ember.block.Block(shape=SHAPE)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)
    block.set_Nb(NB)
    block.set_fluid(FLUID)
    block.set_P_T(P_MEAN, T_MEAN)
    Vx, Vr = _turn(VX * 0.99, 0.0, chi)
    block.set_Vx(Vx * np.ones(SHAPE))
    block.set_Vr(Vr * np.ones(SHAPE))
    block.set_Vt(0.0)
    block.set_wdist(0.0)
    return block


def _finish(grid, chi):
    """Prescribe the boundaries and pair the connectivity."""
    grid.set_L_ref(L)
    inlet = grid.patches.inlet[0]
    block_in = grid[0]
    inlet.set_Po_To(float(block_in.Po[0].mean()), float(block_in.To[0].mean()))
    inlet.set_Alpha(0.0)
    # The flow enters along the duct, which makes the machine-frame pitch angle
    # the duct angle. This is the whole point of the angle staying axis-relative.
    inlet.set_Beta(chi)
    grid.patches.outlet[0].set_P(P_MEAN)

    grid.calculate_wdist()
    grid.connectivity.periodic.pair()
    if grid.patches.mixing:
        grid.connectivity.mixing.pair()
    return grid


def make_duct(chi, bow):
    """One block with a non-reflecting inlet and outlet at either end."""
    block = _make_block(chi, bow)
    block.patches.extend(
        [
            ember.patch.InletPatch(i=0),
            ember.patch.OutletPatch(i=-1),
            ember.patch.PeriodicPatch(k=0),
            ember.patch.PeriodicPatch(k=-1),
        ]
    )
    return _finish(ember.grid.Grid((block,)), chi)


def make_stage(chi, bow):
    """Two blocks with a non-reflecting mixing plane between them."""
    block_up = _make_block(chi, bow, i_block=0)
    block_dn = _make_block(chi, bow, i_block=1)
    block_up.patches.extend(
        [
            ember.patch.InletPatch(i=0),
            ember.patch.MixingPatch(i=-1),
            ember.patch.PeriodicPatch(k=0),
            ember.patch.PeriodicPatch(k=-1),
        ]
    )
    block_dn.patches.extend(
        [
            ember.patch.MixingPatch(i=0),
            ember.patch.OutletPatch(i=-1),
            ember.patch.PeriodicPatch(k=0),
            ember.patch.PeriodicPatch(k=-1),
        ]
    )
    return _finish(ember.grid.Grid((block_up, block_dn)), chi)


def assert_physical(grid):
    """Every block still holds a state a gas could be in."""
    for block in grid:
        assert np.all(np.isfinite(block.conserved)), "Non-finite conserved variables"
        assert np.all(block.rho > 0), "Non-positive density"
        assert np.all(block.P > 0), "Non-positive pressure"
        assert np.all(block.T > 0), "Non-positive temperature"


def mdot(patch):
    """Mass flow through a patch face, from rho V.dA rather than an assumed normal."""
    return abs(float(average.flow_mass(patch.block_view.squeeze())))


# ---------------------------------------------------------------------------
# A duct between an inlet and an outlet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chi, bow", ORIENTATIONS)
def test_duct_run_stays_finite(chi, bow):
    """A march with both ends non-reflecting completes at any orientation."""
    grid = make_duct(chi, bow)
    ember.solver.Solver(n_step=20, n_step_avg=1, n_step_log=20, n_stage=4).run(grid)
    assert_physical(grid)


@pytest.mark.parametrize("chi, bow", ORIENTATIONS)
def test_duct_converges_to_the_prescribed_state(chi, bow):
    """The residual falls and the boundaries hold what they were told to hold.

    Staying finite only says nothing blew up. This says the conditions are
    imposing the right thing: a boundary resolving the velocity onto the wrong
    normal would still march, and would settle at the wrong exit pressure and
    the wrong inlet angle.
    """
    grid = make_duct(chi, bow)
    history = ember.solver.Solver(
        n_step=200, n_step_avg=1, n_step_log=10, n_stage=4
    ).run(grid)
    assert not history.diverged
    assert_physical(grid)

    # Settled, and past the transient. Judged on the level it settles at
    # rather than on how far it fell: a turned duct changes area along its
    # length and starts with a real transient to decay, while an axial one
    # starts within 1e-9 of its own answer and has nothing to fall from. A
    # boundary imposing the wrong thing leaves a standing residual instead,
    # around the 2e-4 the turned cases start at.
    resid = np.abs(np.asarray(history.residual, dtype=float)[: history.i_log + 1, 4])
    assert resid[-1] < 1e-4, resid
    assert resid[-1] < 0.85 * resid.max(), resid

    # The outlet holds the pressure it was given, as a pitchwise mean.
    outlet = grid.patches.outlet[0]
    P_exit = float(np.mean(outlet.block_view.P))
    assert P_exit == pytest.approx(P_MEAN, rel=2e-3)

    # The inlet holds the angle it was given, measured from the machine axis.
    inlet = grid.patches.inlet[0]
    b = inlet.block_view
    Beta = np.degrees(np.arctan2(np.asarray(b.Vr), np.asarray(b.Vx)))
    # Wrapped, so chi = 180 is not compared the long way round.
    delta = np.radians(Beta - chi)
    delta = np.degrees(np.arctan2(np.sin(delta), np.cos(delta)))
    assert np.abs(np.mean(delta)) < 2.0, np.mean(delta)


@pytest.mark.parametrize("chi, bow", ORIENTATIONS)
def test_duct_conserves_mass_end_to_end(chi, bow):
    """What goes in comes out, whichever way the duct points."""
    grid = make_duct(chi, bow)
    ember.solver.Solver(n_step=200, n_step_avg=1, n_step_log=20, n_stage=4).run(grid)

    mdot_in = mdot(grid.patches.inlet[0])
    mdot_out = mdot(grid.patches.outlet[0])
    assert mdot_in > 0.0
    assert abs(mdot_out - mdot_in) / mdot_in < 0.02


# ---------------------------------------------------------------------------
# Two blocks and a mixing plane
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chi, bow", ORIENTATIONS)
def test_mixing_plane_run_stays_finite(chi, bow):
    """A plane between two blocks marches at any orientation.

    The frame a plane works in cannot come from its class -- both sides are the
    same class -- so it is settled against the flow on the first exchange. At
    chi = 270 the frame the geometry would give points upstream, so this is the
    case that exercises the settle rather than merely tolerating it.
    """
    grid = make_stage(chi, bow)
    ember.solver.Solver(n_step=20, n_step_avg=1, n_step_log=20, n_stage=4).run(grid)
    assert_physical(grid)

    # Settled, opposite, and each side reading its own flow direction: the
    # upstream side is an outflow and the downstream side an inflow.
    patch_up, patch_dn = grid.patches.mixing
    assert patch_up._sign_settled and patch_dn._sign_settled
    assert {patch_up._sign_interior, patch_dn._sign_interior} == {-1, 1}


@pytest.mark.parametrize("chi, bow", ORIENTATIONS)
def test_mixing_plane_conserves_mass(chi, bow):
    """Mass flow matches across the plane and through both ends of the stage."""
    grid = make_stage(chi, bow)
    ember.solver.Solver(n_step=200, n_step_avg=1, n_step_log=20, n_stage=4).run(grid)
    assert_physical(grid)

    patch_up, patch_dn = grid.patches.mixing
    flows = [
        mdot(grid.patches.inlet[0]),
        mdot(patch_up),
        mdot(patch_dn),
        mdot(grid.patches.outlet[0]),
    ]
    assert flows[0] > 0.0
    spread = (max(flows) - min(flows)) / flows[0]
    assert spread < 0.05, flows
