"""Tests for the mixing plane.

Modules tested: ember.mixing, ember.mixing_communicator

The patch class adds no numerics of its own -- the characteristic split,
the Hilbert transform, the mean-mode Newton step and the harmonic relations are
all inherited from the non-reflecting inlet and outlet, and are tested in
test_nonreflecting.py, test_inlet.py and
test_outlet.py. What is left here is the mixing plane itself.

Test cases:
- Pairing: the two sides pair across the plane, same-side and foreign patches do
  not, differing pitchwise resolution is allowed
- Collections: both sides appear under mixing and not under the
  plain non-reflecting inlet/outlet lists, and row stations find them
- Target: shape, pitch-uniformity, lazy seeding, copy semantics, set_adjustment
- Physics: matched flow is a fixed point, a cross-plane mismatch relaxes to
  matched pitch-mean fluxes, a pitchwise harmonic is absorbed rather than
  reflected, and a solver run stays finite
- Stalled and reversed stations: a station whose cross-plane mean axial velocity
  is zero survives the exchange, and the clip on that mean bounds it in
  magnitude without turning a reversed station round
- Chains: several planes in one grid stay independent of one another, with a
  middle block carrying an inflow side and an outflow side at once
- Reflective mode: the flag pairs only with itself, both faces end up holding
  one common pitch-uniform state, and mass crosses the plane exactly
"""

import pickle

import numpy as np
import pytest

import ember.solver
from ember import average
from ember.grid import Grid
from ember.mixing_communicator import MixingCommunicator
from ember.patch import InletPatch, MixingPatch, OutletPatch, PeriodicPatch
from nonreflecting_util import LEN_M, harmonic, make_block, seed_chic, turn

# The exchange and the boundary conditions are both heavily under-relaxed by
# default, which is right for a solver run and far too slow for a test that
# iterates the boundary alone. These drive the same fixed point, faster.
RF_EXCHANGE_FAST = 0.5
SIGMA_FAST = 0.5


def make_chain(states, npitch=17, chi=0.0, bow=0.0):
    """Blocks butted end to end, each junction a non-reflecting mixing plane.

    One entry in ``states`` per block, a dict of overrides for
    :func:`nonreflecting_util.make_block`, so each block can start at a
    different flow state and the planes between them see a genuine mismatch.
    ``npitch`` is a scalar or one value per block; the two sides of a plane may
    differ in it, since only pitch means cross.

    ``chi`` turns the whole chain in the meridional plane and ``bow`` curves
    the planes between the blocks, so the same chain can be built axial,
    conical, radially outward or radially inward. The blocks are butted along
    the duct axis rather than along x, which at ``chi = 0`` is the same thing.

    Returns the grid and a list of ``(outflow side, inflow side)`` pairs, one
    per plane, ordered upstream to downstream.
    """
    npitches = [npitch] * len(states) if np.isscalar(npitch) else list(npitch)

    blocks = []
    for i_block, (state, npitch_block) in enumerate(zip(states, npitches, strict=True)):
        block = make_block(npitch=npitch_block, chi=chi, bow=bow, **state)
        dx, dr = turn(i_block * LEN_M, 0.0, chi)
        block.set_x(block.x + dx)
        block.set_r(block.r + dr)
        blocks.append(block)

    planes = []
    for i, (block_up, block_dn) in enumerate(zip(blocks[:-1], blocks[1:])):
        patch_up = MixingPatch(i=-1, label=f"plane{i}_up")
        patch_dn = MixingPatch(i=0, label=f"plane{i}_dn")
        block_up.patches.append(patch_up)
        block_dn.patches.append(patch_dn)
        planes.append((patch_up, patch_dn))

    return Grid(blocks), planes


def make_pair(npitch_up=17, npitch_dn=17, up=None, dn=None, chi=0.0, bow=0.0, **kwargs):
    """Two blocks joined by one non-reflecting mixing plane.

    ``up`` and ``dn`` are per-side overrides of the flow state, so the two
    sides can be started mismatched. Returns the grid, the upstream (outflow)
    patch and the downstream (inflow) patch.
    """
    grid, planes = make_chain(
        [{**kwargs, **(up or {})}, {**kwargs, **(dn or {})}],
        npitch=(npitch_up, npitch_dn),
        chi=chi,
        bow=bow,
    )
    ((patch_up, patch_dn),) = planes
    return grid, patch_up, patch_dn


def communicator(grid, rf_exchange=RF_EXCHANGE_FAST, sigma=SIGMA_FAST):
    """Communicator for every plane in a grid, with the patches sped up."""
    for patch in grid.patches.mixing:
        patch.sigma = sigma
        patch.rf_exchange = rf_exchange
    return MixingCommunicator(grid, grid.connectivity.mixing.pair())


def exchanged(*args, rf_exchange=RF_EXCHANGE_FAST, sigma=SIGMA_FAST, **kwargs):
    """A single paired plane plus its communicator, ready to exchange."""
    grid, patch_up, patch_dn = make_pair(*args, **kwargs)
    return grid, patch_up, patch_dn, communicator(grid, rf_exchange, sigma)


def relax(patches, comm, n_iter):
    """Iterate the exchange and every boundary condition to the fixed point."""
    for _ in range(n_iter):
        comm.exchange()
        for patch in patches:
            patch.update_soln()
            patch.advance()
            patch.apply()


def flux_gap(patch_up, patch_dn):
    """Largest pitch-mean flux mismatch across the plane, relative to its scale.

    Each of the five components is scaled on its own, floored against the
    largest of them so that a component which is identically zero either side
    (the radial momentum flux of a swirl-only mean state) does not divide by
    zero.
    """
    patch_up.set_flux_avg()
    patch_dn.set_flux_avg()
    flux_up = patch_up.flux_avg_nd
    flux_dn = patch_dn.flux_avg_nd
    scale = np.maximum(np.abs(flux_up), np.abs(flux_dn)).max(axis=0)
    scale = np.maximum(scale, 1e-6 * scale.max())
    return np.abs(flux_dn - flux_up).max(axis=0) / scale


# Pairing


def test_pairs_across_the_plane():
    """The two sides of the plane pair with each other, in both directions."""
    grid, patch_up, patch_dn = make_pair()
    pairs = grid.connectivity.mixing.pair()
    assert pairs == {(0, 0): ((1, 0), False), (1, 0): ((0, 0), False)}


def test_side_is_read_off_the_geometry():
    """One class serves both sides; which one it is comes from the mesh.

    The two sides are constructed identically and differ only in the face they
    are attached to, so nothing is left for a caller to get right that the mesh
    does not already say.
    """
    grid, patch_up, patch_dn = make_pair()
    assert type(patch_up) is type(patch_dn)
    # Interior on the -x side of the upstream block's exit face, on the +x side
    # of the downstream block's inlet face.
    assert patch_up._sign_interior == -1
    assert patch_dn._sign_interior == 1
    # And so the two prescribe different rows of the same exchanged target.
    assert patch_up._split_leaving == ([0], [4])
    assert patch_dn._split_entering == ([1, 2, 3, 4], [0, 1, 2, 3])


def test_pairs_with_unequal_pitchwise_resolution():
    """Only pitch means cross the plane, so the two sides may be resolved differently."""
    grid, patch_up, patch_dn = make_pair(npitch_up=17, npitch_dn=13)
    assert grid.connectivity.mixing.pair()


def test_same_side_patches_do_not_pair():
    """Two outflow sides face the same way, so they are not two sides of a plane."""
    grid, patch_up, _ = make_pair()
    other = MixingPatch(i=-1, label="mix_other")
    grid[1].patches.append(other)
    assert patch_up.check_match(other) is None


def test_plain_nonreflecting_patch_does_not_pair():
    """A prescribed-pressure outlet is not half of a mixing plane."""
    grid, _, patch_dn = make_pair()
    plain = OutletPatch(i=-1, label="plain")
    grid[0].patches.append(plain)
    assert patch_dn.check_match(plain) is None
    assert plain.check_match(patch_dn) is None


# Collections and grid wiring


def test_collections_separate_from_plain_nonreflecting():
    """Both sides list under mixing and under neither plain list.

    The plain lists drive Grid.apply_bconds and the inlet-row search in
    Grid._order_row_groups, so a mixing face leaking into them would be applied
    twice per stage and could misidentify which row is first.
    """
    grid, patch_up, patch_dn = make_pair()
    assert grid.patches.mixing == [patch_up, patch_dn]
    assert grid.patches.inlet == []
    assert grid.patches.outlet == []
    # Still a permeable, non-wall face for the boundary-flux machinery.
    assert patch_up in grid[0].patches.permeable


def test_row_stations_find_both_sides():
    """Each row's mixing face is its own upstream or downstream station."""
    grid, patch_up, patch_dn = make_pair()
    grid[0].patches.append(InletPatch(i=0))
    grid[1].patches.append(OutletPatch(i=-1))
    for face in (0, -1):
        grid[0].patches.append(PeriodicPatch(k=face))
        grid[1].patches.append(PeriodicPatch(k=face))
    grid.connectivity.periodic.pair()

    (_, dn0), (up1, _) = grid.row_station_bid_pid
    # The upstream row's mixing face is an outflow side, so its exit station.
    assert (0, 0) in dn0
    # The downstream row's mixing face is an inflow side, so its inlet station.
    assert (1, 0) in up1


def test_communicator_is_the_nonreflecting_one():
    """Connectivity builds the bcond-space exchange for this patch type."""
    grid, _, _ = make_pair()
    comm = grid.connectivity.mixing._get_communicator()
    assert isinstance(comm, MixingCommunicator)


# Target handling


def test_target_is_pitch_uniform():
    """The exchange writes one value per span station, not a nodal field."""
    grid, patch_up, patch_dn, comm = exchanged()
    comm.exchange()

    nspan = patch_dn.shape[patch_dn.span_dim]
    assert patch_dn.get_target().shape == (nspan, 5)
    for name in ("ho_nd", "s_nd", "Vr_nd", "Vt_nd"):
        target = getattr(patch_dn, name)
        assert target.shape[patch_dn.pitch_dim] == 1
    assert patch_up.P_nd.shape[patch_up.pitch_dim] == 1


def test_target_seeds_from_own_pitch_mean():
    """An unexchanged patch seeds its target from the face it is attached to."""
    grid, patch_up, patch_dn = make_pair()
    assert not patch_dn._target_set.any()

    target = patch_dn.get_target()
    assert patch_dn._target_set.all()
    b = patch_dn.block_view
    for idx, field in enumerate((b.ho_nd, b.s_nd, b.Vr_nd, b.Vt_nd, b.P_nd)):
        expect = patch_dn._pitch_mean(field).squeeze()
        assert np.allclose(target[:, idx], expect, rtol=1e-6)


def test_copy_keeps_target_views_live():
    """A copied patch keeps its target, with the published attributes still views on it."""
    grid, patch_up, patch_dn, comm = exchanged()
    comm.exchange()

    clone = patch_dn.copy()
    clone.attach_to_block(grid[1])
    assert np.allclose(clone.get_target(), patch_dn.get_target())

    # Writing the target must move the published attribute with it, or the
    # condition would keep driving to a stale value.
    clone._target[..., 0] += 1.0
    assert np.allclose(clone.ho_nd, clone._target[..., 0])


def test_update_bconds_refreshes_the_frozen_mean_state():
    """The plane re-derives its reference every step, like the inlet and outlet.

    apply() builds the reference only if it has none, so without a per-step
    update_soln the plane stays linearised about whatever state it first saw --
    for a whole run, that is the initial guess -- and its characteristic split
    is frozen with it, leaving a station that reverses during the transient
    still treated as forward-running.

    Both blocks are reversed at the same station: entering/leaving now comes
    from the communicator's shared direction (:cite:t:`Holmes2008` p.5's
    "these average primitives can then be averaged across the interface, so
    as to ensure that ... both sides of the interface agree"), derived from
    the symmetrised interface state rather than either side's own interior.
    Reversing one side alone would only pull that average partway, which is
    correct behaviour but not what this test is after -- see
    :func:`test_shared_direction_overrides_local` for the partial case.
    """
    grid, patch_up, patch_dn, comm = exchanged()
    grid.update_bconds()
    assert not patch_up._entering[3]

    # Reverse one span station over both blocks' whole axial extent, after the
    # plane has already built a reference on the forward flow.
    for block in (grid[0], grid[1]):
        Vx = block.Vx.copy()
        Vx[:, 3, :] = -20.0
        block.set_Vx(Vx)

    grid.update_bconds()

    assert patch_up._entering[3]
    assert not patch_up._entering[[0, 1, 2, 4, 5, 6]].any()


def test_reversed_station_on_the_inflow_side_takes_the_exchanged_pressure():
    """The downstream side carries reversal too, which is what it never used to.

    A stalled or separated row pushes flow back upstream through the interface,
    and the downstream side of a mixing plane is exactly where that shows up.
    Four of that station's characteristics turn outgoing and one quantity is
    left to prescribe -- static pressure, which is row 4 of the same exchanged
    target the upstream side reads.

    Both sides are reversed at the station and the exchange re-run:
    entering/leaving is the communicator's shared direction now (see
    :func:`test_update_bconds_refreshes_the_frozen_mean_state`), derived from
    the symmetrised interface state, so a call to ``patch.update_soln()``
    alone no longer picks up a change made only to one side's interior.
    """
    grid, patch_up, patch_dn, comm = exchanged(rf_exchange=0.05)
    comm.exchange()

    # Reverse one span station of both blocks over their whole axial extent,
    # hard enough to stay reversed: raising the static pressure at a station
    # the flow leaves raises its axial velocity with it.
    for block in (grid[0], grid[1]):
        Vx = block.Vx.copy()
        Vx[:, 3, :] = -100.0
        block.set_Vx(Vx)
    comm.exchange()

    patch_dn.update_soln()
    assert not patch_dn._entering[3]
    assert patch_dn._entering[[0, 1, 2, 4, 5, 6]].all()

    # Only c_down still enters that station; every other characteristic is the
    # interior's, where at a forward station only c_up would be.
    mask = np.broadcast_to(patch_dn._mask_out, patch_dn.shape + (5,))
    np.testing.assert_array_equal(mask[0, 3, 0], [True, False, True, True, True])
    np.testing.assert_array_equal(mask[0, 0, 0], [True, False, False, False, False])

    for _ in range(60):
        comm.exchange()
        patch_up.update_soln()
        patch_up.advance()
        patch_up.apply()
        patch_dn.update_soln()
        patch_dn.advance()
        patch_dn.apply()

    P_face = np.asarray(patch_dn._pitch_mean(patch_dn.block_view.P_nd)).squeeze()
    assert P_face[3] == pytest.approx(patch_dn.get_target()[3, 4], rel=5e-3)
    assert float(patch_dn._pitch_mean(patch_dn.block_view.Vx_nd).ravel()[3]) < 0.0


def test_exchange_survives_a_stalled_span_station():
    """A station whose cross-plane mean axial velocity is zero goes through exchange.

    The two reversal tests above reverse the flow and then only iterate the
    patches, so the clip in _prepare_pair that holds the symmetrised mean axial
    Mach number away from zero is never reached with a stalled mean. It divides
    by that mean two lines later, so a station sitting exactly at zero is the
    case that has to stay finite.

    Station 3 and not station 0: _write_targets extrapolates the hub and casing
    rows from their neighbours, so a fault driven at the hub is overwritten
    before it can be seen -- and a fault at station 1 is what gets copied into
    the hub row.
    """
    grid, patch_up, patch_dn, comm = exchanged()
    comm.exchange()

    # Zero on both sides, so their average is exactly zero rather than merely
    # small: the clip is only wrong at exactly zero.
    for block in grid:
        Vx = block.Vx.copy()
        Vx[:, 3, :] = 0.0
        block.set_Vx(Vx)

    comm.exchange()

    assert np.all(np.isfinite(patch_up.get_target()))
    assert np.all(np.isfinite(patch_dn.get_target()))

    for patch in (patch_up, patch_dn):
        patch.update_soln()
        patch.advance()
        patch.apply()

    for block in grid:
        assert np.all(np.isfinite(block.conserved))


def test_clip_bounds_the_mean_axial_mach_in_magnitude_only():
    """The clip bounds the mean axial Mach number in magnitude, not in direction.

    A station the flow leaves through has to stay reversed in the state the
    Jacobians are evaluated on, or both sides would be linearised about a flow
    running the other way. A station sitting exactly at zero has no direction to
    keep, so it takes the downstream one -- what it must not take is zero, which
    is the value the clip exists to keep out of the Jacobians.
    """
    grid, patch_up, patch_dn, comm = exchanged()

    # Both small enough that |Max| < Ma_clip, so both stations are clipped.
    for block in grid:
        Vx = block.Vx.copy()
        Vx[:, 2, :] = -0.2
        Vx[:, 4, :] = 0.0
        block.set_Vx(Vx)

    b_avg, _ = comm._prepare_pair(patch_up, patch_dn, flip=False)

    Ma_clip = MixingCommunicator.Ma_clip
    Max = np.asarray(b_avg.Max).ravel()
    assert Max[2] == pytest.approx(-Ma_clip, rel=1e-3)
    assert Max[4] == pytest.approx(Ma_clip, rel=1e-3)
    # The untouched stations are above the clip and left alone by it.
    assert Max[0] > Ma_clip
    assert Max[6] == pytest.approx(Max[0], rel=1e-5)


def test_sides_must_agree_on_rf_exchange():
    """One plane relaxes at one rate, so a pair holding two is a configuration error.

    The exchange writes a single shared target and reads the factor off the
    first side, so a disagreement would silently take one side's value.
    """
    grid, patch_up, patch_dn = make_pair()
    patch_up.rf_exchange = 0.5
    patch_dn.rf_exchange = 0.25

    with pytest.raises(ValueError, match="disagree on rf_exchange"):
        MixingCommunicator(grid, grid.connectivity.mixing.pair())


def test_rf_exchange_sets_the_rate_the_target_moves_at():
    """The factor is read off the patches, so changing it changes the exchange.

    Half the relaxation moves the target half as far from the same baseline, so
    the increment scales with it directly.
    """
    increments = {}
    for rf_exchange in (0.5, 0.25):
        grid, patch_up, patch_dn, comm = exchanged(
            rf_exchange=rf_exchange, up={"P": 1.05e5}, dn={"P": 0.95e5}
        )
        comm.exchange()
        ((key, _),) = comm.pairs.items()
        increments[rf_exchange] = comm.get_stats(*key)["du"]

    np.testing.assert_allclose(increments[0.25], 0.5 * increments[0.5], rtol=1e-5)


def test_rf_exchange_is_read_at_every_exchange_not_cached():
    """A communicator built before the value changed still picks the change up.

    This is what lets the solver retune a plane on a grid whose communicator was
    already built and cached by an earlier apply_bconds.
    """
    grid, patch_up, patch_dn, comm = exchanged(
        rf_exchange=0.5, up={"P": 1.05e5}, dn={"P": 0.95e5}
    )
    ((key, _),) = comm.pairs.items()

    comm.exchange()
    du_before = comm.get_stats(*key)["du"].copy()

    # Same baseline again, so only the factor differs between the two calls.
    grid2, _, _, comm2 = exchanged(rf_exchange=0.5, up={"P": 1.05e5}, dn={"P": 0.95e5})
    for patch in grid2.patches.mixing:
        patch.rf_exchange = 0.25
    comm2.exchange()
    ((key2, _),) = comm2.pairs.items()

    np.testing.assert_allclose(comm2.get_stats(*key2)["du"], 0.5 * du_before, rtol=1e-5)


def test_copy_carries_rf_exchange():
    """It is configuration, so it travels with the patch like the other settings."""
    _, patch_up, _ = make_pair()
    patch_up.rf_exchange = 0.123

    assert patch_up.copy().rf_exchange == pytest.approx(0.123)


def test_rf_exchange_survives_a_pickle_round_trip():
    """It lives on the patch precisely so a restart keeps it.

    The communicator that used to hold it is dropped by Grid.__getstate__ and
    rebuilt from its defaults, so a value held there would silently revert on
    every restart.
    """
    patch = MixingPatch(i=-1)
    patch.rf_exchange = 0.321

    assert pickle.loads(pickle.dumps(patch)).rf_exchange == pytest.approx(0.321)


def test_rf_exchange_back_fills_on_a_patch_pickled_without_it():
    """Old EMB files predate the attribute and must still load.

    BasePatch.__setstate__ runs _setup() before applying the pickled state for
    exactly this reason, so the default is in place before the old state lands
    on top of it.
    """
    _, patch_up, _ = make_pair()
    state = patch_up.__getstate__()
    del state["rf_exchange"]

    revived = MixingPatch(i=-1)
    revived.__setstate__(state)

    assert revived.rf_exchange == pytest.approx(0.02)


@pytest.mark.parametrize(
    "name", ["set_adjustment", "set_Alpha", "set_Beta", "set_P", "set_ho_s"]
)
def test_no_setters_of_its_own(name):
    """Every row comes from the exchange, so none of the conditions' setters exist.

    They used to be inherited and then overridden to raise, which meant a caller
    could reach a method that did nothing useful. There is nothing to override
    now: this class is not an inlet or an outlet, it is the condition plus an
    exchange.
    """
    grid, patch_up, patch_dn = make_pair()
    for patch in (patch_up, patch_dn):
        assert not hasattr(patch, name)


def test_angle_rows_are_not_addressable():
    """The mix target space carries velocities in rows 2-3, so the angles are not there."""
    grid, patch_up, patch_dn = make_pair()
    for name in ("tanAlpha", "sinBeta"):
        with pytest.raises(AttributeError, match="so it has no"):
            getattr(patch_dn, name)
    # And what is there instead reads back.
    assert patch_dn.Vr_nd is not None
    assert patch_dn.Vt_nd is not None


# Orientation, and settling the frame against the flow

# Axial, conical, radially outward, running backwards, radially inward. The
# frame a plane works in has to point along the through-flow whatever the
# orientation, and the geometry cannot say which way that is: both sides of a
# plane are the same class, and which is upstream is a property of the machine.
CHI_PLANE = [0.0, 30.0, 90.0, 180.0, 270.0]

# The coefficients of the harmonic relation live on each side of a plane: the
# upstream side is one the flow leaves and the downstream side one it enters.
LIVE_UP = ("coef_t", "coef_t_hilbert", "coef_down", "coef_down_hilbert")
LIVE_DN = ("coef_local", "coef_hilbert")


def _frame_normal_velocity(patch):
    """Pitch-mean velocity along the patch's frame axis, per span station."""
    with patch._resolved():
        patch.set_block_avg()
        return np.asarray(patch.block_avg.Vx_nd).reshape(-1).copy()


def _mdot_gap(patch_up, patch_dn):
    """Mass-flow mismatch across the plane, relative to the flow through it.

    Taken as the integral of rho V.dA over each face, which is frame-free: it
    dots the velocity with the face's own area vector rather than assuming a
    direction for it.
    """
    mdot_up = abs(float(average.flow_mass(patch_up.block_view.squeeze())))
    mdot_dn = abs(float(average.flow_mass(patch_dn.block_view.squeeze())))
    return (mdot_dn - mdot_up) / max(mdot_up, mdot_dn)


@pytest.mark.parametrize("chi", CHI_PLANE)
def test_the_frame_settles_along_the_flow_at_any_orientation(chi):
    """Both sides end up working in a frame that runs downstream.

    This is what keeps the plane non-reflecting rather than merely balanced.
    The harmonic relations are derived for mean flow along the frame axis and
    the condition zeroes the harmonics where the flow opposes it, so a frame
    pointing upstream would leave a plane that still matches the mean fluxes
    while reflecting every harmonic that reached it -- and at chi = 270 the
    provisional frame the geometry gives does point upstream.
    """
    grid, patch_up, patch_dn, comm = exchanged(chi=chi)
    comm.exchange()

    # Settled, and the two sides are still opposite, which is what pairing
    # needs of them.
    assert patch_up._sign_settled and patch_dn._sign_settled
    assert patch_up._sign_interior == -patch_dn._sign_interior

    # A plane is an outflow on the upstream side and an inflow on the
    # downstream one, whichever way the duct points.
    assert patch_up._sign_interior == -1
    assert patch_dn._sign_interior == 1

    # Both frames run with the flow, so the normal velocity is positive on
    # each: the through-flow leaves the upstream block and enters the
    # downstream one, and both read that as the same sign in their own frame.
    assert (_frame_normal_velocity(patch_up) > 0.0).all()
    assert (_frame_normal_velocity(patch_dn) > 0.0).all()


@pytest.mark.parametrize("chi", CHI_PLANE)
def test_each_side_runs_the_relation_its_flow_direction_calls_for(chi):
    """And only that one, at every orientation.

    The upstream side is one the flow leaves and runs Giles Eq. 5.32; the
    downstream side is one it enters and runs Eq. 5.17. Which is live follows
    from the settled frame, so this is the structural statement that the settle
    put both sides on the absorbing branch rather than the zeroed one.

    Only the branch is asserted, not the coefficients: unlike a single face
    turned in place, a turned *chain* is a different duct -- its two blocks sit
    at different radii, so a radial one is a diffuser where the axial one has
    constant area, and the mean states either side genuinely differ.
    """
    _, up, dn, comm = exchanged(chi=chi)
    relax((up, dn), comm, 3)

    for patch, live, dead in ((up, LIVE_UP, LIVE_DN), (dn, LIVE_DN, LIVE_UP)):
        assert set(live) <= set(patch._ref)
        assert not set(dead) & set(patch._ref)
        for key in live:
            coef = np.asarray(patch._ref[key])
            assert np.isfinite(coef).all()
        # The Hilbert coefficient is the non-reflecting part of the relation,
        # so a plane that absorbs harmonics has to carry a nonzero one.
        hilbert = "coef_hilbert" if live is LIVE_DN else "coef_t_hilbert"
        assert np.abs(np.asarray(patch._ref[hilbert])).min() > 1e-3


@pytest.mark.parametrize("chi", CHI_PLANE)
def test_a_turned_plane_conserves_mass_across_the_interface(chi):
    """Mass flow matches across the plane whatever the orientation.

    The pitch-mean fluxes themselves cannot be compared on a turned chain --
    the two sides have different annulus areas, so the same mass flow is
    carried at a different flux -- but the mass flow is what has to balance
    either way, and it is the quantity a canted or radial interface would get
    wrong if it were resolving the velocity onto the wrong normal.
    """
    grid, patch_up, patch_dn, comm = exchanged(
        chi=chi, up={"P": 1.05e5, "Vx": 105.0}, dn={"P": 0.95e5, "Vx": 95.0}
    )
    gap_before = abs(_mdot_gap(patch_up, patch_dn))
    assert gap_before > 1e-2

    relax((patch_up, patch_dn), comm, 100)

    gap_after = abs(_mdot_gap(patch_up, patch_dn))
    assert gap_after < 5e-3, f"{gap_before} -> {gap_after}"


def test_an_axial_chain_still_matches_its_mean_fluxes_exactly(chi=0.0):
    """The axial chain has the same geometry either side, so the fluxes match too.

    The stronger statement the turned cases cannot make, kept so the weaker one
    above is not the only thing standing behind Saxer Eq. 5.65.
    """
    grid, patch_up, patch_dn, comm = exchanged(
        chi=chi, up={"P": 1.05e5, "Vx": 105.0}, dn={"P": 0.95e5, "Vx": 95.0}
    )
    relax((patch_up, patch_dn), comm, 100)
    assert flux_gap(patch_up, patch_dn).max() < 1e-4


def test_a_grid_at_rest_does_not_freeze_an_arbitrary_frame():
    """With no through-flow there is no direction to settle against, so it waits.

    Freezing whichever way the geometry happened to point would be worse than
    waiting a step: a run started from rest would carry that frame for its
    whole life, and half the time it would be the reflecting one.
    """
    # Cross-flow but no through-flow, so the face carries no net mass flux.
    grid, patch_up, patch_dn, comm = exchanged(chi=270.0, Vx=0.0, Vr=20.0)
    comm.exchange()
    assert not patch_up._sign_settled
    assert not patch_dn._sign_settled

    # It still ran, and left the two sides opposite so the exchange is sound.
    assert patch_up._sign_interior == -patch_dn._sign_interior


def test_a_late_settle_turns_the_target_it_already_wrote():
    """A target seeded in the provisional frame is turned with the frame.

    The exchange runs before the boundary conditions do, so a grid started from
    rest can have a target written in one frame and then settle into the other.
    Row 2 of that target is the velocity in the surface, which reverses when the
    frame turns through pi; leaving it would drive the plane toward a
    cross-flow of the wrong sign.
    """
    grid, patch_up, patch_dn, comm = exchanged(chi=270.0, Vx=0.0, Vr=20.0)
    comm.exchange()
    assert not patch_up._sign_settled
    before = [np.copy(p._target[..., 2]) for p in (patch_up, patch_dn)]
    signs = [p._sign_interior for p in (patch_up, patch_dn)]
    # The seeded cross-flow is what makes the turn visible at all.
    assert np.abs(before[0]).max() > 1e-3

    # Start the flow and exchange again: now there is a direction to settle on.
    for block in grid:
        block.set_Vx(block.Vx + turn(100.0, 0.0, 270.0)[0] * np.ones(block.shape))
        block.set_Vr(block.Vr + turn(100.0, 0.0, 270.0)[1] * np.ones(block.shape))
    comm.exchange()

    assert patch_up._sign_settled and patch_dn._sign_settled
    assert patch_up._sign_interior == -1 and patch_dn._sign_interior == 1
    for patch, was, sign_was in zip((patch_up, patch_dn), before, signs):
        if patch._sign_interior != sign_was:
            # Turned, so what was written in the old frame was turned with it.
            # Compared against the seed rather than the current target, which
            # this exchange has also moved.
            assert np.sign(np.sum(patch._target[..., 2])) != np.sign(np.sum(was))


def test_the_frame_does_not_re_settle_when_a_station_reverses():
    """Reversal is the split's business, not the frame's.

    Moving the frame under a target already written in it would be a different
    and much worse thing than switching a station to the other characteristic
    split, which is what the condition does and has always done.
    """
    grid, patch_up, patch_dn, comm = exchanged(chi=90.0)
    comm.exchange()
    settled = [p._sign_interior for p in (patch_up, patch_dn)]

    # Reverse the through-flow outright on both blocks.
    for block in grid:
        block.set_Vx(-block.Vx)
        block.set_Vr(-block.Vr)
    relax((patch_up, patch_dn), comm, 5)

    assert [p._sign_interior for p in (patch_up, patch_dn)] == settled


# Physics


def test_matched_flow_is_a_fixed_point():
    """With the same state either side, the exchange and both conditions do nothing."""
    grid, patch_up, patch_dn, comm = exchanged()
    before = [patch.block_view.conserved_nd.copy() for patch in (patch_up, patch_dn)]

    relax((patch_up, patch_dn), comm, 20)

    for patch, start in zip((patch_up, patch_dn), before):
        assert np.allclose(patch.block_view.conserved_nd, start, atol=1e-5, rtol=1e-4)


def test_mismatch_relaxes_to_matched_mean_fluxes():
    """A cross-plane jump relaxes until the pitch-mean fluxes agree.

    This is the property Saxer Eq. 5.65 asserts: mass, momentum and energy
    fluxes match across the interface once the exchange has converged.
    """
    grid, patch_up, patch_dn, comm = exchanged(
        up={"P": 1.05e5, "Vx": 105.0}, dn={"P": 0.95e5, "Vx": 95.0}
    )
    gap_before = flux_gap(patch_up, patch_dn)
    assert gap_before.max() > 1e-2

    relax((patch_up, patch_dn), comm, 100)

    # An 18 percent mass-flux jump converges to float32 round-off, a few times
    # 1e-6, well inside this.
    gap_after = flux_gap(patch_up, patch_dn)
    assert gap_after.max() < 1e-4, f"{gap_before} -> {gap_after}"


def test_a_reversed_station_relaxes_like_any_other():
    """The plane converges with a span station reversed, as it does without one.

    The twin of :func:`test_mismatch_relaxes_to_matched_mean_fluxes`, differing
    only in the reversed station, and held to the same limit.

    It used to diverge outright: with the split frozen forward (pre-Change-1)
    the reversal at that station deepened by an order of magnitude, the target
    followed it, and the mean state went axially supersonic within about a
    dozen iterations -- the two-block form of what ended a LISA turbine run.
    With the split switched (Change 1) and the direction shared (Change 2) the
    target is correctly oriented, which is what lets the integrating form
    (Change 4) accumulate toward exact flux balance instead of the standing
    offset a proportional relaxation would leave; the physical clamp is the
    anti-windup net if a station still winds up. This needs the lower,
    integrator-scaled gain of :attr:`~ember.mixing.MixingPatch.rf_exchange`'s
    production default rather than the fast test gain most of this module
    uses -- the stiffer, direction-switched feedback of a reversed station
    does not tolerate ``RF_EXCHANGE_FAST``, only :func:`test_mismatch_relaxes_to_matched_mean_fluxes`'s
    unreversed one does.

    Reversed on both blocks, so the mean the exchange symmetrises is genuinely
    reversed there rather than merely small.
    """
    grid, patch_up, patch_dn, comm = exchanged(
        up={"P": 1.05e5, "Vx": 105.0},
        dn={"P": 0.95e5, "Vx": 95.0},
        rf_exchange=0.05,
    )
    for block in grid:
        Vx = block.Vx.copy()
        Vx[:, 3, :] = -20.0
        block.set_Vx(Vx)

    relax((patch_up, patch_dn), comm, 350)

    # The same limit the all-forward case converges to; that one reaches a few
    # times 1e-6, so this is not a tight ask.
    gap = flux_gap(patch_up, patch_dn)
    assert gap.max() < 1e-4, f"flux gap {gap}"
    # And the target stays a physical state rather than integrating away.
    assert np.abs(patch_up.get_target()).max() < 30.0


def test_exchange_leaves_harmonics_alone():
    """Only the pitch mean crosses the plane; the exchange must not touch harmonics."""
    grid, patch_up, patch_dn, comm = exchanged()
    patch_up.update_soln()

    wave = np.zeros(patch_up.shape + (5,), dtype=np.float32)
    phase = 2.0 * np.pi * patch_up.block_view.t / patch_up.block.pitch
    wave[..., 1] = 0.01 * np.cos(phase)
    seed_chic(patch_up, wave)

    seeded = patch_up.block_view.conserved_nd.copy()
    comm.exchange()
    assert np.array_equal(patch_up.block_view.conserved_nd, seeded)

    # And the target the exchange produced carries no pitchwise variation.
    assert np.ptp(patch_up.P_nd, axis=patch_up.pitch_dim).max() == 0.0


def test_harmonic_acoustic_is_absorbed():
    """An acoustic harmonic reaching the plane leaves no pressure harmonic behind.

    The inherited outflow relation reduces without swirl to
    ``c_up = -c_down + 2*Mn*H[c_t]/sqrt(1 - M^2)``, so a pure downstream-running
    acoustic harmonic is met by an equal and opposite upstream-running one and
    the pressure harmonic ``(c_up + c_down)/2`` cancels. Exercised here with the
    target coming from the exchange rather than from ``set_P``, which is the
    only thing this class changes.
    """
    grid, patch_up, patch_dn, comm = exchanged(sigma=1.0, Vt=0.0)
    relax((patch_up, patch_dn), comm, 5)

    wave = np.zeros(patch_up.shape + (5,), dtype=np.float32)
    phase = 2.0 * np.pi * patch_up.block_view.t / patch_up.block.pitch
    wave[..., 1] = 0.01 * np.cos(phase)
    seed_chic(patch_up, wave)

    amp_before = np.abs(harmonic(patch_up, patch_up.block_view.P_nd)).max()
    comm.exchange()
    patch_up.advance()
    patch_up.apply()
    amp_after = np.abs(harmonic(patch_up, patch_up.block_view.P_nd)).max()

    assert amp_after < 0.01 * amp_before, f"{amp_before} -> {amp_after}"


# Several planes in one grid


# Three blocks, the middle one carrying an inflow side and an outflow side at
# once. Blocks 0 and 1 match; block 2 is off, so plane 1 has work to do and
# plane 0 does not. That asymmetry is what the independence tests read.
CHAIN_MATCHED = [{}, {}, {}]
CHAIN_SKEWED = [{}, {}, {"P": 0.9e5, "Vx": 90.0}]


def test_chain_pairs_adjacent_blocks_only():
    """Each plane joins consecutive blocks; the ends do not pair through the middle."""
    grid, planes = make_chain(CHAIN_MATCHED)
    pairs = grid.connectivity.mixing.pair()

    # Block 1 owns two patches, appended inflow side first.
    assert pairs == {
        (0, 0): ((1, 0), False),
        (1, 0): ((0, 0), False),
        (1, 1): ((2, 0), False),
        (2, 0): ((1, 1), False),
    }
    # The far ends face the same way and are not a plane, whatever the geometry.
    patch_first_up = planes[0][0]
    patch_last_dn = planes[1][1]
    assert patch_first_up.check_match(patch_last_dn) is None


def test_chain_middle_block_sides_are_distinct():
    """The middle block's two patches hold separate targets, not one aliased buffer."""
    grid, planes = make_chain(CHAIN_MATCHED)
    dn_of_plane0 = planes[0][1]
    up_of_plane1 = planes[1][0]

    assert dn_of_plane0.block is up_of_plane1.block

    comm = communicator(grid)
    comm.exchange()

    # Seeded by the exchange, so the buffers now exist to be compared.
    assert dn_of_plane0._target is not up_of_plane1._target

    before = up_of_plane1.get_target().copy()
    dn_of_plane0._target[...] += 1.0
    assert np.array_equal(up_of_plane1.get_target(), before)


def test_chain_planes_are_independent():
    """A mismatch at one plane leaves the other plane's target untouched.

    The communicator carries one set of scratch buffers across every pair and
    keys its per-pair state by patch identity, so this is the test that a
    second plane in the grid cannot corrupt the first.
    """
    grid_matched, planes_matched = make_chain(CHAIN_MATCHED)
    communicator(grid_matched).exchange()

    grid_skewed, planes_skewed = make_chain(CHAIN_SKEWED)
    communicator(grid_skewed).exchange()

    # Plane 1 straddles the mismatch and must have moved.
    tgt_matched = planes_matched[1][1].get_target()
    tgt_skewed = planes_skewed[1][1].get_target()
    assert not np.allclose(tgt_skewed, tgt_matched, rtol=1e-3)

    # Plane 0 sees the same flow either way and must not have.
    for side in (0, 1):
        assert np.allclose(
            planes_skewed[0][side].get_target(),
            planes_matched[0][side].get_target(),
            rtol=1e-6,
            atol=1e-7,
        )


def test_chain_exchange_matches_planes_taken_one_at_a_time():
    """Exchanging both planes together gives what exchanging each alone gives.

    Run with a different pitchwise count on every block, so the two planes have
    different shapes and the shared scratch buffers are sliced differently on
    each pair -- the case where a leak between them would show up.
    """
    npitch = (17, 13, 9)
    grid_both, planes_both = make_chain(CHAIN_SKEWED, npitch=npitch)
    communicator(grid_both).exchange()

    for iplane in (0, 1):
        grid_one, planes_one = make_chain(CHAIN_SKEWED, npitch=npitch)
        # Exchange this plane alone by handing the communicator only its pair.
        all_pairs = grid_one.connectivity.mixing.pair()
        keys = [(iplane, 0), (iplane + 1, 0)] if iplane == 0 else [(1, 1), (2, 0)]
        one_pair = {k: all_pairs[k] for k in keys}
        for patch in grid_one.patches.mixing:
            patch.rf_exchange = RF_EXCHANGE_FAST
        MixingCommunicator(grid_one, one_pair).exchange()

        for side in (0, 1):
            assert np.allclose(
                planes_both[iplane][side].get_target(),
                planes_one[iplane][side].get_target(),
                rtol=1e-6,
                atol=1e-7,
            ), f"plane {iplane} side {side} changed when the other plane was present"


def test_chain_relaxes_every_plane():
    """Mismatches at both planes converge together to matched pitch-mean fluxes."""
    states = [{}, {"P": 1.05e5, "Vx": 105.0}, {"P": 0.95e5, "Vx": 95.0}]
    grid, planes = make_chain(states)
    comm = communicator(grid)
    patches = grid.patches.mixing

    gaps_before = [flux_gap(up, dn) for up, dn in planes]
    assert all(gap.max() > 1e-2 for gap in gaps_before)

    relax(patches, comm, 150)

    for iplane, (up, dn) in enumerate(planes):
        gap = flux_gap(up, dn)
        assert gap.max() < 1e-4, f"plane {iplane}: {gaps_before[iplane]} -> {gap}"


def test_chain_stats_are_kept_per_plane():
    """Diagnostics are keyed per pair, so two planes do not share one record."""
    grid, planes = make_chain(CHAIN_SKEWED)
    comm = communicator(grid)
    comm.exchange()

    keys = list(comm.pairs)
    assert len(keys) == 2
    du = [comm.get_stats(*key)["du"] for key in keys]
    # Plane 0 sits in matched flow and plane 1 does not, so their increments
    # cannot be the same record.
    assert not np.allclose(du[0], du[1])


@pytest.mark.parametrize("reflective", [False, True])
def test_solver_run_stays_finite(reflective):
    """A two-block run across the plane completes without NaN or non-physical values."""
    grid, patch_up, patch_dn = make_pair(npitch_up=9, npitch_dn=9, ni=9, nspan=9)
    grid[0].patches.append(InletPatch(i=0))
    grid[1].patches.append(OutletPatch(i=-1))
    for block in grid:
        block.set_wdist(0.0)
        for face in (0, -1):
            block.patches.append(PeriodicPatch(k=face))
    grid.set_L_ref(float(np.ptp(grid[0].x)))

    inlet = grid.patches.inlet[0]
    inlet.set_Po_To(float(grid[0].Po[0].mean()), float(grid[0].To[0].mean()))
    inlet.set_Alpha(0.0)
    inlet.set_Beta(0.0)
    grid.patches.outlet[0].set_P(float(grid[1].P[-1].mean()))
    grid.connectivity.periodic.pair()
    grid.connectivity.mixing.pair()

    ember.solver.Solver(
        n_step=20,
        n_step_avg=1,
        n_step_log=20,
        n_stage=4,
        mix_reflective=reflective,
    ).run(grid)

    for block in grid:
        assert np.all(np.isfinite(block.conserved)), "Non-finite conserved variables"
        assert np.all(block.rho > 0), "Non-positive density"
        assert np.all(block.P > 0), "Non-positive pressure"
        assert np.all(block.T > 0), "Non-positive temperature"


# Reflective mode


def reflective_pair(**kwargs):
    """A paired plane whose two sides impose the mixed-out state directly.

    The flag is set before anything pairs, since check_match reads it. These
    are unit tests on the patches and the communicator, with no march to stamp
    Solver.mix_reflective onto them, so they set the private attribute the
    solver would; test_solver_sets_reflective_on_every_plane covers that path.
    """
    grid, patch_up, patch_dn = make_pair(**kwargs)
    for patch in (patch_up, patch_dn):
        patch._reflective = True
    comm = MixingCommunicator(grid, grid.connectivity.mixing.pair())
    return grid, patch_up, patch_dn, comm


def ripple(patch, amp=0.05):
    """Scale the face by a pitchwise sinusoid, so its mean is not the whole story.

    Every conserved variable is scaled together, which is a pure density
    ripple: the state stays physical at any amplitude and the velocities are
    untouched, while the mass flux the face carries now varies around the pitch.
    """
    b = patch.block_view
    t = np.asarray(b.t)
    fac = 1.0 + amp * np.sin(2.0 * np.pi * (t - t.min()) / patch.block.pitch)
    b.conserved_nd[...] *= fac[..., None]
    b.update_cached_conserved()


def pitch_mean_cons(patch):
    """Circumferential mean of the face's conserved variables, ``(nspan, 5)``.

    Written out here rather than taken from set_block_avg, which is the call
    the code under test makes.
    """
    cons = np.asarray(patch.block_view.conserved_nd)
    w = np.asarray(patch.weight_pitch)[..., None]
    return (cons * w).sum(axis=patch.pitch_dim).squeeze()


def annulus_mass(patch):
    """Mass flow through the whole annulus at this face, as rho V.dA."""
    passage = float(average.flow_mass(patch.block_view.squeeze()))
    return abs(passage) * patch.block.Nb


def test_reflective_does_not_pair_with_the_default_plane():
    """The exchange is one thing or the other, so a pair cannot be half of each."""
    grid, patch_up, patch_dn = make_pair()
    patch_up._reflective = True
    assert patch_up.check_match(patch_dn) is None
    assert patch_dn.check_match(patch_up) is None

    # And with both sides agreeing it pairs exactly as before.
    patch_dn._reflective = True
    assert grid.connectivity.mixing.pair() == {
        (0, 0): ((1, 0), False),
        (1, 0): ((0, 0), False),
    }


def test_reflective_faces_hold_one_common_uniform_state():
    """Both faces come out of apply() pitch-uniform, equal, and hub to casing.

    Nothing is extrapolated at the ends and nothing is relaxed: every node of
    both faces holds the span station's mixed-out state outright.
    """
    grid, patch_up, patch_dn, comm = reflective_pair(
        up={"P": 1.05e5, "Vx": 105.0}, dn={"P": 0.95e5, "Vx": 95.0}
    )
    ripple(patch_up)
    ripple(patch_dn, amp=-0.03)

    comm.exchange()
    patch_up.apply()
    patch_dn.apply()

    cons_up = patch_up.block_view.conserved_nd
    cons_dn = patch_dn.block_view.conserved_nd
    for patch, cons in ((patch_up, cons_up), (patch_dn, cons_dn)):
        first = np.take(cons, [0], axis=patch.pitch_dim)
        assert np.allclose(cons, first, rtol=0.0, atol=0.0), "face is not pitch-uniform"

    # The two sides carry different pitchwise resolutions, so compare the one
    # value per span station each of them now holds -- ends included.
    span_up = patch_up.get_uniform()
    span_dn = patch_dn.get_uniform()
    assert span_up.shape == (patch_up.shape[patch_up.span_dim], 5)
    assert np.allclose(span_up, span_dn, rtol=1e-6)


def test_reflective_state_is_the_mean_of_the_two_circumferential_means():
    """The imposed state is exactly the average of the two sides' pitch means."""
    grid, patch_up, patch_dn, comm = reflective_pair(
        up={"P": 1.05e5, "Vx": 105.0}, dn={"P": 0.95e5, "Vx": 95.0}
    )
    ripple(patch_up)

    mean_up = pitch_mean_cons(patch_up)
    mean_dn = pitch_mean_cons(patch_dn)
    comm.exchange()

    assert np.allclose(patch_up.get_uniform(), 0.5 * (mean_up + mean_dn), rtol=1e-6)


def test_reflective_plane_conserves_mass_across_it():
    """Both faces pass the same annulus mass flow, to round-off, in one exchange.

    Not a convergence statement, unlike the same test on the default plane: the
    face flow is built from the boundary nodes and the face areas alone, so two
    faces holding the same pitch-uniform state pass the same flow immediately
    and identically.
    """
    grid, patch_up, patch_dn, comm = reflective_pair(
        up={"P": 1.05e5, "Vx": 105.0}, dn={"P": 0.95e5, "Vx": 95.0}
    )
    ripple(patch_up)
    assert abs(_mdot_gap(patch_up, patch_dn)) > 1e-2

    comm.exchange()
    patch_up.apply()
    patch_dn.apply()

    mdot_up = annulus_mass(patch_up)
    mdot_dn = annulus_mass(patch_dn)
    assert abs(mdot_dn - mdot_up) / mdot_up < 1e-6, f"{mdot_up} vs {mdot_dn}"


def test_reflective_uniformising_keeps_the_mass_flow_of_the_face():
    """Area-averaging the conserved variables leaves the face's own mass flow alone.

    The face mass flux is linear in the conserved vector and the pitch weights
    are the quadrature the face quads use, so the mixed-out state carries the
    mass flow the pitchwise-varying one did. That is what makes the plane safe
    to hard-reset onto: it does not move the row's mass flow by uniformising
    it. The momentum and energy fluxes are quadratic and do move, which is why
    the state imposed is not a flux-conserving mixed-out state.
    """
    grid, patch_up, patch_dn = make_pair()
    patch_up._reflective = True
    ripple(patch_up, amp=0.2)
    before = annulus_mass(patch_up)

    patch_up.apply()  # seeds from its own pitch mean and imposes it
    assert abs(annulus_mass(patch_up) - before) / before < 1e-6


def test_reflective_apply_seeds_from_its_own_mean_before_any_exchange():
    """A face applied before the first exchange imposes its own mixed-out state.

    apply() runs every stage and the exchange only once a step, and a frozen
    averaging window skips the exchange entirely, so the face must never be
    left imposing the zeros it was allocated with.
    """
    grid, patch_up, patch_dn = make_pair()
    patch_up._reflective = True
    ripple(patch_up)

    expect = pitch_mean_cons(patch_up)
    patch_up.apply()
    assert np.allclose(patch_up.get_uniform(), expect, rtol=1e-6)


def test_reflective_plane_freezes_no_reference_state():
    """None of the characteristic machinery is built or stepped in this mode.

    update_soln and advance are the two per-timestep entry points that build
    and step the frozen reference the non-reflecting condition works from; a
    reflective plane has no such state, and running them would freeze a
    reference and settle a frame that nothing then reads.
    """
    grid, patch_up, patch_dn, comm = reflective_pair()
    comm.exchange()
    for patch in (patch_up, patch_dn):
        patch.update_soln()
        patch.advance()
        patch.apply()
        assert patch._ref is None
        assert not patch._sign_settled
        # The side is still known from the geometry, which is what pairing and
        # the row stations read.
        assert patch._sign_interior in (-1, 1)


def test_reflective_flag_survives_copy_and_pickle():
    """A copied or reloaded patch keeps its treatment.

    Both matter: the multigrid hierarchy copies patches onto each coarse grid,
    and a run restarted from an EMB file unpickles them.
    """
    grid, patch_up, patch_dn, comm = reflective_pair()
    clone = patch_up.copy()
    assert clone._reflective

    # Unattached, as the round trip in test_rf_exchange_survives_a_pickle_round_trip:
    # a patch holds a weak reference to its block, and the grid's own pickle
    # drops and re-attaches it.
    loose = MixingPatch(i=-1)
    loose._reflective = True
    assert pickle.loads(pickle.dumps(loose))._reflective


def test_reflective_flag_back_fills_on_a_patch_pickled_without_it():
    """A patch from before the flag existed reloads as the default plane."""
    _, patch_up, _ = make_pair()
    state = patch_up.__getstate__()
    del state["_reflective"]

    revived = MixingPatch(i=-1)
    revived.__setstate__(state)
    assert not revived._reflective


def test_reflective_flag_migrates_from_the_public_name():
    """A patch pickled while the flag was public reloads as a reflective plane.

    Nothing reads the public name now, so without the migration the plane
    would come back as a Saxer one and only the answer would say so.
    """
    _, patch_up, _ = make_pair()
    state = patch_up.__getstate__()
    del state["_reflective"]
    state["reflective"] = True

    revived = MixingPatch(i=-1)
    revived.__setstate__(state)
    assert revived._reflective


def test_solver_sets_reflective_on_every_plane():
    """Solver.mix_reflective is imposed on both sides, and overrides the patches.

    The stamp is what makes the setting solver-wide: the patch and the
    communicator read the flag from places that never see a Solver, and both
    sides of a plane have to agree for check_match to pair them.
    """
    grid, patch_up, patch_dn = make_pair()

    conf = ember.solver.Solver(n_step=1, mix_reflective=True)
    ember.solver._apply_bcond_relaxation(grid, conf)
    assert patch_up._reflective and patch_dn._reflective

    # The default is imposed, not merely offered, as for rf_inlet.
    ember.solver._apply_bcond_relaxation(grid, ember.solver.Solver(n_step=1))
    assert not patch_up._reflective and not patch_dn._reflective

    # None leaves whatever the patches carry.
    patch_up._reflective = patch_dn._reflective = True
    conf = ember.solver.Solver(n_step=1, mix_reflective=None)
    ember.solver._apply_bcond_relaxation(grid, conf)
    assert patch_up._reflective and patch_dn._reflective
