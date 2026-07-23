"""Tests for the non-reflecting mixing plane.

Modules tested: ember.mixing_nonreflecting,
ember.mixing_communicator.NonReflectingMixingCommunicator

The two patch classes add no numerics of their own -- the characteristic split,
the Hilbert transform, the mean-mode Newton step and the harmonic relations are
all inherited from the non-reflecting inlet and outlet, and are tested in
test_nonreflecting.py, test_inlet_nonreflecting.py and
test_outlet_nonreflecting.py. What is left here is the mixing plane itself.

Test cases:
- Pairing: the two sides pair across the plane, same-side and foreign patches do
  not, differing pitchwise resolution is allowed
- Collections: both sides appear under mixing_nonreflecting and not under the
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
"""

import numpy as np
import pytest

import ember.solver
from ember.grid import Grid
from ember.mixing_communicator import NonReflectingMixingCommunicator
from ember.patch import (
    InletPatch,
    NonReflectingMixingPatch,
    NonReflectingOutletPatch,
    OutletPatch,
    PeriodicPatch,
)
from nonreflecting_util import harmonic, make_block, seed_chic

# The exchange and the boundary conditions are both heavily under-relaxed by
# default, which is right for a solver run and far too slow for a test that
# iterates the boundary alone. These drive the same fixed point, faster.
RF_MIX_FAST = 0.5
SIGMA_FAST = 0.5


def make_chain(states, npitch=17):
    """Blocks butted end to end, each junction a non-reflecting mixing plane.

    One entry in ``states`` per block, a dict of overrides for
    :func:`nonreflecting_util.make_block`, so each block can start at a
    different flow state and the planes between them see a genuine mismatch.
    ``npitch`` is a scalar or one value per block; the two sides of a plane may
    differ in it, since only pitch means cross.

    Returns the grid and a list of ``(outflow side, inflow side)`` pairs, one
    per plane, ordered upstream to downstream.
    """
    npitches = [npitch] * len(states) if np.isscalar(npitch) else list(npitch)

    blocks = []
    x_next = 0.0
    for state, npitch_block in zip(states, npitches, strict=True):
        block = make_block(npitch=npitch_block, **state)
        block.set_x(block.x - block.x.min() + x_next)
        x_next += float(np.ptp(block.x))
        blocks.append(block)

    planes = []
    for i, (block_up, block_dn) in enumerate(zip(blocks[:-1], blocks[1:])):
        patch_up = NonReflectingMixingPatch(i=-1, label=f"plane{i}_up")
        patch_dn = NonReflectingMixingPatch(i=0, label=f"plane{i}_dn")
        block_up.patches.append(patch_up)
        block_dn.patches.append(patch_dn)
        planes.append((patch_up, patch_dn))

    return Grid(blocks), planes


def make_pair(npitch_up=17, npitch_dn=17, up=None, dn=None, **kwargs):
    """Two blocks joined by one non-reflecting mixing plane.

    ``up`` and ``dn`` are per-side overrides of the flow state, so the two
    sides can be started mismatched. Returns the grid, the upstream (outflow)
    patch and the downstream (inflow) patch.
    """
    grid, planes = make_chain(
        [{**kwargs, **(up or {})}, {**kwargs, **(dn or {})}],
        npitch=(npitch_up, npitch_dn),
    )
    ((patch_up, patch_dn),) = planes
    return grid, patch_up, patch_dn


def communicator(grid, rf_mix=RF_MIX_FAST, sigma=SIGMA_FAST):
    """Communicator for every plane in a grid, with the patches sped up."""
    comm = NonReflectingMixingCommunicator(
        grid, grid.connectivity.mixing_nonreflecting.pair(), rf_mix=rf_mix
    )
    for patch in grid.patches.mixing_nonreflecting:
        patch.sigma = sigma
    return comm


def exchanged(*args, rf_mix=RF_MIX_FAST, sigma=SIGMA_FAST, **kwargs):
    """A single paired plane plus its communicator, ready to exchange."""
    grid, patch_up, patch_dn = make_pair(*args, **kwargs)
    return grid, patch_up, patch_dn, communicator(grid, rf_mix, sigma)


def relax(patches, comm, n_iter):
    """Iterate the exchange and every boundary condition to the fixed point."""
    for _ in range(n_iter):
        comm.exchange()
        for patch in patches:
            patch.update_soln()
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
    pairs = grid.connectivity.mixing_nonreflecting.pair()
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
    assert grid.connectivity.mixing_nonreflecting.pair()


def test_same_side_patches_do_not_pair():
    """Two outflow sides face the same way, so they are not two sides of a plane."""
    grid, patch_up, _ = make_pair()
    other = NonReflectingMixingPatch(i=-1, label="mix_other")
    grid[1].patches.append(other)
    assert patch_up.check_match(other) is None


def test_plain_nonreflecting_patch_does_not_pair():
    """A prescribed-pressure outlet is not half of a mixing plane."""
    grid, _, patch_dn = make_pair()
    plain = NonReflectingOutletPatch(i=-1, label="plain")
    grid[0].patches.append(plain)
    assert patch_dn.check_match(plain) is None
    assert plain.check_match(patch_dn) is None


# Collections and grid wiring


def test_collections_separate_from_plain_nonreflecting():
    """Both sides list under mixing_nonreflecting and under neither plain list.

    The plain lists drive Grid.apply_bconds and the inlet-row search in
    Grid._order_row_groups, so a mixing face leaking into them would be applied
    twice per stage and could misidentify which row is first.
    """
    grid, patch_up, patch_dn = make_pair()
    assert grid.patches.mixing_nonreflecting == [patch_up, patch_dn]
    assert grid.patches.inlet_nonreflecting == []
    assert grid.patches.outlet_nonreflecting == []
    assert grid.patches.mixing == []
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

    (up0, dn0), (up1, dn1) = grid.row_station_bid_pid
    # The upstream row's mixing face is an outflow side, so its exit station.
    assert (0, 0) in dn0
    # The downstream row's mixing face is an inflow side, so its inlet station.
    assert (1, 0) in up1


def test_communicator_is_the_nonreflecting_one():
    """Connectivity builds the bcond-space exchange for this patch type."""
    grid, _, _ = make_pair()
    comm = grid.connectivity.mixing_nonreflecting._get_communicator()
    assert isinstance(comm, NonReflectingMixingCommunicator)


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


def test_reversed_station_takes_its_inflow_state_from_the_exchange():
    """The outflow side needs nothing configured to carry a reversed station.

    Rows 0-3 of the exchanged target are the four quantities a reversed station
    has to be given, so it is driven toward the flow standing on the other side
    of the plane -- which is where the flow entering through it comes from.
    """
    grid, patch_up, patch_dn, comm = exchanged(dn={"Vt": 90.0})
    comm.exchange()

    # Reverse one span station of the upstream block's exit face and its
    # interior neighbour, so the outflow side has to switch its split there.
    block = grid[0]
    Vx = block.Vx.copy()
    Vx[-2:, 3, :] = -20.0
    block.set_Vx(Vx)

    patch_up.update_soln()
    assert patch_up._entering[3]
    # Nothing was prescribed and nothing was seeded either: the exchange had
    # already filled every row, so the seed stood aside.
    target = patch_up.get_target()
    np.testing.assert_allclose(
        [float(row.ravel()[3]) for row in patch_up._backflow()], target[3, 0:4], rtol=0
    )

    for _ in range(60):
        patch_up.update_soln()
        patch_up.apply()

    b = patch_up.block_view
    got = [
        float(patch_up._pitch_mean(f).ravel()[3])
        for f in (b.ho_nd, b.s_nd, b.Vr_nd, b.Vt_nd)
    ]
    np.testing.assert_allclose(got, patch_up.get_target()[3, 0:4], rtol=5e-3)


def test_reversed_station_on_the_inflow_side_takes_the_exchanged_pressure():
    """The downstream side carries reversal too, which is what it never used to.

    A stalled or separated row pushes flow back upstream through the interface,
    and the downstream side of a mixing plane is exactly where that shows up.
    Four of that station's characteristics turn outgoing and one quantity is
    left to prescribe -- static pressure, which is row 4 of the same exchanged
    target the upstream side reads.
    """
    grid, patch_up, patch_dn, comm = exchanged()
    comm.exchange()

    # Reverse one span station of the downstream block over its whole axial
    # extent, hard enough to stay reversed: raising the static pressure at a
    # station the flow leaves raises its axial velocity with it.
    block = grid[1]
    Vx = block.Vx.copy()
    Vx[:, 3, :] = -100.0
    block.set_Vx(Vx)

    patch_dn.update_soln()
    assert not patch_dn._entering[3]
    assert patch_dn._entering[[0, 1, 2, 4, 5, 6]].all()

    # Only c_down still enters that station; every other characteristic is the
    # interior's, where at a forward station only c_up would be.
    mask = np.broadcast_to(patch_dn._mask_out, patch_dn.shape + (5,))
    np.testing.assert_array_equal(mask[0, 3, 0], [True, False, True, True, True])
    np.testing.assert_array_equal(mask[0, 0, 0], [True, False, False, False, False])

    for _ in range(60):
        patch_dn.update_soln()
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

    Max = np.asarray(b_avg.Max).ravel()
    assert Max[2] == pytest.approx(-0.01, rel=1e-3)
    assert Max[4] == pytest.approx(0.01, rel=1e-3)
    # The untouched stations are above the clip and left alone by it.
    assert Max[0] > 0.01
    assert Max[6] == pytest.approx(Max[0], rel=1e-5)


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
    pairs = grid.connectivity.mixing_nonreflecting.pair()

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
        all_pairs = grid_one.connectivity.mixing_nonreflecting.pair()
        keys = [(iplane, 0), (iplane + 1, 0)] if iplane == 0 else [(1, 1), (2, 0)]
        one_pair = {k: all_pairs[k] for k in keys}
        NonReflectingMixingCommunicator(
            grid_one, one_pair, rf_mix=RF_MIX_FAST
        ).exchange()

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
    patches = grid.patches.mixing_nonreflecting

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


def test_solver_run_stays_finite():
    """A two-block run across the plane completes without NaN or non-physical values."""
    grid, patch_up, patch_dn = make_pair(npitch_up=9, npitch_dn=9, ni=9, nspan=9)
    grid[0].patches.append(InletPatch(i=0))
    grid[1].patches.append(OutletPatch(i=-1))
    for block in grid:
        block.set_wdist(0.0)
        for face in (0, -1):
            block.patches.append(PeriodicPatch(k=face))
    grid.set_L_ref(float(np.ptp(grid[0].x)))

    grid.patches.inlet[0].set_Po_To_Alpha_Beta(
        float(grid[0].Po[0].mean()), float(grid[0].To[0].mean()), 0.0, 0.0
    )
    grid.patches.outlet[0].set_P(float(grid[1].P[-1].mean()))
    grid.connectivity.periodic.pair()
    grid.connectivity.mixing_nonreflecting.pair()

    ember.solver.Solver(n_step=20, n_step_avg=1, n_step_log=20, n_stage=4).run(grid)

    for block in grid:
        assert np.all(np.isfinite(block.conserved)), "Non-finite conserved variables"
        assert np.all(block.rho > 0), "Non-positive density"
        assert np.all(block.P > 0), "Non-positive pressure"
        assert np.all(block.T > 0), "Non-positive temperature"
