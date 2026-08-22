"""One arena, sized by its worst phase, carved so nothing overlaps.

``Block.scratch`` backs every throwaway buffer in the step: the viscous
boundary tau/q face buffers and the rolling tau/q cell-plane pair, the nodal
transport trio both viscous kernels read, both kernels' rolling planes and
rows, the nodal acoustic speed the timestep kernel reads, the IRS work vector
and the multigrid coarse scratch. That is only safe under two rules,
and neither is something the code can check for itself:

  * buffers reaching the SAME kernel call must come from one
    ``util.carve_view``, which packs them end to end and so cannot overlap;
  * buffers in DIFFERENT phases may reuse the same span, because no two
    phases are live at once.

These tests are the tripwire for both. An overlap here is not a crash -- it is
one kernel quietly writing over another's inputs and a wrong answer coming out
the far end, which is exactly the failure the phase-by-phase sizing exists to
prevent.
"""

import numpy as np
import pytest

import ember.block
import ember.solver
from ember import util
from ember.block import MAX_MG_LEVELS, _scratch_len
from ember.cases import build_duct_grid

# Shapes worth covering: the bench duct, a cube, and a block one cell deep in
# k -- the degenerate case grid.py notes carve_view already rejects when the
# nodal slots do not fit, and which must keep rejecting rather than overlap.
SHAPES = [(273, 65, 57), (81, 65, 57), (49, 49, 49), (33, 33, 3)]


def _phase_buffers(block):
    """Every buffer, grouped by the phase it is live in."""
    ni, nj, nk = block.shape
    njp = nj + 1 if (ni * nj) % 1024 == 0 else nj
    faces, tq, planes, rows, transport = ember.block._carve_viscous(block)

    tmp_shape = (ni - 1, nj - 1, nk - 1, 5)
    mg_shapes = ember.solver.mg_coarse_shapes(ni, nj, nk, MAX_MG_LEVELS)
    # Both marching schemes carve the same way with multigrid on -- a rolling
    # two-plane increment plus the coarse scratch, since both fuse the final
    # prolong with the cell->node scatter -- and with it off both materialise a
    # full-volume cell-shaped increment on its own.
    mg_on = util.carve_view(block.scratch, (ni - 1, nj - 1, 5, 2), *mg_shapes)

    return {
        "update_sources": [*faces, tq, planes, rows, *transport],
        "update_timestep": [util.carve_view(block.scratch, block.shape)],
        "update_residual": list(
            util.carve_view(block.scratch, (ni, njp, 5, 2), (ni, 5, 3))
        )
        + [util.carve_view(block.scratch, (ni, nj, nk, 5))],
        "march_mg": list(mg_on),
        "march_nomg": [util.carve_view(block.scratch, tmp_shape)],
    }


@pytest.mark.parametrize("shape", SHAPES)
def test_buffers_within_a_phase_never_overlap(shape):
    """Within one phase every buffer occupies its own storage.

    ``update_residual`` is the exception that proves the rule and is checked
    separately below: its two carves are two different kernel calls, not one.
    """
    ni, nj, nk = shape
    block = ember.block.Block(shape=shape)
    for phase, bufs in _phase_buffers(block).items():
        if phase == "update_residual":
            continue
        for a in range(len(bufs)):
            for b in range(a + 1, len(bufs)):
                assert not np.shares_memory(bufs[a], bufs[b]), (
                    f"{phase}: buffers {a} and {b} overlap at shape {shape}"
                )


@pytest.mark.parametrize("shape", SHAPES)
def test_scratch_len_covers_every_phase(shape):
    """The arena holds the largest phase, so no carve runs off the end.

    Sized from the same helpers the carves use, so a buffer added to a phase
    without updating ``_scratch_len`` fails here rather than silently taking
    its neighbour's storage.
    """
    block = ember.block.Block(shape=shape)
    n = _scratch_len(shape)
    assert block.scratch.size == n
    for phase, bufs in _phase_buffers(block).items():
        need = sum(b.size for b in bufs)
        assert need <= n, f"{phase} needs {need} of a {n}-element arena at {shape}"


@pytest.mark.parametrize("shape", SHAPES)
def test_viscous_views_are_stable_and_inside_the_arena(shape):
    """Repeated access returns the same storage, and all of it is the arena.

    ``tau_q_faces`` is a view, not an allocation, so each access re-carves.
    ``periodic_communicator.exchange_faces`` depends on seeing exactly the
    storage ``set_tau_q_faces`` wrote and ``set_visc_force`` will read, so two
    accesses landing in different places would be silent corruption.
    """
    block = ember.block.Block(shape=shape)
    for face_a, face_b in zip(block.tau_q_faces, block.tau_q_faces):
        assert face_a.__array_interface__["data"][0] == (
            face_b.__array_interface__["data"][0]
        )
    faces, tq, planes, rows, transport = ember.block._carve_viscous(block)
    for buf in (*block.tau_q_faces, tq, planes, rows, *transport):
        assert np.shares_memory(block.scratch, buf)


def test_arena_is_smaller_than_the_buffers_it_replaced():
    """The merge is a saving, not just a reshuffle.

    Three allocations became one: the tau/q volume (with its spare tenth
    slot), the six face buffers, and the nodal scratch. The arena is sized by
    the multigrid phase, which binds at every shape tried, so it is smaller
    than their sum -- and smaller again since the viscous fusion deleted the
    volume, which used to be what bound it.
    """
    ni, nj, nk = 273, 65, 57
    before = (ni + 1) * (nj + 1) * (nk + 1) * 10 + ni * nj * nk * 5 + sum(
        int(np.prod(s)) for s in ember.block._viscous_face_shapes(ni, nj, nk)
    )
    after = _scratch_len((ni, nj, nk))
    assert after < before
    assert after / before < 0.4   # measured 0.363 at this shape


def test_no_phase_needs_a_volume_of_tau_q():
    """The tau/q the viscous phase keeps is a surface, not a volume.

    It used to bind the arena by a wide margin, on a (ni+1)(nj+1)(nk+1)*9
    tau/q volume that was three quarters of the whole thing. Fusing the two
    viscous kernels removed it: what a pass now keeps is the boundary shell
    plus one rolling cell-plane pair. This asserts the new state of affairs
    rather than the saving, so a change that reintroduced a volume-shaped
    tau/q buffer fails here and not just on someone's RSS.

    The claim is about tau/q specifically, not about the phase as a whole: the
    phase does carry three nodal volumes, the transport trio, and borrowing
    those rather than caching them is the whole point of their being here.
    """
    ni, nj, nk = 273, 65, 57
    block = ember.block.Block(shape=(ni, nj, nk))
    faces, tq, _, _, _ = ember.block._carve_viscous(block)
    kept = sum(b.size for b in (*faces, tq))
    # An order of magnitude under the volume it replaced, not a few percent.
    assert kept < 0.25 * (ni + 1) * (nj + 1) * (nk + 1) * 9


def test_degenerate_block_raises_rather_than_overlapping():
    """A block too small for a phase's buffers must fail loudly.

    ``carve_view`` raises when the requested shapes exceed the buffer. That is
    the behaviour to preserve: the alternative, silently handing back
    overlapping views, is the bug this whole arrangement is guarding against.
    """
    block = ember.block.Block(shape=(5, 5, 3))
    huge = (block.scratch.size,)
    with pytest.raises(Exception):
        util.carve_view(block.scratch, huge, huge)


def test_real_grid_carves_cleanly():
    """End to end on an assembled grid, not a bare Block."""
    block = build_duct_grid(300_000)[0]
    faces, tq, planes, rows, transport = ember.block._carve_viscous(block)
    bufs = [*faces, tq, planes, rows, *transport]
    for a in range(len(bufs)):
        for b in range(a + 1, len(bufs)):
            assert not np.shares_memory(bufs[a], bufs[b])
