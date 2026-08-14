"""Tests for Block.freeze(), which returns a read-only block.

A frozen block is a value: the flow field and the metadata it was built with
can no longer be changed, so a converged or designed state can be handed on
without a caller being able to edit it in place.

Frozen-ness tracks ``_data.flags.writeable`` rather than being a flag of its
own, which is what makes the two properties below hold automatically:

* every view shares the backing array, so slices, ``flat`` and the transposes
  are frozen too, including the writeable-by-design ``conserved_nd`` hot path
  that no Python-level guard would cover;
* ``copy()`` and ``empty()`` allocate a new array, so they are writeable --
  which is how a frozen design becomes the basis of a mixed-out result.

Test cases:
- test_a_new_block_is_not_frozen: freeze() is opt-in, nothing else changes
- test_freeze_preserves_data: values survive freezing
- test_freeze_preserves_metadata: fluid, L_ref, Nb, label, patches survive
- test_freeze_returns_an_independent_block: the original stays writeable
- test_frozen_data_setter_raises: set_* on flow data is refused, and says how
- test_frozen_metadata_setter_raises: set_* on metadata is refused
- test_frozen_backing_array_is_read_only: the array flag itself is set
- test_frozen_conserved_nd_is_read_only: the solver hot-path view is covered
- test_views_of_a_frozen_block_are_frozen: slices and reshapes stay frozen
- test_setting_through_a_view_of_a_frozen_block_raises: and cannot be written
- test_derived_properties_are_still_readable: reading caches, so it must work
- test_clear_cache_is_allowed: a cache is not data
- test_copy_of_a_frozen_block_is_writeable: copy() thaws
- test_copy_without_patches_of_a_frozen_block_is_writeable: so does this one
- test_empty_from_a_frozen_block_is_writeable: as does empty()
- test_mean_of_a_frozen_block_is_not_frozen: a reduction owns its data
- test_freeze_is_idempotent: freezing twice is harmless
- test_frozen_survives_a_pickle_round_trip: the guarantee outlives the file
"""

import pickle

import numpy as np
import pytest

import ember.fluid


@pytest.fixture
def block(small_block):
    """A block with every kind of metadata set, ready to be frozen."""
    # small_block sets coordinates and a P, T state but no velocity, so the
    # momentum variables would read as uninitialised.
    small_block.set_Vx(50.0)
    small_block.set_Vr(0.0)
    small_block.set_Vt(20.0)
    small_block.set_Nb(37)
    small_block.set_label("rotor")
    small_block.set_L_ref(0.1)
    return small_block


@pytest.fixture
def frozen(block):
    return block.freeze()


#
# THE BASICS
#


def test_a_new_block_is_not_frozen(block):
    """freeze() is opt-in: every existing path is untouched by it."""
    assert not block.frozen

    block.set_Vx(10.0)  # does not raise


def test_freeze_preserves_data(block, frozen):
    assert frozen.frozen
    assert frozen.shape == block.shape
    np.testing.assert_array_equal(frozen.conserved, block.conserved)
    np.testing.assert_array_equal(frozen.x, block.x)


def test_freeze_preserves_metadata(block, frozen):
    assert frozen.Nb == block.Nb
    assert frozen.label == block.label
    assert frozen.L_ref == block.L_ref
    assert len(frozen.patches) == len(block.patches)
    # Po needs the equation of state, so it raises if the fluid was not carried
    # across -- a functional check rather than an identity one.
    np.testing.assert_array_equal(frozen.Po, block.Po)


def test_freeze_returns_an_independent_block(block, frozen):
    """The design code holding the original cannot reach into the frozen value."""
    before = np.array(frozen.Vx)

    block.set_Vx(block.Vx + 100.0)

    np.testing.assert_array_equal(frozen.Vx, before)
    assert not block.frozen


#
# WHAT IS REFUSED
#


def test_frozen_data_setter_raises(frozen):
    """The message has to say what to do instead, not just that it failed."""
    with pytest.raises(ValueError, match="frozen Block") as excinfo:
        frozen.set_Vx(10.0)

    assert "copy()" in str(excinfo.value)


@pytest.mark.parametrize(
    "write",
    [
        lambda b: b.set_fluid(
            ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
        ),
        lambda b: b.set_L_ref(0.2),
        lambda b: b.set_Nb(41),
        lambda b: b.set_Omega(100.0),
        lambda b: b.set_label("stator"),
        lambda b: b.set_triangulated(False),
    ],
    ids=["fluid", "L_ref", "Nb", "Omega", "label", "triangulated"],
)
def test_frozen_metadata_setter_raises(frozen, write):
    """Metadata is shared by every view, so freezing data alone is not enough."""
    with pytest.raises(ValueError, match="frozen Block"):
        write(frozen)


def test_frozen_backing_array_is_read_only(frozen):
    assert not frozen._data.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        frozen._data[:] = 0.0


def test_frozen_conserved_nd_is_read_only(frozen):
    """`conserved_nd` is a deliberately writeable view for the solver.

    It bypasses every setter, so it is the one place a guard written in Python
    would leave open. The array flag closes it.
    """
    with pytest.raises(ValueError, match="read-only"):
        frozen.conserved_nd[:] = 0.0


#
# VIEWS
#


def _views(block):
    return {
        "getitem": block[0],
        "getitem_slice": block[1:3],
        "flat": block.flat,
        "transpose": block.transpose(),
        "squeeze": block[0:1].squeeze(),
        "flip": block.flip(0),
        "view": block.view(),
    }


VIEWS = ["getitem", "getitem_slice", "flat", "transpose", "squeeze", "flip", "view"]


@pytest.mark.parametrize("name", VIEWS)
def test_views_of_a_frozen_block_are_frozen(frozen, name):
    assert _views(frozen)[name].frozen


@pytest.mark.parametrize("name", VIEWS)
def test_setting_through_a_view_of_a_frozen_block_raises(frozen, name):
    with pytest.raises(ValueError, match="frozen"):
        _views(frozen)[name].set_Vx(10.0)


#
# WHAT IS STILL ALLOWED
#


def test_derived_properties_are_still_readable(frozen):
    """Reading a derived property writes it into the per-instance cache.

    So frozen cannot mean "no attribute may be written": it means the data and
    the metadata are fixed. Each property is read twice, once to fill the cache
    and once to hit it.
    """
    for _ in range(2):
        assert np.all(np.isfinite(frozen.Po))
        assert np.all(np.isfinite(frozen.Ma))
        assert np.all(np.isfinite(frozen.rho))
        assert np.all(np.isfinite(frozen.vol))


def test_clear_cache_is_allowed(frozen):
    """A cache is not data, so dropping it does not change the value."""
    before = np.array(frozen.Po)
    frozen.clear_cache()

    np.testing.assert_array_equal(frozen.Po, before)


#
# THAWING
#


def test_copy_of_a_frozen_block_is_writeable(frozen):
    """This is how a frozen design becomes the basis of a mixed-out result."""
    thawed = frozen.copy()

    assert not thawed.frozen
    thawed.set_Vx(10.0)
    np.testing.assert_allclose(thawed.Vx, 10.0)


def test_copy_without_patches_of_a_frozen_block_is_writeable(frozen):
    """copy(keep_patches=False) writes metadata, so it must see a thawed array."""
    thawed = frozen.copy(keep_patches=False)

    assert not thawed.frozen
    assert len(thawed.patches) == 0


def test_empty_from_a_frozen_block_is_writeable(frozen):
    """`empty()` is how an ideal state is built beside a real one."""
    ideal = frozen.flat[0].empty()

    assert not ideal.frozen
    ideal.set_P_s(1e5, float(frozen.flat[0].s))


def test_mean_of_a_frozen_block_is_not_frozen(frozen):
    """A reduction allocates its own array, so it is a fresh writeable value."""
    averaged = frozen.mean(axis=0)

    assert not averaged.frozen


#
# EDGES
#


def test_freeze_is_idempotent(frozen):
    twice = frozen.freeze()

    assert twice.frozen
    np.testing.assert_array_equal(twice.conserved, frozen.conserved)


def test_frozen_survives_a_pickle_round_trip(frozen):
    """numpy does not preserve the flag itself, so the state has to carry it."""
    restored = pickle.loads(pickle.dumps(frozen))

    assert restored.frozen
    with pytest.raises(ValueError, match="frozen"):
        restored.set_Vx(10.0)


def test_an_unfrozen_block_survives_a_pickle_round_trip(block):
    restored = pickle.loads(pickle.dumps(block))

    assert not restored.frozen
    restored.set_Vx(10.0)
