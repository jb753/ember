"""``Block.fac_lam``, the prescribed laminar fraction, through every consumer.

The field weights both halves of the turbulence model linearly, so one number
means one thing wherever it is read:

    mu_t = (1 - fac_lam) * mu_mix          eight-corner cell mean
    cf   = fac_lam * 2/Re + (1 - fac_lam) * cf_law    four-corner wall-face mean

What is pinned here:

- the wall law at the scalar level (``wall_core``/``wall_yplus``): the laminar
  end is exactly the discrete no-slip stress, the midpoint is the midpoint;
- the mixing length at the block level, through both viscous kernels: zero
  eddy viscosity at ``fac_lam = 1``, half of it at ``0.5``, and the shell
  producer agreeing that a laminar cell is one with no mixing length;
- that the wall blend reaches the row forms and the per-cell i-face wrapper
  and nothing else, by switching the mixing length off and differencing;
- the setter's range check, that the value does not rescale with ``L_ref``;
- and that an ``.emb`` written before the field existed still loads, reading
  zeros.

``fac_lam = 0`` leaving every kernel bit for bit unchanged is not repeated
here: the viscous, residual and SFD goldens run with the default zero field
and are the gate for it.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

import ember.block
import ember.block_util
import ember.fortran
import ember.grid
from ember import util
from ember.fluid import PerfectFluid
from ember.inviscid import InviscidPatch
from ember.periodic import PeriodicPatch

import viscous_util
from test_viscous_phases_golden import _build_block

_HELPERS = ember.fortran.viscous_helpers
_DATA = Path(__file__).parent / "data"

PR_TURB = 1.0
LAWS = [pytest.param(0, id="fit"), pytest.param(1, id="reichardt")]

# wall_core geometry giving d = vol/|dA| = 0.01, and a Reynolds number well
# into the log layer, where the laws and 2/Re are far apart.
DA = np.array([0.0, 0.0, 2.0], dtype=np.float32)
VOL = np.float32(0.02)
RE = 1.0e4


def _core(fl, law):
    """wall_core at Re = RE with unit rho and V; returns its six outputs."""
    mu = np.float32(0.01 / RE)
    one, zero = np.float32(1.0), np.float32(0.0)
    return _HELPERS.wall_core(
        one, DA, VOL, zero, zero, mu, law, np.float32(fl), one, one, zero, zero
    )


def _yplus(fl, law):
    mu = np.float32(0.01 / RE)
    one, zero = np.float32(1.0), np.float32(0.0)
    return _HELPERS.wall_yplus(
        one, DA, VOL, zero, zero, mu, law, np.float32(fl), one, one, zero, zero
    )


@pytest.mark.parametrize("law", LAWS)
def test_laminar_wall_is_the_no_slip_stress(law):
    """At ``fac_lam = 1`` cf is 2/Re and y+ is sqrt(Re), whatever the law."""
    _V, _dA, _Vts, cf_turb, Re, _tau = _core(0.0, law)
    _V, _dA, _Vts, cf_lam, _Re, _tau = _core(1.0, law)
    np.testing.assert_allclose(Re, RE, rtol=1e-5)
    # Vacuity guard: the test point must be where the law is not laminar.
    assert cf_turb > 2.0 * (2.0 / Re)
    np.testing.assert_allclose(cf_lam, 2.0 / Re, rtol=1e-6)
    np.testing.assert_allclose(_yplus(1.0, law), np.sqrt(Re), rtol=1e-5)


@pytest.mark.parametrize("law", LAWS)
def test_half_laminar_wall_is_the_midpoint(law):
    """The blend is linear in ``fac_lam``: half way gives the mean cf and tau."""
    _V, _dA, _Vts, cf0, _Re, tau0 = _core(0.0, law)
    _V, _dA, _Vts, cf1, _Re, tau1 = _core(1.0, law)
    _V, _dA, _Vts, cfh, _Re, tauh = _core(0.5, law)
    np.testing.assert_allclose(cfh, 0.5 * (cf0 + cf1), rtol=1e-6)
    np.testing.assert_allclose(tauh, 0.5 * (tau0 + tau1), rtol=1e-6)


def _viscous_pair(block):
    """Both viscous kernels on ``block``; returns (faces, fvisc, mu_turb)."""
    faces = [np.array(f) for f in viscous_util.fill_faces(block, PR_TURB)]
    fvisc = viscous_util.run_visc_force(block, PR_TURB)
    return faces, fvisc, np.array(block.mu_turb[:-1, :-1, :-1])


def _pair_at(fac_lam, wdist=None):
    block = _build_block()
    if wdist is not None:
        block.set_wdist(wdist)
    block.set_fac_lam(fac_lam)
    return _viscous_pair(block)


def test_default_is_fully_turbulent():
    """A block that never set it reads zeros, and computes as if it had."""
    block = _build_block()
    assert np.all(block.fac_lam == 0.0)
    _f0, fvisc0, mut0 = _viscous_pair(block)
    _f1, fvisc1, mut1 = _pair_at(0.0)
    np.testing.assert_array_equal(fvisc0, fvisc1)
    np.testing.assert_array_equal(mut0, mut1)


def test_laminar_cells_carry_no_eddy_viscosity():
    """``fac_lam = 1`` zeroes mu_turb; ``0.5`` halves it, cell for cell."""
    _f, _fv, mut0 = _pair_at(0.0)
    _f, _fv, muth = _pair_at(0.5)
    _f, _fv, mut1 = _pair_at(1.0)
    # Vacuity guard: the fixture's mixing length is live in most cells.
    assert np.mean(mut0 > 0.0) > 0.5
    assert np.all(mut1 == 0.0)
    np.testing.assert_allclose(muth, 0.5 * mut0, rtol=1e-6)


def test_shell_producer_agrees_a_laminar_cell_has_no_mixing_length():
    """``set_tau_q_faces`` at ``fac_lam = 1`` matches it with ``wdist = 0``.

    The shell has no wall function in it, so the only thing the field can do
    there is the mixing length, and a laminar cell and a cell with zero mixing
    length are the same cell. The per-cell i faces and the row-form j/k faces
    both have to agree for this to hold.
    """
    wdist0 = np.zeros(_build_block().shape, dtype=np.float32)
    faces_lam, _fv, _mut = _pair_at(1.0)
    faces_nol, _fv, _mut = _pair_at(0.0, wdist=wdist0)
    for got, want in zip(faces_lam, faces_nol):
        np.testing.assert_array_equal(got, want)


def test_wall_blend_reaches_only_the_wall_cells():
    """With the mixing length off, ``fac_lam`` moves only wall-adjacent cells.

    Zero wall distance takes the eddy viscosity out at both ends, so the
    difference between ``fac_lam = 1`` and ``0`` is the wall shear alone. The
    fixture's i and j faces are no-slip walls (k is periodic): the j walls go
    through the row form, the i walls through the per-cell wrapper, and both
    must move while every interior cell stays bit for bit where it was.
    """
    wdist0 = np.zeros(_build_block().shape, dtype=np.float32)
    _f, fvisc0, _m = _pair_at(0.0, wdist=wdist0)
    _f, fvisc1, _m = _pair_at(1.0, wdist=wdist0)
    moved = np.any(fvisc1 != fvisc0, axis=-1)

    interior = np.ones(moved.shape, dtype=bool)
    interior[0, :, :] = interior[-1, :, :] = False
    interior[:, 0, :] = interior[:, -1, :] = False
    assert not np.any(moved[interior])
    for wall_cells in (moved[0], moved[-1], moved[:, 0], moved[:, -1]):
        assert np.all(wall_cells)


def test_yplus_reports_the_laminar_wall():
    """``wall_yplus`` sees the field: y+ drops where the wall is laminar."""
    block = _build_block()
    yp0 = ember.block_util.wall_yplus(block)
    block.set_fac_lam(1.0)
    yp1 = ember.block_util.wall_yplus(block)
    for key in ("yplus_i1", "yplus_ni", "yplus_j1", "yplus_nj"):
        assert np.all(yp1[key] < yp0[key]), key


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan, np.inf])
def test_setter_refuses_out_of_range(bad):
    block = _build_block()
    with pytest.raises(ValueError, match="fac_lam"):
        block.set_fac_lam(bad)


def test_fac_lam_does_not_rescale_with_L_ref():
    """Dimensionless: the stored value survives a change of length scale.

    Contrast ``test_block_core.test_wdist_roundtrip_L_ref``, where it is the
    dimensional distance that is preserved and the stored one that moves.
    """
    block = _build_block()
    fac = np.linspace(0.0, 1.0, block.size, dtype=np.float32).reshape(block.shape)
    block.set_fac_lam(fac)
    block.set_L_ref(0.37)
    np.testing.assert_array_equal(block.fac_lam, fac)


#
# READING A FILE WRITTEN BEFORE THE FIELD EXISTED
#
# `_data` and its name-to-column map are pickled per block, so an `.emb` from
# before `fac_lam` holds ten columns where the class now expects eleven. The
# fixture was written by the pre-change code from the builder below, which
# the test repeats to know what the old columns should hold.
#


def _fixture_block():
    """The block tests/data/block_pre_fac_lam.emb was written from."""
    shape = (6, 5, 4)
    block = ember.block.Block(shape=shape)
    block.set_Nb(36)
    xrt = util.linmesh3((0.0, 0.1), (0.5, 0.6), (0.0, 2.0 * np.pi / 36), shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
    block.set_P_T(101325.0, 300.0)
    block.set_Vx(100.0 + 10.0 * xrt[..., 1])
    block.set_Vt(40.0)
    block.set_Vr(0.0)
    block.set_wdist(0.001 + 0.01 * (xrt[..., 1] - 0.5))
    block.set_mu_turb(np.full(shape, 2e-4))
    block.set_L_ref(0.05)
    block.patches.append(PeriodicPatch(k=0))
    block.patches.append(PeriodicPatch(k=-1))
    block.patches.append(InviscidPatch(i=0))
    return block


def test_an_emb_written_before_fac_lam_still_reads():
    """The old file loads, reads zeros for the new field, and nothing moved."""
    (block,) = ember.grid.Grid.read_emb(_DATA / "block_pre_fac_lam.emb")
    want = _fixture_block()

    assert block._data.shape[-1] == len(ember.block.Block._data_keys)
    np.testing.assert_array_equal(block.fac_lam, 0.0)
    for name in ("xrt_nd", "conserved_nd", "wdist_nd", "mu_turb"):
        np.testing.assert_array_equal(getattr(block, name), getattr(want, name))


def test_an_old_emb_takes_a_laminar_fraction_and_keeps_it(tmp_path):
    """Once migrated the block is an ordinary one: set, write, read back."""
    grid = ember.grid.Grid.read_emb(_DATA / "block_pre_fac_lam.emb")
    grid[0].set_fac_lam(0.25)
    grid.write_emb(tmp_path / "new.emb")
    (block,) = ember.grid.Grid.read_emb(tmp_path / "new.emb")
    np.testing.assert_array_equal(block.fac_lam, 0.25)


def test_migration_gives_a_view_its_own_column():
    """A view pickled with its parent shares its index map; both migrate.

    The unpickler preserves that sharing but hands each its own copy of the
    data, so extending the shared map in place would leave the second one
    with an index past the end of its array.
    """
    # No patches: they hold weak references and are cleared by write_emb,
    # which this bare pickle does not go through.
    shape = (6, 5, 4)
    parent = ember.block.Block(shape=shape)
    parent.set_wdist(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
    parent.set_mu_turb(np.full(shape, 2e-4))
    view = parent[1:4]
    nold = len(ember.block.Block._data_keys) - 1
    # Make both look as they did when written before the field: one column
    # short, and a shared map that does not name it.
    del parent._data_inds["fac_lam"]
    parent._data = np.asfortranarray(parent._data[..., :nold])
    view._data = np.asfortranarray(view._data[..., :nold])
    assert view._data_inds is parent._data_inds

    parent2, view2 = pickle.loads(pickle.dumps((parent, view)))
    for block in (parent2, view2):
        np.testing.assert_array_equal(block.fac_lam, 0.0)
        np.testing.assert_array_equal(block.mu_turb, np.float32(2e-4))
    np.testing.assert_array_equal(view2.wdist_nd, parent2.wdist_nd[1:4])
