"""Tests for unstructured surface blocks (ember.unstructured).

Module tested: ember.unstructured

Test cases:
- test_set_faces_validates: dtype, shape and index range are checked
- test_faces_read_only_copy: faces are stored as a locked int32 copy
- test_faces_unset_raises: reading faces before set_faces raises
- test_dA_follows_new_faces: replacing the faces changes the area vectors
- test_reorders_raise: indexing, flip, reshape and the like raise
- test_concatenate_raises: block concatenation rejects unstructured blocks
- test_empty_returns_block: empty() gives a structured Block without faces
- test_copy_freeze_pickle_keep_faces: faces survive copy, freeze and pickle
- test_shared_vertices: triangulating a structured cut stores each node once
- test_matches_structured: integrals and averages agree with the structured cut
"""

import pickle

import numpy as np
import pytest

import ember.average
import ember.block_util
import ember.fluid
from ember import util
from ember.block import Block
from ember.cut import triangulate_to_unstructured
from ember.unstructured import UnstructuredBlock


@pytest.fixture
def cut():
    """A 2D structured cut with nonuniform flow."""
    shape = (5, 6)
    xrt = util.linmesh3([0.0, 0.0], [0.8, 1.2], [0.0, 0.3], (1, *shape))[0]
    block = Block(shape=shape)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72))
    r = xrt[..., 1]
    block.set_P_T(1e5 + 1e3 * r, 300.0 + 10.0 * r)
    block.set_Vxrt(np.stack([100.0 + 20.0 * r, 0.0 * r, 30.0 * r], axis=-1))
    block.set_Nb(12)
    return block


@pytest.fixture
def tri(cut):
    return triangulate_to_unstructured(cut)


def test_set_faces_validates():
    u = UnstructuredBlock(4)
    with pytest.raises(TypeError, match="integer"):
        u.set_faces(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="shape"):
        u.set_faces(np.zeros((2, 4), dtype=int))
    with pytest.raises(ValueError, match="lie in"):
        u.set_faces([[0, 1, 4]])
    with pytest.raises(ValueError, match="lie in"):
        u.set_faces([[0, -1, 2]])
    u.set_faces([[0, 1, 2], [1, 2, 3]])
    assert u.n_face == 2
    assert u.n_vert == 4


def test_faces_read_only_copy():
    u = UnstructuredBlock(3)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    u.set_faces(faces)
    faces[0, 0] = 2
    assert u.faces[0, 0] == 0
    assert u.faces.dtype == np.int32
    with pytest.raises(ValueError):
        u.faces[0, 0] = 1


def test_faces_unset_raises():
    with pytest.raises(ValueError, match="set_faces"):
        _ = UnstructuredBlock(3).faces


def test_dA_follows_new_faces(tri):
    dA = tri.dA.copy()
    tri.set_faces(tri.faces[:, ::-1])
    np.testing.assert_allclose(tri.dA, -dA)


def test_reorders_raise(tri):
    with pytest.raises(TypeError):
        _ = tri[0]
    for op in (
        lambda b: b.flip(0),
        lambda b: b.reshape((2, -1)),
        lambda b: b.squeeze(),
        lambda b: b.transpose(),
        lambda b: b.mean(),
        lambda b: b.nanmean(),
    ):
        with pytest.raises(TypeError, match="UnstructuredBlock"):
            op(tri)


def test_concatenate_raises(tri):
    with pytest.raises(TypeError, match="UnstructuredBlock"):
        ember.block_util.concatenate(tri, tri)


def test_empty_returns_block(tri):
    out = tri.empty(shape=(2, 3))
    assert type(out) is Block
    assert out.shape == (2, 3)
    assert "faces" not in out._metadata
    assert out.fluid is tri.fluid
    assert out.Nb == tri.Nb
    assert len(out.patches) == 0


def test_copy_freeze_pickle_keep_faces(tri):
    for other in (tri.copy(), tri.freeze(), pickle.loads(pickle.dumps(tri))):
        assert isinstance(other, UnstructuredBlock)
        np.testing.assert_array_equal(other.faces, tri.faces)
        assert not other.faces.flags.writeable
        np.testing.assert_allclose(other.dA, tri.dA)
    with pytest.raises(ValueError, match="frozen"):
        tri.freeze().set_faces(tri.faces)


def test_shared_vertices(cut, tri):
    ni, nj = cut.shape
    assert tri.n_vert == ni * nj
    assert tri.n_face == 2 * (ni - 1) * (nj - 1)
    # Node (i, j) is vertex i + ni * j
    np.testing.assert_allclose(tri.P, np.asarray(cut.P).ravel(order="F"))
    # Every vertex is used, and only the nodes' own values appear
    assert np.unique(tri.faces).size == tri.n_vert


def test_matches_structured(cut, tri):
    av = ember.average
    np.testing.assert_allclose(av.total_area(tri), av.total_area(cut), rtol=1e-6)
    np.testing.assert_allclose(av.flow_mass(tri), av.flow_mass(cut), rtol=1e-3)
    np.testing.assert_allclose(
        av.mass_average(tri.s, tri), av.mass_average(cut.s, cut), rtol=1e-3
    )
    np.testing.assert_allclose(
        av.area_average(tri.P, tri), av.area_average(cut.P, cut), rtol=1e-4
    )
    np.testing.assert_allclose(av.mix_out(tri).P, av.mix_out(cut).P, rtol=1e-4)
