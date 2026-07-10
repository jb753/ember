"""Shared test fixtures for ember tests."""

import ast
import pytest
import numpy as np
from ember.block import Block
from ember.fluid import PerfectFluid


# ---------------------------------------------------------------------------
# Class member ordering checker
# ---------------------------------------------------------------------------

_PROP_DECORATORS = {
    "property",
    "cached_array",
    "cached_object",
    "derived_array",
    "scratch_array",
}
_GROUP_ORDER = ["private", "classmethod", "set", "get", "other_public", "property"]
_GROUP_SORTED = {"set", "get", "other_public", "property"}


def _SORT_KEY(s):
    return s.lstrip("_").lower()


def _member_group(name, dec_names):
    if name.startswith("_"):
        return "private"
    if "classmethod" in dec_names or "staticmethod" in dec_names:
        return "classmethod"
    if name.startswith("set_"):
        return "set"
    if name.startswith("get_"):
        return "get"
    if _PROP_DECORATORS & dec_names:
        return "property"
    return "other_public"


def _decorator_names(node):
    names = set()
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.add(dec.id)
        elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            names.add(dec.func.id)
        elif isinstance(dec, ast.Attribute):
            names.add(dec.attr)
    return names


def assert_class_member_order(src, class_name):
    """Assert that a class in *src* follows the standard member ordering:

    1. private (methods and properties, ``_*``)
    2. classmethods / staticmethods
    3. ``set_*`` methods, alphabetical
    4. ``get_*`` methods, alphabetical
    5. other public methods, alphabetical
    6. public properties, alphabetical

    Sort order is case-insensitive with leading underscores stripped.
    """
    tree = ast.parse(src)
    cls = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == class_name
        ),
        None,
    )
    assert cls is not None, f"Class {class_name!r} not found in source"

    members = []
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dec_names = _decorator_names(node)
        if dec_names & {"setter", "deleter"}:
            continue
        members.append((node.name, _member_group(node.name, dec_names)))

    # Check group ordering
    seen_rank = -1
    for name, group in members:
        rank = _GROUP_ORDER.index(group)
        assert rank >= seen_rank, (
            f"{class_name}: '{name}' ({group}) appears after a later group"
        )
        seen_rank = rank

    # Check alphabetical within sorted groups
    for group in _GROUP_SORTED:
        names = [n for n, g in members if g == group]
        assert names == sorted(names, key=_SORT_KEY), (
            f"{class_name}: [{group}] not in alphabetical order\n"
            f"  got:      {names}\n"
            f"  expected: {sorted(names, key=_SORT_KEY)}"
        )


def _make_block(shape):
    """Create a block with given shape, fluid, coordinates and flow state.

    Geometry is a surface of revolution: j is spanwise (varying x), k is
    pitchwise (constant x, r).
    """
    block = Block(shape=shape)
    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    block.set_fluid(fluid)
    ni, nj, nk = shape
    x = np.linspace(0.0, 1.0, nj).reshape(1, -1, 1) * np.ones(shape)
    r = np.ones(shape) * 0.5
    t = np.linspace(0.0, 0.2, nk).reshape(1, 1, -1) * np.ones(shape)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)
    block.set_P_T(101325.0, 300.0)
    return block


@pytest.fixture
def block_10_20_30():
    """Block with shape (10, 20, 30) for testing patches."""
    return _make_block((10, 20, 30))


@pytest.fixture
def small_block():
    """Small block with shape (5, 5, 5) for testing patches."""
    return _make_block((5, 5, 5))


@pytest.fixture
def block_10():
    """Block with shape (10, 10, 10) for testing patches."""
    return _make_block((10, 10, 10))
