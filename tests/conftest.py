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


# ---------------------------------------------------------------------------
# Analytic real gas, for exercising RealFluid without CoolProp
# ---------------------------------------------------------------------------


class VanDerWaals:
    """Van der Waals gas with constant specific heat at constant volume.

    A genuinely non-ideal but exactly-known equation of state, so a
    :class:`~ember.fluid.RealFluid` fitted to it can be checked against closed
    form. Internal energy carries the attraction term, ``u = cv*T - a*rho``,
    which is what makes the compressibility factor vary across the box and so
    exercises the entropy integral rather than leaving it trivially constant.
    """

    def __init__(self, Rgas=51.2, cv=1400.0, a=15.0, b=2.0e-3):
        self.Rgas = Rgas
        self.cv = cv
        self.a = a
        self.b = b

    def get_T(self, rho, u):
        return (u + self.a * rho) / self.cv

    def get_P(self, rho, u):
        T = self.get_T(rho, u)
        return rho * self.Rgas * T / (1.0 - self.b * rho) - self.a * rho**2

    def get_s(self, rho, u):
        T = self.get_T(rho, u)
        return self.cv * np.log(T) + self.Rgas * np.log(1.0 / rho - self.b)

    def get_h(self, rho, u):
        return u + self.get_P(rho, u) / rho


def fit_real_fluid(model, rho_lim, u_lim, order=10, ni=40, **kwargs):
    """Fit a RealFluid to an analytic model over a box, with the datum centred.

    The datum is placed at the centre of the fit box rather than at some
    convenient round number. It has to lie inside the box -- the constructor
    locates it by inverting the fitted surface -- and centring it leaves the
    widest margin on all sides for tests that step away from it.

    Parameters
    ----------
    model : object
        Anything with ``get_P``, ``get_T`` and ``get_s`` taking ``(rho, u)``,
        and a ``Rgas`` attribute.
    rho_lim, u_lim : tuple
        ``(min, max)`` bounds of the fit box, in SI on the model's own datum.
    order : int, optional
        Maximum polynomial order.
    ni : int, optional
        Sample points along each axis.
    **kwargs
        Passed through to :class:`~ember.fluid.RealFluid`.

    Returns
    -------
    ember.fluid.RealFluid

    """
    import ember.realgas_fit as rgf
    from ember.fluid import RealFluid

    rho_g, u_g = np.meshgrid(
        np.linspace(*rho_lim, ni), np.linspace(*u_lim, ni), indexing="ij"
    )
    rho, u = rho_g.ravel(), u_g.ravel()
    result = rgf.fit(
        rho=rho,
        u=u,
        P=model.get_P(rho, u),
        T=model.get_T(rho, u),
        s=model.get_s(rho, u),
        Rgas=model.Rgas,
        rho_lim=rho_lim,
        u_lim=u_lim,
        rho_isochor=float(np.mean(rho_lim)),
        order=order,
    )

    rho_mid = float(np.mean(rho_lim))
    u_mid = float(np.mean(u_lim))
    kwargs.setdefault("P_dtm", float(model.get_P(rho_mid, u_mid)))
    kwargs.setdefault("T_dtm", float(model.get_T(rho_mid, u_mid)))
    kwargs.setdefault("mu", 1.0e-5)
    kwargs.setdefault("Pr", 1.0)
    return RealFluid(**result.kwargs, **kwargs)
