r"""Physical limits of the wall function in ``_fortran/viscous.f90``.

``wall_core`` is the single point where the solver decides what a wall does,
and two of its choices are consequential but almost invisible in the source:

* the length scale ``d = vol/|dA|`` -- the wall-adjacent cell thickness. That
  is the right length for this cell-vertex scheme, because the velocity fed
  to the closure is the first off-wall *node* (``kface(Vx, i, j, k+dk)``), a
  distance ``vol/A_wall`` away. A cell-centred code sampling its centroid
  would need ``d/2``. Substituting one for the other still yields a plausible,
  converging solution with the skin friction wrong by a factor of two.

* the laminar switch ``Re < 127.53373025``, which is the *root* of the two
  ``cf`` branches for the coefficients ``a1/a2/a3``. Nothing in the source
  computes it, so editing the coefficients without recomputing it opens a
  silent discontinuity in ``cf``.

Below the switch ``cf = 2/Re``, and the closure collapses to the discrete
no-slip stress exactly -- not asymptotically::

    tau = cf*0.5*rho*V^2 = (2*mu/(rho*V*d))*0.5*rho*V^2 = mu*V/d
    y+  = Re*sqrt(cf/2)  = sqrt(Re)

which is what makes a point assertion possible here, with no refinement study
and no analytic reference flow. It also pins ``d`` *physically*: change the
length scale and ``tau == mu*V/d`` fails with a diagnosable message rather
than a numerical diff.

Method: the scalar tests call ``viscous_helpers``' module procedures directly
(f2py exposes them, they are ``public`` in the module), so they test the
production kernel with no fixture, block or ``fvisc`` extraction in the way.
The block-level test then covers what a scalar call cannot -- that the six
call sites hand ``wall_core`` the right cell's volume and the right face's
area.

Companion to ``tests/test_wall_yplus.py``, which checks ``wall_yplus_field``
against an expression-for-expression numpy transcription of ``wall_core``.
That test pins the formula by duplicating it; this one checks the formula
means what it claims. It also reaches the ``Re < 127.53`` branch, which no
other fixture in the suite does.
"""

import numpy as np
import pytest

import ember.block
import ember.block_util
import ember.fortran
from ember import util
from ember.fluid import PerfectFluid
from ember.inviscid import InviscidPatch

_HELPERS = ember.fortran.viscous_helpers

# Root of cf_lam = 2/Re and the turbulent curve fit, transcribed from
# wall_core. Used only to place test points relative to the switch and as a
# vacuity guard -- never as an expected value.
RE_SWITCH = 127.53373025

# Geometry giving d = vol/|dA| = 0.01 exactly.
DA = np.array([0.0, 0.0, 2.0], dtype=np.float32)
VOL = np.float32(0.02)
D_EXACT = 0.01

# float32 kernel; the identities below were measured to hold at ~1e-7.
RTOL = 1e-5


def _mu_for(Re, rho=1.0, V=1.0, d=D_EXACT):
    """Laminar viscosity placing wall_core's Re at the requested value."""
    return np.float32(rho * V * d / Re)


def _core(mu, rho=1.0, Vx=1.0, Vr=0.0, Vt=0.0, r=1.0, Omega_block=0.0, Omega_wall=0.0):
    """wall_core with this file's fixed geometry; returns its six outputs."""
    return _HELPERS.wall_core(
        np.float32(r),
        DA,
        VOL,
        np.float32(Omega_block),
        np.float32(Omega_wall),
        np.float32(mu),
        np.float32(rho),
        np.float32(Vx),
        np.float32(Vr),
        np.float32(Vt),
    )


# Straight axial flow at three points across the laminar branch, plus one
# case with swirl and a wall turning relative to the block, so Vt_slip is
# exercised beyond the Omega_wall == Omega_block degeneracy that every other
# viscous test leaves it in.
_LAMINAR_CASES = [
    pytest.param({"Re": 5.0}, id="Re5"),
    pytest.param({"Re": 30.0}, id="Re30"),
    pytest.param({"Re": 100.0}, id="Re100"),
    pytest.param(
        {
            "Re": 40.0,
            "Vx": 0.7,
            "Vr": 0.3,
            "Vt": 0.5,
            "r": 1.3,
            "Omega_block": 0.2,
            "Omega_wall": 0.05,
        },
        id="swirl_rotating_wall",
    ),
]


def _case_kwargs(case):
    """Split a _LAMINAR_CASES entry into the mu that hits its Re, plus the
    remaining wall_core arguments."""
    kwargs = dict(case)
    Re_target = kwargs.pop("Re")
    # V is the magnitude of the slip velocity, which is what sets Re -- solve
    # for it the same way wall_core does rather than assuming Vx.
    Vt_slip = kwargs.get("Vt", 0.0) - (
        kwargs.get("Omega_wall", 0.0) - kwargs.get("Omega_block", 0.0)
    ) * kwargs.get("r", 1.0)
    V = np.sqrt(kwargs.get("Vx", 1.0) ** 2 + kwargs.get("Vr", 0.0) ** 2 + Vt_slip**2)
    return _mu_for(Re_target, rho=1.0, V=V), kwargs


@pytest.mark.parametrize("case", _LAMINAR_CASES)
def test_laminar_branch_is_discrete_no_slip_stress(case):
    r"""Below the switch the wall function *is* a no-slip wall.

    ``cf = 2/Re`` gives ``tau = mu*V/d`` identically -- the viscous stress
    across the first cell for a linear profile from zero at the wall to ``V``
    at ``d``. So the closure carries no modelling once the near-wall mesh is
    fine enough, and its mesh dependence vanishes in that limit.

    This is the assertion that pins ``d = vol/|dA|``: ``d`` is taken from
    ``wall_core``'s own returned ``dA_mag``, so halving it (a cell-centred
    length) or swapping in the geometric ``wdist`` breaks the identity
    immediately and says so in the failure message.
    """
    mu, kwargs = _case_kwargs(case)
    V, dA_mag, _Vt_slip, cf, Re, tau = _core(mu, **kwargs)

    assert Re < RE_SWITCH, f"Re={Re} is above the switch; this case tests nothing"
    assert cf == pytest.approx(2.0 / Re, rel=RTOL), "not on the laminar branch"

    d = VOL / dA_mag
    assert d == pytest.approx(D_EXACT, rel=RTOL)
    np.testing.assert_allclose(
        tau,
        mu * V / d,
        rtol=RTOL,
        err_msg="laminar-branch tau is not the discrete no-slip stress mu*V/d",
    )


@pytest.mark.parametrize("case", _LAMINAR_CASES)
def test_laminar_branch_yplus_is_sqrt_re(case):
    r"""``y+ == sqrt(Re)`` below the switch.

    With ``cf = 2/Re``, ``y+ = Re*sqrt(cf/2) = sqrt(Re)``. The identity holds
    only if ``wall_yplus`` and ``wall_core`` share one ``d``, one ``V`` and
    one ``rho`` -- the drift ``wall_core``'s extraction exists to prevent
    (see its comment in ``viscous.f90``), promised there by structure and a
    comment, and enforced here.

    ``sqrt(RE_SWITCH) = 11.29`` is also the classical sublayer/log-law
    intersection, so the branch changes over exactly where the sublayer ends.
    """
    mu, kwargs = _case_kwargs(case)
    _V, _dA_mag, _Vt_slip, _cf, Re, _tau = _core(mu, **kwargs)
    yplus = _HELPERS.wall_yplus(
        np.float32(kwargs.get("r", 1.0)),
        DA,
        VOL,
        np.float32(kwargs.get("Omega_block", 0.0)),
        np.float32(kwargs.get("Omega_wall", 0.0)),
        np.float32(mu),
        np.float32(1.0),
        np.float32(kwargs.get("Vx", 1.0)),
        np.float32(kwargs.get("Vr", 0.0)),
        np.float32(kwargs.get("Vt", 0.0)),
    )

    assert Re < RE_SWITCH, f"Re={Re} is above the switch; this case tests nothing"
    np.testing.assert_allclose(
        yplus,
        np.sqrt(Re),
        rtol=RTOL,
        err_msg="laminar-branch y+ is not sqrt(Re): Re and y+ use different d/V/rho",
    )


def test_cf_continuous_across_laminar_switch():
    r"""``cf`` has no jump at ``Re = 127.53373025``.

    That constant is the root of ``2/Re`` and the turbulent fit for the
    current ``a1/a2/a3``; it is written as a literal and derived nowhere. Edit
    the coefficients without recomputing it and ``cf`` -- and so ``y+`` --
    steps discontinuously, with every existing test still green.

    Both values are read from the Fortran on either side of the switch, so
    this asserts nothing about what the coefficients *are*, only that the two
    branches still meet. Across a bracket of +-1e-4 in Re the true jump is
    1.4e-4, which is just ``cf``'s own smooth variation over that interval; a
    1% error in ``a3`` raises it to 6.8e-3 and in ``a2`` to 4.0e-3. The 1e-3
    threshold therefore keeps ~7x headroom while still catching a coefficient
    edit of more than roughly 0.15%.
    """
    eps = 1e-4
    _V, _dA, _Vts, cf_lam, Re_lam, _tau = _core(_mu_for(RE_SWITCH * (1.0 - eps)))
    _V, _dA, _Vts, cf_turb, Re_turb, _tau = _core(_mu_for(RE_SWITCH * (1.0 + eps)))

    # Guard that the bracket really straddles the branch, so a future change
    # to the switch cannot quietly make this a two-point test of one branch.
    assert Re_lam < RE_SWITCH < Re_turb
    assert cf_lam == pytest.approx(2.0 / Re_lam, rel=RTOL), "low side is not laminar"
    assert cf_turb != pytest.approx(2.0 / Re_turb, rel=1e-8), "high side is not the fit"

    jump = abs(cf_turb - cf_lam) / cf_lam
    assert jump < 1e-3, (
        f"cf jumps by {jump:.3e} across Re={RE_SWITCH}: the switch constant no "
        "longer matches the curve-fit coefficients a1/a2/a3 and must be "
        "recomputed as the root of cf_lam = cf_turb"
    )


def test_wall_func_flux_carries_tau():
    r"""``wall_func``'s flux vector is ``tau`` times the face area, directed
    along the slip velocity, and its work term is the wall speed times the
    moment term.

    Ties the flux assembly to the stress that ``wall_core`` returned, so the
    ``vec``/``r``/``Omega_wall`` bookkeeping cannot drift from the physics
    the other tests here pin down. ``flow(3)`` carries a factor ``r``
    (angular momentum), hence the division below.
    """
    kwargs = {
        "Vx": 0.7,
        "Vr": 0.3,
        "Vt": 0.5,
        "r": 1.3,
        "Omega_block": 0.2,
        "Omega_wall": 0.05,
    }
    mu = _mu_for(40.0)
    _V, dA_mag, _Vt_slip, _cf, _Re, tau = _core(mu, **kwargs)
    flow = _HELPERS.wall_func(
        np.float32(kwargs["r"]),
        DA,
        VOL,
        np.float32(kwargs["Omega_block"]),
        np.float32(kwargs["Omega_wall"]),
        np.float32(mu),
        np.float32(1.0),
        np.float32(kwargs["Vx"]),
        np.float32(kwargs["Vr"]),
        np.float32(kwargs["Vt"]),
    )

    magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2 + (flow[2] / kwargs["r"]) ** 2)
    np.testing.assert_allclose(
        magnitude,
        tau * dA_mag,
        rtol=RTOL,
        err_msg="wall_func's momentum flux is not tau times the face area",
    )
    np.testing.assert_allclose(
        flow[3],
        kwargs["Omega_wall"] * flow[2],
        rtol=RTOL,
        err_msg="wall_func's work term is not Omega_wall times its moment term",
    )


# --------------------------------------------------------------------------
# Block level: the scalar tests above verify wall_core given a vol and a dA.
# They cannot catch a wrong cell or face index at the six call sites, e.g.
# vol(i,j,k+(dk-1)/2) picking the far cell instead of the wall-adjacent one.
# --------------------------------------------------------------------------

SHAPE = (7, 9, 9)
NB = 36
# Three orders above production mu (1.8e-5), which puts the whole field on
# the laminar branch. Every other viscous fixture in the suite sits at
# Re ~ 1e5-1e7 and so has never executed that branch.
MU_LAMINAR = 0.05
VX_UNIFORM = 100.0


def _build_uniform_block():
    """Uniform axial flow, stationary, frictionless i-faces and default
    no-slip j/k walls.

    Geometry follows ``tests/test_wall_yplus.py``'s fixture; the flow field is
    reduced to a uniform one so that rho and V are identical on every wall
    face and the only thing varying across the k1 face is the cell geometry
    -- which is what this test is about.
    """
    pitch = 2.0 * np.pi / NB

    block = ember.block.Block(shape=SHAPE)
    block.set_Nb(NB)
    xrt = util.linmesh3((0.0, 0.15), (0.5, 0.9), (0.0, pitch), SHAPE)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=MU_LAMINAR, Pr=0.72))

    block.set_P_T(101325.0, np.full(SHAPE, 300.0, dtype=np.float32))
    block.set_Vx(np.full(SHAPE, VX_UNIFORM, dtype=np.float32))
    block.set_Vr(np.zeros(SHAPE, dtype=np.float32))
    block.set_Vt(np.zeros(SHAPE, dtype=np.float32))
    block.set_Omega(0.0)
    block.set_wdist(np.full(SHAPE, 0.008, dtype=np.float32))

    block.patches.append(InviscidPatch(i=0))
    block.patches.append(InviscidPatch(i=-1))
    # j and k faces carry no patch -> default no-slip walls.
    return block


def test_block_yplus_matches_first_cell_reynolds():
    r"""On a real block, ``y+`` on the k1 wall is ``sqrt(Re)`` built from
    *that* cell's volume over *that* face's area.

    Every input is read from the block's own ``_nd`` accessors -- the same
    ones ``block_util.wall_yplus`` feeds to the kernel -- so this checks the
    plumbing rather than the formula: that the wall-adjacent cell volume and
    the wall face area reaching ``wall_core`` are the ones this face's
    geometry implies. Indexing the far cell, or the opposite face, moves ``d``
    and breaks it.

    The ``Re < RE_SWITCH`` guard is not decoration. ``test_wall_yplus.py``'s
    reference handles both branches but its laminar arm has never been
    evaluated, because the fixture sits three orders above the switch; without
    this assertion the same drift would silently empty this test too.
    """
    block = _build_uniform_block()
    got = np.asarray(ember.block_util.wall_yplus(block)["yplus_k1"], dtype=np.float64)

    rho = np.asarray(block.conserved_nd, dtype=np.float64)[..., 0]
    Vx = np.asarray(block.Vx_nd, dtype=np.float64)
    Vr = np.asarray(block.Vr_nd, dtype=np.float64)
    Vt = np.asarray(block.Vt_rel_nd, dtype=np.float64)
    vol = np.asarray(block.vol_nd, dtype=np.float64)
    dAk = np.asarray(block.dAk_nd, dtype=np.float64)
    mu = float(block.mu_nd)

    def mean4(a, node):
        """kface corner average at a node plane: (i,j)/(i+1,j)/(i,j+1)/(i+1,j+1)."""
        return 0.25 * (
            a[:-1, :-1, node] + a[1:, :-1, node] + a[:-1, 1:, node] + a[1:, 1:, node]
        )

    # k1 wall: face plane 0, wall-adjacent cell 0, velocity at node plane 1.
    V = np.sqrt(mean4(Vx, 1) ** 2 + mean4(Vr, 1) ** 2 + mean4(Vt, 1) ** 2 + 1e-9)
    d = vol[:, :, 0] / np.sqrt((dAk[:, :, :, 0] ** 2).sum(axis=0))
    Re = mean4(rho, 1) * V * d / mu

    assert Re.max() < RE_SWITCH, (
        f"Re reaches {Re.max():.1f}, above the switch at {RE_SWITCH}: this "
        "fixture no longer exercises the laminar branch and the identity "
        "below is vacuous -- lower MU_LAMINAR"
    )
    np.testing.assert_allclose(
        got,
        np.sqrt(Re),
        rtol=RTOL,
        err_msg="block y+ does not match sqrt(Re) built from vol/|dAk| on this face",
    )
