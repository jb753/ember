"""Isolated sanity checks for the polar source in ``set_timestep_sources``.

The polar source is the radial-momentum body force on a cylindrical-polar
mesh,

    S = (P + rho*Vt^2) / r        [per unit volume]

It is physically positive (radially outward): centrifugal ``rho*Vt^2/r``
plus the net radial pressure force on the converging pitchwise faces
``P/r``. It is accumulated into the radial-momentum component
(``net_flow[..., 2]``, i.e. Fortran component 3) and must enter the
residual with the SAME sign convention as the inter-cell flux balance
``dU = flux_in - flux_out + f_body`` (see ``residual.f90``). Hence a
positive ``S`` must push the radial momentum outward, i.e. add ``+vol*S``.

Reference: Multall builds the same term as
``SOURCE = (P + rho*Vt^2)*vol/r`` and ADDS it to the radial-momentum
change.

The source is added by the timestep kernel's ``add_sources`` pass, which also
recomputes ``dt_vol``; the timestep inputs below are arbitrary but valid, and
the SFD half is off (zero gain, zero-size filter state), so ``net_flow`` sees
the polar source and nothing else.
"""

from ember import fortran
import numpy as np

typ = np.float32


def _fort(x):
    return np.asfortranarray(x, dtype=typ)


def set_polar_source(cons, r, p, p_offset, vol, net_flow):
    """Add the polar source to ``net_flow`` in place, and nothing else."""
    ni, nj, nk = r.shape
    fortran.set_timestep_sources(
        dt_vol=_fort(np.zeros((ni - 1, nj - 1, nk - 1))),
        a=_fort(np.full((ni, nj, nk), 340.0)),
        cons=cons,
        r=r,
        omega=0.0,
        dai=_fort(np.ones((3, ni, nj - 1, nk - 1))),
        daj=_fort(np.ones((3, ni - 1, nj, nk - 1))),
        dak=_fort(np.ones((3, ni - 1, nj - 1, nk))),
        mu_turb=_fort(np.zeros((ni, nj, nk))),
        vol=vol,
        rf=1.0,
        fac_visc=1.0,
        p=p,
        p_offset=p_offset,
        f_body=net_flow,
        cons_filt=_fort(np.zeros((0, 0, 0, 5))),
        add_sources=1,
        gain_filt=0.0,
        cfl=0.0,
        delta_filt=1.0,
    )


def _build_case(ni, nj, nk, rho0, r0, Vt0, P0, vol0):
    """Construct a uniform-field test case with an analytic source.

    Every nodal field is constant, so its 8-corner cell average is exactly the
    constant: ``r0``, ``P0``, and from ``cons`` a density ``rho0`` and angular
    momentum ``rho*r*Vt`` that recover ``Vt = Vt0`` exactly.
    """
    nci, ncj, nck = ni - 1, nj - 1, nk - 1

    cons = np.zeros((ni, nj, nk, 5), dtype=typ)
    cons[..., 0] = rho0
    cons[..., 3] = rho0 * r0 * Vt0  # rho * r * Vt (angular momentum)

    r = np.full((ni, nj, nk), r0, dtype=typ)
    P = np.full((ni, nj, nk), P0, dtype=typ)
    vol = np.full((nci, ncj, nck), vol0, dtype=typ)
    net_flow = np.zeros((nci, ncj, nck, 5), dtype=typ)

    return (
        _fort(cons),
        _fort(r),
        _fort(P),
        _fort(vol),
        _fort(net_flow),
    )


def test_sign_is_outward():
    """A swirling, pressurised cell must drive radial momentum OUTWARD (+)."""
    rho0, r0, Vt0, P0, vol0 = 1.2, 0.5, 30.0, 1.0e5, 2.0e-3
    cons, r, P, vol, net_flow = _build_case(5, 4, 4, rho0, r0, Vt0, P0, vol0)

    set_polar_source(cons=cons, r=r, p=P, p_offset=0.0, vol=vol, net_flow=net_flow)

    assert np.all(net_flow[..., 2] > 0.0), (
        "polar source must add positive (outward) radial momentum; "
        f"got {net_flow[..., 2].ravel()[0]:.3e}"
    )


def test_magnitude_matches_analytic():
    """net_flow[...,2] == +vol*(P + rho*Vt^2)/r exactly."""
    rho0, r0, Vt0, P0, vol0 = 1.2, 0.5, 30.0, 1.0e5, 2.0e-3
    cons, r, P, vol, net_flow = _build_case(5, 4, 4, rho0, r0, Vt0, P0, vol0)

    set_polar_source(cons=cons, r=r, p=P, p_offset=0.0, vol=vol, net_flow=net_flow)

    S = (P0 + rho0 * Vt0**2) / r0
    expected = vol0 * S
    np.testing.assert_allclose(net_flow[..., 2], expected, rtol=1e-4)


def test_only_radial_component_touched():
    """Mass, axial, angular-momentum and energy components stay untouched."""
    cons, r, P, vol, net_flow = _build_case(5, 4, 4, 1.2, 0.5, 30.0, 1.0e5, 2e-3)

    set_polar_source(cons=cons, r=r, p=P, p_offset=0.0, vol=vol, net_flow=net_flow)

    for comp in (0, 1, 3, 4):
        assert np.all(net_flow[..., comp] == 0.0), f"component {comp} changed"


def test_zero_source_when_quiescent():
    """No swirl and P == p_offset gives an exactly zero source."""
    P0 = 1.0e5
    cons, r, P, vol, net_flow = _build_case(5, 4, 4, 1.2, 0.5, 0.0, P0, 2e-3)

    set_polar_source(cons=cons, r=r, p=P, p_offset=P0, vol=vol, net_flow=net_flow)

    assert np.all(net_flow[..., 2] == 0.0)


def test_accumulates_into_existing():
    """The source adds to a pre-existing net_flow rather than overwriting."""
    rho0, r0, Vt0, P0, vol0 = 1.2, 0.5, 30.0, 1.0e5, 2.0e-3
    cons, r, P, vol, net_flow = _build_case(5, 4, 4, rho0, r0, Vt0, P0, vol0)
    seed = 7.0
    net_flow[..., 2] = seed

    set_polar_source(cons=cons, r=r, p=P, p_offset=0.0, vol=vol, net_flow=net_flow)

    S = (P0 + rho0 * Vt0**2) / r0
    np.testing.assert_allclose(net_flow[..., 2], seed + vol0 * S, rtol=1e-4)
