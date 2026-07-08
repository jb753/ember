"""Isolated sanity checks for ``ember.fortran.set_polar_source``.

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

Reference: Multall ``tblock`` builds the same term as
``SOURCE = (P + rho*Vt^2)*vol/r`` and ADDS it to the radial-momentum
change (tblock-p-2_3_1.f:7070, :7626).
"""

from ember import fortran
import numpy as np

typ = np.float32


def _fort(x):
    return np.asfortranarray(x, dtype=typ)


def _build_case(ni, nj, nk, rho0, r0, Vt0, P0, vol0):
    """Construct a uniform-field test case with an analytic source.

    Nodal ``r`` and ``P`` are constant so their cell averages are exactly
    ``r0`` and ``P0``. ``cons_cell`` is already cell-centred: density in
    component 1 and angular momentum ``rho*r*Vt`` in component 4, so the
    routine recovers ``Vt = Vt0`` exactly.
    """
    nci, ncj, nck = ni - 1, nj - 1, nk - 1

    cons_cell = np.zeros((nci, ncj, nck, 5), dtype=typ)
    cons_cell[..., 0] = rho0
    cons_cell[..., 3] = rho0 * r0 * Vt0  # rho * r * Vt (angular momentum)

    r = np.full((ni, nj, nk), r0, dtype=typ)
    P = np.full((ni, nj, nk), P0, dtype=typ)
    vol = np.full((nci, ncj, nck), vol0, dtype=typ)
    net_flow = np.zeros((nci, ncj, nck, 5), dtype=typ)

    return (
        _fort(cons_cell),
        _fort(r),
        _fort(P),
        _fort(vol),
        _fort(net_flow),
    )


def test_sign_is_outward():
    """A swirling, pressurised cell must drive radial momentum OUTWARD (+)."""
    rho0, r0, Vt0, P0, vol0 = 1.2, 0.5, 30.0, 1.0e5, 2.0e-3
    cons_cell, r, P, vol, net_flow = _build_case(5, 4, 4, rho0, r0, Vt0, P0, vol0)

    fortran.set_polar_source(
        cons_cell=cons_cell, r=r, p=P, p_offset=0.0, vol=vol, net_flow=net_flow
    )

    assert np.all(net_flow[..., 2] > 0.0), (
        "polar source must add positive (outward) radial momentum; "
        f"got {net_flow[..., 2].ravel()[0]:.3e}"
    )


def test_magnitude_matches_analytic():
    """net_flow[...,2] == +vol*(P + rho*Vt^2)/r exactly."""
    rho0, r0, Vt0, P0, vol0 = 1.2, 0.5, 30.0, 1.0e5, 2.0e-3
    cons_cell, r, P, vol, net_flow = _build_case(5, 4, 4, rho0, r0, Vt0, P0, vol0)

    fortran.set_polar_source(
        cons_cell=cons_cell, r=r, p=P, p_offset=0.0, vol=vol, net_flow=net_flow
    )

    S = (P0 + rho0 * Vt0**2) / r0
    expected = vol0 * S
    np.testing.assert_allclose(net_flow[..., 2], expected, rtol=1e-4)


def test_only_radial_component_touched():
    """Mass, axial, angular-momentum and energy components stay untouched."""
    cons_cell, r, P, vol, net_flow = _build_case(5, 4, 4, 1.2, 0.5, 30.0, 1.0e5, 2e-3)

    fortran.set_polar_source(
        cons_cell=cons_cell, r=r, p=P, p_offset=0.0, vol=vol, net_flow=net_flow
    )

    for comp in (0, 1, 3, 4):
        assert np.all(net_flow[..., comp] == 0.0), f"component {comp} changed"


def test_zero_source_when_quiescent():
    """No swirl and P == p_offset gives an exactly zero source."""
    P0 = 1.0e5
    cons_cell, r, P, vol, net_flow = _build_case(5, 4, 4, 1.2, 0.5, 0.0, P0, 2e-3)

    fortran.set_polar_source(
        cons_cell=cons_cell, r=r, p=P, p_offset=P0, vol=vol, net_flow=net_flow
    )

    assert np.all(net_flow[..., 2] == 0.0)


def test_accumulates_into_existing():
    """The source adds to a pre-existing net_flow rather than overwriting."""
    rho0, r0, Vt0, P0, vol0 = 1.2, 0.5, 30.0, 1.0e5, 2.0e-3
    cons_cell, r, P, vol, net_flow = _build_case(5, 4, 4, rho0, r0, Vt0, P0, vol0)
    seed = 7.0
    net_flow[..., 2] = seed

    fortran.set_polar_source(
        cons_cell=cons_cell, r=r, p=P, p_offset=0.0, vol=vol, net_flow=net_flow
    )

    S = (P0 + rho0 * Vt0**2) / r0
    np.testing.assert_allclose(net_flow[..., 2], seed + vol0 * S, rtol=1e-4)
