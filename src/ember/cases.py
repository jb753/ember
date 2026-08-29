"""Canonical test cases assembled from the core building blocks.

This module holds the case-construction scaffolding used by the CLI drivers in
``tools/``. :func:`build_duct_grid` meshes a single square duct, sets the mean
flow and boundary conditions, and applies the initial-condition perturbations.
"""

import numpy as np

import ember.block
import ember.grid
import ember.fluid
import ember.patch
import ember.set_iterative
from ember import util


def build_duct_grid(
    ncell=1_000_000,
    *,
    cluster=True,
    ER=1.05,
    perturb_vx=0.01,
    perturb_seed=0,
    ho_frac=0.01,
    s_frac=0.01,
    vx_ramp=0.01,
    side=0.1,
    r_mid_ratio=5.0,
    length_ratio=3.0,
    nj=65,
    nk=57,
    periodic_k=None,
    Ma_bulk=0.3,
    Po=1e5,
    To=300.0,
):
    """Assemble the square-duct grid at its perturbed initial condition.

    The duct cross-section is a square of side ``side``, bent onto a mean radius
    ``r_mid_ratio * side`` and repeated around the annulus at ``Nb`` passages,
    with ``Nb`` chosen so the arc subtended by one passage at the mean radius is
    as close to ``side`` as an integer passage count allows (a square section).
    The streamwise mesh is uniform; the two cross-stream directions are either
    clustered towards both walls at expansion ratio ``ER`` (``cluster=True``) or
    uniform (``cluster=False``). The mean flow is fixed by a bulk Mach number at
    a given stagnation state, then perturbed with a random velocity ripple plus
    a deterministic streamwise ramp in enthalpy, entropy, and velocity.

    Parameters
    ----------
    ncell : int
        Target cell count. The streamwise node count ``ni`` is derived from it
        (``nj`` and ``nk`` fixed) and rounded so ``ni - 1`` is a multiple of 8,
        making the block friendly to the two-level multigrid coarsening.
    cluster : bool
        Cluster the two cross-stream directions towards both walls (``True``) or
        space them uniformly (``False``).
    ER : float
        Expansion ratio for the wall clustering; ignored when ``cluster`` is
        ``False``.
    perturb_vx : float
        Amplitude of the random axial-velocity ripple (fraction of ``Vx``).
    perturb_seed : int
        Seed for the velocity-ripple RNG.
    ho_frac : float
        Stagnation enthalpy raised by this fraction of the local dynamic
        enthalpy.
    s_frac : float
        Entropy raised by the entropy-equivalent of that enthalpy offset.
    vx_ramp : float
        Streamwise ramp applied to ``Vx``: inlet unchanged, outlet ``1 + vx_ramp``.
    side, r_mid_ratio, length_ratio : float
        Cross-section side, mean-radius ratio, and length ratio of the duct.
    nj, nk : int
        Cross-stream node counts. Must be odd for symmetric clustering.
    periodic_k : {None, "full", "hmesh"}, optional
        Make the k (pitchwise) faces periodic to each other instead of walls.
        ``None`` leaves the duct closed, as before. ``"full"`` makes the whole
        of both k faces periodic. ``"hmesh"`` makes two streamwise intervals
        periodic with a wall between them, the topology
        :attr:`~ember.block.Block.i_perk` describes, so that a single seam
        carries both a periodic and a wall region.
    Ma_bulk, Po, To : float
        Bulk Mach number and inlet stagnation pressure and temperature.

    Returns
    -------
    ember.grid.Grid
        The assembled, non-dimensionalised grid with wall distance computed.
    """
    r_mid = r_mid_ratio * side
    r_low = r_mid - 0.5 * side
    length = length_ratio * side

    # ni derived from the target cell count, then rounded so ni-1 is a multiple
    # of 8 for the two-level multigrid coarsening.
    ni = ncell // (nj * nk)
    ni = ((ni - 1 + 4) // 8) * 8 + 1
    # Below this the duct is too short to march: a handful of streamwise cells
    # gives the inlet and outlet patches no interior between them, and ni=1
    # (ncell < nj*nk) yields zero cells, which the Fortran kernels reject.
    if ni < 25:
        raise ValueError(
            f"ncell={ncell} gives only ni={ni} streamwise nodes "
            f"(nj={nj}, nk={nk}); need ni >= 25, i.e. ncell >= {25 * nj * nk}"
        )

    Nb = round(2.0 * np.pi * r_mid / side)  # pitch subtends ~side at mean radius
    pitch = 2.0 * np.pi / Nb

    # cluster_symmetric with ER=1.0 reduces to a uniform linspace, so the
    # uniform mesh is just the ER=1.0 case of the same construction.
    ER_eff = ER if cluster else 1.0
    xv = np.linspace(0.0, length, ni, dtype=np.float32)
    rv = r_low + side * util.cluster_symmetric(nj, ER_eff)
    tv = pitch * (util.cluster_symmetric(nk, ER_eff) - 0.5)
    xm, rm, tm = np.meshgrid(xv, rv, tv, indexing="ij")
    xrt = np.stack((xm, rm, tm), axis=-1).astype(np.float32)

    fluid = ember.fluid.PerfectFluid(
        cp=1005.0, gamma=1.4, mu=2.0e-5, Pr=0.72, T_dtm=400.0
    )

    block = ember.block.Block(shape=(ni, nj, nk))
    block.set_xrt(xrt)
    block.set_Nb(Nb)
    block.set_fluid(fluid)

    rho_o, e_o = fluid.set_P_T(Po, To)
    ho = fluid.get_h(rho_o, e_o)
    so = fluid.get_s(rho_o, e_o)
    a_o = fluid.get_a(rho_o, e_o)
    Vbar = Ma_bulk * a_o
    ember.set_iterative.set_ho_s_Ma_Alpha_Beta(block, ho, so, Ma_bulk, 0.0, 0.0)

    U = Vbar / np.inf
    Omega = U / r_mid
    block.set_Omega(Omega)
    block.set_Vt(Omega * block.r)

    # Boundary conditions consistent with the mean flow: inlet fixes stagnation
    # conditions and swirl angle, outlet fixes static pressure with a backflow
    # state for any transient reverse flow. Non-reflecting throughout, so a wave
    # the perturbed initial condition throws at either end leaves the domain
    # instead of bouncing back down the duct.
    block.patches["inlet"] = ember.patch.InletPatch(i=0)
    block.patches["outlet"] = ember.patch.OutletPatch(i=-1)

    Po_in = block.Po[0].mean()
    To_in = block.To[0].mean()
    Alpha_in = block.Alpha[0].mean()
    P_out = block.P[-1].mean()
    T_out = block.T[-1].mean()
    block.patches["inlet"].set_Po_To(Po_in, To_in)
    block.patches["inlet"].set_Alpha(Alpha_in)
    block.patches["inlet"].set_Beta(0.0)
    block.patches["outlet"].set_P(P_out)
    block.patches["outlet"].set_backflow_ho_s(ho, so)
    block.patches["outlet"].set_backflow_Vt(0.0)

    # Optional pitchwise periodicity. The duct's k faces are walls by default;
    # making them periodic gives a block that is periodic to ITSELF in k, the
    # topology the fused viscous seam study needs.
    #
    # Appended HERE, before grid.calculate_wdist() below and before anything
    # reads a wall array, because block.ijk_wall_visc, block.i_perk and
    # block._face_wall_arrays_slip are all cached_object and their docstrings
    # forbid modifying patches after first access. Appending after the grid is
    # built would leave wallk1 at 0.0 across the whole seam AND i_perk at
    # (0, 0), silently -- the wall distance would be wrong too.
    if periodic_k is not None:
        if periodic_k == "full":
            i_lims = [(0, -1)]
        elif periodic_k == "hmesh":
            # Periodic over the upstream and downstream thirds with a "blade"
            # (wall) between: i_perk reads back (i_LE, i_TE) rather than the
            # degenerate (ni, 1) of the full-span case.
            i_le = (ni - 1) // 3
            i_lims = [(0, i_le), (ni - 1 - i_le, -1)]
        else:
            raise ValueError(
                f"periodic_k must be None, 'full' or 'hmesh', got {periodic_k!r}"
            )
        for i_lim in i_lims:
            block.patches.append(ember.patch.PeriodicPatch(k=0, i=i_lim))
            block.patches.append(ember.patch.PeriodicPatch(k=-1, i=i_lim))

    # Velocity ripple, applied before the non-dimensional reference is set (as in
    # the original example/script ordering).
    rng = np.random.default_rng(perturb_seed)
    Vx = block.Vx
    block.set_Vx(
        Vx * (1.0 + perturb_vx * rng.standard_normal(Vx.shape)).astype(Vx.dtype)
    )

    grid = ember.grid.Grid([block])
    grid.set_L_ref(side)
    grid.set_fluid(
        fluid.change_datum(P_out, T_out).change_ref(rho_o, Vbar, block.Rgas.mean())
    )
    grid.calculate_wdist()

    # Deterministic thermodynamic ramp: offset ho and s, then ramp Vx along the
    # duct, so the solution has to relax back to the boundary-consistent state.
    V = np.asarray(block.V)
    ho_field = np.asarray(block.ho)
    s_field = np.asarray(block.s)
    T_field = np.asarray(block.T)
    h_static = ho_field - 0.5 * V**2
    dh = ho_frac * 0.5 * V**2
    ds = s_frac * 0.5 * V**2 / T_field
    block.set_h_s(h_static + dh, s_field + ds)  # velocity preserved

    Vx = np.asarray(block.Vx)
    ramp = np.linspace(1.0, 1.0 + vx_ramp, ni, dtype=Vx.dtype)
    block.set_Vx(Vx * ramp[:, None, None])

    return grid


def er_for_duct_yplus(yplus=30.0, *, correlation="white", x_ref_frac=1.0, **kwargs):
    """Solve :func:`build_duct_grid`'s ``ER`` for a wall cell at ``yplus``.

    Clustering picked by eye is how a duct ends up with its first cell deep
    inside the viscous sublayer, where the wall function changes branch, on a
    mesh whose wall cells are tens of times longer than they are tall. Here the
    expansion ratio is solved for instead: a first pass builds the duct
    UNIFORMLY (``cluster=False``) purely to read back the mean flow and the
    geometry, since ``build_duct_grid`` fixes ``rho``, ``U`` and ``mu`` from
    ``Ma_bulk``, ``Po`` and ``To``, so none of them is known until a grid
    exists. A flat-plate skin-friction correlation evaluated at
    ``x_ref_frac`` of the duct length then sets the friction velocity, hence
    the wall spacing that puts the first node at ``yplus``, and
    :func:`~ember.util.cluster_symmetric`'s first spacing is bisected onto it.

    The correlation is used as a spacing SCALE, not as a prediction of the
    duct's actual wall shear.

    Parameters
    ----------
    yplus : float
        Target wall distance, in wall units, of the first node off the wall.
    correlation : {"white", "prandtl"}
        Local skin-friction correlation for a smooth flat plate at
        Reynolds number ``Re_x = rho*U*x/mu``. ``"white"`` is
        ``cf = 0.455 / ln(0.06*Re_x)**2`` (White, *Viscous Fluid Flow*, 3rd
        ed., eq. 6-73), a log-law fit over roughly ``1e5 < Re_x < 1e9`` and so
        the default; ``"prandtl"`` is the 1/7-power result
        ``cf = 0.0592 * Re_x**-0.2`` quoted in Schlichting, *Boundary-Layer
        Theory*, stated for ``5e5 < Re_x < 1e7``.
    x_ref_frac : float
        Fraction of the duct length at which the correlation is evaluated.
    **kwargs
        Passed to :func:`build_duct_grid` for the probe build, so the ER comes
        out for the grid actually wanted. ``cluster`` is forced off.

    Returns
    -------
    float
        Expansion ratio to pass back to :func:`build_duct_grid` as ``ER``.

    Raises
    ------
    ValueError
        If the target needs a wall cell COARSER than the uniform mesh already
        gives, which no amount of clustering can reach: clustering only
        refines.
    """
    probe = build_duct_grid(**{**kwargs, "cluster": False})[0]
    i0, j0, k0 = probe.ni // 2, probe.nj // 2, probe.nk // 2
    rho = float(probe.rho[i0, j0, k0])
    mu = float(probe.mu[i0, j0, k0])
    U = float(
        np.sqrt(
            probe.Vx[i0, j0, k0] ** 2
            + probe.Vr[i0, j0, k0] ** 2
            + probe.Vt[i0, j0, k0] ** 2
        )
    )
    length = float(probe.x.max() - probe.x.min())
    side = float(probe.r.max() - probe.r.min())
    nj = probe.nj

    Re_x = rho * U * x_ref_frac * length / mu
    if correlation == "white":
        cf = 0.455 / np.log(0.06 * Re_x) ** 2
    elif correlation == "prandtl":
        cf = 0.0592 * Re_x**-0.2
    else:
        raise ValueError(f"unknown correlation {correlation!r}")
    u_tau = np.sqrt(cf * 0.5 * U**2)
    d_target = yplus * mu / (rho * u_tau)

    # cluster_symmetric lays a pure geometric series on each half-width, so the
    # first spacing falls monotonically with ER and a bisection is enough.
    def first(ER):
        return side * float(np.diff(util.cluster_symmetric(nj, ER))[0])

    lo, hi = 1.0 + 1e-7, 3.0
    if first(lo) < d_target:
        raise ValueError(
            f"yplus={yplus} needs a wall cell of {d_target:.4e}, which is "
            f"COARSER than the uniform mesh already gives ({first(lo):.4e}). "
            "Clustering can only refine the wall, so this target is "
            f"unreachable at nj={nj}."
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if first(mid) > d_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
