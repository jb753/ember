"""Canonical test cases assembled from the core building blocks.

This module holds the case-construction scaffolding shared by the square-duct
gallery example (``examples/run_duct.py``) and the CLI sweep driver
(``scripts/run_duct.py``): a single :func:`build_duct_grid` that meshes the
duct, sets the mean flow and boundary conditions, and applies the
initial-condition perturbations.
"""

import numpy as np

import ember.block
import ember.grid
import ember.fluid
import ember.patch
import ember.set_iter
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
        cp=1005.0, gamma=1.4, mu=1.0e-3, Pr=0.72, T_dtm=400.0
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
    ember.set_iter.set_ho_s_Ma_Alpha_Beta(block, ho, so, Ma_bulk, 0.0, 0.0)

    U = Vbar / np.inf
    Omega = U / r_mid
    block.set_Omega(Omega)
    block.set_Vt(Omega * block.r)

    # Boundary conditions consistent with the mean flow: inlet fixes stagnation
    # conditions and swirl angle, outlet fixes static pressure with a backflow
    # state for any transient reverse flow.
    block.patches["inlet"] = ember.patch.InletPatch(i=0)
    block.patches["outlet"] = ember.patch.OutletPatch(i=-1)

    Po_in = block.Po[0].mean()
    To_in = block.To[0].mean()
    Alpha_in = block.Alpha[0].mean()
    P_out = block.P[-1].mean()
    T_out = block.T[-1].mean()
    block.patches["inlet"].set_Po_To_Alpha_Beta(Po_in, To_in, Alpha_in, 0.0)
    block.patches["outlet"].set_P(P_out)
    block.patches["outlet"].set_backflow(ho, so, 0.0, 0.0)

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
