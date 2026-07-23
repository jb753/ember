r"""Configure and run the Navier--Stokes solver.

ember integrates the compressible Navier--Stokes equations to a steady state
using an explicit pseudo-time march, accelerated by multigrid and residual
smoothing. The finite-volume discretisation is second-order accurate in space
on structured multi-block grids; cell-centred residuals are distributed equally
back to nodal conserved variables. This page describes the solver loop driven
by :meth:`Solver.run` and the configuration parameters in :class:`Solver` that
control each stage.

Overview of one time step
--------------------------

:meth:`Solver.run` performs the following operations each step:

1. **Boundary conditions**: :meth:`~ember.grid.Grid.update_bconds` updates
   boundary patch targets (mass flow throttle, radial equilibrium,
   mixing-plane exchange) and :meth:`~ember.grid.Grid.apply_bconds` imposes
   those boundary conditions by modifying :attr:`~ember.block.Block.conserved_nd`.
2. **NaN check**: :meth:`~ember.grid.Grid.check_nan` aborts the run early if
   the conserved state has gone non-finite. Sets a flag on the returned
   :class:`~ember.convergence_history.ConvergenceHistory` and leaves the
   invalid field in place for inspection.
3. **Source terms**: :meth:`~ember.grid.Grid.update_sources` updates viscous
   forces and the polar coordinates source term needed to balance the radial
   momentum equation. With the Runge--Kutta integrator,
   :attr:`~Solver.n_stage` ``> 0``, this runs once every full step
   with the source terms held constant for all stages. When using the scree
   integrator, :attr:`~Solver.n_stage` ``== 0``, the source terms are
   recomputed every few steps to save cost.
4. **Update time step**: :meth:`~ember.grid.Grid.update_timestep` computes
   the time step and stores it pre-divided by cell volume in
   attr:``ember.block.Block.block.dt_vol_nd``.
5. **Residual**: :meth:`~ember.grid.Grid.update_residual` calculates the
   unintegrated net-flow residual, with optional implicit residual smoothing
   , :attr:`~Solver.sf_resid`, or negative feedback limiter
   , :attr:`~Solver.dampin`.
6. **Convergence logging**: every :attr:`~Solver.n_step_log` steps,
   :meth:`~ember.convergence_history.ConvergenceHistory.record_convergence`
   and :meth:`~ember.convergence_history.ConvergenceHistory.format_message`
   record and print a :class:`~ember.convergence_history.ConvergenceHistory`
   row using the current residual.
7. **March**: advance the solution with the selected integrator -- Denton's
   scree march, :func:`scree_step`, or Jameson multi-stage Runge--Kutta,
   :func:`rk_step` -- optionally accelerated by multigrid.
8. **Smoothing**: :meth:`~ember.grid.Grid.smooth` applies a
   constant-coefficient blended second- and fourth-order filter to the
   post-march :attr:`ember.block.Block.conserved_nd` field, to provide artificial dissipation and
   suppress odd-even decoupling.
9. **Pseudotime averaging**: over the final :attr:`~Solver.n_step_avg`
    steps, :meth:`~ember.grid.Grid.accumulate_avg` accumulates the conserved
    state into a running average, which
    :meth:`~ember.grid.Grid.finalise_average` uses to replace the
    instantaneous state once the run completes.

.. _march-schemes:

Time integrators
-----------------

:attr:`~Solver.n_stage` selects one of two integrators, applied to every
block each step:

:func:`scree_step` -- :attr:`~Solver.n_stage` ``== 0``
Implements Denton's basic scree scheme (two steps forward, one step back)

.. math::

    \mathcal{U}_{n+1} = \mathcal{U}_n + \left[ 2 \left(\frac{d\mathcal{U}}{dt}\right)_n
    - \left(\frac{d\mathcal{U}}{dt}\right)_{n-1} \right] \Delta t

The residual from the previous step, :math:`(d\mathcal{U}/dt)_{n-1}`, is kept in
:attr:`~ember.block.Block.store` between steps. This scheme is only first-order accurate in pseudotime, unlike Adams--Bashforth, but experience shows it requires less artificial dissipation and is more robust.

:func:`rk_step`  -- :attr:`~Solver.n_stage` ``>= 1``
-- Jameson's classic Runge--Kutta scheme. Every full step
snapshots the starting conserved state
:math:`\mathcal{U}_n`, saving it to :attr:`~ember.block.Block.store`.
Then, :attr:`~Solver.n_stage` substages call
:func:`advance_rk_stage_mg`, each stage :math:`k` marching relative to the starting snapshot but with the residual freshly evaluated on the most recent stage:

.. math::

    \mathcal{U}^{(0)} &= \mathcal{U}_n \\
    \mathcal{U}^{(k)} &= \mathcal{U}_n
    + \alpha_k\,\Delta t \left(\frac{d\mathcal{U}}{dt}\right)^{(k-1)},
    \qquad k = 1, \dots, m \\
    \mathcal{U}_{n+1} &= \mathcal{U}^{(m)}

for :math:`m =` :attr:`~Solver.n_stage` stages with coefficients
:math:`\alpha_k = 1 / (m - k + 1)`, so the final stage takes the full step
:math:`\alpha_m = 1`.

Both integrators share the :ref:`multigrid` acceleration and the constant-coefficient
:ref:`smoothing` step; they differ only in the march formula and in how many
residual evaluations occur per step. At the end of the step, :attr:`~ember.block.Block.conserved_nd` contains the advanced solution.

.. _cfl:

CFL number
----------

The integrators of the previous section scale each cell's residual by
:attr:`~Solver.cfl` and the local volumetric timestep
:math:`\Delta t_\mathrm{vol}` set by :meth:`~ember.grid.Grid.update_timestep`,
so the :math:`\Delta t` appearing in the march formulae is
:math:`\mathrm{cfl}\,\Delta t_\mathrm{vol}` per cell. The timestep is the
reciprocal of the larger of a convective and a turbulent-diffusion spectral
radius (a max-of-directional-radii variant of the JST/Blazek definition):

.. math::

    \Delta t_\mathrm{vol} = \frac{1}{\max(\lambda_\mathrm{conv},\,\lambda_\mathrm{diff})}

For each direction :math:`d \in \{i, j, k\}`, with :math:`\mathbf{S}_d` the mean
of the two opposing face-area vectors, :math:`\mathbf{V}_\mathrm{rel}` the
relative-frame velocity, :math:`a` the speed of sound, :math:`\mu_t` the
turbulent viscosity, :math:`\rho` the density and :math:`\mathcal{V}` the cell
volume, the directional convective radius and the two combined radii are

.. math::

    \Lambda_d &= \left| \mathbf{V}_\mathrm{rel} \cdot \mathbf{S}_d \right|
    + a \left\| \mathbf{S}_d \right\| \\
    \lambda_\mathrm{conv} &= \max_d \Lambda_d \\
    \lambda_\mathrm{diff} &= f_\mathrm{visc}\, \frac{\mu_t}{\rho}\,
    \frac{\max_d \left\| \mathbf{S}_d \right\|^2}{\mathcal{V}}

Taking the max over directions (rather than Blazek's sum) makes
:attr:`~Solver.cfl` the true 1D Courant limit, so it stays
aspect-ratio-independent for both limits and a single value scales them
consistently; :attr:`~Solver.fac_visc` tightens only the diffusion radius
so the viscous march tolerates the same CFL as the inviscid one. Setting
:math:`\mu_t = 0` drops :math:`\lambda_\mathrm{diff}` and recovers the bare
convective limit. The 4-stage Runge--Kutta march is stable up to
:math:`\mathrm{cfl} \approx 2\sqrt{2}`; the scree scheme is stable up to :math:`\mathrm{cfl} \approx 0.6` on a uniform mesh.

:attr:`~Solver.cfl` is a single constant applied uniformly to every cell
for the entire run -- there is no per-cell adaptive CFL field and no
tolerance-driven backoff inside the solver. A larger CFL converges faster but
risks divergence; implicit residual smoothing (:attr:`~Solver.sf_resid`)
damps high-frequency residual content and so tolerates a substantially higher
CFL for a given scheme.

.. _smoothing:

Smoothing
---------

A constant-coefficient blend of second- and fourth-difference operators, :meth:`ember.grid.Grid.smooth`, is applied to :attr:`ember.block.Block.conserved_nd` after
each step to suppress odd--even decoupling and high-frequency content
introduced by the march and multigrid corrections.
:attr:`~Solver.sf4` and :attr:`~Solver.sf2` are coefficients on the
fourth- and second-difference terms, each scaled by the run's ``cfl`` to make the effective dissipation independent of the time step.

.. _multigrid:

Multigrid
---------

**In-step Denton block-sum multigrid.** Coarse-grid corrections are computed
in place within a single march call over one grid -- coarse block-sum
corrections are folded directly into the fine-grid increment before it is
scattered onto the conserved state -- rather than as a classical
restrict/prolong V-cycle across separate coarse grids. Both integrators honor
two knobs:

- :attr:`~Solver.n_levels` -- number of coarse levels; 0 disables
  multigrid. Each block's cell counts must be an exact multiple of the coarsest
  block size ``2**n_levels`` in every direction, or :meth:`Solver.run` raises before
  marching.
- :attr:`~Solver.fac_mgrid` -- scaling on the coarse correction, with
  successively coarser levels damped further; 0 also disables multigrid.

:attr:`~Solver.sf_resid` additionally drives implicit residual smoothing
(Jameson IRS) on the fine-grid residual via
:meth:`~ember.grid.Grid.update_residual`, independent of whether multigrid is
enabled.

**Full multigrid startup.** :meth:`Solver.run_fmg` runs the same solver
coarse-to-fine as a startup schedule, rather than within a single step. It
builds :attr:`~Solver.n_levels` progressively-halved grids
(:meth:`~ember.grid.Grid.resample`), solves the coarsest first, then
prolongs each converged solution onto the next finer grid as its initial
guess (:meth:`~ember.grid.Grid.interp_from`) and calls :meth:`Solver.run`
again with the in-step multigrid depth set to that level's index -- so the
coarsest level runs with no in-step multigrid and the finest runs at the full
requested :attr:`~Solver.n_levels`, identical to calling :meth:`Solver.run`
directly on the finest grid. With ``n_levels <= 0`` it reduces to a single call
to :meth:`Solver.run`.

.. _body-forces:

Body forces and viscous model
------------------------------

The cell-centred body-force buffer (``block.F_body_nd``) accumulates all
source terms before they are added to the residual, rebuilt by
:meth:`ember.grid.Grid.update_sources`:

- Viscous shear stresses and heat flux, computed unless
  ``Solver.inviscid`` is set. The viscous pass is phased across the
  whole grid (tau/q on every block, then a periodic-seam halo exchange, then
  face-flux accumulation) so block-to-block periodic interfaces stay
  consistent.
- A polar (axisymmetric) source term to balance the cylindrical coordinate
  metric.
- An optional selective-frequency-damping (SFD) force when
  ``Solver.gain_filt`` is nonzero.

The mixing-length turbulent viscosity uses a fixed turbulent Prandtl number
of 1.0 and is evaluated from the absolute-frame vorticity magnitude.
``Solver.fac_visc`` multiplies the turbulent-diffusion timestep radius
independently of this, tightening the viscous stability limit to recover the
inviscid stable CFL where needed.

.. _boundary-coupling:

Boundary patches and inter-block coupling
------------------------------------------

Inlet, outlet, and mixing-plane patches each relax their own state towards a
target every step, with their own relaxation factor rather than a single
solver-wide setting:

- :class:`~ember.inlet.InletPatch` relaxes the face velocity from its
  characteristic solve with ``rf = 1.0`` by default (no relaxation); because
  that target is well conditioned, ``rf < 1`` only adds startup lag.
- :class:`~ember.mixing.MixingPatch` holds no relaxation of its own: it imposes
  whatever target the exchange last wrote.
- :class:`~ember.mixing_communicator.MixingCommunicator` relaxes the
  mixing-plane target exchanged between adjacent blocks with ``rf_mix``
  (default 0.05), which is the only damping the reflecting plane has.
- :class:`~ember.outlet.OutletPatch` takes its own relaxation factor via
  ``set_adjustment(rf=...)``.

:meth:`ember.grid.Grid.update_bconds` advances the slowly-varying boundary
targets once per step (mixing-plane exchange, inlet velocity snapshot,
outlet PID/spanwise target); :meth:`ember.grid.Grid.apply_bconds` then
imposes the full set of physical boundary conditions and closes periodic
seams every time it is called, including between Runge--Kutta substages.

.. _logging-and-averaging:

Logging, averaging, and convergence history
---------------------------------------------

Convergence diagnostics are recorded into a
:class:`~ember.convergence_history.ConvergenceHistory` every
``Solver.n_step_log`` steps: mean residual, mass flow / stagnation
enthalpy / entropy at row interfaces, and outlet throttle state
(:meth:`ember.grid.Grid.get_convergence`).

Pseudotime averaging of the conserved variables accumulates over the final
``Solver.n_step_avg`` steps of the run
(:meth:`ember.grid.Grid.accumulate_avg`). On completion, the time-averaged
state replaces the instantaneous state
(:meth:`ember.grid.Grid.finalise_average`) -- skipped if the run diverged, so
the invalid field is preserved for inspection rather than overwritten by a
partially-accumulated average.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import ember
from ember import util
from ember.convergence_history import ConvergenceHistory
from ember.grid import DivergenceError

logger = logging.getLogger(__name__)


class BaseSolver(ABC):
    """Common interface for in-place flow solvers.

    A solver is constructed from its configuration and run with
    ``solver.run(grid)``, which advances ``grid`` in place and returns the
    convergence history. Both the built-in :class:`Solver` and external-solver
    plugins (e.g. ``ember.plugins.ts.TS3Solver``) implement this contract so
    they are drop-in interchangeable.
    """

    @abstractmethod
    def run(self, grid):
        """Solve on ``grid`` in place; return a ``ConvergenceHistory``."""
        raise NotImplementedError


@dataclass
class Solver(BaseSolver):
    """Configuration for the explicit time-marching solver.

    Also the entry point: build one with the parameters below and call
    :meth:`run` (or :meth:`run_fmg`) to march a grid in place.
    """

    n_step: int

    n_step_log: int = 10
    """Number of steps between convergence log messages."""

    n_step_avg: int = 1
    """Number of steps to average over."""

    cfl: float = 0.4
    """Constant CFL number for the march"""

    sf4: float = 0.008
    """Fourth-order smoothing factor."""

    sf2: float = 0.002
    """Second-order smoothing factor."""

    dampin: float | None = None
    """Negative-feedback damping factor on integrated residual."""

    inviscid: bool = False
    """Skip viscous terms in the sources evaluation."""

    fac_visc: float = 1.0
    """Multiplier on the turbulent-diffusion timestep radius; >1 tightens the
    viscous limit to recover the inviscid stable CFL."""

    sf_resid: float = 0.0
    """Implicit residual smoothing factor. Applied to the fine residual by
    :meth:`~ember.grid.Grid.update_residual` (``sf``) and, on both integrators,
    to the coarse block-restricted residual of the multigrid correction
    (:func:`advance_rk_stage_mg`'s ``sf_irs`` for RK, :func:`scree_step`'s
    ``sf_irs`` for scree). The coarse smoothing needs ``n_levels > 0`` to have
    any effect."""

    gain_filt: float = 0.0
    """Selective frequency damping gain."""

    delta_filt: float = 1.0
    """Selective frequency damping filter width (higher is smoother)."""

    n_stage: int = 0
    """Number of time integration stages per step. 0 for scree, >=1 for RK."""

    n_levels: int = 0
    """Number of coarse multigrid levels; 0 disables multigrid. Honored by
    both integrators (:func:`scree_step` and :func:`rk_step`)."""

    fac_mgrid: float = 0.2
    """Scaling factor on multigrid corrections. Honored by both integrators
    (:func:`scree_step` and :func:`rk_step`)."""

    def run(self, grid):
        """Drive ``grid`` through ``n_step`` steps in place; return the history.

        See :func:`_run` for the stage-by-stage march; this is the public
        :class:`BaseSolver` entry point.
        """
        return _run(grid, self)

    def run_fmg(self, grid):
        """Full-multigrid startup on ``grid`` in place; see :func:`_run_fmg`.

        Returns a list of per-level :class:`ConvergenceHistory`, coarsest first.
        Not part of the :class:`BaseSolver` contract (plugins have no FMG
        analogue).
        """
        return _run_fmg(grid, self)


def scree_step(grid, cfl, fac_mgrid=0.0, n_levels=0, sf_irs=0.0):
    """Advance every block one Denton scree step in place."""
    # Preconditions: dt_vol_nd populated and cached P/T consistent with
    # conserved_nd on entry. The caller invalidates caches and applies boundary
    # conditions between steps, and smooths once on the post-step state
    # (Grid.smooth, shared with the RK path); no smoothing happens here.
    #
    # fac_mgrid == 0 scales every coarse correction to identically zero, so
    # collapse to the plain no-MG dispatch (which also makes sf_irs inert)
    # rather than paying restrict/prolong per level for a guaranteed-zero push.
    n_levels_eff = n_levels if fac_mgrid > 0.0 else 0

    for block in grid:
        ni, nj, nk = block.shape
        cell_shape = (ni - 1, nj - 1, nk - 1, 5)
        # Borrow the nodal scratch (shape (ni,nj,nk,5)) as the cell-shaped
        # increment buffer -- a zero-copy view. block.store is likewise sized to
        # the nodal shape (shared with the RK path) but the Denton residual
        # history is cell-shaped, so carve a leading cell-shaped view of it too.
        tmp = util.carve_view(block.scratch, cell_shape)
        store_cell = util.carve_view(block.store, cell_shape)
        # residual_nd is read here (a cache hit from the loop's get_convergence, or
        # a fresh evaluation) BEFORE the kernel runs. Evaluating residual_nd
        # borrows block.scratch (IRS work) and tau_q_halo (rolling flow
        # buffers), both reused as scratch by the kernels below, so it must be
        # fully materialised first -- passing it as an argument guarantees
        # that ordering.
        if n_levels_eff > 0:
            # Denton block-sum multigrid over the scheme-agnostic engine
            # (mg_coarse_correction), like advance_rk_stage_mg's coarse path but
            # for the scree fine term. Every level -- fine and coarse -- uses the
            # Denton-lagged quantity q = 2*residual - store, not plain residual;
            # verified against multall's TSTEP, which sums the lagged
            # STORE = F1*DELTA + F2*DIFF into its block accumulators. For coarse
            # level l = 1..n_levels (block size b = 2**l) the correction scales by
            # coef_l = cfl*fac_mgrid/b**2 * 2**-(l-1) (advance_rk_stage_mg's
            # formula at alpha=1, since scree takes one full-weight step), with
            # the coarse timestep the volume-weighted mean of dt_vol over the
            # block. sf_irs > 0 selects the coarse-IRS kernel; sf_irs == 0 selects
            # the plain _noirs kernel, which enters no smoothing code at all (no
            # Fortran-side IRS branch -- the two share the engine and differ only
            # in the smoother passed). The fine term is not smoothed here: it
            # already carries the residual the caller's update_residual smoothed.
            # Coarse scratch is carved from tau_q_halo, dead outside the viscous
            # pass (already completed and consumed before this call).
            kernel = (
                ember.fortran.scree_mg_irs
                if sf_irs > 0.0
                else ember.fortran.scree_mg_noirs
            )
            kernel(
                cons=block.conserved_nd,
                residual=block.residual_nd,
                store=store_cell,
                dt_vol=block.dt_vol_nd,
                vol=block.vol_nd,
                cfl=cfl,
                fmgrid=fac_mgrid,
                sf_irs=sf_irs,
                n_levels=n_levels_eff,
                tmp=tmp,
                **_mg_coarse_carve(block, ni, nj, nk, n_levels_eff),
            )
        else:
            # Multigrid off: fine term only, no coarse scratch. Forms
            # q = 2*residual - store in store, builds the increment cfl*dt_vol*q,
            # rolls the history (store <- residual) and frozen-scatters the
            # increment straight onto cons -- bypassing the setters so the P/T
            # cache stays frozen. Untouched by fac_mgrid/n_levels.
            ember.fortran.scree_plain(
                cons=block.conserved_nd,
                residual=block.residual_nd,
                store=store_cell,
                dt_vol=block.dt_vol_nd,
                cfl=cfl,
                tmp=tmp,
            )


def _mg_coarse_scratch_sizes(ni, nj, nk, n_levels, np=5):
    """Element counts for the hier2 kernels' flat packed scratch.

    ``n_corr`` sizes ``corr_all``, which holds every coarse level's scaled
    correction back-to-back (coarsest level first, where the cascade seeds) --
    the sum of per-level element counts. ``cres``/``triw`` are reused per level
    rather than packed, so they only need the largest (level-1) slice: ``n_res``
    is that coarse residual size and ``n_tri`` its IRS Thomas-coefficient size.
    All three are carved once per call from ``block.tau_q_halo``, never
    reallocated -- see :func:`advance_rk_stage_mg` and :func:`scree_step`.
    """
    n_corr = 0
    for lvl in range(1, n_levels + 1):
        b = 2**lvl
        n_corr += ((ni - 1) // b) * ((nj - 1) // b) * ((nk - 1) // b) * np
    nc1i, nc1j, nc1k = (ni - 1) // 2, (nj - 1) // 2, (nk - 1) // 2
    n_res = nc1i * nc1j * nc1k * np if n_levels > 0 else 0
    n_tri = 2 * (nc1i + nc1j + nc1k) if n_levels > 0 else 0
    return n_corr, n_res, n_tri


def _mg_coarse_carve(block, ni, nj, nk, n_levels_eff):
    """Carve the hier2 kernels' scratch from ``block.tau_q_halo`` (dead outside
    the viscous pass, which has completed and been consumed before this call, so
    it is free private memory here). Returns the argument dict shared by both the
    RK and scree fused kernels.
    """
    nc1i, nc1j, nc1k = (ni - 1) // 2, (nj - 1) // 2, (nk - 1) // 2
    n_corr, n_res, n_tri = _mg_coarse_scratch_sizes(ni, nj, nk, n_levels_eff)
    acc_sz = nc1i * nc1j * nc1k * 5
    aplane, bb, dtblk, rawbuf, sdt, sv, corr_all, acc0, acc1, cres, triw = (
        util.carve_view(
            block.tau_q_halo,
            (ni - 1, nc1j),
            (ni - 1, nj - 1, nc1k, 5),
            (nc1i, nc1j, nc1k),
            (nc1i, nc1j, nc1k, 5),
            (nc1i, nc1j, nc1k),
            (nc1i, nc1j, nc1k),
            (n_corr,),
            (acc_sz,),
            (acc_sz,),
            (n_res,),
            (n_tri,),
        )
    )
    return dict(
        dtblk=dtblk,
        aplane=aplane,
        bb=bb,
        rawbuf=rawbuf,
        sdt=sdt,
        sv=sv,
        corr_all=corr_all,
        acc0=acc0,
        acc1=acc1,
        cres=cres,
        triw=triw,
    )


def advance_rk_stage_mg(grid, alpha, cfl, fac_mgrid, n_levels, sf_irs=0.0):
    r"""One Jameson RK stage, optionally with Denton block-sum multigrid.

    The single RK stage integrator. Each stage marches every block off its
    step-start conserved snapshot (``block.store``, seeded by the caller) using
    the residual evaluated on the previous stage's state. In one pass it
    assembles a cell-centred increment -- the fine RK term plus the injected
    coarse block corrections of ``n_levels`` coarse levels -- and scatters it
    onto the snapshot (multall's DO 1500 combine, then ``cell_to_node``)::

        dU_cell = alpha*cfl*dt_vol*residual                          (fine)
                + sum_l  inject_l( coef_l * dt_coarse_l * restrict_l(residual) )
        cons    = snapshot + cell_to_node(dU_cell)

    ``n_levels`` counts the coarse levels only. ``n_levels == 0`` (the default)
    is the trivial subcase: the coarse loop is empty, so the stage reduces to a
    plain Jameson RK step ``cons = snapshot + alpha*cfl*dt_vol*residual``. For
    ``l = 1..n_levels`` the coarse block has ``b = 2**l`` and
    ``coef_l = alpha*cfl*fac_mgrid/b**2 * 2**-(l-1)``. The ``2**-(l-1)`` damps
    successively coarser levels: level 1 (finest coarse, ``b=2``) carries the
    full ``fac_mgrid``, level 2 ``fac_mgrid/2``, level 3 ``fac_mgrid/4``, and so on.

    ``dt_coarse_l`` is the volume-weighted mean of ``dt_vol`` over the coarse
    block, ``sum(dt_vol*vol)/sum(vol)``, which is why the kernels take
    ``block.vol_nd``. This mirrors multall's ``STEP1 =
    CFL*FBLK*PERPMIN/VSOUND/VOLB``: our ``dt_vol*vol`` is the per-cell
    ``perp/(a+V)`` that multall sums into ``PERPMIN``, and the ``1/b**2`` stays
    in ``coef_l``. Sampling ``dt_vol`` at the block's centre cell instead --
    what this used to do -- is wrong by the local clustering ratio on a
    stretched mesh. On a uniform mesh the two agree identically.
    Scaling the block push by the same ``alpha`` as the fine term keeps the stage
    consistent; the final stage (``alpha=1``) therefore lands the full-weight
    coarse correction, matching Denton, while earlier stages damp it like the
    fine residual. Prolongation is **trilinear interpolation**, cascaded coarsest
    -> finest through factor-2 hops (inlined into the fused kernel below).

    The whole per-block body -- fine term, all coarse levels, and the final
    scatter -- runs in one fused Fortran kernel (``rk_mg_irs``/``rk_mg_noirs``,
    thin wrappers over the shared scheme-agnostic engine ``mg_coarse_correction``),
    with no per-level Python crossings or numpy temporaries. With
    ``n_levels == 0`` (or ``fac_mgrid == 0``) the coarse machinery is skipped
    entirely by the ``rk_plain`` kernel (fine term + scatter, no coarse scratch).
    Restriction is **hierarchical**: only level 1
    reads the fine grid, coarser levels reduce the running accumulators
    (``rawbuf`` for the residual, ``sdt``/``sv`` for the volume-weighted dt),
    cutting restriction reads from ``n_levels x N`` to ~``1.14 x N``. Prolongation
    is **cascaded**: the packed per-level corrections (``corr_all``) accumulate
    coarsest -> finest through the ``acc0``/``acc1`` ping-pong, so only the final
    factor-2 hop writes the fine grid (fused with the fine term).
    ``block.scratch`` is borrowed as the cell-shaped increment workspace (nodal,
    free between kernel calls); the coarse timestep (``dtblk``), the restriction
    accumulators, ``corr_all``/``acc0``/``acc1``, the separable-prolong scratch
    (``aplane``, ``bb``) and the coarse-IRS buffers (``cres``, ``triw``) are all
    carved from ``block.tau_q_halo`` at non-overlapping offsets (see
    :func:`_mg_coarse_carve`) -- dead outside the viscous pass and a distinct
    buffer from ``scratch``, so they survive alongside the increment within the
    call. The scatter reads the snapshot from ``block.store`` and writes
    ``conserved_nd`` directly (frozen pressure, bypasses the P/T cache).

    ``dtblk`` is rebuilt inside the kernel on every call, so for RK it is
    recomputed once per stage even though ``dt_vol`` only changes once per step.
    That redundancy is deliberate: confining ``dtblk``'s live range to a single
    kernel call is what makes it safe to borrow ``tau_q_halo``, which
    :meth:`~ember.grid.Grid.update_residual` clobbers between stages. The
    pre-pass costs under 1.15 fine-cell passes of two multiply-adds per level,
    against a full residual evaluation already paid every stage.

    No boundary masking is applied here: ``grid.apply_bconds`` re-imposes the
    inlet/outlet/mixing/cusp targets between stages and at the next step top, so
    the coarse push cannot leave a BC-controlled node inconsistent -- exactly as
    for the fine RK term, which is likewise unmasked.

    ``sf_irs`` (0 disables it, the default) applies implicit residual
    smoothing (Jameson IRS) to the coarse block-restricted residual at every
    level, exactly like the fine-grid smoothing ``Grid.update_residual``
    already applies via its ``sf`` argument -- both are driven by the same
    ``Solver.sf_resid`` value (see :func:`rk_step`). ``sf_irs > 0``
    dispatches ``rk_mg_irs``; ``sf_irs == 0`` (the default) dispatches
    ``rk_mg_noirs``, which enters no smoothing code at all. The two share
    ``mg_coarse_correction`` and differ only in the coarse-residual smoother
    passed to it, so the choice is a Python-side branch
    with no ``sf_irs`` test inside the engine (the fine term is never smoothed
    here -- it already carries the fine residual the caller smoothed). The
    per-level scratch it needs (``cres``, ``triw``) is carved by
    :func:`_mg_coarse_carve` from ``block.tau_q_halo`` -- caller-owned, no
    per-call allocation.

    Assumes ``block.dt_vol_nd`` and ``block.residual_nd`` are populated and the
    caller refreshes P/T, boundary conditions and the residual between stages.
    Requires :func:`_validate_mg` to have passed.

    ``fac_mgrid == 0`` scales every coarse correction to identically zero, so it
    collapses to the plain-RK fast path (``n_levels`` passed as 0, empty coarse
    loop) rather than running restrict/prolong for a guaranteed-zero push -- and
    makes ``sf_irs`` inert, exactly as in :func:`scree_step`.
    """
    # fac_mgrid == 0 makes the coarse loop a no-op; collapse to no-MG dispatch.
    n_levels_eff = n_levels if fac_mgrid > 0.0 else 0
    for block in grid:
        ni, nj, nk = block.shape
        if n_levels_eff > 0:
            # Multigrid-on RK wrappers over the scheme-agnostic engine
            # (mg_coarse_correction). sf_irs > 0 selects the coarse-IRS kernel;
            # otherwise the plain _noirs kernel, which enters no smoothing code
            # (the two share the engine and differ only in the smoother passed --
            # no Fortran-side IRS branch). Coarse scratch is carved from
            # tau_q_halo, dead outside the viscous pass. The engine leaves the
            # coarse correction in acc0; the wrapper's final factor-2 hop is
            # fused with the cell->node scatter, so instead of a full-volume
            # increment it takes a rolling two-plane buffer carved from scratch.
            rbuf = util.carve_view(block.scratch, (ni - 1, nj - 1, 5, 2))
            kernel = (
                ember.fortran.rk_mg_irs if sf_irs > 0.0 else ember.fortran.rk_mg_noirs
            )
            kernel(
                cons=block.conserved_nd,
                snapshot=block.store,
                residual=block.residual_nd,
                dt_vol=block.dt_vol_nd,
                vol=block.vol_nd,
                alpha=alpha,
                cfl=cfl,
                fmgrid=fac_mgrid,
                sf_irs=sf_irs,
                n_levels=n_levels_eff,
                rbuf=rbuf,
                **_mg_coarse_carve(block, ni, nj, nk, n_levels_eff),
            )
        else:
            # Multigrid off: plain Jameson RK fine-term stage, no coarse scratch.
            tmp = util.carve_view(block.scratch, (ni - 1, nj - 1, nk - 1, 5))
            ember.fortran.rk_plain(
                cons=block.conserved_nd,
                snapshot=block.store,
                residual=block.residual_nd,
                dt_vol=block.dt_vol_nd,
                alpha=alpha,
                cfl=cfl,
                tmp=tmp,
            )


@util.profile
def rk_step(grid, conf):
    """Advance every block one Jameson multi-stage RK step in place."""
    # Snapshot the bconds-consistent step-top state U_n into block.store (the
    # caller's residual build did not touch conserved_nd); every stage marches
    # off this frozen snapshot. Smoothing is not applied here -- Grid.smooth runs
    # once on the post-step state, shared with the scree path.
    for block in grid:
        block.store[...] = block.conserved_nd
    for i_stage in range(conf.n_stage):
        # Stage coefficient alpha_k = 1/(n_stage - k); the final stage takes the
        # full step. advance_rk_stage_mg marches off the snapshot and folds in the
        # Denton block-sum multigrid correction (empty coarse loop when
        # conf.n_levels == 0, i.e. a plain RK stage). conf.sf_resid is the
        # coarse-level IRS coefficient, so a nonzero sf_resid smooths both the
        # fine residual (caller's update_residual) and the coarse block-restricted
        # residual. Stage 0 reuses the step-top P/T flush and residual; later
        # stages march off a changed conserved_nd, so P/T and bconds are
        # refreshed below before the next advance.
        alpha = 1.0 / (conf.n_stage - i_stage)
        advance_rk_stage_mg(
            grid, alpha, conf.cfl, conf.fac_mgrid, conf.n_levels, conf.sf_resid
        )
        grid.update_cached_conserved()
        grid.apply_bconds()
        # The residual just feeds the next stage's advance. The final stage has
        # no next stage, and nothing reads residual_nd between here and the next
        # step's top-of-loop rebuild (smooth/accumulate_avg/get_convergence do
        # not, and the pre-march update_residual recomputes it first), so skip
        # the redundant final rebuild.
        if i_stage < conf.n_stage - 1:
            grid.update_residual(dampin=conf.dampin, sf=conf.sf_resid)


def _validate_mg(grid, n_levels):
    """Raise if any block cannot be evenly divided into the coarsest MG blocks.

    Denton block-sum multigrid with ``n_levels`` coarse levels groups cells into
    blocks of linear size up to ``2**n_levels``. We require exact division (no
    runt edge blocks), so every cell dimension ``ni-1, nj-1, nk-1`` must be a
    multiple of the coarsest block size. Since the sizes are powers of two,
    divisibility by the coarsest implies it for every finer level, so one check
    per dimension suffices. No-op when ``n_levels <= 0``.
    """
    if n_levels <= 0:
        return
    b_coarse = 2**n_levels
    for i_block, block in enumerate(grid):
        ni, nj, nk = block.shape
        for name, n_cell in (("i", ni - 1), ("j", nj - 1), ("k", nk - 1)):
            if n_cell % b_coarse != 0:
                raise ValueError(
                    f"Block {i_block} has {n_cell} cells in {name} but "
                    f"multigrid n_levels={n_levels} needs a multiple of the "
                    f"coarsest block size {b_coarse}."
                )


@util.profile
def _run(grid, conf):
    """Drive a grid through ``n_step`` explicit time-marching steps.

    See the module-level `Overview of one time step`_ for the stage-by-stage
    ordering within each step.

    ``dt_vol`` is relaxed in place by :meth:`~ember.grid.Grid.update_timestep`
    (the kernel blends ``rf*new + (1-rf)*old``); the initial call before the
    step loop uses ``rf=1.0`` to seed the uninitialised buffer, ``rf=0.2``
    thereafter.

    Returns
    -------
    ConvergenceHistory
        The recorded history, already trimmed to the steps it logged, so every
        row holds data and no ``isfinite`` masking is needed. If the march blew
        up, the step loop breaks early,
        :attr:`~ember.convergence_history.ConvergenceHistory.diverged` is True, and ``grid`` keeps the
        invalid field for inspection (the pseudotime average is not finalised,
        since it would overwrite ``conserved_nd`` with a buffer that
        ``accumulate_avg`` may never have populated).
    """

    # Fail fast if the grid cannot be evenly blocked for multigrid.
    _validate_mg(grid, conf.n_levels)

    # Initialise timesteps
    grid.update_timestep(rf=1.0, fac_visc=conf.fac_visc)

    # One record per log step: i_step % n_step_log == 0 fires ceil(n_step /
    # n_step_log) times over range(n_step); floor division would under-allocate.
    n_log = -(-conf.n_step // conf.n_step_log)
    hist = ConvergenceHistory.from_grid(n_log, grid)

    for i_step in range(conf.n_step):
        #
        # We overwrote conserved_nd in place last step
        # So flush the cache to recalculate P and T
        grid.update_cached_conserved()

        grid.update_bconds()  # Throttle/radial equilibrium targets
        grid.apply_bconds()

        try:
            grid.check_nan()
        except DivergenceError as err:
            logger.error("Solver diverged at step %d: %s", i_step, err)
            hist.diverged = True
            break

        # Refresh source terms every step for the multi-stage RK march, which
        # re-evaluates the residual each substep and needs current sources; the
        # scree march reuses one residual per step, so lag the expensive viscous
        # pass to every fifth step there. Recompute the timestep every step
        # regardless so dt_vol tracks the flow through fast transients: a lagged
        # dt_vol is sized for an already-stale state and overshoots the
        # stability limit during a cold start, and the timestep refresh is cheap.
        n_step_source = 5 if conf.n_stage == 0 else 1
        if i_step % n_step_source == 0:
            # grid.update_filter(conf.delta_filt)
            grid.update_sources(conf.inviscid, conf.gain_filt)
        grid.update_timestep(rf=0.2, fac_visc=conf.fac_visc)

        # Prepare the residual
        grid.update_residual(dampin=conf.dampin, sf=conf.sf_resid)

        # Convergence logging of the pre-march state
        if i_step % conf.n_step_log == 0:
            hist.record_convergence(i_step, grid.get_convergence())
            logger.info(
                "%s",
                hist.format_message(n_step=conf.n_step),
            )

        # Take a step with the selected integrator.  Both reuse the first
        # residual evaluated above, RK then recalculates each substep
        if conf.n_stage == 0:
            scree_step(grid, conf.cfl, conf.fac_mgrid, conf.n_levels, conf.sf_resid)
        else:
            rk_step(grid, conf)

        # Smooth the post-step conserved solution
        grid.smooth(conf.sf4 * conf.cfl, conf.sf2 * conf.cfl)

        # Pseudotime avearge over last n_step_avg steps
        if i_step >= (conf.n_step - conf.n_step_avg):
            grid.accumulate_avg(conf.n_step_avg)

    # Copy the final average back in the primary storage. Skipped on divergence:
    # the loop broke before accumulate_avg ran, so this would overwrite the
    # invalid conserved_nd with a zeroed average buffer and destroy the evidence.
    if not hist.diverged:
        grid.finalise_average()

    # A completed march logs on every one of the ceil(n_step / n_step_log) rows
    # that from_grid allocated, so this only bites when the loop broke early:
    # the caller never sees the unwritten NaN tail a divergence leaves behind.
    return hist.trim()


def _run_fmg(grid, conf):
    """Full-multigrid startup: solve coarse-to-fine, prolonging each guess.

    ``conf.n_levels`` is the single grid-hierarchy depth. With ``n_levels == 0``
    this is exactly :func:`run(grid, conf) <run>`. Otherwise it builds
    ``n_levels`` grids successively halved by :meth:`~ember.grid.Grid.resample`
    at factor ``0.5``, solves the coarsest, and
    :meth:`~ember.grid.Grid.interp_from`'s the solution up onto each finer grid
    as its initial guess. Sequencing level ``i`` (``0`` = coarsest) is marched
    with in-step Denton block-sum multigrid depth ``i`` -- the same grid
    hierarchy ``n_levels`` already names -- so the coarsest runs plain and the
    finest runs at full ``n_levels``, identical to :meth:`Solver.run` on the finest.

    The single validation ``_validate_mg(grid, n_levels)`` is sufficient for the
    whole chain: level ``i`` holds ``N / 2**(n_levels - i)`` cells per dimension
    and its depth-``i`` march needs those divisible by ``2**i``, which reduces to
    ``N`` (the finest cell count) divisible by ``2**n_levels`` at every level.
    That divisibility also keeps every ``resample(0.5)`` node-coincident with the
    finer grid (an exact subset of its nodes), so the coarse coordinates are the
    true geometry, not interpolated.

    Every level runs the same ``conf`` apart from ``n_levels`` (fixed
    ``n_step`` per level). ``grid`` is the finest level and is mutated in place
    to carry the final solution, matching :meth:`Solver.run`.

    Parameters
    ----------
    grid : Grid
        Finest grid, already carrying its initial guess. Mutated in place.
    conf : Solver
        Solver configuration; ``conf.n_levels`` sets the hierarchy depth.

    Returns
    -------
    list of ConvergenceHistory
        Per-level histories, coarsest first, finest last.
    """
    _validate_mg(grid, conf.n_levels)
    if conf.n_levels <= 0:
        return [_run(grid, conf)]

    # Build finest -> coarsest, then reverse. resample carries the already-set
    # fine guess down, so the coarsest starts from the coarsened cold start.
    chain = [grid]
    for _ in range(conf.n_levels):
        chain.append(chain[-1].resample(0.5))
    chain.reverse()  # chain[-1] is the original `grid`

    histories = []
    for i, level_grid in enumerate(chain):  # i == in-step MG depth for this mesh
        if i > 0:
            level_grid.interp_from(chain[i - 1])  # prolong previous solution
        logger.info(
            "FMG level %d/%d, shape(s) %s",
            i,
            conf.n_levels,
            [block.shape for block in level_grid],
        )
        histories.append(_run(level_grid, replace(conf, n_levels=i)))
    return histories
