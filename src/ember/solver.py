"""Explicit time-marching solver with a lagged-pressure loop.

:func:`run` drives a whole grid to a steady state, selecting one of two
integrators on each step according to :attr:`SolverConfig.n_stage`:

* ``n_stage == 0`` -- Denton's basic "scree" march (:func:`scree_step`),

    F_{n+1} = F_n + [2*(dF/dt)_n - (dF/dt)_{n-1}] * dt    (Denton 2017, Eq 4)

  which builds the unscaled residual exactly as multall (F1=2, F2=-1, F3=0),
  scales it by dt_vol*CFL, and distributes it to the nodes by accumulating
  straight onto ``conserved_nd`` -- bypassing the setters so the P/T cache
  versions are left untouched (frozen pressure);

* ``n_stage >= 1`` -- a Jameson multi-stage Runge-Kutta step (:func:`rk_step`),
  optionally accelerated by Denton block-sum multigrid
  (:func:`advance_rk_stage_mg`).

Both integrators share the same lagged-pressure loop: the conserved cache is
flushed once per step (one full-field pressure evaluation), the body force and
dt_vol are refreshed only every few steps on that fresh P, the march and the
constant-coefficient smoother (:meth:`Grid.smooth`) then run on the frozen
state, and the boundary conditions touch only boundary slices. This keeps
exactly one full-field P recompute per step; everything else reuses the cache.
"""

import logging
from dataclasses import dataclass

import ember
from ember import util
from ember.convergence_history import ConvergenceHistory
from ember.grid import SolverDivergedError

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """Configuration for the explicit time-marching solver."""

    n_step: int

    n_step_log: int = 10
    """Number of steps between convergence log messages."""

    n_step_source: int = 1
    """Number of steps between source term refreshes."""

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
    """Implicit residual smoothing factor"""

    gain_filt: float = 0.0
    """Selective frequency damping gain."""

    delta_filt: float = 1.0
    """Selective frequency damping filter width (higher is smoother)."""

    n_stage: int = 0
    """Number of time integration stages per step. 0 for scree, >=1 for RK."""

    n_levels: int = 0
    """Number of coarse multigrid levels; 0 disables multigrid."""

    fac_mgrid: float = 0.2
    """Scaling factor on multigrid corrections."""


def scree_step(grid, cfl):
    """Advance `block` one scree step in place.

    Assumes ``block.dt_vol_nd`` is populated and the block's cached
    P/T are consistent with ``conserved_nd`` on entry. The caller is
    responsible for invalidating caches and applying boundary conditions
    between steps.

    The whole step is one fused Fortran kernel (``scree_advance``): it builds
    the extrapolated, scaled increment ``(2*residual - store)*CFL*dt_vol``,
    rolls the residual history (``store <- residual``), and scatters the
    cell-centred increment straight onto ``conserved_nd`` -- bypassing the
    setters so the P/T cache versions stay frozen.

    ``block.store`` is the persistent previous-step residual buffer
    (``(dF/dt)_{n-1}``); the kernel reads it then overwrites it with the
    current residual for the next step. The transient increment buffer is
    borrowed from ``block.scratch`` as a cell-shaped, zero-copy view of the
    leading elements -- nothing outside the kernel reads it, so only its
    element count matters, not its indexing.

    Smoothing is not applied here -- the caller runs :meth:`Grid.smooth` once on
    the post-step state, shared with the RK path.
    """

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
        # a fresh evaluation) BEFORE the kernel runs. residual_nd itself uses
        # block.scratch as its flow_i workspace, so it must be fully materialised
        # before scree_advance reuses scratch as tmp -- passing it as an argument
        # guarantees that ordering.
        ember.fortran.scree_advance(
            cons=block.conserved_nd,
            residual=block.residual_nd,
            store=store_cell,
            dt_vol=block.dt_vol_nd,
            cfl=cfl,
            tmp=tmp,
        )


def _mg_irs_scratch_sizes(ni, nj, nk, n_levels, np=5):
    """Element counts for advance_rk_stage_mg_fused_irs's flat packed scratch.

    coarse_res_buf/tri_work_buf hold each level's slice back-to-back (level 1
    -- the largest, b=2 -- first, each successive level 8x smaller), so this
    is just the sum of per-level element counts. The caller carves buffers of
    exactly this size once per call (from ``block.tau_q_halo``, never
    reallocated) -- see :func:`advance_rk_stage_mg`.
    """
    n_res = n_tri = 0
    for lvl in range(1, n_levels + 1):
        b = 2**lvl
        nib, njb, nkb = (ni - 1) // b, (nj - 1) // b, (nk - 1) // b
        n_res += nib * njb * nkb * np
        n_tri += 2 * (nib + njb + nkb)
    return n_res, n_tri


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
    Scaling the block push by the same ``alpha`` as the fine term keeps the stage
    consistent; the final stage (``alpha=1``) therefore lands the full-weight
    coarse correction, matching Denton, while earlier stages damp it like the
    fine residual. There is **no coarse-correction smoothing**, and prolongation
    is **trilinear interpolation** (inlined into the fused kernel below).

    The whole per-block body -- fine term, all coarse levels, and the final
    scatter -- runs in one fused Fortran kernel
    (:func:`ember.fortran.advance_rk_stage_mg_fused_opt`), with no per-level
    Python crossings or numpy temporaries. That variant restructures the coarse
    path for speed (~2x on the coarse levels at production sizes): restrict is a
    coarse-cell register reduction folding in the zero+scale passes, and the
    trilinear prolong is done as separable 1-D interpolations (a cached per-kb
    plane instead of an 8-way gather), with the fine term folded into level 1.
    ``block.scratch`` is borrowed as the cell-shaped increment workspace (nodal,
    free between kernel calls); the coarse block-sum accumulator (``corr``,
    sized to the finest coarse level with coarser levels using only its leading
    corner) and the separable-prolong scratch (``aplane``, ``bb``) are all
    carved from ``block.tau_q_halo`` at non-overlapping offsets -- dead outside
    the viscous pass and a distinct buffer from ``scratch``, so they survive
    alongside the increment within the call. The scatter reads the snapshot
    from ``block.store`` and writes ``conserved_nd`` directly (frozen pressure,
    bypasses the P/T cache).

    No boundary masking is applied here: ``grid.apply_bconds`` re-imposes the
    inlet/outlet/mixing/cusp targets between stages and at the next step top, so
    the coarse push cannot leave a BC-controlled node inconsistent -- exactly as
    for the fine RK term, which is likewise unmasked.

    ``sf_irs`` (0 disables it, the default) applies implicit residual
    smoothing (Jameson IRS) to the coarse block-restricted residual at every
    level, exactly like the fine-grid smoothing ``Grid.update_residual``
    already applies via its ``sf`` argument -- both are driven by the same
    ``SolverConfig.sf_resid`` value (see :func:`rk_step`). This selects the
    experimental :func:`ember.fortran.advance_rk_stage_mg_fused_irs` kernel
    instead of ``_opt``; ``sf_irs=0`` makes its smoothing step an exact no-op
    (see the kernel's docstring), so it is only selected when ``sf_irs > 0``,
    keeping the default (no coarse IRS) path byte-identical to before. The
    extra per-level scratch it needs (``coarse_res_buf``, ``tri_work_buf``,
    sized by :func:`_mg_irs_scratch_sizes`) is likewise carved from
    ``block.tau_q_halo`` -- caller-owned, no per-call allocation.

    Assumes ``block.dt_vol_nd`` and ``block.residual_nd`` are populated and the
    caller refreshes P/T, boundary conditions and the residual between stages.
    Requires :func:`_validate_mg` to have passed.
    """
    for block in grid:
        ni, nj, nk = block.shape
        nc1i, nc1j, nc1k = (ni - 1) // 2, (nj - 1) // 2, (nk - 1) // 2
        tmp = util.carve_view(block.scratch, (ni - 1, nj - 1, nk - 1, 5))
        if sf_irs > 0.0 and n_levels > 0:
            # The prolong scratch (aplane, bb), the coarse block-sum
            # accumulator (corr), and the coarse-IRS scratch (coarse_res_buf,
            # tri_work_buf) are all carved from tau_q_halo -- dead outside the
            # viscous pass and a distinct buffer from block.scratch, so they
            # survive alongside tmp within the call.
            n_res, n_tri = _mg_irs_scratch_sizes(ni, nj, nk, n_levels)
            aplane, bb, corr, coarse_res_buf, tri_work_buf = util.carve_view(
                block.tau_q_halo,
                (ni - 1, nc1j),
                (ni - 1, nj - 1, nc1k, 5),
                (nc1i, nc1j, nc1k, 5),
                (n_res,),
                (n_tri,),
            )
            ember.fortran.advance_rk_stage_mg_fused_irs(
                cons=block.conserved_nd,
                snapshot=block.store,
                residual=block.residual_nd,
                dt_vol=block.dt_vol_nd,
                alpha=alpha,
                cfl=cfl,
                fmgrid=fac_mgrid,
                sf_irs=sf_irs,
                n_levels=n_levels,
                tmp=tmp,
                corr=corr,
                aplane=aplane,
                bb=bb,
                coarse_res_buf=coarse_res_buf,
                tri_work_buf=tri_work_buf,
            )
        else:
            # The prolong scratch (aplane, bb) and the coarse block-sum
            # accumulator (corr) are all carved from tau_q_halo -- dead outside
            # the viscous pass (the caller rebuilt the residual, tau_q_halo's
            # other borrower, before this call) and a distinct buffer from
            # block.scratch, so they survive alongside tmp within the call.
            aplane, bb, corr = util.carve_view(
                block.tau_q_halo,
                (ni - 1, nc1j),
                (ni - 1, nj - 1, nc1k, 5),
                (nc1i, nc1j, nc1k, 5),
            )
            ember.fortran.advance_rk_stage_mg_fused_opt(
                cons=block.conserved_nd,
                snapshot=block.store,
                residual=block.residual_nd,
                dt_vol=block.dt_vol_nd,
                alpha=alpha,
                cfl=cfl,
                fmgrid=fac_mgrid,
                n_levels=n_levels,
                tmp=tmp,
                corr=corr,
                aplane=aplane,
                bb=bb,
            )


@util.profile
def rk_step(grid, conf):
    """Advance `grid` one Jameson RK pseudo-time step in place.

    The RK counterpart of :func:`scree_step`: a full ``conf.n_stage``-stage
    sweep. Snapshots ``U_n`` into ``block.store`` (the bconds-consistent state
    from the step top -- the caller's residual build did not touch
    ``conserved_nd``), then marches every stage off that frozen snapshot with
    :func:`advance_rk_stage_mg` (which folds in the Denton block-sum multigrid
    correction; with ``conf.n_levels == 0`` the coarse loop is empty and it is a
    plain RK stage). ``conf.sf_resid`` is passed through as the coarse-level
    IRS coefficient too, so a nonzero ``sf_resid`` smooths both the fine
    residual (``Grid.update_residual``, run by the caller before this) and the
    per-level coarse block-restricted residual inside the multigrid stage.

    Stage 0 reuses the step-top P/T flush and the residual the caller built
    before this call; later stages march off a changed ``conserved_nd``, so P/T,
    boundary conditions and the residual are refreshed first. Stage coefficients
    come from ``conf.alphas`` (``alpha_k = 1/(n_stage-k+1)``).

    Each stage rebuilds the residual for the *next* stage to consume, so the
    final stage skips it: nothing reads ``residual_nd`` before the next step's
    top-of-loop rebuild. This trims one full residual evaluation per step (five
    down to ``n_stage``).

    Smoothing is not applied here -- the caller runs :meth:`Grid.smooth` once on
    the post-step state, shared with the scree path.
    """
    # Snapshot U_n into block.store; each stage marches off this snapshot.
    for block in grid:
        block.store[...] = block.conserved_nd
    # Loop over substeps
    for i_stage in range(conf.n_stage):
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
def run(grid, conf):
    """Drive a grid through ``n_step`` explicit time-marching steps.

    Implements the lagged-pressure ordering so each step pays exactly one
    full-field pressure evaluation:

    0. flush the conserved cache so the residual sees a fresh full-field P;
    1. every ``n_refresh`` steps, rebuild the body force and relax ``dt_vol``
       (both read the just-refreshed P/a);
    1b. every ``conf.n_step_log`` steps, record and print a convergence message
       (:meth:`Grid.get_convergence` + :meth:`ConvergenceHistory.format_message`).
       Placed after the body-force refresh but before the march so its residual
       read serves as the step's single full-field recompute, which
       :func:`scree_step` then reuses; ``show_cfl=False`` since the fixed-CFL march
       has no per-cell CFL field to report;
    2. march every block with the selected integrator (:func:`scree_step` or the
       RK :func:`advance_rk_stage_mg` sweep) -- writes ``conserved_nd`` in place
       without bumping the cache, so P/T stay frozen for the rest of the step;
    3. smooth every block with the constant-coefficient kernel
       (:meth:`Grid.smooth`), which needs no P/T and so reuses the frozen state;
    4. refresh the boundary targets (:meth:`Grid.update_bconds` -- interior-P
       snapshot, outlet throttle/equilibrium) and impose them
       (:meth:`Grid.apply_bconds`); the patches read/write only boundary slices,
       so no full-field P recompute rides along.

    ``dt_vol`` is relaxed in place by :meth:`Grid.update_timestep` (the kernel blends
    ``rf*new + (1-rf)*old``); the first refresh uses ``rf=1.0`` to initialise the
    uninitialised buffer, ``rf_dt`` thereafter. CFL is the fixed module-level
    :data:`CFL`.

    Returns
    -------
    Grid
        ``grid``, for chaining.
    """

    # Fail fast if the grid cannot be evenly blocked for multigrid.
    _validate_mg(grid, conf.n_levels)

    # Initialise timesteps
    grid.update_timestep(rf=1.0, fac_visc=conf.fac_visc)

    hist = ConvergenceHistory.from_grid(conf.n_step, grid)

    for i_step in range(conf.n_step):
        #
        # We overwrote conserved_nd in place last step
        # So flush the cache to recalculate P and T
        grid.update_cached_conserved()

        grid.update_bconds()  # Throttle/radial equilibrium targets
        grid.apply_bconds()

        try:
            grid.check_nan()
        except SolverDivergedError as err:
            logger.error("Solver diverged at step %d: %s", i_step, err)
            break

        # Refresh source terms on the n_step_source cadence -- the viscous pass
        # is the expensive part -- but recompute the timestep every step so
        # dt_vol tracks the flow through fast transients. A lagged dt_vol is
        # sized for an already-stale state and overshoots the stability limit
        # during a cold start; the timestep refresh itself is cheap.
        if i_step % conf.n_step_source == 0:
            # grid.update_filter(conf.delta_filt)
            grid.update_sources(conf.inviscid, conf.gain_filt)
        grid.update_timestep(rf=0.2, fac_visc=conf.fac_visc)

        # Prepare the residual
        grid.update_residual(dampin=conf.dampin, sf=conf.sf_resid)

        # Convergence logging of the pre-march state
        if i_step % conf.n_step_log == 0:
            hist.record_step(i_step)
            hist.record_convergence(grid.get_convergence())
            logger.info(
                "%s",
                hist.format_message(
                    i_finest=0, n_step=conf.n_step, n_levels=1, show_cfl=False
                ),
            )

        # Take a step with the selected integrator.  Both reuse the first
        # residual evaluated above, RK then recalculates each substep
        if conf.n_stage == 0:
            scree_step(grid, conf.cfl)
        else:
            rk_step(grid, conf)

        # Smooth the post-step conserved solution
        grid.smooth(conf.sf4 * conf.cfl, conf.sf2 * conf.cfl)

        # Pseudotime avearge over last n_step_avg steps
        if i_step >= (conf.n_step - conf.n_step_avg):
            grid.accumulate_avg(conf.n_step_avg)

    # Copy the final average back in the primary storage
    grid.finalise_average()

    return hist
