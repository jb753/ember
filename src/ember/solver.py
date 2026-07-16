"""Explicit time-marching solver with a lagged-pressure loop.

:func:`run` drives a whole grid to a steady state, selecting one of two
integrators on each step according to :attr:`SolverConfig.n_stage`:

* ``n_stage == 0`` -- Denton's basic "scree" march (:func:`scree_step`),

    F_{n+1} = F_n + [2*(dF/dt)_n - (dF/dt)_{n-1}] * dt    (Denton 2017, Eq 4)

  which builds the unscaled residual exactly as multall (F1=2, F2=-1, F3=0),
  scales it by dt_vol*CFL, and distributes it to the nodes by accumulating
  straight onto ``conserved_nd`` -- bypassing the setters so the P/T cache
  versions are left untouched (frozen pressure); optionally accelerated by
  Denton block-sum multigrid at ``n_levels >= 1``, restricting the same
  lagged ``[2*(dF/dt)_n - (dF/dt)_{n-1}]`` quantity at every coarse level
  (matching multall's ``TSTEP``, which sums its own lagged ``STORE`` into
  the multigrid block accumulators);

* ``n_stage >= 1`` -- a Jameson multi-stage Runge-Kutta step (:func:`rk_step`),
  optionally accelerated by Denton block-sum multigrid
  (:func:`advance_rk_stage_mg`).

Both integrators share the same lagged-pressure loop: the conserved cache is
flushed once per step (one full-field pressure evaluation), the body force and
dt_vol are refreshed only every few steps on that fresh P, the march and the
constant-coefficient smoother (:meth:`~ember.grid.Grid.smooth`) then run on the frozen
state, and the boundary conditions touch only boundary slices. This keeps
exactly one full-field P recompute per step; everything else reuses the cache.
"""

import logging
from dataclasses import dataclass, replace

import ember
from ember import util
from ember.convergence_history import ConvergenceHistory
from ember.grid import DivergenceError

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """Configuration for the explicit time-marching solver."""

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


def scree_step(grid, cfl, fac_mgrid=0.0, n_levels=0, sf_irs=0.0):
    """Advance `block` one scree step in place.

    Assumes ``block.dt_vol_nd`` is populated and the block's cached
    P/T are consistent with ``conserved_nd`` on entry. The caller is
    responsible for invalidating caches and applying boundary conditions
    between steps.

    With ``n_levels == 0`` (the default) the whole step is the branch-free
    ``scree_plain`` kernel: it forms the extrapolated fine quantity
    ``q = 2*residual - store`` in place in ``store`` (``scree_form_q``), builds
    the scaled increment ``CFL*dt_vol*q`` (``fine_term``), rolls the residual
    history (``store <- residual``) and scatters the cell-centred increment
    straight onto ``conserved_nd`` -- bypassing the setters so the P/T cache
    versions stay frozen. This path is untouched by ``fac_mgrid``/``n_levels``.

    With ``n_levels >= 1``, ``scree_mg_irs``/``scree_mg_noirs`` additionally
    injects a
    Denton block-sum multigrid correction at ``n_levels`` coarse levels, like
    :func:`advance_rk_stage_mg`'s coarse path but for the scree fine term.
    Every level -- fine and coarse -- restricts/uses the same Denton-lagged
    quantity ``(2*residual - store)``, not plain ``residual``: this was
    verified against multall's ``TSTEP`` reference, which computes the
    lagged ``STORE = F1*DELTA + F2*DIFF`` first and sums exactly that into
    its multigrid block accumulators (not the raw flux imbalance). For
    ``l = 1..n_levels`` the coarse block has ``b = 2**l`` and
    ``coef_l = cfl*fac_mgrid/b**2 * 2**-(l-1)`` -- :func:`advance_rk_stage_mg`'s
    formula with ``alpha=1``, since scree takes one full-weight step per call
    rather than a partial RK sub-stage. The coarse timestep multiplying
    ``coef_l`` is the volume-weighted mean of ``dt_vol`` over the block, not
    ``dt_vol`` at the block's centre cell (see :func:`advance_rk_stage_mg`).

    ``block.store`` is the persistent previous-step residual buffer
    (``(dF/dt)_{n-1}``); the kernel reads it then overwrites it with the
    current residual for the next step. The transient increment buffer is
    borrowed from ``block.scratch`` as a cell-shaped, zero-copy view of the
    leading elements -- nothing outside the kernel reads it, so only its
    element count matters, not its indexing. When ``n_levels >= 1``, the
    coarse block-sum accumulator (``corr``), the coarse timestep (``dtblk``)
    and separable-prolong scratch (``aplane``, ``bb``) are carved from
    ``block.tau_q_halo`` -- dead outside the viscous pass at this point in the
    step, exactly as for :func:`advance_rk_stage_mg` -- so no per-step
    allocation is needed for either integrator.

    ``sf_irs`` (0 disables it, the default) applies implicit residual smoothing
    (Jameson IRS) to the coarse block-restricted residual at every level, the
    scree counterpart of :func:`advance_rk_stage_mg`'s ``sf_irs`` and driven by
    the same ``SolverConfig.sf_resid`` value (see :func:`rk_step`). ``sf_irs > 0``
    dispatches the ``scree_mg_irs`` kernel; ``sf_irs == 0`` (the default)
    dispatches ``scree_mg_noirs``, which enters no smoothing code at all. The two
    share ``mg_coarse_correction`` and differ only in the coarse-residual smoother
    handed to it (``smooth_residual_tri`` vs the ``mg_smooth_noop`` do-nothing),
    so the branch lives here on the Python side -- there is no ``sf_irs`` test
    inside the Fortran engine. Its scratch (sized by
    :func:`_mg_coarse_scratch_sizes`, carved by :func:`_mg_coarse_carve`) is taken
    from ``block.tau_q_halo``, so no per-step allocation is needed. The fine term
    is not smoothed here: it already carries the fine residual smoothed by the
    caller's :meth:`~ember.grid.Grid.update_residual` ``sf``, exactly as for the
    RK path.

    Smoothing is not applied here -- the caller runs :meth:`~ember.grid.Grid.smooth` once on
    the post-step state, shared with the RK path.

    Multigrid is dispatched only when it has both levels *and* nonzero strength:
    ``fac_mgrid == 0`` scales every coarse correction to identically zero, so
    routing to the plain ``scree_plain`` kernel avoids paying the full
    restrict/prolong per level for a guaranteed-zero contribution (and makes
    ``sf_irs`` inert too). This matches ``n_levels == 0`` to floating-point
    rounding -- the coarse path contributes nothing either way.
    """

    # fac_mgrid == 0 makes the coarse loop a no-op; collapse to no-MG dispatch.
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
            # Multigrid-on scree wrappers over the scheme-agnostic engine
            # (mg_coarse_correction). sf_irs > 0 selects the coarse-IRS kernel;
            # sf_irs == 0 selects the plain _noirs kernel, which enters no
            # smoothing code at all (no Fortran-side IRS branch -- the two share
            # the engine and differ only in the smoother passed). Scratch is
            # carved from tau_q_halo, dead outside the viscous pass (already
            # completed and consumed before this call), same reuse as the RK path.
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
            # Multigrid off: fine term only, no coarse scratch. Forms q in store,
            # rolls the history and frozen-scatters onto cons.
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
    ``SolverConfig.sf_resid`` value (see :func:`rk_step`). ``sf_irs > 0``
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
        tmp = util.carve_view(block.scratch, (ni - 1, nj - 1, nk - 1, 5))
        if n_levels_eff > 0:
            # Multigrid-on RK wrappers over the scheme-agnostic engine
            # (mg_coarse_correction). sf_irs > 0 selects the coarse-IRS kernel;
            # otherwise the plain _noirs kernel, which enters no smoothing code
            # (the two share the engine and differ only in the smoother passed --
            # no Fortran-side IRS branch). Coarse scratch is carved from
            # tau_q_halo, dead outside the viscous pass.
            kernel = (
                ember.fortran.rk_mg_irs
                if sf_irs > 0.0
                else ember.fortran.rk_mg_noirs
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
                tmp=tmp,
                **_mg_coarse_carve(block, ni, nj, nk, n_levels_eff),
            )
        else:
            # Multigrid off: plain Jameson RK fine-term stage, no coarse scratch.
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

    Smoothing is not applied here -- the caller runs :meth:`~ember.grid.Grid.smooth` once on
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

    1. flush the conserved cache so the residual sees a fresh full-field P;
    2. every ``n_refresh`` steps, rebuild the body force and relax ``dt_vol``
       (both read the just-refreshed P/a);
    3. every ``conf.n_step_log`` steps, record and print a convergence message
       (:meth:`~ember.grid.Grid.get_convergence` +
       :meth:`~ember.convergence_history.ConvergenceHistory.format_message`).
       Placed after the body-force refresh but before the march so its residual
       read serves as the step's single full-field recompute, which
       :func:`scree_step` then reuses;
    4. march every block with the selected integrator (:func:`scree_step` or the
       RK :func:`advance_rk_stage_mg` sweep) -- writes ``conserved_nd`` in place
       without bumping the cache, so P/T stay frozen for the rest of the step;
    5. smooth every block with the constant-coefficient kernel
       (:meth:`~ember.grid.Grid.smooth`), which needs no P/T and so reuses the
       frozen state;
    6. refresh the boundary targets (:meth:`~ember.grid.Grid.update_bconds` --
       interior-P snapshot, outlet throttle/equilibrium) and impose them
       (:meth:`~ember.grid.Grid.apply_bconds`); the patches read/write only
       boundary slices, so no full-field P recompute rides along.

    ``dt_vol`` is relaxed in place by :meth:`~ember.grid.Grid.update_timestep` (the kernel blends
    ``rf*new + (1-rf)*old``); the first refresh uses ``rf=1.0`` to initialise the
    uninitialised buffer, ``rf_dt`` thereafter. CFL is the fixed module-level
    :data:`CFL`.

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

    hist = ConvergenceHistory.from_grid(conf.n_step, conf.n_step_log, grid)

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


def run_fmg(grid, conf):
    """Full-multigrid startup: solve coarse-to-fine, prolonging each guess.

    ``conf.n_levels`` is the single grid-hierarchy depth. With ``n_levels == 0``
    this is exactly :func:`run(grid, conf) <run>`. Otherwise it builds
    ``n_levels`` grids successively halved by :meth:`~ember.grid.Grid.resample`
    at factor ``0.5``, solves the coarsest, and
    :meth:`~ember.grid.Grid.interp_from`'s the solution up onto each finer grid
    as its initial guess. Sequencing level ``i`` (``0`` = coarsest) is marched
    with in-step Denton block-sum multigrid depth ``i`` -- the same grid
    hierarchy ``n_levels`` already names -- so the coarsest runs plain and the
    finest runs at full ``n_levels``, identical to :func:`run` on the finest.

    The single validation ``_validate_mg(grid, n_levels)`` is sufficient for the
    whole chain: level ``i`` holds ``N / 2**(n_levels - i)`` cells per dimension
    and its depth-``i`` march needs those divisible by ``2**i``, which reduces to
    ``N`` (the finest cell count) divisible by ``2**n_levels`` at every level.
    That divisibility also keeps every ``resample(0.5)`` node-coincident with the
    finer grid (an exact subset of its nodes), so the coarse coordinates are the
    true geometry, not interpolated.

    Every level runs the same ``conf`` apart from ``n_levels`` (fixed
    ``n_step`` per level). ``grid`` is the finest level and is mutated in place
    to carry the final solution, matching :func:`run`.

    Parameters
    ----------
    grid : Grid
        Finest grid, already carrying its initial guess. Mutated in place.
    conf : SolverConfig
        Solver configuration; ``conf.n_levels`` sets the hierarchy depth.

    Returns
    -------
    list of ConvergenceHistory
        Per-level histories, coarsest first, finest last.
    """
    _validate_mg(grid, conf.n_levels)
    if conf.n_levels <= 0:
        return [run(grid, conf)]

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
        histories.append(run(level_grid, replace(conf, n_levels=i)))
    return histories
