Navier--Stokes algorithm
========================

ember integrates the compressible Navier--Stokes equations to a steady state
using an explicit pseudo-time march accelerated by in-place multigrid. The
discretisation is finite-volume in space on structured, multi-block,
curvilinear grids; cell-centred residuals are distributed back to nodal
conserved variables. This page describes the solver loop driven by
:func:`ember.solver.run` and the configuration parameters in
:class:`ember.solver.SolverConfig` that control each stage.

Overview of one time step
-------------------------

:func:`ember.solver.run` follows a *lagged-pressure* ordering, so each step
pays for exactly one full-field pressure/temperature evaluation:

1. **Cache flush**: invalidate the cached pressure/temperature so the residual
   sees a fresh full-field evaluation from the conserved state the previous
   step wrote in place.
2. **Boundary refresh**: update boundary-patch targets (throttle, radial
   equilibrium, mixing-plane exchange) and impose the boundary conditions.
3. **NaN check**: abort the run early (recording ``ConvergenceHistory.diverged``)
   if the conserved state has gone non-finite.
4. **Sources**: rebuild the body-force buffer -- every step when using the
   Runge--Kutta integrator (``SolverConfig.n_stage >= 1``), every fifth step
   when using the scree march (``n_stage == 0``, which only evaluates the
   residual once per step and so can tolerate a lagged viscous pass).
   ``dt_vol`` is relaxed every step regardless, since a stale timestep can
   overshoot the stability limit during a transient.
5. **Residual**: build the unintegrated net-flow residual, with optional
   implicit residual smoothing (``SolverConfig.sf_resid``) and a change
   limiter (``SolverConfig.dampin``).
6. **Convergence logging**: every ``SolverConfig.n_step_log`` steps, record and
   print a :class:`~ember.convergence_history.ConvergenceHistory` row from
   this residual.
7. **March**: advance the solution with the selected integrator -- Denton's
   scree march or Jameson multi-stage Runge--Kutta (see below) -- optionally
   accelerated by in-place multigrid.
8. **Smoothing**: apply a constant-coefficient artificial-dissipation blend to
   the post-march conserved state.
9. **Pseudotime averaging**: over the final ``SolverConfig.n_step_avg`` steps,
   accumulate the conserved state into a running average, which replaces the
   instantaneous state once the run completes.

There is no separate CFL-adaptation phase: CFL is a single constant scalar
for the whole run (see below), not a per-cell field recomputed each step.

.. _march-schemes:

Time integrators
-----------------

``SolverConfig.n_stage`` selects one of two integrators, applied to every
block each step:

**Scree march** (``n_stage == 0``, the default) -- :func:`ember.solver.scree_step`
implements Denton's basic second-order-accurate march (Denton 2017, Eq. 4)::

    F_{n+1} = F_n + [2*(dF/dt)_n - (dF/dt)_{n-1}] * dt

The extrapolated quantity ``q = 2*residual - store`` is formed from the
current residual and the previous step's residual (held in ``block.store``),
scaled by ``CFL * dt_vol``, and scattered straight onto the conserved state --
bypassing the normal setters so the pressure/temperature cache stays frozen
through the step (Denton-style "smooth with the old pressure").

**Jameson multi-stage Runge--Kutta** (``n_stage >= 1``) -- :func:`ember.solver.rk_step`
snapshots the step-start conserved state, then runs ``n_stage`` substages of
:func:`ember.solver.advance_rk_stage_mg` with stage coefficients
``alpha_k = 1 / (n_stage - k)``. Only the last substage skips rebuilding the
residual, since nothing consumes it again before the next step's top-of-loop
rebuild.

Both integrators share the multigrid acceleration described below and the
constant-coefficient smoothing step; they differ only in the march formula and
in how many times per step the residual is rebuilt. The
:doc:`run_duct example <auto_examples/run_duct>` compares concrete tunings for
the same case: ``scree, CFL=0.4``; plain ``RK4, CFL=4.0``; and
``RK4 + implicit residual smoothing (sf_resid=1.0), CFL=8.0`` -- all with two
multigrid levels (``n_levels=2, fac_mgrid=0.4``).

.. _cfl:

CFL number
----------

CFL is a single constant, ``SolverConfig.cfl`` (default 0.4), applied
uniformly to every cell for the entire run -- there is no per-cell adaptive
CFL field and no tolerance-driven backoff inside the solver. A larger CFL
converges faster but risks divergence; implicit residual smoothing
(``SolverConfig.sf_resid``) damps high-frequency residual content and so
tolerates a substantially higher CFL for a given scheme (see the example
above).

A suitable constant CFL for a given case, scheme, and ``sf_resid``/``fac_mgrid``
combination is not chosen by the solver -- it is found offline with
``scripts/duct_cfl_descend.sh``, a developer tuning harness that repeatedly
reruns a case at descending trial CFL values (halving the step on each
convergent trial) to bracket the largest CFL that still converges. This is a
sweep tool built on top of the solver, not part of the solver's runtime
algorithm.

.. _smoothing:

Smoothing
---------

A constant-coefficient blend of second- and fourth-difference operators
(:meth:`ember.grid.Grid.smooth`) is applied to the conserved variables after
each step to suppress odd--even decoupling and high-frequency content
introduced by the march and multigrid corrections.

- ``SolverConfig.sf4``, ``SolverConfig.sf2`` -- coefficients on the
  fourth- and second-difference terms, each scaled by the run's ``cfl`` at the
  call site (``grid.smooth(sf4 * cfl, sf2 * cfl)``).

Unlike the sensor-driven JST smoother this solver used previously, the
current smoother has **no shock sensor and no pressure/temperature
dependence** -- it never touches the pressure cache, so it is safe to run on
the frozen post-march state. (The old adaptive, curvature-sensor smoother
still exists in ``src/ember/_fortran/smooth_v2.f90`` but is no longer called
from the solver; it is exercised only by its unit tests.)

.. _multigrid:

Multigrid
---------

**In-step Denton block-sum multigrid.** Unlike a classical restrict/prolong
V-cycle across separate coarse grids, multigrid here is computed in place
within a single march call over one grid: coarse block-sum corrections are
folded directly into the fine-grid increment before it is scattered onto the
conserved state. It is controlled by ``SolverConfig.n_levels`` (number of
coarse levels; 0 disables multigrid) and ``SolverConfig.fac_mgrid`` (scaling
on the coarse correction; 0 also disables it). Cell counts on each block must
be an exact multiple of the coarsest block size ``2**n_levels`` in every
direction (checked by ``ember.solver._validate_mg`` before the run starts).

Both integrators route their per-block march through the same
scheme-agnostic Fortran engine, ``mg_coarse_correction`` in
``src/ember/_fortran/scree.f90``, operating on a single pre-formed fine
quantity ``q`` (``residual`` for RK, ``2*residual - store`` for scree). Around
it, six branch-free kernels dispatch on scheme and on whether multigrid /
implicit residual smoothing are active:

======================== ============================== ==============================
Multigrid                RK                             Scree
======================== ============================== ==============================
off (fine term only)     ``rk_plain``                   ``scree_plain``
on, no coarse smoothing  ``rk_mg_noirs``                ``scree_mg_noirs``
on, coarse IRS           ``rk_mg_irs``                  ``scree_mg_irs``
======================== ============================== ==============================

The choice between the ``_noirs``/``_irs`` pair is made in Python
(``sf_resid > 0`` selects IRS); the Fortran engine itself carries no
conditional on it. For coarse level ``l = 1..n_levels`` (block size
``b = 2**l``), the correction is scaled by
``coef_l = alpha * cfl * fac_mgrid / b**2 * 2**-(l-1)``, damping successively
coarser levels (``alpha = 1`` for scree, the RK substage weight for RK). The
coarse timestep is the volume-weighted mean of ``dt_vol`` over each coarse
block, not the value at the block's centre cell, which would be biased on a
stretched mesh.

Restriction is **hierarchical**: only the finest coarse level reads the fine
grid directly; each coarser level reduces the previous level's running
accumulators, since an eight-way block-sum is associative. Prolongation is
**cascaded**: per-level scaled corrections accumulate coarsest-to-finest
through factor-2 trilinear interpolation hops, so only the final hop writes
the fine grid (fused with the fine-term scatter). All multigrid scratch is
carved from buffers that are dead at that point in the step (chiefly
``block.tau_q_halo``), so no per-step allocation is needed.

``SolverConfig.sf_resid`` also drives implicit residual smoothing (Jameson
IRS) on the fine-grid residual itself, via ``Grid.update_residual``,
independent of whether multigrid is enabled.

**Full multigrid startup.** :func:`ember.solver.run_fmg` runs the same solver
coarse-to-fine as a startup schedule, rather than within a single step. It
builds ``SolverConfig.n_levels`` progressively-halved grids
(:meth:`ember.grid.Grid.resample`), solves the coarsest first, then
prolongs each converged solution onto the next finer grid as its initial
guess (:meth:`ember.grid.Grid.interp_from`) and calls :func:`ember.solver.run`
again with the in-step multigrid depth set to that level's index -- so the
coarsest level runs with no in-step multigrid and the finest runs at the full
requested ``n_levels``, identical to calling :func:`~ember.solver.run` directly
on the finest grid. With ``n_levels <= 0`` it reduces to a single call to
:func:`~ember.solver.run`.

.. _body-forces:

Body forces and viscous model
------------------------------

The cell-centred body-force buffer (``block.F_body_nd``) accumulates all
source terms before they are added to the residual, rebuilt by
:meth:`ember.grid.Grid.update_sources`:

- Viscous shear stresses and heat flux, computed unless
  ``SolverConfig.inviscid`` is set. The viscous pass is phased across the
  whole grid (tau/q on every block, then a periodic-seam halo exchange, then
  face-flux accumulation) so block-to-block periodic interfaces stay
  consistent.
- A polar (axisymmetric) source term to balance the cylindrical coordinate
  metric.
- An optional selective-frequency-damping (SFD) force when
  ``SolverConfig.gain_filt`` is nonzero.

The mixing-length turbulent viscosity uses a fixed turbulent Prandtl number
of 1.0 and is evaluated from the absolute-frame vorticity magnitude.
``SolverConfig.fac_visc`` multiplies the turbulent-diffusion timestep radius
independently of this, tightening the viscous stability limit to recover the
inviscid stable CFL where needed.

.. _boundary-coupling:

Boundary patches and inter-block coupling
------------------------------------------

Inlet, outlet, and mixing-plane patches each relax their own state towards a
target every step, with their own relaxation factor rather than a single
solver-wide setting:

- :class:`~ember.inlet.InletPatch` relaxes its interior pressure datum with
  ``rf = 0.2`` by default.
- :class:`~ember.mixing.MixingPatch` relaxes similarly, with ``rf = 1.0`` by
  default (no relaxation).
- :class:`~ember.mixing_communicator.MixingCommunicator` relaxes the
  mixing-plane target exchanged between adjacent blocks with a separate
  ``rf_mix`` (default 0.1).
- :class:`~ember.outlet.OutletPatch` takes its own relaxation factor via
  ``set_adjustment(rf=...)``.

:meth:`ember.grid.Grid.update_bconds` advances the slowly-varying boundary
targets once per step (mixing-plane exchange, inlet pressure snapshot,
outlet PID/spanwise target); :meth:`ember.grid.Grid.apply_bconds` then
imposes the full set of physical boundary conditions and closes periodic
seams every time it is called, including between Runge--Kutta substages.

.. _logging-and-averaging:

Logging, averaging, and convergence history
---------------------------------------------

Convergence diagnostics are recorded into a
:class:`~ember.convergence_history.ConvergenceHistory` every
``SolverConfig.n_step_log`` steps: mean residual, mass flow / stagnation
enthalpy / entropy at row interfaces, and outlet throttle state
(:meth:`ember.grid.Grid.get_convergence`).

Pseudotime averaging of the conserved variables accumulates over the final
``SolverConfig.n_step_avg`` steps of the run
(:meth:`ember.grid.Grid.accumulate_avg`). On completion, the time-averaged
state replaces the instantaneous state
(:meth:`ember.grid.Grid.finalise_average`) -- skipped if the run diverged, so
the invalid field is preserved for inspection rather than overwritten by a
partially-accumulated average.

The full set of configuration parameters, and the Python entry points
referenced throughout this page, are documented in :doc:`api/solver`.
