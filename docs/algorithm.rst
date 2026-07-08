Navier--Stokes algorithm
========================

ember integrates the compressible Navier--Stokes equations in time using an
explicit Runge--Kutta scheme accelerated by geometric multigrid. The
discretisation is finite-volume in space on structured, multi-block,
curvilinear grids; cell-centred residuals are distributed back to nodal
conserved variables. This page describes the solver loop driven by
``ember.run.loop`` and the configuration parameters in
``SolverConfig`` that control each stage.

Overview of one time step
-------------------------

A single time step on the finest active grid consists of:

1. **Pre-step**: apply periodic and mixing-plane communications, average
   cusp nodes, recompute body forces (polar source and viscous stresses),
   and update boundary-patch targets.
2. **Runge--Kutta sweep**: four substages of a low-storage RK integrator,
   each evaluating fluxes, integrating the residual, and distributing the
   change to nodes.
3. **Post-step**: scale the error estimate, adapt the CFL field, refresh
   the volumetric timestep, and check for NaNs.
4. **Multigrid leg**: on the down leg, restrict residuals to coarser
   grids; on the up leg, prolongate corrections back to the finest level.
5. **Smoothing**: apply JST artificial dissipation to damp high-frequency
   content introduced by prolongation.

.. _runge-kutta:

Runge--Kutta integrator
-----------------------

The solver uses a four-stage, low-storage Runge--Kutta scheme with
hard-coded coefficients ``A_RK`` and ``B_RK``. At each substage:

- the residual is integrated into the cell-centred error buffer, weighted
  by the previous stage's ``A`` coefficient;
- the accumulated cell change is distributed to surrounding nodes with
  weight ``B``.

After the final substage the cell error is scaled by ``B_RK[-1]`` and
absolute-valued to give the per-step error estimate used by CFL
adaptation.

.. _cfl-adaptation:

CFL adaptation
--------------

ember runs with a *spatially varying* CFL field rather than a single
global number. After each step, the cell-wise error estimate is compared
against a relative tolerance ``SolverConfig.rtol``, and
the CFL number in each cell is adjusted to keep the local error near
tolerance.

Relevant configuration:

- ``SolverConfig.cfl_min``,
  ``SolverConfig.cfl_max`` --- bounds on the adapted
  CFL field.
- ``SolverConfig.cfl_bnd_max`` --- separate, tighter
  cap applied at boundary cells.
- ``SolverConfig.rtol`` --- target relative error per
  step driving the adaptation.
- ``SolverConfig.delta_filt``,
  ``SolverConfig.gain_filt`` --- low-pass filter
  applied to the conserved-variable reference state that the error is
  measured against.
- ``SolverConfig.fac_restart`` --- when restarting,
  scales the seed CFL field by this factor before clipping to
  ``cfl_min``; set to 0 to disable reuse of a prior CFL guess.

The volumetric timestep ``dt_vol`` is refreshed every 50 steps from the
current solution; CFL is then used to scale ``dt_vol`` into the actual
step taken in each cell.

.. _artificial-dissipation:

Artificial dissipation (smoothing)
----------------------------------

A JST-style blend of second- and fourth-difference operators is applied
to the conserved variables after each step to suppress odd--even
decoupling and shock oscillations.

- ``SolverConfig.sf2P``,
  ``SolverConfig.sf2T`` --- coefficients on the
  second-difference (shock-sensing) term. The nodal curvature sensor is
  evaluated separately on pressure and on temperature, and the active
  second-difference coefficient is the elementwise maximum of the two,
  ``max(sf2P * sensor_P, sf2T * sensor_T)``. The temperature term catches
  contact discontinuities (constant pressure, jump in temperature) that a
  pressure-only sensor misses (Swanson, Radespiel & Turkel, AIAA-97-1945).
- ``SolverConfig.sf4`` --- coefficient on the
  background fourth-difference term.
- ``SolverConfig.sf2_min`` --- floor on the second-
  difference coefficient, useful for stabilising strongly distorted
  meshes.
- ``SolverConfig.fac_mg_smooth`` --- per-level scaling
  applied as ``fac_mg_smooth ** i_level`` on coarse grids.

.. _multigrid-cycle:

Multigrid cycle
---------------

ember supports both a fixed multigrid cycle and a full-multigrid (FMG)
startup schedule.

**Grid hierarchy.** The solver builds
``SolverConfig.n_levels`` grids by successive
factor-of-two coarsening from the input fine grid. Cell counts on the
fine grid must satisfy ``(n_i - 1) % 2 == 0`` along every axis so that
restriction is exact, and coarse blocks must retain at least 5 nodes per
direction.

**Down leg.** Starting from the currently finest active level, the
solution is advanced, then the nodal solution is restricted to the next
coarser grid via pure subsampling (every other node). The coarse
``f_body`` buffer is loaded with the difference between fine and coarse
net flow so that the coarse march is driven by the fine-grid residual.

**Up leg.** Coarse corrections are prolongated back down the grid
hierarchy, scaled by
``SolverConfig.fac_mgrid``. Setting ``fac_mgrid = 0``
disables multigrid altogether (single-grid mode). No level is
re-advanced on the way up, so the cycle is a sawtooth rather than a
full V-cycle.

**Full multigrid.**
``SolverConfig.full_mgrid`` enables a startup schedule
in which only the coarsest level is active for the first
``SolverConfig.n_step`` steps; the next finer level is
then initialised by trilinear interpolation from the coarser solution
and the active range expands by one level. After ``n_levels`` phases
all levels are active and the run continues with the full grid
hierarchy.

``SolverConfig.i_level_stop`` truncates the FMG
schedule before reaching the finest level, which is useful when the
finest grid is too expensive to converge but a coarse-level solution is
sufficient for postprocessing or as a restart seed. At end of run, the
remaining finer levels are cascade-initialised from the stop level so
that a full-resolution state is still produced.

.. _body-forces:

Body forces and viscous model
-----------------------------

The cell-centred ``f_body`` buffer accumulates all source terms before
they are added to the convective residual:

- A polar (axisymmetric) source term to balance the cylindrical
  coordinate metric.
- Viscous shear stresses and heat flux, computed unless
  ``SolverConfig.inviscid`` is set.

Relevant configuration:

- ``SolverConfig.Pr_turb`` --- turbulent Prandtl
  number used to convert turbulent viscosity into heat flux.

The mixing-length turbulent viscosity is driven by the relative-frame
(rotating block frame) vorticity magnitude, with no absolute-frame
``+2*Omega`` correction.

.. _boundary-coupling:

Boundary patches and inter-block coupling
-----------------------------------------

Inlet, outlet, and mixing-plane patches all use first-order relaxation
to drive their state towards a target each step. The relaxation factors
are configurable:

- ``SolverConfig.rf_inlet_rho`` --- interior density
  relaxation factor used to stabilise the boundary condition at low Mach
  number. Applied to all nodes on
  :class:`~ember.inlet.InletPatch`, to nodes where flow enters the
  block on a :class:`~ember.mixing.MixingPatch`, and to backflow nodes
  with reversed flow re-entering the domain on
  :class:`~ember.outlet.OutletPatch`.
- ``SolverConfig.rf_mix`` --- separate relaxation
  factor applied inside
  :class:`~ember.mixing_communicator.MixingCommunicator` for the
  mixing-plane target exchange between adjacent blocks.

Periodic and mixing-plane communicators are built once per level during
solver setup and exchanged at the top of every step.

.. _logging-and-averaging:

Logging, averaging, and convergence history
-------------------------------------------

Convergence diagnostics are recorded into a
:class:`~ember.convergence_history.ConvergenceHistory` every
``SolverConfig.n_step_log`` steps: mean residual,
mean CFL, mass flow / stagnation enthalpy / entropy at row interfaces,
and outlet throttle state.

Time averaging of the conserved variables starts
``SolverConfig.n_step_avg`` steps before the end of
the run (see ``SolverConfig.i_step_avg``), so that
averaging always overlaps the final, fully-converged full-hierarchy phase
regardless of FMG activity. On completion, the time-averaged state
replaces the instantaneous state on the finest grid.

The full set of configuration parameters is documented separately in
:doc:`solver_configuration`.

Entry point
-----------

.. note::
   This page describes the legacy ``ember.run`` solver loop, removed in
   favour of the simpler constant-CFL march in ``ember.scree.loop``
   (see ``ember.scree.ScreeConfig`` for its configuration). Retained
   for background on the algorithm ideas (CFL adaptation, full multigrid)
   that the current solver does not (yet) reimplement.
