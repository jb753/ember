Version history
===============

ember roughly follows `semantic versioning <https://semver.org/>`_ starting at
0.1.0. However, until 1.0.0, minor releases may make breaking changes to the
public API without a deprecation period.

.. _v0.3.0:

0.3.0 (unreleased)
------------------

* Add ``Solver.fac_mgrid_bnd``, the multigrid correction scaling used, in
  place of ``fac_mgrid``, on the coarse cells that touch a block boundary, at
  every level. The default None uses ``fac_mgrid`` there too and reproduces the
  uniform correction exactly, so existing runs are unchanged; a smaller value
  weakens the coarse push where the block sums straddle the face and the
  boundary conditions have to absorb it. Honored by both integrators.
* Rename the throttle pressure correction from ``P_throttle`` to
  ``dP_throttle``, in ``OutletPatch.get_throttle_stats``, ``ConvergenceStep``
  and the ``ConvergenceHistory`` column. Breaking, the last of these being a
  public attribute; a ``.cnv`` written before the rename is relabelled as it
  unpickles, so existing files still read.
* Add ``OutletPatch.P_throttle``, the static pressure level a throttled outlet
  has arrived at [Pa], being the prescribed pressure plus the correction. This
  is what to record to reproduce an operating point later, and what to
  re-prescribe to hold one: ``set_P(patch.P_throttle)`` before
  ``set_throttle(None)``, since clearing a throttle reverts to the pressure
  that was prescribed rather than freezing the one it reached.
* Add ``Solver.mix_reflective``, running every mixing plane as a reflective
  one: at each span station both faces are set to the average of the two sides'
  circumferential means, in place of the characteristic exchange. Imposed on
  every plane of every level like ``rf_inlet``, so a run's treatment follows
  from its configuration; None leaves each plane alone. It reflects every
  pitchwise harmonic and conserves mass but not the momentum or energy flux,
  so it is a robustness tool rather than a substitute for the default plane.
* Delete ``Block.xlen_sq_nd`` and ``Block.conserved_cell_nd``, calculate them
  inline in the Fortran kernels instead.
* Add ``RealFluid`` real-gas equation of state and fitting module, following
  Wheeler (2024) polynomial entropy fit.
* Make ``Block.cp_nd`` a nodal array rather than a scalar, pass fields
  of viscous and thermal transport properties to the source term kernels.
* Add ``to_dict`` and ``from_dict`` to fluid classes.
* Add ``Fluid.get_kappa`` thermal conductivity to the fluid API.
* Add ``change_visc()`` to fluid clasess, returning a new instance with
  viscosity scaled up or down. Perfect gases simply alter their scalar viscosity;
  real gases carry a ``scale_visc`` factor that multiplies the true fitted viscosity.
* Exchange viscous/thermal halos via six face arrays instead of a full volume
  array.
* Run the change limiter (``Solver.dampin``) before implicit residual
  smoothing (``Solver.sf_resid``) rather than after, fused into the kernel
  already traversing the volume. IRS is linear and the limiter is nonlinear in
  a global block mean, so the two orderings do not commute: at the default
  ``sf_resid=1.0, dampin=25`` the residual differs by ~19% of the field scale,
  growing with ``sf_resid``. The converged answer is unchanged, only the path
  to it, and the fusion is worth -6% serial and -11% at 100-rank saturation.
* ``Block.scratch`` is now the sole scratch array, and all transient
  workspace for the Fortan kernels is carved out of it.
* Various performance improvements to reduce memory footprint.

.. _v0.2.0:

0.2.0 (2026-08-18)
------------------

* Renamed ``ember.set_iter`` to ``ember.set_iterative``.
* Removed ``BlockRestart`` and its helpers.
* Removed ``ember.geometry``; its functions are now private helpers of their
  callers.
* Made ``ember.struct`` and the Plot3D I/O module private.
* Removed ``Grid.apply_rotation``. It assumed a mesh topology --- that ``j=-1``
  is the casing --- of any grid handed to it, which is not ember's to assume,
  and its ``tip_gap`` and ``shroud`` row types were in any case
  indistinguishable: every wall already takes its block's ``Omega``, so
  ``tip_gap`` overrode five faces with the value they already had and left the
  casing turning with the blade. Set ``Block.Omega`` for the frame, and add a
  ``RotatingPatch`` only for a wall that is *not* at that speed.
* Pruned dead code from ``ember.util`` and made its single-caller helpers
  private.
* Replaced the reflecting mixing plane with the non-reflecting one.
* ``Solver`` is now a frozen immutable class, derive a variant with
  ``dataclasses.replace`` instead of assigning.
* Changed ``Solver`` defaults: ``cfl`` 0.4 to 5.0, ``dampin`` ``None`` to 25,
  ``sf_resid`` 0.0 to 1.0, and ``n_levels`` 0 to 3.
* ``Block.flat`` is now a property, and flattens in Fortran order.
* Split ``interp_from`` into ``Grid.interp_from_arrays`` and
  ``Grid.interp_from_grid``, and transfer primitives rather than conserved
  variables to prevent datum offset errors.
* Generalised non-reflecting patches to any surface of revolution.
* Added ``OutletPatch.Kp`` and ``OutletPatch.Ki``, exposing the throttle gains
  set by ``set_throttle``.
* Added ``InletPatch.Po``, ``InletPatch.To``, ``InletPatch.Alpha``,
  ``InletPatch.Beta`` and ``OutletPatch.P``, reading back the currently
  prescribed boundary state in physical units.
* Added ``wall_yplus`` for y+ post-processing.
* Added ``Block.memory_usage``.
* Added ``Block.freeze()``, returning an immutable copy.
* Added ``block_util.repeat_pitchwise``, viewing a periodic passage as a
  cascade. A periodic solver computes one passage, which is rarely all anyone
  wants to look at: the passage-to-passage picture is what shows a wake meeting
  the next leading edge, or a shock crossing a pitch. Returns one copy per
  passage rather than a concatenated block, and drops patches, a rotated copy
  being a view of the flow rather than a member of a connected grid.
* Added ``util.unwrap_meridional``, giving the conformal coordinate
  ``m' = int dm/r`` that a blade-to-blade view is drawn on.
* Fixed ``mix_out`` returning slightly different states for a cut and its
  k-reversed twin. The Newton loop now iterates to its fixed point instead of
  stopping at the first iterate inside ``atol``, cutting that difference from
  ~3e-4 to ~2e-6.
* Added precompiled wheels for macOS (arm64) and Windows (AMD64), alongside
  the existing Linux (x86_64) wheels.
* Fixed ``mass_average`` raising on a partial reduction. The averaging already
  honoured the ``axes`` argument, but the zero-flux guard beneath it assumed a
  scalar, so averaging over the pitch alone and keeping the span raised about
  an ambiguous truth value instead of returning a profile.
* Fixed ``matvec`` broadcast dispatch order for ``(ni,1,1)`` matrices.
* ``yaml_util`` now uses libyaml's ``CSafeLoader``/``CSafeDumper`` where
  available for speed.
* Fixed YAML being read and written in the locale's encoding rather than
  UTF-8, so a file carrying a degree sign or a non-ASCII name round-tripped
  losslessly only by coincidence of environment.
* Fixed YAML writing a ``Path`` raising ``RepresenterError`` on Windows.
* Fixed the sign of the cusp seam correction.
* Restored the smooth-then-damp order in ``update_residual``.
* Fixed the mixing-length viscosity returning ``-inf`` on flows with no
  resolved shear when built with gfortran 13.

.. _v0.1.0:

0.1.0 (2026-08-06)
------------------

* First public release on the Python Package Index.
