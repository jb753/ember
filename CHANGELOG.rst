Changelog
=========

ember follows `semantic versioning <https://semver.org/>`_ from 0.1.0 onwards.
Until 1.0.0, minor releases may make breaking changes to the public API
without a deprecation period.

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
* ``Solver`` is now frozen. Nothing in the package assigned to a solver field
  --- the working state of a march lives on the grid, and the configuration is
  only read --- so this states that rather than changing it. Beyond
  immutability it makes a ``Solver`` a value: two with the same settings
  compare equal, it is hashable, and a frozen dataclass elsewhere may inherit
  from it. Derive a variant with ``dataclasses.replace`` instead of assigning.
* Changed ``Solver`` defaults: ``cfl`` 0.4 to 5.0, ``dampin`` ``None`` to 25,
  ``sf_resid`` 0.0 to 1.0, and ``n_levels`` 0 to 3. Multigrid and a higher CFL
  are now on by default, so a march that set none of these will converge
  differently --- and faster. Note ``n_levels=3`` needs cell counts divisible
  by ``2**n_levels``; set ``n_levels=0`` for a grid that is not.
* ``Block.flat`` is now a property, and flattens in Fortran order. It reshaped
  the column-major backing array in C order, which numpy silently satisfies
  with a copy for any block with more than one spatial axis, so the promised
  view semantics were quietly lost --- writes through one object were not seen
  by the other. It now shares storage for any freshly allocated shape, and
  raises rather than copies where a non-contiguous slice has no contiguous
  flattening. Pair it with ``reshape(..., order="F")``.
* Split ``interp_from`` into ``Grid.interp_from_arrays`` and
  ``Grid.interp_from_grid``, and transfer primitives rather than conserved
  variables. ``rhoe`` is not a property of the flow alone --- it depends on the
  gas through ``cv`` and on where its energy datum sits --- so interpolating
  between blocks whose fluids differ changed the state it was meant to
  preserve, silently: a datum 600 K apart turned 400 K into 1000 K. Pressure,
  temperature and velocity cross unchanged. A field read back from a file is
  now a first-class input rather than something to wrap a block around, and a
  negative temperature is impossible rather than merely asserted against.
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
  ``m' = int dm/r`` that a blade-to-blade view is drawn on. It belongs to the
  curve rather than to any grid: one integration along the polyline sets the
  datum, so points from different blocks, or from different cuts of the same
  machine, come back on a common scale with nothing to reconcile.
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
  available, falling back to the pure-Python classes otherwise. Same API,
  faster scanner and emitter.
* Fixed YAML being read and written in the locale's encoding rather than
  UTF-8, so a file carrying a degree sign or a non-ASCII name round-tripped
  losslessly only by coincidence of environment.
* Fixed writing a ``Path`` raising ``RepresenterError`` on Windows. PyYAML
  dispatches a representer on the exact runtime type, and only ``Path`` and
  ``PosixPath`` were registered --- ``Path("...")`` builds a ``WindowsPath``
  there, which had no representer at all. Registered for ``PurePath`` along
  the mro instead, covering every concrete path class.
* Fixed the sign of the cusp seam correction.
* Restored the smooth-then-damp order in ``update_residual``.
* Fixed the mixing-length viscosity returning ``-inf`` on flows with no
  resolved shear when built with gfortran 13, which diverged the march within
  one step.
* Added API documentation pages for ``cases``, ``communicators``,
  ``block_util``, ``patch``, ``util``, and ``yaml_util``.

0.1.0 (2026-08-06)
------------------

* First public release on the Python Package Index.
