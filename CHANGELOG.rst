Changelog
=========

ember follows `semantic versioning <https://semver.org/>`_ from 0.1.0 onwards.
Until 1.0.0, minor releases may make breaking changes to the public API
without a deprecation period.

0.2.0 (unreleased)
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
* Generalised non-reflecting patches to any surface of revolution.
* Added ``OutletPatch.Kp`` and ``OutletPatch.Ki``, exposing the throttle gains
  set by ``set_throttle``.
* Added ``InletPatch.Po``, ``InletPatch.To``, ``InletPatch.Alpha``,
  ``InletPatch.Beta`` and ``OutletPatch.P``, reading back the currently
  prescribed boundary state in physical units.
* Added ``wall_yplus`` for y+ post-processing.
* Added ``Block.memory_usage``.
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
