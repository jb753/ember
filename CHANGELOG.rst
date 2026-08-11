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
* Pruned dead code from ``ember.util`` and made its single-caller helpers
  private.
* Replaced the reflecting mixing plane with the non-reflecting one.
* Generalised non-reflecting patches to any surface of revolution.
* Added ``OutletPatch.Kp`` and ``OutletPatch.Ki``, exposing the throttle gains
  set by ``set_throttle``.
* Added ``wall_yplus`` for y+ post-processing.
* Added ``Block.memory_usage``.
* Added precompiled wheels for macOS (arm64) and Windows (AMD64), alongside
  the existing Linux (x86_64) wheels.
* Fixed ``matvec`` broadcast dispatch order for ``(ni,1,1)`` matrices.
* Fixed the sign of the cusp seam correction.
* Restored the smooth-then-damp order in ``update_residual``.
* Added API documentation pages for ``cases``, ``communicators``,
  ``block_util``, ``patch``, ``util``, and ``yaml_util``.

0.1.0 (2026-08-06)
------------------

* First public release on the Python Package Index.
