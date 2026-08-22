Version history
===============

ember follows `semantic versioning <https://semver.org/>`_ from 0.1.0 onwards.
Until 1.0.0, minor releases may make breaking changes to the public API
without a deprecation period.

.. _v0.3.0:

0.3.0 (unreleased)
------------------

* Added ``RealFluid``, a real-gas equation of state, and ``ember.realgas_fit``,
  the offline tool that produces the coefficients it evaluates. Both follow
  Wheeler (2024): a polynomial surface is fitted to the compressibility factor
  over a box in density and internal energy, and integrated analytically to give
  entropy, with temperature and pressure derived from that one surface rather
  than fitted alongside it. That is what separates it from a lookup table, which
  reproduces its own tabulated values but not the relations between them, and so
  creates spurious entropy wherever gradients are steep. The fitting module
  needs CoolProp only to sample a table --- ``sample_coolprop``, which imports it
  lazily --- so a fluid built from coefficients someone else fitted carries no
  such dependency. ``RealFluid`` implements the whole ``_Fluid`` interface, so it
  drops in wherever ``PerfectFluid`` went; the surface evaluation and the Newton
  inversions behind ``set_P_T`` and friends are in Fortran.

* ``RealFluid``'s Newton solves now work in Fortran order and allocate nothing
  per iteration. The scalar solve forced its inputs C contiguous to reach the
  surface kernel, so every call from the solver --- whose block fields are
  Fortran contiguous --- copied both arrays on the way in, and both solves
  rebuilt their whole update as fresh temporaries at every step. They now write
  each step through buffers allocated once, which is also what settles the
  memory order: the kernel pairs density against energy element by element, and
  numpy takes an expression's layout from all of its inputs, so a C-ordered
  target alone was enough to hand back an iterate that walks against the
  density the wrong way. Measured at 512x512, about 15% off a scalar solve and
  5-10% off a two-dimensional one, with the peak memory of the latter down by
  about a tenth. Two consequences for callers: a single-precision scalar solve
  hands back a Fortran-ordered energy whatever it was given, and the derivative
  getters return zero-dimensional arrays rather than numpy scalars where they
  are handed scalars.

* ``Block.cp_nd`` is now a nodal array rather than a single number --- which is
  what it had always documented itself as. The distinction is invisible for a
  perfect gas, whose specific heat is constant, and the viscous kernel behind
  it took cp as a scalar, so a real gas's conductivity was built from a
  specific heat frozen at the datum. That made it a function of where the datum
  sat rather than of the flow: shifting the datum with ``change_datum``, which
  re-labels ``u`` and ``s`` and is meant to change nothing physical, moved it
  by as much as cp varies over the whole fit box. The kernel now takes the
  field and averages it to the cell, for about 7% of that kernel's time.

* Added ``to_dict`` and ``from_dict`` on the fluid classes, giving a fitted
  equation of state a portable form. A ``RealFluid`` surface costs a property
  table and an offline fit to produce, and until now the only thing that
  persisted one was a pickle of the grid it happened to be attached to ---
  which also carried every derived surface alongside the twelve numbers that
  actually define the fluid. The dict holds plain floats and nested lists, so
  ``json`` or a plain YAML dumper can write it, and it names its own class in a
  ``type`` key, so ``_Fluid.from_dict`` reads one back without the caller
  knowing which equation of state wrote it.

* ``RealFluid`` now defaults its datum to its own state at the centre of the
  fit box, rather than to 1 bar and 300 K. A fitted surface exists only inside
  its box and the datum has to lie in there, so a fixed pair of numbers can
  only ever belong to some other fluid's box: ambient falls inside an air-like
  fit and hundreds of bar outside a dense one, which is the case a real gas is
  for. The datum is read off the fitted surface, so it sits exactly on the
  surface it is the origin of. Passing ``P_dtm`` or ``T_dtm`` still overrides,
  either one on its own.

* The viscous kernels now take viscosity and conductivity as nodal fields, so
  a real gas's fitted transport surfaces reach the flow instead of stopping at
  the fluid. ``Block.mu_nd`` becomes a field like ``Block.cp_nd`` --- which it
  had always documented itself as --- and is joined by ``Block.kappa_nd`` and a
  dimensional ``Block.kappa``. The kernel takes that conductivity directly and
  no longer takes a laminar Prandtl number: deriving conductivity back from a
  ratio the fluid does not store is a round trip, and the ratio is only defined
  by the two surfaces in the first place. The mixing-length clamp
  ``3000*mu`` and the wall functions' ``Re = rho*V*d/mu`` follow the local
  value too, the latter face-averaged at the same node plane as the density it
  divides. Measured at about 13% of ``set_tau_q_soa`` for the two fields
  together, in line with the 7% one field cost when ``cp`` went nodal;
  ``set_visc_force`` is unchanged within noise, since it reads viscosity only
  at wall faces. Perfect-gas results move by float32 round-off rather than not
  at all --- averaging eight copies of one float32 to a cell is not exact --
  and a perfect gas now fills a whole nodal array with one repeated constant,
  as it already did for ``cp``. Two consequences for callers: ``Block.mu_nd``
  needs an initialised flow field, where before it was a property of the fluid
  and the reference length alone, and ``RealFluid`` no longer carries the
  block-wide ``_mu_nd`` and ``_Pr`` it used to hand the solver.

* ``RealFluid`` now fits viscosity and thermal conductivity as surfaces of
  density and internal energy, after Appendix B of the same paper, rather than
  taking two constants. They are ordinary least-squares fits over the same box
  and in the same coordinates as the compressibility factor, and they take no
  part in the thermodynamic consistency argument: nothing relates transport to
  entropy, and neither surface is ever differentiated, so they cost one
  polynomial evaluation and no derivatives. Each is normalised by its own value
  at the centre of the fit box, so the coefficients are of order unity and the
  physical scale is a single readable number. **Breaking**: ``RealFluid`` takes
  ``delta``, ``gamma``, ``mu_c`` and ``kappa_c`` in place of ``mu`` and ``Pr``,
  and ``realgas_fit.fit`` requires viscosity and conductivity samples --
  ``sample_coolprop`` returns them, so a pipeline built as
  ``fit(**sample_coolprop(...))`` needs no change. ``get_kappa`` joins the fluid
  interface, with ``PerfectFluid`` deriving it from the ``Pr`` it still takes
  and ``RealFluid`` deriving ``Pr`` from its two surfaces instead. The solver
  does not see any of this yet: its viscous kernel takes one viscosity and one
  Prandtl number per block, and what it now gets are the surfaces evaluated at
  the centre of the fit box.

* Added ``change_visc`` on the fluid classes, a factory in the manner of
  ``change_datum`` returning the same fluid with its viscosity scaled by a
  factor. Sweeping Reynolds number needs nothing else to move --- the geometry,
  the boundary conditions and the thermodynamics all stay put --- and the
  Prandtl number is untouched, so the conductivity scales with the viscosity
  and the Peclet number follows the Reynolds number. ``PerfectFluid`` changes
  its stored ``mu``; ``RealFluid`` carries the factor as a new ``scale_visc``
  constructor argument multiplying the viscosity, so that it will still mean
  the same thing once the transport properties are fitted surfaces of density
  and internal energy rather than the constants they are today.

* Fixed ``RealFluid``'s inversions returning an array of one where they were
  given a single state. ``asfortranarray`` forces ``ndmin=1``, so laying the
  iterate out for the kernel promoted a 0-d solve to shape ``(1,)`` -- and the
  result then refused to be printed, ``ndarray.__format__`` rejecting a format
  spec at any rank. The convergence history reports the inlet and outlet state
  on every log line, so a real-gas run raised a TypeError there rather than
  logging. The solves now keep 0-d at 0-d and hand back a numpy scalar, which
  is what ``PerfectFluid`` always did.

* Added a ``periodic_k`` argument to ``ember.cases.build_duct_grid``, which
  makes the duct's pitchwise faces periodic to each other instead of walls:
  ``"full"`` for the whole face, ``"hmesh"`` for two streamwise intervals with
  a wall between, the topology ``Block.i_perk`` describes. The patches have to
  be appended during construction rather than afterwards, because
  ``ijk_wall_visc`` and ``i_perk`` are cached on first access and the wall
  distance is computed from the patches -- appending later leaves the seam
  reading as a wall, silently. Written for the viscous seam study, where a
  block periodic to itself in k is the case under test.

* Added ``Block.tau_q_faces``, six two-layer surface buffers holding the
  boundary tau/q that the viscous face-flux phase reads, with
  ``set_tau_q_faces`` to produce them in an O(surface) pass and
  ``Grid.connectivity.periodic.exchange_faces`` to exchange them across
  periodic patches. Together these give a fused viscous kernel a halo source
  that is not a full-volume array, which is what lets one serve arbitrary
  block connectivity rather than only a block periodic to itself in k.
  Production's ``set_tau_q_soa``/``set_visc_force`` path is untouched and
  still uses ``tau_q_halo``.

* ``Block.scratch`` is now the single scratch arena and is flat rather than
  ``(ni, nj, nk, 5)``. ``Block.tau_q_halo`` and ``Block.tau_q_faces`` are
  views into it rather than allocations of their own, and ``tau_q_halo`` drops
  its spare tenth slot. Three allocations become one, sized by whichever step
  phase needs the most: 67.4 MB to 43.6 MB at a 1M-cell block, with peak RSS
  over a short run falling 351 MB to 341 MB. Timing is unchanged
  (``set_residual`` 24.479 to 24.476 ns/cell, the viscous pair 56.4 to 56.6,
  both inside their error bars).

  Consumers that carve from the arena must take every buffer a single kernel
  call needs from ONE ``util.carve_view``, which packs them end to end and so
  cannot overlap; buffers in different phases may share a span because no two
  phases are live at once. ``ember.solver`` gains ``mg_coarse_shapes`` so its
  callers can fold the multigrid scratch into their own carve, replacing
  ``_mg_coarse_carve``. Multigrid ``n_levels`` is now capped at
  ``ember.block.MAX_MG_LEVELS`` (3), validated in ``_validate_mg`` alongside
  the divisibility rule, because the arena is sized for that depth.

* ``set_visc_force`` is faster, by two independent changes to how it does the
  same arithmetic. The wall-function face helpers were not being inlined in the
  production build and could not vectorize regardless, because the
  Reynolds-number branch of the skin-friction fit sits in the middle of their
  arithmetic; they are now row forms split into three phases over a fixed i-tile,
  so only the branch stays scalar and the phases carrying the divides and square
  roots vectorize. Separately, the fused k walk now runs over j-panels of fixed
  cell AREA, which bounds the rolling face-flow buffers it carries across each k
  step to a fixed number of bytes whatever the block's aspect ratio, keeping them
  in L2 rather than round-tripping through a last-level cache that eight ranks
  share. Together, -33.7% at a 1M-cell block and -40.1% at 2M with all eight
  cores of a socket busy, -19.6% over the viscous pair end to end; serial the
  split is the other way round, -20.6% at 300k falling to -8.9% at 2M. The panel
  change is bitwise; the row forms move ``fvisc`` by 0.625 ulp of the field
  scale, confined to the two-cell shell the wall faces touch.

* ``set_residual`` now walks k over j-panels of fixed cell area, the same
  change made to ``set_visc_force`` above and for the same reason: what the
  fused walk carries from one k step to the next is a rolling pair of k-face
  flow planes, 720 KB at a 1M-cell block, so each plane was evicted before the
  next step read it and every component round-tripped through a last-level
  cache that all the ranks on a socket share. Panelling bounds what is live at
  once without changing what is read. With eight cores of a socket busy,
  -16.0% at 1M cells and -33.0% at 2M, -3.8% at 300k, neutral at 100k. Run
  serially it is instead about 2% slower at 1M and above, because a panel
  recomputes its lowest j-face row and a lone rank has the whole cache anyway;
  the contended figure is the one that describes a real solver run. The
  residual is bitwise unchanged --- only the change limiter's block-mean
  scale factor moves, by a few ulps, its summation order having changed.

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

.. _v0.1.0:

0.1.0 (2026-08-06)
------------------

* First public release on the Python Package Index.
