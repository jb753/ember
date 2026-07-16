# Viscous body-force kernels: features, a failed optimization, and a cache-blocking plan

Scope: the two Fortran kernels that build the viscous body force in
`src/ember/_fortran/viscous.f90`, driven by `Grid.update_sources`
(`src/ember/grid.py`). This note records how they work, documents a
component-first memory-layout optimization that was implemented, proven correct,
and then reverted because it regressed the real build, and lays out a plan for
k-slab cache blocking together with a benchmarking protocol that would actually
be representative of production.

**Status: the k-slab blocking of section 3 is implemented and measured; see
section 5 for the design as built and the A/B results (set_visc_force -30% to
-36%, update_sources -13% to -20%, production `kb = 8`).**

---

## 1. The two kernels

`Grid.update_sources` assembles `block.F_body_nd` in two Fortran passes with a
single periodic seam exchange between them:

```
for block in grid:            # phase 1
    set_tau_q_soa(...)        # per-cell tau, q, mu_turb
grid.connectivity.periodic.exchange_halos()   # one seam exchange (swap_by_ijk)
for block in grid:            # phase 2
    set_visc_force(...)       # face fluxes -> accumulate (negated) into F_body_nd
# then: polar source, optional SFD force
```

The phasing (all tau/q first, then one exchange, then all face fluxes) keeps the
seam consistent for block-to-block periodic interfaces, where a per-block
exchange would read a stale neighbour halo.

### 1.1 `set_tau_q_soa` (phase 1) -- `viscous.f90:310`

Per cell, computes:

- the viscous stress tensor `tau` (6 components: xx, rr, tt, xr, xt, rt),
- the heat flux `q` (3 components: x, r, t),
- the mixing-length turbulent viscosity `mu_turb`.

Key features:

- **Two-stage, SoA structure.** Stage 1 computes velocity gradients (Green-Gauss
  over the six faces, plus the Multall `GRADVEL` radial metric correction) and
  cell metrics into row temporaries `gVx/gVr/gVt(ni-1,3)`, `vct/rcr/ivr/rhoc(ni-1)`.
  Stage 2 assembles `tau`, `mu_turb` (`= min(rho * l^2 * |omega|, visc_lim)`,
  `visc_lim = 3000*mu`), and `q` (`lambda = mu*cp/Pr_lam + mu_turb*cp/Pr_turb`).
- **Vectorization.** The vector axis is `i` (the contiguous dim-1, the SIMD
  lane). Stage 1 vectorizes over `i` (32-byte AVX2 under the production flags).
  Stage 2 does **not** loop-vectorize: it carries a `sqrt` (vorticity magnitude)
  and a `min` (the mixing-length clamp) and scatters 9 component stores, so GCC
  falls back to weak per-iteration SLP. This was investigated (see section 2);
  fusing the two stages or splitting the row temps into rank-1 arrays does not
  help, and fusing measurably regresses because it drags the vectorized Stage-1
  work into the un-vectorizable Stage-2 loop.
- **Frame.** `Vt` is the block-relative tangential velocity; the strain rate is
  frame-invariant and the mixing-length vorticity is taken directly in the
  relative frame (no absolute-frame `+2*Omega` correction). `Pr_turb` is fixed at
  1.0 for the grid march.
- **Halo fill.** Writes owned cells (Fortran indices `2..ni`, etc.) then fills the
  single-sided face-halo slots with the nearest edge value (`+edge`). Corner/edge
  halo slots are left untouched.
- **Cost profile.** Compute-bound: ~50-60 ns/cell, roughly flat as the grid
  grows (see the sweep in section 2).

### 1.2 `set_visc_force` (phase 2) -- `viscous.f90:480`

Takes `tau_cell`/`q_cell` (possibly seam-exchanged) and accumulates the viscous
body force into `fvisc = F_body_nd[..., 1:]` (4 components: 3 momenta + energy).

Per face direction (i-faces, j-faces, k-faces), it:

1. scales the boundary halos of `tau`/`q` by `(2*wall - 1)` (wall=0 -> `-edge`
   so the face average is zero; wall=1 -> `+edge` single-sided stress);
2. runs a **face-flux loop** that averages the two adjacent cells' tau/q, dots
   with the face area, adds the viscous work term, and writes
   `flow_scratch(ni,nj,nk,4)`. **These loops vectorize (32-byte AVX2)** and are
   the bulk of the cost;
3. injects the **wall function** at the wall-adjacent face (`i=2`/`i=ni-1`, etc.)
   via `wall_func_iface/jface/kface`, blended by the wall mask;
4. runs an **accumulate loop** `fvisc += flow_scratch(lo_face) - flow_scratch(hi_face)`
   (also vectorizes).

After the three directions it:

- couples the **cusp seam** (a modelled trailing edge joining the `k=1` and
  `k=nk` faces) by averaging the two one-sided viscous fluxes so both faces carry
  the same shared seam flux (only when `i_cusp_start > 0`);
- **zeros** `fvisc` at wall-adjacent cells (the wall friction is applied at the
  off-wall face above; any residual viscous content is discarded);
- **negates** `fvisc` so the polar/SFD forces added afterward are not flipped.

Key cost feature: **`set_visc_force` is ~2x the cost of `set_tau_q_soa`** and is
**memory-bandwidth-bound at scale.** It makes three full-volume passes over
`tau_cell`/`q_cell`, one per face direction. Per-cell cost rises from ~82 ns/cell
when the tau/q working set is cache-resident to ~128 ns/cell once it spills L3.

### 1.3 The shared `tau_q_halo` scratch buffer

`block.tau_q_halo` is one Fortran-contiguous scratch buffer allocated
`(ni+1, nj+1, nk+1, 10)`. It is *reinterpreted* by each borrower and its contents
never persist between borrowers:

- the viscous pass carves `tau_cell(...,0:6)` and `q_cell(...,6:9)`
  (component-last today);
- the fused inviscid residual (`set_residual`) borrows it as `(ni,nj,nk,10)`
  flow scratch;
- `solver.py` multigrid borrows it flat (element count only).

The periodic seam exchange (`periodic_communicator.exchange_halos`) copies
neighbour edge-cell tau/q into local halo slots via the Fortran `swap_by_ijk`
(`indexing.f90:65`), which copies all components at given `(i,j,k)` slot indices.

---

## 2. The failed optimization: component-first tau/q layout

### 2.1 Idea

`set_visc_force` is memory-bound and reads `tau_cell`/`q_cell` three times. With
the component-**last** layout `(ni+1, nj+1, nk+1, 9)`, a single cell's 9 tau/q
values live in 9 different planes, `(ni+1)(nj+1)(nk+1)` elements apart -- nine
concurrent memory streams with no per-cell locality. Making the component the
**fastest** axis, `(6, ni+1, nj+1, nk+1)` / `(3, ...)`, packs each cell's tau/q
into one ~36-byte line, which should cut the memory traffic for the memory-bound
consumer.

### 2.2 What was done

The change was implemented end-to-end and is entirely reversible:

- both kernels re-indexed to component-first (`tau_cell(c,i,j,k)`);
- `swap_by_ijk` transposed to component-first (`h(:,i,j,k)`, a contiguous move);
- `grid.py` and `exchange_halos` carve the buffer component-first with
  `util.carve_view` (verified zero-copy: the views are F-contiguous,
  `OWNDATA=False`);
- the two direct-call viscous tests updated; the phase golden regenerated
  (its stored `tau_cell`/`q_cell` change shape but not values).

### 2.3 Correctness

Fully correct. The whole suite passed (1408 tests), and the goldens routed
through `update_sources` (`test_set_F_body_golden`, `test_residual_golden`) and
the seam test (`test_viscous_periodic`) passed **without regeneration**. Output
was bitwise-equivalent to float32 precision (differences at relative ~1e-7, i.e.
`-Ofast` reassociation between the two layouts, never exceeding the golden
tolerances).

### 2.4 Why it was reverted

Measured in an **isolated 2-file scratchpad build** (`gfortran viscous.f90 +
driver`, same optimization flags), `set_visc_force` looked ~14% faster at large
sizes -- locality appeared to beat the lost SIMD (component-first makes the tau
reads stride-6, a "complicated access pattern", so the face loops no longer
vectorize).

Measured in the **real f2py whole-program build** (`make compile`, all ~20
`_fortran/*.f90` compiled together with `-flto -fwhole-program`), the result
inverted:

| kernel (single block, 80 x 64 x 64) | baseline (comp-last) | component-first | delta |
| --- | --- | --- | --- |
| `set_tau_q_soa`   | 11.35 ms | 10.22 ms | -10% |
| `set_visc_force`  | 13.23 ms | 18.81 ms | **+42%** |
| `update_sources` (end-to-end) | 88.8 ns/cell | 100.5 ns/cell | **+11%** |

Root cause: in the full whole-program build, GCC vectorizes the component-**last**
face loops well (32-byte AVX2) and schedules the inlined `wall_func_*` helpers
better via cross-file IPA. Component-first loses that SIMD, and the loss
dominates the locality gain. The 2-file scratchpad build under-optimized the
baseline, which made component-first look like a win. `set_tau_q_soa` was
actually faster component-first, but it cannot adopt the layout alone -- it shares
the buffer with the consumer, which must agree on layout.

### 2.5 Lessons

1. **Benchmark in the real build.** A standalone compile of one kernel does not
   reproduce the whole-program IPA/vectorization of the f2py `.so`. This single
   fact flipped the sign of the result.
2. **The vectorizer report is a hint, not a verdict.** `-fopt-info-vec` is
   emitted per file at compile time and does not reflect whole-program inlining.
3. **Measure both kernels and end-to-end.** A change can help one kernel (here
   `set_tau_q_soa`) and hurt the dominant one (`set_visc_force`); only the
   end-to-end number decides.
4. **Keep the vectorization.** For these kernels the component-last SIMD face
   loops are the thing to protect; any layout change that strides the `i` reads
   is suspect.

Conclusion for layout: **do not change the tau/q component ordering.**
`set_tau_q_soa` is already near-optimal and should be left alone.

---

## 3. Plan: k-slab cache blocking for `set_visc_force`

The layout lever failed because it fought vectorization. Cache blocking is
orthogonal to layout: it keeps the component-**last**, vectorizing face loops
exactly as they are, and instead attacks the real bottleneck -- the three
full-volume streaming passes over `tau_cell`/`q_cell`.

### 3.1 Target

Today, for a block of `N = (ni-1)(nj-1)(nk-1)` cells, `set_visc_force` streams
`tau_cell`/`q_cell` (~9 floats/cell) from memory roughly three times (once per
face direction), plus a full `flow_scratch` write+read (~4 floats/cell) per
direction. When the tau/q working set
(`10*(ni+1)(nj+1)(nk+1)*4` bytes) exceeds L2/L3 this traffic sets the runtime.

Goal: read `tau_cell`/`q_cell` from memory ~once instead of ~three times, by
processing the volume in slabs small enough that a slab's tau/q stays hot in L2
across all three face directions.

### 3.2 Blocking scheme

Tile the outer `k` loop into slabs of `KB` cell-planes (`KB` a tuning
parameter). For each slab `k in [k0, k0+KB)`:

1. compute the **i-face** and **j-face** fluxes for the slab's cells, and the
   **k-face** fluxes for faces `k0 .. k0+KB` (one extra face plane on the high
   side, shared with the next slab);
2. accumulate `fvisc` for the slab's cells from those fluxes;
3. advance to the next slab.

Because a slab reads only tau/q planes `[k0-1 .. k0+KB]` (the owned planes plus
one halo on each k-side), and all three directions consume them before the slab
is retired, each tau/q plane is fetched from memory about once. Choose `KB` so
that `KB` planes of tau/q plus the slab's `flow_scratch` fit in L2 (target a few
hundred KB); `KB` in the range 4-16 is the likely sweet spot and must be swept.

### 3.3 Complications to handle (each has a concrete resolution)

1. **k-face overlap.** Cell `k` needs k-face fluxes at `k` and `k+1`; the shared
   high face of a slab is the low face of the next. Either recompute that one
   face plane at the slab boundary (cheap, 1 plane) or carry it forward in a
   one-plane buffer.
2. **Cusp seam.** The seam couples the `k=1` and `k=nk` faces across the whole
   `i`-range -- it is inherently non-local in `k` and cannot be done slab-locally.
   Resolution: keep the current cusp coupling as a **separate O(surface) pass**
   over just the seam faces, run after the slab sweep, exactly as today (it reads
   `flow_scratch(i,j,1)` and `flow_scratch(i,j,nk)`), so preserve those two face
   planes in a small side buffer rather than in the full `flow_scratch`.
3. **Wall-function injection at `k=2`/`k=nk-1`.** These live only in the first and
   last slab; inject them there. The `i=2`/`i=ni-1` and `j=2`/`j=nj-1` wall
   injections are per-slab as before.
4. **Wall-adjacent zeroing and final negation.** Both are per-cell and can run
   inside the slab as each slab's `fvisc` cells are finalized, or as a cheap final
   O(N) pass -- whichever benchmarks better.
5. **`flow_scratch` footprint.** With blocking, `flow_scratch` only needs to be
   slab-sized `(ni, nj, KB+1, 4)`, not full-volume. This shrinks its footprint
   and is a prerequisite for the fusion below.

### 3.4 Complementary optimization: rolling-buffer fusion

Independently of blocking, the per-direction pattern
`fvisc(cell) = flow(lo_face) - flow(hi_face)` can be fused with the face-flux
loop using a rolling buffer: as the sweep advances one face at a time, keep only
the previous face's flux and difference on the fly, eliminating the full
`flow_scratch` array and its write+read traffic (~8 floats/cell/direction).

- The i- and j-direction sweeps roll cleanly.
- The k-direction rolling conflicts with the cusp (which needs the `k=1` face
  retained until `k=nk`); keep the k=1/k=nk seam faces in the small side buffer
  from 3.3.2 and roll the interior.

Fusion and blocking compose: blocking keeps tau/q hot; fusion removes the
`flow_scratch` round-trip.

### 3.5 Constraints the implementation must preserve

- **Bitwise-equivalent physics** (to float reassociation): the golden tests
  (`test_set_F_body_golden`, `test_residual_golden`, `test_viscous_phases_golden`)
  and the seam test (`test_viscous_periodic`) must pass. If the change only
  reorders arithmetic, the `update_sources` goldens should pass within their
  existing `rtol=1e-4` tolerance without regeneration; do not regenerate to
  "make it pass".
- **Do not touch the component ordering** (section 2 conclusion) -- keep the
  vectorizing component-last face loops.
- The cusp seam averaging, the wall functions, the wall-adjacent zeroing, and the
  final negation must all be reproduced exactly.

### 3.6 Expected outcome and risk

If the kernel is genuinely bandwidth-bound at production block sizes, cutting
tau/q traffic from ~3x to ~1x and removing the `flow_scratch` round-trip could
approach the arithmetic-intensity ceiling -- but the component-first episode is a
warning that expected wins must be **measured in the real build** before being
believed. Blocking also adds address arithmetic and a boundary-face recompute
that could erode the gain at small block sizes; the crossover must be measured.

---

## 4. How to compile and benchmark so results represent production

The single most important lesson from section 2: **the isolated build lied.**
Follow this protocol.

### 4.1 Build exactly as production builds

- Build via the real path: `make compile` (which runs `uv pip install -e .`,
  invoking f2py with the meson backend and compiling **all**
  `src/ember/_fortran/*.f90` together). Do **not** benchmark a standalone
  `gfortran viscous.f90` compile -- whole-program IPA across the full file set
  changes inlining and vectorization of the hot loops, and that is precisely what
  inverted the component-first result.
- Production flags (`setup.py`, do not alter for a representative run):
  `-Ofast <march> -funroll-all-loops -finline-functions -finline-limit=10000
  --param early-inlining-insns=200 -flto -fwhole-program -fno-trapping-math
  -freciprocal-math -fipa-pta -floop-nest-optimize -fvect-cost-model=unlimited`.
  `-flto -fwhole-program` is the part a single-file build cannot reproduce.
- Match `-march` to the deployment target. Default is `-march=haswell` (portable
  AVX2 baseline). For a tuned perf run, `EMBER_MARCH="-march=native -mtune=native"`.
  Hold `-march` fixed across an A/B comparison.

### 4.2 Measure through the real call path

- Call the kernels the way production does -- from Python via
  `ember.fortran.set_visc_force` / `ember.fortran.set_tau_q_soa`, and the whole
  path via `Grid.update_sources`. This includes the real f2py marshaling.
- **Verify zero-copy.** Check that the arrays f2py receives are F-contiguous and
  not copied: `arr.flags['F_CONTIGUOUS']` is `True` and `arr.flags['OWNDATA']` is
  `False` for carved views. A silent f2py copy (wrong order/contiguity) will
  dominate and invalidate the measurement.

### 4.3 Use representative sizes, and sweep the cache boundary

- The tau/q working set is `10*(ni+1)*(nj+1)*(nk+1)*4` bytes. The kernel's
  behaviour changes qualitatively as this crosses L2 then L3, so **sweep a range
  of block sizes** and report **ns/cell**, not a single size.
- Use realistic per-**block** sizes. Production multi-block grids have blocks
  commonly `1e4`-`1e5` cells; a single `80 x 64 x 64` (~320k cells, ~14 MB tau/q)
  is already at the large end. A change that helps only when the working set
  spills L3 (e.g. `96^3` ~ 35 MB) may be irrelevant or harmful at typical block
  sizes -- test both regimes.

### 4.4 Measure both kernels and end-to-end, against a clean baseline

- Time `set_tau_q_soa`, `set_visc_force`, and `update_sources` **separately**. A
  change can help one and hurt another; the end-to-end `update_sources` number is
  the decision variable.
- A/B protocol: build the baseline and the candidate with **identical flags on
  the same machine**. Use a `git worktree` (or `git stash` + `make compile`) so
  each variant is a full, freshly-linked `.so`. Interleave baseline/candidate
  runs to cancel thermal/frequency drift.
- Correctness gate first: run the golden and seam tests before trusting any
  timing. Never report a speed number from a kernel that fails a golden.

### 4.5 Stabilize the measurement

- Single-thread: `OMP_NUM_THREADS=1` (the kernels are not OpenMP, but this avoids
  interference from any threaded dependency).
- Pin the CPU (`taskset -c <core>`) to prevent migration; if possible disable
  turbo / fix the frequency governor to reduce run-to-run variance.
- Warm up (a few untimed calls), then time many reps and report the **median**
  (and min) ns/cell, not a single run.
- Beware in-place mutation across repeated calls: `set_visc_force` scales the
  tau/q boundary halos by `(2*wall-1)` in place. Re-initialising per call, or
  accepting the bounded sign flip, keeps the compute cost representative -- just
  be consistent between baseline and candidate.

### 4.6 Optional: read the vectorizer report as a hint only

`gfortran <production flags minus -flto> -fopt-info-vec-all=report.txt -c
viscous.f90` shows which loops vectorize, useful for spotting a regression like
the lost face-loop SIMD. But it is per-file and pre-IPA; treat it as a lead to
confirm with a real end-to-end benchmark, never as the result itself.

---

## 5. Implemented: k-slab blocking, design as built and measured results

The plan of section 3 was implemented (July 2026) with two refinements, and
benchmarked with the protocol of section 4 via `scripts/bench_viscous.py`.

### 5.1 Design deltas from the section 3 plan

- **Carry plane without a side buffer.** Each slab's k-face loop leaves its top
  face plane in flow_scratch slot `kb+1`; the next slab's i/j phases write
  local slots `1..kb` only, so the plane survives them, and the next k-phase
  copies it to slot 1 before overwriting slot `kb+1`. No separate carry array,
  and the carried plane preserves any k-wall-function injection for both
  adjacent cells.
- **Cusp seam by recompute, not save.** Instead of preserving seam face planes
  in a side buffer (plan 3.3.2), both seam cells receive the identical
  post-sweep correction `0.5*(flow(k=nk) - flow(k=1))` (for the k=1 cell,
  avg - flow(1); for the k=nk-1 cell, flow(nk) - avg; the two coincide). The
  two raw seam-face fluxes are recomputed by the `kface_flow` helper --
  tau/q are unchanged after the entry halo scaling and neither seam plane takes
  a wall injection, so the recompute is exact. This removes all scratch
  capacity concerns and is O(cusp surface). Same arithmetic as before up to
  float reassociation; nk=2 with a cusp (seam cells coincide) is unsupported.
- **kb is a runtime argument** inferred by f2py from the flow_scratch shape.
  The caller (`Grid.update_sources`) carves `(ni, nj, kb+1, 4)` zero-copy from
  `block.scratch` with `util.carve_view` and clamps `kb = min(_KB_VISC, nk-1)`
  (`ember.grid._KB_VISC = 8`). `kb = nk-1` degenerates to a single slab.
- The face-loop bodies are verbatim from the unblocked kernel (only the
  flow_scratch k-index changed to the slab-local slot), so the i-vectorization
  is untouched. Wall injections run per slab; the k=2 / k=nk-1 injections run
  in whichever slab computes that face plane, immediately after its face loop.
  Wall-adjacent zeroing and the final negation are unchanged full passes.

Correctness: the whole suite passes with **no golden regeneration** (goldens
`test_set_F_body_golden`, `test_residual_golden`, `test_viscous_phases_golden`;
seam `test_viscous_periodic`). A new `test_set_visc_force_kb_consistent`
asserts **bitwise-identical** fvisc for `kb in {1, 2, 3}` (short last slab
included) against the single-slab `kb = nk-1` reference -- the per-cell
arithmetic and the i trip counts the vectorizer sees are the same for every
kb, so exact equality is the correct bar for slab bookkeeping.

### 5.2 A/B results

Protocol: `make compile` both sides (baseline = pre-tiling HEAD via
`git stash`), default `-march=haswell` flags, single thread pinned to one core
(`sched_setaffinity`), warmup then median ns/cell with kb variants interleaved
round-robin; a repeat tiled run agreed within ~2 ns/cell. Machine: Xeon
E5-2640 v3 (Haswell, 256 KB L2/core, 20 MB L3/socket). The tau/q working set
is 2.1 MB at 48x32x32, 14 MB at 80x64x64 (~L3), 36+ MB at 96^3 and up (L3
spill).

`set_visc_force`, median ns/cell:

| size | baseline | kb=1 | kb=2 | kb=4 | kb=8 | kb=16 | 1 slab |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48x32x32  | 39.0 | 29.4 | 28.0 | 27.4 | 27.1 | 26.9 | 27.3 |
| 64x48x48  | 36.6 | 28.3 | 26.4 | 25.5 | 25.0 | 24.9 | 25.4 |
| 80x64x64  | 43.2 | 34.5 | 31.9 | 30.3 | 29.8 | 29.8 | 37.9 |
| 96x96x96  | 46.4 | 34.4 | 31.8 | 30.4 | 30.1 | 30.9 | 42.0 |
| 128x96x96 | 45.6 | 33.3 | 30.6 | 29.3 | 29.0 | 33.2 | 40.8 |

`update_sources` end to end (the decision variable), median ns/cell:

| size | baseline | kb=4 | kb=8 | kb=16 |
| --- | --- | --- | --- | --- |
| 48x32x32  | 78.4 | 66.5 | 66.1 | 65.4 |
| 64x48x48  | 86.2 | 65.6 | 65.8 | 65.8 |
| 80x64x64  | 87.9 | 73.6 | 73.4 | 72.8 |
| 96x96x96  | 90.3 | 72.5 | 72.0 | 72.7 |
| 128x96x96 | 84.9 | 69.6 | 69.5 | 74.2 |

`set_tau_q_soa` is unchanged within noise (it was not touched).

### 5.3 Reading of the results

- **The win is real at every size, not just past the cache cliff**: -30% to
  -36% on `set_visc_force`, -13% to -20% end to end. There is no small-size
  crossover where blocking loses.
- Two separable effects: (a) at cache-resident sizes the single-slab column
  already beats baseline (27.3 vs 39.0 at 48x32x32) -- that is the compact,
  reused slab scratch replacing the baseline's full-volume `flow_scratch`
  round-trip; (b) at L3-spilling sizes multi-slab blocking adds its own large
  step (29.0 vs 40.8 at 128x96x96) -- that is the tau/q traffic cut from ~3x
  to ~1x, as designed.
- **kb choice**: 4-16 are near-tied at small sizes; kb=16 degrades at the
  largest planes (128x96 plane footprint) while kb=4-8 hold. `kb = 8` is the
  production default (`ember.grid._KB_VISC`); the clamp `min(kb, nk-1)` covers
  thin blocks.
- The rolling-buffer fusion of section 3.4 was deliberately deferred: with the
  slab scratch already L2-resident, its remaining upside is small and it would
  complicate the i-direction SIMD. Revisit only with a fresh A/B.
