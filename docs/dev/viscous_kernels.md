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
-36%, update_sources -13% to -20%, production `kb = 8`). Section 6 applies
the same tiling to `set_residual`; section 7 adds the rolling-buffer fusion
of section 3.4 on top (a further -8% to -16% on set_visc_force); the same
fusion for `set_residual` was first tried bitwise-exactly and rejected
(section 8), then landed as a slab-tiled port under a bounded float32
tolerance with a conditional anti-aliasing pad (section 9, -3% to -12%).
A second-level j-panel tile on top of the k-slabs was tried and rejected
(section 10): panels lose 2-5% at small/mid sizes and gain at most ~1% at
the largest, so the slab-tiled section 9 kernel remains production.
Section 11 drops `set_residual`'s separate `vt_rel` nodal array (derived
inline from `vt`/`r`/`Omega` instead), cutting one field from the
per-node footprint all three direction sweeps re-touch: implemented,
gauge-corrected wins at every size (-0.4% to -14.5%), no regressions.
Section 12 records two further candidates in the same family; the first
(fusing the i- and j-direction sweeps into one dU write) is implemented
and measured in section 13 (`set_residual` -7% to -12% gauge-corrected,
no regression, no golden regeneration), the second (precomputing per-node
`pm`/`mf` once) remains untried.
Section 14 moves to the RK multigrid path: the coarse-level IRS smoother in
`rk_mg_irs`/`scree_mg_irs` was handed the untiled `smooth_residual_tri`,
whose i-direction Thomas solve the link-stage vectorization report shows as
scalar; swapping it for the already-existing `smooth_residual_tri_tiled`
(the fine-grid smoother) vectorizes the coarse i-solve and wins -20% to -27%
on the isolated coarse smoother at the dominant level and -1.6% to -6.8%
gauge-corrected on the full RK stage, bitwise-identical, no golden
regeneration.
Section 15 continues the RK path once the report confirms every remaining hot
loop already vectorizes: the last lever is memory traffic, so the final
prolongation hop (`mg_prolong2x_fine`) is fused with the cell->node scatter
(`cell_to_node_generic`) into a rolling two-plane `mg_prolong2x_fine_scatter`,
eliminating the full-volume increment `tmp` write+read round-trip. Wins -11%
to -30% on the full RK stage (raw, both `rk_mg_irs` and `rk_mg_noirs`, every
size, both repeats), ~1 ulp scatter reassociation, no golden regeneration.**

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
  Stage 2 does **not** loop-vectorize, and falls back to weak per-iteration
  SLP. The `sqrt` (vorticity magnitude) and `min` (mixing-length clamp) are
  **not** the cause -- both have native AVX2 vector forms under `-Ofast`, and
  a real-build probe (July 2026) with both eliminated failed identically. The
  vectorizer reports "no vectype" on the first row-temp read; the trigger
  survives making the row temps dummy arguments, dropping
  `-floop-nest-optimize`, and stripping the loop to a minimal
  read-gVx/store-tau skeleton, so the root cause (a data-ref analysis
  failure, exact ingredient unisolated) is subtler than any one statement.
  Earlier experiments also showed fusing the two stages or splitting the row
  temps into rank-1 arrays does not help, and fusing measurably regresses
  because it drags the vectorized Stage-1 work into the un-vectorizable
  Stage-2 loop.
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

### 4.6 The vectorizer report: use the link-stage one

`EMBER_OPT_REPORT=<path> make compile` writes the vectorization report
(`-fopt-info-vec-all`) of the **real production build** to `<path>`. The
flag is injected at link time (`setup.py` appends it to `LDFLAGS`), where
GCC's LTO backend does the whole-program codegen -- so this report describes
the code that actually runs, post-IPA, with cross-file inlining applied.
Expect only the innermost (i) loops to vectorize; "missed" entries on outer
j/k/m loops are normal.

Two report sources to distrust:

- a standalone `gfortran <flags minus -flto> -fopt-info-vec-all=r.txt -c
  file.f90` is per-file and pre-IPA -- a hint only (this is what inverted the
  section 2 result);
- a compile-stage report from the LTO build itself (e.g. via
  `-ffat-lto-objects`) describes the *discarded* per-TU codegen and can flag
  spurious misses (observed: "no vectype" complaints on loops that the link
  stage vectorizes cleanly at 32 bytes).

In all cases the report is a lead to confirm with a real end-to-end
benchmark, never the result itself.

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

---

## 6. Same treatment for `set_residual` (the fused inviscid residual)

`set_residual` (`residual.f90`), which backs `residual_nd` and runs per RK
stage per level, had the same disease at larger scale: three full-volume face
passes each re-streaming ~10 nodal fields, plus fifteen slots of full-volume
flow scratch (`flow_i` 5 + `flow_jk` 10) round-tripped into the fused dU
accumulate. Measured pre-tiling: 27.5 ns/cell cache-resident rising to 43.3 at
96^3.

### 6.1 Design (mirrors section 5)

- Slab sweep over `kb` cell planes inside `set_residual`; the three face-flow
  helpers gained slab/face-range arguments and write slab-local planes; the
  fused dU accumulate runs per slab. Scratch shrinks to `flow_i(ni,nj,kb,5)`
  (carved from `block.scratch`) and `flow_jk(ni,nj,kb+1,10)` (carved from
  `tau_q_halo`).
- k-face carry: each slab leaves its top k-face plane in `flow_jk` slot kb+1,
  components 6:10 -- untouched by the next slab's i/j phases, which write
  `flow_i` and components 1:5 only -- and the next slab copies it to slot 1.
- Cusp seam: `correct_cusp_kface` (which averaged/rebuilt the seam flux planes
  before accumulation) became `correct_cusp_kface_du`, a deferred O(surface)
  pass that recomputes the raw wall-masked seam fluxes from the nodal fields
  and adds corrected-minus-raw deltas to dU. Exact recompute for the same
  reason as `kface_flow` (nothing mutates the nodal inputs); nk=2 with a cusp
  is unsupported, as in the viscous kernel.
- One shared slab-depth knob: `_KB_VISC` was renamed `ember.grid._KB_SLAB` and
  drives both tiled kernels.

Correctness: `test_residual_kb_consistent` asserts bitwise-identical dU for
`kb in {1,2,3}` vs the single-slab reference. The residual golden was
**regenerated** -- justified, not to "make it pass": the old and new kernels
compiled standalone at -O0 (strict FP, no reassociation possible) produce
**bitwise identical** dU for every kb, proving the source arithmetic is
unchanged; the golden shift is -Ofast codegen only (different FMA/vectorization
choices on the reshaped loops), 1 ulp of the flux scale (max rel 1.5e-7),
tripping one near-cancelling interior cell's region-scaled atol by 1.5x.

### 6.2 A/B results

Same protocol and machine as section 5.2; repeat tiled run within ~1 ns/cell.

`set_residual`, median ns/cell:

| size | baseline | kb=1 | kb=2 | kb=4 | kb=8 | kb=16 | 1 slab |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48x32x32  | 24.1 | 25.7 | 24.9 | 24.6 | 24.3 | 23.9 | 24.6 |
| 64x48x48  | 31.0 | 30.2 | 26.8 | 26.0 | 26.0 | 26.7 | 30.4 |
| 80x64x64  | 40.1 | 32.9 | 31.8 | 31.4 | 31.3 | 31.9 | 38.7 |
| 96x96x96  | 43.6 | 32.2 | 31.2 | 30.8 | 31.3 | 33.6 | 42.5 |
| 128x96x96 | 39.7 | 28.0 | 27.4 | 27.0 | 27.7 | 31.5 | 38.4 |

`update_residual` end to end tracks the kernel within ~1 ns/cell (the slab
carve overhead is negligible).

### 6.3 Reading

- **-22% to -32% at L3-spilling sizes; neutral at cache-resident sizes**
  (48x32x32: 24.3 vs 24.1 at kb=8 -- within noise). Unlike the viscous kernel
  there is no small-size win, because the baseline's flow scratch was already
  reused per direction; the entire gain here is the traffic cut, and it
  appears exactly where the working set spills, as designed.
- The single-slab column reproduces the baseline (38.4 vs 39.7 at the largest
  size) -- the restructure itself costs nothing; the win is all blocking.
- **kb choice**: the residual optimum is kb=4, the viscous optimum kb=8, but
  each is within ~2-3% of the other's optimum at every size, so the shared
  `_KB_SLAB = 8` stands. Not worth splitting the knob.
- Since `set_residual` runs every RK stage, this feeds straight into
  wall-clock per iteration at production block sizes.

### 6.4 Build-hygiene gotcha found on the way

Running `gfortran -fsyntax-only src/ember/_fortran/<file>.f90` from the repo
root drops `<module>.mod` files there; a later `make compile` syntax check can
then resolve `use` against the stale module and fail (or worse, pass wrongly)
after the source is reverted/changed. Delete stray `*.mod` from the repo root
before builds, or run syntax checks in a temp directory.

---

## 7. Implemented: rolling-buffer fusion for `set_visc_force` (section 3.4)

The fusion deferred in section 5.3 was revisited (July 2026) with a fresh A/B
per the section 4 protocol, and **adopted**: the expected "small upside"
turned out to be a solid -8% to -16% on `set_visc_force` at the production
`kb = 8`, -4% to -7% end to end.

### 7.1 Design as built

The k-slab sweep is unchanged (kb knob, wall injections, cusp recompute
correction, zeroing, negation); only the staging of face fluxes moved from
the slab-sized `flow_scratch(ni,nj,kb+1,4)` into rolling buffers, with every
face-loop body kept verbatim (i remains the innermost SIMD axis):

- **i-direction**: one face row `rows(ni,4, slot 1)` per `(j,k)`; the
  `i=2`/`i=ni-1` wall injections apply to the row; the row is differenced
  into `fvisc` immediately (assignment, as before).
- **j-direction**: an alternating face-row pair (rows slots 2/3); face row j
  is computed, wall-injected if it is the `j=2`/`j=nj-1` face, then cell row
  `j-1` accumulates the difference and the slots swap.
- **k-direction**: an alternating face-plane pair `planes(ni,nj,4,2)`; same
  rolling pattern per face plane. The inter-slab carry copy of section 5.1
  **dissolves**: the plane pair persists across the slab boundary (the
  intervening i/j phases touch only rows), so the previous slab's top plane
  is simply still there.

Scratch shrinks from `(kb+1)` full planes to 2 planes + 3 rows, carved
together zero-copy by `util.carve_view(block.scratch, (ni,nj,4,2), (ni,4,3))`
(fits `block.scratch`'s 5 nodal slots for `nk >= 3`, or `nk = 2` with
`nj >= 6`; the carve raises if not).

Correctness: the whole suite passes with **no golden regeneration** -- the
fusion only re-orders when each face flux is staged and consumed, not any
per-cell arithmetic -- and `test_set_visc_force_kb_consistent` still passes
bitwise (kb now changes only the direction-interleaving order).

### 7.2 A/B results

Same protocol and machine as section 5.2 (`make compile` both sides, baseline
= pre-fusion HEAD via `git stash`, `-march=haswell`, single thread pinned,
median ns/cell, kb variants interleaved). Baseline column is the tiled kernel
at its production `kb = 8`.

`set_visc_force`, median ns/cell:

| size | tiled kb=8 | kb=1 | kb=2 | kb=4 | kb=8 | kb=16 | 1 slab |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48x32x32  | 29.8 | 25.5 | 25.4 | 25.3 | 25.1 | 25.0 | 25.0 |
| 64x48x48  | 26.6 | 23.0 | 22.8 | 22.7 | 22.6 | 22.6 | 22.5 |
| 80x64x64  | 30.6 | 28.3 | 27.8 | 27.5 | 27.3 | 27.3 | 31.6 |
| 96x96x96  | 30.5 | 29.0 | 28.3 | 28.0 | 27.9 | 28.3 | 34.8 |
| 128x96x96 | 29.2 | 27.7 | 27.1 | 26.9 | 26.9 | 27.7 | 34.4 |

`update_sources` end to end (the decision variable), median ns/cell:

| size | tiled kb=8 | kb=1 | kb=2 | kb=4 | kb=8 | kb=16 | 1 slab |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 48x32x32  | 68.7 | 64.3 | 64.1 | 64.0 | 63.9 | 63.8 | 63.7 |
| 64x48x48  | 65.5 | 61.3 | 61.6 | 61.2 | 61.0 | 61.3 | 61.1 |
| 80x64x64  | 73.2 | 71.2 | 70.5 | 70.1 | 69.9 | 70.0 | 74.6 |
| 96x96x96  | 72.1 | 70.4 | 69.7 | 69.4 | 69.3 | 69.6 | 76.1 |
| 128x96x96 | 69.3 | 67.4 | 66.8 | 66.5 | 66.6 | 67.4 | 74.2 |

`set_tau_q_soa` is unchanged within noise (untouched).

**Cross-build noise gauge.** The residual kernel was identical in both
builds, so its A/B delta measures run-to-run + codegen drift: up to
~2.9 ns/cell, systematic per size (e.g. -2.8 at 48x32x32, +1.9 at 80x64x64 --
whole-program LTO re-inlines around the changed viscous file). The fusion win
exceeds this co-measured drift at every size, and at 80x64x64 wins *against*
a +1.9 unfavourable drift; only at 128x96x96 could drift account for up to
half the end-to-end delta.

### 7.3 Reading

- The fusion removes the slab scratch write+read round-trip; since that
  scratch was already L2-resident, the win is mostly the ~8 floats/cell/
  direction of L2 traffic plus the removed carry copy -- worth more than
  section 5.3 guessed. The feared i-SIMD complication did not materialise:
  the face-loop bodies are verbatim and still vectorize.
- **kb sensitivity is now nearly flat** (kb=1 within ~2% of kb=8 everywhere):
  with no slab scratch, kb only paces how the three directions interleave
  over the tau/q planes. The blocking still matters at L3-spilling sizes
  (1-slab column: 34.4 vs 26.9 at 128x96x96) -- fusion did not replace
  tiling, it composed with it, exactly as section 3.4 predicted. `_KB_SLAB
  = 8` stands.
- At cache-resident sizes the 1-slab column now matches kb=8 (25.0 / 22.5):
  the small-size win of section 5.3(a) -- compact reused scratch -- is fully
  captured by the rolling buffers alone.
- Follow-on candidate: the same fusion applies to `set_residual`'s
  `flow_i`/`flow_jk` slab scratch (section 6); that is a separate change and
  needs its own A/B. **Tried and rejected -- see section 8.**

---

## 8. Tried and rejected: rolling-buffer fusion for `set_residual`

The section 7 fusion was ported to `set_residual` (July 2026), passed every
correctness gate, won at four of five sizes -- and was **reverted** because it
regressed ~10% at the largest size, failing the no-regression adoption rule.
Recorded here per the section 2 precedent: negative results get written up.

### 8.1 Design tried

`set_residual` differs from the viscous kernel in that dU is written once by
a single fused expression (i, j, k face differences + f_body, fixed term
order). Preserving that arithmetic bitwise forces all six face flows per cell
to coexist, which leads to a **fully fused single sweep** rather than
per-direction rolling: per cell plane k, a rolling k-face plane pair
`planes(ni,nj,5,2)`; per cell row j, an i-face row and a rolling j-face row
pair `rows(ni,5,3)`; dU assembled per row with the identical expression. The
three slab-based face-flow helpers became row/plane-granularity helpers
(`iface_flow_row`, `jface_flow_row`, `kface_flow_plane`, accum/put bodies
verbatim). A cell plane touches only nodal planes k..k+1, so the sweep
streams every nodal input once by construction -- **full fusion subsumes the
k-slab tiling**, and the `kb` argument left the kernel entirely.

Correctness: whole suite green. The residual golden tripped exactly as in
section 6 (one near-cancelling interior cell, 1 ulp at the flux scale, 1.49x
its region atol); the -O0 strict-FP standalone proof (old vs new kernel over
five shapes including ni=2 / nj=2 / nk=2 boundaries and the cusp path) was
**bitwise identical**, so the shift was again -Ofast codegen only and the
golden was regenerated during the trial (and restored on revert).

### 8.2 A/B results and the rejection

Same protocol and machine as 5.2/7.2; baseline = tiled `set_residual` at HEAD
via `git stash`. This time the **viscous kernel was identical in both
builds** and serves as the noise gauge: its cross-build delta was at most
0.5 ns/cell, so the deltas below are real.

`set_residual`, median ns/cell (tiled at production kb=8 vs fused):

| size | tiled kb=8 | fused | delta |
| --- | --- | --- | --- |
| 48x32x32  | 24.5 | 23.3 | -4.7% |
| 64x48x48  | 26.4 | 23.6 | -10.6% |
| 80x64x64  | 30.9 | 30.5 | -1.4% |
| 96x96x96  | 32.2 | 30.7 | -4.7% |
| 128x96x96 | 27.5 | 30.3 | **+9.9%** |

`update_residual` end to end tracks the kernel within ~0.5 ns/cell.

The regression at 128x96x96 is +2.7 ns/cell against a 0.2 ns/cell gauge --
unambiguous. The adoption rule (win above noise, regress nowhere) fails, so
the kernel change was reverted; the tiled section 6 kernel remains
production.

### 8.3 Reading

- The tiled baseline is anomalously strong at 128x96x96 (27.5, faster than
  its own 31-32 at 80x64x64 / 96^3 -- visible already in the section 6
  tables), while the fused kernel is flat (30.3 vs 30.7). The regression is
  as much "the staged version exploits this size unusually well" (clean
  ni=128 vector runs and line-aligned streaming through simple full-volume
  loops) as it is a fusion cost.
- The fused sweep's plane pair (`10*ni*nj*4` bytes, ~1 MB at 128x96) spills
  L2, and the pa plane is re-read a full cell-plane's nodal traffic after it
  was written; but 96^3 spills too and still won, so L2 residency alone does
  not explain the flip.
- Possible rescue: tile the sweep over j-panels so the rolling plane pair
  stays cache-sized at any plane dimension, restoring the small-size wins
  without the large-plane exposure. Untried -- would need its own A/B against
  both the tiled kernel and this fused variant.
- The full patch (kernel, callers, tests, bench support) is preserved in the
  repo stash: `git stash list` -- "set_residual rolling fusion (rejected)".

Contrast with section 7: the same idea won cleanly for `set_visc_force`
because there the accumulate is per-direction (rolling preserves arithmetic
direction-by-direction, tiling is retained and composes), whereas here full
fusion had to replace tiling and gave up the slab structure that the large
size apparently rewards. **Superseded by section 9**, which recovers the
slab structure by relaxing the bitwise constraint.

---

## 9. Implemented: slab-tiled rolling fusion for `set_residual` (relaxed tolerance)

Section 8's failure was forced by the bitwise constraint: the staged kernel's
single seven-term dU expression demanded an all-directions sweep, which cost
the slab tiling. Relaxing to a bounded float32 tolerance (July 2026) allows
the **section 7 viscous structure to port verbatim**: per-direction
accumulation (`dU = i-diff + f_body; += j-diff; += k-diff`), each direction
fused through rolling buffers (section 8's row/plane helpers reused as-is),
all inside the kb-slab sweep. The inter-slab k-plane carry is automatic, as
in the viscous kernel. `_KB_SLAB` drives both kernels again.

### 9.1 Tolerance and correctness

Only the final per-cell sum is reassociated (three partial sums instead of
one expression); every face-flow value is computed by identical arithmetic.
Quantified at -O0 (strict FP) old vs new over five shapes including
ni/nj/nk = 2 boundaries and the cusp: ~51% of dU values differ, with maximum
difference **1.2 ulp of the flux scale** (max rel 2.3e-5). The residual
golden was regenerated on that bounded-ulp argument (the bitwise -O0 proof of
section 6/8 does not apply to a deliberate reassociation).
`test_residual_kb_consistent` remains **bitwise** across kb, as for the
viscous kernel. The link-stage vectorization report
(`EMBER_OPT_REPORT`, section 4.6) confirms every hot inner loop -- the three
face-flow helpers and the three accumulates -- vectorizes at 32 bytes.

### 9.2 The ni=128 anomaly and the anti-aliasing pad

The first A/B won -4 to -8% at three sizes but regressed +2-3% at
128x96x96. An extended ladder (160x96x96, 128x128x128, 192x128x128)
localized the deficit to **power-of-two ni**, not size: ni=160/192 won up to
-12% while both ni=128 sizes lost. Root cause: with `ni*nj*4` bytes an exact
page multiple (128x96x4 = 48 KB), the k-accumulate's ten concurrent
component streams (5 components x plane pair) 4K-alias into the same L1
sets. The same mechanism makes **every** residual variant -- the tiled
baseline included -- allocation-sensitive at ni=128: across processes the
baseline itself wobbled 27.5-30.2 ns/cell there while the co-measured
viscous gauge held to a few tenths. (Earlier readings of the baseline as
"anomalously good at 128x96x96" were partly this lottery.)

Fix: one padding j-row in the plane buffer, applied **conditionally** --
`njp = nj+1` iff `ni*nj` is a multiple of 1024 -- because an unconditional
pad measurably hurt (~+5%) the small blocks it cannot help. `njp` is a
runtime argument; the pad is arithmetic-neutral (the extra row is never
touched).

### 9.3 A/B results (conditional pad vs tiled kb=8, median ns/cell)

Same protocol; the identical viscous kernel is the cross-build gauge.

| size | tiled | port+cond-pad | delta | note |
| --- | --- | --- | --- | --- |
| 48x32x32    | 24.8 | 24.0 | -3.5% | pad inactive |
| 64x48x48    | 26.3 | 24.0 | -8.5% | |
| 80x64x64    | 31.3 | 30.3 | -3.1% | |
| 96x96x96    | 31.9 | 30.4 | -4.6% | |
| 128x96x96   | 28.2/30.2 | 28.7/29.6 | +1.6%/-1.9% | tie within the lottery band |
| 160x96x96   | 30.5 | 26.8 | -12.0% | |
| 128x128x128 | 31.7 | 29.6 | -6.9% | ~-5% gauge-corrected |
| 192x128x128 | 38.6 | 31.6 | -18.2% | ~-12% gauge-corrected (gauge drifted -2.6) |

`update_residual` end to end tracks the kernel within ~1 ns/cell. Wins at
eight of nine configs; the ninth (128x96x96) flips sign between processes
inside its own +-2.5 ns/cell alignment band -- no reproducible regression.

### 9.4 Notes

- **kb**: near-flat 4-16 at most sizes, but kb=16 degrades at large planes
  and the biggest size prefers kb=4 (36.4 vs 38.2 at 192x128x128 pre-pad).
  `_KB_SLAB = 8` retained; revisit only if production blocks grow past ~1M
  cells.
- **Benchmarking gotcha for the protocol of section 4**: at power-of-two ni,
  per-process heap layout swings residual-kernel timings by +-2-3 ns/cell
  (and even the viscous kernel by ~2.6 at 192x128x128). Never conclude from
  a single process at such sizes; use the co-measured unchanged-kernel gauge
  and repeat runs.
- The section 8 full-fusion variant remains in the repo stash for reference;
  this port supersedes it.

---

## 10. Tried and rejected: second-level j-panel tile for `set_residual`

The remaining untiled lever in `set_residual` after section 9 was the plane
dimension itself: the per-slab working set (nodal panel ~`10*ni*nj*(kb+1)`,
dU slab, plane pair) scales with the full `ni*nj` plane, and two section 9
observations hinted it mattered -- kb=16 degrades at large planes, and the
largest size preferred kb=4. Section 8.3 had already proposed the fix:
tile j into panels so the active set scales with `ni*jbw` instead. Tried
(July 2026) and **rejected**: panels lose or tie everywhere, with the only
gain ~1% at the very largest size.

### 10.1 Design tried

A `jbw`-wide j-panel loop wrapped around the k-slab sweep (panel outer,
slab inner); per panel the sweep is the unmodified section 9 slab sweep
restricted to cell rows `j0..j1`: `kface_flow_plane` gained a j-range, the
j-direction's rolling pair starts at face `j0` (an interior panel-boundary
face row is computed twice, once per adjoining panel -- same cost shape as
the per-k `j=1` row), and the k-plane carry works per panel because pa/pb
reset at each panel start. Per-cell arithmetic and accumulation order are
unchanged for any jbw, so the kernel is **bitwise identical** across jbw
(verified by a `test_residual_jb_consistent` sweep; goldens passed without
regeneration), and `jbw = nj-1` degenerates exactly to the section 9
kernel. That degeneracy made the A/B within-build: baseline (`jfull`) and
candidate panel widths interleaved round-robin in one process, no
cross-build gauge needed. `jfull` also reproduced the section 9.3 numbers
within noise at every size, confirming the restructure itself is free.

### 10.2 A/B results and the rejection

Same protocol and machine; two full repeat processes (r1/r2 below, the
pow2-ni lottery shifts whole columns by 1-2 ns/cell but never the
within-run ordering). `set_residual` at production kb=8, median ns/cell:

| size | jfull | j8 | j16 | j32 | j64 |
| --- | --- | --- | --- | --- | --- |
| 48x32x32    | 23.7/23.0 | 26.4/25.7 | 24.9/24.2 | - | - |
| 64x48x48    | 24.0/24.2 | 25.8/26.1 | 24.9/25.2 | 24.6/24.9 | - |
| 80x64x64    | 29.3/29.2 | 29.4/29.4 | 30.3/30.3 | 30.8/30.7 | - |
| 96x96x96    | 30.3/30.2 | 30.8/30.8 | 31.0/30.9 | 31.5/31.5 | 31.2/31.1 |
| 128x96x96   | 30.1/28.2 | 31.6/30.3 | 31.9/30.3 | 31.4/29.7 | 30.9/29.1 |
| 160x96x96   | 26.0/26.6 | 27.8/28.9 | 27.5/28.2 | 26.9/27.7 | 26.6/27.3 |
| 128x128x128 | 29.4/29.1 | 32.6/32.0 | 31.7/31.4 | 30.6/30.4 | 29.9/29.6 |
| 192x128x128 | 31.4/30.5 | 33.3/32.6 | 32.2/31.4 | 31.4/30.6 | 31.0/30.2 |

`update_residual` end to end tracks the kernel within ~1 ns/cell. kb
crosses (kb=4 and kb=16 at j16) were also measured: both sit at or above
the kb=8 column at every size, so panels do not move the slab-depth
optimum either -- `_KB_SLAB = 8` stands.

Full width wins or ties at seven of eight sizes; the trend is monotone
(wider panels always better). The lone exception, 192x128x128, shows a
consistent ~0.3 ns/cell (~1%) j64 edge in both processes -- but the same
j64 loses 1.7-2.7% at four other sizes, failing the adoption rule (win
above noise, regress nowhere).

### 10.3 Reading

- The hypothesis was wrong at its root: the slab-tiled kernel already
  fetches everything from DRAM ~once, so panel tiling can only convert
  L3-resident re-touches (the three per-direction dU read-modify-writes
  and nodal re-reads within a slab) into L2-resident ones -- and on this
  Haswell that L3 traffic is evidently not the bottleneck.
- The panel costs are real and visible: one extra j-face row recompute per
  panel boundary per k (~`1/jbw` of the j-face work, matching the observed
  j8 penalty), plus j-restricted sweeps chopping the long contiguous
  plane walks the hardware prefetchers ride into `ni*jbw` chunks across
  ten-plus concurrent streams.
- The section 9.4 large-plane kb sensitivity is better read as slab
  working set vs L3 capacity, and its remedy -- if production blocks ever
  grow past ~1M cells -- is simply a smaller kb at large planes, not a
  second tiling dimension.
- The patch (kernel, caller knob `_JB_PANEL`, jb-consistency test, bench
  jbw sweep) is preserved in the repo stash: `git stash list` --
  "set_residual j-panel tiling (rejected)". Raw data:
  `scripts/bench_panelres_ab.csv` (labels panel-r1/panel-r2).

---

## 11. Implemented: derive `vt_rel` inline in `set_residual` instead of a stored nodal array

Section 10's reading was that the slab-tiled kernel already fetches every
nodal field from DRAM ~once, so further *spatial* tiling had nothing left
to cut. The remaining lever is *field count*: `set_residual` carried
`vt_rel(ni,nj,nk)` as its own stored nodal array, used only in the
mass-flux term `mf(3) = ... + w*cons(i,j,k,1)*vt_rel(i,j,k)`. But
`vt_rel` is just `Vt - Omega*r` (`block.py:592-608`,
`_Vt_rel_nd_uninit`), and `set_residual` already receives `vt`, `r`, and
the per-block scalar `Omega` as separate arguments -- so the whole array
is redundant and can be derived per node instead of streamed from its
own buffer.

### 11.1 Design

`vt_rel` removed from all 5 places it appeared in `residual.f90`
(`iface_flow_row`, `jface_flow_row`, `kface_flow_plane`,
`correct_cusp_kface_du`, and `set_residual` itself, which threads it to
the other four); every `accum()` now computes
`mf(3) = mf(3) + w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k))` inline.
No tiling, kb bookkeeping, wall masking, or cusp logic touched. The two
call sites (`Grid.update_residual` in `grid.py`, and
`scripts/bench_viscous.py`'s `make_residual_call`) simply stop passing
`vt_rel=block.Vt_rel_nd`; the Python-side `Vt_rel_nd` cached array itself
is untouched and still backs the `set_visc_force`/`set_tau_q_soa` calls
and the rothalpy calc in `block.py:502`, which still consume it.

### 11.2 Correctness

Not claimed bitwise: the old value came from two separate float32 numpy
ops (`np.multiply` then `np.subtract`, no fusion); the new Fortran
`vt(i,j,k) - Omega*r(i,j,k)` compiled under production `-Ofast` may
legally FMA-fuse into a single fused-multiply-subtract with different
rounding. Measured directly rather than assumed: `test_residual_matches_golden`
**passed without regeneration**. Quantified anyway --
max abs diff **0.0009766** against a field scale of 13420.06 (~0.6 ulp of
the flux scale, smaller than section 6's 1 ulp and section 9's 1.2 ulp),
369/1920 cells differ (19.2%), max relative diff among differing cells
1.25e-4 -- comfortably inside the golden's existing `rtol=1e-4` /
region-scaled `atol`. `test_residual_kb_consistent` (kb in {1,2,3} vs.
kb=nk-1) passes unaffected, as expected since kb bookkeeping isn't
touched. Full suite: 1414 passed, 1 skipped (pre-existing, unrelated).

### 11.3 A/B results

Protocol: `scripts/bench_viscous.py`, `make compile` both sides via
`git stash`, single thread pinned, production `kb = 8`, median ns/cell.
Two full repeat processes (r1/r2). This session's cross-build noise floor
(the co-measured, **untouched** `set_visc_force`/`set_tau_q_soa` gauge)
was unusually wide -- up to +-4.5% swing between r1 and r2, worse than
the ~0.2-2.9 ns/cell bands in sections 8-10 -- so raw deltas are reported
gauge-corrected (`set_residual` delta minus the co-measured `set_visc_force`
delta at the same size/rep, cancelling common-mode drift):

| size | r1 raw | r1 gauge | r1 corrected | r2 raw | r2 gauge | r2 corrected |
| --- | --- | --- | --- | --- | --- | --- |
| 48x32x32    | +0.09% | +0.47% | -0.37% | -3.01% | -0.58% | -2.44% |
| 64x48x48    | -8.96% | +0.57% | -9.53% | +0.68% | +2.03% | -1.35% |
| 80x64x64    | -5.23% | +1.21% | -6.45% | -3.32% | +0.04% | -3.35% |
| 96x96x96    | -0.53% | +0.60% | -1.13% | -5.52% | +1.56% | -7.08% |
| 128x96x96   | -1.24% | +3.06% | -4.30% | -15.43% | -0.96% | -14.47% |

`update_residual` end to end tracks `set_residual` within ~1 ns/cell at
every point (both reps). Every gauge-corrected cell in both repeats is a
win (-0.4% to -14.5%); the one raw uncorrected uptick (r2, 64x48x48,
+0.68%) sits inside that run's own gauge noise (+2.03% at the same
size/rep) and resolves to a win once corrected. No regression survives
gauge correction at any size in either repeat.

### 11.4 Reading

- Unlike section 10's j-panel tile, this is not spatial tiling -- it
  removes one field's worth of *redundant re-reads*: every interior node
  is a corner of up to ~4 face calls per direction across the 3 direction
  sweeps, so `vt_rel` was being fetched from its own buffer repeatedly
  per slab. Deriving it from already-resident `vt`/`r` costs one multiply-
  subtract per touch instead of one more array stream.
- The win growing with size (largest at 128x96x96, the most L3-pressured
  configuration) suggests at least part of the benefit is real L2/L3
  traffic reduction, not purely redundant arithmetic -- partially
  refining, not contradicting, section 10.3's "L3 traffic is not the
  bottleneck" reading: that finding was specific to *adding* a second
  tiling dimension, not to *removing* a field from the existing slab
  window.
- This composes for free with sections 6-10's tiling/fusion structure
  (no kb, wall-masking, or cusp-correction interaction), so it stacks
  with the current production kernel rather than competing with it.
- Raw data: `scripts/bench_vtrel_ab.csv`, labels baseline/candidate (r1)
  and baseline_r2/candidate_r2 (r2).

---

## 12. Untried: further per-node redundancy reduction in `set_residual`

Section 11's result -- removing one field from the per-node data that all
three direction sweeps re-touch produced real, size-scaling wins --
suggests the remaining untried levers in this family are worth a proper
A/B too, rather than further spatial tiling (section 10 closed that
door). Two candidates, not yet implemented:

### 12.1 Fuse the i- and j-direction sweeps into one pass -- IMPLEMENTED (section 13)

Currently the i-direction and j-direction sweeps are two separate
full-slab passes over `(j,k)`: the i-sweep assigns `dU` once, then the
j-sweep is a wholly separate pass that re-reads and accumulates into it.
Both already iterate row-by-row over the same `(j,k)` grid and don't
depend on each other's rolling buffer (the i-direction needs no carry
across j; the j-direction rolling pair only needs the previous j-face
still resident), so they can be interleaved in a single loop body: for
each `(j,k)`, call `iface_flow_row` and assign `dU`, then immediately
call `jface_flow_row` for the next j-face and add its difference, while
the row is still hot. This cuts `dU` from 3 touches per slab (write,
RMW, RMW) to 2.

Expected to be **arithmetically free**: same three-term accumulation
(`dU = i-diff + f_body; += j-diff; += k-diff`), same summation order,
just computed back to back instead of across two full-slab passes -- no
new reassociation beyond what section 9 already accepted, so in
principle no golden change at all (unlike section 11, which did shift
the golden by a sub-ulp amount). Untried -- needs its own A/B against the
section 9/11 baseline across the standard size ladder.

### 12.2 Precompute per-node `pm`/`mf` once instead of per face

`accum()` currently re-derives `pm(6)`/`mf(3)` from the raw nodal fields
fresh at every face call. Each interior node is a corner of up to ~4 face
calls per direction (trapezoidal face averaging shares corners between
neighbouring faces), so across the three direction sweeps a node's
`pm`/`mf` ingredients are recomputed roughly 4x per direction rather than
once. A prepass over the slab's nodal window that computes and stores
`pm`/`mf` once per node (9 values, `i` kept fastest-varying to preserve
the section 2 SIMD lesson), with the three face helpers summing
precomputed corners instead of recomputing them, would sum the same 4
addends per face in the same order -- bitwise identical in principle,
unlike section 11's inline derivation (no new floating-point operation
order, just caching an already-computed value).

Caveat carried over from when this idea was first raised: given section
10's finding that L3/DRAM traffic wasn't the bottleneck for *spatial*
tiling, this idea's benefit is more plausibly redundant *compute* (fewer
multiply/subtract ops) and fewer distinct array streams during face
assembly than raw DRAM bytes saved. Section 11's result -- a real,
size-growing win from removing a field -- partially refines that
picture and makes this more worth trying, but the compute-vs-memory-bound
question should still be checked (e.g. `perf stat` port/IPC counters)
before investing in the more invasive boundary-handling work this needs:
interior nodes precomputed in bulk, wall-adjacent faces handled as an
`O(surface)` correction pass, mirroring the pattern
`correct_cusp_kface_du` already uses for the cusp seam. Untried -- needs
its own A/B, and is more invasive than 12.1, so worth trying 12.1 first.

---

## 13. Implemented: fuse the i- and j-direction sweeps in `set_residual`

Section 12.1 was implemented (July 2026) and **adopted**: fusing the two
separate full-slab `dU` passes (the i-sweep's write and the j-sweep's
read-modify-write) into a single per-`(j,k)`-row write wins -7% to -12%
on `set_residual` gauge-corrected, consistently at every size and in both
repeat processes, with no regression and no golden regeneration.

### 13.1 Design as built

The two `do k = k0, k1` sweeps inside the slab loop became one. Per cell
row `(j,k)`: compute the i-face row into `rows` slot 1, advance the
rolling j-face pair (slots 2/3, primed with the `j=1` boundary face once
per `k`), then write `dU` **once** with both directions folded into a
single expression:

```
dU(i,j,k,m) = rows(i,m,1) - rows(i+1,m,1) + f_body(i,j,k,m)   ! i-diff + f_body
            + rows(i,m,ja) - rows(i,m,jb)                     ! + j-diff
```

This cuts `dU` from three touches per slab (i-write, j-RMW, k-RMW) to two
(fused i+j-write, k-RMW). Key properties:

- **No extra scratch.** The i-face row (slot 1) and the rolling j-face
  pair (slots 2/3) are disjoint slots that already coexist in the
  `rows(ni, 5, 3)` buffer -- today's separate sweeps just used slot 1 while
  slots 2/3 sat idle, and vice versa. The `block.tau_q_halo` carve is
  byte-for-byte identical; no new `carve_view` argument.
- **i-SIMD untouched.** The face-loop helper bodies are verbatim; the
  fused accumulate is still an inner `do i` over contiguous `dU`/`rows`,
  and the link-stage report shows it vectorizes at 32 bytes as before.
- The k-direction sweep, cusp correction, wall masking, and kb bookkeeping
  are unchanged. `kb` still only paces direction interleaving, so
  `test_residual_kb_consistent` stays **bitwise** across kb.

### 13.2 Correctness

The reassociation is confined to the final per-cell sum (the seven-term
`dU` is now assembled as `i-diff + f_body + j-diff` in one write, then
`+= k-diff`, rather than three RMWs). At -O0 this is the identical
left-to-right association as the old two-statement accumulate, so the only
source of difference is `-Ofast` codegen (FMA/vectorization) on the
reshaped loop. Measured, not assumed: `test_residual_matches_golden`
**passed without regeneration** -- max abs diff **0.000977** against a
field scale of 13420.06 (**7.28e-8 of scale**, ~0.6 ulp of the flux scale,
matching section 11), 46.8% of cells differ, comfortably inside the
golden's `rtol=1e-4` / region-scaled `atol`. `test_residual_kb_consistent`
passes unaffected. Full suite: **1414 passed, 1 skipped** (pre-existing,
unrelated).

### 13.3 A/B results

Protocol: `scripts/bench_viscous.py`, `make compile` both sides via
`git stash`, single thread pinned, production `kb = 8`, median ns/cell.
Two full repeat processes (r1/r2). Same machine as section 5.2 (Xeon
E5-2640 v3). The co-measured, **untouched** `set_visc_force` (kb=8) is the
cross-build drift gauge; deltas are reported both raw and gauge-corrected
(`set_residual` delta minus the co-measured `set_visc_force` delta at the
same size/rep).

`set_residual`, median ns/cell and gauge-corrected delta:

| size | base r1/r2 | cand r1/r2 | raw r1/r2 | gauge-corr r1/r2 |
| --- | --- | --- | --- | --- |
| 48x32x32  | 23.0 / 23.1 | 21.2 / 21.5 | -7.9% / -6.8% | -8.2% / (see note) |
| 64x48x48  | 24.6 / 25.2 | 22.6 / 22.6 | -8.3% / -10.3% | -7.8% / -9.8% |
| 80x64x64  | 30.4 / 30.6 | 27.7 / 27.6 | -8.9% / -9.9% | -8.6% / -7.8% |
| 96x96x96  | 30.4 / 29.2 | 27.6 / 27.3 | -9.0% / -6.5% | -7.3% / -4.6% |
| 128x96x96 | 26.6 / 24.9 | 23.6 / 23.1 | -11.2% / -7.5% | -11.9% / -9.5% |

`update_residual` end to end tracks `set_residual` within ~1 ns/cell at
every point in both reps (raw -6% to -16%; the larger small-size figures
are baseline process warmup, not extra kernel gain).

**Note on the r2 48x32x32 gauge.** The median gauge there was corrupted:
the baseline `set_visc_force` median spiked to 30.4 (vs its own 25.1 min)
from an interference burst, throwing the gauge to -15.3% (and
`set_tau_q_soa` to -24.7% -- impossible for unchanged kernels). Using the
interference-robust **min** ns/cell instead, that gauge is +2.3% and the
cell corrects to **-9.1%**, in line with every other cell. Min-based
gauge-corrected deltas confirm the whole table (-8.3%, -8.3%, -7.5%,
-8.7%, -16.1% for r1; -9.1%, -10.6%, -9.2%, -6.8%, -6.1% for r2), so no
regression survives at any size in either repeat.

### 13.4 Reading

- The win is the removed `dU` traffic: one fewer full read-modify-write
  of the 5-component `dU` field per slab. Unlike section 10's rejected
  j-panel tile (which tried to convert L3 re-touches to L2 and found L3
  wasn't the bottleneck), this removes a pass outright, so it helps at
  every size -- there is no cache-cliff crossover where it loses, and it
  grows slightly at the most L3-pressured size (128x96x96), the same
  signature as section 11's field removal.
- The feared arithmetic reassociation was a sub-ulp non-event (golden held
  without regeneration), and the i-SIMD survived verbatim, so this stacks
  cleanly on the section 9/11 production kernel.
- The remaining untried lever is section 12.2 (precompute per-node
  `pm`/`mf` once); it is more invasive (boundary correction pass) and its
  benefit is more plausibly redundant *compute* than memory traffic, so
  the compute-vs-memory-bound question (`perf stat` port/IPC) should be
  settled before attempting it.
- Raw data: `scripts/bench_ijfuse_ab.csv`, labels `base_r1`/`base_r2`
  (pre-fusion HEAD) and `cand_r1`/`cand_r2` (i+j fused).

---

## 14. Implemented: vectorize the coarse-grid IRS i-solve in the RK/MG path

Sections 5-13 tuned the two `update_sources`/`update_residual` kernels. The
same measure-first method applies to the **RK multigrid** path
(`solver.advance_rk_stage_mg` -> `rk_mg_irs`/`rk_mg_noirs` in `scree.f90`,
over the shared `mg_coarse_correction` engine). This change (July 2026) was
found from the compiler vectorization report and **adopted**: it wins -20% to
-27% on the isolated coarse smoother at the dominant level and -1.6% to -6.8%
gauge-corrected on the full RK stage, bitwise-identical, no golden change.

### 14.1 The finding, from the link-stage report

The coarse-level IRS (Jameson ADI, exact factored tridiagonal
`smooth_residual_tri`, `residual.f90:824`) is applied per multigrid level to
the block-restricted coarse residual, passed to `mg_coarse_correction` as the
`smoother` dummy procedure (`scree.f90`). Its three per-direction Thomas solves
have very different SIMD fates: the j- and k-solves run the recurrence along j
(resp. k) with the innermost loop over the stride-1 `i`, so they vectorize; the
**i-solve runs the recurrence along the unit-stride axis itself, so it cannot
vectorize**. The link-stage `EMBER_OPT_REPORT` build (section 4.6) confirms it
directly -- on the inlined coarse copies operating on `cres`:

```
residual.f90:864: missed: not vectorized: no vectype for stmt: ... cres_47 ...
residual.f90:867: missed: not vectorized: no vectype for stmt: ... cres_47 ...   (forward + back-sub)
residual.f90:880/885/891: optimized: loop vectorized using 32 byte vectors        (j-solve)
residual.f90:904/911:     optimized: loop vectorized using 32 byte vectors        (k-solve)
```

A transpose-tiled variant that fixes exactly this **already exists** --
`smooth_residual_tri_tiled` (`residual.f90:975`), identical maths, i-solve run
over a BJ=8-wide transposed tile of j-lines so the innermost loop is over the
independent lanes -- and is **already the production fine-grid smoother**
(`Grid.update_residual`, `grid.py:1464`). But the coarse MG-IRS path was never
switched to it. The report shows the tiled i-solve lanes vectorizing at 32
bytes (`residual.f90:1026/1031/1037/1042`).

### 14.2 The change and correctness

The two IRS wrappers `rk_mg_irs` (`scree.f90:637`) and `scree_mg_irs`
(`scree.f90:545`) now pass `smooth_residual_tri_tiled` to the engine instead of
`smooth_residual_tri` (the `external` decl and the call argument in each; four
lines, plus banner comments). Nothing else moves: both smoothers have the
identical `(dU, sf, work, ni, nj, nk)` signature and the same `2*(nib+njb+nkb)`
Thomas-work sizing the engine already carves, so the caller-side scratch
(`_mg_coarse_carve`, `solver.py:430`) is untouched. (`smooth_residual_tri` was
kept as the reference implementation at the time; it was later removed as
production-dead once the tiled variant took over -- recover from git history if
needed.)

Correctness is **bitwise-identical, no golden regeneration**: the tiled i-solve
does the same per-line Thomas recurrence in the same i-order, only the lane
grouping differs, so every coarse residual value is unchanged to the bit.
Verified: the after-swap link report shows the scalar `cres` i-solve instances
**gone** (replaced by the 32-byte-vectorized tiled lanes); `test_mg_irs`,
`test_scree_mg`, `test_residual_smoothing` and the full suite pass
(1414 passed, 1 skipped), with the residual/scree-mg goldens holding **without
regeneration**.

### 14.3 A/B results

Protocol per section 4: `make compile` both sides via `git stash` of the
`scree.f90` swap (the fine-grid `smooth_residual_tri_tiled` call is unchanged
on both sides), single core pinned, `OMP_NUM_THREADS=1`, warmup then median
ns/cell with variants interleaved round-robin, two repeat processes (r1/r2).
Harness: the rewritten `scripts/mg_irs_bench.py` (the old one referenced
removed f2py names and used mean-time; it now drives the real
`advance_rk_stage_mg` path and mirrors `bench_viscous.py`'s rigor). Sizes are
node dims; cell dims (node-1) are divisible by 8 for `n_levels = 3`.

**Isolated coarse smoother** (`smooth_residual_tri` vs `smooth_residual_tri_tiled`
called directly at each MG level's coarse shape -- a within-build comparison,
both f2py entries exist on both sides), median ns/coarse-cell:

| fine (nodes) | level 1 (fine/2)        | level 2 (fine/4)       | level 3 (fine/8)      |
| --- | --- | --- | --- |
| 49x33x33   | 22.96 -> 18.30  (-20.3%) | 23.17 -> 21.78 (-6.0%)  | 46.10 -> 53.55 (+16.1%) |
| 65x49x49   | 27.04 -> 19.83  (-26.7%) | 23.42 -> 22.60 (-3.5%)  | 27.56 -> 34.10 (+23.7%) |
| 81x65x65   | 30.00 -> 22.71  (-24.3%) | 20.24 -> 17.47 (-13.7%) | 26.26 -> 25.51 (-2.8%)  |
| 97x97x97   | 29.57 -> 22.54  (-23.8%) | 25.16 -> 19.94 (-20.7%) | 21.77 -> 22.60 (+3.8%)  |
| 129x97x97  | 31.07 -> 24.52  (-21.1%) | 27.96 -> 20.27 (-27.5%) | 23.34 -> 22.51 (-3.6%)  |

Level 1 -- the largest coarse grid, carrying the full `fac_mgrid` weight and 8x
the cells of level 2 -- is a solid **-20% to -27%** everywhere; level 2 mostly
wins; level 3 (a handful of coarse cells, ~1/64 the level-1 count) is noise and
sometimes loses to the transpose overhead, but its cost is negligible.

**Full RK stage** (`advance_rk_stage_mg`, `sf_irs = 0.5` -> `rk_mg_irs`, the
changed kernel), gauge-corrected against the co-measured `rk_mg_noirs`
(`sf_irs = 0`, structurally unchanged by the swap), median ns/cell:

| size (nodes) | base irs r1/r2 | cand irs r1/r2 | gauge-corr r1/r2 |
| --- | --- | --- | --- |
| 49x33x33   | 22.85 / 22.46 | 21.64 / 22.04 | -2.6% / -2.0% |
| 65x49x49   | 21.90 / 21.87 | 21.93 / 22.38 | -4.3% / -6.8% |
| 81x65x65   | 29.07 / 28.65 | 27.85 / 29.25 | -4.0% / -4.7% |
| 97x97x97   | 33.98 / 32.81 | 35.11 / 36.32 | -3.0% / -2.0% |
| 129x97x97  | 32.33 / 33.46 | 33.71 / 30.04 | -3.7% / -1.6% |

The **gauge is load-bearing here**: raw `irs` deltas span -10% to +11% because
the unchanged `rk_mg_noirs` gauge itself drifted up to +-12% between the two
builds (whole-program LTO re-inlining around the changed `scree.f90`, plus the
pow2-plane heap-layout lottery of section 9.4 at ni=97/129) -- exactly the
cross-build drift section 7.2 documents. Gauge-corrected (candidate `irs` delta
minus the co-measured `noirs` delta at the same size/rep), every cell in both
repeats is a clean **-1.6% to -6.8%** win with no regression.

### 14.4 Reading

- The full-stage win (-1.6% to -6.8%) is smaller than the isolated-smoother win
  (-20% to -27%) because the RK stage also does restriction, cascaded
  prolongation and the fused scatter -- all unchanged -- and only the i-solve
  (one of three directions) was scalar. The smoothing is a real fraction of the
  stage, so vectorizing its slowest third is worth a few percent end to end,
  for free.
- This was pure oversight recovery: the fix already existed and was in
  production on the fine grid; only the coarse dispatch lagged. The vectorization
  report is what surfaced it -- the coarse smoother's scalar i-solve is invisible
  in a timing number alone.
- No cache-cliff crossover: the win is compute-side (SIMD on the recurrence
  lanes), so it holds across the size ladder rather than appearing only past L3.
- Build note: gfortran 13.3's `-Werror=line-truncation` rejects one pre-existing
  133-char line in `viscous.f90:759` (over the 132-col free-form limit); wrapped
  onto a continuation line (arithmetically identical) so the whole file set
  builds under the production flags. Unrelated to the optimization.
- Raw data: `scripts/bench_mgirs_ab.csv`, labels `base_r1`/`base_r2` (untiled
  coarse smoother) and `cand_r1`/`cand_r2` (tiled).

---

## 15. Implemented: fuse the final prolongation hop with the RK scatter

Section 14 fixed the last vectorization miss in the RK multigrid path. A fresh
link-stage report (section 4.6) on that build confirms **every remaining hot
loop already vectorizes at 32 bytes** -- the restriction over coarse-i
(`scree.f90:418/453/475`), the final prolong-fine write (`:223`), and the
cell->node scatter interior (`distribute.f90:58`). With no SIMD lever left, the
only remaining inefficiency is memory traffic. This change (July 2026) removes
the largest removable piece and was **adopted**: -11% to -30% on the full RK
stage.

### 15.1 The finding

Per RK MG stage, the increment is written once and read once, back to back:
`mg_coarse_correction`'s final hop `mg_prolong2x_fine` (`scree.f90:183`) writes
the full-volume 5-component cell increment `tmp`, and the wrapper's scatter
`cell_to_node_generic` (`distribute.f90:32`), `cons = snapshot +
cell_to_node(tmp)`, immediately re-reads all of it. That ~40 B/cell write+read
round-trip is exactly the class the viscous k-rolling fusion (section 7) and
`set_residual` k-plane carry (sections 6/9) removed.

### 15.2 The change

New `mg_prolong2x_fine_scatter` (`scree.f90`): it builds the increment one fine
k-plane at a time into a rolling two-plane buffer `rbuf(ni-1,nj-1,5,2)` and
scatters each finished node plane straight into `cons`, so the increment is
never materialised full-volume. Cell plane `kc` feeds node plane `kc` (as the
k-upper plane) and, at the ends, the two k-boundary node planes; the per-node
weighting (interior 1/8 of 8 cells, i/j faces 1/4 of 4, edges 1/2 of 2, corners
1 of 1) and the term order mirror `cell_to_node_generic` exactly. The final hop
was lifted out of the shared engine `mg_coarse_correction` (which now leaves the
coarse correction in `acc0`); the **RK** wrappers `rk_mg_irs`/`rk_mg_noirs` call
the fused routine, while the **scree** wrappers keep the unfused
`mg_prolong2x_fine` + `scree_roll_and_scatter` tail (its scatter is in-place and
rolls the Denton history). The caller `advance_rk_stage_mg` carves the rolling
`rbuf` from `block.scratch` in place of the full `tmp` for the MG path.

The interior scatter loop stays i-innermost and branch-free, so it vectorizes at
32 bytes exactly as `cell_to_node_generic` did (link report confirms).

### 15.3 Correctness

Not bitwise: splitting the increment production into rolling planes and summing
the node average in the fused loop lets `-Ofast` pick different FMA/vector
codegen, a bounded reassociation. Quantified standalone (fused vs
`mg_prolong2x_fine` + `cell_to_node_generic`) over 11 shapes incl.
ni/nj/nk = 2 boundaries: max abs diff **2.4e-7 on O(1) data (~1-2 ulp)**, one
shape exactly bitwise -- a real bug would not be bitwise for any shape. The full
suite passes with **no golden regeneration** (the RK-path shift stays inside the
existing tolerances). Two test edits: the `rk_mg_*` scratch argument `tmp`
became `rbuf` (`tests/test_mg_irs.py`), and
`test_rk_plain_matches_mg_at_fac_mgrid_zero` relaxed from byte-equality to
`allclose` (rk_plain still scatters unfused, so the two now agree to ~1 ulp
rather than exactly). `test_scree_mg` is unaffected (scree keeps the old path).

### 15.4 A/B results

Protocol per section 4: `make compile` both sides via `git stash` of the fusion
(`scree.f90` + `solver.py` + the test), single core pinned, warmup, round-robin
interleave, median ns/cell, two repeat processes (r1/r2). The fusion changes
**both** `rk_mg_irs` and `rk_mg_noirs`, so neither can be the gauge; the
co-measured drift gauge is `rk_plain` (`fac_mgrid=0`, multigrid off), which the
fusion leaves untouched.

`advance_rk_stage_mg`, `rk_mg_irs` (sf_irs=0.5), median ns/cell:

| size (nodes) | base r1/r2 | cand r1/r2 | raw r1/r2 | gauge-corr r1/r2 |
| --- | --- | --- | --- | --- |
| 49x33x33  | 14.76/14.42 | 13.16/12.81 | -10.8% / -11.2% | -11.3% / -8.6% |
| 65x49x49  | 14.63/14.92 | 11.73/11.51 | -19.8% / -22.8% | -26.5% / -25.6% |
| 81x65x65  | 14.26/14.54 | 10.93/10.81 | -23.4% / -25.7% | -21.9% / -23.4% |
| 97x97x97  | 13.53/14.69 | 10.98/10.78 | -18.9% / -26.6% | -26.2% / -31.2% |
| 129x97x97 | 13.48/13.75 | 11.39/11.46 | -15.5% / -16.7% | -33.6% / -31.7% |

`rk_mg_noirs` tracks a touch larger (raw -13% to -30%). Min ns/cell
(interference-robust) agrees: irs -12% to -23%, noirs -14% to -28%, every
size, both repeats.

### 15.5 Reading

- The win is the removed `tmp` round-trip: two fewer full-volume passes over the
  5-component increment per stage. It holds at every size (no cache-cliff
  crossover) because the report already showed the path SIMD-clean -- this is a
  pure traffic cut, not a vectorization fix.
- **The gauge drifted against the change.** `rk_plain` was up to ~+18% slower in
  the fused build (whole-program LTO re-inlining around the enlarged
  `scree.f90`), so the raw deltas already understate the kernel win;
  gauge-correcting widens it. The **raw** figure (-11% to -30%) is the
  conservative floor and is itself a clear, consistent win, so adoption does not
  rest on the noisy gauge correction.
- scree was deliberately left on the unfused path (its in-place, history-rolling
  scatter does not fit the RK snapshot+scatter shape); revisit only with its own
  A/B.
- Raw data: `scripts/bench_mgfuse_ab.csv`, labels `base_r1`/`base_r2` (unfused)
  and `cand_r1`/`cand_r2` (fused); `plain` variant rows are the rk_plain gauge.
