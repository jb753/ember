# Plan: collapse the IRS smoother's memory traffic

## Goal

Cut the memory traffic of implicit residual smoothing and its immediate
neighbour (`set_residual`'s change-limiter scaling pass) by ~3x, without
changing a single bit of the result.

`smooth_residual_tri_tiled` (`src/ember/_fortran/residual.f90:901`) is three
sequential ADI direction solves, each in place on the full
`dU(nci,ncj,nck,5)`. Counting full-array touches:

| phase | reads | writes |
| --- | --- | --- |
| i-solve (gather -> tile -> scatter) | 1 | 1 |
| j-solve (forward, back-substitution) | 2 | 2 |
| k-solve (forward, back-substitution) | 2 | 2 |
| **IRS total** | **5** | **5** |
| `scale_du` (`residual.f90:749`, when `dampin > 0`) | 1 | 1 |
| **post-sweep total** | **6** | **6** |

At the harness case (273x65x57, 974,848 cells) `dU` is 19.5 MB, so one IRS
call moves ~195 MB and the post-sweep total is ~234 MB. The arithmetic is
~3 flops per element per direction (one FMA + one multiply forward, one FMA
back), so ~9 flops per element against 40 bytes moved: **~0.2 flop/byte**.

Nothing in the loop bodies is the problem. The link-stage `-fopt-info-vec-all`
report is clean -- every innermost loop vectorises at 32 bytes. The pass
structure is the problem, and it has never had the treatment every other hot
kernel in this file has had (k-slab blocking, rolling-buffer fusion, merged
traversals). The only IRS tuning on record is `BJ`, which is a within-L1 fix
to the i-solve, not a traffic fix.

**Target: 6R+6W -> 2R+2W**, bitwise identical.

> **Amended after Phase 0's measurement.** The traffic argument above is
> sound but it is *not the whole cost*, and it does not rank the three
> direction solves. Measured, the i-solve takes 53.6% of the time for 20% of
> the traffic, because its transpose gather/scatter compiles to scalar moves.
> Read "Rule 8 check" in Phase 0 and "Revised priorities" before the design
> sections below -- the phase ordering changed, though no phase was dropped.

## Success criteria

- Bitwise-identical `dU` against the current kernel, at every phase, on the
  `swirl()` state. This is not a tolerance to be negotiated -- see
  "Why this is bitwise" below. Any deviation is a bug in the new code.
- No golden regeneration, no convergence re-verification, no numerics review.
- A measured win on the isolated smoother at 1M cells that survives the size
  ladder, in the paired-within-launch statistic.
- `tests/` green, including `test_mg_irs.py` and `test_scree_mg.py`.

## Why this is bitwise (and why that is the point)

Every restructuring below reorders *traversal*, never *arithmetic*:

- **Blocking the j-solve over i.** For a fixed `(i,k,m)`, the j-recurrence
  reads and writes only that j-line. Lines at different `i` are completely
  independent. Restricting the loop to a strip of `i` changes the order in
  which independent lines are solved and nothing else.
- **Blocking the k-solve over i.** Same argument, one axis over.
- **Fusing the j- and k-solves inside one strip.** The k-solve for column
  `(i,j)` consumes the j-solve's output at that column. Inside a strip, every
  column the k-solve needs has already been j-solved. The dependency is
  respected exactly; only the interleaving changes.
- **Fusing the i-solve into `scale_du`.** `scale_du` is pointwise. The i-solve
  for row `(j,k,m)` depends only on that row's scaled values. Doing them
  back-to-back per row is the same computation on the same operands.

There is no reassociation anywhere, unlike the fusions already adopted for
`set_residual` (which carry a documented ~1.2 ulp tolerance). That makes the
correctness gate exact equality, which is a far stronger gate than any of the
prior kernel work in this directory could use.

**The one ordering fact to preserve:** production today runs
sweep -> cusp correction -> `scale_du` -> IRS-i -> IRS-j -> IRS-k. The cusp
correction (`residual.f90:715`) modifies `dU` on the two seam planes *after*
the sweep and *before* `scale_du`, and the change limiter runs before IRS (the
deliberate reordering documented at `residual.f90:566`, worth ~19% of the
field scale at `sf=1, dampin=25`). The design below preserves that exact
order. Nothing here re-opens that question.

## Design

### Pass structure

Two passes over `dU` replace six:

```
PASS 1  (full volume, per component, per BJ-block of j-lines)
        scale (change limiter)  (x)  i-direction Thomas solve
        -> 1R + 1W

PASS 2  (per i-strip of width W, per component: full j and k extent)
        j-direction Thomas solve (forward + back)
        k-direction Thomas solve (forward + back)
        -> 1R + 1W, with all four sub-passes inside an L2-resident block
```

Pass 1 already visits every element to apply the limiter, so the i-solve rides
along free: the tile gather *is* the scale's read, and the tile scatter *is*
its write. `scale_du`'s separate round-trip disappears entirely.

Pass 2's block is `W x ncj x nck` for one component. At `W=32` on the harness
case that is 32*64*56*4 = 458 KB -- comfortably L2-resident on both this
machine's P-cores (2 MB) and Sapphire Rapids (2 MB). The four sub-passes
(j forward, j back, k forward, k back) then hit L2, not DRAM.

`W` is a tuning parameter to sweep, like `BJ` and `_KB_SLAB` before it. Start
at 32 and sweep 16/32/64/128 on the real build.

### Two consumers, two deliverables

`smooth_residual_tri_tiled` has two callers, and they need different amounts
of this:

- **Fine grid** -- `Grid.update_residual` (`grid.py:1597`), immediately after
  `set_residual`. Gets the full design: pass 1 fused into `set_residual`,
  pass 2 as a new standalone kernel.
- **Coarse grid** -- `scree.f90`'s `mg_coarse_correction`
  (`scree.f90:596,652`), which gathers a coarse residual `cres` and smooths
  it. There is no `set_residual` there and no limiter to fuse into, so it
  keeps a standalone three-direction smoother.

Phases 1 and 2 both live entirely inside the standalone smoother and therefore
benefit **both** consumers with no interface change anywhere. Only Phase 3
fuses into `set_residual`, and it benefits the fine path only. So the first
two phases carry most of the win and nearly all of the safety; do them first
and independently, and leave the interface alone until they have landed.

### On "one code path" -- `sf = 0` and `dampin = 0`

Correct that we do not need a duplicated kernel: one `set_residual`, one
smoother, no `set_residual_irs` twin. But `sf = 0` is a no-op in *arithmetic*,
not in *cost*, and the distinction matters here more than usual:

- `tri_coeffs` at `sf = 0` gives `minv = 1`, `cp = 0`, so the solve is exactly
  the identity -- but an unguarded fused kernel would still pay the tile
  gather, the three tile passes and the scatter to compute it.
- **`sf_resid` defaults to `0.0`** (`solver.py:330`). IRS-off is the common
  case, not the rare one. Making it pay for a smoother it disabled would be a
  regression on the default configuration.

So: one code path, with the `sf > 0` and `dampin > 0` tests **hoisted to phase
level** -- once per kernel call, outside every loop -- rather than eliminated.
Four combinations of the two flags, two hoisted booleans, no duplicated loop
bodies (the bodies live in contained subroutines called from a common driver).

The one place a branch could leak into an inner loop is pass 1, which is
"scale then i-solve" when `dampin > 0` and "i-solve alone" when it is not.
Options, in preference order:

1. `if (do_damp)` around the scale statement inside the gather loop, relying
   on loop unswitching to hoist it. Check the opt report; if the loop still
   vectorises, this is free and keeps one body.
2. Two contained subroutines for the gather (with and without the scale),
   sharing the rest. Small duplication, no branch.

Do **not** make it branch-free by folding `dampin` into a reciprocal: the
existing code keeps `fdamp/dampin` as a division specifically so the
arithmetic matches production exactly (`residual.f90:855`), and a reciprocal
multiply would break the bitwise property this whole plan rests on.

**Countervailing house style, stated so the choice is deliberate:** `scree.f90`
dispatches `mg_smooth_noop` vs the real smoother as a dummy-procedure argument
precisely to avoid a runtime `sf` test -- "so the smoothing step is
structurally absent (no `sf_irs<=0` test, no `tri_coeffs` call), rather than
relying on `smooth_residual_tri`'s internal guard" (`scree.f90:72`). That
precedent stands where it is, because there the smoother call sits inside a
per-level loop. A hoisted guard evaluated once per `set_residual` call is a
different situation and a plain `if` is the right tool.

## Phases

Each phase is independently measurable, independently revertible, and
bitwise-gated before it is timed.

### Phase 0 -- land the alias-versioning fix, and check the premise

- **Land fix 1** (measured: -1.95% +/- 0.81%, faster in 14/15 launches,
  bitwise). Hoist the four alias-versioned j/k recurrence loops
  (`residual.f90:998,1004,1024,1032`) into contained subroutines taking
  separate dummies. This is the `scale_du` trick, it removes the runtime
  overlap test GCC currently emits, and it collapses the k-solve's `(j,i)`
  nest into one flat vector loop. Already prototyped and gated as the `irsna`
  arm (`bench/subroutines/residual_irs_noalias.f90`).
  Do this first so Phase 1's number is measured against a clean baseline.
- **Fix 2 (tile in `work`) is rejected**: +0.50% +/- 0.53%, faster in 2/5
  launches -- a coin flip -- against six edit sites across `residual.f90`,
  `scree.f90`, `grid.py` and `solver.py`, and a new cross-language `BJ`
  invariant whose drift mode is a silent out-of-bounds write into live
  scratch. Recorded in the history section of `bench/README.md`, not built.
- **Rule 8 check -- DONE, and it FAILED the traffic model.** Measured with
  `bench/subroutines/residual_irs_dirs.f90` (per-direction switches, updated
  to track production's `BJ` and line-kernel structure), 6 launches, 2 P-cores,
  1M cells. The control is sound: `irsijk` (all three directions, through the
  switches) matches `irs` at -0.45% +/- 0.78%, 3/6 launches, so the switches
  themselves cost nothing. The three directions sum to 100.9% of the whole,
  so the split is additive and real:

  | direction | ns/cell | share of time | share of traffic |
  | --- | --- | --- | --- |
  | i-solve | 4.629 | **53.6%** | 20% |
  | j-solve | 1.751 | 20.3% | 40% |
  | k-solve | 2.330 | 27.0% | 40% |

  **The i-solve costs 2.7x more per byte moved than the j-solve.** The kernel
  is not simply bandwidth-bound and the traffic model does not rank these
  phases. Cause confirmed by disassembly rather than inferred: the compiled
  `smooth_residual_tri_tiled_` contains **266 scalar `vmovss` moves** (plus
  `vinsertps`/`vinsertf128` to assemble vectors) against 190 vector `vmovups`,
  and **no gather/scatter instructions at all**. The three tile passes
  vectorise, but the transpose gather and scatter that feed them are done one
  element at a time, strided across `dU` at `nci*4 = 1088` bytes. Per
  `(j0-block, k, m)` that is ~2*nci*nb = 17,400 scalar moves against ~3,300
  vector ops of actual solve. This is precisely the failure mode the Gate 1
  note warns about -- "a 'vectorized' loop can still be gather/scatter-based
  rather than unit-stride; the report doesn't distinguish them".

  **Consequence: the phases below are re-ordered.** See "Revised priorities".

### Revised priorities (after the Phase 0 measurement)

The original ordering assumed traffic ranked the phases. It does not. Revised,
with each phase's share of the measured time attached:

| new order | phase | targets | share of IRS time |
| --- | --- | --- | --- |
| 1 | **vectorise the i-solve transpose** (new) | 266 scalar moves | **53.6%** |
| 2 | fuse j+k into i-strips (was Phase 1) | 4R+4W -> 1R+1W | 47.3% |
| 3 | fold i-solve into the scale pass (was Phase 2) | 1R+1W | part of the 53.6% |

Phase 2 keeps its full modelled value -- j and k really are traffic-shaped, and
cutting 4R+4W to 1R+1W is still the largest single traffic reduction available.
But it can only ever address 47% of the kernel, so it is no longer the headline.

Phase 3 is worth less than originally modelled: it removes one of the i-solve's
two full-array touches, but the i-solve's cost is dominated by the transpose,
not by that traffic. Do it after Phase 1, and re-price it then -- if Phase 1
makes the transpose cheap, Phase 3's remaining traffic saving becomes the
dominant term again and its value goes back up.

### Phase 1 (new) -- vectorise the i-solve transpose

Replace the element-at-a-time gather and scatter with a blocked transpose:
read 8 contiguous runs of 8 floats from `dU` (unit-stride vector loads),
transpose the 8x8 block, write it into the tile. Same for the scatter in
reverse.

- Bitwise: a transpose moves values, it does not compute with them.
- No interface change; entirely inside `smooth_residual_tri_tiled`. Benefits
  both consumers, like the old Phase 1.
- The staging buffer must be a small fixed-size `(8,8)` local so it stays in
  registers/L1 -- the point is to move the strided access off `dU` and onto a
  block the compiler can keep hot, not merely to move it elsewhere.
- **Gate on the disassembly, not the opt report.** The success criterion is
  the scalar `vmovss` count collapsing and unit-stride `vmovups` rising;
  `-fopt-info-vec-all` said "loop vectorized" about the current code and was
  useless here. Use `codegen_gauge.py`'s `mix` output plus a direct
  `objdump -d` count.
- If gfortran will not emit shuffles for an 8x8 block transpose even fully
  unrolled, the fallback is to accept scalar moves but make them hit an
  L1-resident staging block rather than `dU` strided at 1088 bytes -- most of
  the cost is the strided access pattern, not the scalar-ness as such.
- Re-run the per-direction split afterwards. If the i-solve drops toward its
  20% traffic share, the transpose was the whole story and Phase 2's modelled
  value can be trusted for what remains.

**RESULT: adopted.** -8.27% +/- 0.29%, faster in 8/8 launches, bitwise
identical, no interface change. `TB` is steeply optimal at the vector lane
count -- TB=4 is +43.8% and TB=16 is +9.0% against TB=8 -- which confirms the
mechanism is the register-level shuffle network, not cache blocking. Landed in
`residual.f90`; `bench/subroutines/residual_irs_{dirs,transpose}.f90` track it.

Note the static disassembly gate was **inconclusive** and the timing decided
it: scalar `vmovss` went *up* (266 -> 550), because the four remainder loops in
each of gather/scatter contribute instructions that never execute at sizes
where `nci` and `nb` are multiples of `TB`. Static instruction counts cannot
distinguish hot from cold code; use them to form a hypothesis, not to settle
one.

**Re-measured split (6 launches, control `irsijk` = `irs` at -0.16%, 3/6):**

| direction | ns/cell (was) | share of time (was) | share of traffic |
| --- | --- | --- | --- |
| i-solve | 3.794 (4.629) | 48.4% (53.6%) | 20% |
| j-solve | 1.815 (1.751) | 23.1% (20.3%) | 40% |
| k-solve | 2.408 (2.330) | 30.7% (27.0%) | 40% |

The i-solve fell 18%, but at 48.4% of the time for 20% of the traffic it is
**still 2.4x the j-solve's cost per byte** (was 2.7x). So the transpose was
about a third of the excess, not all of it. The most likely remainder is the
tile recurrence's inner loop length: it runs over `nb = BJ = 32` lanes, i.e.
only 4 AVX2 iterations per call against the j-solve's 272 and the k-solve's
17,408, so loop control and recurrence latency are amortised over far less
work. That is a `BJ` question -- and `BJ` trades inner-loop length against the
L1 residency it was tuned for (34 KB at BJ=32, 68 KB at BJ=64). Re-sweeping it
is explicitly out of scope here and belongs on the production target.

**Consequence for Phase 2:** j and k are now 53.8% of the time and move 80% of
the traffic, making them the largest remaining item. Phase 2 proceeds as
planned.

### Phase 2 -- fuse j+k into i-strips (both consumers)

Restructure `smooth_residual_tri_tiled` into two passes: i-solve as today,
then a fused j+k solve blocked over i-strips. **5R+5W -> 2R+2W.**

- No interface change. `work` is unchanged (the Thomas coefficients are the
  same six vectors). `scree.f90` and `grid.py` are untouched.
- Gate: bitwise against the current kernel on `swirl()`, at 100k/300k/1M/2M.
- Sweep `W` in 16/32/64/128 on the real build.
- Measure via the `irs` arm set now in the harness.

**Risk: strip contiguity.** An i-strip of width `W` is `W*4` contiguous bytes
per `(j,k)`, with `nci*4` bytes between j-rows. At `W=16` that is exactly one
64-byte line; at `W=32`, two. But `nci` is not generally a multiple of 16
(the harness case has `nci=272`, so rows are 1088 bytes = 17 lines), so
successive rows' strips are not line-aligned relative to each other and a
strip will straddle an extra line. Expect to lose some of the modelled saving.
Mitigations, in order: sweep `W`; try aligning strips to line boundaries by
starting the strip ladder at an offset; if it still disappoints, fall back to
the intermediate design below.

**RESULT: adopted, with a caveat on its attribution.** `W` is steeply optimal
at 64 (917 KB block, 8 AVX2 iterations per line): W=32 -5.6%, **W=64 -11.0%**,
W=128 (1.83 MB, past L2) -1.2%, W=256 -2.5%. Bitwise, 8/8 launches, no
interface change. Landed in `residual.f90`.

**The modelled 4R+4W -> 1R+1W overstated the target.** Production's j-solve
back-substitution re-reads what its forward pass just wrote out of one
(nci,ncj) plane -- 70 KB, L2-resident -- not out of DRAM, and the k-solve gets
the same plane-to-plane reuse. So each solve already cost ~1R+1W of DRAM and
fusing them saves one of the two, ~40 bytes/cell. The traffic table at the top
of this plan counts *touches*, not DRAM traffic, and the two diverge wherever
a pass fits in cache. Worth remembering before modelling the next kernel.

**Rule 7 flag, and its resolution: IDENTICAL-CODE FOLDING was corrupting the
comparison.** Phase 2 first measured -11.0% +/- 0.39% (8/8) as an arm against a
Phase-1 production, then only -5.0% once the fusion was also in production.
Two tight 8-launch paired measurements disagreeing far outside their error
bars, so: between-build, not noise. Rule 9's own tool found it in one call --
production's `smooth_residual_tri_tiled_` fingerprinted as **13 instructions
over 3 functions**, and the disassembly was:

```
smooth_residual_tri_tiled_:
    jmp    smooth_residual_tri_jk_@plt
```

Once the arm was promoted, its source was byte-identical to production, so GCC
folded the two functions and left production a tail-call thunk. Two failures at
once:

- **The Rule 9 gate passed on a stub.** Any fingerprint comparison of that
  symbol was hashing three instructions.
- **The thunk biased the timing by ~8%** on byte-identical source: production
  measured 7.543 ns/cell folded-and-thunked versus 6.932 standalone.

Rebuilt without the source-identical arm (`residual_irs_jk.f90` deleted --
production *is* that arm now, and `irstr` still covers the unfused-j/k state
for A/B), production is a real 3502-instruction function again. `irstr`
fingerprints identically across both builds, so it is a valid fixed reference.

**`codegen_gauge.py` now detects thunks and warns**, since this trap is
invisible otherwise and will recur every time an adopted arm is left in the
build. **General rule: delete a bench arm once it is promoted, or the two fold
together and every measurement involving either becomes suspect.**

**Caveat on the `TB` and `W` sweeps above.** Both ran with an arm in the build
whose source matched production at the time (`irsna` during the `TB` sweep,
`irstr` during the `W` sweep), so production was very likely thunked in those
runs too and their *absolute* "vs prod" percentages are not trustworthy. The
*rankings* are: the baseline carried the same bias at every `TB` and every `W`,
while the arm under test changed, so TB=8 and W=64 remain the right constants.
Re-run either sweep on a clean build before quoting a number from it.

**Corrected cumulative -- one build, no folding, drift-paired, 8 launches at
1M on 2 P-cores:**

| state | ns/cell | vs production |
| --- | --- | --- |
| Phase 0 baseline (`irsna`, alias fix only) | 8.675 | +25.38% +/- 0.49%, 0/8 |
| after Phase 1 (`irstr`) | 7.880 | +13.84% +/- 0.38%, 0/8 |
| **production now** | **6.932** | -- |

Phase 1 **-9.2%**, Phase 2 **-12.2%**, together **-20.2%** against the Phase-0
baseline, or **~-22%** against the original kernel with the alias fix included.
The first Phase 2 measurement (-11.0%) was much closer to the truth than the
contaminated re-measurement (-5.0%) that appeared to contradict it.

**Fallback if the strip design underperforms:** block each direction
independently -- `(i,k)` tiles for the j-solve, `(i,j)` tiles for the k-solve
-- so forward and back-substitution share one cache residency without any
cross-direction fusion. That is **5R+5W -> 3R+3W**, a strict subset of the
above, and it keeps the access pattern much closer to today's.

### Phase 3 -- fuse the i-solve into `set_residual`'s scale pass (fine path)

Fold pass 1 into `set_residual`, so the change limiter's scaling and the
i-direction solve share one traversal. **2R+2W -> 1R+1W** downstream of the
residual sweep.

- `set_residual` gains `sf` and an IRS `work` buffer; `Grid.update_residual`
  stops calling the i-solve separately and calls the Phase 1 j+k kernel
  afterwards. The standalone three-direction smoother stays for `scree.f90`.
- `scale_du`'s traversal is currently `k, j, m, i`; it becomes
  `m, k, j0-block-of-BJ, ...` so the i-solve's transpose tile can be filled
  as the scale is applied.
- Ordering is preserved exactly: sweep -> cusp -> scale -> i -> j -> k.
- Gate: bitwise against Phase 1's output, at both `dampin = 0` and
  `dampin = 2`, and at both `sf = 0` and `sf > 0` (four combinations -- this
  is where the hoisted-guard design gets tested).
- Verify `sf = 0, dampin = 0` costs nothing it did not cost before: fingerprint
  `set_residual` and confirm the IRS-off path is unchanged, or measure it
  directly.

**RESULT: adopted, but built differently from the design above.** The plan had
`set_residual` gain `sf` and a work buffer and do the fused pass itself. That
would have put the fusion inside the hottest kernel in the code for a benefit
that only exists when IRS is on -- and **every current entry point defaults
`sf_resid=0.0`** (`tools/run_duct.py`, `tools/run_throttle.py`,
`SolverConfig`); only `pgo_train.py` enables it, deliberately, to exercise it.
(`residual.f90` cited "run.py defaults sf_resid=1.0, dampin=25"; `run.py` no
longer exists, so that comment was stale.)

Built instead so the fusion lives in the SMOOTHER, not in `set_residual`:

- `set_residual` gains one `intent(out) ravg(5)` -- the block means it already
  accumulates during its sweep -- and nothing else. Its hot sweep is untouched.
- `grid.update_residual` passes `dampin=0` when `sf > 0`, which suppresses the
  scaling pass but not the reduction, and hands `ravg` to the new
  `smooth_residual_scale_tri`, which applies the limiter inside its i-solve
  gather.
- The standalone `smooth_residual_tri_tiled` is unchanged for `scree.f90`.
- Every IRS primitive moved into `residual_helpers` so the two smoothers share
  one copy. That is also the structural fix for the folding hazard above: code
  that exists once cannot be folded with itself.

**-5.02% +/- 0.44%, faster in 8/8 launches** on the combined `set_residual`+IRS
path (19.882 -> 18.903 ns/cell), bitwise.

**The default path is unaffected**, which was the risk worth checking:
`set_residual` alone measures 12.671 -> 12.646 ns/cell (-0.2%, inside noise)
and grows by 6 instructions (13034 -> 13040), the five divides and the store.

**The traffic model overestimated a third time.** One full-volume read/write
pair at ~22 GB/s should have been ~1.8 ns/cell; the measured saving is
0.98 ns/cell, about half. Three for three now -- treat a DRAM-traffic estimate
on this code as an upper bound of roughly twice the achievable saving.

**Coverage gap closed.** `residual.f90` recorded that no test drove
`update_residual` with `dampin` set and `sf > 0` together -- exactly the
combination this changes. `tests/test_irs_fused_damp.py` now gates all four
(`dampin`, `sf`) combinations bitwise against the unfused sequence, plus a
guard that both flags actually alter the result so the gate cannot pass
vacuously.

### Phase 4 -- adopt, document, or revert

Per phase: adopt if it wins at every size with no regression; record the
number and the reasoning in `bench/README.md`'s history section either way.
A rejected arm with a measured number is a result worth keeping -- most of
that section is rejections.

## Out of scope

- **Scattering the final k back-substitution into the march** (the shape of
  the already-adopted `mg_prolong2x_fine_scatter` win, -11% to -30%). A real
  further 1R+1W, but it needs a call-site restructure across `grid.py` and the
  march kernel. Price it after Phase 2, as its own plan.
- **Replacing the recurrences with a truncated FIR.** The constant-coefficient
  Neumann Toeplitz inverse decays geometrically, so ~8 taps would reach
  float32 and all three directions would become blockable in one pass. Not
  bitwise, needs a real error analysis in ulps, needs boundary care. Only
  worth building if Phases 1-2 disappoint.
- **Re-sweeping `BJ`.** Hardware-tuned constant, already documented as needing
  a re-sweep per target. Orthogonal to this work.
- Anything that reopens the damp-vs-IRS ordering question.

## Measurement protocol on this machine

The harness's protocol was written for a homogeneous Haswell workstation.
This box is a Core Ultra 5 135U (Meteor Lake) and needs three adjustments,
already made:

- **Hybrid cores.** CPUs 0-3 are two P-cores plus SMT siblings (4.4 GHz),
  4-11 are eight E-cores (3.6 GHz), 12-13 are two LP E-cores on the SoC tile
  (2.1 GHz). Consecutive CPU ids span core classes, so "median across ranks"
  would be the median of a multi-modal distribution. `run_all_arms.sh` now
  takes an explicit `CPUS` list; use `CPUS="0 2"` (the two physical P-cores,
  siblings idle) or `CPUS="4 5 6 7 8 9 10 11"` for a homogeneous contended
  regime. `NRANKS=16` cannot run here at all -- CPUs 14-15 do not exist.
- **Thermal drift.** A 15-launch run drifted ~8% between launch 5 and launch
  8, putting 4.3% of stdev on the unpaired medians and burying a 2% effect
  present in 14 of 15 individual launches. `analyze_multi` now also reports
  the **paired-within-launch** difference, which cancels drift because
  `run_all_arms.sh` is launch-outer/arm-inner. Quote the paired number for
  differences; quote the unpaired one for absolute spread.
- **`uv run` re-syncs by default**, rebuilding the extension without
  `EMBER_BENCH_KERNELS` and silently dropping every arm between the build you
  fingerprinted and the run you timed. The driver scripts now set
  `UV_NO_SYNC=1`, and `bench_prod_baseline.py` fails loudly on a missing arm
  instead of raising `KeyError` inside one rank while its peers hang.

**Threat to validity, stated plainly:** every number produced here is
gfortran on a mobile Meteor Lake part. The harness history is gfortran on a
Haswell workstation. Production is ifort on Sapphire Rapids. These are three
different machines and results do not transfer between them. A ~2x traffic
reduction should survive any of them -- that is the point of choosing a
traffic-bound target -- but the single-digit-percent tuning results (`W`, the
strip alignment) must be re-swept on the production target before they are
trusted there.
