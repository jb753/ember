# Plan: attributing section 26's win (`nodal` and `tbaos` arms)

> **Status: this plan's run completed and its outcome is superseded.** The
> `viscous_kernels.md`/`kernel_benchmark_methodology.md` sections this plan
> refers to ("section 20", "section 26", "methodology section N") have been
> replaced by `bench/README.md`, whose "corrected instrument" table gives the
> final numbers for every arm named here (`nodal` +11.3%, `tbaos` +98.6%
> vs `prod`, both on the corrected harness -- production wins outright). The
> harness this plan drives (`tools/bench_residual_staged.py`,
> `tools/run_residual_staged.sh`) was retired along with it; use
> `bench/residual_arms.py` (Gate-2 correctness) and `bench/run_all_arms.sh` /
> `bench/run_prod_baseline.sh` (timing) instead, per `bench/README.md`. This
> document is kept for the attribution reasoning (section 1) and the
> pre-registered predictions (section 5), which are still a useful worked
> example of the methodology; its own run commands below are historical and
> point at tools that no longer exist.

The first intermediate step towards acting on section 26. Section 26.5 names
it: production's **single fused sweep reading the nodal primitives** -- undo
section 20 only, keep everything else. That is the `nodal` arm. A second arm,
`tbaos`, was added before running the ladder so that the four kernels form an
incremental chain rather than two isolated points (section 1.1). This is the
build and measurement plan for both.

**Scope of this document.** gfortran local only. This machine (Xeon E5-2640
v3, Haswell, gfortran 14.2) is not in a SLURM cluster and no ifort/Sapphire
run is planned here. That bounds what the experiment can conclude -- see
section 6 -- and the bound is the main thing to keep straight while running
it.

Protocol throughout: `bench/README.md` (formerly `kernel_benchmark_methodology.md`).
Rules cited by number are from that document.

---

## 1. The question

Section 26's `multall` arm beats production by 13-50% and bundles **three**
changes:

1. nodal primitives staged/read instead of derived per face corner;
2. five passes instead of one fused sweep;
3. SoA face-area geometry instead of component-first `dA(3,i,j,k)`.

Section 26.4 attributes the win to divides, i.e. entirely to (1). That is a
mechanism claim with no attribution control behind it. The `nodal` arm is that
control: it applies (1) alone, to production's own fused sweep.

Why this one first, of the three:

- it is the only one of the three that is a small edit to production rather
  than a rewrite -- (2) inverts the fusion decisions of sections 8/13, (3)
  touches `geometry.f90`, `block.py` and every consumer of `dA*_nd`;
- section 26.4's stated mechanism predicts it carries most of the effect, so
  it is the cheapest test of the loudest claim;
- section 20 measured its exact inverse on the production target, so there is
  a real prior to score against (section 5).

### 1.1 A second arm, so the ladder is a chain

`nodal` alone leaves the other two changes bundled: the residual gap
`multall - nodal` is "five passes + SoA geometry" with no way to split it. One
more arm fixes that. `tbaos` (`residual_multall_aos.f90`) is `multall`'s design
on ember's own component-first `dA(3,i,j,k)`, which makes the ladder an
incremental chain where every adjacent pair differs by exactly one thing:

| step | isolates |
| --- | --- |
| `prod` -> `nodal` | nodal primitives instead of derived-per-corner |
| `nodal` -> `tbaos` | the five-pass split, with the per-node staging it forces |
| `tbaos` -> `multall` | SoA face-area geometry instead of AoS |

A third arm, `prodsoa`, closes the one remaining hole (section 1.2).

The middle step bundles the split with staging on purpose: Rule 1 says
co-optimised choices move together, and a five-pass design that re-derives
primitives per corner is section 25's `split` arm, which is already measured
and already loses.

`tbaos` also answers a question `multall` cannot. Splitting the face areas is
free *in principle* (grid geometry, built once, Rule 3) but not free *in
ember*: `dA` is built in `geometry.f90` and consumed by the viscous kernels,
the multigrid restriction and the boundary conditions, none of which are under
test. So `tbaos` is what a residual port gets **without touching any of
that** -- the change one could actually propose -- and `multall` is the ceiling
it reaches only after a wider layout change. Section 26.1 asserted the AoS
penalty without pricing it; `multall - tbaos` is that price.

### 1.2 What the seven arms cover, and the one hole `prodsoa` fills

Two of the three axes are a **complete 2x2**, all at ember's AoS geometry:

| | primitives derived per corner | primitives read nodally |
| --- | --- | --- |
| **one fused sweep** | `prod` | `nodal` |
| **five passes** | `staged` | `tbaos` |

so the nodal-primitive effect is measured **twice**, once in each pass
structure. `prod - nodal` against `staged - tbaos` is the interaction test for
section 26.5's central claim that the representation and the pass structure
are not independent choices: equal deltas mean the changes are separable and
the attribution is additive, unequal deltas quantify the coupling that section
26.5 asserted. `split` remains the control for mass-flux staging inside the
five-pass family.

The third axis was the hole. With `multall - tbaos` alone, geometry layout is
measured at **one corner only, inside the five-pass family**, and the sign is
not obvious in general:

- a five-pass momentum loop wants **one** `dA` component and AoS makes it pay
  for a line of three -- section 26.1's argument;
- a fused sweep wants **all three** in one expression (`mdot = mf1*dA1 +
  mf2*dA2 + mf3*dA3`), so component-first is arguably the *right* layout there
  and SoA turns one stream into three.

`prodsoa` (`residual_prod_soa.f90`) settles it: production's kernel, arithmetic
untouched, on the nine component arrays. It is also the only variant in the
study that needs **no residual rewrite at all** -- it is a change to
`geometry.f90` and the `dA` consumers -- so if it wins it is the cheapest
proposal available, and if it loses then the layout effect exists only in the
presence of the pass split, which must be known before anyone proposes a
layout change on the strength of `multall - tbaos`.

Deliberately still not built, and why:

- **staged `AVGPI/AVGPJ/AVGPK`** (section 26.1's deferred item) extends the
  chain *past* `multall`; it is a ceiling, not an attribution, and only matters
  if the five-pass design wins;
- **fused + staged `rowt`/`rvt`** -- a precompute buffer in a fused sweep is
  the shape sections 18 and 20 already measured as a loss;
- **five passes + cons-derived + SoA**, the eighth cube cell -- `split`
  already shows cons-derived five-pass loses badly, so it is a strawman by
  Rule 1.

Two useful outcomes, both cheap to reach:

- `nodal` recovers most of `multall`'s win -> divides are the story, and the
  five-pass rewrite is not worth pricing further on this hardware;
- `nodal` recovers little of it -> the win lives in the pass structure or the
  geometry layout, and section 26's headline is not the small change it looks
  like.

---

## 2. What the change is

Narrower than a literal revert of section 20, for three reasons worth stating
before any code is written.

**`vx/vr/vt/ho` are already arguments.** `Grid.update_residual`
(`grid.py:1567-1570`) passes `block.Vx_nd/Vr_nd/Vt_nd/ho_nd` into
`set_residual` on every evaluation, and `set_residual` already declares them
(`residual.f90:607-609`). Today only the O(surface) cusp pass consumes them.
Nothing changes in `grid.py`, in the signature, or in `block.py`. This also
satisfies Rule 4 with the same citation section 26.2 used.

**The target form is already in the file, verbatim.** The reverted bodies are
`residual.f90:503-505` (`pm1/pm2/pm3` from `vx`, `vr`, `r*vt`) and `:512`
(`mf3 = cons1*(vt - Omega*r)`) inside `correct_cusp_kface_du`. The edit is
transcription into the three hot helpers, not derivation.

**No staged array.** multall stages `rowt`/`rvt` per node because five passes
cannot redo the work five times. A fused sweep forms `r*vt` and
`cons1*(vt - Omega*r)` per corner for two flops. Staging them here would
re-bundle a second variable *and* take the shape section 18 lost with (a
buffer that writes more than it saves). Explicitly out of scope.

### The edit

Three helpers, `iface_flow_row`/`jface_flow_row`/`kface_flow_plane`'s inner
`accum_corners` (`residual.f90:111-132`, `:220-241`, `:337-358`):

| now | becomes |
| --- | --- |
| `g1..g4 = 1/cons(...,1)` | deleted |
| `pm1 = sum 0.25*cons(...,2)*g` | `pm1 = sum 0.25*vx(...)` |
| `pm2 = sum 0.25*cons(...,3)*g` | `pm2 = sum 0.25*vr(...)` |
| `pm3 = sum 0.25*cons(...,4)*g` | `pm3 = sum 0.25*r(...)*vt(...)` |
| `mf3 = sum w*(cons(...,4)/r - Omega*cons(...,1)*r)` | `mf3 = sum w*cons(...,1)*(vt - Omega*r)` |

`pm4/pm5/pm6`, `mf1`, `mf2` are untouched. `mf1`/`mf2` stay pure `cons(...,2)`
/ `cons(...,3)` sums -- that matters for the gate (section 4).

Everything else is retained verbatim: hand-scalarized `pm1..pm6`/`mf1..mf3`,
hand-unrolled corners, rolling buffers, the fused `dU` write, the folded
change limiter, and the shared `correct_cusp_kface_du`. Sections 17/19 priced
that structure at 2-2.7x; none of it is under test.

**Divides removed: all of them.** Four reciprocals plus four `cons4/r` per
face, ~3 faces/cell = ~24 divides/cell -> 0. Note the `mf3` divide goes too:
`rho*Vt_rel = cons1*(vt - Omega*r)` needs no reciprocal. So this arm tests
section 26.4's mechanism at full strength, not partially.

### Source

Do **not** take `residual.f90` wholesale from before section 20. Production
has since folded the change limiter into `set_residual` (`0384c83`), which
came after. Edit forward from current `residual.f90`; use `git show
3e59b42:src/ember/_fortran/residual.f90` (the commit before `a12df87` landed
`consa`) as the reference for the three reverted helper bodies, and
`residual_consa.f90` as the shape template for a standalone arm file -- it is
the same transformation in the other direction.

### File

`src/ember/_fortran/residual_nodal.f90`:

- entry point `set_residual_nodal` at **file scope**, helpers in a private
  `residual_nodal_helpers` module (methodology section 2);
- `use residual_helpers, only: correct_cusp_kface_du` -- the cusp pass is
  production's own routine, called unmodified, exactly as `consa` and the
  section 25/26 arms do. `residual.f90` is a provider in
  `check_compile.sh`'s hand-ordered list, so no ordering hazard;
- signature identical to `set_residual`, including `planes`/`rows` (5-wide
  rolling buffers) and `kb`, so the harness reuses production's scratch carve
  unchanged;
- header states what is deliberately not reproduced: no staged nodal
  primitives, no SoA geometry, no pass split (methodology section 3).

`setup.py` globs `_fortran/*.f90`, so no build-file edit.

---

### 2.1 The `tbaos` edit

`residual_multall.f90` copied with the nine SoA component arrays replaced by
the three AoS ones. Stage 1 (mass flux) needs all three components in the same
expression, so `dA1(i,j,k)` simply becomes `dA(1,i,j,k)` and the layout costs
nothing there -- that is the case AoS is good at. Stage 2 is where it should
bite: each `m` pass wants **one** component, so `dA(i,j,k)` becomes
`dA(c,i,j,k)` with `c` passed as a **literal** from each call site (1, 2, 3, 3
for m = 2..5) so it constant-folds to a fixed offset with stride 3.

`stage_primitives` is `use`d from `residual_multall_helpers` rather than
copied, so the staging -- which is not under test in this pair -- has bitwise
identical codegen. Same for `scale_du_all` and `correct_cusp_kface_du`. The
AoS arm needs no separate `dAk` for the cusp pass, since its `dak` already is
the AoS array production's routine wants.

**Ordering hazard, and `check_compile.sh` needed a fix.** `residual_multall_aos.f90`
`use`s `residual_multall_helpers`, and the pre-flight compiles providers first
then the glob. The methodology doc says a rename can break this; what actually
broke it is subtler and worth recording: **`ls` collates locale-aware and
ignores the underscore**, so `residual_multall_aos.f90` sorts *before*
`residual_multall.f90` (comparing `...multallaos` against `...multallf90`), not
after as byte order would give. `residual_staged.f90` and
`residual_multall.f90` are now both listed as providers, which is honest
anyway: the arms deliberately share helpers.

---

## 3. Harness changes

Extend `tools/bench_residual_staged.py` rather than forking it -- all arms in
one `.so`, one process, round-robin interleaved (methodology section 2).

1. `ARMS = ("prod", "staged", "split", "multall", "nodal", "tbaos")`;
2. an `ENTRY` dict mapping arm name -> symbol. The old code built the symbol
   as `"set_residual_" + name`, which silently reported `tbaos` as "not in
   this build" because its kernel is `set_residual_multall_aos`. Both the
   caller table and the missing-arm check now read `ENTRY`;
3. `private["nodal"] = dict(planes=planes5, rows=rows5, kb=nk - 1)` -- the
   same 5-wide carve production uses. No new scratch, no new SoA geometry, no
   `del base["dai"]`;
4. `private["tbaos"]` takes `multall`'s staging scratch (`planes`/`rows`/`fi`/
   `fj`/`fk` plus `rowt`/`rvt`) and reads `dai`/`daj`/`dak` from `common`.
   Sharing `rowt`/`rvt` with `multall` is safe: both recompute them from `cons`
   on every call;
5. `private["prodsoa"]` takes production's 5-wide carve and `kb` plus the nine
   SoA component arrays, and no staging scratch. Like `multall` it drops
   `dai`/`daj` from `common` and keeps the AoS `dak` for the shared cusp
   routine;
6. `run_residual_staged.sh`'s aggregate block: add all three to `arms`.

The module docstring's traffic predictions are for the section 25 arms; add
the `nodal` prediction (section 5) rather than editing theirs.

> **The committed `tools/bench_residual_staged.jsonl` becomes stale on the
> first run.** Adding a fifth arm changes the round-robin cache footprint every
> other arm sees. Numbers from this run may only be compared with each other,
> never against the committed section 26 table. `run_residual_staged.sh`
> already `rm -f`s the results file, so the ladder re-times all five arms
> together and the comparison stays within one process. Do not splice.

---

## 4. Gates

Nothing is timed until all three pass (methodology section 5).

**Gate 0 -- build.** `make compile`, which pre-flights with `-Wall -Werror
-Warray-temporaries -Wfatal-errors`. A temporary is a build failure. Prefer
contiguous slices.

**Gate 1 -- vectorization, link-stage.**

```bash
EMBER_MARCH="-march=native -mtune=native" \
EMBER_OPT_REPORT=tools/opt_report_nodal.txt make compile
```

Every innermost `i` loop in the three new helpers vectorized; outer `j`/`k`
misses are normal. Two failure modes to look for specifically, both plausible
here and neither fatal to the study if named:

- **Gather.** Section 16 found production ~45% `vgatherdps` despite a clean
  report, and this arm adds three more nodal streams to an already
  gather-heavy access pattern. The report does not distinguish gather from
  unit stride; if the result is surprising, disassemble.
- **Register pressure, in the good direction.** Section 20.2 anticipated spill
  from four extra live values and did not get it; this arm removes them. If
  Gate 1 shows something worse than production, that is the thing to explain
  before timing.

**Gate 1 outcome (recorded, because it changed the kernel).** The arm was
first written with `accum_corners` as a contained subroutine in all three
helpers, exactly as production has it. GCC then refused to inline it in
`kface_flow_plane_nd` -- link-stage report `statement clobbers memory:
accum_corners`, all three k-face `i` loops scalar, driver down to 769 `%ymm`
against production's 2269 -- while production's own k-face loops vectorized.
The trigger is the three extra host-associated arrays (`vx/vr/vt`) pushing the
helper past an inline-cost threshold; it is a heuristic cliff, not a resource
limit, and timing it would have measured GCC's inliner rather than the nodal
representation.

gfortran has no `always_inline` attribute, so the fix is the one this codebase
already applied to the same failure under ifort (`470d6f8`): `accum_corners`
is manually inlined in the j- and k-helpers. Arithmetic, corner order and
summation order are unchanged. Both helpers had to move together -- inlining
the k-helper alone shifted the budget and cost the j-helper *its*
vectorization, which is worth knowing: **on this build the three face helpers
share one inlining budget, so a change to any of them can silently
de-vectorize another.** Check all three every time, not just the one edited.

Final status, per helper, in one build:

| helper | prod | nodal |
| --- | --- | --- |
| i-face interior `i` loop | missed | missed |
| j-face `i` loops | vectorized | vectorized |
| k-face `i` loops | vectorized | vectorized |

Parity, which is the condition for timing. (Production's i-face loop not
vectorizing under gfortran is pre-existing and unrelated to this arm.) Divides
in the driver: 50 -> 6, and the remaining six are the change limiter's
`avg`/`ravg` reciprocals, not the sweep. Zero `vgather` in either arm.

**Gate 2 -- numerical agreement.** Existing `swirl()` gate covers this arm
correctly and already does the `update_cached_conserved()` call Rule 6
requires -- this arm reads the nodal arrays, so it needs it for exactly
section 26.2's reason.

**Gate 2 outcome.** At 300k on the swirled state the arm agrees to 2.9e-10
absolute, **2.0e-07 of the dU field scale** at `dampin=0` and 5.9e-07 at
`dampin=2` -- roughly 2x tighter than `staged` and 4x tighter than
`split`/`multall`, and two orders inside the goldens' rtol. Residual goldens
pass unregenerated; full suite 1727 passed.

The sub-proof, as predicted but with its quantifier corrected by the grid:
`mf1`/`mf2` are untouched, and on this grid `dAi(3) = dAj(3) = 0` (measured:
`dAi` is axial-only, `dAj` radial-only, `dAk` theta-only), so the i- and
j-face mass fluxes consume only `mf1`/`mf2` and are bitwise. **The entire mass
deviation is therefore the k-face `mf3` term**, the one mass quantity that
changed -- 1.5e-11 absolute, 8.4e-08 of the mass scale, nonzero on 61544 of
286720 cells. That isolation is what the swirled state buys; on the unswirled
duct `mf3` is multiplied by a zero area and the check is vacuous (Rule 5).

**Gate 2 outcome for `prodsoa`: bitwise, at both `dampin` settings.** Nothing
weaker was acceptable here and nothing weaker was needed -- the arm changes
only how `dA` is indexed, so bitwise agreement proves the layout change alters
no arithmetic anywhere in the kernel, and **every nanosecond of difference in
the ladder is the layout and nothing else**. This is the cleanest arm in the
study; contrast `multall`, which cannot be bitwise anywhere.

It needed the same manual inlining as `nodal`, for the same reason: three `dA`
dummies per direction instead of one pushed `accum_corners` past GCC's inline
threshold and the k-face loops went scalar. Note the ordering -- **the
inlining fix was applied before the bitwise check, and the check still comes
out bitwise**, which independently confirms that manual inlining is a codegen
change and not a numerics one.

**Gate 2 outcome for `tbaos`, and a failed prediction.** The arm was written
expecting **bitwise** agreement with `multall` -- identical operands, identical
order, only the `dA` indexing differs -- and that was going to be the gate. It
is not bitwise: 1.1e-09 max, 7.6e-07 of scale. What the per-component
breakdown shows is better than the prediction was:

| component | tbaos vs multall | cells differing | tbaos vs prod | multall vs prod |
| --- | --- | --- | --- | --- |
| mass | **0** | **0.00%** | 1.1642e-10 | 1.1642e-10 |
| x-mom | 1.4552e-11 | 1.38% | 6.5484e-10 | 6.5484e-10 |
| r-mom | 1.0987e-09 | 3.29% | 6.3665e-10 | 6.3665e-10 |
| rVt-mom | 7.2760e-12 | 0.87% | 2.1828e-11 | 2.1828e-11 |
| energy | 1.1642e-10 | 1.63% | 1.1059e-09 | 1.1059e-09 |

- **Mass is bitwise**, and that carries a proof: the mass residual is the pure
  six-point sum of the staged mass fluxes, so it proves the three AoS stage-1
  helpers reproduce the SoA ones exactly, both cross-stream directions and the
  wall masks included. Stage 1 is exactly where AoS changes nothing, and the
  measurement agrees to the last bit.
- The differences are confined to the stage-2 passes, which is precisely where
  the `dA` access changed from contiguous to stride-3, and they are last-bit:
  a fifth of one ulp of the r-momentum flow scale, on 1-3% of cells.
  `-Ofast` reassociation and FMA contraction after strided-load lowering, the
  class the methodology doc already documents.
- **The two arms deviate from production identically in every component**, to
  all printed digits. That is the strongest available evidence there is no
  indexing bug: a mis-indexed `dA` component would mix axial with radial face
  areas and show up as an O(1) relative error, not 1e-6 of scale, and it would
  certainly not leave the deviation-from-prod unchanged.

Everything else lands at the reassociation level of the other arms: `Vx_nd` is
itself `cons2/cons1` computed upstream in float32, so bitwise agreement in the
momentum components is not expected and its absence is not a bug. Quantify
deviations in **ulps of the face flows being differenced**, not relative to
the residual.

Then:

```bash
uv run pytest tests/test_residual_golden.py -q   # unregenerated
uv run pytest tests -q
```

---

## 5. Predictions, pre-registered

Written before the run, scored in the write-up including failures
(methodology section 7).

**Traffic.** Adds `vx`, `vr`, `vt` (+12 B/cell). Drops `cons(...,4)` from the
hot sweep **if** it goes fully dead there -- `cons` is component-last, so a
dead component is a real dropped stream. *Verify that by reading the sweep;
do not assume it.* Net ~+8 B/cell against production's modelled ~152, i.e.
~+5%. Section 20 measured the reverse direction at -12.5 B/cell modelled but
only -4.1% realised, and section 20.4 concluded compulsory-byte accounting is
~2x optimistic for heavily-reused nodal fields. Expect the traffic penalty
here to be similarly overstated.

1. **Serial: `nodal` beats production, by more than the traffic model says it
   should lose.** Divides go 24/cell -> 0; section 26.4 prices Haswell's
   `vdivps` at ~13 ns/cell against production's measured 39-52.
2. **Contended: `nodal` still beats production, but by less than serial**,
   because the extra ~8 B/cell bites where bandwidth is contended and the
   divider pressure is unchanged.
3. **`nodal` lands between production and `multall` at every size**, closer to
   `multall`. If it does not -- if it matches or beats `multall` -- then the
   pass split and SoA geometry are worth nothing or less than nothing, which
   is a stronger result than the one being sought.
4. **Production's 300k L2 hump (section 25.4) shrinks or moves.** The 5-wide
   rolling buffer is unchanged, but three more streamed nodal fields change
   the L2 working set. If the hump vanishes, the section 25.4 explanation
   needs revisiting.

For `tbaos`:

5. **`tbaos` lands between `nodal` and `multall` at every size**, making the
   chain monotone. If it does not -- if AoS ever *beats* SoA -- then stage 1's
   three-components-per-expression reads are paying for the stage-2 passes'
   line, and section 26.1's layout argument is wrong in sign.
6. **The `multall - tbaos` gap widens with block size**, because the AoS
   penalty is a memory-traffic effect (a line of three fetched for one
   component) and should be invisible while the face areas are cache-resident.
   A gap that is flat across the ladder means the penalty is not what section
   26.1 says it is.

For `prodsoa`:

7. **`prodsoa` loses to `prod`, or ties.** Every face in a fused sweep
   consumes all three components in one expression, so AoS lines are used in
   full and SoA converts one stream into three. A win here would mean
   production's `dA` reads are not the compulsory-traffic item the section 25
   model assumed, and the model would need revisiting (Rule 8).
8. **`prod - prodsoa` and `tbaos - multall` have opposite signs.** That is the
   sharp form of the whole layout question: the same layout change helping one
   pass structure and hurting the other is exactly section 26.5's
   co-optimisation claim, restated on the third axis and measured rather than
   asserted. If instead SoA helps both, the layout is a free-standing win and
   the cheapest proposal in the study is a `geometry.f90` change with no
   residual rewrite at all.

Predictions 3, 5 and 8 are the ones that carry the study. Note that both are
*bracketing* claims, and per Rule 2 their quantifier is: at these four sizes,
on this machine, under gfortran.

---

## 6. Decision rules, and what this run cannot decide

**It cannot adopt anything.** Section 20 measured this arm's exact inverse on
ifort/Sapphire at **-0.3% serial, -4.5% saturated, 179/200 rank-pairs won**.
So the honest prior for `nodal` on the production target is a **~+4.5%
regression saturated**, and this machine cannot test that. Section 26.4 says
why the two machines should disagree: Sapphire's AVX-512 divider is several
times faster per element, so the ~24 redundant divides that dominate here cost
far less there.

Pre-registered outcomes, phrased as what they rule out:

| gfortran/Haswell result | conclusion |
| --- | --- |
| `nodal` recovers most of `multall`'s win | section 26.4's mechanism confirmed; the win is divides, so it is a hardware-specific result and the five-pass rewrite need not be priced further on this machine |
| `nodal` recovers little of it | the win is pass structure and/or SoA geometry; section 26.4's mechanism is wrong and needs re-deriving before anything is proposed for production |
| `nodal` beats `multall` | the pass split costs more than the staging saves; the strongest form of the first row |
| `nodal` loses to production | divides are not the binding constraint even here; re-open the whole section 26 analysis (Rule 7 first: re-run, densify, remove the harness as a suspect) |

And for the second step, which is the one with a production consequence:

| gfortran/Haswell result | conclusion |
| --- | --- |
| `tbaos` close to `multall` | the SoA layout is worth little; a residual-only port captures nearly the whole win without touching `geometry.f90` -- the cheapest real proposal on the table |
| `tbaos` close to `nodal` | the win is mostly the geometry layout, so it is not a residual-kernel change at all; it is a `dA` layout change, and should be evaluated as one (against the viscous, multigrid and bcond consumers) |
| `tbaos` beats `multall` | prediction 5 fails in sign; section 26.1's layout argument is wrong and the `multall` arm's SoA hoist deserves re-examination as a possible unfair advantage |

In **every** row the production decision stays blocked on an ifort/Sapphire
measurement. What this run buys is that the Sapphire run, when it happens,
knows which single variable to measure instead of a three-change bundle.

Keep the arm regardless of the ruling -- section 22's precedent, restated in
methodology section 9: a near-bitwise rejected variant is retained because the
ruling can differ on other hardware, and this arm is *expected* to rule
differently there.

---

## 7. Run

```bash
# build all five arms into one .so, with the link-stage report
EMBER_MARCH="-march=native -mtune=native" \
EMBER_OPT_REPORT=tools/opt_report_nodal.txt make compile

# Gate 2 only (fast)
taskset -c 0 uv run python tools/bench_residual_staged.py \
    --mode serial --ncell 300000 --check-only

uv run pytest tests/test_residual_golden.py -q
uv run pytest tests -q

# ladder: serial + 6-rank socket-contended, four sizes, aggregate + PDF
./tools/run_residual_staged.sh 6 100000 300000 1000000 2000000
```

Ranks pinned to cores 0-5, one socket, for the reason in methodology section
6. Report it as "6-rank socket-contended", never "saturated". Anything
anomalous gets Rule 7 before it gets an explanation.

Artifacts: `residual_nodal.f90` and `residual_multall_aos.f90` (tracked);
`bench_residual_staged.jsonl` regenerated with six arms (tracked);
`opt_report_nodal.txt` and the PDF untracked and regenerated.

**Gate 1 status of the whole build, for reference when re-running.** All 15
inner `i` loops of `tbaos` vectorized, zero missed. `nodal` and `prod` keep
the parity recorded above -- re-checked after `tbaos` was added, since the
face helpers share an inlining budget and a new arm can move it. No `vgather`
in any of the four arms. Driver divide counts: `prod` 70, `nodal` 6, `multall`
9, `tbaos` 9.

---

## 8. Write-up

New section 27 in `viscous_kernels.md`, structured as sections 20 and 26 are:
the change, the gates, the ladder in both regimes, then a **Reading** section
that scores the six predictions above -- including the failures, of which the
`tbaos`-is-bitwise one has already failed and been replaced by a better result
-- and states the machine/compiler threat to validity in the body, not as a
footnote. Lead with the **attribution table** (`prod` -> `nodal` -> `tbaos`
-> `multall`, one delta per step), not with a speed number: the deltas are the
transferable claim and the absolute numbers are not.

Update section 26.5's "highest-value next probe" bullet to point at section 27
once it lands.
