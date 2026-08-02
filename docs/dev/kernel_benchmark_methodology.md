# Kernel A/B methodology: comparing designs, not implementations

How the `set_residual` pass-structure study (`viscous_kernels.md` sections 25
and 26) was run, written up as a reusable protocol. It exists because that
study reached the **wrong conclusion first**, published a ruling, and then
reversed it -- and the reversal was caused by a methodological error, not by a
measurement error. Everything below is written to make that specific class of
mistake harder to repeat.

The study asked: multall/multall evaluates the five conserved-variable
residuals in five passes over the mesh; ember does it in one fused sweep.
Which is faster, and why is multall competitive in production?

- First answer (section 25): the five-pass design loses by 15-27%.
- Second answer (section 26): the *faithful* five-pass design wins by 13-50%.
- Third answer (section 27, after the harness and the build were fixed): the
  five-pass design loses by **65%**, and every other candidate loses too.

> ## READ THIS FIRST: the third answer invalidated the instrument, not just
> ## the ruling
>
> Sections 25 and 26 were measured on a harness and a build that could not
> support the precision they were quoted at. Two independent faults, either of
> which alone would have been enough to invert a ranking:
>
> 1. **The baseline was compiled wrong.** Production's `set_residual` was
>    inline-budget-limited, and the benchmark build was accidentally relieving
>    that by adding files. Compiled properly it runs **73.2 -> 46.7 ns/cell,
>    -36%**. Every arm in sections 25 and 26 was racing a handicapped
>    production. See section 2.1.
> 2. **The harness leaked.** Round-robin interleaving in one process made a
>    ratio depend on which *other* arms were in the build -- a 16-point swing
>    at fixed binary, size and rank count -- and free-running ranks made the
>    contention state an unrecorded variable. See sections 6.1 and 6.2.
>
> On the corrected instrument, at 1M cells and 16 ranks, **production beats
> everything tested**: `prodsoa` +5.3%, `rinv` +6.4%, `nodal` +11.3%,
> `multall` +65.6%, `tbaos` +98.6%, `staged` +104.9%, `split` +110.5%.
>
> The rules below marked **[REVISED]** are ones sections 25/26 followed and
> that are now known to be wrong. The rest still stand: none of the *design*
> reasoning was at fault, and Rules 1-8 all survived. What failed was the
> measurement apparatus, and it failed silently for two studies.

---

## 1. The central hazard: a design is not one variable

The first study built two five-pass kernels and kept everything else the same
as production -- same nodal representation (velocities re-derived from the
conserved state at every face corner), same component-first `dA(3,i,j,k)`
geometry. That felt like good experimental hygiene: change one thing at a
time.

It was not. Production's nodal representation is *itself* a consequence of
being a single fused pass. Section 20 chose to re-derive `Vx/Vr/r*Vt` from
`cons` at every face corner -- paying ~12x redundant reciprocals -- precisely
because a single bandwidth-bound sweep would rather burn flops than stream
three more fields. multall made the opposite choice, staging primitives once
per node, because five passes cannot afford to redo that work five times.

So "five passes with ember's data structures" is not a point on the design
axis being explored. It is a hybrid neither code would ever write, and it is
worse than both endpoints. Measuring it and calling the result "the multall
design" was the error.

> **Rule 1. Identify the choices the incumbent has co-optimised, and vary them
> together.** Before building an arm, ask what *else* in the incumbent exists
> only because of the thing you are changing. If a data-layout or
> representation decision was justified by the very structure under test, it
> must move with it, or the arm is a straw man.

Symptom to watch for: an arm that loses on *every* axis at once, with no
regime where it is even competitive. A genuine alternative design usually wins
somewhere. Section 25's `split` arm was +57 to +104% everywhere -- that should
have prompted the question "what is this design assuming that we have denied
it?" rather than a ruling.

> **Rule 2. Never generalise from an arm to a family.** Section 25 claimed
> "one kernel bounds the whole staged family" because the arm built was
> strictly cheaper than multall's literal form. That was true *at a fixed nodal
> representation* and false in general. Bounding arguments must state the
> quantifier.

---

## 2. Build protocol

Inherited from `viscous_kernels.md` section 4, which exists because an
isolated two-file build once inverted the sign of a result (section 2).

- **Build the real way.** `make compile` (`uv pip install -e .`), which f2py
  compiles every `src/ember/_fortran/*.f90` together with `-flto
  -fwhole-program`. A standalone `gfortran one_kernel.f90` does not reproduce
  whole-program IPA and will mis-rank arms.
- **[REVISED] Arms are no longer all built into one `.so`.** The original rule
  said they must be, so that a single process could compare them round-robin
  with no cross-build drift. That reasoning was sound about drift and wrong
  about everything else: putting the arms in one program makes them perturb
  *each other's compilation*, and comparing them in one process makes them
  perturb each other's *cache and phase*. Both effects are larger than the
  differences being measured. `setup.py` now excludes benchmark-only sources
  by default; `EMBER_ARMS=nodal,multall` builds specific ones back in. See
  section 2.1 for the mechanism and the gate that replaces the old rule.
- **Hold flags fixed across arms** -- automatic, given the above. Set
  `EMBER_MARCH="-march=native -mtune=native"` for a tuned run and keep it
  constant.
- **Entry points at file scope**, helpers in a private module, so f2py wraps
  the driver and not the helpers. Follow `residual_consa.f90`'s shape.

### Ordering hazard

`tools/check_compile.sh` pre-flights syntax with a hand-ordered file list:
module *providers* first (`residual.f90`, `viscous.f90`), then everything else
alphabetically. An arm that `use`s another arm's module (here
`residual_multall.f90` uses `residual_staged_helpers` for `scale_du_all`) is
only safe because `staged` sorts before `multall`. f2py/meson does its own
dependency resolution, so this affects the pre-flight only -- but a rename can
break the commit hook without breaking the build.

Sharing a routine like `scale_du_all` across arms is deliberate: it guarantees
identical codegen for the part that is *not* under test.

### 2.1 The build is a variable. Treat it as one.

> **Rule 9. A kernel's machine code depends on what else is in the program.
> Verify that the baseline is byte-identical between the builds you are
> comparing, or the comparison is meaningless.**

The build is `-Ofast -flto -fwhole-program`, and GCC's inline budgets --
`inline-unit-growth`, `large-unit-insns`, `large-function-growth` -- are
**unit-level**. Growing the program grows the absolute inlining budget, which
silently changes decisions for functions that did not change. Measured on
production's own `set_residual`, varying only which benchmark files existed in
`_fortran/`:

| build | insns | `vrcpps` | fingerprint |
| --- | --- | --- | --- |
| production only (what ships) | 7,818 | 50 | `81b468af` |
| + `residual_nodal.f90` | 7,421 | 49 | `9acc8829` |
| + all benchmark arms | 10,726 | 101 | `63c75ac1` |

Sections 25 and 26 ran in the bottom row. Their `prod` had 37% more
instructions and twice the reciprocals of the shipped kernel, and was ~20%
*faster* than what ships -- so the arms were measured against a baseline that
existed nowhere else.

Three consequences, all now enforced:

- **Benchmark sources are excluded by default** (`setup.py`'s
  `BENCHMARK_ONLY`, a denylist so the failure mode is building too much rather
  than shipping a broken package). This also stopped the wheel carrying ten
  dead residual kernels.
- **Fingerprint the baseline, don't time it.** `tools/codegen_gauge.py` hashes
  the recursive closure of a symbol's machine code, normalised so that
  layout-only differences do not register. Timing cannot do this job: the
  launch-to-launch noise floor is ~0.4%, the same order as the effects being
  chased. Codegen identity is exact.
- **Pinning the unit budgets decouples the build entirely.** With
  `--param=inline-unit-growth=1000000` and the `large-*` equivalents,
  `set_residual` fingerprints identically under `EMBER_ARMS` unset / `nodal` /
  `all`. Note the syntax: f2py re-splits `--param X=Y`, so the `--param=X=Y`
  form is required. Available via `EMBER_FFLAGS_EXTRA`; **not** a default,
  because it changes production codegen and that is a production decision.

**And it is worth 36%.** Production-only build, 1M cells, 16 ranks, 10
launches each: default flags **73.248 ns/cell**, pinned budgets **46.733**.
Goldens pass unregenerated, full suite passes. The face helpers were simply
not being inlined. This is probably gfortran-specific -- `INTEL_FLAGS` already
carries `-inline-factor=10000`, which scales ifort's size limits including
`-inline-max-per-compile` -- but it means this box was not a valid proxy for
production until it was fixed, and it may account for part of sections 17/19's
"gfortran ~4.7x slower than ifort", which has been read as a compiler-quality
fact rather than a build-configuration artefact.

> **Corollary, learned the hard way.** If a Gate 1 failure is fixed by *hand*
> inlining in the candidate (as `residual_nodal.f90`, `residual_prod_soa.f90`
> and `residual_rinv.f90` all did), the candidate now has forced inlining that
> the incumbent does not. `nodal` measured **-4.4%** that way and **+11.3%**
> once the budgets were pinned and both kernels were inlined by the same
> rules. A codegen fix applied to one arm is a thumb on the scale.

---

## 3. Arm design: isolation and attribution

Four arms, each answering one question:

| arm | what it is | what it isolates |
| --- | --- | --- |
| `prod` | production, untouched | baseline |
| `staged` | 5 passes, mass flux staged into `fi/fj/fk` | the pass structure |
| `split` | 5 passes, mass flux recomputed per pass | staging vs recomputation |
| `multall` | 5 passes + staged nodal primitives + SoA geometry | the faithful design |

**Include an attribution control.** `split` exists only so that a loss can be
attributed to the split rather than to the staging. Without it the study would
have concluded "staging is bad" when in fact staging was worth 21-45% *within*
the five-pass family. Attribution controls are cheap (here, ~40 lines on a
shared driver) and they are what turn a number into an explanation.

**Write the drivers out separately rather than sharing one branchy driver.**
A shared `if (recompute)` risks perturbing codegen for one arm and not the
other, which corrupts the comparison being made. Duplicating ~100 lines of
sweep is the right trade; sharing the *helpers* is not.

**Do not build strictly-dominated variants.** multall's literal form
materialises `FLUXI/FLUXJ/FLUXK` per variable and re-reads them; fusing the
flux build with the six-face difference is strictly cheaper, so only the fused
form was built. State the domination argument explicitly, and state its
quantifier (see Rule 2).

**Record what was deliberately not reproduced.** The `multall` arm omits
multall's staged face-averaged pressures (`AVGPI/AVGPJ/AVGPK`). Writing that
down in the kernel header turns an omission into a stated scope boundary and a
named next step.

---

## 4. Fairness: what may be hoisted out of the timed region

The `multall` arm needs nine separate face-area component arrays where ember
has one component-first array. Transposing per call would be absurd; hoisting
it entirely could be cheating. The test is not "is it expensive" but:

> **Rule 3. An input may be prepared outside the timed region if and only if
> the real port would also prepare it outside the per-step loop.**

- Face areas are **grid geometry**, built once in multall's `FIND_AREAS`, never
  rebuilt per step. Splitting them is a startup cost in any real port, so the
  harness allocates them once. Fair.
- `rowt` and `rvt` are **solution-dependent**, so the kernel recomputes them
  on every call. Only their storage is preallocated, exactly as any block
  scratch buffer is. Fair.

The second fairness question was subtler and needed checking, not assuming:
the `multall` arm reads the nodal `vx/vr/vt/ho` arrays. That is a free lunch
unless ember already materialises them. It does -- `Grid.update_residual`
passes `block.Vx_nd/Vr_nd/Vt_nd/ho_nd` into `set_residual` on every
evaluation, and they are cached derived arrays invalidated each step. Verified
by reading the call site, not assumed from the signature.

> **Rule 4. Any input an arm gets "for free" must be traced to a real
> production consumer.** Cite the call site in the write-up.

---

## 5. Correctness gating

No timing number is reported until three gates pass. This ordering is not
bureaucracy: two of the three gates caught real problems in this study.

### Gate 0 -- build

`make compile`, which pre-flights with `-Wall -Werror -Warray-temporaries
-Wfatal-errors`. `-Warray-temporaries` is the one that matters: passing an
array slice to an explicit-shape dummy can silently generate a copy that would
dominate the measurement. A temporary is a build failure, which is correct.
Prefer contiguous slices (`fi(:,j,k)`) over sequence association.

### Gate 1 -- vectorization, from the **link-stage** report

`EMBER_OPT_REPORT=<path> make compile` injects `-fopt-info-vec-all` at link
time, where GCC's LTO backend does the real codegen. A compile-stage report
describes discarded per-TU codegen and reports spurious misses (section 4.6).

Check that every innermost `i` loop vectorized. Expect outer `j`/`k`/`m` loops
to be reported as missed -- that is normal. Watch for a loop that is
"vectorized" via gather/scatter rather than unit stride; the report does not
distinguish them, and section 16 found production is ~45% `vgatherdps` despite
a clean report. Disassembly, not the report, settles that.

Useful non-failure: GCC turned one plane-copy loop into `memcpy`, which is
better than vectorizing it.

### Gate 2 -- numerical agreement, on a **non-degenerate state**

This gate found the study's worst trap.

> **`build_duct_grid` cannot detect an error in the j- or k-face mass flux.**
> The duct is axially straight with `Vr = Vt = 0`, so `dAj(1) = dAk(1) = 0`
> and the cross-stream mass fluxes are *identically zero*. Zero times a wrong
> number is still zero. Timing is unaffected -- the work happens either way,
> there are no data-dependent branches -- so the benchmark was valid while the
> correctness check was vacuous over a third of the kernel.

The harness therefore seeds cross-stream momentum (`swirl()`) before gating.
The state need not be converged or even physical; it only has to make every
term in every helper non-degenerate.

> **Rule 5. Before trusting a gate, ask which terms of the kernel the test
> state actually exercises.** A symmetric or degenerate case is the normal
> state of a synthetic benchmark grid, and it hides exactly the code a new
> kernel is most likely to get wrong.

A second trap followed from the first. `swirl()` writes `conserved_nd`
directly. Production re-derives its primitives from `cons`, so it never
noticed; the `multall` arm reads the *nodal* arrays, so without a
`block.update_cached_conserved()` call after the write the two arms solve
**different states** and the gate reports a failure that does not exist.

> **Rule 6. When arms disagree about where a quantity comes from, any direct
> write to state must be followed by the cache invalidation the solver would
> do.**

**On exactness.** Aim for bitwise, isolate what *is* bitwise, and quantify the
rest instead of hand-waving:

- `staged`'s mass component is bitwise identical to production. That is not a
  nicety -- the mass residual is the pure six-point sum of the staged mass
  fluxes, so bitwise agreement **proves all three staging helpers reproduce
  production's mass flux exactly**, both cross-stream directions and the wall
  masks included. Look for a component whose agreement carries a proof.
- The rest agrees to <= 1.2e-09 absolute, <= 2.3e-06 of the dU field scale.
  Under `-Ofast` (reassociation, FMA contraction, `-freciprocal-math`) a
  quantity computed in a dedicated helper will not match one computed inline
  among five others. Aligning the source form verbatim was tried and does not
  fix it.
- Quantify deviations in **ulps of the quantities being differenced**, not
  just relative to the result. A residual is a small difference of large face
  flows, so relative-to-result overstates the error badly. Here the deviation
  is ~1e-5 of one ulp of the face flows.
- The `multall` arm cannot be bitwise anywhere and is not expected to be: it
  stages `r*Vt` per node where production forms `cons4/cons1` per corner. Its
  agreeing *no worse* than the other arms is itself evidence it is correct.

Then run the goldens (`tests/test_residual_golden.py`) unregenerated, and the
full suite. Never report a speed number from a kernel that fails a golden.

---

## 6. Measurement protocol

### Two regimes

- **serial** -- one process, `taskset -c 0`. The diagnostic number.
- **contended** -- N independent processes, each with its own grid, all timing
  the same window. This is the production-representative regime: ember runs
  many ranks per node, where DRAM bandwidth is contended.

### 6.1 [REVISED] Contention without a job scheduler -- and why a shared start time is not enough

This machine is not in a SLURM cluster, so the contended arm is N background
processes pinned with `taskset`. The original design had them rendezvous on a
shared wall-clock start time (`EMBER_BENCH_START`, a 180 s fixed sleep) and
then free-run. **That is not sufficient, and it silently corrupted two
studies.**

Free-running ranks drift out of phase, so each rank is timed against a
different, unrecorded mixture of what its neighbours happen to be doing. The
symptom: sweeping the *arm set* at fixed binary, size and rank count moved
`multall` vs `prod` at 1M contended from **-17.5% (four arms) to -1.1% (two
arms)** -- a 16-point swing generated entirely by the harness. With only two
arms the ranks stay roughly in phase, so the bandwidth-hungry arm is scored
against six concurrent copies of itself; with more arms that self-contention
is diluted.

> **Rule 10. In a contended run, synchronise the ranks before every timed
> call.** Otherwise the contention state is a free variable, and it is a
> larger effect than the kernel differences being measured.

`tools/bench_prod_baseline.py` implements a lock-free shared-memory barrier
(one `int64` slot per rank, each rank writing only its own, spin never sleep;
microseconds against a ~41 ms call). It does double duty:

- **as the startup rendezvous**, replacing the 180 s sleep. That constant was
  ~11x oversized -- the slowest of 16 ranks was ready at 16.2 s -- and was
  ~98% of the wall clock of every contended configuration ever run, which is
  what made repeat launches look unaffordable;
- **before every timed call**, so every rank is inside the same kernel at the
  same time. That is both a defined condition and what production does.

Two implementation notes. CPython's `resource_tracker` unlinks a shared
segment when *any* attaching process exits, so non-owners must
`resource_tracker.unregister` or the owner's segment vanishes underneath it.
And use a unique segment name per launch, so a crashed run cannot poison the
next one.

### 6.2 [REVISED] One arm per process

Round-robin interleaving was introduced to cancel drift between arms. With one
arm per process there is no drift to cancel, and the coupling it introduced --
each arm running with its neighbours' footprints in cache -- disappears.
Cancel drift at the **launch** level instead: run launch-outer, arm-inner, so
each arm is sampled across the whole time window rather than in one contiguous
block (`tools/run_all_arms.sh`).

**Pin all ranks to one socket.** Six Haswell cores ask for more than one
socket's memory controller can deliver (~10-15 GB/s sustained per core against
a ~59 GB/s controller), so cores 0-5 give genuine contention. Spread across
two sockets they would not contend, and a bandwidth-heavy arm would be
flattered. Pinning is part of the experiment, not tidiness.

**Name the regime honestly.** Six of sixteen cores on a Haswell is not the
100-rank sapphire regime. It is reported as "6-rank socket-contended", never
"saturated".

### Sweep sizes, always

A kernel's ranking changes as its working set crosses L2 and L3. In this study
the staged arrays are ~12 MB at 1M cells but ~1.2 MB at 100k, where they stay
L3-resident and flatter the staged arms. Sweep at least 100k / 300k / 1M / 2M,
report **ns/cell**, and plot the curve (to PDF). A win that decays
monotonically with size is not a win.

The sweep is also what turned an anomaly into a finding: production is ~40%
slower per cell at 300k than at 150k or 1M, a reproducible L2 hump where its
5-wide rolling plane buffer is ~80% of a 256 KB L2. A four-point ladder found
it; a single size would have hidden it or mistaken it for noise.

### 6.3 [REVISED] Stabilising, and where the replication belongs

- `OMP_NUM_THREADS=1`; pin with `taskset`. On this box, 16 ranks means cores
  0-15, which is 8 physical cores on *each* socket -- SMT siblings are 16-31
  and are left idle. That is a different regime from 6 ranks on one socket
  (two memory controllers, plus NUMA), not a bigger one; never splice them.
- >= 5 warmup calls, each behind the barrier.
- **20-50 reps, and no more.** See below.
- **Repeat the LAUNCH 5-10 times.** This is the replication that matters.
- Restore any input an arm mutates, **untimed**, between reps. (Not needed
  here: every `dU` element is assigned before it is read.)

> **Rule 11. Replicate at the launch, not the rep. N reps in one process are
> N correlated views of a single draw.**

The old rule -- 50 reps, report median and min, quote the spread across ranks
-- treated rep count as the precision knob and rank spread as the error bar.
Both are wrong. All 50 reps share one draw of page placement, allocation
alignment, core assignment, thermal state and rank phase; so do all the ranks
in a launch. Section 20 got this right (two independent repeat processes,
r1/r2, agreeing within 0.6 points) and this document dropped it, which is how
sections 25 and 26 came to quote single draws to two decimal places.

The arithmetic, measured rather than assumed (`tools/bench_rep_convergence.py`,
2000-rep traces):

- per-rep scatter is ~3% of the median, but the trace is **right-skewed** with
  rare severe outliers (p95 +5.5%, p99 +8.6%, p100 +49%) and drifts <1% over
  2000 reps. The outliers, not drift, are what make short runs unreliable;
- a contiguous-block bootstrap (**not** i.i.d. -- consecutive reps are
  correlated and shuffling understates the reps required) gives, for +/-1%
  within one launch: **500 reps** for the median, **10** for the min;
- `Var = sigma^2_launch + sigma^2_rep/n`. With sigma_launch ~3% and a 50-rep
  standard error of ~0.8%, going 50 -> 500 reps moves the total from 3.1% to
  3.01%. Ten times the wall clock for a tenth of a point. Five launches gives
  1.4%; ten gives 1.0%.

> **Rule 12. Use `min` for serial and `median` for contended.** `min` is
> ~2x more precise at 10 reps than the median is at 500, because interference
> only ever adds time -- but under contention it preferentially samples the
> instants when neighbouring ranks were *between* calls, i.e. it erases the
> contention the regime exists to measure. Ranking `multall` on `min` turned a
> -1.1% into -17.8%. Also note `min` is biased low and the bias grows with n,
> so mins may only be compared at equal rep count.

**Result of applying all of this**: `prod` at 1M / 16 ranks reproduces to
**+/-0.40% launch-to-launch** (10 launches, 30 reps), against +/-6% for the
same quantity under the old harness. Cross-build, with the codegen
fingerprint identical, two independent launch sets agreed to **0.18%**.

### Isolating a suspicious result

When one point inverted the ranking, three things were done before it was
written up, and all three are cheap:

1. **Re-run it.** 300k reproduced to within 0.6 ns/cell.
2. **Densify around it.** 150k/200k/250k/300k/400k/500k/700k showed a *band*,
   not a spike -- which changes the explanation entirely.
3. **Remove the harness as a suspect.** Round-robin interleaving means each
   arm runs with the previous arm's footprint in cache. Production was
   re-timed **alone in its own process**, with no other arm present, and the
   hump survived. Only then was it attributed to production.

> **Rule 7. A result that contradicts your headline gets re-run, densified,
> and re-measured with the harness removed as a suspect -- before it is
> explained.**

---

## 7. Analysis: traffic models and their failure modes

A per-cell compulsory-traffic model is worth building before measuring: it is
cheap, it forces the layout details into the open, and it is falsifiable.

Count every stream once per pass that touches it, and respect layout:
component-first `dA(3,i,j,k)` costs its full 36 B/cell in **any** pass that
touches **any** axis, because the line holds all three. That single detail
dominated the model and is invisible in a flop count.

Then score the predictions in the write-up, including the ones that failed.
In this study, of three pre-registered predictions:

1. "Both split arms ~2.5x prod contended" -- **wrong** (+15 to +91%);
   compulsory traffic over-predicts when staged arrays are re-read from cache.
2. "Serial gap much smaller, because staging removes the divides" -- **right**
   for `staged`, and badly understated for `split`.
3. "`staged` loses to `split` contended, because the write plus five re-reads
   cost more than the `cons` re-reads it saves" -- **wrong, and the most
   useful of the three**: `staged` beat `split` in every size and both
   regimes.

Prediction 3 failed for the reason section 22 already established from the
other direction: **what the passes share decides the outcome, and here what
they share is compute, not bytes.** Counting bytes ranked the two five-pass
arms backwards.

The final result made the same point harder: the `multall` arm moves ~2.2x
production's bytes and still wins under contention, because on this machine
the kernel is **divide-limited**, not bandwidth-limited. Production issues ~24
divides per cell (four reciprocals plus four `cons4/r` per face, ~3 faces per
cell); Haswell's 256-bit `vdivps` at ~10-13 cycles throughput makes that ~13
ns/cell of pure divider pressure against a measured 39-52.

> **Rule 8. A traffic model ranks arms only when the kernel is
> bandwidth-bound. Check that premise before trusting the ranking** -- an arm
> that moves more bytes and still wins is telling you the premise is false,
> not that the measurement is wrong.

---

## 8. Pre-registered decision rules

Fix the kill criteria before running, so the result is not read to taste:

- loses > 20% contended at 1M cells -> rejected;
- wins serially but loses contended -> contended decides, since production
  runs many ranks per node;
- wins in **both** regimes -> the traffic model is wrong; stop and find out
  why before acting.

The third branch is the one that fired for the `multall` arm, and having
written it down in advance is what turned a surprising number into an
investigation rather than a victory lap.

---

## 9. Threats to validity: state what does not transfer

Every number here is **gfortran 14.2 on a Xeon E5-2640 v3 (Haswell)**.
Production runs ifort on Sapphire Rapids. Concretely:

- Sections 17/19 measured gfortran ~4.7x slower than ifort on this kernel,
  with a different bottleneck mix. These numbers support a kill decision (a
  2x loss shows up under any compiler) but not a marginal one.
- **The headline result is at risk specifically.** Sapphire's AVX-512 divider
  is several times faster per element than Haswell's, so production's
  redundant divides cost far less there -- and section 20 measured the change
  that *introduced* that redundancy as a win on exactly that machine. The
  ranking may invert. It must be re-measured under ifort on sapphire.
- The 300k L2 hump is expected to be Haswell-only: sapphire's 2 MB L2 holds
  the same buffer out to ~6M cells.
- L2 is 256 KB/core here; any conclusion phrased in terms of "fits in L2"
  carries that number with it.

**Keep rejected kernels.** Section 22's precedent: a bitwise-or-near-bitwise
rejected variant is retained for re-measurement, because the ruling can differ
on other hardware. Section 25's arms are retained for exactly the reason
section 26 then demonstrated -- and section 27 then demonstrated again, in the
other direction.

**The build configuration does not transfer either.** The -36% from pinning
the unit-level inline budgets is almost certainly gfortran-only:
`INTEL_FLAGS` already carries `-inline-factor=10000`, which scales ifort's
size limits including `-inline-max-per-compile`. The check on the production
machine is cheap and worth doing: build with and without that flag and diff
`set_residual`'s fingerprint. Identical means ifort was never budget-limited
and the win is ours alone; different means there is a real production win
sitting untouched.

---

## 10. Reproducing this study

```bash
PIN="--param=inline-unit-growth=1000000 --param=large-unit-insns=1000000 \
--param=large-function-growth=1000000 --param=large-function-insns=1000000"

# build the arms you want, with the budgets pinned so the baseline is
# invariant, plus the link-stage vectorization report
EMBER_ARMS=all EMBER_FFLAGS_EXTRA="$PIN" \
EMBER_MARCH="-march=native -mtune=native" \
EMBER_OPT_REPORT=tools/opt_report_staged.txt make compile

# Rule 9 gate: the baseline must fingerprint identically to the build you are
# comparing against. If it does not, stop -- the comparison is not valid.
uv run python tools/codegen_gauge.py set_residual_

# Gate 2 (fast), then goldens and the full suite
taskset -c 0 uv run python tools/bench_residual_staged.py \
    --mode serial --ncell 300000 --check-only
uv run pytest tests/test_residual_golden.py -q
uv run pytest tests -q

# every arm, 16 ranks, 10 interleaved launches, barriered (~30 min)
./tools/run_all_arms.sh

# one arm on its own (~4 min), and how many reps a launch needs
RESULTS=tools/bench_x.jsonl ./tools/run_prod_baseline.sh 10 16 1000000 30 nodal
taskset -c 0 uv run python tools/bench_rep_convergence.py --arm prod --reps 2000
```

The old ladder (`run_residual_staged.sh`) is kept only to reproduce the
superseded sections 25/26 numbers. **Do not use it for new rulings**: it
interleaves arms in one process and free-runs the ranks.

Artifacts:

| file | role |
| --- | --- |
| `src/ember/_fortran/residual_staged.f90` | `staged` and `split` arms |
| `src/ember/_fortran/residual_multall.f90` | faithful `multall` arm |
| `src/ember/_fortran/residual_multall_aos.f90` | `tbaos`: multall design on ember's AoS `dA` |
| `src/ember/_fortran/residual_nodal.f90` | `nodal`: fused sweep reading nodal primitives |
| `src/ember/_fortran/residual_prod_soa.f90` | `prodsoa`: production on SoA geometry (bitwise) |
| `src/ember/_fortran/residual_rinv.f90` | `rinv`: `1/r` staged as geometry |
| `tools/bench_residual_staged.py` | gates, kwargs plumbing (**superseded for timing**) |
| `tools/run_residual_staged.sh` | old ladder driver (**superseded**) |

**Corrected instrument** -- use these, not the two marked superseded:

| file | role |
| --- | --- |
| `tools/bench_prod_baseline.py` | one arm per process, rank barrier, launch-replicated |
| `tools/run_prod_baseline.sh` | launch-repeat driver for a single arm |
| `tools/run_all_arms.sh` | launch-outer/arm-inner sweep over every arm |
| `tools/codegen_gauge.py` | machine-code fingerprint; the gate for cross-build A/B |
| `tools/bench_rep_convergence.py` | how many reps a launch actually needs |
| `tools/run_harness_isolation.sh` | arm-set sweep that exposed the harness fault |
| `tools/bench_all_arms.jsonl` | the corrected results in section 10.1 |

Two of those are **not tracked**, and are regenerated rather than committed:
`tools/opt_report_*.txt` (rebuilt by `EMBER_OPT_REPORT=... make compile`,
matching the existing convention for `opt_report_damp.txt`), and the PDF
(`*.pdf` is gitignored repo-wide; `run_residual_staged.sh` regenerates it
from the committed jsonl in seconds).

Note that `tools/bench_residual_variants.py` and `tools/bench_setdamp.py` are
**stale**: both call `set_residual` without `dt_vol`/`dampin` and predate the
change-limiter fold. Do not copy their kwargs dict without fixing it.

---

## 10.1 The corrected result, and what it cost to get

At 1M cells, 16 ranks, 10 interleaved launches per arm, 30 reps, barriered,
one fingerprint-verified binary with the inline budgets pinned:

| arm | ns/cell | half-range | vs `prod` | section 26 said |
| --- | --- | --- | --- | --- |
| **`prod`** | **46.342** | 1.01% | -- | -- |
| `prodsoa` | 48.790 | 0.89% | +5.3% | (not built) |
| `rinv` | 49.292 | 0.78% | +6.4% | (not built) |
| `nodal` | 51.592 | 0.52% | +11.3% | (not built) |
| `multall` | 76.726 | 0.12% | **+65.6%** | **-17.7%** |
| `tbaos` | 92.024 | 0.20% | +98.6% | (not built) |
| `staged` | 94.955 | 0.25% | +104.9% | +11.1% |
| `split` | 97.550 | 0.42% | +110.5% | +49.5% |

Ratios carry ~0.9%, so every gap is resolved many times over. Reading:

- **Ember's fused single-pass residual wins, and not narrowly.** The paper
  question "why is multall competitive?" needs no exotic answer on this
  hardware: with both kernels compiled properly and measured on an instrument
  that does not leak, it is not competitive here.
- **Section 20 is vindicated three times.** Every attempt to undo it loses:
  `nodal` +11.3%, and both cheaper variants proposed as "the smallest possible
  production change" lose too -- `rinv` +6.4% (staging `1/r` costs more
  traffic than the reciprocal chain costs; the disassembly had already shown
  `-Ofast -freciprocal-math` compiles those divides to `vrcpps`, so section
  26.4's "~13 ns/cell of divider pressure" never described this binary) and
  `prodsoa` +5.3% (SoA geometry hurts a fused sweep, which reads all three
  components in one expression).
- **Nothing tested beat production.**

Still open: this is 1M only, and section 26's largest claims were at 100k and
300k where `multall`'s staged volumes go L3-resident. Those sizes need the same
treatment before the reversal is called general. And it remains gfortran on
Haswell.

---

## 11. Checklist

Before building an arm:

- [ ] What has the incumbent co-optimised with the thing I am changing? Does it move too? (Rule 1)
- [ ] Is there an attribution control that separates the two changes I am bundling?
- [ ] What am I deliberately not reproducing, and is that written in the kernel header?

Before trusting a gate:

- [ ] Does the test state exercise every term, or is it degenerate? (Rule 5)
- [ ] Does any direct state write need a cache invalidation? (Rule 6)
- [ ] Is there a component whose bitwise agreement proves something specific?
- [ ] Are deviations quantified in ulps of the quantities differenced?

Before trusting the instrument (new -- this is where two studies died):

- [ ] Is the **baseline** compiled the same in every build being compared?
      Fingerprint it, don't time it (Rule 9)
- [ ] Did a Gate 1 fix give one arm hand-forced inlining the incumbent lacks?
- [ ] Are the ranks **barriered** before every timed call (Rule 10)?
- [ ] Is the replication at the **launch**, with the spread across launches
      quoted as the error bar (Rule 11)?
- [ ] Right estimator for the regime: `min` serial, `median` contended
      (Rule 12)?
- [ ] Is the quoted precision actually achievable? Measure it on the baseline
      before quoting any difference smaller than it

Before reporting a number:

- [ ] Link-stage vectorization report, not compile-stage
- [ ] Flags fixed; benchmark sources excluded from the default build
- [ ] Size ladder, not a single size
- [ ] Contended regime named honestly, including which sockets and whether
      SMT siblings are idle
- [ ] Anything anomalous re-run, densified, and re-measured without the harness
- [ ] Predictions scored, including the failures
- [ ] Machine and compiler stated, with what does not transfer

Before generalising:

- [ ] Does the bounding argument state its quantifier? (Rule 2)
- [ ] Is the traffic model's bandwidth-bound premise actually true here? (Rule 8)
