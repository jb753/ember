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

- First answer (section 25): the five-pass design loses by 15-27%. **Wrong.**
- Second answer (section 26): the *faithful* five-pass design wins by 13-50%.

Same machine, same build, same harness, same protocol. The difference was
entirely in what the arms were allowed to change.

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
- **All arms in one `.so`.** `setup.py` globs the `_fortran` directory, so a
  new `residual_*.f90` is picked up automatically. This is the single most
  valuable structural choice in the harness: arms are compared **within one
  process, round-robin interleaved**, so there is no cross-build LTO drift, no
  gauge-kernel correction, and a surprising result can be re-checked against
  the exact binary that produced it.
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

### Contention without a job scheduler

This machine is not in a SLURM cluster, so the contended arm is N background
processes pinned with `taskset`, rendezvousing on a shared wall-clock start
time (`EMBER_BENCH_START`) rather than an MPI barrier. Ranks are independent
processes with no communicator; a spin-until-T gate is enough to overlap the
timed windows and avoids an mpi4py dependency. Rank index arrives by
environment variable in place of `SLURM_PROCID`.

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

### Stabilising

- `OMP_NUM_THREADS=1`; pin with `taskset`.
- >= 10 warmup calls -- this also first-touches staged arrays, keeping page
  faults out of the timed window.
- 50 reps, report **median and min** ns/cell, plus the spread across ranks.
- Interleave arms round-robin within each rep so thermal and frequency drift
  hits all arms equally.
- Restore any input an arm mutates, **untimed**, between reps. (Not needed
  here: every `dU` element is assigned before it is read.)

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
section 26 then demonstrated.

---

## 10. Reproducing this study

```bash
# build all arms into one .so, with the link-stage vectorization report
EMBER_MARCH="-march=native -mtune=native" \
EMBER_OPT_REPORT=tools/opt_report_staged.txt make compile

# Gate 2 only (fast)
taskset -c 0 uv run python tools/bench_residual_staged.py \
    --mode serial --ncell 300000 --check-only

# goldens, then full suite
uv run pytest tests/test_residual_golden.py -q
uv run pytest tests -q

# full ladder: serial + 6-rank socket-contended, four sizes, aggregate + PDF
./tools/run_residual_staged.sh 6 100000 300000 1000000 2000000
```

Artifacts:

| file | role |
| --- | --- |
| `src/ember/_fortran/residual_staged.f90` | `staged` and `split` arms |
| `src/ember/_fortran/residual_multall.f90` | faithful `multall` arm |
| `tools/bench_residual_staged.py` | gates, kwargs plumbing, interleaved timing |
| `tools/run_residual_staged.sh` | local (no-SLURM) ladder driver, aggregation, plot |
| `tools/bench_residual_staged.jsonl` | raw per-rank results |
| `tools/bench_staged_dip.jsonl` | dense serial sweep isolating the L2 hump |
| `tools/bench_residual_staged.pdf` | ns/cell vs block size, all arms, both regimes |

Two of those are **not tracked**, and are regenerated rather than committed:
`tools/opt_report_*.txt` (rebuilt by `EMBER_OPT_REPORT=... make compile`,
matching the existing convention for `opt_report_damp.txt`), and the PDF
(`*.pdf` is gitignored repo-wide; `run_residual_staged.sh` regenerates it
from the committed jsonl in seconds).

Note that `tools/bench_residual_variants.py` and `tools/bench_setdamp.py` are
**stale**: both call `set_residual` without `dt_vol`/`dampin` and predate the
change-limiter fold. Do not copy their kwargs dict without fixing it.

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

Before reporting a number:

- [ ] Link-stage vectorization report, not compile-stage
- [ ] All arms in one `.so`, interleaved, flags fixed
- [ ] Size ladder, not a single size
- [ ] Contended ranks pinned to one socket, regime named honestly
- [ ] Anything anomalous re-run, densified, and re-measured without the harness
- [ ] Predictions scored, including the failures
- [ ] Machine and compiler stated, with what does not transfer

Before generalising:

- [ ] Does the bounding argument state its quantifier? (Rule 2)
- [ ] Is the traffic model's bandwidth-bound premise actually true here? (Rule 8)
