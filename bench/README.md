# Benchmarking ember's hot kernels

This directory holds the harness for A/B-benchmarking `set_residual` and its
neighbours (`set_visc_force`, `set_tau_q_soa`, the IRS smoother, the RK
multigrid path), plus a condensed record of what was found the hard way. It
replaces `docs/dev/kernel_benchmark_methodology.md` and
`docs/dev/viscous_kernels.md`, which grew to ~3000 lines documenting a
benchmarking story that twice reached the wrong answer before the instrument
itself was fixed. Read **"Dos and don'ts"** below before trusting any number
you produce here, then **"Kernel optimization history"** for what's already
known.

## Performance tuning for a new CPU

The hot kernels carry blocking constants tuned against a specific cache
hierarchy and vector width. They are correct on any machine — every one of
them is bitwise-neutral, affecting only the order in which work is done — but
a value tuned for one CPU can cost tens of percent on another. Re-sweep them
when moving to a new architecture, and do it **in the real build** (`make
compile`, which compiles the whole `_fortran/` tree with `-flto
-fwhole-program`); a standalone single-file compile has inverted results here
before.

| constant | where | what bounds it |
| --- | --- | --- |
| `IRS_BJ = 32` | `_fortran/residual.f90` | **Vector-chain count, then nothing.** Sets how many independent lanes the i-solve's recurrence runs over. Below 32 it goes latency-bound (16 costs +19%); above 32 it is flat (64 and 128 are within ±2%), so the tile spilling L1d no longer matters. The least fussy of these constants — but only because the transpose is blocked; it was a sharp L1d-capacity optimum before that. |
| `IRS_TB = 8` | `_fortran/residual.f90` | **SIMD lane count** (8 for AVX2, 16 for AVX-512). Edge of the transpose block, so one staged row is exactly one vector load. Steeply optimal — 4 and 16 are both large regressions on AVX2. |
| `IRS_W = 64` | `_fortran/residual.f90` | **L2 capacity.** The i-strip carried through the fused j+k solves is `IRS_W*(nj-1)*(nk-1)*4` bytes — 917 KB on a 273×65×57 block, inside a 2 MB L2. Also sets the vector loop length, so it trades off in two directions at once. |
| `_KB_SLAB = 8` | `grid.py` | **Nothing, any more -- do not sweep it.** It was the k-slab depth for both kernels, but the fused rewrites of `set_residual` and `set_visc_force` subsumed slab blocking: what is left of the slab loop is a pure re-nesting of `do k = 1, nk-1`, and `set_visc_force` consumes `kb` only as a sanity guard. Measured at 1M cells: `kb` in 2..nk-1 is bitwise identical and within 1% on time. The panel constants above are what bound the working set now. Kept as a dummy so the bench arms share one kwargs dict with production. |
| `RES_JAREA = 4400` | `_fortran/residual.f90` | **L2 capacity.** j-panel AREA (in cells) of `set_residual`'s k walk, the counterpart of `VISC_JAREA` below and swept the same way. The carry it bounds is the rolling k-face plane pair, `ni*njp*5*2*4` bytes. |
| `RES_JMIN = 16` | `_fortran/residual.f90` | **Redundant-work floor, not a cache bound.** A panel recomputes its own lowest j-face row, so `1/jbw` of the j-face flux work is wasted; on a long-i block the area rule alone drives `jbw` to single digits and that overhead reaches several percent. Floor 16 is worth about 1.8% over no floor at 2M cells, in BOTH regimes. Do not raise it to 32: that fits fewer panels in L2 and costs -33% -> -21% contended at 2M, for 1.5% serial. |
| `WALL_TW = 64` | `_fortran/viscous.f90` | **L1d capacity, loosely.** i-tile the wall-function row forms carry their phase-A temps in. Eleven `WALL_TW`-float stack arrays, so 64 is under 3 KB and stays L1-resident across the three phases. Flat over a wide range -- it only has to be long enough to amortise the loop overhead and short enough not to spill. |
| `VISC_JAREA = 4400` | `_fortran/viscous.f90` | **L2 capacity.** j-panel AREA (in cells) of `set_visc_force`'s k walk: the panel is `VISC_JAREA/ni` rows deep, so the carry it bounds -- the rolling `planes` pair and one tau/q plane -- is a fixed number of BYTES whatever the block's aspect ratio, which a fixed row count would not be. 4400 cells puts both inside a 256 KB L2. Re-sweep on a machine with a different L2, and see the adopted entry for why the win is mostly a contended one. |
| `njp` pad rule | `grid.py` | **Page size** -- but worth almost nothing here now; see the negative result below before spending time on it. Pads the rolling face-flow plane by one j-row when `ni*nj` is a multiple of 1024 (= 4096-byte page / 4-byte float), to dodge a 4K-aliasing penalty at power-of-two plane sizes. Kept because it is one line and costs nothing to keep, not because this machine can measure it. |

Build flags matter as much as the constants:

- `EMBER_MARCH` defaults to `-march=haswell` for portable wheels. Set
  `EMBER_MARCH="-march=native -mtune=native"` when building for one machine.
- The pinned GCC inline budgets in `setup.py`
  (`inline-unit-growth`, `large-function-growth`) are worth **−53% serial** on
  `set_residual` — without them its face helpers are simply not inlined. Do
  not drop them. Both are needed; `inline-unit-growth` alone is worth only
  2.3% but is necessary for the other to bind.
- **Do not enable PGO** (`EMBER_PGO=use`). On this code it is a **+169%
  regression** — profile data makes GCC un-inline exactly the helpers whose
  inlining is worth the 53%.

`bench/README.md` documents the benchmarking harness and the protocol these
numbers were measured under, including several ways of measuring them that
produced confidently wrong answers.
## Why this harness looks the way it does

The `set_residual` pass-structure question ("multall/multall evaluates the
five residuals in five passes; ember does it in one fused sweep -- which is
faster, and why is ember's kernel competitive?") was answered three times:

1. First: the five-pass design loses by 15-27%.
2. Second: the *faithful* five-pass design wins by 13-50%.
3. Third, after fixing the harness and the build: the five-pass design loses
   by **65%**, and every other candidate loses too.

Two independent faults, either alone enough to invert a ranking:

- **The baseline was compiled wrong.** The build is `-flto -fwhole-program`,
  and GCC's inline budgets (`inline-unit-growth`, `large-unit-insns`,
  `large-function-growth`) are **unit-level**: adding benchmark files to
  `_fortran/` silently grew the inlining budget and changed production's own
  codegen. Compiled properly, `set_residual` runs **73.2 -> 46.7 ns/cell**
  (see "Adopted" list below) -- every earlier arm was racing a handicapped
  production.
- **The harness leaked.** Round-robin interleaving of all arms in one process
  made a ranking depend on which *other* arms were in the build (a 16-point
  swing at fixed binary/size/rank count), and free-running ranks made the
  contention state an unrecorded variable.

At 1M cells / 16 ranks, on the corrected instrument, production beats every
arm tested:

| arm | ns/cell | vs `prod` |
| --- | --- | --- |
| **`prod`** | **46.342** | -- |
| `prodsoa` | 48.790 | +5.3% |
| `rinv` | 49.292 | +6.4% |
| `nodal` | 51.592 | +11.3% |
| `multall` | 76.726 | +65.6% |
| `tbaos` | 92.024 | +98.6% |
| `staged` | 94.955 | +104.9% |
| `split` | 97.550 | +110.5% |

Ember's fused single-pass residual wins, and not narrowly. This is gfortran
on a Haswell workstation, not the production ifort/Sapphire target -- see
"Threats to validity" below.

## What's in this directory

| file | role |
| --- | --- |
| `residual_arms.py` | Shared library: grid/state setup (`build_case`), the scratch-carving kwargs builder (`build_kwargs`), one callable per arm (`callers`), the non-degenerate correctness state (`swirl`), the correctness gate (`check_correctness`), and an LLC-flush helper. Import this rather than copying any of it. Two of its arms, `padnjp` and `alias4k`, are not variant kernels at all: they call production with a different rolling-plane j-extent, to price the `njp` pad rule against the aliasing it prevents (both are bitwise identical to `prod` by construction). Run standalone (`uv run python bench/residual_arms.py --ncell 300000`) for a fast Gate-2 correctness pre-flight; it does **not** time anything -- that's `bench_prod_baseline.py`. |
| `bench_prod_baseline.py` | The corrected instrument: one arm per process, rank-barriered before every timed call, replication at the launch. `--analyze` summarises a results file. Normally driven by the two scripts below rather than invoked directly. |
| `run_prod_baseline.sh` | Launch-repeat driver for a **single** arm (default `prod`). `bench/run_prod_baseline.sh [launches] [nranks] [ncell] [reps] [arm]`. |
| `run_all_arms.sh` | Launch-outer/arm-inner sweep over every arm, on one fingerprint-verified binary (`EMBER_BENCH_KERNELS=all` build with the inline budgets pinned). This is what produced the table above. |
| `bench_rep_convergence.py` | Answers "how many reps does one launch need?" via a contiguous-block bootstrap over a long single-process trace. Orthogonal question to launch-replication -- see its docstring. |
| `codegen_gauge.py` | Fingerprints a compiled symbol's machine code (recursive call closure, layout-normalised, hashed). The gate for any cross-build comparison: **`prod` must fingerprint identically between the builds you're comparing, or the comparison is meaningless** (Rule 9 below). |
| `run_flag_sweep.sh` | One-factor-at-a-time compiler flag sweep, fingerprint-gated so a no-op flag is never timed. `MODE=serial\|contended`, `CONFIGS="name1 name2"` to run a subset. |
| `irs_arms.py` | The IRS counterpart of `residual_arms.py`, for the implicit residual smoother (`smooth_residual_tri_tiled`) rather than `set_residual`. Same shape: `callers_irs`, a bitwise gate for the arms that compute the exact operator (`check_correctness`), a quantified gate for the ones that deliberately approximate it (`check_jacobi`), and `callers_update` for the `set_residual`+IRS pair, whose fusion spans both kernels. Also `check_denormals`, because the smoother runs in place and repeated reps compound. Drive it with `bench_prod_baseline.py --kernel irs` (or `--kernel update`); run standalone for a Gate-2 pre-flight. |
| `visc_arms.py` | The viscous counterpart, for `set_visc_force` (`--kernel visc`) and `set_tau_q_soa` (`--kernel tauq`). Same shape as the two above, plus `halo_restorer`: `set_visc_force` is **not idempotent** -- its entry pass scales the tau/q halo slots by `(2*wall-1)` in place -- so a repeated-rep instrument must restore those six faces (O(surface)) between calls. On the duct case that turns out to be insurance rather than a repair, for two reasons that are accidents of the case; see the module docstring before assuming it can be dropped. |
| `run_visc_baseline.sh` | Production `set_visc_force` over the size ladder in the 8-rank socket-contended regime, plus one `set_tau_q_soa` point for the phase split. |
| `run_so_ab.sh` | A/B of two or more BUILDS of the same production kernel, swapping `src/ember/fortran*.so` between launches. The instrument for changing production itself, where freezing a copy of the old kernel as an arm would put two near-duplicate bodies in one translation unit and have their inline budgets starve each other (Rule 9's +56% swing). Launch-outer, build-inner, so drift cannot alias onto one build; `bench_prod_baseline.py --label` records which build a row came from and the paired-within-launch analysis differences them inside a launch. Build each candidate with `make compile` and copy the `.so` aside first. |
| `visc_arms.py` (`callers_pair`) | `--kernel viscpair` times the two viscous phases as a UNIT (`unfused` = `set_tau_q_soa` then `set_visc_force`; `fused` = the experimental single call), because a fusion spanning both kernels cannot be attributed to either. Same shape as `irs_arms.callers_update`. Its gate covers `mu_turb` as well as `fvisc` -- a fused kernel producing right forces from a wrong mixing length would pass an `fvisc`-only gate -- and asserts both arms are idempotent. |
| `rk_arms.py` | The RK stage counterpart, for `advance_rk_stage_mg`'s kernels (`--kernel rk`). Its arms are CONFIGURATIONS of production rather than competing implementations: `rk` (`rk_mg_irs`, the solver defaults -- 3 coarse levels, coarse-IRS on), `rknoirs` (`rk_mg_noirs`, what `sf_resid=0` dispatches) and `rkplain` (`rk_plain`, the `n_levels=0` path), so the differences price the coarse smoother and the whole restrict/prolong machinery. They compute different things, so there is no cross-arm value gate; the gate is IDEMPOTENCE -- each kernel reads the frozen step-top snapshot and writes `cons`, so a repeated rep must reproduce the first bitwise or the reps are not timing one operation. Run standalone for that check. |
| `subroutines/` | The non-production Fortran arms themselves (see below) -- not compiled by default, only via `EMBER_BENCH_KERNELS`. |
| `results/` | Tracked `.jsonl` result files from the corrected instrument (`bench_all_arms.jsonl`, `bench_prod_baseline.jsonl`, `bench_flagsweep_{serial,phase2}.jsonl`) plus wherever you point `--json`/`--out`/`RESULTS`/`OUT`. Untracked `.pdf`/`.npz` plots regenerate from the jsonl in seconds; don't commit them. |

The production kernel this harness exercises, `set_residual`, lives in
`src/ember/_fortran/residual.f90` and is always built. Its non-production
arms (`residual_staged.f90`, `residual_multall.f90`, `residual_multall_aos.f90`,
`residual_nodal.f90`, `residual_prod_soa.f90`, `residual_rinv.f90`,
`residual_consa.f90`, `residual_irs_dirs.f90`, `residual_irs_km.f90`,
`perfect.f90`, and others) live in `bench/subroutines/` instead of
`src/ember/_fortran/`, so the default build's glob never sees them at all --
no denylist to keep in sync, and no cross-file provider-ordering problem for
`tools/check_compile.sh`'s pre-commit syntax check (which only ever globs
`src/ember/_fortran/`). Build a kernel back in with
`EMBER_BENCH_KERNELS=<name>[,<name>...]` or `EMBER_BENCH_KERNELS=all`; any
`.f90` file dropped into `bench/subroutines/` is automatically selectable by
name (`setup.py`'s `select_bench_kernels()` resolves its `use` dependencies
within that directory too, so e.g. asking for `multall` also pulls in
`residual_staged.f90` for `scale_du_all`). `make compile` itself (f2py's
meson backend) does real dependency-graph resolution regardless of file
order, so nothing in `bench/subroutines/` needs a naming convention either --
name new files however reads best.

**Arms are snapshots, and a breaking change in production does not oblige you
to update them.** An arm exists to answer one question at one moment; once that
question is answered its value is as a record of what was measured, not as
running code. So when a production kernel changes shape -- an argument promoted
from a scalar to a field, a parameter dropped, a formula moved -- the arms under
`bench/subroutines/` that share its helpers may stop compiling, and that is an
acceptable state for them to be in. Nothing in the default build, the tests or
CI touches them, so the breakage costs nothing until someone asks for that arm
by name with `EMBER_BENCH_KERNELS`. Fix an arm when you next want to *run* it,
against whatever production looks like then, and skip it otherwise: porting a
change into an arm nobody is about to time is work spent on a comparison that
may never be made, and it quietly rewrites what the arm measured.

## Dos and don'ts

Rules earned from two studies that reached the wrong conclusion before this
list existed. Numbering matches the original methodology document so old
cross-references still resolve.

**Building and comparing:**

- **Build the real way.** `make compile` (`uv pip install -e .`), which f2py
  compiles every `src/ember/_fortran/*.f90` together with `-flto
  -fwhole-program`. A standalone `gfortran one_kernel.f90` does not reproduce
  whole-program IPA and has inverted a result before (component-first tau/q
  layout looked +14% faster in an isolated 2-file build, +42% *slower* in the
  real build).
- **Arms are not all built into one `.so`.** Putting every arm in the same
  program perturbs each other's compilation (unit-level inline budgets) and,
  if timed round-robin in one process, perturbs each other's cache/phase too.
  `setup.py` excludes `bench/subroutines/` by default; `EMBER_BENCH_KERNELS=nodal,multall`
  builds specific ones back in for a targeted comparison.
- **Rule 9 -- fingerprint the baseline, don't time it.** `codegen_gauge.py`
  hashes a symbol's normalised machine code. Launch-to-launch timing noise is
  ~0.4%, the same order as effects under investigation; codegen identity is
  exact. Verify `prod` fingerprints identically across any two builds you
  compare, every time.
- **Corollary.** If a vectorization-gate failure is fixed by *hand-inlining
  in the candidate only* (not in production), the candidate now has forced
  inlining the incumbent lacks -- a codegen fix applied to one arm is a thumb
  on the scale. (`nodal` measured -4.4% that way, +11.3% once both kernels
  were inlined by the same rule.)
- **Rule 1 -- identify what the incumbent has co-optimised, and vary it
  together.** Before building an arm, ask what else in the incumbent exists
  *because of* the thing you're changing. A five-pass kernel bolted onto
  ember's own nodal representation and AoS geometry is a hybrid neither code
  would write, and it is worse than either real design -- measuring it and
  calling the result "the multall design" was the original section-25 error.
- **Rule 2 -- never generalise from one arm to a family.** "This kernel is
  strictly cheaper than the literal form, so it bounds the whole family" is
  true only at a fixed nodal representation. State the quantifier.
- **Rule 3 -- an input may be prepared outside the timed region iff a real
  port would also prepare it outside the per-step loop.** Grid geometry
  (face areas) is built once and never touched again: fair to hoist.
  Solution-dependent quantities are recomputed every call even if only their
  *storage* is preallocated.
- **Rule 4 -- trace any "free" input to its real production consumer.** If an
  arm reads arrays a real kernel wouldn't normally have, cite the call site
  that proves production already materialises them (or don't build the arm).
- **Rule 8 -- a traffic model ranks arms only when the kernel is actually
  bandwidth-bound; check that premise before trusting the ranking.** An arm
  that moves *more* bytes and still wins is telling you the premise is false.
  (`multall` moved ~2.2x production's bytes and still won under the leaky
  harness -- because on that instrument the ranking was wrong for unrelated
  reasons, but the diagnostic is real: production turned out to be
  divide-limited in one study and memory-bandwidth-limited in another,
  depending on compiler and concurrency. Measure, don't assume.)
- **Rule 13 -- never suppress a build diagnostic to make a comparison run.**
  A `-Wno-missing-profile` added so unexercised PGO translation units
  wouldn't trip `-Werror` silently hid "no profile data found on *every*
  file" for an entire study; the reported "+0.6%, no effect" was actually
  `-fprofile-use` with zero training data. Re-run properly, PGO turned out to
  be a **+169% regression**. If you suppress a diagnostic, verify the thing
  you suppressed.

**Correctness gates (run before trusting any timing number):**

- **Gate 0 -- build.** `-Wall -Werror -Warray-temporaries -Wfatal-errors`.
  `-Warray-temporaries` matters most: an array-slice-to-explicit-shape-dummy
  copy would dominate a measurement silently. Prefer contiguous slices.
- **Gate 1 -- vectorization, from the link-stage report.**
  `EMBER_OPT_REPORT=<path> make compile` injects `-fopt-info-vec-all` at
  link time, where the real whole-program codegen happens. A compile-stage
  report describes discarded per-TU codegen and reports spurious misses.
  Check every innermost `i` loop vectorized; outer `j`/`k`/`m` misses are
  normal. A "vectorized" loop can still be gather/scatter-based rather than
  unit-stride -- the report doesn't distinguish them; disassemble
  (`objdump -d`, or read `codegen_gauge.py`'s `mix` output) if a result is
  surprising.
- **Rule 5 -- before trusting a gate, ask which terms of the kernel the test
  state actually exercises.** `build_duct_grid` is axially straight
  (`Vr = Vt = 0`), so the j- and k-face mass fluxes are *identically* zero
  there -- a wrong j/k-face kernel would pass that gate silently. Use
  `residual_arms.swirl()` (seeds cross-stream momentum to ~5% of axial) to
  make every term in every helper non-degenerate before gating.
- **Rule 6 -- any direct write to solver state must be followed by the cache
  invalidation the solver would do.** `swirl()` writes `conserved_nd`
  directly; arms that re-derive primitives from `cons` (production) don't
  notice a missing invalidation, but arms that read cached nodal arrays
  (`vx`/`vr`/`vt`/`ho`) will silently solve a *different* state without
  `block.update_cached_conserved()` afterward -- a phantom "bug" that isn't
  one.
- **On exactness.** Aim for bitwise where you can prove it, quantify the rest
  instead of hand-waving. A component whose agreement is bitwise for a
  *structural* reason (e.g. the mass residual is a pure sum of staged mass
  fluxes) proves something specific about the arms being identical there --
  call that out explicitly. Quantify remaining deviations in **ulps of the
  quantities being differenced**, not just relative to the final result (a
  residual is a small difference of large face flows, so relative-to-result
  overstates error badly).

**Measurement protocol:**

- **Two regimes: serial (`taskset -c 0`) and contended (N pinned ranks, one
  socket).** Pin all ranks to one socket -- spread across sockets they don't
  actually contend and a bandwidth-heavy arm gets flattered. Name the regime
  honestly (e.g. "6-rank socket-contended"), never "saturated" unless it is.
- **Rule 10 -- synchronise ranks before every timed call, not just at
  startup.** A shared wall-clock start time is not enough: free-running
  ranks drift out of phase, so each is timed against a different, unrecorded
  mixture of what its neighbours are doing (a 16-point swing was traced to
  exactly this). `bench_prod_baseline.py`'s `Barrier` is a lock-free
  shared-memory rendezvous, cheap enough (microseconds) to use before every
  call, not just once.
- **Rule 11 -- replicate at the LAUNCH, not the rep.** N reps in one process
  are N correlated views of a single draw of page placement, allocation
  alignment, core assignment and thermal state. Per-rep scatter is ~3% but
  right-skewed with rare severe outliers (p99 +8.6%, p100 +49% in a 2000-rep
  trace); the thing that actually varies launch-to-launch is a *different*
  ~3% draw each time. Keep reps modest (`bench_rep_convergence.py` will tell
  you how many you actually need -- usually far fewer than assumed) and
  repeat the **launch** 5-10 times.
- **Rule 12 -- `min` for serial, `median` for contended.** Under contention,
  `min` preferentially samples the instants when neighbouring ranks happened
  to be *between* calls, erasing the contention the regime exists to
  measure (ranking one arm on `min` turned a real -1% into a fabricated
  -18%). `min` is also biased low with a bias that grows with rep count, so
  only compare mins at equal rep count.
- **Sweep sizes, always, and report ns/cell.** A kernel's ranking changes as
  its working set crosses L2/L3. Sweep at least 100k/300k/1M/2M cells and
  plot the curve (to PDF, never PNG) -- a win that decays monotonically with
  size is not a durable win, and an anomaly at one size (e.g. production's
  L2 hump around 300k on Haswell) is real information, not noise to average
  away.
- **Rule 7 -- a result that contradicts your headline gets re-run,
  densified, and re-measured with the harness itself removed as a suspect,
  before it is explained.** Concretely: (1) re-run the single point; (2)
  densify the size ladder around it; (3) time the "suspicious" kernel alone
  in its own process, with no other arm sharing the build or the run, and
  see if the anomaly survives.

**Threats to validity -- state what does not transfer:**

- Every number in this directory's history was measured on **gfortran on a
  Haswell workstation**. Production runs **ifort on Sapphire Rapids**. A
  ~2x loss under gfortran is a safe kill signal under any compiler; a
  marginal (single-digit-percent) result is not, and several results in the
  history below flipped sign between compilers or hardware generations
  (AVX2 vs AVX-512 divide/gather cost, L1/L2 size, `-inline-factor` size
  limits). Anything you plan to act on in production needs its own ifort/
  Sapphire measurement using this same harness before you trust it there.
- The build-flag wins below (inline-budget pinning) are `setup.py` defaults
  already; re-verify they still hold before changing `GFORTRAN_FLAGS`.

## Quick recipes

```bash
# Fast correctness pre-flight (no timing) before trusting anything below
uv run python bench/residual_arms.py --ncell 300000

# Fingerprint-verify the baseline before any cross-build comparison
uv run python bench/codegen_gauge.py set_residual_

# One arm, launch-replicated (the diagnostic number)
taskset -c 0 bench/run_prod_baseline.sh 10 1 1000000 30 prod

# Every arm, on one fingerprint-verified binary (needs EMBER_BENCH_KERNELS=all)
EMBER_BENCH_KERNELS=all EMBER_MARCH="-march=native -mtune=native" make compile
bench/run_all_arms.sh

# How many reps does a launch actually need?
taskset -c 0 uv run python bench/bench_rep_convergence.py \
    --ncell 1000000 --reps 2000 --arm prod

# One-factor-at-a-time compiler flag sweep
bench/run_flag_sweep.sh                       # serial screen
MODE=contended bench/run_flag_sweep.sh

# Viscous kernels: baseline ladder, then an A/B of the fvisc-fusion arms
bench/run_visc_baseline.sh 10 30
EMBER_BENCH_KERNELS=viscous_fused make compile
KERNEL=visc ARMS="visc viscijk viscpol2" NRANKS=8 NCELL=1000000 \
    RESULTS=bench/results/bench_visc_pol2_1000000.jsonl bench/run_all_arms.sh
```

---

## Kernel optimization history

Condensed record of what changed `src/ember/_fortran/viscous.f90` and
`residual.f90` (and the RK multigrid path in `scree.f90`/`distribute.f90`)
into their current shape, and what was tried and didn't work. Full
blow-by-blow investigation narrative, per-size A/B tables, and rejected
prototype code are not reproduced here -- recover them from git history
(`docs/dev/viscous_kernels.md`, deleted when this directory replaced it) if
you need the detail. This section is deliberately a summary: what changed,
the headline number, and where it lives.

### Adopted (shipped in production today)

- **Vectorized wall-function row forms**, `set_visc_force`
  (`wall_row_kface`/`wall_row_jface` in `viscous.f90`). The scalar
  `wall_func_kface`/`_jface` were not being inlined in the production build
  (5 symbols in the call closure, 15.7% of kernel time) and their i-loops
  could not vectorize anyway: `wall_core`'s Reynolds-number branch sits in the
  middle of the arithmetic, leaving 4 scalar divides and 2 scalar sqrt per
  wall face cell. Split into three phases over a `WALL_TW`-wide i-tile --
  face averages and `Re`, then the branch alone, then stress and flux --
  so only the middle phase stays scalar and the two that carry the divides
  and roots vectorize. Worth **-20.6% / -12.1% / -8.9% serial** at 300k/1M/2M
  cells; much less contended, because the wall work is compute-bound and
  DRAM contention masks it. Not bitwise: 0.625 ulp of the field scale,
  confined entirely to the two-cell shell the wall faces touch (the wall
  injection writes the SECOND face plane, not the first). The deviation is
  `vrcpps`+Newton vector reciprocals under `-Ofast`, not reassociation --
  `-fno-associative-math -ffp-contract=off` does not collapse it.
- **j-panel tiling of `set_visc_force`'s k walk** (`VISC_JAREA`). The fused
  walk carries a rolling `planes(ni,nj,4,2)` pair plus a tau/q plane across
  each k step; at 273x65x57 that is 568 KB and 639 KB, both past the 256 KB
  L2, so the carry round-tripped through L3 -- and at 8 ranks the socket's
  20 MB L3 is entirely consumed by it. Running the whole k walk over j-panels
  of fixed AREA bounds the concurrent working set (not the traffic) to L2.
  **-10.5% / -16.8% / -33.7% / -40.1%** at 100k/300k/1M/2M cells in the
  8-rank socket-contended regime, growing with size exactly as a capacity
  effect should; **bitwise identical** to the un-panelled kernel. Panels are
  floored at four rows wide because a panel duplicates its lowest j-face row,
  so `1/jbw` of the j-face work is redundant.

  Together the two are **-19.6%** on the viscous pair end to end
  (`set_tau_q_soa` + `exchange_halos` + `set_visc_force`) at 1M cells,
  8 ranks. Measured with `run_so_ab.sh`; results in
  `results/visc_jpanel_final_*.jsonl` and `visc_jpanel_pair_1000000.jsonl`.
- **Cascade ping-pong in `mg_coarse_correction`** (`scree.f90`), plus its
  four per-call `alloca`s. The coarsest-to-finest prolongation cascade
  copied its accumulator back to `acc0` after every hop, purely so the
  caller would find the result there; a perf profile put `memmove` at ~7% of
  the RK stage. The parity of the hop count says where the cascade ends, so
  choosing the STARTING buffer by that parity lands it in `acc0` with no
  copy-back. Separately, the level tables (`dib`/`djb`/`dkb`/`offc`) were
  automatic arrays sized by `n_levels`, i.e. four `alloca`s and a
  `stack_save`/`stack_restore` pair per call, with memory clobbers across
  the body; `ember.block.MAX_MG_LEVELS` is a compile-time bound, so they are
  now fixed-size. **-3.2 / -2.5 / -2.6% contended** and **-4.4 / -2.8 /
  -1.1% serial** at 300k/1M/2M, faster in 47 of 48 launches, and **bitwise
  identical at every `n_levels` of 1, 2 and 3** (both cascade parities).
  Removing the automatic arrays also removed a `__builtin_trap` GCC had been
  emitting in this routine -- it was the guard on the runtime-sized local.
- **Prolongation weights computed, not tabulated** (`scree.f90`). The three
  prolongations each built six index arrays and three weight arrays the
  length of the fine directions, per call -- six more `alloca`s, ~4.7 KB of
  stack at a 1M-cell block, recomputing pure geometry every call. The
  bracket is a closed form in the fine index, so j and k now evaluate it per
  outer-loop iteration (`mg_bracket2x`), and the contiguous i direction
  drops it entirely: `mg_interp_i2x` is driven by the COARSE index and emits
  both fine cells of each coarse pair, which turns the gathers the index
  arrays forced into contiguous reads. **-5.2 to -5.8% serial** across the
  ladder, 8 of 8 launches at every size; **-2.1% contended at 300k and flat
  at 1M/2M**, which is the signature of a compute win on a phase that
  contention makes bandwidth-bound. Not bitwise: at most 1 ulp of field
  scale, on 1 to 70 cells of a million (the constant-folded weights contract
  differently from the runtime ones). `scree.f90` now has **no `alloca` and
  no `__builtin_trap` at all**.

  Together with the ping-pong above, against the kernel as it stood:
  **-4.3 / -2.7 / -2.1% contended** and **-10.8 / -7.9 / -6.6% serial** at
  300k/1M/2M, faster in 48 of 48 launches. Results in
  `results/rk_pingpong_*.jsonl`, `results/rk_weights_*.jsonl` and
  `results/rk_final_*.jsonl`; the baseline ladder the three arms establish
  is in `results/rk_arms_*.jsonl`.
- **j-panel tiling of `set_residual`'s k walk** (`RES_JAREA`, `RES_JMIN`),
  the same lever as the viscous panel above, applied to the larger kernel.
  The carry here is the rolling k-face plane pair, `ni*njp*5*2*4` bytes --
  720 KB on a 273x65x57 block -- so plane `k`, written on step `k`, was
  evicted before step `k+1` read it. **-3.8% / -16.0% / -33.0%** at
  300k/1M/2M cells, 8 ranks on a socket, every launch of 10; neutral at
  100k (+0.8%, where the panel does not engage at all -- `jbw` covers the
  whole block -- so that is the loop's own codegen cost, not the tiling).
  The residual itself is **bitwise identical**; the only output that moves
  is the change limiter's block-mean `ravg`, whose summation order the
  panel changes, and it moves dU by 1e-7 to 1.6e-6 relative -- a few ulps
  of a scalar that scales the whole field.

  **Unlike the viscous panel, this one costs serial time**: -5.7% at 300k
  but **+2.3% / +2.2%** at 1M/2M. That is the redundant j-face row (see
  `RES_JMIN`), which serial has nothing to buy with it -- one rank owns the
  whole 20 MB L3 and the un-panelled carry fits. It is adopted on the
  contended regime because that is the one production runs in: every rank
  of a socket evaluating residuals at once is the normal case, and a 33%
  win there is not comparable to a 2% loss in a regime nobody runs.
  Results in `results/res_panel_final_*.jsonl`, sweep in
  `results/res_jarea_ab_*.jsonl`.
- **k-slab cache blocking**, `set_visc_force` and `set_residual`
  (`viscous.f90`, `residual.f90`). Both stream their nodal working set in
  `kb`-plane slabs instead of full-volume passes, so tau/q (or the residual's
  nodal fields) are fetched from memory roughly once instead of ~3x.
  -30% to -36% on `set_visc_force`, -13% to -20% end-to-end
  (`update_sources`); -3% to -18% on `set_residual` depending on size, with a
  conditional anti-aliasing pad (`njp = nj+1` iff `ni*nj` is a multiple of
  1024) to dodge a 4K-aliasing penalty at power-of-two plane sizes. Slab
  depth `ember.grid._KB_SLAB = 8` drives both kernels; re-sweep only if
  production blocks grow past ~1M cells or the target machine's L2/L3
  changes materially.
- **Rolling-buffer fusion**, `set_visc_force`. The per-direction
  `fvisc = flow(lo_face) - flow(hi_face)` pattern is fused into the face-flux
  loop with rolling buffers instead of a full slab-sized scratch array,
  removing its write+read round-trip. A further -8% to -16% on top of
  k-slab blocking. (The equivalent full-fusion rewrite of `set_residual` was
  tried, passed every correctness gate, and regressed +10% at the largest
  size tested -- rejected in that form; a bitwise-relaxed, still-slab-tiled
  port of the same idea *is* adopted for `set_residual`, see next.)
- **Slab-tiled rolling fusion for `set_residual`** (relaxed float32
  tolerance, ~1.2 ulp of the flux scale). Per-direction accumulation through
  rolling row/plane buffers inside the existing kb-slab sweep. -3% to -18%
  depending on size, no size where it loses.
- **Derive `vt_rel` inline instead of storing it as its own nodal array**
  (`residual.f90`). `Vt_rel = Vt - Omega*r` is two scalars ember already has
  in registers; storing it as a fifth streamed field was pure redundant
  traffic given every interior node is a corner of ~12 face evaluations.
  -0.4% to -14.5%, growing with size (more L3-pressured = more benefit).
- **Fuse the i- and j-direction sweeps in `set_residual`** into one
  per-`(j,k)`-row `dU` write instead of two separate full-slab passes,
  cutting `dU` from three touches per slab to two. -7% to -12%, no
  regression at any size, no golden regeneration (reassociation is
  sub-ulp).
- **Vectorize the coarse-grid IRS i-solve** in the RK multigrid path
  (`scree.f90`'s `rk_mg_irs`/`scree_mg_irs`): swap the untiled
  `smooth_residual_tri` for the already-existing, already-production-on-the-
  fine-grid `smooth_residual_tri_tiled`. Pure oversight recovery -- the
  transpose-tiled i-solve already existed and was already used elsewhere.
  Bitwise identical. -20% to -27% on the isolated coarse smoother at the
  dominant level, -1.6% to -6.8% on the full RK stage.
- **Fuse the final RK-multigrid prolongation hop with the cell->node
  scatter** (`mg_prolong2x_fine_scatter` in `scree.f90`): builds the
  increment one rolling k-plane at a time and scatters straight into `cons`
  instead of materialising the full-volume increment and re-reading it.
  Not bitwise (~1-2 ulp), no golden regeneration needed. -11% to -30% on the
  full RK stage. `scree`'s own (non-RK) path keeps the unfused route
  deliberately -- its in-place, history-rolling scatter doesn't fit this
  shape.
- **Derive `Vx`/`Vr`/`r*Vt` from `cons` instead of streaming them separately**
  (`residual_consa.f90` design, folded into production). `cons` already
  contains `rho*Vx` etc., so `Vx = cons2/cons1` is exact and free -- this
  drops 2 of 9 streamed nodal fields for one extra reciprocal per corner,
  paid ~12x redundantly rather than precomputed (a precompute buffer would
  write more than it saves, the same mistake the rejected box-filter rewrite
  below made). Flat serial (~0%), -4.5% saturated -- the signature of a real
  bandwidth reduction that only shows up once bandwidth is the constraint.
  **Deriving `P`/`ho` from `cons` too (the natural next step) was
  deliberately NOT done**: no kernel in `_fortran/` may reference
  `gamma`/`Rgas`/`T_dtm` directly. The equation of state lives entirely
  behind `_Fluid.get_P`/`get_h` (`fluid.py`) so other fluids can be added;
  `Vx = cons2/cons1` is the velocity's *definition* for any fluid, but `P`
  and `ho` are thermodynamic and must stay pre-evaluated inputs. This is an
  architectural boundary, not a performance one -- don't cross it chasing
  traffic.
- **Merge `damp_residual`'s ten full-volume sweeps into one traversal per
  `(j,k)` plane** (`residual.f90`), so `dt_vol` and each `dU` plane are read
  once per plane instead of five times. -28.7% serial, -30.8% saturated,
  winning on every rank measured -- the largest single kernel-level win
  found. Branch-free by construction (a flat-component guard is precomputed
  into a zeroed reciprocal rather than a `cycle` inside the sweep). **The
  first attempt put the component index innermost** ("one pass instead of
  ten") and was **3x slower** -- component-last data makes that stride
  `(ni-1)*(nj-1)*(nk-1)` elements apart and destroys i-vectorization
  entirely. Component index stays outside the spatial loops; only the
  spatial traversal is shared across components.
- **Collapse `fvisc` to a single store per cell in `set_visc_force`**
  (`viscous.f90`). Production visited `fvisc` four times per cell: the
  i-direction sweep assigned it, the j- and k-direction sweeps
  read-modify-wrote it, and the trailing polar-source pass RMW'd component 2.
  The rolling buffers to collapse that already existed -- nothing forced the
  three face differences into three separate visits except the order they were
  written in. Now a single walk over k face planes, with the i/j scan that
  closes cell plane k-1 summing all three differences and the polar source
  into one store. **-20.7% +/- 0.16% at 1M cells, 8-rank socket-contended,
  faster in 10/10 launches.** Staged arms attribute it: i-into-j fusion alone
  (4 touches -> 3) is -16.9%, adding k (-> 2) is -20.8%, adding polar (-> 1)
  is a further tie-to-win depending on size.
  **The k-slab loop disappears with the fusion** and `kb` becomes inert: slab
  blocking existed to keep a slab's tau/q hot across three separate direction
  sweeps, and once fused, face plane k reads halo planes k and k+1 while the
  i/j scan reads halo plane k, so a single k walk *is* the blocked schedule.
  Not bitwise: ~2.5 ulp of the fvisc field scale, spread evenly through the
  interior and absent at the wall edges (the rounding signature, not the
  wrong-index signature). Most of it is GCC reassociating one loop shape and
  not the other -- `-fno-associative-math` drops it to 0.12 ulp, and
  `-ffp-contract=off` changes nothing further. Every golden test passes
  unchanged.
  **The polar source cannot simply ride along in the fused store**: production
  adds it *after* the wall-zeroing pass, because it is a geometric source
  rather than viscous content and the wall mask must not eat it. Interior
  cells take it in the store; the boundary shell takes it in an O(surface)
  pass afterwards, partitioned so every shell cell is visited exactly once
  (the zeroing loops may overlap at edges because a repeated multiply is
  harmless, but a repeated *add* is not).
  **The first version of that shell pass cost 1.55%**, because the
  `i=1`/`i=ni-1` cells form a sheet at fixed i that can only be reached with
  stride `ni-1`: Gate 1 showed one such block gather-vectorized and the other
  not vectorized at all. The fix uses an asymmetry -- a row interior in j and
  k carries no j- or k-mask on its end cells, so those two cells can take
  their `walli1`/`wallni` mask *and* their polar source inside the fused
  store, where the row is in L1 and every access is unit-stride. The sheet
  then leaves the O(surface) pass entirely and the polar row loop becomes one
  unbroken vectorized `i = 1..ni-1`. **-1.55% +/- 0.12%, 10/10 launches**, and
  it is what makes the polar fusion free rather than costly at 1M.
- **Retune the IRS transpose-tile width, `BJ`, from 8 to 32** (one
  `parameter` in `residual.f90`). Production's `BJ=8` was tuned for AVX2 (8
  float32 lanes); Sapphire Rapids is AVX-512 (16 lanes) with a larger L1d.
  Bitwise identical for every width (BJ only groups independent j-lines).
  -26.4% serial / -8.1% saturated at BJ=32; **BJ=64 gives most of the win
  back** because its 68 KB tile spills Sapphire's 48 KB L1d. **The optimum is
  set by L1 capacity, not vector width** -- re-sweep on any new target
  hardware rather than assuming "match the SIMD lanes".
- **`set_residual_damp_split`**: move the change-limiter's scaling loop into
  its own contained subroutine so the compiler can no longer assume an
  output dependence between it and the main dU sweep. Fixes a
  `PARTIAL LOOP WAS VECTORIZED` blemish in the opt-report but is a *timing
  tie* (+/-0.2%, inside noise) -- adopted for codegen hygiene, not speed.
  Worth recording as a caution: not every opt-report blemish is a cost, and
  not every clean report is free of one (see the gather finding below).
  **Fusing damp into `set_residual` entirely (`set_residual_damp`) is NOT
  adopted**, despite an attractive standalone number: it reorders IRS vs.
  the change limiter's application (a real numerics change, no timing result
  addresses it) and its block-mean reduction runs before the cusp
  correction, silently wrong on any cusped block.
- **Raise `--param=vect-max-version-for-alias-checks` from its default 10 to
  200** (a `setup.py` default). GCC versions a loop for vectorization when it
  cannot prove the accesses do not alias, but only up to this many runtime
  checks; past the limit it silently declines to vectorize at all and reports
  a bare `couldn't vectorize loop` with no reason attached.
  `set_tau_q_soa`'s **stage-2 row loop -- tau, `mu_turb` and q, a full-volume
  loop -- was over the limit and had never vectorized**: seven variable-size
  automatic row temps, which GCC lowers to pointers it cannot disambiguate
  from the dummy arrays, against five dummies. At 200 it vectorizes at 32
  bytes. Stage 1 was always under the limit and always versioned, which is why
  the miss looked like a property of stage 2's body rather than a budget.
  **Caution -- this is a whole-program flag and only one kernel's loops were
  checked.** `set_residual` and the IRS smoother certainly moved codegen under
  it and are UNTIMED; the full test suite passes, but "vectorizes better" is
  not "faster" (see the PGO entry below, adopted on exactly that reasoning and
  worth +169%). Time those two before treating this as settled.
- **Pin GCC's unit-level inline budgets**
  (`--param=inline-unit-growth=1000000 --param=large-function-growth=1000000`,
  now a `setup.py` default). The single largest effect ever found on this
  kernel, and not a kernel change: `set_residual`'s face helpers were simply
  not being inlined by default. **-53% serial, -36% at 16-rank contention,
  -17% on a full timestep.** The minimal *exact* pair is
  `inline-unit-growth` + `large-function-growth`; `inline-unit-growth` is
  worth only 2.3% alone but is necessary -- without it the other budget
  collapses to a weaker constraint (a real, non-monotonic interaction,
  confirmed by exact codegen fingerprints, not noise). **PGO
  (`EMBER_PGO=use`) is a 2.7x regression (+169%) and must not be used on
  this code** -- profile data makes GCC's inlining decisions actively worse
  here (it un-inlines exactly the helpers whose inlining is worth 53%).
  Probably gfortran-only: `INTEL_FLAGS` already carries
  `-inline-factor=10000`; diff `set_residual`'s fingerprint with and without
  the pinned budgets on the production compiler before assuming this result
  transfers.
- **Extract `wall_core` out of `wall_func`** (`viscous.f90`'s
  `viscous_helpers` module), so a new diagnostic (`wall_yplus`/
  `wall_yplus_iface/jface/kface`/`wall_yplus_field`, post-processing y+ for
  `block_util.wall_yplus`) shares its Re/skin-friction physics with the
  production wall function instead of duplicating it. `set_visc_force`'s own
  source is untouched -- same call sites, same names, only `wall_func`'s
  callee moves -- and in the real (no `EMBER_BENCH_KERNELS`) build,
  `set_visc_force_`'s codegen fingerprint is **bitwise identical**
  before/after (`32b4f45a7fd9b39a`, 17560 insns, both sides of the refactor):
  GCC fully absorbs the extra call under whole-program inlining. **Zero
  regression, proven by codegen identity, not inferred from noisy timing.**
  2026-08-07, gfortran/Haswell, this harness.
  **A same-build A/B against a frozen pre-refactor copy (the usual protocol
  for a kernel change, see `viscij`/`viscijk`/`viscpol` above) was tried
  first and gave a false alarm**: with both `set_visc_force` (refactored)
  and a full frozen `set_visc_force_pre` (~370 lines each, near-duplicates)
  sharing one translation unit, `visc` timed 57.1 ns/cell and `viscpre`
  89.3 ns/cell at 1M cells/8-rank contention -- `viscpre` +56.5% slower than
  the kernel it was frozen FROM. Isolating `visc` alone (no other arm in the
  build, Rule 7) reproduced its own plain-build fingerprint exactly and timed
  58.2 ns/cell, tight and reproducible (0.39% stdev over 10 launches) --
  proving the two-arm number was contaminated, not a real property of either
  arm's code. **New finding for the cross-cutting section below: two large
  (~300+ instruction) near-duplicate kernel bodies sharing one unit-level
  inline-budget translation unit can starve each other's inlining in either
  direction, not just perturb instruction count by a few percent** -- the
  existing "4 identical / 5 different" finding was a codegen-identity check;
  this is the same mechanism costing real, large (>50%), asymmetric timing
  swings. **Prefer fingerprint identity over same-build timing when a
  refactor is provably code-motion-only** (unchanged call sites/signatures);
  reach for a same-build A/B only when the fingerprint actually differs and
  a timing verdict is unavoidable, and even then, isolate any arm whose
  result looks anomalous (Rule 7) before trusting it, especially for
  full-kernel-sized arms.

### Rejected or negative results worth remembering

- **Component-first tau/q layout** (`(6,ni+1,...)` instead of
  `(ni+1,...,6)`). Looked +14% faster in an isolated 2-file build; **+42%
  slower** in the real whole-program build, because component-last is what
  lets the face-flux loops vectorize at 32 bytes, and losing that SIMD
  dominates any locality gained. The lesson that shaped every protocol rule
  above: **never trust a standalone single-file compile for this codebase.**
- **Removing `set_tau_q_soa`'s ten automatic arrays -- all three ways.** The
  kernel declares its row temps as `gVx(ni-1,3)`, `vct(ni-1)` and seven more,
  which is an `alloca` per call and the one thing kernel scratch here is not
  supposed to do. It is also, measurably, the fastest arrangement of the
  three available, so it stays -- with this entry to stop the next person
  re-deriving it.

  - **Fixed-size stack tiles plus an i-tile loop** (the shape that worked for
    the wall-function rows): **+12.9% at 1M and +183.9% at 2M**, serial. The
    cost is per loop ENTRY, not per cell: this loop only vectorizes under
    the raised `--param=vect-max-version-for-alias-checks`, and that
    versioning prologue is re-executed for every tile. Tile width buys it
    back exactly as that explains -- 64-wide +184%, 128-wide +87%, 256-wide
    breaks even, 512-wide +1%. **A kernel whose vectorization depends on
    alias versioning must not have its rows tiled.**
  - **Fusing the two stages into one all-scalar loop.** Legitimate on the
    dependences -- stage 2 only ever reads stage 1's output at its own `i`,
    so nothing needs to cross an iteration and the arrays are simply not
    needed. -0.8% at 1M serial, but **+55.7% at 2M serial and +44.9%
    contended**, and not bitwise (2-3 ulp: the split loop rounds through
    memory at each store, the fused one keeps the values in registers).
  - **Caller-owned row scratch from the arena** (`tqrow(ni-1,16)`, carved in
    `_carve_viscous`), which keeps the loop structure, the vectorization and
    the arithmetic exactly as they were and is bitwise identical. Still
    **about +2% at every size, in both regimes** -- consistent across six
    points, and the only change is where the memory comes from. Two follow-up
    experiments say what it is NOT. **It is not the alias check**: `!GCC$
    ivdep` on both stage loops does remove the versioning (the opt report's
    "loop versioned for vectorization because of possible aliasing" is gone
    for both, and they still vectorize at 32 bytes), but paired against the
    same build without it the hint is a wash -- +2.2% / -1.4% serial at
    1M/2M, flat contended. One runtime check per (j,k) row is amortised over
    ni cells; it only mattered when tiling multiplied it. **And it is not
    where the buffer lives**: allocating the row temps outside the arena
    entirely, so they cannot conflict with the tau/q volume the same loop
    streams, measures the same 1-4% above baseline. What is left is that a
    local automatic array simply tells the compiler more than any dummy can.

  The lesson generalises past this kernel: **`alloca` in a hot kernel is
  worth removing only if you can show it costs something.** Here it is one
  allocation per call, of ~35 KB at 2M cells, against a kernel that runs
  ~10^6 cells -- and every arrangement that removes it perturbs something
  the compiler was relying on. Results in `results/tauq_tile_*.jsonl` and
  `results/tauq_arena_*.jsonl`.
- **The `njp` 4K-aliasing pad, re-measured after the j-panel.** The rule
  exists so the ten component streams of the k-accumulate (5 components x
  the plane pair) cannot land on the same cache sets. Forcing exactly that
  pathology -- the `alias4k` arm, which picks the smallest j-extent making
  the component stride `ni*njp` an exact multiple of 1024 floats -- now
  costs **+0.1% to +0.8%**, contended and serial alike, and applying the pad
  unconditionally (`padnjp`) costs about the same again. Neither is clearly
  outside the ~0.5% uncertainty on a ratio. Plausibly the panel is what
  shrank it: the plane pair is now walked in narrow j-strips, so far fewer
  sets are live at once. Note also that the rule **cannot fire on the duct
  bench at all** -- `build_duct_grid` forces `ni-1` to a multiple of 8, so
  `ni` is odd and `ni*nj` is a multiple of 1024 only if `nj` is, which no
  sane cross-section is. The pad is kept, but do not go looking for percent
  in it here.

  Getting this wrong is instructive. An unbarriered 8-rank screen of the
  same question showed a **fabricated 11-12% win** for whichever j-extent
  was measured LAST in each process -- free-running ranks drift apart over
  a sweep, so the final configuration is effectively timed uncontended.
  Rule 10 is not a formality; it is the difference between +0.6% and -12%
  on the same code.
- **A second-level j-panel tile on top of k-slab tiling**, `set_residual`.
  The slab-tiled kernel already fetches every nodal field from DRAM ~once;
  panelling the j-dimension on top only converts already-cheap L3 re-touches
  into L2 re-touches, while its own overhead (one extra j-face-row recompute
  per panel boundary, shorter contiguous plane walks) is real. Loses or ties
  at 7 of 8 sizes tested. **L3 traffic was not the remaining bottleneck** --
  don't add a second tiling dimension speculatively.

  **Superseded, and worth understanding why.** A j-panel IS production
  today (see the adopted entry above), and this result is not wrong: it was
  measured on a kernel that still fetched through a k-slab, and in a regime
  where one rank owns the whole L3. Both halves matter. Once the fused
  rewrite made the k walk carry a rolling PLANE PAIR from step to step, the
  thing to bound stopped being nodal-field traffic (which the slab already
  handled) and became the carry itself; and once eight ranks share the L3,
  a carry that fits it alone no longer fits. Serially, the panel still
  loses about 2% at 1M cells and above, exactly as this entry found. The
  lesson is narrower than "don't panel": a tiling dimension is worth what
  the machine's contention makes it worth, so measure it in the regime
  production actually runs in.
- **The "idiomatic" rewrite, the separable box-filter tiled rewrite, and the
  naive textbook kernel** (all measured against production in the same
  `.so`, same arithmetic where checked). All are **1.5x-2.7x slower**, under
  ifort, than production's hand-scalarized, hand-unrolled kernel. The
  mechanism: ifort refuses to vectorize a loop that assigns into an
  accumulator **array** (`pm(:)`, `mf(:)`) because it can't disprove a flow
  dependence between elements -- hand-scalarizing into named scalars removes
  that dependence and is what makes the face loops vectorize at all. This is
  **compiler-specific**: under gfortran the ranking inverts (production's
  `accum_corners` doesn't vectorize there either, so the "clean" rewrite
  wins) -- re-verify hand-optimizations like this per compiler/version, and
  never take a "just write it properly" refactor on faith for this file.
  Corollary from the naive-kernel measurement: **being clever in the wrong
  dimension is worse than not being clever at all** -- a fully-vectorized,
  cache-resident tiled rewrite still lost to four thoughtlessly-written
  full-volume passes, because the tiled rewrite traded cheap arithmetic
  (already vectorized) for a 4.6x increase in scratch traffic on a kernel
  that is memory-bandwidth-bound at production concurrency.
- **Merging the IRS k-solve's component loop** (mirroring the
  `damp_residual` win). Regressed +9.8% saturated, losing on every rank.
  **The general rule this pair of experiments (with `damp_residual`)
  establishes: merging per-component loops only pays when the components
  share a real per-cell input.** `damp_residual`'s five components
  genuinely share one `dt_vol(i,j,k)` value; the IRS k-solve's components
  share only scalars already in registers, so merging just multiplied the
  concurrent working set (10 streams x 68 KB instead of one at a time) for
  no shared saving. Traffic-counting predicted a win in both cases; only
  measuring what the loops actually *share* explained why one merge won and
  the other lost.
- **`-qopt-zmm-usage=high`** (prefer 512-bit ZMM over 256-bit YMM encodings
  on AVX-512 hardware). Regresses +13-16% at small/mid sizes (the classic
  AVX-512 frequency/warm-up tax on loops too short to amortise it); the one
  win at the largest size tested was a single, unrepeated data point. Not
  adopted.
- **The multall/multall five-pass residual family** (`staged`, `split`,
  `multall`, `nodal`, `tbaos`, `prodsoa`, `rinv`). See the harness-fix summary
  at the top of this file -- every one of these loses to production once
  measured on a correctly-compiled baseline with a non-leaking harness. Kept
  in `bench/subroutines/` (excluded from the shipped build by default,
  available via `EMBER_BENCH_KERNELS`) because the ranking is expected to
  differ on ifort/Sapphire, where the AVX-512
  divider is several times faster per element than Haswell's and production's
  redundant reciprocals cost much less.

- **The `fvisc` fusion LOSES at the smallest size on the duct ladder**
  (86016 cells: +10.6% for the i+j+k arm, +47.3% for the first polar arm),
  wins at 286720 (-9.7%), 974848 (-21.1%) and 1921024 (-21.3%), and every
  arm's penalty shrinks monotonically with size. That is the instruction-bound
  signature: the fusion's cost is fixed per-row overhead, its benefit is
  `fvisc` traffic that only exists once the working set stops fitting in
  cache. **Read the small-size point with care before generalising**:
  `build_duct_grid` grows `ni` only, so the ladder is really an `ni` ladder
  (shapes 25/81/273/537 x 65 x 57) and the 86016 point is the only one where
  i is the *short* axis, with inner loops 3 AVX2 vectors long. Whether that
  regression is a size threshold or an aspect-ratio artefact was not
  determined. Adopted anyway: production blocks are at the sizes where it
  wins, and the kernel is memory-bound there.

- **Fusing `set_tau_q_soa` into `set_visc_force`** (experimental,
  `bench/subroutines/viscous_tauq_fused.f90`, NOT adopted). tau/q stops
  round-tripping through memory: the adopted `set_visc_force` is a single walk
  over k face planes consuming tau/q cell planes k-1 and k and nothing else, so
  tau/q is produced inside the walk into a rolling plane pair. `tau_cell`/
  `q_cell` demote to halo source only. **Bitwise** on both `fvisc` and
  `mu_turb`. On the pair at 1M cells: **-20.9% serial, but only -8.2% at
  8-rank socket contention**, 10/10 launches in both.
  **The win shrinks under contention, which is the opposite of a bandwidth
  saving** and is the whole finding. The rolling pair is 1.24 MB per rank at
  that shape: 6% of the 20 MB L3 for one rank, but 9.9 MB across 8 ranks, so
  half of each rank's 2.5 MB share is the buffer itself, evicting the nodal
  fields both halves re-read. The fusion trades DRAM traffic for L3 pressure
  that scales with rank count while the traffic saving does not -- so going to
  higher concurrency makes it *worse*, not better. Static spill density is up
  ~40% (36.4 vs 25.9 stack vector moves per 1k insns) but cannot be the
  explanation: spills cost the same in both regimes.
  **The lever, untried: j-tile the walk to ~12 rows**, putting the pair at
  ~250 KB in the PRIVATE per-core L2, where its cost stops depending on how
  many ranks share the socket. Unlike the j-panel tiling rejected for
  `set_residual` above, this is not a speculative second tiling dimension --
  it shrinks a buffer the fusion itself introduced, and there is a measured
  regime-dependence pointing at it.
  Not adoptable as it stands regardless: no cusp seam support, and integration
  needs an O(surface) boundary-only phase 1, the `grid.py` resequencing around
  `exchange_halos`, and a dedicated buffer.

### Cross-cutting findings, not tied to one kernel

- **What a whole step actually costs** (1M-cell duct, `n_stage=4`, `cfl=2`,
  solver defaults, `kernprof` on the `util.profile` decorators). Percent of
  `_run`, serial then 8-rank socket-contended: `advance_rk_stage_mg`
  21.0/22.5, `set_residual` 22.8/21.0, the IRS smoother 16.2/14.9,
  `update_primitive` 7.7/7.6, `set_visc_force` 6.5/6.5, `set_tau_q_soa`
  6.6/6.0, `damp_residual` 3.4/4.6, `apply_bconds` 3.3/4.2, and
  `Grid.smooth`, `update_timestep`, `update_bconds` about 2-3 each. Three
  things fall out of it. `update_residual` is 49% of a step because it runs
  FOUR times (once at the top, once per RK stage bar the last) against
  `update_sources`' once, so a percent off the residual is worth four times
  a percent off the viscous pass. `advance_rk_stage_mg` is the largest
  single phase under contention and had never been benchmarked. And
  `exchange_halos` is 0.004% -- whatever separates the viscous pair's
  end-to-end number from `set_visc_force`'s own, it is not the halo
  exchange.
- **Kernel wins are contention wins; a single-rank run will not show them.**
  The wall-clock cost of a step at 1M cells, before the viscous work,
  after it, and after the residual panel as well: 379 / 377 / 378 ns per
  cell per step serial -- nothing -- against 724 / 707 / 666 with eight
  cores of the socket busy, which is **-8.0%**. Serially the two changes
  very nearly cancel (the viscous rows win ~0.8% of the step, the residual
  panel gives back ~0.5%). Quote the contended figure, and do not conclude
  from an idle-machine timing that a change did nothing.


- **`set_residual` is ~45% `vgatherdps` under ifort**, spread evenly across
  all three face directions, inherent to the 4-corner-average pattern in
  `accum_corners` reading `vx(i,j,k)`/`vx(i,j+1,k)`/etc. This is present in
  the **current, unmodified production kernel**, not an artefact of any
  rejected rewrite. No fix was found or is proposed; ifort's gather-based
  vectorization, slow per-instruction as it is, still beats gfortran's fully
  scalar version of the same loop by ~4.7x. Any future attempt to "fix" the
  gathers needs an ifort A/B against *this* production baseline, not a
  standalone harness or a gfortran baseline -- both have inverted rankings
  on this exact kernel before.
- **Production is memory-bandwidth-bound at production concurrency, not
  serially.** `set_residual` goes from ~9.5 ns/cell serial to ~24 ns/cell at
  100-rank saturation (~522 GB/s aggregate, close to this node's realistic
  STREAM ceiling) -- so a change's serial number and its saturated number can
  and do disagree in *direction*, not just magnitude (`consa`: flat serial,
  real win saturated; the "clean" rewrites: penalty *shrinks* under
  saturation because they're instruction-bound and DRAM contention partially
  masks it; the tiled rewrites: penalty *grows* under saturation because
  they're traffic-bound and contention compounds it). **Always measure both
  regimes and use the penalty-direction as a diagnostic of which resource a
  candidate is actually spending.**
- **A compulsory-traffic model only ranks candidates when the kernel is
  actually bandwidth-bound** (Rule 8 above) -- treat its predictions as an
  upper bound on a win, not an exact one: heavily-reused nodal fields are
  partially L3-resident, so dropping a stream saves less than naive
  byte-counting predicts (predicted -11%, measured -4.1% for `consa`).
- **Pinned inline budgets make a kernel's codegen independent of the other
  arms in the build -- but not without limit, and not of DELETIONS.** Two
  counts against the assumption stated in `run_all_arms.sh`'s header, both
  found with `codegen_gauge.py`: (1) deleting four dead helper routines from
  `viscous.f90` moved `set_visc_force` from `f03ddf9e` to `348aa81a`,
  11849 -> 12292 instructions -- removing translation-unit content freed
  budget and GCC inlined *more* into the live kernel (the timing difference
  was inside launch-to-launch drift, but the codegen identity was not);
  (2) `set_visc_force` fingerprinted identically with four viscous arms in the
  build and *differently* with five. Neither invalidates a within-build A/B,
  which is why `visc` is always re-measured alongside its arms rather than
  compared to a stored baseline -- but never assume a cross-build number is
  comparable without re-running the gauge.
  (3) **This is not always harmless even within one build**: freezing a full
  ~370-line pre-refactor copy of `set_visc_force` alongside production for a
  same-build A/B (the `wall_core` extraction study, "Adopted" above) produced
  a +56% timing swing that vanished when the frozen arm was isolated in its
  own build -- two large near-duplicate kernel bodies competing for the same
  unit-level inline budget can starve each other's inlining by a lot, not
  just perturb instruction count by a few percent. A within-build A/B is only
  as trustworthy as the arms sharing the unit are small/dissimilar (the
  fvisc-fusion ladder's arms, ~90% shared code, have never shown this); two
  full independent kernel bodies is a different regime -- isolate (Rule 7)
  before trusting a timing result that surprises you.
- **A `PARTIAL LOOP WAS VECTORIZED` or a clean `LOOP WAS VECTORIZED` report
  is a lead, never a verdict, in either direction.** A blemished report can
  cost nothing (the damp-split scaling loop); a clean report can hide a slow
  gather-based implementation (the gather finding above). Only a real,
  correctly-built, correctly-barriered timing comparison decides.

