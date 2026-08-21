# Prototype: exchange-free fused viscous pass for a k-self-periodic block

## Context

The two viscous kernels hand ~40% of the pair's DRAM traffic to each other
through `tau_cell`/`q_cell`: 9 nodal planes written by `set_tau_q_soa` (plus
their write-allocate reads) and streamed straight back by `set_visc_force`.
Fusing them removes all of it, and a fused arm already exists --
`set_visc_force_tqf` in `bench/subroutines/viscous_tauq_fused.f90` -- but it
is blocked by the grid-wide periodic-seam exchange that has to run *between*
the phases.

For a block periodic to itself in k -- the H-mesh case, seam over
`i` in `[1, i_LE]` and `[i_TE, ni]` with blade walls between -- that exchange
is a copy from the block's own far edge into its own near halo. The fused
kernel can read the far plane directly instead, so the k-seam exchange
disappears and with it the phase barrier between the two kernels. That is what
this prototype measures.

Scope is a bench arm, a bench case, one `ember.cases` case-builder argument,
and one new assertion in an existing test.

## Three findings that shape it

**The wall mask carries the per-cell periodic/wall distinction, but it is not
a periodicity flag.** `wallk1`/`wallnk` are already arguments to both kernels
and are 1.0 wherever the k face is not a viscous wall, per `(i,j)` -- finer
than `i_perk`'s two i-intervals. So the kernel needs no `i_perk` and no
i-banding: the seam select is a blend against a mask it already has.

But `ijk_wall_visc` (`block.py:2780`) is built from `_face_wall_arrays_slip`
and means "not a **viscous wall**", not "periodic". It is 1.0 for every
permeable patch and for slip patches too. A slip endwall, a mixing plane, a
non-matching patch or an inlet on a k face all read `wallk1 == 1` while
nothing exchanges their halo -- the unfused path leaves them at +edge, and a
kernel that trusted the mask alone would read the far k plane there and be
silently wrong.

The sharpest example is on the target geometry itself. `CuspPatch` is in
`PERMEABLE_TYPES` (`patch.py:187-197`) and "must be on a constant-k face"
(`cusp.py:25`), so on a real H-mesh the trailing-edge cusp interval of the k
face reads `wallk1 == 1` while being neither wall nor periodic --
`tests/test_viscous_cusp_seam.py:130-135` builds exactly that block.
`InviscidPatch` does the same through `SLIP_TYPES = PERMEABLE_TYPES +
(InviscidPatch,)` (`patch.py:200`), which is why `ijk_wall_visc` and
`ijk_wall_conv` disagree about it (`tests/test_wdist.py:262`). The mask is the
right *in-kernel* predicate only once the caller has established that non-wall
and periodic coincide on the k faces.

This also makes the arm's existing refusal of cusped cases load-bearing rather
than incidental: a cusp is precisely a k-face interval that the mask calls
non-wall and the seam select would mishandle.

**`block.i_perk` is the eligibility sentinel.** `i_perk` (`block.py:2725`)
returns `(0, 0)` exactly when the block has no k-face `PeriodicPatch`, so it
is the switch between the two paths:

```
i_perk == (0, 0)  ->  general path (exchange, unfused or plain tqf)
i_perk != (0, 0)  ->  candidate for the seam-free fused path
```

It reads correctly for every shape in play: a full-span
`PeriodicPatch(k=0)`/`(k=-1)` gives `(ni, 1)`, an H-mesh pair of i-intervals
gives `(i_LE, i_TE)`, and `build_duct_grid` gives `(0, 0)`. It is already
cached on the block and needs no new state. It does not enter the Fortran
signature -- the kernel still keys on the mask.

**Self-periodic pairing works, but only after `clear()`.** `pair()` caches, and
`build_duct_grid` populates the cache during construction, so patches appended
afterwards are invisible until `grid.connectivity.periodic.clear()`. With that
call, a single block pairs to itself: `{(0,2): (0,3), (0,3): (0,2)}`. Without
it the pair dict is silently empty, the seam is never exchanged, and the block
runs with a zero-gradient seam.

The same trap sits one level up and is the more dangerous of the two.
`build_case` (`bench/residual_arms.py:104`) calls `grid.update_sources`, which
runs the viscous kernels, which splat `**block.ijk_wall_visc` (`grid.py:1723`).
That is a `cached_object` whose docstring says outright "patches must not be
modified after first access", and `i_perk`, `i_cusp` and
`_face_wall_arrays_slip` are all warm by the time `build_case` returns. Append
patches after that and `wallk1` stays 0.0 across the whole seam *and* `i_perk`
stays `(0, 0)` -- the sentinel silently says "general path" and the gate fails
looking like a kernel bug. So the patches go in at construction time, not
afterwards (see below).

## Design

### 1. The case (`src/ember/cases.py`, not the bench)

`build_duct_grid` has only `InletPatch(i=0)` and `OutletPatch(i=-1)`, so
`i_perk == (0, 0)` and `wallk1 == 0` everywhere: **the existing bench case has
no periodic seam, and the current `tqf` arm has therefore never exercised the
halo path it models.**

Add a `periodic_k` argument to `build_duct_grid` (`cases.py:19`) that appends
`PeriodicPatch(k=0)` and `PeriodicPatch(k=-1)` at the same point the inlet and
outlet are appended (`cases.py:122-123`) -- i.e. before `calculate_wdist()`
and before anything touches a wall array. That is what
`tests/test_viscous_periodic.py:_build_periodic_block` already does, and it
sidesteps the cache-warming trap entirely. Two variants:

* full-span patches -- every k face periodic, the simplest gate and the
  best-case timing;
* patches restricted to two i-intervals -- the H-mesh shape, mixing periodic
  and wall along the same seam, which is the variant that actually exercises
  the mask blend.

Two consequences of the case change to check, not assume:

* The duct's only patches today are inlet and outlet, so **all four**
  cross-stream faces are walls. Making both k faces periodic removes two of the
  four, so the wall-distance field changes, hence `xlength`, hence
  `mut = min(rho*xlength*vm, 3000*mu)`, and the clamp will bind over much more
  of the field. `calculate_wdist()` must run after the patches exist, and the
  full-span variant must be checked to be well posed with walls on two faces
  only. The existing arm's `~35 ulp` / `0.03 ulp` figures were measured on the
  old field and do not transfer -- re-measure them here.
* Timing is *not* biased by the mask change. `wall_func_kface` is called
  unconditionally and blended by `wfac`, so a mask of 0 and a mask of 1 cost
  the same. Assert this rather than leaving it to the reader.

### 2. The kernel (new arm beside `viscous_tauq_fused.f90`)

The existing arm touches the exchanged halo in exactly two places, both
`load_halo_kplane` calls (`viscous_tauq_fused.f90:246` for the `k=1` slot and
`:349` for `k=nk`). Those are the only edits.

* Pre-pass: produce cell plane `nk-1`'s tau/q into a saved plane before the k
  loop. One plane of duplicated work, O(surface). It needs only the interior
  `(i,j)` of the plane -- a k-halo plane's i/j edges are never read, per
  `load_halo_kplane`'s own comment -- so it skips `load_halo_ijedge`.
* Stash cell plane 1 when the loop produces it, into a fourth `tq` slot.
* At `k=1`, after `tb` (plane 1) is produced, fill `ta` by blend rather than
  by halo read:
  `ta = wallk1*plane_nk1 + (1 - wallk1)*(-tb)`
  -- the periodic side reads the far plane; the wall side is the `+edge` ghost
  negated, which is exactly what `load_halo_kplane` computes today from the
  `(2*wall-1)` scaling.
* At `k=nk`, symmetrically: `tb = wallnk*plane_1 + (1 - wallnk)*(-ta)`.
* `tau_cell`/`q_cell` stay in the signature for `load_halo_ijedge` only -- the
  i/j halo edges are unrelated to the k seam and keep their existing path.

The blend is exact: `ijk_wall_visc` casts a boolean, so the mask is exactly
0.0 or 1.0. Two harmless notes so nobody rediscovers them: it evaluates both
sides, so `0.0 * x` is not zero for non-finite `x`; and `+0.0` on a `-0.0`
flips the sign of zero. Neither reaches `fvisc`.

**How the pre-pass reuses the producer body: by duplication.** The producer is
~90 lines (`viscous_tauq_fused.f90:249-342`) and Fortran gives no clean
alternative here. Hoisting it into a module procedure is directly
contraindicated -- the comment at `viscous_tauq_fused.f90:208-214` records that
making the row temps dummy arguments cost this exact loop its vectorization,
because GCC will not version for alias checks against a dummy, and hoisting
would move both the timing signal and the ulp figure. A `block` construct is
ruled out: it silently drops unrelated subroutines from the f2py build. So the
prototype duplicates, and says so.

Ordering note: the pre-pass is what avoids restructuring `fvisc`. The
alternative -- deferring the `k=1` face flux until plane `nk-1` exists -- would
touch `fvisc` plane 1 a second time at the end, breaking the
single-store-per-cell property that `set_visc_force` was optimised for.

**Buffer cost, which is the main risk to the result.** `tq` goes from 2 slots
to 4: 1.3 MB to 2.6 MB at the 1M shape. The arm header names `tq`'s L3
residency as "the main risk to the whole idea", and this doubles it. That is
not just "the small end of the ladder will lose" -- it is a direct attack on
the mechanism under test, and it may eat the fusion win outright at the
contended sizes where the traffic actually binds.

`callers_pair`'s scratch guard hard-codes the old size and must be updated:

```python
need = (ni + 1) * (nj + 1) * 9 * 2 + ni * nj * 4 * 2 + ni * 4 * 3
```

Check `b.scratch.size` actually holds a fourth slot before assuming the
`SystemExit` stays untriggered.

Worth evaluating as a cheaper shape: at `k=nk` the kernel needs the *face
flux*, not tau/q, so a 4-component `planes` slot stashed from `k=1` could
replace a 9-component `tq` plane. Gate it, do not assume it --
`dAk(:,i,j,1)` and `dAk(:,i,j,nk)` come from different node coordinates and
will not agree bitwise, which is exactly why the cusp correction
(`viscous.f90:978`) has to average the two seam face flows in the first place.

### 3. The caller (`bench/visc_arms.py`)

A `callers_pair` variant that runs the fused arm with **no** `exchange_halos`
call. Eligibility is `i_perk != (0, 0)` plus two guards, both O(surface) and
both run once at setup, refusing with `SystemExit` beside the existing cusp
refusal:

1. **Self-pairing.** Every k-face periodic pair maps the block to itself:
   `all(bid == nxbid for (bid, pid), ((nxbid, nxpid), _) in pairs.items())`,
   restricted to `const_dim == 2` patches. Cross-block k pairing is
   constructible -- a pitch split into two blocks would pair A's `k=0` to B's
   `k=nk` and `check_match` (`periodic.py:56`) would accept it -- and nothing
   in the codebase rules it out.
2. **Coverage.** The non-wall part of the k face is *exactly* the periodic
   part. Rebuild a k-face indicator from `patches.periodic` with
   `const_dim == 2` via `patch.get_ijk_face()` -- the same call
   `_get_face_wall_arrays` uses at `block.py:946` -- and require it to equal
   `wallk1`/`wallnk`. Five lines, no assumption about patch shape, and it
   catches the cusp, slip-endwall and partial-j-span cases that comparing
   against `i_perk`'s i-intervals would miss.

**The unfused reference must gain the exchange.** Today
`callers_pair` builds `out = {"unfused": lambda: (tauq(), visc())}`
(`visc_arms.py:263`) -- no `exchange_halos`. On a self-periodic case that
reference computes a **zero-gradient seam**, because `set_tau_q_soa` overwrites
every k halo slot with the +edge value on its way out (the "Fill boundary halo
slots with +edge" loops at the end of the subroutine), destroying whatever
`seed_tau_q`'s exchange left there. Gated against that, the fused arm would
differ at `k=1`/`k=nk-1` -- and the natural reading of a deviation concentrated
at the seam is "the seam handling is wrong", which would be a misdiagnosis of
the reference. So `unfused` becomes `lambda: (tauq(), exchange(), visc())`.
`callers_pair` takes only `b` today and needs the grid or the communicator
passed in; that propagates to `check_pair` and `bench_prod_baseline.py:171`.

One knock-on for the module docstring: on a periodic k face `wallk1 == 1`, so
`zero_wall_fvisc_border` no longer zeroes the seam row and the halo values
reach `fvisc` for the first time. `halo_restorer` stops being "insurance" and
becomes load-bearing.

### 4. What this does and does not remove

The design removes the **k-seam** exchange and the phase barrier it imposes.
It does not remove all exchange. `load_halo_ijedge` still reads the i/j halo
edge slots on every k plane, and those are produced by `set_tau_q_soa`'s +edge
fill, which the fused arm never runs -- in the bench it free-rides on
`seed_tau_q`, and the frozen state makes the stale edges the right values. In
a real integration the O(surface) boundary kernel the arm header postulates
still has to run before the fused pass to produce them, and on a real H-mesh
the i faces are block interfaces needing a genuine exchange.

## Correctness gate

**Primary gate: physics, in `tests/test_viscous_periodic.py`.** That test
already builds a single k-periodic block carrying two full wavelengths of `Vx`
across the pitch, and asserts (A) that toggling the exchange changes only the
two seam-adjacent k-cells, and (B) that each seam cell's force equals its
interior twin half a domain away. A correct seam-free kernel must reproduce
`fx_exchange` **with no exchange at all**. Add that as a third variant of
`_fvisc_x` with `comm=None`, asserted equal to `fx_exchange`. It runs in
seconds on a 5x5x9 block, it is a physics statement rather than a tolerance,
and it catches a wrong far-plane index that an ulp comparison at 300k cells
would bury in reassociation noise.

**Secondary gate: ulps, in the bench**, against unfused
`set_tau_q_soa` + `exchange_halos` + `set_visc_force` on the self-periodic
case, before any timing:

* `mu_turb` bitwise -- the producer is `set_tau_q_soa` expression for
  expression, as the existing arm already demonstrates. The pre-pass writes
  `mu_turb(:,:,nk-1)` twice, once in the pre-pass and once at `k = nk-1`;
  that is idempotent, and `check_pair`'s existing `idempotent` field asserts it
  rather than arguing it.
* `fvisc` to the arm's known ~35 ulp of field scale at production flags,
  collapsing to ~0.03 ulp under `-fno-associative-math -ffp-contract=off` --
  both figures re-measured on the new case, not inherited.
* Run the H-mesh variant too: a kernel that ignored the mask would pass the
  full-span case and fail this one.

Two notes while there. `swap_by_ijk` (`indexing.f90:65`) receives the same
array as both `h1` and `h2` for a self-pair. The pairs dict holds both keys, so
`exchange_halos` (`periodic_communicator.py:196`) runs the swap **twice** with
roles reversed; it is idempotent because the owned-edge-cell slots are interior
and never written -- not because of the read-into-`tmp`-first ordering. The
residual hazard is an f2py copy-in of the aliased buffer silently reverting one
side of the seam, but `tests/test_viscous_periodic.py` is green today and could
not be if that happened, so a one-line assert after the exchange is confidence,
not an open risk.

And the arm refuses cusped cases. The duct has `i_cusp == (0, 0)`, so this
constrains which case can be built -- but note that the "H-mesh variant" is
therefore a **mask-shape mock, not an H-mesh**: a real H-mesh with a seam split
by a blade has a cusp. It exercises the mask blend, which is its job; the
result must not be read as "validated on the target geometry".

Worth recording in passing: `viscous_tauq_fused.f90:26-32` says the cusp
correction is blocked because "plane 1 is long gone from the rolling pair by
the time the walk reaches nk". This design stashes plane 1 *and* plane `nk-1`,
which is exactly what the correction needs. Not required scope, but it is the
reason this arm is the one that could become production-ready.

## Measurement: what was actually found

Measured on the `hmesh` case at 1M cells, 10 launches, paired within launch,
`gp-111`, arms `unfused` / `fused` / `fused_selfk` (see Verification below).

```
                 serial (1 rank)        contended (8 ranks)
  fused           -11.30% +/- 0.23      -6.99% +/- 0.53     10/10 launches
  fused_selfk     -22.88% +/- 0.34     -14.96% +/- 0.47     10/10 launches
```

So the seam-free arm is faster, consistently and by a wide margin. **But
almost none of that margin is the thing this study is about**, and the
three-way split plus a control arm is what showed it.

`exchange_halos` was measured directly at **0.612 ms/call = 0.63 ns/cell** at
the 1M shape, against an unfused pair of 55.3 ns/cell. It moves 23,040 halo
points (0.83 MB); it does not copy the 42 MB buffer, so the f2py aliasing
worry was unfounded. Removing it is worth about **1% of the pair**, not the
12 points that separate the two fused arms.

The rest was attributed with two timing controls, both wrong by
construction and gated by nothing: `viscous_tauq_ctl.f90` keeps the pre-pass
and the 4-slot `tq` but restores the parent's halo reads, and
`viscous_tauq_noij.f90` additionally reads NOTHING from the full-volume
buffer. Measured through `run_all_arms.sh` (serial, 1M cells, 10 launches,
launch-outer/arm-inner, paired within launch):

```
  unfused                                              54.503 ns/cell
  fused        2 slots, no pre-pass, halo reads        48.514   -11.08% +/- 0.23
  fused_ctl    4 slots, pre-pass,    halo reads        41.831   -23.36% +/- 0.21
  fused_noij   4 slots, pre-pass,    no volume reads   41.304   -24.27% +/- 0.28
  fused_selfk  4 slots, pre-pass,    seam select       42.222   -22.59% +/- 0.21

  tqf -> ctl     pre-pass + 4-slot tq          6.68 ns/cell   13.8%
  ctl -> selfk   seam select vs halo read     -0.39 ns/cell   the select costs MORE
  selfk -> noij  drop the i/j edge reads       0.92 ns/cell    2.2%
  exchange removed                             0.63 ns/cell   (not harness-measured)
```

**The seam-free design nets out near zero**: 0.63 ns/cell saved on the
exchange, 0.39 given back because the select is marginally dearer than the
halo read it replaces. The exchange itself was measured directly at 0.612
ms/call = 0.63 ns/cell against an unfused pair of 54.5 -- it moves 23,040 halo
points (0.83 MB) and does not copy the 42 MB buffer, so the f2py aliasing
worry was unfounded.

**The dominant effect, 6.68 ns/cell or 13.8%, is the pre-pass plus the 4-slot
buffer** -- and `fused_ctl` still reads the exchanged halo, so the parent arm
can have all of it without any seam machinery. The mechanism is NOT
established: `ctl` does strictly more work than `tqf` (an extra plane of
producer, double the rolling buffer) and is 13.8% faster. Codegen differs
sharply -- 20,615 insns over 7 functions for `tqf` against 29,487 over 5 for
`ctl` -- but the instruction growth is just the duplicated producer, which
leaves the speed-up unexplained.

A third control, `viscous_tauq_nosig.f90`, drops tau_cell/q_cell from the
SIGNATURE rather than merely leaving them unread, separating the cost of
reading the volume from the cost of it being an argument at all (serial, 1M
cells, 10 launches, paired):

```
  fused_noij   unread, still an argument   41.108 ns/cell  -24.65% +/- 0.20
  fused_nosig  out of the signature        41.212 ns/cell  -24.54% +/- 0.15
  fused_selfk  reads the i/j edges         42.176 ns/cell  -22.77% +/- 0.16

  reading the volume in the k loop   1.07 ns/cell  1.88pp   (replicates 0.92)
  having it as an argument           0.10 ns/cell  0.11pp +/- 0.25  -- NULL
```

So the whole cost is the loop reads; passing a 37.8 MB array nobody reads is
free, as f2py marshals one descriptor per call. Both runs agree on `selfk`
(42.176 vs 42.222) and `unfused` (54.555 vs 54.503) across independent
launches.

WHAT NO ARM HERE MEASURES. Every control still has the block ALLOCATE the
full-volume buffer, and seed_tau_q still fills it once before the reps. So
the case for a compact surface buffer rests on allocation footprint and on
phase 1's 37.8 MB of writes -- and phase 1's writes are outside every timed
window in this study. That is the larger untested quantity, and it is the
thing the "O(surface) boundary kernel" design is actually for.

A METHOD WARNING, recorded because it cost a wrong conclusion. An earlier
pass timed these arms in ONE process with a bare perf_counter loop, and got
`noij` 2.85 ns/cell SLOWER than `selfk` -- the opposite sign to the harness
result above. In-process comparison lets arm order, page placement and
allocation address leak into the number. bench/README.md's one-arm-per-process
launch-replicated protocol is not ceremony; on these arms it flips signs.

**The actionable finding is therefore not the seam removal.** It is that
`set_visc_force_tqf` appears to leave ~14% on the table for a reason nobody
predicted, reachable without a periodic block, without the eligibility guards,
and without giving up the cusp correction. That is the thread to pull next.

## Verification

```bash
EMBER_BENCH_KERNELS=viscous_tauq_selfk,viscous_tauq_fused UV_NO_SYNC=1 uv pip install -e .
uv run pytest -q tests/test_viscous_periodic.py                      # primary gate
uv run python bench/visc_arms.py --ncell 300000 --periodic-k hmesh   # ulp gate
uv run python bench/visc_arms.py --ncell 300000 --periodic-k full
uv run pytest -q                                     # cases.py gains an argument
uv run python bench/codegen_gauge.py set_visc_force_ # same prod codegen either side
LAUNCHES=10 NRANKS=1 NCELL=1000000 REPS=30 KERNEL=viscpair PERIODIC_K=hmesh \
  ARMS="unfused fused fused_selfk" bash bench/run_all_arms.sh
```

`EMBER_BENCH_KERNELS` selects by FILE name (`select_bench_kernels`,
`setup.py:164`), so the token is `viscous_tauq_selfk` -- not the
`visc_arms.ENTRY` key and not the Fortran entry point. Those are three
separate names and all three appear in this workflow.

`bench/subroutines/viscous_fused.f90` is unbuildable for a pre-existing reason
(an ambiguous `scale_visc_halos` between its own helper module and
production's, dating from the commit that created it), so select only the arms
needed rather than `EMBER_BENCH_KERNELS=all`.

## Postscript: the surface-buffer path, and what it revealed

`Block.tau_q_faces` (six two-layer surface buffers), `set_tau_q_faces` (an
O(surface) boundary producer), `exchange_faces` and
`set_visc_force_tqf_faces` now exist and are gated: the consumer reproduces
production's fvisc to ~4.8 ulp of field scale with the ENTIRE full-volume
buffer poisoned to NaN, on a full-span seam, a subset seam, and a sealed block
with no periodic patches at all. The topology restriction is gone -- that part
of the design worked.

The timing did not. Serial, 1M cells, 10 launches, paired within launch:

```
  unfused      p1_soa + exchange_halos + set_visc_force   55.500 ns/cell
  faces        p1_faces + exchange_faces + consumer       59.315   +6.96%  0/10
  faces_nop1   consumer only                              47.768  -14.05% 10/10
  fused        exchange_halos + set_visc_force_tqf        48.558  -12.56% 10/10
```

`faces` and `unfused` are the first two SELF-CONTAINED arms in this study:
each produces the halo it reads, inside the timed window. Every other fused
number here -- including every one recorded above -- free-rides on seed_tau_q,
which fills the volume buffer once, untimed, before the reps. There was no
boundary producer to charge it to until now.

Phase costs, each measured through the harness rather than inferred:

```
  p1_soa   25.562      p1_faces  6.478      the boundary producer saves 19.1
  exchange_halos 0.63  exchange_faces 0.56  the face exchange is the cheaper one
  fused consumer 47.768  vs  p1_soa + visc = 54.87  ->  the fusion saves 7.07
```

**So the tau/q fusion, costed honestly, does not beat production.** It saves
7.07 ns/cell of volume work and gives back 6.5 producing the halo plus ~4.5 of
interaction the separate measurements do not show (running the producer
immediately before the consumer evicts the consumer's working set; the parts
sum to 54.8 against a measured 59.3). `fused`'s -12.5% was the free ride, not
the fusion.

The producer can be improved but not enough to change that. It runs at 87 ns
per boundary cell against the row form's 25.6, because `tau_q_at_cell` is a
scalar per-cell call; the j and k faces could use the row form directly, the
two i faces pin the axis it vectorises over. Fixing the 90% would put `faces`
near 55.2 -- break-even.

WHAT NOT TO DO NEXT. Do not retire `set_visc_force_tqf_selfk` (staging step 5).
It was to be retired once the faces path was proven, and the faces path is
currently slower than the arm it would replace.

## Why the fusion gives the round trip back: stream count, confirmed

The surface buffers do remove the round trip. Measured by LLC misses x 64 B,
differencing a 12-rep run against a 2-rep one so setup cancels:

```
  p1_soa     19.4 MB per call        p1_faces    4.0 MB per call
```

15.4 MB saved, and that matches the arithmetic: the tau/q span of tau_q_halo
is 37.8 MB, written once and read back, ~116 B/cell, which at the ~6.4 GB/s a
single process gets here is 18 ns/cell against a measured 19.1.

But the PAIR only improves 27.5 -> 26.3 MB. The fused consumer alone moves
11.4 MB where production's phase 2 moves ~8, and the two phases together move
far more than their parts (4.0 + 11.4 = 15.4 against a measured 26.3).

WHAT THE MECHANISM IS NOT. `l1d_pend_miss.fb_full` -- cycles with every L1
fill buffer occupied, the direct signature of more outstanding misses than the
core can track -- is 0.3% of cycles for the fused consumer, the LOWEST of any
arm here (p1_soa and unfused both sit at 1.1%). It is not fill-buffer
saturation.

WHAT IT IS. Two arms, `streams_hi` and `streams_lo`, differ in one thing:
whether mu, cp and kappa are three separate buffers holding identical values
or one buffer passed three times. Same arithmetic, same results, same
instruction stream, different stream count.

```
  streams_hi   150.9M cycles   10.14 MB          (10 launches, paired)
  streams_lo   144.6M cycles    7.37 MB          -7.86% +/- 0.30, 10/10

  time saved 2.44 ms;  traffic saved 2.77 MB, worth 0.43 ms at 6.4 GB/s
```

**82% of the gain is not explained by the bytes.** The fused loop is limited
by how many streams it walks, not by what it moves -- which is what
viscous_tauq_fused.f90's header predicted ("roughly doubles the concurrent
stream count, the failure mode behind the rejected j-panel tiling and the IRS
k-solve merge") and what these arms now measure. Not the fill buffers, so
most likely the L2 streamer's stream-tracking capacity plus DRAM row-buffer
thrashing across many open pages.

SO THE LEVER IS THE CONSUMER, NOT THE HALO SOURCE. `kb` is already in the
fused kernel's signature and inert. Slab-blocking the k walk so the nodal
fields stream through in panels is the change that would let the 15.4 MB the
surface buffers save actually reach the bottom line.
