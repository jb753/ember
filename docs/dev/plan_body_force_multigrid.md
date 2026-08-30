# Plan: block-sum multigrid as a frozen body force

> **Status: not started.** Design and test plan only. Nothing here is built.
>
> **Revised 2026-08-30** after a review against the source. Corrected: the
> `fac_mgrid=0.2` evidence run diverged at step 4010 rather than completing 10k
> steps; the restriction normalisation (`R = P^T/8**l`, without which the gain
> is wrong by up to 512x); the sign of the two-grid operator, `(I + P G R)`;
> the restriction kernel's pass order (k, j, i -- the reverse of prolongation);
> the scree cadence, which the original wiring did not actually fix; the
> comparison gain, `6*fac_mgrid` total rather than `2*fac_mgrid` per level; and
> several citations. Added: the self-feedback loop through `residual_nd`
> (risk 7, test 5.11), which is the one hazard the reformulation introduces
> rather than inherits, and the tension between mechanism 1 and what
> `mg_stability.md` actually measured (1.1).
>
> **Revised again 2026-08-30** after a second review against the source. The
> substantive fixes: **test 5.11's loop form was inverted** -- pinning
> `residual_nd` while `F_mg_nd` evolves decouples the two, so the assertion
> passed on the broken code and failed on the correct code, and 5.11 is the gate
> on sequencing step 3. **Test 5.4 probed the fine Nyquist mode** and credited
> full weighting with an attenuation the plain block sum already has exactly
> (`D_2(pi) = 0`); it now probes the mid-band, where the two restrictions
> actually differ and where E1 puts the growing mode. Also: the operator order
> is `P G R`, not the `G P R` section 5.5 had (they coincide only at
> `expon_mgrid = 2`, and `run.py` ships 1.414); `A_h` in the 5.5 model must be
> the negated second-difference; there is no `Config` class -- it is the frozen
> `Solver` dataclass, and the RK path needs `dataclasses.replace` because
> `rk_step` reads `conf` directly; `G(0)` collided with `mg_stability.md`'s own
> definition (which includes the fine term) and is now `G_c`, with the 2.34-vs-
> 2.4 predictions reconciled; the residual vector quoted "by step 4000" was
> step 3990's; the `F_body_nd`/`F_mg_nd` writes need the `flags.writeable`
> toggle; equations are now labelled `(MG1)`/`(MG2)` so they no longer collide
> with section numbers; and "no fixed point" is now the accurate "unstable fixed
> point".

The turbine's `run_final` stage **diverges** when the multigrid coarse
correction is on, and holds flat when it is off. At `fac_mgrid=0` the five
residual norms settle to `(1.68, 5.34, 2.07, 4.98, 5.02)e-8` by about step 2500
and hold there to step 10000 (`turbine/results/run_final_fm0_10k.log`, 10000
steps, no divergence). At `fac_mgrid=0.2` they bottom at
`(1.71, 5.37, 2.17, 4.87, 4.96)e-8` around step 1200-1400 (a componentwise
minimum over that range, not any one record), climb monotonically to
`(1.34, 2.97, 2.45, 2.09, 2.20)e-7` at step 4000, the last record before the
run dies -- between 4.3x and 11.3x the bottom depending on which norm you read
-- and the run then fails outright:

```
Solver diverged at step 4010: NaN in conserved_nd density of block 0 ('row0'):
5785 node(s), bbox i[144:144]/144 j[0:88]/88 k[0:64]/64,
touches [i-hi, j-lo, j-hi, k-lo, k-hi]
```

`run_final_fm02_10k.log` therefore records 4010 steps, not 10000, despite its
name. The growth is slow, but it is not benign and it does not level off. The
two runs differ in `fac_mgrid` and nothing else -- both are
`cfl=4.0, n_stage=4, n_levels=3, expon_mgrid=2.0, sf_resid=1.0, gain_filt=1.0,
delta_filt=0.2` from the same `initial.emb`, per the config line each log
prints.

`docs/dev/mg_stability.md` has the one-step von Neumann model for the related
duct CFL penalty; this document is a *reformulation* of the correction, not
another analysis of the existing one. Section 1.1 states where the two
diagnoses agree and where this plan goes beyond what that document measured.

The reformulation: stop applying the coarse correction as an in-stage state
increment and instead apply it as a **cell-centred body force**, assembled once
per step from the start-of-step residual and frozen across the whole RK cycle,
entering the march through `F_body` exactly as the viscous, polar and SFD
sources already do.

---

## 1. What this fixes, and what it does not

### 1.1 The three mechanisms in the current scheme

From the analysis in this session (see also `mg_stability.md`):

1. **Non-adjoint transfer operators, no coarse operator.** Restriction is a
   plain block sum (`scree.f90:838-855`); prolongation is cascaded factor-2
   trilinear interpolation, explicitly "a genuine operator change from a direct
   factor-b prolong" (`scree.f90:708`). The coarse "solve" is a single scaled
   block-Jacobi sweep, `coef * dtblk * restrict(residual)` -- there is no
   coarse discretisation `A_H`. The two-grid operator is therefore not a
   projection and is non-normal, so nothing bounds its spectrum below 1.

   **This is a structural observation, not a measured diagnosis, and
   `mg_stability.md` does not support the strong form of it.** That document
   measured the composite restrict -> smooth -> prolong transfer directly (its
   section "The prolongation is NOT the explanation (measured)") and found it
   well behaved: `G_c = 2.35` against 2.34 predicted, with clustering moving
   it under 2 per cent at every wavenumber (`G_c` is the coarse transfer alone;
   `mg_stability.md`'s own `G(0)` includes the fine term and is 2.35 + 1 = 3.4
   at those settings -- see the note in 2.3). Its E1 experiment found the growing
   mode to be "a long wave of 8 to 10 cells per wave, **not a checkerboard**",
   and it attributes the duct CFL penalty to the *wavenumber band over which
   the coarse correction still has amplitude*, not to a block-Nyquist
   eigenvalue. An earlier draft of this plan claimed growth "for modes near the
   block Nyquist frequency"; E1 contradicts that and it has been removed.

   What survives is compatible with both readings. Full-weighting restriction
   narrows the band over which the correction retains amplitude, which is the
   lever `mg_stability.md` identifies, *and* makes the transfer pair adjoint,
   which is the lever this plan identifies. Test 5.5 is what decides whether
   either buys anything, and it must be read against E1's long-wave mode rather
   than a checkerboard (see 5.5).

2. **The correction bypasses the fine smoothing.** It is injected after
   `Grid.update_residual`'s IRS, as a raw state jump -- "the fine term is never
   smoothed here" (`solver.py:762`, `advance_rk_stage_mg`'s docstring). The
   *coarse* levels are smoothed, each by its own IRS pass at `sf_irs` inside
   the engine (`scree.f90:859-861`, `914-915`); what never happens is the
   correction passing through the fine smoother. See 3.3 for what the
   reformulation does with that coarse pass.

3. **Intra-cycle stage feedback.** The coarse push is recomputed every RK
   stage off the evolving residual, so stage `N`'s push perturbs stage `N+1`'s
   residual which drives stage `N+1`'s push.

### 1.2 Node/cell mismatch blocks the naive "make R = P^T" fix

The final prolongation hop maps coarse-cell centres directly to fine **nodes**
(`mg_bracket2x_node`, `mg_prolong2x_fine_scatter`), scattering onto
`conserved_nd`. Its transpose acts on nodal data; restriction sums the cell
residual. `R` and `P^T` do not act on the same space, so pairing them requires
dragging `cell_to_node` (and its transpose) into the coarse transfer.

### 1.3 What the body-force form changes

`F_body` is **cell-centred** (`f_body(ni-1,nj-1,nk-1,5)`, folded into the cell
residual by `set_residual`, `residual.f90:1122`). Making the coarse correction
a body force therefore makes every coarse<->fine transfer cell-to-cell:

- **Fixes the node/cell mismatch (1.2).** Restriction (fine cells -> coarse)
  and prolongation (coarse -> fine cells) live in one space, so
  `R = P^T / 8**l` is available as a clean (scaled) adjoint pair -- see 2.1 for
  why the scalar is there. The lone `cell_to_node` is applied once, downstream, to
  the *combined* residual (fine flux term + `F_body`), exactly as it already
  is for the fine term. The node-targeted hop and its clamped `MG_W_LO/HI`
  weights -- the "ill-conditioned enough to turn the prolongation into an
  amplifier" case (`Block.weight_mgrid` docstring, `block.py:4083-4085`, which
  points at `_mg_hop_weights_node` at `block.py:1062` and `_mg_project`) -- are
  deleted.
- **Fixes (2).** `F_mg` is added to the residual and then sees the full RK +
  IRS low-pass before it moves the state. The aliased block-Nyquist content in
  `R r` is damped by the same smoother that damps fine high-k content.
- **Fixes (3).** `F_mg` is frozen for the RK cycle. The integrator sees a
  constant forcing, integrated by the RK stability region exactly like the
  polar source. No stage-to-stage loop. It does **not** remove every feedback
  path -- it replaces the intra-cycle one with a per-step one that has to be
  broken explicitly; see 2.1 and risk 7.
- **Restriction runs once per step, not once per stage** -- a 4x reduction in
  call count on RK4. Not a 4x cost saving: a separable three-pass weighted
  restriction costs materially more per call than the fused 8-point block sum
  it replaces, and the final prolongation hop that used to be fused into the
  stage kernel now needs its own pass. The net is expected to be a win, but it
  is not quantified here. The once-per-step refresh is also a crude temporal
  low-pass in its own right.
- **No `dtblk`.** The per-cell fine-term integration (`alpha*cfl*dt_vol`)
  supplies the timestep weighting automatically -- small cells under a block
  get a small push, large cells a large push -- which is arguably better than
  one harmonic-mean block timestep on a stretched mesh.

### 1.4 What it does NOT fix on its own

Nothing here forces `R = P^T`. If restriction stays a plain sum while
prolongation stays trilinear, the adjoint mismatch (mechanism 1) survives; the
body-force form only routes it through a gentler `dt`-scaled channel
(`M = I - dt*(I + P G R) A_h` instead of a jump), so an eigenvalue `> 1` still
grows, just diffusively.

Note the sign. `residual_nd` is the rate of accumulation `dU/dt` (its docstring,
`block.py:3556`), and `F_mg` is *added* to it (section 2.2), so the coarse correction
**augments** the fine push: the operator carries `(I + P G R)`, not the
`(I - P A_H^-1 R A_h)` of a textbook coarse-grid correction, which subtracts.
Earlier drafts wrote `(I - PR)` here and in sections 4 and 5.5, which reads as
an operator that annihilates smooth modes -- the opposite of what this scheme
does. `G` is the diagonal of per-level gains `g_l`.

**Therefore this plan adopts normalised full-weighting restriction
(`R = P^T / 8**l`) as part of the same change.** That is the actual fix for
mechanism 1; the body-force form is what makes it a small, local change
(section 3.2). Whether mechanism 1 is what the turbine is actually dying of is
a separate question that only test 5.5, read against E1's measured mode,
answers -- see the caveat in 1.1.

---

## 2. Formulation

### 2.1 The force

Assembled once per step, frozen across the RK cycle:

```
F_mg  =  sum_{l=1}^{L}  g_l * P_l( R_l( r_step ) )                        (MG1)
```

(Equations are labelled `(MG1)`, `(MG2)`; a bare `(2.3)` or `(2.4)` is always a
section number. An earlier draft numbered the equations `(2.1)`/`(2.2)`, which
collided with the section numbers and made "`F_mg` is added to it (2.2)" read as
a reference to the gain formula rather than to the sign discussion.)

- `r_step` -- the fine cell residual **with the previous step's `F_mg` removed**:
  `r_step = residual_nd - F_mg_nd`. This subtraction is load-bearing, not
  hygiene. `residual_nd` is "net-flow residual **+ body forces**"
  (`block.py:3556`), and `F_mg` is one of those body forces, so restricting
  `residual_nd` raw would feed `F_mg` back into itself once per step with loop
  gain `sum_l g_l` -- which is `6*fac_mgrid = 1.2 > 1` at the turbine's settings
  (2.3). That recursion does have a fixed point -- `F* = G r/(1 - G) = -6 r` at
  that gain -- but with `|G| > 1` it is unstable, so the iteration diverges from
  any starting point, independently of anything `A_h` does. Subtracting
  `F_mg_nd` breaks the loop exactly: `residual_nd = flux + other_sources +
  F_mg`, so `r_step` is `flux + other_sources` and `F_mg` at step `n+1` has no
  dependence on `F_mg` at step `n` at all. Both arrays are cell-shaped
  `(ni-1, nj-1, nk-1, 5)`, so it is one axpy. See risk 7 and test 5.11 -- and
  note that this identity is what test 5.11 has to reproduce, not sidestep.

  What remains is the one-step explicit lag: `residual_nd` at assembly time is
  the previous step's penultimate-stage residual (`rk_step` skips the
  final-stage rebuild, `solver.py:869-871`), evaluated before that step's
  `Grid.smooth`. Same explicit coupling SFD uses for its filter (`solver.py`,
  the `update_filter` comment at ~1055). At convergence `r -> 0` so `F_mg -> 0`
  and the lag introduces no bias (verified: test 5.4).
- `R_l` -- restriction from fine cells to level-`l` coarse cells,
  `b_l = 2**l`. **Cascaded factor-2 full weighting, normalised**: `R_l` is the
  transpose of the cascaded factor-2 trilinear prolongation, divided by `8**l`.

  The normalisation is not optional and cannot be dropped. `P` preserves
  constants (`P 1_coarse = 1_fine`, partition of unity per pass), so the raw
  transpose has column sums of about `8` per hop and `P^T 1_fine ~ 8**l * 1`.
  The raw transpose is therefore a *sum*-like operator, not an average, and
  using it in (MG1) would leave `g_l` wrong by `8**l` -- 512x at level 3.
  Dividing by `8**l` makes `R_l` an averaging operator, so `R_l 1 ~ 1` and the
  gain algebra in 2.3 closes. Adjointness survives the scaling: `R = P^T/8**l`
  is a *scaled* adjoint, and a positive scalar is all that the eigenvalue
  argument needs -- it is what makes `P G R` self-adjoint-like rather than
  non-normal, and the scalar is absorbed into `g_l` anyway.

  `R_l 1 = 1` is **exact only where the column sums are exactly 8**, which
  holds on a uniform mesh (each 1D pass has weights `0.25`/`0.75` and its
  column sums are exactly 2) but not on a stretched one. The departure is
  bounded by the same weight excursion that bounds the prolongation weights
  (a few per cent at expansion ratio 1.2-1.5, `_mg_project` docstring), and it
  is the error bar on (MG2) rather than a defect. Test 5.3 asserts equality on
  a uniform mesh and records the deviation on a stretched one.

  Each hop is a separable weighted average (three 1D passes), so restriction
  telescopes hop-by-hop exactly as the current block sum does -- level `l` is
  one more hop applied to level `l-1`'s result -- so the fine grid is read once
  and the coarser levels reduce small accumulators: `1 + 1/8 + 1/64 = 1.14`
  fine-cell traversals in total, not `L`. (Each traversal is three separable
  passes, so that is a cell-visit count, not a flop count.)
- `P_l` -- prolongation from level-`l` coarse cells to fine cells. Cascaded
  factor-2 trilinear, geometry-aware (volume-weighted centroid position),
  **all hops cell-targeted** including the finest. Reuses the existing
  `_mg_hop_weights` cell-hop machinery; the `_mg_hop_weights_node` path is not
  used.
- `g_l` -- per-level gain, section 2.3.

### 2.2 Sign and where it enters

`set_residual` computes `dU = flux_terms + f_body` (`residual.f90:1122`), and
the state update is `U += alpha*cfl*dt_vol*dU` (`rk_plain`). `F_mg` is added
into `F_body` with a `+` sign, so the coarse correction pushes the state the
same way the block-averaged residual would -- identical in sign and integration
to the current direct-injection scheme.

### 2.3 Gain calibration

Matching the low-wavenumber acceleration of the current direct-injection
scheme. Current scheme, per level, adds to the state (`advance_rk_stage_mg`
docstring, `scree.f90:796`):

```
dU_state,l  ~  coef_l * dtblk_l * cres_l
coef_l      =  alpha*cfl * fac_mgrid / b_l^2 * expon_mgrid^{-(l-1)}
cres_l      =  block SUM of r over b_l^3 cells      ~  b_l^3 * <r>_block
dtblk_l                                             ~  <dt_vol>_block
```

so `dU_state,l ~ alpha*cfl * fac_mgrid * b_l * expon^{-(l-1)} * <dt_vol> * <r>`.

Body force (MG1) integrated by the fine term at stage `alpha`:
`dU_state,l = alpha*cfl*dt_vol_cell * g_l * P_l(R_l(r))`. `P_l` preserves
constants and `R_l` is normalised to average (section 2.1), so
`P_l(R_l(r)) ~ <r>_block`
-- this step is where the `1/8**l` in `R_l` is spent; without it the right-hand
side is `b_l**3 <r>_block` and everything below is wrong by that factor.
Equating, with `<dt_vol>/dt_vol_cell ~ 1`:

```
g_l  =  fac_mgrid * b_l * expon_mgrid^{-(l-1)}  =  fac_mgrid * 2**l * expon_mgrid^{-(l-1)}   (MG2)
```

For `expon_mgrid = 2` (the turbine's `run_final` setting) this collapses to a
**level-independent** `g_l = 2 * fac_mgrid`. For `expon_mgrid = 1.414` it
grows with level (0.40, 0.57, 0.80 at `fac_mgrid=0.2`).

**The total smooth-mode gain matters on its own.** For a mode `P_l R_l` passes
unchanged, `F_mg = (sum_l g_l) * r`. At `expon_mgrid = 2`, `n_levels = 3` that
sum is `6 * fac_mgrid`: 1.2 at `fac_mgrid = 0.2`, 2.4 at 0.4. The coarse
correction is therefore *larger than the fine residual it corrects*, which is
not new -- `mg_stability.md` measures exactly this on the duct, `G_c = 2.35` at
`fac_mgrid = 0.4` against the `6*fac_mgrid = 2.4` this algebra predicts in the
`theta -> 0` limit, an independent confirmation of (MG2). It is new only in that
the body-force form routes that gain through the residual, where a sum above 1
is what makes the `r_step` subtraction of 2.1 mandatory rather than tidy.

**Note the symbol.** `mg_stability.md:159` defines `G(0) = 1 + 2*fac_mgrid *
n_levels`, which is `3.4` at `fac_mgrid = 0.4, n_levels = 3` -- the leading `1`
is the fine term. This document is only ever talking about the coarse transfer
*without* that `1`, so it writes `G_c = G(0) - 1 = 2.35`. Do not read the two
numbers as disagreeing. The same distinction applies to the two predictions
quoted for it: `2.34` is `mg_stability.md`'s hat model evaluated at its longest
tabulated wave, 72 cells per wave (`mg_stability.md:945`), and `2.4` is the same
model's `theta -> 0` limit. `2.35` measured sits between them, as it should.

(MG2) is a starting point, not a claim of equivalence -- the body-force scheme
has a different stability boundary and its own optimal gain. Introduce a fresh
knob rather than overloading `fac_mgrid`:

- `mgrid_bf` : bool, default `False` -- selects the body-force scheme.
- `fac_mgrid` : reused as the `fac_mgrid` in (MG2). `n_levels`, `expon_mgrid`
  reused unchanged.

With `mgrid_bf=False` the code path, kernels and results are **byte-identical**
to today (test 5.1).

### 2.4 Interaction with the scree march

`update_sources` runs every step for RK (`n_stage != 0`) but only every 5th step
for scree (`solver.py:1044-1046`, `n_step_source`). It also **zeroes and
rebuilds** `F_body_nd` from scratch on each of those calls (its docstring,
`grid.py:1587`).

Assembling `F_mg` in its own per-step pass is therefore not by itself enough:
if the `F_mg_nd -> F_body_nd` addition lives inside `update_sources`, then on
the scree march a force freshly assembled on steps 1-4 of every five is simply
discarded, and the lag is five steps, not one. This is precisely the lag SFD
already carries -- `update_filter` runs every step, but the filter only reaches
`F_body` through `update_sources` -- so the `update_filter` analogy argues the
opposite of what an earlier draft claimed.

The fix is to apply the MG force as a **delta** (section 3.1):
`update_mg_source` holds the previous `F_mg_nd`, assembles `F_mg_new`, and does
`F_body_nd += F_mg_new - F_mg_nd` before storing `F_mg_new` back. When
`update_sources` does run it re-zeroes `F_body_nd`, rebuilds the other sources
and re-adds the current `F_mg_nd` at the end, so the delta chain restarts from
an exact value at most every fifth step and float32 drift cannot accumulate
past four increments. Ordering in `_run` is `update_sources` first, then
`update_mg_source`.

The turbine uses RK4, where `n_step_source == 1` and the delta reduces to the
obvious thing, so this is a correctness-parity concern for the duct/scree
tests, not for the target case.

---

## 3. Code changes

### 3.1 New per-step assembly pass

`ember/grid.py`:

- `Grid.update_mg_source(fac_mgrid, n_levels, expon_mgrid)` -- new method.
  For each block:
  1. form `r_step = block.residual_nd - block.F_mg_nd` into scratch (2.1 --
     this is what breaks the self-feedback loop);
  2. restrict `r_step` through `L` cascaded full-weighting hops, normalising
     each hop by `1/8`;
  3. scale each level by `g_l` (MG2) and prolong back through `L` cascaded
     trilinear hops, accumulating into a scratch `F_mg_new`;
  4. apply the delta: `block.F_body_nd += F_mg_new - block.F_mg_nd`, then copy
     `F_mg_new` into `block.F_mg_nd` (2.4).

  At `fac_mgrid == 0` or `n_levels == 0` the transfer work in steps 2-3 is
  skipped and `F_mg_new` is taken as zero, but step 4 still runs, so turning
  the correction off mid-run drives `F_mg_nd` to zero through the same delta
  rather than stranding a stale force in `F_body_nd`. It is not an early
  return.

  `F_body_nd` and `F_mg_nd` are read-only cached arrays, so the writes in
  step 4 need the `flags.writeable` toggle `Grid.update_sources` already
  brackets its own assembly with (`grid.py:1619`, `1775`; `Grid.update_residual`
  does the same for `residual_nd` at `grid.py:1545`). `F_body_nd`'s docstring
  currently names two owners, "``Grid.update_sources`` and the FAS
  coarse-forcing assembly" (`block.py:3093-3106`), but no FAS assembly exists in
  the tree today -- so that docstring is stale and this change makes
  `update_mg_source` the genuine second writer. Update it in the same commit.

  Borrows `block.scratch` -- a distinct phase from the stage march, so the arena
  is free (section 3.4).
- `Grid.update_sources` -- after the SFD sub-phase, add `block.F_mg_nd` into
  `block.F_body_nd` (one axpy per block) when `mgrid_bf` is active. This is the
  re-seed after the method's own `F_body_nd.fill(0.0)`, not the primary path;
  the primary path is the delta above. `F_mg_nd` is a no-key cached buffer like
  `F_body_nd`, cell-shaped `(ni-1, nj-1, nk-1, 5)`, zeroed on allocation.

`ember/solver.py`, `_run` step loop:

```
grid.update_sources(conf.inviscid, conf.gain_filt)                       # every step (RK) / every 5th (scree)
grid.update_mg_source(conf.fac_mgrid, conf.n_levels, conf.expon_mgrid)   # every step, if conf.mgrid_bf
```

both placed before `update_residual` so the step-top residual, every RK stage
(including stage 0), and every between-stage `update_residual` all see the same
frozen `F_mg`. `update_sources` runs first because it is the one that zeroes
`F_body_nd`. `update_mg_source` reads `residual_nd` as it stands here (previous
step's penultimate stage, pre-`Grid.smooth` -- the intended one-step lag).

### 3.2 Restriction kernel (the `R = P^T` fix)

`ember/_fortran/`:

- New `mg_restrict2x_acc` -- separable factor-2 full-weighting restriction, the
  transpose of `mg_prolong2x_acc` scaled by `1/8`. One hop scatters each fine
  cell's value onto its bracket with the prolongation's own weights: for each
  coarse cell `c`,
  `(1/2) * [ sum_{f: lo(f)=c} (1-w_f) * F[f] + sum_{f: hi(f)=c} w_f * F[f] ]`
  per 1D pass, three passes, same weight arrays as prolongation
  (`weight_mgrid_cell`), read in transpose.

  **The passes must run in the reverse order to prolongation: k, then j, then
  i.** Prolongation is a composition `P = P_k P_j P_i` -- `_mg_hop_weights`
  resolves `i` first, then `j`, then `k`, and each pass is *anchored on the
  positions the previous one produced* (`Pci`, `Pcij` in that function). The
  transpose of a composition reverses it, `P^T = P_i^T P_j^T P_k^T`, so the
  restriction must consume `wk` first and `wi` last. An earlier draft specified
  "i then j then k", which is not the adjoint of anything and would fail test
  5.2 on a stretched mesh (it happens to coincide on a uniform one, where the
  passes commute, so a uniform-mesh test would not catch it).
- The finest restriction hop is fine-cell -> coarse-cell, so it needs a
  cell-hop weight set for level 0->1. Add `Block.weight_mgrid_cell` (all hops
  cell-targeted, built from `_mg_hop_weights` at every level including the
  finest) alongside the existing `weight_mgrid` (which keeps its node-targeted
  final hop for the legacy path). Cached, geometry-keyed, dropped by
  `clear_cache`.

### 3.3 Config plumbing

- `ember/solver.py` -- add `mgrid_bf: bool = False` to the frozen `Solver`
  dataclass (`solver.py:326`). There is no class named `Config`: `Solver` *is*
  the configuration object, and the `conf` that `_run(grid, conf)` and
  `rk_step(grid, conf)` take is the `Solver` instance passed as `self` from
  `Solver.run` (`solver.py:491`).
- `run.py` `DEFAULTS` -- add `"mgrid_bf": False`; CLI `--mgrid-bf`
  (store_true).
- `advance_rk_stage_mg` / `scree_step` -- no signature change, but the two
  integrators are reached differently and only one of them can be told anything.
  `_run` calls `scree_step(grid, conf.cfl, fac_mgrid=..., n_levels=..., ...)`
  with explicit keywords (`solver.py:1078-1085`), so the scree path can simply
  be handed `fac_mgrid=0.0, n_levels=0`. The RK path is `rk_step(grid, conf)`
  (`solver.py:1087`), and `rk_step` reads `conf.fac_mgrid` and `conf.n_levels`
  off the dataclass itself (`solver.py:862-869`) -- there is no argument to
  override. So `_run` must build `dataclasses.replace(conf, fac_mgrid=0.0,
  n_levels=0)` once outside the step loop and pass that to `rk_step` when
  `conf.mgrid_bf` is set (`replace` is already the idiom here -- `_run_fmg` uses
  it at `solver.py:1196`). Either way the integrator takes the plain
  (`rk_plain` / `scree_plain`) fast path, the in-stage coarse machinery is fully
  bypassed, and the correction arrives only through `F_body`.
  `_validate_mg(grid, conf.n_levels)` still runs on the *configured* level count
  -- so it must be given the original `conf`, not the replaced one, which is
  what the body-force path needs too.
- **`sf_irs` no longer reaches the coarse levels, by design.** The current
  engine IRS-smooths each restricted level before scaling it
  (`smoother(cres, sf_irs, ...)`, `scree.f90:859-861` and `914-915`).
  `update_mg_source` has no smoother, so under `mgrid_bf` a nonzero `sf_resid`
  smooths the fine residual only. That is the intent -- the argument in 1.3 is
  that the downstream fine IRS applied to `flux + F_body` replaces it, and
  applying both would low-pass the correction twice -- but it is a real
  behaviour change and not a free consequence of the reformulation. If the
  section 6 duct sweep shows it costs convergence rate at `sf_resid = 1`, the
  coarse smoother comes back as a per-level pass inside `update_mg_source`.

### 3.4 Scratch arena

`ember/block.py` `_scratch_len` -- add the `update_mg_source` phase to the
per-phase accounting. It needs: two fine-cell-shaped buffers, `r_step` and
`F_mg_new` (`(ni-1)*(nj-1)*(nk-1)*5` each -- the residual subtraction and the
prolongation target of 3.1, both of which the in-stage kernel had no analogue
for because it fused straight onto the state), the cascaded restriction
accumulators (hop-by-hop, so ~`nc1` sized), the packed per-level scaled
corrections (`n_corr`, reused from `_mg_coarse_scratch_sizes`), and the prolong
ping-pong + separable-pass scratch (`acc0`/`acc1`/`aplane`/`bb`, already sized
by `mg_coarse_shapes`). No node-shaped buffers, no IRS buffers, no `dtblk`. The
two fine-shaped buffers are the reason this phase may NOT fit inside the
existing multigrid stage phase despite dropping the node-shaped work -- the
sizing below decides it, and if it binds, the arena grows. This
sizing is computed, not asserted -- `_scratch_len` gains the term and
`tests/test_scratch_arena.py::test_scratch_len_covers_every_phase` verifies it
across the standard shape set. (The multigrid stage phase is the one that binds
today, per that file's `test_arena_is_smaller_than_the_buffers_it_replaced`.)

### 3.5 No persistent history

The once-per-step refresh is the temporal smoothing. There is no EMA and no
history buffer (that was a separate band-aid proposal for the legacy scheme).
`F_mg_nd` is recomputed each step from scratch.

---

## 4. What "correct" means here

Five properties, each with an isolated test in section 5:

1. **Adjoint transfer, up to the normalising scalar.**
   `<P x, y>_fine = 8**l * <x, R y>_coarse` to float32 rounding, for the
   cell-hop weights, at every level and on a stretched mesh. Equivalently
   `R = P^T / 8**l`. The scalar is the normalisation of 2.1 and is absorbed
   into `g_l`; what the property asserts is that `R` and `P^T` are parallel,
   which is what removes the non-normality.
2. **Constant preservation.** `P 1_coarse = 1_fine` exactly (partition of
   unity: each pass blends two weights summing to 1, and the cell-targeted hops
   clip to `[0, 1]`, so this holds on any mesh). `R 1_fine = 1_coarse` exactly
   on a uniform mesh; on a stretched mesh it holds only to the departure of the
   transposed column sums from 8, which is bounded but nonzero (section 2.1). Property
   1 and an exact `R 1 = 1` on a stretched mesh are **not** simultaneously
   achievable, and the plan takes property 1 as the binding one.
3. **Consistency.** `F_mg(r) -> 0` as `r -> 0`, componentwise and to machine
   zero when `r == 0` exactly. No steady-state bias: the fixed point of the
   outer iteration still has `r* = 0`.
4. **Contractivity.** The numerically-formed two-grid operator
   `M = I - dt*(I + P G R) A_h` (`G` the diagonal of per-level gains `g_l`;
   the `+` per 1.4) has spectral radius `<= 1` on a model problem where the
   plain-sum + trilinear pairing has `rho > 1`.

   **The operator order is `P G R`, not `G P R`.** `G` is diagonal on the
   *stacked coarse levels*, so it has to sit between `R` and `P`; `G P R` puts
   it in the fine space, where it is not even the right shape unless every `g_l`
   is the same number. The two coincide only at `expon_mgrid = 2`, where (MG2)
   makes `g_l = 2*fac_mgrid` for every level and `G` degenerates to a scalar --
   and `run.py` ships `expon_mgrid = 1.414`, where the gains are
   `(0.40, 0.57, 0.80)` at `fac_mgrid = 0.2` and the order matters. Write it
   `P G R` everywhere, section 5.5 included.
5. **No self-feedback.** `F_mg` assembled at step `n+1` does not depend on
   `F_mg` at step `n`, *given the coupling the march actually presents* --
   `residual_nd` at step `n+1` already contains step `n`'s `F_mg`, so the
   property is that `F_mg` is a function of `flux + other_sources` alone. That
   qualifier is the whole content of the property and test 5.11 has to reproduce
   it rather than pin `residual_nd` independently of `F_mg_nd`. Without the
   `r_step` subtraction of 2.1 the recursion becomes
   `F_{n+1} = G(flux + other + F_n)` with smooth-mode gain
   `sum_l g_l = 6*fac_mgrid`, which exceeds 1 at the turbine's settings.

Plus the usual harness contracts:

6. **Off is off.** `mgrid_bf=False` reproduces today byte-for-byte.
7. **Frozen.** `F_mg` does not change across the four RK stages of a step.
8. **Wiring.** A nonzero gain actually reaches `conserved_nd`; the assembly
   runs once per step, not per stage, and (scree) reaches `F_body_nd` every
   step, not every fifth.

---

## 5. Isolated tests

New file `tests/test_mg_body_force.py` unless noted. Patterns follow
`test_mg_weights.py` (weight/transfer algebra), `test_scree_mg.py` (kernel
parity contracts) and `test_sfd.py` (march wiring).

### 5.1 `mgrid_bf=False` is byte-identical to today

`test_off_is_byte_identical` -- build a small duct grid, run N steps twice with
identical seeds, once on `main`-equivalent config and once with
`mgrid_bf=False` explicitly set. `conserved_nd`, `residual_nd` and the
convergence history must match bit-for-bit. Guards every downstream refactor.

### 5.2 Adjoint of the transfer pair -- THE test for mechanism 1

`test_restriction_is_prolongation_transpose`:

- Build a single `Block` on a **stretched** mesh (wall clustering in `j`,
  shear in `k` -- the `test_mg_weights.py` "shape that made the LISA rotor
  diverge" geometry).
- For each hop `h = 1..L`, extract that hop's weight slices for `P`
  (`weight_mgrid_cell`) and form the **explicit dense matrices** `P_h`
  (shape `n_fine_h x n_coarse_h`) and `R_h` (`n_coarse_h x n_fine_h`) by
  applying the kernels to unit vectors, one column at a time:
  `P_h[:,k] = mg_prolong2x_acc(e_k)` for `e_k` in the coarse space, and
  `R_h[:,k] = mg_restrict2x_acc(e_k)` for `e_k` in the fine space. Applying an
  operator to a unit vector yields a **column**, not a row; an earlier draft
  wrote `R_h[k,:]`, which transposes `R` a second time and makes the assertion
  below pass for the wrong kernel.
- Both kernels **accumulate** onto their output (`mg_prolong2x_acc` is
  `out += interp_2x(src)`, `scree.f90:311-314`), so zero the output buffer
  before every unit-vector call or the columns are cumulative sums.
- Assert `R_h == P_h.T / 8` to `< 1e-6` relative (float32 kernels). The `/8` is
  the per-hop normalisation of 2.1; without it this assertion is testing an
  operator the gain derivation does not use.
- Assert the same for the **cascaded** operators `P = P_1 P_2 ... P_L` (level
  `L` to fine) and `R = R_L ... R_2 R_1`: `R == P.T / 8**L`.
- Do it for `np=1` and `np=5` (the transfer must not mix equations).

This is the property that pulls the two-grid eigenvalues back below 1 and is
the whole reason the reformulation is worth doing. If it fails, nothing else
matters.

### 5.3 Constant preservation

`test_transfer_preserves_constants`:

- `P_h @ ones == ones` to `< 1e-6`, per hop and for the cascade -- prolongation
  of a flat coarse field is flat (partition of unity: the two per-direction
  weights sum to 1, and the cell-targeted hops clip to `[0,1]`). Holds on a
  stretched mesh as well as a uniform one.
- `R_h @ ones == ones` to `< 1e-6` **on a uniform mesh only**, per hop and for
  the cascade. This is exact there because each 1D pass carries weights
  `0.25`/`0.75` and its transposed column sums are exactly 2, so the `/8` of
  2.1 normalises it exactly.
- On a **stretched** mesh do not assert equality: `R = P^T/8**l` and
  `P 1 = 1` together force the transposed column sums to be whatever the
  geometry makes them, and they are only approximately 8. Instead record
  `max|R 1 - 1|` and assert it stays under a pinned tolerance (start at 0.1 and
  tighten to whatever the clustered duct geometry actually produces). That
  number is the error bar on the (MG2) gain and belongs in the test output, not
  hidden behind an `assert_allclose`.
- Note on notation: `_h` is a single factor-2 **hop**, for which the
  normalisation is `8`; the composite operator to level `l` is `L`-many hops
  and normalises by `b_l**3 = 8**l`. An earlier draft mixed the two.

### 5.4 Consistency: the force vanishes with the residual

`test_force_vanishes_at_convergence` (kernel level):

- `update_mg_source` fed `residual_nd == 0` and `F_mg_nd == 0` writes
  `F_mg_nd == 0` exactly (bit zero, not `< eps`).
- Fed `residual_nd = c` (a per-equation constant field) and `F_mg_nd == 0`,
  `F_mg_nd` is the correction to a constant -- assert it equals
  `sum_l g_l * c` per equation, which is `6*fac_mgrid*c` at
  `expon_mgrid = 2, n_levels = 3` (2.3). Exact on a **uniform** mesh only,
  where `R 1 = 1` is exact; on a stretched mesh assert to the tolerance test
  5.3 pinned. A constant residual is a genuine imbalance, so a nonzero force
  there is correct -- and note that this constant is `1.2c` at
  `fac_mgrid = 0.2`, i.e. the correction exceeds the residual, which is what
  makes test 5.11 necessary.
- Fed `residual_nd = A * sin(2 pi x / L)` with wavelength `>> b_L` cells,
  `F_mg_nd` recovers `~ (sum_l g_l) * residual` (smooth mode passes the
  transfer nearly unchanged).

  **Do not probe this at the fine Nyquist mode.** An earlier draft asked for
  wavelength `= 2` cells and claimed the resulting attenuation was "the property
  a plain sum lacks". It is not: `mg_stability.md:170-172` gives the block-sum
  transfer `D_b(theta) = sin(b*theta/2)/sin(theta/2)`, which at `theta = pi,
  b = 2` is `sin(pi)/1 = 0` exactly -- each 2-cell block sums to zero, so the
  plain sum annihilates that mode too. The assertion would pass on either
  restriction and discriminate nothing, and it probes the one band E1 has
  already ruled out as the growing mode (1.1).

  Probe the **mid-band** instead, at 4 and 8 cells per wave, which is where the
  `D_b/b` factor separating full weighting from the plain sum actually lives and
  where E1 puts the mode that grows. Assert the ratio `|F_mg_nd|/|residual|`
  drops by at least 10x between the long wave and 4 cells/wave, and record the
  8 cells/wave value rather than asserting on it -- that number is the one to
  compare against `M_fw`'s wavenumber sweep in 5.5, and pinning a threshold on
  it before the sweep has run would be guessing.

### 5.5 Contractivity on a model problem

`test_two_grid_operator_is_contractive` (small, dense, offline):

- 1D or 2D scalar model on a mildly stretched grid, `dt` at the explicit
  stability limit. Size the grid so it resolves a **long** wave of 8 to 10 cells
  per wave, because that -- not a checkerboard -- is the mode
  `mg_stability.md`'s E1 measured as the one that actually grows (1.1). A model
  that only goes unstable at the Nyquist mode is reproducing a different
  instability than the turbine's and proves nothing here.
- `A_h` is the **negated** second-difference, i.e. positive semi-definite. The
  sign is what makes the minus in `M = I - dt*(...)A_h` correct:
  `residual = dU/dt = -A_h U`, so a plain step is `M = I - dt*A_h` and decays.
  Taken as the bare (negative semi-definite) second-difference, that same
  expression grows for a trivial reason that has nothing to do with the transfer
  pair, and every result below is meaningless.
- Form `M_sum = I - dt*(I + P G R_sum) A_h` with **plain-sum** restriction +
  trilinear prolong (the current pairing) and show `rho(M_sum) > 1`. Note the
  `+`: the correction augments the fine push (sections 1.4 and 2.2). If it comes
  out `-`, the model is a textbook coarse-grid correction and is not this
  scheme. Note also the operator order `P G R` -- `G` is diagonal on the stacked
  coarse levels, not on the fine grid (section 4, property 4); an earlier draft
  wrote `G P R`, which is only defensible at `expon_mgrid = 2` where every
  `g_l` is equal.
- Form `M_fw` = same but normalised full-weighting restriction
  (`R = P^T/8**l`), show `rho(M_fw) <= 1` (allow `1 + 1e-6`).
- **Also report `rho(M_sum)` and `rho(M_fw)` as a function of wavenumber**, not
  just the maximum, and check the result against E1's measured band. If
  `M_fw` is contractive only because it damps the Nyquist end while leaving the
  8-10 cells/wave band above 1, the reformulation does not address the observed
  failure and step 2 of the sequencing should stop there.
- Sweep `g` (the gain) and report the stability boundary `g_max` for `M_fw`.
  Compare it against the **total** gain `sum_l g_l = 6*fac_mgrid`, not the
  per-level `g_l`: 1.2 at `fac_mgrid = 0.2` (2.3). Pre-register the
  expectation: `g_max >= 6*fac_mgrid` at `fac_mgrid = 0.2`, i.e. (MG2) is
  usable as-is. An earlier draft compared against `2*fac_mgrid`, which is one
  level's share and understates what the scheme actually applies by 3x.

Not a march test -- pure linear algebra on assembled matrices, fast, and it is
the quantitative check that the reformulation removes the growth rather than
slowing it.

### 5.6 Frozen across the RK cycle

`test_force_frozen_over_stages` -- instrument `advance_rk_stage_mg` (or wrap
`set_residual`) to record `F_body_nd` at each of the 4 stages of one step;
assert the `F_mg` contribution is identical across all four. Mirrors
`test_sfd.py::test_the_filter_advances_every_step` in spirit.

### 5.7 Assembly cadence

`test_mg_source_assembled_once_per_step` -- spy on `Grid.update_mg_source`;
assert one call per step on the RK march, and (separate duct/scree case) one
call per step on the scree march too (not one per five, unlike the viscous
sources).

`test_mg_force_reaches_f_body_every_scree_step` -- the call count is not the
property that matters; the delta of 2.4 is. On a scree march (`n_stage=0`),
record `F_body_nd` at the top of ten consecutive steps and assert its MG
contribution tracks `F_mg_nd` on **every** step, including the four in five on
which `update_sources` does not run. This is the test that would have caught
the original design, where the addition lived inside `update_sources` and the
force was silently discarded on four steps out of five.

### 5.8 End-to-end wiring

`test_gain_reaches_conserved` -- duct grid, short march, `mgrid_bf=True` with a
nonzero `fac_mgrid` vs `fac_mgrid=0`; assert `conserved_nd` differs by more
than rounding. The `mgrid_bf` analogue of
`test_sfd.py::test_the_damping_changes_the_solution`.

### 5.9 Scratch arena

Extend `tests/test_scratch_arena.py` -- the new `update_mg_source` phase is
covered by the existing "every phase fits, sizes are computed not literal"
sweep across the standard shape set.

### 5.10 Golden

`test_set_F_body_golden.py` -- add an `mgrid_bf` arm (fixed grid, fixed
residual, committed `F_mg` golden) so a silent change in the assembled MG
force is caught, same as the viscous/polar/SFD golden already does.

### 5.11 The force does not feed on itself

`test_force_is_independent_of_previous_force` -- the test for property 5, and
the one that guards the `r_step = residual_nd - F_mg_nd` subtraction of 2.1.

- Set `residual_nd = r` and `F_mg_nd = 0`; call `update_mg_source`; record
  `F_a`.
- Reset to `residual_nd = r + d` and `F_mg_nd = d` for an arbitrary nonzero
  cell field `d` -- the state the march actually presents, since `residual_nd`
  already contains the previous step's `F_mg` (`block.py:3556`: "net-flow
  residual **+ body forces**"). Call `update_mg_source`; assert the result
  equals `F_a` to float32 rounding.
- Without the subtraction the second call returns `F_a + G(d)`, and iterating
  it multiplies the force by `sum_l g_l = 6*fac_mgrid` per step -- 1.2 at
  `fac_mgrid = 0.2`, so the error compounds rather than decays.
- Then the loop form, since the above only tests one step. **The loop must
  re-establish the march's coupling before every call**, not pin `residual_nd`:
  hold the *flux-plus-other-sources* part fixed at a constant `c` and set
  `residual_nd = c + F_mg_nd` immediately before each of twenty consecutive
  `update_mg_source` calls. With the subtraction in place `r_step` is `c` every
  time, so `F_mg_nd` is `6*fac_mgrid*c` from call 1 onwards and must be
  identical after call 2 and call 20. With the loop open, `r_step` is
  `c + F_n` and the force grows geometrically by 1.2x per call at
  `fac_mgrid = 0.2` -- an unmistakable ~38x over that span.

  An earlier draft instead pinned `residual_nd = c` across all twenty calls,
  which inverts the test. Pinning it decouples `residual_nd` from `F_mg_nd`,
  so the *broken* code sees `r_step = c` every call and returns a constant
  `1.2c` (the assertion passes), while the *correct* code sees
  `r_step = c - F_n` and produces `1.2c, -0.24c, 1.488c, -0.586c, ...` --
  amplitude growing 1.2x per call, sign alternating, assertion failed. Since
  5.11 gates step 3 of the sequencing, that draft would have either blocked the
  work or waved through the very omission it exists to catch. `residual_nd`
  containing `F_mg` is not incidental to this test; it is the entire property
  (section 2.1, section 4 property 5).

---

## 6. Integration / validation runs

Not unit tests; run by hand after 1-5 pass.

- **Duct MG case** (`tools/run_duct_mg.py`, the `mg_stability.md` workhorse):
  sweep `fac_mgrid` for `mgrid_bf=True` at `n_levels=3`, `expon_mgrid=2`,
  `sf_resid` in `{0, 1}`. Compare the CFL penalty against the direct-injection
  scheme's measured limits -- `mg_stability.md` section 2 ("What was observed")
  for the raw brackets and section 4 ("Validation against the measured CFL
  limits") for the model comparison. (Its section 7 is "Experiments that would
  test or break this", not the penalty data; an earlier draft cited it.)
  Success = the body-force scheme's penalty is lower at matched asymptotic
  convergence rate, or it tolerates a higher `fac_mgrid` before the penalty
  appears. Run `sf_resid = 1` deliberately, since that is the arm where
  dropping the coarse IRS (3.3) could show up.
- **Turbine `run_final`**, 10k steps, three arms:
  - A: `fac_mgrid=0` (multigrid off) -- have it, 10000 steps, converged flat.
  - B: `fac_mgrid=0.2`, direct injection -- have it, but it is a 4010-step run
    that ended in NaN, not a 10k comparison. It bounds when B fails; it gives
    no deep-convergence number to compare C against.
  - C: `fac_mgrid=0.2` (or `g_max`-informed), `mgrid_bf=True`.
  Success = C reaches 10000 steps without diverging and converges deeper than
  A. Merely "not drifting like B" is too weak a bar now that B is known to
  diverge outright.
- Full-multigrid startup (`Solver.run_fmg`, `test_fmg.py`) still converges with
  `mgrid_bf` on -- the FMG schedule and the in-step correction are orthogonal,
  but check the coarse levels' histories.

---

## 7. Risks and open questions

1. **Cascaded full-weighting restriction is the transpose of *cascaded*
   trilinear prolongation, not of a direct factor-b prolong.** That is fine --
   `P` and `R` only need to be transposes of *each other*, which cascaded
   hops-vs-transposed-hops guarantees hop by hop. But the composite `P R` is
   then a wider stencil than a two-grid textbook `P R`; the contractivity test
   (5.5) must use the *cascaded* operators, not idealised ones.
2. **Gain (MG2) is derived by magnitude-matching, not by stability analysis.**
   Test 5.5 gives `g_max`; if `g_max < 6*fac_mgrid` (the total, 1.2 at
   `fac_mgrid = 0.2` -- 2.3) the turbine arm C needs a detuned gain and the
   "drop-in `fac_mgrid`" story weakens.
3. **One-step residual lag.** Near convergence harmless (5.4). During a fast
   transient the lagged `r_step` is stale; the turbine's `run_final` starts
   from a converged `initial.emb` so this is low-risk there, but the duct cold
   start may need `mgrid_bf` gated on until the residual settles, or a first-N
   -step ramp.
4. **No boundary masking** (explicit user decision). The frozen `F_mg` at a
   BC-controlled node is fought by `apply_bconds` every stage. At convergence
   `F_mg -> 0` so the steady state is unaffected, but the transient may show a
   thin BC-adjacent artefact. Watch the near-boundary residual in run C; add
   masking only if it bites.
5. **`weight_mgrid_cell` roughly doubles the cached weight storage** on any
   block that runs both paths in one process (tests). `weight_mgrid` costs
   ~14 bytes/fine-cell, most of it hop 1's node-targeted `wk` (one float per
   node); `weight_mgrid_cell`'s hop 1 is cell-targeted and so is somewhat
   smaller, making this a bit under 2x rather than exactly 2x. Acceptable
   either way; drop via `clear_cache`.
6. **scree parity.** Hoisting the MG assembly out of the 5-step source cadence
   (2.4) changes the scree march's per-step cost. Negligible (no viscous in
   the MG pass) but `test_scree_mg.py`'s timing-adjacent assumptions should be
   re-read.
7. **Self-feedback through the residual -- the sharpest risk in this plan, and
   one the current scheme does not have.** `residual_nd` includes `F_body`, so
   assembling `F_mg` from it closes a loop with smooth-mode gain
   `sum_l g_l = 6*fac_mgrid = 1.2` at the turbine's settings. That recursion
   does have a fixed point, `F* = G r/(1 - G) = -6 r`, but with `|G| > 1` it is
   unstable, so the iteration runs away from any starting point whatever `A_h`
   does. Section 2.1's `r_step = residual_nd - F_mg_nd` subtraction is what
   closes it, test 5.11 is what proves it, and neither is optional. The
   direct-injection scheme escapes this because it writes to the state and never
   into the residual -- so this is a hazard the reformulation *introduces*, not
   one it inherits. Note also that the two-grid model of section 4 property 4
   and test 5.5 does not represent it: 5.5 passing tells you nothing about
   this loop.
8. **The gain sum exceeds 1 even with the loop broken.** Independently of
   risk 7, `F_mg` is 1.2x the residual it corrects for a smooth mode. That is
   how the existing scheme is calibrated (`mg_stability.md` measures a coarse
   transfer `G_c = 2.35` at `fac_mgrid = 0.4`; its own `G(0)` for the same
   settings is `1 + G_c = 3.4`, see 2.3) and is not new, but the body-force form
   pushes it through the RK stability region rather than as a state jump, so
   the setting that is stable there need not be stable here. `g_max` from
   test 5.5 is the number that decides it.

---

## 8. Rollback

`mgrid_bf` defaults `False`; the legacy in-stage path (`advance_rk_stage_mg`
with `fac_mgrid > 0`, `mg_coarse_correction`, `mg_prolong2x_fine_scatter`,
`weight_mgrid`) is untouched and remains the default. Removing the feature is
deleting `update_mg_source`, `F_mg_nd`, `mgrid_bf`, `weight_mgrid_cell`,
`mg_restrict2x_acc`, the `F_mg_nd` re-seed at the end of `update_sources`, and
the `_run` branch. No golden or config migration for users who never set
`mgrid_bf`.

The one piece that is not local is the `F_body_nd` delta bookkeeping in
`update_mg_source` (2.4). It writes into a buffer `update_sources` is today the
only real owner of, and this change makes `update_mg_source` a genuine second
writer -- so removing the feature means checking that nothing else has since
come to rely on `F_body_nd` being mutated outside `update_sources`, and
restoring the docstring's ownership list (3.1) to match.

---

## 9. Sequencing

1. `mg_restrict2x_acc` + `weight_mgrid_cell` + tests 5.2, 5.3 (pure transfer
   algebra, no march). This is the load-bearing part; if 5.2 will not pass,
   stop.
2. Model-problem contractivity 5.5 (offline linear algebra). Confirms the
   reformulation removes the growth and fixes `g_max`.
3. `update_mg_source` + `F_mg_nd` + `_scratch_len` + tests 5.4, 5.9, **5.11**.
   5.11 gates the rest: with the feedback loop open the march diverges for
   reasons no later test explains.
4. `_run` wiring + `Config`/`run.py` + tests 5.1, 5.6, 5.7, 5.8.
5. Golden 5.10.
6. Full suite (`make test`), then the section 6 runs.
