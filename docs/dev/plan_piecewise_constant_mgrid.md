# Plan: piecewise-constant multigrid, and nothing else

> **Status: kernel and tests built behind `mgrid_pwc`; validation part-run.**
> Sections 1-4 and 7 steps 1-2 are done. Of section 5: the duct has a first
> matched-gain point (5.2) and the turbine arm is mid-flight with a
> preliminary POSITIVE result (5.7). The screen's duct margin did not
> reproduce; the turbine stability difference did. See 5.2 and 5.7.
>
> The whole change: **replace the cascaded trilinear prolongation with plain
> injection.** Every fine cell under a coarse block takes that block's
> correction, unaltered. Restriction stays the block sum. Phase 1 of
> `mg_coarse_correction` -- the hierarchical restriction and the coarse
> timestep -- is untouched.
>
> This deletes far more than it adds, and the evidence says it converges better
> than what it replaces. It is the opposite direction from
> `plan_body_force_multigrid.md` and the stagewise correction plan (branch
> `stagewise-mg-correction`, commit `7a66715`, not present on this branch),
> which both made the transfers *more* elaborate (adjoint, geometry-aware,
> cell-to-cell) and both measured as worth nothing.

---

## 1. The scheme

For fine cell `c`, at RK stage `alpha`:

```
dU_c  =  alpha*cfl*dt_vol_c*r_c  +  sum_{l=1}^{L} coef_l * dtblk_l * cres_l( c >> l )     (PC1)
```

`cres_l` is the block sum of the residual over the `b_l**3 = 8**l` cells of the
level-`l` block, `dtblk_l` the block's volume-weighted harmonic mean of
`dt_vol`, and `coef_l = alpha*cfl*fac_mgrid/b_l**2 * expon_mgrid**-(l-1)` --
**all three exactly as master already computes them**. `c >> l` is integer index
division: which level-`l` block the cell belongs to.

The only thing that changes is what the second term used to be: a cascade of
factor-2 trilinear interpolations, the last of them targeting the fine NODES
through geometry-derived weights. It becomes a lookup.

`c >> l` is spelled `(i+1)/2` per hop, with no bracket and no clamp. That is
safe only because `ember.solver._validate_mg` forces every cell dimension to be
a multiple of `2**n_levels`, so each level's grid is exactly twice the next
coarser one. The cascade degrades gracefully without that rule --
`mg_bracket2x` clamps -- and injection does not: it would read past the end of
the coarser slot. If that check is ever relaxed, this breaks silently.

### 1.1 Why this should be better and not merely simpler

- **The correction is what a coarse solve would give.** A real coarse grid
  produces one number per coarse cell. (PC1) applies exactly that, uniform over
  the block. The trilinear cascade instead smears each coarse value over its
  neighbours, which is a defensible interpolation of a coarse *solution* but not
  of a coarse *correction increment*.
- **Adjointness is exact and free, for the cell-to-cell pair.** For injection,
  `P[f,c] = 1` iff `f` is in block `c`, so `P^T` IS the block sum -- exactly
  adjoint on any mesh, with no normalisation, no weights and no geometry. The
  property `plan_body_force_multigrid.md` spent a whole branch engineering falls
  out of doing less. **Scope the claim honestly:** the operator the solver
  actually applies is `S . I`, injection composed with the cell->node scatter
  (2.1), and `(S . I)^T` is not the block sum. What is exact here is the
  cell-to-cell pair, which is the pair the previous branch was trying to make
  adjoint. Claiming more would repeat that branch's mistake of optimising a
  property that is not the one in the loop.
- **The block sum is also the physically right restriction.** A coarse cell's
  residual genuinely is the sum of its children's: the interior face fluxes
  cancel and only the coarse cell's own boundary survives.
- **The ill-conditioned hop disappears.** `_mg_hop_weights_node` records that on
  the LISA rotor it produced blend coefficients of **-25 at the casing**, applied
  to the coarse correction at full weight, which is why `MG_W_LO`/`MG_W_HI` exist
  as a backstop at all. Injection has no weights to be ill-conditioned.

### 1.2 What the correction is worth, in units of the fine term

Worth writing down before any of the evidence, because it is exact, it is the
same for both prolongations (both preserve constants), and it explains where the
divergence cliff sits.

At DC, `cres_l = r * b_l**3`, so the total coarse push relative to the fine term
`alpha*cfl*dt_vol*r` is, where `dtblk_l == dt_vol`,

```
G(0) / fine  =  fac_mgrid * sum_{l=1}^{L} b_l * expon_mgrid**-(l-1)
```

At `expon_mgrid = 2`, `n_levels = 3` that sum is `2*1 + 4*(1/2) + 8*(1/4) = 6`:
**the three levels contribute equally**, and the coarse push is `6*fac_mgrid`
fine terms. So `fac_mgrid = 0.3` applies 1.8 fine increments on top of the fine
increment and `0.4` applies 2.4. `mg_stability.md` measures this directly --
`G(0) = 2.354` against a predicted 2.34 at `fac_mgrid = 0.4` -- so the formula
is confirmed on the real kernel, and the cliff between 0.3 and 0.4 is a
total-gain cliff, not something peculiar to injection.

This is also the analytic target for test 4.3 and the thing risk 5 is really
about: with `expon_mgrid = 2` the level balance is flat, so a level-limited
result and a gain-limited one are distinguishable.

### 1.3 The evidence, and how far it goes

Screened in numpy, 250 steps, `cfl=3`, `n_levels=3`, `expon_mgrid=2`,
`sf_resid=1`, 300k-cell duct, decades of energy residual. "Master" is the
in-tree scheme: block-sum restriction, cascaded trilinear prolongation with a
node-targeted final hop.

| `fac_mgrid=0.2` | uniform | clustered |
|---|---|---|
| multigrid off | +1.31 | +1.31 |
| master (trilinear cascade, node-targeted final hop) | +1.41 | +1.51 |
| trilinear + adjoint cell-to-cell transfers | +1.37 | +1.48 |
| **piecewise-constant + block sum** | **+1.70** | **+1.65** |

`fac_mgrid` sweep, clustered duct, same settings -- injection against the
trilinear cascade:

| `fac_mgrid` | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|---|
| piecewise-constant | +1.44 | +1.54 | +1.58 | +1.65 | **+1.74** | DIVERGED |
| trilinear cascade | -- | +1.44 | -- | +1.48 | -- | +0.99 |

Monotone up to a cliff: the optimum sits at `fac_mgrid = 0.3`, immediately
before divergence at 0.4 -- which 1.2 places at 2.4 fine increments of coarse
push. That is +1.74 against master's best MEASURED +1.51, so about +0.23
decades -- but see caveat 4. The uniform-mesh sweep was stopped partway
(0.05 -> +1.42, 0.1 -> +1.50) once the clustered result was clear.

**Five caveats, all load-bearing.**

1. The screen ran in the *residual-correction* framing of the stagewise plan
   (branch `stagewise-mg-correction`, commit `7a66715`), not in master's
   in-stage path. The port must re-measure rather than assume the result
   carries.
2. **Piecewise-constant is not a like-for-like swap at fixed `fac_mgrid`.** Both
   prolongations preserve constants, so the DC gain of 1.2 is identical, but
   trilinear attenuates the mid-band and injection does not -- so injection
   delivers more effective gain per unit `fac_mgrid` and its optimum sits lower.
   It diverged at 0.4 where trilinear survived. Comparing at matched `fac_mgrid`
   is the wrong test; section 5 sweeps.
3. The duct is a case where master already works, so the best any alternative
   can score is a modest margin. It scored one, which is more than anything else
   has, but it is still a 250-step duct run and not the turbine.
4. **The comparison is not yet like for like.** Injection has six sweep points
   and master only two (0.2 and 0.4), so "+0.23 decades over master's best" is
   six samples against two and flatters injection by however much master's own
   optimum sits above +1.51. Closing master's sweep is the first thing 5.2 does,
   and the margin should be re-read afterwards.
5. **The row-to-row differences are unreplicated single runs.** No repeat and no
   window sensitivity was taken, so nothing here bounds the noise that risk 3 is
   about -- 5.1 fixes that before any of the rest is read. The "multigrid off"
   row scoring +1.31 on both meshes, when every other row separates them, is the
   kind of thing that needs a rerun rather than an explanation.

---

## 2. Code changes

### 2.1 The correction becomes a CELL quantity

Master applies the correction at the nodes: the final cascade hop targets fine
nodes directly, fused into `mg_prolong2x_fine_scatter`, precisely so the
correction is never averaged through a cell centre. That kernel's own header
explains why -- interpolating onto cell centres and then scattering evaluates the
interpolant at the mean of the eight surrounding centroids rather than at the
node, a first-order error in the clustering ratio.

**That objection does not survive the change to injection.** Within a block the
correction is constant, so cell-averaging it changes nothing at all; it differs
from the node-targeted form only at block faces, where the average of the two
adjoining blocks' corrections is the desired behaviour rather than an error. It
is a one-cell smoothing of the staircase, applied exactly where the staircase is.
The scatter is a partition of unity everywhere (interior 1/8 of 8 cells, faces
1/4 of 4, edges 1/2 of 2, corners 1 of 1), including at the domain boundary, so
it preserves the constant and does not dilute the correction anywhere.

Concretely: fine cell `i` sits in level-1 block `(i+1)/2`, so node `i` -- fed by
cells `i-1` and `i` -- straddles a block face exactly when `i` is odd. Half the
nodes in each direction take one block's correction unaltered, half take the
mean of two.

So the correction is added into the fine cell increment alongside the fine term
and both ride the existing `cell_to_node` scatter. The kernel's "the two halves
land at the nodes by different routes" becomes one route, and the node-targeted
machinery is not merely unused but unnecessary.

### 2.2 Phase 2: collapse in place, do not cascade

Do not prolong level by level onto the fine grid, and do not build a separate
accumulator either. `corr_all` already holds every level's scaled correction in
its own compact, disjoint slot, packed coarsest first, so the collapse runs
**in place inside `corr_all`**: coarsest first, each level's slot gains the
injected total from the slot above it.

```fortran
do lvl = 2, n_levels
    ! corr_all[lvl] += inject(corr_all[lvl-1]); the two slots are disjoint
    ! regions of one array, and every target element is written once.
    call mg_inject_acc(corr_all(offc(lvl-1)+1), dib(lvl-1), djb(lvl-1), dkb(lvl-1), &
                       corr_all(offc(lvl)+1),   dib(lvl),   djb(lvl),   dkb(lvl), np)
end do
```

where `mg_inject_acc` is `out(i,j,k,ip) += src((i+1)/2,(j+1)/2,(k+1)/2,ip)`.

`1 + 1/8 + 1/64 = 1.14` traversals of the level-1 grid, i.e. about 0.14
fine-cell-equivalents. Then the fine grid sees **one read** from the finest
coarse slot, `corr_all(offc(n_levels)+1)` at `(nc1i,nc1j,nc1k,np)`:

```fortran
corr = src((i+1)/2, (j+1)/2, (k+1)/2, ip)
```

from an array an eighth the size, whose `i` index advances every second fine
cell, so the value is reused across the inner loop and stays in cache.

**This needs no accumulator at all.** An earlier draft of this plan proposed two
level-1-sized buffers `tot`/`nxt` ping-ponging like `acc0`/`acc1`; running in
`corr_all` instead deletes `acc0` AND `acc1` (609,280 elements each at
273x65x57) rather than renaming one and keeping its partner, and removes two
names from the buffer list instead of zero.

### 2.3 The fine application

Injection makes the correction a cell quantity, so `mg_prolong2x_fine_scatter`
has nothing left to interpolate. Its replacement `mg_pwc_fine_scatter` keeps the
rolling two-plane structure -- which is what avoids a full-volume increment --
and drops `fill_cbuf` with it:

```fortran
rbuf(i,j,ip,cur) = scale*dt_vol(i,j,kc)*q(i,j,kc,ip) &
                 + src((i+1)/2, (j+1)/2, (kc+1)/2, ip)
```

then the same `emit_kint`/`emit_kbnd` node planes, term for term and in the same
summation order as `cell_to_node_generic`, minus the `+ cbuf(...)` on every
line. `aplane`, `bb` and `cbuf` go with it.

The alternative -- materialise the increment full-volume and call the existing
`cell_to_node_generic`, as `rk_plain` does -- is less code but **grows the
arena**: `mg_pwc` scratch plus a `(ni-1)(nj-1)(nk-1)*5` increment is 29.3 MB at
273x65x57 against the 25.02 MB the arena holds today. Rejected on that.

### 2.4 What this actually touches

The final hop is not in `mg_coarse_correction`; it is in
`mg_prolong2x_fine_scatter`, called from the four wrappers `scree_mg_irs`,
`scree_mg_noirs`, `rk_mg_irs`, `rk_mg_noirs`. So `mgrid_pwc` cannot be "which
Phase 2 runs" -- it selects a different fine application too, and the file's own
rule (*"configuration is resolved by which blocks are called, never by a runtime
`if`"*) forbids branching inside the scatter. The change is therefore:

- Factor Phase 1 out of `mg_coarse_correction` into `mg_restrict_levels`
  (arithmetic untouched, so the cascade path stays byte-identical).
  `mg_coarse_correction` becomes that call plus the cascade.
- New blocks: `mg_inject_acc`, `mg_collapse_pwc`, `mg_pwc_fine_scatter`.
- **Four new wrappers**, `scree_mgpwc_irs`, `scree_mgpwc_noirs`,
  `rk_mgpwc_irs`, `rk_mgpwc_noirs`, each a three-call straight line
  (`mg_restrict_levels` -> `mg_collapse_pwc` -> `mg_pwc_fine_scatter`), with the
  scree pair keeping their `scree_form_q` / `scree_roll` bookends.
- Python: `MG_PWC_NAMES` / `mg_pwc_shapes` (a strict subset of the cascade's
  twelve), and a second dispatch arm in each of `scree_step` and
  `advance_rk_stage_mg`. The pwc arm never touches `block.weight_mgrid`, so the
  geometry ladder is not built at all.

### 2.5 What becomes dead

Nothing here is deleted in the first pass -- the flag of 2.6 keeps both paths
alive until section 5 has run. Listed so the size of the prize is visible:

- Fortran: `mg_prolong2x_acc`, `mg_interp_i2x`, `mg_interp_i2x_node`,
  `mg_bracket2x`, `mg_bracket2x_node`, `mg_weight_offsets`, and all of
  `mg_prolong2x_fine_scatter`.
- Python: `Block.weight_mgrid`, `_mg_hop_weights`, `_mg_hop_weights_node`,
  `_mg_project`, `_mg_centroid_ladder`, `_mg_weight_lengths`,
  `_mg_index_bracket`, `_mg_index_bracket_node`, `MG_W_LO`/`MG_W_HI`. About
  14 bytes per fine cell of cached weight storage goes with them, and so does
  the geometry ladder that builds it.
- Scratch: `cbuf`, `aplane`, `bb`, `acc0`, `acc1` leave the buffer list; nothing
  is added.
- `tests/test_mg_weights.py` (564 lines) covers only the deleted weights.

**What that is worth, measured rather than estimated.** Element counts from
`mg_coarse_shapes`, float32, at the two shapes `_scratch_len`'s docstring
already quotes:

| phase | 273x65x57 master | 273x65x57 pwc | 49^3 master | 49^3 pwc |
|---|---|---|---|---|
| multigrid | 25.02 MB | **9.82 MB** | 2.89 MB | **1.11 MB** |
| update_residual | 20.96 | 20.96 | 2.45 | 2.45 |
| update_sources | 19.24 | 19.24 | 2.67 | 2.67 |
| scree/RK, no MG | 19.50 | 19.50 | 2.21 | 2.21 |
| **arena** | **25.02** | **20.96** | **2.89** | **2.67** |

So the multigrid phase falls by 61 per cent, but **`_scratch_len` falls by only
16 per cent** (7.6 per cent on the cube), because the multigrid phase stops
being the binding one: `update_residual` binds at 273x65x57 and
`update_sources` at 49^3. Two knock-ons that belong to step 6, not to this
paragraph's headline:

- `_scratch_len`'s docstring says "The MULTIGRID phase binds, at every shape
  tried". That becomes false and has to be rewritten.
- The same docstring justifies borrowing the nodal transport trio into the arena
  as taking "space in an arena the multigrid phase was already sizing". After
  the deletion it is `update_sources` -- the trio's own phase -- that sets the
  arena on a cube. The trade is still favourable (2.67 against 2.89) but the
  recorded reasoning no longer holds and should be restated.

### 2.6 Config

- `Solver.mgrid_pwc: bool = False` -- select injection instead of the cascade.
  Both integrators honour it.
- `run.py` `DEFAULTS` and `tools/run_duct_mg.py` gain the flag, so the
  section 5 sweeps are one command line.
- `fac_mgrid`, `n_levels`, `expon_mgrid`, `sf_resid` are reused unchanged, and
  `coef_l` is **not** recalibrated -- see 1.3 caveat 2. The gain that suits
  injection is found by sweeping, not by rescaling the formula.
- While the flag exists the arena stays sized for the cascade: `Block.scratch`
  is allocated once from `_scratch_len(shape)`, which cannot see `mgrid_pwc`.
  The pwc buffer list is a strict subset of the cascade's, so no sizing change
  is needed to keep both paths safe, and none of 2.5's saving is realised until
  step 6.

---

## 3. What "correct" means here

1. **Block-uniform increment.** The state increment from the coarse correction
   is identical for every fine cell of a coarse block, before the cell-to-node
   scatter; after it, identical on the interior nodes of a block and the mean of
   two blocks on the faces between them. Test 4.1 pins it.
2. **Exact adjointness of the cell-to-cell pair.** The restriction is `P^T` for
   the injection `P`, on any mesh, with no scaling -- and not of the applied
   operator, which includes the scatter (1.1). Test 4.2.
3. **Constant preservation.** Injecting a constant coarse field gives a constant
   fine field, exactly, on any mesh -- trivially true, and worth one assertion
   because it is what fixes the DC gain equal to the cascade's.
4. **Consistency.** The correction vanishes with the residual, so the steady
   state is unchanged.
5. **Off is off.** `mgrid_pwc=False` reproduces master bit for bit, and never
   carves the pwc buffers.

---

## 4. Tests

New file `tests/test_mg_pwc.py`, except where noted.

### 4.1 `test_the_increment_is_uniform_over_a_coarse_block`

The property the whole plan rests on. Drive one block with a residual and a
`dt_vol` that varies strongly *between adjacent cells* (a smooth ramp will not
do -- any prolongation reproduces a linear field, so a ramp cannot tell injection
from interpolation; this exact mistake was made once already, in
`tests/test_mg_correction.py` on branch `stagewise-mg-correction`).

**The pre-scatter cell increment is never materialised** -- the correction is
fused into the rolling-plane scatter -- and the post-scatter increment is *not*
block-constant, because block-face nodes are averaged, which is 2.1's whole
point. So the assertion is made in its post-scatter form: recover the coarse
contribution by differencing `fac_mgrid > 0` against `fac_mgrid = 0`, then
assert it is constant over the nodes interior to each level-1 block (even node
index in every direction) to float32 rounding, and exactly the two-block mean at
the odd ones. The same measurement on the trilinear path satisfies neither.

### 4.2 `test_restriction_is_the_transpose_of_injection`

Build the dense `P` (injection) and `R` (block sum) by applying the kernels to
unit vectors and assert `R == P.T` exactly -- no scaling, on a sheared mesh
clustered to a wall. This pins the cell-to-cell pair of 3.2 and nothing more; it
is close to true by construction, which is the point -- it is what the previous
two branches spent their effort buying. Assert alongside it that the *applied*
transfer is `S . I` and therefore NOT self-adjoint, so the scope of the claim is
in the test file rather than only in this document.

### 4.3 `test_a_constant_residual_gives_the_calibrated_gain`

The DC gain of 1.2. Assert the total correction against
`r * sum_l coef_l * dtblk_l * b_l**3` -- note the residual value `r`, which an
earlier draft of this section dropped -- and against the cascade.

**This must hold `dt_vol` constant.** With a constant residual the level
correction is `coef_l * dtblk_l * cres_l`, and `dtblk_l` varies block to block
on a stretched mesh; trilinear smears that variation across blocks and injection
does not, so the two paths genuinely disagree there. The two-path equality is a
statement about the prolongations, so it is tested where nothing else varies:
uniform `dt_vol`, where `sum_l b_l * expon_mgrid**-(l-1)` is checkable in closed
form as well.

### 4.4 `test_off_is_byte_identical`

`mgrid_pwc=False` versus master's default on a short duct march: `conserved_nd`,
`residual_nd` and the convergence history bit for bit. Covers the Phase-1
factoring of 2.4 as much as the flag.

### 4.5 `test_the_correction_reaches_conserved`

End to end: a nonzero `fac_mgrid` under `mgrid_pwc` moves `conserved_nd` by more
than rounding, on both integrators.

### 4.6 Golden and arena

Extend `tests/test_set_F_body_golden.py`'s pattern with a committed golden for
the assembled coarse increment under `mgrid_pwc` (fixed grid, fixed residual),
since 4.1-4.3 pin properties rather than numbers. `tests/test_scratch_arena.py`
gains a case asserting the pwc carve fits the arena; its multigrid-phase buffer
list and the sizing re-check of 2.5 belong to step 6, when the cascade goes.

---

## 5. Validation

Unit tests do not settle this. Run these in order and stop at the first failure.

### 5.1 Noise floor, first

Everything below is read in tenths of a decade, and 1.3 caveat 5 is that nothing
bounds the noise. Before any comparison: repeat one setting (say `mgrid_pwc` at
`fac_mgrid=0.2`, clustered) and record decades at 200 and at 250 steps. If the
spread approaches 0.1 decades, the whole table in 1.3 is one sample per cell and
5.2's gate needs more than one run per point.

### 5.2 The gain sweep, best against best

`tools/run_duct_mg.py`, both meshes, `cfl=3`, `fac_mgrid` in
{0.05, 0.1, 0.15, 0.2, 0.3, 0.4}, `mgrid_pwc` on and off. **Compare the best of
each curve, not matched `fac_mgrid`** (1.3, caveat 2). Add `n_levels` in {2, 3}
on the clustered mesh: the staircase amplitude scales with `b_l`, 8 cells wide
at level 3, so dropping a level is the cheapest direct probe of risk 1 and it
costs one extra column.

**Gate: injection's best must beat the cascade's best on both meshes, by more
than 5.1's noise floor.** The screen says +1.65/+1.70 against +1.51/+1.41, so
the expectation is roughly +0.15 to +0.3 decades. If it merely ties, this is a
simplification with no convergence argument -- still arguably worth doing for
the deletion in 2.5, but that is a different decision and should be taken
deliberately.

**MEASURED, one matched gain so far.** `fac_mgrid = 0.4`, `cfl = 3`, 250 steps,
`tools/run_duct_mg.py`'s fixed 209x73x65 mesh (992k nodes), decades of energy
residual:

| mesh | cascade | injection | gap |
|---|---|---|---|
| clustered | **+1.73** | +1.66 | -0.07 |
| uniform | +1.89 | **+1.92** | +0.02 |

**The screen did not reproduce.** It predicted injection DIVERGING at 0.4
clustered against the cascade's +0.99; instead nothing diverged, the cascade
reached +1.73, and injection came in slightly behind it. Caveat 1 is realised:
the residual-correction framing does not carry into the in-stage path.

The gaps are systematic rather than scatter -- each pair starts from an
identical peak residual and the gap accumulates monotonically (clustered -0.01
at step 20, -0.03 at 60, -0.05 at 160, -0.07 at 245) -- so 5.1's noise floor is
not what limits this comparison. Residual is still falling monotonically at 250
steps in every arm, so this measures the transient, not an asymptote.

Consequence for caveat 2: the cliff at 0.4 was the whole reason to believe
injection delivers more effective gain per unit `fac_mgrid`, and it is absent.
Matched-gain is therefore a more defensible reading than this plan first gave
it credit for. The remaining gains are still worth sweeping, but the
expectation should now be that the two schemes are close on the duct at any
gain -- which is what 2.1 predicts anyway, since both preserve constants and so
share the DC gain of 1.2, differing only in the band above it.

### 5.3 The composite transfer, measured

`mg_stability.md` already has the harness: drive the kernel with a single
Fourier mode along `j` (uniform in `i` and `k`, `dt_vol = 1`), difference
`fac_mgrid = 0.4` against 0, and read off the true composite
restrict -> smooth -> prolong transfer and the mode leakage. It is recorded there
for the cascade (0.637 at 8 cells/wave rising to 2.354 at 72; leakage under 3
per cent except 23 per cent at 6 cells/wave). Run it on injection.

This is the direct measurement of both halves of the argument that 1.3 caveat 2
and risk 1 currently make in words -- how much more mid-band gain injection
delivers, and how much energy the staircase puts into the alias band -- and it
predicts 5.4 for a fraction of the cost.

### 5.4 CFL headroom

Repeat 5.2's best settings at `cfl` in {4, 5, 6}. Master's no-multigrid baseline
diverges between 5 and 6 on the clustered duct and above 6 on the uniform one,
and both existing multigrid schemes lose stability *earlier* than that -- between
4 and 5 clustered. If injection pushes that back it is worth more than the
convergence margin; the multigrid CFL penalty is what `mg_stability.md` is about.

### 5.5 Cost

`us/node/step` against no-multigrid and against the cascade, on the clustered
duct. Master's cascade measured +14% over no multigrid. Phase 1 is untouched and
is the half that reads the fine grid, so the prolongation saving is bounded;
**estimate +5 to 7%, and measure rather than quote that.**

### 5.6 Full multigrid startup

`Solver.run_fmg` still converges with `mgrid_pwc` on -- the FMG schedule and the
in-step correction are orthogonal, but the coarse levels' histories are worth a
look. Check first that every level `run_fmg` builds satisfies `_validate_mg`'s
divisibility rule at the configured `n_levels`; injection has no clamp to fall
back on (section 1).

### 5.7 The turbine

`turbine/run_final`, 10000 steps, three arms: `fac_mgrid=0` (have it, converged
flat), `fac_mgrid=0.2` cascade (have it: diverged at step 4010), and the best
`fac_mgrid` from 5.2 under `mgrid_pwc`.

**Pre-check, before spending 10000 steps.** The quantity that decides this is
`dtblk_l / dt_vol_c` at the wall: a block's coarse timestep is the
volume-weighted harmonic mean over the block, which on a clustered mesh is set
by the large cells, and injection hands the smallest cell in the block that
block's full correction with nothing blending it toward the neighbour. The
kernel's own comment records up to 2.0x on the clustered duct against 1.01x
uniform, and `mg_stability.md` has a whole section on the tail. Compute
`max_c sum_l coef_l * dtblk_l * b_l**3 / (alpha*cfl*dt_vol_c)` on the turbine
mesh -- static geometry, no march needed, minutes not hours -- and read it
against the 6*`fac_mgrid` that 1.2 says a uniform mesh would give. That number
is the single best predictor available of whether this arm survives.

Success = reaching 10000 steps without diverging and converging deeper than the
multigrid-off arm. **If it diverges**, walk `fac_mgrid` down (0.2, 0.1, 0.05)
before concluding anything: 1.3 shows the cliff is sharp and the duct optimum is
not obliged to transfer. A converged turbine at a lower gain than the duct's
optimum is still the result this line of work is for; only a divergence at every
gain is a failure.

This is the case the whole line of work exists for. Every duct measurement is on
a case master already handles, where the ceiling is a small margin; the turbine
is the case master *fails*.

### 5.7a  PRELIMINARY RESULT -- the stability difference is real

Injection at `cfl=4`, `fac_mgrid=0.2`, `n_levels=3`, every other setting
identical to the cascade arm. **The run is mid-flight; these are the first
~3000 of 10000 steps and nothing here is a finished result.** Energy residual:

| step | multigrid off | cascade | injection |
|---|---|---|---|
| 1000 | 6.016e-08 | 5.160e-08 | 5.140e-08 |
| 1410 | 5.346e-08 | **4.947e-08** (its minimum) | 4.910e-08 |
| 2000 | 5.200e-08 | 5.055e-08 | 4.860e-08 |
| 2500 | 5.083e-08 | 5.757e-08 | 4.850e-08 |
| 2750 | 5.053e-08 | **6.770e-08** (+37%) | 4.850e-08 |
| 10000 | 5.024e-08 | DIVERGED at ~4010 | -- |

Three things, in decreasing order of confidence.

1. **Injection passed the cascade's turning point still falling.** Up to step
   1410 the two agreed to within 0.5% -- expected, since they share the DC gain
   -- and they separate only once the cascade turns. At the same distance past
   its own minimum the cascade was already climbing monotonically
   (4.947, 4.948, 4.949, 4.948, 4.949, 4.950, ...); injection was not. This is
   the first measurement in the whole line of work that discriminates the two
   prolongations, and it discriminates them on the axis that matters.
2. **Injection has plateaued, not turned.** Minimum 4.850e-08 at step ~2140 and
   flat to three significant figures for ~850 steps since. Flat is not the same
   as stable: the cascade took ~2600 steps to go from its turn to the blow-up,
   so a longer fuse is not yet excluded. The full-precision history only exists
   once the run writes its `.cnv`.
3. **The convergence margin over doing nothing is small.** The multigrid-off arm
   ends at 5.024e-08 and is still falling. If injection holds its plateau the
   final margin is about 3.3%, or **0.014 decades** -- real, but not an
   acceleration. 5.7's success condition has two halves, and injection looks set
   to clear the stability half decisively while barely clearing the other. That
   is a different and more modest claim than the screen's +0.23 decades, and it
   is the one the evidence supports.

Two more facts belonging here:

- **`cfl=5`, `fac_mgrid=0.4` under injection diverged at step 6**, NaN density
  across essentially all of row0. 1.2 puts that gain at 2.4 fine increments of
  coarse push, so an effective step near `cfl*3.4`; blowing up is unsurprising,
  blowing up at step 6 is faster than gain alone comfortably explains. **No
  cascade control was run at those settings, so this number is currently
  uninterpretable** and should not be cited until it is.
- **The kernel in use was verified, not assumed.** Instrumenting the Fortran
  entry points and driving `run.py`'s own path gives 24 calls (2 steps x 4
  stages x 3 blocks) to `rk_mg_irs` with the flag off and 24 to `rk_mgpwc_irs`
  with it on, zero to the other family either way, and states differing across
  9.49M of 9.70M nodes. Worth recording because the two arms agree to 0.5% for
  the first 1400 steps, which is also exactly what a silent fall-through would
  look like.

The mesh clears `_validate_mg` at `n_levels=3`: all three blocks (145x89x65,
97x89x41, 129x89x65) divide by 8 in every direction, so injection's unclamped
`(i+1)/2` is safe here.

---

## 6. Risks

1. **The block-face staircase.** Injection puts a jump in the correction at
   every coarse block face -- high-wavenumber content added to the state, which
   the artificial dissipation and the fine IRS then have to absorb. The
   cell-to-node scatter softens it by one cell (2.1) and the duct screen was
   fine, but the duct is smooth. 5.3 measures the alias content directly; watch
   the near-boundary and near-seam residual in 5.7. Multi-block seams are the
   sharper version of this: the correction is built per block, so at a seam two
   independent staircases meet with nothing averaging across.
2. **The `dtblk`/`dt_vol` tail.** The known stability mechanism (`mg_stability.md`:
   the arithmetic -> harmonic fix, then the min-clamp experiment that falsified
   the tail as the *explanation* of the clustered CFL penalty). Falsified as the
   explanation is not the same as harmless here: the cascade blends a wall cell's
   correction toward its neighbour block and injection does not, so whatever the
   tail is worth, injection is the scheme most exposed to it. 5.7's pre-check is
   the cheap way to find out before the turbine does.
3. **The evidence is from another framing.** 1.3, caveat 1. If 5.2 fails to
   reproduce the screen in the in-stage path, believe 5.2 and stop.
4. **The margin may be noise.** +0.15 decades over 250 steps on one case is not
   a large signal, and 1.3 caveat 5 is that no repeat was taken. 5.1 now
   measures the floor before 5.2's sweep is read against it.
5. **`expon_mgrid`.** The per-level decay was tuned against the cascade's
   attenuation. Injection has none, so the level balance may want revisiting --
   but 1.2 shows `expon_mgrid = 2` weights the three levels *equally*, so
   "level-limited" and "gain-limited" are distinguishable: sweep `fac_mgrid` and
   `n_levels` first (5.2) and only touch `expon_mgrid` if dropping a level helps
   where lowering the gain does not.
6. **Deleting `test_mg_weights.py`.** It is 564 lines pinning real properties of
   a real subsystem, and deleting it is right only if the subsystem goes with it.
   Do not delete it while the cascade is still selectable.

---

## 7. Rollback and sequencing

`mgrid_pwc` defaults False and the cascade is untouched, so rollback is deleting
the flag, the four pwc wrappers and the three new blocks.

1. **Kernel** (done): the Phase-1 factoring, the in-place collapse of 2.2, the
   fine application of 2.3, and the four wrappers and dispatch of 2.4, behind
   `mgrid_pwc`. Nothing is removed, so the arena does not move (2.6).
2. **Tests** (done): 4.1 to 4.5, then the golden and arena of 4.6.
3. **5.1 then 5.2, the gate.** Stop here if injection does not beat the cascade
   at its own best gain by more than the noise floor.
4. 5.3, then 5.4 and 5.5.
5. **5.7, the turbine**, its pre-check first. This is the answer the whole
   exercise is for and it should not be deferred behind the deletion.
6. Only then, if 5.2 and 5.7 both hold: delete everything in 2.5, drop the flag,
   make injection the only path, let the arena fall, and rewrite the two
   `_scratch_len` docstring claims that 2.5 lists as casualties.
