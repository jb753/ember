# Plan: piecewise-constant multigrid, and nothing else

> **Status: not started.** Design and test plan only. Nothing here is built.
>
> The whole change: **replace the cascaded trilinear prolongation with plain
> injection.** Every fine cell under a coarse block takes that block's
> correction, unaltered. Restriction stays the block sum. Phase 1 of
> `mg_coarse_correction` -- the hierarchical restriction and the coarse
> timestep -- is untouched.
>
> This deletes far more than it adds, and the evidence says it converges better
> than what it replaces. It is the opposite direction from
> `plan_body_force_multigrid.md` and `plan_stagewise_mg_correction.md`, which
> both made the transfers *more* elaborate (adjoint, geometry-aware,
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

### 1.1 Why this should be better and not merely simpler

- **The correction is what a coarse solve would give.** A real coarse grid
  produces one number per coarse cell. (PC1) applies exactly that, uniform over
  the block. The trilinear cascade instead smears each coarse value over its
  neighbours, which is a defensible interpolation of a coarse *solution* but not
  of a coarse *correction increment*.
- **Adjointness is exact and free.** For injection, `P[f,c] = 1` iff `f` is in
  block `c`, so `P^T` IS the block sum -- exactly adjoint on any mesh, with no
  normalisation, no weights and no geometry. The property
  `plan_body_force_multigrid.md` spent a whole branch engineering falls out of
  doing less.
- **The block sum is also the physically right restriction.** A coarse cell's
  residual genuinely is the sum of its children's: the interior face fluxes
  cancel and only the coarse cell's own boundary survives.
- **The ill-conditioned hop disappears.** `_mg_hop_weights_node` records that on
  the LISA rotor it produced blend coefficients of **-25 at the casing**, applied
  to the coarse correction at full weight, which is why `MG_W_LO`/`MG_W_HI` exist
  as a backstop at all. Injection has no weights to be ill-conditioned.

### 1.2 The evidence, and how far it goes

Screened in numpy, 250 steps, `cfl=3`, `n_levels=3`, `expon_mgrid=2`,
`sf_resid=1`, 300k-cell duct, decades of energy residual:

| `fac_mgrid=0.2` | uniform | clustered |
|---|---|---|
| multigrid off | +1.31 | +1.31 |
| direct injection (master) | +1.41 | +1.51 |
| trilinear + adjoint transfers | +1.37 | +1.48 |
| **piecewise-constant + block sum** | **+1.70** | **+1.65** |

`fac_mgrid` sweep, clustered duct, same settings -- injection against the
trilinear cascade of `plan_stagewise_mg_correction.md`:

| `fac_mgrid` | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|---|
| piecewise-constant | +1.44 | +1.54 | +1.58 | +1.65 | **+1.74** | DIVERGED |
| trilinear cascade | -- | +1.44 | -- | +1.48 | -- | +0.99 |

Monotone up to a cliff: the optimum sits at `fac_mgrid = 0.3`, immediately
before divergence at 0.4. That is +1.74 against master's best MEASURED +1.51,
so about +0.23 decades -- but see caveat 4. The uniform-mesh sweep was stopped
partway (0.05 -> +1.42, 0.1 -> +1.50) once the clustered result was clear.

**Three caveats, all load-bearing.**

1. The screen ran in the *residual-correction* framing of
   `plan_stagewise_mg_correction.md`, not in master's in-stage path. The port
   must re-measure rather than assume the result carries.
2. **Piecewise-constant is not a like-for-like swap at fixed `fac_mgrid`.** Both
   prolongations preserve constants, so the DC gain is identical, but trilinear
   attenuates the mid-band and injection does not -- so injection delivers more
   effective gain per unit `fac_mgrid` and its optimum sits lower. It diverged at
   0.4 where trilinear survived. Comparing at matched `fac_mgrid` is the wrong
   test; section 6 sweeps.
3. The duct is a case where master already works, so the best any alternative
   can score is a modest margin. It scored one, which is more than anything else
   has, but it is still a 250-step duct run and not the turbine.
4. **The comparison is not yet like for like.** Injection has six sweep points
   and master's direct injection only two (0.2 and 0.4), so "+0.23 decades over
   master's best" is six samples against two and flatters injection by however
   much master's own optimum sits above +1.51. Closing master's sweep is the
   first thing section 5.1 does, and the margin should be re-read afterwards.

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

So the correction is added into the fine cell increment alongside the fine term
and both ride the existing `cell_to_node` scatter. The kernel's "the two halves
land at the nodes by different routes" becomes one route, and the node-targeted
machinery is not merely unused but unnecessary.

### 2.2 Phase 2 of `mg_coarse_correction`: collapse, do not cascade

Do not prolong level by level onto the fine grid. Accumulate every level onto
the level-1 grid first, coarsest first, doubling each pass:

```fortran
! seed with the coarsest correction
call mg_copy(corr_all(offc(1)+1), tot, dib(1)*djb(1)*dkb(1)*np)
do lvl = 2, n_levels
    do ip = 1, np
      do kb = 1, dkb(lvl)
        do jb = 1, djb(lvl)
          do ib = 1, dib(lvl)
            nxt(ib,jb,kb,ip) = corr_lvl(ib,jb,kb,ip) &
                             + tot((ib+1)/2, (jb+1)/2, (kb+1)/2, ip)
```

`1 + 1/8 + 1/64 = 1.14` traversals of the level-1 grid, i.e. about 0.14
fine-cell-equivalents. Then the fine grid sees **one read**:

```fortran
corr = tot((i+1)/2, (j+1)/2, (k+1)/2, ip)
```

from an array an eighth the size, whose `i` index advances every second fine
cell, so the value is reused across the inner loop and stays in cache. Two
buffers of level-1 size (`tot`, `nxt`) replace the whole cascade; alternate
between them by hop parity exactly as `acc0`/`acc1` do today, or ping-pong in
place.

### 2.3 What becomes dead

Nothing here is deleted in the first pass -- the flag of 2.4 keeps both paths
alive until section 6 has run. Listed so the size of the prize is visible:

- Fortran: `mg_prolong2x_acc`, `mg_interp_i2x`, `mg_interp_i2x_node`,
  `mg_bracket2x`, `mg_bracket2x_node`, `mg_weight_offsets`, and the node half of
  `mg_prolong2x_fine_scatter`.
- Python: `Block.weight_mgrid`, `_mg_hop_weights`, `_mg_hop_weights_node`,
  `_mg_project`, `_mg_centroid_ladder`, `_mg_weight_lengths`,
  `_mg_index_bracket`, `_mg_index_bracket_node`, `MG_W_LO`/`MG_W_HI`. About
  14 bytes per fine cell of cached weight storage goes with them, and so does
  the geometry ladder that builds it.
- Scratch: `cbuf`, `aplane`, `bb`, `acc1` leave `MG_COARSE_NAMES`; `acc0`
  becomes `tot` and gains a partner `nxt` at the same size. `bb` alone is
  `(ni,nj,nc1k,5)`, about 2.5M elements at 273x65x57, so **the multigrid arena
  should fall by roughly half**, and with it `_scratch_len`, since the multigrid
  phase is what binds it.
- `tests/test_mg_weights.py` (564 lines) covers only the deleted weights.

### 2.4 Config

- `Solver.mgrid_pwc: bool = False` -- select injection instead of the cascade.
  Both integrators honour it; it changes only which Phase 2 runs.
- `run.py` `DEFAULTS` and `tools/run_duct_mg.py` gain the flag, so the
  section 6 sweeps are one command line.
- `fac_mgrid`, `n_levels`, `expon_mgrid`, `sf_resid` are reused unchanged, and
  `coef_l` is **not** recalibrated -- see 1.2 caveat 2. The gain that suits
  injection is found by sweeping, not by rescaling the formula.

---

## 3. What "correct" means here

1. **Block-uniform increment.** The state increment from the coarse correction
   is identical for every fine cell of a coarse block, before the cell-to-node
   scatter. This is the defining property and test 4.1 pins it.
2. **Exact adjointness, for free.** The restriction is `P^T` for the injection
   `P`, on any mesh, with no scaling. Test 4.2.
3. **Constant preservation.** Injecting a constant coarse field gives a constant
   fine field, exactly, on any mesh -- trivially true, and worth one assertion
   because it is what fixes the DC gain equal to the cascade's.
4. **Consistency.** The correction vanishes with the residual, so the steady
   state is unchanged.
5. **Off is off.** `mgrid_pwc=False` reproduces master bit for bit, and never
   builds `tot`/`nxt`.

---

## 4. Tests

New file `tests/test_mg_pwc.py`, except where noted.

### 4.1 `test_the_increment_is_uniform_over_a_coarse_block`

The property the whole plan rests on. Drive one block with a residual and a
`dt_vol` that varies strongly *between adjacent cells* (a smooth ramp will not
do -- any prolongation reproduces a linear field, so a ramp cannot tell injection
from interpolation; this exact mistake was made once already in
`test_mg_correction.py`). Recover the coarse contribution to the cell increment
and assert it is constant within each level-1 block to float32 rounding, and that
the same measurement on the trilinear path is not.

### 4.2 `test_restriction_is_the_transpose_of_injection`

Build the dense `P` (injection) and `R` (block sum) by applying the kernels to
unit vectors and assert `R == P.T` exactly -- no scaling, on a sheared mesh
clustered to a wall. One assertion, and it replaces the whole adjointness
apparatus of the previous two branches.

### 4.3 `test_a_constant_residual_gives_the_calibrated_gain`

A constant residual must produce the same total correction the cascade does, so
the two schemes agree at DC and any difference between them is genuinely in the
band above it. Assert both paths against `sum_l coef_l * dtblk_l * b_l**3` and
against each other.

### 4.4 `test_off_is_byte_identical`

`mgrid_pwc=False` versus master's default on a short duct march: `conserved_nd`,
`residual_nd` and the convergence history bit for bit.

### 4.5 `test_the_correction_reaches_conserved`

End to end: a nonzero `fac_mgrid` under `mgrid_pwc` moves `conserved_nd` by more
than rounding, on both integrators.

### 4.6 Golden and arena

Extend `tests/test_set_F_body_golden.py`'s pattern with a committed golden for
the assembled coarse increment under `mgrid_pwc` (fixed grid, fixed residual),
since 4.1-4.3 pin properties rather than numbers. Update
`tests/test_scratch_arena.py`'s multigrid phase to the new buffer list and
re-check the sizing, which should fall.

---

## 5. Validation

Unit tests do not settle this. Run these in order and stop at the first failure.

### 5.1 The gain sweep, best against best

`tools/run_duct_mg.py`, both meshes, `cfl=3`, `fac_mgrid` in
{0.05, 0.1, 0.15, 0.2, 0.3, 0.4}, `mgrid_pwc` on and off. **Compare the best of
each curve, not matched `fac_mgrid`** (1.2, caveat 2).

**Gate: injection's best must beat the cascade's best on both meshes.** The
screen says +1.65/+1.70 against +1.51/+1.41, so the expectation is roughly
+0.15 to +0.3 decades. If it merely ties, this is a simplification with no
convergence argument -- still arguably worth doing for the deletion in 2.3, but
that is a different decision and should be taken deliberately.

### 5.2 CFL headroom

Repeat 5.1's best settings at `cfl` in {4, 5, 6}. Master's no-multigrid baseline
diverges between 5 and 6 on the clustered duct and above 6 on the uniform one,
and both existing multigrid schemes lose stability *earlier* than that -- between
4 and 5 clustered. If injection pushes that back it is worth more than the
convergence margin; the multigrid CFL penalty is what `mg_stability.md` is about.

### 5.3 Cost

`us/node/step` against no-multigrid and against the cascade, on the clustered
duct. Master's cascade measured +14% over no multigrid. Phase 1 is untouched and
is the half that reads the fine grid, so the prolongation saving is bounded;
**estimate +5 to 7%, and measure rather than quote that.**

### 5.4 Full multigrid startup

`Solver.run_fmg` still converges with `mgrid_pwc` on -- the FMG schedule and the
in-step correction are orthogonal, but the coarse levels' histories are worth a
look.

### 5.5 The turbine

`turbine/run_final`, 10000 steps, three arms: `fac_mgrid=0` (have it, converged
flat), `fac_mgrid=0.2` cascade (have it: diverged at step 4010), and the best
`fac_mgrid` from 5.1 under `mgrid_pwc`.

This is the case the whole line of work exists for, and **no scheme has yet been
run on it.** Every duct measurement to date is on a case master already handles,
where the ceiling is a small margin; the turbine is the case master *fails*.
Success = reaching 10000 steps without diverging and converging deeper than the
multigrid-off arm.

---

## 6. Risks

1. **The block-face staircase.** Injection puts a jump in the correction at
   every coarse block face -- high-wavenumber content added to the state, which
   the artificial dissipation and the fine IRS then have to absorb. The
   cell-to-node scatter softens it by one cell (2.1) and the duct screen was
   fine, but the duct is smooth. Watch the near-boundary and near-seam residual
   in 5.5.
2. **The evidence is from another framing.** 1.2, caveat 1. If 5.1 fails to
   reproduce the screen in the in-stage path, believe 5.1 and stop.
3. **The margin may be noise.** +0.15 decades over 250 steps on one case is not
   a large signal. 5.1's sweep across six gains and two meshes is what turns it
   into one; a single matched-gain comparison would not.
4. **Deleting `test_mg_weights.py`.** It is 564 lines pinning real properties of
   a real subsystem, and deleting it is right only if the subsystem goes with it.
   Do not delete it while the cascade is still selectable.
5. **`expon_mgrid`.** The per-level decay was tuned against the cascade's
   attenuation. Injection has none, so the level balance may want revisiting --
   but sweep `fac_mgrid` first and only touch `expon_mgrid` if the sweep looks
   level-limited rather than gain-limited.

---

## 7. Rollback and sequencing

`mgrid_pwc` defaults False and the cascade is untouched, so rollback is deleting
the flag, Phase 2's injection branch, and `tot`/`nxt`.

1. Kernel: the collapse of 2.2 and the cell-quantity correction of 2.1, behind
   `mgrid_pwc`. Phase 1 untouched. Scratch gains `tot`/`nxt`; nothing is removed
   yet, so the arena does not move.
2. Tests 4.1 to 4.5, then the golden and arena of 4.6.
3. **Validation 5.1, the gate.** Stop here if injection does not beat the
   cascade at its own best gain.
4. 5.2 and 5.3.
5. **5.5, the turbine.** This is the answer the whole exercise is for and it
   should not be deferred behind the deletion.
6. Only then, if 5.1 and 5.5 both hold: delete everything in 2.3, drop the flag,
   make injection the only path, and let the arena and `_scratch_len` fall.
