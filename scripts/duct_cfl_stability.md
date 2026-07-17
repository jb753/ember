# Square-duct empirical CFL stability limits

Empirical maximum-stable-CFL measurements for the three marching schemes
(scree, RK4, RK4+IRS), obtained by bisecting the CFL number with
`scripts/duct_cfl_sweep.sh` (which drives `scripts/run_duct.py` and keys purely on
its exit code). The measurements bracket the discovery that the old
sum-of-directional-radii timestep normalisation inflated the CFL number to ~2.3x
the physical Courant number, and validate the change to a max-of-directional-radii
normalisation (commit `b4e5202`).

## Case and run settings (common to every measurement)

Built by `ember.cases.build_duct_grid`, run via `scripts/run_duct.py`.

### Geometry / grid
| quantity | value |
|----------|-------|
| `ncell` (target) | 250 000 |
| grid `ni x nj x nk` | 65 x 65 x 57 (240 825 nodes) |
| `side` | 0.1 m |
| `r_mid_ratio` -> `r_mid` | 5.0 -> 0.5 m |
| `length_ratio` -> length | 3.0 -> 0.3 m |
| passages `Nb` = round(2*pi*r_mid/side) | 31 |
| mesh | **uniform** (`cluster=False`; `--cluster` not passed) |
| dx, dr, arc (uniform spacing) | 0.004688, 0.001563, 0.001810 m |

### Fluid / mean flow
| quantity | value |
|----------|-------|
| fluid | `PerfectFluid(cp=1005, gamma=1.4, mu=1e-3, Pr=0.72, T_dtm=400)` |
| `Ma_bulk` | 0.3 |
| `Po`, `To` | 1e5 Pa, 300 K |
| sound speed a (approx) | 344 m/s |
| axial velocity Vx (approx) | 104 m/s |
| viscous | yes (`--inviscid` NOT passed) |

### Initial-condition perturbations (defaults)
`perturb_vx = 0.01`, `perturb_seed = 0`, `ho_frac = 0.01`, `s_frac = 0.01`,
`vx_ramp = 0.01`.

### Solver settings held fixed across the sweep
| setting | value |
|---------|-------|
| `n_step` | 1000 |
| `fac_mgrid` | 0.0 (multigrid correction off; isolates the plain scheme) |
| `n_levels` | 2 (inert while `fac_mgrid = 0`) |
| bisection `tol` | 0.05 |

### The three schemes (differ only in these)
| label | `n_stage` | `sf_resid` | `n_step_source` |
|-------|-----------|------------|-----------------|
| scree | 0 | 0.0 | 5 |
| RK4 | 4 | 0.0 | 1 |
| RK4+IRS | 4 | 1.0 | 1 |

### Exact `run_duct.py` invocation (per bisection sample)
```
uv run scripts/run_duct.py \
    --n-step 1000 --fac-mgrid 0.0 --n-levels 2 \
    --n-stage <N> --sf-resid <S> --n-step-source <K> \
    --cfl <C> --ncell 250000
```
Divergence is detected inside `run_duct.py` (NaN in the final conserved state via
`Grid.check_nan`); it exits 1 on divergence, 0 otherwise, and the shell bisects on
that exit code.

## Results

| scheme | 1D theory | old **sum** norm | new **max** norm | max / theory |
|--------|-----------|------------------|------------------|--------------|
| scree | 1/sqrt(3) = 0.577 | 1.31 | 0.562 | 0.97x |
| RK4 | 2*sqrt(2) = 2.828 | 6.44 | 2.831 | 1.00x |
| RK4+IRS | 2*sqrt(10) ~= 6.32 * | 13.35 | 6.207 | 0.98x |

\* IRS with smoothing factor beta = `sf_resid` = 1 extends the explicit limit by
sqrt(1 + 4*beta) = sqrt(5), so 2*sqrt(2) * sqrt(5) = 2*sqrt(10) ~= 6.32.

Under the **sum** normalisation every scheme sat ~2.3x above its 1D theory
(1.31/0.577 = 2.27, 6.44/2.828 = 2.28). Under the **max** normalisation all three
land within ~3% of the textbook Courant number.

### CFL brackets used
| run | scheme(s) | `--cfl-lo` | `--cfl-hi` |
|-----|-----------|-----------|-----------|
| sum norm, full sweep | scree, RK4, RK4+IRS | 0.2 | 10 |
| sum norm, rerun | RK4+IRS (`--case`) | 0.2 | 20 |
| max norm | scree (`--case`) | 0.3 | 1.0 |
| max norm | RK4 (`--case`) | 2.0 | 3.4 |
| max norm | RK4+IRS (`--case`) | 3.5 | 9.0 |

Under the sum norm RK4+IRS was still stable at `--cfl-hi 10`, so it was re-run to
20 to bracket the crossing (landed 13.35).

## Multigrid coarse-grid correction sweep (RK4+IRS)

Separate study: how the coarse-grid correction fraction `fac_mgrid` affects the
RK4+IRS stability limit, measured with `scripts/duct_cfl_descend.sh` (descend from
a high CFL by `--dcfl` to bracket the limit, then bisect). Distinct run settings
from the normalisation study above:

| setting | value |
|---------|-------|
| scheme | RK4+IRS (`n_stage=4`, `sf_resid=1.0`) |
| `ncell` (target) | 1 000 000 (grid 273 x 65 x 57, 1 011 465 nodes) |
| `n_step` | 500 |
| `n_levels` | 2 |
| bisection `tol` | 0.05 |
| descend `cfl_start`, `dcfl` | 12.0, 1.0 |

### Results (partial)

| `fac_mgrid` | max stable CFL | unstable at |
|-------------|----------------|-------------|
| 0.0 | 6.34 | 6.375 |
| 0.2 | 5.31 | 5.344 |
| 0.4 | (in progress) | - |

Enabling coarse-grid correction *lowers* the RK4+IRS stability limit on this case
(6.34 -> 5.31 as `fac_mgrid` goes 0.0 -> 0.2). The `0.4` leg is still running.

### Reproduce
```
scripts/duct_cfl_descend.sh --fac-mgrid 0.0 --fac-mgrid 0.2 --fac-mgrid 0.4
```

## Multigrid sweep, convergence criterion (RK4+IRS, 250k-cell case)

A later re-run of the same idea with two changes from the study above: the
default 250k-cell case grid (not 1M), and a **convergence** pass/fail instead of
mere non-divergence. `run_duct.py` now exits 0 only when the energy residual
falls >= 1 decade from its peak
(`ConvergenceHistory.check_convergence(decay=1.0)`), exits 2 if it runs
but stalls short of that, and exits 1 if it diverges. So the number below is the
highest CFL at which the case actually *converges*, which sits below the bare
divergence ceiling (near the limit the residual stalls without blowing up).

The search is descent-only, never bisection: hold the highest CFL known not to
converge, step DOWN by `--dcfl`, and halve `--dcfl` on each overshoot into the
converging band, stopping once the step drops below `--tol`. CFL only ever moves
downward.

| setting | value |
|---------|-------|
| scheme | RK4+IRS (`n_stage=4`, `sf_resid=1.0`) |
| convergence bar | energy residual down >= 1 decade (`decay=1.0`) |
| `ncell` (target) | 250 000 (grid 65 x 65 x 57, 240 825 nodes) |
| `n_step` | 500 |
| `n_levels` | 2 |
| descend `cfl_start`, `dcfl`, `tol` | 12.0, 1.0, 0.1 |

### Results

Run at two grid sizes to check mesh dependence. The 250k column uses the
settings table above; the 500k column doubles the streamwise resolution (grid
137 x 65 x 57, 507 585 nodes, `--ncell 500000`), everything else identical.

| `fac_mgrid` | max conv. CFL (250k) | fails at | max conv. CFL (500k) | fails at |
|-------------|----------------------|----------|----------------------|----------|
| 0.0 | 5.625 | 5.750 | 5.500 | 5.625 |
| 0.2 | 4.625 | 4.750 | 4.625 | 4.750 |
| 0.4 | 3.875 | 4.000 | 3.875 | 4.000 |

Enabling coarse-grid correction *lowers* the max converging CFL on this case,
monotonically: 5.625 -> 4.625 -> 3.875 as `fac_mgrid` goes 0.0 -> 0.2 -> 0.4, a
loss of ~1.75 in CFL headroom by 0.4. On this duct the Denton block-sum
correction is destabilising rather than accelerating. (The `fac_mgrid=0.0` leg
at `n_levels=2` matches the coarse-correction-off baseline, as expected: a zero
weight makes the MG depth almost irrelevant.)

The limits are essentially **grid-independent**: doubling the streamwise cell
count leaves 0.2 and 0.4 unchanged and moves 0.0 by a single `tol`-step
(5.625 -> 5.500), so the numbers reflect the scheme rather than the mesh.

### Reproduce
```
# 250k-cell case (run_duct default ncell)
scripts/duct_cfl_descend.sh --tol 0.1 \
    --fac-mgrid 0.0 --fac-mgrid 0.2 --fac-mgrid 0.4 \
    -- --n-levels 2 --n-step 500

# 500k-cell case (mesh-dependence check)
scripts/duct_cfl_descend.sh --tol 0.1 \
    --fac-mgrid 0.0 --fac-mgrid 0.2 --fac-mgrid 0.4 \
    -- --n-levels 2 --n-step 500 --ncell 500000
```

## Theory basis

### 1D von Neumann limits
- **RK4** (Jameson 4-stage, alpha = 1/4, 1/3, 1/2, 1; classical RK4 linear
  amplification): imaginary-axis limit `2*sqrt(2) ~= 2.828`.
- **scree** (Denton's `u^{n+1} = u^n - dt*(2 R^n - R^{n-1})` residual
  extrapolation): the two-level recurrence
  `g^2 - (1 - 2is) g - is = 0`, `s = nu*sin(theta)`, is stable for
  `s <= 1/sqrt(3)`, giving `CFL_max = 1/sqrt(3) ~= 0.577` (confirmed numerically to
  6 digits).

### Why the sum normalisation gave ~2.3x
`set_timestep_spectral` set `dt = cfl*vol / sum_d Lambda_d`, with
`Lambda_d = (|V.dA_d| + a*||dA_d||)`. On this mesh:
- `Sigma_d Lambda_d / Lambda_max = 2.30` (sum vs the single stiffest direction),
  which equals the measured/theory ratio because, with the solver's smoothing, the
  binding mode is the stiffest single grid direction (radius `Lambda_max`), not the
  summed corner mode.
- Multi-D undamped von Neumann (Blazek sum normalisation) gives scree 0.907, RK4
  4.44 -> a factor 1.57 above 1D (l1-vs-l2 acoustic over-count: `a` counted once
  per direction). The remaining ~1.45x to the measured 6.44 is the solver's
  artificial dissipation damping the corner mode. `1.57 * 1.45 = 2.28`.

### The fix
`lam_conv = max(lam_i, lam_j, lam_k)` (and `lam_diff` likewise on the max of the
directional viscous radii). Normalising on the largest single-direction radius
makes `cfl` the true 1D Courant number while staying aspect-ratio independent, at
the cost of being less conservative than the sum (it relies on the smoothing
keeping the corner modes subcritical; the sweep remains the stability backstop).

## Reproduce
```
# max-normalised limits (current tree)
scripts/duct_cfl_sweep.sh --ncell 250000 --case scree   --cfl-lo 0.3 --cfl-hi 1.0
scripts/duct_cfl_sweep.sh --ncell 250000 --case RK4     --cfl-lo 2.0 --cfl-hi 3.4
scripts/duct_cfl_sweep.sh --ncell 250000 --case RK4+IRS --cfl-lo 3.5 --cfl-hi 9.0
```
Each bisection is ~5-10 `run_duct.py` invocations at 1000 steps on ~240k nodes
(order 5-10 min per case).
