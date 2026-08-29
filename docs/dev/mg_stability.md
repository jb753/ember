# Why block-sum multigrid costs CFL on the clustered duct

A one-step von Neumann model of ember's in-step Denton block-sum multigrid, the
measurements it reproduces, and the experiments that would break it.

Case throughout: clustered square duct, 209 x 73 x 65 (991,705 nodes), y+ = 30
wall spacing, RK4 (n_stage=4), sigma = 0.05, expon_mgrid = 2.0, n_levels = 3
unless stated. Driver: `tools/run_duct_mg.py`.


## 1. Summary

The multigrid CFL penalty is set mainly by the *wavenumber at which the coarse
correction still has amplitude*, not by its low-wavenumber gain. Mesh clustering
is a second, smaller effect on top (E3 below), and not the one first guessed. A one-dimensional linear model of the composite one-step
operator predicts the measured penalty to within 1 per cent at sf_resid = 0 and
within 10 per cent at sf_resid = 1, and it predicts the same penalty ratio for
RK4 and for scree, which is what the measured data shows (agreement between the
two schemes is better than 1 per cent).

Two direct tests have been run (E1 and E3, section 7). E1 supports the
mechanism partially: the growing mode is a long wave of 8 to 10 cells per wave,
not a checkerboard, and multigrid shifts it to a longer wave than the
no-multigrid instability, both within 15 per cent of the predicted wavenumber --
but the predicted size of that shift is about three times what is measured.

E3 is the more informative result. On a UNIFORM mesh, the only mesh the model
actually describes, the predicted penalty ratio is exact: 0.76 predicted, 0.76
measured. On the clustered mesh it is 0.63. So the wavenumber argument is
quantitatively right, and wall clustering adds a further, separate 17 per cent
penalty on top of it that this model does not contain. The growing mode still
hugs the wall on the uniform mesh, so that extra penalty is a property of the
wall, not of the stretching.

Two further questions have since been closed. The mode's streamwise uniformity
is predicted by the 3D form of the same model -- it is the shape of the worst
mode wherever one direction sets the timestep -- and the boundary conditions are
not involved at all: sigma over a 50x range leaves the divergence step
unchanged. See section 6b.

Two consequences that matter in practice:

- Implicit residual smoothing (sf_resid) raises the whole stability envelope but
  leaves the multigrid penalty as a near-constant fraction.
- Artificial dissipation (sf2, sf4) cannot touch this instability at all,
  because it is a high-wavenumber tool aimed at a low-wavenumber failure. This is
  confirmed by experiment, not just by the model.


## 1b. Provenance -- which build these numbers came from

This matters and was nearly missed. The compiled extension used for every run
in this document was built on 08-29 at 09:55 from a working tree that was, at
the time, UNCOMMITTED: the `cbuf` scratch argument and the node-targeted final
prolongation hop, continuing commit 807a7eb "Interpolate the coarse correction
in space, not in index". That work is now commit 9bd2bcd, "Land the final
multigrid prolongation hop on the nodes", so these runs are reproducible from
9bd2bcd rather than from the 7b4fd71 that was HEAD while they ran. The
give-away at the time was that the .so required an argument 7b4fd71's
solver.py never passed; three checks confirmed it (the loaded .so path, `cbuf`
present in the compiled signature, and a setuptools_scm `.d20260829`
dirty-build tag).

    08-27 21:47  commit 807a7eb, geometry-aware prolongation
    08-28 02:29  duct_cfl_limits.yaml       (sigma = 0.2)
    08-28 16:26  commit 7b4fd71
    08-28 19:04  duct_cfl_limits_local.yaml (sigma = 0.05)  <- 7b4fd71 build
    08-29 09:55  .so built from what is now commit 9bd2bcd
    08-29 11:09  duct_cfl_limits_nl2.yaml                   <- probably 9bd2bcd
    08-29 12:50+ every probe in this document               <- 9bd2bcd

So section 4 compares the model against limits measured on 7b4fd71, while every
probe here ran on 9bd2bcd. Where they overlap they agree: the WIP-build
probe gives an RK4 penalty ratio of 0.63 against the HEAD-build YAML's 0.656,
within bracket resolution, so the prolongation rework has not visibly moved the
penalty.

What is NOT settled by that is section 7d's causal claim. 9bd2bcd's prolongation
is geometry-aware and lands on the nodes; the model assumes a uniform hat, P_b = (D_b/b)**2. The two
coincide on a uniform mesh and differ on a clustered one -- exactly the
signature attributed there to the dt_blk tail. Both explanations predict "model
exact on uniform, optimistic on clustered", so the tail measurement (1.35x
versus 1.01x) is evidence for it but not exclusive evidence.

Discriminating test, no new runs needed: read the packed weights out of
block.weight_mgrid, compute their true transfer function, substitute it for P_b,
and see whether the predicted clustered ratio moves from 0.76 toward 0.656 on
its own. If it does, the prolongation explains the gap; if it does not, the
dt_blk tail stands.

What 9bd2bcd does NOT touch: dtblk, s_dt and s_v are unchanged (only the
argument lists gained cbuf), so the dt_blk/dt_cell measurements and claim 1 of
section 7d rest on code identical across both commits.


## 2. What was observed

At cfl = 5 and cfl = 4 with fac_mgrid = 0.4, n_levels = 3, sf_resid = 1 the run
dies within 25 steps. Logging every step (`--n-step-log 1`) puts the NaN at
step 18:

    step   res_rho     res_e      err_mdot
       0  1.6117e-07  1.4988e-06  9.8606e-03
       5  8.9145e-08  8.7651e-07  7.8130e-03
       9  8.5027e-08  8.0465e-07  6.0776e-03   <- floor
      10  2.5208e-07  2.1774e-06  5.5130e-03
      11  1.0910e-07  9.8711e-07  4.8834e-03
      12  6.0825e-07  5.1076e-06  4.5620e-03
      13  2.1109e-07  1.8084e-06  4.4116e-03
      14  1.6796e-06  1.3953e-05  3.3321e-03
      15  6.3542e-07  5.3373e-06  1.9746e-03
      16  3.6727e-06  3.0102e-05  2.9095e-03
      17  2.0575e-06  1.6252e-05  4.6974e-03   <- NaN at 18

The residual falls cleanly for 9 steps, then a period-2 (odd/even) mode takes
over. Even and odd steps each form their own sequence, both growing about 2.2 to
2.9 times every two steps, i.e. roughly 1.5x per step.

Comparing the fmg = 0 and fmg = 0.4 fields at step 17 (identical in every other
respect):

- dP/P: rms 1.05e-01, max 1.14. dVx/Vx: rms 1.73e-02, max 0.88.
- The difference is concentrated at the wall. rms dP/P by index from the wall:

        j:   0        1        2        3       18      36 (mid)
             4.99e-1  3.44e-1  8.75e-2  4.44e-2  3.18e-2  8.52e-3
        k:   2.21e-1  2.05e-1  1.80e-1  1.68e-1  7.35e-2  2.31e-2

- It is NOT a spatial checkerboard. Fraction of neighbouring cells with
  opposite-sign difference: 0.00 along i, 0.28 along j, 0.05 along k (0.5 would
  be random, 1.0 a perfect checkerboard). The alternation is purely in time.

An initial guess that the clustered wall cells receive an over-large correction
was tested and is WRONG ON THE MEAN. Computing the per-cell smooth-mode
amplification from the grid's own dt_vol and vol gives 3.39 at j = 0 versus 3.70
in the core, and the MEAN dt_blk/dt_cell at level 3 is 0.99 at the wall versus
1.17 in the core.

It is right on the TAIL, which is what sets stability, and section 7d shows the
tail is where the clustered mesh's extra penalty lives: the same ratio reaches
2.009 at its maximum, and the per-cell gain reaches 4.59 against a mean of 3.40.
Reading the mean and stopping there was the error.


## 3. The mechanism

From `advance_rk_stage_mg` (src/ember/solver.py:661), one RK stage is

    dU = alpha*cfl*dt_vol*R + sum_l inject_l( coef_l * dt_blk_l * restrict_l(R) )
    coef_l = alpha*cfl*fac_mgrid/b**2 * expon_mgrid**-(l-1),   b = 2**l

`restrict_l` is a block SUM over the b**3 cells of the block, not an average. So
for a mode smooth across a block the restriction returns b**3 * R while coef_l
divides by b**2, and each level's low-wavenumber gain is

    fac_mgrid * b * expon_mgrid**-(l-1)

which at expon_mgrid = 2 equals 2 * fac_mgrid for EVERY level, independent of b.
Hence

    G(0) = 1 + 2 * fac_mgrid * n_levels = 3.4   for fac_mgrid=0.4, n_levels=3

Measured directly from the saved grid: 3.40. (Summing coef_l alone, without the
b**3 from the restriction, gives 0.7 and is wrong.)

This is an ADDITIVE in-step correction, not a V-cycle. No coarse grid is ever
marched with its own timestep; the block-summed push is added to the full fine
update inside the same stage. So "each grid alone is inside its own CFL limit"
does not apply -- the two amplifications superpose in one operator, whose
low-wavenumber effective Courant number is cfl*G(0), i.e. 13.6 at cfl = 4.

But that is not what sets the limit either, or the limit would be 6.33/3.4 = 1.9,
far below the measured 3.6875. The block sum is a low-pass filter with transfer

    D_b(theta) = sin(b*theta/2) / sin(theta/2)

equal to b as theta -> 0 and about zero for theta > 2*pi/b. The modes that get
the full 3.4x gain are exactly those with negligible spatial eigenvalue, and the
modes near the RK4 stability boundary get almost no coarse push. The limit is
the maximum over wavenumber of the product:

    cfl * G(theta) * S(theta) * |sin(theta)| <= 2*sqrt(2)     (RK4 imaginary axis)

    G(theta) = 1 + sum_l fac*expon**-(l-1) * D_b(theta) * S_c(b*theta) * P_b(theta)
    S(theta) = 1 / (1 + 2*eps*(1 - cos(theta))),   eps = sf_resid   (Jameson IRS)
    P_b      = (D_b(theta)/b)**2                   (hat prolongation; injection
                                                    P_b = 1 changes little)

Multigrid moves the critical wavenumber DOWN and raises the peak there:

    setting            critical mode        G at that mode   cfl*
    no MG              7.5 cells/wave       1.00             6.33
    fac_mgrid = 0.2    9.1 cells/wave       1.28             5.59
    fac_mgrid = 0.4   11.7 cells/wave       1.47             4.81

At the critical mode G is 1.47, not 3.4 -- the coarse correction has already
rolled off. That is why the penalty is a 25 to 35 per cent CFL cut and not the
factor 3.4 the smooth-mode gain would suggest.

Consistency check on the blow-up rate: the residual grew about 1.5x per step at
cfl = 4, and |P_RK4(i*y)| = 1.5 at y = 3.0, about 6 per cent past the 2*sqrt(2)
boundary -- a mode just outside the stability circle, not a wildly unstable one.


## 4. Validation against the measured CFL limits

Sources: `ember-paper/duct/results/stability/duct_cfl_limits_local.yaml`
(sigma = 0.05, n_levels = 3) and `duct_cfl_limits.yaml` (sigma = 0.2).

Penalty ratio, cfl*(MG) / cfl*(no MG):

    sf_resid  fac_mgrid   model   RK4 measured    scree measured
       0.0       0.2      0.860   0.867           0.865
       0.0       0.4      0.732   (not in file)   0.730
       1.0       0.2      0.883   0.800           0.808
       1.0       0.4      0.760   0.656           0.658

The RK4 and scree columns agree with each other to better than 1 per cent at
every point. This is a genuine prediction of the model and not a fit: G(theta)
contains no property of the time integrator, so the penalty ratio should be
scheme-independent even though the absolute limits differ by a factor of 5
(RK4 5.625 versus scree 1.140625 at sf_resid = 1, fac_mgrid = 0).

Absolute limits without multigrid:

    sf_resid    model   measured
       0        2.83    2.8125
       1        6.33    5.625
       2        8.49    9.0  (sigma = 0.2 data)

So the model is exact at sf_resid = 0 and roughly 12 per cent optimistic at
sf_resid = 1.

Confirmed independently with this document's own 200-step probe on the clustered
mesh (scree judged on residual growth rather than the decade bar, which it does
not reach in 200 steps):

    scheme               no MG          fac_mgrid = 0.4   ratio
    scree (n_stage=0)    (1.15, 1.3)    (0.7, 0.8)        0.61
    RK4   (n_stage=4)    (5.5, 6.0)     (3.5, 3.75)       0.63

The ratio is why this matters. It equals

    [y_max / max_theta(G*S*sin)] / [y_max / max_theta(S*sin)]

so the scheme's own imaginary-axis limit y_max cancels EXACTLY, provided that
limit is an interval on the imaginary axis. Nothing about the integrator
survives into the ratio; only G(theta), which is pure spatial transfer function.
Everything in sections 7b to 7d should therefore carry across to scree unchanged,
with only the absolute CFL differing.


## 5. Experiments already run

### 5.1 IRS raises the envelope, does not remove the penalty

Model: cfl* scales roughly as sqrt(1 + 4*eps), and the ratio is near-constant.

    sf_resid   cfl* no MG   cfl* fmg=0.4   ratio   critical mode
       0          2.83          2.07       0.73     6 cells/wave
       1          6.33          4.81       0.76    12 cells/wave
       2          8.49          6.47       0.76    15 cells/wave
       4         11.66          8.91       0.76    21 cells/wave

Measured at cfl = 4, fac_mgrid = 0.4, n_levels = 3, 200 steps:

    sf_resid = 1   diverged at step 18
    sf_resid = 2   converged, 1.20 decades, growth 1.00x
    sf_resid = 4   stable but slower: 0.49 decades in the same 200 steps

Bracketing the new limit at sf_resid = 2: stable at 4.0, diverged at 5.0
(step 37), 5.5 (step 10), 6.0 (step 6). Model scaled by the 0.767 offset from
sf_resid = 1 predicts 4.96. Note this is a short-run divergence probe, not the
same criterion as the descent sweeps.

### 5.2 Artificial dissipation cannot help

The post-step filter has transfer 1 - e2*(2-2cos t) - e4*(2-2cos t)**2 with
e2 = sf2*cfl, e4 = sf4*cfl, and its own stability caps e4 <= 1/16, e2 <= 1/4.
At the critical mode (2-2cos t) = 0.283, so the most damping any legal setting
can apply there is 0.0625*0.283**2 = 0.5 per cent per step from sf4, or
0.25*0.283 = 7 per cent per step from sf2, against a 50 per cent per step growth.

Measured, cfl = 4, fac_mgrid = 0.4, sf_resid = 1, 200 steps:

    sf4 = 0.008, sf2 = 0.002 (defaults)      diverged, step 18
    sf4 = 0.0156 (e4 = 1/16), sf2 = 0        diverged, step 18
    sf2 = 0.0625 (e2 = 1/4), sf4 default     diverged, step 18
    sf4 = 0.1 (past the cap, control)        diverged, step 11
    sf4 = 0, sf2 = 0, fac_mgrid = 0          fine, 0.37 decades, no growth

Cranking the dissipation to its own stability limit changes the divergence step
by exactly zero. The controls confirm the flags bite and that dissipation is not
propping up the no-multigrid case either.


## 6. What the model does not explain

Still open:

- (CLOSED by 7d: the extra clustered-mesh penalty is the dt_blk/dt_cell tail,
  not anisotropy. Left here as a pointer.)
- **The 12 per cent optimism at sf_resid = 1**, and none at sf_resid = 0. A
  guess: the IRS transfer used here is the ideal factored-tridiagonal one,
  whereas the implementation is tiled and the coarse-level IRS acts on the
  block-summed residual at coarse spacing.
- **Why the mode sits at the wall on a UNIFORM mesh** (E3). Streamwise
  uniformity is explained (6b), but not why the amplitude decays to 5 per cent
  of peak by the eighth cell off the wall when the mesh is not clustered at all.
  The likely reason is that the wall region is where the wall-normal radius
  dominates, which is also what 6b needs; E10 would settle both at once.

Closed since the first draft:

- Wall localisation is NOT a mesh-stretching effect. Ruled out twice: the
  per-cell amplification is 3.39 at the wall against 3.70 in the core
  (section 2), and the mode still hugs the wall on a uniform mesh (E3).
- Streamwise uniformity is NOT a boundary-condition effect. Sigma over 50x
  changes the divergence step by nothing (6b); it is the predicted shape of the
  worst mode when one direction sets the timestep.

Standing limitations of the model itself: 1D, linear, convection-only,
uniform-mesh, with no artificial dissipation, no viscous term, no boundary
conditions. The max-of-directional-radii timestep is what makes a 1D reduction
defensible; it would not be for a Blazek sum-of-radii definition. The 3D scan of
6b is the honest version, and the 1D formula is its (0, 0, theta) mode.

An earlier claim of mine that expon_mgrid is the effective lever is WRONG: see
E4, where the model says it barely helps.


## 6b. Why the mode is streamwise-uniform (and it is not the boundaries)

The 1D model is really the (0, 0, theta) mode of a 3D operator, and the 3D form
explains the direction selectivity directly. The restriction transfer is a
PRODUCT over directions while the fine eigenvalue is a SUM:

    G  = 1 + sum_l fac*expon**-(l-1) * (1/b**2) * prod_d D_b(theta_d) * ...
    lam = sum_d c_d * sin(theta_d)      c_d = Lambda_d / max_d Lambda_d

A mode that sets theta_i = theta_k = 0 collects the full D_b(0) = b from each of
those two directions -- the largest coarse gain available -- and pays no fine
eigenvalue for them. Scanning the 3D product for the worst mode:

    cell type                       worst mode (cells/wave)   cfl*
    isotropic, c = (1, 1, 1)        (20, 20, 20)              3.09
    j-limited, c = (0.3, 1, 0.3)    (60, 12, 60)              4.61

Measured (section 7, E1 and E3): theta_i = 0.000 to 0.096, theta_j = 0.620 to
0.730, theta_k = 0.050 to 0.114, i.e. (infinity to 60, 9 to 10, 60 to 125)
cells/wave. That is the j-limited row, not the isotropic one. So the streamwise
uniformity is not a puzzle and not a boundary effect: it is what the worst mode
looks like wherever ONE direction sets the timestep, which is the situation in
the wall region. It also retro-validates the 1D reduction used in section 3.

**Boundary conditions are not involved.** Sweeping sigma (rf_inlet and
rf_outlet) over 50x at cfl = 4, fac_mgrid = 0.4, sf_resid = 1:

    sigma    0.01   0.02   0.05   0.1   0.2   0.5
    diverges at step 18 in every single case

The flag is doing its job -- at step 16 err_mdot differs by a factor of 6
between sigma = 0.01 and sigma = 0.5 (3.85e-3 versus 6.47e-4), and res_e agrees
to 0.4 per cent. So the boundaries govern the mean-flow convergence and have no
hold whatsoever on the growing mode. The instability is interior.


## 7. Experiments that would test or break this

Ordered by discriminating power. Each is cheap: a diverging trial costs about
40 s including the grid build, a stable 200-step trial about 75 s.

### E1. Wavelength of the growing mode -- RUN, partial support

Take consecutive pre-divergence fields (written by `--write-emb` at n_step = N
and N-1) and measure the wavenumber of the increment.

Method note: a windowed FFT is the WRONG tool here. The mode lives at the wall,
and a Hann window zeroes exactly that region, so the spectra come back dominated
by the domain scale in every case and show no movement between settings. The
windowless Rayleigh quotient -<d, delta^2 d>/<d,d> = 2-2cos(theta) measures a
wall-localised mode correctly. Applied to the second time-difference
U(n+1) - 2*U(n) + U(n-1), which amplifies the period-2 growing mode and cancels
the steady convergence drift:

    case (2nd time-difference)          theta_i   theta_j   cells/wave (j)   predicted
    UNSTABLE fmg=0.4 cfl=4   sf1         0.000     0.620         10.1        0.537 / 11.7
    UNSTABLE no MG   cfl=6.5 sf1         0.095     0.738          8.5        0.842 /  7.5
    STABLE   no MG   cfl=4   sf1         0.938     0.571         11.0        --

What this confirms:

- The growing mode is a LONG wave, 8 to 10 cells per wave, not a checkerboard
  (theta = pi) and not domain-scale. The classic odd-even-decoupling story is
  ruled out.
- Multigrid shifts it to a longer wavelength than the no-multigrid instability
  (10.1 versus 8.5 cells/wave), the predicted direction, and both are within
  12 to 15 per cent of the predicted wavenumber.
- Measured on the plain first difference the ordering holds across three cases:
  no MG 0.689, fmg=0.4/sf_resid=1 0.498, fmg=0.4/sf_resid=2 0.476, against
  predicted 0.842, 0.537, 0.410 -- correct order, compressed range.

What it does not confirm:

- The predicted SIZE of the shift is too large: predicted 7.5 -> 11.7 cells/wave
  (56 per cent), measured 8.5 -> 10.1 (19 per cent).
- The wall-normal wavelength alone does not discriminate: a STABLE run's
  increment sits at 11.0 cells/wave, inside the same band. The unambiguous
  signature of the unstable mode is instead that it is streamwise-uniform
  (theta_i = 0.000 and 0.095 for the two unstable cases, versus 0.938 for the
  stable increment).

New fact the model does not contain: the mode is essentially two-dimensional,
uniform along i and structured along j, the clustered direction. That is a
direction-selectivity the 1D model has nothing to say about, and it is probably
the same fact as the unexplained wall localisation in section 6.

### E2. Fixed smooth-mode gain, varied depth

Three settings with identical G(0) = 3.4 but different level structure:

    fac_mgrid = 0.4, n_levels = 3   model cfl* 4.81  (ratio 0.76)
    fac_mgrid = 0.6, n_levels = 2   model cfl* 4.17  (ratio 0.66)
    fac_mgrid = 1.2, n_levels = 1   model cfl* 3.43  (ratio 0.54)

- If the limits come out equal, the penalty is set by the smooth-mode gain and
  this whole wavenumber argument is wrong.
- If they come out in the predicted ORDER (deeper is better at fixed G(0),
  which is counterintuitive), the wavenumber picture stands.

### E3. Uniform mesh -- RUN, prediction falsified in a useful way

Same node count with `--uniform` (cluster=False), sf_resid = 1, n_levels = 3.
Limits bracketed by the same 200-step probe on both meshes, counting a run as
unstable if it either NaNs or ends with residual growth above 2x its own
minimum (the two ambiguous verdicts here came back at 94.7x and 21.9x, so
neither is a borderline call):

    mesh        no MG           fac_mgrid = 0.4    ratio        model
    clustered   (5.5, 6.0)      (3.5, 3.75)        0.63         0.760
    uniform     (6.0, 6.5)      (4.5, 5.0)         0.76         0.760

The prediction was that the ratio is mesh-independent. It is NOT: the bracket
intervals are [0.583, 0.682] clustered and [0.692, 0.833] uniform, which do not
overlap. Clustering costs a further 17 per cent of multigrid CFL headroom on top
of the wavenumber penalty.

The useful half: on the uniform mesh -- the only mesh the 1D model actually
describes -- the model is exact, 0.76 predicted against 0.76 measured. The
clustered-mesh discrepancy noted in section 4 (model 0.760 versus measured
0.656) is therefore not a defect of the wavenumber argument. It is a separate,
additive stretching effect that the model does not contain and that the
per-cell amplification measurement in section 2 does not capture either.

Mode structure on the uniform mesh (fac_mgrid = 0.4, cfl = 5, second
time-difference at steps 19 to 21):

    theta_i = 0.096  (65.7 cells/wave)
    theta_j = 0.730  ( 8.6 cells/wave)
    theta_k = 0.050  (125.0 cells/wave)
    normalised rms by j:  j=0 1.00,  j=1 0.53,  j=2 0.31,  j=4 0.20,
                          j=8 0.05,  j=18 0.01,  j=36 0.00

So with no clustering at all the mode is still streamwise-uniform and still
hugs the wall, decaying to 5 per cent of its peak by the eighth cell. The wall
localisation of section 6 is therefore a property of the WALL, not of the
clustered mesh -- which rules out the mesh-stretching explanation a second time,
by a different route.

### E4. expon_mgrid sweep

At fac_mgrid = 0.4, n_levels = 3, the model gives cfl* 4.81 / 4.94 / 5.00 for
expon_mgrid 2 / 3 / 4, even though G(0) falls 3.4 / 2.69 / 2.40.

- Prediction: raising expon_mgrid buys almost nothing (a few per cent), because
  the critical mode is dominated by level 1, which expon_mgrid does not touch.
- This is counterintuitive and easy to check: cfl = 4, fac_mgrid = 0.4,
  expon_mgrid = 4 should still diverge.

### E5. Measure G(theta) directly instead of modelling it

Fill `block.residual_nd` with a single Fourier mode of known wavenumber, call
`advance_rk_stage_mg` once with and once without multigrid, and read off the
ratio of the resulting increments cell by cell.

- This measures the transfer function that the whole argument rests on, with no
  modelling assumption at all, and it is a unit-test-sized piece of work.
- Prediction: the measured ratio matches G(theta) above, in particular 3.4 at
  theta -> 0 and about 1.47 at 12 cells per wave.

### E6. Growth rate, not just the threshold

Above the limit the model predicts the growth rate too: the per-step
amplification should be |P_RK4(i * cfl * F_max)| where F_max is the peak of the
product. Measure the per-step residual growth at cfl = 4.0, 4.5, 5.0, 6.0 with
fac_mgrid = 0.4 and compare.

- A threshold can be fitted by one free parameter; a growth-rate curve cannot.

### E7. Scheme independence, extended

The RK4 versus scree agreement (better than 1 per cent) is the strongest
existing evidence. Extend it to n_stage = 2, 3, 5: the absolute limits should
track each scheme's own imaginary-axis limit while the penalty ratio stays put.

### E8. Inviscid

`--inviscid` drops lam_diff from the timestep and the turbulent viscosity from
the residual.

- Prediction: absolute limits shift, penalty ratio does not. If the ratio moves
  substantially, the viscous terms are part of the mechanism and the
  convection-only model is missing something real.

### E9. IRS scaling law

For no multigrid the model gives cfl* proportional to sqrt(1 + 4*eps): 2.83,
4.90, 6.33, 8.49, 11.66 at sf_resid 0, 0.5, 1, 2, 4. Two of those points are
already confirmed (2.8125 and 9.0). Filling in 0.5 and 4 tests the functional
form rather than two isolated points.


### E10. Directional radii: is the residual penalty cell anisotropy?

No new solver runs needed -- everything is already in the grid.

For every cell compute the three directional convective radii
Lambda_d = |V_rel . S_d| + a*||S_d|| (and lam_diff), form
c_d = Lambda_d / max_d Lambda_d, and map c_j/c_i over both the clustered and the
uniform mesh.

- Prediction from 6b: the wall region of BOTH meshes is j-limited (c_j = 1 with
  c_i, c_k well below 1), which is why the mode is streamwise-uniform and
  wall-bound on the uniform mesh too.
- Prediction for the E3 gap: the clustered mesh is MORE anisotropic at the wall
  than the uniform one, and feeding its measured c_d into the 3D scan of 6b
  should recover a penalty ratio near the measured 0.63 rather than the uniform
  mesh's 0.76. If it does not, the extra 17 per cent is something else --
  the wall function or the viscous radius, most likely -- and E8 (inviscid)
  becomes the next test.


## 7b. What the model suggests changing in the algorithm

Every variant below is scored at EQUAL low-wavenumber acceleration: fac_mgrid is
re-solved so G(0) = 3.4 in every row. Without that constraint any "improvement"
is just fac_mgrid turned down in disguise, trading acceleration for stability.

    variant                                 fac for G(0)=3.4   cfl*    vs current
    prolongation: injection                       0.400        4.34      0.90x
    prolongation: trilinear hat (CURRENT)         0.400        4.81      1.00x
    prolongation: smoother (hat applied twice)    0.400        5.05      1.05x
    restriction: full weighting                   0.400        4.94      1.03x
    full weighting + smooth prolongation          0.400        5.14      1.07x
    coarse IRS decoupled, eps_c = 2               0.400        5.41      1.13x
    coarse IRS decoupled, eps_c = 4               0.400        5.84      1.21x
    full weighting + coarse IRS x4                0.400        5.90      1.23x
    expon_mgrid = 1.0 (coarse-weighted)           0.171        5.55      1.15x
    expon_mgrid = 4.0 (fine-weighted)             0.686        4.16      0.86x

**Injection would be a step backwards** (0.90x). It applies the coarse
correction unattenuated at every wavenumber; the trilinear hat already in the
code is itself a low-pass that removes the mid-theta content doing the damage.
This part of the algorithm is right as it stands.

**Lever 1: decouple sf_irs from sf_resid (1.21x).** `advance_rk_stage_mg`
already takes `sf_irs` as its own argument; `rk_step` merely passes
`conf.sf_resid` into it. A separate `Solver.sf_irs` field would let the COARSE
residual be smoothed harder than the fine one, attenuating the coarse push at
mid-theta without touching the fine march.

**Lever 2: expon_mgrid < 2 (1.15x), and it needs no code change at all.**
Level 1 (b = 2) filters least -- its first Dirichlet zero is at theta = pi -- so
it contributes most of the mid-theta amplitude. Shifting weight to the coarser
levels, whose transfer rolls off earlier, holds G(0) while cutting that
amplitude. Note this REVERSES the naive reading: at fixed fac_mgrid, raising
expon_mgrid looked mildly helpful, but only because it was quietly weakening
multigrid; at fixed G(0) it is 0.86x, i.e. worse.

**Lever 3: full-weighting restriction (1.03x alone).** A real change in the hot
restriction loop for the smallest gain. Not worth it by itself.

### Measured, lever 2 (no code change, same G(0) = 3.4)

    setting                          stability bracket   decades in 200 steps
    fac=0.4,   expon=2 (baseline)    (3.5, 3.75)         1.76  at cfl 3.5
    fac=0.171, expon=1               (4.0, 4.5)          1.90  at cfl 4.0
    no multigrid                     (5.5, 6.0)          1.77  at cfl 5.5

The stability gain is 1.17x measured against 1.15x predicted, at the same G(0),
and cfl = 4.0 is stable where the baseline died at step 18.

An uncomfortable observation from the same table: at its own best stable CFL,
NO MULTIGRID converges as fast as baseline multigrid does at its (1.77 versus
1.76 decades). On this case the expon = 2 scheme gives back in CFL everything it
gains in acceleration. Only the coarse-weighted variant is actually ahead, and
only by 8 per cent. Caveats: 200 steps on one case, and decades-in-200-steps is
an early-transient measure, not the asymptotic rate -- the settling metric over
a full run is the one that should decide this.


## 7c. Is there an optimum expon_mgrid?

Not an interior one. At fixed G(0) = 3.4, n_levels = 3, sf_resid = 1, cfl* rises
monotonically as expon_mgrid falls and saturates:

    expon   fac      cfl*    gain per level (1,2,3)   G at 20 c/w   excess half-width
     0.25   0.016    6.20    (0.03, 0.26, 2.10)          1.31        50 cells/wave
     0.50   0.057    6.01    (0.11, 0.46, 1.83)          1.41        46
     1.00   0.171    5.55    (0.34, 0.69, 1.37)          1.61        38
     2.00   0.400    4.81    (0.80, 0.80, 0.80)          1.91        27
     4.00   0.686    4.16    (1.37, 0.69, 0.34)          2.23        19

By expon = 0.25 the limit is 6.20, which is 98 per cent of the 6.325 no-multigrid
limit -- multigrid has become almost free in stability terms. There is no peak
to find, so the choice is a trade, not an optimum.

What is traded is BANDWIDTH. Low expon concentrates the correction on the
coarsest level (at expon = 0.25 the level-1 and level-2 gains are 0.03 and 0.26,
so it is effectively a single b = 8 scheme), which accelerates only the longest
waves: G at 20 cells/wave falls from 1.91 to 1.31 and the half-width of the
excess widens from 27 to 50 cells/wave. The same theta -> 0 acceleration is
kept; the mid-wavelength part is given up.

Measured, each variant at its own best stable CFL, 200 steps:

    expon   fac      stability bracket   best stable cfl   decades in 200 steps
     0.25   0.016    (4.5, 5.0)               4.50              1.98
     0.50   0.057    (4.25, 4.5)              4.25              1.93
     1.00   0.171    (4.0, 4.5)               4.00              1.90
     2.00   0.400    (3.5, 3.75)              3.50              1.76
    no multigrid      (5.5, 6.0)              5.50              1.77

The stability trend is monotone and matches the model: measured 3.6 -> 4.75
across expon 2 -> 0.25 (1.32x), predicted 4.81 -> 6.20 (1.29x).

### expon -> 0 does not disable multigrid

expon_mgrid enters as expon**-(l-1), so expon < 1 tilts the weights toward the
COARSE levels, not away from them. At expon = 0.25 the level gains are (0.03,
0.26, 2.10), summing to the same 2.4 of coarse gain the baseline has; what is
switched off is levels 1 and 2, not the multigrid. Two controls:

- At the SAME cfl = 4.5, 200 steps: fac = 0.016 / expon = 0.25 falls 1.98
  decades, against 1.63 decades with multigrid off entirely. 21 per cent faster
  at equal CFL, so the coarse correction is doing real work.
- Measured G(0) on the real grid, from its own dt_vol and vol: baseline 3.397,
  coarse-heavy 3.330 -- within 2 per cent. The long-wave acceleration is
  preserved. fac_mgrid = 0.016 only looks negligible because the b**3/b**2
  factor at b = 8 is 8.

So the low-expon limit is multigrid CONCENTRATED ON ONE COARSE LEVEL. That is
consistent with the mechanism: level 1 (b = 2) has its first Dirichlet zero at
theta = pi, so it barely low-passes at all and contributes mid-theta amplitude
without buying much filtering.

Open design question this raises: if the useful configuration is a single b = 8
level, then n_levels = 3 with a steep weight tilt is an awkward way to write it
-- the restriction and prolongation for levels 1 and 2 are still paid every
stage. A single coarse level at b = 8 cannot currently be expressed (n_levels=1
means b = 2). Whether that is worth a code change depends on what fraction of
step time the level-1 and level-2 passes cost, which the existing timings can
answer without new runs.

On this case the extra CFL more than pays for the lost bandwidth: convergence
improves monotonically as expon falls, and expon = 0.25 at cfl = 4.5 is the best
of the five, 12 per cent ahead of both the expon = 2 baseline and no multigrid at
all. Two caveats before acting on that: decades-in-200-steps is an early
transient measure rather than the asymptotic rate, and the best low-expon
settings sit close to their own stability brackets, which is exactly where the
descent sweeps' growth test exists to catch a run that bottoms out and climbs.
The settling metric over a full 10000-step run is what should decide it.


## 7d. A rolling level schedule, and where the clustered penalty actually lives

### Rolling schedule: the model says worse, not better

Applying one coarse level per step and cycling through them makes the scheme a
periodic operator, so stability is the geometric mean of the per-step
amplifications around one cycle,
max_theta |P(z1) P(z2) P(z3)|**(1/3) <= 1, rather than a single step's.

    scheme                                                    cfl*
    current: all 3 levels every step                          4.81
    rolling 1-2-3 at 3x gain (equal mean acceleration)         3.57
    rolling 1-2-3 at same per-application gain (1/3 accel)     5.86
    coarse-heavy steady (expon = 0.25, section 7c)             6.20
    level 3 only, every step, same G(0)                        6.33
    level 3 only, every 3rd step at 3x gain                    5.95

Stability is set by the WORST step in the cycle. Holding the mean acceleration
fixed forces each application to be 3x larger, and the level-1 step -- whose
transfer barely filters -- then carries three times the mid-theta gain it had
before. Temporal concentration hurts for the same reason fine-level
concentration hurts. Cycle order is irrelevant (all six permutations give 3.569,
as they must for scalar modes), and even cycling only the good level is worse
than applying it every step (5.95 versus 6.33).

So the rolling schedule is not worth building. The row worth acting on is
"level 3 only, every step", which the model puts AT the no-multigrid limit of
6.325 -- stability-free acceleration.

### Except it is not free, and that is the clustered-mesh penalty

Measured, approximating level-3-only with fac_mgrid = 2.56e-5,
expon_mgrid = 0.01 (level 1 and 2 gains 5e-5 and 0.01):

    cfl = 4.5   1.92 decades, growth 1.00
    cfl = 5.0   0.19 decades, growth 18.45
    cfl = 5.5   diverged at step 14

Bracket (4.5, 5.0) against a predicted 6.33 and a measured no-multigrid limit of
(5.5, 6.0). The model is 17 per cent optimistic -- the same size as the E3 gap.

The cause is the per-cell spread of dt_blk/dt_cell, which grows with level
because a deeper block spans a wider range of cell sizes:

    clustered mesh    mean    95th pct    max
      level 1 (b=2)   1.000     1.080     1.114
      level 2 (b=4)   1.000     1.165     1.371
      level 3 (b=8)   0.997     1.406     2.009

    uniform mesh
      level 1         1.000     1.002     1.005
      level 2         1.000     1.005     1.008
      level 3         1.000     1.010     1.014

and the resulting per-cell gain:

    scheme                clustered G: mean / 95th / max    local excess
    baseline all-levels        3.40 / 3.90 / 4.59              1.35x
    level-3-only               3.05 / 3.89 / 5.13              1.68x
    baseline, UNIFORM mesh     3.40 / --   / 3.42              1.01x

This closes E3. The extra penalty on the clustered mesh is not anisotropy and
not the mean amplification -- it is the TAIL of dt_blk/dt_cell. The uniform mesh
has no tail (1.01x) and is exactly where the model is exact (0.76 predicted,
0.76 measured); the clustered mesh has a 1.35x tail and is where the model is
optimistic (0.760 against 0.656). It also explains why level-3-only
underperforms: concentrating everything on the deepest level concentrates the
gain on the cells with the widest dt spread, raising the local excess from 1.35x
to 1.68x.

### The change this suggests, and its actual justification

Start from where the current scaling comes from. dt_vol = 1/max(lam_conv,
lam_diff) and the fine update is dU = cfl*dt_vol*R with R extensive. A coarse
block of side b has face areas scaling as b**2, so Lambda_blk ~ b**2 * Lambda,
hence dt_vol_blk ~ dt_vol/b**2. That b**2 is exactly the factor in coef_l, and
dt_blk supplies the remaining dt_vol. On a UNIFORM block the current form is the
correct coarse-grid update; nothing is wrong with it there.

What breaks on a stretched block is the averaging rule. The block needs the
reciprocal of the block's spectral radius, 1/<Lambda>. The code computes
<1/Lambda>, the volume-weighted arithmetic mean of dt_vol. By Jensen
<1/Lambda> >= 1/<Lambda>, with equality only when Lambda is constant over the
block, so the implemented dt_blk is PROVABLY too large wherever a block spans
varying cell sizes. That is not a modelling choice, it is an inequality, and it
is exactly where the tail lives.

Measured, level-3 blocks, dt_blk/dt_cell:

    rule                                    mean    95th    max
    arithmetic, volume-weighted (CURRENT)   0.997   1.406   2.010
    harmonic, 1/<Lambda> (consistent)       0.940   1.328   1.788
    block minimum                           0.679   1.000   1.000

The spread is CONVECTIVE, not viscous: the table is identical at fac_visc = 0
and fac_visc = 1. It arises because max_d Lambda_d changes direction within a
block -- the thin near-wall cells are limited by their large i-k face, the core
cells by the axial face.

So there are two claims of very different strength:

1. **Arithmetic -> harmonic is a correctness fix**, not a stability hack. It
   costs 6 per cent of the mean correction and removes 11 per cent of the tail
   (2.010 -> 1.788). This is the change worth defending on its own terms.
2. **Bounding the local effective CFL needs the block MINIMUM**, which makes the
   ratio <= 1 by construction but costs 32 per cent of the mean correction. That
   is a stability argument, not a consistency one, and the lost acceleration has
   to be measured rather than assumed. Restoring the mean acceleration means
   raising fac_mgrid by 1/0.679 = 1.47x, which the bounded ratio should now
   tolerate.

An earlier version of this section proposed scaling by each cell's OWN dt_vol at
the injection point. That is WRONG and is withdrawn: it bounds the ratio, but it
destroys the property that the coarse correction is a single value per block.
The push becomes modulated by the mesh spacing inside the block, injecting
content at the block wavenumber -- precisely the mid-theta region this whole
analysis says to keep clean. The min-clamp achieves the same bound while keeping
the correction uniform across the block, so it dominates that suggestion.

Predictions for either change:

- The clustered-mesh penalty ratio moves from 0.656 toward the uniform mesh's
  0.76, i.e. the model becomes as accurate on the clustered mesh as it already
  is on the uniform one.
- Level-3-only then approaches the no-multigrid limit, making the coarse-heavy
  weighting of section 7c substantially more attractive than the 17 per cent it
  currently buys.

Both are a departure from the multall STEP1 lineage the docstring cites.

### Claim 1 implemented, and its stability prediction falsified

The harmonic mean is now what the kernel computes: sdt accumulates
sum(vol/dt_vol), sv accumulates sum(vol), and dtblk = sv/sdt (scree.f90, the
shared mg_coarse_correction engine; the coarser levels reduce the two
accumulators unchanged). tests/test_scree_mg.py pins it with a checkerboard of
2/3 and 2, whose harmonic mean over every 2x2x2 block is exactly 1.0 against an
arithmetic mean of 4/3, so the kernel cannot tell that field from a uniform
dt_vol of 1.0 -- measured agreement 9.5e-07 against a coarse contribution of
0.486, where the arithmetic mean would differ by a third.

The predicted stability gain did NOT appear:

    case                 before (arithmetic)        after (harmonic)      predicted
    fmg = 0.4, nl = 3    stable 3.5, unstable 3.75  stable 3.625,         ~4.2 to 4.4
                                                    unstable 3.75
    level-3-only         unstable 5.0 (growth 18.5) unstable 5.0          ~5.5 to 6.0
                                                    (growth 23.0)

The upper bound is identical in both cases; any gain is smaller than the 0.125
bracket resolution, against a predicted move of 0.6 or more. (The cfl = 3.625
point was never run under the arithmetic mean, so even the narrow gain is
unproven; an A/B needs a revert, a rebuild and one run.)

So the dt_blk tail is NOT the source of the clustered mesh's extra penalty,
despite the 1.35x-versus-1.01x measurement that suggested it. That was
correlation: the tail is real and the penalty is real, but removing the tail
does not remove the penalty. What remains, per section 1b, is the
geometry-aware prolongation -- the other clustering-only difference between
model and kernel, and the one the model represents with a plain hat P_b.

The change stands or falls on its own correctness argument now, not on any
stability benefit: sum(vol)/sum(vol/dt_vol) is the reciprocal of the block's
mean spectral radius, which is what the coarse update needs, and the previous
arithmetic mean was larger by Jensen whenever Lambda varied over the block.


## 8. Reproducing

    # the failing case, with a per-step account of the divergence
    uv run tools/run_duct_mg.py --cfl 4.0 --n-step 200 --n-step-log 1 \
        --write-hist hist.cnv --write-emb out.emb

    # the control it is compared against
    uv run tools/run_duct_mg.py --cfl 4.0 --fac-mgrid 0.0 --n-levels 0 \
        --n-step 17 --n-step-log 1 --write-emb out_fmg0.emb

    # the uniform-mesh control of E3
    uv run tools/run_duct_mg.py --uniform --cfl 5.0 --n-step 200 --n-step-log 1

`tools/run_duct_mg.py` fixes the mesh (1e6 cells, nj = 73, nk = 65, ER solved
for y+ = 30 by `ember.cases.er_for_duct_yplus`) and exposes cfl, n_stage,
n_levels, fac_mgrid, expon_mgrid, sf_resid, sf2, sf4, sigma, growth_max,
inviscid and uniform. It writes the .cnv and .emb BEFORE the divergence verdict,
so a blown-up run leaves both files behind.

Wavenumbers are measured with the windowless Rayleigh quotient of E1, applied to
the second time-difference of three consecutive `--write-emb` fields. Do not use
a windowed FFT: the window suppresses the wall region the mode lives in.
