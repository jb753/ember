# Plan: make the non-reflecting mixing plane faithful to Holmes (2008)

## Goal

Restore accurate flux conservation at the mixing plane (including at part-span
reversed stations, e.g. the rotor-exit tip separation that currently leaves a
12.7% mass error at the rotor/stator2 plane) while keeping the plane robust and
non-reflecting for the m>0 modes.

The exchange in `mixing_communicator.py:NonReflectingMixingCommunicator` is
already the Holmes control loop (his Eq. 15): flux difference -> `A^-1`
(`flux_to_primitive`) -> `B` (`primitive_to_chic`) -> `D` split -> back to
target, under-relaxed. We depart from Holmes in four places. This plan closes
all four, ending with his integrating controller.

## Success criteria

- Turbine continuation from `results/final.emb` stays finite (no supersonic
  guard trip) and converges (max residual back to ~1e-7).
- `err_mdot` at the rotor/stator2 plane drops from 12.7% to the forward-plane
  level, and the forward planes' small standing offset (0.68% at plane 1) also
  shrinks once the integrator removes the proportional steady-state error.
- Existing `tests/test_mixing_nonreflecting.py` stays green (with the
  reversed-station tests re-derived to the new, conservative fixed point).
- A new regression test: a *sustained one-sided* reversal converges to matched
  pitch-mean fluxes (the case that does NOT converge today).

## The four gaps

| # | Gap | Holmes | Current ember | Change |
|---|-----|--------|---------------|--------|
| 1 | Direction split | `D` switched by sign(vn) (p.4-5) | frozen forward (`_write_targets` 372-375) | switch the acoustic on symmetrised sign |
| 2 | Shared direction | average across interface so both sides agree (p.5) | each patch keys `_entering` off its own face (`nonreflecting.py` 690, 706) | communicator hands one direction to both patches |
| 3 | Controller gain | "turned down to ensure stability" | turbine runs `rf_exchange=1.0` | lower to ~0.02-0.05 |
| 4 | Relaxation form | integrate correction onto aux cells -> `ΔF -> 0` | proportional, re-anchored to baseline (leaves offset) | accumulate on the target, with anti-windup |

Already correct and kept: flux-difference form (not Giles' mixed-out inversion),
`A^-1`/`B`, the `Ma_clip` (Holmes Eq. 16), the symmetrised `b_avg` for the
Jacobians.

---

## Change 1 - direction-switched split (DONE, committed 8777b49)

**Where:** `mixing_communicator.py:_write_targets`, lines 372-391.

**Was:** `v1 = dchic[...,0]` (P bucket), `v2 = dchic[...,1:]` (inflow bucket),
regardless of flow direction.

**Now:** per span station, pick which acoustic feeds the pressure target by the
sign of the symmetrised axial Mach:

```
p_row = np.where(b_avg.Max.ravel() < 0.0, 1, 0)   # Vx-a (fwd) or Vx+a (rev) -> P bucket
```

- P bucket (`dchic_up`): acoustic row `p_row` only.
- inflow bucket (`dchic_dn`): the other acoustic + convective rows 2,3,4.

Convective rows always feed the inflow quantities; only the acoustic swaps.

**Why this is right (verified against both Holmes and ember's own mechanics,
2026-07-24):** it's tempting to read `nonreflecting.py`'s "the acoustic split
is fixed by the geometry" as ruling this out, but that sentence is about a
different fact -- which chic row (0 or 1) a *patch* treats as its own incoming
acoustic is fixed by `_sign_interior` alone (`_calc_split`'s
`acoustic = 1 if sign_interior>0 else 0`), independent of entering/leaving.
What flips under reversal is which *target row* (0-3 "inflow" vs 4 "P") that
fixed acoustic feeds, because `entering`/`leaving` itself flips. Tracing both
patches of a plane through a reversal: forward flow has patch1 (downstream
inlet) entering (wants rows 0-3 fed by `c_down` + convectives) and patch2
(upstream exit) leaving (wants row 4 fed by `c_up`); a reversed station flips
patch1 to leaving (now wants row 4 fed by *its* fixed acoustic, `c_down`) and
patch2 to entering (now wants rows 0-3 fed by *its* fixed acoustic, `c_up`).
So the acoustic driving the shared target's P-row must swap row0->row1 exactly
when the station reverses -- which is what the code does. This also matches
Holmes Eq. 10-11 (p.4): his selection matrix `D` is defined per span station
"depending on the sign of the normal velocity", i.e. by local flow direction,
not by a fixed upstream/downstream label -- confirming the switch is meant to
be per-station, not frozen.

**Risk:** stiffer feedback; unstable if run at high gain (this is why our first
attempt blew up). Mitigated by Change 3.

---

## Change 2 - single shared direction

**Where:** the communicator writes it; the patch consumes it.
`nonreflecting.py:_calc_reference` (690) currently calls `set_block_avg()` on the
patch's own face and derives `_entering` (706) from that local mean.

**Design:** the exchange already builds the symmetrised state `b_avg`. Have it
compute one direction per span station (with hysteresis, mirroring the patch's
existing `_frac_rev_off`, line 168) and stamp it on both patches, e.g.
`patch._entering_shared`. `NonReflectingMixingPatch._calc_entering` returns the
shared array when present, else falls back to the local computation (so inlet /
outlet are untouched).

**Why:** at a mixed-sign station (one side reversed, one forward - the tip band)
the communicator's split and the two patches' prescribed rows must reference the
same direction, or the target rows are built from one set of characteristics and
read as another. Holmes' cross-interface averaging exists precisely to remove
this.

**Ordering dependency:** the exchange must run *before* the patches'
`update_soln` each step so the shared direction is fresh. Verify the solver loop
order in `solver.py` / `grid.update_bconds`; the exchange (`comm.exchange`) and
the patch reference build must not race.

---

## Change 3 - controller gain

**Where:** configuration, not code. `run.py:DEFAULTS["rf_exchange"]` is 1.0;
the solver pushes it onto the patches (`solver.py:739-745`).

**New:** lower the mixing-plane `rf_exchange` to ~0.02-0.05 for the switched
split. For the integrating form (Change 4) the gain is an integrator gain with a
tighter stability limit - start at 0.02 and tune down if the transient rings.

No structural change; just a value, and it is read per-exchange so it can be
retuned without rebuilding the communicator.

---

## Change 4 - integrating, not proportional (the core of "full Holmes")

**Where:** `mixing_communicator.py:_write_targets`, lines 391-407, and the
`self._baseline` computation in `_prepare_pair` (320-323).

**Now (proportional):** `target_n = baseline_n + rf * correction_n`, where
`baseline` is the live symmetrised mean. No memory -> steady-state offset even at
forward stations; correction is re-derived each step and never accumulates.

**New (integrating):** accumulate the correction on the persistent target, as
Holmes does on the auxiliary cells:

```
target_n = target_{n-1} + rf * correction_n
```

`target_{n-1}` is the current `_target` (symmetrised across the two sides, as the
reflecting communicator already does with `0.5*(target1+target2)`). At the fixed
point `target_n = target_{n-1}` forces `rf*correction = 0`, i.e. `ΔF = 0` -
exact flux balance. This is the property proportional relaxation cannot deliver.

**This is the form that ran away before** (old ember "integrating onto its own
previous target"), but that runaway was with the *un-switched* split feeding a
wrong-direction correction. With Changes 1-3 the accumulated correction is
correctly oriented, which is exactly the configuration Holmes reports as stable
under reversal (his Table 2). Anti-windup is the safety net, not the mechanism.

**Anti-windup (required safety net):**

1. *Leaky integrator* - keep `self._baseline` and bleed the target toward it:
   ```
   target_n = target_{n-1} + rf*correction_n - leak*(target_{n-1} - baseline_n)
   ```
   `leak = 0` is pure Holmes (zero steady-state error, can wind up); small `leak`
   bounds the windup at the cost of a tiny residual `ΔF ~ leak/rf`. Start at
   `leak = 0`, engage only if a hard station winds up.
2. *Physical clamp* - after the update, reject targets that imply non-physical
   state (rho<=0, p<=0, |Ma|>=1) by falling back to the baseline at those
   stations. This directly prevents the "target walks to a state the flow was
   never in" failure the old version hit.
3. The existing `Ma_clip` (Eq. 16) stays - it keeps the Jacobians conditioned
   near vn=0 and only earns its keep once Change 1 exists.

**Simplification available:** with `leak=0` the `self._baseline` symmetrisation
can be deleted; keep it only if the leaky term is used.

---

## Test plan

**Unit (`tests/test_mixing_nonreflecting.py`):**

- New: `test_split_switches_on_symmetrised_sign` - a reversed symmetrised mean
  routes the *other* acoustic to the pressure target row.
- New: `test_sustained_one_sided_reversal_conserves` - the harness that does not
  converge today (rotor-exit reversed, stator2 inlet forward, interiors
  sustained) now relaxes to `flux_gap < 1e-4`. This is the regression proof.
- Re-derive `test_a_reversed_station_relaxes_like_any_other` and the two
  reversed-station target tests to the new conservative fixed point (they encode
  the *old* proportional behaviour).
- New: `test_shared_direction_overrides_local` - a mixed-sign station uses the
  communicator's direction on both patches.

**Integration:**

- Turbine continuation from `final.emb` at `rf_exchange=0.02`: assert finite,
  residual reconverges, `err_mdot` at rotor/stator2 drops below (target) ~1%.
  Compare against the Stage-0 control (12.7%).

**Golden:** `test_residual_golden` and friends may shift slightly as the
integrator removes the forward-plane offset; regenerate goldens deliberately
with a noted reason, not blindly.

---

## Rollout (staged, each independently revertible)

- **Stage 0 - control.** Record current `err_mdot` (12.7%) and residual history
  as the baseline to beat. No code change.
- **Stage 1 - Change 1 + Change 3.** Switched split, proportional still, low
  gain. Cheapest, safest; likely recovers most of the error. Gate: suite green,
  turbine stable, `err_mdot` down.
- **Stage 2 - Change 2.** Shared direction; closes the mixed-sign tip band. Gate:
  tip-band conservation, no chattering.
- **Stage 3 - Change 4.** Integrator + anti-windup; drives `ΔF -> 0` and removes
  the forward-plane offset. Gate: `err_mdot` near machine level, still stable.

Each stage is a small, localized diff. The proportional form remains the fallback
at every stage, so a regression at Stage 3 rolls back to a still-improved Stage 2.

## Open questions to resolve during implementation

- Does the solver loop guarantee exchange-before-reference every step, or is
  there a stage where the patch rebuilds its reference from a stale shared
  direction? (Change 2 dependency.)
- Is the integrator gain limit tighter than the clip can protect at the deepest
  reversed stations, or does `leak=0` hold? (Determines whether the leaky term
  ships on by default.)
- Do we want Holmes' "mixed vs unmixed" flux choice (his Eq. 4 vs 5, entropy vs
  normal-momentum flux) exposed as a knob, or fix one? Out of scope for
  conservation, but it is the same machinery.
