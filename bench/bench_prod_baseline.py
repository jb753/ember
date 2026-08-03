#!/usr/bin/env -S uv run
"""Reproducibility baseline: how well can we measure `prod` at all?

Everything else in this study is a RATIO against production. That ratio can
only be trusted to the precision with which production itself is measurable,
and that had never been established. This measures it, for one kernel, one
size, one regime -- deliberately the smallest experiment that can answer the
question.

Corrected on every count the earlier harness got wrong:

  ONE ARM. No round-robin, so no arm-set dependence. Interleaving was
  introduced to cancel drift between arms; with a single arm there is no
  drift to cancel and it only added a confound (sweeping the arm set moved a
  ratio by 16 points).

  BARRIER, NOT A SLEEP. Ranks synchronise through shared memory: once at
  startup, replacing the 180 s fixed rendezvous (the slowest rank of 16 was
  ready at 16.2 s, so the constant was ~11x oversized and was ~98% of the
  wall clock of every contended run), and again BEFORE EVERY TIMED CALL. The
  per-call barrier is the substantive fix: without it ranks free-run and
  drift out of phase, so each rank is timed against a different, unrecorded
  mixture of what its neighbours happen to be doing. With it, every rank is
  inside the same kernel at the same time -- which is both a defined
  condition and what production actually does.

  REPLICATION AT THE LAUNCH, NOT THE REP. 50 reps in one process are 50
  correlated views of a single draw of page placement, allocation alignment,
  core assignment and thermal state; the per-rep scatter is ~3% while the
  same configuration moved 10 points between launches. Variance is
  sigma^2_launch + sigma^2_rep/n, and with sigma_launch ~ 3% and a 50-rep
  standard error of ~0.8%, more reps chase the term that is already
  negligible. Reps are therefore kept low and the driver repeats LAUNCHES.

  NO FLUSH. At 1M, prod streams ~152 MB against a 20 MB L3 (3.3 MB per rank
  once shared), so nothing survives between calls and there is nothing to
  flush. Flushing here would only inject 48 MB of traffic that is untimed for
  the rank doing it and squarely inside its neighbours' timed windows.

  MIN IS NOT USED. Under contention the minimum preferentially samples the
  instants when neighbours were between calls, i.e. it erases the very
  contention being measured. Median per rank, median across ranks.

Usage (normally via bench/run_prod_baseline.sh):
    EMBER_BENCH_RANK=0 EMBER_BARRIER=name-0 uv run python \\
        bench/bench_prod_baseline.py --nranks 16 --ncell 1000000 --reps 30
"""

import argparse
import json
import os
import statistics
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

BARRIER_TIMEOUT = 300.0


class Barrier:
    """Lock-free sense-reversing barrier over shared memory.

    Each rank writes only its OWN slot and reads the others, so there is no
    atomicity requirement: an int64 store is single-copy atomic on x86-64 and
    cache coherence makes it visible. Spin, never sleep -- the whole point is
    that the barrier costs microseconds against a ~41 ms call.
    """

    def __init__(self, name, rank, nranks):
        self.rank, self.nranks, self.gen = rank, nranks, 0
        size = 8 * nranks
        if rank == 0:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            self.owner = True
            np.ndarray((nranks,), dtype=np.int64, buffer=self.shm.buf)[:] = -1
        else:
            self.owner = False
            t0 = time.time()
            while True:
                try:
                    self.shm = shared_memory.SharedMemory(name=name)
                    # CPython registers every attach with the resource
                    # tracker, which then unlinks the segment when THIS
                    # process exits -- so a non-owner finishing first
                    # destroys the segment under the owner's feet. Only the
                    # creator should own its lifetime.
                    resource_tracker.unregister(self.shm._name, "shared_memory")
                    break
                except (FileNotFoundError, ValueError):
                    # FileNotFoundError: rank 0 has not created it yet.
                    # ValueError ("cannot mmap an empty file"): it has been
                    # created but not yet sized -- shm_open and ftruncate are
                    # two syscalls, and a non-owner that lands between them
                    # sees a zero-length segment. Both are the same "not ready
                    # yet" condition and both must retry; catching only the
                    # first leaves a race that widens the fewer ranks there
                    # are (fewer ranks start sooner after rank 0).
                    if time.time() - t0 > BARRIER_TIMEOUT:
                        raise
                    time.sleep(0.01)
        self.slots = np.ndarray((nranks,), dtype=np.int64, buffer=self.shm.buf)

    def wait(self):
        self.gen += 1
        g = self.gen
        self.slots[self.rank] = g
        t0 = time.time()
        while True:
            if (self.slots >= g).all():
                return
            if time.time() - t0 > BARRIER_TIMEOUT:
                raise RuntimeError(f"barrier timeout at gen {g}: {self.slots}")

    def close(self):
        self.shm.close()
        if self.owner:
            # Give the others a moment to detach before removing the segment.
            time.sleep(0.5)
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="prod")
    ap.add_argument(
        "--kernel",
        default="residual",
        choices=("residual", "irs", "update"),
        help="which kernel's arm set to time: set_residual "
        "(residual_arms.py) or the IRS smoother (irs_arms.py)",
    )
    ap.add_argument("--ncell", type=int, default=1_000_000)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--nranks", type=int, default=16)
    ap.add_argument("--launch", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rank = int(os.environ["EMBER_BENCH_RANK"])
    barrier = Barrier(os.environ["EMBER_BARRIER"], rank, args.nranks)

    from residual_arms import DAMPIN, build_case, callers

    t0 = time.perf_counter()
    grid, b = build_case(args.ncell)
    du = b.residual_nd
    du.flags.writeable = True
    if args.kernel == "update":
        # Phase 3: the limiter+IRS fusion spans set_residual and the smoother,
        # so the pair is timed together. dU is an output of the first call, so
        # unlike --kernel irs there is nothing to seed.
        import irs_arms

        irs_arms.swirl_state(b)
        built = irs_arms.callers_update(b, du)
    elif args.kernel == "irs":
        # The smoother consumes the residual, so seed dU with a real one
        # (untimed: it is the previous kernel's output, and pricing it here
        # would be timing set_residual). See irs_arms.seed_du.
        import irs_arms

        irs_arms.swirl_state(b)
        irs_arms.seed_du(b)
        built = irs_arms.callers_irs(b, du)
    else:
        built = callers(b, du, DAMPIN)
    if args.arm not in built:
        # The arm's symbol is not in the .so. Almost always means the build
        # did not include it (EMBER_BENCH_KERNELS unset, or `uv run` re-synced
        # and rebuilt without it -- see UV_NO_SYNC in the driver scripts).
        # Fail here with the reason rather than as a KeyError inside one rank
        # while its peers hang on the barrier.
        raise SystemExit(
            f"arm {args.arm!r} is not in the built extension "
            f"(have: {sorted(built)}). Rebuild with "
            f"EMBER_BENCH_KERNELS=<name> make compile."
        )
    fn = built[args.arm]
    t_build = time.perf_counter() - t0

    ni, nj, nk = b.shape
    ncell = (ni - 1) * (nj - 1) * (nk - 1)

    # Startup rendezvous: no fixed sleep, just wait for the slowest builder.
    barrier.wait()

    for _ in range(args.warmup):
        barrier.wait()
        fn()

    samples = np.empty(args.reps)
    for i in range(args.reps):
        barrier.wait()  # every rank enters the kernel together
        t = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t) / ncell * 1e9

    med = float(np.median(samples))
    extra = {}
    if args.kernel == "irs":
        # The IRS smoother runs in place, so rep n smooths rep n-1's output.
        # That is timing-neutral only while the field stays normal -- assert
        # it rather than assume it (irs_arms.check_denormals).
        import irs_arms

        extra = irs_arms.check_denormals(du)
        if extra["frac_subnormal"] > 1e-6:
            print(
                f"WARNING rank {rank}: {extra['frac_subnormal']:.2%} of dU is "
                f"subnormal after {args.reps} in-place reps -- timing suspect",
                flush=True,
            )
    print(
        f"launch {args.launch:>2} rank {rank:>2} {args.arm:>8}  {med:7.3f} ns/cell  "
        f"(p5 {np.percentile(samples, 5):.2f} p95 {np.percentile(samples, 95):.2f}, "
        f"build {t_build:.1f}s)",
        flush=True,
    )

    if args.json:
        with open(args.json, "a") as fh:
            fh.write(
                json.dumps(
                    dict(
                        **extra,
                        arm=args.arm,
                        launch=args.launch,
                        rank=rank,
                        nranks=args.nranks,
                        ncell=ncell,
                        shape=[ni, nj, nk],
                        reps=args.reps,
                        median=med,
                        min=float(samples.min()),
                        mean=float(samples.mean()),
                        std=float(samples.std()),
                        p5=float(np.percentile(samples, 5)),
                        p95=float(np.percentile(samples, 95)),
                        build_s=t_build,
                    )
                )
                + "\n"
            )

    barrier.wait()  # nobody leaves until everyone has finished
    barrier.close()
    return 0


def analyze(path):
    rows = [json.loads(l) for l in open(path)]
    arms = sorted({r.get("arm", "prod") for r in rows})
    if len(arms) > 1:
        return analyze_multi(rows, arms)
    launches = sorted({r["launch"] for r in rows})
    print(
        f"\n{len(rows)} rank-rows over {len(launches)} launches, "
        f"{rows[0]['nranks']} ranks, ncell={rows[0]['ncell']}, "
        f"{rows[0]['reps']} reps/launch\n"
    )
    print(
        f"{'launch':>6} {'median-of-ranks':>16} {'rank spread':>18} "
        f"{'within-rank p5-p95':>20}"
    )
    per_launch = []
    for L in launches:
        sel = [r for r in rows if r["launch"] == L]
        v = sorted(r["median"] for r in sel)
        m = statistics.median(v)
        per_launch.append(m)
        sp = [r["p95"] / r["median"] - 1 for r in sel]
        print(
            f"{L:>6} {m:>15.3f}  {min(v):>7.2f}-{max(v):<7.2f} "
            f"{100 * statistics.median(sp):>17.1f}%"
        )

    g = statistics.median(per_launch)
    lo, hi = min(per_launch), max(per_launch)
    half = (hi - lo) / 2 / g * 100
    sd = statistics.stdev(per_launch) if len(per_launch) > 1 else 0.0
    print("\nacross launches (the number that matters):")
    print(f"  median          {g:.3f} ns/cell")
    print(f"  range           {lo:.3f} - {hi:.3f}  (half-range {half:+.2f}% of median)")
    print(f"  stdev           {sd:.3f}  ({sd / g * 100:.2f}%)")
    if len(per_launch) > 1:
        sem = sd / len(per_launch) ** 0.5
        print(f"  s.e. of median  {sem:.3f}  ({sem / g * 100:.2f}%)")
    verdict = "YES" if half <= 1.0 else "NO"
    print(
        f"\n  reproducible to +/-1% launch-to-launch? {verdict} "
        f"(half-range {half:.2f}%)"
    )


def _launch_medians(rows, arm, stat="median"):
    out = {}
    for L in sorted({r["launch"] for r in rows if r.get("arm", "prod") == arm}):
        v = [
            r.get(stat, r["median"])
            for r in rows
            if r["launch"] == L and r.get("arm", "prod") == arm
        ]
        out[L] = statistics.median(v)
    return out


def analyze_multi(rows, arms, stat="median"):
    """Compare arms measured in SEPARATE launch sets from the same build.

    Legitimate only because prod's launch-to-launch half-range is ~0.4%: the
    uncertainty on a ratio of two independently measured arms is then well
    under the differences being resolved. It is NOT legitimate across builds
    -- see the gauge note in the module docstring.
    """
    # The incumbent is `prod` for the set_residual arms and `irs` for the
    # smoother arms; a results file only ever holds one kernel's arms.
    baseline = next((a for a in ("prod", "irs", "unfused") if a in arms), arms[0])
    base = _launch_medians(rows, baseline, stat)
    if not base:
        print(f"no `{baseline}` rows: nothing to compare against")
        return
    b = statistics.median(base.values())
    bsd = statistics.stdev(base.values()) if len(base) > 1 else 0.0
    print(
        f"\n{'arm':>8} {'launches':>9} {'median':>9} {'half-range':>11} "
        f"{'stdev':>8} {'vs prod':>9}"
    )
    for arm in arms:
        m = _launch_medians(rows, arm, stat)
        if not m:
            continue
        v = list(m.values())
        med = statistics.median(v)
        half = (max(v) - min(v)) / 2 / med * 100
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        rel = "" if arm == baseline else f"{(med / b - 1) * 100:+8.2f}%"
        print(
            f"{arm:>8} {len(v):>9} {med:>9.3f} {half:>10.2f}% {sd / med * 100:>7.2f}% {rel:>9}"
        )
    if bsd:
        print(
            f"\n  {baseline} stdev {bsd / b * 100:.2f}% -> a ratio of two independently "
            f"measured arms carries ~{(2**0.5) * bsd / b * 100:.2f}%"
        )
    _paired(rows, arms, baseline, stat)


def _paired(rows, arms, baseline, stat):
    """Per-launch paired comparison against the baseline arm.

    The table above compares each arm's median-ACROSS-launches with the
    baseline's, which treats the two as independent samples. They are not:
    run_all_arms.sh is launch-outer/arm-inner, so within one launch every arm
    is measured under the same thermal state, the same page placement and the
    same background load. Differencing inside a launch cancels all of that.

    It matters whenever the machine drifts. On a thermally throttling mobile
    part an IRS screen drifted ~8% between launch 5 and launch 8, which put
    4.3% of stdev on the unpaired medians and buried a 2% effect that was
    nonetheless present in 14 of 15 individual launches. Report both: the
    unpaired number is the honest absolute spread, the paired one is the
    resolution actually available on the difference.

    The sign count is the assumption-free companion to the mean -- under the
    null it is a fair coin, so k of n one way is a plain binomial tail, no
    normality or equal-variance claim required.
    """
    per = {a: _launch_medians(rows, a, stat) for a in arms}
    common = set.intersection(*(set(v) for v in per.values() if v)) if per else set()
    if len(common) < 2 or len(arms) < 2:
        return
    L = sorted(common)
    print(f"\n  paired within launch ({len(L)} launches, drift-cancelled):")
    for arm in arms:
        if arm == baseline or not per[arm]:
            continue
        rat = [per[arm][x] / per[baseline][x] - 1 for x in L]
        mean = statistics.mean(rat)
        sd = statistics.stdev(rat) if len(rat) > 1 else 0.0
        sem = sd / len(rat) ** 0.5
        wins = sum(r < 0 for r in rat)
        print(
            f"{arm:>8} {100 * mean:+8.2f}% +/- {100 * sem:.2f}% (s.e.)   "
            f"faster in {wins}/{len(rat)} launches"
        )


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--analyze":
        analyze(sys.argv[2])
    else:
        sys.exit(main())
