"""Iterative setters on a 97^2 patch: perfect (closed form) vs real (Newton).

A real gas has no closed form for any pair but (rho, u), so these solve by
Newton and each iteration walks all six entropy surfaces again. They sit on the
boundary-condition path -- NonReflecting.apply calls set_rho_s and set_P_rho on
every patch on every step -- so the count of surface evaluations matters as much
as the cost of one, and both are reported. Run from the repo root:

    uv run python bench/bench_fluid_setters.py
"""

import sys
import time

sys.path.insert(0, "tests")
import numpy as np
from conftest import VanDerWaals, fit_real_fluid
from ember.fluid import PerfectFluid, RealFluid

NI = 97
SHAPE = (NI, NI)
N = NI * NI
REP = 5


def timeit(fn, *a):
    fn(*a)
    best = float("inf")
    for _ in range(REP):
        t0 = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def count_evals(fluid, fn, *a):
    """How many surface evaluations one solve costs."""
    n = {"p1": 0, "p2": 0}
    p1, p2 = RealFluid._partials1, RealFluid._partials2
    RealFluid._partials1 = lambda s, r, u: (
        n.__setitem__("p1", n["p1"] + 1),
        p1(s, r, u),
    )[1]
    RealFluid._partials2 = lambda s, r, u: (
        n.__setitem__("p2", n["p2"] + 1),
        p2(s, r, u),
    )[1]
    try:
        fn(*a)
    finally:
        RealFluid._partials1, RealFluid._partials2 = p1, p2
    return n


perfect = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
real = fit_real_fluid(VanDerWaals(), (1.0, 150.0), (3.0e5, 5.0e5), order=8)

rng = np.random.default_rng(0)


def span(lim, lo=0.3, hi=0.7):
    a = lim[0] + lo * (lim[1] - lim[0])
    b = lim[0] + hi * (lim[1] - lim[0])
    return rng.uniform(a, b, SHAPE).astype(np.float32)


print(f"patch {NI}^2 = {N:,} nodes, best of {REP}\n")
print(f"{'setter':10s} {'perfect':>12s} {'real o8':>12s} {'ratio':>8s}   surface evals")
for name in ("set_P_rho", "set_rho_s", "set_h_s", "set_P_T"):
    rho_r, u_r = span(real.rho_lim_nd), span(real.u_lim_nd)
    rho_p, u_p = span((0.5, 5.0)), span((1e4, 3e5))
    args_r = {
        "set_P_rho": (real.get_P(rho_r, u_r), rho_r),
        "set_rho_s": (rho_r, real.get_s(rho_r, u_r)),
        "set_h_s": (real.get_h(rho_r, u_r), real.get_s(rho_r, u_r)),
        "set_P_T": (real.get_P(rho_r, u_r), real.get_T(rho_r, u_r)),
    }[name]
    args_p = {
        "set_P_rho": (perfect.get_P(rho_p, u_p), rho_p),
        "set_rho_s": (rho_p, perfect.get_s(rho_p, u_p)),
        "set_h_s": (perfect.get_h(rho_p, u_p), perfect.get_s(rho_p, u_p)),
        "set_P_T": (perfect.get_P(rho_p, u_p), perfect.get_T(rho_p, u_p)),
    }[name]
    tp = timeit(getattr(perfect, name), *args_p)
    tr = timeit(getattr(real, name), *args_r)
    ev = count_evals(real, getattr(real, name), *args_r)
    print(
        f"{name:10s} {tp:9.3f} ms {tr:9.2f} ms {tr / tp:7.0f}x   "
        f"_partials1 x{ev['p1']}, _partials2 x{ev['p2']}"
    )

# For scale: the fused getter on the same patch.
outs = [np.zeros(SHAPE, np.float32, order="F") for _ in range(3)]
rho_f = np.asfortranarray(span(real.rho_lim_nd))
u_f = np.asfortranarray(span(real.u_lim_nd))
t = timeit(lambda: real.get_P_h_T(rho_f, u_f, *outs))
print(
    f"\nfor scale, get_P_h_T on the same patch: {t:.3f} ms ({t * 1e6 / N:.1f} ns/node)"
)
