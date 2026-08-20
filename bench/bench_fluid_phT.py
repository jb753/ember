"""Time PerfectFluid vs RealFluid get_P_h_T over a 97^3 grid.

Both fluids dispatch to a Fortran kernel for a float32 call with all three
outputs supplied, which is what the solver makes once per Runge-Kutta stage;
anything else falls back to numpy. Run from the repo root, since the RealFluid
case is built by the fitting helper in tests/conftest.py:

    uv run python bench/bench_fluid_phT.py
"""

import sys
import time

sys.path.insert(0, "tests")

import numpy as np

from conftest import VanDerWaals, fit_real_fluid
from ember.fluid import PerfectFluid

NI = 97
SHAPE = (NI, NI, NI)
N = NI**3
REPEAT = 7


def timeit(fn, *args):
    """Best of REPEAT wall times, in milliseconds."""
    fn(*args)  # warm up
    best = float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def make_state(fluid, lo_frac=0.25, hi_frac=0.75):
    """A float32 F-contiguous state filling the grid, inside any fit box."""
    rng = np.random.default_rng(0)
    if hasattr(fluid, "rho_lim_nd"):
        rho_lo, rho_hi = fluid.rho_lim_nd
        u_lo, u_hi = fluid.u_lim_nd
    else:
        rho_lo, rho_hi, u_lo, u_hi = 0.5, 5.0, 1.0e4, 3.0e5

    def _span(lo, hi):
        a = lo + lo_frac * (hi - lo)
        b = lo + hi_frac * (hi - lo)
        return np.asfortranarray(rng.uniform(a, b, SHAPE)).astype(np.float32)

    return _span(rho_lo, rho_hi), _span(u_lo, u_hi)


def main():
    perfect = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    real = fit_real_fluid(VanDerWaals(), (1.0, 150.0), (3.0e5, 5.0e5))
    print(f"grid {NI}^3 = {N:,} nodes, best of {REPEAT}")
    print(f"real fit: alpha {real._alpha.shape}, Sc {real._Sc.shape}")

    for name, fluid in (("PerfectFluid", perfect), ("RealFluid", real)):
        rho, u = make_state(fluid)
        outs = [np.zeros(SHAPE, np.float32, order="F") for _ in range(3)]

        fused = timeit(lambda: fluid.get_P_h_T(rho, u, *outs))
        singles = timeit(
            lambda: (
                fluid.get_P(rho, u, out=outs[0]),
                fluid.get_h(rho, u, out=outs[1]),
                fluid.get_T(rho, u, out=outs[2]),
            )
        )
        print(
            f"{name:13s} get_P_h_T {fused:8.2f} ms "
            f"({fused * 1e6 / N:6.2f} ns/node)   "
            f"P+h+T separately {singles:8.2f} ms  "
            f"(fused is {singles / fused:.2f}x)"
        )


if __name__ == "__main__":
    main()
