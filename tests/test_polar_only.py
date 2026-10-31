"""Tests that set_polar_source matches the polar half of set_timestep_sources.

The polar source exists in two kernels: fused into the timestep walk, which
adds it on a source refresh step, and alone in ``set_polar_source``, which a
refresh step off the ``Solver.n_step_dt`` cadence calls instead. The two must
agree, or the source would change with the cadence. The arithmetic itself is
pinned by ``tests/test_polar_source.py`` against the fused kernel.

Test cases:
- test_matches_fused_kernel: same body force from both kernels on a random,
  nonuniform state with swirl
- test_only_radial_component_touched: the other four components are left as
  they were
"""

import numpy as np

from ember import fortran

typ = np.float32
NI, NJ, NK = 13, 9, 7


def _fort(x):
    return np.asfortranarray(x, dtype=typ)


def _state(seed=0):
    rng = np.random.default_rng(seed)
    cons = np.empty((NI, NJ, NK, 5))
    cons[..., 0] = rng.uniform(0.8, 1.4, (NI, NJ, NK))
    cons[..., 1] = rng.uniform(50.0, 150.0, (NI, NJ, NK))
    cons[..., 2] = rng.uniform(-20.0, 20.0, (NI, NJ, NK))
    r = rng.uniform(0.3, 0.6, (NI, NJ, NK))
    cons[..., 3] = cons[..., 0] * r * rng.uniform(-80.0, 80.0, (NI, NJ, NK))
    cons[..., 4] = rng.uniform(2.0e5, 3.0e5, (NI, NJ, NK))
    P = rng.uniform(0.9e5, 1.1e5, (NI, NJ, NK))
    vol = rng.uniform(1.0e-6, 2.0e-6, (NI - 1, NJ - 1, NK - 1))
    f_body = rng.uniform(-1.0, 1.0, (NI - 1, NJ - 1, NK - 1, 5))
    return _fort(cons), _fort(r), _fort(P), _fort(vol), _fort(f_body)


def _fused(cons, r, P, vol, f_body, P_offset):
    fortran.set_timestep_sources(
        dt_vol=_fort(np.ones((NI - 1, NJ - 1, NK - 1))),
        a=_fort(np.full((NI, NJ, NK), 340.0)),
        cons=cons,
        r=r,
        omega=0.0,
        dai=_fort(np.ones((3, NI, NJ - 1, NK - 1))),
        daj=_fort(np.ones((3, NI - 1, NJ, NK - 1))),
        dak=_fort(np.ones((3, NI - 1, NJ - 1, NK))),
        mu_turb=_fort(np.zeros((NI, NJ, NK))),
        vol=vol,
        rf=0.2,
        fac_visc=1.0,
        p=P,
        p_offset=P_offset,
        f_body=f_body,
        cons_filt=_fort(np.zeros((0, 0, 0, 5))),
        add_sources=1,
        gain_filt=0.0,
        cfl=0.0,
        delta_filt=1.0,
    )


def _alone(cons, r, P, vol, f_body, P_offset):
    fortran.set_polar_source(
        cons=cons, r=r, p=P, p_offset=P_offset, vol=vol, f_body=f_body
    )


def test_matches_fused_kernel():
    cons, r, P, vol, f_body = _state()
    f_fused = f_body.copy(order="F")
    f_alone = f_body.copy(order="F")
    _fused(cons, r, P, vol, f_fused, 1.0e5)
    _alone(cons, r, P, vol, f_alone, 1.0e5)
    # The source moved f_body, so agreement is not just two untouched copies.
    assert not np.allclose(f_alone, f_body)
    # -Ofast may contract the two kernels' arithmetic differently, so allow
    # float32 rounding rather than demanding bit equality.
    np.testing.assert_allclose(f_alone, f_fused, rtol=1e-5, atol=1e-7)


def test_only_radial_component_touched():
    cons, r, P, vol, f_body = _state(seed=1)
    f_alone = f_body.copy(order="F")
    _alone(cons, r, P, vol, f_alone, 0.0)
    for comp in (0, 1, 3, 4):
        np.testing.assert_array_equal(f_alone[..., comp], f_body[..., comp])
