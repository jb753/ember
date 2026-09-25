"""Golden-value test for a short march with the polar source and SFD active.

The kernel goldens (``test_set_F_body_golden``, ``test_residual_golden``) lock
one assembly of the body force; they cannot see how the solver sequences it
across steps. This test marches a small swirling duct for a few steps with a
filter seeded away from the flow and compares the final conserved state and
filter state against a committed golden, for both marching schemes and both
viscous settings. What it locks, that nothing else does:

  * the SFD force reads the filter BEFORE that step's filter update, and the
    update uses that step's freshly blended ``dt_vol``;
  * the filter advances every step, while the body force (polar + SFD) is
    rebuilt only on the scree march's every-fifth-step source cadence;
  * the polar source reaches the march on the inviscid path as well as the
    viscous one.

``N_STEP = 6`` spans one scree refresh at step 5, so the lagged body force on
steps 1-4 and its rebuild are both inside the window.

Regenerate the golden after an *intentional* change to the march:

    uv run python tests/test_sfd_march_golden.py
"""

from pathlib import Path

import numpy as np
import pytest

import ember.solver
from conftest import cell_conserved
from ember.cases import build_duct_grid

GOLDEN_FILE = Path(__file__).parent / "data" / "sfd_march_golden.npz"

# Small enough to march in well under a second, per case.
NCELL = 8000
NJ = NK = 17
N_STEP = 6

# The filter time constant is about six local timesteps (dt ~ 0.035 at this
# cfl), so it moves ~15% of the way to the flow per step, and the gain is
# comparable -- both large enough that getting their order or cadence wrong
# moves the answer far outside the golden's float32 tolerance.
CFL = 2.0
DELTA_FILT = 0.2
GAIN_FILT = 5.0
OFFSET_FRAC = 0.02

CASES = [(n_stage, inviscid) for n_stage in (0, 4) for inviscid in (True, False)]


def _key(n_stage, inviscid):
    return f"stage{n_stage}_{'inv' if inviscid else 'visc'}"


def _grid():
    """Duct with a smooth swirl, so the polar source's rho*Vt^2 term is live."""
    grid = build_duct_grid(NCELL, nj=NJ, nk=NK)
    block = grid[0]
    r = block.r
    r_span = float(r.max() - r.min())
    Vt = 0.1 * block.Vx * np.sin(np.pi * (r - r.min()) / r_span)
    block.set_Vt(Vt.astype(np.float32))

    # Filter seeded off the flow on every cell and equation (one scalar offset,
    # as test_sfd does, so the near-zero radial momentum is offset too).
    cons_cell = cell_conserved(block)
    offset = np.float32(OFFSET_FRAC * np.sqrt(np.mean(cons_cell**2)))
    cons_filt = block.conserved_filt_nd
    cons_filt.flags.writeable = True
    cons_filt[...] = cons_cell + offset
    cons_filt.flags.writeable = False
    return grid


def _march(n_stage, inviscid):
    """March the case and return copies of the final cons and filter state."""
    grid = _grid()
    hist = ember.solver.Solver(
        n_step=N_STEP,
        n_step_log=N_STEP,
        cfl=CFL,
        n_stage=n_stage,
        n_levels=0,
        fac_mgrid=0.0,
        inviscid=inviscid,
        gain_filt=GAIN_FILT,
        delta_filt=DELTA_FILT,
    ).run(grid)
    assert not hist.diverged
    block = grid[0]
    return np.array(block.conserved_nd), np.array(block.conserved_filt_nd)


def _assert_close(actual, expected):
    """float32 tolerance, atol floated per conserved component.

    The components differ by orders of magnitude (rho*r*Vt against rho*e), so
    one field-wide atol would be meaningless for the small ones. The atol is
    ten times the kernel goldens', because round-off compounds over the 24
    stages of a Runge-Kutta march: at theirs, macOS arm64 misses by 10% on a
    few near-zero cells. A 1% change in the SFD gain still misses this by 30x.
    """
    assert actual.shape == expected.shape
    for c in range(expected.shape[-1]):
        atol = 1e-4 * float(np.abs(expected[..., c]).max())
        np.testing.assert_allclose(
            actual[..., c], expected[..., c], rtol=1e-4, atol=atol
        )


@pytest.mark.parametrize("n_stage, inviscid", CASES)
def test_sfd_march_matches_golden(n_stage, inviscid):
    if not GOLDEN_FILE.exists():
        pytest.skip(f"golden missing; regenerate with: uv run python {__file__}")
    cons, filt = _march(n_stage, inviscid)
    golden = np.load(GOLDEN_FILE)
    key = _key(n_stage, inviscid)
    _assert_close(cons, golden[f"{key}_cons"])
    _assert_close(filt, golden[f"{key}_filt"])


if __name__ == "__main__":
    out = {}
    for n_stage, inviscid in CASES:
        key = _key(n_stage, inviscid)
        out[f"{key}_cons"], out[f"{key}_filt"] = _march(n_stage, inviscid)
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_FILE, **out)
    print(f"wrote {GOLDEN_FILE}")
    for key, arr in out.items():
        print(f"  {key}: shape={arr.shape}  sum={arr.sum():.6e}")
