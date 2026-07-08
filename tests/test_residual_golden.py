"""Golden-value integration test for :meth:`ember.block.Block.residual_nd`.

Builds a fixed single-block grid, periodic in theta (the k direction), with a
deterministic swirling and sheared flow and a nonzero wall distance, assembles
the body force via ``grid.update_sources``, then computes the unintegrated net-flow
residual (inviscid face-flux balance + body force) via ``block.residual_nd`` and
compares it against a committed golden reference.

This locks the numerical output of the inviscid residual kernel (``set_residual``
in ``residual.f90`` and its face-flow helpers). ``test_known_good`` used to cover
the residual, but it drove removed Fortran entry points; this golden replaces it
for the production ``residual_nd`` path, so a silent change there would be caught.

Regenerate the golden after an *intentional* change to the residual:

    uv run python tests/test_residual_golden.py
"""

from pathlib import Path

import numpy as np
import pytest

import ember.block
import ember.grid
from ember import util
from ember.fluid import PerfectFluid
from ember.periodic import PeriodicPatch

GOLDEN_FILE = Path(__file__).parent / "data" / "residual_golden.npz"

# Assembly inputs held fixed so the golden is reproducible.
GAIN_FILT = 5.0
SHAPE = (7, 9, 9)  # k (theta) has 8 cells = two wavelengths of the Vx pattern
NB = 36


def _build_grid():
    """Deterministic single-block periodic grid with a swirling sheared flow."""
    pitch = 2.0 * np.pi / NB

    block = ember.block.Block(shape=SHAPE)
    block.set_Nb(NB)
    xrt = util.linmesh3((0.0, 0.15), (0.5, 0.9), (0.0, pitch), SHAPE)
    block.set_x(xrt[..., 0]).set_r(xrt[..., 1]).set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))
    block.set_P_T(101325.0, 300.0)

    # Smooth, exactly theta-periodic velocity field with axial/radial shear and
    # swirl, so the inviscid face flows (mass, momentum, energy) are non-trivial.
    x, r, t = block.x, block.r, block.t
    r_span = float(r.max() - r.min())
    Vx = (
        100.0
        + 20.0 * np.sin(4.0 * np.pi * t / pitch + np.pi / 4.0)
        + 10.0 * (r - r.min()) / r_span
    ).astype(np.float32)
    Vr = (5.0 * np.cos(2.0 * np.pi * t / pitch)).astype(np.float32)
    Vt = (40.0 + 15.0 * np.sin(2.0 * np.pi * x / float(x.max()))).astype(np.float32)
    block.set_Vx(Vx).set_Vr(Vr).set_Vt(Vt)

    # Nonzero rotation so the rotating-frame logic (relative tangential velocity
    # in the mass flux, rothalpy in the energy flux) is exercised.
    block.set_Omega(50.0)

    # Nonzero wall distance => nonzero mixing length, so the body force assembled
    # below (and folded into the residual) carries a real viscous contribution.
    wdist = 0.02 * (1.0 + np.sin(np.pi * (r - r.min()) / r_span))
    block.set_wdist(wdist.astype(np.float32))

    block.patches.append(PeriodicPatch(k=0))
    block.patches.append(PeriodicPatch(k=-1))

    grid = ember.grid.Grid([block])

    # Seed the SFD low-pass filter to a known 2% offset from the current cell
    # state, so the body force has a non-zero SFD term (matching the F_body_nd
    # golden). conserved_filt_nd is a read-only cached buffer, so unlock it.
    cons_filt = block.conserved_filt_nd
    cons_filt.flags.writeable = True
    cons_filt[...] = block.conserved_cell_nd * np.float32(1.02)
    cons_filt.flags.writeable = False

    return grid, block


def _assemble():
    """Build the grid, assemble F_body_nd, and return the residual."""
    grid, block = _build_grid()
    grid.update_sources(inviscid=False, gain_filt=GAIN_FILT)
    grid.update_residual()
    return np.array(block.residual_nd)


# The cell array is (ni-1, nj-1, nk-1, 5). The seven regions below tile the
# block into the interior and the six i/j/k hi/lo boundary faces (the boundary
# slices overlap on edges/corners, which is fine -- each must match the golden).
REGIONS = {
    "interior": (slice(1, -1), slice(1, -1), slice(1, -1)),
    "i_lo": (0, slice(None), slice(None)),
    "i_hi": (-1, slice(None), slice(None)),
    "j_lo": (slice(None), 0, slice(None)),
    "j_hi": (slice(None), -1, slice(None)),
    "k_lo": (slice(None), slice(None), 0),
    "k_hi": (slice(None), slice(None), -1),
}


@pytest.mark.parametrize("region", list(REGIONS))
def test_residual_matches_golden(region):
    if not GOLDEN_FILE.exists():
        pytest.skip(f"golden missing; regenerate with: uv run python {__file__}")
    residual = _assemble()
    golden = np.load(GOLDEN_FILE)["residual"]
    assert residual.shape == golden.shape

    sl = REGIONS[region]
    actual = residual[sl]
    expected = golden[sl]
    # rtol is generous enough for float32 cross-platform reduction order. atol is
    # scaled to each region's own field magnitude rather than a fixed floor, so
    # cells where the flux balance nearly cancels are governed by float32 noise
    # (~1e-5 of the field scale) instead of a runaway relative error.
    atol = 1e-5 * float(np.abs(expected).max())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=atol)


if __name__ == "__main__":
    residual = _assemble()
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_FILE, residual=residual)
    print(
        f"wrote {GOLDEN_FILE}\n"
        f"  shape={residual.shape}  |R|_max={np.abs(residual).max():.6e}  "
        f"sum={residual.sum():.6e}"
    )
