"""Golden-value integration test for :meth:`ember.grid.Grid.update_sources`.

Builds a fixed single-block grid, periodic in theta (the k direction), with a
deterministic swirling and sheared flow and a nonzero wall distance (so the
turbulent mixing length is active). It then assembles the full body force --
viscous shear stresses, the polar source, and the selective-frequency-damping
(SFD) force -- via ``grid.update_sources`` and compares ``block.F_body_nd`` against a
committed golden reference.

This locks the numerical output of the viscous/polar/SFD body-force path. The
structural test :mod:`test_viscous_periodic` only checks seam transparency, so
without this test a silent change in the assembled body force would go undetected.

Regenerate the golden after an *intentional* change to the body force:

    uv run python tests/test_set_F_body_golden.py
"""

from pathlib import Path

import numpy as np
import pytest

import ember.block
import ember.grid
from ember import util
from ember.fluid import PerfectFluid
from ember.periodic import PeriodicPatch

GOLDEN_FILE = Path(__file__).parent / "data" / "F_body_golden.npz"

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
    # swirl, so the viscous (gradient) and polar (rho*Vt^2 + dP) terms are both
    # non-trivial.
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

    # Nonzero rotation so the rotating-frame logic (relative vorticity, viscous
    # work term, wall slip) is actually exercised; Omega*r ~ 35 m/s is a
    # meaningful fraction of the swirl, so absolute and relative differ.
    block.set_Omega(50.0)

    # Nonzero wall distance => nonzero mixing length => turbulent viscosity is
    # exercised (xlen_sq_nd derives from this).
    wdist = 0.02 * (1.0 + np.sin(np.pi * (r - r.min()) / r_span))
    block.set_wdist(wdist.astype(np.float32))

    block.patches.append(PeriodicPatch(k=0))
    block.patches.append(PeriodicPatch(k=-1))

    grid = ember.grid.Grid([block])

    # Seed the SFD low-pass filter to a known 2% offset from the current cell
    # state, so the SFD force term is non-zero (it is ~0 right after the filter
    # is seeded to the current state on first access). conserved_filt_nd is a
    # read-only cached buffer, so unlock it for this seeding write.
    cons_filt = block.conserved_filt_nd
    cons_filt.flags.writeable = True
    cons_filt[...] = block.conserved_cell_nd * np.float32(1.02)
    cons_filt.flags.writeable = False

    return grid, block


def _assemble():
    """Build the grid and return a copy of the assembled F_body."""
    grid, block = _build_grid()
    grid.update_sources(inviscid=False, gain_filt=GAIN_FILT)
    return np.array(block.F_body_nd)  # F_body_nd is read-only after update_sources


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
def test_set_F_body_matches_golden(region):
    if not GOLDEN_FILE.exists():
        pytest.skip(f"golden missing; regenerate with: uv run python {__file__}")
    F_body = _assemble()
    golden = np.load(GOLDEN_FILE)["F_body"]
    assert F_body.shape == golden.shape

    sl = REGIONS[region]
    actual = F_body[sl]
    expected = golden[sl]
    # rtol is generous enough for float32 cross-platform reduction order. atol is
    # scaled to each region's own field magnitude rather than a fixed floor: a few
    # energy cells nearly cancel (|F| ~ 1e-3 against region maxima of ~3-9), so
    # their relative error is dominated by float32 noise at the ~1e-5 absolute
    # level. Floating atol at 1e-5 of the region max absorbs that (~5 orders below
    # the field scale) without masking any real, magnitude-scale regression.
    atol = 1e-5 * float(np.abs(expected).max())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=atol)


if __name__ == "__main__":
    F_body = _assemble()
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_FILE, F_body=F_body)
    print(
        f"wrote {GOLDEN_FILE}\n"
        f"  shape={F_body.shape}  |F|_max={np.abs(F_body).max():.6e}  "
        f"sum={F_body.sum():.6e}"
    )
