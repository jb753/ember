"""Subroutine-level golden-value tests for the two viscous body-force phases.

:meth:`ember.grid.Grid.update_sources` builds the viscous body force in two
Fortran passes (see :mod:`test_set_F_body_golden` for the composed force):

  * phase 1 -- ``ember.fortran.set_tau_q_soa`` fills the per-cell stress tensor
    ``tau_cell``, heat flux ``q_cell`` and mixing-length ``mu_turb``; then
  * phase 2 -- ``ember.fortran.set_visc_force`` turns tau/q into face fluxes and
    accumulates the (negated) viscous force into ``F_body_nd``.

``test_set_F_body_golden`` only locks the *composition* of these two passes with
the polar and SFD terms, so a change in one pass that is masked by the other --
or by the polar/SFD force -- would slip through, and a failure there cannot be
attributed to a single subroutine. These tests lock each pass independently:

  * phase 1 is called directly and its tau/q/mu_turb output is compared to a
    committed golden; and
  * phase 2 is fed a *synthetic, analytic* tau/q (NOT the phase-1 output), so a
    regression in ``set_tau_q_soa`` cannot cascade into the phase-2 golden --
    the two goldens fail independently.

The fixture is a deterministic single-block, theta-periodic, swirling and
sheared flow modelled on ``test_set_F_body_golden``, extended so both phase-1
outputs are meaningfully exercised: a radial/axial temperature gradient gives
q_cell a real signal, and the wall distance is tuned so mu_turb straddles the
mixing-length formula and its limiter. The i/j faces are walls, so the
wall-shear scaling and wall-function branches of phase 2 are exercised; the
block has no cusp patch, so that seam branch stays inactive here.

Regenerate the golden after an *intentional* change to either pass:

    uv run python tests/test_viscous_phases_golden.py
"""

from pathlib import Path

import numpy as np
import pytest

import ember.block
import ember.fortran
from ember import util
from ember.fluid import PerfectFluid
from ember.periodic import PeriodicPatch

GOLDEN_FILE = Path(__file__).parent / "data" / "viscous_phases_golden.npz"

# Fixture inputs held fixed so the goldens are reproducible.
SHAPE = (7, 9, 9)  # k (theta) has 8 cells = two wavelengths of the Vx pattern
NB = 36
PR_TURB = 1.0  # phase-1 turbulent Prandtl, fixed at 1.0 for the grid march


def _build_block():
    """Deterministic single-block periodic block with a swirling sheared flow.

    Modelled on :func:`test_set_F_body_golden._build_grid` but returned as a
    standalone block (both viscous passes are per-block Fortran calls needing no
    grid-level halo exchange; phase 2 here is fed a synthetic seam, below) and
    with a temperature gradient and wall distance chosen to exercise the phase-1
    heat-flux and mixing-length paths (see module docstring).
    """
    pitch = 2.0 * np.pi / NB

    block = ember.block.Block(shape=SHAPE)
    block.set_Nb(NB)
    xrt = util.linmesh3((0.0, 0.15), (0.5, 0.9), (0.0, pitch), SHAPE)
    block.set_x(xrt[..., 0])
    block.set_r(xrt[..., 1])
    block.set_t(xrt[..., 2])
    block.set_fluid(PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72))

    x, r, t = block.x, block.r, block.t
    r_span = float(r.max() - r.min())

    # Radial + axial temperature gradient (theta-uniform, so the seam stays
    # periodic) so the phase-1 heat flux q_cell carries a real signal rather than
    # the metric-closure noise a uniform T would leave.
    Temp = (
        300.0
        + 20.0 * (r - r.min()) / r_span
        + 8.0 * np.sin(2.0 * np.pi * x / float(x.max()))
    ).astype(np.float32)
    block.set_P_T(101325.0, Temp)

    # Smooth, exactly theta-periodic velocity field with axial/radial shear and
    # swirl, so the viscous velocity gradients are non-trivial in all directions.
    Vx = (
        100.0
        + 20.0 * np.sin(4.0 * np.pi * t / pitch + np.pi / 4.0)
        + 10.0 * (r - r.min()) / r_span
    ).astype(np.float32)
    Vr = (5.0 * np.cos(2.0 * np.pi * t / pitch)).astype(np.float32)
    Vt = (40.0 + 15.0 * np.sin(2.0 * np.pi * x / float(x.max()))).astype(np.float32)
    block.set_Vx(Vx)
    block.set_Vr(Vr)
    block.set_Vt(Vt)

    # Nonzero rotation so the rotating-frame logic (relative vorticity in phase 1,
    # viscous work term and wall slip in phase 2) is exercised.
    block.set_Omega(50.0)

    # Wall distance => mixing length (xlen_sq_nd derives from this). The 0.008
    # amplitude is chosen so mu_turb mostly sits on the rho*l^2*|omega| formula
    # (~70% of cells) while a minority saturate the visc_lim clamp, exercising
    # both branches of the phase-1 mixing-length min().
    wdist = 0.008 * (1.0 + np.sin(np.pi * (r - r.min()) / r_span))
    block.set_wdist(wdist.astype(np.float32))

    block.patches.append(PeriodicPatch(k=0))
    block.patches.append(PeriodicPatch(k=-1))
    return block


def _run_phase1():
    """Call ``set_tau_q_soa`` directly; return its tau/q/mu_turb output.

    The tau/q scratch buffer is zeroed first so the corner/edge halo slots the
    subroutine leaves untouched are a deterministic zero rather than stale
    scratch. ``mu_turb`` is captured on its written region only ([:-1, :-1, :-1]);
    the final node in each axis is padding the kernel never writes.
    """
    block = _build_block()

    halo = block.tau_q_halo
    halo.fill(0.0)
    tau_cell = halo[..., 0:6]
    q_cell = halo[..., 6:9]
    mu_turb = block._get_data_by_keys(
        ("mu_turb",), raise_uninit=False, writeable=True
    )

    ember.fortran.set_tau_q_soa(
        cons=block.conserved_nd,
        t=block.T_nd,
        mu=block.mu_nd,
        cp=block.cp_nd,
        pr_lam=block.fluid._Pr,
        pr_turb=PR_TURB,
        xlength=block.xlen_sq_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        r=block.r_nd,
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        tau_cell=tau_cell,
        q_cell=q_cell,
        mu_turb=mu_turb,
    )
    return {
        "tau_cell": np.array(tau_cell),
        "q_cell": np.array(q_cell),
        "mu_turb": np.array(mu_turb[:-1, :-1, :-1]),
    }


def _synthetic_tau_q(shape):
    """Deterministic, smooth analytic tau/q over the halo-padded cell grid.

    Independent of phase 1 by construction: phase 2's golden is locked against
    THIS field, not against ``set_tau_q_soa`` output, so the two passes fail
    independently. Smooth O(1) fields (rather than an RNG fill) keep every
    face-flux difference well-resolved and free of platform RNG dependence.
    Every slot ``set_visc_force`` reads -- owned cells and the single-sided
    boundary halos -- is filled.
    """
    ni1, nj1, nk1 = shape
    ii = np.linspace(0.0, 1.0, ni1)
    jj = np.linspace(0.0, 1.0, nj1)
    kk = np.linspace(0.0, 1.0, nk1)
    gi, gj, gk = np.meshgrid(ii, jj, kk, indexing="ij")

    tau = np.empty((ni1, nj1, nk1, 6), dtype=np.float32)
    for c in range(6):
        phase = 2.0 * np.pi * ((c + 1) * gi + (c + 2) * gj + (c + 3) * gk)
        tau[..., c] = (0.5 + 0.4 * np.sin(phase + 0.3 * c)).astype(np.float32)

    q = np.empty((ni1, nj1, nk1, 3), dtype=np.float32)
    for c in range(3):
        phase = 2.0 * np.pi * ((c + 2) * gi + (c + 1) * gj + (c + 3) * gk)
        q[..., c] = (0.2 * np.cos(phase + 0.5 * c)).astype(np.float32)
    return tau, q


def _run_phase2():
    """Call ``set_visc_force`` on a synthetic tau/q; return the fvisc output."""
    block = _build_block()

    halo = block.tau_q_halo
    tau, q = _synthetic_tau_q(halo.shape[:3])
    halo[..., 0:6] = tau
    halo[..., 6:9] = q
    tau_cell = halo[..., 0:6]
    q_cell = halo[..., 6:9]

    # F_body_nd is a read-only cached buffer; unlock and zero it as update_sources
    # does before accumulating the viscous force into components 1: (momenta+energy).
    fbody = block.F_body_nd
    fbody.flags.writeable = True
    fbody.fill(0.0)

    i_cusp_start, i_cusp_end = block.i_cusp
    ember.fortran.set_visc_force(
        cons=block.conserved_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        omega_block=block.Omega_nd,
        r=block.r_nd,
        mu=block.mu_nd,
        fvisc=fbody[..., 1:],
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        tau_cell=tau_cell,
        q_cell=q_cell,
        flow_scratch=block.scratch[..., 0:4],
        **block.ijk_wall_visc,
        **block.Omega_wall_nd,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
    )
    return np.array(fbody[..., 1:])


def _assert_matches_golden(actual, expected):
    """Compare with the same float32 tolerance policy as test_set_F_body_golden.

    rtol tolerates cross-platform float32 reduction order; atol floats at 1e-5 of
    the field magnitude so near-cancelling cells are not judged against a fixed
    floor while a real magnitude-scale regression is still caught.
    """
    assert actual.shape == expected.shape
    atol = 1e-5 * float(np.abs(expected).max())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=atol)


# ---- phase 1: set_tau_q_soa ------------------------------------------------


@pytest.mark.parametrize("field", ["tau_cell", "q_cell", "mu_turb"])
def test_set_tau_q_soa_matches_golden(field):
    if not GOLDEN_FILE.exists():
        pytest.skip(f"golden missing; regenerate with: uv run python {__file__}")
    out = _run_phase1()
    golden = np.load(GOLDEN_FILE)
    _assert_matches_golden(out[field], golden[field])


# ---- phase 2: set_visc_force -----------------------------------------------

# The cell array is (ni-1, nj-1, nk-1, 4). The seven regions tile the block into
# the interior and the six i/j/k hi/lo boundary faces (overlapping on
# edges/corners is fine -- each must match the golden). The boundary faces carry
# the wall-shear scaling and wall-function contributions of set_visc_force.
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
def test_set_visc_force_matches_golden(region):
    if not GOLDEN_FILE.exists():
        pytest.skip(f"golden missing; regenerate with: uv run python {__file__}")
    fvisc = _run_phase2()
    golden = np.load(GOLDEN_FILE)["fvisc"]
    sl = REGIONS[region]
    _assert_matches_golden(fvisc[sl], golden[sl])


if __name__ == "__main__":
    phase1 = _run_phase1()
    fvisc = _run_phase2()
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_FILE, fvisc=fvisc, **phase1)
    print(f"wrote {GOLDEN_FILE}")
    for name, arr in {**phase1, "fvisc": fvisc}.items():
        print(
            f"  {name:9s} shape={arr.shape}  "
            f"|.|_max={np.abs(arr).max():.6e}  sum={arr.sum():.6e}"
        )
