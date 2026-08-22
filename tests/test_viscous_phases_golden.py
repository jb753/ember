"""Subroutine-level golden-value tests for the two viscous body-force phases.

:meth:`ember.grid.Grid.update_sources` builds the viscous body force in two
Fortran passes (see :mod:`test_set_F_body_golden` for the composed force):

  * phase 1 -- ``ember.fortran.set_tau_q_faces`` fills the stress tensor and
    heat flux for the cells on the block's BOUNDARY SHELL, into the six
    surface buffers of ``Block.tau_q_faces``, layer 0 the block's own edge
    cells and layer 1 the halo value; then
  * phase 2 -- ``ember.fortran.set_visc_force`` produces INTERIOR tau/q inside
    its own k walk, turns tau/q into face fluxes, accumulates the viscous force
    into ``F_body_nd``, writes ``mu_turb``, and folds in the polar
    (radial-momentum) source in the same final pass (an optimisation: both
    write the same ``F_body_nd`` slots, so this saves a whole separate
    full-array touch -- see the kernel's header comment).

``test_set_F_body_golden`` only locks the *composition* of these passes with
the (inviscid-only, now) polar call and the SFD term, so a change masked by
another pass would slip through there, and a failure cannot be attributed to
a single subroutine. These tests lock each pass independently -- phase 2's
golden below is viscous+polar combined, not viscous alone, since that is now
what ``set_visc_force`` computes:

  * phase 1 is called directly and its six face buffers are compared to a
    committed golden; and
  * phase 2 is fed *synthetic, analytic* face buffers (NOT the phase-1 output),
    so a regression in ``set_tau_q_faces`` cannot cascade into the phase-2
    golden -- the two goldens fail independently. That independence is now
    structural as well as procedural: phase 2 derives every interior tau/q
    itself and reads phase 1 only at the shell.

The fixture is a deterministic single-block, theta-periodic, swirling and
sheared flow modelled on ``test_set_F_body_golden``, extended so both phases'
outputs are meaningfully exercised: a radial/axial temperature gradient gives
the heat flux a real signal, and the wall distance is tuned so mu_turb
straddles the mixing-length formula and its limiter. The i/j faces are walls,
so the wall-shear scaling and wall-function branches of phase 2 are exercised;
the block has no cusp patch, so that seam branch stays inactive here.

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


FACE_NAMES = ("f_i1", "f_ini", "f_j1", "f_jnj", "f_k1", "f_knk")


def _run_phase1(mu=None, kappa=None):
    """Call ``set_tau_q_faces`` directly; return its six face buffers.

    ``mu`` and ``kappa`` default to the block's own nodal fields; pass arrays
    to drive the kernel with transport that varies where this fixture's
    perfect gas leaves it constant.

    The buffers are poisoned first, so a slot the kernel fails to write shows
    up as a NaN here rather than as whatever the arena last held. Every slot of
    all six IS written: the producer covers the whole shell.
    """
    block = _build_block()

    faces = block.tau_q_faces
    for buf in faces:
        buf.fill(np.nan)

    ember.fortran.set_tau_q_faces(
        cons=block.conserved_nd,
        t=block.T_nd,
        mu=block.mu_nd if mu is None else mu,
        cp=block.cp_nd,
        kappa=block.kappa_nd if kappa is None else kappa,
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
        f_i1=faces[0],
        f_ini=faces[1],
        f_j1=faces[2],
        f_jnj=faces[3],
        f_k1=faces[4],
        f_knk=faces[5],
        **block.ijk_wall_visc,
    )
    return {name: np.array(buf) for name, buf in zip(FACE_NAMES, faces)}


def _synthetic_faces(block):
    """Deterministic, smooth analytic tau/q for the six face buffers.

    Independent of phase 1 by construction: phase 2's golden is locked against
    THIS shell, not against ``set_tau_q_faces`` output, so the two passes fail
    independently. Smooth O(1) fields (rather than an RNG fill) keep every
    face-flux difference well-resolved and free of platform RNG dependence.
    Both layers of every buffer are filled -- the edge cell and the halo value
    -- and they are given DIFFERENT fields, so a kernel that read the wrong
    layer would not pass.
    """
    out = []
    for buf in block.tau_q_faces:
        na, _, nb, _ = buf.shape
        ga, gc, gb = np.meshgrid(
            np.linspace(0.0, 1.0, na), np.arange(9.0), np.linspace(0.0, 1.0, nb),
            indexing="ij",
        )
        phase = 2.0 * np.pi * ((gc + 1.0) * ga + (gc + 2.0) * gb)
        layer0 = 0.5 + 0.4 * np.sin(phase + 0.3 * gc)
        layer1 = 0.2 * np.cos(phase + 0.5 * gc)
        out.append(np.stack([layer0, layer1], axis=-1).astype(np.float32))
    return out


def _run_phase2(jbw=0, mu=None, kappa=None):
    """Call ``set_visc_force`` on a synthetic shell; return fvisc and mu_turb.

    ``set_visc_force`` folds the polar (radial-momentum) source into its own
    final pass over ``fvisc`` (see the kernel's header comment), so this golden
    locks viscous+polar combined, not viscous alone. It also produces every
    interior tau/q itself and writes ``mu_turb``, so that field is part of
    THIS phase's golden and not phase 1's.

    ``jbw`` is the j-panel width; 0 mirrors production and sizes it from the
    kernel's own VISC_JAREA. ``mu``/``kappa`` override the block's own nodal
    transport, as in :func:`_run_phase1`.
    """
    block = _build_block()

    for buf, synth in zip(block.tau_q_faces, _synthetic_faces(block)):
        buf[...] = synth

    # F_body_nd is a read-only cached buffer; unlock and zero it as update_sources
    # does before accumulating the viscous force into components 1: (momenta+energy).
    fbody = block.F_body_nd
    fbody.flags.writeable = True
    fbody.fill(0.0)
    mu_turb = block._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True)

    i_cusp_start, i_cusp_end = block.i_cusp
    # One carve for the whole viscous phase: every buffer below reaches this
    # one call, so carving them together is what makes them disjoint.
    faces, tq, planes, rows, transport = ember.block._carve_viscous(block)
    ember.fortran.set_visc_force(
        cons=block.conserved_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        omega_block=block.Omega_nd,
        r=block.r_nd,
        mu=block.mu_nd if mu is None else mu,
        p=block.P_nd,
        p_offset=block.P_offset_nd,
        fvisc=fbody[..., 1:],
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        t=block.T_nd,
        cp=block.cp_nd,
        kappa=block.kappa_nd if kappa is None else kappa,
        pr_turb=PR_TURB,
        xlength=block.xlen_sq_nd,
        mu_turb=mu_turb,
        f_i1=faces[0],
        f_ini=faces[1],
        f_j1=faces[2],
        f_jnj=faces[3],
        f_k1=faces[4],
        f_knk=faces[5],
        tq=tq,
        planes=planes,
        rows=rows,
        **block.ijk_wall_visc,
        **block.Omega_wall_nd,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
        jbw_in=jbw,
    )
    # mu_turb's final node in each axis is padding the kernel never writes.
    return np.array(fbody[..., 1:]), np.array(mu_turb[:-1, :-1, :-1])


def _assert_matches_golden(actual, expected):
    """Compare with the same float32 tolerance policy as test_set_F_body_golden.

    rtol tolerates cross-platform float32 reduction order; atol floats at 1e-5 of
    the field magnitude so near-cancelling cells are not judged against a fixed
    floor while a real magnitude-scale regression is still caught.
    """
    assert actual.shape == expected.shape
    atol = 1e-5 * float(np.abs(expected).max())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=atol)


# ---- phase 1: set_tau_q_faces ----------------------------------------------


@pytest.mark.parametrize("field", FACE_NAMES)
def test_set_tau_q_faces_matches_golden(field):
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
    fvisc, _ = _run_phase2()
    golden = np.load(GOLDEN_FILE)["fvisc"]
    sl = REGIONS[region]
    _assert_matches_golden(fvisc[sl], golden[sl])


def test_set_visc_force_mu_turb_matches_golden():
    """The mixing-length viscosity is phase 2's output now, not phase 1's.

    It is the one thing the fused kernel produces that is not a force, so a
    producer that drifted while still giving the right fvisc -- a plausible
    failure, since fvisc is a difference of large face flows and mu_turb is
    not -- would show up here and nowhere else in this file.
    """
    if not GOLDEN_FILE.exists():
        pytest.skip(f"golden missing; regenerate with: uv run python {__file__}")
    _, mu_turb = _run_phase2()
    _assert_matches_golden(mu_turb, np.load(GOLDEN_FILE)["mu_turb"])


@pytest.mark.parametrize("jbw", [4, 5, 8])
def test_set_visc_force_panel_consistent(jbw):
    """The j-panel width must not change the result.

    Compare each panel width against the single-panel jbw = nj-1 reference,
    which degenerates to an unpanelled walk. The per-cell arithmetic is
    identical for every width -- a panel only changes which rows are live at
    once -- so the comparison is exact. What it catches is panel bookkeeping:
    a panel duplicates its lowest j-face row and its producer rows at jp0-1
    and jp1+1, and getting those bounds wrong leaves a seam one row wide that
    no other test in this file would separate from ordinary reassociation.
    jbw = 5 exercises a short last panel (nj-1 = 8 = 5 + 3).

    This is the successor to the old k-slab consistency test: the fused walk
    has no k slab (each tau/q plane is consumed the moment it is produced, so
    a single walk over k IS the blocked schedule), and the j panel is the
    blocking that remains.
    """
    ref_f, ref_m = _run_phase2(jbw=SHAPE[1] - 1)
    out_f, out_m = _run_phase2(jbw=jbw)
    np.testing.assert_array_equal(out_f, ref_f)
    np.testing.assert_array_equal(out_m, ref_m)


def _corner(a):
    """Nodal array averaged to cells, as the kernel's stage-1 row temps do."""
    a = np.asarray(a, dtype=np.float64)
    return 0.125 * (
        a[:-1, :-1, :-1]
        + a[1:, :-1, :-1]
        + a[:-1, 1:, :-1]
        + a[1:, 1:, :-1]
        + a[:-1, :-1, 1:]
        + a[1:, :-1, 1:]
        + a[:-1, 1:, 1:]
        + a[1:, 1:, 1:]
    )


def test_phase1_reads_the_transport_fields_cell_by_cell():
    """Shell tau follows the local viscosity and q the local conductivity.

    Every other test in this file hands the kernel a perfect gas, whose
    transport is one number repeated over the whole field -- which a kernel
    that ignored the fields entirely would reproduce exactly. This one hands
    it fields that vary, and checks the consequence in closed form: the
    stress is linear in the cell's own corner-averaged viscosity and the heat
    flux in its conductivity, so each cell's ratio to the constant-transport
    run is known without re-deriving the kernel's arithmetic.

    The two fields vary along different axes, so a kernel that crossed them
    would not pass either.

    Both of the producer's shapes are checked: the k1 face, which the vectorized
    ROW body writes, and the i1 face, which pins that axis and falls back to
    the per-cell ``tau_q_at_cell``. They are the same arithmetic twice over,
    and this is the test that would notice if one of them stopped being.

    The mixing-length viscosity the ratio needs comes from phase 2 (it is that
    kernel's output now), evaluated on the same state and the same transport.
    """
    block = _build_block()
    ni, nj, nk = SHAPE
    mu0 = np.asarray(block.mu_nd, dtype=np.float64)
    ka0 = np.asarray(block.kappa_nd, dtype=np.float64)
    ramp_i = (np.arange(ni) / (ni - 1.0))[:, None, None]
    ramp_j = (np.arange(nj) / (nj - 1.0))[None, :, None]
    mu_f = np.asfortranarray(mu0 * (1.0 + 0.4 * ramp_i), dtype=np.float32)
    ka_f = np.asfortranarray(ka0 * (1.0 + 0.4 * ramp_j), dtype=np.float32)

    base, got = _run_phase1(), _run_phase1(mu=mu_f, kappa=ka_f)
    _, mut_a = _run_phase2()
    _, mut_b = _run_phase2(mu=mu_f, kappa=ka_f)

    cpc = _corner(block.cp_nd)
    fac_a, fac_b = _corner(mu0) + mut_a, _corner(mu_f) + mut_b
    lam_a = _corner(ka0) + mut_a * cpc / PR_TURB
    lam_b = _corner(ka_f) + mut_b * cpc / PR_TURB

    # Each face buffer is (a, 9, b, 2) over the cells of one boundary face;
    # take the same cells out of the cell-shaped ratios, with the component
    # axis moved into place.
    # knk is written by the vectorized ROW body, i1 by the per-cell fallback.
    # Both are chosen for carrying cells where the mixing-length CLAMP binds
    # (60% and 28% of them here): where it does not, mut is independent of the
    # laminar viscosity and swamps it, so the predicted ratio is 1 and the
    # comparison, while still correct, says nothing. k1 has no clamped cell at
    # all on this fixture, which is why it is not one of the two.
    faces = {"f_knk": np.s_[:, :, -1], "f_i1": np.s_[0, :, :]}
    for name, cells in faces.items():
        tau_ratio = np.moveaxis(np.broadcast_to(
            (fac_b / fac_a)[cells][..., None], (fac_a[cells].shape + (6,))), -1, 1)
        q_ratio = np.moveaxis(np.broadcast_to(
            (lam_b / lam_a)[cells][..., None], (lam_a[cells].shape + (3,))), -1, 1)
        ratio = np.concatenate([tau_ratio, q_ratio], axis=1)
        # Layer 0, the block's own edge cell. Layer 1 is that times a mask the
        # transport cannot reach, so it carries no independent information.
        ref = np.asarray(base[name], dtype=np.float64)[..., 0]
        expected = ref * ratio
        actual = np.asarray(got[name], dtype=np.float64)[..., 0]
        atol = 1e-5 * float(np.abs(expected).max())
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=atol,
                                   err_msg=f"{name} does not scale with transport")
        # Not vacuous: varying the transport has to MOVE the output by much
        # more than the tolerance the comparison allows, or a kernel ignoring
        # the fields entirely would pass. It moves by a few percent here rather
        # than the ramp's 40%, because the clamped mixing-length viscosity
        # swamps the laminar one over most of the shell -- so this is asserted
        # against the tolerance rather than against the ramp.
        assert np.abs(actual - ref).max() > 100.0 * atol


if __name__ == "__main__":
    phase1 = _run_phase1()
    fvisc, mu_turb = _run_phase2()
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_FILE, fvisc=fvisc, mu_turb=mu_turb, **phase1)
    print(f"wrote {GOLDEN_FILE}")
    for name, arr in {**phase1, "fvisc": fvisc, "mu_turb": mu_turb}.items():
        print(
            f"  {name:9s} shape={arr.shape}  "
            f"|.|_max={np.abs(arr).max():.6e}  sum={arr.sum():.6e}"
        )
