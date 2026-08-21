"""Arm definitions for the viscous study: `set_visc_force` and `set_tau_q_soa`.

The viscous kernels (src/ember/_fortran/viscous.f90) carry a long adopted
history in bench/README.md -- k-slab blocking, rolling-buffer fusion -- but
none of it was ever measured on THIS harness: every tracked result in
bench/results/ is a `set_residual` or IRS number. This module is the viscous
counterpart of `residual_arms.py` and `irs_arms.py`, so
`bench_prod_baseline.py --kernel visc` (or `--kernel tauq`) can time them under
the protocol the README mandates: one arm per process, rank-barriered before
every call, replicated at the launch.

Two kernels, timed separately because they are separate calls with a grid-wide
periodic halo exchange between them (grid.py's update_sources):

  tauq    set_tau_q_soa    -- phase 1, cell stress tensor and heat flux
  visc    set_visc_force   -- phase 2, face fluxes accumulated into fvisc

The `visc` arm set is where the fvisc-fusion study lives: production touches
fvisc four times (the i-sweep assigns it, the j- and k-sweeps read-modify-write
it, the trailing polar-source pass RMWs component 2), and the rolling buffers
needed to collapse that to a single store already exist in the kernel.

WHY REPEATED CALLS NEED A RESTORE HOOK. Unlike `set_residual`, `set_visc_force`
is NOT idempotent: its entry pass scales the tau/q halo slots by (2*wall - 1)
IN PLACE, so rep n reads rep n-1's halos. On the duct case this happens to be
harmless twice over, and both reasons are accidents of the case rather than
properties of the kernel:

  - block.ijk_wall_visc casts a boolean, so the mask is exactly 0.0 or 1.0 and
    the factor is exactly -1 or +1: reps flip a sign, they do not decay. A
    blended mask (0 < wall < 1) would contract the halos geometrically toward
    subnormal, which IS a timing bias -- the hazard irs_arms.check_denormals
    exists for.
  - the flip is invisible in fvisc anyway, because the halo tau/q only reaches
    the boundary faces of the wall-adjacent cells, and those cells are
    multiplied by the same zero mask at the end of the kernel. Verified: four
    consecutive calls with no restore are bitwise identical here.

So `halo_restorer` is insurance, not a repair. It snapshots the six halo faces
(O(surface), not O(volume): ~0.4% of the buffer at 1M cells, microseconds
against a call of tens of milliseconds) and restores them before each call, so
every rep does bitwise-identical work on bitwise-identical input regardless of
which case is being run. The driver calls it OUTSIDE the timed window and
before the barrier.

fvisc itself needs no restore: the i-direction sweep assigns every cell before
anything reads it, so each call re-initialises its own output.
"""

import numpy as np

from ember import util
from ember.grid import _KB_SLAB

from residual_arms import build_case, swirl  # noqa: F401  (re-export)

VISC_ARMS = ("visc", "viscij", "viscijk", "viscpol", "viscpol2")
TAUQ_ARMS = ("tauq", "tauq_tau", "tauq_q")

ENTRY = {
    "visc": "set_visc_force",
    # The fvisc-fusion ladder (bench/subroutines/viscous_fused.f90). Each arm
    # removes one more visit to fvisc; all three sum each cell's contributions
    # in production's order and so should gate bitwise.
    "viscij": "set_visc_force_ij",  # i fused into j:   4 touches -> 3
    "viscijk": "set_visc_force_ijk",  # k fused in too:   4 touches -> 2
    "viscpol": "set_visc_force_pol",  # polar fused too:  4 touches -> 1
    # viscpol with the i=1/i=ni-1 sheet moved out of the O(surface)
    # pass and into the fused store, where it is unit-stride.
    "viscpol2": "set_visc_force_pol2",
    "tauq": "set_tau_q_soa",
    # Timing controls, wrong by construction: set_tau_q_soa split by
    # consumer, to price a tau/q split. See viscous_tauq_split.f90.
    "tauq_tau": "set_tau_q_tau_only",
    "tauq_q": "set_tau_q_q_only",
}


def seed_tau_q(grid, b):
    """Run phase 1 so tau/q hold a real field for phase 2, once, untimed.

    Rule 3: a real port evaluates set_tau_q_soa immediately before
    set_visc_force every step, so tau/q are emphatically not a hoistable
    input -- but they are the PREVIOUS kernel's output, not this one's, and
    pricing them inside the set_visc_force window would time set_tau_q_soa.
    Built once here, exactly as grid.py's update_sources builds them, including
    the periodic seam exchange that runs between the two phases.

    Must be called after swirl(): tau/q are a function of the state, so seeding
    them before the state is perturbed would leave phase 2 reading a tau/q
    field that does not belong to the velocities it also reads.
    """
    import ember.fortran as F

    halo = b.tau_q_halo
    F.set_tau_q_soa(
        cons=b.conserved_nd,
        t=b.T_nd,
        mu=b.mu_nd,
        cp=b.cp_nd,
        kappa=b.kappa_nd,
        pr_turb=1.0,
        xlength=b.xlen_sq_nd,
        vol=b.vol_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        r=b.r_nd,
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_rel_nd,
        tau_cell=halo[..., 0:6],
        q_cell=halo[..., 6:9],
        mu_turb=b._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True),
    )
    b._versions["mu_turb"] += 1
    grid.connectivity.periodic.exchange_halos()
    return halo


def halo_restorer(b):
    """Snapshot the tau/q halo faces; return a zero-argument restore callable.

    See the module docstring for why this exists. Only the six outer faces of
    the (ni+1, nj+1, nk+1, 9) buffer are ever touched by set_visc_force's entry
    scaling, so this is O(surface): at 1M cells it is ~0.4% of the buffer and
    a few microseconds against a call of tens of milliseconds.
    """
    halo = b.tau_q_halo
    ni, nj, nk = b.shape
    faces = (
        (np.s_[0, :, :, :], None),
        (np.s_[ni, :, :, :], None),
        (np.s_[:, 0, :, :], None),
        (np.s_[:, nj, :, :], None),
        (np.s_[:, :, 0, :], None),
        (np.s_[:, :, nk, :], None),
    )
    saved = [(sl, np.array(halo[sl], copy=True)) for sl, _ in faces]

    def restore():
        for sl, val in saved:
            halo[sl] = val

    return restore


def callers_visc(b, active_arms=VISC_ARMS):
    """One zero-argument callable per active set_visc_force arm."""
    import ember.fortran as F

    ni, nj, nk = b.shape
    halo = b.tau_q_halo
    i_cusp_start, i_cusp_end = b.i_cusp
    # Rolling face-flow buffers carved from block.scratch exactly as grid.py
    # does -- NOT from tau_q_halo, which is this kernel's tau/q input and
    # whose docstring forbids aliasing a second array into the same call.
    planes, rows = util.carve_view(b.scratch, (ni, nj, 4, 2), (ni, 4, 3))
    kw = dict(
        cons=b.conserved_nd,
        cons_cell=b.conserved_cell_nd,
        vol=b.vol_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        omega_block=b.Omega_nd,
        r=b.r_nd,
        mu=b.mu_nd,
        p=b.P_nd,
        p_offset=b.P_offset_nd,
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_rel_nd,
        tau_cell=halo[..., 0:6],
        q_cell=halo[..., 6:9],
        planes=planes,
        rows=rows,
        kb=min(_KB_SLAB, nk - 1),
        **b.ijk_wall_visc,
        **b.Omega_wall_nd,
        i_cusp_start=i_cusp_start,
        i_cusp_end=i_cusp_end,
    )
    # F_body_nd is served read-only; the kernel's fvisc dummy is its
    # components 2-5 (mass excluded), a contiguous trailing slice.
    b.F_body_nd.flags.writeable = True
    fvisc = b.F_body_nd[..., 1:]

    out = {}
    for name in active_arms:
        fn = getattr(F, ENTRY[name], None)
        if fn is None:
            continue
        out[name] = lambda fn=fn: fn(fvisc=fvisc, **kw)
    return out


def callers_tauq(b, active_arms=TAUQ_ARMS):
    """One zero-argument callable per active set_tau_q_soa arm.

    Phase 1 is idempotent (every owned cell and every halo slot it reads is
    assigned before it is read), so it needs no restore hook.
    """
    import ember.fortran as F

    halo = b.tau_q_halo
    mu_turb = b._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True)
    kw = dict(
        cons=b.conserved_nd,
        t=b.T_nd,
        mu=b.mu_nd,
        cp=b.cp_nd,
        kappa=b.kappa_nd,
        pr_turb=1.0,
        xlength=b.xlen_sq_nd,
        vol=b.vol_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        r=b.r_nd,
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_rel_nd,
        tau_cell=halo[..., 0:6],
        q_cell=halo[..., 6:9],
        mu_turb=mu_turb,
    )
    out = {}
    for name in active_arms:
        fn = getattr(F, ENTRY[name], None)
        if fn is None:
            continue
        out[name] = lambda fn=fn: fn(**kw)
    return out


def selfk_ineligible(grid, b):
    """Why this block cannot take the seam-free k path, or None if it can.

    The seam-free arm replaces the exchanged k halo with a read of the block's
    own far cell plane, selected per (i,j) against wallk1/wallnk. That is only
    equivalent to the exchange under three conditions, checked here once at
    setup (all O(surface), none of them in the timed window).

    THE SENTINEL is block.i_perk: it is (0, 0) exactly when the block has no
    k-face PeriodicPatch, so it is the switch between the general (exchange)
    path and this one. It reads (ni, 1) for a full-span seam and (i_LE, i_TE)
    for the H-mesh shape.

    THE MASK IS NOT THE PREDICATE. block.ijk_wall_visc means "not a viscous
    wall", not "periodic": it is 1.0 for every PERMEABLE_TYPES patch and, via
    SLIP_TYPES (patch.py:200), for InviscidPatch too. Two ways that bites on
    real geometry -- a slip endwall on a k face, and CuspPatch, which is in
    PERMEABLE_TYPES and must sit on a constant-k face (cusp.py:25), so an
    H-mesh trailing edge reads non-wall while being neither wall nor periodic.
    Nothing exchanges those halos, so a kernel trusting the mask alone would
    read the far plane there and be silently wrong. Hence the coverage check:
    the non-wall part of each k face must be EXACTLY the periodic part.

    Self-pairing is checked because a pitch split into two blocks would pair
    A's k=0 to B's k=nk -- periodic.check_match accepts that -- and the far
    plane the kernel reads would then belong to the wrong block.
    """
    if b.i_perk == (0, 0):
        return "i_perk == (0, 0): no k-face PeriodicPatch, use the general path"

    for (bid, pid), ((nxbid, _), _) in grid.connectivity.periodic.pair().items():
        if grid[bid].patches[pid].const_dim != 2:
            continue  # i/j periodicity is unrelated to the k seam
        if bid != nxbid:
            return (
                f"k-face periodic patch {(bid, pid)} pairs to block {nxbid}, "
                "not to itself: the far k plane is another block's"
            )

    # Rebuild the k-face indicator from k-face PeriodicPatches alone, exactly
    # as Block._get_face_wall_arrays builds the real one (block.py:912), and
    # require the non-wall part of the mask to be precisely that.
    kperi = np.zeros(b.shape_kface, dtype=np.uint8)
    for patch in b.patches.periodic:
        if patch.const_dim == 2:
            kperi[*patch.get_ijk_face().T] += 1
    for name, kslot in (("wallk1", 0), ("wallnk", -1)):
        nonwall = np.asarray(b.ijk_wall_visc[name])[:, :, 0] > 0
        covered = kperi[:, :, kslot] > 0
        if not np.array_equal(nonwall, covered):
            n = int(np.count_nonzero(nonwall & ~covered))
            return (
                f"{name} is non-wall at {n} face cells that no k-face "
                "PeriodicPatch covers (slip patch, cusp, or another permeable "
                "type): their halo is never exchanged"
            )
    return None


def callers_pair(grid, b):
    """Time the viscous PAIR: tau/q then face fluxes, unfused vs fused.

    The saving spans both kernels -- tau/q stops round-tripping through memory
    entirely -- so neither can be timed alone, exactly as the IRS fused-limiter
    study had to time set_residual+IRS together (irs_arms.callers_update).

      unfused      set_tau_q_soa, exchange_halos, set_visc_force -- production
      fused        exchange_halos, set_visc_force_tqf -- tau/q produced inside
                   the k walk, halo still exchanged between the phases
      fused_selfk  set_visc_force_tqf_selfk -- no exchange at all: the k seam
                   is read from the block's own far plane

    THE EXCHANGE IS IN THE TIMED WINDOW, and it did not use to be. grid.py's
    update_sources runs it between the two phases every step, so by Rule 3 it
    is part of the pair's per-step cost -- but the arms as originally written
    timed `tauq(); visc()` with no exchange, and quoted the resulting -20.9%
    as if the exchange had been in the loop. It had not: seed_tau_q runs the
    only exchange in the bench once, untimed, before the reps.

    Splitting the fused arm in two is what makes the two savings separable:
    `unfused` vs `fused` prices the tau/q DRAM round trip with the exchange
    held constant, and `fused` vs `fused_selfk` prices the exchange alone. The
    exchange is a Python loop over pairs.items() around one swap_by_ijk call,
    so it is O(surface) work behind a fixed Python overhead that does not
    scale with ncell -- keeping it in its own column stops it flattering the
    small end of the size ladder.

    All three are idempotent, for different reasons: `unfused` because phase 1
    rewrites every tau/q slot including the halos that phase 2 then scales in
    place, and the two fused arms because they never write tau_cell/q_cell at
    all (exchange_halos on a self-pair is idempotent too -- it reads only
    interior cells). Asserted in check_pair rather than assumed.
    """
    import ember.fortran as F

    ni, nj, nk = b.shape
    if b.i_cusp[0] > 0:
        # The fused arm does not apply the cusp seam correction (see the header
        # of bench/subroutines/viscous_tauq_fused.f90). Refuse rather than
        # silently compare a kernel that is missing a term.
        raise SystemExit(
            "callers_pair: this case is cusped (i_cusp=%r) and set_visc_force_tqf "
            "does not implement the cusp seam correction." % (b.i_cusp,)
        )

    unfused = dict(callers_tauq(b), **callers_visc(b, ("visc",)))
    tauq, visc = unfused["tauq"], unfused["visc"]
    # Hoisted so the timed lambda does no attribute lookup the other arms are
    # not also paying; the communicator itself is cached on the connectivity.
    exchange = grid.connectivity.periodic.exchange_halos

    out = {"unfused": lambda: (tauq(), exchange(), visc())}

    fn = getattr(F, "set_visc_force_tqf", None)
    fn_selfk = getattr(F, "set_visc_force_tqf_selfk", None)
    if fn is None and fn_selfk is None:
        return out

    # The rolling tau/q plane pair and the tau/q row temps, carved from
    # block.scratch alongside the face-flow buffers -- NOT from tau_q_halo,
    # which is this kernel's halo input.
    # The seam-free arm needs FOUR tq slots, not two: the rolling pair plus a
    # saved cell plane nk-1 (produced by the pre-pass) and a stashed cell
    # plane 1. That is the buffer growth the study's own risk note is about --
    # 2.6 MB rather than 1.3 MB at the 1M shape, against a kernel whose main
    # hazard is tq falling out of cache.
    n_tq = 4 if fn_selfk is not None else 2
    need = (ni + 1) * (nj + 1) * 9 * n_tq + ni * nj * 4 * 2 + ni * 4 * 3
    if need > b.scratch.size:
        raise SystemExit(
            f"callers_pair: block.scratch holds {b.scratch.size} floats, the "
            f"fused arms need {need}. A real integration wants its own buffer."
        )
    planes, rows, tq = util.carve_view(
        b.scratch, (ni, nj, 4, 2), (ni, 4, 3), (ni + 1, nj + 1, 9, n_tq)
    )
    halo = b.tau_q_halo
    b.F_body_nd.flags.writeable = True
    kw = dict(
        cons=b.conserved_nd,
        cons_cell=b.conserved_cell_nd,
        vol=b.vol_nd,
        dai=b.dAi_nd,
        daj=b.dAj_nd,
        dak=b.dAk_nd,
        omega_block=b.Omega_nd,
        r=b.r_nd,
        mu=b.mu_nd,
        p=b.P_nd,
        p_offset=b.P_offset_nd,
        fvisc=b.F_body_nd[..., 1:],
        vx=b.Vx_nd,
        vr=b.Vr_nd,
        vt=b.Vt_rel_nd,
        t=b.T_nd,
        cp=b.cp_nd,
        kappa=b.kappa_nd,
        pr_turb=1.0,
        xlength=b.xlen_sq_nd,
        mu_turb=b._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True),
        tau_cell=halo[..., 0:6],
        q_cell=halo[..., 6:9],
        tq=tq,
        planes=planes,
        rows=rows,
        kb=min(_KB_SLAB, nk - 1),
        **b.ijk_wall_visc,
        **b.Omega_wall_nd,
        i_cusp_start=b.i_cusp[0],
        i_cusp_end=b.i_cusp[1],
    )
    # `fused` still needs the exchange: it reads the k halo slots exchange_halos
    # fills (viscous_tauq_fused.f90's header calls tau_cell/q_cell a HALO SOURCE
    # ONLY input). `fused_selfk` does not -- that is the whole point of it.
    if fn is not None:
        kw_tqf = dict(kw, tq=tq[..., :2])
        out["fused"] = lambda: (exchange(), fn(**kw_tqf))
    # Timing controls. Both are WRONG BY CONSTRUCTION and exist only to
    # attribute the gap between `fused` and `fused_selfk`, which is far larger
    # than the exchange it was supposed to come from:
    #   fused_ctl   pre-pass + 4-slot tq, but the parent's halo reads put back
    #   fused_noij  reads nothing at all from the full-volume tau/q buffer
    # check_pair does not gate them -- it only ever compares `fused` and
    # `fused_selfk` -- so they cannot be mistaken for candidate kernels.
    for arm, sym in (("fused_ctl", "set_visc_force_tqf_ctl"),
                     ("fused_noij", "set_visc_force_tqf_noij")):
        fn_ctl = getattr(F, sym, None)
        if fn_ctl is not None:
            out[arm] = lambda fn_ctl=fn_ctl: fn_ctl(**kw)

    # fused_faces is the first SELF-CONTAINED fused arm: it produces its own
    # halo source (set_tau_q_faces, O(surface)), exchanges it (exchange_faces)
    # and consumes it, all inside the timed window. Every other fused arm here
    # free-rides on seed_tau_q, which fills the full-volume halo once, untimed,
    # before the reps -- so their numbers omit the cost of producing the halo
    # they read, while `unfused` pays set_tau_q_soa in full. This arm is the
    # first that can be compared with `unfused` without that asymmetry.
    # set_visc_force_tqf with a four-slot tq dummy and nothing else changed:
    # the bisection between viscous_tauq_ctl's pre-pass and its bigger buffer.
    # Component-first packings of the velocity and transport streams: six
    # arrays become two in the hot loops, same values and same bytes. Packed
    # here at setup, outside the timed window -- a real integration would
    # change Block's axis order instead, which is why this is a bench arm.
    fn_packed = getattr(F, "set_visc_force_tqf_packed", None)
    if fn_packed is not None:
        vel = np.asfortranarray(
            np.stack([np.asarray(b.Vx_nd), np.asarray(b.Vr_nd),
                      np.asarray(b.Vt_rel_nd)], axis=0).astype(np.float32))
        trans = np.asfortranarray(
            np.stack([np.asarray(b.mu_nd), np.asarray(b.cp_nd),
                      np.asarray(b.kappa_nd)], axis=0).astype(np.float32))
        kw_packed = dict(kw, tq=tq[..., :2], vel=vel, trans=trans)
        # Same shape as `fused`: it reads the exchanged k halo, so it needs
        # the exchange in front of it.
        out["packed"] = lambda: (exchange(), fn_packed(**kw_packed))

    fn_ptrans = getattr(F, "set_visc_force_tqf_ptrans", None)
    if fn_ptrans is not None:
        trans_p = np.asfortranarray(
            np.stack([np.asarray(b.mu_nd), np.asarray(b.cp_nd),
                      np.asarray(b.kappa_nd)], axis=0).astype(np.float32))
        kw_pt = dict(kw, tq=tq[..., :2], trans=trans_p)
        out["packed_trans"] = lambda: (exchange(), fn_ptrans(**kw_pt))

    fn_pad = getattr(F, "set_visc_force_tqf_pad", None)
    if fn_pad is not None:
        out["fused_pad"] = lambda: fn_pad(**kw)

    fn_faces = getattr(F, "set_visc_force_tqf_faces", None)
    if fn_faces is not None:
        faces = b.tau_q_faces
        kw_p1 = dict(
            cons=b.conserved_nd, t=b.T_nd, mu=b.mu_nd, cp=b.cp_nd,
            kappa=b.kappa_nd, pr_turb=1.0, xlength=b.xlen_sq_nd, vol=b.vol_nd,
            dai=b.dAi_nd, daj=b.dAj_nd, dak=b.dAk_nd, r=b.r_nd, vx=b.Vx_nd,
            vr=b.Vr_nd, vt=b.Vt_rel_nd,
            f_i1=faces[0], f_ini=faces[1], f_j1=faces[2], f_jnj=faces[3],
            f_k1=faces[4], f_knk=faces[5],
            **b.ijk_wall_visc,
        )
        kw_faces = {k: v for k, v in kw.items()
                    if k not in ("tau_cell", "q_cell")}
        kw_faces.update(
            f_i1=faces[0], f_ini=faces[1], f_j1=faces[2], f_jnj=faces[3],
            f_k1=faces[4], f_knk=faces[5], tq=tq[..., :2],
        )
        p1_faces = F.set_tau_q_faces
        exch_faces = grid.connectivity.periodic.exchange_faces
        out["faces"] = lambda: (p1_faces(**kw_p1), exch_faces(),
                                fn_faces(**kw_faces))
        # Phase 1 alone, both forms, so the split between producer and
        # consumer is measured rather than inferred by subtraction.
        out["p1_faces"] = lambda: p1_faces(**kw_p1)
        out["p1_soa"] = tauq

        # STREAM-COUNT CONTROLS. Identical arithmetic, identical results, and
        # the same instruction stream -- the only difference is how many
        # DISTINCT arrays the loop walks. `streams_hi` hands the kernel three
        # separate buffers holding the same values; `streams_lo` hands it one
        # buffer three times. If the fused loop is limited by the number of
        # concurrent streams rather than by the bytes it moves, these two must
        # differ; if they do not, that hypothesis is dead.
        same = np.array(b.mu_nd, copy=True, order="F")
        kw_hi = dict(kw_faces, mu=np.array(same, copy=True, order="F"),
                     cp=np.array(same, copy=True, order="F"),
                     kappa=np.array(same, copy=True, order="F"))
        kw_lo = dict(kw_faces, mu=same, cp=same, kappa=same)
        out["streams_hi"] = lambda: fn_faces(**kw_hi)
        out["streams_lo"] = lambda: fn_faces(**kw_lo)

        # The same controlled A/B on the VELOCITY streams, which is the
        # interesting case: Vx, Vr and Vt_rel are all recoverable from cons, r
        # and Omega, every one of which the kernel already takes. If collapsing
        # them is worth much, deriving them on the fly (into a rolling node
        # plane, as tq does for tau/q) trades three volume streams for three
        # divisions per node. If it is worth little, it cannot pay for them.
        vsame = np.array(b.Vx_nd, copy=True, order="F")
        kw_vhi = dict(kw_faces, vx=np.array(vsame, copy=True, order="F"),
                      vr=np.array(vsame, copy=True, order="F"),
                      vt=np.array(vsame, copy=True, order="F"))
        kw_vlo = dict(kw_faces, vx=vsame, vr=vsame, vt=vsame)
        out["vel_hi"] = lambda: fn_faces(**kw_vhi)
        out["vel_lo"] = lambda: fn_faces(**kw_vlo)
        # The consumer alone, for comparison with `fused` on equal footing
        # (both then exclude their halo production).
        out["faces_nop1"] = lambda: fn_faces(**kw_faces)

    # fused_nosig drops tau_cell/q_cell from the SIGNATURE, so it needs its own
    # kwargs dict. Paired with fused_noij it separates the cost of reading the
    # 37.8 MB buffer in the k loop from the cost of it merely being an
    # argument. NEITHER measures the case for a compact surface buffer: the
    # block still allocates the volume and seed_tau_q still fills it.
    fn_nosig = getattr(F, "set_visc_force_tqf_nosig", None)
    if fn_nosig is not None:
        kw_nosig = {k: v for k, v in kw.items() if k not in ("tau_cell", "q_cell")}
        out["fused_nosig"] = lambda: fn_nosig(**kw_nosig)

    if fn_selfk is not None:
        # i_perk is the sentinel, so a block with no k-face PeriodicPatch is
        # not an error -- it is simply one the general path handles, and the
        # other two arms are still a valid comparison on it. A block that IS
        # k-periodic but fails a guard is an error: the caller asked for the
        # seam case and the geometry cannot support the select.
        ineligible = selfk_ineligible(grid, b)
        if ineligible is None:
            out["fused_selfk"] = lambda: fn_selfk(**kw)
        elif b.i_perk != (0, 0):
            raise SystemExit(f"callers_pair: seam-free arm not eligible: {ineligible}")
    return out


def check_pair(grid, b):
    """Gate the fused pair against the unfused one, on fvisc AND mu_turb.

    mu_turb matters as much as fvisc here: it is the other output of the phase
    the fusion absorbs, it is consumed downstream by timestep_diffusion, and a
    fused kernel that produced the right forces from a wrong mixing length
    would pass an fvisc-only gate.

    Expect ~ulp agreement, not bitwise: same arithmetic, different loop shape,
    and -Ofast reassociates the two shapes differently (measured at 2.5 ulp for
    the fvisc fusion). Deviations are quantified at the field scale, since
    fvisc is a small difference of large face flows and pointwise ulps blow up
    where it passes through zero.
    """
    fns = callers_pair(grid, b)
    arms = [n for n in ("fused", "fused_selfk", "faces", "packed") if n in fns]
    if not arms:
        return {}
    fvisc, mu_turb = b.F_body_nd, b.mu_turb

    # Reset the halo state before each fused run. `unfused` ends with
    # set_visc_force, which scales the tau/q halos by (2*wall-1) IN PLACE; a
    # fused run afterwards would read those already scaled values and scale
    # them again, putting the wrong sign on every wall face. That cannot happen
    # in the bench (one arm per process) but it can here, and it would be
    # silent. Re-running phase 1 is also exactly what a real integration
    # leaves behind: halos at +edge.
    #
    # NB this is not the explanation for the deviation reported below -- that
    # is compiler reassociation, and it is unchanged with or without this
    # reset. The reset is here because the hazard is real, not because it
    # fixed anything.
    callers_tauq(b)["tauq"]()
    # Snapshot the WHOLE buffer, not just the six faces halo_restorer covers.
    # The faces arm poisons all of it to prove it reads none of it, and an arm
    # running afterwards would otherwise exchange poisoned owned-edge cells
    # into its own seam -- a NaN that looks like a seam bug and is not.
    halo_snapshot = np.array(b.tau_q_halo, copy=True)

    def restore():
        b.tau_q_halo[...] = halo_snapshot

    def run(name):
        if name != "unfused":
            restore()
        if name == "faces":
            # Poison the full-volume buffer before every faces run. The arm
            # claims never to touch it; if that is wrong the gate should fail
            # loudly here rather than quietly agree because the buffer happens
            # to hold the right values.
            b.tau_q_halo[...] = np.nan
        fns[name]()
        return np.array(fvisc, copy=True), np.array(mu_turb, copy=True)

    base, base2 = run("unfused"), run("unfused")

    results = {}
    for arm in arms:
        got, got2 = run(arm), run(arm)
        for n, field in enumerate(("fvisc", "mu_turb")):
            ref, val = base[n], got[n]
            idem = np.array_equal(base[n], base2[n]) and np.array_equal(got[n], got2[n])
            diff = np.abs(val - ref)
            scale = float(np.abs(ref).max())
            results[(arm, field)] = dict(
                bitwise=bool(np.array_equal(val, ref)),
                max_abs=float(diff.max()),
                rel=float(diff.max() / scale) if scale else 0.0,
                max_ulp=(
                    float(diff.max() / np.spacing(np.float32(scale))) if scale else 0.0
                ),
                idempotent=bool(idem),
                # Where the deviation sits matters more than its size for the
                # seam arms: a k-seam bug lands on the k=1 / k=nk-1 cell planes
                # and nowhere else, while compiler reassociation is spread
                # through the interior. Reported so the two are never confused.
                seam_only=bool(_seam_only(diff)) if diff.max() > 0 else True,
            )
    return results


def _seam_only(diff):
    """True if every differing cell lies on the k=1 or k=nk-1 cell plane.

    The reason this is reported: a wrong far-plane index in the seam select
    shows up exactly on the two seam cell planes, whereas the reassociation
    the fused arms are already known to carry is spread through the interior
    (93% of differing entries strictly interior, per the arm header). Without
    this split, "35 ulp" alone cannot tell the two apart.
    """
    if diff.ndim < 3:
        return False
    hit = diff > 0
    interior = hit.copy()
    interior[:, :, 0] = False
    interior[:, :, -1] = False
    return not interior.any()


def check_correctness(b, active_arms=VISC_ARMS):
    """Compare each set_visc_force arm against production on identical input.

    The input restore is mandatory here, not decorative: without it arm n
    would see the halo signs arm n-1 left behind (module docstring), and every
    arm after the first would be gated against a different tau/q field.
    """
    restore = halo_restorer(b)
    fns = callers_visc(b, active_arms)
    fvisc = b.F_body_nd

    restore()
    fns["visc"]()
    base = np.array(fvisc, copy=True)
    scale = float(np.abs(base).max())

    results = {}
    for name, fn in fns.items():
        if name == "visc":
            continue
        restore()
        fn()
        got = np.array(fvisc, copy=True)
        diff = np.abs(got - base)
        # Ulps AT THE FIELD SCALE, not pointwise. fvisc is a small difference
        # of large face flows, so a cell whose own value is ~1e-15 has an
        # enormous pointwise ulp count for an absolute deviation that is
        # nothing -- an early version of this gate reported 131072 ulps for a
        # deviation of 1 ulp of the largest value in the field. The README's
        # rule is to quantify in ulps of the quantities being DIFFERENCED; the
        # face flows are not visible from here, and they are larger than the
        # residual, so ulps of max|fvisc| is the conservative stand-in.
        results[name] = dict(
            bitwise=bool(np.array_equal(got, base)),
            max_abs=float(diff.max()),
            rel=float(diff.max() / scale) if scale else 0.0,
            max_ulp=float(diff.max() / np.spacing(np.float32(scale))) if scale else 0.0,
        )
    return results


def main():
    """Standalone Gate-2 correctness pre-flight. Times nothing."""
    import argparse

    import ember.fortran as F

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--ncell", type=int, default=300_000)
    ap.add_argument(
        "--periodic-k",
        default=None,
        choices=("full", "hmesh"),
        help="make the duct's k faces periodic to each other: 'full' for the "
        "whole face, 'hmesh' for two i-intervals with a wall between. Without "
        "it the duct has no seam at all, which is why the fused arm has never "
        "exercised the halo path it models.",
    )
    args = ap.parse_args()

    grid, b = build_case(args.ncell, periodic_k=args.periodic_k)
    ni, nj, nk = b.shape
    active = [a for a in VISC_ARMS if a == "visc" or getattr(F, ENTRY[a], None)]
    print(f"grid {ni} x {nj} x {nk}  ncell={args.ncell}  cusp={b.i_cusp[0] > 0}")
    print(f"periodic_k={args.periodic_k!r}  i_perk={b.i_perk}  "
          f"k-seam non-wall fraction="
          f"{float(np.asarray(b.ijk_wall_visc['wallk1']).mean()):.3f}")
    print(f"seam-free eligibility: {selfk_ineligible(grid, b) or 'ELIGIBLE'}")
    print(f"arms in this build: {active}")

    # Rule 5: build_duct_grid is axially straight, so Vr = Vt = 0 and the
    # wall function's Vt_slip, tau(6)'s swirl term and the polar source's
    # rho*Vt^2 are all degenerate. Swirl before seeding tau/q, since tau/q
    # are a function of the state.
    swirl(b)
    seed_tau_q(grid, b)

    print("\ncorrectness gate (swirled state, so the swirl terms are non-zero):")
    for name, r in check_correctness(b, active).items():
        ok = "BITWISE" if r["bitwise"] else f"DIFFERS max {r['max_abs']:.3e}"
        print(f"  {name:>8}  {ok}  ({r['rel']:.3e} of scale, {r['max_ulp']:.2f} ulp)")

    pair = check_pair(grid, b)
    if pair:
        print("\nfused arms vs unfused (set_tau_q_soa + exchange_halos + "
              "set_visc_force):")
        for (arm, field), r in pair.items():
            ok = "BITWISE" if r["bitwise"] else f"DIFFERS max {r['max_abs']:.3e}"
            where = "" if r["bitwise"] else (
                "  SEAM-ONLY" if r["seam_only"] else "  interior"
            )
            print(
                f"  {arm:>12} {field:>8}  {ok}  ({r['rel']:.3e} of scale, "
                f"{r['max_ulp']:.2f} ulp){where}  idempotent={r['idempotent']}"
            )
        if not all(r["idempotent"] for r in pair.values()):
            print("  FAIL: an arm is not idempotent -- repeated reps would not "
                  "be measuring the same work")
            return 1
        # A deviation confined to the two seam cell planes is the seam select,
        # not reassociation: the fused arms' known ~35 ulp is spread through
        # the interior. Fail loudly rather than let it read as compiler noise.
        seam = [k for k, r in pair.items() if not r["bitwise"] and r["seam_only"]]
        if seam:
            print(f"  FAIL: deviation confined to the k seam for {seam} -- "
                  "that is the seam handling, not reassociation")
            return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
