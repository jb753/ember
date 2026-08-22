"""Arm definitions for the viscous study: `set_tau_q_faces` and `set_visc_force`.

The viscous pass is two kernels with a grid-wide periodic seam exchange between
them (grid.py's update_sources), so the unit that means anything is the PAIR:

  p1     set_tau_q_faces   -- boundary tau/q, O(surface), into the six face
                              buffers of Block.tau_q_faces
  k2     set_visc_force    -- interior tau/q produced inside its own k walk,
                              face fluxes, wall functions, cusp seam and the
                              polar source, in one pass over fvisc
  prod   p1 + exchange_faces + k2, which is exactly what update_sources runs

`--kernel viscpair` times `prod`; the two phases are also timed separately, so
the split between producer and consumer is measured rather than inferred by
subtraction. Everything here is production -- there is no second implementation
of the viscous pass to compare against any more, and the ladder of fused arms
that got the codebase to this one is history (bench/README.md keeps the
numbers; the arms themselves are in git).

WHAT THIS HARNESS STILL GATES. Correctness of the kernels lives in pytest now
(test_viscous_phases_golden locks both phases against committed goldens,
test_viscous_cusp_seam and test_viscous_periodic gate the two seams). What
cannot live there is the property this harness's own method depends on:

  IDEMPOTENCE. A rep-based instrument calls the same kernel thirty times in a
  row and assumes every rep does identical work on identical input. That was
  NOT true of the previous viscous kernel -- its entry pass scaled the tau/q
  halo slots by (2*wall - 1) in place, so rep n read rep n-1's halos, and the
  harness carried a restore hook for it. It is true now, because the sign is
  applied once by set_tau_q_faces when it writes layer 1 and the consumer only
  reads it. `check_pair` asserts that rather than assuming it, and the restore
  hook is gone.

THE PANEL SWEEP. set_visc_force takes its j-panel width as an argument
(`jbw_in`, 0 meaning size it from the kernel's own VISC_JAREA), so the arms
below sweep it. That is a live study, not a control: the panel bounds the
kernel's carry, and it is worth about 4% serially and about 46 points at 8-rank
socket contention -- the single largest effect measured on this kernel. Sweep
it on a machine with a different L2 before changing VISC_JAREA.
"""

import numpy as np

import ember.block

from residual_arms import build_case, swirl  # noqa: F401  (re-export)

# Panel widths in cell rows. 0 is production (VISC_JAREA / ni, floored at 4);
# the rest bracket it, and jbw large enough is a single panel -- the unpanelled
# walk, which is what the panel has to beat.
PANEL_WIDTHS = (4, 8, 16, 32, 4096)


def _p1_kwargs(b):
    """Arguments for phase 1, the O(surface) boundary producer."""
    faces = b.tau_q_faces
    return dict(
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
        f_i1=faces[0],
        f_ini=faces[1],
        f_j1=faces[2],
        f_jnj=faces[3],
        f_k1=faces[4],
        f_knk=faces[5],
        **b.ijk_wall_visc,
    )


def _k2_kwargs(b):
    """Arguments for phase 2, the fused walk.

    Every buffer it takes from the arena comes from ONE `_carve_viscous`, which
    is what makes them disjoint -- carving them apart would overlap them.
    """
    faces, tq, planes, rows = ember.block._carve_viscous(b)
    b.F_body_nd.flags.writeable = True
    return dict(
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
        f_i1=faces[0],
        f_ini=faces[1],
        f_j1=faces[2],
        f_jnj=faces[3],
        f_k1=faces[4],
        f_knk=faces[5],
        tq=tq,
        planes=planes,
        rows=rows,
        **b.ijk_wall_visc,
        **b.Omega_wall_nd,
        i_cusp_start=b.i_cusp[0],
        i_cusp_end=b.i_cusp[1],
        jbw_in=0,
    )


def seed_tau_q(grid, b):
    """Run phase 1 once, untimed, so the face buffers hold a real field.

    Rule 3: a real port evaluates the producer immediately before the consumer
    every step, so the boundary tau/q are emphatically not a hoistable input --
    but they are the PREVIOUS kernel's output, not this one's, and pricing them
    inside the `k2` window would time phase 1 twice.

    Must be called after swirl(): tau/q are a function of the state, so seeding
    them before the state is perturbed would leave phase 2 reading a shell that
    does not belong to the velocities it also reads.
    """
    import ember.fortran as F

    F.set_tau_q_faces(**_p1_kwargs(b))
    grid.connectivity.periodic.exchange_faces()
    return b.tau_q_faces


def callers_pair(grid, b):
    """Time the viscous pair and each of its phases, plus the panel sweep.

    Arms:

      prod   the whole pass, as update_sources runs it
      p1     set_tau_q_faces alone
      k2     set_visc_force alone, on the shell seed_tau_q left
      jbwN   set_visc_force alone at panel width N (see PANEL_WIDTHS)

    `prod` is the only arm that pays for phase 1 and the exchange, so it is the
    one to quote end to end; `p1` and `k2` say where its time goes.
    """
    import ember.fortran as F

    kw_p1, kw_k2 = _p1_kwargs(b), _k2_kwargs(b)
    # Hoisted so the timed lambdas do no attribute lookup one another is not
    # also paying; the communicator itself is cached on the connectivity.
    p1 = F.set_tau_q_faces
    k2 = F.set_visc_force
    exchange = grid.connectivity.periodic.exchange_faces

    out = {
        "prod": lambda: (p1(**kw_p1), exchange(), k2(**kw_k2)),
        "p1": lambda: p1(**kw_p1),
        "k2": lambda: k2(**kw_k2),
    }
    for jbw in PANEL_WIDTHS:
        kw = dict(kw_k2, jbw_in=jbw)
        out[f"jbw{jbw}"] = lambda kw=kw: k2(**kw)
    return out


def check_pair(grid, b):
    """Gate what the timing method depends on: idempotence, and the panel.

    Two calls of each arm on identical input must give identical output, or a
    rep-based measurement is timing a moving target. And every panel width must
    give the SAME answer -- exactly, since a panel changes only which rows are
    live at once -- or the sweep below is comparing different computations.

    Returns a dict of arm -> dict(idempotent=, matches_prod_panel=).
    """
    fns = callers_pair(grid, b)
    fvisc, mu_turb = b.F_body_nd, b.mu_turb

    def run(name):
        fns[name]()
        return np.array(fvisc, copy=True), np.array(mu_turb, copy=True)

    # Production's own panelling, the reference every other width must match.
    ref = run("k2")
    results = {}
    for name in ("prod", "k2", *(f"jbw{j}" for j in PANEL_WIDTHS)):
        first, second = run(name), run(name)
        results[name] = dict(
            idempotent=all(np.array_equal(x, y) for x, y in zip(first, second)),
            matches_prod_panel=(
                all(np.array_equal(x, y) for x, y in zip(first, ref))
                if name.startswith("jbw") else None
            ),
        )
    return results


def main():
    """Standalone correctness pre-flight for the harness's own assumptions."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--ncell", type=int, default=300_000)
    ap.add_argument(
        "--periodic-k",
        default=None,
        choices=("full", "hmesh"),
        help="make the duct's k faces periodic to each other: 'full' for the "
        "whole face, 'hmesh' for two i-intervals with a wall between. Without "
        "it the duct has no seam at all and exchange_faces moves nothing.",
    )
    args = ap.parse_args()

    grid, b = build_case(args.ncell, periodic_k=args.periodic_k)
    ni, nj, nk = b.shape
    print(f"grid {ni} x {nj} x {nk}  ncell={args.ncell}  cusp={b.i_cusp[0] > 0}")
    print(f"periodic_k={args.periodic_k!r}  i_perk={b.i_perk}  "
          f"k-seam non-wall fraction="
          f"{float(np.asarray(b.ijk_wall_visc['wallk1']).mean()):.3f}")
    print(f"arena {b.scratch.size * 4 / 1024**2:.2f} MB")

    # Rule 5: build_duct_grid is axially straight, so Vr = Vt = 0 and the
    # wall function's Vt_slip, tau(6)'s swirl term and the polar source's
    # rho*Vt^2 are all degenerate. Swirl before seeding tau/q, since tau/q
    # are a function of the state.
    swirl(b)
    seed_tau_q(grid, b)

    print("\nharness preconditions:")
    bad = 0
    for name, r in check_pair(grid, b).items():
        panel = "" if r["matches_prod_panel"] is None else (
            "  panel-invariant" if r["matches_prod_panel"]
            else "  PANEL CHANGES THE ANSWER"
        )
        print(f"  {name:>8}  idempotent={r['idempotent']}{panel}")
        bad += (not r["idempotent"]) or (r["matches_prod_panel"] is False)
    if bad:
        print("  FAIL: see above -- a rep-based measurement would be timing "
              "a moving target, or the panel widths are not the same kernel")
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
