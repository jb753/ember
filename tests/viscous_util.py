"""Drive the two viscous kernels the way :meth:`ember.grid.Grid.update_sources`
does, for tests that need the pair rather than the grid method.

Three test modules run the viscous pair on a hand-built block -- the cusp seam,
the periodic seam, and the polar/mu-limit comparison -- and each needs it
slightly differently: one overrides ``i_cusp``, one toggles the seam exchange,
one only wants the composed force. The call itself is thirty-odd arguments, so
three copies of it would be three chances for a test to drift from production
and start gating something production does not do.

:mod:`test_viscous_phases_golden` deliberately does NOT use this: it feeds
phase 2 a synthetic boundary shell instead of phase 1's output, which is what
makes its two goldens fail independently.
"""

import numpy as np

import ember.block
import ember.fortran


def fill_faces(block, pr_turb):
    """Phase 1: boundary tau/q into ``block.tau_q_faces``. Returns the six."""
    faces = block.tau_q_faces
    ember.fortran.set_tau_q_faces(
        cons=block.conserved_nd,
        t=block.T_nd,
        mu=block.mu_nd,
        cp=block.cp_nd,
        kappa=block.kappa_nd,
        pr_turb=pr_turb,
        wdist=block.wdist_nd,
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
    return faces


def run_visc_force(block, pr_turb, i_cusp=None, jbw=0):
    """Phase 2: interior tau/q, face fluxes and the polar source into F_body.

    Zeroes ``F_body_nd`` first and hands it back locked, as update_sources
    leaves it. ``i_cusp`` defaults to the block's own; pass ``(0, 0)`` to run
    with the cusp seam correction switched off, which is how a test isolates
    it. ``jbw`` is the j-panel width, 0 meaning the kernel's own VISC_JAREA.

    Returns ``fvisc``, a float64 copy of ``F_body_nd[..., 1:]``.
    """
    fbody = block.F_body_nd
    fbody.flags.writeable = True
    fbody.fill(0.0)
    mu_turb = block._get_data_by_keys(("mu_turb",), raise_uninit=False, writeable=True)

    i_cusp_start, i_cusp_end = block.i_cusp if i_cusp is None else i_cusp
    faces, tq, planes, rows, transport = ember.block._carve_viscous(block)
    ember.fortran.set_visc_force(
        cons=block.conserved_nd,
        vol=block.vol_nd,
        dai=block.dAi_nd,
        daj=block.dAj_nd,
        dak=block.dAk_nd,
        omega_block=block.Omega_nd,
        r=block.r_nd,
        mu=block.mu_nd,
        p=block.P_nd,
        p_offset=block.P_offset_nd,
        fvisc=fbody[..., 1:],
        vx=block.Vx_nd,
        vr=block.Vr_nd,
        vt=block.Vt_rel_nd,
        t=block.T_nd,
        cp=block.cp_nd,
        kappa=block.kappa_nd,
        pr_turb=pr_turb,
        wdist=block.wdist_nd,
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
    # The kernel is mu_turb's producer, so mark it initialised for any later
    # read through the public property.
    block._versions["mu_turb"] += 1
    fbody.flags.writeable = False
    return np.array(fbody[..., 1:], dtype=np.float64)


def run_pair(block, pr_turb, comm=None, i_cusp=None, jbw=0):
    """Both phases with the seam exchange between them, as update_sources runs.

    ``comm`` is a :class:`~ember.periodic_communicator.PeriodicCommunicator`,
    or None to skip the exchange -- which leaves every periodic face reading
    the ``(2*wall - 1)`` ghost its producer seeded instead of the neighbour's
    edge cell, and is how a test measures what the seam is worth.
    """
    fill_faces(block, pr_turb)
    if comm is not None:
        comm.exchange_faces()
    return run_visc_force(block, pr_turb, i_cusp=i_cusp, jbw=jbw)
