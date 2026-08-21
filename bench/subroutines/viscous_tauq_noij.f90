! TIMING CONTROL ONLY -- WRONG ANSWER BY CONSTRUCTION, NEVER GATE IT.
!
! set_visc_force_tqf_selfk with load_halo_ijedge deleted, so the kernel
! reads NOTHING from the full-volume tau/q buffer -- not the k planes
! (the seam select already replaced those) and now not the i/j edges
! either. The i/j edges of tq are left holding the previous plane's
! values, which makes fvisc wrong on the i/j boundary shell but leaves
! the arithmetic, the loop structure and the working set identical.
!
! It exists to bracket one question: how much is there to gain by
! feeding the fused kernel a compact surface buffer instead of letting
! it reach into the 37.8 MB volume? This arm is the upper bound on that
! gain -- a real boundary kernel would still have to read something.
!
! EXPERIMENTAL: viscous fusion with NO periodic k-seam exchange.
!
! set_visc_force_tqf already fuses set_tau_q_soa into set_visc_force, removing
! the 9-plane tau/q round trip through DRAM. It still needs the grid-wide
! periodic halo exchange to run BETWEEN the phases, though, because at k=1 it
! reads the halo slot that exchange filled. That exchange is a phase barrier
! across the whole grid.
!
! For a block periodic to ITSELF in k -- the H-mesh case -- that exchange is a
! copy from the block's own far cell plane into its own near halo. This arm
! reads the far plane directly instead, so for the k seam there is nothing to
! exchange and no barrier. The two load_halo_kplane calls are the only things
! that change; everything else is set_visc_force_tqf character for character.
!
! WHAT THE SEAM SELECT IS. At k=1 production's halo slot holds, per (i,j):
!   periodic face (wallk1 = 1): the exchanged far cell plane, scaled by
!                               (2*1-1) = +1, i.e. cell plane nk-1 unchanged;
!   wall face     (wallk1 = 0): set_tau_q_soa's +edge fill, scaled by
!                               (2*0-1) = -1, i.e. MINUS cell plane 1.
! Both are available inside the walk -- plane nk-1 from a pre-pass, plane 1 as
! it is produced -- so the halo read becomes a blend against a mask the kernel
! already takes as an argument. The mask is exactly 0.0 or 1.0
! (block.ijk_wall_visc casts a boolean), so the blend is an exact select.
!
! WHAT THE CALLER MUST GUARANTEE, because this kernel cannot check it:
!
!   * every k-face periodic patch is paired to THIS block, not a neighbour;
!   * the non-wall part of each k face is EXACTLY the periodic part.
!
! The second is not implied by the first and is the easy one to get wrong:
! wallk1 means "not a viscous wall", so it is 1.0 for a slip endwall, for a
! mixing plane, and for CuspPatch -- which is in PERMEABLE_TYPES and must sit
! on a constant-k face, so a real H-mesh trailing edge reads non-wall while
! being neither wall nor periodic. Nothing exchanges those halos. Reading the
! far plane there would be silently wrong, so bench/visc_arms.py's
! selfk_ineligible() refuses any case that does not satisfy both, gated on
! block.i_perk as the sentinel.
!
! COST AGAINST set_visc_force_tqf, both of which this arm pays to buy the
! exchange removal:
!   * one extra cell plane of producer work (the pre-pass for plane nk-1),
!     O(surface), duplicated source rather than a shared procedure -- see the
!     note on the row temps in the parent arm, moving them to dummy arguments
!     costs the loop its vectorization, and `block` breaks the f2py build;
!   * tq grows from 2 rolling slots to 4 (a saved plane nk-1 and a stashed
!     plane 1): 1.3 MB -> 2.6 MB at the 1M bench shape. That is the main risk
!     to the whole idea, since the parent arm's header already names tq's L3
!     residency as its dominant hazard.
!
! Like its parent, THE CUSP SEAM CORRECTION IS NOT APPLIED and the arm refuses
! cusped cases in the caller. Note though that this arm stashes both plane 1
! and plane nk-1, which is exactly what that correction needs and exactly what
! the parent arm lacked.

module viscous_noij_helpers
    implicit none
    private
    public :: unused_placeholder

contains

    ! Nothing: this control reads no halo at all.
    subroutine unused_placeholder()
    end subroutine unused_placeholder

end module viscous_noij_helpers


subroutine set_visc_force_tqf_noij( &
    cons, cons_cell, vol, dAi, dAj, dAk, &
    Omega_block, r, mu, P, P_offset, &
    fvisc, &
    Vx, Vr, Vt, &
    T, cp, kappa, Pr_turb, xlength, &
    mu_turb, &
    tau_cell, &
    q_cell, &
    tq, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    Omega_walli1_nd, Omega_wallj1_nd, Omega_wallk1_nd, &
    Omega_wallni_nd, Omega_wallnj_nd, Omega_wallnk_nd, &
    i_cusp_start, i_cusp_end, &
    kb, ni, nj, nk)

    use viscous_helpers
    implicit none

    integer, intent(in) :: ni, nj, nk, kb
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: cons_cell(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega_block
    real, intent(in) :: mu(ni, nj, nk)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(inout) :: fvisc(ni-1, nj-1, nk-1, 4)
    real, intent(in) :: Vx(ni, nj, nk)
    real, intent(in) :: Vr(ni, nj, nk)
    real, intent(in) :: Vt(ni, nj, nk)
    real, intent(in) :: T(ni, nj, nk)
    real, intent(in) :: cp(ni, nj, nk)
    real, intent(in) :: kappa(ni, nj, nk)
    real, intent(in) :: Pr_turb
    real, intent(in) :: xlength(ni-1, nj-1, nk-1)
    ! Cell-centred mixing-length viscosity, written at the cell's low-corner
    ! node exactly as set_tau_q_soa writes it; consumed downstream by
    ! timestep_diffusion, so it keeps its full-volume write.
    real, intent(inout) :: mu_turb(ni, nj, nk)
    ! HALO SOURCE ONLY. Only slots 1/ni+1, 1/nj+1, 1/nk+1 are read -- the
    ! values exchange_halos fills. The interior is neither read nor written:
    ! this kernel produces it. intent(in), so unlike production this kernel
    ! does not mutate its own input and is therefore idempotent.
    real, intent(in) :: tau_cell(ni+1, nj+1, nk+1, 6)
    real, intent(in) :: q_cell(ni+1, nj+1, nk+1, 3)
    real, intent(inout) :: planes(ni, nj, 4, 2)
    real, intent(inout) :: rows(ni, 4, 3)
    ! Rolling tau/q CELL-plane pair, halo-indexed in i and j exactly as
    ! tau_cell is, with slots 1-6 tau and 7-9 q. Slot ta holds cell plane
    ! k-1, slot tb cell plane k.
    real, intent(inout) :: tq(ni+1, nj+1, 9, 4)
    real, intent(in) :: walli1(nj-1, nk-1)
    real, intent(in) :: wallni(nj-1, nk-1)
    real, intent(in) :: wallj1(ni-1, nk-1)
    real, intent(in) :: wallnj(ni-1, nk-1)
    real, intent(in) :: wallk1(ni-1, nj-1)
    real, intent(in) :: wallnk(ni-1, nj-1)
    real, intent(in) :: Omega_walli1_nd(nj-1, nk-1)
    real, intent(in) :: Omega_wallni_nd(nj-1, nk-1)
    real, intent(in) :: Omega_wallj1_nd(ni-1, nk-1)
    real, intent(in) :: Omega_wallnj_nd(ni-1, nk-1)
    real, intent(in) :: Omega_wallk1_nd(ni-1, nj-1)
    real, intent(in) :: Omega_wallnk_nd(ni-1, nj-1)
    integer, intent(in) :: i_cusp_start, i_cusp_end

    integer :: i, j, k, jc, kc
    logical :: k_interior, row_interior
    integer :: sa, sb, pa, pb, stmp
    real :: tauf(6), qf(3), Vf(3), rf
    real :: wvisc(3), Vabs, wf(4), wfac
    integer :: ta, tb, tp, ts, c
    ! Row temps for the tau/q stage, AUTOMATIC exactly as set_tau_q_soa
    ! declares them. A caller-preallocated buffer was tried first and cost the
    ! stage-1 loop its vectorization: GCC versions this loop with a runtime
    ! alias check (opt-report: "loop versioned for vectorization because of
    ! possible aliasing") and will not do so against a dummy argument. Keeping
    ! them automatic also leaves the arm differing from production only in the
    ! thing under test.
    real :: gVx(ni-1, 3), gVr(ni-1, 3), gVt(ni-1, 3)
    real :: vct(ni-1), rcr(ni-1), ivr(ni-1), rhoc(ni-1)
    real :: cpc(ni-1), muc(ni-1), kac(ni-1)
    real :: visc_lim, lambda
    ! Scalars for the hand-inlined polar source (see the note at its first
    ! use): GCC inlines polar_src into production's set_visc_force but not
    ! into this larger fused body, and a call in the loop blocks
    ! vectorization outright.
    real :: prhoc, prhorVtc, prc, pPc, pVtc
    real :: f1, f2, f3, f4, f5, f6, g1, g2, g3
    real :: t1, t2, t3, t4, t5, t6, w1, w2, w3, vm, mut, fac

    ! kb is inert -- the fused schedule subsumes k-slab blocking -- and
    ! i_cusp_start/i_cusp_end are inert because THIS KERNEL DOES NOT APPLY THE
    ! CUSP SEAM CORRECTION (see the file header). All three stay in the
    ! signature so the arm shares one kwargs dict with production, and are
    ! consumed in this guard rather than silenced.
    ! tau_cell/q_cell stay in the signature so this control shares one
    ! kwargs dict with the arms it is compared against, but nothing reads
    ! them any more. One element each, in a test that is never true, is
    ! what keeps -Werror=unused-dummy-argument quiet: two cache lines per
    ! call, against a kernel that moves hundreds of megabytes.
    if (kb < 1 .or. i_cusp_start < 0 .or. i_cusp_end < 0 &
        .or. tau_cell(1,1,1,1) /= tau_cell(1,1,1,1) &
        .or. q_cell(1,1,1,1) /= q_cell(1,1,1,1)) return

    pa = 1
    pb = 2
    ta = 1
    tb = 2
    ! Fixed slots, never rolled: tp holds cell plane nk-1 (the k=1 halo's
    ! periodic side), ts holds cell plane 1 (the k=nk halo's periodic side).
    tp = 3
    ts = 4

    ! ===== PRE-PASS: cell plane nk-1 into slot tp =====
    ! One plane of duplicated producer work, O(surface). Aiming the producer
    ! at tp by assigning k and tb, rather than by editing the body, is what
    ! keeps the body below character-identical to the parent arm's -- and so
    ! keeps mu_turb bitwise. mu_turb(:,:,nk-1) is written here and again by
    ! the walk at k = nk-1, with the same expression on the same inputs.
    ! Only the interior (i,j) is produced: a k-halo plane's i/j edges are
    ! never read, so this needs no load_halo_ijedge.
    k = nk - 1
    tb = tp
        do j = 1, nj-1
        ! Stage 1: velocity gradients + cell metrics, vectorizable over i.
        do i = 1, ni-1
            ivr(i) = 0.25e0 / vol(i,j,k)
            rcr(i) = 0.125e0 * (r(i,j,k)   + r(i+1,j,k)   + r(i,j+1,k)   + r(i+1,j+1,k) &
                              + r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
            rhoc(i) = 0.125e0 * (cons(i,j,k,1)   + cons(i+1,j,k,1)   + cons(i,j+1,k,1)   + cons(i+1,j+1,k,1) &
                               + cons(i,j,k+1,1) + cons(i+1,j,k+1,1) + cons(i,j+1,k+1,1) + cons(i+1,j+1,k+1,1))
            cpc(i) = 0.125e0 * (cp(i,j,k)   + cp(i+1,j,k)   + cp(i,j+1,k)   + cp(i+1,j+1,k) &
                              + cp(i,j,k+1) + cp(i+1,j,k+1) + cp(i,j+1,k+1) + cp(i+1,j+1,k+1))
            muc(i) = 0.125e0 * (mu(i,j,k)   + mu(i+1,j,k)   + mu(i,j+1,k)   + mu(i+1,j+1,k) &
                              + mu(i,j,k+1) + mu(i+1,j,k+1) + mu(i,j+1,k+1) + mu(i+1,j+1,k+1))
            kac(i) = 0.125e0 * (kappa(i,j,k)   + kappa(i+1,j,k)   + kappa(i,j+1,k)   + kappa(i+1,j+1,k) &
                              + kappa(i,j,k+1) + kappa(i+1,j,k+1) + kappa(i,j+1,k+1) + kappa(i+1,j+1,k+1))
            ! --- Vx ---
            f1 = Vx(i,j,k)+Vx(i,j+1,k)+Vx(i,j,k+1)+Vx(i,j+1,k+1)
            f2 = Vx(i+1,j,k)+Vx(i+1,j+1,k)+Vx(i+1,j,k+1)+Vx(i+1,j+1,k+1)
            f3 = Vx(i,j,k)+Vx(i+1,j,k)+Vx(i,j,k+1)+Vx(i+1,j,k+1)
            f4 = Vx(i,j+1,k)+Vx(i+1,j+1,k)+Vx(i,j+1,k+1)+Vx(i+1,j+1,k+1)
            f5 = Vx(i,j,k)+Vx(i+1,j,k)+Vx(i,j+1,k)+Vx(i+1,j+1,k)
            f6 = Vx(i,j,k+1)+Vx(i+1,j,k+1)+Vx(i,j+1,k+1)+Vx(i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVx(i,1) = g1*ivr(i)
            gVx(i,3) = g3*ivr(i)
            gVx(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vr ---
            f1 = Vr(i,j,k)+Vr(i,j+1,k)+Vr(i,j,k+1)+Vr(i,j+1,k+1)
            f2 = Vr(i+1,j,k)+Vr(i+1,j+1,k)+Vr(i+1,j,k+1)+Vr(i+1,j+1,k+1)
            f3 = Vr(i,j,k)+Vr(i+1,j,k)+Vr(i,j,k+1)+Vr(i+1,j,k+1)
            f4 = Vr(i,j+1,k)+Vr(i+1,j+1,k)+Vr(i,j+1,k+1)+Vr(i+1,j+1,k+1)
            f5 = Vr(i,j,k)+Vr(i+1,j,k)+Vr(i,j+1,k)+Vr(i+1,j+1,k)
            f6 = Vr(i,j,k+1)+Vr(i+1,j,k+1)+Vr(i,j+1,k+1)+Vr(i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVr(i,1) = g1*ivr(i)
            gVr(i,3) = g3*ivr(i)
            gVr(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vt ---
            f1 = Vt(i,j,k)+Vt(i,j+1,k)+Vt(i,j,k+1)+Vt(i,j+1,k+1)
            f2 = Vt(i+1,j,k)+Vt(i+1,j+1,k)+Vt(i+1,j,k+1)+Vt(i+1,j+1,k+1)
            f3 = Vt(i,j,k)+Vt(i+1,j,k)+Vt(i,j,k+1)+Vt(i+1,j,k+1)
            f4 = Vt(i,j+1,k)+Vt(i+1,j+1,k)+Vt(i,j+1,k+1)+Vt(i+1,j+1,k+1)
            f5 = Vt(i,j,k)+Vt(i+1,j,k)+Vt(i,j+1,k)+Vt(i+1,j+1,k)
            f6 = Vt(i,j,k+1)+Vt(i+1,j,k+1)+Vt(i,j+1,k+1)+Vt(i+1,j+1,k+1)
            vct(i) = (f1+f2)*0.125e0
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVt(i,1) = g1*ivr(i)
            gVt(i,3) = g3*ivr(i)
            gVt(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
        end do
        ! Stage 2: tau, mixing-length mu_turb, and q -- store with stride-1
        ! per-component writes; vectorizable over i.
        do i = 1, ni-1
            t1 = gVx(i,1)
            t2 = gVr(i,2)
            t3 = gVt(i,3)
            t4 = gVx(i,2) + gVr(i,1)
            t5 = gVx(i,3) + gVt(i,1)
            t6 = gVr(i,3) + gVt(i,2) - vct(i)/rcr(i)
            w1 = gVt(i,2) - gVr(i,3) + vct(i)/rcr(i)
            w2 = gVx(i,3) - gVt(i,1)
            w3 = gVr(i,1) - gVx(i,2)
            vm = sqrt(w1*w1 + w2*w2 + w3*w3)
            visc_lim = 3000e0 * muc(i)
            mut = min(rhoc(i) * xlength(i,j,k) * vm, visc_lim)
            mu_turb(i,j,k) = mut
            fac = (muc(i) + mut) * 0.5e0
            tq(i+1,j+1,1,tb) = t1*fac
            tq(i+1,j+1,2,tb) = t2*fac
            tq(i+1,j+1,3,tb) = t3*fac
            tq(i+1,j+1,4,tb) = t4*fac
            tq(i+1,j+1,5,tb) = t5*fac
            tq(i+1,j+1,6,tb) = t6*fac
            lambda = kac(i) + mut * cpc(i) / Pr_turb
            f1 = T(i,j,k)+T(i,j+1,k)+T(i,j,k+1)+T(i,j+1,k+1)
            f2 = T(i+1,j,k)+T(i+1,j+1,k)+T(i+1,j,k+1)+T(i+1,j+1,k+1)
            f3 = T(i,j,k)+T(i+1,j,k)+T(i,j,k+1)+T(i+1,j,k+1)
            f4 = T(i,j+1,k)+T(i+1,j+1,k)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            f5 = T(i,j,k)+T(i+1,j,k)+T(i,j+1,k)+T(i+1,j+1,k)
            f6 = T(i,j,k+1)+T(i+1,j,k+1)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            tq(i+1,j+1,7,tb) = (f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k) &
                  -f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            tq(i+1,j+1,9,tb) = (f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k) &
                  -f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            tq(i+1,j+1,8,tb) = ((f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k) &
                  -f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))*ivr(i) &
                  + 0.125e0*(f1+f2)/rcr(i)) * (lambda*0.5e0)
        end do
        end do
    tb = 2

    do k = 1, nk
    ! ===== PRODUCE tau/q for cell plane k into slot tb =====
    ! This is set_tau_q_soa's per-(j,k) body, verbatim but for the store
    ! target: the whole point of the fusion is that cell plane k is consumed
    ! by the k-face flux below and by the i/j scan on the next iteration, and
    ! by nothing else, so it never needs to reach memory.
    if (k <= nk-1) then
        do j = 1, nj-1
        ! Stage 1: velocity gradients + cell metrics, vectorizable over i.
        do i = 1, ni-1
            ivr(i) = 0.25e0 / vol(i,j,k)
            rcr(i) = 0.125e0 * (r(i,j,k)   + r(i+1,j,k)   + r(i,j+1,k)   + r(i+1,j+1,k) &
                              + r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
            rhoc(i) = 0.125e0 * (cons(i,j,k,1)   + cons(i+1,j,k,1)   + cons(i,j+1,k,1)   + cons(i+1,j+1,k,1) &
                               + cons(i,j,k+1,1) + cons(i+1,j,k+1,1) + cons(i,j+1,k+1,1) + cons(i+1,j+1,k+1,1))
            cpc(i) = 0.125e0 * (cp(i,j,k)   + cp(i+1,j,k)   + cp(i,j+1,k)   + cp(i+1,j+1,k) &
                              + cp(i,j,k+1) + cp(i+1,j,k+1) + cp(i,j+1,k+1) + cp(i+1,j+1,k+1))
            muc(i) = 0.125e0 * (mu(i,j,k)   + mu(i+1,j,k)   + mu(i,j+1,k)   + mu(i+1,j+1,k) &
                              + mu(i,j,k+1) + mu(i+1,j,k+1) + mu(i,j+1,k+1) + mu(i+1,j+1,k+1))
            kac(i) = 0.125e0 * (kappa(i,j,k)   + kappa(i+1,j,k)   + kappa(i,j+1,k)   + kappa(i+1,j+1,k) &
                              + kappa(i,j,k+1) + kappa(i+1,j,k+1) + kappa(i,j+1,k+1) + kappa(i+1,j+1,k+1))
            ! --- Vx ---
            f1 = Vx(i,j,k)+Vx(i,j+1,k)+Vx(i,j,k+1)+Vx(i,j+1,k+1)
            f2 = Vx(i+1,j,k)+Vx(i+1,j+1,k)+Vx(i+1,j,k+1)+Vx(i+1,j+1,k+1)
            f3 = Vx(i,j,k)+Vx(i+1,j,k)+Vx(i,j,k+1)+Vx(i+1,j,k+1)
            f4 = Vx(i,j+1,k)+Vx(i+1,j+1,k)+Vx(i,j+1,k+1)+Vx(i+1,j+1,k+1)
            f5 = Vx(i,j,k)+Vx(i+1,j,k)+Vx(i,j+1,k)+Vx(i+1,j+1,k)
            f6 = Vx(i,j,k+1)+Vx(i+1,j,k+1)+Vx(i,j+1,k+1)+Vx(i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVx(i,1) = g1*ivr(i)
            gVx(i,3) = g3*ivr(i)
            gVx(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vr ---
            f1 = Vr(i,j,k)+Vr(i,j+1,k)+Vr(i,j,k+1)+Vr(i,j+1,k+1)
            f2 = Vr(i+1,j,k)+Vr(i+1,j+1,k)+Vr(i+1,j,k+1)+Vr(i+1,j+1,k+1)
            f3 = Vr(i,j,k)+Vr(i+1,j,k)+Vr(i,j,k+1)+Vr(i+1,j,k+1)
            f4 = Vr(i,j+1,k)+Vr(i+1,j+1,k)+Vr(i,j+1,k+1)+Vr(i+1,j+1,k+1)
            f5 = Vr(i,j,k)+Vr(i+1,j,k)+Vr(i,j+1,k)+Vr(i+1,j+1,k)
            f6 = Vr(i,j,k+1)+Vr(i+1,j,k+1)+Vr(i,j+1,k+1)+Vr(i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVr(i,1) = g1*ivr(i)
            gVr(i,3) = g3*ivr(i)
            gVr(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vt ---
            f1 = Vt(i,j,k)+Vt(i,j+1,k)+Vt(i,j,k+1)+Vt(i,j+1,k+1)
            f2 = Vt(i+1,j,k)+Vt(i+1,j+1,k)+Vt(i+1,j,k+1)+Vt(i+1,j+1,k+1)
            f3 = Vt(i,j,k)+Vt(i+1,j,k)+Vt(i,j,k+1)+Vt(i+1,j,k+1)
            f4 = Vt(i,j+1,k)+Vt(i+1,j+1,k)+Vt(i,j+1,k+1)+Vt(i+1,j+1,k+1)
            f5 = Vt(i,j,k)+Vt(i+1,j,k)+Vt(i,j+1,k)+Vt(i+1,j+1,k)
            f6 = Vt(i,j,k+1)+Vt(i+1,j,k+1)+Vt(i,j+1,k+1)+Vt(i+1,j+1,k+1)
            vct(i) = (f1+f2)*0.125e0
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVt(i,1) = g1*ivr(i)
            gVt(i,3) = g3*ivr(i)
            gVt(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
        end do
        ! Stage 2: tau, mixing-length mu_turb, and q -- store with stride-1
        ! per-component writes; vectorizable over i.
        do i = 1, ni-1
            t1 = gVx(i,1)
            t2 = gVr(i,2)
            t3 = gVt(i,3)
            t4 = gVx(i,2) + gVr(i,1)
            t5 = gVx(i,3) + gVt(i,1)
            t6 = gVr(i,3) + gVt(i,2) - vct(i)/rcr(i)
            w1 = gVt(i,2) - gVr(i,3) + vct(i)/rcr(i)
            w2 = gVx(i,3) - gVt(i,1)
            w3 = gVr(i,1) - gVx(i,2)
            vm = sqrt(w1*w1 + w2*w2 + w3*w3)
            visc_lim = 3000e0 * muc(i)
            mut = min(rhoc(i) * xlength(i,j,k) * vm, visc_lim)
            mu_turb(i,j,k) = mut
            fac = (muc(i) + mut) * 0.5e0
            tq(i+1,j+1,1,tb) = t1*fac
            tq(i+1,j+1,2,tb) = t2*fac
            tq(i+1,j+1,3,tb) = t3*fac
            tq(i+1,j+1,4,tb) = t4*fac
            tq(i+1,j+1,5,tb) = t5*fac
            tq(i+1,j+1,6,tb) = t6*fac
            lambda = kac(i) + mut * cpc(i) / Pr_turb
            f1 = T(i,j,k)+T(i,j+1,k)+T(i,j,k+1)+T(i,j+1,k+1)
            f2 = T(i+1,j,k)+T(i+1,j+1,k)+T(i+1,j,k+1)+T(i+1,j+1,k+1)
            f3 = T(i,j,k)+T(i+1,j,k)+T(i,j,k+1)+T(i+1,j,k+1)
            f4 = T(i,j+1,k)+T(i+1,j+1,k)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            f5 = T(i,j,k)+T(i+1,j,k)+T(i,j+1,k)+T(i+1,j+1,k)
            f6 = T(i,j,k+1)+T(i+1,j,k+1)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            tq(i+1,j+1,7,tb) = (f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k) &
                  -f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            tq(i+1,j+1,9,tb) = (f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k) &
                  -f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            tq(i+1,j+1,8,tb) = ((f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k) &
                  -f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))*ivr(i) &
                  + 0.125e0*(f1+f2)/rcr(i)) * (lambda*0.5e0)
        end do
        end do
        ! i/j halo edges of this plane, taking the (2*wall-1) scaling on the
        ! way in. Production scales tau_cell in place; doing it here instead
        ! leaves the caller's buffer untouched. The i/j edges are unrelated to
        ! the k seam and keep the parent arm's path.
        if (k == 1) then
            ! Stash cell plane 1 for the k=nk halo, and fill the k=1 halo by
            ! select. Both AFTER the producer, unlike the parent arm's
            ! load_halo_kplane: the wall side of the k=1 halo is minus cell
            ! plane 1, which does not exist until the producer has run.
            do c = 1, 9
            do j = 1, nj-1
            do i = 1, ni-1
                tq(i+1,j+1,c,ts) = tq(i+1,j+1,c,tb)
            end do
            end do
            end do
            do c = 1, 9
            do j = 1, nj-1
            do i = 1, ni-1
                tq(i+1,j+1,c,ta) = wallk1(i,j)*tq(i+1,j+1,c,tp) &
                                 - (1.0e0 - wallk1(i,j))*tq(i+1,j+1,c,tb)
            end do
            end do
            end do
        end if
    else
        ! k = nk: the halo above the last cell plane. Periodic side is cell
        ! plane 1 (stashed); wall side is minus cell plane nk-1, which is
        ! sitting in ta.
        do c = 1, 9
        do j = 1, nj-1
        do i = 1, ni-1
            tq(i+1,j+1,c,tb) = wallnk(i,j)*tq(i+1,j+1,c,ts) &
                             - (1.0e0 - wallnk(i,j))*tq(i+1,j+1,c,ta)
        end do
        end do
        end do
    end if

    ! --- k-face plane k into the rolling pair ---
    do j = 1, nj-1
    do i = 1, ni-1
        tauf(1) = (tq(i+1, j+1, 1, ta) + tq(i+1, j+1, 1, tb)) * 0.5e0
        tauf(2) = (tq(i+1, j+1, 2, ta) + tq(i+1, j+1, 2, tb)) * 0.5e0
        tauf(3) = (tq(i+1, j+1, 3, ta) + tq(i+1, j+1, 3, tb)) * 0.5e0
        tauf(4) = (tq(i+1, j+1, 4, ta) + tq(i+1, j+1, 4, tb)) * 0.5e0
        tauf(5) = (tq(i+1, j+1, 5, ta) + tq(i+1, j+1, 5, tb)) * 0.5e0
        tauf(6) = (tq(i+1, j+1, 6, ta) + tq(i+1, j+1, 6, tb)) * 0.5e0
        qf(1)   = (tq(i+1, j+1, 7, ta) + tq(i+1, j+1, 7, tb)) * 0.5e0
        qf(2)   = (tq(i+1, j+1, 8, ta) + tq(i+1, j+1, 8, tb)) * 0.5e0
        qf(3)   = (tq(i+1, j+1, 9, ta) + tq(i+1, j+1, 9, tb)) * 0.5e0
        Vf(1) = (Vx(i,j,k) + Vx(i+1,j,k) + Vx(i,j+1,k) + Vx(i+1,j+1,k)) * 0.25e0
        Vf(2) = (Vr(i,j,k) + Vr(i+1,j,k) + Vr(i,j+1,k) + Vr(i+1,j+1,k)) * 0.25e0
        Vf(3) = (Vt(i,j,k) + Vt(i+1,j,k) + Vt(i,j+1,k) + Vt(i+1,j+1,k)) * 0.25e0
        rf     = (r(i,j,k)  + r(i+1,j,k)  + r(i,j+1,k)  + r(i+1,j+1,k))  * 0.25e0
        Vabs = Vf(3) + Omega_block * rf
        planes(i,j,1,pb) = tauf(1)*dAk(1,i,j,k) + tauf(4)*dAk(2,i,j,k) + tauf(5)*dAk(3,i,j,k)
        planes(i,j,2,pb) = tauf(4)*dAk(1,i,j,k) + tauf(2)*dAk(2,i,j,k) + tauf(6)*dAk(3,i,j,k)
        planes(i,j,3,pb) = (tauf(5)*dAk(1,i,j,k) + tauf(6)*dAk(2,i,j,k) + tauf(3)*dAk(3,i,j,k)) * rf
        wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
        wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
        wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
        planes(i,j,4,pb) = (wvisc(1)-qf(1))*dAk(1,i,j,k) &
                         + (wvisc(2)-qf(2))*dAk(2,i,j,k) &
                         + (wvisc(3)-qf(3))*dAk(3,i,j,k)
    end do
    end do
    if (k == 2) then
        do j = 1, nj-1
        do i = 1, ni-1
            wfac = 1.0e0 - wallk1(i,j)
            call wall_func_kface(r, dAk, vol, Omega_block, Omega_wallk1_nd(i,j), mu, cons(:,:,:,1), Vx, Vr, Vt, i, j, 1, 1, wf)
            planes(i,j,1,pb) = wallk1(i,j)*planes(i,j,1,pb) + wfac*wf(1)
            planes(i,j,2,pb) = wallk1(i,j)*planes(i,j,2,pb) + wfac*wf(2)
            planes(i,j,3,pb) = wallk1(i,j)*planes(i,j,3,pb) + wfac*wf(3)
            planes(i,j,4,pb) = wallk1(i,j)*planes(i,j,4,pb) + wfac*wf(4)
        end do
        end do
    end if
    if (k == nk-1) then
        do j = 1, nj-1
        do i = 1, ni-1
            wfac = 1.0e0 - wallnk(i,j)
            call wall_func_kface(r, dAk, vol, Omega_block, Omega_wallnk_nd(i,j), mu, cons(:,:,:,1), Vx, Vr, Vt, i, j, nk, -1, wf)
            planes(i,j,1,pb) = wallnk(i,j)*planes(i,j,1,pb) + wfac*wf(1)
            planes(i,j,2,pb) = wallnk(i,j)*planes(i,j,2,pb) + wfac*wf(2)
            planes(i,j,3,pb) = wallnk(i,j)*planes(i,j,3,pb) + wfac*wf(3)
            planes(i,j,4,pb) = wallnk(i,j)*planes(i,j,4,pb) + wfac*wf(4)
        end do
        end do
    end if

    ! --- cell plane kc = k-1: i/j scan, one store per cell ---
    if (k > 1) then
        kc = k - 1
        k_interior = (kc >= 2 .and. kc <= nk-2)
        sa = 2
        sb = 3
        do j = 1, nj
            do i = 1, ni-1
                tauf(1) = (tq(i+1, j, 1, ta) + tq(i+1, j+1, 1, ta)) * 0.5e0
                tauf(2) = (tq(i+1, j, 2, ta) + tq(i+1, j+1, 2, ta)) * 0.5e0
                tauf(3) = (tq(i+1, j, 3, ta) + tq(i+1, j+1, 3, ta)) * 0.5e0
                tauf(4) = (tq(i+1, j, 4, ta) + tq(i+1, j+1, 4, ta)) * 0.5e0
                tauf(5) = (tq(i+1, j, 5, ta) + tq(i+1, j+1, 5, ta)) * 0.5e0
                tauf(6) = (tq(i+1, j, 6, ta) + tq(i+1, j+1, 6, ta)) * 0.5e0
                qf(1)   = (tq(i+1, j, 7, ta) + tq(i+1, j+1, 7, ta)) * 0.5e0
                qf(2)   = (tq(i+1, j, 8, ta) + tq(i+1, j+1, 8, ta)) * 0.5e0
                qf(3)   = (tq(i+1, j, 9, ta) + tq(i+1, j+1, 9, ta)) * 0.5e0
                Vf(1) = (Vx(i,j,kc) + Vx(i+1,j,kc) + Vx(i,j,kc+1) + Vx(i+1,j,kc+1)) * 0.25e0
                Vf(2) = (Vr(i,j,kc) + Vr(i+1,j,kc) + Vr(i,j,kc+1) + Vr(i+1,j,kc+1)) * 0.25e0
                Vf(3) = (Vt(i,j,kc) + Vt(i+1,j,kc) + Vt(i,j,kc+1) + Vt(i+1,j,kc+1)) * 0.25e0
                rf     = (r(i,j,kc)  + r(i+1,j,kc)  + r(i,j,kc+1)  + r(i+1,j,kc+1))  * 0.25e0
                Vabs = Vf(3) + Omega_block * rf
                rows(i,1,sb) = tauf(1)*dAj(1,i,j,kc) + tauf(4)*dAj(2,i,j,kc) + tauf(5)*dAj(3,i,j,kc)
                rows(i,2,sb) = tauf(4)*dAj(1,i,j,kc) + tauf(2)*dAj(2,i,j,kc) + tauf(6)*dAj(3,i,j,kc)
                rows(i,3,sb) = (tauf(5)*dAj(1,i,j,kc) + tauf(6)*dAj(2,i,j,kc) + tauf(3)*dAj(3,i,j,kc)) * rf
                wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
                wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
                wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
                rows(i,4,sb) = (wvisc(1)-qf(1))*dAj(1,i,j,kc) &
                             + (wvisc(2)-qf(2))*dAj(2,i,j,kc) &
                             + (wvisc(3)-qf(3))*dAj(3,i,j,kc)
            end do
            if (j == 2) then
                do i = 1, ni-1
                    wfac = 1.0e0 - wallj1(i,kc)
                    call wall_func_jface(r, dAj, vol, Omega_block, Omega_wallj1_nd(i,kc), &
                        mu, cons(:,:,:,1), Vx, Vr, Vt, i, 1, kc, 1, wf)
                    rows(i,1,sb) = wallj1(i,kc)*rows(i,1,sb) + wfac*wf(1)
                    rows(i,2,sb) = wallj1(i,kc)*rows(i,2,sb) + wfac*wf(2)
                    rows(i,3,sb) = wallj1(i,kc)*rows(i,3,sb) + wfac*wf(3)
                    rows(i,4,sb) = wallj1(i,kc)*rows(i,4,sb) + wfac*wf(4)
                end do
            end if
            if (j == nj-1) then
                do i = 1, ni-1
                    wfac = 1.0e0 - wallnj(i,kc)
                    call wall_func_jface(r, dAj, vol, Omega_block, Omega_wallnj_nd(i,kc), &
                        mu, cons(:,:,:,1), Vx, Vr, Vt, i, nj, kc, -1, wf)
                    rows(i,1,sb) = wallnj(i,kc)*rows(i,1,sb) + wfac*wf(1)
                    rows(i,2,sb) = wallnj(i,kc)*rows(i,2,sb) + wfac*wf(2)
                    rows(i,3,sb) = wallnj(i,kc)*rows(i,3,sb) + wfac*wf(3)
                    rows(i,4,sb) = wallnj(i,kc)*rows(i,4,sb) + wfac*wf(4)
                end do
            end if
            if (j > 1) then
                jc = j - 1
                row_interior = k_interior .and. (jc >= 2 .and. jc <= nj-2)
                do i = 1, ni
                    tauf(1) = (tq(i, jc+1, 1, ta) + tq(i+1, jc+1, 1, ta)) * 0.5e0
                    tauf(2) = (tq(i, jc+1, 2, ta) + tq(i+1, jc+1, 2, ta)) * 0.5e0
                    tauf(3) = (tq(i, jc+1, 3, ta) + tq(i+1, jc+1, 3, ta)) * 0.5e0
                    tauf(4) = (tq(i, jc+1, 4, ta) + tq(i+1, jc+1, 4, ta)) * 0.5e0
                    tauf(5) = (tq(i, jc+1, 5, ta) + tq(i+1, jc+1, 5, ta)) * 0.5e0
                    tauf(6) = (tq(i, jc+1, 6, ta) + tq(i+1, jc+1, 6, ta)) * 0.5e0
                    qf(1)   = (tq(i, jc+1, 7, ta) + tq(i+1, jc+1, 7, ta)) * 0.5e0
                    qf(2)   = (tq(i, jc+1, 8, ta) + tq(i+1, jc+1, 8, ta)) * 0.5e0
                    qf(3)   = (tq(i, jc+1, 9, ta) + tq(i+1, jc+1, 9, ta)) * 0.5e0
                    Vf(1) = (Vx(i,jc,kc) + Vx(i,jc+1,kc) + Vx(i,jc,kc+1) + Vx(i,jc+1,kc+1)) * 0.25e0
                    Vf(2) = (Vr(i,jc,kc) + Vr(i,jc+1,kc) + Vr(i,jc,kc+1) + Vr(i,jc+1,kc+1)) * 0.25e0
                    Vf(3) = (Vt(i,jc,kc) + Vt(i,jc+1,kc) + Vt(i,jc,kc+1) + Vt(i,jc+1,kc+1)) * 0.25e0
                    rf     = (r(i,jc,kc)  + r(i,jc+1,kc)  + r(i,jc,kc+1)  + r(i,jc+1,kc+1))  * 0.25e0
                    Vabs = Vf(3) + Omega_block * rf
                    rows(i,1,1) = tauf(1)*dAi(1,i,jc,kc) + tauf(4)*dAi(2,i,jc,kc) + tauf(5)*dAi(3,i,jc,kc)
                    rows(i,2,1) = tauf(4)*dAi(1,i,jc,kc) + tauf(2)*dAi(2,i,jc,kc) + tauf(6)*dAi(3,i,jc,kc)
                    rows(i,3,1) = (tauf(5)*dAi(1,i,jc,kc) + tauf(6)*dAi(2,i,jc,kc) + tauf(3)*dAi(3,i,jc,kc)) * rf
                    wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
                    wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
                    wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
                    rows(i,4,1) = (wvisc(1)-qf(1))*dAi(1,i,jc,kc) &
                                + (wvisc(2)-qf(2))*dAi(2,i,jc,kc) &
                                + (wvisc(3)-qf(3))*dAi(3,i,jc,kc)
                end do
                wfac = 1.0e0 - walli1(jc,kc)
                call wall_func_iface(r, dAi, vol, Omega_block, Omega_walli1_nd(jc,kc), &
                    mu, cons(:,:,:,1), Vx, Vr, Vt, 1, jc, kc, 1, wf)
                rows(2,1,1) = walli1(jc,kc)*rows(2,1,1) + wfac*wf(1)
                rows(2,2,1) = walli1(jc,kc)*rows(2,2,1) + wfac*wf(2)
                rows(2,3,1) = walli1(jc,kc)*rows(2,3,1) + wfac*wf(3)
                rows(2,4,1) = walli1(jc,kc)*rows(2,4,1) + wfac*wf(4)
                wfac = 1.0e0 - wallni(jc,kc)
                call wall_func_iface(r, dAi, vol, Omega_block, Omega_wallni_nd(jc,kc), &
                    mu, cons(:,:,:,1), Vx, Vr, Vt, ni, jc, kc, -1, wf)
                rows(ni-1,1,1) = wallni(jc,kc)*rows(ni-1,1,1) + wfac*wf(1)
                rows(ni-1,2,1) = wallni(jc,kc)*rows(ni-1,2,1) + wfac*wf(2)
                rows(ni-1,3,1) = wallni(jc,kc)*rows(ni-1,3,1) + wfac*wf(3)
                rows(ni-1,4,1) = wallni(jc,kc)*rows(ni-1,4,1) + wfac*wf(4)
                ! Production's association, not merely its order: its j and
                ! k accumulates are `fvisc = fvisc + hi - lo`, i.e. ((x + hi) - lo),
                ! NOT x + (hi - lo). Grouping the differences instead re-rounds
                ! and costs bitwise agreement (measured: ~1 ulp of the field
                ! scale). Left-to-right, exactly as written below.
                do i = 1, ni-1
                    fvisc(i,jc,kc,1) = (rows(i+1,1,1) - rows(i,1,1)) &
                                     + rows(i,1,sb) - rows(i,1,sa) &
                                     + planes(i,jc,1,pb) - planes(i,jc,1,pa)
                    fvisc(i,jc,kc,2) = (rows(i+1,2,1) - rows(i,2,1)) &
                                     + rows(i,2,sb) - rows(i,2,sa) &
                                     + planes(i,jc,2,pb) - planes(i,jc,2,pa)
                    fvisc(i,jc,kc,3) = (rows(i+1,3,1) - rows(i,3,1)) &
                                     + rows(i,3,sb) - rows(i,3,sa) &
                                     + planes(i,jc,3,pb) - planes(i,jc,3,pa)
                    fvisc(i,jc,kc,4) = (rows(i+1,4,1) - rows(i,4,1)) &
                                     + rows(i,4,sb) - rows(i,4,sa) &
                                     + planes(i,jc,4,pb) - planes(i,jc,4,pa)
                end do
                ! Wall mask and polar source, both finished here while the
                ! row is still in L1. For a row interior in j and k the ONLY
                ! mask its end cells carry is walli1/wallni -- no j- or k-mask
                ! applies -- so those two cells can be masked now, and the
                ! polar loop then covers the whole row unbroken and
                ! unit-stride. That is what removes the i=1/i=ni-1 sheet from
                ! the O(surface) pass, where fvisc could only ever be reached
                ! with stride ni-1 (opt-report: one such block gather-
                ! vectorized, the other not vectorized at all).
                ! Order matches production: i-mask, then polar. The cusp
                ! correction cannot interfere -- it touches only kc=1 and
                ! kc=nk-1, which are not interior rows.
                if (row_interior) then
                    fvisc(1,jc,kc,1) = fvisc(1,jc,kc,1) * walli1(jc,kc)
                    fvisc(1,jc,kc,2) = fvisc(1,jc,kc,2) * walli1(jc,kc)
                    fvisc(1,jc,kc,3) = fvisc(1,jc,kc,3) * walli1(jc,kc)
                    fvisc(1,jc,kc,4) = fvisc(1,jc,kc,4) * walli1(jc,kc)
                    fvisc(ni-1,jc,kc,1) = fvisc(ni-1,jc,kc,1) * wallni(jc,kc)
                    fvisc(ni-1,jc,kc,2) = fvisc(ni-1,jc,kc,2) * wallni(jc,kc)
                    fvisc(ni-1,jc,kc,3) = fvisc(ni-1,jc,kc,3) * wallni(jc,kc)
                    fvisc(ni-1,jc,kc,4) = fvisc(ni-1,jc,kc,4) * wallni(jc,kc)
                    do i = 1, ni-1
                        prhoc    = cons_cell(i, jc, kc, 1)
                        prhorVtc = cons_cell(i, jc, kc, 4)
                        prc = 0.125e0 * ( &
                            r(i,jc,kc) + r(i+1,jc,kc) + r(i,jc+1,kc) + r(i+1,jc+1,kc) + &
                            r(i,jc,kc+1) + r(i+1,jc,kc+1) + r(i,jc+1,kc+1) + r(i+1,jc+1,kc+1))
                        pPc = 0.125e0 * ( &
                            P(i,jc,kc) + P(i+1,jc,kc) + P(i,jc+1,kc) + P(i+1,jc+1,kc) + &
                            P(i,jc,kc+1) + P(i+1,jc,kc+1) + P(i,jc+1,kc+1) + P(i+1,jc+1,kc+1))
                        pVtc = prhorVtc / (prhoc * prc)
                        fvisc(i,jc,kc,2) = fvisc(i,jc,kc,2) &
                            + vol(i,jc,kc) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
                    end do
                end if
            end if
            stmp = sa
            sa = sb
            sb = stmp
        end do
    end if
    stmp = pa
    pa = pb
    pb = stmp
    stmp = ta
    ta = tb
    tb = stmp
    end do

    call zero_wall_fvisc_border(fvisc, walli1, wallj1, wallk1, wallni, wallnj, wallnk, ni, nj, nk)

    ! ===== Polar source on the boundary shell, AFTER the wall zeroing =====
    ! Interior cells took their polar source inside the fused store above; the
    ! shell could not. Production adds the polar source after the zeroing pass
    ! because it is a geometric source, not viscous content, so the wall mask
    ! must not eat it -- and the fused store runs before that pass.
    !
    ! The four blocks below partition the shell so every cell in it is visited
    ! EXACTLY once. This is stricter than the zeroing loops need to be: those
    ! may overlap at edges and corners because a repeated multiply by the same
    ! mask is harmless, but a repeated ADD is not. Each high-face block is also
    ! guarded, so a degenerate dimension (one cell plane, where the low and
    ! high faces are the same cells) does not double-add either.
    do j = 1, nj-1
    do i = 1, ni-1
        prhoc    = cons_cell(i, j, 1, 1)
        prhorVtc = cons_cell(i, j, 1, 4)
        prc = 0.125e0 * ( &
            r(i,j,1) + r(i+1,j,1) + r(i,j+1,1) + r(i+1,j+1,1) + &
            r(i,j,1+1) + r(i+1,j,1+1) + r(i,j+1,1+1) + r(i+1,j+1,1+1))
        pPc = 0.125e0 * ( &
            P(i,j,1) + P(i+1,j,1) + P(i,j+1,1) + P(i+1,j+1,1) + &
            P(i,j,1+1) + P(i+1,j,1+1) + P(i,j+1,1+1) + P(i+1,j+1,1+1))
        pVtc = prhorVtc / (prhoc * prc)
        fvisc(i,j,1,2) = fvisc(i,j,1,2) &
            + vol(i,j,1) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
    end do
    end do
    if (nk-1 > 1) then
        do j = 1, nj-1
        do i = 1, ni-1
            prhoc    = cons_cell(i, j, nk-1, 1)
            prhorVtc = cons_cell(i, j, nk-1, 4)
            prc = 0.125e0 * ( &
                r(i,j,nk-1) + r(i+1,j,nk-1) + r(i,j+1,nk-1) + r(i+1,j+1,nk-1) + &
                r(i,j,nk-1+1) + r(i+1,j,nk-1+1) + r(i,j+1,nk-1+1) + r(i+1,j+1,nk-1+1))
            pPc = 0.125e0 * ( &
                P(i,j,nk-1) + P(i+1,j,nk-1) + P(i,j+1,nk-1) + P(i+1,j+1,nk-1) + &
                P(i,j,nk-1+1) + P(i+1,j,nk-1+1) + P(i,j+1,nk-1+1) + P(i+1,j+1,nk-1+1))
            pVtc = prhorVtc / (prhoc * prc)
            fvisc(i,j,nk-1,2) = fvisc(i,j,nk-1,2) &
                + vol(i,j,nk-1) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
        end do
        end do
    end if
    do k = 2, nk-2
    do i = 1, ni-1
        prhoc    = cons_cell(i, 1, k, 1)
        prhorVtc = cons_cell(i, 1, k, 4)
        prc = 0.125e0 * ( &
            r(i,1,k) + r(i+1,1,k) + r(i,1+1,k) + r(i+1,1+1,k) + &
            r(i,1,k+1) + r(i+1,1,k+1) + r(i,1+1,k+1) + r(i+1,1+1,k+1))
        pPc = 0.125e0 * ( &
            P(i,1,k) + P(i+1,1,k) + P(i,1+1,k) + P(i+1,1+1,k) + &
            P(i,1,k+1) + P(i+1,1,k+1) + P(i,1+1,k+1) + P(i+1,1+1,k+1))
        pVtc = prhorVtc / (prhoc * prc)
        fvisc(i,1,k,2) = fvisc(i,1,k,2) &
            + vol(i,1,k) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
    end do
    end do
    if (nj-1 > 1) then
        do k = 2, nk-2
        do i = 1, ni-1
            prhoc    = cons_cell(i, nj-1, k, 1)
            prhorVtc = cons_cell(i, nj-1, k, 4)
            prc = 0.125e0 * ( &
                r(i,nj-1,k) + r(i+1,nj-1,k) + r(i,nj-1+1,k) + r(i+1,nj-1+1,k) + &
                r(i,nj-1,k+1) + r(i+1,nj-1,k+1) + r(i,nj-1+1,k+1) + r(i+1,nj-1+1,k+1))
            pPc = 0.125e0 * ( &
                P(i,nj-1,k) + P(i+1,nj-1,k) + P(i,nj-1+1,k) + P(i+1,nj-1+1,k) + &
                P(i,nj-1,k+1) + P(i+1,nj-1,k+1) + P(i,nj-1+1,k+1) + P(i+1,nj-1+1,k+1))
            pVtc = prhorVtc / (prhoc * prc)
            fvisc(i,nj-1,k,2) = fvisc(i,nj-1,k,2) &
                + vol(i,nj-1,k) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
        end do
        end do
    end if

end subroutine set_visc_force_tqf_noij
