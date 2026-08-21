! EXPERIMENTAL: viscous fusion whose halo source is six SURFACE buffers.
!
! set_visc_force_tqf produces interior tau/q inside its own k walk and reads
! only the boundary shell -- but it reads that shell out of a full-volume
! (ni+1, nj+1, nk+1, 9) array, because that is where phase 1's "+edge" fill and
! exchange_halos put it. That one dependency is what makes phase 1 O(volume)
! for an O(surface) need, and it is why set_visc_force_tqf_selfk had to delete
! the exchange -- restricting itself to k-self-periodic blocks -- to get free
! of it.
!
! This arm takes Block.tau_q_faces instead: six (a, 9, b, 2) buffers, layer 2
! holding the halo value. Phase 1 becomes set_tau_q_faces, an O(surface) pass,
! and the exchange becomes exchange_faces. Nothing about the kernel depends on
! the block's topology any more:
!
!   * no seam select and no wall mask on the halo path -- set_tau_q_faces has
!     already applied (2*wall-1) to layer 2, so a wall face arrives as -edge
!     and a permeable one as +edge;
!   * no pre-pass and no stashed plane 1, so tq is back to TWO rolling slots
!     rather than the four set_visc_force_tqf_selfk needed;
!   * no i_perk, no self-pairing check, no coverage check. A block connected to
!     neighbours is served exactly as one periodic to itself.
!
! The i/j edge reads are also unit-stride now. Reading tau_cell(1, j+1, kh, c)
! strides (ni+1) floats per element and fetches a cache line per 4-byte value;
! f_i1(j, c, k, 2) is contiguous in j. That is worth about 1.9% by the
! fused_noij control, so it is the smaller half of the story -- the point is
! phase 1's traffic and the topology independence.
!
! NOT PRODUCTION-READY: like its parents, THE CUSP SEAM CORRECTION IS NOT
! APPLIED. Note though that f_k1 and f_knk hold cell planes 1 and nk-1 at once,
! which is exactly what that correction needs and what the rolling pair could
! never give it.

module viscous_faces_helpers
    implicit none
    private
    public :: load_kface, load_ijedge_faces

contains

    ! One k-direction halo plane, from its face buffer into a rolling tq slot.
    ! No mask: the face buffer's layer 2 already carries (2*wall-1), applied
    ! once by set_tau_q_faces rather than on every read.
    subroutine load_kface(fk, plane, ni, nj)
        implicit none
        integer, intent(in) :: ni, nj
        real, intent(in) :: fk(ni-1, 9, nj-1, 2)
        real, intent(inout) :: plane(ni+1, nj+1, 9)
        integer :: i, j, c
        do j = 1, nj-1
        do c = 1, 9
        do i = 1, ni-1
            plane(i+1,j+1,c) = fk(i,c,j,2)
        end do
        end do
        end do
    end subroutine load_kface

    ! The four i/j halo edges of one cell plane, from their face buffers.
    ! Both reads are unit-stride in the face buffer; the i-edge WRITE into the
    ! halo-shaped plane still strides (ni+1), which is the plane's layout, not
    ! this buffer's.
    subroutine load_ijedge_faces(f_i1, f_ini, f_j1, f_jnj, plane, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: k, ni, nj, nk
        real, intent(in) :: f_i1(nj-1, 9, nk-1, 2), f_ini(nj-1, 9, nk-1, 2)
        real, intent(in) :: f_j1(ni-1, 9, nk-1, 2), f_jnj(ni-1, 9, nk-1, 2)
        real, intent(inout) :: plane(ni+1, nj+1, 9)
        integer :: i, j, c
        do c = 1, 9
        do j = 1, nj-1
            plane(1,j+1,c)    = f_i1(j,c,k,2)
            plane(ni+1,j+1,c) = f_ini(j,c,k,2)
        end do
        end do
        do c = 1, 9
        do i = 1, ni-1
            plane(i+1,1,c)    = f_j1(i,c,k,2)
            plane(i+1,nj+1,c) = f_jnj(i,c,k,2)
        end do
        end do
    end subroutine load_ijedge_faces

end module viscous_faces_helpers

subroutine set_visc_force_tqf_faces( &
    cons, cons_cell, vol, dAi, dAj, dAk, &
    Omega_block, r, mu, P, P_offset, &
    fvisc, &
    Vx, Vr, Vt, &
    T, cp, kappa, Pr_turb, xlength, &
    mu_turb, &
    f_i1, f_ini, f_j1, f_jnj, f_k1, f_knk, &
    tq, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    Omega_walli1_nd, Omega_wallj1_nd, Omega_wallk1_nd, &
    Omega_wallni_nd, Omega_wallnj_nd, Omega_wallnk_nd, &
    i_cusp_start, i_cusp_end, &
    kb, ni, nj, nk)

    use viscous_helpers
    use viscous_faces_helpers
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
    ! HALO SOURCE, and the only one: six surface buffers, layer 2 of each
    ! holding the halo value with (2*wall-1) already applied by
    ! set_tau_q_faces and the periodic part already overwritten by
    ! exchange_faces. The kernel neither writes them nor reads layer 1.
    real, intent(in) :: f_i1(nj-1, 9, nk-1, 2), f_ini(nj-1, 9, nk-1, 2)
    real, intent(in) :: f_j1(ni-1, 9, nk-1, 2), f_jnj(ni-1, 9, nk-1, 2)
    real, intent(in) :: f_k1(ni-1, 9, nj-1, 2), f_knk(ni-1, 9, nj-1, 2)
    real, intent(inout) :: planes(ni, nj, 4, 2)
    real, intent(inout) :: rows(ni, 4, 3)
    ! Rolling tau/q CELL-plane pair, halo-indexed in i and j exactly as
    ! tau_cell is, with slots 1-6 tau and 7-9 q. Slot ta holds cell plane
    ! k-1, slot tb cell plane k.
    real, intent(inout) :: tq(ni+1, nj+1, 9, 2)
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
    integer :: ta, tb
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
    if (kb < 1 .or. i_cusp_start < 0 .or. i_cusp_end < 0) return

    pa = 1
    pb = 2
    ta = 1
    tb = 2

    do k = 1, nk
    ! ===== PRODUCE tau/q for cell plane k into slot tb =====
    ! This is set_tau_q_soa's per-(j,k) body, verbatim but for the store
    ! target: the whole point of the fusion is that cell plane k is consumed
    ! by the k-face flux below and by the i/j scan on the next iteration, and
    ! by nothing else, so it never needs to reach memory.
    if (k == 1) then
        call load_kface(f_k1, tq(1,1,1,ta), ni, nj)
    end if
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
        ! i/j halo edges of this plane, straight out of their face buffers.
        ! No scaling on the way in and no mask argument: set_tau_q_faces
        ! applied (2*wall-1) once, when it wrote layer 2.
        call load_ijedge_faces(f_i1, f_ini, f_j1, f_jnj, &
            tq(1,1,1,tb), k, ni, nj, nk)
    else
        call load_kface(f_knk, tq(1,1,1,tb), ni, nj)
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

end subroutine set_visc_force_tqf_faces
