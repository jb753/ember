! EXPERIMENTAL, bench only: set_visc_force_tqf with the velocity and
! transport streams PACKED component-first.
!
! The fused loop is stream-limited, not bandwidth-limited: collapsing three
! arrays into one is worth 7.86% for mu/cp/kappa and 5.54% for Vx/Vr/Vt, and
! ~82% of each is not explained by the bytes it also saves. This arm keeps the
! bytes and takes only the streams -- vel(3,i,j,k) and trans(3,i,j,k) hold the
! same values in the layout dAi(3,i,j-1,k-1) already uses, so all three
! components of a node arrive together instead of from three separate streams.
!
! Six streams become two in the hot loops. Nothing else changes: same
! arithmetic, same results, same bytes read.
!
! The caller packs them in numpy at setup, outside the timed window. That is
! the point of doing this as a bench arm first -- a real integration would
! change Block._Vxrt_nd_uninit's axis order and give the transport properties
! a stacked field, which touches Vx_nd/Vr_nd/Vt_nd's contiguity and every
! other consumer of them.
!
! EXPERIMENTAL: tau/q evaluation fused into set_visc_force.
!
! set_tau_q_soa writes 9 full-volume float32 fields that set_visc_force then
! streams straight back in -- ~72 B/cell of pure DRAM round trip -- and both
! kernels independently stream dAi/dAj/dAk (36 B/cell) and the nodal
! cons/r/Vx/Vr/Vt. Fusing them drops all of that second pass.
!
! It is only tractable because the adopted set_visc_force is a single walk over
! k face planes: at face plane k it consumes tau/q cell planes k-1 and k and
! NOTHING else. That is a producer/consumer window two planes deep, so tau/q
! can be produced inside the walk into a rolling pair (`tq`) instead of read
! from a full-volume buffer. Both halves are the production kernels verbatim --
! set_tau_q_soa's two row loops as the producer, set_visc_force's face loops as
! the consumer -- with only the tau/q store/load target changed.
!
! tau_cell/q_cell stay in the signature but are demoted to HALO SOURCE ONLY:
! the kernel reads the slots exchange_halos fills and never touches the
! interior. That models the real design, where phase 1 shrinks to an O(surface)
! boundary kernel run before the exchange. A side benefit is that the
! (2*wall-1) halo scaling moves out of the caller's buffer and into the copy
! into `tq`, so unlike production this kernel does not mutate its own input and
! is idempotent.
!
! NOT PRODUCTION-READY, and not only because it is unmeasured:
!
!   * THE CUSP SEAM CORRECTION IS NOT APPLIED. It needs tau/q at cell planes 1
!     and nk-1 simultaneously, and plane 1 is long gone from the rolling pair
!     by the time the walk reaches nk. Supporting it means stashing plane 1 and
!     a kface_flow variant taking explicit planes -- real work, no effect on
!     the timing signal (it is O(surface)), and not worth doing before the
!     fusion is known to be worth anything. bench/visc_arms.py refuses to build
!     a cusped case against this arm so it cannot be used wrongly by accident.
!   * The halo slots are assumed already exchanged.
!
! EXACTNESS, measured (bench/visc_arms.py --ncell 300000, swirled state):
!   mu_turb  BITWISE -- the producer is set_tau_q_soa expression for
!            expression, and this proves it: same gradients, same mixing
!            length, therefore the same tau and q.
!   fvisc    ~35 ulp of the field scale at production flags, collapsing to
!            0.03 ulp with -fno-associative-math -ffp-contract=off. So it is
!            GCC reassociating a much larger loop body differently, not a
!            logic error -- and the deviation is spread through the interior
!            (93% of differing entries strictly interior, max at an interior
!            cell), where halo values cannot reach, which rules out the
!            boundary handling independently.
!   * `tq` is 2*(ni+1)*(nj+1)*9 floats -- 1.3 MB at the 1M bench shape, so L3-
!     rather than L2-resident per rank. That is the main risk to the whole
!     idea: the fused inner loop roughly doubles the concurrent stream count,
!     the failure mode behind the rejected j-panel tiling and the IRS k-solve
!     merge.

module viscous_packed_helpers
    implicit none
    private
    public :: load_halo_kplane, load_halo_ijedge

contains

    ! One k-direction halo plane into a rolling slot, scaled by (2*wall-1).
    ! Only the interior (i,j) of a k-halo plane is ever read (the k-face flux
    ! loop), so the i/j edges of this plane are left alone.
    subroutine load_halo_kplane(tau_cell, q_cell, plane, wall, kh, ni, nj, nk)
        implicit none
        integer, intent(in) :: kh, ni, nj, nk
        real, intent(in) :: tau_cell(ni+1, nj+1, nk+1, 6)
        real, intent(in) :: q_cell(ni+1, nj+1, nk+1, 3)
        real, intent(inout) :: plane(ni+1, nj+1, 9)
        real, intent(in) :: wall(ni-1, nj-1)
        integer :: i, j, c
        do c = 1, 6
        do j = 1, nj-1
        do i = 1, ni-1
            plane(i+1,j+1,c) = tau_cell(i+1,j+1,kh,c) * (2.0e0*wall(i,j) - 1.0e0)
        end do
        end do
        end do
        do c = 1, 3
        do j = 1, nj-1
        do i = 1, ni-1
            plane(i+1,j+1,6+c) = q_cell(i+1,j+1,kh,c) * (2.0e0*wall(i,j) - 1.0e0)
        end do
        end do
        end do
    end subroutine load_halo_kplane

    ! The four i/j halo edges of one cell plane, scaled by (2*wall-1). kh is
    ! the halo k index of the plane, kc the cell k index the wall masks use.
    ! Corners are never read (the i-face row reads i-halo at interior j, the
    ! j-face row j-halo at interior i) and are left alone.
    subroutine load_halo_ijedge(tau_cell, q_cell, plane, &
        walli1, wallni, wallj1, wallnj, kh, kc, ni, nj, nk)
        implicit none
        integer, intent(in) :: kh, kc, ni, nj, nk
        real, intent(in) :: tau_cell(ni+1, nj+1, nk+1, 6)
        real, intent(in) :: q_cell(ni+1, nj+1, nk+1, 3)
        real, intent(inout) :: plane(ni+1, nj+1, 9)
        real, intent(in) :: walli1(nj-1, nk-1), wallni(nj-1, nk-1)
        real, intent(in) :: wallj1(ni-1, nk-1), wallnj(ni-1, nk-1)
        integer :: i, j, c
        do c = 1, 6
        do j = 1, nj-1
            plane(1,j+1,c)    = tau_cell(1,j+1,kh,c)    * (2.0e0*walli1(j,kc) - 1.0e0)
            plane(ni+1,j+1,c) = tau_cell(ni+1,j+1,kh,c) * (2.0e0*wallni(j,kc) - 1.0e0)
        end do
        end do
        do c = 1, 3
        do j = 1, nj-1
            plane(1,j+1,6+c)    = q_cell(1,j+1,kh,c)    * (2.0e0*walli1(j,kc) - 1.0e0)
            plane(ni+1,j+1,6+c) = q_cell(ni+1,j+1,kh,c) * (2.0e0*wallni(j,kc) - 1.0e0)
        end do
        end do
        do c = 1, 6
        do i = 1, ni-1
            plane(i+1,1,c)    = tau_cell(i+1,1,kh,c)    * (2.0e0*wallj1(i,kc) - 1.0e0)
            plane(i+1,nj+1,c) = tau_cell(i+1,nj+1,kh,c) * (2.0e0*wallnj(i,kc) - 1.0e0)
        end do
        end do
        do c = 1, 3
        do i = 1, ni-1
            plane(i+1,1,6+c)    = q_cell(i+1,1,kh,c)    * (2.0e0*wallj1(i,kc) - 1.0e0)
            plane(i+1,nj+1,6+c) = q_cell(i+1,nj+1,kh,c) * (2.0e0*wallnj(i,kc) - 1.0e0)
        end do
        end do
    end subroutine load_halo_ijedge

end module viscous_packed_helpers


subroutine set_visc_force_tqf_packed( &
    cons, cons_cell, vol, dAi, dAj, dAk, &
    Omega_block, r, mu, P, P_offset, &
    fvisc, &
    Vx, Vr, Vt, &
    vel, trans, &
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
    use viscous_packed_helpers
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
    ! Component-FIRST packings of the velocities and the transport
    ! properties, the layout dAi/dAj/dAk already use. The hot loops read
    ! these; Vx/Vr/Vt and mu stay only because the wall functions take
    ! whole arrays.
    real, intent(in) :: vel(3, ni, nj, nk)
    real, intent(in) :: trans(3, ni, nj, nk)
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
    ! cp and kappa are read only through `trans` now; they stay in the
    ! signature so this arm shares one kwargs dict with its parent.
    if (kb < 1 .or. i_cusp_start < 0 .or. i_cusp_end < 0 &
        .or. cp(1,1,1) /= cp(1,1,1) .or. kappa(1,1,1) /= kappa(1,1,1)) return

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
        call load_halo_kplane(tau_cell, q_cell, tq(1,1,1,ta), wallk1, 1, ni, nj, nk)
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
            cpc(i) = 0.125e0 * (trans(2,i,j,k)   + trans(2,i+1,j,k)   + trans(2,i,j+1,k)   + trans(2,i+1,j+1,k) &
                              + trans(2,i,j,k+1) + trans(2,i+1,j,k+1) + trans(2,i,j+1,k+1) + trans(2,i+1,j+1,k+1))
            muc(i) = 0.125e0 * (trans(1,i,j,k)   + trans(1,i+1,j,k)   + trans(1,i,j+1,k)   + trans(1,i+1,j+1,k) &
                              + trans(1,i,j,k+1) + trans(1,i+1,j,k+1) + trans(1,i,j+1,k+1) + trans(1,i+1,j+1,k+1))
            kac(i) = 0.125e0 * (trans(3,i,j,k)   + trans(3,i+1,j,k)   + trans(3,i,j+1,k)   + trans(3,i+1,j+1,k) &
                              + trans(3,i,j,k+1) + trans(3,i+1,j,k+1) + trans(3,i,j+1,k+1) + trans(3,i+1,j+1,k+1))
            ! --- Vx ---
            f1 = vel(1,i,j,k)+vel(1,i,j+1,k)+vel(1,i,j,k+1)+vel(1,i,j+1,k+1)
            f2 = vel(1,i+1,j,k)+vel(1,i+1,j+1,k)+vel(1,i+1,j,k+1)+vel(1,i+1,j+1,k+1)
            f3 = vel(1,i,j,k)+vel(1,i+1,j,k)+vel(1,i,j,k+1)+vel(1,i+1,j,k+1)
            f4 = vel(1,i,j+1,k)+vel(1,i+1,j+1,k)+vel(1,i,j+1,k+1)+vel(1,i+1,j+1,k+1)
            f5 = vel(1,i,j,k)+vel(1,i+1,j,k)+vel(1,i,j+1,k)+vel(1,i+1,j+1,k)
            f6 = vel(1,i,j,k+1)+vel(1,i+1,j,k+1)+vel(1,i,j+1,k+1)+vel(1,i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVx(i,1) = g1*ivr(i)
            gVx(i,3) = g3*ivr(i)
            gVx(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vr ---
            f1 = vel(2,i,j,k)+vel(2,i,j+1,k)+vel(2,i,j,k+1)+vel(2,i,j+1,k+1)
            f2 = vel(2,i+1,j,k)+vel(2,i+1,j+1,k)+vel(2,i+1,j,k+1)+vel(2,i+1,j+1,k+1)
            f3 = vel(2,i,j,k)+vel(2,i+1,j,k)+vel(2,i,j,k+1)+vel(2,i+1,j,k+1)
            f4 = vel(2,i,j+1,k)+vel(2,i+1,j+1,k)+vel(2,i,j+1,k+1)+vel(2,i+1,j+1,k+1)
            f5 = vel(2,i,j,k)+vel(2,i+1,j,k)+vel(2,i,j+1,k)+vel(2,i+1,j+1,k)
            f6 = vel(2,i,j,k+1)+vel(2,i+1,j,k+1)+vel(2,i,j+1,k+1)+vel(2,i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVr(i,1) = g1*ivr(i)
            gVr(i,3) = g3*ivr(i)
            gVr(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vt ---
            f1 = vel(3,i,j,k)+vel(3,i,j+1,k)+vel(3,i,j,k+1)+vel(3,i,j+1,k+1)
            f2 = vel(3,i+1,j,k)+vel(3,i+1,j+1,k)+vel(3,i+1,j,k+1)+vel(3,i+1,j+1,k+1)
            f3 = vel(3,i,j,k)+vel(3,i+1,j,k)+vel(3,i,j,k+1)+vel(3,i+1,j,k+1)
            f4 = vel(3,i,j+1,k)+vel(3,i+1,j+1,k)+vel(3,i,j+1,k+1)+vel(3,i+1,j+1,k+1)
            f5 = vel(3,i,j,k)+vel(3,i+1,j,k)+vel(3,i,j+1,k)+vel(3,i+1,j+1,k)
            f6 = vel(3,i,j,k+1)+vel(3,i+1,j,k+1)+vel(3,i,j+1,k+1)+vel(3,i+1,j+1,k+1)
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
        ! leaves the caller's buffer untouched.
        call load_halo_ijedge(tau_cell, q_cell, tq(1,1,1,tb), &
            walli1, wallni, wallj1, wallnj, k+1, k, ni, nj, nk)
    else
        call load_halo_kplane(tau_cell, q_cell, tq(1,1,1,tb), wallnk, nk+1, ni, nj, nk)
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
        Vf(1) = (vel(1,i,j,k) + vel(1,i+1,j,k) + vel(1,i,j+1,k) + vel(1,i+1,j+1,k)) * 0.25e0
        Vf(2) = (vel(2,i,j,k) + vel(2,i+1,j,k) + vel(2,i,j+1,k) + vel(2,i+1,j+1,k)) * 0.25e0
        Vf(3) = (vel(3,i,j,k) + vel(3,i+1,j,k) + vel(3,i,j+1,k) + vel(3,i+1,j+1,k)) * 0.25e0
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
                Vf(1) = (vel(1,i,j,kc) + vel(1,i+1,j,kc) + vel(1,i,j,kc+1) + vel(1,i+1,j,kc+1)) * 0.25e0
                Vf(2) = (vel(2,i,j,kc) + vel(2,i+1,j,kc) + vel(2,i,j,kc+1) + vel(2,i+1,j,kc+1)) * 0.25e0
                Vf(3) = (vel(3,i,j,kc) + vel(3,i+1,j,kc) + vel(3,i,j,kc+1) + vel(3,i+1,j,kc+1)) * 0.25e0
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
                    Vf(1) = (vel(1,i,jc,kc) + vel(1,i,jc+1,kc) + vel(1,i,jc,kc+1) + vel(1,i,jc+1,kc+1)) * 0.25e0
                    Vf(2) = (vel(2,i,jc,kc) + vel(2,i,jc+1,kc) + vel(2,i,jc,kc+1) + vel(2,i,jc+1,kc+1)) * 0.25e0
                    Vf(3) = (vel(3,i,jc,kc) + vel(3,i,jc+1,kc) + vel(3,i,jc,kc+1) + vel(3,i,jc+1,kc+1)) * 0.25e0
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

end subroutine set_visc_force_tqf_packed
