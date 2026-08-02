! =====================================================================
! set_residual_multall -- the FAITHFUL multall/multall residual design.
! Benchmark-only companion; nothing calls it in production.
!
! Section 25 measured multall's five-pass structure and rejected it, but
! that measurement was compromised: both arms there inherited two ember
! design choices that a real five-pass code would never make. This arm
! removes both, so it prices multall's design rather than a hybrid.
!
! DIFFERENCE 1 -- PRIMITIVES ARE STAGED PER NODE, NOT DERIVED PER FACE.
! multall's SET_FLUX block-copy loop (multall-p-2_3_1.f:6357) computes VX,
! VR, VT, WT, ROWT, HO and P once per node per timestep, so its per-
! variable flux loops contain no divides at all (:6825):
!
!     AVGVX        = VX(I,J,K)+VX(I,J+1,K)+VX(I,J+1,K+1)+VX(I,J,K+1)
!     FLUXI(I,J,K) = 0.25*(AVGVX*FIMAS(I,J,K) + AIX(N)*AVGPI(I,J,K))
!
! Production ember does the opposite by deliberate choice: section 20's
! `consa` derives Vx/Vr/r*Vt from cons inline at every face corner, paying
! ~12x redundant reciprocals (each node is a corner of ~12 faces) to drop
! three streamed nodal fields. That is the right trade for ONE fused
! bandwidth-bound pass and a terrible one under a five-way split -- which
! is exactly what section 25's `split` arm measured (+57 to +104%).
!
! So this arm reads the nodal vx/vr/ho arrays ember already carries, and
! stages the two it does not:
!
!     rowt(i,j,k) = rho*Vt_rel = cons4/r - Omega*cons1*r   (1 divide/node)
!     rvt (i,j,k) = r*Vt       = r*vt                      (1 mul/node)
!
! After that stage-1 does no divides (mf = cons2, cons3, rowt) and every
! stage-2 pass is a four-point average of ONE nodal array times a staged
! mass flux, plus a pressure term -- multall's inner loop exactly.
!
! It also collapses the four per-component helpers to two, since vx, vr,
! rvt and ho are now interchangeable arguments to a single `q` dummy. That
! is not a coincidence: it is what multall's uniformity buys.
!
! DIFFERENCE 2 -- GEOMETRY IS SoA, NOT AoS. multall stores face areas as
! separate AIX/AIR/AIT (AJ*, AK*) arrays, so its axial-momentum pass reads
! only AIX/AJX/AKX -- one component per direction. Ember's dA(3,i,j,k) is
! component-first, so touching one axis pulls the line holding all three,
! and section 25's traffic model charged every pass the full 36 B/cell for
! that reason. That is a cost of ember's LAYOUT, not of the five-pass idea.
! This arm takes nine separate component arrays.
!
! Splitting them is free in a real port and is NOT cheating the benchmark:
! face areas are grid geometry, built once at startup, never rebuilt per
! step. The harness therefore allocates the nine arrays outside the timed
! region, exactly as multall's AIX/AIR/AIT are set up once in FIND_AREAS.
!
! NOT REPRODUCED: multall also stages the face-averaged pressures
! AVGPI/AVGPJ/AVGPK (multall-p-2_3_1.f:6455), three more full face volumes,
! so its momentum passes do not re-average P either. Skipped deliberately:
! it would save the m=2/m=3 passes one four-point average each while adding
! three face-volume streams, and leaving it out keeps this arm a clean
! isolation of the two differences above. If the arm wins, that is the next
! thing to add.
!
! NUMERICS: this arm is NOT bitwise against production anywhere, and cannot
! be. Production forms r*Vt as cons4/cons1 per corner; here it is r*vt
! staged per node. Same quantity, different rounding, and mf3 likewise
! moves its divide from the corner to the node. Expect agreement at the
! ~1e-6-of-scale level, as section 25's arms show.
! =====================================================================

module residual_multall_helpers
    implicit none
    private
    public :: stage_primitives
    public :: tbmflux_iface_row, tbmflux_jface_row, tbmflux_kface_plane
    public :: tbp_iface_row, tbp_jface_row, tbp_kface_plane
    public :: tbrp_iface_row, tbrp_jface_row, tbrp_kface_plane

contains

    ! ---- stage 0: the two nodal primitives ember does not already hold --
    ! One divide and one multiply per NODE, replacing production's four
    ! reciprocals plus four divides per FACE CORNER. This is the whole
    ! point of the multall design.
    subroutine stage_primitives(cons, r, vt, Omega, rowt, rvt, ni, nj, nk)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: r(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: Omega
        real, intent(inout) :: rowt(ni, nj, nk), rvt(ni, nj, nk)
        integer :: i, j, k
        do k = 1, nk
        do j = 1, nj
        !DIR$ IVDEP
        do i = 1, ni
            rowt(i,j,k) = cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)
            rvt(i,j,k)  = r(i,j,k)*vt(i,j,k)
        end do
        end do
        end do
    end subroutine stage_primitives


    ! ---- stage 1: face mass flux (multall's FIMAS/FJMAS/FKMAS) ----------
    ! mf = (rho*Vx, rho*Vr, rho*Vt_rel) = (cons2, cons3, rowt): three plain
    ! nodal reads, no arithmetic beyond the corner average. The wall mask
    ! weights mf and only mf, so it folds in exactly here and the stage-2
    ! helpers need no wall handling.

    pure subroutine tbmflux_iface_row(cons, rowt, dA1, dA2, dA3, wall_lo, wall_hi, mrow, j, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: rowt(ni, nj, nk)
        real, intent(in) :: dA1(ni, nj-1, nk-1), dA2(ni, nj-1, nk-1), dA3(ni, nj-1, nk-1)
        real, intent(in) :: wall_lo, wall_hi
        real, intent(inout) :: mrow(ni)
        integer :: i
        real :: mf1, mf2, mf3, w

        w = 0.25e0*wall_lo
        call accum_mf(1, w, mf1, mf2, mf3)
        mrow(1) = mf1*dA1(1,j,k) + mf2*dA2(1,j,k) + mf3*dA3(1,j,k)

        !DIR$ IVDEP
        do i = 2, ni-1
            call accum_mf(i, 0.25e0, mf1, mf2, mf3)
            mrow(i) = mf1*dA1(i,j,k) + mf2*dA2(i,j,k) + mf3*dA3(i,j,k)
        end do

        w = 0.25e0*wall_hi
        call accum_mf(ni, w, mf1, mf2, mf3)
        mrow(ni) = mf1*dA1(ni,j,k) + mf2*dA2(ni,j,k) + mf3*dA3(ni,j,k)

    contains
        pure subroutine accum_mf(i, w, mf1, mf2, mf3)
            integer, intent(in) :: i
            real, intent(in) :: w
            real, intent(out) :: mf1, mf2, mf3
            mf1 = w*cons(i,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i,j,k+1,2) + w*cons(i,j+1,k+1,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i,j,k+1,3) + w*cons(i,j+1,k+1,3)
            mf3 = w*rowt(i,j,k) + w*rowt(i,j+1,k) + w*rowt(i,j,k+1) + w*rowt(i,j+1,k+1)
        end subroutine accum_mf
    end subroutine tbmflux_iface_row


    pure subroutine tbmflux_jface_row(cons, rowt, dA1, dA2, dA3, wall_lo, wall_hi, mrow, jf, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: rowt(ni, nj, nk)
        real, intent(in) :: dA1(ni-1, nj, nk-1), dA2(ni-1, nj, nk-1), dA3(ni-1, nj, nk-1)
        real, intent(in) :: wall_lo(ni-1, nk-1), wall_hi(ni-1, nk-1)
        real, intent(inout) :: mrow(ni-1)
        integer :: i
        real :: mf1, mf2, mf3

        if (jf == 1) then
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, 0.25e0*wall_lo(i,k), mf1, mf2, mf3)
                mrow(i) = mf1*dA1(i,jf,k) + mf2*dA2(i,jf,k) + mf3*dA3(i,jf,k)
            end do
        else if (jf == nj) then
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, 0.25e0*wall_hi(i,k), mf1, mf2, mf3)
                mrow(i) = mf1*dA1(i,jf,k) + mf2*dA2(i,jf,k) + mf3*dA3(i,jf,k)
            end do
        else
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, 0.25e0, mf1, mf2, mf3)
                mrow(i) = mf1*dA1(i,jf,k) + mf2*dA2(i,jf,k) + mf3*dA3(i,jf,k)
            end do
        end if

    contains
        pure subroutine accum_mf(i, w, mf1, mf2, mf3)
            integer, intent(in) :: i
            real, intent(in) :: w
            real, intent(out) :: mf1, mf2, mf3
            mf1 = w*cons(i,jf,k,2) + w*cons(i+1,jf,k,2) + w*cons(i,jf,k+1,2) + w*cons(i+1,jf,k+1,2)
            mf2 = w*cons(i,jf,k,3) + w*cons(i+1,jf,k,3) + w*cons(i,jf,k+1,3) + w*cons(i+1,jf,k+1,3)
            mf3 = w*rowt(i,jf,k) + w*rowt(i+1,jf,k) + w*rowt(i,jf,k+1) + w*rowt(i+1,jf,k+1)
        end subroutine accum_mf
    end subroutine tbmflux_jface_row


    pure subroutine tbmflux_kface_plane(cons, rowt, dA1, dA2, dA3, wall_lo, wall_hi, mplane, kf, ni, nj, nk)
        implicit none
        integer, intent(in) :: kf, ni, nj, nk
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: rowt(ni, nj, nk)
        real, intent(in) :: dA1(ni-1, nj-1, nk), dA2(ni-1, nj-1, nk), dA3(ni-1, nj-1, nk)
        real, intent(in) :: wall_lo(ni-1, nj-1), wall_hi(ni-1, nj-1)
        real, intent(inout) :: mplane(ni-1, nj-1)
        integer :: i, j
        real :: mf1, mf2, mf3

        if (kf == 1) then
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, j, 0.25e0*wall_lo(i,j), mf1, mf2, mf3)
                mplane(i,j) = mf1*dA1(i,j,kf) + mf2*dA2(i,j,kf) + mf3*dA3(i,j,kf)
            end do
            end do
        else if (kf == nk) then
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, j, 0.25e0*wall_hi(i,j), mf1, mf2, mf3)
                mplane(i,j) = mf1*dA1(i,j,kf) + mf2*dA2(i,j,kf) + mf3*dA3(i,j,kf)
            end do
            end do
        else
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, j, 0.25e0, mf1, mf2, mf3)
                mplane(i,j) = mf1*dA1(i,j,kf) + mf2*dA2(i,j,kf) + mf3*dA3(i,j,kf)
            end do
            end do
        end if

    contains
        pure subroutine accum_mf(i, j, w, mf1, mf2, mf3)
            integer, intent(in) :: i, j
            real, intent(in) :: w
            real, intent(out) :: mf1, mf2, mf3
            mf1 = w*cons(i,j,kf,2) + w*cons(i+1,j,kf,2) + w*cons(i,j+1,kf,2) + w*cons(i+1,j+1,kf,2)
            mf2 = w*cons(i,j,kf,3) + w*cons(i+1,j,kf,3) + w*cons(i,j+1,kf,3) + w*cons(i+1,j+1,kf,3)
            mf3 = w*rowt(i,j,kf) + w*rowt(i+1,j,kf) + w*rowt(i,j+1,kf) + w*rowt(i+1,j+1,kf)
        end subroutine accum_mf
    end subroutine tbmflux_kface_plane


    ! ---- stage 2a: components 2 and 3 -- flow = avg(q)*mdot + avg(dp)*dA
    ! q is vx for m=2 and vr for m=3: one nodal array, one four-point
    ! average, no divides. dA is the single component this pass needs.

    pure subroutine tbp_iface_row(P, P_offset, q, dA, mrow, row, j, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: dA(ni, nj-1, nk-1)
        real, intent(in) :: mrow(ni)
        real, intent(inout) :: row(ni)
        integer :: i
        real :: pm, pm5
        !DIR$ IVDEP
        do i = 1, ni
            pm = 0.25e0*q(i,j,k) + 0.25e0*q(i,j+1,k) + 0.25e0*q(i,j,k+1) + 0.25e0*q(i,j+1,k+1)
            pm5 = 0.25e0*(P(i,j,k) - P_offset) + 0.25e0*(P(i,j+1,k) - P_offset) &
                + 0.25e0*(P(i,j,k+1) - P_offset) + 0.25e0*(P(i,j+1,k+1) - P_offset)
            row(i) = pm*mrow(i) + pm5*dA(i,j,k)
        end do
    end subroutine tbp_iface_row


    pure subroutine tbp_jface_row(P, P_offset, q, dA, mrow, row, jf, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: dA(ni-1, nj, nk-1)
        real, intent(in) :: mrow(ni-1)
        real, intent(inout) :: row(ni)
        integer :: i
        real :: pm, pm5
        !DIR$ IVDEP
        do i = 1, ni-1
            pm = 0.25e0*q(i,jf,k) + 0.25e0*q(i+1,jf,k) + 0.25e0*q(i,jf,k+1) + 0.25e0*q(i+1,jf,k+1)
            pm5 = 0.25e0*(P(i,jf,k) - P_offset) + 0.25e0*(P(i+1,jf,k) - P_offset) &
                + 0.25e0*(P(i,jf,k+1) - P_offset) + 0.25e0*(P(i+1,jf,k+1) - P_offset)
            row(i) = pm*mrow(i) + pm5*dA(i,jf,k)
        end do
    end subroutine tbp_jface_row


    pure subroutine tbp_kface_plane(P, P_offset, q, dA, mplane, plane, kf, njp, ni, nj, nk)
        implicit none
        integer, intent(in) :: kf, njp, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: dA(ni-1, nj-1, nk)
        real, intent(in) :: mplane(ni-1, nj-1)
        real, intent(inout) :: plane(ni, njp)
        integer :: i, j
        real :: pm, pm5
        do j = 1, nj-1
        !DIR$ IVDEP
        do i = 1, ni-1
            pm = 0.25e0*q(i,j,kf) + 0.25e0*q(i+1,j,kf) + 0.25e0*q(i,j+1,kf) + 0.25e0*q(i+1,j+1,kf)
            pm5 = 0.25e0*(P(i,j,kf) - P_offset) + 0.25e0*(P(i+1,j,kf) - P_offset) &
                + 0.25e0*(P(i,j+1,kf) - P_offset) + 0.25e0*(P(i+1,j+1,kf) - P_offset)
            plane(i,j) = pm*mplane(i,j) + pm5*dA(i,j,kf)
        end do
        end do
    end subroutine tbp_kface_plane


    ! ---- stage 2b: components 4 and 5 -- pressure term carries a radius.
    ! q is rvt (= r*Vt, staged) for m=4 and ho for m=5; pfac is 1 and Omega.
    ! Only this pair still re-derives its pressure factor from nodal r and
    ! P, because multall's own AVGP staging is deliberately not reproduced
    ! here (see the header).

    pure subroutine tbrp_iface_row(P, P_offset, r, q, dA, mrow, pfac, row, j, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset, pfac
        real, intent(in) :: dA(ni, nj-1, nk-1)
        real, intent(in) :: mrow(ni)
        real, intent(inout) :: row(ni)
        integer :: i
        real :: pm, pm6, dp1, dp2, dp3, dp4
        !DIR$ IVDEP
        do i = 1, ni
            dp1 = P(i,j,k)     - P_offset
            dp2 = P(i,j+1,k)   - P_offset
            dp3 = P(i,j,k+1)   - P_offset
            dp4 = P(i,j+1,k+1) - P_offset
            pm = 0.25e0*q(i,j,k) + 0.25e0*q(i,j+1,k) + 0.25e0*q(i,j,k+1) + 0.25e0*q(i,j+1,k+1)
            pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i,j+1,k)*dp2 &
                + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i,j+1,k+1)*dp4
            row(i) = pm*mrow(i) + pfac*pm6*dA(i,j,k)
        end do
    end subroutine tbrp_iface_row


    pure subroutine tbrp_jface_row(P, P_offset, r, q, dA, mrow, pfac, row, jf, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset, pfac
        real, intent(in) :: dA(ni-1, nj, nk-1)
        real, intent(in) :: mrow(ni-1)
        real, intent(inout) :: row(ni)
        integer :: i
        real :: pm, pm6, dp1, dp2, dp3, dp4
        !DIR$ IVDEP
        do i = 1, ni-1
            dp1 = P(i,jf,k)     - P_offset
            dp2 = P(i+1,jf,k)   - P_offset
            dp3 = P(i,jf,k+1)   - P_offset
            dp4 = P(i+1,jf,k+1) - P_offset
            pm = 0.25e0*q(i,jf,k) + 0.25e0*q(i+1,jf,k) + 0.25e0*q(i,jf,k+1) + 0.25e0*q(i+1,jf,k+1)
            pm6 = 0.25e0*r(i,jf,k)*dp1 + 0.25e0*r(i+1,jf,k)*dp2 &
                + 0.25e0*r(i,jf,k+1)*dp3 + 0.25e0*r(i+1,jf,k+1)*dp4
            row(i) = pm*mrow(i) + pfac*pm6*dA(i,jf,k)
        end do
    end subroutine tbrp_jface_row


    pure subroutine tbrp_kface_plane(P, P_offset, r, q, dA, mplane, pfac, plane, kf, njp, ni, nj, nk)
        implicit none
        integer, intent(in) :: kf, njp, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset, pfac
        real, intent(in) :: dA(ni-1, nj-1, nk)
        real, intent(in) :: mplane(ni-1, nj-1)
        real, intent(inout) :: plane(ni, njp)
        integer :: i, j
        real :: pm, pm6, dp1, dp2, dp3, dp4
        do j = 1, nj-1
        !DIR$ IVDEP
        do i = 1, ni-1
            dp1 = P(i,j,kf)     - P_offset
            dp2 = P(i+1,j,kf)   - P_offset
            dp3 = P(i,j+1,kf)   - P_offset
            dp4 = P(i+1,j+1,kf) - P_offset
            pm = 0.25e0*q(i,j,kf) + 0.25e0*q(i+1,j,kf) + 0.25e0*q(i,j+1,kf) + 0.25e0*q(i+1,j+1,kf)
            pm6 = 0.25e0*r(i,j,kf)*dp1 + 0.25e0*r(i+1,j,kf)*dp2 &
                + 0.25e0*r(i,j+1,kf)*dp3 + 0.25e0*r(i+1,j+1,kf)*dp4
            plane(i,j) = pm*mplane(i,j) + pfac*pm6*dA(i,j,kf)
        end do
        end do
    end subroutine tbrp_kface_plane

end module residual_multall_helpers


subroutine set_residual_multall( &
    cons, P, P_offset, &
    r, Omega, &
    dai1, dai2, dai3, daj1, daj2, daj3, dak1, dak2, dak3, &
    f_body, &
    dU, &
    vx, vr, vt, ho, &
    rowt, rvt, &
    fi, fj, fk, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    dAk, dt_vol, dampin, &
    njp, ni, nj, nk &
    )

    use residual_helpers, only: correct_cusp_kface_du
    use residual_staged_helpers, only: scale_du_all
    use residual_multall_helpers

    implicit none

    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega
    ! Face areas as nine separate component arrays -- multall's
    ! AIX/AIR/AIT, AJX/AJR/AJT, AKX/AKR/AKT. Grid geometry, built once.
    real, intent(in) :: dai1(ni, nj-1, nk-1), dai2(ni, nj-1, nk-1), dai3(ni, nj-1, nk-1)
    real, intent(in) :: daj1(ni-1, nj, nk-1), daj2(ni-1, nj, nk-1), daj3(ni-1, nj, nk-1)
    real, intent(in) :: dak1(ni-1, nj-1, nk), dak2(ni-1, nj-1, nk), dak3(ni-1, nj-1, nk)
    real, intent(in) :: f_body(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: vx(ni, nj, nk)
    real, intent(in) :: vr(ni, nj, nk)
    real, intent(in) :: vt(ni, nj, nk)
    real, intent(in) :: ho(ni, nj, nk)
    real, intent(in) :: walli1(nj-1, nk-1)
    real, intent(in) :: wallni(nj-1, nk-1)
    real, intent(in) :: wallj1(ni-1, nk-1)
    real, intent(in) :: wallnj(ni-1, nk-1)
    real, intent(in) :: wallk1(ni-1, nj-1)
    real, intent(in) :: wallnk(ni-1, nj-1)
    integer, intent(in) :: i_cusp_start, i_cusp_end
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! Staged nodal primitives (multall's ROWT and its AVGR*AVGVT pairing).
    real, intent(inout) :: rowt(ni, nj, nk), rvt(ni, nj, nk)
    ! Staged face mass fluxes (multall's FIMAS / FJMAS / FKMAS).
    real, intent(inout) :: fi(ni, nj-1, nk-1)
    real, intent(inout) :: fj(ni-1, nj, nk-1)
    real, intent(inout) :: fk(ni-1, nj-1, nk)
    real, intent(inout) :: planes(ni, njp, 2)
    real, intent(inout) :: rows(ni, 3)
    ! AoS dAk retained solely for correct_cusp_kface_du, which is
    ! production's own routine and is called unmodified so the seam
    ! correction is identical in every arm.
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dampin
    integer, intent(in) :: njp, ni, nj, nk

    integer :: i, j, k, m, ja, jb, pa, pb, stmp, ncell
    real :: pfac
    real :: avg(5), ravg(5)

    do m = 1, 5
        avg(m) = 0.0e0
    end do

    ! ---- stage 0: nodal primitives, one divide per node ----------------
    call stage_primitives(cons, r, vt, Omega, rowt, rvt, ni, nj, nk)

    ! ---- stage 1: face mass flux, once for all five components --------
    do k = 1, nk-1
        do j = 1, nj-1
            call tbmflux_iface_row(cons, rowt, dai1, dai2, dai3, walli1(j,k), wallni(j,k), &
                                   fi(:,j,k), j, k, ni, nj, nk)
        end do
    end do
    do k = 1, nk-1
        do j = 1, nj
            call tbmflux_jface_row(cons, rowt, daj1, daj2, daj3, wallj1, wallnj, &
                                   fj(:,j,k), j, k, ni, nj, nk)
        end do
    end do
    do k = 1, nk
        call tbmflux_kface_plane(cons, rowt, dak1, dak2, dak3, wallk1, wallnk, &
                                 fk(:,:,k), k, ni, nj, nk)
    end do

    ! ---- pass m = 1: mass. The staged fluxes ARE the flows. -----------
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        dU(i,j,k,1) = fi(i,j,k) - fi(i+1,j,k) + f_body(i,j,k,1) &
                    + fj(i,j,k) - fj(i,j+1,k) &
                    + fk(i,j,k) - fk(i,j,k+1)
        avg(1) = avg(1) + abs(dU(i,j,k,1) * dt_vol(i,j,k))
    end do
    end do
    end do

    ! ---- passes m = 2..5 ----------------------------------------------
    do m = 2, 5
        if (m == 5) then
            pfac = Omega
        else
            pfac = 1.0e0
        end if

        pa = 1
        pb = 2
        call kflow_plane(1, planes(:,:,pa))

        do k = 1, nk-1
            ja = 2
            jb = 3
            call jflow_row(1, k, rows(:,ja))
            call kflow_plane(k+1, planes(:,:,pb))
            do j = 1, nj-1
                call iflow_row(j, k, rows(:,1))
                call jflow_row(j+1, k, rows(:,jb))
                do i = 1, ni-1
                    dU(i,j,k,m) = rows(i,1) - rows(i+1,1) + f_body(i,j,k,m) &
                                + rows(i,ja) - rows(i,jb) &
                                + planes(i,j,pa) - planes(i,j,pb)
                    avg(m) = avg(m) + abs(dU(i,j,k,m) * dt_vol(i,j,k))
                end do
                stmp = ja
                ja = jb
                jb = stmp
            end do
            stmp = pa
            pa = pb
            pb = stmp
        end do
    end do

    if (i_cusp_start > 0 .and. nk > 2) then
        call correct_cusp_kface_du(vx, vr, vt, ho, P, P_offset, r, cons, &
                                   Omega, dAk, wallk1, wallnk, dU, &
                                   i_cusp_start, i_cusp_end, ni, nj, nk)
    end if

    if (dampin > 0.0e0) then
        ncell = (ni-1)*(nj-1)*(nk-1)
        do m = 1, 5
            avg(m) = avg(m) / ncell
            if (avg(m) > 0.0e0) then
                ravg(m) = 1.0e0 / avg(m)
            else
                ravg(m) = 0.0e0
            end if
        end do
        call scale_du_all(dU, dt_vol, ravg, dampin, ni, nj, nk)
    end if

contains

    ! Components 2..5 differ only in WHICH nodal array is averaged and
    ! which dA component is used -- vx/dA1, vr/dA2, rvt/dA3, ho/dA3. That
    ! uniformity is what staging the primitives buys, and it is why multall
    ! can write one flux loop per variable in six lines.
    subroutine iflow_row(j, k, row)
        integer, intent(in) :: j, k
        real, intent(inout) :: row(ni)
        if (m == 2) then
            call tbp_iface_row(P, P_offset, vx, dai1, fi(:,j,k), row, j, k, ni, nj, nk)
        else if (m == 3) then
            call tbp_iface_row(P, P_offset, vr, dai2, fi(:,j,k), row, j, k, ni, nj, nk)
        else if (m == 4) then
            call tbrp_iface_row(P, P_offset, r, rvt, dai3, fi(:,j,k), pfac, row, j, k, ni, nj, nk)
        else
            call tbrp_iface_row(P, P_offset, r, ho, dai3, fi(:,j,k), pfac, row, j, k, ni, nj, nk)
        end if
    end subroutine iflow_row

    subroutine jflow_row(jf, k, row)
        integer, intent(in) :: jf, k
        real, intent(inout) :: row(ni)
        if (m == 2) then
            call tbp_jface_row(P, P_offset, vx, daj1, fj(:,jf,k), row, jf, k, ni, nj, nk)
        else if (m == 3) then
            call tbp_jface_row(P, P_offset, vr, daj2, fj(:,jf,k), row, jf, k, ni, nj, nk)
        else if (m == 4) then
            call tbrp_jface_row(P, P_offset, r, rvt, daj3, fj(:,jf,k), pfac, row, jf, k, ni, nj, nk)
        else
            call tbrp_jface_row(P, P_offset, r, ho, daj3, fj(:,jf,k), pfac, row, jf, k, ni, nj, nk)
        end if
    end subroutine jflow_row

    subroutine kflow_plane(kf, plane)
        integer, intent(in) :: kf
        real, intent(inout) :: plane(ni, njp)
        if (m == 2) then
            call tbp_kface_plane(P, P_offset, vx, dak1, fk(:,:,kf), plane, kf, njp, ni, nj, nk)
        else if (m == 3) then
            call tbp_kface_plane(P, P_offset, vr, dak2, fk(:,:,kf), plane, kf, njp, ni, nj, nk)
        else if (m == 4) then
            call tbrp_kface_plane(P, P_offset, r, rvt, dak3, fk(:,:,kf), pfac, plane, kf, njp, ni, nj, nk)
        else
            call tbrp_kface_plane(P, P_offset, r, ho, dak3, fk(:,:,kf), pfac, plane, kf, njp, ni, nj, nk)
        end if
    end subroutine kflow_plane

end subroutine set_residual_multall
