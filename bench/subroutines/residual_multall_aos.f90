! =====================================================================
! BENCHMARK ARM `tbaos` -- the multall five-pass residual on ember's own
! component-first dA(3,i,j,k) geometry.
!
! Nothing in production calls this. See docs/dev/plan_nodal_primitives.md
! and docs/dev/kernel_benchmark_methodology.md.
!
! Why it exists
! -------------
! Section 26's `multall` arm bundles three changes and cannot attribute its
! 13-50% win among them. With this arm and `nodal`, the ladder becomes an
! incremental chain in which every adjacent pair differs by ONE thing:
!
!   prod   -> nodal   nodal primitives instead of derived-per-corner
!   nodal  -> tbaos   the five-pass split (with the per-node staging that
!                     a five-pass design forces -- Rule 1: they are
!                     co-optimised and must move together)
!   tbaos  -> multall  SoA face-area geometry instead of AoS dA(3,i,j,k)
!
! The last step is the one this arm adds, and it is the one section 26.1
! asserted without measuring: "ember's dA(3,i,j,k) is component-first, so
! touching one axis pulls the line holding all three... that is a cost of
! ember's LAYOUT, not of the five-pass idea." True or not, the size of
! that cost was never priced. Here it is `multall` minus `tbaos`.
!
! It is also the honest number for a real port. Splitting the face areas
! is free in principle (grid geometry, built once in FIND_AREAS, Rule 3)
! but not free in ember: dA is built in geometry.f90 and consumed by the
! viscous kernels, the multigrid restriction and the boundary conditions,
! none of which are under test here. `tbaos` is therefore what a port of
! multall's residual gets WITHOUT touching any of that -- the change one
! could actually propose -- and `multall` is the ceiling it would reach
! after the layout change as well.
!
! What differs from residual_multall.f90
! -------------------------------------
! Only the geometry indexing. The nine component arrays become the three
! AoS arrays, so:
!
!   stage 1 (mass flux)  dA1(i,j,k), dA2(...), dA3(...)  ->  dA(1..3,i,j,k)
!                        -- no change in bytes touched: it needs all
!                        three components anyway, which is the case AoS
!                        is good at.
!   stage 2 (m = 2..5)   dA(i,j,k) with the caller picking the array
!                        ->  dA(c,i,j,k) with the caller picking c as a
!                        LITERAL at each call site, so it constant-folds.
!                        -- this is where AoS should hurt: the pass wants
!                        one component and pays for a line of three.
!
! Everything else is residual_multall.f90 verbatim: the same staging, the
! same pass structure, the same rolling buffers, the same fused dU write.
! `stage_primitives` is USED FROM the multall arm rather than copied, which
! guarantees identical codegen for the part that is not under test
! (methodology section 2). Likewise `scale_du_all` and the shared
! `correct_cusp_kface_du`.
!
! FILE ORDER: this arm `use`s residual_multall_helpers, and
! tools/check_compile.sh pre-flights with providers first then
! alphabetically -- "residual_multall.f90" sorts before
! "residual_multall_aos.f90" ('.' < '_'), so the pre-flight is safe. Do not
! rename it to something that sorts earlier.
!
! NUMERICS: identical arithmetic to `multall` in every term -- same
! operands, same order -- so the two are expected to agree BITWISE. That
! is a real gate: any difference between them is a codegen or indexing
! bug, not float rounding. Against production it sits at the same
! ~1e-6-of-scale level as `multall`.
! =====================================================================

module residual_multall_aos_helpers
    implicit none
    private
    public :: tbaos_mflux_iface_row, tbaos_mflux_jface_row, tbaos_mflux_kface_plane
    public :: tbaos_p_iface_row, tbaos_p_jface_row, tbaos_p_kface_plane
    public :: tbaos_rp_iface_row, tbaos_rp_jface_row, tbaos_rp_kface_plane

contains

    ! ---- stage 1: face mass flux (multall's FIMAS/FJMAS/FKMAS) ----------
    ! Reads all three dA components, so AoS costs nothing here.

    pure subroutine tbaos_mflux_iface_row(cons, rowt, dA, wall_lo, wall_hi, mrow, j, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: rowt(ni, nj, nk)
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: wall_lo, wall_hi
        real, intent(inout) :: mrow(ni)
        integer :: i
        real :: mf1, mf2, mf3, w

        w = 0.25e0*wall_lo
        call accum_mf(1, w, mf1, mf2, mf3)
        mrow(1) = mf1*dA(1,1,j,k) + mf2*dA(2,1,j,k) + mf3*dA(3,1,j,k)

        !DIR$ IVDEP
        do i = 2, ni-1
            call accum_mf(i, 0.25e0, mf1, mf2, mf3)
            mrow(i) = mf1*dA(1,i,j,k) + mf2*dA(2,i,j,k) + mf3*dA(3,i,j,k)
        end do

        w = 0.25e0*wall_hi
        call accum_mf(ni, w, mf1, mf2, mf3)
        mrow(ni) = mf1*dA(1,ni,j,k) + mf2*dA(2,ni,j,k) + mf3*dA(3,ni,j,k)

    contains
        pure subroutine accum_mf(i, w, mf1, mf2, mf3)
            integer, intent(in) :: i
            real, intent(in) :: w
            real, intent(out) :: mf1, mf2, mf3
            mf1 = w*cons(i,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i,j,k+1,2) + w*cons(i,j+1,k+1,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i,j,k+1,3) + w*cons(i,j+1,k+1,3)
            mf3 = w*rowt(i,j,k) + w*rowt(i,j+1,k) + w*rowt(i,j,k+1) + w*rowt(i,j+1,k+1)
        end subroutine accum_mf
    end subroutine tbaos_mflux_iface_row


    pure subroutine tbaos_mflux_jface_row(cons, rowt, dA, wall_lo, wall_hi, mrow, jf, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: rowt(ni, nj, nk)
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: wall_lo(ni-1, nk-1), wall_hi(ni-1, nk-1)
        real, intent(inout) :: mrow(ni-1)
        integer :: i
        real :: mf1, mf2, mf3

        if (jf == 1) then
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, 0.25e0*wall_lo(i,k), mf1, mf2, mf3)
                mrow(i) = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
            end do
        else if (jf == nj) then
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, 0.25e0*wall_hi(i,k), mf1, mf2, mf3)
                mrow(i) = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
            end do
        else
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, 0.25e0, mf1, mf2, mf3)
                mrow(i) = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
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
    end subroutine tbaos_mflux_jface_row


    pure subroutine tbaos_mflux_kface_plane(cons, rowt, dA, wall_lo, wall_hi, mplane, kf, ni, nj, nk)
        implicit none
        integer, intent(in) :: kf, ni, nj, nk
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: rowt(ni, nj, nk)
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
        real, intent(in) :: wall_lo(ni-1, nj-1), wall_hi(ni-1, nj-1)
        real, intent(inout) :: mplane(ni-1, nj-1)
        integer :: i, j
        real :: mf1, mf2, mf3

        if (kf == 1) then
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, j, 0.25e0*wall_lo(i,j), mf1, mf2, mf3)
                mplane(i,j) = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
            end do
            end do
        else if (kf == nk) then
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, j, 0.25e0*wall_hi(i,j), mf1, mf2, mf3)
                mplane(i,j) = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
            end do
            end do
        else
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, j, 0.25e0, mf1, mf2, mf3)
                mplane(i,j) = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
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
    end subroutine tbaos_mflux_kface_plane


    ! ---- stage 2a: components 2 and 3 ----------------------------------
    ! One nodal array, one four-point average, one dA COMPONENT -- and on
    ! this layout that component costs a whole 12-byte line. `c` is passed
    ! as a literal from each call site so it constant-folds to a fixed
    ! offset with stride 3.

    pure subroutine tbaos_p_iface_row(P, P_offset, q, dA, c, mrow, row, j, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: c, j, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: mrow(ni)
        real, intent(inout) :: row(ni)
        integer :: i
        real :: pm, pm5
        !DIR$ IVDEP
        do i = 1, ni
            pm = 0.25e0*q(i,j,k) + 0.25e0*q(i,j+1,k) + 0.25e0*q(i,j,k+1) + 0.25e0*q(i,j+1,k+1)
            pm5 = 0.25e0*(P(i,j,k) - P_offset) + 0.25e0*(P(i,j+1,k) - P_offset) &
                + 0.25e0*(P(i,j,k+1) - P_offset) + 0.25e0*(P(i,j+1,k+1) - P_offset)
            row(i) = pm*mrow(i) + pm5*dA(c,i,j,k)
        end do
    end subroutine tbaos_p_iface_row


    pure subroutine tbaos_p_jface_row(P, P_offset, q, dA, c, mrow, row, jf, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: c, jf, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: mrow(ni-1)
        real, intent(inout) :: row(ni)
        integer :: i
        real :: pm, pm5
        !DIR$ IVDEP
        do i = 1, ni-1
            pm = 0.25e0*q(i,jf,k) + 0.25e0*q(i+1,jf,k) + 0.25e0*q(i,jf,k+1) + 0.25e0*q(i+1,jf,k+1)
            pm5 = 0.25e0*(P(i,jf,k) - P_offset) + 0.25e0*(P(i+1,jf,k) - P_offset) &
                + 0.25e0*(P(i,jf,k+1) - P_offset) + 0.25e0*(P(i+1,jf,k+1) - P_offset)
            row(i) = pm*mrow(i) + pm5*dA(c,i,jf,k)
        end do
    end subroutine tbaos_p_jface_row


    pure subroutine tbaos_p_kface_plane(P, P_offset, q, dA, c, mplane, plane, kf, njp, ni, nj, nk)
        implicit none
        integer, intent(in) :: c, kf, njp, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
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
            plane(i,j) = pm*mplane(i,j) + pm5*dA(c,i,j,kf)
        end do
        end do
    end subroutine tbaos_p_kface_plane


    ! ---- stage 2b: components 4 and 5 (pressure term carries a radius) --

    pure subroutine tbaos_rp_iface_row(P, P_offset, r, q, dA, c, mrow, pfac, row, j, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: c, j, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset, pfac
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
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
            row(i) = pm*mrow(i) + pfac*pm6*dA(c,i,j,k)
        end do
    end subroutine tbaos_rp_iface_row


    pure subroutine tbaos_rp_jface_row(P, P_offset, r, q, dA, c, mrow, pfac, row, jf, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: c, jf, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset, pfac
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
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
            row(i) = pm*mrow(i) + pfac*pm6*dA(c,i,jf,k)
        end do
    end subroutine tbaos_rp_jface_row


    pure subroutine tbaos_rp_kface_plane(P, P_offset, r, q, dA, c, mplane, pfac, plane, kf, njp, ni, nj, nk)
        implicit none
        integer, intent(in) :: c, kf, njp, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk), q(ni, nj, nk)
        real, intent(in) :: P_offset, pfac
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
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
            plane(i,j) = pm*mplane(i,j) + pfac*pm6*dA(c,i,j,kf)
        end do
        end do
    end subroutine tbaos_rp_kface_plane

end module residual_multall_aos_helpers


! =====================================================================
! Driver: residual_multall.f90's, with the nine SoA component arrays
! replaced by ember's three AoS dA arrays. The staging, the pass
! structure, the rolling buffers, the fused dU write and the deferred
! cusp correction are unchanged, so the only variable between this arm
! and `multall` is the face-area layout.
!
! dAk serves both the k-direction passes and correct_cusp_kface_du here,
! where the `multall` arm had to carry a separate AoS copy for the cusp.
! =====================================================================
subroutine set_residual_multall_aos( &
    cons, P, P_offset, &
    r, Omega, &
    dai, daj, dak, &
    f_body, &
    dU, &
    vx, vr, vt, ho, &
    rowt, rvt, &
    fi, fj, fk, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    dt_vol, dampin, &
    njp, ni, nj, nk &
    )

    use residual_helpers, only: correct_cusp_kface_du
    use residual_staged_helpers, only: scale_du_all
    use residual_multall_helpers, only: stage_primitives
    use residual_multall_aos_helpers

    implicit none

    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega
    ! Ember's own component-first face areas -- the whole point of the arm.
    real, intent(in) :: dai(3, ni, nj-1, nk-1)
    real, intent(in) :: daj(3, ni-1, nj, nk-1)
    real, intent(in) :: dak(3, ni-1, nj-1, nk)
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
    ! Shared with the `multall` arm, not copied: identical codegen for the
    ! part that is not under test.
    call stage_primitives(cons, r, vt, Omega, rowt, rvt, ni, nj, nk)

    ! ---- stage 1: face mass flux, once for all five components --------
    do k = 1, nk-1
        do j = 1, nj-1
            call tbaos_mflux_iface_row(cons, rowt, dai, walli1(j,k), wallni(j,k), &
                                       fi(:,j,k), j, k, ni, nj, nk)
        end do
    end do
    do k = 1, nk-1
        do j = 1, nj
            call tbaos_mflux_jface_row(cons, rowt, daj, wallj1, wallnj, &
                                       fj(:,j,k), j, k, ni, nj, nk)
        end do
    end do
    do k = 1, nk
        call tbaos_mflux_kface_plane(cons, rowt, dak, wallk1, wallnk, &
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
                                   Omega, dak, wallk1, wallnk, dU, &
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
    ! which dA component is indexed -- vx/1, vr/2, rvt/3, ho/3. The
    ! component is a literal at every call site so it constant-folds.
    subroutine iflow_row(j, k, row)
        integer, intent(in) :: j, k
        real, intent(inout) :: row(ni)
        if (m == 2) then
            call tbaos_p_iface_row(P, P_offset, vx, dai, 1, fi(:,j,k), row, j, k, ni, nj, nk)
        else if (m == 3) then
            call tbaos_p_iface_row(P, P_offset, vr, dai, 2, fi(:,j,k), row, j, k, ni, nj, nk)
        else if (m == 4) then
            call tbaos_rp_iface_row(P, P_offset, r, rvt, dai, 3, fi(:,j,k), pfac, row, j, k, ni, nj, nk)
        else
            call tbaos_rp_iface_row(P, P_offset, r, ho, dai, 3, fi(:,j,k), pfac, row, j, k, ni, nj, nk)
        end if
    end subroutine iflow_row

    subroutine jflow_row(jf, k, row)
        integer, intent(in) :: jf, k
        real, intent(inout) :: row(ni)
        if (m == 2) then
            call tbaos_p_jface_row(P, P_offset, vx, daj, 1, fj(:,jf,k), row, jf, k, ni, nj, nk)
        else if (m == 3) then
            call tbaos_p_jface_row(P, P_offset, vr, daj, 2, fj(:,jf,k), row, jf, k, ni, nj, nk)
        else if (m == 4) then
            call tbaos_rp_jface_row(P, P_offset, r, rvt, daj, 3, fj(:,jf,k), pfac, row, jf, k, ni, nj, nk)
        else
            call tbaos_rp_jface_row(P, P_offset, r, ho, daj, 3, fj(:,jf,k), pfac, row, jf, k, ni, nj, nk)
        end if
    end subroutine jflow_row

    subroutine kflow_plane(kf, plane)
        integer, intent(in) :: kf
        real, intent(inout) :: plane(ni, njp)
        if (m == 2) then
            call tbaos_p_kface_plane(P, P_offset, vx, dak, 1, fk(:,:,kf), plane, kf, njp, ni, nj, nk)
        else if (m == 3) then
            call tbaos_p_kface_plane(P, P_offset, vr, dak, 2, fk(:,:,kf), plane, kf, njp, ni, nj, nk)
        else if (m == 4) then
            call tbaos_rp_kface_plane(P, P_offset, r, rvt, dak, 3, fk(:,:,kf), pfac, plane, kf, njp, ni, nj, nk)
        else
            call tbaos_rp_kface_plane(P, P_offset, r, ho, dak, 3, fk(:,:,kf), pfac, plane, kf, njp, ni, nj, nk)
        end if
    end subroutine kflow_plane

end subroutine set_residual_multall_aos
