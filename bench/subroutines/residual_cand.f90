! =====================================================================
! Idiomatic rewrite of set_residual, for benchmarking only.
!
! This is the kernel someone would write from scratch, knowing the
! algorithm but not the optimisation history: it drops every hand
! optimisation the production kernel in residual.f90 carries, keeping
! only the *algorithmic* structure (rolling face buffers, one dU touch,
! deferred cusp correction). Exposed to f2py as set_residual_clean and
! compared against production in the same .so by
! tools/bench_residual_variants.py -- nothing calls it in production.
!
! What is deliberately given up relative to residual.f90:
!
!   1. pm/mf are arrays again (pm(6), mf(3)) accumulated by a single
!      accum() call per corner, instead of nine hand-scalarized locals
!      pm1..pm6/mf1..mf3 with the four corners hand-unrolled. This is
!      the form residual.f90:39-47 says ifort 2022.1.0 refused to
!      vectorize ("vector dependence prevents vectorization"); the point
!      of this file is to re-test that claim under the current compiler
!      and flags, and to price it if it still holds. correct_cusp_kface_du
!      in residual.f90 still uses exactly this style (it is O(surface), so
!      it was never worth hand-optimising) -- so this is the house style,
!      not an invention.
!
!   2. One loop body per helper instead of three. Production duplicates
!      each face loop three times (low boundary / interior / high
!      boundary) so the interior copy sees a compile-time wfac = 1.0.
!      Here the wall factor is read per-i from a contiguous wfac row that
!      the caller selects, so there is a single loop. Costs one extra
!      unit-stride stream and loses the constant-folding of w = 0.25*1.0.
!
!   3. No kb dummy and no slab loop. In production the `do k0 = 1, nk-1,
!      kb` nest is already a pure re-nesting of `do k = 1, nk-1` (pa/pb
!      and the k=1 face prime sit outside both loops; there is no
!      per-slab prologue or epilogue), so dropping it changes nothing but
!      the source. This one is free and should be bitwise.
!
! The face-flow arithmetic, the corner summation order, and the final dU
! expression are unchanged, so any difference against production should
! be float32 reassociation only (~1 ulp of the flux scale), not physics.
! =====================================================================
module residual_clean_helpers
    implicit none
    private
    public :: iface_flow_row_clean, jface_flow_row_clean, kface_flow_plane_clean

contains

    ! Assemble, per face, the 5 inviscid flows from face-averaged per-mass
    ! factors pm(6) and mass-flux factors mf(3):
    !   pm = (Vx, Vr, r*Vt_abs, ho, P-P_offset, r*(P-P_offset))
    !   mf = (rho*Vx, rho*Vr, rho*Vt_rel)
    ! The wall mask weights mf only. Granularity is one row (i/j) or one
    ! plane (k) so the caller can roll small buffers.

    pure subroutine iface_flow_row_clean(vx, vr, vt, ho, P, P_offset, r, &
                                         cons, Omega, dA, &
                                         wfac, row, j, k, ni, nj, nk)
        ! Inviscid face flows on the ni i-faces of cell row (j,k).
        ! i-face corners: (i, j:j+1, k:k+1). wfac(i) is the per-face wall
        ! factor: 1.0 in the interior, the wall mask at i=1 and i=ni.

        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: wfac(ni)
        real, intent(inout) :: row(ni, 5)

        integer :: i
        real :: pm(6), mf(3), mdot

        do i = 1, ni
            pm = 0.0e0
            mf = 0.0e0
            call accum(pm, mf, i, j,   k,   wfac(i))
            call accum(pm, mf, i, j+1, k,   wfac(i))
            call accum(pm, mf, i, j,   k+1, wfac(i))
            call accum(pm, mf, i, j+1, k+1, wfac(i))
            mdot = mf(1)*dA(1,i,j,k) + mf(2)*dA(2,i,j,k) + mf(3)*dA(3,i,j,k)
            row(i,1) = mdot
            row(i,2) = pm(1)*mdot + pm(5)*dA(1,i,j,k)
            row(i,3) = pm(2)*mdot + pm(5)*dA(2,i,j,k)
            row(i,4) = pm(3)*mdot + pm(6)*dA(3,i,j,k)
            row(i,5) = pm(4)*mdot + Omega*pm(6)*dA(3,i,j,k)
        end do

    contains
        pure subroutine accum(pm, mf, i, j, k, wfac)
            real, intent(inout) :: pm(6), mf(3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real :: dp, w
            dp = P(i,j,k) - P_offset
            pm(1) = pm(1) + 0.25e0*vx(i,j,k)
            pm(2) = pm(2) + 0.25e0*vr(i,j,k)
            pm(3) = pm(3) + 0.25e0*r(i,j,k)*vt(i,j,k)
            pm(4) = pm(4) + 0.25e0*ho(i,j,k)
            pm(5) = pm(5) + 0.25e0*dp
            pm(6) = pm(6) + 0.25e0*r(i,j,k)*dp
            w = 0.25e0*wfac
            mf(1) = mf(1) + w*cons(i,j,k,2)
            mf(2) = mf(2) + w*cons(i,j,k,3)
            mf(3) = mf(3) + w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k))
        end subroutine accum
    end subroutine iface_flow_row_clean


    pure subroutine jface_flow_row_clean(vx, vr, vt, ho, P, P_offset, r, &
                                         cons, Omega, dA, &
                                         wfac, row, jf, k, ni, nj, nk)
        ! Inviscid face flows on the (ni-1) j-faces of face row jf at cell
        ! plane k. j-face corners: (i:i+1, jf, k:k+1). wfac(i) is the
        ! per-face wall factor row (all ones for interior jf).

        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: wfac(ni-1)
        real, intent(inout) :: row(ni, 5)

        integer :: i
        real :: pm(6), mf(3), mdot

        do i = 1, ni-1
            pm = 0.0e0
            mf = 0.0e0
            call accum(pm, mf, i,   jf, k,   wfac(i))
            call accum(pm, mf, i+1, jf, k,   wfac(i))
            call accum(pm, mf, i,   jf, k+1, wfac(i))
            call accum(pm, mf, i+1, jf, k+1, wfac(i))
            mdot = mf(1)*dA(1,i,jf,k) + mf(2)*dA(2,i,jf,k) + mf(3)*dA(3,i,jf,k)
            row(i,1) = mdot
            row(i,2) = pm(1)*mdot + pm(5)*dA(1,i,jf,k)
            row(i,3) = pm(2)*mdot + pm(5)*dA(2,i,jf,k)
            row(i,4) = pm(3)*mdot + pm(6)*dA(3,i,jf,k)
            row(i,5) = pm(4)*mdot + Omega*pm(6)*dA(3,i,jf,k)
        end do

    contains
        pure subroutine accum(pm, mf, i, j, k, wfac)
            real, intent(inout) :: pm(6), mf(3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real :: dp, w
            dp = P(i,j,k) - P_offset
            pm(1) = pm(1) + 0.25e0*vx(i,j,k)
            pm(2) = pm(2) + 0.25e0*vr(i,j,k)
            pm(3) = pm(3) + 0.25e0*r(i,j,k)*vt(i,j,k)
            pm(4) = pm(4) + 0.25e0*ho(i,j,k)
            pm(5) = pm(5) + 0.25e0*dp
            pm(6) = pm(6) + 0.25e0*r(i,j,k)*dp
            w = 0.25e0*wfac
            mf(1) = mf(1) + w*cons(i,j,k,2)
            mf(2) = mf(2) + w*cons(i,j,k,3)
            mf(3) = mf(3) + w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k))
        end subroutine accum
    end subroutine jface_flow_row_clean


    pure subroutine kface_flow_plane_clean(vx, vr, vt, ho, P, P_offset, r, &
                                           cons, Omega, dA, &
                                           wfac, plane, kf, njp, ni, nj, nk)
        ! Inviscid face flows on the (ni-1)x(nj-1) k-face plane kf.
        ! k-face corners: (i:i+1, j:j+1, kf). njp (nj or nj+1) is the plane
        ! buffer's padded j-extent -- see set_residual_clean. wfac(i,j) is
        ! the per-face wall factor plane (all ones for interior kf).

        implicit none
        integer, intent(in) :: kf, njp, ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
        real, intent(in) :: wfac(ni-1, nj-1)
        real, intent(inout) :: plane(ni, njp, 5)

        integer :: i, j
        real :: pm(6), mf(3), mdot

        do j = 1, nj-1
        do i = 1, ni-1
            pm = 0.0e0
            mf = 0.0e0
            call accum(pm, mf, i,   j,   kf, wfac(i,j))
            call accum(pm, mf, i+1, j,   kf, wfac(i,j))
            call accum(pm, mf, i,   j+1, kf, wfac(i,j))
            call accum(pm, mf, i+1, j+1, kf, wfac(i,j))
            mdot = mf(1)*dA(1,i,j,kf) + mf(2)*dA(2,i,j,kf) + mf(3)*dA(3,i,j,kf)
            plane(i,j,1) = mdot
            plane(i,j,2) = pm(1)*mdot + pm(5)*dA(1,i,j,kf)
            plane(i,j,3) = pm(2)*mdot + pm(5)*dA(2,i,j,kf)
            plane(i,j,4) = pm(3)*mdot + pm(6)*dA(3,i,j,kf)
            plane(i,j,5) = pm(4)*mdot + Omega*pm(6)*dA(3,i,j,kf)
        end do
        end do

    contains
        pure subroutine accum(pm, mf, i, j, k, wfac)
            real, intent(inout) :: pm(6), mf(3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real :: dp, w
            dp = P(i,j,k) - P_offset
            pm(1) = pm(1) + 0.25e0*vx(i,j,k)
            pm(2) = pm(2) + 0.25e0*vr(i,j,k)
            pm(3) = pm(3) + 0.25e0*r(i,j,k)*vt(i,j,k)
            pm(4) = pm(4) + 0.25e0*ho(i,j,k)
            pm(5) = pm(5) + 0.25e0*dp
            pm(6) = pm(6) + 0.25e0*r(i,j,k)*dp
            w = 0.25e0*wfac
            mf(1) = mf(1) + w*cons(i,j,k,2)
            mf(2) = mf(2) + w*cons(i,j,k,3)
            mf(3) = mf(3) + w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k))
        end subroutine accum
    end subroutine kface_flow_plane_clean

end module residual_clean_helpers


! =====================================================================
! Idiomatic set_residual. Same algorithm as the production kernel: a
! single k sweep with rolling face buffers (an i-face row, an alternating
! j-face row pair, an alternating k-face plane pair), all three direction
! contributions folded into one dU write, and the cusp seam deferred to an
! O(surface) correction pass afterwards.
!
! Differs from production only in *how* the face loops are written -- see
! this file's header. No kb dummy: the production slab loop is already a
! pure re-nesting of `do k = 1, nk-1`.
! =====================================================================
subroutine set_residual_clean( &
    cons, P, P_offset, &
    r, Omega, dAi, dAj, dAk, &
    f_body, &
    dU, &
    vx, vr, vt, ho, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    njp, ni, nj, nk &
    )

    use residual_clean_helpers
    use residual_helpers, only: correct_cusp_kface_du

    implicit none

    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
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
    ! Rolling flow scratch, as in production: planes holds the k-face plane
    ! pair (slots pa/pb), rows holds the i-face row (slot 1) and the j-face
    ! row pair (slots ja/jb alternating 2/3). njp is planes' padded j-extent
    ! (nj+1 when ni*nj*4 bytes is a whole page multiple, nj otherwise).
    real, intent(inout) :: planes(ni, njp, 5, 2)
    real, intent(inout) :: rows(ni, 5, 3)
    integer, intent(in) :: njp, ni, nj, nk

    integer :: i, j, k, m, ja, jb, pa, pb, stmp
    ! Wall-factor rows/planes handed to the face helpers. The interior faces
    ! take the all-ones buffers; only the six boundary faces take a real wall
    ! mask. Filling these once per call is what lets each helper be a single
    ! loop instead of three near-identical copies.
    real :: ones_j(ni-1), ones_k(ni-1, nj-1)
    real :: wfac_i(ni)

    ones_j = 1.0e0
    ones_k = 1.0e0
    ! wfac_i's interior stays 1.0 for the whole call; only slots 1 and ni are
    ! rewritten per cell row.
    wfac_i = 1.0e0

    pa = 1
    pb = 2

    ! Prime the rolling k-face plane with face k=1 (the sweep below always
    ! has face k resident in pa on entry to cell layer k).
    call kface_flow_plane_clean(vx, vr, vt, ho, P, P_offset, r, cons, &
                                Omega, dAk, wallk1, planes(:,:,:,pa), &
                                1, njp, ni, nj, nk)

    do k = 1, nk-1
        ja = 2
        jb = 3
        ! Prime the rolling j-face pair with the j=1 boundary face.
        call jface_flow_row_clean(vx, vr, vt, ho, P, P_offset, r, cons, &
                                  Omega, dAj, wallj1(:,k), rows(:,:,ja), &
                                  1, k, ni, nj, nk)
        ! Advance the rolling k-face pair: pa holds face k already, pb gets
        ! face k+1 fresh. Only the k=nk face carries a wall mask.
        if (k+1 == nk) then
            call kface_flow_plane_clean(vx, vr, vt, ho, P, P_offset, r, cons, &
                                        Omega, dAk, wallnk, planes(:,:,:,pb), &
                                        k+1, njp, ni, nj, nk)
        else
            call kface_flow_plane_clean(vx, vr, vt, ho, P, P_offset, r, cons, &
                                        Omega, dAk, ones_k, planes(:,:,:,pb), &
                                        k+1, njp, ni, nj, nk)
        end if
        do j = 1, nj-1
            ! i-face wall factors: interior faces are unmasked, the i=1 and
            ! i=ni faces carry this row's wall mask (a scalar per row). Only
            ! the two end slots change per row -- the interior stays 1.0 from
            ! the one-off fill above.
            wfac_i(1) = walli1(j,k)
            wfac_i(ni) = wallni(j,k)
            call iface_flow_row_clean(vx, vr, vt, ho, P, P_offset, r, cons, &
                                      Omega, dAi, wfac_i, &
                                      rows(:,:,1), j, k, ni, nj, nk)
            if (j+1 == nj) then
                call jface_flow_row_clean(vx, vr, vt, ho, P, P_offset, r, cons, &
                                          Omega, dAj, wallnj(:,k), rows(:,:,jb), &
                                          j+1, k, ni, nj, nk)
            else
                call jface_flow_row_clean(vx, vr, vt, ho, P, P_offset, r, cons, &
                                          Omega, dAj, ones_j, rows(:,:,jb), &
                                          j+1, k, ni, nj, nk)
            end if
            do m = 1, 5
            do i = 1, ni-1
                dU(i,j,k,m) = rows(i,m,1) - rows(i+1,m,1) + f_body(i,j,k,m) &
                            + rows(i,m,ja) - rows(i,m,jb) &
                            + planes(i,j,m,pa) - planes(i,j,m,pb)
            end do
            end do
            stmp = ja
            ja = jb
            jb = stmp
        end do
        stmp = pa
        pa = pb
        pb = stmp
    end do

    ! Cusp seam: non-local in k (couples the k=1 and k=nk faces), applied as
    ! a deferred O(surface) correction. nk=2 (seam cells coincide) unsupported.
    ! Shared verbatim with production -- it is O(surface) and already written
    ! in the idiomatic pm/mf array style, so there is nothing to vary here.
    if (i_cusp_start > 0 .and. nk > 2) then
        call correct_cusp_kface_du(vx, vr, vt, ho, P, P_offset, r, cons, &
                                   Omega, dAk, wallk1, wallnk, dU, &
                                   i_cusp_start, i_cusp_end, ni, nj, nk)
    end if

end subroutine set_residual_clean
