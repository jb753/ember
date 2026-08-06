! A/B arms for the two opt-report blemishes in production's
! smooth_residual_tri_tiled (src/ember/_fortran/residual.f90):
!
!   1. ALIAS VERSIONING. All four j/k recurrence inner loops report
!      "loop versioned for vectorization because of possible aliasing"
!      (residual.f90:998,1004,1024,1032 in the link-stage -fopt-info-vec-all
!      report). GCC cannot prove dU(:,j,k,m) and dU(:,j-1,k,m) are disjoint,
!      so it emits a runtime overlap test and two copies of each body. The
!      i-solve's tile loops are NOT versioned -- `tile` is a local.
!
!   2. STACK ALLOCA. `tile(BJ, ni-1)` is an automatic array, so entry/exit
!      emit __builtin_alloca_with_align / __builtin_stack_restore, both
!      flagged "statement clobbers memory" (residual.f90:926,928).
!
! Two arms so the two fixes can be attributed separately:
!
!   smooth_residual_tri_na   fix 1 only. Production's signature, unchanged.
!   smooth_residual_tri_nat  fix 1 + fix 2. Carries the tile in `work`, so it
!                            needs a longer work array and takes its length
!                            as an argument.
!
! Fix 1 is the `scale_du` trick already used in set_residual (residual.f90's
! contains block): hoist the sweep into a contained subroutine taking the two
! operands as SEPARATE dummies. The standard forbids a caller from aliasing
! dummies when one of them is defined, so the compiler may assume disjointness
! and drop the runtime test.
!
! The line kernels are shaped to serve both directions. The j-solve's operands
! are dU(:,j,k,m) and dU(:,j+-1,k,m), contiguous runs of nci. The k-solve's are
! dU(:,:,k,m) and dU(:,:,k+-1,m), contiguous runs of nci*ncj -- so the same
! flat 1D kernel covers it, and the k-solve's (j,i) nest collapses into one
! long vector loop as a side effect. Both are reached by sequence association
! from an element reference, which is unambiguous and cannot raise an array
! temporary.
!
! BOTH ARMS ARE BITWISE IDENTICAL TO PRODUCTION BY CONSTRUCTION: same
! operations, same order, same operands. Nothing here is a numerics change,
! only a codegen one. If either fails the bitwise gate, it is a bug in the
! arm, not a tolerance to relax.
! =====================================================================


! ---------------------------------------------------------------------
! Arm `irsna`: fix 1 only. Signature identical to production's
! smooth_residual_tri_tiled, so it is a drop-in for a differential test.
! ---------------------------------------------------------------------
subroutine smooth_residual_tri_na(dU, sf, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    ! Production's tile width. Kept identical so this arm differs from
    ! production in exactly the one thing under test.
    integer, parameter :: BJ = 32
    real    :: tile(BJ, ni-1)               ! (lane, i) transposed i-solve pad

    if (sf <= 0.0e0) return

    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    bcpi = 0
    bmii = nci
    bcpj = 2*nci
    bmij = 2*nci + ncj
    bcpk = 2*nci + 2*ncj
    bmik = 2*nci + 2*ncj + nck
    call tri_coeffs(sf, nci, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci))
    call tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))

    ! ---- i-direction: unchanged from production (its tile loops are not
    ! alias-versioned, so there is nothing to fix here) ----
    if (nci >= 2) then
        call isolve(dU, sf, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci), &
                    tile, nci, ncj, nck)
    end if

    ! ---- j-direction: recurrence along j, one contiguous nci-run per line ----
    if (ncj >= 2) then
        do m = 1, 5
        do k = 1, nck
            call line_scale(dU(1,1,k,m), work(bmij+1), nci)
            do j = 2, ncj
                call line_fwd(dU(1,j,k,m), dU(1,j-1,k,m), sf, work(bmij+j), nci)
            end do
            do j = ncj-1, 1, -1
                call line_back(dU(1,j,k,m), dU(1,j+1,k,m), work(bcpj+j), nci)
            end do
        end do
        end do
    end if

    ! ---- k-direction: recurrence along k, one contiguous nci*ncj-run per
    ! plane (production's (j,i) nest collapses into a single vector loop) ----
    if (nck >= 2) then
        do m = 1, 5
            call line_scale(dU(1,1,1,m), work(bmik+1), nci*ncj)
            do k = 2, nck
                call line_fwd(dU(1,1,k,m), dU(1,1,k-1,m), sf, work(bmik+k), nci*ncj)
            end do
            do k = nck-1, 1, -1
                call line_back(dU(1,1,k,m), dU(1,1,k+1,m), work(bcpk+k), nci*ncj)
            end do
        end do
    end if

contains

    ! x = x*mm
    subroutine line_scale(x, mm, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: mm
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) * mm
        end do
    end subroutine line_scale

    ! Forward sweep: x = (x + sf*xprev)*mm. x and xprev are separate dummies,
    ! one defined and one not, so the compiler may assume they are disjoint --
    ! which is the whole point of this arm.
    subroutine line_fwd(x, xprev, e, mm, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: e, mm
        real, intent(in)    :: xprev(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = (x(ii) + e*xprev(ii)) * mm
        end do
    end subroutine line_fwd

    ! Back substitution: x = x - cc*xnext.
    subroutine line_back(x, xnext, cc, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: cc
        real, intent(in)    :: xnext(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) - cc*xnext(ii)
        end do
    end subroutine line_back

    ! Production's transpose-tiled i-solve, verbatim apart from taking the
    ! tile and the two coefficient vectors as dummies.
    subroutine isolve(d, e, cpi, minvi, tl, nc_i, nc_j, nc_k)
        implicit none
        integer, intent(in) :: nc_i, nc_j, nc_k
        real, intent(in)    :: e
        real, intent(in)    :: cpi(nc_i), minvi(nc_i)
        real, intent(inout) :: d(nc_i, nc_j, nc_k, 5)
        real, intent(inout) :: tl(BJ, nc_i)
        integer :: i, jj, j0, kk, mm2, nb
        real    :: cc, pv

        do mm2 = 1, 5
        do kk = 1, nc_k
        do j0 = 1, nc_j, BJ
            nb = min(BJ, nc_j - j0 + 1)
            do i = 1, nc_i
                do jj = 1, nb
                    tl(jj,i) = d(i, j0+jj-1, kk, mm2)
                end do
            end do
            pv = minvi(1)
            do jj = 1, nb
                tl(jj,1) = tl(jj,1) * pv
            end do
            do i = 2, nc_i
                pv = minvi(i)
                do jj = 1, nb
                    tl(jj,i) = (tl(jj,i) + e*tl(jj,i-1)) * pv
                end do
            end do
            do i = nc_i-1, 1, -1
                cc = cpi(i)
                do jj = 1, nb
                    tl(jj,i) = tl(jj,i) - cc*tl(jj,i+1)
                end do
            end do
            do i = 1, nc_i
                do jj = 1, nb
                    d(i, j0+jj-1, kk, mm2) = tl(jj,i)
                end do
            end do
        end do
        end do
        end do
    end subroutine isolve

    ! Identical to production's tri_coeffs (residual.f90).
    subroutine tri_coeffs(e, n, cp, minv)
        implicit none
        real, intent(in)     :: e
        integer, intent(in)  :: n
        real, intent(out)    :: cp(n), minv(n)
        integer :: ii

        if (n == 1) then
            minv(1) = 1.0e0
            cp(1)   = 0.0e0
            return
        end if

        minv(1) = 1.0e0 / (1.0e0 + e)
        cp(1)   = -e * minv(1)
        do ii = 2, n-1
            minv(ii) = 1.0e0 / ((1.0e0 + 2.0e0*e) + e*cp(ii-1))
            cp(ii)   = -e * minv(ii)
        end do
        minv(n) = 1.0e0 / ((1.0e0 + e) + e*cp(n-1))
        cp(n)   = 0.0e0
    end subroutine tri_coeffs

end subroutine smooth_residual_tri_na


! ---------------------------------------------------------------------
! Arm `irsnat`: fix 1 + fix 2. The i-solve tile is carried in `work` instead
! of being an automatic array, which removes the alloca/stack_restore pair.
!
! The cost is an interface change: `work` must be
!   2*(nci+ncj+nck) + BJ*nci
! elements instead of 2*(nci+ncj+nck), and the length is passed explicitly
! (f2py cannot parse a dimension expression containing BJ). In production this
! would mean touching grid.py's nwork, solver.py's _mg_coarse_scratch_sizes,
! and scree.f90's two `smoother` call sites -- which is why it is measured
! before it is adopted, not the other way round.
! ---------------------------------------------------------------------
subroutine smooth_residual_tri_nat(dU, sf, work, nwork, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk, nwork
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(inout) :: work(nwork)

    integer :: j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik, btile
    integer, parameter :: BJ = 32

    if (sf <= 0.0e0) return

    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    bcpi = 0
    bmii = nci
    bcpj = 2*nci
    bmij = 2*nci + ncj
    bcpk = 2*nci + 2*ncj
    bmik = 2*nci + 2*ncj + nck
    btile = 2*(nci + ncj + nck)
    call tri_coeffs(sf, nci, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci))
    call tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))

    if (nci >= 2) then
        call isolve(dU, sf, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci), &
                    work(btile+1:btile+BJ*nci), nci, ncj, nck)
    end if

    if (ncj >= 2) then
        do m = 1, 5
        do k = 1, nck
            call line_scale(dU(1,1,k,m), work(bmij+1), nci)
            do j = 2, ncj
                call line_fwd(dU(1,j,k,m), dU(1,j-1,k,m), sf, work(bmij+j), nci)
            end do
            do j = ncj-1, 1, -1
                call line_back(dU(1,j,k,m), dU(1,j+1,k,m), work(bcpj+j), nci)
            end do
        end do
        end do
    end if

    if (nck >= 2) then
        do m = 1, 5
            call line_scale(dU(1,1,1,m), work(bmik+1), nci*ncj)
            do k = 2, nck
                call line_fwd(dU(1,1,k,m), dU(1,1,k-1,m), sf, work(bmik+k), nci*ncj)
            end do
            do k = nck-1, 1, -1
                call line_back(dU(1,1,k,m), dU(1,1,k+1,m), work(bcpk+k), nci*ncj)
            end do
        end do
    end if

contains

    subroutine line_scale(x, mm, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: mm
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) * mm
        end do
    end subroutine line_scale

    subroutine line_fwd(x, xprev, e, mm, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: e, mm
        real, intent(in)    :: xprev(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = (x(ii) + e*xprev(ii)) * mm
        end do
    end subroutine line_fwd

    subroutine line_back(x, xnext, cc, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: cc
        real, intent(in)    :: xnext(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) - cc*xnext(ii)
        end do
    end subroutine line_back

    subroutine isolve(d, e, cpi, minvi, tl, nc_i, nc_j, nc_k)
        implicit none
        integer, intent(in) :: nc_i, nc_j, nc_k
        real, intent(in)    :: e
        real, intent(in)    :: cpi(nc_i), minvi(nc_i)
        real, intent(inout) :: d(nc_i, nc_j, nc_k, 5)
        real, intent(inout) :: tl(BJ, nc_i)
        integer :: i, jj, j0, kk, mm2, nb
        real    :: cc, pv

        do mm2 = 1, 5
        do kk = 1, nc_k
        do j0 = 1, nc_j, BJ
            nb = min(BJ, nc_j - j0 + 1)
            do i = 1, nc_i
                do jj = 1, nb
                    tl(jj,i) = d(i, j0+jj-1, kk, mm2)
                end do
            end do
            pv = minvi(1)
            do jj = 1, nb
                tl(jj,1) = tl(jj,1) * pv
            end do
            do i = 2, nc_i
                pv = minvi(i)
                do jj = 1, nb
                    tl(jj,i) = (tl(jj,i) + e*tl(jj,i-1)) * pv
                end do
            end do
            do i = nc_i-1, 1, -1
                cc = cpi(i)
                do jj = 1, nb
                    tl(jj,i) = tl(jj,i) - cc*tl(jj,i+1)
                end do
            end do
            do i = 1, nc_i
                do jj = 1, nb
                    d(i, j0+jj-1, kk, mm2) = tl(jj,i)
                end do
            end do
        end do
        end do
        end do
    end subroutine isolve

    subroutine tri_coeffs(e, n, cp, minv)
        implicit none
        real, intent(in)     :: e
        integer, intent(in)  :: n
        real, intent(out)    :: cp(n), minv(n)
        integer :: ii

        if (n == 1) then
            minv(1) = 1.0e0
            cp(1)   = 0.0e0
            return
        end if

        minv(1) = 1.0e0 / (1.0e0 + e)
        cp(1)   = -e * minv(1)
        do ii = 2, n-1
            minv(ii) = 1.0e0 / ((1.0e0 + 2.0e0*e) + e*cp(ii-1))
            cp(ii)   = -e * minv(ii)
        end do
        minv(n) = 1.0e0 / ((1.0e0 + e) + e*cp(n-1))
        cp(n)   = 0.0e0
    end subroutine tri_coeffs

end subroutine smooth_residual_tri_nat
