! Diagnostic-only: smooth_residual_tri_tiled with per-direction switches so
! each of the three Thomas solves can be timed in isolation. Identical
! arithmetic; the flags only skip whole solves. Not for production use --
! it exists to answer "which IRS direction actually costs the time?".
subroutine smooth_residual_tri_dirs(dU, sf, work, do_i, do_j, do_k, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    integer, intent(in) :: do_i, do_j, do_k
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: i, j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: cc, mm
    ! Must track production's BJ (residual.f90), or the i-direction share of
    ! the split this arm exists to measure is not production's.
    integer, parameter :: BJ = 32
    ! Likewise for production's transpose block edge.
    integer, parameter :: TB = 8
    integer :: j0, nb
    real    :: tile(BJ, ni-1)               ! (lane, i) transposed i-solve pad

    if (sf <= 0.0e0) return

    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    ! Base offsets of the six coefficient vectors packed into work:
    ! [cpi | minvi | cpj | minvj | cpk | minvk], lengths nci,nci,ncj,ncj,nck,nck.
    bcpi = 0
    bmii = nci
    bcpj = 2*nci
    bmij = 2*nci + ncj
    bcpk = 2*nci + 2*ncj
    bmik = 2*nci + 2*ncj + nck
    call tri_coeffs(sf, nci, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci))
    call tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))

    ! ---- i-direction: transpose-tiled. A BJ-wide block of j-lines is gathered
    ! into tile(lane, i); the recurrence then runs along i with the innermost
    ! loop over the BJ contiguous, independent lanes -> vectorises + hides the
    ! recurrence latency. Scatter back afterwards. (nci >= 2) ----
    if (nci >= 2 .and. do_i /= 0) then
        do m = 1, 5
        do k = 1, nck
        do j0 = 1, ncj, BJ
            nb = min(BJ, ncj - j0 + 1)
            call gather_tile(dU(1,j0,k,m), tile, nci, nb)
            mm = work(bmii+1)
            do i = 1, nb
                tile(i,1) = tile(i,1) * mm
            end do
            do i = 2, nci
                mm = work(bmii+i)
                call tile_fwd(tile(1,i), tile(1,i-1), sf, mm, nb)
            end do
            do i = nci-1, 1, -1
                cc = work(bcpi+i)
                call tile_back(tile(1,i), tile(1,i+1), cc, nb)
            end do
            call scatter_tile(dU(1,j0,k,m), tile, nci, nb)
        end do
        end do
        end do
    end if

    ! ---- j- and k-directions, through the same three line kernels production
    ! uses (residual.f90). Kept in step with production deliberately: this arm
    ! measures how production's time SPLITS across directions, so any structural
    ! difference here would answer a question about a kernel nobody runs. ----
    if (ncj >= 2 .and. do_j /= 0) then
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

    if (nck >= 2 .and. do_k /= 0) then
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

    ! Production's blocked transpose (residual.f90), verbatim.
    subroutine gather_tile(src, tl, n, nb_in)
        implicit none
        integer, intent(in) :: n, nb_in
        real, intent(in)    :: src(n, nb_in)
        real, intent(inout) :: tl(BJ, n)
        real    :: blk(TB, TB)
        integer :: i0, jj0, ii, jj, nfull_i, nfull_j
        nfull_i = (n / TB) * TB
        nfull_j = (nb_in / TB) * TB
        do jj0 = 1, nfull_j, TB
            do i0 = 1, nfull_i, TB
                do jj = 1, TB
                    do ii = 1, TB
                        blk(ii,jj) = src(i0+ii-1, jj0+jj-1)
                    end do
                end do
                do ii = 1, TB
                    do jj = 1, TB
                        tl(jj0+jj-1, i0+ii-1) = blk(ii,jj)
                    end do
                end do
            end do
        end do
        do jj = 1, nfull_j
            do ii = nfull_i+1, n
                tl(jj,ii) = src(ii,jj)
            end do
        end do
        do jj = nfull_j+1, nb_in
            do ii = 1, n
                tl(jj,ii) = src(ii,jj)
            end do
        end do
    end subroutine gather_tile

    subroutine scatter_tile(dst, tl, n, nb_in)
        implicit none
        integer, intent(in) :: n, nb_in
        real, intent(inout) :: dst(n, nb_in)
        real, intent(in)    :: tl(BJ, n)
        real    :: blk(TB, TB)
        integer :: i0, jj0, ii, jj, nfull_i, nfull_j
        nfull_i = (n / TB) * TB
        nfull_j = (nb_in / TB) * TB
        do jj0 = 1, nfull_j, TB
            do i0 = 1, nfull_i, TB
                do ii = 1, TB
                    do jj = 1, TB
                        blk(ii,jj) = tl(jj0+jj-1, i0+ii-1)
                    end do
                end do
                do jj = 1, TB
                    do ii = 1, TB
                        dst(i0+ii-1, jj0+jj-1) = blk(ii,jj)
                    end do
                end do
            end do
        end do
        do jj = 1, nfull_j
            do ii = nfull_i+1, n
                dst(ii,jj) = tl(jj,ii)
            end do
        end do
        do jj = nfull_j+1, nb_in
            do ii = 1, n
                dst(ii,jj) = tl(jj,ii)
            end do
        end do
    end subroutine scatter_tile

    subroutine tile_fwd(x, xprev, e, pv, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: e, pv
        real, intent(in)    :: xprev(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = (x(ii) + e*xprev(ii)) * pv
        end do
    end subroutine tile_fwd

    subroutine tile_back(x, xnext, cc_in, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: cc_in
        real, intent(in)    :: xnext(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) - cc_in*xnext(ii)
        end do
    end subroutine tile_back

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

    ! Thomas forward-sweep factors for the constant-coefficient Neumann
    ! tridiagonal along a line of length n: a = c = -sf, b = 1+2sf interior,
    ! b = 1+sf at the two ends. Returns cp (eliminated super-diagonal) and
    ! minv = 1/pivot, so a line solve is:
    !   x(1)   = d(1)*minv(1)
    !   x(i)   = (d(i) + sf*x(i-1))*minv(i)          i = 2..n   (forward)
    !   x(i)   = x(i) - cp(i)*x(i+1)                 i = n-1..1 (back-sub)
    ! An n=1 line has no neighbours -> operator is the identity (minv=1).
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

        ! Row 1: b = 1 + sf (single neighbour), c = -sf.
        minv(1) = 1.0e0 / (1.0e0 + e)
        cp(1)   = -e * minv(1)
        do ii = 2, n-1
            minv(ii) = 1.0e0 / ((1.0e0 + 2.0e0*e) + e*cp(ii-1))
            cp(ii)   = -e * minv(ii)
        end do
        ! Row n: b = 1 + sf (single neighbour), c = 0.
        minv(n) = 1.0e0 / ((1.0e0 + e) + e*cp(n-1))
        cp(n)   = 0.0e0
    end subroutine tri_coeffs

end subroutine smooth_residual_tri_dirs
