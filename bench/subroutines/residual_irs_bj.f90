! Diagnostic-only: smooth_residual_tri_tiled at several transpose-tile
! widths BJ. Production fixes BJ = 8 with the comment "AVX = 8 float32
! lanes", but this machine is AVX-512 (16 float32 lanes), so BJ = 8 fills
! only half a zmm register in the i-solve's lane loop. The i-solve is 68%
! of IRS (9.0 of 13.5 ns/cell, measured per-direction), so its tile width
! is worth a sweep. Arithmetic is identical for every BJ -- the tile only
! groups independent j-lines -- so all variants must agree bitwise.


subroutine smooth_residual_tri_bj16(dU, sf, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: i, j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: cc, mm
    integer, parameter :: BJ = 16           ! swept tile width
    integer :: jj, j0, nb
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
    if (nci >= 2) then
        do m = 1, 5
        do k = 1, nck
        do j0 = 1, ncj, BJ
            nb = min(BJ, ncj - j0 + 1)
            do i = 1, nci
                do jj = 1, nb
                    tile(jj,i) = dU(i, j0+jj-1, k, m)
                end do
            end do
            mm = work(bmii+1)
            do jj = 1, nb
                tile(jj,1) = tile(jj,1) * mm
            end do
            do i = 2, nci
                mm = work(bmii+i)
                do jj = 1, nb
                    tile(jj,i) = (tile(jj,i) + sf*tile(jj,i-1)) * mm
                end do
            end do
            do i = nci-1, 1, -1
                cc = work(bcpi+i)
                do jj = 1, nb
                    tile(jj,i) = tile(jj,i) - cc*tile(jj,i+1)
                end do
            end do
            do i = 1, nci
                do jj = 1, nb
                    dU(i, j0+jj-1, k, m) = tile(jj,i)
                end do
            end do
        end do
        end do
        end do
    end if

    ! ---- j-direction: recurrence along j, innermost vector loop over i.
    ! The per-plane factors are loop-invariant over i, so hoist to scalars. ----
    if (ncj >= 2) then
        do m = 1, 5
        do k = 1, nck
            mm = work(bmij+1)
            do i = 1, nci
                dU(i,1,k,m) = dU(i,1,k,m) * mm
            end do
            do j = 2, ncj
                mm = work(bmij+j)
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j-1,k,m)) * mm
                end do
            end do
            do j = ncj-1, 1, -1
                cc = work(bcpj+j)
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j+1,k,m)
                end do
            end do
        end do
        end do
    end if

    ! ---- k-direction: recurrence along k, innermost vector loop over i ----
    if (nck >= 2) then
        do m = 1, 5
            mm = work(bmik+1)
            do j = 1, ncj
            do i = 1, nci
                dU(i,j,1,m) = dU(i,j,1,m) * mm
            end do
            end do
            do k = 2, nck
                mm = work(bmik+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j,k-1,m)) * mm
                end do
                end do
            end do
            do k = nck-1, 1, -1
                cc = work(bcpk+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j,k+1,m)
                end do
                end do
            end do
        end do
    end if

contains

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

end subroutine smooth_residual_tri_bj16

subroutine smooth_residual_tri_bj32(dU, sf, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: i, j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: cc, mm
    integer, parameter :: BJ = 32           ! swept tile width
    integer :: jj, j0, nb
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
    if (nci >= 2) then
        do m = 1, 5
        do k = 1, nck
        do j0 = 1, ncj, BJ
            nb = min(BJ, ncj - j0 + 1)
            do i = 1, nci
                do jj = 1, nb
                    tile(jj,i) = dU(i, j0+jj-1, k, m)
                end do
            end do
            mm = work(bmii+1)
            do jj = 1, nb
                tile(jj,1) = tile(jj,1) * mm
            end do
            do i = 2, nci
                mm = work(bmii+i)
                do jj = 1, nb
                    tile(jj,i) = (tile(jj,i) + sf*tile(jj,i-1)) * mm
                end do
            end do
            do i = nci-1, 1, -1
                cc = work(bcpi+i)
                do jj = 1, nb
                    tile(jj,i) = tile(jj,i) - cc*tile(jj,i+1)
                end do
            end do
            do i = 1, nci
                do jj = 1, nb
                    dU(i, j0+jj-1, k, m) = tile(jj,i)
                end do
            end do
        end do
        end do
        end do
    end if

    ! ---- j-direction: recurrence along j, innermost vector loop over i.
    ! The per-plane factors are loop-invariant over i, so hoist to scalars. ----
    if (ncj >= 2) then
        do m = 1, 5
        do k = 1, nck
            mm = work(bmij+1)
            do i = 1, nci
                dU(i,1,k,m) = dU(i,1,k,m) * mm
            end do
            do j = 2, ncj
                mm = work(bmij+j)
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j-1,k,m)) * mm
                end do
            end do
            do j = ncj-1, 1, -1
                cc = work(bcpj+j)
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j+1,k,m)
                end do
            end do
        end do
        end do
    end if

    ! ---- k-direction: recurrence along k, innermost vector loop over i ----
    if (nck >= 2) then
        do m = 1, 5
            mm = work(bmik+1)
            do j = 1, ncj
            do i = 1, nci
                dU(i,j,1,m) = dU(i,j,1,m) * mm
            end do
            end do
            do k = 2, nck
                mm = work(bmik+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j,k-1,m)) * mm
                end do
                end do
            end do
            do k = nck-1, 1, -1
                cc = work(bcpk+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j,k+1,m)
                end do
                end do
            end do
        end do
    end if

contains

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

end subroutine smooth_residual_tri_bj32

subroutine smooth_residual_tri_bj64(dU, sf, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: i, j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: cc, mm
    integer, parameter :: BJ = 64           ! swept tile width
    integer :: jj, j0, nb
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
    if (nci >= 2) then
        do m = 1, 5
        do k = 1, nck
        do j0 = 1, ncj, BJ
            nb = min(BJ, ncj - j0 + 1)
            do i = 1, nci
                do jj = 1, nb
                    tile(jj,i) = dU(i, j0+jj-1, k, m)
                end do
            end do
            mm = work(bmii+1)
            do jj = 1, nb
                tile(jj,1) = tile(jj,1) * mm
            end do
            do i = 2, nci
                mm = work(bmii+i)
                do jj = 1, nb
                    tile(jj,i) = (tile(jj,i) + sf*tile(jj,i-1)) * mm
                end do
            end do
            do i = nci-1, 1, -1
                cc = work(bcpi+i)
                do jj = 1, nb
                    tile(jj,i) = tile(jj,i) - cc*tile(jj,i+1)
                end do
            end do
            do i = 1, nci
                do jj = 1, nb
                    dU(i, j0+jj-1, k, m) = tile(jj,i)
                end do
            end do
        end do
        end do
        end do
    end if

    ! ---- j-direction: recurrence along j, innermost vector loop over i.
    ! The per-plane factors are loop-invariant over i, so hoist to scalars. ----
    if (ncj >= 2) then
        do m = 1, 5
        do k = 1, nck
            mm = work(bmij+1)
            do i = 1, nci
                dU(i,1,k,m) = dU(i,1,k,m) * mm
            end do
            do j = 2, ncj
                mm = work(bmij+j)
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j-1,k,m)) * mm
                end do
            end do
            do j = ncj-1, 1, -1
                cc = work(bcpj+j)
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j+1,k,m)
                end do
            end do
        end do
        end do
    end if

    ! ---- k-direction: recurrence along k, innermost vector loop over i ----
    if (nck >= 2) then
        do m = 1, 5
            mm = work(bmik+1)
            do j = 1, ncj
            do i = 1, nci
                dU(i,j,1,m) = dU(i,j,1,m) * mm
            end do
            end do
            do k = 2, nck
                mm = work(bmik+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j,k-1,m)) * mm
                end do
                end do
            end do
            do k = nck-1, 1, -1
                cc = work(bcpk+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j,k+1,m)
                end do
                end do
            end do
        end do
    end if

contains

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

end subroutine smooth_residual_tri_bj64

subroutine smooth_residual_tri_bj128(dU, sf, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: i, j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: cc, mm
    integer, parameter :: BJ = 128           ! swept tile width
    integer :: jj, j0, nb
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
    if (nci >= 2) then
        do m = 1, 5
        do k = 1, nck
        do j0 = 1, ncj, BJ
            nb = min(BJ, ncj - j0 + 1)
            do i = 1, nci
                do jj = 1, nb
                    tile(jj,i) = dU(i, j0+jj-1, k, m)
                end do
            end do
            mm = work(bmii+1)
            do jj = 1, nb
                tile(jj,1) = tile(jj,1) * mm
            end do
            do i = 2, nci
                mm = work(bmii+i)
                do jj = 1, nb
                    tile(jj,i) = (tile(jj,i) + sf*tile(jj,i-1)) * mm
                end do
            end do
            do i = nci-1, 1, -1
                cc = work(bcpi+i)
                do jj = 1, nb
                    tile(jj,i) = tile(jj,i) - cc*tile(jj,i+1)
                end do
            end do
            do i = 1, nci
                do jj = 1, nb
                    dU(i, j0+jj-1, k, m) = tile(jj,i)
                end do
            end do
        end do
        end do
        end do
    end if

    ! ---- j-direction: recurrence along j, innermost vector loop over i.
    ! The per-plane factors are loop-invariant over i, so hoist to scalars. ----
    if (ncj >= 2) then
        do m = 1, 5
        do k = 1, nck
            mm = work(bmij+1)
            do i = 1, nci
                dU(i,1,k,m) = dU(i,1,k,m) * mm
            end do
            do j = 2, ncj
                mm = work(bmij+j)
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j-1,k,m)) * mm
                end do
            end do
            do j = ncj-1, 1, -1
                cc = work(bcpj+j)
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j+1,k,m)
                end do
            end do
        end do
        end do
    end if

    ! ---- k-direction: recurrence along k, innermost vector loop over i ----
    if (nck >= 2) then
        do m = 1, 5
            mm = work(bmik+1)
            do j = 1, ncj
            do i = 1, nci
                dU(i,j,1,m) = dU(i,j,1,m) * mm
            end do
            end do
            do k = 2, nck
                mm = work(bmik+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j,k-1,m)) * mm
                end do
                end do
            end do
            do k = nck-1, 1, -1
                cc = work(bcpk+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j,k+1,m)
                end do
                end do
            end do
        end do
    end if

contains

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

end subroutine smooth_residual_tri_bj128
