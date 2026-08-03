! =====================================================================
! smooth_residual_tri_km -- IRS with the k-direction solve's component
! loop merged into its k sweeps. Benchmark-only companion to
! smooth_residual_tri_tiled (residual.f90); nothing calls it.
!
! Section 21 left the IRS smoother as the largest single item in
! Grid.update_residual (31.6 ns/cell saturated, against set_residual's
! 22.6). Inspecting its three direction solves, only the k-solve carries
! the redundancy that made damp_residual worth merging:
!
!   i-solve  transpose-tiled; each dU element is read and written once
!            per solve already. Nothing to merge.
!   j-solve  works one (i,j) plane at a time (~68 KB, L2-resident), and
!            the forward/back passes reuse that plane from cache.
!            Nothing to merge.
!   k-solve  the recurrence runs along k, so k must be the outer loop and
!            each pass streams the whole 3.7 MB per-component volume.
!            Production wraps that in `do m = 1, 5`, giving TEN
!            full-volume streams (five forward, five back-substitution).
!
! Merging m into the k sweeps makes it two streams instead of ten,
! saving ~30 MB of traffic per call at 1M cells. Only the k-solve is
! changed; the i- and j-solves are production's verbatim.
!
! Arithmetic is untouched -- same operations, same per-line recurrence
! order, only the order in which independent components are visited --
! so this should be bitwise identical.
! =====================================================================

subroutine smooth_residual_tri_km(dU, sf, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: i, j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: cc, mm
    integer, parameter :: BJ = 8            ! tile width (AVX = 8 float32 lanes)
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
    ! The component loop is merged INTO the k sweeps rather than wrapping
    ! them. Production runs the whole forward sweep and the whole
    ! back-substitution once per component, and each of those streams the
    ! full 3.7 MB volume -- so five components cost ten full-volume streams.
    ! Merged, one forward and one back-substitution pass cover all five: the
    ! working set per k step is five (i,j) planes (~340 KB at this size),
    ! comfortably L2-resident, so the whole solve costs two streams instead
    ! of ten.
    !
    ! m sits OUTSIDE the i loop, not innermost: dU is component-last, so
    ! dU(:,j,k,m) is contiguous in i only for fixed m. Making m innermost
    ! strides the i reads by the whole volume and destroys vectorization --
    ! measured at +208% when tried on damp_residual (section 21.2).
    if (nck >= 2) then
        mm = work(bmik+1)
        do m = 1, 5
        do j = 1, ncj
        do i = 1, nci
            dU(i,j,1,m) = dU(i,j,1,m) * mm
        end do
        end do
        end do
        do k = 2, nck
            mm = work(bmik+k)
            do m = 1, 5
            do j = 1, ncj
            do i = 1, nci
                dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j,k-1,m)) * mm
            end do
            end do
            end do
        end do
        do k = nck-1, 1, -1
            cc = work(bcpk+k)
            do m = 1, 5
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

end subroutine smooth_residual_tri_km
