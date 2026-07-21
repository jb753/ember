! Blended 4th and 2nd order smoothing with constant, isotropic factors.
!
! A shock-sensor-free smoother: no shock sensor, no CFL scaling, no L weighting.
! sf2 and sf4 are independent: sf2 drives 2nd-order smoothing and sf4 drives
! 4th-order smoothing with no coupling between them.
!
! The 4th-order term is applied as M = S^T S, where S is the 2nd-difference
! operator taken only where its centred stencil fits (no boundary closure is
! invented). M is therefore symmetric positive semi-definite by construction,
! so I - (sf4/6) M is non-amplifying for every sf4 <= 0.75: ||A^k||_2 = 1 at all
! k. The boundary rows fall out as (1,-2,1) at i=1 and (-2,5,-4,1) at i=2.
!
! The trade: those rows are 2nd differences, so near a face the 4th-order term
! degrades to 2nd-order dissipation and a cubic is NOT reproduced at i=1,2 and
! i=ni-1,ni. Linear fields still are, everywhere. The earlier scheme shared one
! biased 4th difference between i=1 and i=2 to keep cubic exactness at the wall;
! that made the boundary block defective and amplified the non-cubic residual
! like sf4*sqrt(k) under repeated application.
!
! Separable rolling-plane version. The smoother is a sum of three independent
! 1-D operators applied to the *original* field, so the work is done as a
! forward sweep over k-planes, each plane assembled in three vectorising passes:
!
!   PASS K  base + k-operator     -> xs(:,:,slot)      (do j, do i)
!   PASS I  += i-operator         (i interior do i; i=1,2,ni-1,ni branch-free)
!   PASS J  += j-operator         (j interior/biased, all do i)
!
! Because each output plane reads only the *input* field, results are written
! back into x in place with a two-plane lag (the interior k-stencil reaches
! k +/- 2): after plane k is assembled, input plane k-2 has had its last reader,
! so x(:,:,k-2) is overwritten from the rolling buffer. The high-k biased
! stencils (k=nk-1,nk) reach back to plane nk-4, so the top five planes are held
! in the buffer and flushed after the sweep instead of rolled. This removes the
! full-volume xs work array and the separate x = xs copyback of the earlier
! all-at-once version: x is streamed once in and once out per component.
!
! Results match the earlier version to a bounded float32 tolerance (the three
! directional contributions are summed as three partial sums rather than one
! expression, a legal -Ofast reassociation); it is not bitwise identical.
!
! xs is a pre-allocated (ni,nj,kr) rolling buffer (intent inout), kr>=min(6,nk)
! planes; the caller carves it zero-copy from the block scratch.
! Requires ni,nj,nk >= 5.
!
subroutine smooth3d_const( &
        x, &        ! Array to smooth (ni,nj,nk,np)
        sf4, sf2, & ! 4th and 2nd order smoothing factors
        xs, &       ! Rolling plane buffer (ni,nj,kr)
        ni, nj, nk, np, kr &
    )

    integer, intent(in) :: ni, nj, nk, np, kr
    real,    intent(in) :: sf4, sf2
    real, intent(inout) :: x(ni, nj, nk, np)
    real, intent(inout) :: xs(ni, nj, kr)

    real, parameter :: c16 = 1e0 / 6e0

    real :: sum_sf, base
    real :: d4, d4b, s2, s4
    integer :: i, j, k, ip, slot, ps

    if (ni < 5 .or. nj < 5 .or. nk < 5) then
        stop 'smooth3d_const: ni,nj,nk must be >= 5'
    end if
    if (kr < min(6, nk)) then
        stop 'smooth3d_const: xs must have >= min(6,nk) planes'
    end if

    sum_sf = 3e0 * (sf2 + sf4)
    base = 1e0 - sum_sf

    do ip = 1, np

        do k = 1, nk

            slot = mod(k - 1, kr) + 1

            ! ============================================================
            ! PASS K: base + k-direction operator into the plane buffer
            ! (k-mode branch is outside the i,j loops, so the bodies are
            !  branch-free and vectorise over i).
            ! ============================================================
            if (k == 1) then
                do j = 1, nj
                    do i = 1, ni
                        d4 = (x(i,j,1,ip) - 2e0*x(i,j,2,ip) + x(i,j,3,ip)) * c16
                        xs(i,j,slot) = base*x(i,j,1,ip) &
                            + sf2*(2e0*x(i,j,2,ip) - x(i,j,3,ip)) &
                            + sf4*(x(i,j,1,ip) - d4)
                    end do
                end do
            else if (k == 2) then
                do j = 1, nj
                    do i = 1, ni
                        d4 = (-2e0*x(i,j,1,ip) + 5e0*x(i,j,2,ip) &
                            - 4e0*x(i,j,3,ip) + x(i,j,4,ip)) * c16
                        xs(i,j,slot) = base*x(i,j,2,ip) &
                            + sf2*((x(i,j,1,ip) + x(i,j,3,ip)) * 0.5e0) &
                            + sf4*(x(i,j,2,ip) - d4)
                    end do
                end do
            else if (k == nk-1) then
                do j = 1, nj
                    do i = 1, ni
                        d4 = (-2e0*x(i,j,nk,ip) + 5e0*x(i,j,nk-1,ip) &
                            - 4e0*x(i,j,nk-2,ip) + x(i,j,nk-3,ip)) * c16
                        xs(i,j,slot) = base*x(i,j,nk-1,ip) &
                            + sf2*((x(i,j,nk-2,ip) + x(i,j,nk,ip)) * 0.5e0) &
                            + sf4*(x(i,j,nk-1,ip) - d4)
                    end do
                end do
            else if (k == nk) then
                do j = 1, nj
                    do i = 1, ni
                        d4 = (x(i,j,nk,ip) - 2e0*x(i,j,nk-1,ip) + x(i,j,nk-2,ip)) * c16
                        xs(i,j,slot) = base*x(i,j,nk,ip) &
                            + sf2*(2e0*x(i,j,nk-1,ip) - x(i,j,nk-2,ip)) &
                            + sf4*(x(i,j,nk,ip) - d4)
                    end do
                end do
            else
                do j = 1, nj
                    do i = 1, ni
                        s2 = (x(i,j,k-1,ip) + x(i,j,k+1,ip)) * 0.5e0
                        s4 = (-x(i,j,k-2,ip) + 4e0*x(i,j,k-1,ip) &
                            + 4e0*x(i,j,k+1,ip) - x(i,j,k+2,ip)) * c16
                        xs(i,j,slot) = base*x(i,j,k,ip) + sf2*s2 + sf4*s4
                    end do
                end do
            end if

            ! ============================================================
            ! PASS I: add the i-direction operator (reads plane k only).
            ! i interior vectorises; i=1,2,ni-1,ni are 4 branch-free columns.
            ! ============================================================
            do j = 1, nj
                do i = 3, ni-2
                    s2 = (x(i-1,j,k,ip) + x(i+1,j,k,ip)) * 0.5e0
                    s4 = (-x(i-2,j,k,ip) + 4e0*x(i-1,j,k,ip) &
                        + 4e0*x(i+1,j,k,ip) - x(i+2,j,k,ip)) * c16
                    xs(i,j,slot) = xs(i,j,slot) + sf2*s2 + sf4*s4
                end do
                d4 = (x(1,j,k,ip) - 2e0*x(2,j,k,ip) + x(3,j,k,ip)) * c16
                d4b = (-2e0*x(1,j,k,ip) + 5e0*x(2,j,k,ip) &
                    - 4e0*x(3,j,k,ip) + x(4,j,k,ip)) * c16
                xs(1,j,slot) = xs(1,j,slot) &
                    + sf2*(2e0*x(2,j,k,ip) - x(3,j,k,ip)) + sf4*(x(1,j,k,ip) - d4)
                xs(2,j,slot) = xs(2,j,slot) &
                    + sf2*((x(1,j,k,ip) + x(3,j,k,ip)) * 0.5e0) + sf4*(x(2,j,k,ip) - d4b)
                d4 = (x(ni,j,k,ip) - 2e0*x(ni-1,j,k,ip) + x(ni-2,j,k,ip)) * c16
                d4b = (-2e0*x(ni,j,k,ip) + 5e0*x(ni-1,j,k,ip) &
                    - 4e0*x(ni-2,j,k,ip) + x(ni-3,j,k,ip)) * c16
                xs(ni-1,j,slot) = xs(ni-1,j,slot) &
                    + sf2*((x(ni-2,j,k,ip) + x(ni,j,k,ip)) * 0.5e0) + sf4*(x(ni-1,j,k,ip) - d4b)
                xs(ni,j,slot) = xs(ni,j,slot) &
                    + sf2*(2e0*x(ni-1,j,k,ip) - x(ni-2,j,k,ip)) + sf4*(x(ni,j,k,ip) - d4)
            end do

            ! ============================================================
            ! PASS J: add the j-direction operator (reads plane k only).
            ! All bodies vectorise over i; the j-mode branch is outside do i.
            ! ============================================================
            do i = 1, ni
                d4 = (x(i,1,k,ip) - 2e0*x(i,2,k,ip) + x(i,3,k,ip)) * c16
                xs(i,1,slot) = xs(i,1,slot) &
                    + sf2*(2e0*x(i,2,k,ip) - x(i,3,k,ip)) + sf4*(x(i,1,k,ip) - d4)
            end do
            do i = 1, ni
                d4 = (-2e0*x(i,1,k,ip) + 5e0*x(i,2,k,ip) &
                    - 4e0*x(i,3,k,ip) + x(i,4,k,ip)) * c16
                xs(i,2,slot) = xs(i,2,slot) &
                    + sf2*((x(i,1,k,ip) + x(i,3,k,ip)) * 0.5e0) + sf4*(x(i,2,k,ip) - d4)
            end do
            do j = 3, nj-2
                do i = 1, ni
                    s2 = (x(i,j-1,k,ip) + x(i,j+1,k,ip)) * 0.5e0
                    s4 = (-x(i,j-2,k,ip) + 4e0*x(i,j-1,k,ip) &
                        + 4e0*x(i,j+1,k,ip) - x(i,j+2,k,ip)) * c16
                    xs(i,j,slot) = xs(i,j,slot) + sf2*s2 + sf4*s4
                end do
            end do
            do i = 1, ni
                d4 = (-2e0*x(i,nj,k,ip) + 5e0*x(i,nj-1,k,ip) &
                    - 4e0*x(i,nj-2,k,ip) + x(i,nj-3,k,ip)) * c16
                xs(i,nj-1,slot) = xs(i,nj-1,slot) &
                    + sf2*((x(i,nj-2,k,ip) + x(i,nj,k,ip)) * 0.5e0) + sf4*(x(i,nj-1,k,ip) - d4)
            end do
            do i = 1, ni
                d4 = (x(i,nj,k,ip) - 2e0*x(i,nj-1,k,ip) + x(i,nj-2,k,ip)) * c16
                xs(i,nj,slot) = xs(i,nj,slot) &
                    + sf2*(2e0*x(i,nj-1,k,ip) - x(i,nj-2,k,ip)) + sf4*(x(i,nj,k,ip) - d4)
            end do

            ! ============================================================
            ! Lagged in-place writeback: plane k-2 is now dead for every
            ! remaining interior stencil. Planes nk-4..nk are held (the high-k
            ! biased stencils still read them) and flushed after the sweep.
            ! ============================================================
            if (k-2 >= 1 .and. k-2 <= nk-5) then
                ps = mod(k-3, kr) + 1
                x(:,:,k-2,ip) = xs(:,:,ps)
            end if

        end do

        ! Flush the retained top planes (up to five) back into x.
        do k = max(1, nk-4), nk
            ps = mod(k-1, kr) + 1
            x(:,:,k,ip) = xs(:,:,ps)
        end do

    end do

end subroutine smooth3d_const
