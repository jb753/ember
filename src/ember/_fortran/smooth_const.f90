! Blended 4th and 2nd order smoothing with constant, isotropic factors.
!
! A shock-sensor-free smoother: no shock sensor, no CFL scaling, no L weighting.
! sf2 and sf4 are independent: sf2 drives 2nd-order smoothing and sf4 drives
! 4th-order smoothing with no coupling between them.
!
! At i=1,2 and i=ni-1,ni (and equivalently for j, k) an asymmetric biased
! 4th-order difference is used instead of falling back to 2nd order, so the
! stencil leaves any cubic polynomial exactly unchanged everywhere.
!
! Boundary-loops version: the interior loop (i=3..ni-2, j=3..nj-2, k=3..nk-2)
! is branch-free, with boundary slabs handled in separate loops.
!
! Partition (each point covered exactly once):
!   i-slabs: i=1,2 and i=ni-1,ni (all j,k)
!   j-slabs: j=1,2 and j=nj-1,nj (i interior, all k)
!   k-slabs: k=1,2 and k=nk-1,nk (i,j interior)
!   interior: i=3..ni-2, j=3..nj-2, k=3..nk-2 (branch-free)
!
! xs must be a pre-allocated (ni,nj,nk) work array (intent inout).
! Requires ni,nj,nk >= 5.
!
subroutine smooth3d_const( &
        x, &        ! Array to smooth (ni,nj,nk,np)
        sf4, sf2, & ! 4th and 2nd order smoothing factors
        xs, &       ! Work array (ni,nj,nk), overwritten each component
        ni, nj, nk, np &
    )

    integer, intent(in) :: ni, nj, nk, np
    real,    intent(in) :: sf4, sf2
    real, intent(inout) :: x(ni, nj, nk, np)
    real, intent(inout) :: xs(ni, nj, nk)

    real :: sum_sf
    real :: xs2_i, xs2_j, xs2_k
    real :: xs4_i, xs4_j, xs4_k
    real :: d4i, d4j, d4k
    integer :: i, j, k, ip

    if (ni < 5 .or. nj < 5 .or. nk < 5) then
        stop 'smooth3d_const: ni,nj,nk must be >= 5'
    end if

    sum_sf = 3e0 * (sf2 + sf4)

    do ip = 1, np

        ! ----------------------------------------------------------------
        ! Interior: i=3..ni-2, j=3..nj-2, k=3..nk-2   zero branches
        ! ----------------------------------------------------------------
        do k = 3, nk-2
            do j = 3, nj-2
                do i = 3, ni-2

                    xs2_i = (x(i-1,j,k,ip) + x(i+1,j,k,ip)) * 0.5e0
                    xs4_i = (-x(i-2,j,k,ip) + 4e0*x(i-1,j,k,ip) &
                            + 4e0*x(i+1,j,k,ip) - x(i+2,j,k,ip)) * (1e0/6e0)

                    xs2_j = (x(i,j-1,k,ip) + x(i,j+1,k,ip)) * 0.5e0
                    xs4_j = (-x(i,j-2,k,ip) + 4e0*x(i,j-1,k,ip) &
                            + 4e0*x(i,j+1,k,ip) - x(i,j+2,k,ip)) * (1e0/6e0)

                    xs2_k = (x(i,j,k-1,ip) + x(i,j,k+1,ip)) * 0.5e0
                    xs4_k = (-x(i,j,k-2,ip) + 4e0*x(i,j,k-1,ip) &
                            + 4e0*x(i,j,k+1,ip) - x(i,j,k+2,ip)) * (1e0/6e0)

                    xs(i,j,k) = (1e0 - sum_sf) * x(i,j,k,ip) &
                              + sf2 * (xs2_i + xs2_j + xs2_k) &
                              + sf4 * (xs4_i + xs4_j + xs4_k)
                end do
            end do
        end do

        ! ----------------------------------------------------------------
        ! Low k-slab: k=1,2; i=3..ni-2, j=3..nj-2
        ! d4k is the same for k=1 and k=2 (forward biased from k=1)
        ! ----------------------------------------------------------------
        do j = 3, nj-2
            do i = 3, ni-2
                xs2_i = (x(i-1,j,1,ip) + x(i+1,j,1,ip)) * 0.5e0
                xs4_i = (-x(i-2,j,1,ip) + 4e0*x(i-1,j,1,ip) &
                        + 4e0*x(i+1,j,1,ip) - x(i+2,j,1,ip)) * (1e0/6e0)
                xs2_j = (x(i,j-1,1,ip) + x(i,j+1,1,ip)) * 0.5e0
                xs4_j = (-x(i,j-2,1,ip) + 4e0*x(i,j-1,1,ip) &
                        + 4e0*x(i,j+1,1,ip) - x(i,j+2,1,ip)) * (1e0/6e0)
                xs2_k = 2e0*x(i,j,2,ip) - x(i,j,3,ip)
                d4k   = (x(i,j,1,ip) - 4e0*x(i,j,2,ip) + 6e0*x(i,j,3,ip) &
                       - 4e0*x(i,j,4,ip) + x(i,j,5,ip)) * (1e0/6e0)
                xs4_k = x(i,j,1,ip) - d4k
                xs(i,j,1) = (1e0 - sum_sf) * x(i,j,1,ip) &
                          + sf2 * (xs2_i + xs2_j + xs2_k) &
                          + sf4 * (xs4_i + xs4_j + xs4_k)

                ! k=2 reuses d4k and the same i,j stencils
                xs2_i = (x(i-1,j,2,ip) + x(i+1,j,2,ip)) * 0.5e0
                xs4_i = (-x(i-2,j,2,ip) + 4e0*x(i-1,j,2,ip) &
                        + 4e0*x(i+1,j,2,ip) - x(i+2,j,2,ip)) * (1e0/6e0)
                xs2_j = (x(i,j-1,2,ip) + x(i,j+1,2,ip)) * 0.5e0
                xs4_j = (-x(i,j-2,2,ip) + 4e0*x(i,j-1,2,ip) &
                        + 4e0*x(i,j+1,2,ip) - x(i,j+2,2,ip)) * (1e0/6e0)
                xs2_k = (x(i,j,1,ip) + x(i,j,3,ip)) * 0.5e0
                xs4_k = x(i,j,2,ip) - d4k
                xs(i,j,2) = (1e0 - sum_sf) * x(i,j,2,ip) &
                          + sf2 * (xs2_i + xs2_j + xs2_k) &
                          + sf4 * (xs4_i + xs4_j + xs4_k)
            end do
        end do

        ! ----------------------------------------------------------------
        ! High k-slab: k=nk-1,nk; i=3..ni-2, j=3..nj-2
        ! d4k backward biased from k=nk
        ! ----------------------------------------------------------------
        do j = 3, nj-2
            do i = 3, ni-2
                ! k=nk-1
                xs2_i = (x(i-1,j,nk-1,ip) + x(i+1,j,nk-1,ip)) * 0.5e0
                xs4_i = (-x(i-2,j,nk-1,ip) + 4e0*x(i-1,j,nk-1,ip) &
                        + 4e0*x(i+1,j,nk-1,ip) - x(i+2,j,nk-1,ip)) * (1e0/6e0)
                xs2_j = (x(i,j-1,nk-1,ip) + x(i,j+1,nk-1,ip)) * 0.5e0
                xs4_j = (-x(i,j-2,nk-1,ip) + 4e0*x(i,j-1,nk-1,ip) &
                        + 4e0*x(i,j+1,nk-1,ip) - x(i,j+2,nk-1,ip)) * (1e0/6e0)
                xs2_k = (x(i,j,nk-2,ip) + x(i,j,nk,ip)) * 0.5e0
                d4k   = (x(i,j,nk-4,ip) - 4e0*x(i,j,nk-3,ip) + 6e0*x(i,j,nk-2,ip) &
                       - 4e0*x(i,j,nk-1,ip) + x(i,j,nk,ip)) * (1e0/6e0)
                xs4_k = x(i,j,nk-1,ip) - d4k
                xs(i,j,nk-1) = (1e0 - sum_sf) * x(i,j,nk-1,ip) &
                             + sf2 * (xs2_i + xs2_j + xs2_k) &
                             + sf4 * (xs4_i + xs4_j + xs4_k)

                ! k=nk reuses d4k
                xs2_i = (x(i-1,j,nk,ip) + x(i+1,j,nk,ip)) * 0.5e0
                xs4_i = (-x(i-2,j,nk,ip) + 4e0*x(i-1,j,nk,ip) &
                        + 4e0*x(i+1,j,nk,ip) - x(i+2,j,nk,ip)) * (1e0/6e0)
                xs2_j = (x(i,j-1,nk,ip) + x(i,j+1,nk,ip)) * 0.5e0
                xs4_j = (-x(i,j-2,nk,ip) + 4e0*x(i,j-1,nk,ip) &
                        + 4e0*x(i,j+1,nk,ip) - x(i,j+2,nk,ip)) * (1e0/6e0)
                xs2_k = 2e0*x(i,j,nk-1,ip) - x(i,j,nk-2,ip)
                xs4_k = x(i,j,nk,ip) - d4k
                xs(i,j,nk) = (1e0 - sum_sf) * x(i,j,nk,ip) &
                           + sf2 * (xs2_i + xs2_j + xs2_k) &
                           + sf4 * (xs4_i + xs4_j + xs4_k)
            end do
        end do

        ! ----------------------------------------------------------------
        ! Low j-slab: j=1,2; i=3..ni-2; all k
        ! d4j forward biased from j=1, shared between j=1 and j=2
        ! k-stencil branches still needed here
        ! ----------------------------------------------------------------
        do k = 1, nk
            ! k-direction stencils (with boundary branches)
            do i = 3, ni-2
                xs2_i = (x(i-1,1,k,ip) + x(i+1,1,k,ip)) * 0.5e0
                xs4_i = (-x(i-2,1,k,ip) + 4e0*x(i-1,1,k,ip) &
                        + 4e0*x(i+1,1,k,ip) - x(i+2,1,k,ip)) * (1e0/6e0)
                xs2_j = 2e0*x(i,2,k,ip) - x(i,3,k,ip)
                d4j   = (x(i,1,k,ip) - 4e0*x(i,2,k,ip) + 6e0*x(i,3,k,ip) &
                       - 4e0*x(i,4,k,ip) + x(i,5,k,ip)) * (1e0/6e0)
                xs4_j = x(i,1,k,ip) - d4j
                if (k == 1) then
                    xs2_k = 2e0*x(i,1,2,ip) - x(i,1,3,ip)
                    xs4_k = x(i,1,1,ip) - (x(i,1,1,ip) - 4e0*x(i,1,2,ip) + 6e0*x(i,1,3,ip) &
                                         - 4e0*x(i,1,4,ip) + x(i,1,5,ip)) * (1e0/6e0)
                else if (k == nk) then
                    xs2_k = 2e0*x(i,1,nk-1,ip) - x(i,1,nk-2,ip)
                    xs4_k = x(i,1,nk,ip) - (x(i,1,nk-4,ip) - 4e0*x(i,1,nk-3,ip) + 6e0*x(i,1,nk-2,ip) &
                                           - 4e0*x(i,1,nk-1,ip) + x(i,1,nk,ip)) * (1e0/6e0)
                else if (k <= 2) then
                    xs2_k = (x(i,1,k-1,ip) + x(i,1,k+1,ip)) * 0.5e0
                    xs4_k = x(i,1,k,ip) - (x(i,1,1,ip) - 4e0*x(i,1,2,ip) + 6e0*x(i,1,3,ip) &
                                         - 4e0*x(i,1,4,ip) + x(i,1,5,ip)) * (1e0/6e0)
                else if (k >= nk-1) then
                    xs2_k = (x(i,1,k-1,ip) + x(i,1,k+1,ip)) * 0.5e0
                    xs4_k = x(i,1,k,ip) - (x(i,1,nk-4,ip) - 4e0*x(i,1,nk-3,ip) + 6e0*x(i,1,nk-2,ip) &
                                         - 4e0*x(i,1,nk-1,ip) + x(i,1,nk,ip)) * (1e0/6e0)
                else
                    xs2_k = (x(i,1,k-1,ip) + x(i,1,k+1,ip)) * 0.5e0
                    xs4_k = (-x(i,1,k-2,ip) + 4e0*x(i,1,k-1,ip) &
                            + 4e0*x(i,1,k+1,ip) - x(i,1,k+2,ip)) * (1e0/6e0)
                end if
                xs(i,1,k) = (1e0 - sum_sf) * x(i,1,k,ip) &
                          + sf2 * (xs2_i + xs2_j + xs2_k) &
                          + sf4 * (xs4_i + xs4_j + xs4_k)

                ! j=2 reuses d4j and i-stencils
                xs2_i = (x(i-1,2,k,ip) + x(i+1,2,k,ip)) * 0.5e0
                xs4_i = (-x(i-2,2,k,ip) + 4e0*x(i-1,2,k,ip) &
                        + 4e0*x(i+1,2,k,ip) - x(i+2,2,k,ip)) * (1e0/6e0)
                xs2_j = (x(i,1,k,ip) + x(i,3,k,ip)) * 0.5e0
                xs4_j = x(i,2,k,ip) - d4j
                if (k == 1) then
                    xs2_k = 2e0*x(i,2,2,ip) - x(i,2,3,ip)
                    xs4_k = x(i,2,1,ip) - (x(i,2,1,ip) - 4e0*x(i,2,2,ip) + 6e0*x(i,2,3,ip) &
                                         - 4e0*x(i,2,4,ip) + x(i,2,5,ip)) * (1e0/6e0)
                else if (k == nk) then
                    xs2_k = 2e0*x(i,2,nk-1,ip) - x(i,2,nk-2,ip)
                    xs4_k = x(i,2,nk,ip) - (x(i,2,nk-4,ip) - 4e0*x(i,2,nk-3,ip) + 6e0*x(i,2,nk-2,ip) &
                                           - 4e0*x(i,2,nk-1,ip) + x(i,2,nk,ip)) * (1e0/6e0)
                else if (k <= 2) then
                    xs2_k = (x(i,2,k-1,ip) + x(i,2,k+1,ip)) * 0.5e0
                    xs4_k = x(i,2,k,ip) - (x(i,2,1,ip) - 4e0*x(i,2,2,ip) + 6e0*x(i,2,3,ip) &
                                         - 4e0*x(i,2,4,ip) + x(i,2,5,ip)) * (1e0/6e0)
                else if (k >= nk-1) then
                    xs2_k = (x(i,2,k-1,ip) + x(i,2,k+1,ip)) * 0.5e0
                    xs4_k = x(i,2,k,ip) - (x(i,2,nk-4,ip) - 4e0*x(i,2,nk-3,ip) + 6e0*x(i,2,nk-2,ip) &
                                         - 4e0*x(i,2,nk-1,ip) + x(i,2,nk,ip)) * (1e0/6e0)
                else
                    xs2_k = (x(i,2,k-1,ip) + x(i,2,k+1,ip)) * 0.5e0
                    xs4_k = (-x(i,2,k-2,ip) + 4e0*x(i,2,k-1,ip) &
                            + 4e0*x(i,2,k+1,ip) - x(i,2,k+2,ip)) * (1e0/6e0)
                end if
                xs(i,2,k) = (1e0 - sum_sf) * x(i,2,k,ip) &
                          + sf2 * (xs2_i + xs2_j + xs2_k) &
                          + sf4 * (xs4_i + xs4_j + xs4_k)
            end do
        end do

        ! ----------------------------------------------------------------
        ! High j-slab: j=nj-1,nj; i=3..ni-2; all k
        ! d4j backward biased from j=nj
        ! ----------------------------------------------------------------
        do k = 1, nk
            do i = 3, ni-2
                ! j=nj-1
                xs2_i = (x(i-1,nj-1,k,ip) + x(i+1,nj-1,k,ip)) * 0.5e0
                xs4_i = (-x(i-2,nj-1,k,ip) + 4e0*x(i-1,nj-1,k,ip) &
                        + 4e0*x(i+1,nj-1,k,ip) - x(i+2,nj-1,k,ip)) * (1e0/6e0)
                xs2_j = (x(i,nj-2,k,ip) + x(i,nj,k,ip)) * 0.5e0
                d4j   = (x(i,nj-4,k,ip) - 4e0*x(i,nj-3,k,ip) + 6e0*x(i,nj-2,k,ip) &
                       - 4e0*x(i,nj-1,k,ip) + x(i,nj,k,ip)) * (1e0/6e0)
                xs4_j = x(i,nj-1,k,ip) - d4j
                if (k == 1) then
                    xs2_k = 2e0*x(i,nj-1,2,ip) - x(i,nj-1,3,ip)
                    xs4_k = x(i,nj-1,1,ip) - (x(i,nj-1,1,ip) - 4e0*x(i,nj-1,2,ip) + 6e0*x(i,nj-1,3,ip) &
                                             - 4e0*x(i,nj-1,4,ip) + x(i,nj-1,5,ip)) * (1e0/6e0)
                else if (k == nk) then
                    xs2_k = 2e0*x(i,nj-1,nk-1,ip) - x(i,nj-1,nk-2,ip)
                    xs4_k = x(i,nj-1,nk,ip) - (x(i,nj-1,nk-4,ip) - 4e0*x(i,nj-1,nk-3,ip) + 6e0*x(i,nj-1,nk-2,ip) &
                                              - 4e0*x(i,nj-1,nk-1,ip) + x(i,nj-1,nk,ip)) * (1e0/6e0)
                else if (k <= 2) then
                    xs2_k = (x(i,nj-1,k-1,ip) + x(i,nj-1,k+1,ip)) * 0.5e0
                    xs4_k = x(i,nj-1,k,ip) - (x(i,nj-1,1,ip) - 4e0*x(i,nj-1,2,ip) + 6e0*x(i,nj-1,3,ip) &
                                             - 4e0*x(i,nj-1,4,ip) + x(i,nj-1,5,ip)) * (1e0/6e0)
                else if (k >= nk-1) then
                    xs2_k = (x(i,nj-1,k-1,ip) + x(i,nj-1,k+1,ip)) * 0.5e0
                    xs4_k = x(i,nj-1,k,ip) - (x(i,nj-1,nk-4,ip) - 4e0*x(i,nj-1,nk-3,ip) + 6e0*x(i,nj-1,nk-2,ip) &
                                             - 4e0*x(i,nj-1,nk-1,ip) + x(i,nj-1,nk,ip)) * (1e0/6e0)
                else
                    xs2_k = (x(i,nj-1,k-1,ip) + x(i,nj-1,k+1,ip)) * 0.5e0
                    xs4_k = (-x(i,nj-1,k-2,ip) + 4e0*x(i,nj-1,k-1,ip) &
                            + 4e0*x(i,nj-1,k+1,ip) - x(i,nj-1,k+2,ip)) * (1e0/6e0)
                end if
                xs(i,nj-1,k) = (1e0 - sum_sf) * x(i,nj-1,k,ip) &
                             + sf2 * (xs2_i + xs2_j + xs2_k) &
                             + sf4 * (xs4_i + xs4_j + xs4_k)

                ! j=nj reuses d4j
                xs2_i = (x(i-1,nj,k,ip) + x(i+1,nj,k,ip)) * 0.5e0
                xs4_i = (-x(i-2,nj,k,ip) + 4e0*x(i-1,nj,k,ip) &
                        + 4e0*x(i+1,nj,k,ip) - x(i+2,nj,k,ip)) * (1e0/6e0)
                xs2_j = 2e0*x(i,nj-1,k,ip) - x(i,nj-2,k,ip)
                xs4_j = x(i,nj,k,ip) - d4j
                if (k == 1) then
                    xs2_k = 2e0*x(i,nj,2,ip) - x(i,nj,3,ip)
                    xs4_k = x(i,nj,1,ip) - (x(i,nj,1,ip) - 4e0*x(i,nj,2,ip) + 6e0*x(i,nj,3,ip) &
                                          - 4e0*x(i,nj,4,ip) + x(i,nj,5,ip)) * (1e0/6e0)
                else if (k == nk) then
                    xs2_k = 2e0*x(i,nj,nk-1,ip) - x(i,nj,nk-2,ip)
                    xs4_k = x(i,nj,nk,ip) - (x(i,nj,nk-4,ip) - 4e0*x(i,nj,nk-3,ip) + 6e0*x(i,nj,nk-2,ip) &
                                           - 4e0*x(i,nj,nk-1,ip) + x(i,nj,nk,ip)) * (1e0/6e0)
                else if (k <= 2) then
                    xs2_k = (x(i,nj,k-1,ip) + x(i,nj,k+1,ip)) * 0.5e0
                    xs4_k = x(i,nj,k,ip) - (x(i,nj,1,ip) - 4e0*x(i,nj,2,ip) + 6e0*x(i,nj,3,ip) &
                                          - 4e0*x(i,nj,4,ip) + x(i,nj,5,ip)) * (1e0/6e0)
                else if (k >= nk-1) then
                    xs2_k = (x(i,nj,k-1,ip) + x(i,nj,k+1,ip)) * 0.5e0
                    xs4_k = x(i,nj,k,ip) - (x(i,nj,nk-4,ip) - 4e0*x(i,nj,nk-3,ip) + 6e0*x(i,nj,nk-2,ip) &
                                          - 4e0*x(i,nj,nk-1,ip) + x(i,nj,nk,ip)) * (1e0/6e0)
                else
                    xs2_k = (x(i,nj,k-1,ip) + x(i,nj,k+1,ip)) * 0.5e0
                    xs4_k = (-x(i,nj,k-2,ip) + 4e0*x(i,nj,k-1,ip) &
                            + 4e0*x(i,nj,k+1,ip) - x(i,nj,k+2,ip)) * (1e0/6e0)
                end if
                xs(i,nj,k) = (1e0 - sum_sf) * x(i,nj,k,ip) &
                           + sf2 * (xs2_i + xs2_j + xs2_k) &
                           + sf4 * (xs4_i + xs4_j + xs4_k)
            end do
        end do

        ! ----------------------------------------------------------------
        ! Low i-slab: i=1,2; all j,k
        ! d4i forward biased from i=1, shared between i=1 and i=2
        ! j and k branches still needed
        ! ----------------------------------------------------------------
        do k = 1, nk
            do j = 1, nj
                xs2_i = 2e0*x(2,j,k,ip) - x(3,j,k,ip)
                d4i   = (x(1,j,k,ip) - 4e0*x(2,j,k,ip) + 6e0*x(3,j,k,ip) &
                       - 4e0*x(4,j,k,ip) + x(5,j,k,ip)) * (1e0/6e0)
                xs4_i = x(1,j,k,ip) - d4i

                if (j <= 2) then
                    xs2_j = 2e0*x(1,2,k,ip) - x(1,3,k,ip)
                    xs4_j = x(1,j,k,ip) - (x(1,1,k,ip) - 4e0*x(1,2,k,ip) + 6e0*x(1,3,k,ip) &
                                         - 4e0*x(1,4,k,ip) + x(1,5,k,ip)) * (1e0/6e0)
                    if (j == 2) xs2_j = (x(1,1,k,ip) + x(1,3,k,ip)) * 0.5e0
                else if (j >= nj-1) then
                    xs2_j = 2e0*x(1,nj-1,k,ip) - x(1,nj-2,k,ip)
                    xs4_j = x(1,j,k,ip) - (x(1,nj-4,k,ip) - 4e0*x(1,nj-3,k,ip) + 6e0*x(1,nj-2,k,ip) &
                                         - 4e0*x(1,nj-1,k,ip) + x(1,nj,k,ip)) * (1e0/6e0)
                    if (j == nj-1) xs2_j = (x(1,nj-2,k,ip) + x(1,nj,k,ip)) * 0.5e0
                else
                    xs2_j = (x(1,j-1,k,ip) + x(1,j+1,k,ip)) * 0.5e0
                    xs4_j = (-x(1,j-2,k,ip) + 4e0*x(1,j-1,k,ip) &
                            + 4e0*x(1,j+1,k,ip) - x(1,j+2,k,ip)) * (1e0/6e0)
                end if

                if (k <= 2) then
                    xs2_k = 2e0*x(1,j,2,ip) - x(1,j,3,ip)
                    xs4_k = x(1,j,k,ip) - (x(1,j,1,ip) - 4e0*x(1,j,2,ip) + 6e0*x(1,j,3,ip) &
                                         - 4e0*x(1,j,4,ip) + x(1,j,5,ip)) * (1e0/6e0)
                    if (k == 2) xs2_k = (x(1,j,1,ip) + x(1,j,3,ip)) * 0.5e0
                else if (k >= nk-1) then
                    xs2_k = 2e0*x(1,j,nk-1,ip) - x(1,j,nk-2,ip)
                    xs4_k = x(1,j,k,ip) - (x(1,j,nk-4,ip) - 4e0*x(1,j,nk-3,ip) + 6e0*x(1,j,nk-2,ip) &
                                         - 4e0*x(1,j,nk-1,ip) + x(1,j,nk,ip)) * (1e0/6e0)
                    if (k == nk-1) xs2_k = (x(1,j,nk-2,ip) + x(1,j,nk,ip)) * 0.5e0
                else
                    xs2_k = (x(1,j,k-1,ip) + x(1,j,k+1,ip)) * 0.5e0
                    xs4_k = (-x(1,j,k-2,ip) + 4e0*x(1,j,k-1,ip) &
                            + 4e0*x(1,j,k+1,ip) - x(1,j,k+2,ip)) * (1e0/6e0)
                end if

                xs(1,j,k) = (1e0 - sum_sf) * x(1,j,k,ip) &
                          + sf2 * (xs2_i + xs2_j + xs2_k) &
                          + sf4 * (xs4_i + xs4_j + xs4_k)

                ! i=2 reuses d4i
                xs2_i = (x(1,j,k,ip) + x(3,j,k,ip)) * 0.5e0
                xs4_i = x(2,j,k,ip) - d4i

                if (j <= 2) then
                    xs2_j = 2e0*x(2,2,k,ip) - x(2,3,k,ip)
                    xs4_j = x(2,j,k,ip) - (x(2,1,k,ip) - 4e0*x(2,2,k,ip) + 6e0*x(2,3,k,ip) &
                                         - 4e0*x(2,4,k,ip) + x(2,5,k,ip)) * (1e0/6e0)
                    if (j == 2) xs2_j = (x(2,1,k,ip) + x(2,3,k,ip)) * 0.5e0
                else if (j >= nj-1) then
                    xs2_j = 2e0*x(2,nj-1,k,ip) - x(2,nj-2,k,ip)
                    xs4_j = x(2,j,k,ip) - (x(2,nj-4,k,ip) - 4e0*x(2,nj-3,k,ip) + 6e0*x(2,nj-2,k,ip) &
                                         - 4e0*x(2,nj-1,k,ip) + x(2,nj,k,ip)) * (1e0/6e0)
                    if (j == nj-1) xs2_j = (x(2,nj-2,k,ip) + x(2,nj,k,ip)) * 0.5e0
                else
                    xs2_j = (x(2,j-1,k,ip) + x(2,j+1,k,ip)) * 0.5e0
                    xs4_j = (-x(2,j-2,k,ip) + 4e0*x(2,j-1,k,ip) &
                            + 4e0*x(2,j+1,k,ip) - x(2,j+2,k,ip)) * (1e0/6e0)
                end if

                if (k <= 2) then
                    xs2_k = 2e0*x(2,j,2,ip) - x(2,j,3,ip)
                    xs4_k = x(2,j,k,ip) - (x(2,j,1,ip) - 4e0*x(2,j,2,ip) + 6e0*x(2,j,3,ip) &
                                         - 4e0*x(2,j,4,ip) + x(2,j,5,ip)) * (1e0/6e0)
                    if (k == 2) xs2_k = (x(2,j,1,ip) + x(2,j,3,ip)) * 0.5e0
                else if (k >= nk-1) then
                    xs2_k = 2e0*x(2,j,nk-1,ip) - x(2,j,nk-2,ip)
                    xs4_k = x(2,j,k,ip) - (x(2,j,nk-4,ip) - 4e0*x(2,j,nk-3,ip) + 6e0*x(2,j,nk-2,ip) &
                                         - 4e0*x(2,j,nk-1,ip) + x(2,j,nk,ip)) * (1e0/6e0)
                    if (k == nk-1) xs2_k = (x(2,j,nk-2,ip) + x(2,j,nk,ip)) * 0.5e0
                else
                    xs2_k = (x(2,j,k-1,ip) + x(2,j,k+1,ip)) * 0.5e0
                    xs4_k = (-x(2,j,k-2,ip) + 4e0*x(2,j,k-1,ip) &
                            + 4e0*x(2,j,k+1,ip) - x(2,j,k+2,ip)) * (1e0/6e0)
                end if

                xs(2,j,k) = (1e0 - sum_sf) * x(2,j,k,ip) &
                          + sf2 * (xs2_i + xs2_j + xs2_k) &
                          + sf4 * (xs4_i + xs4_j + xs4_k)
            end do
        end do

        ! ----------------------------------------------------------------
        ! High i-slab: i=ni-1,ni; all j,k
        ! d4i backward biased from i=ni
        ! ----------------------------------------------------------------
        do k = 1, nk
            do j = 1, nj
                xs2_i = (x(ni-2,j,k,ip) + x(ni,j,k,ip)) * 0.5e0
                d4i   = (x(ni-4,j,k,ip) - 4e0*x(ni-3,j,k,ip) + 6e0*x(ni-2,j,k,ip) &
                       - 4e0*x(ni-1,j,k,ip) + x(ni,j,k,ip)) * (1e0/6e0)
                xs4_i = x(ni-1,j,k,ip) - d4i

                if (j <= 2) then
                    xs2_j = 2e0*x(ni-1,2,k,ip) - x(ni-1,3,k,ip)
                    xs4_j = x(ni-1,j,k,ip) - (x(ni-1,1,k,ip) - 4e0*x(ni-1,2,k,ip) + 6e0*x(ni-1,3,k,ip) &
                                            - 4e0*x(ni-1,4,k,ip) + x(ni-1,5,k,ip)) * (1e0/6e0)
                    if (j == 2) xs2_j = (x(ni-1,1,k,ip) + x(ni-1,3,k,ip)) * 0.5e0
                else if (j >= nj-1) then
                    xs2_j = 2e0*x(ni-1,nj-1,k,ip) - x(ni-1,nj-2,k,ip)
                    xs4_j = x(ni-1,j,k,ip) - (x(ni-1,nj-4,k,ip) - 4e0*x(ni-1,nj-3,k,ip) + 6e0*x(ni-1,nj-2,k,ip) &
                                            - 4e0*x(ni-1,nj-1,k,ip) + x(ni-1,nj,k,ip)) * (1e0/6e0)
                    if (j == nj-1) xs2_j = (x(ni-1,nj-2,k,ip) + x(ni-1,nj,k,ip)) * 0.5e0
                else
                    xs2_j = (x(ni-1,j-1,k,ip) + x(ni-1,j+1,k,ip)) * 0.5e0
                    xs4_j = (-x(ni-1,j-2,k,ip) + 4e0*x(ni-1,j-1,k,ip) &
                            + 4e0*x(ni-1,j+1,k,ip) - x(ni-1,j+2,k,ip)) * (1e0/6e0)
                end if

                if (k <= 2) then
                    xs2_k = 2e0*x(ni-1,j,2,ip) - x(ni-1,j,3,ip)
                    xs4_k = x(ni-1,j,k,ip) - (x(ni-1,j,1,ip) - 4e0*x(ni-1,j,2,ip) + 6e0*x(ni-1,j,3,ip) &
                                            - 4e0*x(ni-1,j,4,ip) + x(ni-1,j,5,ip)) * (1e0/6e0)
                    if (k == 2) xs2_k = (x(ni-1,j,1,ip) + x(ni-1,j,3,ip)) * 0.5e0
                else if (k >= nk-1) then
                    xs2_k = 2e0*x(ni-1,j,nk-1,ip) - x(ni-1,j,nk-2,ip)
                    xs4_k = x(ni-1,j,k,ip) - (x(ni-1,j,nk-4,ip) - 4e0*x(ni-1,j,nk-3,ip) + 6e0*x(ni-1,j,nk-2,ip) &
                                            - 4e0*x(ni-1,j,nk-1,ip) + x(ni-1,j,nk,ip)) * (1e0/6e0)
                    if (k == nk-1) xs2_k = (x(ni-1,j,nk-2,ip) + x(ni-1,j,nk,ip)) * 0.5e0
                else
                    xs2_k = (x(ni-1,j,k-1,ip) + x(ni-1,j,k+1,ip)) * 0.5e0
                    xs4_k = (-x(ni-1,j,k-2,ip) + 4e0*x(ni-1,j,k-1,ip) &
                            + 4e0*x(ni-1,j,k+1,ip) - x(ni-1,j,k+2,ip)) * (1e0/6e0)
                end if

                xs(ni-1,j,k) = (1e0 - sum_sf) * x(ni-1,j,k,ip) &
                             + sf2 * (xs2_i + xs2_j + xs2_k) &
                             + sf4 * (xs4_i + xs4_j + xs4_k)

                ! i=ni reuses d4i
                xs2_i = 2e0*x(ni-1,j,k,ip) - x(ni-2,j,k,ip)
                xs4_i = x(ni,j,k,ip) - d4i

                if (j <= 2) then
                    xs2_j = 2e0*x(ni,2,k,ip) - x(ni,3,k,ip)
                    xs4_j = x(ni,j,k,ip) - (x(ni,1,k,ip) - 4e0*x(ni,2,k,ip) + 6e0*x(ni,3,k,ip) &
                                          - 4e0*x(ni,4,k,ip) + x(ni,5,k,ip)) * (1e0/6e0)
                    if (j == 2) xs2_j = (x(ni,1,k,ip) + x(ni,3,k,ip)) * 0.5e0
                else if (j >= nj-1) then
                    xs2_j = 2e0*x(ni,nj-1,k,ip) - x(ni,nj-2,k,ip)
                    xs4_j = x(ni,j,k,ip) - (x(ni,nj-4,k,ip) - 4e0*x(ni,nj-3,k,ip) + 6e0*x(ni,nj-2,k,ip) &
                                          - 4e0*x(ni,nj-1,k,ip) + x(ni,nj,k,ip)) * (1e0/6e0)
                    if (j == nj-1) xs2_j = (x(ni,nj-2,k,ip) + x(ni,nj,k,ip)) * 0.5e0
                else
                    xs2_j = (x(ni,j-1,k,ip) + x(ni,j+1,k,ip)) * 0.5e0
                    xs4_j = (-x(ni,j-2,k,ip) + 4e0*x(ni,j-1,k,ip) &
                            + 4e0*x(ni,j+1,k,ip) - x(ni,j+2,k,ip)) * (1e0/6e0)
                end if

                if (k <= 2) then
                    xs2_k = 2e0*x(ni,j,2,ip) - x(ni,j,3,ip)
                    xs4_k = x(ni,j,k,ip) - (x(ni,j,1,ip) - 4e0*x(ni,j,2,ip) + 6e0*x(ni,j,3,ip) &
                                          - 4e0*x(ni,j,4,ip) + x(ni,j,5,ip)) * (1e0/6e0)
                    if (k == 2) xs2_k = (x(ni,j,1,ip) + x(ni,j,3,ip)) * 0.5e0
                else if (k >= nk-1) then
                    xs2_k = 2e0*x(ni,j,nk-1,ip) - x(ni,j,nk-2,ip)
                    xs4_k = x(ni,j,k,ip) - (x(ni,j,nk-4,ip) - 4e0*x(ni,j,nk-3,ip) + 6e0*x(ni,j,nk-2,ip) &
                                          - 4e0*x(ni,j,nk-1,ip) + x(ni,j,nk,ip)) * (1e0/6e0)
                    if (k == nk-1) xs2_k = (x(ni,j,nk-2,ip) + x(ni,j,nk,ip)) * 0.5e0
                else
                    xs2_k = (x(ni,j,k-1,ip) + x(ni,j,k+1,ip)) * 0.5e0
                    xs4_k = (-x(ni,j,k-2,ip) + 4e0*x(ni,j,k-1,ip) &
                            + 4e0*x(ni,j,k+1,ip) - x(ni,j,k+2,ip)) * (1e0/6e0)
                end if

                xs(ni,j,k) = (1e0 - sum_sf) * x(ni,j,k,ip) &
                           + sf2 * (xs2_i + xs2_j + xs2_k) &
                           + sf4 * (xs4_i + xs4_j + xs4_k)
            end do
        end do

        x(:,:,:,ip) = xs

    end do

end subroutine smooth3d_const
