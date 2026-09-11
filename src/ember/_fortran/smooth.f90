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

! ======================================================================
! ======================================================================

! Adaptive blended 2nd/4th-order smoothing with a JST shock sensor.
!
! Same stencil family and the same boundary closures as smooth3d_const, but
! the 2nd-order factor is a per-node, per-direction shock sensor instead of a
! constant. There is no L (length-scale) weighting and no CFL scaling: the
! operator is a pure function of x and the sensor, exactly as smooth3d_const
! is a pure function of x.
!
! Sensor: elementwise max of the JST normalised-curvature sensor evaluated on
! pressure and on temperature. The temperature term catches contact
! discontinuities (constant P, jump in T) that the pressure-only sensor misses
! (Swanson, Radespiel & Turkel, AIAA-97-1945).
!
! Blend (Jameson, Schmidt & Turkel, AIAA-81-1259):
!
!   sf2n = max(sf2P * nu_P, sf2T * nu_T)
!   sf4n = max(sf4 - sf2n, 0)
!
! so the sensor subtracts from the background 4th-order factor and clips at
! zero: inside a shock the 4th-order term switches off entirely, and the total
! damping per direction is max(sf2n, sf4) rather than a sum.
!
! Sensor is anisotropic: each direction uses its own normalised-curvature
! sensor, with no per-node max coupling across directions. The isotropic max
! (a feature sensed in any direction activating 2nd-order dissipation in all
! three) was removed because at a cusp trailing edge the physical cross-seam
! loading jump tripped the streamwise switch, clipping the 4th-order
! streamwise term and leaving a streamwise 2 dx mode undamped.
!
! Per direction, with s2 = sf2n and s4 = sf4n, the 1-D contribution is:
!
!   m = 1        -(s2 + s4/6) * (x_1 - 2 x_2 + x_3)
!   m = 2         (s2/2) * (x_1 - 2 x_2 + x_3)
!                   - (s4/6) * (-2 x_1 + 5 x_2 - 4 x_3 + x_4)
!   3..n-2        (s2/2) * (x_-1 - 2 x_0 + x_+1)
!                   - (s4/6) * (x_-2 - 4 x_-1 + 6 x_0 - 4 x_+1 + x_+2)
!   m = n-1       (s2/2) * (x_n-2 - 2 x_n-1 + x_n)
!                   - (s4/6) * (-2 x_n + 5 x_n-1 - 4 x_n-2 + x_n-3)
!   m = n        -(s2 + s4/6) * (x_n - 2 x_n-1 + x_n-2)
!
! Every row has zero sum, so constants are preserved, and every row is a
! second difference or a fourth difference of the *unshifted* stencil, so
! linear fields are preserved at every node including the faces. The interior
! rows annihilate cubics; the four boundary rows do not (they are the S^T S
! rows of smooth3d_const, which degrade to 2nd-order dissipation near a face).
!
! CAVEAT vs smooth3d_const: the non-amplification guarantee of smooth3d_const
! rests on M = S^T S with *constant* coefficients, which makes M symmetric
! positive semi-definite. Here s2 and s4 vary per node and per direction, so M
! is no longer symmetric and that proof does not carry over. Linear exactness
! is a per-row property and does survive; non-amplification is not claimed.
!
! Because L and CFL are gone the operator is exactly separable, so it is
! assembled as three independent passes over the *original* field accumulated
! into dx (pass I assigns, passes J and K add), then applied as x = x + dx.
!
! Working arrays (caller pre-allocates):
!   sf2n(ni,nj,nk,3)   - nodal 2nd-order factors (P and T limbs max'd in place)
!   dx(ni,nj,nk)       - accumulated delta per component
!
! The shared helper used by smooth3d_adaptive, below:
!   accumulate_curvature_sensor -- weighted Jameson sensor on a scalar at nodes,
!                                  max-accumulated over the 3 directions
!
! Requires ni,nj,nk >= 5.
!
subroutine smooth3d_adaptive( &
        x, P, T, &
        sf4, sf2P, sf2T, &
        sf2n, dx, &
        ni, nj, nk, np &
    )

    integer, intent(in)    :: ni, nj, nk, np
    real,    intent(in)    :: sf4, sf2P, sf2T
    real,    intent(inout) :: x(ni, nj, nk, np)
    real,    intent(in)    :: P(ni, nj, nk)
    real,    intent(in)    :: T(ni, nj, nk)
    real,    intent(inout) :: sf2n(ni, nj, nk, 3)
    real,    intent(inout) :: dx(ni, nj, nk)

    real, parameter :: c16 = 1e0 / 6e0

    integer :: i, j, k, ip
    real :: s2, s4

    if (ni < 5 .or. nj < 5 .or. nk < 5) then
        stop 'smooth3d_adaptive: ni,nj,nk must be >= 5'
    end if

    ! ================================================================
    ! Step A: combined P/T sensor and blend
    ! ================================================================

    sf2n = 0e0
    call accumulate_curvature_sensor(P, sf2P, sf2n, ni, nj, nk)
    call accumulate_curvature_sensor(T, sf2T, sf2n, ni, nj, nk)

    ! ================================================================
    ! Step B: three separable passes into dx, then apply
    ! ================================================================

    do ip = 1, np

        ! ------------------------------------------------------------
        ! PASS I (assigns dx). The i interior vectorises; i=1,2,ni-1,ni
        ! are four branch-free columns.
        ! ------------------------------------------------------------
        do k = 1, nk
            do j = 1, nj
                do i = 3, ni-2
                    s2 = sf2n(i,j,k,1)
                    s4 = max(sf4 - s2, 0e0)
                    dx(i,j,k) = &
                        s2 * 0.5e0 * (x(i-1,j,k,ip) - 2e0*x(i,j,k,ip) + x(i+1,j,k,ip)) &
                        - s4 * c16 * (x(i-2,j,k,ip) - 4e0*x(i-1,j,k,ip) &
                            + 6e0*x(i,j,k,ip) - 4e0*x(i+1,j,k,ip) + x(i+2,j,k,ip))
                end do

                s2 = sf2n(1,j,k,1)
                s4 = max(sf4 - s2, 0e0)
                dx(1,j,k) = -(s2 + s4 * c16) &
                    * (x(1,j,k,ip) - 2e0*x(2,j,k,ip) + x(3,j,k,ip))

                s2 = sf2n(2,j,k,1)
                s4 = max(sf4 - s2, 0e0)
                dx(2,j,k) = &
                    s2 * 0.5e0 * (x(1,j,k,ip) - 2e0*x(2,j,k,ip) + x(3,j,k,ip)) &
                    - s4 * c16 * (-2e0*x(1,j,k,ip) + 5e0*x(2,j,k,ip) &
                        - 4e0*x(3,j,k,ip) + x(4,j,k,ip))

                s2 = sf2n(ni-1,j,k,1)
                s4 = max(sf4 - s2, 0e0)
                dx(ni-1,j,k) = &
                    s2 * 0.5e0 * (x(ni-2,j,k,ip) - 2e0*x(ni-1,j,k,ip) + x(ni,j,k,ip)) &
                    - s4 * c16 * (-2e0*x(ni,j,k,ip) + 5e0*x(ni-1,j,k,ip) &
                        - 4e0*x(ni-2,j,k,ip) + x(ni-3,j,k,ip))

                s2 = sf2n(ni,j,k,1)
                s4 = max(sf4 - s2, 0e0)
                dx(ni,j,k) = -(s2 + s4 * c16) &
                    * (x(ni,j,k,ip) - 2e0*x(ni-1,j,k,ip) + x(ni-2,j,k,ip))
            end do
        end do

        ! ------------------------------------------------------------
        ! PASS J (adds). The j-class branch is outside do i, so every
        ! body vectorises over i.
        ! ------------------------------------------------------------
        do k = 1, nk
            do i = 1, ni
                s2 = sf2n(i,1,k,2)
                s4 = max(sf4 - s2, 0e0)
                dx(i,1,k) = dx(i,1,k) - (s2 + s4 * c16) &
                    * (x(i,1,k,ip) - 2e0*x(i,2,k,ip) + x(i,3,k,ip))
            end do
            do i = 1, ni
                s2 = sf2n(i,2,k,2)
                s4 = max(sf4 - s2, 0e0)
                dx(i,2,k) = dx(i,2,k) &
                    + s2 * 0.5e0 * (x(i,1,k,ip) - 2e0*x(i,2,k,ip) + x(i,3,k,ip)) &
                    - s4 * c16 * (-2e0*x(i,1,k,ip) + 5e0*x(i,2,k,ip) &
                        - 4e0*x(i,3,k,ip) + x(i,4,k,ip))
            end do
            do j = 3, nj-2
                do i = 1, ni
                    s2 = sf2n(i,j,k,2)
                    s4 = max(sf4 - s2, 0e0)
                    dx(i,j,k) = dx(i,j,k) &
                        + s2 * 0.5e0 * (x(i,j-1,k,ip) - 2e0*x(i,j,k,ip) + x(i,j+1,k,ip)) &
                        - s4 * c16 * (x(i,j-2,k,ip) - 4e0*x(i,j-1,k,ip) &
                            + 6e0*x(i,j,k,ip) - 4e0*x(i,j+1,k,ip) + x(i,j+2,k,ip))
                end do
            end do
            do i = 1, ni
                s2 = sf2n(i,nj-1,k,2)
                s4 = max(sf4 - s2, 0e0)
                dx(i,nj-1,k) = dx(i,nj-1,k) &
                    + s2 * 0.5e0 * (x(i,nj-2,k,ip) - 2e0*x(i,nj-1,k,ip) + x(i,nj,k,ip)) &
                    - s4 * c16 * (-2e0*x(i,nj,k,ip) + 5e0*x(i,nj-1,k,ip) &
                        - 4e0*x(i,nj-2,k,ip) + x(i,nj-3,k,ip))
            end do
            do i = 1, ni
                s2 = sf2n(i,nj,k,2)
                s4 = max(sf4 - s2, 0e0)
                dx(i,nj,k) = dx(i,nj,k) - (s2 + s4 * c16) &
                    * (x(i,nj,k,ip) - 2e0*x(i,nj-1,k,ip) + x(i,nj-2,k,ip))
            end do
        end do

        ! ------------------------------------------------------------
        ! PASS K (adds). The k-class branch is outside do j, do i.
        ! ------------------------------------------------------------
        do j = 1, nj
            do i = 1, ni
                s2 = sf2n(i,j,1,3)
                s4 = max(sf4 - s2, 0e0)
                dx(i,j,1) = dx(i,j,1) - (s2 + s4 * c16) &
                    * (x(i,j,1,ip) - 2e0*x(i,j,2,ip) + x(i,j,3,ip))
            end do
        end do
        do j = 1, nj
            do i = 1, ni
                s2 = sf2n(i,j,2,3)
                s4 = max(sf4 - s2, 0e0)
                dx(i,j,2) = dx(i,j,2) &
                    + s2 * 0.5e0 * (x(i,j,1,ip) - 2e0*x(i,j,2,ip) + x(i,j,3,ip)) &
                    - s4 * c16 * (-2e0*x(i,j,1,ip) + 5e0*x(i,j,2,ip) &
                        - 4e0*x(i,j,3,ip) + x(i,j,4,ip))
            end do
        end do
        do k = 3, nk-2
            do j = 1, nj
                do i = 1, ni
                    s2 = sf2n(i,j,k,3)
                    s4 = max(sf4 - s2, 0e0)
                    dx(i,j,k) = dx(i,j,k) &
                        + s2 * 0.5e0 * (x(i,j,k-1,ip) - 2e0*x(i,j,k,ip) + x(i,j,k+1,ip)) &
                        - s4 * c16 * (x(i,j,k-2,ip) - 4e0*x(i,j,k-1,ip) &
                            + 6e0*x(i,j,k,ip) - 4e0*x(i,j,k+1,ip) + x(i,j,k+2,ip))
                end do
            end do
        end do
        do j = 1, nj
            do i = 1, ni
                s2 = sf2n(i,j,nk-1,3)
                s4 = max(sf4 - s2, 0e0)
                dx(i,j,nk-1) = dx(i,j,nk-1) &
                    + s2 * 0.5e0 * (x(i,j,nk-2,ip) - 2e0*x(i,j,nk-1,ip) + x(i,j,nk,ip)) &
                    - s4 * c16 * (-2e0*x(i,j,nk,ip) + 5e0*x(i,j,nk-1,ip) &
                        - 4e0*x(i,j,nk-2,ip) + x(i,j,nk-3,ip))
            end do
        end do
        do j = 1, nj
            do i = 1, ni
                s2 = sf2n(i,j,nk,3)
                s4 = max(sf4 - s2, 0e0)
                dx(i,j,nk) = dx(i,j,nk) - (s2 + s4 * c16) &
                    * (x(i,j,nk,ip) - 2e0*x(i,j,nk-1,ip) + x(i,j,nk-2,ip))
            end do
        end do

        x(:,:,:,ip) = x(:,:,:,ip) + dx(:,:,:)

    end do

end subroutine smooth3d_adaptive


! ----------------------------------------------------------------------
! Weighted nodal normalised-curvature sensor on a scalar field Q, accumulated
! into sf2n as an elementwise max:
!
!   sf2n(:,:,:,d) = max(sf2n(:,:,:,d), w * |Q_-1 - 2 Q_0 + Q_+1|
!                                        / (Q_-1 + 2 Q_0 + Q_+1))
!
! along d. At domain boundaries the stencil is shifted (uses 1,2,3 or
! n-2,n-1,n). Q must be strictly positive (pressure, temperature, density).
!
! Accumulating rather than assigning lets the P and T limbs of the blend share
! one array: zero sf2n, then call once per scalar. The raw sensor is >= 0, so
! for the non-negative weights this is called with, seeding from zero gives the
! same result as an assign-then-max on two separate arrays.
subroutine accumulate_curvature_sensor(Q, w, sf2n, ni, nj, nk)
    integer, intent(in)    :: ni, nj, nk
    real,    intent(in)    :: w
    real,    intent(in)    :: Q(ni, nj, nk)
    real,    intent(inout) :: sf2n(ni, nj, nk, 3)

    sf2n(2:ni-1, :, :, 1) = max(sf2n(2:ni-1, :, :, 1), w * &
        abs(Q(1:ni-2, :, :) - 2e0*Q(2:ni-1, :, :) + Q(3:ni, :, :)) &
        /  (Q(1:ni-2, :, :) + 2e0*Q(2:ni-1, :, :) + Q(3:ni, :, :)))
    sf2n(1, :, :, 1) = max(sf2n(1, :, :, 1), w * &
        abs(Q(1, :, :) - 2e0*Q(2, :, :) + Q(3, :, :)) &
        /  (Q(1, :, :) + 2e0*Q(2, :, :) + Q(3, :, :)))
    sf2n(ni, :, :, 1) = max(sf2n(ni, :, :, 1), w * &
        abs(Q(ni, :, :) - 2e0*Q(ni-1, :, :) + Q(ni-2, :, :)) &
        /  (Q(ni, :, :) + 2e0*Q(ni-1, :, :) + Q(ni-2, :, :)))

    sf2n(:, 2:nj-1, :, 2) = max(sf2n(:, 2:nj-1, :, 2), w * &
        abs(Q(:, 1:nj-2, :) - 2e0*Q(:, 2:nj-1, :) + Q(:, 3:nj, :)) &
        /  (Q(:, 1:nj-2, :) + 2e0*Q(:, 2:nj-1, :) + Q(:, 3:nj, :)))
    sf2n(:, 1, :, 2) = max(sf2n(:, 1, :, 2), w * &
        abs(Q(:, 1, :) - 2e0*Q(:, 2, :) + Q(:, 3, :)) &
        /  (Q(:, 1, :) + 2e0*Q(:, 2, :) + Q(:, 3, :)))
    sf2n(:, nj, :, 2) = max(sf2n(:, nj, :, 2), w * &
        abs(Q(:, nj, :) - 2e0*Q(:, nj-1, :) + Q(:, nj-2, :)) &
        /  (Q(:, nj, :) + 2e0*Q(:, nj-1, :) + Q(:, nj-2, :)))

    sf2n(:, :, 2:nk-1, 3) = max(sf2n(:, :, 2:nk-1, 3), w * &
        abs(Q(:, :, 1:nk-2) - 2e0*Q(:, :, 2:nk-1) + Q(:, :, 3:nk)) &
        /  (Q(:, :, 1:nk-2) + 2e0*Q(:, :, 2:nk-1) + Q(:, :, 3:nk)))
    sf2n(:, :, 1, 3) = max(sf2n(:, :, 1, 3), w * &
        abs(Q(:, :, 1) - 2e0*Q(:, :, 2) + Q(:, :, 3)) &
        /  (Q(:, :, 1) + 2e0*Q(:, :, 2) + Q(:, :, 3)))
    sf2n(:, :, nk, 3) = max(sf2n(:, :, nk, 3), w * &
        abs(Q(:, :, nk) - 2e0*Q(:, :, nk-1) + Q(:, :, nk-2)) &
        /  (Q(:, :, nk) + 2e0*Q(:, :, nk-1) + Q(:, :, nk-2)))
end subroutine accumulate_curvature_sensor
