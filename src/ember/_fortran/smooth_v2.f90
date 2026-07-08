! Optimised smooth3d with combined 2nd/4th-order stencil and pre-averaged CFL
!
! Two optimisations over smooth3d:
!   1. Sensor computed WITHOUT L scaling or boundary clipping.
!      L is factored out and applied once per direction when assembling dx.
!   2. CFL is averaged from cells to nodes ONCE (for all 5 components)
!      before the property loop, rather than recomputed per-component.
!
! Sensor: elementwise max of the JST normalised-curvature sensor evaluated
! on pressure and on temperature. The temperature term catches contact
! discontinuities (constant P, jump in T) that the pressure-only sensor
! misses (Swanson, Radespiel & Turkel, AIAA-97-1945).
!
! Working arrays (caller pre-allocates):
!   sf2n(ni,nj,nk,3)   - nodal 2nd-order factors (no L)
!   sf2n_T(ni,nj,nk,3) - scratch for the temperature-based sensor
!   cfln(ni,nj,nk,5)   - node-averaged CFL
!   dx(ni,nj,nk)       - accumulated delta per component
!
! This file also hosts the shared helpers used by smooth3d:
!   nodal_curvature_sensor -- raw Jameson sensor on a scalar at nodes (3 directions)
!   cfl_cell_to_node       -- 1/8,1/4,1/2,1 weighted average of CFL
!
subroutine smooth3d( &
        x, P, T, L, cfl, &
        sf4, sf2P, sf2T, sf2min, &
        sf2n, sf2n_T, cfln, dx, &
        ni, nj, nk, np &
    )

    integer, intent(in)    :: ni, nj, nk, np
    real,    intent(in)    :: sf4, sf2P, sf2T, sf2min
    real,    intent(inout) :: x(ni, nj, nk, np)
    real,    intent(in)    :: P(ni, nj, nk)
    real,    intent(in)    :: T(ni, nj, nk)
    real,    intent(in)    :: L(ni, nj, nk, 3)
    real,    intent(in)    :: cfl(ni-1, nj-1, nk-1, 5)
    real,    intent(inout) :: sf2n(ni, nj, nk, 3)
    real,    intent(inout) :: sf2n_T(ni, nj, nk, 3)
    real,    intent(inout) :: cfln(ni, nj, nk, 5)
    real,    intent(inout) :: dx(ni, nj, nk)

    integer :: i, j, k, ip
    real :: sf24_d, sf44_d, sft_d, sfb_d
    real :: dxl, tgt

    ! ================================================================
    ! Step A: Combined P/T sensor and sf2n (no L, no boundary clipping)
    ! ================================================================

    call nodal_curvature_sensor(P, sf2n,   ni, nj, nk)
    call nodal_curvature_sensor(T, sf2n_T, ni, nj, nk)
    sf2n = max(sf2P * sf2n, sf2T * sf2n_T)
    sf2n = max(sf2n, sf2min)

    ! Anisotropic sensor: each direction uses its own normalised-curvature
    ! sensor, with no per-node max coupling across directions. The isotropic
    ! max (a feature sensed in any direction activating 2nd-order dissipation
    ! in all three) was removed because at a cusp trailing edge the physical
    ! cross-seam loading jump tripped the streamwise switch, clipping the
    ! 4th-order streamwise term and leaving a streamwise 2 dx mode undamped.

    ! ================================================================
    ! Step B: Pre-average CFL from cells to nodes (all 5 components)
    ! ================================================================

    call cfl_cell_to_node(cfl, cfln, ni, nj, nk)


    ! ================================================================
    ! Step C: Component loop with combined stencil
    ! ================================================================

    do ip = 1, np

        ! ----- Interior (i=3..ni-2, j=3..nj-2, k=3..nk-2) -----
        do k = 3, nk-2
            do j = 3, nj-2
                do i = 3, ni-2
                    ! i-direction (4th-order)
                    call combined_coefs(sf2n(i,j,k,1), sf4, sft_d, sf24_d, sf44_d)
                    dxl = L(i,j,k,1) * ( &
                        sft_d * x(i,j,k,ip) &
                        + sf24_d * (x(i-1,j,k,ip) + x(i+1,j,k,ip)) &
                        + sf44_d * (x(i-2,j,k,ip) + x(i+2,j,k,ip)))

                    ! j-direction (4th-order)
                    call combined_coefs(sf2n(i,j,k,2), sf4, sft_d, sf24_d, sf44_d)
                    dxl = dxl + L(i,j,k,2) * ( &
                        sft_d * x(i,j,k,ip) &
                        + sf24_d * (x(i,j-1,k,ip) + x(i,j+1,k,ip)) &
                        + sf44_d * (x(i,j-2,k,ip) + x(i,j+2,k,ip)))

                    ! k-direction (4th-order)
                    call combined_coefs(sf2n(i,j,k,3), sf4, sft_d, sf24_d, sf44_d)
                    dxl = dxl + L(i,j,k,3) * ( &
                        sft_d * x(i,j,k,ip) &
                        + sf24_d * (x(i,j,k-1,ip) + x(i,j,k+1,ip)) &
                        + sf44_d * (x(i,j,k-2,ip) + x(i,j,k+2,ip)))

                    dx(i,j,k) = dxl * cfln(i,j,k,ip)
                end do
            end do
        end do

        ! ----- LOW i-BOUNDARY (i=1,2, all j,k) -----
        do k = 1, nk
            do j = 1, nj
                do i = 1, 2
                    ! i-direction: 2nd-order only
                    sfb_d = sf2n(i,j,k,1)
                    if (i == 1) then
                        tgt = x(2,j,k,ip)
                    else
                        tgt = (x(1,j,k,ip) + x(3,j,k,ip)) * 0.5e0
                    end if
                    dxl = L(i,j,k,1) * sfb_d * (tgt - x(i,j,k,ip))

                    ! j-direction
                    sfb_d = sf2n(i,j,k,2)
                    if (j == 1) then
                        tgt = x(i,2,k,ip)
                        dxl = dxl + L(i,j,k,2) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (j == nj) then
                        tgt = x(i,nj-1,k,ip)
                        dxl = dxl + L(i,j,k,2) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (j >= 3 .and. j <= nj-2) then
                        call combined_coefs(sf2n(i,j,k,2), sf4, sft_d, sf24_d, sf44_d)
                        dxl = dxl + L(i,j,k,2) * ( &
                            sft_d * x(i,j,k,ip) &
                            + sf24_d * (x(i,j-1,k,ip) + x(i,j+1,k,ip)) &
                            + sf44_d * (x(i,j-2,k,ip) + x(i,j+2,k,ip)))
                    else
                        tgt = (x(i,j-1,k,ip) + x(i,j+1,k,ip)) * 0.5e0
                        dxl = dxl + L(i,j,k,2) * sfb_d * (tgt - x(i,j,k,ip))
                    end if

                    ! k-direction
                    sfb_d = sf2n(i,j,k,3)
                    if (k == 1) then
                        tgt = x(i,j,2,ip)
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (k == nk) then
                        tgt = x(i,j,nk-1,ip)
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (k >= 3 .and. k <= nk-2) then
                        call combined_coefs(sf2n(i,j,k,3), sf4, sft_d, sf24_d, sf44_d)
                        dxl = dxl + L(i,j,k,3) * ( &
                            sft_d * x(i,j,k,ip) &
                            + sf24_d * (x(i,j,k-1,ip) + x(i,j,k+1,ip)) &
                            + sf44_d * (x(i,j,k-2,ip) + x(i,j,k+2,ip)))
                    else
                        tgt = (x(i,j,k-1,ip) + x(i,j,k+1,ip)) * 0.5e0
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    end if

                    dx(i,j,k) = dxl * cfln(i,j,k,ip)
                end do
            end do
        end do

        ! ----- HIGH i-BOUNDARY (i=ni-1,ni, all j,k) -----
        do k = 1, nk
            do j = 1, nj
                do i = ni-1, ni
                    ! i-direction: 2nd-order only
                    sfb_d = sf2n(i,j,k,1)
                    if (i == ni) then
                        tgt = x(ni-1,j,k,ip)
                    else
                        tgt = (x(ni-2,j,k,ip) + x(ni,j,k,ip)) * 0.5e0
                    end if
                    dxl = L(i,j,k,1) * sfb_d * (tgt - x(i,j,k,ip))

                    ! j-direction
                    sfb_d = sf2n(i,j,k,2)
                    if (j == 1) then
                        tgt = x(i,2,k,ip)
                        dxl = dxl + L(i,j,k,2) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (j == nj) then
                        tgt = x(i,nj-1,k,ip)
                        dxl = dxl + L(i,j,k,2) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (j >= 3 .and. j <= nj-2) then
                        call combined_coefs(sf2n(i,j,k,2), sf4, sft_d, sf24_d, sf44_d)
                        dxl = dxl + L(i,j,k,2) * ( &
                            sft_d * x(i,j,k,ip) &
                            + sf24_d * (x(i,j-1,k,ip) + x(i,j+1,k,ip)) &
                            + sf44_d * (x(i,j-2,k,ip) + x(i,j+2,k,ip)))
                    else
                        tgt = (x(i,j-1,k,ip) + x(i,j+1,k,ip)) * 0.5e0
                        dxl = dxl + L(i,j,k,2) * sfb_d * (tgt - x(i,j,k,ip))
                    end if

                    ! k-direction
                    sfb_d = sf2n(i,j,k,3)
                    if (k == 1) then
                        tgt = x(i,j,2,ip)
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (k == nk) then
                        tgt = x(i,j,nk-1,ip)
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (k >= 3 .and. k <= nk-2) then
                        call combined_coefs(sf2n(i,j,k,3), sf4, sft_d, sf24_d, sf44_d)
                        dxl = dxl + L(i,j,k,3) * ( &
                            sft_d * x(i,j,k,ip) &
                            + sf24_d * (x(i,j,k-1,ip) + x(i,j,k+1,ip)) &
                            + sf44_d * (x(i,j,k-2,ip) + x(i,j,k+2,ip)))
                    else
                        tgt = (x(i,j,k-1,ip) + x(i,j,k+1,ip)) * 0.5e0
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    end if

                    dx(i,j,k) = dxl * cfln(i,j,k,ip)
                end do
            end do
        end do

        ! ----- LOW j-BOUNDARY (j=1,2, interior i, all k) -----
        do k = 1, nk
            do j = 1, 2
                do i = 3, ni-2
                    ! i-direction (4th-order)
                    call combined_coefs(sf2n(i,j,k,1), sf4, sft_d, sf24_d, sf44_d)
                    dxl = L(i,j,k,1) * ( &
                        sft_d * x(i,j,k,ip) &
                        + sf24_d * (x(i-1,j,k,ip) + x(i+1,j,k,ip)) &
                        + sf44_d * (x(i-2,j,k,ip) + x(i+2,j,k,ip)))

                    ! j-direction: 2nd-order only
                    sfb_d = sf2n(i,j,k,2)
                    if (j == 1) then
                        tgt = x(i,2,k,ip)
                    else
                        tgt = (x(i,1,k,ip) + x(i,3,k,ip)) * 0.5e0
                    end if
                    dxl = dxl + L(i,j,k,2) * sfb_d * (tgt - x(i,j,k,ip))

                    ! k-direction
                    sfb_d = sf2n(i,j,k,3)
                    if (k == 1) then
                        tgt = x(i,j,2,ip)
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (k == nk) then
                        tgt = x(i,j,nk-1,ip)
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (k >= 3 .and. k <= nk-2) then
                        call combined_coefs(sf2n(i,j,k,3), sf4, sft_d, sf24_d, sf44_d)
                        dxl = dxl + L(i,j,k,3) * ( &
                            sft_d * x(i,j,k,ip) &
                            + sf24_d * (x(i,j,k-1,ip) + x(i,j,k+1,ip)) &
                            + sf44_d * (x(i,j,k-2,ip) + x(i,j,k+2,ip)))
                    else
                        tgt = (x(i,j,k-1,ip) + x(i,j,k+1,ip)) * 0.5e0
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    end if

                    dx(i,j,k) = dxl * cfln(i,j,k,ip)
                end do
            end do
        end do

        ! ----- HIGH j-BOUNDARY (j=nj-1,nj, interior i, all k) -----
        do k = 1, nk
            do j = nj-1, nj
                do i = 3, ni-2
                    ! i-direction (4th-order)
                    call combined_coefs(sf2n(i,j,k,1), sf4, sft_d, sf24_d, sf44_d)
                    dxl = L(i,j,k,1) * ( &
                        sft_d * x(i,j,k,ip) &
                        + sf24_d * (x(i-1,j,k,ip) + x(i+1,j,k,ip)) &
                        + sf44_d * (x(i-2,j,k,ip) + x(i+2,j,k,ip)))

                    ! j-direction: 2nd-order only
                    sfb_d = sf2n(i,j,k,2)
                    if (j == nj) then
                        tgt = x(i,nj-1,k,ip)
                    else
                        tgt = (x(i,nj-2,k,ip) + x(i,nj,k,ip)) * 0.5e0
                    end if
                    dxl = dxl + L(i,j,k,2) * sfb_d * (tgt - x(i,j,k,ip))

                    ! k-direction
                    sfb_d = sf2n(i,j,k,3)
                    if (k == 1) then
                        tgt = x(i,j,2,ip)
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (k == nk) then
                        tgt = x(i,j,nk-1,ip)
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    else if (k >= 3 .and. k <= nk-2) then
                        call combined_coefs(sf2n(i,j,k,3), sf4, sft_d, sf24_d, sf44_d)
                        dxl = dxl + L(i,j,k,3) * ( &
                            sft_d * x(i,j,k,ip) &
                            + sf24_d * (x(i,j,k-1,ip) + x(i,j,k+1,ip)) &
                            + sf44_d * (x(i,j,k-2,ip) + x(i,j,k+2,ip)))
                    else
                        tgt = (x(i,j,k-1,ip) + x(i,j,k+1,ip)) * 0.5e0
                        dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))
                    end if

                    dx(i,j,k) = dxl * cfln(i,j,k,ip)
                end do
            end do
        end do

        ! ----- LOW k-BOUNDARY (k=1,2, interior i,j) -----
        do k = 1, 2
            do j = 3, nj-2
                do i = 3, ni-2
                    ! i-direction (4th-order)
                    call combined_coefs(sf2n(i,j,k,1), sf4, sft_d, sf24_d, sf44_d)
                    dxl = L(i,j,k,1) * ( &
                        sft_d * x(i,j,k,ip) &
                        + sf24_d * (x(i-1,j,k,ip) + x(i+1,j,k,ip)) &
                        + sf44_d * (x(i-2,j,k,ip) + x(i+2,j,k,ip)))

                    ! j-direction (4th-order)
                    call combined_coefs(sf2n(i,j,k,2), sf4, sft_d, sf24_d, sf44_d)
                    dxl = dxl + L(i,j,k,2) * ( &
                        sft_d * x(i,j,k,ip) &
                        + sf24_d * (x(i,j-1,k,ip) + x(i,j+1,k,ip)) &
                        + sf44_d * (x(i,j-2,k,ip) + x(i,j+2,k,ip)))

                    ! k-direction: 2nd-order only
                    sfb_d = sf2n(i,j,k,3)
                    if (k == 1) then
                        tgt = x(i,j,2,ip)
                    else
                        tgt = (x(i,j,1,ip) + x(i,j,3,ip)) * 0.5e0
                    end if
                    dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))

                    dx(i,j,k) = dxl * cfln(i,j,k,ip)
                end do
            end do
        end do

        ! ----- HIGH k-BOUNDARY (k=nk-1,nk, interior i,j) -----
        do k = nk-1, nk
            do j = 3, nj-2
                do i = 3, ni-2
                    ! i-direction (4th-order)
                    call combined_coefs(sf2n(i,j,k,1), sf4, sft_d, sf24_d, sf44_d)
                    dxl = L(i,j,k,1) * ( &
                        sft_d * x(i,j,k,ip) &
                        + sf24_d * (x(i-1,j,k,ip) + x(i+1,j,k,ip)) &
                        + sf44_d * (x(i-2,j,k,ip) + x(i+2,j,k,ip)))

                    ! j-direction (4th-order)
                    call combined_coefs(sf2n(i,j,k,2), sf4, sft_d, sf24_d, sf44_d)
                    dxl = dxl + L(i,j,k,2) * ( &
                        sft_d * x(i,j,k,ip) &
                        + sf24_d * (x(i,j-1,k,ip) + x(i,j+1,k,ip)) &
                        + sf44_d * (x(i,j-2,k,ip) + x(i,j+2,k,ip)))

                    ! k-direction: 2nd-order only
                    sfb_d = sf2n(i,j,k,3)
                    if (k == nk) then
                        tgt = x(i,j,nk-1,ip)
                    else
                        tgt = (x(i,j,nk-2,ip) + x(i,j,nk,ip)) * 0.5e0
                    end if
                    dxl = dxl + L(i,j,k,3) * sfb_d * (tgt - x(i,j,k,ip))

                    dx(i,j,k) = dxl * cfln(i,j,k,ip)
                end do
            end do
        end do

        ! Apply accumulated delta
        x(:,:,:,ip) = x(:,:,:,ip) + dx(:,:,:)

    end do

contains

    ! Calculate combined 2nd and 4th smoothing stencil
    pure subroutine combined_coefs(sf2n_d, sf4_in, sft, sf24, sf44)
        real, intent(in)  :: sf2n_d, sf4_in
        real, intent(out) :: sft, sf24, sf44
        real :: sf4n
        sf4n = max(sf4_in - sf2n_d, 0e0)
        sft  = -(sf2n_d + sf4n)
        sf24 = sf2n_d * 0.5e0 + sf4n * (2.0e0 / 3.0e0)
        sf44 = -sf4n * (1.0e0 / 6.0e0)
    end subroutine

end subroutine smooth3d


! ======================================================================
! H-mesh variant of smooth3d for blocks that are periodic to self in the
! k direction (k=nk coincident with k=1) over a leading-edge interval
! (i = 1 .. i_le) and a trailing-edge interval (i = i_te .. ni). Over those
! two intervals the k-boundary smoothing stencil and the curvature sensor
! wrap around the k=1/k=nk seam so the smoothing is continuous across the
! block boundary; in the bladed region (i_le+1 .. i_te-1) the original
! clipped k-boundary treatment is retained. i_le / i_te are 1-based
! inclusive; pass 0 for an absent interval (i_te==0 disables the downstream
! wrap, i_le==0 the upstream). With both 0 this is identical to smooth3d.
! ======================================================================
subroutine smooth3d_hmesh( &
        x, P, T, L, cfl, &
        sf4, sf2P, sf2T, sf2min, &
        sf2n, sf2n_T, cfln, dx, &
        ni, nj, nk, np, i_le, i_te &
    )

    integer, intent(in)    :: ni, nj, nk, np, i_le, i_te
    real,    intent(in)    :: sf4, sf2P, sf2T, sf2min
    real,    intent(inout) :: x(ni, nj, nk, np)
    real,    intent(in)    :: P(ni, nj, nk)
    real,    intent(in)    :: T(ni, nj, nk)
    real,    intent(in)    :: L(ni, nj, nk, 3)
    real,    intent(in)    :: cfl(ni-1, nj-1, nk-1, 5)
    real,    intent(inout) :: sf2n(ni, nj, nk, 3)
    real,    intent(inout) :: sf2n_T(ni, nj, nk, 3)
    real,    intent(inout) :: cfln(ni, nj, nk, 5)
    real,    intent(inout) :: dx(ni, nj, nk)

    integer :: i, j, k, ip
    integer :: ia, ic
    real :: dxl

    ! ================================================================
    ! Step A: combined P/T sensor, then wrap the k=1/k=nk sensor over
    ! the periodic intervals before scaling/clipping (as in smooth3d).
    ! ================================================================

    call nodal_curvature_sensor(P, sf2n,   ni, nj, nk)
    call nodal_curvature_sensor(T, sf2n_T, ni, nj, nk)
    call fix_kperk_sensor(P, sf2n,   ni, nj, nk, i_le, i_te)
    call fix_kperk_sensor(T, sf2n_T, ni, nj, nk, i_le, i_te)
    sf2n = max(sf2P * sf2n, sf2T * sf2n_T)
    sf2n = max(sf2n, sf2min)

    ! Anisotropic sensor (see smooth3d): each direction keeps its own
    ! normalised-curvature sensor with no per-node max coupling. The isotropic
    ! max was removed because at the cusp k=1/k=nk seam the wrapped k-sensor
    ! carries the physical cross-seam loading jump; coupling it into the
    ! i-direction switch clipped the 4th-order streamwise dissipation along
    ! the seam and left a streamwise 2 dx mode undamped.

    ! ================================================================
    ! Step B: pre-average CFL from cells to nodes (all 5 components)
    ! ================================================================

    call cfl_cell_to_node(cfl, cfln, ni, nj, nk)

    ! i-segment bounds within the interior i range (3 .. ni-2):
    !   upstream wrapped : 3 .. ia
    !   bladed  clipped  : ia+1 .. ic-1
    !   downstream wrapped : ic .. ni-2
    ! Clamped so the three segments tile 3..ni-2 exactly and absent
    ! intervals collapse to empty do-loops.
    if (i_le > 0) then
        ia = min(i_le, ni-2)
    else
        ia = 2
    end if
    if (i_te > 0) then
        ic = max(i_te, ia+1, 3)
    else
        ic = ni-1
    end if

    ! ================================================================
    ! Step C: component loop
    ! ================================================================

    do ip = 1, np

        ! ----- Interior (i=3..ni-2, j=3..nj-2, k=3..nk-2) -----
        do k = 3, nk-2
            do j = 3, nj-2
                do i = 3, ni-2
                    dxl = L(i,j,k,1) * id4(i,j,k,ip) &
                        + L(i,j,k,2) * jd4(i,j,k,ip) &
                        + L(i,j,k,3) * kdelta_clipped(i,j,k,ip)
                    dx(i,j,k) = dxl * cfln(i,j,k,ip)
                end do
            end do
        end do

        ! ----- LOW/HIGH i-BOUNDARY (i=1,2,ni-1,ni; all j,k) -----
        ! perk is constant down each i column, so it is hoisted out.
        call do_ibnd(ip, 1)
        call do_ibnd(ip, 2)
        call do_ibnd(ip, ni-1)
        call do_ibnd(ip, ni)

        ! ----- LOW/HIGH j-BOUNDARY (j=1,2,nj-1,nj; interior i; all k) -----
        call do_jbnd(ip, 3,    ia,   1, 2,      .true.)
        call do_jbnd(ip, ia+1, ic-1, 1, 2,      .false.)
        call do_jbnd(ip, ic,   ni-2, 1, 2,      .true.)
        call do_jbnd(ip, 3,    ia,   nj-1, nj,  .true.)
        call do_jbnd(ip, ia+1, ic-1, nj-1, nj,  .false.)
        call do_jbnd(ip, ic,   ni-2, nj-1, nj,  .true.)

        ! ----- LOW/HIGH k-BOUNDARY (k=1,2,nk-1,nk; interior i,j) -----
        call do_kbnd(ip, 3,    ia,   1, 2,       .true.)
        call do_kbnd(ip, ia+1, ic-1, 1, 2,       .false.)
        call do_kbnd(ip, ic,   ni-2, 1, 2,       .true.)
        call do_kbnd(ip, 3,    ia,   nk-1, nk,   .true.)
        call do_kbnd(ip, ia+1, ic-1, nk-1, nk,   .false.)
        call do_kbnd(ip, ic,   ni-2, nk-1, nk,   .true.)

        ! Apply accumulated delta
        x(:,:,:,ip) = x(:,:,:,ip) + dx(:,:,:)

    end do

contains

    ! --- k-boundary slab: interior i,j; 4th-order i and j; wrapped or
    !     clipped k. ilo..ihi is one i-segment, klo..khi is {1,2}/{nk-1,nk}.
    subroutine do_kbnd(ipp, ilo, ihi, klo, khi, wrap)
        integer, intent(in) :: ipp, ilo, ihi, klo, khi
        logical, intent(in) :: wrap
        integer :: ii, jj, kk
        real :: d
        do kk = klo, khi
            do jj = 3, nj-2
                do ii = ilo, ihi
                    d = L(ii,jj,kk,1) * id4(ii,jj,kk,ipp) &
                      + L(ii,jj,kk,2) * jd4(ii,jj,kk,ipp)
                    if (wrap) then
                        d = d + L(ii,jj,kk,3) * kdelta_wrapped(ii,jj,kk,ipp)
                    else
                        d = d + L(ii,jj,kk,3) * kdelta_clipped(ii,jj,kk,ipp)
                    end if
                    dx(ii,jj,kk) = d * cfln(ii,jj,kk,ipp)
                end do
            end do
        end do
    end subroutine

    ! --- j-boundary slab: interior i; 4th-order i; 2nd-order j; all k. ---
    subroutine do_jbnd(ipp, ilo, ihi, jlo, jhi, wrap)
        integer, intent(in) :: ipp, ilo, ihi, jlo, jhi
        logical, intent(in) :: wrap
        integer :: ii, jj, kk
        real :: d
        do jj = jlo, jhi
            do ii = ilo, ihi
                do kk = 1, nk
                    d = L(ii,jj,kk,1) * id4(ii,jj,kk,ipp) &
                      + L(ii,jj,kk,2) * jdb(ii,jj,kk,ipp)
                    if (wrap) then
                        d = d + L(ii,jj,kk,3) * kdelta_wrapped(ii,jj,kk,ipp)
                    else
                        d = d + L(ii,jj,kk,3) * kdelta_clipped(ii,jj,kk,ipp)
                    end if
                    dx(ii,jj,kk) = d * cfln(ii,jj,kk,ipp)
                end do
            end do
        end do
    end subroutine

    ! --- i-boundary column (i=ival): 2nd-order i; full j; all k. The
    !     periodic flag is constant down the column and resolved here. ---
    subroutine do_ibnd(ipp, ival)
        integer, intent(in) :: ipp, ival
        integer :: jj, kk
        logical :: wrap
        real :: d
        wrap = (ival <= i_le) .or. (i_te > 0 .and. ival >= i_te)
        do jj = 1, nj
            do kk = 1, nk
                d = L(ival,jj,kk,1) * idb(ival,jj,kk,ipp) &
                  + L(ival,jj,kk,2) * jdfull(ival,jj,kk,ipp)
                if (wrap) then
                    d = d + L(ival,jj,kk,3) * kdelta_wrapped(ival,jj,kk,ipp)
                else
                    d = d + L(ival,jj,kk,3) * kdelta_clipped(ival,jj,kk,ipp)
                end if
                dx(ival,jj,kk) = d * cfln(ival,jj,kk,ipp)
            end do
        end do
    end subroutine

    ! --- 4th-order i contribution (interior i), pre-L. ---
    pure function id4(i, j, k, ip) result(dk)
        integer, intent(in) :: i, j, k, ip
        real :: dk, sft_d, sf24_d, sf44_d
        call combined_coefs(sf2n(i,j,k,1), sf4, sft_d, sf24_d, sf44_d)
        dk = sft_d * x(i,j,k,ip) &
           + sf24_d * (x(i-1,j,k,ip) + x(i+1,j,k,ip)) &
           + sf44_d * (x(i-2,j,k,ip) + x(i+2,j,k,ip))
    end function

    ! --- 4th-order j contribution (interior j), pre-L. ---
    pure function jd4(i, j, k, ip) result(dk)
        integer, intent(in) :: i, j, k, ip
        real :: dk, sft_d, sf24_d, sf44_d
        call combined_coefs(sf2n(i,j,k,2), sf4, sft_d, sf24_d, sf44_d)
        dk = sft_d * x(i,j,k,ip) &
           + sf24_d * (x(i,j-1,k,ip) + x(i,j+1,k,ip)) &
           + sf44_d * (x(i,j-2,k,ip) + x(i,j+2,k,ip))
    end function

    ! --- 2nd-order i contribution at the i boundary (i=1,2,ni-1,ni). ---
    pure function idb(i, j, k, ip) result(dk)
        integer, intent(in) :: i, j, k, ip
        real :: dk, tgt
        if (i == 1) then
            tgt = x(2,j,k,ip)
        else if (i == ni) then
            tgt = x(ni-1,j,k,ip)
        else
            tgt = (x(i-1,j,k,ip) + x(i+1,j,k,ip)) * 0.5e0
        end if
        dk = sf2n(i,j,k,1) * (tgt - x(i,j,k,ip))
    end function

    ! --- 2nd-order j contribution at the j boundary (j=1,2,nj-1,nj). ---
    pure function jdb(i, j, k, ip) result(dk)
        integer, intent(in) :: i, j, k, ip
        real :: dk, tgt
        if (j == 1) then
            tgt = x(i,2,k,ip)
        else if (j == nj) then
            tgt = x(i,nj-1,k,ip)
        else
            tgt = (x(i,j-1,k,ip) + x(i,j+1,k,ip)) * 0.5e0
        end if
        dk = sf2n(i,j,k,2) * (tgt - x(i,j,k,ip))
    end function

    ! --- j contribution for the full j range (used in i-boundary column):
    !     2nd-order at j=1,2,nj-1,nj, 4th-order in the j interior. ---
    pure function jdfull(i, j, k, ip) result(dk)
        integer, intent(in) :: i, j, k, ip
        real :: dk, sft_d, sf24_d, sf44_d, tgt
        if (j == 1) then
            dk = sf2n(i,j,k,2) * (x(i,2,k,ip) - x(i,j,k,ip))
        else if (j == nj) then
            dk = sf2n(i,j,k,2) * (x(i,nj-1,k,ip) - x(i,j,k,ip))
        else if (j >= 3 .and. j <= nj-2) then
            call combined_coefs(sf2n(i,j,k,2), sf4, sft_d, sf24_d, sf44_d)
            dk = sft_d * x(i,j,k,ip) &
               + sf24_d * (x(i,j-1,k,ip) + x(i,j+1,k,ip)) &
               + sf44_d * (x(i,j-2,k,ip) + x(i,j+2,k,ip))
        else
            tgt = (x(i,j-1,k,ip) + x(i,j+1,k,ip)) * 0.5e0
            dk = sf2n(i,j,k,2) * (tgt - x(i,j,k,ip))
        end if
    end function

    ! --- k contribution, original clipped boundary treatment, pre-L. ---
    pure function kdelta_clipped(i, j, k, ip) result(dk)
        integer, intent(in) :: i, j, k, ip
        real :: dk, sft_d, sf24_d, sf44_d
        if (k == 1) then
            dk = sf2n(i,j,k,3) * (x(i,j,2,ip) - x(i,j,k,ip))
        else if (k == nk) then
            dk = sf2n(i,j,k,3) * (x(i,j,nk-1,ip) - x(i,j,k,ip))
        else if (k >= 3 .and. k <= nk-2) then
            call combined_coefs(sf2n(i,j,k,3), sf4, sft_d, sf24_d, sf44_d)
            dk = sft_d * x(i,j,k,ip) &
               + sf24_d * (x(i,j,k-1,ip) + x(i,j,k+1,ip)) &
               + sf44_d * (x(i,j,k-2,ip) + x(i,j,k+2,ip))
        else
            dk = sf2n(i,j,k,3) * ((x(i,j,k-1,ip) + x(i,j,k+1,ip)) * 0.5e0 - x(i,j,k,ip))
        end if
    end function

    ! --- k contribution, wrapped across the k=1/k=nk seam, pre-L. Full
    !     4th-order at every k; k=nk is the duplicate of k=1, so the four
    !     boundary planes take explicit wrapped neighbour indices. ---
    pure function kdelta_wrapped(i, j, k, ip) result(dk)
        integer, intent(in) :: i, j, k, ip
        real :: dk, sft_d, sf24_d, sf44_d
        integer :: km1, km2, kp1, kp2
        if (k == 1) then
            km1 = nk-1; km2 = nk-2; kp1 = 2;    kp2 = 3
        else if (k == 2) then
            km1 = 1;    km2 = nk-1; kp1 = 3;    kp2 = 4
        else if (k == nk-1) then
            km1 = nk-2; km2 = nk-3; kp1 = nk;   kp2 = 2
        else if (k == nk) then
            km1 = nk-1; km2 = nk-2; kp1 = 2;    kp2 = 3
        else
            km1 = k-1;  km2 = k-2;  kp1 = k+1;  kp2 = k+2
        end if
        call combined_coefs(sf2n(i,j,k,3), sf4, sft_d, sf24_d, sf44_d)
        dk = sft_d * x(i,j,k,ip) &
           + sf24_d * (x(i,j,km1,ip) + x(i,j,kp1,ip)) &
           + sf44_d * (x(i,j,km2,ip) + x(i,j,kp2,ip))
    end function

    ! Calculate combined 2nd and 4th smoothing stencil (copy of the helper
    ! in smooth3d; contained here for host association of this routine).
    pure subroutine combined_coefs(sf2n_d, sf4_in, sft, sf24, sf44)
        real, intent(in)  :: sf2n_d, sf4_in
        real, intent(out) :: sft, sf24, sf44
        real :: sf4n
        sf4n = max(sf4_in - sf2n_d, 0e0)
        sft  = -(sf2n_d + sf4n)
        sf24 = sf2n_d * 0.5e0 + sf4n * (2.0e0 / 3.0e0)
        sf44 = -sf4n * (1.0e0 / 6.0e0)
    end subroutine

end subroutine smooth3d_hmesh


! ----------------------------------------------------------------------
! Overwrite the k-direction (dir 3) curvature sensor at k=1 and k=nk with
! a wrapped centred stencil over the k-periodic i intervals (i<=i_le or
! i>=i_te), so the sensor has no seam at the k=1/k=nk boundary. k=nk is
! the duplicate of k=1, so both planes use neighbours (nk-1, 2). Only the
! one-sided k=1/k=nk entries need fixing; k=2..nk-1 are already centred.
subroutine fix_kperk_sensor(Q, sf2n, ni, nj, nk, i_le, i_te)
    integer, intent(in)    :: ni, nj, nk, i_le, i_te
    real,    intent(in)    :: Q(ni, nj, nk)
    real,    intent(inout) :: sf2n(ni, nj, nk, 3)

    integer :: i, j
    logical :: perk

    do j = 1, nj
        do i = 1, ni
            perk = (i <= i_le) .or. (i_te > 0 .and. i >= i_te)
            if (.not. perk) cycle
            sf2n(i,j,1,3) = &
                abs(Q(i,j,nk-1) - 2e0*Q(i,j,1) + Q(i,j,2)) &
                /  (Q(i,j,nk-1) + 2e0*Q(i,j,1) + Q(i,j,2))
            sf2n(i,j,nk,3) = &
                abs(Q(i,j,nk-1) - 2e0*Q(i,j,nk) + Q(i,j,2)) &
                /  (Q(i,j,nk-1) + 2e0*Q(i,j,nk) + Q(i,j,2))
        end do
    end do
end subroutine fix_kperk_sensor


! ----------------------------------------------------------------------
! Raw nodal normalised-curvature sensor on a scalar field Q (no sf2/sf2min
! scaling). sf2n(:,:,:,d) = |Q_-1 - 2 Q_0 + Q_+1| / (Q_-1 + 2 Q_0 + Q_+1)
! along d. At domain boundaries the stencil is shifted (uses 1,2,3 or
! n-2,n-1,n). Q must be strictly positive (pressure, temperature, density).
subroutine nodal_curvature_sensor(Q, sf2n, ni, nj, nk)
    integer, intent(in)    :: ni, nj, nk
    real,    intent(in)    :: Q(ni, nj, nk)
    real,    intent(inout) :: sf2n(ni, nj, nk, 3)

    sf2n(2:ni-1, :, :, 1) = &
        abs(Q(1:ni-2, :, :) - 2e0*Q(2:ni-1, :, :) + Q(3:ni, :, :)) &
        /  (Q(1:ni-2, :, :) + 2e0*Q(2:ni-1, :, :) + Q(3:ni, :, :))
    sf2n(1, :, :, 1) = &
        abs(Q(1, :, :) - 2e0*Q(2, :, :) + Q(3, :, :)) &
        /  (Q(1, :, :) + 2e0*Q(2, :, :) + Q(3, :, :))
    sf2n(ni, :, :, 1) = &
        abs(Q(ni, :, :) - 2e0*Q(ni-1, :, :) + Q(ni-2, :, :)) &
        /  (Q(ni, :, :) + 2e0*Q(ni-1, :, :) + Q(ni-2, :, :))

    sf2n(:, 2:nj-1, :, 2) = &
        abs(Q(:, 1:nj-2, :) - 2e0*Q(:, 2:nj-1, :) + Q(:, 3:nj, :)) &
        /  (Q(:, 1:nj-2, :) + 2e0*Q(:, 2:nj-1, :) + Q(:, 3:nj, :))
    sf2n(:, 1, :, 2) = &
        abs(Q(:, 1, :) - 2e0*Q(:, 2, :) + Q(:, 3, :)) &
        /  (Q(:, 1, :) + 2e0*Q(:, 2, :) + Q(:, 3, :))
    sf2n(:, nj, :, 2) = &
        abs(Q(:, nj, :) - 2e0*Q(:, nj-1, :) + Q(:, nj-2, :)) &
        /  (Q(:, nj, :) + 2e0*Q(:, nj-1, :) + Q(:, nj-2, :))

    sf2n(:, :, 2:nk-1, 3) = &
        abs(Q(:, :, 1:nk-2) - 2e0*Q(:, :, 2:nk-1) + Q(:, :, 3:nk)) &
        /  (Q(:, :, 1:nk-2) + 2e0*Q(:, :, 2:nk-1) + Q(:, :, 3:nk))
    sf2n(:, :, 1, 3) = &
        abs(Q(:, :, 1) - 2e0*Q(:, :, 2) + Q(:, :, 3)) &
        /  (Q(:, :, 1) + 2e0*Q(:, :, 2) + Q(:, :, 3))
    sf2n(:, :, nk, 3) = &
        abs(Q(:, :, nk) - 2e0*Q(:, :, nk-1) + Q(:, :, nk-2)) &
        /  (Q(:, :, nk) + 2e0*Q(:, :, nk-1) + Q(:, :, nk-2))
end subroutine nodal_curvature_sensor


! ----------------------------------------------------------------------
! Average cell-centred CFL (ni-1,nj-1,nk-1,5) onto nodes (ni,nj,nk,5).
! Weights: 1/8 interior, 1/4 face, 1/2 edge, 1/1 corner.
subroutine cfl_cell_to_node(cfl, cfln, ni, nj, nk)
    integer, intent(in)    :: ni, nj, nk
    real,    intent(in)    :: cfl(ni-1, nj-1, nk-1, 5)
    real,    intent(inout) :: cfln(ni, nj, nk, 5)

    integer :: i, j, k, ic

    do k = 2, nk-1
        do j = 2, nj-1
            do i = 2, ni-1
                do ic = 1, 5
                    cfln(i,j,k,ic) = ( &
                        cfl(i-1,j-1,k-1,ic) + cfl(i,j-1,k-1,ic) + &
                        cfl(i-1,j,k-1,ic)   + cfl(i,j,k-1,ic)   + &
                        cfl(i-1,j-1,k,ic)   + cfl(i,j-1,k,ic)   + &
                        cfl(i-1,j,k,ic)     + cfl(i,j,k,ic)       &
                    ) * 0.125e0
                end do
            end do
        end do
    end do

    do k = 2, nk-1
        do j = 2, nj-1
            do ic = 1, 5
                cfln(1,j,k,ic) = ( &
                    cfl(1,j-1,k-1,ic) + cfl(1,j,k-1,ic) + &
                    cfl(1,j-1,k,ic)   + cfl(1,j,k,ic)     &
                ) * 0.25e0
                cfln(ni,j,k,ic) = ( &
                    cfl(ni-1,j-1,k-1,ic) + cfl(ni-1,j,k-1,ic) + &
                    cfl(ni-1,j-1,k,ic)   + cfl(ni-1,j,k,ic)     &
                ) * 0.25e0
            end do
        end do
    end do
    do k = 2, nk-1
        do i = 2, ni-1
            do ic = 1, 5
                cfln(i,1,k,ic) = ( &
                    cfl(i-1,1,k-1,ic) + cfl(i,1,k-1,ic) + &
                    cfl(i-1,1,k,ic)   + cfl(i,1,k,ic)     &
                ) * 0.25e0
                cfln(i,nj,k,ic) = ( &
                    cfl(i-1,nj-1,k-1,ic) + cfl(i,nj-1,k-1,ic) + &
                    cfl(i-1,nj-1,k,ic)   + cfl(i,nj-1,k,ic)     &
                ) * 0.25e0
            end do
        end do
    end do
    do j = 2, nj-1
        do i = 2, ni-1
            do ic = 1, 5
                cfln(i,j,1,ic) = ( &
                    cfl(i-1,j-1,1,ic) + cfl(i,j-1,1,ic) + &
                    cfl(i-1,j,1,ic)   + cfl(i,j,1,ic)     &
                ) * 0.25e0
                cfln(i,j,nk,ic) = ( &
                    cfl(i-1,j-1,nk-1,ic) + cfl(i,j-1,nk-1,ic) + &
                    cfl(i-1,j,nk-1,ic)   + cfl(i,j,nk-1,ic)     &
                ) * 0.25e0
            end do
        end do
    end do

    do i = 2, ni-1
        do ic = 1, 5
            cfln(i,1,1,ic)   = (cfl(i-1,1,1,ic)         + cfl(i,1,1,ic))         * 0.5e0
            cfln(i,nj,1,ic)  = (cfl(i-1,nj-1,1,ic)      + cfl(i,nj-1,1,ic))      * 0.5e0
            cfln(i,1,nk,ic)  = (cfl(i-1,1,nk-1,ic)      + cfl(i,1,nk-1,ic))      * 0.5e0
            cfln(i,nj,nk,ic) = (cfl(i-1,nj-1,nk-1,ic)   + cfl(i,nj-1,nk-1,ic))   * 0.5e0
        end do
    end do
    do j = 2, nj-1
        do ic = 1, 5
            cfln(1,j,1,ic)   = (cfl(1,j-1,1,ic)         + cfl(1,j,1,ic))         * 0.5e0
            cfln(ni,j,1,ic)  = (cfl(ni-1,j-1,1,ic)      + cfl(ni-1,j,1,ic))      * 0.5e0
            cfln(1,j,nk,ic)  = (cfl(1,j-1,nk-1,ic)      + cfl(1,j,nk-1,ic))      * 0.5e0
            cfln(ni,j,nk,ic) = (cfl(ni-1,j-1,nk-1,ic)   + cfl(ni-1,j,nk-1,ic))   * 0.5e0
        end do
    end do
    do k = 2, nk-1
        do ic = 1, 5
            cfln(1,1,k,ic)   = (cfl(1,1,k-1,ic)         + cfl(1,1,k,ic))         * 0.5e0
            cfln(ni,1,k,ic)  = (cfl(ni-1,1,k-1,ic)      + cfl(ni-1,1,k,ic))      * 0.5e0
            cfln(1,nj,k,ic)  = (cfl(1,nj-1,k-1,ic)      + cfl(1,nj-1,k,ic))      * 0.5e0
            cfln(ni,nj,k,ic) = (cfl(ni-1,nj-1,k-1,ic)   + cfl(ni-1,nj-1,k,ic))   * 0.5e0
        end do
    end do

    do ic = 1, 5
        cfln(1,1,1,ic)     = cfl(1,1,1,ic)
        cfln(ni,1,1,ic)    = cfl(ni-1,1,1,ic)
        cfln(1,nj,1,ic)    = cfl(1,nj-1,1,ic)
        cfln(ni,nj,1,ic)   = cfl(ni-1,nj-1,1,ic)
        cfln(1,1,nk,ic)    = cfl(1,1,nk-1,ic)
        cfln(ni,1,nk,ic)   = cfl(ni-1,1,nk-1,ic)
        cfln(1,nj,nk,ic)   = cfl(1,nj-1,nk-1,ic)
        cfln(ni,nj,nk,ic)  = cfl(ni-1,nj-1,nk-1,ic)
    end do
end subroutine cfl_cell_to_node
