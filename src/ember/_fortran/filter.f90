! Selective frequency damping (SFD) low-pass filter update.
!
! Evolves the temporally low-pass-filtered conserved state cons_filt by a
! first-order exponential moving average toward the current cell-centred state:
!
!   dt = cfl * dt_vol * vol                              ! local physical timestep
!   cons_filt += dt * (cons_cell - cons_filt) / delta_filt
!
! The acoustic speed folded into dt_vol (= dl_min / ((a + V_rel) * vol)) keeps
! dt finite as V -> 0, so the filter stays stable. cons_filt is read by the SFD
! body force (apply_sfd_force) in Grid.update_F_body.
!
! This was previously fused into adapt_cfl; it is split out because it is
! independent of CFL adaptation and of the time-integration scheme. Two variants
! differ only in whether cfl is a per-cell/per-equation array or a scalar.
!
! update_filter_array: cfl is (ni-1, nj-1, nk-1, 5).
!
subroutine update_filter_array( &
        cons_filt, &
        cons_cell, &
        cfl, &
        dt_vol, &
        vol, &
        delta_filt, &
        ni, nj, nk &
    )

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(inout) :: cons_filt(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: cons_cell(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: cfl(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: delta_filt

    integer :: i, j, k, eq
    real :: dt

    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        do eq = 1, 5
            dt = cfl(i, j, k, eq) * dt_vol(i, j, k) * vol(i, j, k)
            cons_filt(i, j, k, eq) = cons_filt(i, j, k, eq) + &
                dt * (cons_cell(i, j, k, eq) - cons_filt(i, j, k, eq)) / delta_filt
        end do
    end do
    end do
    end do

end subroutine update_filter_array


! update_filter_scalar: cfl is a single scalar shared by all cells/equations.
!
subroutine update_filter_scalar( &
        cons_filt, &
        cons_cell, &
        cfl, &
        dt_vol, &
        vol, &
        delta_filt, &
        ni, nj, nk &
    )

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(inout) :: cons_filt(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: cons_cell(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: cfl
    real, intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: delta_filt

    integer :: i, j, k, eq
    real :: dt

    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        dt = cfl * dt_vol(i, j, k) * vol(i, j, k)
        do eq = 1, 5
            cons_filt(i, j, k, eq) = cons_filt(i, j, k, eq) + &
                dt * (cons_cell(i, j, k, eq) - cons_filt(i, j, k, eq)) / delta_filt
        end do
    end do
    end do
    end do

end subroutine update_filter_scalar
