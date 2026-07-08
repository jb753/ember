! Set the polar source term for radial momentum
! Accounts for the difference in areas between lower and upper r faces
! S = (rho*Vt^2 + DP_offset)/r * vol
! The units of S are Newtons - it is a body force that
! must be added to net_flow
! set_polar_source_v2: net_flow components-last layout (ni-1,nj-1,nk-1,5).
!
subroutine set_polar_source( &
    cons_cell, r, P, P_offset, vol, net_flow, &
    ni, nj, nk &
    )

    implicit none

    real, intent (in)    :: cons_cell(ni-1, nj-1, nk-1, 5)
    real, intent (in)    :: r(ni, nj, nk)
    real, intent (in)    :: P(ni, nj, nk)
    real, intent (in)    :: P_offset
    real, intent (in)    :: vol(ni-1, nj-1, nk-1)
    real, intent (inout) :: net_flow(ni-1, nj-1, nk-1, 5)
    integer, intent (in) :: ni, nj, nk

    integer :: i, j, k
    real :: rhoc, Pc, rc, Vtc, rhorVtc, S

    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        rhoc    = cons_cell(i, j, k, 1)
        rhorVtc = cons_cell(i, j, k, 4)
        rc      = avg_cell(r, i, j, k)
        Pc      = avg_cell(P, i, j, k)
        Vtc = rhorVtc / (rhoc * rc)
        S   = ((Pc - P_offset) + rhoc * Vtc ** 2) / rc
        net_flow(i, j, k, 3) = net_flow(i, j, k, 3) + vol(i, j, k) * S
    end do
    end do
    end do

contains

    pure function avg_cell(x, i, j, k) result(avg)
        implicit none
        real, intent(in) :: x(ni,nj,nk)
        integer, intent(in) :: i, j, k
        real :: avg
        avg = 0.125e0 * ( &
            x(i,j,k) + x(i+1,j,k) + x(i,j+1,k) + x(i+1,j+1,k) + &
            x(i,j,k+1) + x(i+1,j,k+1) + x(i,j+1,k+1) + x(i+1,j+1,k+1))
    end function avg_cell

end subroutine set_polar_source
