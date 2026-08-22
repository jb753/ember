! Set the polar source term for radial momentum
! Accounts for the difference in areas between lower and upper r faces
! S = (rho*Vt^2 + DP_offset)/r * vol
! The units of S are Newtons - it is a body force that
! must be added to net_flow
! set_polar_source_v2: net_flow components-last layout (ni-1,nj-1,nk-1,5).
!
subroutine set_polar_source( &
    cons, r, P, P_offset, vol, net_flow, &
    ni, nj, nk &
    )

    implicit none

    ! Node-centered conserved variables, averaged to the cell below. Only rho
    ! and rho*r*Vt are read, so only those two are ever averaged.
    real, intent (in)    :: cons(ni, nj, nk, 5)
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
        rhoc    = avg_cell5(cons, 1, i, j, k)
        rhorVtc = avg_cell5(cons, 4, i, j, k)
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

    ! avg_cell over one component of a five-component nodal array. Takes the
    ! component index rather than a cons(:,:,:,m) section, which against an
    ! explicit-shape dummy risks a copy-in/copy-out of the whole volume.
    pure function avg_cell5(x, m, i, j, k) result(avg)
        implicit none
        real, intent(in) :: x(ni,nj,nk,5)
        integer, intent(in) :: m, i, j, k
        real :: avg
        avg = 0.125e0 * ( &
            x(i,j,k,m) + x(i+1,j,k,m) + x(i,j+1,k,m) + x(i+1,j+1,k,m) + &
            x(i,j,k+1,m) + x(i+1,j,k+1,m) + x(i,j+1,k+1,m) + x(i+1,j+1,k+1,m))
    end function avg_cell5

end subroutine set_polar_source
