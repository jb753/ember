! Geometric calculations for face areas and cell volumes

subroutine get_dAi(xrt, dAi, ni, nj, nk)
    ! Calculate constant i index face areas
    ! Inlined dA_Gauss calculation to avoid temporary array creation

    implicit none

    integer, intent(in) :: ni, nj, nk
    double precision, intent(in) :: xrt(ni, nj, nk, 3)
    double precision, intent(inout) :: dAi(ni, nj-1, nk-1, 3)

    integer :: i, j, k, m
    double precision :: A(3), B(3), C(3), D(3)
    double precision :: xrt_loc(4, 3), xrrt(4, 3)
    double precision :: theta_mean, face_center(3)
    double precision :: dv(4, 3), vm(4, 3)
    double precision :: F(4, 3, 3), dl(4, 3, 3)

    do k = 1, nk-1
        do j = 1, nj-1
            do i = 1, ni
                ! Get vertices for this face
                A = xrt(i, j, k, :)      ! A(i,j,k)
                B = xrt(i, j, k+1, :)    ! B(i,j,k+1)
                C = xrt(i, j+1, k+1, :)  ! C(i,j+1,k+1)
                D = xrt(i, j+1, k, :)    ! D(i,j+1,k)

                ! Inlined dA_Gauss calculation
                ! Assemble vertices
                xrt_loc(1, :) = A
                xrt_loc(2, :) = B
                xrt_loc(3, :) = C
                xrt_loc(4, :) = D

                ! Shift theta origins to center of face
                theta_mean = (xrt_loc(1, 3) + xrt_loc(2, 3) + xrt_loc(3, 3) + xrt_loc(4, 3)) * 0.25d0
                xrt_loc(:, 3) = xrt_loc(:, 3) - theta_mean

                ! Convert to pseudo-Cartesian: [x, r, r*theta]
                xrrt(:, 1) = xrt_loc(:, 1)
                xrrt(:, 2) = xrt_loc(:, 2)
                xrrt(:, 3) = xrt_loc(:, 2) * xrt_loc(:, 3)

                ! Subtract face center to reduce round-off error
                face_center = (xrrt(1, :) + xrrt(2, :) + xrrt(3, :) + xrrt(4, :)) * 0.25d0
                do m = 1, 4
                    xrrt(m, :) = xrrt(m, :) - face_center
                end do

                ! Edge vectors
                dv(1, :) = xrrt(2, :) - xrrt(1, :)
                dv(2, :) = xrrt(3, :) - xrrt(2, :)
                dv(3, :) = xrrt(4, :) - xrrt(3, :)
                dv(4, :) = xrrt(1, :) - xrrt(4, :)

                ! Edge midpoint vertices
                vm(1, :) = 0.5d0 * (xrrt(1, :) + xrrt(2, :))
                vm(2, :) = 0.5d0 * (xrrt(2, :) + xrrt(3, :))
                vm(3, :) = 0.5d0 * (xrrt(3, :) + xrrt(4, :))
                vm(4, :) = 0.5d0 * (xrrt(4, :) + xrrt(1, :))

                ! Vector field F with zeroed elements
                F(:, 1, 1) = 0.0d0
                F(:, 1, 2) = vm(:, 2)
                F(:, 1, 3) = vm(:, 3)
                F(:, 2, 1) = vm(:, 1)
                F(:, 2, 2) = 0.0d0
                F(:, 2, 3) = vm(:, 3)
                F(:, 3, 1) = vm(:, 1)
                F(:, 3, 2) = vm(:, 2)
                F(:, 3, 3) = 0.0d0

                ! Edge normal components
                dl(:, 1, 1) = dv(:, 1)
                dl(:, 1, 2) = -dv(:, 3)
                dl(:, 1, 3) = dv(:, 2)
                dl(:, 2, 1) = dv(:, 3)
                dl(:, 2, 2) = dv(:, 2)
                dl(:, 2, 3) = -dv(:, 1)
                dl(:, 3, 1) = -dv(:, 2)
                dl(:, 3, 2) = dv(:, 1)
                dl(:, 3, 3) = dv(:, 3)

                ! Apply Gauss' theorem for area - direct assignment
                dAi(i, j, k, 1) = 0.0d0
                dAi(i, j, k, 2) = 0.0d0
                dAi(i, j, k, 3) = 0.0d0
                do m = 1, 4
                    dAi(i, j, k, 1) = dAi(i, j, k, 1) + F(m, 1, 1) * dl(m, 1, 1) &
                        + F(m, 1, 2) * dl(m, 1, 2) + F(m, 1, 3) * dl(m, 1, 3)
                    dAi(i, j, k, 2) = dAi(i, j, k, 2) + F(m, 2, 1) * dl(m, 2, 1) &
                        + F(m, 2, 2) * dl(m, 2, 2) + F(m, 2, 3) * dl(m, 2, 3)
                    dAi(i, j, k, 3) = dAi(i, j, k, 3) + F(m, 3, 1) * dl(m, 3, 1) &
                        + F(m, 3, 2) * dl(m, 3, 2) + F(m, 3, 3) * dl(m, 3, 3)
                end do
                dAi(i, j, k, :) = dAi(i, j, k, :) * 0.5d0
            end do
        end do
    end do

end subroutine get_dAi


subroutine get_dAj(xrt, dAj, ni, nj, nk)
    ! Calculate constant j index face areas
    ! Inlined dA_Gauss calculation to avoid temporary array creation

    implicit none

    integer, intent(in) :: ni, nj, nk
    double precision, intent(in) :: xrt(ni, nj, nk, 3)
    double precision, intent(inout) :: dAj(ni-1, nj, nk-1, 3)

    integer :: i, j, k, m
    double precision :: A(3), B(3), C(3), D(3)
    double precision :: xrt_loc(4, 3), xrrt(4, 3)
    double precision :: theta_mean, face_center(3)
    double precision :: dv(4, 3), vm(4, 3)
    double precision :: F(4, 3, 3), dl(4, 3, 3)

    do k = 1, nk-1
        do j = 1, nj
            do i = 1, ni-1
                ! Get vertices for this face
                A = xrt(i, j, k, :)      ! A(i,j,k)
                B = xrt(i, j, k+1, :)    ! B(i,j,k+1)
                C = xrt(i+1, j, k+1, :)  ! C(i+1,j,k+1)
                D = xrt(i+1, j, k, :)    ! D(i+1,j,k)

                ! Inlined dA_Gauss calculation
                ! Assemble vertices
                xrt_loc(1, :) = A
                xrt_loc(2, :) = B
                xrt_loc(3, :) = C
                xrt_loc(4, :) = D

                ! Shift theta origins to center of face
                theta_mean = (xrt_loc(1, 3) + xrt_loc(2, 3) + xrt_loc(3, 3) + xrt_loc(4, 3)) * 0.25d0
                xrt_loc(:, 3) = xrt_loc(:, 3) - theta_mean

                ! Convert to pseudo-Cartesian: [x, r, r*theta]
                xrrt(:, 1) = xrt_loc(:, 1)
                xrrt(:, 2) = xrt_loc(:, 2)
                xrrt(:, 3) = xrt_loc(:, 2) * xrt_loc(:, 3)

                ! Subtract face center to reduce round-off error
                face_center = (xrrt(1, :) + xrrt(2, :) + xrrt(3, :) + xrrt(4, :)) * 0.25d0
                do m = 1, 4
                    xrrt(m, :) = xrrt(m, :) - face_center
                end do

                ! Edge vectors
                dv(1, :) = xrrt(2, :) - xrrt(1, :)
                dv(2, :) = xrrt(3, :) - xrrt(2, :)
                dv(3, :) = xrrt(4, :) - xrrt(3, :)
                dv(4, :) = xrrt(1, :) - xrrt(4, :)

                ! Edge midpoint vertices
                vm(1, :) = 0.5d0 * (xrrt(1, :) + xrrt(2, :))
                vm(2, :) = 0.5d0 * (xrrt(2, :) + xrrt(3, :))
                vm(3, :) = 0.5d0 * (xrrt(3, :) + xrrt(4, :))
                vm(4, :) = 0.5d0 * (xrrt(4, :) + xrrt(1, :))

                ! Vector field F with zeroed elements
                F(:, 1, 1) = 0.0d0
                F(:, 1, 2) = vm(:, 2)
                F(:, 1, 3) = vm(:, 3)
                F(:, 2, 1) = vm(:, 1)
                F(:, 2, 2) = 0.0d0
                F(:, 2, 3) = vm(:, 3)
                F(:, 3, 1) = vm(:, 1)
                F(:, 3, 2) = vm(:, 2)
                F(:, 3, 3) = 0.0d0

                ! Edge normal components
                dl(:, 1, 1) = dv(:, 1)
                dl(:, 1, 2) = -dv(:, 3)
                dl(:, 1, 3) = dv(:, 2)
                dl(:, 2, 1) = dv(:, 3)
                dl(:, 2, 2) = dv(:, 2)
                dl(:, 2, 3) = -dv(:, 1)
                dl(:, 3, 1) = -dv(:, 2)
                dl(:, 3, 2) = dv(:, 1)
                dl(:, 3, 3) = dv(:, 3)

                ! Apply Gauss' theorem for area - direct assignment (negated for dAj)
                dAj(i, j, k, 1) = 0.0d0
                dAj(i, j, k, 2) = 0.0d0
                dAj(i, j, k, 3) = 0.0d0
                do m = 1, 4
                    dAj(i, j, k, 1) = dAj(i, j, k, 1) + F(m, 1, 1) * dl(m, 1, 1) &
                        + F(m, 1, 2) * dl(m, 1, 2) + F(m, 1, 3) * dl(m, 1, 3)
                    dAj(i, j, k, 2) = dAj(i, j, k, 2) + F(m, 2, 1) * dl(m, 2, 1) &
                        + F(m, 2, 2) * dl(m, 2, 2) + F(m, 2, 3) * dl(m, 2, 3)
                    dAj(i, j, k, 3) = dAj(i, j, k, 3) + F(m, 3, 1) * dl(m, 3, 1) &
                        + F(m, 3, 2) * dl(m, 3, 2) + F(m, 3, 3) * dl(m, 3, 3)
                end do
                dAj(i, j, k, :) = -dAj(i, j, k, :) * 0.5d0
            end do
        end do
    end do

end subroutine get_dAj


subroutine get_dAk(xrt, dAk, ni, nj, nk)
    ! Calculate constant k index face areas
    ! Inlined dA_Gauss calculation to avoid temporary array creation

    implicit none

    integer, intent(in) :: ni, nj, nk
    double precision, intent(in) :: xrt(ni, nj, nk, 3)
    double precision, intent(inout) :: dAk(ni-1, nj-1, nk, 3)

    integer :: i, j, k, m
    double precision :: A(3), B(3), C(3), D(3)
    double precision :: xrt_loc(4, 3), xrrt(4, 3)
    double precision :: theta_mean, face_center(3)
    double precision :: dv(4, 3), vm(4, 3)
    double precision :: F(4, 3, 3), dl(4, 3, 3)

    do k = 1, nk
        do j = 1, nj-1
            do i = 1, ni-1
                ! Get vertices for this face
                A = xrt(i, j, k, :)      ! A(i,j,k)
                B = xrt(i, j+1, k, :)    ! B(i,j+1,k)
                C = xrt(i+1, j+1, k, :)  ! C(i+1,j+1,k)
                D = xrt(i+1, j, k, :)    ! D(i+1,j,k)

                ! Inlined dA_Gauss calculation
                ! Assemble vertices
                xrt_loc(1, :) = A
                xrt_loc(2, :) = B
                xrt_loc(3, :) = C
                xrt_loc(4, :) = D

                ! Shift theta origins to center of face
                theta_mean = (xrt_loc(1, 3) + xrt_loc(2, 3) + xrt_loc(3, 3) + xrt_loc(4, 3)) * 0.25d0
                xrt_loc(:, 3) = xrt_loc(:, 3) - theta_mean

                ! Convert to pseudo-Cartesian: [x, r, r*theta]
                xrrt(:, 1) = xrt_loc(:, 1)
                xrrt(:, 2) = xrt_loc(:, 2)
                xrrt(:, 3) = xrt_loc(:, 2) * xrt_loc(:, 3)

                ! Subtract face center to reduce round-off error
                face_center = (xrrt(1, :) + xrrt(2, :) + xrrt(3, :) + xrrt(4, :)) * 0.25d0
                do m = 1, 4
                    xrrt(m, :) = xrrt(m, :) - face_center
                end do

                ! Edge vectors
                dv(1, :) = xrrt(2, :) - xrrt(1, :)
                dv(2, :) = xrrt(3, :) - xrrt(2, :)
                dv(3, :) = xrrt(4, :) - xrrt(3, :)
                dv(4, :) = xrrt(1, :) - xrrt(4, :)

                ! Edge midpoint vertices
                vm(1, :) = 0.5d0 * (xrrt(1, :) + xrrt(2, :))
                vm(2, :) = 0.5d0 * (xrrt(2, :) + xrrt(3, :))
                vm(3, :) = 0.5d0 * (xrrt(3, :) + xrrt(4, :))
                vm(4, :) = 0.5d0 * (xrrt(4, :) + xrrt(1, :))

                ! Vector field F with zeroed elements
                F(:, 1, 1) = 0.0d0
                F(:, 1, 2) = vm(:, 2)
                F(:, 1, 3) = vm(:, 3)
                F(:, 2, 1) = vm(:, 1)
                F(:, 2, 2) = 0.0d0
                F(:, 2, 3) = vm(:, 3)
                F(:, 3, 1) = vm(:, 1)
                F(:, 3, 2) = vm(:, 2)
                F(:, 3, 3) = 0.0d0

                ! Edge normal components
                dl(:, 1, 1) = dv(:, 1)
                dl(:, 1, 2) = -dv(:, 3)
                dl(:, 1, 3) = dv(:, 2)
                dl(:, 2, 1) = dv(:, 3)
                dl(:, 2, 2) = dv(:, 2)
                dl(:, 2, 3) = -dv(:, 1)
                dl(:, 3, 1) = -dv(:, 2)
                dl(:, 3, 2) = dv(:, 1)
                dl(:, 3, 3) = dv(:, 3)

                ! Apply Gauss' theorem for area - direct assignment
                dAk(i, j, k, 1) = 0.0d0
                dAk(i, j, k, 2) = 0.0d0
                dAk(i, j, k, 3) = 0.0d0
                do m = 1, 4
                    dAk(i, j, k, 1) = dAk(i, j, k, 1) + F(m, 1, 1) * dl(m, 1, 1) &
                        + F(m, 1, 2) * dl(m, 1, 2) + F(m, 1, 3) * dl(m, 1, 3)
                    dAk(i, j, k, 2) = dAk(i, j, k, 2) + F(m, 2, 1) * dl(m, 2, 1) &
                        + F(m, 2, 2) * dl(m, 2, 2) + F(m, 2, 3) * dl(m, 2, 3)
                    dAk(i, j, k, 3) = dAk(i, j, k, 3) + F(m, 3, 1) * dl(m, 3, 1) &
                        + F(m, 3, 2) * dl(m, 3, 2) + F(m, 3, 3) * dl(m, 3, 3)
                end do
                dAk(i, j, k, :) = dAk(i, j, k, :) * 0.5d0
            end do
        end do
    end do

end subroutine get_dAk


subroutine get_vol(xrt, dAi, dAj, dAk, vol, ni, nj, nk)
    ! Calculate cell volumes using Gauss's theorem in polar coordinates
    !
    ! For each cell, computes volume as divergence of field F = [x, r^2/2, r*theta]
    ! using face fluxes: vol = (1/3) * sum over faces of (F . dA)
    !
    ! This explicit loop implementation minimizes memory usage and works with -Ofast

    implicit none

    integer, intent(in) :: ni, nj, nk
    double precision, intent(in) :: xrt(ni, nj, nk, 3)
    double precision, intent(in) :: dAi(3, ni, nj-1, nk-1)
    double precision, intent(in) :: dAj(3, ni-1, nj, nk-1)
    double precision, intent(in) :: dAk(3, ni-1, nj-1, nk)
    double precision, intent(inout) :: vol(ni-1, nj-1, nk-1)

    ! Local variables for face-centered values
    double precision :: xi, ri, rti  ! Face-centered x, r, r*theta for i-faces
    double precision :: xj, rj, rtj  ! Face-centered x, r, r*theta for j-faces
    double precision :: xk, rk, rtk  ! Face-centered x, r, r*theta for k-faces
    double precision :: Fi_dot_dAi_lo, Fi_dot_dAi_hi  ! Fluxes through i-faces
    double precision :: Fj_dot_dAj_lo, Fj_dot_dAj_hi  ! Fluxes through j-faces
    double precision :: Fk_dot_dAk_lo, Fk_dot_dAk_hi  ! Fluxes through k-faces
    double precision :: div_F  ! Divergence of field F
    integer :: i, j, k

    do k = 1, nk-1
        do j = 1, nj-1
            do i = 1, ni-1
                ! === i-faces (lower and upper in i-direction) ===

                ! Lower i-face: average 4 nodes at (i, j:j+1, k:k+1)
                xi = 0.25d0 * (xrt(i, j, k, 1) + xrt(i, j+1, k, 1) + &
                               xrt(i, j+1, k+1, 1) + xrt(i, j, k+1, 1))
                ri = 0.25d0 * (xrt(i, j, k, 2) + xrt(i, j+1, k, 2) + &
                               xrt(i, j+1, k+1, 2) + xrt(i, j, k+1, 2))
                rti = 0.25d0 * (xrt(i, j, k, 2) * xrt(i, j, k, 3) + &
                                xrt(i, j+1, k, 2) * xrt(i, j+1, k, 3) + &
                                xrt(i, j+1, k+1, 2) * xrt(i, j+1, k+1, 3) + &
                                xrt(i, j, k+1, 2) * xrt(i, j, k+1, 3))

                ! F_i = [xi, ri/2, rti], dot with dAi
                Fi_dot_dAi_lo = xi * dAi(1, i, j, k) + &
                                (ri * 0.5d0) * dAi(2, i, j, k) + &
                                rti * dAi(3, i, j, k)

                ! Upper i-face: average 4 nodes at (i+1, j:j+1, k:k+1)
                xi = 0.25d0 * (xrt(i+1, j, k, 1) + xrt(i+1, j+1, k, 1) + &
                               xrt(i+1, j+1, k+1, 1) + xrt(i+1, j, k+1, 1))
                ri = 0.25d0 * (xrt(i+1, j, k, 2) + xrt(i+1, j+1, k, 2) + &
                               xrt(i+1, j+1, k+1, 2) + xrt(i+1, j, k+1, 2))
                rti = 0.25d0 * (xrt(i+1, j, k, 2) * xrt(i+1, j, k, 3) + &
                                xrt(i+1, j+1, k, 2) * xrt(i+1, j+1, k, 3) + &
                                xrt(i+1, j+1, k+1, 2) * xrt(i+1, j+1, k+1, 3) + &
                                xrt(i+1, j, k+1, 2) * xrt(i+1, j, k+1, 3))

                Fi_dot_dAi_hi = xi * dAi(1, i+1, j, k) + &
                                (ri * 0.5d0) * dAi(2, i+1, j, k) + &
                                rti * dAi(3, i+1, j, k)

                ! === j-faces (lower and upper in j-direction) ===

                ! Lower j-face: average 4 nodes at (i:i+1, j, k:k+1)
                xj = 0.25d0 * (xrt(i, j, k, 1) + xrt(i+1, j, k, 1) + &
                               xrt(i+1, j, k+1, 1) + xrt(i, j, k+1, 1))
                rj = 0.25d0 * (xrt(i, j, k, 2) + xrt(i+1, j, k, 2) + &
                               xrt(i+1, j, k+1, 2) + xrt(i, j, k+1, 2))
                rtj = 0.25d0 * (xrt(i, j, k, 2) * xrt(i, j, k, 3) + &
                                xrt(i+1, j, k, 2) * xrt(i+1, j, k, 3) + &
                                xrt(i+1, j, k+1, 2) * xrt(i+1, j, k+1, 3) + &
                                xrt(i, j, k+1, 2) * xrt(i, j, k+1, 3))

                Fj_dot_dAj_lo = xj * dAj(1, i, j, k) + &
                                (rj * 0.5d0) * dAj(2, i, j, k) + &
                                rtj * dAj(3, i, j, k)

                ! Upper j-face: average 4 nodes at (i:i+1, j+1, k:k+1)
                xj = 0.25d0 * (xrt(i, j+1, k, 1) + xrt(i+1, j+1, k, 1) + &
                               xrt(i+1, j+1, k+1, 1) + xrt(i, j+1, k+1, 1))
                rj = 0.25d0 * (xrt(i, j+1, k, 2) + xrt(i+1, j+1, k, 2) + &
                               xrt(i+1, j+1, k+1, 2) + xrt(i, j+1, k+1, 2))
                rtj = 0.25d0 * (xrt(i, j+1, k, 2) * xrt(i, j+1, k, 3) + &
                                xrt(i+1, j+1, k, 2) * xrt(i+1, j+1, k, 3) + &
                                xrt(i+1, j+1, k+1, 2) * xrt(i+1, j+1, k+1, 3) + &
                                xrt(i, j+1, k+1, 2) * xrt(i, j+1, k+1, 3))

                Fj_dot_dAj_hi = xj * dAj(1, i, j+1, k) + &
                                (rj * 0.5d0) * dAj(2, i, j+1, k) + &
                                rtj * dAj(3, i, j+1, k)

                ! === k-faces (lower and upper in k-direction) ===

                ! Lower k-face: average 4 nodes at (i:i+1, j:j+1, k)
                xk = 0.25d0 * (xrt(i, j, k, 1) + xrt(i+1, j, k, 1) + &
                               xrt(i+1, j+1, k, 1) + xrt(i, j+1, k, 1))
                rk = 0.25d0 * (xrt(i, j, k, 2) + xrt(i+1, j, k, 2) + &
                               xrt(i+1, j+1, k, 2) + xrt(i, j+1, k, 2))
                rtk = 0.25d0 * (xrt(i, j, k, 2) * xrt(i, j, k, 3) + &
                                xrt(i+1, j, k, 2) * xrt(i+1, j, k, 3) + &
                                xrt(i+1, j+1, k, 2) * xrt(i+1, j+1, k, 3) + &
                                xrt(i, j+1, k, 2) * xrt(i, j+1, k, 3))

                Fk_dot_dAk_lo = xk * dAk(1, i, j, k) + &
                                (rk * 0.5d0) * dAk(2, i, j, k) + &
                                rtk * dAk(3, i, j, k)

                ! Upper k-face: average 4 nodes at (i:i+1, j:j+1, k+1)
                xk = 0.25d0 * (xrt(i, j, k+1, 1) + xrt(i+1, j, k+1, 1) + &
                               xrt(i+1, j+1, k+1, 1) + xrt(i, j+1, k+1, 1))
                rk = 0.25d0 * (xrt(i, j, k+1, 2) + xrt(i+1, j, k+1, 2) + &
                               xrt(i+1, j+1, k+1, 2) + xrt(i, j+1, k+1, 2))
                rtk = 0.25d0 * (xrt(i, j, k+1, 2) * xrt(i, j, k+1, 3) + &
                                xrt(i+1, j, k+1, 2) * xrt(i+1, j, k+1, 3) + &
                                xrt(i+1, j+1, k+1, 2) * xrt(i+1, j+1, k+1, 3) + &
                                xrt(i, j+1, k+1, 2) * xrt(i, j+1, k+1, 3))

                Fk_dot_dAk_hi = xk * dAk(1, i, j, k+1) + &
                                (rk * 0.5d0) * dAk(2, i, j, k+1) + &
                                rtk * dAk(3, i, j, k+1)

                ! === Divergence theorem: sum of face fluxes ===
                div_F = (Fi_dot_dAi_hi - Fi_dot_dAi_lo) + &
                        (Fj_dot_dAj_hi - Fj_dot_dAj_lo) + &
                        (Fk_dot_dAk_hi - Fk_dot_dAk_lo)

                ! Volume = divergence / 3
                vol(i, j, k) = div_F / 3.0d0

            end do
        end do
    end do

end subroutine get_vol


subroutine get_zeta(xrt, zeta, ni, nj, nk)
    ! Calculate normalized arc length along i, j, k grid lines
    !
    ! Computes normalized arc length [0 to 1] for all three grid directions
    ! and stores them in a single stacked array.
    !
    ! Parameters
    ! ----------
    ! xrt : real(ni, nj, nk, 3)
    !     Polar coordinates at three-dimensional block nodes
    ! zeta : real(ni, nj, nk, 3)
    !     Output normalized arc lengths [0 to 1] where:
    !     zeta(:,:,:,1) = zetai (arc length along i)
    !     zeta(:,:,:,2) = zetaj (arc length along j)
    !     zeta(:,:,:,3) = zetak (arc length along k)
    ! ni, nj, nk : integer
    !     Grid dimensions

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: xrt(ni, nj, nk, 3)
    real, intent(inout) :: zeta(ni, nj, nk, 3)

    real :: xyz(ni, nj, nk, 3)
    real :: dx(3), ds, total_length
    integer :: i, j, k

    ! Convert to pseudo-Cartesian for metric-aware distance calculations
    xyz(:, :, :, 1) = xrt(:, :, :, 1)  ! x
    xyz(:, :, :, 2) = xrt(:, :, :, 2)  ! r
    xyz(:, :, :, 3) = xrt(:, :, :, 2) * xrt(:, :, :, 3)  ! r*theta

    ! Initialize output
    zeta = 0.0e0

    ! Compute zetai (arc length along i-direction)
    do k = 1, nk
        do j = 1, nj
            ! Cumulative distance along i
            do i = 2, ni
                dx = xyz(i, j, k, :) - xyz(i-1, j, k, :)
                ds = sqrt(dx(1)**2 + dx(2)**2 + dx(3)**2)
                zeta(i, j, k, 1) = zeta(i-1, j, k, 1) + ds
            end do

            ! Normalize to [0, 1]
            total_length = zeta(ni, j, k, 1)
            if (total_length > 0.0e0) then
                zeta(:, j, k, 1) = zeta(:, j, k, 1) / total_length
            end if
        end do
    end do

    ! Compute zetaj (arc length along j-direction)
    do k = 1, nk
        do i = 1, ni
            ! Cumulative distance along j
            do j = 2, nj
                dx = xyz(i, j, k, :) - xyz(i, j-1, k, :)
                ds = sqrt(dx(1)**2 + dx(2)**2 + dx(3)**2)
                zeta(i, j, k, 2) = zeta(i, j-1, k, 2) + ds
            end do

            ! Normalize to [0, 1]
            total_length = zeta(i, nj, k, 2)
            if (total_length > 0.0e0) then
                zeta(i, :, k, 2) = zeta(i, :, k, 2) / total_length
            end if
        end do
    end do

    ! Compute zetak (arc length along k-direction)
    do j = 1, nj
        do i = 1, ni
            ! Cumulative distance along k
            do k = 2, nk
                dx = xyz(i, j, k, :) - xyz(i, j, k-1, :)
                ds = sqrt(dx(1)**2 + dx(2)**2 + dx(3)**2)
                zeta(i, j, k, 3) = zeta(i, j, k-1, 3) + ds
            end do

            ! Normalize to [0, 1]
            total_length = zeta(i, j, nk, 3)
            if (total_length > 0.0e0) then
                zeta(i, j, :, 3) = zeta(i, j, :, 3) / total_length
            end if
        end do
    end do

end subroutine get_zeta
