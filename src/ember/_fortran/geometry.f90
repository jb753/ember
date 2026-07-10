! Geometric calculations for face areas and cell volumes

subroutine dA_cross(A, B, C, D, dA)
    ! Area vector of the quadrilateral face with corner nodes A, B, C, D
    !
    ! Each component of dA is the signed area of the projection of the face on
    ! to the plane perpendicular to that component, which is half the cross
    ! product of the face diagonals,
    !
    !     dA = 1/2 * (D - B) x (C - A)
    !
    ! exact for a warped face whose four nodes are not coplanar.
    !
    ! Nodes are supplied in polar coordinates and converted to pseudo-Cartesian
    ! here.  Circumferential angles are measured from the mean angle of the four
    ! nodes, so that the components of dA are resolved in the cylindrical basis
    ! at the face centre.

    implicit none

    double precision, intent(in) :: A(3), B(3), C(3), D(3)
    double precision, intent(inout) :: dA(3)

    double precision :: xrt_loc(4, 3), xrrt(4, 3)
    double precision :: theta_mean
    double precision :: d1(3), d2(3)

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

    ! Diagonals BD and AC.  Differencing the nodes directly is what keeps the
    ! round-off small, so no shift to the face centre is needed.
    d1 = xrrt(4, :) - xrrt(2, :)
    d2 = xrrt(3, :) - xrrt(1, :)

    dA(1) = 0.5d0 * (d1(2) * d2(3) - d1(3) * d2(2))
    dA(2) = 0.5d0 * (d1(3) * d2(1) - d1(1) * d2(3))
    dA(3) = 0.5d0 * (d1(1) * d2(2) - d1(2) * d2(1))

end subroutine dA_cross


subroutine get_dAi(xrt, dAi, ni, nj, nk)
    ! Calculate constant i index face areas

    implicit none

    integer, intent(in) :: ni, nj, nk
    double precision, intent(in) :: xrt(ni, nj, nk, 3)
    double precision, intent(inout) :: dAi(ni, nj-1, nk-1, 3)

    integer :: i, j, k
    double precision :: A(3), B(3), C(3), D(3), dA_loc(3)

    do k = 1, nk-1
        do j = 1, nj-1
            do i = 1, ni
                A = xrt(i, j, k, :)      ! A(i,j,k)
                B = xrt(i, j, k+1, :)    ! B(i,j,k+1)
                C = xrt(i, j+1, k+1, :)  ! C(i,j+1,k+1)
                D = xrt(i, j+1, k, :)    ! D(i,j+1,k)

                call dA_cross(A, B, C, D, dA_loc)
                dAi(i, j, k, :) = dA_loc
            end do
        end do
    end do

end subroutine get_dAi


subroutine get_dAj(xrt, dAj, ni, nj, nk)
    ! Calculate constant j index face areas

    implicit none

    integer, intent(in) :: ni, nj, nk
    double precision, intent(in) :: xrt(ni, nj, nk, 3)
    double precision, intent(inout) :: dAj(ni-1, nj, nk-1, 3)

    integer :: i, j, k
    double precision :: A(3), B(3), C(3), D(3), dA_loc(3)

    do k = 1, nk-1
        do j = 1, nj
            do i = 1, ni-1
                A = xrt(i, j, k, :)      ! A(i,j,k)
                B = xrt(i+1, j, k, :)    ! B(i+1,j,k)
                C = xrt(i+1, j, k+1, :)  ! C(i+1,j,k+1)
                D = xrt(i, j, k+1, :)    ! D(i,j,k+1)

                call dA_cross(A, B, C, D, dA_loc)
                dAj(i, j, k, :) = dA_loc
            end do
        end do
    end do

end subroutine get_dAj


subroutine get_dAk(xrt, dAk, ni, nj, nk)
    ! Calculate constant k index face areas

    implicit none

    integer, intent(in) :: ni, nj, nk
    double precision, intent(in) :: xrt(ni, nj, nk, 3)
    double precision, intent(inout) :: dAk(ni-1, nj-1, nk, 3)

    integer :: i, j, k
    double precision :: A(3), B(3), C(3), D(3), dA_loc(3)

    do k = 1, nk
        do j = 1, nj-1
            do i = 1, ni-1
                A = xrt(i, j, k, :)      ! A(i,j,k)
                B = xrt(i, j+1, k, :)    ! B(i,j+1,k)
                C = xrt(i+1, j+1, k, :)  ! C(i+1,j+1,k)
                D = xrt(i+1, j, k, :)    ! D(i+1,j,k)

                call dA_cross(A, B, C, D, dA_loc)
                dAk(i, j, k, :) = dA_loc
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
    double precision :: tc  ! Mean angle of the eight cell corner nodes
    integer :: i, j, k

    do k = 1, nk-1
        do j = 1, nj-1
            do i = 1, ni-1
                ! Measure theta from the cell centre.  The face area vectors are
                ! resolved in the cylindrical basis at each face centre, so F must
                ! share a single angular origin across the six faces of this cell.
                ! Using the global theta instead makes the volume depend on where
                ! the angular origin happens to sit, which is unphysical.
                tc = 0.125d0 * (xrt(i, j, k, 3) + xrt(i+1, j, k, 3) + &
                                xrt(i, j+1, k, 3) + xrt(i+1, j+1, k, 3) + &
                                xrt(i, j, k+1, 3) + xrt(i+1, j, k+1, 3) + &
                                xrt(i, j+1, k+1, 3) + xrt(i+1, j+1, k+1, 3))

                ! === i-faces (lower and upper in i-direction) ===

                ! Lower i-face: average 4 nodes at (i, j:j+1, k:k+1)
                xi = 0.25d0 * (xrt(i, j, k, 1) + xrt(i, j+1, k, 1) + &
                               xrt(i, j+1, k+1, 1) + xrt(i, j, k+1, 1))
                ri = 0.25d0 * (xrt(i, j, k, 2) + xrt(i, j+1, k, 2) + &
                               xrt(i, j+1, k+1, 2) + xrt(i, j, k+1, 2))
                rti = 0.25d0 * (xrt(i, j, k, 2) * (xrt(i, j, k, 3) - tc) + &
                                xrt(i, j+1, k, 2) * (xrt(i, j+1, k, 3) - tc) + &
                                xrt(i, j+1, k+1, 2) * (xrt(i, j+1, k+1, 3) - tc) + &
                                xrt(i, j, k+1, 2) * (xrt(i, j, k+1, 3) - tc))

                ! F_i = [xi, ri/2, rti], dot with dAi
                Fi_dot_dAi_lo = xi * dAi(1, i, j, k) + &
                                (ri * 0.5d0) * dAi(2, i, j, k) + &
                                rti * dAi(3, i, j, k)

                ! Upper i-face: average 4 nodes at (i+1, j:j+1, k:k+1)
                xi = 0.25d0 * (xrt(i+1, j, k, 1) + xrt(i+1, j+1, k, 1) + &
                               xrt(i+1, j+1, k+1, 1) + xrt(i+1, j, k+1, 1))
                ri = 0.25d0 * (xrt(i+1, j, k, 2) + xrt(i+1, j+1, k, 2) + &
                               xrt(i+1, j+1, k+1, 2) + xrt(i+1, j, k+1, 2))
                rti = 0.25d0 * (xrt(i+1, j, k, 2) * (xrt(i+1, j, k, 3) - tc) + &
                                xrt(i+1, j+1, k, 2) * (xrt(i+1, j+1, k, 3) - tc) + &
                                xrt(i+1, j+1, k+1, 2) * (xrt(i+1, j+1, k+1, 3) - tc) + &
                                xrt(i+1, j, k+1, 2) * (xrt(i+1, j, k+1, 3) - tc))

                Fi_dot_dAi_hi = xi * dAi(1, i+1, j, k) + &
                                (ri * 0.5d0) * dAi(2, i+1, j, k) + &
                                rti * dAi(3, i+1, j, k)

                ! === j-faces (lower and upper in j-direction) ===

                ! Lower j-face: average 4 nodes at (i:i+1, j, k:k+1)
                xj = 0.25d0 * (xrt(i, j, k, 1) + xrt(i+1, j, k, 1) + &
                               xrt(i+1, j, k+1, 1) + xrt(i, j, k+1, 1))
                rj = 0.25d0 * (xrt(i, j, k, 2) + xrt(i+1, j, k, 2) + &
                               xrt(i+1, j, k+1, 2) + xrt(i, j, k+1, 2))
                rtj = 0.25d0 * (xrt(i, j, k, 2) * (xrt(i, j, k, 3) - tc) + &
                                xrt(i+1, j, k, 2) * (xrt(i+1, j, k, 3) - tc) + &
                                xrt(i+1, j, k+1, 2) * (xrt(i+1, j, k+1, 3) - tc) + &
                                xrt(i, j, k+1, 2) * (xrt(i, j, k+1, 3) - tc))

                Fj_dot_dAj_lo = xj * dAj(1, i, j, k) + &
                                (rj * 0.5d0) * dAj(2, i, j, k) + &
                                rtj * dAj(3, i, j, k)

                ! Upper j-face: average 4 nodes at (i:i+1, j+1, k:k+1)
                xj = 0.25d0 * (xrt(i, j+1, k, 1) + xrt(i+1, j+1, k, 1) + &
                               xrt(i+1, j+1, k+1, 1) + xrt(i, j+1, k+1, 1))
                rj = 0.25d0 * (xrt(i, j+1, k, 2) + xrt(i+1, j+1, k, 2) + &
                               xrt(i+1, j+1, k+1, 2) + xrt(i, j+1, k+1, 2))
                rtj = 0.25d0 * (xrt(i, j+1, k, 2) * (xrt(i, j+1, k, 3) - tc) + &
                                xrt(i+1, j+1, k, 2) * (xrt(i+1, j+1, k, 3) - tc) + &
                                xrt(i+1, j+1, k+1, 2) * (xrt(i+1, j+1, k+1, 3) - tc) + &
                                xrt(i, j+1, k+1, 2) * (xrt(i, j+1, k+1, 3) - tc))

                Fj_dot_dAj_hi = xj * dAj(1, i, j+1, k) + &
                                (rj * 0.5d0) * dAj(2, i, j+1, k) + &
                                rtj * dAj(3, i, j+1, k)

                ! === k-faces (lower and upper in k-direction) ===

                ! Lower k-face: average 4 nodes at (i:i+1, j:j+1, k)
                xk = 0.25d0 * (xrt(i, j, k, 1) + xrt(i+1, j, k, 1) + &
                               xrt(i+1, j+1, k, 1) + xrt(i, j+1, k, 1))
                rk = 0.25d0 * (xrt(i, j, k, 2) + xrt(i+1, j, k, 2) + &
                               xrt(i+1, j+1, k, 2) + xrt(i, j+1, k, 2))
                rtk = 0.25d0 * (xrt(i, j, k, 2) * (xrt(i, j, k, 3) - tc) + &
                                xrt(i+1, j, k, 2) * (xrt(i+1, j, k, 3) - tc) + &
                                xrt(i+1, j+1, k, 2) * (xrt(i+1, j+1, k, 3) - tc) + &
                                xrt(i, j+1, k, 2) * (xrt(i, j+1, k, 3) - tc))

                Fk_dot_dAk_lo = xk * dAk(1, i, j, k) + &
                                (rk * 0.5d0) * dAk(2, i, j, k) + &
                                rtk * dAk(3, i, j, k)

                ! Upper k-face: average 4 nodes at (i:i+1, j:j+1, k+1)
                xk = 0.25d0 * (xrt(i, j, k+1, 1) + xrt(i+1, j, k+1, 1) + &
                               xrt(i+1, j+1, k+1, 1) + xrt(i, j+1, k+1, 1))
                rk = 0.25d0 * (xrt(i, j, k+1, 2) + xrt(i+1, j, k+1, 2) + &
                               xrt(i+1, j+1, k+1, 2) + xrt(i, j+1, k+1, 2))
                rtk = 0.25d0 * (xrt(i, j, k+1, 2) * (xrt(i, j, k+1, 3) - tc) + &
                                xrt(i+1, j, k+1, 2) * (xrt(i+1, j, k+1, 3) - tc) + &
                                xrt(i+1, j+1, k+1, 2) * (xrt(i+1, j+1, k+1, 3) - tc) + &
                                xrt(i, j+1, k+1, 2) * (xrt(i, j+1, k+1, 3) - tc))

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
