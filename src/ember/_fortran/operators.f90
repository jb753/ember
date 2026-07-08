! Divergence and gradient operators

subroutine grad(x, gradx, vol, dAi, dAj, dAk, r, ni, nj, nk)
    ! Gradient at cell centers using grad_cell pure function

    implicit none

    real, intent (in)  :: x(ni, nj, nk)

    real, intent (in)  :: dAi(3, ni, nj-1, nk-1)
    real, intent (in)  :: dAj(3, ni-1, nj, nk-1)
    real, intent (in)  :: dAk(3, ni-1, nj-1, nk)
    real, intent (in)  :: vol(ni-1, nj-1, nk-1)

    real, intent (inout)  :: gradx(ni, nj, nk, 3)

    integer, intent (in)  :: ni
    integer, intent (in)  :: nj
    integer, intent (in)  :: nk

    real, intent (in)  :: r(ni, nj, nk)

    ! Local variables
    integer :: i, j, k
    real :: inv_vol, rc_cell
    real :: g(3)

    ! Compute gradient for each cell using grad_cell
    do k = 1, nk-1
        do j = 1, nj-1
            do i = 1, ni-1
                inv_vol = 0.25e0 / vol(i, j, k)
                rc_cell = avg_cell(r, i, j, k)

                g = grad_cell(x, r, rc_cell, dAi, dAj, dAk, inv_vol, i, j, k, ni, nj, nk)
                gradx(i, j, k, 1) = g(1)
                gradx(i, j, k, 2) = g(2)
                gradx(i, j, k, 3) = g(3)
            end do
        end do
    end do

contains

    pure function avg_cell(x, i, j, k) result(avg)
        implicit none
        real, intent(in) :: x(:,:,:)
        integer, intent(in) :: i, j, k
        real :: avg
        avg = 0.125e0 * ( &
            x(i,j,k) + x(i+1,j,k) + x(i,j+1,k) + x(i+1,j+1,k) + &
            x(i,j,k+1) + x(i+1,j,k+1) + x(i,j+1,k+1) + x(i+1,j+1,k+1))
    end function avg_cell

    pure function iface(x, i, j, k) result(sum4)
        implicit none
        real, intent(in) :: x(:,:,:)
        integer, intent(in) :: i, j, k
        real :: sum4
        sum4 = x(i,j,k) + x(i,j+1,k) + x(i,j,k+1) + x(i,j+1,k+1)
    end function iface

    pure function jface(x, i, j, k) result(sum4)
        implicit none
        real, intent(in) :: x(:,:,:)
        integer, intent(in) :: i, j, k
        real :: sum4
        sum4 = x(i,j,k) + x(i+1,j,k) + x(i,j,k+1) + x(i+1,j,k+1)
    end function jface

    pure function kface(x, i, j, k) result(sum4)
        implicit none
        real, intent(in) :: x(:,:,:)
        integer, intent(in) :: i, j, k
        real :: sum4
        sum4 = x(i,j,k) + x(i+1,j,k) + x(i,j+1,k) + x(i+1,j+1,k)
    end function kface

    pure function iface_r(x, r, i, j, k) result(sum4)
        implicit none
        real, intent(in) :: x(:,:,:), r(:,:,:)
        integer, intent(in) :: i, j, k
        real :: sum4
        sum4 = x(i,j,k)/r(i,j,k) + x(i,j+1,k)/r(i,j+1,k) + &
               x(i,j,k+1)/r(i,j,k+1) + x(i,j+1,k+1)/r(i,j+1,k+1)
    end function iface_r

    pure function jface_r(x, r, i, j, k) result(sum4)
        implicit none
        real, intent(in) :: x(:,:,:), r(:,:,:)
        integer, intent(in) :: i, j, k
        real :: sum4
        sum4 = x(i,j,k)/r(i,j,k) + x(i+1,j,k)/r(i+1,j,k) + &
               x(i,j,k+1)/r(i,j,k+1) + x(i+1,j,k+1)/r(i+1,j,k+1)
    end function jface_r

    pure function kface_r(x, r, i, j, k) result(sum4)
        implicit none
        real, intent(in) :: x(:,:,:), r(:,:,:)
        integer, intent(in) :: i, j, k
        real :: sum4
        sum4 = x(i,j,k)/r(i,j,k) + x(i+1,j,k)/r(i+1,j,k) + &
               x(i,j+1,k)/r(i,j+1,k) + x(i+1,j+1,k)/r(i+1,j+1,k)
    end function kface_r

    pure function grad_cell(x, r, rc, dAi, dAj, dAk, inv_vol, i, j, k, ni, nj, nk) result(grad)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: x(ni,nj,nk), r(ni,nj,nk)
        real, intent(in) :: rc
        real, intent(in) :: dAi(3, ni, nj-1, nk-1), dAj(3, ni-1, nj, nk-1), dAk(3, ni-1, nj-1, nk)
        real, intent(in) :: inv_vol
        integer, intent(in) :: i, j, k
        real :: grad(3)
        real :: xi_lo, xi_hi, xj_lo, xj_hi, xk_lo, xk_hi

        xi_lo  = iface(x,i,j,k)
        xi_hi  = iface(x,i+1,j,k)
        xj_lo  = jface(x,i,j,k)
        xj_hi  = jface(x,i,j+1,k)
        xk_lo  = kface(x,i,j,k)
        xk_hi  = kface(x,i,j,k+1)

        grad(1) = -(xi_lo  * dAi(1,i,j,k) - xi_hi  * dAi(1,i+1,j,k) &
                  + xj_lo  * dAj(1,i,j,k) - xj_hi  * dAj(1,i,j+1,k) &
                  + xk_lo  * dAk(1,i,j,k) - xk_hi  * dAk(1,i,j,k+1))

        grad(3) = -(xi_lo  * dAi(3,i,j,k) - xi_hi  * dAi(3,i+1,j,k) &
                  + xj_lo  * dAj(3,i,j,k) - xj_hi  * dAj(3,i,j+1,k) &
                  + xk_lo  * dAk(3,i,j,k) - xk_hi  * dAk(3,i,j,k+1))

        xi_lo = iface_r(x,r,i,j,k)
        xi_hi = iface_r(x,r,i+1,j,k)
        xj_lo = jface_r(x,r,i,j,k)
        xj_hi = jface_r(x,r,i,j+1,k)
        xk_lo = kface_r(x,r,i,j,k)
        xk_hi = kface_r(x,r,i,j,k+1)

        grad(2) = -(xi_lo * dAi(2,i,j,k) - xi_hi * dAi(2,i+1,j,k) &
                  + xj_lo * dAj(2,i,j,k) - xj_hi * dAj(2,i,j+1,k) &
                  + xk_lo * dAk(2,i,j,k) - xk_hi * dAk(2,i,j,k+1)) * rc

        grad = grad * inv_vol
    end function grad_cell

end subroutine grad
