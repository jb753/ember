! Block setter operations for efficient state updates

subroutine set_rho_u_vxrt_write(rho, u, vx, vr, vt, r, out, ni, nj, nk)
    ! Fused counterpart of Block.set_rho_u_Vxrt_nd's numpy body: computes the
    ! five conserved columns from non-dimensional state and velocity in one
    ! pass, into a caller-owned contiguous buffer (Block.scratch). The
    ! caller still has to copy the result into Block._data itself -- that
    ! target is a non-contiguous slice for a patch's block_view, which f2py
    ! refuses to accept directly as an intent(inout) argument -- but doing
    ! the arithmetic in one fused pass rather than the numpy chain is still a
    ! net win even after that copy (see block.py's set_rho_u_Vxrt_nd).
    !
    ! rho, u, vx, vr, vt, r : (ni,nj,nk)   non-dimensional state and velocity
    ! out                   : (ni,nj,nk,5) out: [rho, rhoVx, rhoVr, rhorVt, rhoe]

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: rho(ni,nj,nk), u(ni,nj,nk)
    real, intent(in) :: vx(ni,nj,nk), vr(ni,nj,nk), vt(ni,nj,nk), r(ni,nj,nk)
    real, intent(inout) :: out(ni,nj,nk,5)

    integer :: i, j, k
    real :: e

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                e = 0.5 * (vx(i,j,k)**2 + vr(i,j,k)**2 + vt(i,j,k)**2) + u(i,j,k)
                out(i,j,k,1) = rho(i,j,k)
                out(i,j,k,2) = rho(i,j,k) * vx(i,j,k)
                out(i,j,k,3) = rho(i,j,k) * vr(i,j,k)
                out(i,j,k,4) = rho(i,j,k) * r(i,j,k) * vt(i,j,k)
                out(i,j,k,5) = rho(i,j,k) * e
            end do
        end do
    end do

end subroutine set_rho_u_vxrt_write
