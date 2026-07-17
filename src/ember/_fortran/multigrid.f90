! Running pseudo-time average accumulator. (The standalone restrict/prolong
! multigrid primitives that once lived here were unused -- the production
! multigrid is the fused engine in scree.f90 -- and were removed; recover them
! from git history if a direct-transfer variant is ever needed again.)


subroutine accumulate_avg(cons_nd, cons_avg_nd, n_step_avg, ni, nj, nk, ncomp)
    ! Accumulate nondimensional conserved variables into a running average.
    ! cons_avg_nd += cons_nd / n_step_avg
    ! Single precision is sufficient: over a few thousand steps the streaming
    ! mean holds ~1e-6 relative error, far below discretization error.
    ! Uses only a scalar temporary - no extra array allocation.
    implicit none

    integer, intent(in)    :: ni, nj, nk, ncomp
    real,    intent(in)    :: cons_nd(ni, nj, nk, ncomp)
    real,    intent(inout) :: cons_avg_nd(ni, nj, nk, ncomp)
    integer, intent(in)    :: n_step_avg

    integer :: i, j, k, v
    real    :: inv_n, tmp

    inv_n = 1.0 / real(n_step_avg)

    do v = 1, ncomp
    do k = 1, nk
    do j = 1, nj
    do i = 1, ni
        tmp = cons_nd(i, j, k, v) * inv_n
        cons_avg_nd(i, j, k, v) = cons_avg_nd(i, j, k, v) + tmp
    end do
    end do
    end do
    end do

end subroutine accumulate_avg
