! Multigrid operations for coarse/fine grid transfers


! restrict_cell: components-last layout with n outermost loop.
! Each n-slice is a contiguous (ni-1,nj-1,nk-1) array; the spatial
! loops then do stride-1 reads/writes within that slice.
!
subroutine restrict_cell(cell_fine, cell_coarse, &
                            ni_fine, nj_fine, nk_fine, ni_coarse, nj_coarse, nk_coarse, ncomp)

    implicit none

    integer, intent(in) :: ni_fine, nj_fine, nk_fine
    integer, intent(in) :: ni_coarse, nj_coarse, nk_coarse
    integer, intent(in) :: ncomp

    real, intent(in)    :: cell_fine  (ni_fine-1,   nj_fine-1,   nk_fine-1,   ncomp)
    real, intent(inout) :: cell_coarse(ni_coarse-1, nj_coarse-1, nk_coarse-1, ncomp)

    integer :: ic, jc, kc, n
    integer :: ifs, jfs, kfs

    do n = 1, ncomp
        do kc = 1, nk_coarse-1
        do jc = 1, nj_coarse-1
        do ic = 1, ni_coarse-1
            ifs = 2*ic - 1
            jfs = 2*jc - 1
            kfs = 2*kc - 1
            cell_coarse(ic, jc, kc, n) = cell_coarse(ic, jc, kc, n) + &
                cell_fine(ifs,   jfs,   kfs,   n) + &
                cell_fine(ifs+1, jfs,   kfs,   n) + &
                cell_fine(ifs,   jfs+1, kfs,   n) + &
                cell_fine(ifs+1, jfs+1, kfs,   n) + &
                cell_fine(ifs,   jfs,   kfs+1, n) + &
                cell_fine(ifs+1, jfs,   kfs+1, n) + &
                cell_fine(ifs,   jfs+1, kfs+1, n) + &
                cell_fine(ifs+1, jfs+1, kfs+1, n)
        end do
        end do
        end do
    end do

end subroutine restrict_cell


subroutine restrict_node(conserved_fine, conserved_coarse, &
                         ni_fine, nj_fine, nk_fine, ni_coarse, nj_coarse, nk_coarse, ncomp)
    ! Simple restriction from fine to coarse using direct node sampling
    !
    ! This is the simplest possible restriction operator - just copy every other
    ! node from the fine grid to the coarse grid. Equivalent to numpy slicing [::2, ::2, ::2, :].
    !
    ! Parameters
    ! ----------
    ! conserved_fine : real(ni_fine, nj_fine, nk_fine, ncomp)
    !     Fine grid conserved variables at nodes (input)
    ! conserved_coarse : real(ni_coarse, nj_coarse, nk_coarse, ncomp)
    !     Coarse grid conserved variables at nodes (output)
    ! ni_fine, nj_fine, nk_fine : integer
    !     Fine grid dimensions (number of nodes)
    ! ni_coarse, nj_coarse, nk_coarse : integer
    !     Coarse grid dimensions (number of nodes)
    ! ncomp : integer
    !     Number of conserved components (typically 5)

    implicit none

    integer, intent(in) :: ni_fine, nj_fine, nk_fine
    integer, intent(in) :: ni_coarse, nj_coarse, nk_coarse
    integer, intent(in) :: ncomp

    real, intent(in) :: conserved_fine(ni_fine, nj_fine, nk_fine, ncomp)
    real, intent(inout) :: conserved_coarse(ni_coarse, nj_coarse, nk_coarse, ncomp)

    integer :: ic, jc, kc, n, if, jf, kf

    do n = 1, ncomp
    do kc = 1, nk_coarse
        do jc = 1, nj_coarse
            do ic = 1, ni_coarse
                ! Map coarse index to fine index (every other node)
                ! In Fortran 1-based indexing: ic=1 -> if=1, ic=2 -> if=3, ic=3 -> if=5
                if = 2*ic - 1
                jf = 2*jc - 1
                kf = 2*kc - 1

                    conserved_coarse(ic, jc, kc, n) = conserved_fine(if, jf, kf, n)
            end do
        end do
    end do
    end do

end subroutine restrict_node


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


subroutine prolongate_correction(correction, fine, fac_mgrid, &
                                 ni_coarse, nj_coarse, nk_coarse, &
                                 ni_fine, nj_fine, nk_fine, ncomp)
    ! Trilinearly interpolate a pre-built coarse-grid correction onto the fine
    ! grid, scale by fac_mgrid, and add to the fine field.
    !
    ! The correction is built and smoothed at the Python layer before this call;
    ! pressure and CFL for that smoothing are materialised from the coarse
    ! solution first, so the correction may overwrite the coarse conserved buffer.

    implicit none
    integer, intent(in) :: ni_coarse, nj_coarse, nk_coarse
    integer, intent(in) :: ni_fine, nj_fine, nk_fine, ncomp
    real, intent(in)    :: correction(ni_coarse, nj_coarse, nk_coarse, ncomp)
    real, intent(inout) :: fine(ni_fine, nj_fine, nk_fine, ncomp)
    real, intent(in) :: fac_mgrid

    integer :: ic, jc, kc, ic1, jc1, kc1, if, jf, kf, n
    real :: c000, c100, c010, c110, c001, c101, c011, c111
    real :: wi, wj, wk

    ! Trilinear interpolation of correction to ALL fine grid nodes, scaled by
    ! fac_mgrid.  At each upper boundary (e.g. if=ni_fine), ic=ni_coarse and
    ! wi=0, so every stencil term referencing ic1 vanishes; the clamp just
    ! keeps the index valid.
    do n = 1, ncomp
        do kf = 1, nk_fine
            do jf = 1, nj_fine
                do if = 1, ni_fine
                    ic = (if + 1) / 2
                    jc = (jf + 1) / 2
                    kc = (kf + 1) / 2
                    wi = 0.5 * (if - (2*ic - 1))
                    wj = 0.5 * (jf - (2*jc - 1))
                    wk = 0.5 * (kf - (2*kc - 1))
                    ic1 = min(ic + 1, ni_coarse)
                    jc1 = min(jc + 1, nj_coarse)
                    kc1 = min(kc + 1, nk_coarse)

                    c000 = correction(ic,  jc,  kc,  n)
                    c100 = correction(ic1, jc,  kc,  n)
                    c010 = correction(ic,  jc1, kc,  n)
                    c110 = correction(ic1, jc1, kc,  n)
                    c001 = correction(ic,  jc,  kc1, n)
                    c101 = correction(ic1, jc,  kc1, n)
                    c011 = correction(ic,  jc1, kc1, n)
                    c111 = correction(ic1, jc1, kc1, n)

                    fine(if, jf, kf, n) = fine(if, jf, kf, n) + fac_mgrid * ( &
                        (1.0-wi)*(1.0-wj)*(1.0-wk)*c000 + &
                        wi*(1.0-wj)*(1.0-wk)*c100 + &
                        (1.0-wi)*wj*(1.0-wk)*c010 + &
                        wi*wj*(1.0-wk)*c110 + &
                        (1.0-wi)*(1.0-wj)*wk*c001 + &
                        wi*(1.0-wj)*wk*c101 + &
                        (1.0-wi)*wj*wk*c011 + &
                        wi*wj*wk*c111)
                end do
            end do
        end do
    end do

end subroutine prolongate_correction
