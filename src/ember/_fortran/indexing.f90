! Functions for accessing 4D arrays using unstructured lists of ijk

! Retrieve data from the 4D array x at the given list of ijk
! Return in an unstructured list
! Given two 4D arrays and lists of ijk indexes into each,
! average the variables at corresponding indexes and assign
! back to both the original arrays
subroutine average_by_ijk(x1, x2, ijk1, ijk2, rf, ni1, nj1, nk1, ni2, nj2, nk2, npt, nv)

    integer, intent (in)  :: npt
    integer, intent (in)  :: ni1
    integer, intent (in)  :: nj1
    integer, intent (in)  :: nk1
    integer, intent (in)  :: ni2
    integer, intent (in)  :: nj2
    integer, intent (in)  :: nk2
    integer, intent (in) :: nv
    real, intent (in) :: rf

    real, intent (inout) :: x1(ni1, nj1, nk1, nv)
    real, intent (inout) :: x2(ni2, nj2, nk2, nv)
    integer*2, intent (in) :: ijk1(npt, 3)
    integer*2, intent (in) :: ijk2(npt, 3)

    integer :: ipt
    real :: avg(nv)


    integer :: i1
    integer :: j1
    integer :: k1

    integer :: i2
    integer :: j2
    integer :: k2

    ! If we have some points
    if (npt > 0) then
        ! Loop over all points
        do ipt = 1,npt

            ! Extract indices
            i1 = ijk1(ipt, 1)
            j1 = ijk1(ipt, 2)
            k1 = ijk1(ipt, 3)
            i2 = ijk2(ipt, 1)
            j2 = ijk2(ipt, 2)
            k2 = ijk2(ipt, 3)

            ! Get average
            avg = 0.5e0*(x1(i1, j1, k1, :) + x2(i2, j2, k2, :))
            x1(i1, j1, k1, :) = avg * rf + x1(i1, j1, k1, :)*(1.0e0-rf)
            x2(i2, j2, k2, :) = avg * rf + x2(i2, j2, k2, :)*(1.0e0-rf)

        end do
    end if

end subroutine


! Swap values between two 4D arrays at given lists of ijk indices.
! h1[hs1[ipt]] <-> h2[ec2[ipt]]  and  h2[hs2[ipt]] <-> h1[ec1[ipt]]
! i.e. write neighbour's owned edge-cell value into local halo slot.
! Uses a scalar tmp(nv) so no heap allocation occurs.
subroutine swap_by_ijk(h1, h2, hs1, ec1, hs2, ec2, &
        ni1, nj1, nk1, ni2, nj2, nk2, npt, nv)

    integer, intent(in) :: npt, nv
    integer, intent(in) :: ni1, nj1, nk1, ni2, nj2, nk2
    real, intent(inout) :: h1(ni1, nj1, nk1, nv)
    real, intent(inout) :: h2(ni2, nj2, nk2, nv)
    integer*2, intent(in) :: hs1(npt, 3)   ! halo slot indices into h1 (1-based)
    integer*2, intent(in) :: ec1(npt, 3)   ! owned edge-cell indices in h1 (1-based)
    integer*2, intent(in) :: hs2(npt, 3)   ! halo slot indices into h2 (1-based)
    integer*2, intent(in) :: ec2(npt, 3)   ! owned edge-cell indices in h2 (1-based)

    integer :: ipt
    real :: tmp(nv)

    do ipt = 1, npt
        tmp = h1(ec1(ipt,1), ec1(ipt,2), ec1(ipt,3), :)
        h1(hs1(ipt,1), hs1(ipt,2), hs1(ipt,3), :) = h2(ec2(ipt,1), ec2(ipt,2), ec2(ipt,3), :)
        h2(hs2(ipt,1), hs2(ipt,2), hs2(ipt,3), :) = tmp
    end do

end subroutine
