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


! Copy tau/q from one block face buffer's OWNED layer into another's HALO
! layer, at matched index lists. This is the whole periodic seam exchange for
! the viscous pass: the boundary tau/q the two kernels hand each other is
! O(surface), so this moves it, and nothing walks a volume.
!
! Two properties follow from the face buffers carrying their owned and halo
! values in separate layers:
!
!   * It is a copy, not a swap. src is read only at layer 1 and dst written
!     only at layer 2, so the two never collide and no temporary is needed --
!     not even when a face is paired to itself, which is what a block periodic
!     to itself in theta does.
!   * It moves one direction only. PeriodicCommunicator prunes its pairs to
!     one key per pair, so the caller makes this call twice per pair, once
!     each way.
!
! Indices are Fortran 1-based cell coordinates within the face: (j,k) on an
! i face, (i,k) on a j face, (i,j) on a k face. Which face each buffer is, and
! therefore how its two spatial extents are named, is settled by the caller.
subroutine copy_faces_by_ij(dst, src, idx_dst, idx_src, &
        na_d, nb_d, na_s, nb_s, npt, nv)

    integer, intent(in) :: na_d, nb_d, na_s, nb_s, npt, nv
    real, intent(inout) :: dst(na_d, nv, nb_d, 2)
    real, intent(in) :: src(na_s, nv, nb_s, 2)
    integer*2, intent(in) :: idx_dst(npt, 2)   ! (a, b) into dst, 1-based
    integer*2, intent(in) :: idx_src(npt, 2)   ! (a, b) into src, 1-based

    integer :: ipt, c

    do ipt = 1, npt
        do c = 1, nv
            dst(idx_dst(ipt,1), c, idx_dst(ipt,2), 2) = &
                src(idx_src(ipt,1), c, idx_src(ipt,2), 1)
        end do
    end do

end subroutine
