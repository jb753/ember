! Signed distance to a piecewise-linear curve in the meridional plane

subroutine signed_distance(xr, xq, d, nseg, npt)
    ! Signed distance from every query point to the nearest of the nseg-1
    ! segments joining consecutive nodes of the polyline xr.
    !
    ! Components are the slowest axis of xr and xq, matching the Fortran
    ! ordering of ember's nodal arrays, so an (ni, nj, nk, 2) coordinate array
    ! reshapes to (npt, 2) for this kernel without a copy, and the x and r
    ! components arrive as two separate contiguous streams.
    !
    ! The point loop is innermost and tiled: with the segment loop outside it,
    ! the running minimum and the tile of query points stay in cache across
    ! every segment, so the points are read once rather than once per segment.
    ! The update is a merge() rather than an if, because with a branch here
    ! gfortran emits scalar code and the kernel runs 5x slower.  Vectorising it
    ! also needs the -ffast-math that the package's -Ofast implies; under a
    ! plain -O3 this form is no faster than the naive one.

    implicit none

    integer, intent(in) :: nseg, npt
    double precision, intent(in) :: xr(nseg, 2), xq(npt, 2)
    double precision, intent(out) :: d(npt)

    ! Points per tile.  2048 doubles of running minimum plus the two coordinate
    ! streams is 48 kB, which stays inside a typical 32-48 kB L1 with the
    ! streams themselves evicting rather than the minimum.
    integer, parameter :: NTILE = 2048

    integer :: i, s, i0, i1
    double precision :: bx, br, ax, ar, Lsq, h, px, pr, cross, di, sgn

    do i0 = 1, npt, NTILE
        i1 = min(i0 + NTILE - 1, npt)

        ! With no segments at all this sentinel is what the caller sees, but
        ! the caller screens that case out before reaching the kernel.
        d(i0:i1) = huge(1.0d0)

        do s = 1, nseg - 1
            bx = xr(s + 1, 1) - xr(s, 1)
            br = xr(s + 1, 2) - xr(s, 2)

            ! Floor the squared length so a repeated node cannot divide by zero
            Lsq = max(bx * bx + br * br, 1.0d-9)

            do i = i0, i1
                ax = xq(i, 1) - xr(s, 1)
                ar = xq(i, 2) - xr(s, 2)

                ! Fraction along the segment of the projected point, clamped to
                ! the segment so the distance is to the segment, not to its line
                h = min(max((ax * bx + ar * br) / Lsq, 0.0d0), 1.0d0)

                ! Perpendicular from the segment to the point
                px = ax - bx * h
                pr = ar - br * h

                di = sqrt(px * px + pr * pr)

                ! Sign from the side of the segment the point falls on.  The
                ! normal to (bx, br) is (-br, bx), so the sign is that of
                ! dot((px, pr), (-br, bx)).
                cross = pr * bx - px * br

                ! A zero cross product signs the distance zero rather than
                ! picking a side, reproducing the numpy reference exactly:
                ! sign(di, cross) would return +di instead.  This matters for a
                ! point beyond a segment end and exactly on its axis, where the
                ! perpendicular is parallel to the segment and the reference
                ! returns zero for a point that is not on the curve.
                sgn = merge(0.0d0, sign(1.0d0, cross), cross == 0.0d0)

                ! Compare unsigned, store signed, as the reference does
                d(i) = merge(di * sgn, d(i), di < abs(d(i)))
            end do
        end do
    end do

end subroutine signed_distance
