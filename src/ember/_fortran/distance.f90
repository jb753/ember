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
    !
    ! The running minimum is of the squared distance, carrying the winning
    ! segment's sign, so the square root is taken once per point rather than
    ! once per segment.

    implicit none

    integer, intent(in) :: nseg, npt

    ! Single precision throughout, as ember stores coordinates, which is all
    ! the precision a cut can have; it also doubles the vector width.
    real, intent(in) :: xr(nseg, 2), xq(npt, 2)
    real, intent(out) :: d(npt)

    ! Points per tile.  2048 reals of running minimum plus the two coordinate
    ! streams is 24 kB, which stays inside a typical 32-48 kB L1 with the
    ! streams themselves evicting rather than the minimum.
    integer, parameter :: NTILE = 2048

    integer :: i, s, i0, i1
    real :: bx, br, ax, ar, rLsq, h, px, pr, cross, dsq, sgn

    do i0 = 1, npt, NTILE
        i1 = min(i0 + NTILE - 1, npt)

        ! Squared distance to the nearest segment so far, signed as that
        ! segment puts the point. With no segments at all this sentinel is
        ! what the caller sees, but the caller screens that case out before
        ! reaching the kernel.
        d(i0:i1) = huge(1.0)

        do s = 1, nseg - 1
            bx = xr(s + 1, 1) - xr(s, 1)
            br = xr(s + 1, 2) - xr(s, 2)

            ! Floor the squared length so a repeated node cannot divide by zero
            rLsq = 1.0 / max(bx * bx + br * br, 1.0e-9)

            do i = i0, i1
                ax = xq(i, 1) - xr(s, 1)
                ar = xq(i, 2) - xr(s, 2)

                ! Fraction along the segment of the projected point, clamped to
                ! the segment so the distance is to the segment, not to its line
                h = min(max((ax * bx + ar * br) * rLsq, 0.0), 1.0)

                ! Perpendicular from the segment to the point
                px = ax - bx * h
                pr = ar - br * h

                dsq = px * px + pr * pr

                ! Sign from the side of the segment the point falls on.  The
                ! normal to (bx, br) is (-br, bx), so the sign is that of
                ! dot((px, pr), (-br, bx)).
                cross = pr * bx - px * br

                ! A point beyond a segment end and on its axis has no side: the
                ! perpendicular is parallel to the segment and the cross product
                ! is zero.  Under -Ofast it comes out as +/- round-off instead,
                ! whose sign differs between the vectorised loop body and the
                ! scalar remainder, so equal points landed on opposite sides by
                ! their place in the array.  Anything within 1e-6 of |a| |b|,
                ! a few times the single precision round-off and compared in
                ! squares to keep sqrt out, is therefore on the axis and takes
                ! the positive side.  Not a zero sign, which the numpy
                ! reference gives: stored as zero, the running minimum below
                ! can then never be beaten by a nearer segment.
                sgn = merge(1.0, sign(1.0, cross), &
                            cross * cross * rLsq <= 1.0e-12 * (ax * ax + ar * ar))

                ! Compare unsigned, store signed, as the reference does
                d(i) = merge(dsq * sgn, d(i), dsq < abs(d(i)))
            end do
        end do

        do i = i0, i1
            d(i) = sign(sqrt(abs(d(i))), d(i))
        end do
    end do

end subroutine signed_distance
