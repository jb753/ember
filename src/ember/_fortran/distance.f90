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


subroutine cut_distance(xr, xq, d, keep, nseg, ni, nj, nk)
    ! Signed distance from every node of a block to a cut curve, exact only
    ! near the curve: the field the cutting kernels take their zero level from.
    !
    ! The cut is the polyline xr carried on past both ends as straight rays,
    ! the first segment's line backwards and the last's forwards, so a curve
    ! of one segment is exactly the line through it. The curve and its rays
    ! split the plane in two, and the sign says which side a node is on:
    ! positive to the left of the direction the curve runs, as in
    ! signed_distance above. That needs the curve and rays not to cross
    ! themselves.
    !
    ! The cutting kernels use the distance only at the two ends of a grid
    ! edge that the cut crosses, and need nothing but its sign elsewhere.
    ! Searching every segment for the nearest is what costs, so it is done
    ! only where it is used. The block is split into tiles, each with a
    ! bounding box over its nodes and their neighbours one node on in every
    ! direction. Where no piece of the cut enters that box, no edge from a
    ! node of the tile crosses the cut, and the tile takes the sign of its
    ! first node, found by counting crossings along a ray from it, and the
    ! magnitude huge(1.0), larger than any distance in the band. Where a
    ! piece does enter, every node of the tile gets its exact distance, from
    ! a search over just the pieces that could be nearest to the tile, so
    ! both ends of an edge that crosses the cut are always exact.
    !
    ! Near the cut, the nearest piece's side of the point is the side of the
    ! whole cut, so the two ways of finding the sign agree. Away from it,
    ! taking the sign from the nearest piece, as signed_distance does, would
    ! flip it wherever the nearest piece switches to one that puts the point
    ! on its other side, making a sign change where the distance never
    ! passes through zero; counting crossings has no such flips.
    !
    ! Components are the slowest axis of xr and xq, as in signed_distance.

    implicit none

    integer, intent(in) :: nseg, ni, nj, nk

    real, intent(in) :: xr(nseg, 2), xq(ni, nj, nk, 2)
    real, intent(out) :: d(ni, nj, nk)

    ! Room for the pieces one tile keeps, supplied by the caller
    integer, intent(inout) :: keep(nseg + 1)

    ! Nodes per tile along each axis. A smaller tile hugs the cut more
    ! closely, so fewer nodes get the exact search, at the cost of more
    ! tiles to screen and wider halos relative to the tile.
    integer, parameter :: TI = 32, TJ = 4, TK = 4

    ! Length a ray stands in for, beyond any distance in the plane but with
    ! room to square it in single precision
    real, parameter :: BIG = 1.0e18

    ! The pieces of the cut: 0 the ray back from the first node, 1 to
    ! nseg - 1 the segments, nseg the ray on from the last node. Each runs
    ! from o + hmin b to o + hmax b, b along the direction of the curve, so
    ! that its left is the cut's left.
    integer :: p, n, n_keep

    integer :: i, j, k, i0, i1, j0, j1, k0, k1
    real :: x_lo, x_hi, r_lo, r_hi, d_max, d_top, gx, gr
    real :: ox, orr, bx, br, hmin, hmax, rLsq, ex, er, ax, ar, h, px, pr, dsq, cross, sgn
    real :: s_even, a0, a1, alpha, theta, ux0, ur0, ux1, ur1
    logical :: touch

    real, parameter :: PI = 3.14159265

    ! Directions the curve runs in along its two rays
    call unit(xr(2, 1) - xr(1, 1), xr(2, 2) - xr(1, 2), ux0, ur0)
    call unit(xr(nseg, 1) - xr(nseg - 1, 1), xr(nseg, 2) - xr(nseg - 1, 2), ux1, ur1)

    ! Far off, the left of the cut is the arc of directions turning
    ! anticlockwise from the forward ray to the backward one. A ray cast
    ! through the middle of the larger of the two arcs ends on a known side,
    ! well clear of both rays: left if it is the left arc, and right if not.
    a0 = atan2(-ur0, -ux0)
    a1 = atan2(ur1, ux1)
    alpha = modulo(a0 - a1, 2.0 * PI)
    if (alpha >= PI) then
        theta = a1 + 0.5 * alpha
        s_even = 1.0
    else
        theta = a0 + 0.5 * (2.0 * PI - alpha)
        s_even = -1.0
    end if
    ex = cos(theta)
    er = sin(theta)

    do k0 = 1, nk, TK
    k1 = min(k0 + TK - 1, nk)
    do j0 = 1, nj, TJ
    j1 = min(j0 + TJ - 1, nj)
    do i0 = 1, ni, TI
        i1 = min(i0 + TI - 1, ni)

        ! Box over the tile and the nodes one on from it every way, so that
        ! it holds every grid edge from a node of the tile. Four reductions in
        ! one loop over i, which unlike minval and maxval here vectorises.
        x_lo = huge(1.0)
        x_hi = -huge(1.0)
        r_lo = huge(1.0)
        r_hi = -huge(1.0)
        do k = max(k0 - 1, 1), min(k1 + 1, nk)
            do j = max(j0 - 1, 1), min(j1 + 1, nj)
                do i = max(i0 - 1, 1), min(i1 + 1, ni)
                    x_lo = min(x_lo, xq(i, j, k, 1))
                    x_hi = max(x_hi, xq(i, j, k, 1))
                    r_lo = min(r_lo, xq(i, j, k, 2))
                    r_hi = max(r_hi, xq(i, j, k, 2))
                end do
            end do
        end do

        ! Whether any piece of the cut enters the box: its own box overlaps,
        ! and the box's corners are not all to one side of its line
        touch = .false.
        do p = 0, nseg
            call piece(p, ox, orr, bx, br, hmin, hmax)
            if (min(ox + bx * hmin, ox + bx * hmax) > x_hi .or. max(ox + bx * hmin, ox + bx * hmax) < x_lo) cycle
            if (min(orr + br * hmin, orr + br * hmax) > r_hi .or. max(orr + br * hmin, orr + br * hmax) < r_lo) cycle
            gx = bx * (r_lo - orr) - br * (x_lo - ox)
            gr = bx * (r_hi - orr) - br * (x_hi - ox)
            ax = bx * (r_hi - orr) - br * (x_lo - ox)
            ar = bx * (r_lo - orr) - br * (x_hi - ox)
            if (min(gx, gr, ax, ar) <= 0.0 .and. max(gx, gr, ax, ar) >= 0.0) then
                touch = .true.
                exit
            end if
        end do

        if (.not. touch) then
            sgn = side(xq(i0, j0, k0, 1), xq(i0, j0, k0, 2))
            d(i0:i1, j0:j1, k0:k1) = sgn * huge(1.0)
            cycle
        end if

        ! No node of the tile is further than d_top from its nearest piece,
        ! the least over pieces of the furthest corner of the box, since
        ! distance to a piece is convex in the point
        d_top = huge(1.0)
        do p = 0, nseg
            d_max = max(dist_sq(x_lo, r_lo, p), dist_sq(x_hi, r_lo, p), &
                        dist_sq(x_lo, r_hi, p), dist_sq(x_hi, r_hi, p))
            d_top = min(d_top, d_max)
        end do

        ! So a piece further than that from the whole box is never nearest.
        ! The margin covers round-off between this bound and the distances
        ! below, so no piece that could tie is dropped.
        n_keep = 0
        do p = 0, nseg
            call piece(p, ox, orr, bx, br, hmin, hmax)
            gx = max(0.0, min(ox + bx * hmin, ox + bx * hmax) - x_hi, x_lo - max(ox + bx * hmin, ox + bx * hmax))
            gr = max(0.0, min(orr + br * hmin, orr + br * hmax) - r_hi, r_lo - max(orr + br * hmin, orr + br * hmax))
            if (gx * gx + gr * gr <= d_top * 1.001) then
                n_keep = n_keep + 1
                keep(n_keep) = p
            end if
        end do

        ! Nearest kept piece, as in signed_distance, with the fraction along
        ! each piece clamped to its own range
        d(i0:i1, j0:j1, k0:k1) = huge(1.0)
        do n = 1, n_keep
            call piece(keep(n), ox, orr, bx, br, hmin, hmax)
            rLsq = 1.0 / max(bx * bx + br * br, 1.0e-9)
            do k = k0, k1
                do j = j0, j1
                    do i = i0, i1
                        ax = xq(i, j, k, 1) - ox
                        ar = xq(i, j, k, 2) - orr
                        h = min(max((ax * bx + ar * br) * rLsq, hmin), hmax)
                        px = ax - bx * h
                        pr = ar - br * h
                        dsq = px * px + pr * pr
                        cross = pr * bx - px * br
                        sgn = merge(1.0, sign(1.0, cross), &
                                    cross * cross * rLsq <= 1.0e-12 * (ax * ax + ar * ar))
                        d(i, j, k) = merge(dsq * sgn, d(i, j, k), dsq < abs(d(i, j, k)))
                    end do
                end do
            end do
        end do

        do k = k0, k1
            do j = j0, j1
                do i = i0, i1
                    d(i, j, k) = sign(sqrt(abs(d(i, j, k))), d(i, j, k))
                end do
            end do
        end do
    end do
    end do
    end do

contains

    subroutine unit(vx, vr, ux, ur)
        ! Unit vector along (vx, vr), floored against a repeated node
        real, intent(in) :: vx, vr
        real, intent(out) :: ux, ur
        real :: length
        length = sqrt(max(vx * vx + vr * vr, 1.0e-18))
        ux = vx / length
        ur = vr / length
    end subroutine unit

    subroutine piece(q, qx, qr, qbx, qbr, qhmin, qhmax)
        ! Origin, direction and range of piece q of the cut
        integer, intent(in) :: q
        real, intent(out) :: qx, qr, qbx, qbr, qhmin, qhmax
        if (q == 0) then
            qx = xr(1, 1)
            qr = xr(1, 2)
            qbx = ux0
            qbr = ur0
            qhmin = -BIG
            qhmax = 0.0
        else if (q == nseg) then
            qx = xr(nseg, 1)
            qr = xr(nseg, 2)
            qbx = ux1
            qbr = ur1
            qhmin = 0.0
            qhmax = BIG
        else
            qx = xr(q, 1)
            qr = xr(q, 2)
            qbx = xr(q + 1, 1) - xr(q, 1)
            qbr = xr(q + 1, 2) - xr(q, 2)
            qhmin = 0.0
            qhmax = 1.0
        end if
    end subroutine piece

    real function dist_sq(qx, qr, q)
        ! Squared distance from the point (qx, qr) to piece q
        real, intent(in) :: qx, qr
        integer, intent(in) :: q
        real :: sx, sr, sbx, sbr, shmin, shmax, t
        call piece(q, sx, sr, sbx, sbr, shmin, shmax)
        t = ((qx - sx) * sbx + (qr - sr) * sbr) / max(sbx * sbx + sbr * sbr, 1.0e-9)
        t = min(max(t, shmin), shmax)
        dist_sq = (qx - sx - sbx * t)**2 + (qr - sr - sbr * t)**2
    end function dist_sq

    real function side(qx, qr)
        ! +1 left of the cut, -1 right, by the parity of the crossings of
        ! the cut along the ray from (qx, qr) in direction (ex, er). A
        ! piece crosses the ray's line where its ends lie on opposite sides
        ! of it, an end on the line counting as on the left, so a ray
        ! through a node crosses one of the two pieces that meet there.
        real, intent(in) :: qx, qr
        integer :: q, n_cross
        real :: sx, sr, sbx, sbr, shmin, shmax, s_a, s_b, denom, t
        n_cross = 0
        do q = 0, nseg
            call piece(q, sx, sr, sbx, sbr, shmin, shmax)
            s_a = ex * (sr + sbr * shmin - qr) - er * (sx + sbx * shmin - qx)
            s_b = ex * (sr + sbr * shmax - qr) - er * (sx + sbx * shmax - qx)
            if ((s_a >= 0.0) .eqv. (s_b >= 0.0)) cycle
            denom = ex * sbr - er * sbx
            if (denom == 0.0) cycle
            ! Where along the ray the piece's line is met
            t = ((sx - qx) * sbr - (sr - qr) * sbx) / denom
            if (t > 0.0) n_cross = n_cross + 1
        end do
        side = merge(s_even, -s_even, modulo(n_cross, 2) == 0)
    end function side

end subroutine cut_distance
