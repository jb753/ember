! Locating where a cut crosses a structured grid: marching cubes triangles
! and the first crossing along each grid line

subroutine marching_cubes( &
    a, b, na, nb, c0, vert_ijk, tritable, edge_ijk, lo, hi, frac, n_face, &
    ic_row, i_cut, ni, nj, nk, cap)
    ! Locate every triangle vertex a marching cubes cut produces, in one pass.
    !
    ! The cut is the zero level of the signed distance d = na a + nb b - c0,
    ! worked out at each node as it is needed rather than read from a field.
    ! A straight cut passes the meridional coordinates as a and b, with the
    ! unit normal to its line and the line's offset, so no distance field is
    ! ever built; any other passes a distance field already worked out as a,
    ! with na = 1 and nb = c0 = 0, which reproduce it exactly.
    !
    ! Each cell's corner sign pattern has one bit per corner, set where the
    ! distance is negative. Zero and all-ones are the uncut cells; any other
    ! pattern indexes the triangle table, whose triangles are written out as
    ! the cell is reached. Cells are walked i fastest, as the field is stored.
    !
    ! Each row of cells along i is done in three loops. The first finds every
    ! pattern in the row, a corner at a time so that each corner's offset is
    ! fixed across the loop over i, which reads the field contiguously and
    ! has no branch to stop it vectorising, and counts the cut cells. A row
    ! with none, as most are, stops there. Otherwise the second lists the cut
    ! cells without branching on each, which would mispredict, and the third
    ! visits just those.
    !
    ! A vertex sits on a cell edge, between two nodes of the grid, so it is
    ! reported as the pair of nodes it lies between, as zero-based indices
    ! into the field flattened i fastest, and the fraction of the way along
    ! that the distance vanishes. Each pair runs from the lower node to the
    ! higher, so cells sharing an edge report it identically.
    !
    ! How many triangles a cut makes is not known until the grid has been
    ! walked, so the caller supplies room for cap of them. Triangles past
    ! that are counted but not written, and n_face returns the full count:
    ! larger than cap, the caller makes room for them all and calls again.
    !
    ! Interpolating the data itself is left to the caller, which keeps this
    ! free of the number of variables, of the bulk data array, and of the
    ! precision that array happens to be stored in. Every table is supplied
    ! by the caller rather than repeated here, so that the Python and Fortran
    ! sides cannot drift apart.

    implicit none

    integer, intent(in) :: ni, nj, nk, cap

    ! The two fields and the coefficients the distance is formed from, in the
    ! single precision of the coordinates, as is everything worked out here
    real, intent(in) :: a(ni, nj, nk), b(ni, nj, nk)
    real, intent(in) :: na, nb, c0

    ! Zero-based (i, j, k) offset of each of the 8 corners of a cell, which
    ! fixes the order of the bits in the pattern
    integer, intent(in) :: vert_ijk(8, 3)

    ! Triangles per pattern, as edge numbers in threes, ending at -1
    integer, intent(in) :: tritable(256, 16)

    ! The two corner offsets of each of the 12 cell edges
    integer, intent(in) :: edge_ijk(12, 2, 3)

    ! Nodes each vertex of each triangle lies between, and how far along
    integer, intent(inout) :: lo(3, cap), hi(3, cap)
    real, intent(inout) :: frac(3, cap)

    ! Triangles the cut makes, written or not
    integer, intent(out) :: n_face

    ! Room for the patterns of one row of cells, and for the i of each cell
    ! in it that the cut crosses, supplied by the caller
    integer, intent(inout) :: ic_row(ni - 1), i_cut(ni - 1)

    integer :: i, j, k, c, ic, bit, row, n, v, e, t, n_cut, m
    integer :: oi, oj, ok
    integer :: ilo(3), ihi(3), swap(3)
    real :: d_lo, d_hi

    t = 0

    do k = 1, nk - 1
        do j = 1, nj - 1
            ! One bit per corner, set where the distance is negative
            ic_row = 0
            bit = 1
            do c = 1, 8
                oi = vert_ijk(c, 1)
                oj = j + vert_ijk(c, 2)
                ok = k + vert_ijk(c, 3)
                do i = 1, ni - 1
                    ic_row(i) = ic_row(i) + merge(bit, 0, &
                        na * a(i + oi, oj, ok) + nb * b(i + oi, oj, ok) - c0 < 0.0)
                end do
                bit = bit * 2
            end do

            ! Cells with corners on both sides of the cut
            n_cut = 0
            do i = 1, ni - 1
                n_cut = n_cut + merge(1, 0, ic_row(i) /= 0 .and. ic_row(i) /= 255)
            end do
            if (n_cut == 0) cycle

            ! List them. Every i is stored and only a cut one kept.
            n_cut = 0
            do i = 1, ni - 1
                i_cut(n_cut + 1) = i
                n_cut = n_cut + merge(1, 0, ic_row(i) /= 0 .and. ic_row(i) /= 255)
            end do

            do m = 1, n_cut
                i = i_cut(m)
                ic = ic_row(i)

                row = ic + 1
                do n = 1, 16, 3
                    ! A sentinel edge closes the pattern's list of triangles
                    if (tritable(row, n) < 0) exit

                    t = t + 1
                    if (t > cap) cycle

                    do v = 1, 3
                        e = tritable(row, n + v - 1) + 1

                        ! Zero-based node indices at the ends of the edge
                        ilo(1) = i - 1 + edge_ijk(e, 1, 1)
                        ilo(2) = j - 1 + edge_ijk(e, 1, 2)
                        ilo(3) = k - 1 + edge_ijk(e, 1, 3)
                        ihi(1) = i - 1 + edge_ijk(e, 2, 1)
                        ihi(2) = j - 1 + edge_ijk(e, 2, 2)
                        ihi(3) = k - 1 + edge_ijk(e, 2, 3)

                        ! Four of the twelve edges run against the index
                        ! direction. Turn them round, so the neighbouring cell
                        ! sharing the edge gives the same pair and fraction.
                        if (any(ihi < ilo)) then
                            swap = ilo
                            ilo = ihi
                            ihi = swap
                        end if

                        d_lo = na * a(ilo(1) + 1, ilo(2) + 1, ilo(3) + 1) &
                             + nb * b(ilo(1) + 1, ilo(2) + 1, ilo(3) + 1) - c0
                        d_hi = na * a(ihi(1) + 1, ihi(2) + 1, ihi(3) + 1) &
                             + nb * b(ihi(1) + 1, ihi(2) + 1, ihi(3) + 1) - c0

                        ! The cut crosses this edge, so one end is negative
                        ! and the fraction lies in [0, 1]. Formed from two
                        ! terms, the distance here can round differently from
                        ! the vectorised pattern loop's, so a node within
                        ! round-off of the cut can lose its sign: clamp, and
                        ! take the midpoint of an edge whose ends then agree.
                        if (d_hi == d_lo) then
                            frac(v, t) = 0.5
                        else
                            frac(v, t) = min(max(-d_lo / (d_hi - d_lo), 0.0), 1.0)
                        end if

                        lo(v, t) = ilo(1) + ni * (ilo(2) + nj * ilo(3))
                        hi(v, t) = ihi(1) + ni * (ihi(2) + nj * ihi(3))
                    end do
                end do
            end do
        end do
    end do

    n_face = t

end subroutine marching_cubes


subroutine first_j_crossing( &
    a, b, na, nb, c0, j_cut, frac, d_lo, n_cut, ni, nj, nk)
    ! Find where each grid line along j first crosses a cut, in one pass.
    !
    ! The cut is the zero level of the signed distance d = na a + nb b - c0,
    ! formed node by node exactly as in marching_cubes above, so a straight
    ! cut passes the meridional coordinates and its line's coefficients, and
    ! any other a distance field worked out already, with na = 1, nb = c0 = 0.
    !
    ! Every (i, k) line is walked along j to its first interval whose ends
    ! differ in sign. A node exactly on the cut has a sign of its own, so it
    ! is a change on both sides of itself. The crossing is reported as the
    ! zero-based j of the interval's lower end and the fraction of the way
    ! along it that the distance vanishes, clamped to [0, 1]. A line whose
    ! ends there are too close in distance to interpolate between is left
    ! uncut, rather than moved on to its next crossing, as is a line that
    ! never crosses: both come back with j_cut = -1.
    !
    ! The lines of one k are walked together, a j at a time, so the loop over
    ! i reads the field contiguously and has no branch to stop it vectorising.
    ! A line already crossed is carried along unchanged, and once every line
    ! of the k has crossed, the rest of j is skipped. The walk keeps only
    ! where each line crosses: selecting a fraction as well, a real chosen by
    ! a test on an integer, stops gfortran vectorising it. The fractions are
    ! worked out after, once per line.

    implicit none

    integer, intent(in) :: ni, nj, nk

    ! The two fields and the coefficients the distance is formed from
    real, intent(in) :: a(ni, nj, nk), b(ni, nj, nk)
    real, intent(in) :: na, nb, c0

    ! Zero-based j of each line's crossing, -1 where there is none, and how
    ! far along the interval it lies
    integer, intent(inout) :: j_cut(ni, nk)
    real, intent(inout) :: frac(ni, nk)

    ! Room for the distance at the current j of every line of one k,
    ! supplied by the caller
    real, intent(inout) :: d_lo(ni)

    ! Lines that cross
    integer, intent(out) :: n_cut

    ! Marks a line that has crossed between ends too close to interpolate
    ! between, while the walk still has to tell it from one not yet crossed
    integer, parameter :: DEGENERATE = -2

    integer :: i, j, k, n_left
    real :: d_hi
    logical :: hit

    n_cut = 0

    do k = 1, nk
        do i = 1, ni
            d_lo(i) = na * a(i, 1, k) + nb * b(i, 1, k) - c0
            j_cut(i, k) = -1
        end do

        do j = 1, nj - 1
            do i = 1, ni
                d_hi = na * a(i, j + 1, k) + nb * b(i, j + 1, k) - c0

                ! Not crossed before, and the two ends differ in sign
                hit = j_cut(i, k) == -1 .and. &
                    ((d_lo(i) > 0.0 .neqv. d_hi > 0.0) .or. &
                     (d_lo(i) < 0.0 .neqv. d_hi < 0.0))
                j_cut(i, k) = merge( &
                    merge(j - 1, DEGENERATE, abs(d_hi - d_lo(i)) >= 1.0e-12), &
                    j_cut(i, k), hit)

                d_lo(i) = d_hi
            end do

            ! Counted in a loop of its own, which, unlike a count taken in
            ! the walk, leaves both loops to vectorise
            n_left = 0
            do i = 1, ni
                n_left = n_left + merge(1, 0, j_cut(i, k) == -1)
            end do
            if (n_left == 0) exit
        end do

        ! Fraction along each crossed line's interval, from the same
        ! expression as the walk, so the same distances and signs
        do i = 1, ni
            if (j_cut(i, k) >= 0) then
                j = j_cut(i, k) + 1
                d_lo(i) = na * a(i, j, k) + nb * b(i, j, k) - c0
                d_hi = na * a(i, j + 1, k) + nb * b(i, j + 1, k) - c0
                frac(i, k) = min(max(-d_lo(i) / (d_hi - d_lo(i)), 0.0), 1.0)
                n_cut = n_cut + 1
            else
                j_cut(i, k) = -1
                frac(i, k) = 0.0
            end if
        end do
    end do

end subroutine first_j_crossing
