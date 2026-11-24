! Triangle topology for a marching cubes cut

subroutine marching_cubes( &
    dist, vert_ijk, tritable, edge_ijk, lo, hi, frac, n_face, &
    ni, nj, nk, cap)
    ! Locate every triangle vertex a marching cubes cut produces, in one pass.
    !
    ! Each cell's corner sign pattern is found from the signed distance at its
    ! eight corners, one bit per corner set where the distance is negative.
    ! Zero and all-ones are the uncut cells; any other pattern indexes the
    ! triangle table, whose triangles are written out as the cell is reached.
    ! Cells are walked i fastest, the order the field is stored in.
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

    ! Signed distance at every node, zero on the cut, in the single precision
    ! of the coordinates it is measured from, as is everything worked out
    ! from it here
    real, intent(in) :: dist(ni, nj, nk)

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

    integer :: i, j, k, c, ic, bit, row, n, v, e, t
    integer :: ilo(3), ihi(3), swap(3)
    real :: d_lo, d_hi

    t = 0

    do k = 1, nk - 1
        do j = 1, nj - 1
            do i = 1, ni - 1
                ic = 0
                bit = 1
                do c = 1, 8
                    if (dist(i + vert_ijk(c, 1), &
                             j + vert_ijk(c, 2), &
                             k + vert_ijk(c, 3)) < 0.0) ic = ic + bit
                    bit = bit * 2
                end do

                ! All corners on one side: the cut misses this cell
                if (ic == 0 .or. ic == 255) cycle

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

                        d_lo = dist(ilo(1) + 1, ilo(2) + 1, ilo(3) + 1)
                        d_hi = dist(ihi(1) + 1, ihi(2) + 1, ihi(3) + 1)

                        ! The cut crosses this edge, so exactly one end is
                        ! negative and the two distances cannot be equal.
                        ! That also puts the fraction in [0, 1] without it
                        ! having to be clamped.
                        frac(v, t) = -d_lo / (d_hi - d_lo)

                        lo(v, t) = ilo(1) + ni * (ilo(2) + nj * ilo(3))
                        hi(v, t) = ihi(1) + ni * (ihi(2) + nj * ihi(3))
                    end do
                end do
            end do
        end do
    end do

    n_face = t

end subroutine marching_cubes
