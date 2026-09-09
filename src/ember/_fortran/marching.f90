! Triangle topology for a marching cubes cut

subroutine marching_cubes_edges( &
    dist, cells, offsets, icube, tritable, edge_ijk, &
    ijk_lo, ijk_hi, frac, ni, nj, nk, ncut, ntri)
    ! Locate every triangle vertex a marching cubes cut produces.
    !
    ! Walks only the cells the caller has already found to be cut, and for
    ! each one reads the triangle list out of tritable. A vertex sits on a
    ! cell edge, between two nodes of the grid, so it is reported as the pair
    ! of nodes it lies between and the fraction of the way along that the
    ! signed distance vanishes.
    !
    ! Interpolating the data itself is left to the caller, which keeps this
    ! free of the number of variables, of the bulk data array, and of the
    ! precision that array happens to be stored in.
    !
    ! Every table this needs is supplied by the caller rather than repeated
    ! here, so that the Python and Fortran sides cannot drift apart.
    !
    ! All grid indices, in cells and in the two output arrays, are zero-based
    ! as the caller uses them; only the lookups into the tables are shifted.

    implicit none

    integer, intent(in) :: ni, nj, nk, ncut, ntri

    ! Signed distance at every node, zero on the cut
    double precision, intent(in) :: dist(ni, nj, nk)

    ! Zero-based (i, j, k) of each cut cell, and the index of the first
    ! triangle it contributes, so that cells can be walked in any order and
    ! still write their triangles to the place the caller expects them
    integer, intent(in) :: cells(ncut, 3)
    integer, intent(in) :: offsets(ncut)

    ! Which of the 256 corner sign patterns each cut cell has
    integer, intent(in) :: icube(ncut)

    ! Triangles per pattern, as edge numbers in threes, ending at -1
    integer, intent(in) :: tritable(256, 16)

    ! The two corner offsets of each of the 12 cell edges
    integer, intent(in) :: edge_ijk(12, 2, 3)

    ! Nodes each vertex lies between, and how far along it lies
    integer, intent(inout) :: ijk_lo(ntri, 3, 3)
    integer, intent(inout) :: ijk_hi(ntri, 3, 3)
    double precision, intent(inout) :: frac(ntri, 3)

    integer :: c, n, v, e, t, row
    integer :: ijk_cell(3), lo(3), hi(3)
    double precision :: d_lo, d_hi

    do c = 1, ncut
        ijk_cell = cells(c, :)
        row = icube(c) + 1
        t = offsets(c)

        do n = 1, 16, 3
            ! A sentinel edge closes the pattern's list of triangles
            if (tritable(row, n) < 0) exit

            t = t + 1
            do v = 1, 3
                e = tritable(row, n + v - 1) + 1

                lo = ijk_cell + edge_ijk(e, 1, :)
                hi = ijk_cell + edge_ijk(e, 2, :)

                d_lo = dist(lo(1) + 1, lo(2) + 1, lo(3) + 1)
                d_hi = dist(hi(1) + 1, hi(2) + 1, hi(3) + 1)

                ! The cut crosses this edge, so exactly one end is negative
                ! and the two distances cannot be equal.  That also puts the
                ! fraction in [0, 1] without it having to be clamped.
                frac(t, v) = -d_lo / (d_hi - d_lo)

                ijk_lo(t, v, :) = lo
                ijk_hi(t, v, :) = hi
            end do
        end do
    end do

end subroutine marching_cubes_edges


subroutine marching_cubes_index(dist, vert_ijk, icube, ni, nj, nk)
    ! Which of the 256 corner sign patterns each cell of the grid has.
    !
    ! One bit per corner, set where the signed distance is negative, so that
    ! zero and all-ones are the uncut cells and everything between indexes the
    ! triangle table.  Walking the corners of a cell together reads the
    ! distance field once, rather than once per corner as a slice at a time
    ! over the whole field would.
    !
    ! The corner offsets come from the caller so that the order of the bits is
    ! defined in one place only.

    implicit none

    integer, intent(in) :: ni, nj, nk

    double precision, intent(in) :: dist(ni, nj, nk)

    ! Zero-based (i, j, k) offset of each of the 8 corners of a cell
    integer, intent(in) :: vert_ijk(8, 3)

    integer, intent(inout) :: icube(ni - 1, nj - 1, nk - 1)

    integer :: i, j, k, v, ic, bit

    do k = 1, nk - 1
        do j = 1, nj - 1
            do i = 1, ni - 1
                ic = 0
                bit = 1
                do v = 1, 8
                    if (dist(i + vert_ijk(v, 1), &
                             j + vert_ijk(v, 2), &
                             k + vert_ijk(v, 3)) < 0.0d0) ic = ic + bit
                    bit = bit * 2
                end do
                icube(i, j, k) = ic
            end do
        end do
    end do

end subroutine marching_cubes_index
