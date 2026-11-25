! Interpolation routines for upsampling

subroutine upsample_1d(zeta_fine, zeta_coarse, y_coarse, y_fine, n_fine, n_coarse)
    ! Upsample coarse data to fine grid using linear interpolation
    !
    ! Interpolates coarse grid data onto a fine grid using linear interpolation.
    ! Assumes coarse grid points correspond to every other fine grid point
    ! (i.e., coarse points are at even indices 0, 2, 4, ... of the fine grid).
    !
    ! Parameters
    ! ----------
    ! zeta_fine : real(n_fine)
    !     Fine grid normalized arc length values [0 to 1] (input)
    ! zeta_coarse : real(n_coarse)
    !     Coarse grid normalized arc length values [0 to 1] (input)
    ! y_coarse : real(n_coarse)
    !     Data values at coarse grid points (input)
    ! y_fine : real(n_fine)
    !     Linearly interpolated data at fine grid points (output)
    ! n_fine : integer
    !     Number of fine grid points
    ! n_coarse : integer
    !     Number of coarse grid points

    implicit none

    integer, intent(in) :: n_fine, n_coarse
    real, intent(in) :: zeta_fine(n_fine)
    real, intent(in) :: zeta_coarse(n_coarse)
    real, intent(in) :: y_coarse(n_coarse)
    real, intent(inout) :: y_fine(n_fine)

    integer :: i, i_coarse_left, i_coarse_right
    real :: zeta_left, zeta_right, zeta_target
    real :: frac, denom

    ! Copy coarse values to corresponding fine grid points (every other point)
    ! In Fortran, array indices start at 1
    do i = 1, n_coarse
        y_fine(2*i - 1) = y_coarse(i)
    end do

    ! Interpolate intermediate points (odd indices in 0-based, even in 1-based)
    do i = 2, n_fine-1, 2
        ! Find bracketing coarse points
        ! For Fortran 1-based indexing: fine index i corresponds to 0-based index i-1
        ! 0-based odd indices: 1, 3, 5, ... map to 1-based even indices: 2, 4, 6, ...
        i_coarse_left = i / 2
        i_coarse_right = i_coarse_left + 1

        ! Get zeta values
        zeta_left = zeta_coarse(i_coarse_left)
        zeta_right = zeta_coarse(i_coarse_right)
        zeta_target = zeta_fine(i)

        ! Linear interpolation weight
        denom = zeta_right - zeta_left
        if (abs(denom) < 1.0e-12) then
            ! Handle zero-length interval (use simple average)
            frac = 0.5e0
        else
            frac = (zeta_target - zeta_left) / denom
        end if

        ! Interpolate
        y_fine(i) = y_coarse(i_coarse_left) + frac * (y_coarse(i_coarse_right) - y_coarse(i_coarse_left))
    end do

end subroutine upsample_1d

subroutine map_coordinates_3d(data_in, coords_i, coords_j, coords_k, data_out, &
                               ni, nj, nk, ni_new, nj_new, nk_new, nprop)
    implicit none
    integer, intent(in) :: ni, nj, nk, ni_new, nj_new, nk_new, nprop
    real, intent(in)    :: data_in(ni, nj, nk, nprop)
    real, intent(in)    :: coords_i(ni_new)
    real, intent(in)    :: coords_j(nj_new)
    real, intent(in)    :: coords_k(nk_new)
    real, intent(out)   :: data_out(ni_new, nj_new, nk_new, nprop)

    integer, allocatable :: i0(:), j0(:), k0(:)
    real,    allocatable :: wi(:), wj(:), wk(:)
    integer :: in, jn, kn, n, ii, jj, kk
    real    :: ci, cj, ck
    integer :: i0i, j0j, k0k, i1i, j1j, k1k
    real    :: wi0, wi1, wj0, wj1, wk0, wk1
    real    :: c000, c100, c010, c110, c001, c101, c011, c111

    allocate(i0(ni_new), wi(ni_new))
    allocate(j0(nj_new), wj(nj_new))
    allocate(k0(nk_new), wk(nk_new))

    do in = 1, ni_new
        ci = coords_i(in)
        ii = min(max(int(ci), 0), ni - 2)
        i0(in) = ii + 1
        wi(in) = ci - real(ii)
    end do

    do jn = 1, nj_new
        cj = coords_j(jn)
        jj = min(max(int(cj), 0), nj - 2)
        j0(jn) = jj + 1
        wj(jn) = cj - real(jj)
    end do

    do kn = 1, nk_new
        ck = coords_k(kn)
        kk = min(max(int(ck), 0), nk - 2)
        k0(kn) = kk + 1
        wk(kn) = ck - real(kk)
    end do

    do kn = 1, nk_new
        ! Clamp indices into [1, n]; the upper neighbour degenerates to the
        ! base node when an axis has extent 1 (e.g. a patch face plane). There
        ! the paired weights sum to 1, so the axis contributes an identity and
        ! no read falls outside data_in.
        k0k = min(max(k0(kn), 1), nk);  k1k = min(k0k + 1, nk)
        wk0 = 1.0 - wk(kn);  wk1 = wk(kn)
        do jn = 1, nj_new
            j0j = min(max(j0(jn), 1), nj);  j1j = min(j0j + 1, nj)
            wj0 = 1.0 - wj(jn);  wj1 = wj(jn)
            do in = 1, ni_new
                i0i = min(max(i0(in), 1), ni);  i1i = min(i0i + 1, ni)
                wi0 = 1.0 - wi(in);  wi1 = wi(in)
                do n = 1, nprop
                    c000 = data_in(i0i, j0j, k0k, n)
                    c100 = data_in(i1i, j0j, k0k, n)
                    c010 = data_in(i0i, j1j, k0k, n)
                    c110 = data_in(i1i, j1j, k0k, n)
                    c001 = data_in(i0i, j0j, k1k, n)
                    c101 = data_in(i1i, j0j, k1k, n)
                    c011 = data_in(i0i, j1j, k1k, n)
                    c111 = data_in(i1i, j1j, k1k, n)
                    data_out(in, jn, kn, n) = &
                        wi0*wj0*wk0*c000 + wi1*wj0*wk0*c100 + &
                        wi0*wj1*wk0*c010 + wi1*wj1*wk0*c110 + &
                        wi0*wj0*wk1*c001 + wi1*wj0*wk1*c101 + &
                        wi0*wj1*wk1*c011 + wi1*wj1*wk1*c111
                end do
            end do
        end do
    end do

    deallocate(i0, wi, j0, wj, k0, wk)

end subroutine map_coordinates_3d

subroutine tri_interp_linear(tri_xy, tri_var, qpts, out, ntri, nq, nvar)
    ! Linear interpolation of scattered triangle data onto query points.
    !
    ! For each query point, finds the (first) triangle containing it by brute
    ! force -- with a cheap axis-aligned bounding-box reject -- and writes the
    ! barycentric blend of that triangle's three vertex values. Query points
    ! that fall in no triangle (outside the triangulated region) are left as
    ! NaN, signalling the caller to fill them by nearest-neighbour fallback.
    !
    ! Single precision throughout: barycentric coordinates are O(1), so an
    ! absolute tolerance band on the inclusion test absorbs fp32 edge rounding,
    ! and an area floor skips degenerate slivers.
    !
    ! Parameters
    ! ----------
    ! tri_xy  : real(ntri, 3, 2)     per-triangle vertex (x, y) coordinates
    ! tri_var : real(ntri, 3, nvar)  per-triangle vertex variable values
    ! qpts    : real(nq, 2)          query point (x, y) coordinates
    ! out     : real(nq, nvar)       interpolated values; NaN where no triangle
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    implicit none
    integer, intent(in)  :: ntri, nq, nvar
    real,    intent(in)  :: tri_xy(ntri, 3, 2)
    real,    intent(in)  :: tri_var(ntri, 3, nvar)
    real,    intent(in)  :: qpts(nq, 2)
    real,    intent(out) :: out(nq, nvar)

    real, parameter :: eps = 1.0e-5      ! barycentric inclusion tolerance
    real, parameter :: area_eps = 1.0e-12 ! twice-area floor for sliver reject

    ! Per-triangle precomputed terms for the barycentric solve and bbox reject.
    real, allocatable :: x3(:), y3(:), e1(:), e2(:), e3(:), e4(:), invdet(:)
    real, allocatable :: xmin(:), xmax(:), ymin(:), ymax(:)
    logical, allocatable :: valid(:)
    real    :: x1, y1, x2, y2, xv3, yv3, det
    real    :: px, py, dx, dy, a, b, c
    integer :: t, q, n

    out = ieee_value(0.0, ieee_quiet_nan)

    allocate(x3(ntri), y3(ntri), e1(ntri), e2(ntri), e3(ntri), e4(ntri))
    allocate(invdet(ntri), xmin(ntri), xmax(ntri), ymin(ntri), ymax(ntri))
    allocate(valid(ntri))

    do t = 1, ntri
        x1 = tri_xy(t, 1, 1); y1 = tri_xy(t, 1, 2)
        x2 = tri_xy(t, 2, 1); y2 = tri_xy(t, 2, 2)
        xv3 = tri_xy(t, 3, 1); yv3 = tri_xy(t, 3, 2)
        det = (y2 - yv3) * (x1 - xv3) + (xv3 - x2) * (y1 - yv3)
        valid(t) = abs(det) > area_eps
        if (valid(t)) then
            invdet(t) = 1.0 / det
        else
            invdet(t) = 0.0
        end if
        x3(t) = xv3;  y3(t) = yv3
        e1(t) = y2 - yv3;  e2(t) = xv3 - x2
        e3(t) = yv3 - y1;  e4(t) = x1 - xv3
        xmin(t) = min(x1, x2, xv3);  xmax(t) = max(x1, x2, xv3)
        ymin(t) = min(y1, y2, yv3);  ymax(t) = max(y1, y2, yv3)
    end do

    do q = 1, nq
        px = qpts(q, 1)
        py = qpts(q, 2)
        do t = 1, ntri
            if (.not. valid(t)) cycle
            if (px < xmin(t) - eps .or. px > xmax(t) + eps) cycle
            if (py < ymin(t) - eps .or. py > ymax(t) + eps) cycle
            dx = px - x3(t)
            dy = py - y3(t)
            a = (e1(t) * dx + e2(t) * dy) * invdet(t)
            b = (e3(t) * dx + e4(t) * dy) * invdet(t)
            c = 1.0 - a - b
            if (a >= -eps .and. b >= -eps .and. c >= -eps) then
                do n = 1, nvar
                    out(q, n) = a * tri_var(t, 1, n) &
                              + b * tri_var(t, 2, n) &
                              + c * tri_var(t, 3, n)
                end do
                exit
            end if
        end do
    end do

    deallocate(x3, y3, e1, e2, e3, e4, invdet)
    deallocate(xmin, xmax, ymin, ymax, valid)

end subroutine tri_interp_linear

subroutine bilinear_scattered(Q_src, u_src, v_src, uv_tgt, Q_out, &
                               ni_s, nj_s, ni_t, nj_t, nvar)
    ! Bilinear interpolation from a structured 2D source grid onto scattered
    ! query points. No internal heap allocation -- uses only scalar temporaries.
    !
    ! Parameters
    ! ----------
    ! Q_src  : real(ni_s, nj_s, nvar)  source conserved variables (F-order)
    ! u_src  : real(ni_s)              source u grid axis (monotone increasing)
    ! v_src  : real(nj_s)              source v grid axis (monotone increasing)
    ! uv_tgt : real(ni_t, nj_t, 2)    target parametric coords [u, v] in [0,1]
    ! Q_out  : real(ni_t, nj_t, nvar) output array written in-place
    implicit none
    integer, intent(in)    :: ni_s, nj_s, ni_t, nj_t, nvar
    real,    intent(in)    :: Q_src(ni_s, nj_s, nvar)
    real,    intent(in)    :: u_src(ni_s)
    real,    intent(in)    :: v_src(nj_s)
    real,    intent(in)    :: uv_tgt(ni_t, nj_t, 2)
    real,    intent(inout) :: Q_out(ni_t, nj_t, nvar)

    integer :: it, jt, i0, j0, n
    real    :: u, v, wu1, wu0, wv1, wv0

    do jt = 1, nj_t
        do it = 1, ni_t
            u = uv_tgt(it, jt, 1)
            v = uv_tgt(it, jt, 2)

            i0 = 1
            do while (i0 < ni_s - 1 .and. u_src(i0+1) <= u)
                i0 = i0 + 1
            end do
            wu1 = (u - u_src(i0)) / (u_src(i0+1) - u_src(i0))
            wu0 = 1.0 - wu1

            j0 = 1
            do while (j0 < nj_s - 1 .and. v_src(j0+1) <= v)
                j0 = j0 + 1
            end do
            wv1 = (v - v_src(j0)) / (v_src(j0+1) - v_src(j0))
            wv0 = 1.0 - wv1

            do n = 1, nvar
                Q_out(it, jt, n) = wu0*wv0*Q_src(i0,   j0,   n) &
                                  + wu1*wv0*Q_src(i0+1, j0,   n) &
                                  + wu0*wv1*Q_src(i0,   j0+1, n) &
                                  + wu1*wv1*Q_src(i0+1, j0+1, n)
            end do
        end do
    end do

end subroutine bilinear_scattered

subroutine pchip_resample_3d(data_in, coords_i, coords_j, coords_k, &
                             lo_i, hi_i, lo_j, hi_j, lo_k, hi_k, data_out, &
                             slopes, ni, nj, nk, ni_new, nj_new, nk_new, nprop)
    ! Resample block coordinates at fractional indices, smoothly
    !
    ! Each property is interpolated along every old grid line, in each
    ! direction, by a monotone piecewise cubic (PCHIP, Fritsch-Butland slopes
    ! on the unit index spacing), one segment between critical indices at a
    ! time, so a curve depends only on its own segment's nodes, as the block
    ! across an interface ending there sees them. A new node is the trilinear
    ! Gordon-Hall blend of the twelve edge curves of its old cell, less twice
    ! the trilinear blend of the cell's corners: that reproduces each edge
    ! curve, uses a face's own data on it, and treats the directions alike, so
    ! two blocks meeting across an interface agree however their axes run.
    ! Sums are offsets from the cell's first corner, so single precision
    ! loses nothing at large coordinates.
    !
    ! Parameters
    ! ----------
    ! data_in : real(ni, nj, nk, nprop)
    !     Values at the old nodes (input)
    ! coords_i, coords_j, coords_k : real(*_new)
    !     Zero-based fractional old indices of the new nodes in each direction
    ! lo_i, hi_i : integer(ni), and likewise for j and k
    !     First and last node of the segment holding the interval from each
    !     node to the next, one-based; the last entry repeats the one before
    ! data_out : real(ni_new, nj_new, nk_new, nprop)
    !     Values at the new nodes (output)
    ! slopes : real(ni, nj, nk, 6)
    !     Scratch: one property's slopes along i, j, k into the interval after
    !     each node (1-3) and out of the interval before it (4-6)

    implicit none
    integer, intent(in) :: ni, nj, nk, ni_new, nj_new, nk_new, nprop
    real, intent(in)    :: data_in(ni, nj, nk, nprop)
    real, intent(in)    :: coords_i(ni_new), coords_j(nj_new), coords_k(nk_new)
    integer, intent(in) :: lo_i(ni), hi_i(ni), lo_j(nj), hi_j(nj), lo_k(nk), hi_k(nk)
    real, intent(out)   :: data_out(ni_new, nj_new, nk_new, nprop)
    real, intent(inout) :: slopes(ni, nj, nk, 6)

    integer :: p, i, j, k, in, jn, kn, q, r, s
    integer :: ii(0:1), jj(0:1), kk(0:1)
    real :: wa(0:1), wb(0:1), wc(0:1), c0, acc, ta, tb, tc

    do p = 1, nprop

        ! Slopes at every old node along each direction, within the segment
        ! after the node (1-3) and the one before it (4-6)
        slopes = 0.0
        do k = 1, nk
            do j = 1, nj
                do i = 1, ni
                    if (i < ni) slopes(i, j, k, 1) = slope_i(i, lo_i(i), hi_i(i))
                    if (i > 1) slopes(i, j, k, 4) = slope_i(i, lo_i(i - 1), hi_i(i - 1))
                    if (j < nj) slopes(i, j, k, 2) = slope_j(j, lo_j(j), hi_j(j))
                    if (j > 1) slopes(i, j, k, 5) = slope_j(j, lo_j(j - 1), hi_j(j - 1))
                    if (k < nk) slopes(i, j, k, 3) = slope_k(k, lo_k(k), hi_k(k))
                    if (k > 1) slopes(i, j, k, 6) = slope_k(k, lo_k(k - 1), hi_k(k - 1))
                end do
            end do
        end do

        ! Gordon-Hall blend within each old cell
        do kn = 1, nk_new
            call cell_of(coords_k(kn), nk, kk, tc)
            wc(1) = tc; wc(0) = 1.0 - tc
            do jn = 1, nj_new
                call cell_of(coords_j(jn), nj, jj, tb)
                wb(1) = tb; wb(0) = 1.0 - tb
                do in = 1, ni_new
                    call cell_of(coords_i(in), ni, ii, ta)
                    wa(1) = ta; wa(0) = 1.0 - ta
                    ! An old node: copy it, exactly
                    if (wa(0) * wa(1) == 0.0 .and. wb(0) * wb(1) == 0.0 &
                        .and. wc(0) * wc(1) == 0.0) then
                        data_out(in, jn, kn, p) = data_in(ii(nint(ta)), &
                            jj(nint(tb)), kk(nint(tc)), p)
                        cycle
                    end if
                    c0 = data_in(ii(0), jj(0), kk(0), p)
                    acc = 0.0
                    do r = 0, 1
                        do q = 0, 1
                            ! Edges along i, j and k, each weighted by the
                            ! bilinear blend across the other two
                            acc = acc + wb(q) * wc(r) * edge( &
                                data_in(ii(0), jj(q), kk(r), p), &
                                data_in(ii(1), jj(q), kk(r), p), &
                                slopes(ii(0), jj(q), kk(r), 1), &
                                slopes(ii(1), jj(q), kk(r), 4), ta)
                            acc = acc + wa(q) * wc(r) * edge( &
                                data_in(ii(q), jj(0), kk(r), p), &
                                data_in(ii(q), jj(1), kk(r), p), &
                                slopes(ii(q), jj(0), kk(r), 2), &
                                slopes(ii(q), jj(1), kk(r), 5), tb)
                            acc = acc + wa(q) * wb(r) * edge( &
                                data_in(ii(q), jj(r), kk(0), p), &
                                data_in(ii(q), jj(r), kk(1), p), &
                                slopes(ii(q), jj(r), kk(0), 3), &
                                slopes(ii(q), jj(r), kk(1), 6), tc)
                            do s = 0, 1
                                acc = acc - 2.0 * wa(s) * wb(q) * wc(r) &
                                    * (data_in(ii(s), jj(q), kk(r), p) - c0)
                            end do
                        end do
                    end do
                    data_out(in, jn, kn, p) = c0 + acc
                end do
            end do
        end do

    end do

contains

    real function slope_i(m, a, b)
        ! Slope at node m along i, within segment a..b
        integer, intent(in) :: m, a, b
        slope_i = node_slope(b - a + 1, m - a + 1, &
            data_in(max(m - 2, a), j, k, p), data_in(max(m - 1, a), j, k, p), &
            data_in(m, j, k, p), &
            data_in(min(m + 1, b), j, k, p), data_in(min(m + 2, b), j, k, p))
    end function slope_i

    real function slope_j(m, a, b)
        ! Slope at node m along j, within segment a..b
        integer, intent(in) :: m, a, b
        slope_j = node_slope(b - a + 1, m - a + 1, &
            data_in(i, max(m - 2, a), k, p), data_in(i, max(m - 1, a), k, p), &
            data_in(i, m, k, p), &
            data_in(i, min(m + 1, b), k, p), data_in(i, min(m + 2, b), k, p))
    end function slope_j

    real function slope_k(m, a, b)
        ! Slope at node m along k, within segment a..b
        integer, intent(in) :: m, a, b
        slope_k = node_slope(b - a + 1, m - a + 1, &
            data_in(i, j, max(m - 2, a), p), data_in(i, j, max(m - 1, a), p), &
            data_in(i, j, m, p), &
            data_in(i, j, min(m + 1, b), p), data_in(i, j, min(m + 2, b), p))
    end function slope_k

    subroutine cell_of(c, n, b, t)
        ! One-based end nodes of the old interval holding fractional index c,
        ! and the fraction of the way along it; the last node closes the last
        ! interval at t = 1, and a direction of one node has t = 0
        real, intent(in) :: c
        integer, intent(in) :: n
        integer, intent(out) :: b(0:1)
        real, intent(out) :: t
        if (n == 1) then
            b = 1
            t = 0.0
        else
            b(0) = min(max(int(c), 0), n - 2)
            t = min(max(c - real(b(0)), 0.0), 1.0)
            b(0) = b(0) + 1
            b(1) = b(0) + 1
        end if
    end subroutine cell_of

    real function edge(y0, y1, m0, m1, t)
        ! Cubic Hermite from y0 to y1 at fraction t, as an offset from c0
        real, intent(in) :: y0, y1, m0, m1, t
        if (t == 0.0) then
            edge = y0 - c0
        else if (t == 1.0) then
            edge = y1 - c0
        else
            edge = (y0 - c0) + t * t * (3.0 - 2.0 * t) * (y1 - y0) &
                + t * (1.0 - t) ** 2 * m0 + t * t * (t - 1.0) * m1
        end if
    end function edge

    real function node_slope(n, m, ym2, ym1, y0, yp1, yp2)
        ! PCHIP slope at node m of a line of n nodes, on unit spacing, given
        ! the values two either side (clamped into the line): the harmonic
        ! mean of the secants inside, zero at an extremum; three-point at the
        ! ends, kept to the shape of the data; the secant on two nodes
        integer, intent(in) :: n, m
        real, intent(in) :: ym2, ym1, y0, yp1, yp2
        real :: d0, d1
        if (n == 1) then
            node_slope = 0.0
        else if (n == 2) then
            node_slope = merge(yp1 - y0, y0 - ym1, m == 1)
        else if (m == 1 .or. m == n) then
            if (m == 1) then
                d0 = yp1 - y0; d1 = yp2 - yp1
            else
                d0 = y0 - ym1; d1 = ym1 - ym2
            end if
            node_slope = 0.5 * (3.0 * d0 - d1)
            if (node_slope * d0 <= 0.0) then
                node_slope = 0.0
            else if (d0 * d1 < 0.0 .and. abs(node_slope) > abs(3.0 * d0)) then
                node_slope = 3.0 * d0
            end if
        else
            d0 = y0 - ym1; d1 = yp1 - y0
            if (d0 * d1 > 0.0) then
                node_slope = 2.0 * d0 * d1 / (d0 + d1)
            else
                node_slope = 0.0
            end if
        end if
    end function node_slope

end subroutine pchip_resample_3d
