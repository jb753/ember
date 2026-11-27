! Structured traverse cuts: where a cut crosses each family of grid lines,
! the patches those crossings make, and the arc length along the cut curve

subroutine crossing_counts( &
    a, b, na, nb, c0, sgn, cnt_i, cnt_j, cnt_k, prev_j, prev_k, n_pos, &
    ni, nj, nk)
    ! Count how many times every grid line crosses a cut, in one pass.
    !
    ! The cut is the zero level of d = na a + nb b - c0, formed node by node
    ! as in marching_cubes. Each node's side, 1 where d >= 0 and 0 elsewhere,
    ! is written to sgn, so that crossing_patch walks exactly the same signs
    ! however the compiler happens to round d in another loop.
    !
    ! Lines along i are counted a row at a time. Lines along j and k are
    ! counted by comparing each row with the one before it in j, and with the
    ! same row one plane before in k, both kept in caller scratch, so every
    ! loop runs over i and reads contiguously.

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: a(ni, nj, nk), b(ni, nj, nk)
    real, intent(in) :: na, nb, c0

    ! Side of the cut at every node
    integer(kind=1), intent(inout) :: sgn(ni, nj, nk)

    ! Sign changes along each line of i, of j and of k
    integer, intent(inout) :: cnt_i(nj, nk), cnt_j(ni, nk), cnt_k(ni, nj)

    ! Signs of the previous row in j and of every row of the previous plane
    integer(kind=1), intent(inout) :: prev_j(ni), prev_k(ni, nj)

    ! Nodes on the positive side, to tell a block the cut misses
    integer, intent(out) :: n_pos

    integer :: i, j, k, c

    n_pos = 0
    cnt_j = 0
    cnt_k = 0

    do k = 1, nk
        do j = 1, nj
            do i = 1, ni
                sgn(i, j, k) = int(merge(1, 0, &
                    na * a(i, j, k) + nb * b(i, j, k) - c0 >= 0.0), kind=1)
            end do

            c = 0
            do i = 1, ni
                n_pos = n_pos + sgn(i, j, k)
            end do
            do i = 1, ni - 1
                c = c + abs(sgn(i + 1, j, k) - sgn(i, j, k))
            end do
            cnt_i(j, k) = c

            if (j > 1) then
                do i = 1, ni
                    cnt_j(i, k) = cnt_j(i, k) + abs(sgn(i, j, k) - prev_j(i))
                end do
            end if
            do i = 1, ni
                prev_j(i) = sgn(i, j, k)
            end do

            if (k > 1) then
                do i = 1, ni
                    cnt_k(i, j) = cnt_k(i, j) + abs(sgn(i, j, k) - prev_k(i, j))
                end do
            end if
            do i = 1, ni
                prev_k(i, j) = sgn(i, j, k)
            end do
        end do
    end do

end subroutine crossing_counts


subroutine crossing_patch( &
    a, b, na, nb, c0, sgn, data, ax, nth, p0, q0, out, edge, cnt, &
    ni, nj, nk, nvar, n1, n2)
    ! Interpolate a block's data to the nth crossing of each line of one axis.
    !
    ! The lines run along axis ax (1, 2 or 3 for i, j, k). They are indexed
    ! by the other two axes in order, starting from the zero-based stations
    ! p0 and q0 and running n1 and n2 lines along each: the patch the
    ! crossings make. The walk reads the signs crossing_counts wrote, and the
    ! caller has checked from its counts that every one of these lines
    ! crosses at least nth times.
    !
    ! Each variable is interpolated from the edge's node on the negative
    ! side to its node on the positive side, whichever way the block's index
    ! runs along it, so two blocks that share the edge give the same value.
    ! Lines along j and k are walked a plane or a row at a time, over i, so
    ! the walk reads contiguously; the interpolation goes a variable at a
    ! time.

    implicit none

    integer, intent(in) :: ni, nj, nk, nvar, n1, n2
    real, intent(in) :: a(ni, nj, nk), b(ni, nj, nk)
    real, intent(in) :: na, nb, c0
    integer(kind=1), intent(in) :: sgn(ni, nj, nk)
    real, intent(in) :: data(ni, nj, nk, nvar)
    integer, intent(in) :: ax, nth, p0, q0

    ! Data at the crossings, over the patch
    real, intent(inout) :: out(n1, n2, nvar)

    ! Caller scratch: one-based station of the crossing edge's lower end on
    ! each line, and crossings so far
    integer, intent(inout) :: edge(n1, n2), cnt(n1, n2)

    integer :: p, q, s, i, j, k, v, i0, j0, k0, i1, j1, k1
    integer :: ia, ja, ka, ib, jb, kb
    real :: da, db, f
    logical :: chg

    cnt = 0
    edge = 0

    if (ax == 1) then
        ! Lines along i, indexed by (j, k)
        do q = 1, n2
            k = q0 + q
            do p = 1, n1
                j = p0 + p
                do s = 1, ni - 1
                    if (sgn(s + 1, j, k) /= sgn(s, j, k)) then
                        cnt(p, q) = cnt(p, q) + 1
                        if (cnt(p, q) == nth) then
                            edge(p, q) = s
                            exit
                        end if
                    end if
                end do
            end do
        end do
    else if (ax == 2) then
        ! Lines along j, indexed by (i, k): walk j for a row of i at once
        do q = 1, n2
            k = q0 + q
            do s = 1, nj - 1
                do p = 1, n1
                    i = p0 + p
                    chg = sgn(i, s + 1, k) /= sgn(i, s, k)
                    cnt(p, q) = cnt(p, q) + merge(1, 0, chg)
                    edge(p, q) = merge(s, edge(p, q), &
                        chg .and. cnt(p, q) == nth)
                end do
            end do
        end do
    else
        ! Lines along k, indexed by (i, j): walk k for a plane at once
        do s = 1, nk - 1
            do q = 1, n2
                j = q0 + q
                do p = 1, n1
                    i = p0 + p
                    chg = sgn(i, j, s + 1) /= sgn(i, j, s)
                    cnt(p, q) = cnt(p, q) + merge(1, 0, chg)
                    edge(p, q) = merge(s, edge(p, q), &
                        chg .and. cnt(p, q) == nth)
                end do
            end do
        end do
    end if

    ! Interpolate along each line's crossing edge, negative end to positive
    do q = 1, n2
        do p = 1, n1
            s = edge(p, q)
            if (ax == 1) then
                i0 = s; j0 = p0 + p; k0 = q0 + q
                i1 = s + 1; j1 = j0; k1 = k0
            else if (ax == 2) then
                i0 = p0 + p; j0 = s; k0 = q0 + q
                i1 = i0; j1 = s + 1; k1 = k0
            else
                i0 = p0 + p; j0 = q0 + q; k0 = s
                i1 = i0; j1 = j0; k1 = s + 1
            end if
            if (sgn(i0, j0, k0) == 0) then
                ia = i0; ja = j0; ka = k0; ib = i1; jb = j1; kb = k1
            else
                ia = i1; ja = j1; ka = k1; ib = i0; jb = j0; kb = k0
            end if
            da = na * a(ia, ja, ka) + nb * b(ia, ja, ka) - c0
            db = na * a(ib, jb, kb) + nb * b(ib, jb, kb) - c0
            if (da == db) then
                f = 0.5
            else
                f = min(max(da / (da - db), 0.0), 1.0)
            end if
            do v = 1, nvar
                out(p, q, v) = data(ia, ja, ka, v) &
                    + (data(ib, jb, kb, v) - data(ia, ja, ka, v)) * f
            end do
        end do
    end do

end subroutine crossing_patch


subroutine polyline_arc(xr, xq, arc, npts, npt)
    ! Arc length along a polyline of the foot of each query point.
    !
    ! The first segment runs on backwards past the start and the last on
    ! forwards past the end, so points beyond the ends get arc lengths below
    ! zero or past the total. Each point takes the foot on the nearest piece.

    implicit none

    integer, intent(in) :: npts, npt
    real, intent(in) :: xr(npts, 2), xq(npt, 2)
    real, intent(inout) :: arc(npt)

    integer :: n, s
    real :: dx, dr, len2, u, ex, er, dist, best, cum, seg

    do n = 1, npt
        best = huge(1.0)
        cum = 0.0
        do s = 1, npts - 1
            dx = xr(s + 1, 1) - xr(s, 1)
            dr = xr(s + 1, 2) - xr(s, 2)
            len2 = dx * dx + dr * dr
            seg = sqrt(len2)
            u = ((xq(n, 1) - xr(s, 1)) * dx + (xq(n, 2) - xr(s, 2)) * dr) / len2
            if (s > 1) u = max(u, 0.0)
            if (s < npts - 1) u = min(u, 1.0)
            ex = xq(n, 1) - xr(s, 1) - u * dx
            er = xq(n, 2) - xr(s, 2) - u * dr
            dist = ex * ex + er * er
            if (dist < best) then
                best = dist
                arc(n) = cum + u * seg
            end if
            cum = cum + seg
        end do
    end do

end subroutine polyline_arc


subroutine fill_hole( &
    zv, tv, var, faces, zl, tl, zr, tr, tt, eps, out, hits, &
    nv, nvar, nf, n, m)
    ! Fill the interior rows of one hole from the triangles that cover it.
    !
    ! Row j runs straight in (zeta, theta) from (zl(j), tl(j)) to
    ! (zr(j), tr(j)), theta in pitches, and its nodes sit at theta tt(:, j).
    ! Along it f_j = zeta - z_j(theta) is zero, where z_j is the row's line,
    ! and f_j falls as j rises, the rows stacking from hub to casing.
    !
    ! Each triangle is first moved whole pitches to the hole's middle at its
    ! own span, the middle and half-width read off the row nearest its
    ! centroid; it and its copies a pitch either side are each skipped if
    ! their theta range misses the hole. Its vertices
    ! are bracketed between rows by bisection on f_j, and it is sliced
    ! exactly against each row it spans: a segment with ends on two of its
    ! edges, along which every node of the row within eps of the segment
    ! takes the value linear in theta. Rows 1 and n, on the endwalls, are
    ! left to the caller, which interpolates along the walls themselves.
    !
    ! hits counts the writes to each node, so a node left at zero lies on no
    ! triangle.

    implicit none

    integer, intent(in) :: nv, nvar, nf, n, m
    real, intent(in) :: zv(nv), tv(nv), var(nv, nvar)
    integer, intent(in) :: faces(3, nf)
    real, intent(in) :: zl(n), tl(n), zr(n), tr(n), tt(m, n), eps
    real, intent(inout) :: out(n, m, nvar)
    integer, intent(inout) :: hits(n, m)

    integer :: f, c, j, jlo, jhi, jv(3), e, k, ka, kb, n_pt, idx(3), p
    real :: tz(3), tq(3), t0q(3), zc, tc, mid, half, s, fv(3), w
    integer :: ds
    real :: pt_t(2), pt_w(2)
    integer :: pt_a(2), pt_b(2)
    real :: ta, tb, a, b, wt
    integer :: lo, hi

    hits = 0

    do f = 1, nf
        do c = 1, 3
            idx(c) = faces(c, f) + 1
            tz(c) = zv(idx(c))
            tq(c) = tv(idx(c))
        end do

        ! Middle and half-width of the hole at the triangle's span
        zc = (tz(1) + tz(2) + tz(3)) / 3.0
        j = side_row(zc)
        mid = 0.5 * (tl(j) + tr(j))
        half = 0.5 * (tr(j) - tl(j))
        tc = (tq(1) + tq(2) + tq(3)) / 3.0
        s = real(nint(mid - tc))
        do c = 1, 3
            t0q(c) = tq(c) + s
        end do

        ! A hole up to a pitch wide can need a copy a pitch either side too
        do ds = -1, 1
        do c = 1, 3
            tq(c) = t0q(c) + real(ds)
        end do
        if (max(tq(1), tq(2), tq(3)) < mid - half - 1.0e-3) cycle
        if (min(tq(1), tq(2), tq(3)) > mid + half + 1.0e-3) cycle

        ! Rows the triangle can meet
        do c = 1, 3
            jv(c) = row_below(tz(c), tq(c))
        end do
        jlo = max(min(jv(1), jv(2), jv(3)), 2)
        jhi = min(max(jv(1), jv(2), jv(3)) + 1, n - 1)

        do j = jlo, jhi
            do c = 1, 3
                fv(c) = tz(c) - row_z(j, tq(c))
            end do
            if (min(fv(1), fv(2), fv(3)) > 0.0) cycle
            if (max(fv(1), fv(2), fv(3)) < 0.0) cycle

            ! Ends of the slice: vertices on the row, then edges it crosses
            n_pt = 0
            do c = 1, 3
                if (fv(c) == 0.0 .and. n_pt < 2) then
                    n_pt = n_pt + 1
                    pt_t(n_pt) = tq(c)
                    pt_w(n_pt) = 0.0
                    pt_a(n_pt) = idx(c)
                    pt_b(n_pt) = idx(c)
                end if
            end do
            do e = 1, 3
                ka = e
                kb = mod(e, 3) + 1
                if (fv(ka) * fv(kb) < 0.0 .and. n_pt < 2) then
                    w = fv(ka) / (fv(ka) - fv(kb))
                    n_pt = n_pt + 1
                    pt_t(n_pt) = tq(ka) + w * (tq(kb) - tq(ka))
                    pt_w(n_pt) = w
                    pt_a(n_pt) = idx(ka)
                    pt_b(n_pt) = idx(kb)
                end if
            end do
            if (n_pt < 2) cycle
            if (pt_t(1) <= pt_t(2)) then
                lo = 1; hi = 2
            else
                lo = 2; hi = 1
            end if
            ta = pt_t(lo)
            tb = pt_t(hi)

            ! Nodes of the row on the segment
            do k = 1, m
                if (tt(k, j) < ta - eps .or. tt(k, j) > tb + eps) cycle
                if (tb > ta) then
                    wt = min(max((tt(k, j) - ta) / (tb - ta), 0.0), 1.0)
                else
                    wt = 0.5
                end if
                do p = 1, nvar
                    a = var(pt_a(lo), p) &
                        + pt_w(lo) * (var(pt_b(lo), p) - var(pt_a(lo), p))
                    b = var(pt_a(hi), p) &
                        + pt_w(hi) * (var(pt_b(hi), p) - var(pt_a(hi), p))
                    out(j, k, p) = a + wt * (b - a)
                end do
                hits(j, k) = hits(j, k) + 1
            end do
        end do
        end do
    end do

contains

    real function row_z(j, t)
        ! Arc length of row j's line at theta t
        integer, intent(in) :: j
        real, intent(in) :: t
        row_z = zl(j) + (t - tl(j)) / (tr(j) - tl(j)) * (zr(j) - zl(j))
    end function row_z

    integer function side_row(z)
        ! Row whose left end is nearest below arc length z, at least 1
        real, intent(in) :: z
        integer :: lo_, hi_, mid_
        lo_ = 1
        hi_ = n
        do while (hi_ - lo_ > 1)
            mid_ = (lo_ + hi_) / 2
            if (zl(mid_) <= z) then
                lo_ = mid_
            else
                hi_ = mid_
            end if
        end do
        side_row = lo_
    end function side_row

    integer function row_below(z, t)
        ! Highest row whose line passes at or below (z, t), 0 if none
        real, intent(in) :: z, t
        integer :: lo_, hi_, mid_
        lo_ = 0
        hi_ = n + 1
        do while (hi_ - lo_ > 1)
            mid_ = (lo_ + hi_) / 2
            if (z - row_z(mid_, t) >= 0.0) then
                lo_ = mid_
            else
                hi_ = mid_
            end if
        end do
        row_below = lo_
    end function row_below

end subroutine fill_hole


subroutine union_labels(a, b, label, parent, n_label, n, npair)
    ! Label the groups that pairs of indices join, 0 to n_label - 1.
    !
    ! Union-find with path halving: each pair joins the roots of its two
    ! ends, the lower root becoming the parent. Labels then number the roots
    ! in order of their least member, so they are the same however the pairs
    ! are listed. a and b are zero-based; parent is caller scratch.

    implicit none

    integer, intent(in) :: n, npair
    integer, intent(in) :: a(npair), b(npair)
    integer, intent(inout) :: label(n), parent(n)
    integer, intent(out) :: n_label

    integer :: p, ra, rb, i

    do i = 1, n
        parent(i) = i
    end do

    do p = 1, npair
        ra = root(a(p) + 1)
        rb = root(b(p) + 1)
        if (ra < rb) then
            parent(rb) = ra
        else if (rb < ra) then
            parent(ra) = rb
        end if
    end do

    n_label = 0
    do i = 1, n
        ra = root(i)
        if (ra == i) then
            label(i) = n_label
            n_label = n_label + 1
        else
            label(i) = label(ra)
        end if
    end do

contains

    integer function root(i0)
        integer, intent(in) :: i0
        integer :: r
        r = i0
        do while (parent(r) /= r)
            parent(r) = parent(parent(r))
            r = parent(r)
        end do
        root = r
    end function root

end subroutine union_labels


subroutine edge_topology( &
    faces, start, bucket, uses, nbr, mark, loops, loop_start, &
    n_edge, n_bnd, n_loop, n_used, status, nv, nf, cap)
    ! Edges of a triangle mesh, the boundary, and its boundary loops.
    !
    ! Each triangle side is an undirected edge between its lower and higher
    ! vertex. Edges are bucketed by their lower vertex with a counting sort
    ! into caller scratch, and duplicates within a bucket counted: an edge of
    ! one triangle only is on the boundary. Triangles with two corners the
    ! same, left by merging vertices, are skipped.
    !
    ! Every boundary vertex of a manifold mesh has two boundary neighbours,
    ! and the loops are traced from them, each listed in order into loops
    ! from loop_start(l) to loop_start(l + 1) - 1, zero-based vertex ids.
    ! status is 0, or 1 where a vertex has more than two boundary edges and
    ! the boundary does not trace into loops. n_used counts the vertices
    ! some triangle uses, for the Euler characteristic V - E + F.

    implicit none

    integer, intent(in) :: nv, nf, cap
    integer, intent(in) :: faces(3, nf)
    integer, intent(inout) :: start(nv + 1), bucket(3 * nf), uses(3 * nf)
    integer, intent(inout) :: nbr(2, nv), mark(nv), loops(cap), loop_start(cap + 1)
    integer, intent(out) :: n_edge, n_bnd, n_loop, n_used, status

    integer :: f, c, u, v, lo, hi, p, q, s0, s1, cur, prev, nxt, l
    logical :: ok

    status = 0
    start = 0
    nbr = -1

    ! Count edges per lower vertex, and the vertices used
    do f = 1, nf
        if (degenerate(f)) cycle
        do c = 1, 3
            u = faces(c, f) + 1
            v = faces(mod(c, 3) + 1, f) + 1
            lo = min(u, v)
            start(lo + 1) = start(lo + 1) + 1
        end do
    end do
    start(1) = 1
    do u = 1, nv
        start(u + 1) = start(u + 1) + start(u)
    end do

    ! Place each edge's higher vertex in its lower vertex's bucket
    uses = 0
    do u = 1, nv
        nbr(1, u) = start(u)
    end do
    do f = 1, nf
        if (degenerate(f)) cycle
        do c = 1, 3
            u = faces(c, f) + 1
            v = faces(mod(c, 3) + 1, f) + 1
            lo = min(u, v)
            hi = max(u, v)
            bucket(nbr(1, lo)) = hi
            nbr(1, lo) = nbr(1, lo) + 1
        end do
    end do

    ! Count uses within each bucket, kept on the first copy of each edge
    n_edge = 0
    n_bnd = 0
    do u = 1, nv
        s0 = start(u)
        s1 = start(u + 1) - 1
        do p = s0, s1
            if (bucket(p) < 0) cycle
            uses(p) = 1
            do q = p + 1, s1
                if (bucket(q) == bucket(p)) then
                    uses(p) = uses(p) + 1
                    bucket(q) = -bucket(q)
                end if
            end do
            n_edge = n_edge + 1
        end do
    end do

    ! Boundary neighbours of each vertex
    nbr = -1
    do u = 1, nv
        do p = start(u), start(u + 1) - 1
            if (bucket(p) < 0 .or. uses(p) /= 1) cycle
            n_bnd = n_bnd + 1
            v = bucket(p)
            call add_nbr(u, v, ok)
            if (.not. ok) status = 1
            call add_nbr(v, u, ok)
            if (.not. ok) status = 1
        end do
    end do

    ! Vertices used
    mark = 0
    do f = 1, nf
        if (degenerate(f)) cycle
        do c = 1, 3
            mark(faces(c, f) + 1) = 1
        end do
    end do
    n_used = sum(mark)

    ! Trace the loops
    n_loop = 0
    loop_start(1) = 1
    l = 0
    if (status == 0) then
        do u = 1, nv
            if (nbr(1, u) <= 0) cycle
            ! Start a loop here; visited vertices are marked by negating
            n_loop = n_loop + 1
            prev = u
            cur = nbr(1, u)
            l = l + 1
            if (l <= cap) loops(l) = u - 1
            nbr(1, u) = -nbr(1, u)
            do while (cur /= u)
                l = l + 1
                if (l <= cap) loops(l) = cur - 1
                if (abs(nbr(1, cur)) /= prev) then
                    nxt = abs(nbr(1, cur))
                else
                    nxt = abs(nbr(2, cur))
                end if
                nbr(1, cur) = -abs(nbr(1, cur))
                prev = cur
                cur = nxt
                if (l > nv) then
                    status = 1
                    exit
                end if
            end do
            if (n_loop + 1 <= cap + 1) loop_start(n_loop + 1) = l + 1
            if (status /= 0) exit
        end do
    end if

contains

    logical function degenerate(f0)
        integer, intent(in) :: f0
        degenerate = faces(1, f0) == faces(2, f0) .or. &
            faces(2, f0) == faces(3, f0) .or. faces(3, f0) == faces(1, f0)
    end function degenerate

    subroutine add_nbr(u0, v0, ok0)
        integer, intent(in) :: u0, v0
        logical, intent(out) :: ok0
        ok0 = .true.
        if (nbr(1, u0) == -1) then
            nbr(1, u0) = v0
        else if (nbr(2, u0) == -1) then
            nbr(2, u0) = v0
        else
            ok0 = .false.
        end if
    end subroutine add_nbr

end subroutine edge_topology


subroutine slide_rows(tn, data, theta0, pitch, it, factor, out, n, m, nvar, mm)
    ! Slide every row of a closed sheet round to start at theta0.
    !
    ! Row j has m columns, the last its first a pitch on, at theta tn(j, :)
    ! in pitches. Its nodes all move by the same amount, so that its first
    ! is at theta0: the row's spacing is kept, and its columns stay on one
    ! grid line each, straight wherever the rows' spacings are alike. Each
    ! interval is then split into factor equal parts, mm = factor (m - 1) + 1
    ! nodes in all, and every variable interpolated at the new nodes by a
    ! periodic monotone cubic Hermite through the row's own (Fritsch and
    ! Butland's slopes), which never overshoots the data either side of an
    ! interval. Variable it of the data is theta in radians,
    ! set from the new positions; the flow is periodic, so data a pitch
    ! round is the same.

    implicit none

    integer, intent(in) :: n, m, nvar, it, factor, mm
    real, intent(in) :: tn(n, m), data(n, m, nvar), theta0, pitch
    real, intent(inout) :: out(n, mm, nvar)

    integer :: j, k, r, q, nn, lo, hi, mid, k0, k1, km, kp, v
    real :: shift, a, b, tq, xs, h0, hm, hp, s, d0, dm, dp, m0, m1
    real :: h00, h10, h01, h11

    nn = m - 1
    do j = 1, n
        shift = tn(j, 1) - theta0
        do k = 1, nn
            a = tn(j, k) - shift
            b = tn(j, k + 1) - shift
            do r = 0, factor - 1
                q = (k - 1) * factor + r + 1
                tq = a + (b - a) * real(r) / real(factor)

                ! Interval of the row's own nodes holding tq, a pitch round
                xs = tn(j, 1) + modulo(tq - tn(j, 1), 1.0)
                lo = 1
                hi = nn + 1
                do while (hi - lo > 1)
                    mid = (lo + hi) / 2
                    if (tn(j, mid) <= xs) then
                        lo = mid
                    else
                        hi = mid
                    end if
                end do
                k0 = lo
                k1 = mod(lo, nn) + 1
                km = mod(lo - 2 + nn, nn) + 1
                kp = mod(lo + 1, nn) + 1

                ! Interval widths, wrapping a pitch at the ends of the row
                h0 = width(k0)
                hm = width(km)
                hp = width(k1)
                s = (xs - tn(j, k0)) / h0
                h00 = (2.0 * s - 3.0) * s * s + 1.0
                h10 = ((s - 2.0) * s + 1.0) * s
                h01 = (3.0 - 2.0 * s) * s * s
                h11 = (s - 1.0) * s * s

                do v = 1, nvar
                    d0 = (data(j, k1, v) - data(j, k0, v)) / h0
                    dm = (data(j, k0, v) - data(j, km, v)) / hm
                    dp = (data(j, kp, v) - data(j, k1, v)) / hp
                    m0 = slope(dm, d0, hm, h0)
                    m1 = slope(d0, dp, h0, hp)
                    out(j, q, v) = h00 * data(j, k0, v) + h10 * h0 * m0 &
                        + h01 * data(j, k1, v) + h11 * h0 * m1
                end do
                out(j, q, it) = tq * pitch
            end do
        end do

        ! The last column is the first a pitch on
        do v = 1, nvar
            out(j, mm, v) = out(j, 1, v)
        end do
        out(j, mm, it) = (theta0 + 1.0) * pitch
    end do

contains

    real function slope(dl, dr, hl, hr)
        ! Monotone slope at a node between differences dl and dr over widths
        ! hl and hr: zero at an extremum, else their weighted harmonic mean
        real, intent(in) :: dl, dr, hl, hr
        real :: wl, wr
        if (dl * dr <= 0.0) then
            slope = 0.0
        else
            wl = 2.0 * hr + hl
            wr = hr + 2.0 * hl
            slope = (wl + wr) / (wl / dl + wr / dr)
        end if
    end function slope

    real function width(k)
        ! Width of the row's interval from node k, the last one wrapping
        integer, intent(in) :: k
        if (k < nn) then
            width = tn(j, k + 1) - tn(j, k)
        else
            width = tn(j, 1) + 1.0 - tn(j, nn)
        end if
    end function width

end subroutine slide_rows
