! Nine-point stencils of the elliptic grid generation equations

subroutine ttm_stencil(X, sg, ss, PQ, pitch, w, src, n_node, n_reg)
    ! Coefficients and right-hand side of the Thompson-Thames-Mastin equations
    ! in Thomas-Middlecoff form,
    !
    !     alpha (r_xixi + P r_xi) - 2 beta r_xieta + gamma (r_etaeta + Q r_eta) = 0,
    !
    ! frozen at the current node coordinates, at each of n_reg stencils.
    !
    ! A stencil is nine node numbers in the order P, E, NE, N, NW, W, SW, S, SE,
    ! taken from any block structure, so the kernel knows nothing of blocks:
    ! it gathers through the numbers it is given. Each neighbour is seen with
    ! its own pitch shift in y, for a stencil that straddles a periodic
    ! boundary.
    !
    ! Parameters
    ! ----------
    ! X : double(n_node, 2)
    !     Node coordinates (x, y).
    ! sg : integer(n_reg, 9)
    !     Node numbers of each stencil, counted from zero.
    ! ss : integer(n_reg, 9)
    !     Pitch shift of each stencil node, in whole pitches.
    ! PQ : double(n_reg, 2)
    !     Control functions P and Q at each stencil centre.
    ! pitch : double
    !     Period in y, or zero.
    ! w : double(n_reg, 9), out
    !     Stencil coefficients normalised to a unit centre coefficient, so that
    !     the row of the linear system is sum_k w(n, k) X(sg(n, k)) = src(n).
    ! src : double(n_reg, 2), out
    !     Right-hand side: the lagged control function source, and in y the
    !     constant the pitch shifts of the neighbours contribute.

    implicit none

    integer, intent(in) :: n_node, n_reg
    double precision, intent(in) :: X(n_node, 2)
    integer, intent(in) :: sg(n_reg, 9), ss(n_reg, 9)
    double precision, intent(in) :: PQ(n_reg, 2)
    double precision, intent(in) :: pitch
    double precision, intent(out) :: w(n_reg, 9)
    double precision, intent(out) :: src(n_reg, 2)

    integer :: n, k
    double precision :: xE, yE, xN, yN, xW, yW, xS, yS
    double precision :: xi_x, xi_y, eta_x, eta_y
    double precision :: a, b, c, rdiag, shift_sum

    do n = 1, n_reg
        ! The four edge neighbours, which are all the metrics need.
        xE = X(sg(n, 2) + 1, 1)
        yE = X(sg(n, 2) + 1, 2) + ss(n, 2) * pitch
        xN = X(sg(n, 4) + 1, 1)
        yN = X(sg(n, 4) + 1, 2) + ss(n, 4) * pitch
        xW = X(sg(n, 6) + 1, 1)
        yW = X(sg(n, 6) + 1, 2) + ss(n, 6) * pitch
        xS = X(sg(n, 8) + 1, 1)
        yS = X(sg(n, 8) + 1, 2) + ss(n, 8) * pitch

        xi_x = 0.5d0 * (xE - xW)
        xi_y = 0.5d0 * (yE - yW)
        eta_x = 0.5d0 * (xN - xS)
        eta_y = 0.5d0 * (yN - yS)

        a = eta_x * eta_x + eta_y * eta_y
        b = xi_x * eta_x + xi_y * eta_y
        c = xi_x * xi_x + xi_y * xi_y
        rdiag = 1.0d0 / (2.0d0 * (a + c))

        ! Unit centre; edge neighbours from the second differences, corners
        ! from the cross derivative.
        w(n, 1) = 1.0d0
        w(n, 2) = -a * rdiag
        w(n, 3) = 0.5d0 * b * rdiag
        w(n, 4) = -c * rdiag
        w(n, 5) = -0.5d0 * b * rdiag
        w(n, 6) = -a * rdiag
        w(n, 7) = 0.5d0 * b * rdiag
        w(n, 8) = -c * rdiag
        w(n, 9) = -0.5d0 * b * rdiag

        shift_sum = 0.0d0
        do k = 2, 9
            shift_sum = shift_sum + w(n, k) * ss(n, k)
        end do

        src(n, 1) = (a * PQ(n, 1) * xi_x + c * PQ(n, 2) * eta_x) * rdiag
        src(n, 2) = (a * PQ(n, 1) * xi_y + c * PQ(n, 2) * eta_y) * rdiag &
            - shift_sum * pitch
    end do

end subroutine ttm_stencil

subroutine winslow_sweeps(X, sg, ss, &
        irr, irr_ptr, irr_col, irr_w, irr_c, &
        sld, sld_ptr, sld_col, sld_w, sld_c, &
        cp_node, cp_corner, cp_dy, cp_f, &
        wn0, wn1, dy0, dy1, line, s_line, floor_, seg0, seg1, &
        pitch, n_sweep, m_irr, m_sld, m_cp, m_w, &
        n_node, n_reg, n_irr, n_irr_nb, n_sld, n_sld_nb, n_cp, n_w, n_line)
    ! Point-Jacobi sweeps of Winslow's equations, in place on X.
    !
    ! Each sweep, in turn:
    !   1. every regular node to the solution of its nine-point stencil, and
    !      every singular point to its weighted row, both from the coordinates
    !      the sweep started from;
    !   2. every sliding node to the weighted mean y of its row, plus a
    !      constant, from the coordinates after 1;
    !   3. every node near a corner drawn towards it, by a factor it keeps of
    !      its distance;
    !   4. every wall node slid along the wall line to the foot of the
    !      perpendicular from the node outside it, no further than half way to
    !      either neighbour, the gaps between them kept above a floor, the wall
    !      nodes closing in a ring whose last is its first a turn on.
    !
    ! As in ttm_stencil, the kernel knows nothing of blocks: it works through
    ! the node numbers and pitch shifts it is given, counted from zero. Rows
    ! of singular and sliding nodes are compressed sparse rows. The wall line
    ! is closed, its last point its first, and the nearest point on it to a
    ! node is found by stepping from where it was the sweep before, in seg0
    ! and seg1, which the caller seeds.
    !
    ! Coordinates are single precision; the wall line and arc lengths along it
    ! are double. The groups' counts m_irr, m_sld, m_cp and m_w are given apart
    ! from their arrays' lengths, which f2py will not let be zero: an empty
    ! group comes as one unused entry.

    implicit none

    integer, intent(in) :: n_node, n_reg, n_irr, n_irr_nb, n_sld, n_sld_nb
    integer, intent(in) :: n_cp, n_w, n_line, n_sweep
    integer, intent(in) :: m_irr, m_sld, m_cp, m_w
    real, intent(inout) :: X(n_node, 2)
    integer, intent(in) :: sg(n_reg, 9), ss(n_reg, 9)
    integer, intent(in) :: irr(n_irr), irr_ptr(n_irr + 1), irr_col(n_irr_nb)
    real, intent(in) :: irr_w(n_irr_nb), irr_c(n_irr)
    integer, intent(in) :: sld(n_sld), sld_ptr(n_sld + 1), sld_col(n_sld_nb)
    real, intent(in) :: sld_w(n_sld_nb), sld_c(n_sld)
    integer, intent(in) :: cp_node(n_cp), cp_corner(n_cp)
    real, intent(in) :: cp_dy(n_cp), cp_f(n_cp)
    integer, intent(in) :: wn0(n_w), wn1(n_w)
    real, intent(in) :: dy0(n_w), dy1(n_w)
    double precision, intent(in) :: line(n_line, 2), s_line(n_line)
    double precision, intent(in) :: floor_(n_w)
    integer, intent(inout) :: seg0(n_w), seg1(n_w)
    real, intent(in) :: pitch

    real :: Xnew(n_reg, 2), Xirr(n_irr, 2), y_sld(n_sld)
    double precision :: s0(n_w), s1(n_w), s_new(n_w), s_total, v, lo, hi
    real :: xE, yE, xN, yN, xW, yW, xS, yS, xNE, yNE, xNW, yNW, xSW, ySW
    real :: xSE, ySE, xi_x, xi_y, eta_x, eta_y, a, b, c, r
    real :: xc, yc
    double precision :: px, py, fx, fy
    integer :: it, n, i, q, half

    s_total = s_line(n_line)
    half = m_w / 2

    do it = 1, n_sweep

        ! 1. Regular nodes and singular points, from X as it is.
        do n = 1, n_reg
            xE = X(sg(n, 2) + 1, 1)
            yE = X(sg(n, 2) + 1, 2) + ss(n, 2) * pitch
            xNE = X(sg(n, 3) + 1, 1)
            yNE = X(sg(n, 3) + 1, 2) + ss(n, 3) * pitch
            xN = X(sg(n, 4) + 1, 1)
            yN = X(sg(n, 4) + 1, 2) + ss(n, 4) * pitch
            xNW = X(sg(n, 5) + 1, 1)
            yNW = X(sg(n, 5) + 1, 2) + ss(n, 5) * pitch
            xW = X(sg(n, 6) + 1, 1)
            yW = X(sg(n, 6) + 1, 2) + ss(n, 6) * pitch
            xSW = X(sg(n, 7) + 1, 1)
            ySW = X(sg(n, 7) + 1, 2) + ss(n, 7) * pitch
            xS = X(sg(n, 8) + 1, 1)
            yS = X(sg(n, 8) + 1, 2) + ss(n, 8) * pitch
            xSE = X(sg(n, 9) + 1, 1)
            ySE = X(sg(n, 9) + 1, 2) + ss(n, 9) * pitch

            xi_x = 0.5 * (xE - xW)
            xi_y = 0.5 * (yE - yW)
            eta_x = 0.5 * (xN - xS)
            eta_y = 0.5 * (yN - yS)
            a = eta_x * eta_x + eta_y * eta_y
            b = xi_x * eta_x + xi_y * eta_y
            c = xi_x * xi_x + xi_y * xi_y
            r = 1.0 / (2.0 * (a + c))

            Xnew(n, 1) = (a * (xE + xW) + c * (xN + xS) &
                - 0.5 * b * (xNE - xSE + xSW - xNW)) * r
            Xnew(n, 2) = (a * (yE + yW) + c * (yN + yS) &
                - 0.5 * b * (yNE - ySE + ySW - yNW)) * r
        end do

        do n = 1, m_irr
            Xirr(n, 1) = 0.0
            Xirr(n, 2) = irr_c(n)
            do q = irr_ptr(n) + 1, irr_ptr(n + 1)
                Xirr(n, 1) = Xirr(n, 1) + irr_w(q) * X(irr_col(q) + 1, 1)
                Xirr(n, 2) = Xirr(n, 2) + irr_w(q) * X(irr_col(q) + 1, 2)
            end do
        end do

        do n = 1, n_reg
            X(sg(n, 1) + 1, :) = Xnew(n, :)
        end do
        do n = 1, m_irr
            X(irr(n) + 1, :) = Xirr(n, :)
        end do

        ! 2. Sliding nodes in y.
        do n = 1, m_sld
            y_sld(n) = sld_c(n)
            do q = sld_ptr(n) + 1, sld_ptr(n + 1)
                y_sld(n) = y_sld(n) + sld_w(q) * X(sld_col(q) + 1, 2)
            end do
        end do
        do n = 1, m_sld
            X(sld(n) + 1, 2) = y_sld(n)
        end do

        ! 3. Nodes near the corners drawn in.
        do n = 1, m_cp
            xc = X(cp_corner(n) + 1, 1)
            yc = X(cp_corner(n) + 1, 2) + cp_dy(n)
            X(cp_node(n) + 1, 1) = xc + (X(cp_node(n) + 1, 1) - xc) * cp_f(n)
            X(cp_node(n) + 1, 2) = yc + (X(cp_node(n) + 1, 2) - yc) * cp_f(n)
        end do

        ! 4. Wall nodes along the wall line.
        if (m_w > 0) then
            do i = 1, m_w
                px = dble(X(wn0(i) + 1, 1))
                py = dble(X(wn0(i) + 1, 2) + dy0(i))
                call nearest_on_line(px, py, seg0(i), s0(i))
                px = dble(X(wn1(i) + 1, 1))
                py = dble(X(wn1(i) + 1, 2) + dy1(i))
                call nearest_on_line(px, py, seg1(i), s1(i))
            end do

            ! Each on the side of the cut its node is.
            do i = 1, half
                if (s0(i) > 0.75d0 * s_total) s0(i) = s0(i) - s_total
                if (s1(i) > 0.75d0 * s_total) s1(i) = s1(i) - s_total
            end do
            do i = half + 1, m_w
                if (s0(i) < 0.25d0 * s_total) s0(i) = s0(i) + s_total
                if (s1(i) < 0.25d0 * s_total) s1(i) = s1(i) + s_total
            end do
            s0(m_w) = s0(1) + s_total
            s1(m_w) = s1(1) + s_total

            do i = 1, m_w
                if (i == 1) then
                    lo = 0.5d0 * (s0(m_w - 1) - s_total + s0(1))
                else
                    lo = 0.5d0 * (s0(i - 1) + s0(i))
                end if
                if (i == m_w) then
                    hi = 0.5d0 * (s0(m_w) + s0(2) + s_total)
                else
                    hi = 0.5d0 * (s0(i) + s0(i + 1))
                end if
                s_new(i) = min(max(s1(i), lo), hi)
            end do

            ! No gap below its floor: forward from the cut, back from a turn on.
            v = s_new(1) - floor_(1)
            do i = 2, m_w
                v = max(v, s_new(i) - floor_(i))
                s_new(i) = v + floor_(i)
            end do
            s_new(m_w) = s_new(1) + s_total
            v = s_new(m_w) - floor_(m_w)
            do i = m_w - 1, 1, -1
                v = min(v, s_new(i) - floor_(i))
                s_new(i) = v + floor_(i)
            end do

            do i = 1, m_w
                call point_at(modulo(s_new(i), s_total), fx, fy)
                X(wn0(i) + 1, 1) = real(fx)
                X(wn0(i) + 1, 2) = real(fy) - dy0(i)
            end do
        end if

    end do

contains

    subroutine nearest_on_line(px, py, k, s)
        ! Arc length of the point on the line nearest (px, py), stepping from
        ! sample k to the nearest sample, then on the segment either side.
        double precision, intent(in) :: px, py
        integer, intent(inout) :: k
        double precision, intent(out) :: s
        integer :: m, ka, kb, j
        double precision :: d, dn, ax, ay, bx, by, t, best

        m = n_line - 1
        k = modulo(k, m)
        d = dist2(k, px, py)
        do
            dn = dist2(modulo(k + 1, m), px, py)
            if (dn >= d) exit
            k = modulo(k + 1, m)
            d = dn
        end do
        do
            dn = dist2(modulo(k - 1, m), px, py)
            if (dn >= d) exit
            k = modulo(k - 1, m)
            d = dn
        end do

        best = huge(1.0d0)
        s = s_line(k + 1)
        do j = -1, 0
            ka = k + j
            kb = ka + 1
            ! Samples counted round the closed line, the last being the first.
            if (ka < 0) then
                ka = m - 1
                kb = m
            end if
            ax = line(ka + 1, 1)
            ay = line(ka + 1, 2)
            bx = line(kb + 1, 1)
            by = line(kb + 1, 2)
            t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) &
                / ((bx - ax)**2 + (by - ay)**2)
            t = min(max(t, 0.0d0), 1.0d0)
            dn = (ax + t * (bx - ax) - px)**2 + (ay + t * (by - ay) - py)**2
            if (dn < best) then
                best = dn
                s = s_line(ka + 1) + t * (s_line(kb + 1) - s_line(ka + 1))
            end if
        end do
    end subroutine nearest_on_line

    double precision function dist2(kk, qx, qy)
        ! Squared distance from (qx, qy) to sample kk of the line.
        integer, intent(in) :: kk
        double precision, intent(in) :: qx, qy
        dist2 = (line(kk + 1, 1) - qx)**2 + (line(kk + 1, 2) - qy)**2
    end function dist2

    subroutine point_at(s, qx, qy)
        ! The point on the line at arc length s, from zero to its whole length.
        double precision, intent(in) :: s
        double precision, intent(out) :: qx, qy
        integer :: lo_, hi_, mid
        double precision :: t
        lo_ = 1
        hi_ = n_line
        do while (hi_ - lo_ > 1)
            mid = (lo_ + hi_) / 2
            if (s_line(mid) <= s) then
                lo_ = mid
            else
                hi_ = mid
            end if
        end do
        t = (s - s_line(lo_)) / (s_line(hi_) - s_line(lo_))
        qx = line(lo_, 1) + t * (line(hi_, 1) - line(lo_, 1))
        qy = line(lo_, 2) + t * (line(hi_, 2) - line(lo_, 2))
    end subroutine point_at

end subroutine winslow_sweeps
