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
    integer :: i0i, j0j, k0k
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
        k0k = k0(kn);  wk0 = 1.0 - wk(kn);  wk1 = wk(kn)
        do jn = 1, nj_new
            j0j = j0(jn);  wj0 = 1.0 - wj(jn);  wj1 = wj(jn)
            do in = 1, ni_new
                i0i = i0(in);  wi0 = 1.0 - wi(in);  wi1 = wi(in)
                do n = 1, nprop
                    c000 = data_in(i0i,   j0j,   k0k,   n)
                    c100 = data_in(i0i+1, j0j,   k0k,   n)
                    c010 = data_in(i0i,   j0j+1, k0k,   n)
                    c110 = data_in(i0i+1, j0j+1, k0k,   n)
                    c001 = data_in(i0i,   j0j,   k0k+1, n)
                    c101 = data_in(i0i+1, j0j,   k0k+1, n)
                    c011 = data_in(i0i,   j0j+1, k0k+1, n)
                    c111 = data_in(i0i+1, j0j+1, k0k+1, n)
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
