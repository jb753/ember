! Real-gas equation of state, batched.
!
! THIS FILE IS OWNED BY `RealFluid` (fluid.py) AND IS NOT A SOLVER KERNEL.
! Nothing in the solver may call it. The equation of state lives behind the
! `_Fluid` interface, and `_Fluid.get_P_h_T` has a base implementation in numpy
! that keeps any fluid correct. This routine is only what `RealFluid`
! dispatches to for its own override, which is why the fitted coefficients
! arrive as arguments rather than being referenced from anywhere.
!
! It exists because the numpy path spends 98% of its time moving memory that
! the arithmetic never needed. Pressure and temperature come from the first
! partials of one entropy surface, each a Legendre polynomial in normalised
! density and internal energy. Evaluated pointwise, that is ~310 flops against
! a few hundred coefficients that fit in L1. Evaluated with numpy's legval2d it
! is the same arithmetic, but the inner call materialises an intermediate of
! shape (order, nodes) -- 41 MB per surface on a 97^3 grid, streamed out to
! memory and back to produce 3.7 MB of answer. This routine walks the nodes
! once and keeps the coefficients in cache.
!
! Pointwise and flat: the Fluid API is shape-agnostic (nodal volumes, patch
! faces, single values), so the caller flattens and passes the length.
!
! The coefficient arrays are the entropy surface already differentiated, in
! the normalised coordinates x = rho*xa + xb and y = u*ya + yb:
!
!     s_rho = legval2d(x, y, scx)*xa + legval(y, sl)/rho
!     s_u   = (legval2d(x, y, scy) + legval(y, sly)*log(rho))*ya
!
! and then T = 1/s_u, P = -rho^2*T*s_rho, h = u + P/rho. Differentiating drops
! one order per axis, so the four arrays have four different extents and each
! carries its own; the shared basis is built to the largest of them.
!
! The basis is built by the three-term Legendre recurrence and contracted
! explicitly, rather than by the Clenshaw sweep numpy uses. Clenshaw would have
! to run once per surface, whereas one basis serves all four here -- about a
! third of the flops. The two agree to a few ulp, not to the bit; the tests in
! test_update_primitive.py hold this path to the accuracy of the numpy one
! against an analytic gas, which is the property that matters.
!
! Nodes are processed in tiles, and every loop below runs over the tile rather
! than over the polynomial order. Taken a node at a time the recurrence is a
! dependency chain -- each term needs the previous two -- and GCC could
! vectorise nothing but a two-wide crumb of it; the contractions did vectorise,
! but over a trip count of eleven, which wastes most of a register. Across
! nodes the chains are independent and the trip count is the tile, so all
! eleven loops vectorise at 32 bytes and log() goes through libmvec eight at a
! time. Worth 126 -> 20 ns/node on a 97^3 grid.
!
! The coefficient surfaces are mostly zeros. A total-order fit keeps only the
! terms with i + j <= order, which after differentiating leaves a triangle:
! 55 of Sc_x's 121 coefficients at order 10, and 55 of Sc_y's 120. The caller
! passes the number of rows worth visiting in each column, so those terms are
! skipped rather than multiplied. The answer is unchanged to the bit, since
! the skipped coefficients are exactly zero. Worth 20 -> 15 ns/node.

subroutine set_P_h_T_real( &
    rho, u, &
    scx, nzx, scy, nzy, sl, sly, &
    xa, xb, ya, yb, &
    P, h, T, &
    nax, nbx, nay, nby, nsl, nsly, n &
    )

    implicit none

    integer, intent (in) :: nax, nbx, nay, nby, nsl, nsly, n
    real, intent (in)    :: rho(n), u(n)
    real, intent (in)    :: scx(nax, nbx), scy(nay, nby), sl(nsl), sly(nsly)

    ! Rows to visit in each column: everything past the last nonzero
    ! coefficient. A dense fit passes the full extents and nothing is skipped.
    integer, intent (in) :: nzx(nbx), nzy(nby)
    real, intent (in)    :: xa, xb, ya, yb
    real, intent (inout) :: P(n), h(n), T(n)

    integer, parameter :: MAXORD = 31

    ! Nodes per tile. Every inner loop below runs over the tile, so this sets
    ! the vector length the whole routine is built around, and it wants to be
    ! large: each inner loop costs a setup and a runtime alias check, and only
    ! the tile amortises them. Measured on a 97^3 grid, ns/node --
    !   8: 64.1   16: 53.6   32: 33.4   64: 23.7   128: 20.9   256: 20.0   512: 24.9
    ! The rise at 512 is the working set leaving L1. Only the first na rows of
    ! the basis are ever touched, about twelve of the thirty-two, so the live
    ! footprint at 256 is ~24 KB rather than the 64 KB the arrays reserve.
    integer, parameter :: NTILE = 256

    integer :: i, i0, j, nj, a, b, na, nb

    ! (a + 1)*P_{a+1}(x) = (2a + 1)*x*P_a(x) - a*P_{a-1}(x), folded at compile
    ! time. Written as the recurrence rather than as a wall of literals so the
    ! derivation stays readable.
    integer :: k
    real, parameter :: w1(1:MAXORD) = [(real(2 * k + 1) / real(k + 1), k = 1, MAXORD)]
    real, parameter :: w2(1:MAXORD) = [(real(k) / real(k + 1), k = 1, MAXORD)]

    real :: sc, srho, su, Ti, Pi
    real :: rhov(NTILE), uv(NTILE), xv(NTILE), yv(NTILE), lnrv(NTILE)
    real :: cxv(NTILE), cyv(NTILE), colv(NTILE), Mv(NTILE), Myv(NTILE)

    ! Node index first, so a basis term is contiguous across the tile and the
    ! loops that build and consume it vectorise along j. Per node the
    ! recurrence is a dependency chain and nothing can be done about it; across
    ! nodes the chains are independent, which is the whole reason for tiling.
    real :: Px(NTILE, 0:MAXORD), Qy(NTILE, 0:MAXORD)

    na = max(nax, nay)
    nb = max(max(nbx, nby), max(nsl, nsly))

    do i0 = 1, n, NTILE

        nj = min(NTILE, n - i0 + 1)

        do j = 1, nj
            rhov(j) = rho(i0 + j - 1)
            uv(j) = u(i0 + j - 1)
            xv(j) = rhov(j) * xa + xb
            yv(j) = uv(j) * ya + yb
            lnrv(j) = log(rhov(j))
        end do

        do j = 1, nj
            Px(j, 0) = 1.0
            Qy(j, 0) = 1.0
        end do

        if (na > 1) then
            do j = 1, nj
                Px(j, 1) = xv(j)
            end do
        end if

        do a = 1, na - 2
            do j = 1, nj
                Px(j, a + 1) = w1(a) * xv(j) * Px(j, a) - w2(a) * Px(j, a - 1)
            end do
        end do

        if (nb > 1) then
            do j = 1, nj
                Qy(j, 1) = yv(j)
            end do
        end if

        do b = 1, nb - 2
            do j = 1, nj
                Qy(j, b + 1) = w1(b) * yv(j) * Qy(j, b) - w2(b) * Qy(j, b - 1)
            end do
        end do

        ! Each coefficient is a scalar broadcast across the tile, and the basis
        ! term it multiplies is a contiguous run of nj floats. The column sum
        ! is held back so the inner loop is a single fused multiply-add.
        do j = 1, nj
            cxv(j) = 0.0
            cyv(j) = 0.0
        end do

        do b = 1, nbx
            if (nzx(b) == 0) cycle
            do j = 1, nj
                colv(j) = 0.0
            end do
            do a = 1, nzx(b)
                sc = scx(a, b)
                do j = 1, nj
                    colv(j) = colv(j) + sc * Px(j, a - 1)
                end do
            end do
            do j = 1, nj
                cxv(j) = cxv(j) + colv(j) * Qy(j, b - 1)
            end do
        end do

        do b = 1, nby
            if (nzy(b) == 0) cycle
            do j = 1, nj
                colv(j) = 0.0
            end do
            do a = 1, nzy(b)
                sc = scy(a, b)
                do j = 1, nj
                    colv(j) = colv(j) + sc * Px(j, a - 1)
                end do
            end do
            do j = 1, nj
                cyv(j) = cyv(j) + colv(j) * Qy(j, b - 1)
            end do
        end do

        ! The log multiplier and its derivative, both functions of y alone.
        do j = 1, nj
            Mv(j) = 0.0
            Myv(j) = 0.0
        end do

        do b = 1, nsl
            sc = sl(b)
            do j = 1, nj
                Mv(j) = Mv(j) + sc * Qy(j, b - 1)
            end do
        end do

        do b = 1, nsly
            sc = sly(b)
            do j = 1, nj
                Myv(j) = Myv(j) + sc * Qy(j, b - 1)
            end do
        end do

        ! Entropy partials, then the state. Operation order follows RealFluid's
        ! _partials1 and get_P_h_T.
        do j = 1, nj
            srho = cxv(j) * xa + Mv(j) / rhov(j)
            su = (cyv(j) + Myv(j) * lnrv(j)) * ya
            Ti = 1.0 / su
            Pi = -(rhov(j) * rhov(j)) * Ti * srho
            i = i0 + j - 1
            T(i) = Ti
            P(i) = Pi
            h(i) = uv(j) + Pi / rhov(j)
        end do

    end do

end subroutine set_P_h_T_real
