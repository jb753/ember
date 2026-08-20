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

subroutine set_P_h_T_real( &
    rho, u, &
    scx, scy, sl, sly, &
    xa, xb, ya, yb, &
    P, h, T, &
    nax, nbx, nay, nby, nsl, nsly, n &
    )

    implicit none

    integer, intent (in) :: nax, nbx, nay, nby, nsl, nsly, n
    real, intent (in)    :: rho(n), u(n)
    real, intent (in)    :: scx(nax, nbx), scy(nay, nby), sl(nsl), sly(nsly)
    real, intent (in)    :: xa, xb, ya, yb
    real, intent (inout) :: P(n), h(n), T(n)

    integer, parameter :: MAXORD = 31
    integer :: i, a, b, na, nb

    ! (a + 1)*P_{a+1}(x) = (2a + 1)*x*P_a(x) - a*P_{a-1}(x), folded at compile
    ! time. Written as the recurrence rather than as a wall of literals so the
    ! derivation stays readable.
    integer :: k
    real, parameter :: w1(1:MAXORD) = [(real(2 * k + 1) / real(k + 1), k = 1, MAXORD)]
    real, parameter :: w2(1:MAXORD) = [(real(k) / real(k + 1), k = 1, MAXORD)]

    real :: x, y, lnr, rhoi, srho, su, Ti, Pi, cx, cy, col, M, My

    ! Fixed size, not automatic: an automatic array is an alloca per call that
    ! the optimiser treats as clobbering memory. The caller guarantees the
    ! order fits (see RealFluid.get_P_h_T).
    real :: Px(0:MAXORD)
    real :: Qy(0:MAXORD)

    na = max(nax, nay)
    nb = max(max(nbx, nby), max(nsl, nsly))

    do i = 1, n

        rhoi = rho(i)
        x = rhoi * xa + xb
        y = u(i) * ya + yb
        lnr = log(rhoi)

        Px(0) = 1.0
        if (na > 1) Px(1) = x
        do a = 1, na - 2
            Px(a + 1) = w1(a) * x * Px(a) - w2(a) * Px(a - 1)
        end do

        Qy(0) = 1.0
        if (nb > 1) Qy(1) = y
        do b = 1, nb - 2
            Qy(b + 1) = w1(b) * y * Qy(b) - w2(b) * Qy(b - 1)
        end do

        ! Column by column: the coefficients are column-major, so the inner
        ! loop is a contiguous dot product, and each Qy is touched once per
        ! column instead of once per term.
        cx = 0.0
        do b = 1, nbx
            col = 0.0
            do a = 1, nax
                col = col + scx(a, b) * Px(a - 1)
            end do
            cx = cx + col * Qy(b - 1)
        end do

        cy = 0.0
        do b = 1, nby
            col = 0.0
            do a = 1, nay
                col = col + scy(a, b) * Px(a - 1)
            end do
            cy = cy + col * Qy(b - 1)
        end do

        ! The log multiplier and its derivative, both functions of y alone.
        M = 0.0
        do b = 1, nsl
            M = M + sl(b) * Qy(b - 1)
        end do

        My = 0.0
        do b = 1, nsly
            My = My + sly(b) * Qy(b - 1)
        end do

        ! Entropy partials, then the state. Operation order follows RealFluid's
        ! _partials1 and get_P_h_T.
        srho = cx * xa + M / rhoi
        su = (cy + My * lnr) * ya

        Ti = 1.0 / su
        Pi = -(rhoi * rhoi) * Ti * srho

        T(i) = Ti
        P(i) = Pi
        h(i) = u(i) + Pi / rhoi

    end do

end subroutine set_P_h_T_real
