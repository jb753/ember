! Kinematic primitives from the conserved variables, in one pass.
!
! This is the fluid-agnostic half of the primitive evaluation: velocity is
! defined by the conserved variables for ANY fluid (Vx = rhoVx/rho and so on),
! so nothing here touches the equation of state. The thermodynamic half --
! (rho, u) -> P, h, T -- stays behind the Fluid interface, where a real-gas or
! tabulated fluid can override it. See `_Fluid.get_P_h_T` in fluid.py.
!
! WHY THIS KERNEL EXISTS. Evaluated lazily in numpy, this chain is five
! separate full-volume passes -- Vxrt, then halfVsq, then u -- writing and
! re-reading halfVsq and u purely as intermediates, ~120 B/cell in total. An
! 8-rank contended profile of a 1M-cell run put the whole primitive chain at
! ~17% of a timestep. Fused, halfVsq and u never leave registers on the way to
! being stored, and the pass reads 24 B and writes 20 B per node.
!
! halfVsq IS still stored, despite being an intermediate here: `ho_nd` needs it
! (ho = h + halfVsq, and h comes from the fluid), and two non-hot consumers in
! block.py read it directly.

subroutine set_primitive_kinematic( &
    cons, r, &
    vxrt, u, halfvsq, &
    ni, nj, nk &
    )

    implicit none

    integer, intent (in) :: ni, nj, nk
    ! Conserved: rho, rho*Vx, rho*Vr, rho*r*Vt, rho*e.
    real, intent (in)    :: cons(ni, nj, nk, 5)
    real, intent (in)    :: r(ni, nj, nk)
    real, intent (inout) :: vxrt(ni, nj, nk, 3)
    real, intent (inout) :: u(ni, nj, nk)
    real, intent (inout) :: halfvsq(ni, nj, nk)

    integer :: i, j, k
    real :: rho, Vx, Vr, Vt, hv

    ! Divisions written out rather than hoisted into a reciprocal: this mirrors
    ! the numpy path (block.py `_Vxrt_nd_uninit`, which divides by rho and then
    ! by r), and -freciprocal-math lets the compiler make that transformation
    ! itself if it pays.
    do k = 1, nk
    do j = 1, nj
    do i = 1, ni
        rho = cons(i, j, k, 1)
        Vx = cons(i, j, k, 2) / rho
        Vr = cons(i, j, k, 3) / rho
        ! Vt = rhorVt/rho, then /r -- the same two-step split numpy uses to
        ! avoid materialising a rho*r temporary.
        Vt = cons(i, j, k, 4) / rho / r(i, j, k)
        hv = 0.5e0 * (Vx * Vx + Vr * Vr + Vt * Vt)
        vxrt(i, j, k, 1) = Vx
        vxrt(i, j, k, 2) = Vr
        vxrt(i, j, k, 3) = Vt
        halfvsq(i, j, k) = hv
        u(i, j, k) = cons(i, j, k, 5) / rho - hv
    end do
    end do
    end do

end subroutine set_primitive_kinematic
