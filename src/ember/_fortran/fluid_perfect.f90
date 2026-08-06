! Perfect-gas equation of state, batched.
!
! THIS FILE IS OWNED BY `PerfectFluid` (fluid.py) AND IS NOT A SOLVER KERNEL.
! Nothing in the solver may call it. The equation of state lives behind the
! `_Fluid` interface so that other fluids -- real gas, tabulated -- can be
! added, and `_Fluid.get_P_h_T` has a base implementation in numpy that keeps
! any such fluid correct. This routine is only what `PerfectFluid` dispatches
! to for its own override, which is why the gas constants arrive as arguments
! rather than being referenced from anywhere.
!
! It exists because P, h and T share almost all of their work: T = u/cv + T_dtm
! is computed inside P, and evaluating the three separately re-reads rho and u
! three times over. One pass reads 8 B and writes 12 B per node instead of
! ~36 B of traffic across three passes.
!
! Pointwise and flat: the Fluid API is shape-agnostic (nodal volumes, patch
! faces, single values), so the caller flattens and passes the length.

subroutine set_P_h_T_perfect( &
    rho, u, &
    cv, T_dtm, Rgas, gamma, &
    P, h, T, &
    n &
    )

    implicit none

    integer, intent (in) :: n
    real, intent (in)    :: rho(n), u(n)
    real, intent (in)    :: cv, T_dtm, Rgas, gamma
    real, intent (inout) :: P(n), h(n), T(n)

    integer :: i
    real :: Ti, h_offset

    ! Loop-invariant part of the enthalpy datum offset.
    h_offset = T_dtm * Rgas

    ! Operation order matches PerfectFluid's numpy implementations exactly:
    !   get_T:  u/cv, then + T_dtm
    !   get_P:  (u/cv + T_dtm), then * rho, then * Rgas
    !   get_h:  gamma*u, then + T_dtm*Rgas
    do i = 1, n
        Ti = u(i) / cv + T_dtm
        T(i) = Ti
        P(i) = Ti * rho(i) * Rgas
        h(i) = gamma * u(i) + h_offset
    end do

end subroutine set_P_h_T_perfect
