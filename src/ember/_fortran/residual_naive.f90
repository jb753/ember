! =====================================================================
! set_residual_naive -- the boring, textbook inviscid residual.
! Benchmark-only companion to production set_residual (residual.f90);
! nothing calls it in production.
!
! This is deliberately the most obvious thing that could work, with no
! optimisation of any kind:
!
!   1. Compute ALL i-face flows into a full-volume array.
!   2. Compute ALL j-face flows into a full-volume array.
!   3. Compute ALL k-face flows into a full-volume array.
!   4. Difference them into dU, plus the body force.
!
! Four separate full-volume passes. No tiling, no cache blocking, no
! rolling buffers, no direction fusion, no hand-unrolled corners, no
! scalarized pm/mf, no slab bookkeeping. The face loops are written the
! plainest way: a 4-corner average accumulated by a helper over pm(6) and
! mf(3), exactly as correct_cusp_kface_du (the one part of the production
! file that was never worth optimising) already does.
!
! Its purpose is to be the reference point the other variants are read
! against: sections 17 and 18 priced two rewrites against the *optimised*
! production kernel, but never established what the unoptimised starting
! point actually costs. That number is what says whether the accumulated
! machinery in residual.f90 bought a lot or a little.
!
! Scratch: three full-volume face-flow arrays (~56 MB at 1M cells), which
! is far more than any other variant -- deliberately so. It exceeds
! tau_q_halo alone, so the caller carves flow_k from block.scratch.
! =====================================================================
module residual_naive_helpers
    implicit none
    private
    public :: iface_flows, jface_flows, kface_flows

contains

    ! The three routines below are near-identical on purpose: each is the
    ! straightforward "loop over every face of this direction, average the
    ! four corners, contract with dA" reading. Deduplicating them is
    ! exactly the kind of cleverness this variant exists to omit.

    subroutine iface_flows(vx, vr, vt, ho, P, P_offset, r, cons, Omega, &
                           dA, wall_lo, wall_hi, flow, ni, nj, nk)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: vx(ni,nj,nk), vr(ni,nj,nk), vt(ni,nj,nk)
        real, intent(in) :: ho(ni,nj,nk), P(ni,nj,nk), r(ni,nj,nk)
        real, intent(in) :: cons(ni,nj,nk,5)
        real, intent(in) :: P_offset, Omega
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: wall_lo(nj-1, nk-1), wall_hi(nj-1, nk-1)
        real, intent(inout) :: flow(ni, nj-1, nk-1, 5)

        integer :: i, j, k
        real :: pm(6), mf(3), mdot, w

        do k = 1, nk-1
            do j = 1, nj-1
                do i = 1, ni
                    if (i == 1) then
                        w = wall_lo(j,k)
                    else if (i == ni) then
                        w = wall_hi(j,k)
                    else
                        w = 1.0e0
                    end if
                    pm = 0.0e0
                    mf = 0.0e0
                    call accum(pm, mf, i, j,   k,   w)
                    call accum(pm, mf, i, j+1, k,   w)
                    call accum(pm, mf, i, j,   k+1, w)
                    call accum(pm, mf, i, j+1, k+1, w)
                    mdot = mf(1)*dA(1,i,j,k) + mf(2)*dA(2,i,j,k) &
                         + mf(3)*dA(3,i,j,k)
                    flow(i,j,k,1) = mdot
                    flow(i,j,k,2) = pm(1)*mdot + pm(5)*dA(1,i,j,k)
                    flow(i,j,k,3) = pm(2)*mdot + pm(5)*dA(2,i,j,k)
                    flow(i,j,k,4) = pm(3)*mdot + pm(6)*dA(3,i,j,k)
                    flow(i,j,k,5) = pm(4)*mdot + Omega*pm(6)*dA(3,i,j,k)
                end do
            end do
        end do

    contains
        pure subroutine accum(pm, mf, i, j, k, wfac)
            real, intent(inout) :: pm(6), mf(3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real :: dp, w
            dp = P(i,j,k) - P_offset
            pm(1) = pm(1) + 0.25e0*vx(i,j,k)
            pm(2) = pm(2) + 0.25e0*vr(i,j,k)
            pm(3) = pm(3) + 0.25e0*r(i,j,k)*vt(i,j,k)
            pm(4) = pm(4) + 0.25e0*ho(i,j,k)
            pm(5) = pm(5) + 0.25e0*dp
            pm(6) = pm(6) + 0.25e0*r(i,j,k)*dp
            w = 0.25e0*wfac
            mf(1) = mf(1) + w*cons(i,j,k,2)
            mf(2) = mf(2) + w*cons(i,j,k,3)
            mf(3) = mf(3) + w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k))
        end subroutine accum
    end subroutine iface_flows


    subroutine jface_flows(vx, vr, vt, ho, P, P_offset, r, cons, Omega, &
                           dA, wall_lo, wall_hi, flow, ni, nj, nk)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: vx(ni,nj,nk), vr(ni,nj,nk), vt(ni,nj,nk)
        real, intent(in) :: ho(ni,nj,nk), P(ni,nj,nk), r(ni,nj,nk)
        real, intent(in) :: cons(ni,nj,nk,5)
        real, intent(in) :: P_offset, Omega
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: wall_lo(ni-1, nk-1), wall_hi(ni-1, nk-1)
        real, intent(inout) :: flow(ni-1, nj, nk-1, 5)

        integer :: i, j, k
        real :: pm(6), mf(3), mdot, w

        do k = 1, nk-1
            do j = 1, nj
                do i = 1, ni-1
                    if (j == 1) then
                        w = wall_lo(i,k)
                    else if (j == nj) then
                        w = wall_hi(i,k)
                    else
                        w = 1.0e0
                    end if
                    pm = 0.0e0
                    mf = 0.0e0
                    call accum(pm, mf, i,   j, k,   w)
                    call accum(pm, mf, i+1, j, k,   w)
                    call accum(pm, mf, i,   j, k+1, w)
                    call accum(pm, mf, i+1, j, k+1, w)
                    mdot = mf(1)*dA(1,i,j,k) + mf(2)*dA(2,i,j,k) &
                         + mf(3)*dA(3,i,j,k)
                    flow(i,j,k,1) = mdot
                    flow(i,j,k,2) = pm(1)*mdot + pm(5)*dA(1,i,j,k)
                    flow(i,j,k,3) = pm(2)*mdot + pm(5)*dA(2,i,j,k)
                    flow(i,j,k,4) = pm(3)*mdot + pm(6)*dA(3,i,j,k)
                    flow(i,j,k,5) = pm(4)*mdot + Omega*pm(6)*dA(3,i,j,k)
                end do
            end do
        end do

    contains
        pure subroutine accum(pm, mf, i, j, k, wfac)
            real, intent(inout) :: pm(6), mf(3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real :: dp, w
            dp = P(i,j,k) - P_offset
            pm(1) = pm(1) + 0.25e0*vx(i,j,k)
            pm(2) = pm(2) + 0.25e0*vr(i,j,k)
            pm(3) = pm(3) + 0.25e0*r(i,j,k)*vt(i,j,k)
            pm(4) = pm(4) + 0.25e0*ho(i,j,k)
            pm(5) = pm(5) + 0.25e0*dp
            pm(6) = pm(6) + 0.25e0*r(i,j,k)*dp
            w = 0.25e0*wfac
            mf(1) = mf(1) + w*cons(i,j,k,2)
            mf(2) = mf(2) + w*cons(i,j,k,3)
            mf(3) = mf(3) + w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k))
        end subroutine accum
    end subroutine jface_flows


    subroutine kface_flows(vx, vr, vt, ho, P, P_offset, r, cons, Omega, &
                           dA, wall_lo, wall_hi, flow, ni, nj, nk)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: vx(ni,nj,nk), vr(ni,nj,nk), vt(ni,nj,nk)
        real, intent(in) :: ho(ni,nj,nk), P(ni,nj,nk), r(ni,nj,nk)
        real, intent(in) :: cons(ni,nj,nk,5)
        real, intent(in) :: P_offset, Omega
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
        real, intent(in) :: wall_lo(ni-1, nj-1), wall_hi(ni-1, nj-1)
        real, intent(inout) :: flow(ni-1, nj-1, nk, 5)

        integer :: i, j, k
        real :: pm(6), mf(3), mdot, w

        do k = 1, nk
            do j = 1, nj-1
                do i = 1, ni-1
                    if (k == 1) then
                        w = wall_lo(i,j)
                    else if (k == nk) then
                        w = wall_hi(i,j)
                    else
                        w = 1.0e0
                    end if
                    pm = 0.0e0
                    mf = 0.0e0
                    call accum(pm, mf, i,   j,   k, w)
                    call accum(pm, mf, i+1, j,   k, w)
                    call accum(pm, mf, i,   j+1, k, w)
                    call accum(pm, mf, i+1, j+1, k, w)
                    mdot = mf(1)*dA(1,i,j,k) + mf(2)*dA(2,i,j,k) &
                         + mf(3)*dA(3,i,j,k)
                    flow(i,j,k,1) = mdot
                    flow(i,j,k,2) = pm(1)*mdot + pm(5)*dA(1,i,j,k)
                    flow(i,j,k,3) = pm(2)*mdot + pm(5)*dA(2,i,j,k)
                    flow(i,j,k,4) = pm(3)*mdot + pm(6)*dA(3,i,j,k)
                    flow(i,j,k,5) = pm(4)*mdot + Omega*pm(6)*dA(3,i,j,k)
                end do
            end do
        end do

    contains
        pure subroutine accum(pm, mf, i, j, k, wfac)
            real, intent(inout) :: pm(6), mf(3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real :: dp, w
            dp = P(i,j,k) - P_offset
            pm(1) = pm(1) + 0.25e0*vx(i,j,k)
            pm(2) = pm(2) + 0.25e0*vr(i,j,k)
            pm(3) = pm(3) + 0.25e0*r(i,j,k)*vt(i,j,k)
            pm(4) = pm(4) + 0.25e0*ho(i,j,k)
            pm(5) = pm(5) + 0.25e0*dp
            pm(6) = pm(6) + 0.25e0*r(i,j,k)*dp
            w = 0.25e0*wfac
            mf(1) = mf(1) + w*cons(i,j,k,2)
            mf(2) = mf(2) + w*cons(i,j,k,3)
            mf(3) = mf(3) + w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k))
        end subroutine accum
    end subroutine kface_flows

end module residual_naive_helpers


! =====================================================================
! Four full-volume passes: i-flows, j-flows, k-flows, then difference.
! =====================================================================
subroutine set_residual_naive( &
    cons, P, P_offset, &
    r, Omega, dAi, dAj, dAk, &
    f_body, &
    dU, &
    vx, vr, vt, ho, &
    flow_i, flow_j, flow_k, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    ni, nj, nk &
    )

    use residual_naive_helpers
    use residual_helpers, only: correct_cusp_kface_du

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: f_body(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk)
    real, intent(in) :: vt(ni, nj, nk), ho(ni, nj, nk)
    real, intent(in) :: walli1(nj-1, nk-1), wallni(nj-1, nk-1)
    real, intent(in) :: wallj1(ni-1, nk-1), wallnj(ni-1, nk-1)
    real, intent(in) :: wallk1(ni-1, nj-1), wallnk(ni-1, nj-1)
    integer, intent(in) :: i_cusp_start, i_cusp_end
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! Full-volume face-flow scratch, one array per direction. ~56 MB at
    ! 1M cells -- more than tau_q_halo holds on its own, so the caller
    ! carves flow_k from block.scratch.
    real, intent(inout) :: flow_i(ni, nj-1, nk-1, 5)
    real, intent(inout) :: flow_j(ni-1, nj, nk-1, 5)
    real, intent(inout) :: flow_k(ni-1, nj-1, nk, 5)

    integer :: i, j, k, m

    call iface_flows(vx, vr, vt, ho, P, P_offset, r, cons, Omega, &
                     dAi, walli1, wallni, flow_i, ni, nj, nk)
    call jface_flows(vx, vr, vt, ho, P, P_offset, r, cons, Omega, &
                     dAj, wallj1, wallnj, flow_j, ni, nj, nk)
    call kface_flows(vx, vr, vt, ho, P, P_offset, r, cons, Omega, &
                     dAk, wallk1, wallnk, flow_k, ni, nj, nk)

    do m = 1, 5
        do k = 1, nk-1
            do j = 1, nj-1
                do i = 1, ni-1
                    dU(i,j,k,m) = flow_i(i,j,k,m) - flow_i(i+1,j,k,m) &
                                + f_body(i,j,k,m) &
                                + flow_j(i,j,k,m) - flow_j(i,j+1,k,m) &
                                + flow_k(i,j,k,m) - flow_k(i,j,k+1,m)
                end do
            end do
        end do
    end do

    if (i_cusp_start > 0 .and. nk > 2) then
        call correct_cusp_kface_du(vx, vr, vt, ho, P, P_offset, r, cons, &
                                   Omega, dAk, wallk1, wallnk, dU, &
                                   i_cusp_start, i_cusp_end, ni, nj, nk)
    end if

end subroutine set_residual_naive
