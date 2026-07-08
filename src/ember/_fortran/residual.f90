module residual_helpers
    implicit none
    private
    public :: compute_iface_flows, compute_jface_flows, compute_kface_flows
    public :: correct_cusp_kface

contains

    pure subroutine put_flow(flow, i, j, k, pm, mf, Omega, dA1, dA2, dA3, ni, nj, nk)
        ! Assemble the 5 inviscid face flows from the face-averaged per-mass
        ! factors pm(6), mass-flux factors mf(3) and the face area vector dA,
        ! writing them into flow(i,j,k,:) component-wise (no array temporary).
        ! pm = (Vx, Vr, r*Vt_abs, ho, P-P_offset, r*(P-P_offset)).
        ! mf = (rho*Vx, rho*Vr, rho*Vt_rel).
        implicit none
        integer, intent(in) :: i, j, k, ni, nj, nk
        real, intent(inout) :: flow(ni, nj, nk, 5)
        real, intent(in) :: pm(6), mf(3), Omega, dA1, dA2, dA3
        real :: mdot

        mdot = mf(1)*dA1 + mf(2)*dA2 + mf(3)*dA3
        flow(i,j,k,1) = mdot
        flow(i,j,k,2) = pm(1)*mdot + pm(5)*dA1
        flow(i,j,k,3) = pm(2)*mdot + pm(5)*dA2
        flow(i,j,k,4) = pm(3)*mdot + pm(6)*dA3
        flow(i,j,k,5) = pm(4)*mdot + Omega*pm(6)*dA3
    end subroutine put_flow


    pure subroutine compute_iface_flows(vx, vr, vt, ho, P, P_offset, r, &
                                        cons, vt_rel, Omega, dA, &
                                        wall_lo, wall_hi, flow, ni, nj, nk)
        ! Compute inviscid face flows on all i-faces.
        ! i-face corners: (i, j:j+1, k:k+1)

        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5), vt_rel(ni, nj, nk)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: wall_lo(nj-1, nk-1)
        real, intent(in) :: wall_hi(nj-1, nk-1)
        real, intent(out) :: flow(ni, nj, nk, 5)

        integer :: i, j, k
        real :: pm(6), mf(3)

        ! Low boundary i=1
        do k = 1, nk-1
        do j = 1, nj-1
            pm = 0.0e0; mf = 0.0e0
            call accum(pm, mf, 1, j,   k,   wall_lo(j,k))
            call accum(pm, mf, 1, j+1, k,   wall_lo(j,k))
            call accum(pm, mf, 1, j,   k+1, wall_lo(j,k))
            call accum(pm, mf, 1, j+1, k+1, wall_lo(j,k))
            call put_flow(flow, 1, j, k, pm, mf, Omega, &
                          dA(1,1,j,k), dA(2,1,j,k), dA(3,1,j,k), ni, nj, nk)
        end do
        end do

        ! Interior i=2..ni-1
        do k = 1, nk-1
        do j = 1, nj-1
        do i = 2, ni-1
            pm = 0.0e0; mf = 0.0e0
            call accum(pm, mf, i, j,   k,   1.0e0)
            call accum(pm, mf, i, j+1, k,   1.0e0)
            call accum(pm, mf, i, j,   k+1, 1.0e0)
            call accum(pm, mf, i, j+1, k+1, 1.0e0)
            call put_flow(flow, i, j, k, pm, mf, Omega, &
                          dA(1,i,j,k), dA(2,i,j,k), dA(3,i,j,k), ni, nj, nk)
        end do
        end do
        end do

        ! High boundary i=ni
        do k = 1, nk-1
        do j = 1, nj-1
            pm = 0.0e0; mf = 0.0e0
            call accum(pm, mf, ni, j,   k,   wall_hi(j,k))
            call accum(pm, mf, ni, j+1, k,   wall_hi(j,k))
            call accum(pm, mf, ni, j,   k+1, wall_hi(j,k))
            call accum(pm, mf, ni, j+1, k+1, wall_hi(j,k))
            call put_flow(flow, ni, j, k, pm, mf, Omega, &
                          dA(1,ni,j,k), dA(2,ni,j,k), dA(3,ni,j,k), ni, nj, nk)
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
            mf(3) = mf(3) + w*cons(i,j,k,1)*vt_rel(i,j,k)
        end subroutine accum
    end subroutine compute_iface_flows


    pure subroutine compute_jface_flows(vx, vr, vt, ho, P, P_offset, r, &
                                        cons, vt_rel, Omega, dA, &
                                        wall_lo, wall_hi, flow, ni, nj, nk)
        ! Compute inviscid face flows on all j-faces.
        ! j-face corners: (i:i+1, j, k:k+1)

        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5), vt_rel(ni, nj, nk)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: wall_lo(ni-1, nk-1)
        real, intent(in) :: wall_hi(ni-1, nk-1)
        real, intent(out) :: flow(ni, nj, nk, 5)

        integer :: i, j, k
        real :: pm(6), mf(3)

        ! Low boundary j=1
        do k = 1, nk-1
        do i = 1, ni-1
            pm = 0.0e0; mf = 0.0e0
            call accum(pm, mf, i,   1, k,   wall_lo(i,k))
            call accum(pm, mf, i+1, 1, k,   wall_lo(i,k))
            call accum(pm, mf, i,   1, k+1, wall_lo(i,k))
            call accum(pm, mf, i+1, 1, k+1, wall_lo(i,k))
            call put_flow(flow, i, 1, k, pm, mf, Omega, &
                          dA(1,i,1,k), dA(2,i,1,k), dA(3,i,1,k), ni, nj, nk)
        end do
        end do

        ! Interior j=2..nj-1
        do k = 1, nk-1
        do j = 2, nj-1
        do i = 1, ni-1
            pm = 0.0e0; mf = 0.0e0
            call accum(pm, mf, i,   j, k,   1.0e0)
            call accum(pm, mf, i+1, j, k,   1.0e0)
            call accum(pm, mf, i,   j, k+1, 1.0e0)
            call accum(pm, mf, i+1, j, k+1, 1.0e0)
            call put_flow(flow, i, j, k, pm, mf, Omega, &
                          dA(1,i,j,k), dA(2,i,j,k), dA(3,i,j,k), ni, nj, nk)
        end do
        end do
        end do

        ! High boundary j=nj
        do k = 1, nk-1
        do i = 1, ni-1
            pm = 0.0e0; mf = 0.0e0
            call accum(pm, mf, i,   nj, k,   wall_hi(i,k))
            call accum(pm, mf, i+1, nj, k,   wall_hi(i,k))
            call accum(pm, mf, i,   nj, k+1, wall_hi(i,k))
            call accum(pm, mf, i+1, nj, k+1, wall_hi(i,k))
            call put_flow(flow, i, nj, k, pm, mf, Omega, &
                          dA(1,i,nj,k), dA(2,i,nj,k), dA(3,i,nj,k), ni, nj, nk)
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
            mf(3) = mf(3) + w*cons(i,j,k,1)*vt_rel(i,j,k)
        end subroutine accum
    end subroutine compute_jface_flows


    pure subroutine compute_kface_flows(vx, vr, vt, ho, P, P_offset, r, &
                                        cons, vt_rel, Omega, dA, &
                                        wall_lo, wall_hi, flow, ni, nj, nk)
        ! Compute inviscid face flows on all k-faces.
        ! k-face corners: (i:i+1, j:j+1, k)

        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5), vt_rel(ni, nj, nk)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
        real, intent(in) :: wall_lo(ni-1, nj-1)
        real, intent(in) :: wall_hi(ni-1, nj-1)
        real, intent(out) :: flow(ni, nj, nk, 5)

        integer :: i, j, k
        real :: pm(6), mf(3)

        ! Low boundary k=1
        do j = 1, nj-1
        do i = 1, ni-1
            pm = 0.0e0; mf = 0.0e0
            call accum(pm, mf, i,   j,   1, wall_lo(i,j))
            call accum(pm, mf, i+1, j,   1, wall_lo(i,j))
            call accum(pm, mf, i,   j+1, 1, wall_lo(i,j))
            call accum(pm, mf, i+1, j+1, 1, wall_lo(i,j))
            call put_flow(flow, i, j, 1, pm, mf, Omega, &
                          dA(1,i,j,1), dA(2,i,j,1), dA(3,i,j,1), ni, nj, nk)
        end do
        end do

        ! Interior k=2..nk-1
        do k = 2, nk-1
        do j = 1, nj-1
        do i = 1, ni-1
            pm = 0.0e0; mf = 0.0e0
            call accum(pm, mf, i,   j,   k, 1.0e0)
            call accum(pm, mf, i+1, j,   k, 1.0e0)
            call accum(pm, mf, i,   j+1, k, 1.0e0)
            call accum(pm, mf, i+1, j+1, k, 1.0e0)
            call put_flow(flow, i, j, k, pm, mf, Omega, &
                          dA(1,i,j,k), dA(2,i,j,k), dA(3,i,j,k), ni, nj, nk)
        end do
        end do
        end do

        ! High boundary k=nk
        do j = 1, nj-1
        do i = 1, ni-1
            pm = 0.0e0; mf = 0.0e0
            call accum(pm, mf, i,   j,   nk, wall_hi(i,j))
            call accum(pm, mf, i+1, j,   nk, wall_hi(i,j))
            call accum(pm, mf, i,   j+1, nk, wall_hi(i,j))
            call accum(pm, mf, i+1, j+1, nk, wall_hi(i,j))
            call put_flow(flow, i, j, nk, pm, mf, Omega, &
                          dA(1,i,j,nk), dA(2,i,j,nk), dA(3,i,j,nk), ni, nj, nk)
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
            mf(3) = mf(3) + w*cons(i,j,k,1)*vt_rel(i,j,k)
        end subroutine accum
    end subroutine compute_kface_flows


    subroutine correct_cusp_kface(vx, vr, vt, ho, P, P_offset, r, &
                                  cons, vt_rel, dAk, flow, &
                                  i_cusp_start, i_cusp_end, ni, nj, nk)
        ! Correct k-face flux at cusp faces (matching Multall).
        ! Mass / angular momentum / energy: full TFLUX average across the seam.
        ! Axial and radial momentum: rebuild both faces from seam-averaged
        ! mdot, velocity and pressure, with per-face dAk only.

        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5), vt_rel(ni, nj, nk)
        real, intent(in) :: dAk(3, ni-1, nj-1, nk)
        real, intent(inout) :: flow(ni, nj, nk, 5)
        integer, intent(in) :: i_cusp_start, i_cusp_end

        integer :: i, j
        real :: pm_lo(6), mf_lo(3), mdot_lo
        real :: pm_hi(6), mf_hi(3), mdot_hi
        real :: Vx_avg, Vr_avg, P_avg, mdot_avg, imb_mass

        do j = 1, nj-1
        do i = i_cusp_start, i_cusp_end-1
            ! Seam-averaged per-mass and mass-flux factors at k=1.
            pm_lo = 0.0e0; mf_lo = 0.0e0
            call accum(pm_lo, mf_lo, i,   j,   1)
            call accum(pm_lo, mf_lo, i+1, j,   1)
            call accum(pm_lo, mf_lo, i,   j+1, 1)
            call accum(pm_lo, mf_lo, i+1, j+1, 1)
            mdot_lo = mf_lo(1)*dAk(1,i,j,1) + mf_lo(2)*dAk(2,i,j,1) &
                    + mf_lo(3)*dAk(3,i,j,1)

            ! Seam-averaged factors at k=nk.
            pm_hi = 0.0e0; mf_hi = 0.0e0
            call accum(pm_hi, mf_hi, i,   j,   nk)
            call accum(pm_hi, mf_hi, i+1, j,   nk)
            call accum(pm_hi, mf_hi, i,   j+1, nk)
            call accum(pm_hi, mf_hi, i+1, j+1, nk)
            mdot_hi = mf_hi(1)*dAk(1,i,j,nk) + mf_hi(2)*dAk(2,i,j,nk) &
                    + mf_hi(3)*dAk(3,i,j,nk)

            ! Seam-averaged primitives (Vx, Vr, P) and mdot.
            Vx_avg   = 0.5e0*(pm_lo(1) + pm_hi(1))
            Vr_avg   = 0.5e0*(pm_lo(2) + pm_hi(2))
            P_avg    = 0.5e0*(pm_lo(5) + pm_hi(5))
            mdot_avg = 0.5e0*(mdot_lo + mdot_hi)

            ! Mass (m=1): full TFLUX average.
            imb_mass = 0.5e0*(flow(i,j,1,1) + flow(i,j,nk,1))
            flow(i,j,1,1)  = imb_mass
            flow(i,j,nk,1) = imb_mass

            ! Axial momentum (m=2): rebuild both faces from averaged
            ! mdot*Vx + P*Ax, using each face's own dAk(1,...).
            flow(i,j,1,2)  = mdot_avg*Vx_avg + P_avg*dAk(1,i,j,1)
            flow(i,j,nk,2) = mdot_avg*Vx_avg + P_avg*dAk(1,i,j,nk)

            ! Radial momentum (m=3): same with dAk(2,...).
            flow(i,j,1,3)  = mdot_avg*Vr_avg + P_avg*dAk(2,i,j,1)
            flow(i,j,nk,3) = mdot_avg*Vr_avg + P_avg*dAk(2,i,j,nk)

            ! Angular momentum (m=4) and energy (m=5): full TFLUX average.
            flow(i,j,1,4)  = 0.5e0*(flow(i,j,1,4) + flow(i,j,nk,4))
            flow(i,j,nk,4) = flow(i,j,1,4)
            flow(i,j,1,5)  = 0.5e0*(flow(i,j,1,5) + flow(i,j,nk,5))
            flow(i,j,nk,5) = flow(i,j,1,5)
        end do
        end do

    contains
        pure subroutine accum(pm, mf, i, j, k)
            real, intent(inout) :: pm(6), mf(3)
            integer, intent(in) :: i, j, k
            real :: dp
            dp = P(i,j,k) - P_offset
            pm(1) = pm(1) + 0.25e0*vx(i,j,k)
            pm(2) = pm(2) + 0.25e0*vr(i,j,k)
            pm(3) = pm(3) + 0.25e0*r(i,j,k)*vt(i,j,k)
            pm(4) = pm(4) + 0.25e0*ho(i,j,k)
            pm(5) = pm(5) + 0.25e0*dp
            pm(6) = pm(6) + 0.25e0*r(i,j,k)*dp
            mf(1) = mf(1) + 0.25e0*cons(i,j,k,2)
            mf(2) = mf(2) + 0.25e0*cons(i,j,k,3)
            mf(3) = mf(3) + 0.25e0*cons(i,j,k,1)*vt_rel(i,j,k)
        end subroutine accum
    end subroutine correct_cusp_kface


end module residual_helpers


! =====================================================================
! v3 unscaled residual: single pass over precomputed nodal primitives.
! Per-mass (perm) and mass-flux (mflux) factors are read directly from
! cached nodal arrays (vx, vr, vt, vt_rel, ho) and assembled per face,
! so the only scratch needed is the 5-wide reusable `flow` buffer.
! =====================================================================
subroutine set_residual( &
    cons, P, P_offset, &
    r, Omega, dAi, dAj, dAk, &
    f_body, &
    dU, &
    vx, vr, vt, vt_rel, ho, &
    flow_i, flow_jk, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    ni, nj, nk &
    )

    use residual_helpers

    implicit none

    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: f_body(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: vx(ni, nj, nk)
    real, intent(in) :: vr(ni, nj, nk)
    real, intent(in) :: vt(ni, nj, nk)
    real, intent(in) :: vt_rel(ni, nj, nk)
    real, intent(in) :: ho(ni, nj, nk)
    real, intent(in) :: walli1(nj-1, nk-1)
    real, intent(in) :: wallni(nj-1, nk-1)
    real, intent(in) :: wallj1(ni-1, nk-1)
    real, intent(in) :: wallnj(ni-1, nk-1)
    real, intent(in) :: wallk1(ni-1, nj-1)
    real, intent(in) :: wallnk(ni-1, nj-1)
    integer, intent(in) :: i_cusp_start, i_cusp_end
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! Two transient flow-scratch buffers so all three direction flows coexist
    ! and the dU accumulation fuses into one pass. flow_i (5 slots) holds the
    ! i-face flows; flow_jk (10 slots) holds j-face (1:5) and k-face (6:10).
    ! Caller backs these with block._scratch and block._tau_q_halo, which are
    ! pure transient scratch -- the (i,j,k) layout here is private to this call.
    real, intent(inout) :: flow_i(ni, nj, nk, 5)
    real, intent(inout) :: flow_jk(ni, nj, nk, 10)
    integer, intent(in) :: ni, nj, nk

    integer :: i, j, k, m

    ! Each direction writes its own contiguous 5-slot sub-block. In Fortran
    ! column-major order flow_jk(:,:,:,a:b) is contiguous (last axis varies
    ! slowest), so these slices pass without array temporaries.
    call compute_iface_flows(vx, vr, vt, ho, P, P_offset, r, cons, vt_rel, &
                             Omega, dAi, walli1, wallni, flow_i, ni, nj, nk)
    call compute_jface_flows(vx, vr, vt, ho, P, P_offset, r, cons, vt_rel, &
                             Omega, dAj, wallj1, wallnj, flow_jk(:,:,:,1:5), ni, nj, nk)
    call compute_kface_flows(vx, vr, vt, ho, P, P_offset, r, cons, vt_rel, &
                             Omega, dAk, wallk1, wallnk, flow_jk(:,:,:,6:10), ni, nj, nk)
    if (i_cusp_start > 0) then
        call correct_cusp_kface(vx, vr, vt, ho, P, P_offset, r, cons, vt_rel, &
                                dAk, flow_jk(:,:,:,6:10), i_cusp_start, i_cusp_end, &
                                ni, nj, nk)
    end if

    ! Fused accumulation: dU written once, never read back.
    do m = 1, 5
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        dU(i,j,k,m) = flow_i(i,j,k,m)     - flow_i(i+1,j,k,m)     &
                    + flow_jk(i,j,k,m)    - flow_jk(i,j+1,k,m)    &
                    + flow_jk(i,j,k,m+5)  - flow_jk(i,j,k+1,m+5)  &
                    + f_body(i,j,k,m)
    end do
    end do
    end do
    end do

end subroutine set_residual


! =====================================================================
! Negative-feedback change limiter (ported from multall's DAMP loop).
!
! Soft-clips outlier per-cell changes so the explicit march stays stable
! without globally cutting the timestep. The per-step change is
! dU * dt_vol; per conserved variable, the block mean of its magnitude is
! avg, and each cell is shrunk by 1/(1 + fdamp/dampin) with
! fdamp = |change|/avg. Cells near the mean are barely touched; large
! outliers saturate towards dampin*avg. Operates in place on dU.
! =====================================================================
subroutine damp_residual(dU, dt_vol, dampin, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: dampin

    integer :: i, j, k, m, ncell
    real :: avg, chg, fdamp

    ncell = (ni-1)*(nj-1)*(nk-1)
    do m = 1, 5
        ! Pass 1: block mean of the per-step change magnitude for variable m.
        avg = 0.0e0
        do k = 1, nk-1
        do j = 1, nj-1
        do i = 1, ni-1
            avg = avg + abs(dU(i,j,k,m) * dt_vol(i,j,k))
        end do
        end do
        end do
        avg = avg / ncell
        if (avg <= 0.0e0) cycle          ! flat field: nothing to limit
        ! Pass 2: soft-clip each cell's change relative to the block mean.
        do k = 1, nk-1
        do j = 1, nj-1
        do i = 1, ni-1
            chg   = abs(dU(i,j,k,m) * dt_vol(i,j,k))
            fdamp = chg / avg
            dU(i,j,k,m) = dU(i,j,k,m) / (1.0e0 + fdamp/dampin)
        end do
        end do
        end do
    end do

end subroutine damp_residual


! =====================================================================
! Implicit residual smoothing (Jameson IRS) -- production Jacobi version.
!
! Solves the same system as smooth_residual_gs, (1 - sf*grad^2) R* = R,
! but by n_smooth JACOBI sweeps (each cell reads the PREVIOUS iterate,
! never the current one), so the operator is order-independent and
! symmetric, and -- crucially -- the sweep has no loop-carried
! dependency and vectorises. Both schemes converge to the same R* as
! n_smooth -> inf; at finite sweeps Jacobi damps somewhat less per sweep.
!
! The hot loop is branch-free and divide-free:
!   * Halo padding. The iterate buffers carry a one-cell zero halo, so a
!     missing face neighbour contributes 0 with no per-cell test. The
!     Neumann boundary (reduced neighbour count) is absorbed into inv.
!   * Precomputed reciprocal. inv(i,j,k) = 1/(1 + sf*navail) is built once
!     (navail = 6 interior, less on the shell), turning the per-cell divide
!     into a multiply.
!
! Constant fields are preserved exactly, linear fields in the interior,
! and IRS(0) = 0 so the converged solution is unchanged. Operates in
! place on dU. work is the caller's (ni-1,nj-1,nk-1,5) buffer, used to
! hold the fixed RHS R across the sweeps.
!
! NOTE: the padded ping-pong buffers a,b and inv are allocated per call.
! If this shows up in profiling, hoist them into a persistent per-block
! workspace (cf. block.scratch / util.allocate_or_reuse).
! =====================================================================
subroutine smooth_residual(dU, sf, n_smooth, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk, n_smooth
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(inout) :: work(ni-1, nj-1, nk-1, 5)

    integer :: i, j, k, m, it, nci, ncj, nck
    real :: navail
    real, allocatable :: a(:,:,:,:), b(:,:,:,:), t(:,:,:,:), inv(:,:,:)

    if (n_smooth <= 0 .or. sf <= 0.0e0) return

    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    ! Fixed RHS R for every sweep.
    work = dU

    ! Halo-padded iterate buffers: index 0 and nc+1 planes stay zero forever
    ! (only the interior 1..nc is ever written), so boundary reads need no test.
    allocate(a(0:nci+1, 0:ncj+1, 0:nck+1, 5))
    allocate(b(0:nci+1, 0:ncj+1, 0:nck+1, 5))
    allocate(inv(nci, ncj, nck))
    a = 0.0e0
    b = 0.0e0

    do m = 1, 5
    do k = 1, nck
    do j = 1, ncj
    do i = 1, nci
        a(i,j,k,m) = dU(i,j,k,m)
    end do
    end do
    end do
    end do

    ! Reciprocal denominator, built once: navail = 6 interior, one less per
    ! boundary face. Branches here are fine -- one-time O(N) pass, not hot.
    do k = 1, nck
    do j = 1, ncj
    do i = 1, nci
        navail = 6.0e0
        if (i == 1)   navail = navail - 1.0e0
        if (i == nci) navail = navail - 1.0e0
        if (j == 1)   navail = navail - 1.0e0
        if (j == ncj) navail = navail - 1.0e0
        if (k == 1)   navail = navail - 1.0e0
        if (k == nck) navail = navail - 1.0e0
        inv(i,j,k) = 1.0e0 / (1.0e0 + sf*navail)
    end do
    end do
    end do

    do it = 1, n_smooth
        ! b <- Jacobi(a): branch-free, divide-free, vectorises over i.
        do m = 1, 5
        do k = 1, nck
        do j = 1, ncj
        do i = 1, nci
            b(i,j,k,m) = (work(i,j,k,m) + sf*( &
                a(i-1,j,k,m) + a(i+1,j,k,m) + &
                a(i,j-1,k,m) + a(i,j+1,k,m) + &
                a(i,j,k-1,m) + a(i,j,k+1,m))) * inv(i,j,k)
        end do
        end do
        end do
        end do
        ! Ping-pong a <-> b via descriptor moves (no data copy). Halos of both
        ! stay zero throughout, so boundary reads remain branch-free.
        call move_alloc(a, t)
        call move_alloc(b, a)
        call move_alloc(t, b)
    end do

    ! Latest iterate is in a's interior -> copy back to dU.
    do m = 1, 5
    do k = 1, nck
    do j = 1, ncj
    do i = 1, nci
        dU(i,j,k,m) = a(i,j,k,m)
    end do
    end do
    end do
    end do

    deallocate(a, b, inv)

end subroutine smooth_residual


! =====================================================================
! Implicit residual smoothing (Jameson IRS) -- EXACT factored tridiagonal.
!
! The unfactored operator (1 - sf*grad^2) that smooth_residual/_gs invert
! iteratively is replaced by the ADI-style factored product
!
!   (1 - sf*d2_i) (1 - sf*d2_j) (1 - sf*d2_k) R* = R
!
! where d2_d is the 1D second difference along direction d with zero-
! gradient (Neumann) ends -- exactly the boundary rule the Jacobi/GS
! smoothers use (missing neighbour dropped, diagonal reduced). Because the
! three 1D operators act on orthogonal index directions they commute, so
! the factored inverse is applied as three successive EXACT tridiagonal
! (Thomas) solves, one per direction, in place on dU. No sweep count: each
! direction is solved to the last bit in O(n) per line, so this realises
! the full sf-dependent damping that finite Jacobi sweeps only approach.
!
! The tridiagonal matrix is identical for every line in a given direction
! (constant coefficients a=c=-sf, b=1+2sf interior, b=1+sf at the two ends),
! so its Thomas factors cp(.) and the reciprocal pivots minv(.) are built
! ONCE per direction by tri_coeffs and reused for all lines. The hot solve
! loops are then completely branch-free and divide-free:
!   * i-solve: recurrence runs along the stride-1 index (cache-streamed).
!   * j/k-solve: recurrence runs along j (resp. k) while the innermost loop
!     runs over the stride-1 i index, so each recurrence step is a vector
!     op over i.
!
! Constant fields are preserved exactly (each 1D factor maps a constant to
! itself), and IRS(0)=0, so the converged solution is unchanged. The
! factored operator differs from the unfactored one only by O(sf^2) cross
! terms; on the highest mode it damps by 1/(1+4sf)^3 (vs 1/(1+2*d*sf) for
! the true d-dimensional Laplacian) -- i.e. more aggressively per unit sf.
!
! Scratch: the Thomas solve is in place on dU, so the only workspace is the
! per-direction factors cp(.) and minv(.). work is a 1D buffer holding all
! six vectors back-to-back and must be at least 2*((ni-1)+(nj-1)+(nk-1))
! elements -- e.g. carve a leading slice of block.scratch (nodal (ni,nj,nk,5),
! vastly oversized) with util.carve_view; nothing is allocated here.
! =====================================================================
subroutine smooth_residual_tri(dU, sf, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(inout) :: work(2*((ni-1) + (nj-1) + (nk-1)))

    integer :: i, j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: cc, mm

    if (sf <= 0.0e0) return

    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    ! Base offsets of the six coefficient vectors packed into work:
    ! [cpi | minvi | cpj | minvj | cpk | minvk], lengths nci,nci,ncj,ncj,nck,nck.
    bcpi = 0
    bmii = nci
    bcpj = 2*nci
    bmij = 2*nci + ncj
    bcpk = 2*nci + 2*ncj
    bmik = 2*nci + 2*ncj + nck
    call tri_coeffs(sf, nci, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci))
    call tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))

    ! ---- i-direction: recurrence along the stride-1 index (nci >= 2) ----
    if (nci >= 2) then
        do m = 1, 5
        do k = 1, nck
        do j = 1, ncj
            dU(1,j,k,m) = dU(1,j,k,m) * work(bmii+1)
            do i = 2, nci
                dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i-1,j,k,m)) * work(bmii+i)
            end do
            do i = nci-1, 1, -1
                dU(i,j,k,m) = dU(i,j,k,m) - work(bcpi+i)*dU(i+1,j,k,m)
            end do
        end do
        end do
        end do
    end if

    ! ---- j-direction: recurrence along j, innermost vector loop over i.
    ! The per-plane factors are loop-invariant over i, so hoist to scalars. ----
    if (ncj >= 2) then
        do m = 1, 5
        do k = 1, nck
            mm = work(bmij+1)
            do i = 1, nci
                dU(i,1,k,m) = dU(i,1,k,m) * mm
            end do
            do j = 2, ncj
                mm = work(bmij+j)
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j-1,k,m)) * mm
                end do
            end do
            do j = ncj-1, 1, -1
                cc = work(bcpj+j)
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j+1,k,m)
                end do
            end do
        end do
        end do
    end if

    ! ---- k-direction: recurrence along k, innermost vector loop over i ----
    if (nck >= 2) then
        do m = 1, 5
            mm = work(bmik+1)
            do j = 1, ncj
            do i = 1, nci
                dU(i,j,1,m) = dU(i,j,1,m) * mm
            end do
            end do
            do k = 2, nck
                mm = work(bmik+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j,k-1,m)) * mm
                end do
                end do
            end do
            do k = nck-1, 1, -1
                cc = work(bcpk+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j,k+1,m)
                end do
                end do
            end do
        end do
    end if

contains

    ! Thomas forward-sweep factors for the constant-coefficient Neumann
    ! tridiagonal along a line of length n: a = c = -sf, b = 1+2sf interior,
    ! b = 1+sf at the two ends. Returns cp (eliminated super-diagonal) and
    ! minv = 1/pivot, so a line solve is:
    !   x(1)   = d(1)*minv(1)
    !   x(i)   = (d(i) + sf*x(i-1))*minv(i)          i = 2..n   (forward)
    !   x(i)   = x(i) - cp(i)*x(i+1)                 i = n-1..1 (back-sub)
    ! An n=1 line has no neighbours -> operator is the identity (minv=1).
    subroutine tri_coeffs(e, n, cp, minv)
        implicit none
        real, intent(in)     :: e
        integer, intent(in)  :: n
        real, intent(out)    :: cp(n), minv(n)
        integer :: ii

        if (n == 1) then
            minv(1) = 1.0e0
            cp(1)   = 0.0e0
            return
        end if

        ! Row 1: b = 1 + sf (single neighbour), c = -sf.
        minv(1) = 1.0e0 / (1.0e0 + e)
        cp(1)   = -e * minv(1)
        do ii = 2, n-1
            minv(ii) = 1.0e0 / ((1.0e0 + 2.0e0*e) + e*cp(ii-1))
            cp(ii)   = -e * minv(ii)
        end do
        ! Row n: b = 1 + sf (single neighbour), c = 0.
        minv(n) = 1.0e0 / ((1.0e0 + e) + e*cp(n-1))
        cp(n)   = 0.0e0
    end subroutine tri_coeffs

end subroutine smooth_residual_tri

! =====================================================================
! Transpose-tiled variant of smooth_residual_tri. Identical maths and
! results; only the i-direction solve differs. In smooth_residual_tri the
! i-solve recurrence runs along the unit-stride axis, so it cannot
! vectorise (the compiler reports the j/k solves vectorised but not this
! one). Here a BJ-wide tile of j-lines is transposed into a small (BJ,nci)
! scratch pad so the recurrence's innermost loop runs over the contiguous
! BJ lanes -- independent lines, so it both vectorises and exposes ILP to
! hide the FMA-latency chain -- then the tile is scattered back. The j and
! k solves are unchanged (they already vectorise over stride-1 i).
! =====================================================================
subroutine smooth_residual_tri_tiled(dU, sf, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(inout) :: work(2*((ni-1) + (nj-1) + (nk-1)))

    integer :: i, j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: cc, mm
    integer, parameter :: BJ = 8            ! tile width (AVX = 8 float32 lanes)
    integer :: jj, j0, nb
    real    :: tile(BJ, ni-1)               ! (lane, i) transposed i-solve pad

    if (sf <= 0.0e0) return

    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    ! Base offsets of the six coefficient vectors packed into work:
    ! [cpi | minvi | cpj | minvj | cpk | minvk], lengths nci,nci,ncj,ncj,nck,nck.
    bcpi = 0
    bmii = nci
    bcpj = 2*nci
    bmij = 2*nci + ncj
    bcpk = 2*nci + 2*ncj
    bmik = 2*nci + 2*ncj + nck
    call tri_coeffs(sf, nci, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci))
    call tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))

    ! ---- i-direction: transpose-tiled. A BJ-wide block of j-lines is gathered
    ! into tile(lane, i); the recurrence then runs along i with the innermost
    ! loop over the BJ contiguous, independent lanes -> vectorises + hides the
    ! recurrence latency. Scatter back afterwards. (nci >= 2) ----
    if (nci >= 2) then
        do m = 1, 5
        do k = 1, nck
        do j0 = 1, ncj, BJ
            nb = min(BJ, ncj - j0 + 1)
            do i = 1, nci
                do jj = 1, nb
                    tile(jj,i) = dU(i, j0+jj-1, k, m)
                end do
            end do
            mm = work(bmii+1)
            do jj = 1, nb
                tile(jj,1) = tile(jj,1) * mm
            end do
            do i = 2, nci
                mm = work(bmii+i)
                do jj = 1, nb
                    tile(jj,i) = (tile(jj,i) + sf*tile(jj,i-1)) * mm
                end do
            end do
            do i = nci-1, 1, -1
                cc = work(bcpi+i)
                do jj = 1, nb
                    tile(jj,i) = tile(jj,i) - cc*tile(jj,i+1)
                end do
            end do
            do i = 1, nci
                do jj = 1, nb
                    dU(i, j0+jj-1, k, m) = tile(jj,i)
                end do
            end do
        end do
        end do
        end do
    end if

    ! ---- j-direction: recurrence along j, innermost vector loop over i.
    ! The per-plane factors are loop-invariant over i, so hoist to scalars. ----
    if (ncj >= 2) then
        do m = 1, 5
        do k = 1, nck
            mm = work(bmij+1)
            do i = 1, nci
                dU(i,1,k,m) = dU(i,1,k,m) * mm
            end do
            do j = 2, ncj
                mm = work(bmij+j)
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j-1,k,m)) * mm
                end do
            end do
            do j = ncj-1, 1, -1
                cc = work(bcpj+j)
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j+1,k,m)
                end do
            end do
        end do
        end do
    end if

    ! ---- k-direction: recurrence along k, innermost vector loop over i ----
    if (nck >= 2) then
        do m = 1, 5
            mm = work(bmik+1)
            do j = 1, ncj
            do i = 1, nci
                dU(i,j,1,m) = dU(i,j,1,m) * mm
            end do
            end do
            do k = 2, nck
                mm = work(bmik+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = (dU(i,j,k,m) + sf*dU(i,j,k-1,m)) * mm
                end do
                end do
            end do
            do k = nck-1, 1, -1
                cc = work(bcpk+k)
                do j = 1, ncj
                do i = 1, nci
                    dU(i,j,k,m) = dU(i,j,k,m) - cc*dU(i,j,k+1,m)
                end do
                end do
            end do
        end do
    end if

contains

    ! Thomas forward-sweep factors for the constant-coefficient Neumann
    ! tridiagonal along a line of length n: a = c = -sf, b = 1+2sf interior,
    ! b = 1+sf at the two ends. Returns cp (eliminated super-diagonal) and
    ! minv = 1/pivot, so a line solve is:
    !   x(1)   = d(1)*minv(1)
    !   x(i)   = (d(i) + sf*x(i-1))*minv(i)          i = 2..n   (forward)
    !   x(i)   = x(i) - cp(i)*x(i+1)                 i = n-1..1 (back-sub)
    ! An n=1 line has no neighbours -> operator is the identity (minv=1).
    subroutine tri_coeffs(e, n, cp, minv)
        implicit none
        real, intent(in)     :: e
        integer, intent(in)  :: n
        real, intent(out)    :: cp(n), minv(n)
        integer :: ii

        if (n == 1) then
            minv(1) = 1.0e0
            cp(1)   = 0.0e0
            return
        end if

        ! Row 1: b = 1 + sf (single neighbour), c = -sf.
        minv(1) = 1.0e0 / (1.0e0 + e)
        cp(1)   = -e * minv(1)
        do ii = 2, n-1
            minv(ii) = 1.0e0 / ((1.0e0 + 2.0e0*e) + e*cp(ii-1))
            cp(ii)   = -e * minv(ii)
        end do
        ! Row n: b = 1 + sf (single neighbour), c = 0.
        minv(n) = 1.0e0 / ((1.0e0 + e) + e*cp(n-1))
        cp(n)   = 0.0e0
    end subroutine tri_coeffs

end subroutine smooth_residual_tri_tiled
