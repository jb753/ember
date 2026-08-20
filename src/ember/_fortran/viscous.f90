! Routines for adding viscous effects

! ============================================================
! Module containing the helper functions used by the viscous kernels.
! All arrays are assumed-shape (contiguous) so ni/nj/nk are
! not needed in any signature.
! ============================================================
module viscous_helpers
    implicit none
    private
    public :: iface, jface, kface
    public :: wall_core, wall_func, wall_yplus
    public :: wall_func_iface, wall_func_jface, wall_func_kface
    public :: wall_yplus_iface, wall_yplus_jface, wall_yplus_kface
    public :: kface_flow
    public :: polar_src, scale_visc_halos, zero_wall_fvisc_border

contains

    pure function iface(x, i, j, k) result(sum4)
        implicit none
        real, intent(in), contiguous :: x(:,:,:)
        integer, intent(in) :: i, j, k
        real :: sum4
        sum4 = x(i,j,k) + x(i,j+1,k) + x(i,j,k+1) + x(i,j+1,k+1)
    end function iface

    pure function jface(x, i, j, k) result(sum4)
        implicit none
        real, intent(in), contiguous :: x(:,:,:)
        integer, intent(in) :: i, j, k
        real :: sum4
        sum4 = x(i,j,k) + x(i+1,j,k) + x(i,j,k+1) + x(i+1,j,k+1)
    end function jface

    pure function kface(x, i, j, k) result(sum4)
        implicit none
        real, intent(in), contiguous :: x(:,:,:)
        integer, intent(in) :: i, j, k
        real :: sum4
        sum4 = x(i,j,k) + x(i+1,j,k) + x(i,j+1,k) + x(i+1,j+1,k)
    end function kface

    ! Shared core of wall_func/wall_yplus: the local wall-friction physics --
    ! slip velocity, Reynolds number on the cell-thickness scale
    ! d = vol/dA_mag, and the skin-friction curve fit -- with none of either
    ! caller's own output assembly. Splitting this out means wall_func's flux
    ! and wall_yplus's y+ can never silently drift apart: one Re/d
    ! definition, one curve fit, two three-line callers. See wall_yplus's own
    ! comment for the y+ derivation in terms of these outputs.
    pure subroutine wall_core(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, &
                               V, dA_mag, Vt_slip, cf, Re, tau)
        implicit none
        real, intent(in) :: r, dA(3), vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt
        real, intent(out) :: V, dA_mag, Vt_slip, cf, Re, tau
        real :: d, lnRew
        real, parameter :: a1 = -1.767e-3
        real, parameter :: a2 = 3.177e-2
        real, parameter :: a3 = 2.5614e-1
        ! Vt is relative to block frame; subtract wall velocity in block frame
        Vt_slip = Vt - (Omega_wall - Omega_block) * r
        V = sqrt(Vx**2 + Vr**2 + Vt_slip**2 + 1e-9)
        dA_mag = sqrt(dA(1)**2 + dA(2)**2 + dA(3)**2)
        d = vol / dA_mag
        Re = rho * V * d / mu
        if (Re .lt. 127.53373025e0) then
            cf = 2e0/Re
        else
            ! lnRew moved inside this branch (it used to be computed
            ! unconditionally and wasted below Re=127.5) -- a drive-by
            ! cleanup that fell out of extracting this core, not a change in
            ! what either branch computes.
            lnRew = log(Re)
            cf = (a1 + a2/lnRew + a3/lnRew/lnRew)
        end if
        tau = cf * 0.5e0 * rho * V * V
    end subroutine wall_core

    pure subroutine wall_func(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, flow)
        implicit none
        real, intent(in) :: r, dA(3), vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt
        real, intent(out) :: flow(4)
        real :: V, dA_mag, Vt_slip, cf, Re, tau, vec(3)
        call wall_core(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, &
                       V, dA_mag, Vt_slip, cf, Re, tau)
        vec(1) = Vx     / V * dA_mag
        vec(2) = Vr     / V * dA_mag
        vec(3) = Vt_slip / V * dA_mag
        flow(1) = vec(1) * tau
        flow(2) = vec(2) * tau
        flow(3) = r * vec(3) * tau
        flow(4) = Omega_wall * r * vec(3) * tau
    end subroutine wall_func

    ! y+ at this wall face, using the SAME Re/d wall_func itself uses (not
    ! the geometric wdist -- this reports what the wall function is actually
    ! modelling at this face, not a different distance definition). Closed
    ! form: y+ = rho*u_tau*d/mu, u_tau = sqrt(tau/rho), d = Re*mu/(rho*V)
    ! => y+ = Re*sqrt(cf/2). Diagnostic-only -- see wall_yplus_field, never
    ! called from set_visc_force's per-step hot path.
    pure subroutine wall_yplus(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, yplus)
        implicit none
        real, intent(in) :: r, dA(3), vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt
        real, intent(out) :: yplus
        real :: V, dA_mag, Vt_slip, cf, Re, tau
        call wall_core(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, &
                       V, dA_mag, Vt_slip, cf, Re, tau)
        yplus = Re * sqrt(cf * 0.5e0)
    end subroutine wall_yplus

    pure subroutine wall_func_iface(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, i, j, k, di, flow)
        implicit none
        real, intent(in), contiguous :: r(:,:,:), rho(:,:,:), Vx(:,:,:), Vr(:,:,:), Vt(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall, mu
        integer, intent(in) :: i, j, k, di
        real :: Vxf, Vrf, Vtf, rf, rhof
        real, intent(out) :: flow(4)
        Vxf  = iface(Vx,  i+di, j, k) * 0.25e0
        Vrf  = iface(Vr,  i+di, j, k) * 0.25e0
        Vtf  = iface(Vt,  i+di, j, k) * 0.25e0
        rhof = iface(rho, i+di, j, k) * 0.25e0
        rf   = iface(r, i, j, k) * 0.25e0
        call wall_func(rf, dA(:,i,j,k), vol(i+(di-1)/2,j,k), Omega_block, Omega_wall, mu, rhof, Vxf, Vrf, Vtf, flow)
        flow = flow * di
    end subroutine wall_func_iface

    pure subroutine wall_func_jface(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, i, j, k, dj, flow)
        implicit none
        real, intent(in), contiguous :: r(:,:,:), rho(:,:,:), Vx(:,:,:), Vr(:,:,:), Vt(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall, mu
        integer, intent(in) :: i, j, k, dj
        real :: Vxf, Vrf, Vtf, rf, rhof
        real, intent(out) :: flow(4)
        Vxf  = jface(Vx,  i, j+dj, k) * 0.25e0
        Vrf  = jface(Vr,  i, j+dj, k) * 0.25e0
        Vtf  = jface(Vt,  i, j+dj, k) * 0.25e0
        rhof = jface(rho, i, j+dj, k) * 0.25e0
        rf   = jface(r, i, j, k) * 0.25e0
        call  wall_func(rf, dA(:,i,j,k), vol(i,j+(dj-1)/2,k), Omega_block, Omega_wall, mu, rhof, Vxf, Vrf, Vtf, flow)
        flow = flow * dj
    end subroutine wall_func_jface

    pure subroutine wall_func_kface(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, i, j, k, dk, flow)
        implicit none
        real, intent(in), contiguous :: r(:,:,:), rho(:,:,:), Vx(:,:,:), Vr(:,:,:), Vt(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall, mu
        integer, intent(in) :: i, j, k, dk
        real :: Vxf, Vrf, Vtf, rf, rhof
        real, intent(out) :: flow(4)
        Vxf  = kface(Vx,  i, j, k+dk) * 0.25e0
        Vrf  = kface(Vr,  i, j, k+dk) * 0.25e0
        Vtf  = kface(Vt,  i, j, k+dk) * 0.25e0
        rhof = kface(rho, i, j, k+dk) * 0.25e0
        rf   = kface(r, i, j, k) * 0.25e0
        call wall_func(rf, dA(:,i,j,k), vol(i,j,k+(dk-1)/2), Omega_block, Omega_wall, mu, rhof, Vxf, Vrf, Vtf, flow)
        flow = flow * dk
    end subroutine wall_func_kface

    ! Diagnostic y+ counterparts of wall_func_iface/jface/kface above -- same
    ! face-averaging, calling wall_yplus instead of wall_func. No `* di/dj/dk`
    ! sign multiply: y+ has no direction, unlike a flux vector. Used only by
    ! wall_yplus_field (post-processing), never set_visc_force.
    pure subroutine wall_yplus_iface(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, i, j, k, di, yplus)
        implicit none
        real, intent(in), contiguous :: r(:,:,:), rho(:,:,:), Vx(:,:,:), Vr(:,:,:), Vt(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall, mu
        integer, intent(in) :: i, j, k, di
        real :: Vxf, Vrf, Vtf, rf, rhof
        real, intent(out) :: yplus
        Vxf  = iface(Vx,  i+di, j, k) * 0.25e0
        Vrf  = iface(Vr,  i+di, j, k) * 0.25e0
        Vtf  = iface(Vt,  i+di, j, k) * 0.25e0
        rhof = iface(rho, i+di, j, k) * 0.25e0
        rf   = iface(r, i, j, k) * 0.25e0
        call wall_yplus(rf, dA(:,i,j,k), vol(i+(di-1)/2,j,k), Omega_block, Omega_wall, mu, rhof, Vxf, Vrf, Vtf, yplus)
    end subroutine wall_yplus_iface

    pure subroutine wall_yplus_jface(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, i, j, k, dj, yplus)
        implicit none
        real, intent(in), contiguous :: r(:,:,:), rho(:,:,:), Vx(:,:,:), Vr(:,:,:), Vt(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall, mu
        integer, intent(in) :: i, j, k, dj
        real :: Vxf, Vrf, Vtf, rf, rhof
        real, intent(out) :: yplus
        Vxf  = jface(Vx,  i, j+dj, k) * 0.25e0
        Vrf  = jface(Vr,  i, j+dj, k) * 0.25e0
        Vtf  = jface(Vt,  i, j+dj, k) * 0.25e0
        rhof = jface(rho, i, j+dj, k) * 0.25e0
        rf   = jface(r, i, j, k) * 0.25e0
        call wall_yplus(rf, dA(:,i,j,k), vol(i,j+(dj-1)/2,k), Omega_block, Omega_wall, mu, rhof, Vxf, Vrf, Vtf, yplus)
    end subroutine wall_yplus_jface

    pure subroutine wall_yplus_kface(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, i, j, k, dk, yplus)
        implicit none
        real, intent(in), contiguous :: r(:,:,:), rho(:,:,:), Vx(:,:,:), Vr(:,:,:), Vt(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall, mu
        integer, intent(in) :: i, j, k, dk
        real :: Vxf, Vrf, Vtf, rf, rhof
        real, intent(out) :: yplus
        Vxf  = kface(Vx,  i, j, k+dk) * 0.25e0
        Vrf  = kface(Vr,  i, j, k+dk) * 0.25e0
        Vtf  = kface(Vt,  i, j, k+dk) * 0.25e0
        rhof = kface(rho, i, j, k+dk) * 0.25e0
        rf   = kface(r, i, j, k) * 0.25e0
        call wall_yplus(rf, dA(:,i,j,k), vol(i,j,k+(dk-1)/2), Omega_block, Omega_wall, mu, rhof, Vxf, Vrf, Vtf, yplus)
    end subroutine wall_yplus_kface

    ! One k-face viscous flux at face plane k: identical arithmetic to the
    ! k-direction face loop in set_visc_force. Used only by the O(surface)
    ! cusp-seam correction pass there, never by the hot slab loops (which keep
    ! the body inlined for vectorization). tau_cell/q_cell are halo-indexed
    ! (ni+1, nj+1, nk+1, 6/3), component-last.
    pure subroutine kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, k, flow)
        implicit none
        real, intent(in), contiguous :: tau_cell(:,:,:,:), q_cell(:,:,:,:)
        real, intent(in), contiguous :: Vx(:,:,:), Vr(:,:,:), Vt(:,:,:), r(:,:,:)
        real, intent(in), contiguous :: dAk(:,:,:,:)
        real, intent(in) :: Omega_block
        integer, intent(in) :: i, j, k
        real, intent(out) :: flow(4)
        real :: tauf(6), qf(3), Vf(3), rf, Vabs, wvisc(3)
        tauf(1) = (tau_cell(i+1, j+1, k, 1) + tau_cell(i+1, j+1, k+1, 1)) * 0.5e0
        tauf(2) = (tau_cell(i+1, j+1, k, 2) + tau_cell(i+1, j+1, k+1, 2)) * 0.5e0
        tauf(3) = (tau_cell(i+1, j+1, k, 3) + tau_cell(i+1, j+1, k+1, 3)) * 0.5e0
        tauf(4) = (tau_cell(i+1, j+1, k, 4) + tau_cell(i+1, j+1, k+1, 4)) * 0.5e0
        tauf(5) = (tau_cell(i+1, j+1, k, 5) + tau_cell(i+1, j+1, k+1, 5)) * 0.5e0
        tauf(6) = (tau_cell(i+1, j+1, k, 6) + tau_cell(i+1, j+1, k+1, 6)) * 0.5e0
        qf(1)   = (q_cell(i+1, j+1, k, 1) + q_cell(i+1, j+1, k+1, 1)) * 0.5e0
        qf(2)   = (q_cell(i+1, j+1, k, 2) + q_cell(i+1, j+1, k+1, 2)) * 0.5e0
        qf(3)   = (q_cell(i+1, j+1, k, 3) + q_cell(i+1, j+1, k+1, 3)) * 0.5e0
        Vf(1) = (Vx(i,j,k) + Vx(i+1,j,k) + Vx(i,j+1,k) + Vx(i+1,j+1,k)) * 0.25e0
        Vf(2) = (Vr(i,j,k) + Vr(i+1,j,k) + Vr(i,j+1,k) + Vr(i+1,j+1,k)) * 0.25e0
        Vf(3) = (Vt(i,j,k) + Vt(i+1,j,k) + Vt(i,j+1,k) + Vt(i+1,j+1,k)) * 0.25e0
        rf     = (r(i,j,k)  + r(i+1,j,k)  + r(i,j+1,k)  + r(i+1,j+1,k))  * 0.25e0
        Vabs = Vf(3) + Omega_block * rf
        flow(1) = tauf(1)*dAk(1,i,j,k) + tauf(4)*dAk(2,i,j,k) + tauf(5)*dAk(3,i,j,k)
        flow(2) = tauf(4)*dAk(1,i,j,k) + tauf(2)*dAk(2,i,j,k) + tauf(6)*dAk(3,i,j,k)
        flow(3) = (tauf(5)*dAk(1,i,j,k) + tauf(6)*dAk(2,i,j,k) + tauf(3)*dAk(3,i,j,k)) * rf
        wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
        wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
        wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
        flow(4) = (wvisc(1)-qf(1))*dAk(1,i,j,k) &
                + (wvisc(2)-qf(2))*dAk(2,i,j,k) &
                + (wvisc(3)-qf(3))*dAk(3,i,j,k)
    end subroutine kface_flow

    ! Polar (radial-momentum) source per unit volume for cell (i,j,k):
    !     S = (rho*Vt^2 + (P - P_offset)) / r
    ! Identical arithmetic to production's trailing pass, factored out only so
    ! the hot fused loop and the O(surface) boundary-shell pass cannot drift
    ! apart. Bitwise agreement with production depends on this staying an
    ! expression-for-expression copy of it.
    pure function polar_src(cons_cell, P, r, P_offset, i, j, k) result(S)
        implicit none
        real, intent(in), contiguous :: cons_cell(:,:,:,:), P(:,:,:), r(:,:,:)
        real, intent(in) :: P_offset
        integer, intent(in) :: i, j, k
        real :: S
        real :: rhoc, rhorVtc, rc, Pc, Vtc
        rhoc    = cons_cell(i, j, k, 1)
        rhorVtc = cons_cell(i, j, k, 4)
        rc = 0.125e0 * ( &
            r(i,j,k) + r(i+1,j,k) + r(i,j+1,k) + r(i+1,j+1,k) + &
            r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
        Pc = 0.125e0 * ( &
            P(i,j,k) + P(i+1,j,k) + P(i,j+1,k) + P(i+1,j+1,k) + &
            P(i,j,k+1) + P(i+1,j,k+1) + P(i,j+1,k+1) + P(i+1,j+1,k+1))
        Vtc = rhorVtc / (rhoc * rc)
        S = ((Pc - P_offset) + rhoc * Vtc**2) / rc
    end function polar_src

    ! Scale the tau/q boundary halos by (2*wall - 1): wall=0 gives -edge so the
    ! face average is zero, wall=1 keeps +edge for the single-sided stress.
    subroutine scale_visc_halos(tau_cell, q_cell, &
        walli1, wallj1, wallk1, wallni, wallnj, wallnk, ni, nj, nk)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(inout) :: tau_cell(ni+1, nj+1, nk+1, 6)
        real, intent(inout) :: q_cell(ni+1, nj+1, nk+1, 3)
        real, intent(in) :: walli1(nj-1, nk-1), wallni(nj-1, nk-1)
        real, intent(in) :: wallj1(ni-1, nk-1), wallnj(ni-1, nk-1)
        real, intent(in) :: wallk1(ni-1, nj-1), wallnk(ni-1, nj-1)
        integer :: i, j, k
        do k = 1, nk-1
        do j = 1, nj-1
            tau_cell(1, j+1, k+1, :) = tau_cell(1, j+1, k+1, :) * (2.0e0*walli1(j,k) - 1.0e0)
            q_cell(1, j+1, k+1, :) = q_cell(1, j+1, k+1, :) * (2.0e0*walli1(j,k) - 1.0e0)
        end do
        end do
        do k = 1, nk-1
        do j = 1, nj-1
            tau_cell(ni+1, j+1, k+1, :) = tau_cell(ni+1, j+1, k+1, :) * (2.0e0*wallni(j,k) - 1.0e0)
            q_cell(ni+1, j+1, k+1, :) = q_cell(ni+1, j+1, k+1, :) * (2.0e0*wallni(j,k) - 1.0e0)
        end do
        end do
        do k = 1, nk-1
        do i = 1, ni-1
            tau_cell(i+1, 1, k+1, :) = tau_cell(i+1, 1, k+1, :) * (2.0e0*wallj1(i,k) - 1.0e0)
            q_cell(i+1, 1, k+1, :) = q_cell(i+1, 1, k+1, :) * (2.0e0*wallj1(i,k) - 1.0e0)
        end do
        end do
        do k = 1, nk-1
        do i = 1, ni-1
            tau_cell(i+1, nj+1, k+1, :) = tau_cell(i+1, nj+1, k+1, :) * (2.0e0*wallnj(i,k) - 1.0e0)
            q_cell(i+1, nj+1, k+1, :) = q_cell(i+1, nj+1, k+1, :) * (2.0e0*wallnj(i,k) - 1.0e0)
        end do
        end do
        do j = 1, nj-1
        do i = 1, ni-1
            tau_cell(i+1, j+1, 1, :) = tau_cell(i+1, j+1, 1, :) * (2.0e0*wallk1(i,j) - 1.0e0)
            q_cell(i+1, j+1, 1, :) = q_cell(i+1, j+1, 1, :) * (2.0e0*wallk1(i,j) - 1.0e0)
        end do
        end do
        do j = 1, nj-1
        do i = 1, ni-1
            tau_cell(i+1, j+1, nk+1, :) = tau_cell(i+1, j+1, nk+1, :) * (2.0e0*wallnk(i,j) - 1.0e0)
            q_cell(i+1, j+1, nk+1, :) = q_cell(i+1, j+1, nk+1, :) * (2.0e0*wallnk(i,j) - 1.0e0)
        end do
        end do
    end subroutine scale_visc_halos

    ! zero_wall_fvisc for set_visc_force, whose fused store has ALREADY
    ! applied the i-mask to every row interior in j and k. Only the border of
    ! the i-sheet is left: the rows that also carry a j- or k-mask, and so
    ! could not be finished inside the store. O(nj+nk) instead of O(nj*nk),
    ! which is the strided traffic the opt-report flagged.
    subroutine zero_wall_fvisc_border(fvisc, walli1, wallj1, wallk1, &
        wallni, wallnj, wallnk, ni, nj, nk)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(inout) :: fvisc(ni-1, nj-1, nk-1, 4)
        real, intent(in) :: walli1(nj-1, nk-1), wallni(nj-1, nk-1)
        real, intent(in) :: wallj1(ni-1, nk-1), wallnj(ni-1, nk-1)
        real, intent(in) :: wallk1(ni-1, nj-1), wallnk(ni-1, nj-1)
        integer :: i, j, k, m
        ! i-mask on the j-boundary rows (all k), then on the k-boundary rows
        ! for interior j -- each (j,k) visited exactly once, as production's
        ! full-sheet loop does.
        do m = 1, 4
        do k = 1, nk-1
            fvisc(1,1,k,m)       = fvisc(1,1,k,m)       * walli1(1,k)
            fvisc(ni-1,1,k,m)    = fvisc(ni-1,1,k,m)    * wallni(1,k)
            fvisc(1,nj-1,k,m)    = fvisc(1,nj-1,k,m)    * walli1(nj-1,k)
            fvisc(ni-1,nj-1,k,m) = fvisc(ni-1,nj-1,k,m) * wallni(nj-1,k)
        end do
        end do
        do m = 1, 4
        do j = 2, nj-2
            fvisc(1,j,1,m)       = fvisc(1,j,1,m)       * walli1(j,1)
            fvisc(ni-1,j,1,m)    = fvisc(ni-1,j,1,m)    * wallni(j,1)
            fvisc(1,j,nk-1,m)    = fvisc(1,j,nk-1,m)    * walli1(j,nk-1)
            fvisc(ni-1,j,nk-1,m) = fvisc(ni-1,j,nk-1,m) * wallni(j,nk-1)
        end do
        end do
        do k = 1, nk-1
        do i = 1, ni-1
            fvisc(i,1,k,1)    = fvisc(i,1,k,1)    * wallj1(i,k)
            fvisc(i,1,k,2)    = fvisc(i,1,k,2)    * wallj1(i,k)
            fvisc(i,1,k,3)    = fvisc(i,1,k,3)    * wallj1(i,k)
            fvisc(i,1,k,4)    = fvisc(i,1,k,4)    * wallj1(i,k)
            fvisc(i,nj-1,k,1) = fvisc(i,nj-1,k,1) * wallnj(i,k)
            fvisc(i,nj-1,k,2) = fvisc(i,nj-1,k,2) * wallnj(i,k)
            fvisc(i,nj-1,k,3) = fvisc(i,nj-1,k,3) * wallnj(i,k)
            fvisc(i,nj-1,k,4) = fvisc(i,nj-1,k,4) * wallnj(i,k)
        end do
        end do
        do j = 1, nj-1
        do i = 1, ni-1
            fvisc(i,j,1,1)    = fvisc(i,j,1,1)    * wallk1(i,j)
            fvisc(i,j,1,2)    = fvisc(i,j,1,2)    * wallk1(i,j)
            fvisc(i,j,1,3)    = fvisc(i,j,1,3)    * wallk1(i,j)
            fvisc(i,j,1,4)    = fvisc(i,j,1,4)    * wallk1(i,j)
            fvisc(i,j,nk-1,1) = fvisc(i,j,nk-1,1) * wallnk(i,j)
            fvisc(i,j,nk-1,2) = fvisc(i,j,nk-1,2) * wallnk(i,j)
            fvisc(i,j,nk-1,3) = fvisc(i,j,nk-1,3) * wallnk(i,j)
            fvisc(i,j,nk-1,4) = fvisc(i,j,nk-1,4) * wallnk(i,j)
        end do
        end do
    end subroutine zero_wall_fvisc_border

end module viscous_helpers

! The viscous calculation, in two kernels
! ---------------------------------------
! The grid has (ni x nj x nk) nodes defining (ni-1 x nj-1 x nk-1) cells.
! Face area vectors dAi/dAj/dAk point inward (low-index direction positive).
! The wall arrays (walli1, wallni, etc.) are blending weights: 1 = free-stream
! viscous stress, 0 = wall function. Values in between blend both.
!
! Nodal Vx, Vr, Vt are inputs, cached on the block: Vt is relative
! (frame-subtracted), and Omega*r is added back where absolute velocity is
! needed (the shear-work terms).
!
! `set_tau_q_soa` computes, for every cell, the stress tensor tau(6) and
! heat-flux vector q(3) by a Green-Gauss gradient over the six cell faces,
! each face average being the mean of its four corner nodes. tau and q are
! stored multiplied by 2, so that averaging two adjacent cells recovers the
! correct face value without a further factor.
!
! `set_visc_force` then accumulates the viscous face flows into fvisc. The two
! are separate kernels, not two passes of one, because a grid-wide periodic
! seam halo exchange runs between them (grid.py's update_sources).
!
! Interior faces take tauf as the average of tau_cell from the two adjacent
! cells; boundary faces (i=1, i=ni, j=1, j=nj, k=1, k=nk) take it from the
! single adjacent interior cell (already doubled above, so no extra factor)
! and blend the free-stream viscous stress with a wall-function force
! according to the wall weight.

! Strip-mined / SoA evaluation of tau and q: the per-cell work is split into
! two flat do-i loops over each (j,k) row, with the per-cell intermediates held
! in row temps dimensioned with i as the contiguous axis. This gives the auto-
! vectorizer simple, call-light, unit-stride loops to vectorize over i instead
! of one deep nest with a heavy inlined body.
!
! Three modelling choices are built into the arithmetic below and are not
! obvious from it:
!
!   Normal stresses are 2*mu*strain, with no -2/3 div(V) deviatoric term,
!   matching Multall NEW_LOSS (TXX = 2*VISTOT*DVXDX).
!
!   The radial derivative uses the Multall GRADVEL form: the same plain face
!   sums as the axial/tangential derivatives, dotted with the radial area, plus
!   a single cell-level metric correction -x_cell/rc (0.125*(the two i-face
!   sums) being x_cell). This avoids a second set of six 1/r-weighted face
!   sums. q's flux sum carries the opposite sign convention, so its correction
!   is +T_cell/rc rather than -T_cell/rc, and vanishes for uniform T either way.
!
!   The vorticity magnitude driving mu_turb = rho * l^2 * |omega| is taken in
!   the RELATIVE frame, with no absolute-frame +2*Omega correction, i.e. the
!   block-frame value as differenced from Vt_rel. The strain rate itself is
!   frame-invariant regardless, since the rigid rotation Omega*r contributes
!   only an antisymmetric part.
subroutine set_tau_q_soa( &
    cons, T, mu, cp, Pr_lam, Pr_turb, xlength, vol, dAi, dAj, dAk, &
    r, &
    Vx, Vr, Vt, &
    tau_cell, &
    q_cell, &
    mu_turb, &
    ni, nj, nk)

    use viscous_helpers
    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: T(ni, nj, nk)
    real, intent(in) :: mu, Pr_lam, Pr_turb
    ! Nodal, unlike mu: a real gas's specific heat varies over the field,
    ! and freezing it at one state was worth as much as the whole spread of
    ! cp over a fit box. Averaged to the cell below, like rho.
    real, intent(in) :: cp(ni, nj, nk)
    real, intent(in) :: xlength(ni-1, nj-1, nk-1)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Vx(ni, nj, nk)
    real, intent(in) :: Vr(ni, nj, nk)
    real, intent(in) :: Vt(ni, nj, nk)
    real, intent(inout) :: tau_cell(ni+1, nj+1, nk+1, 6)
    real, intent(inout) :: q_cell(ni+1, nj+1, nk+1, 3)
    ! Cell-centred mixing-length turbulent viscosity, written at the cell's
    ! low-corner node (i,j,k). The final node in each axis is padding that is
    ! not written here and must not be read. intent(inout) so that padding is
    ! left untouched rather than becoming undefined.
    real, intent(inout) :: mu_turb(ni, nj, nk)

    integer :: i, j, k
    real :: visc_lim
    ! Row temps -- i is the contiguous (dim-1) axis, the SIMD lane index.
    real :: gVx(ni-1, 3), gVr(ni-1, 3), gVt(ni-1, 3)
    real :: vct(ni-1), rcr(ni-1), ivr(ni-1), rhoc(ni-1), cpc(ni-1)
    real :: f1, f2, f3, f4, f5, f6, g1, g2, g3
    real :: t1, t2, t3, t4, t5, t6, w1, w2, w3, vm, mut, fac, lambda

    visc_lim = 3000e0 * mu

    do k = 1, nk-1
    do j = 1, nj-1
        ! Stage 1: velocity gradients + cell metrics, vectorizable over i.
        do i = 1, ni-1
            ivr(i) = 0.25e0 / vol(i,j,k)
            rcr(i) = 0.125e0 * (r(i,j,k)   + r(i+1,j,k)   + r(i,j+1,k)   + r(i+1,j+1,k) &
                              + r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
            rhoc(i) = 0.125e0 * (cons(i,j,k,1)   + cons(i+1,j,k,1)   + cons(i,j+1,k,1)   + cons(i+1,j+1,k,1) &
                               + cons(i,j,k+1,1) + cons(i+1,j,k+1,1) + cons(i,j+1,k+1,1) + cons(i+1,j+1,k+1,1))
            cpc(i) = 0.125e0 * (cp(i,j,k)   + cp(i+1,j,k)   + cp(i,j+1,k)   + cp(i+1,j+1,k) &
                              + cp(i,j,k+1) + cp(i+1,j,k+1) + cp(i,j+1,k+1) + cp(i+1,j+1,k+1))
            ! --- Vx ---
            f1 = Vx(i,j,k)+Vx(i,j+1,k)+Vx(i,j,k+1)+Vx(i,j+1,k+1)
            f2 = Vx(i+1,j,k)+Vx(i+1,j+1,k)+Vx(i+1,j,k+1)+Vx(i+1,j+1,k+1)
            f3 = Vx(i,j,k)+Vx(i+1,j,k)+Vx(i,j,k+1)+Vx(i+1,j,k+1)
            f4 = Vx(i,j+1,k)+Vx(i+1,j+1,k)+Vx(i,j+1,k+1)+Vx(i+1,j+1,k+1)
            f5 = Vx(i,j,k)+Vx(i+1,j,k)+Vx(i,j+1,k)+Vx(i+1,j+1,k)
            f6 = Vx(i,j,k+1)+Vx(i+1,j,k+1)+Vx(i,j+1,k+1)+Vx(i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVx(i,1) = g1*ivr(i)
            gVx(i,3) = g3*ivr(i)
            gVx(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vr ---
            f1 = Vr(i,j,k)+Vr(i,j+1,k)+Vr(i,j,k+1)+Vr(i,j+1,k+1)
            f2 = Vr(i+1,j,k)+Vr(i+1,j+1,k)+Vr(i+1,j,k+1)+Vr(i+1,j+1,k+1)
            f3 = Vr(i,j,k)+Vr(i+1,j,k)+Vr(i,j,k+1)+Vr(i+1,j,k+1)
            f4 = Vr(i,j+1,k)+Vr(i+1,j+1,k)+Vr(i,j+1,k+1)+Vr(i+1,j+1,k+1)
            f5 = Vr(i,j,k)+Vr(i+1,j,k)+Vr(i,j+1,k)+Vr(i+1,j+1,k)
            f6 = Vr(i,j,k+1)+Vr(i+1,j,k+1)+Vr(i,j+1,k+1)+Vr(i+1,j+1,k+1)
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVr(i,1) = g1*ivr(i)
            gVr(i,3) = g3*ivr(i)
            gVr(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vt ---
            f1 = Vt(i,j,k)+Vt(i,j+1,k)+Vt(i,j,k+1)+Vt(i,j+1,k+1)
            f2 = Vt(i+1,j,k)+Vt(i+1,j+1,k)+Vt(i+1,j,k+1)+Vt(i+1,j+1,k+1)
            f3 = Vt(i,j,k)+Vt(i+1,j,k)+Vt(i,j,k+1)+Vt(i+1,j,k+1)
            f4 = Vt(i,j+1,k)+Vt(i+1,j+1,k)+Vt(i,j+1,k+1)+Vt(i+1,j+1,k+1)
            f5 = Vt(i,j,k)+Vt(i+1,j,k)+Vt(i,j+1,k)+Vt(i+1,j+1,k)
            f6 = Vt(i,j,k+1)+Vt(i+1,j,k+1)+Vt(i,j+1,k+1)+Vt(i+1,j+1,k+1)
            vct(i) = (f1+f2)*0.125e0
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVt(i,1) = g1*ivr(i)
            gVt(i,3) = g3*ivr(i)
            gVt(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
        end do
        ! Stage 2: tau, mixing-length mu_turb, and q -- store with stride-1
        ! per-component writes; vectorizable over i.
        do i = 1, ni-1
            t1 = gVx(i,1)
            t2 = gVr(i,2)
            t3 = gVt(i,3)
            t4 = gVx(i,2) + gVr(i,1)
            t5 = gVx(i,3) + gVt(i,1)
            t6 = gVr(i,3) + gVt(i,2) - vct(i)/rcr(i)
            w1 = gVt(i,2) - gVr(i,3) + vct(i)/rcr(i)
            w2 = gVx(i,3) - gVt(i,1)
            w3 = gVr(i,1) - gVx(i,2)
            vm = sqrt(w1*w1 + w2*w2 + w3*w3)
            ! The max(0) is not physics -- mut is analytically confined to
            ! [0, visc_lim] already, since rhoc, xlength and vm are all
            ! non-negative and nothing here divides. It contains a gfortran 13
            ! codegen fault: built with the setup.py production flags (the
            ! raised --param=vect-max-version-for-alias-checks is what forces
            ! this loop to vectorize at all), gfortran 13.3 returns -inf from
            ! this line for about two thirds of cells whenever the vorticity
            ! sits at the float32 cancellation noise floor, i.e. a uniform flow
            ! with no real shear -- the chi = 90/270 duct cases in
            ! tests/test_nonreflecting_integration.py. The -inf then travels
            ! fac -> tau_cell -> F_body -> residual and NaNs the whole field
            ! inside one step. gfortran 14.2 is unaffected, as is any build at
            ! -O2, without fast-math, or at the default alias-check budget.
            ! Ubuntu 24.04 (the CI runner) ships gfortran 13.3, so this is a
            ! live target, not a historical one. The clamp costs one vmaxps
            ! and leaves the loop vectorized.
            mut = max(0.0e0, min(rhoc(i) * xlength(i,j,k) * vm, visc_lim))
            mu_turb(i,j,k) = mut
            fac = (mu + mut) * 0.5e0
            tau_cell(i+1,j+1,k+1,1) = t1*fac
            tau_cell(i+1,j+1,k+1,2) = t2*fac
            tau_cell(i+1,j+1,k+1,3) = t3*fac
            tau_cell(i+1,j+1,k+1,4) = t4*fac
            tau_cell(i+1,j+1,k+1,5) = t5*fac
            tau_cell(i+1,j+1,k+1,6) = t6*fac
            lambda = (mu/Pr_lam + mut/Pr_turb) * cpc(i)
            f1 = T(i,j,k)+T(i,j+1,k)+T(i,j,k+1)+T(i,j+1,k+1)
            f2 = T(i+1,j,k)+T(i+1,j+1,k)+T(i+1,j,k+1)+T(i+1,j+1,k+1)
            f3 = T(i,j,k)+T(i+1,j,k)+T(i,j,k+1)+T(i+1,j,k+1)
            f4 = T(i,j+1,k)+T(i+1,j+1,k)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            f5 = T(i,j,k)+T(i+1,j,k)+T(i,j+1,k)+T(i+1,j+1,k)
            f6 = T(i,j,k+1)+T(i+1,j,k+1)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            q_cell(i+1,j+1,k+1,1) = (f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k) &
                  -f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            q_cell(i+1,j+1,k+1,3) = (f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k) &
                  -f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            q_cell(i+1,j+1,k+1,2) = ((f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k) &
                  -f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))*ivr(i) &
                  + 0.125e0*(f1+f2)/rcr(i)) * (lambda*0.5e0)
        end do
    end do
    end do

    ! Fill boundary halo slots with +edge (identical to set_tau_q).
    do k = 1, nk-1
    do j = 1, nj-1
        tau_cell(1, j+1, k+1, :) = tau_cell(2, j+1, k+1, :)
        q_cell(1, j+1, k+1, :) = q_cell(2, j+1, k+1, :)
        tau_cell(ni+1, j+1, k+1, :) = tau_cell(ni, j+1, k+1, :)
        q_cell(ni+1, j+1, k+1, :) = q_cell(ni, j+1, k+1, :)
    end do
    end do
    do k = 1, nk-1
    do i = 1, ni-1
        tau_cell(i+1, 1, k+1, :) = tau_cell(i+1, 2, k+1, :)
        q_cell(i+1, 1, k+1, :) = q_cell(i+1, 2, k+1, :)
        tau_cell(i+1, nj+1, k+1, :) = tau_cell(i+1, nj, k+1, :)
        q_cell(i+1, nj+1, k+1, :) = q_cell(i+1, nj, k+1, :)
    end do
    end do
    do j = 1, nj-1
    do i = 1, ni-1
        tau_cell(i+1, j+1, 1, :) = tau_cell(i+1, j+1, 2, :)
        q_cell(i+1, j+1, 1, :) = q_cell(i+1, j+1, 2, :)
        tau_cell(i+1, j+1, nk+1, :) = tau_cell(i+1, j+1, nk, :)
        q_cell(i+1, j+1, nk+1, :) = q_cell(i+1, j+1, nk, :)
    end do
    end do

end subroutine set_tau_q_soa


! Pass 2 of a split viscous calculation: given tau_cell and q_cell (which may
! have been exchanged across periodic boundaries since eval_tau_q returned),
! compute face fluxes and accumulate into fvisc.
!
! tau_cell and q_cell are halo-dimensioned (ni+1, nj+1, nk+1, 6/3): owned
! cells sit at indices 2..ni (2..nj, 2..nk).  Halo slots 1 and ni+1 (etc.)
! will carry neighbour data after exchange; until then the boundary loops
! below are single-sided and use only the nearest owned cell, as before.
!
! k-slab cache blocking and rolling-buffer fusion
! -----------------------------------------------
! The three face-direction sweeps are tiled over slabs of kb cell planes
! (1 <= kb <= nk-1) so that a slab's tau_cell/q_cell planes stay hot in cache
! across all three directions: tau/q is then streamed from memory roughly once
! instead of once per direction. Within a slab the i-, j- and k-direction
! sweeps run back to back, and each fuses its face-flux loop with its fvisc
! accumulate through a rolling buffer, so no slab-sized flow scratch exists:
!   - i-direction: one face row (rows slot 1) per (j,k), differenced in place;
!   - j-direction: an alternating face-row pair (rows slots 2/3);
!   - k-direction: an alternating face-plane pair (planes slots 1/2).
! The per-cell arithmetic and its ordering (i, then j, then k) are identical
! to the staged version, so the result differs only by float reassociation at
! the cusp seam (see below).
!
! The k direction couples adjacent slabs: a slab's low k-face plane is the
! previous slab's high plane. The rolling plane pair persists across the slab
! boundary (the intervening i/j phases touch only rows), so the carry is
! automatic, and the carried plane preserves the k=2 / k=nk-1 wall-function
! injections for both adjacent cells.
!
! The cusp seam (k=1 face coupled to k=nk) is inherently non-local in k, so it
! is handled by an O(surface) correction pass after the slab sweep instead of
! the pre-accumulation flux averaging the unblocked version used; see the
! comment at the correction loop.
subroutine set_visc_force( &
    cons, cons_cell, vol, dAi, dAj, dAk, &
    Omega_block, r, mu, P, P_offset, &
    fvisc, &
    Vx, Vr, Vt, &
    tau_cell, &
    q_cell, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    Omega_walli1_nd, Omega_wallj1_nd, Omega_wallk1_nd, &
    Omega_wallni_nd, Omega_wallnj_nd, Omega_wallnk_nd, &
    i_cusp_start, i_cusp_end, &
    kb, ni, nj, nk)

    use viscous_helpers
    implicit none

    integer, intent(in) :: ni, nj, nk, kb
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: cons_cell(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega_block, mu
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(inout) :: fvisc(ni-1, nj-1, nk-1, 4)
    real, intent(in) :: Vx(ni, nj, nk)
    real, intent(in) :: Vr(ni, nj, nk)
    real, intent(in) :: Vt(ni, nj, nk)
    real, intent(inout) :: tau_cell(ni+1, nj+1, nk+1, 6)
    real, intent(inout) :: q_cell(ni+1, nj+1, nk+1, 3)
    real, intent(inout) :: planes(ni, nj, 4, 2)
    real, intent(inout) :: rows(ni, 4, 3)
    real, intent(in) :: walli1(nj-1, nk-1)
    real, intent(in) :: wallni(nj-1, nk-1)
    real, intent(in) :: wallj1(ni-1, nk-1)
    real, intent(in) :: wallnj(ni-1, nk-1)
    real, intent(in) :: wallk1(ni-1, nj-1)
    real, intent(in) :: wallnk(ni-1, nj-1)
    real, intent(in) :: Omega_walli1_nd(nj-1, nk-1)
    real, intent(in) :: Omega_wallni_nd(nj-1, nk-1)
    real, intent(in) :: Omega_wallj1_nd(ni-1, nk-1)
    real, intent(in) :: Omega_wallnj_nd(ni-1, nk-1)
    real, intent(in) :: Omega_wallk1_nd(ni-1, nj-1)
    real, intent(in) :: Omega_wallnk_nd(ni-1, nj-1)
    integer, intent(in) :: i_cusp_start, i_cusp_end

    integer :: i, j, k, jc, kc
    logical :: k_interior, row_interior
    integer :: sa, sb, pa, pb, stmp
    real :: tauf(6), qf(3), Vf(3), rf
    real :: wvisc(3), Vabs, wf(4), wfac
    real :: flow1(4), flownk(4), fcorr(4)

    ! kb is inert here -- the fused schedule subsumes k-slab blocking (see the
    ! file header) -- but it stays in the signature so these arms share one
    ! kwargs dict with production. Consumed as a sanity guard rather than
    ! silenced, so a caller passing a nonsense slab depth still fails loudly.
    if (kb < 1) return

    call scale_visc_halos(tau_cell, q_cell, &
        walli1, wallj1, wallk1, wallni, wallnj, wallnk, ni, nj, nk)

    pa = 1
    pb = 2

    do k = 1, nk
    ! --- k-face plane k into the rolling pair ---
    do j = 1, nj-1
    do i = 1, ni-1
        tauf(1) = (tau_cell(i+1, j+1, k, 1) + tau_cell(i+1, j+1, k+1, 1)) * 0.5e0
        tauf(2) = (tau_cell(i+1, j+1, k, 2) + tau_cell(i+1, j+1, k+1, 2)) * 0.5e0
        tauf(3) = (tau_cell(i+1, j+1, k, 3) + tau_cell(i+1, j+1, k+1, 3)) * 0.5e0
        tauf(4) = (tau_cell(i+1, j+1, k, 4) + tau_cell(i+1, j+1, k+1, 4)) * 0.5e0
        tauf(5) = (tau_cell(i+1, j+1, k, 5) + tau_cell(i+1, j+1, k+1, 5)) * 0.5e0
        tauf(6) = (tau_cell(i+1, j+1, k, 6) + tau_cell(i+1, j+1, k+1, 6)) * 0.5e0
        qf(1)   = (q_cell(i+1, j+1, k, 1) + q_cell(i+1, j+1, k+1, 1)) * 0.5e0
        qf(2)   = (q_cell(i+1, j+1, k, 2) + q_cell(i+1, j+1, k+1, 2)) * 0.5e0
        qf(3)   = (q_cell(i+1, j+1, k, 3) + q_cell(i+1, j+1, k+1, 3)) * 0.5e0
        Vf(1) = (Vx(i,j,k) + Vx(i+1,j,k) + Vx(i,j+1,k) + Vx(i+1,j+1,k)) * 0.25e0
        Vf(2) = (Vr(i,j,k) + Vr(i+1,j,k) + Vr(i,j+1,k) + Vr(i+1,j+1,k)) * 0.25e0
        Vf(3) = (Vt(i,j,k) + Vt(i+1,j,k) + Vt(i,j+1,k) + Vt(i+1,j+1,k)) * 0.25e0
        rf     = (r(i,j,k)  + r(i+1,j,k)  + r(i,j+1,k)  + r(i+1,j+1,k))  * 0.25e0
        Vabs = Vf(3) + Omega_block * rf
        planes(i,j,1,pb) = tauf(1)*dAk(1,i,j,k) + tauf(4)*dAk(2,i,j,k) + tauf(5)*dAk(3,i,j,k)
        planes(i,j,2,pb) = tauf(4)*dAk(1,i,j,k) + tauf(2)*dAk(2,i,j,k) + tauf(6)*dAk(3,i,j,k)
        planes(i,j,3,pb) = (tauf(5)*dAk(1,i,j,k) + tauf(6)*dAk(2,i,j,k) + tauf(3)*dAk(3,i,j,k)) * rf
        wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
        wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
        wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
        planes(i,j,4,pb) = (wvisc(1)-qf(1))*dAk(1,i,j,k) &
                         + (wvisc(2)-qf(2))*dAk(2,i,j,k) &
                         + (wvisc(3)-qf(3))*dAk(3,i,j,k)
    end do
    end do
    if (k == 2) then
        do j = 1, nj-1
        do i = 1, ni-1
            wfac = 1.0e0 - wallk1(i,j)
            call wall_func_kface(r, dAk, vol, Omega_block, Omega_wallk1_nd(i,j), mu, cons(:,:,:,1), Vx, Vr, Vt, i, j, 1, 1, wf)
            planes(i,j,1,pb) = wallk1(i,j)*planes(i,j,1,pb) + wfac*wf(1)
            planes(i,j,2,pb) = wallk1(i,j)*planes(i,j,2,pb) + wfac*wf(2)
            planes(i,j,3,pb) = wallk1(i,j)*planes(i,j,3,pb) + wfac*wf(3)
            planes(i,j,4,pb) = wallk1(i,j)*planes(i,j,4,pb) + wfac*wf(4)
        end do
        end do
    end if
    if (k == nk-1) then
        do j = 1, nj-1
        do i = 1, ni-1
            wfac = 1.0e0 - wallnk(i,j)
            call wall_func_kface(r, dAk, vol, Omega_block, Omega_wallnk_nd(i,j), mu, cons(:,:,:,1), Vx, Vr, Vt, i, j, nk, -1, wf)
            planes(i,j,1,pb) = wallnk(i,j)*planes(i,j,1,pb) + wfac*wf(1)
            planes(i,j,2,pb) = wallnk(i,j)*planes(i,j,2,pb) + wfac*wf(2)
            planes(i,j,3,pb) = wallnk(i,j)*planes(i,j,3,pb) + wfac*wf(3)
            planes(i,j,4,pb) = wallnk(i,j)*planes(i,j,4,pb) + wfac*wf(4)
        end do
        end do
    end if

    ! --- cell plane kc = k-1: i/j scan, one store per cell ---
    if (k > 1) then
        kc = k - 1
        k_interior = (kc >= 2 .and. kc <= nk-2)
        sa = 2
        sb = 3
        do j = 1, nj
            do i = 1, ni-1
                tauf(1) = (tau_cell(i+1, j, kc+1, 1) + tau_cell(i+1, j+1, kc+1, 1)) * 0.5e0
                tauf(2) = (tau_cell(i+1, j, kc+1, 2) + tau_cell(i+1, j+1, kc+1, 2)) * 0.5e0
                tauf(3) = (tau_cell(i+1, j, kc+1, 3) + tau_cell(i+1, j+1, kc+1, 3)) * 0.5e0
                tauf(4) = (tau_cell(i+1, j, kc+1, 4) + tau_cell(i+1, j+1, kc+1, 4)) * 0.5e0
                tauf(5) = (tau_cell(i+1, j, kc+1, 5) + tau_cell(i+1, j+1, kc+1, 5)) * 0.5e0
                tauf(6) = (tau_cell(i+1, j, kc+1, 6) + tau_cell(i+1, j+1, kc+1, 6)) * 0.5e0
                qf(1)   = (q_cell(i+1, j, kc+1, 1) + q_cell(i+1, j+1, kc+1, 1)) * 0.5e0
                qf(2)   = (q_cell(i+1, j, kc+1, 2) + q_cell(i+1, j+1, kc+1, 2)) * 0.5e0
                qf(3)   = (q_cell(i+1, j, kc+1, 3) + q_cell(i+1, j+1, kc+1, 3)) * 0.5e0
                Vf(1) = (Vx(i,j,kc) + Vx(i+1,j,kc) + Vx(i,j,kc+1) + Vx(i+1,j,kc+1)) * 0.25e0
                Vf(2) = (Vr(i,j,kc) + Vr(i+1,j,kc) + Vr(i,j,kc+1) + Vr(i+1,j,kc+1)) * 0.25e0
                Vf(3) = (Vt(i,j,kc) + Vt(i+1,j,kc) + Vt(i,j,kc+1) + Vt(i+1,j,kc+1)) * 0.25e0
                rf     = (r(i,j,kc)  + r(i+1,j,kc)  + r(i,j,kc+1)  + r(i+1,j,kc+1))  * 0.25e0
                Vabs = Vf(3) + Omega_block * rf
                rows(i,1,sb) = tauf(1)*dAj(1,i,j,kc) + tauf(4)*dAj(2,i,j,kc) + tauf(5)*dAj(3,i,j,kc)
                rows(i,2,sb) = tauf(4)*dAj(1,i,j,kc) + tauf(2)*dAj(2,i,j,kc) + tauf(6)*dAj(3,i,j,kc)
                rows(i,3,sb) = (tauf(5)*dAj(1,i,j,kc) + tauf(6)*dAj(2,i,j,kc) + tauf(3)*dAj(3,i,j,kc)) * rf
                wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
                wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
                wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
                rows(i,4,sb) = (wvisc(1)-qf(1))*dAj(1,i,j,kc) &
                             + (wvisc(2)-qf(2))*dAj(2,i,j,kc) &
                             + (wvisc(3)-qf(3))*dAj(3,i,j,kc)
            end do
            if (j == 2) then
                do i = 1, ni-1
                    wfac = 1.0e0 - wallj1(i,kc)
                    call wall_func_jface(r, dAj, vol, Omega_block, Omega_wallj1_nd(i,kc), &
                        mu, cons(:,:,:,1), Vx, Vr, Vt, i, 1, kc, 1, wf)
                    rows(i,1,sb) = wallj1(i,kc)*rows(i,1,sb) + wfac*wf(1)
                    rows(i,2,sb) = wallj1(i,kc)*rows(i,2,sb) + wfac*wf(2)
                    rows(i,3,sb) = wallj1(i,kc)*rows(i,3,sb) + wfac*wf(3)
                    rows(i,4,sb) = wallj1(i,kc)*rows(i,4,sb) + wfac*wf(4)
                end do
            end if
            if (j == nj-1) then
                do i = 1, ni-1
                    wfac = 1.0e0 - wallnj(i,kc)
                    call wall_func_jface(r, dAj, vol, Omega_block, Omega_wallnj_nd(i,kc), &
                        mu, cons(:,:,:,1), Vx, Vr, Vt, i, nj, kc, -1, wf)
                    rows(i,1,sb) = wallnj(i,kc)*rows(i,1,sb) + wfac*wf(1)
                    rows(i,2,sb) = wallnj(i,kc)*rows(i,2,sb) + wfac*wf(2)
                    rows(i,3,sb) = wallnj(i,kc)*rows(i,3,sb) + wfac*wf(3)
                    rows(i,4,sb) = wallnj(i,kc)*rows(i,4,sb) + wfac*wf(4)
                end do
            end if
            if (j > 1) then
                jc = j - 1
                row_interior = k_interior .and. (jc >= 2 .and. jc <= nj-2)
                do i = 1, ni
                    tauf(1) = (tau_cell(i, jc+1, kc+1, 1) + tau_cell(i+1, jc+1, kc+1, 1)) * 0.5e0
                    tauf(2) = (tau_cell(i, jc+1, kc+1, 2) + tau_cell(i+1, jc+1, kc+1, 2)) * 0.5e0
                    tauf(3) = (tau_cell(i, jc+1, kc+1, 3) + tau_cell(i+1, jc+1, kc+1, 3)) * 0.5e0
                    tauf(4) = (tau_cell(i, jc+1, kc+1, 4) + tau_cell(i+1, jc+1, kc+1, 4)) * 0.5e0
                    tauf(5) = (tau_cell(i, jc+1, kc+1, 5) + tau_cell(i+1, jc+1, kc+1, 5)) * 0.5e0
                    tauf(6) = (tau_cell(i, jc+1, kc+1, 6) + tau_cell(i+1, jc+1, kc+1, 6)) * 0.5e0
                    qf(1)   = (q_cell(i, jc+1, kc+1, 1) + q_cell(i+1, jc+1, kc+1, 1)) * 0.5e0
                    qf(2)   = (q_cell(i, jc+1, kc+1, 2) + q_cell(i+1, jc+1, kc+1, 2)) * 0.5e0
                    qf(3)   = (q_cell(i, jc+1, kc+1, 3) + q_cell(i+1, jc+1, kc+1, 3)) * 0.5e0
                    Vf(1) = (Vx(i,jc,kc) + Vx(i,jc+1,kc) + Vx(i,jc,kc+1) + Vx(i,jc+1,kc+1)) * 0.25e0
                    Vf(2) = (Vr(i,jc,kc) + Vr(i,jc+1,kc) + Vr(i,jc,kc+1) + Vr(i,jc+1,kc+1)) * 0.25e0
                    Vf(3) = (Vt(i,jc,kc) + Vt(i,jc+1,kc) + Vt(i,jc,kc+1) + Vt(i,jc+1,kc+1)) * 0.25e0
                    rf     = (r(i,jc,kc)  + r(i,jc+1,kc)  + r(i,jc,kc+1)  + r(i,jc+1,kc+1))  * 0.25e0
                    Vabs = Vf(3) + Omega_block * rf
                    rows(i,1,1) = tauf(1)*dAi(1,i,jc,kc) + tauf(4)*dAi(2,i,jc,kc) + tauf(5)*dAi(3,i,jc,kc)
                    rows(i,2,1) = tauf(4)*dAi(1,i,jc,kc) + tauf(2)*dAi(2,i,jc,kc) + tauf(6)*dAi(3,i,jc,kc)
                    rows(i,3,1) = (tauf(5)*dAi(1,i,jc,kc) + tauf(6)*dAi(2,i,jc,kc) + tauf(3)*dAi(3,i,jc,kc)) * rf
                    wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
                    wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
                    wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
                    rows(i,4,1) = (wvisc(1)-qf(1))*dAi(1,i,jc,kc) &
                                + (wvisc(2)-qf(2))*dAi(2,i,jc,kc) &
                                + (wvisc(3)-qf(3))*dAi(3,i,jc,kc)
                end do
                wfac = 1.0e0 - walli1(jc,kc)
                call wall_func_iface(r, dAi, vol, Omega_block, Omega_walli1_nd(jc,kc), &
                    mu, cons(:,:,:,1), Vx, Vr, Vt, 1, jc, kc, 1, wf)
                rows(2,1,1) = walli1(jc,kc)*rows(2,1,1) + wfac*wf(1)
                rows(2,2,1) = walli1(jc,kc)*rows(2,2,1) + wfac*wf(2)
                rows(2,3,1) = walli1(jc,kc)*rows(2,3,1) + wfac*wf(3)
                rows(2,4,1) = walli1(jc,kc)*rows(2,4,1) + wfac*wf(4)
                wfac = 1.0e0 - wallni(jc,kc)
                call wall_func_iface(r, dAi, vol, Omega_block, Omega_wallni_nd(jc,kc), &
                    mu, cons(:,:,:,1), Vx, Vr, Vt, ni, jc, kc, -1, wf)
                rows(ni-1,1,1) = wallni(jc,kc)*rows(ni-1,1,1) + wfac*wf(1)
                rows(ni-1,2,1) = wallni(jc,kc)*rows(ni-1,2,1) + wfac*wf(2)
                rows(ni-1,3,1) = wallni(jc,kc)*rows(ni-1,3,1) + wfac*wf(3)
                rows(ni-1,4,1) = wallni(jc,kc)*rows(ni-1,4,1) + wfac*wf(4)
                ! Production's association, not merely its order: its j and
                ! k accumulates are `fvisc = fvisc + hi - lo`, i.e. ((x + hi) - lo),
                ! NOT x + (hi - lo). Grouping the differences instead re-rounds
                ! and costs bitwise agreement (measured: ~1 ulp of the field
                ! scale). Left-to-right, exactly as written below.
                do i = 1, ni-1
                    fvisc(i,jc,kc,1) = (rows(i+1,1,1) - rows(i,1,1)) &
                                     + rows(i,1,sb) - rows(i,1,sa) &
                                     + planes(i,jc,1,pb) - planes(i,jc,1,pa)
                    fvisc(i,jc,kc,2) = (rows(i+1,2,1) - rows(i,2,1)) &
                                     + rows(i,2,sb) - rows(i,2,sa) &
                                     + planes(i,jc,2,pb) - planes(i,jc,2,pa)
                    fvisc(i,jc,kc,3) = (rows(i+1,3,1) - rows(i,3,1)) &
                                     + rows(i,3,sb) - rows(i,3,sa) &
                                     + planes(i,jc,3,pb) - planes(i,jc,3,pa)
                    fvisc(i,jc,kc,4) = (rows(i+1,4,1) - rows(i,4,1)) &
                                     + rows(i,4,sb) - rows(i,4,sa) &
                                     + planes(i,jc,4,pb) - planes(i,jc,4,pa)
                end do
                ! Wall mask and polar source, both finished here while the
                ! row is still in L1. For a row interior in j and k the ONLY
                ! mask its end cells carry is walli1/wallni -- no j- or k-mask
                ! applies -- so those two cells can be masked now, and the
                ! polar loop then covers the whole row unbroken and
                ! unit-stride. That is what removes the i=1/i=ni-1 sheet from
                ! the O(surface) pass, where fvisc could only ever be reached
                ! with stride ni-1 (opt-report: one such block gather-
                ! vectorized, the other not vectorized at all).
                ! Order matches production: i-mask, then polar. The cusp
                ! correction cannot interfere -- it touches only kc=1 and
                ! kc=nk-1, which are not interior rows.
                if (row_interior) then
                    fvisc(1,jc,kc,1) = fvisc(1,jc,kc,1) * walli1(jc,kc)
                    fvisc(1,jc,kc,2) = fvisc(1,jc,kc,2) * walli1(jc,kc)
                    fvisc(1,jc,kc,3) = fvisc(1,jc,kc,3) * walli1(jc,kc)
                    fvisc(1,jc,kc,4) = fvisc(1,jc,kc,4) * walli1(jc,kc)
                    fvisc(ni-1,jc,kc,1) = fvisc(ni-1,jc,kc,1) * wallni(jc,kc)
                    fvisc(ni-1,jc,kc,2) = fvisc(ni-1,jc,kc,2) * wallni(jc,kc)
                    fvisc(ni-1,jc,kc,3) = fvisc(ni-1,jc,kc,3) * wallni(jc,kc)
                    fvisc(ni-1,jc,kc,4) = fvisc(ni-1,jc,kc,4) * wallni(jc,kc)
                    do i = 1, ni-1
                        fvisc(i,jc,kc,2) = fvisc(i,jc,kc,2) &
                            + vol(i,jc,kc) * polar_src(cons_cell, P, r, P_offset, i, jc, kc)
                    end do
                end if
            end if
            stmp = sa
            sa = sb
            sb = stmp
        end do
    end if
    stmp = pa
    pa = pb
    pb = stmp
    end do

    ! ===== Cusp seam: replace each seam face flux with the two-sided average =====
    ! The seam is non-local in k, so under the sweep both seam cells have
    ! accumulated their raw one-sided flux; replacing it with
    ! avg = 0.5*(flow(k=1) + flow(k=nk)) is the same delta for both cells.
    !
    ! fcorr is a REPLACEMENT DELTA, not a face difference, so it carries the
    ! accumulate's sign convention -- fvisc = high-minus-low here (this kernel
    ! produces the residual's sign directly; there is no trailing negation pass
    ! as there was before commit 2381658745). Deriving it for the k=1 cell,
    ! which holds flow(2) - flow(1): swapping flow(1) for avg adds
    ! flow(1) - avg = 0.5*(flow(1) - flow(nk)). The k=nk-1 cell holds
    ! flow(nk) - flow(nk-1), and swapping flow(nk) for avg adds
    ! avg - flow(nk) = the same thing -- which is why one fcorr serves both.
    !
    ! NB the pre-2381658745 code accumulated low-minus-high and negated
    ! everything at the end, so it added the opposite, 0.5*(flow(nk) - flow(1)),
    ! and the negation flipped accumulate and correction together. When that
    ! commit flipped the accumulate it left this pass alone, on the reasoning
    ! that fcorr's own high-minus-low ordering already matched the new
    ! convention. It does not follow: a replacement delta must flip with the
    ! quantity it corrects regardless of how it is spelled internally, and the
    ! seam cells took +2*fcorr of spurious, anti-diffusive viscous force every
    ! step as a result. See tests/test_viscous_cusp_seam.py.
    !
    ! The two raw seam-face fluxes are recomputed via kface_flow: tau_cell/
    ! q_cell are unchanged since the entry halo scaling and neither seam plane
    ! takes a wall-function injection, so the recompute matches the sweep's
    ! values. (nk=2, where the two seam cells coincide, is not supported here.)
    if (i_cusp_start > 0 .and. nk > 2) then
        do j = 1, nj-1
        do i = i_cusp_start, i_cusp_end-1
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, 1, flow1)
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, nk, flownk)
            fcorr(1) = 0.5e0 * (flow1(1) - flownk(1))
            fcorr(2) = 0.5e0 * (flow1(2) - flownk(2))
            fcorr(3) = 0.5e0 * (flow1(3) - flownk(3))
            fcorr(4) = 0.5e0 * (flow1(4) - flownk(4))
            fvisc(i,j,1,1) = fvisc(i,j,1,1) + fcorr(1)
            fvisc(i,j,1,2) = fvisc(i,j,1,2) + fcorr(2)
            fvisc(i,j,1,3) = fvisc(i,j,1,3) + fcorr(3)
            fvisc(i,j,1,4) = fvisc(i,j,1,4) + fcorr(4)
            fvisc(i,j,nk-1,1) = fvisc(i,j,nk-1,1) + fcorr(1)
            fvisc(i,j,nk-1,2) = fvisc(i,j,nk-1,2) + fcorr(2)
            fvisc(i,j,nk-1,3) = fvisc(i,j,nk-1,3) + fcorr(3)
            fvisc(i,j,nk-1,4) = fvisc(i,j,nk-1,4) + fcorr(4)
        end do
        end do
    end if

    call zero_wall_fvisc_border(fvisc, walli1, wallj1, wallk1, wallni, wallnj, wallnk, ni, nj, nk)

    ! ===== Polar source on the boundary shell, AFTER the wall zeroing =====
    ! Interior cells took their polar source inside the fused store above; the
    ! shell could not. Production adds the polar source after the zeroing pass
    ! because it is a geometric source, not viscous content, so the wall mask
    ! must not eat it -- and the fused store runs before that pass.
    !
    ! The four blocks below partition the shell so every cell in it is visited
    ! EXACTLY once. This is stricter than the zeroing loops need to be: those
    ! may overlap at edges and corners because a repeated multiply by the same
    ! mask is harmless, but a repeated ADD is not. Each high-face block is also
    ! guarded, so a degenerate dimension (one cell plane, where the low and
    ! high faces are the same cells) does not double-add either.
    do j = 1, nj-1
    do i = 1, ni-1
        fvisc(i,j,1,2) = fvisc(i,j,1,2) + vol(i,j,1) * polar_src(cons_cell, P, r, P_offset, i, j, 1)
    end do
    end do
    if (nk-1 > 1) then
        do j = 1, nj-1
        do i = 1, ni-1
            fvisc(i,j,nk-1,2) = fvisc(i,j,nk-1,2) &
                + vol(i,j,nk-1) * polar_src(cons_cell, P, r, P_offset, i, j, nk-1)
        end do
        end do
    end if
    do k = 2, nk-2
    do i = 1, ni-1
        fvisc(i,1,k,2) = fvisc(i,1,k,2) + vol(i,1,k) * polar_src(cons_cell, P, r, P_offset, i, 1, k)
    end do
    end do
    if (nj-1 > 1) then
        do k = 2, nk-2
        do i = 1, ni-1
            fvisc(i,nj-1,k,2) = fvisc(i,nj-1,k,2) &
                + vol(i,nj-1,k) * polar_src(cons_cell, P, r, P_offset, i, nj-1, k)
        end do
        end do
    end if

end subroutine set_visc_force


! =====================================================================
! y+ on all six wall-adjacent boundary faces of a block -- POST-PROCESSING
! ONLY, never called from set_visc_force's per-step hot path. Reuses
! wall_yplus_iface/jface/kface (viscous_helpers), which share their Re/cf/d
! definition with the production wall function via wall_core, so this
! cannot silently drift from what the solver actually modelled at a face.
!
! Costs O(surface) per call and carries none of set_visc_force's k-slab or
! rolling-buffer machinery -- there is no per-step budget to protect here,
! so a plain double-nested loop per face is the right shape. The six output
! arrays are zeroed on entry and only written on wall cells (walli1 etc. are
! 0.0=wall, 1.0=free, per Block.ijk_wall_visc's convention); non-wall and
! non-computed cells stay at that zero fill.
!
! The (i, j, k, di/dj/dk) argument patterns at each of the six call sites
! below are copied from set_visc_force's own six wall_func_*face call
! sites -- keep them in lockstep if that kernel's wall-face indexing ever
! changes.
! =====================================================================
subroutine wall_yplus_field( &
    cons, vol, dAi, dAj, dAk, &
    Omega_block, r, mu, &
    Vx, Vr, Vt, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    Omega_walli1_nd, Omega_wallj1_nd, Omega_wallk1_nd, &
    Omega_wallni_nd, Omega_wallnj_nd, Omega_wallnk_nd, &
    yplus_i1, yplus_j1, yplus_k1, &
    yplus_ni, yplus_nj, yplus_nk, &
    ni, nj, nk)

    use viscous_helpers
    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega_block, mu
    real, intent(in) :: Vx(ni, nj, nk), Vr(ni, nj, nk), Vt(ni, nj, nk)
    real, intent(in) :: walli1(nj-1, nk-1), wallni(nj-1, nk-1)
    real, intent(in) :: wallj1(ni-1, nk-1), wallnj(ni-1, nk-1)
    real, intent(in) :: wallk1(ni-1, nj-1), wallnk(ni-1, nj-1)
    real, intent(in) :: Omega_walli1_nd(nj-1, nk-1), Omega_wallni_nd(nj-1, nk-1)
    real, intent(in) :: Omega_wallj1_nd(ni-1, nk-1), Omega_wallnj_nd(ni-1, nk-1)
    real, intent(in) :: Omega_wallk1_nd(ni-1, nj-1), Omega_wallnk_nd(ni-1, nj-1)
    real, intent(out) :: yplus_i1(nj-1, nk-1), yplus_ni(nj-1, nk-1)
    real, intent(out) :: yplus_j1(ni-1, nk-1), yplus_nj(ni-1, nk-1)
    real, intent(out) :: yplus_k1(ni-1, nj-1), yplus_nk(ni-1, nj-1)

    integer :: i, j, k
    real :: yp

    yplus_i1 = 0.0e0; yplus_ni = 0.0e0
    yplus_j1 = 0.0e0; yplus_nj = 0.0e0
    yplus_k1 = 0.0e0; yplus_nk = 0.0e0

    do j = 1, nj-1
    do i = 1, ni-1
        if (wallk1(i,j) == 0.0e0) then
            call wall_yplus_kface(r, dAk, vol, Omega_block, Omega_wallk1_nd(i,j), mu, &
                                   cons(:,:,:,1), Vx, Vr, Vt, i, j, 1, 1, yp)
            yplus_k1(i,j) = yp
        end if
        if (wallnk(i,j) == 0.0e0) then
            call wall_yplus_kface(r, dAk, vol, Omega_block, Omega_wallnk_nd(i,j), mu, &
                                   cons(:,:,:,1), Vx, Vr, Vt, i, j, nk, -1, yp)
            yplus_nk(i,j) = yp
        end if
    end do
    end do

    do k = 1, nk-1
    do i = 1, ni-1
        if (wallj1(i,k) == 0.0e0) then
            call wall_yplus_jface(r, dAj, vol, Omega_block, Omega_wallj1_nd(i,k), mu, &
                                   cons(:,:,:,1), Vx, Vr, Vt, i, 1, k, 1, yp)
            yplus_j1(i,k) = yp
        end if
        if (wallnj(i,k) == 0.0e0) then
            call wall_yplus_jface(r, dAj, vol, Omega_block, Omega_wallnj_nd(i,k), mu, &
                                   cons(:,:,:,1), Vx, Vr, Vt, i, nj, k, -1, yp)
            yplus_nj(i,k) = yp
        end if
    end do
    end do

    do k = 1, nk-1
    do j = 1, nj-1
        if (walli1(j,k) == 0.0e0) then
            call wall_yplus_iface(r, dAi, vol, Omega_block, Omega_walli1_nd(j,k), mu, &
                                   cons(:,:,:,1), Vx, Vr, Vt, 1, j, k, 1, yp)
            yplus_i1(j,k) = yp
        end if
        if (wallni(j,k) == 0.0e0) then
            call wall_yplus_iface(r, dAi, vol, Omega_block, Omega_wallni_nd(j,k), mu, &
                                   cons(:,:,:,1), Vx, Vr, Vt, ni, j, k, -1, yp)
            yplus_ni(j,k) = yp
        end if
    end do
    end do

end subroutine wall_yplus_field
