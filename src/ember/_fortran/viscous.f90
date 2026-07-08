! Routines for adding viscous effects




! Interpolate from lookup table using log-log interpolation


! ============================================================
! Module containing helper functions used by shear_stress_v3.
! All arrays are assumed-shape (contiguous) so ni/nj/nk are
! not needed in any signature.
! ============================================================
module viscous_helpers
    implicit none
    private
    public :: avg_cell, iface, jface, kface
    public :: wall_func
    public :: wall_func_iface, wall_func_jface, wall_func_kface
    public :: compute_tau, grad_cell_cached, compute_q

contains

    pure function avg_cell(x, i, j, k) result(avg)
        implicit none
        real, intent(in), contiguous :: x(:,:,:)
        integer, intent(in) :: i, j, k
        real :: avg
        avg = 0.125e0 * ( &
            x(i,j,k) + x(i+1,j,k) + x(i,j+1,k) + x(i+1,j+1,k) + &
            x(i,j,k+1) + x(i+1,j,k+1) + x(i,j+1,k+1) + x(i+1,j+1,k+1))
    end function avg_cell

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

    pure subroutine wall_func(r, dA, vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt, flow)
        implicit none
        real, intent(in) :: r, dA(3), vol, Omega_block, Omega_wall, mu, rho, Vx, Vr, Vt
        real, intent(out) :: flow(4)
        real :: V, Re, d, cf, tau, vec(3), dA_mag
        real :: lnRew, Vt_slip
        real, parameter :: a1 = -1.767e-3
        real, parameter :: a2 = 3.177e-2
        real, parameter :: a3 = 2.5614e-1
        ! Vt is relative to block frame; subtract wall velocity in block frame
        Vt_slip = Vt - (Omega_wall - Omega_block) * r
        V = sqrt(Vx**2 + Vr**2 + Vt_slip**2 + 1e-9)
        dA_mag = sqrt(dA(1)**2 + dA(2)**2 + dA(3)**2)
        d = vol / dA_mag
        Re = rho * V * d / mu
        lnRew = log(Re)
        if (Re .lt. 127.53373025e0) then
            cf = 2e0/Re
        else
            cf = (a1 + a2/lnRew + a3/lnRew/lnRew)
        end if
        tau = cf * 0.5e0 * rho * V * V
        vec(1) = Vx     / V * dA_mag
        vec(2) = Vr     / V * dA_mag
        vec(3) = Vt_slip / V * dA_mag
        flow(1) = vec(1) * tau
        flow(2) = vec(2) * tau
        flow(3) = r * vec(3) * tau
        flow(4) = Omega_wall * r * vec(3) * tau
    end subroutine wall_func

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

    ! Compute strain rate tensor tau, mixing-length turbulent viscosity mu_turb,
    ! and cache face-averaged velocities in fv for reuse by the caller.
    !
    ! Vorticity frame for mixing-length turbulence
    ! --------------------------------------------
    ! Vt is stored as the relative tangential velocity (block frame); the strain
    ! rate built from gradVx/gradVr/gradVt is frame-invariant because the rigid
    ! rotation Omega*r contributes only an antisymmetric part. The vorticity
    ! magnitude that drives mu_turb = rho * l^2 * |omega| is taken directly in
    ! this relative frame (no absolute-frame +2*Omega correction).
    pure subroutine compute_tau(Vx, Vr, Vt, cons, i, j, k, inv_vol_4, rc, &
                                xlength_cell, visc_lim, mu, &
                                dAi, dAj, dAk, &
                                tau, mu_turb)
        implicit none
        real, intent(in), contiguous :: Vx(:,:,:), Vr(:,:,:), Vt(:,:,:)
        real, intent(in), contiguous :: cons(:,:,:,:)
        integer, intent(in) :: i, j, k
        real, intent(in) :: inv_vol_4, rc, xlength_cell, visc_lim, mu
        real, intent(in), contiguous :: dAi(:,:,:,:), dAj(:,:,:,:), dAk(:,:,:,:)
        real, intent(out) :: tau(6), mu_turb
        real :: gradVx(3), gradVr(3), gradVt(3), Vc(3), rhoc, vort_mag
        ! face_sums slots from grad_cell_cached; only the two i-faces (1,2) are
        ! read, to recover the cell-centre velocity Vc for the vorticity model.
        real :: fsx(6), fsr(6), fst(6)
        call grad_cell_cached(Vx, rc, inv_vol_4, i, j, k, dAi, dAj, dAk, gradVx, fsx)
        call grad_cell_cached(Vr, rc, inv_vol_4, i, j, k, dAi, dAj, dAk, gradVr, fsr)
        call grad_cell_cached(Vt, rc, inv_vol_4, i, j, k, dAi, dAj, dAk, gradVt, fst)
        Vc(1) = (fsx(1) + fsx(2)) * 0.125e0
        Vc(2) = (fsr(1) + fsr(2)) * 0.125e0
        Vc(3) = (fst(1) + fst(2)) * 0.125e0
        rhoc = avg_cell(cons(:,:,:,1), i, j, k)
        ! Normal stresses are 2*mu*strain (no -2/3 div(V) deviatoric term),
        ! matching Multall NEW_LOSS (TXX = 2*VISTOT*DVXDX).
        tau(1) = gradVx(1)
        tau(2) = gradVr(2)
        tau(3) = gradVt(3)
        tau(4) = gradVx(2) + gradVr(1)
        tau(5) = gradVx(3) + gradVt(1)
        tau(6) = gradVr(3) + gradVt(2) - Vc(3)/rc
        ! Relative-frame vorticity drives the mixing length (no absolute-frame
        ! +2*Omega correction): the block-frame value as differenced from Vt_rel.
        Vc(1) = gradVt(2) - gradVr(3) + Vc(3)/rc
        Vc(2) = gradVx(3) - gradVt(1)
        Vc(3) = gradVr(1) - gradVx(2)
        vort_mag = sqrt(Vc(1)**2 + Vc(2)**2 + Vc(3)**2)
        mu_turb = rhoc * xlength_cell * vort_mag
        mu_turb = min(mu_turb, visc_lim)
        tau = tau * (mu + mu_turb) * 0.5e0
    end subroutine compute_tau

    ! Cell gradient of x by Green-Gauss over the six faces. The radial
    ! derivative uses the Multall GRADVEL form: the same plain face sums as the
    ! axial/tangential derivatives, dotted with the radial area, plus a single
    ! cell-level metric correction -x_cell/rc. This avoids a second set of six
    ! 1/r-weighted face sums (the old per-face metric treatment).
    pure subroutine grad_cell_cached(x, rc, inv_vol, i, j, k, dAi, dAj, dAk, grad, face_sums)
        implicit none
        real, intent(in), contiguous :: x(:,:,:)
        real, intent(in) :: rc, inv_vol
        integer, intent(in) :: i, j, k
        real, intent(in), contiguous :: dAi(:,:,:,:), dAj(:,:,:,:), dAk(:,:,:,:)
        real, intent(out) :: grad(3), face_sums(6)
        face_sums(1) = iface(x,i,j,k)
        face_sums(2) = iface(x,i+1,j,k)
        face_sums(3) = jface(x,i,j,k)
        face_sums(4) = jface(x,i,j+1,k)
        face_sums(5) = kface(x,i,j,k)
        face_sums(6) = kface(x,i,j,k+1)
        grad(1) = -(face_sums(1)*dAi(1,i,j,k) - face_sums(2)*dAi(1,i+1,j,k) &
                  + face_sums(3)*dAj(1,i,j,k) - face_sums(4)*dAj(1,i,j+1,k) &
                  + face_sums(5)*dAk(1,i,j,k) - face_sums(6)*dAk(1,i,j,k+1))
        grad(3) = -(face_sums(1)*dAi(3,i,j,k) - face_sums(2)*dAi(3,i+1,j,k) &
                  + face_sums(3)*dAj(3,i,j,k) - face_sums(4)*dAj(3,i,j+1,k) &
                  + face_sums(5)*dAk(3,i,j,k) - face_sums(6)*dAk(3,i,j,k+1))
        grad(2) = -(face_sums(1)*dAi(2,i,j,k) - face_sums(2)*dAi(2,i+1,j,k) &
                  + face_sums(3)*dAj(2,i,j,k) - face_sums(4)*dAj(2,i,j+1,k) &
                  + face_sums(5)*dAk(2,i,j,k) - face_sums(6)*dAk(2,i,j,k+1))
        grad = grad * inv_vol
        ! cell-level metric correction -x_cell/rc; 0.125*(i-faces) = x_cell.
        grad(2) = grad(2) - 0.125e0 * (face_sums(1) + face_sums(2)) / rc
    end subroutine grad_cell_cached

    pure subroutine compute_q(T, rc, inv_vol_4, mu_turb, i, j, k, &
                              mu, cp, Pr_lam, Pr_turb, dAi, dAj, dAk, q)
        implicit none
        real, intent(in), contiguous :: T(:,:,:)
        real, intent(in) :: rc, inv_vol_4, mu_turb
        integer, intent(in) :: i, j, k
        real, intent(in) :: mu, cp, Pr_lam, Pr_turb
        real, intent(in), contiguous :: dAi(:,:,:,:), dAj(:,:,:,:), dAk(:,:,:,:)
        real, intent(out) :: q(3)
        real :: lambda
        real :: s1, s2, s3, s4, s5, s6
        lambda = mu * cp / Pr_lam + mu_turb * cp / Pr_turb
        s1 = iface(T,i,j,k)
        s2 = iface(T,i+1,j,k)
        s3 = jface(T,i,j,k)
        s4 = jface(T,i,j+1,k)
        s5 = kface(T,i,j,k)
        s6 = kface(T,i,j,k+1)
        q(1) = (s1*dAi(1,i,j,k) - s2*dAi(1,i+1,j,k) &
              + s3*dAj(1,i,j,k) - s4*dAj(1,i,j+1,k) &
              + s5*dAk(1,i,j,k) - s6*dAk(1,i,j,k+1)) * (inv_vol_4 * lambda * 0.5e0)
        q(3) = (s1*dAi(3,i,j,k) - s2*dAi(3,i+1,j,k) &
              + s3*dAj(3,i,j,k) - s4*dAj(3,i,j+1,k) &
              + s5*dAk(3,i,j,k) - s6*dAk(3,i,j,k+1)) * (inv_vol_4 * lambda * 0.5e0)
        ! Multall-style radial term: plain face sums + cell-level metric
        ! correction. q's flux sum carries the opposite sign convention to
        ! grad_cell_cached (no leading minus), so the correction is +T_cell/rc
        ! here (vs -x_cell/rc in grad_cell_cached) to vanish for uniform T.
        q(2) = ((s1*dAi(2,i,j,k) - s2*dAi(2,i+1,j,k) &
               + s3*dAj(2,i,j,k) - s4*dAj(2,i,j+1,k) &
               + s5*dAk(2,i,j,k) - s6*dAk(2,i,j,k+1)) * inv_vol_4 &
               + 0.125e0 * (s1 + s2) / rc) * (lambda * 0.5e0)
    end subroutine compute_q

end module viscous_helpers

! Compute viscous fluxes fvisc for all cells using a three-pass algorithm.
!
! The grid has (ni x nj x nk) nodes defining (ni-1 x nj-1 x nk-1) cells.
! Face area vectors dAi/dAj/dAk point inward (low-index direction positive).
! The wall arrays (walli1, wallni, etc.) are blending weights: 1 = free-stream
! viscous stress, 0 = wall function. Values in between blend both.
!
! Algorithm overview
! ------------------
! Pass 0  Extract primitive velocities Vx, Vr, Vt at all nodes from conserved
!         variables.  Vt is relative (frame-subtracted); Omega*r is added back
!         where absolute velocity is needed (e.g. shear-work terms).
!
! Pass 1  For every cell (i,j,k) compute the stress tensor tau(6) and heat-flux
!         vector q(3) using a Green-Gauss gradient over the six cell faces.
!         Each face average is the mean of its four corner nodes.  tau and q are
!         stored multiplied by 2 so that averaging two adjacent cells recovers
!         the correct face value without an extra factor.
!
! Pass 2  Accumulate viscous face flows into fvisc one coordinate direction at a
!         time, reusing flow_scratch between directions.  For each direction:
!
!         Boundary faces (i=1, i=ni, j=1, j=nj, k=1, k=nk):
!           Face stress is taken from the single adjacent interior cell (already
!           doubled in Pass 1, so no extra factor needed).  The face velocity is
!           the node-average over the four face corners.  Boundary faces blend
!           free-stream viscous stress with a wall-function force according to
!           the wall blending weight.  The outward-pointing HIGH boundary stores
!           the negated flow so the subsequent subtraction cancels correctly.
!
!         Interior faces (e.g. i=2..ni-1):
!           Face stress tauf is the average of tau_cell from the two adjacent
!           cells.  Face velocity is inlined as the average of four nodes at the
!           face (avoiding a function-call overhead in the hot loop).
!
!         After all face flows for a direction are in flow_scratch, fvisc is
!         updated as:  fvisc(i,j,k) += flow_scratch(lo_face) - flow_scratch(hi_face)
!
! Scratch layout (16 nodal slots total):
!   slots 0-5:   tau_cell  (6,ni,nj,nk) -- written pass 1, read pass 2
!   slots 6-8:   q_cell    (3,ni,nj,nk) -- written pass 1, read pass 2
!   slots 9-12:  flow      (ni,nj,nk,4) -- reused per direction in pass 2
!   slots 13-15: Vx,Vr,Vt  (ni,nj,nk)  -- written pass 0, read passes 1+2

! Strip-mined / SoA variant of set_tau_q for vectorization experiments.
!
! Same inputs/outputs as set_tau_q, but the per-cell work is split into two
! flat do-i loops over each (j,k) row, with the per-cell intermediates held in
! row temps dimensioned with i as the contiguous axis. This gives the auto-
! vectorizer simple, call-light, unit-stride loops to vectorize over i instead
! of one deep nest with a heavy inlined body.
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
    real, intent(in) :: mu, cp, Pr_lam, Pr_turb
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
    real :: vct(ni-1), rcr(ni-1), ivr(ni-1), rhoc(ni-1)
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
            mut = min(rhoc(i) * xlength(i,j,k) * vm, visc_lim)
            mu_turb(i,j,k) = mut
            fac = (mu + mut) * 0.5e0
            tau_cell(i+1,j+1,k+1,1) = t1*fac
            tau_cell(i+1,j+1,k+1,2) = t2*fac
            tau_cell(i+1,j+1,k+1,3) = t3*fac
            tau_cell(i+1,j+1,k+1,4) = t4*fac
            tau_cell(i+1,j+1,k+1,5) = t5*fac
            tau_cell(i+1,j+1,k+1,6) = t6*fac
            lambda = mu*cp/Pr_lam + mut*cp/Pr_turb
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
! tau_cell and q_cell are halo-dimensioned (6/3, ni+1, nj+1, nk+1): owned
! cells sit at indices 2..ni (2..nj, 2..nk).  Halo slots 1 and ni+1 (etc.)
! will carry neighbour data after exchange; until then the boundary loops
! below are single-sided and use only the nearest owned cell, as before.
subroutine set_visc_force( &
    cons, vol, dAi, dAj, dAk, &
    Omega_block, r, mu, &
    fvisc, &
    Vx, Vr, Vt, &
    tau_cell, &
    q_cell, &
    flow_scratch, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    Omega_walli1_nd, Omega_wallj1_nd, Omega_wallk1_nd, &
    Omega_wallni_nd, Omega_wallnj_nd, Omega_wallnk_nd, &
    i_cusp_start, i_cusp_end, &
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
    real, intent(inout) :: fvisc(ni-1, nj-1, nk-1, 4)
    real, intent(in) :: Vx(ni, nj, nk)
    real, intent(in) :: Vr(ni, nj, nk)
    real, intent(in) :: Vt(ni, nj, nk)
    real, intent(inout) :: tau_cell(ni+1, nj+1, nk+1, 6)
    real, intent(inout) :: q_cell(ni+1, nj+1, nk+1, 3)
    real, intent(inout) :: flow_scratch(ni, nj, nk, 4)
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

    integer :: i, j, k
    real :: tauf(6), qf(3), Vf(3), rf
    real :: wvisc(3), Vabs, wf(4), wfac

    ! ===== Scale boundary halos by (2*wall-1) =====
    ! wall=0 (wall face): factor=-1, giving -edge so face average is zero.
    ! wall=1 (inlet/outlet/periodic): factor=+1, keeping +edge for single-sided stress.
    ! Periodic halos were already overwritten by exchange_halos() so scaling is harmless.
    ! low-i
    do k = 1, nk-1
    do j = 1, nj-1
        tau_cell(1, j+1, k+1, :) = tau_cell(1, j+1, k+1, :) * (2.0e0*walli1(j,k) - 1.0e0)
        q_cell(1, j+1, k+1, :) = q_cell(1, j+1, k+1, :) * (2.0e0*walli1(j,k) - 1.0e0)
    end do
    end do
    ! high-i
    do k = 1, nk-1
    do j = 1, nj-1
        tau_cell(ni+1, j+1, k+1, :) = tau_cell(ni+1, j+1, k+1, :) * (2.0e0*wallni(j,k) - 1.0e0)
        q_cell(ni+1, j+1, k+1, :) = q_cell(ni+1, j+1, k+1, :) * (2.0e0*wallni(j,k) - 1.0e0)
    end do
    end do
    ! low-j
    do k = 1, nk-1
    do i = 1, ni-1
        tau_cell(i+1, 1, k+1, :) = tau_cell(i+1, 1, k+1, :) * (2.0e0*wallj1(i,k) - 1.0e0)
        q_cell(i+1, 1, k+1, :) = q_cell(i+1, 1, k+1, :) * (2.0e0*wallj1(i,k) - 1.0e0)
    end do
    end do
    ! high-j
    do k = 1, nk-1
    do i = 1, ni-1
        tau_cell(i+1, nj+1, k+1, :) = tau_cell(i+1, nj+1, k+1, :) * (2.0e0*wallnj(i,k) - 1.0e0)
        q_cell(i+1, nj+1, k+1, :) = q_cell(i+1, nj+1, k+1, :) * (2.0e0*wallnj(i,k) - 1.0e0)
    end do
    end do
    ! low-k
    do j = 1, nj-1
    do i = 1, ni-1
        tau_cell(i+1, j+1, 1, :) = tau_cell(i+1, j+1, 1, :) * (2.0e0*wallk1(i,j) - 1.0e0)
        q_cell(i+1, j+1, 1, :) = q_cell(i+1, j+1, 1, :) * (2.0e0*wallk1(i,j) - 1.0e0)
    end do
    end do
    ! high-k
    do j = 1, nj-1
    do i = 1, ni-1
        tau_cell(i+1, j+1, nk+1, :) = tau_cell(i+1, j+1, nk+1, :) * (2.0e0*wallnk(i,j) - 1.0e0)
        q_cell(i+1, j+1, nk+1, :) = q_cell(i+1, j+1, nk+1, :) * (2.0e0*wallnk(i,j) - 1.0e0)
    end do
    end do

    ! ===== Uniform face loops =====
    ! Every face averages the two adjacent halo-indexed cells.
    ! Boundary faces use the ghost values written by eval_tau_q and exchange_halos.

    ! --- i-direction: faces i=1..ni ---
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni
        tauf(1) = (tau_cell(i, j+1, k+1, 1) + tau_cell(i+1, j+1, k+1, 1)) * 0.5e0
        tauf(2) = (tau_cell(i, j+1, k+1, 2) + tau_cell(i+1, j+1, k+1, 2)) * 0.5e0
        tauf(3) = (tau_cell(i, j+1, k+1, 3) + tau_cell(i+1, j+1, k+1, 3)) * 0.5e0
        tauf(4) = (tau_cell(i, j+1, k+1, 4) + tau_cell(i+1, j+1, k+1, 4)) * 0.5e0
        tauf(5) = (tau_cell(i, j+1, k+1, 5) + tau_cell(i+1, j+1, k+1, 5)) * 0.5e0
        tauf(6) = (tau_cell(i, j+1, k+1, 6) + tau_cell(i+1, j+1, k+1, 6)) * 0.5e0
        qf(1)   = (q_cell(i, j+1, k+1, 1) + q_cell(i+1, j+1, k+1, 1)) * 0.5e0
        qf(2)   = (q_cell(i, j+1, k+1, 2) + q_cell(i+1, j+1, k+1, 2)) * 0.5e0
        qf(3)   = (q_cell(i, j+1, k+1, 3) + q_cell(i+1, j+1, k+1, 3)) * 0.5e0
        Vf(1) = (Vx(i,j,k) + Vx(i,j+1,k) + Vx(i,j,k+1) + Vx(i,j+1,k+1)) * 0.25e0
        Vf(2) = (Vr(i,j,k) + Vr(i,j+1,k) + Vr(i,j,k+1) + Vr(i,j+1,k+1)) * 0.25e0
        Vf(3) = (Vt(i,j,k) + Vt(i,j+1,k) + Vt(i,j,k+1) + Vt(i,j+1,k+1)) * 0.25e0
        rf     = (r(i,j,k)  + r(i,j+1,k)  + r(i,j,k+1)  + r(i,j+1,k+1))  * 0.25e0
        Vabs = Vf(3) + Omega_block * rf
        flow_scratch(i,j,k,1) = tauf(1)*dAi(1,i,j,k) + tauf(4)*dAi(2,i,j,k) + tauf(5)*dAi(3,i,j,k)
        flow_scratch(i,j,k,2) = tauf(4)*dAi(1,i,j,k) + tauf(2)*dAi(2,i,j,k) + tauf(6)*dAi(3,i,j,k)
        flow_scratch(i,j,k,3) = (tauf(5)*dAi(1,i,j,k) + tauf(6)*dAi(2,i,j,k) + tauf(3)*dAi(3,i,j,k)) * rf
        wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
        wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
        wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
        flow_scratch(i,j,k,4) = (wvisc(1)-qf(1))*dAi(1,i,j,k) &
                               + (wvisc(2)-qf(2))*dAi(2,i,j,k) &
                               + (wvisc(3)-qf(3))*dAi(3,i,j,k)
    end do
    end do
    end do
    ! ===== Wall function injected as i=2 / i=ni-1 face flow =====
    do k = 1, nk-1
    do j = 1, nj-1
        wfac = 1.0e0 - walli1(j,k)
        call wall_func_iface(r, dAi, vol, Omega_block, Omega_walli1_nd(j,k), mu, cons(:,:,:,1), Vx, Vr, Vt, 1, j, k, 1, wf)
        flow_scratch(2,j,k,1) = walli1(j,k)*flow_scratch(2,j,k,1) + wfac*wf(1)
        flow_scratch(2,j,k,2) = walli1(j,k)*flow_scratch(2,j,k,2) + wfac*wf(2)
        flow_scratch(2,j,k,3) = walli1(j,k)*flow_scratch(2,j,k,3) + wfac*wf(3)
        flow_scratch(2,j,k,4) = walli1(j,k)*flow_scratch(2,j,k,4) + wfac*wf(4)
        wfac = 1.0e0 - wallni(j,k)
        call wall_func_iface(r, dAi, vol, Omega_block, Omega_wallni_nd(j,k), mu, cons(:,:,:,1), Vx, Vr, Vt, ni, j, k, -1, wf)
        flow_scratch(ni-1,j,k,1) = wallni(j,k)*flow_scratch(ni-1,j,k,1) + wfac*wf(1)
        flow_scratch(ni-1,j,k,2) = wallni(j,k)*flow_scratch(ni-1,j,k,2) + wfac*wf(2)
        flow_scratch(ni-1,j,k,3) = wallni(j,k)*flow_scratch(ni-1,j,k,3) + wfac*wf(3)
        flow_scratch(ni-1,j,k,4) = wallni(j,k)*flow_scratch(ni-1,j,k,4) + wfac*wf(4)
    end do
    end do
    ! --- accumulate i-direction ---
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        fvisc(i,j,k,1) = flow_scratch(i,j,k,1) - flow_scratch(i+1,j,k,1)
        fvisc(i,j,k,2) = flow_scratch(i,j,k,2) - flow_scratch(i+1,j,k,2)
        fvisc(i,j,k,3) = flow_scratch(i,j,k,3) - flow_scratch(i+1,j,k,3)
        fvisc(i,j,k,4) = flow_scratch(i,j,k,4) - flow_scratch(i+1,j,k,4)
    end do
    end do
    end do

    ! --- j-direction: faces j=1..nj ---
    do k = 1, nk-1
    do j = 1, nj
    do i = 1, ni-1
        tauf(1) = (tau_cell(i+1, j, k+1, 1) + tau_cell(i+1, j+1, k+1, 1)) * 0.5e0
        tauf(2) = (tau_cell(i+1, j, k+1, 2) + tau_cell(i+1, j+1, k+1, 2)) * 0.5e0
        tauf(3) = (tau_cell(i+1, j, k+1, 3) + tau_cell(i+1, j+1, k+1, 3)) * 0.5e0
        tauf(4) = (tau_cell(i+1, j, k+1, 4) + tau_cell(i+1, j+1, k+1, 4)) * 0.5e0
        tauf(5) = (tau_cell(i+1, j, k+1, 5) + tau_cell(i+1, j+1, k+1, 5)) * 0.5e0
        tauf(6) = (tau_cell(i+1, j, k+1, 6) + tau_cell(i+1, j+1, k+1, 6)) * 0.5e0
        qf(1)   = (q_cell(i+1, j, k+1, 1) + q_cell(i+1, j+1, k+1, 1)) * 0.5e0
        qf(2)   = (q_cell(i+1, j, k+1, 2) + q_cell(i+1, j+1, k+1, 2)) * 0.5e0
        qf(3)   = (q_cell(i+1, j, k+1, 3) + q_cell(i+1, j+1, k+1, 3)) * 0.5e0
        Vf(1) = (Vx(i,j,k) + Vx(i+1,j,k) + Vx(i,j,k+1) + Vx(i+1,j,k+1)) * 0.25e0
        Vf(2) = (Vr(i,j,k) + Vr(i+1,j,k) + Vr(i,j,k+1) + Vr(i+1,j,k+1)) * 0.25e0
        Vf(3) = (Vt(i,j,k) + Vt(i+1,j,k) + Vt(i,j,k+1) + Vt(i+1,j,k+1)) * 0.25e0
        rf     = (r(i,j,k)  + r(i+1,j,k)  + r(i,j,k+1)  + r(i+1,j,k+1))  * 0.25e0
        Vabs = Vf(3) + Omega_block * rf
        flow_scratch(i,j,k,1) = tauf(1)*dAj(1,i,j,k) + tauf(4)*dAj(2,i,j,k) + tauf(5)*dAj(3,i,j,k)
        flow_scratch(i,j,k,2) = tauf(4)*dAj(1,i,j,k) + tauf(2)*dAj(2,i,j,k) + tauf(6)*dAj(3,i,j,k)
        flow_scratch(i,j,k,3) = (tauf(5)*dAj(1,i,j,k) + tauf(6)*dAj(2,i,j,k) + tauf(3)*dAj(3,i,j,k)) * rf
        wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
        wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
        wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
        flow_scratch(i,j,k,4) = (wvisc(1)-qf(1))*dAj(1,i,j,k) &
                               + (wvisc(2)-qf(2))*dAj(2,i,j,k) &
                               + (wvisc(3)-qf(3))*dAj(3,i,j,k)
    end do
    end do
    end do
    ! ===== Wall function injected as j=2 / j=nj-1 face flow =====
    do k = 1, nk-1
    do i = 1, ni-1
        wfac = 1.0e0 - wallj1(i,k)
        call wall_func_jface(r, dAj, vol, Omega_block, Omega_wallj1_nd(i,k), mu, cons(:,:,:,1), Vx, Vr, Vt, i, 1, k, 1, wf)
        flow_scratch(i,2,k,1) = wallj1(i,k)*flow_scratch(i,2,k,1) + wfac*wf(1)
        flow_scratch(i,2,k,2) = wallj1(i,k)*flow_scratch(i,2,k,2) + wfac*wf(2)
        flow_scratch(i,2,k,3) = wallj1(i,k)*flow_scratch(i,2,k,3) + wfac*wf(3)
        flow_scratch(i,2,k,4) = wallj1(i,k)*flow_scratch(i,2,k,4) + wfac*wf(4)
        wfac = 1.0e0 - wallnj(i,k)
        call wall_func_jface(r, dAj, vol, Omega_block, Omega_wallnj_nd(i,k), mu, cons(:,:,:,1), Vx, Vr, Vt, i, nj, k, -1, wf)
        flow_scratch(i,nj-1,k,1) = wallnj(i,k)*flow_scratch(i,nj-1,k,1) + wfac*wf(1)
        flow_scratch(i,nj-1,k,2) = wallnj(i,k)*flow_scratch(i,nj-1,k,2) + wfac*wf(2)
        flow_scratch(i,nj-1,k,3) = wallnj(i,k)*flow_scratch(i,nj-1,k,3) + wfac*wf(3)
        flow_scratch(i,nj-1,k,4) = wallnj(i,k)*flow_scratch(i,nj-1,k,4) + wfac*wf(4)
    end do
    end do
    ! --- accumulate j-direction ---
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        fvisc(i,j,k,1) = fvisc(i,j,k,1) + flow_scratch(i,j,k,1) - flow_scratch(i,j+1,k,1)
        fvisc(i,j,k,2) = fvisc(i,j,k,2) + flow_scratch(i,j,k,2) - flow_scratch(i,j+1,k,2)
        fvisc(i,j,k,3) = fvisc(i,j,k,3) + flow_scratch(i,j,k,3) - flow_scratch(i,j+1,k,3)
        fvisc(i,j,k,4) = fvisc(i,j,k,4) + flow_scratch(i,j,k,4) - flow_scratch(i,j+1,k,4)
    end do
    end do
    end do

    ! --- k-direction: faces k=1..nk ---
    do k = 1, nk
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
        flow_scratch(i,j,k,1) = tauf(1)*dAk(1,i,j,k) + tauf(4)*dAk(2,i,j,k) + tauf(5)*dAk(3,i,j,k)
        flow_scratch(i,j,k,2) = tauf(4)*dAk(1,i,j,k) + tauf(2)*dAk(2,i,j,k) + tauf(6)*dAk(3,i,j,k)
        flow_scratch(i,j,k,3) = (tauf(5)*dAk(1,i,j,k) + tauf(6)*dAk(2,i,j,k) + tauf(3)*dAk(3,i,j,k)) * rf
        wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
        wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
        wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
        flow_scratch(i,j,k,4) = (wvisc(1)-qf(1))*dAk(1,i,j,k) &
                               + (wvisc(2)-qf(2))*dAk(2,i,j,k) &
                               + (wvisc(3)-qf(3))*dAk(3,i,j,k)
    end do
    end do
    end do
    ! --- cusp: couple the viscous k-faces across the seam ---
    ! The seam (a modelled trailing edge) joins the k=1 and k=nk faces. To make
    ! the viscous flux continuous across the cut -- and consistent with the
    ! inviscid coupling in correct_cusp_kface -- the two one-sided viscous
    ! fluxes are averaged so both faces carry the same shared seam flux. This
    ! gives the seam cells a real viscous flux that the residual can balance,
    ! instead of the slip-wall decoupling that left the seam over-constrained
    ! (inviscidly coupled but viscously zeroed) and prevented convergence.
    if (i_cusp_start > 0) then
        do j = 1, nj-1
        do i = i_cusp_start, i_cusp_end-1
            flow_scratch(i,j,1,:)  = 0.5e0 * (flow_scratch(i,j,1,:) + flow_scratch(i,j,nk,:))
            flow_scratch(i,j,nk,:) = flow_scratch(i,j,1,:)
        end do
        end do
    end if
    ! ===== Wall function injected as k=2 / k=nk-1 face flow =====
    ! At k=1 wall cells, overwrite flow_scratch(:,:,2,:) with wall_func_kface
    ! (off-wall stencil). At k=nk wall cells, overwrite flow_scratch(:,:,nk-1,:).
    ! Blend by wall mask: wall=0 -> use wf; wall=1 -> keep original.
    do j = 1, nj-1
    do i = 1, ni-1
        wfac = 1.0e0 - wallk1(i,j)
        call wall_func_kface(r, dAk, vol, Omega_block, Omega_wallk1_nd(i,j), mu, cons(:,:,:,1), Vx, Vr, Vt, i, j, 1, 1, wf)
        flow_scratch(i,j,2,1) = wallk1(i,j)*flow_scratch(i,j,2,1) + wfac*wf(1)
        flow_scratch(i,j,2,2) = wallk1(i,j)*flow_scratch(i,j,2,2) + wfac*wf(2)
        flow_scratch(i,j,2,3) = wallk1(i,j)*flow_scratch(i,j,2,3) + wfac*wf(3)
        flow_scratch(i,j,2,4) = wallk1(i,j)*flow_scratch(i,j,2,4) + wfac*wf(4)
        wfac = 1.0e0 - wallnk(i,j)
        call wall_func_kface(r, dAk, vol, Omega_block, Omega_wallnk_nd(i,j), mu, cons(:,:,:,1), Vx, Vr, Vt, i, j, nk, -1, wf)
        flow_scratch(i,j,nk-1,1) = wallnk(i,j)*flow_scratch(i,j,nk-1,1) + wfac*wf(1)
        flow_scratch(i,j,nk-1,2) = wallnk(i,j)*flow_scratch(i,j,nk-1,2) + wfac*wf(2)
        flow_scratch(i,j,nk-1,3) = wallnk(i,j)*flow_scratch(i,j,nk-1,3) + wfac*wf(3)
        flow_scratch(i,j,nk-1,4) = wallnk(i,j)*flow_scratch(i,j,nk-1,4) + wfac*wf(4)
    end do
    end do
    ! --- accumulate k-direction ---
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        fvisc(i,j,k,1) = fvisc(i,j,k,1) + flow_scratch(i,j,k,1) - flow_scratch(i,j,k+1,1)
        fvisc(i,j,k,2) = fvisc(i,j,k,2) + flow_scratch(i,j,k,2) - flow_scratch(i,j,k+1,2)
        fvisc(i,j,k,3) = fvisc(i,j,k,3) + flow_scratch(i,j,k,3) - flow_scratch(i,j,k+1,3)
        fvisc(i,j,k,4) = fvisc(i,j,k,4) + flow_scratch(i,j,k,4) - flow_scratch(i,j,k+1,4)
    end do
    end do
    end do

    ! ===== Zero fvisc at wall-adjacent cells =====
    ! The wall-adjacent cell is made entirely inviscid: the wall friction is
    ! applied at the i=2 / i=ni-1, j=2 / j=nj-1, k=2 / k=nk-1 face above; any
    ! remaining viscous content (polluted tau_cell from wall-touching
    ! Green-Gauss stencil, transverse-face contributions averaging the wall
    ! cell's tau) is discarded here.
    do k = 1, nk-1
    do j = 1, nj-1
        fvisc(1,j,k,1)    = fvisc(1,j,k,1)    * walli1(j,k)
        fvisc(1,j,k,2)    = fvisc(1,j,k,2)    * walli1(j,k)
        fvisc(1,j,k,3)    = fvisc(1,j,k,3)    * walli1(j,k)
        fvisc(1,j,k,4)    = fvisc(1,j,k,4)    * walli1(j,k)
        fvisc(ni-1,j,k,1) = fvisc(ni-1,j,k,1) * wallni(j,k)
        fvisc(ni-1,j,k,2) = fvisc(ni-1,j,k,2) * wallni(j,k)
        fvisc(ni-1,j,k,3) = fvisc(ni-1,j,k,3) * wallni(j,k)
        fvisc(ni-1,j,k,4) = fvisc(ni-1,j,k,4) * wallni(j,k)
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

    ! ===== Negate so the polar source / body forces are not flipped =====
    ! The accumulated flux is the divergence as +flux; the residual wants its
    ! negation. Doing it here (rather than in the Python caller) keeps F_body_nd's
    ! sign convention internal to the kernel and saves a separate array pass.
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        fvisc(i,j,k,1) = -fvisc(i,j,k,1)
        fvisc(i,j,k,2) = -fvisc(i,j,k,2)
        fvisc(i,j,k,3) = -fvisc(i,j,k,3)
        fvisc(i,j,k,4) = -fvisc(i,j,k,4)
    end do
    end do
    end do

end subroutine set_visc_force
