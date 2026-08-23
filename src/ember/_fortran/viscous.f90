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
    public :: vel_at, iface_vel, jface_vel, kface_vel
    public :: wall_core, wall_func, wall_yplus
    public :: wall_func_iface, wall_func_jface, wall_func_kface
    public :: wall_yplus_iface, wall_yplus_jface, wall_yplus_kface
    public :: kface_flow_tq, tau_q_at_cell
    public :: load_kface, load_ijedge_faces
    public :: polar_src, zero_wall_fvisc_border
    public :: wall_row_kface, wall_row_jface
    public :: VISC_JAREA
    public :: XLEN_FAC

    ! Mixing length squared per unit of SUMMED nodal wall distance: the
    ! turbulent length is kappa*w with von Karman kappa = 0.41 and w the
    ! 8-corner cell average of the nodal wall distance, so
    ! (kappa*w)^2 = (0.41 * sum/8)^2 = (0.41*0.125)^2 * sum^2. Folding the
    ! averaging into the constant is what keeps deriving it at the point of
    ! use down to a sum, a square and one multiply -- see the callers, which
    ! take the nodal wall distance rather than a cell-shaped mixing-length
    ! volume the block would otherwise have to store.
    real, parameter :: XLEN_FAC = (0.41e0 * 0.125e0)**2

    ! Skin-friction curve fit, shared by wall_core and the row forms below so
    ! the two spellings of the same physics cannot drift apart.
    real, parameter :: WALL_A1 = -1.767e-3
    real, parameter :: WALL_A2 = 3.177e-2
    real, parameter :: WALL_A3 = 2.5614e-1
    ! i-tile the row forms work in. Fixed size so their phase-A temps are
    ! plain stack locals rather than automatic arrays sized by ni: kernel
    ! scratch is never allocated per call. 64 floats x ~11 temps is under
    ! 3 KB, so a tile stays in L1 across the three phases.
    integer, parameter :: WALL_TW = 64
    ! j-panel AREA of set_visc_force's k walk, in cells: the panel is
    ! VISC_JAREA/ni rows deep, so the carry it bounds is a fixed number of
    ! bytes whatever the block's aspect ratio. Bounds the concurrent working
    ! set, not the traffic -- see the comment on the panel loop there.
    integer, parameter :: VISC_JAREA = 4400

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

    ! ------------------------------------------------------------------
    ! Velocity from the conserved state.
    !
    ! cons = (rho, rho*Vx, rho*Vr, rho*r*Vt, rho*e), so Vx = c2/c1 is the
    ! velocity's definition rather than an approximation of it, and the
    ! nodal velocity volumes this kernel used to stream are redundant with
    ! cons. Recomputed at every use and never gathered into a buffer -- a
    ! buffer would write more than it saves, which is the trade
    ! bench/README.md records for the same change in set_residual.
    !
    ! DIVIDE PER COMPONENT. Do not "optimise" these into one reciprocal per
    ! corner reused across the three components, which is how set_residual
    ! spells it and what bench/README.md's entry describes. Both reasons are
    ! measured, on a 1M-cell RK4/IRS/MG duct at 8-rank socket contention,
    ! 6 paired launches:
    !
    !   * it is SLOWER here -- the shared-reciprocal form came out +0.13%
    !     against this one's -0.72% at the same point in the work. That trade
    !     was measured on set_residual's short stage-1 body; these bodies hold
    !     eight reciprocals live across three components, and the register
    !     pressure costs more than the divisions save.
    !   * it MISCOMPILES. With -Ofast the shared-reciprocal form puts NaN
    !     through set_visc_force's interior on a rotated duct (a case with
    !     radial through-flow), while set_tau_q_faces' identical stage 1 stays
    !     clean. It is not this source being wrong: -O0 -fcheck=all is clean
    !     and bounds-clean, -fno-tree-vectorize is clean, and the derived
    !     velocities are bitwise equal to the arrays they replaced.
    !     Vectorisation is the trigger and the mechanism is not yet known.
    !
    ! Vt is BLOCK-RELATIVE, which is what every consumer here wants (see
    ! wall_core: "Vt is relative to block frame"). This subroutine is the one
    ! spelling of all four; the vectorised loops below inline it verbatim
    ! rather than call it, because a call in those loops costs them their
    ! vectorisation, but the text must stay identical to this.
    ! ------------------------------------------------------------------
    pure subroutine vel_at(cons, r, Omega, i, j, k, Vx, Vr, Vt)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:)
        real, intent(in) :: Omega
        integer, intent(in) :: i, j, k
        real, intent(out) :: Vx, Vr, Vt
        real :: g
        g = 1.0e0/cons(i,j,k,1)
        Vx = cons(i,j,k,2)*g
        Vr = cons(i,j,k,3)*g
        Vt = cons(i,j,k,4)*g/r(i,j,k) - Omega*r(i,j,k)
    end subroutine vel_at

    ! Face averages of (Vx, Vr, Vt_rel, rho) over the four nodes of a face,
    ! in the same node order as iface/jface/kface sum theirs.
    pure subroutine iface_vel(cons, r, Omega, i, j, k, Vxf, Vrf, Vtf, rhof)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:)
        real, intent(in) :: Omega
        integer, intent(in) :: i, j, k
        real, intent(out) :: Vxf, Vrf, Vtf, rhof
        real :: x1, x2, x3, x4, r1, r2, r3, r4, t1, t2, t3, t4
        call vel_at(cons, r, Omega, i, j,   k,   x1, r1, t1)
        call vel_at(cons, r, Omega, i, j+1, k,   x2, r2, t2)
        call vel_at(cons, r, Omega, i, j,   k+1, x3, r3, t3)
        call vel_at(cons, r, Omega, i, j+1, k+1, x4, r4, t4)
        Vxf = (x1 + x2 + x3 + x4) * 0.25e0
        Vrf = (r1 + r2 + r3 + r4) * 0.25e0
        Vtf = (t1 + t2 + t3 + t4) * 0.25e0
        rhof = iface(cons(:,:,:,1), i, j, k) * 0.25e0
    end subroutine iface_vel

    pure subroutine jface_vel(cons, r, Omega, i, j, k, Vxf, Vrf, Vtf, rhof)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:)
        real, intent(in) :: Omega
        integer, intent(in) :: i, j, k
        real, intent(out) :: Vxf, Vrf, Vtf, rhof
        real :: x1, x2, x3, x4, r1, r2, r3, r4, t1, t2, t3, t4
        call vel_at(cons, r, Omega, i,   j, k,   x1, r1, t1)
        call vel_at(cons, r, Omega, i+1, j, k,   x2, r2, t2)
        call vel_at(cons, r, Omega, i,   j, k+1, x3, r3, t3)
        call vel_at(cons, r, Omega, i+1, j, k+1, x4, r4, t4)
        Vxf = (x1 + x2 + x3 + x4) * 0.25e0
        Vrf = (r1 + r2 + r3 + r4) * 0.25e0
        Vtf = (t1 + t2 + t3 + t4) * 0.25e0
        rhof = jface(cons(:,:,:,1), i, j, k) * 0.25e0
    end subroutine jface_vel

    pure subroutine kface_vel(cons, r, Omega, i, j, k, Vxf, Vrf, Vtf, rhof)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:)
        real, intent(in) :: Omega
        integer, intent(in) :: i, j, k
        real, intent(out) :: Vxf, Vrf, Vtf, rhof
        real :: x1, x2, x3, x4, r1, r2, r3, r4, t1, t2, t3, t4
        call vel_at(cons, r, Omega, i,   j,   k, x1, r1, t1)
        call vel_at(cons, r, Omega, i+1, j,   k, x2, r2, t2)
        call vel_at(cons, r, Omega, i,   j+1, k, x3, r3, t3)
        call vel_at(cons, r, Omega, i+1, j+1, k, x4, r4, t4)
        Vxf = (x1 + x2 + x3 + x4) * 0.25e0
        Vrf = (r1 + r2 + r3 + r4) * 0.25e0
        Vtf = (t1 + t2 + t3 + t4) * 0.25e0
        rhof = kface(cons(:,:,:,1), i, j, k) * 0.25e0
    end subroutine kface_vel

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
        real, parameter :: a1 = WALL_A1
        real, parameter :: a2 = WALL_A2
        real, parameter :: a3 = WALL_A3
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

    pure subroutine wall_func_iface(cons, r, dA, vol, Omega_block, Omega_wall, mu, i, j, k, di, flow)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:), mu(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall
        integer, intent(in) :: i, j, k, di
        real :: Vxf, Vrf, Vtf, rf, rhof, muf
        real, intent(out) :: flow(4)
        call iface_vel(cons, r, Omega_block, i+di, j, k, Vxf, Vrf, Vtf, rhof)
        muf  = iface(mu,  i+di, j, k) * 0.25e0
        rf   = iface(r, i, j, k) * 0.25e0
        call wall_func(rf, dA(:,i,j,k), vol(i+(di-1)/2,j,k), Omega_block, Omega_wall, muf, rhof, Vxf, Vrf, Vtf, flow)
        flow = flow * di
    end subroutine wall_func_iface

    pure subroutine wall_func_jface(cons, r, dA, vol, Omega_block, Omega_wall, mu, i, j, k, dj, flow)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:), mu(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall
        integer, intent(in) :: i, j, k, dj
        real :: Vxf, Vrf, Vtf, rf, rhof, muf
        real, intent(out) :: flow(4)
        call jface_vel(cons, r, Omega_block, i, j+dj, k, Vxf, Vrf, Vtf, rhof)
        muf  = jface(mu,  i, j+dj, k) * 0.25e0
        rf   = jface(r, i, j, k) * 0.25e0
        call  wall_func(rf, dA(:,i,j,k), vol(i,j+(dj-1)/2,k), Omega_block, Omega_wall, muf, rhof, Vxf, Vrf, Vtf, flow)
        flow = flow * dj
    end subroutine wall_func_jface

    pure subroutine wall_func_kface(cons, r, dA, vol, Omega_block, Omega_wall, mu, i, j, k, dk, flow)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:), mu(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall
        integer, intent(in) :: i, j, k, dk
        real :: Vxf, Vrf, Vtf, rf, rhof, muf
        real, intent(out) :: flow(4)
        call kface_vel(cons, r, Omega_block, i, j, k+dk, Vxf, Vrf, Vtf, rhof)
        muf  = kface(mu,  i, j, k+dk) * 0.25e0
        rf   = kface(r, i, j, k) * 0.25e0
        call wall_func(rf, dA(:,i,j,k), vol(i,j,k+(dk-1)/2), Omega_block, Omega_wall, muf, rhof, Vxf, Vrf, Vtf, flow)
        flow = flow * dk
    end subroutine wall_func_kface

    ! ------------------------------------------------------------------
    ! Row forms of wall_func_kface/jface, blending straight into the face-
    ! flow buffer the sweep is filling.
    !
    ! WHY THESE EXIST. The per-cell forms above are called from an `i` loop at
    ! every wall face, and that loop does not vectorize: the opt report says
    ! "unsupported control flow in loop", which is wall_core's Re branch, and
    ! GCC then leaves a real call in the loop as well. The result is four
    ! SCALAR divides and two scalar square roots per wall face cell, on their
    ! own dependence chain -- 15.7% of set_visc_force's samples sit in the
    ! three wall_func_*face symbols, against ~7% of cells being wall faces.
    !
    ! The fix is the three-phase split the branch forces:
    !   A  gather the four-node face averages, the areas and Re
    !   B  the skin-friction curve, which carries the branch
    !   C  tau, the flux vector and the mask blend
    ! Phase B is written as the part that has to stay scalar; GCC in fact
    ! if-converts it and vectorises it too, calling libmvec's _ZGVdN8v_logf
    ! for the turbulent branch (-Ofast is what permits that). Splitting it out
    ! is still what makes A and C vectorise, which is where the divides and
    ! roots are.
    !
    ! Every expression keeps production's operands, order and association, so
    ! the change is which lane does the work, not what the work is. It is not
    ! bitwise all the same: under -Ofast GCC implements some of the vector
    ! divides as vrcpps plus a Newton step, where the scalar form used an
    ! exact vdivss. Measured on the duct case, the deviation is 0.625 ulp of
    ! the fvisc field scale and every differing cell lies in the two-cell
    ! shell the wall faces reach -- not one cell outside it moves, which is
    ! the structural statement that no interior term changed. Turning
    ! -fno-associative-math -ffp-contract=off on does NOT shrink it, so it is
    ! the reciprocals and not reassociation.
    !
    ! Phase A's outputs are fixed-size stack tiles, not automatic arrays: the
    ! row length is not known at compile time and kernel scratch is never
    ! allocated per call (nor may a `block` construct appear in this tree --
    ! it silently drops unrelated subroutines from the f2py build).
    ! ------------------------------------------------------------------
    pure subroutine wall_row_kface(ni, nj, nk, cons, r, dAk, vol, Omega_block, Omega_wall, &
        mu, wall, planes, j, k, dk, pslot)
        implicit none
        integer, intent(in) :: ni, nj, nk, j, k, dk, pslot
        real, intent(in) :: r(ni,nj,nk), mu(ni,nj,nk), cons(ni,nj,nk,5)
        real, intent(in) :: dAk(3,ni-1,nj-1,nk), vol(ni-1,nj-1,nk-1)
        real, intent(in) :: Omega_block, Omega_wall(ni-1), wall(ni-1)
        real, intent(inout) :: planes(ni,nj,4,2)
        integer :: i, i0, t, m, kv, kc
        real :: rf(WALL_TW), Vxf(WALL_TW), Vrf(WALL_TW), Vsf(WALL_TW)
        real :: rhof(WALL_TW), muf(WALL_TW), dAm(WALL_TW), Vm(WALL_TW)
        real :: Rew(WALL_TW), cf(WALL_TW)
        real :: tau, vec1, vec2, vec3, wfac, w, d, lnRew
        real :: q1, q2, q3, q4
        kv = k + dk
        kc = k + (dk - 1) / 2
        do i0 = 1, ni-1, WALL_TW
            m = min(WALL_TW, ni - i0)
            ! --- A: face averages, areas, Reynolds number ---
            do t = 1, m
                i = i0 + t - 1
                ! vel_at inlined, not called: this tile loop is vectorised
                ! and a call would cost it that. The four corners in the node
                ! order kface sums them, with one reciprocal each.
                q1 = 1.0e0/cons(i,j,kv,1)
                q2 = 1.0e0/cons(i+1,j,kv,1)
                q3 = 1.0e0/cons(i,j+1,kv,1)
                q4 = 1.0e0/cons(i+1,j+1,kv,1)
                Vxf(t)  = (cons(i,j,kv,2)*q1 + cons(i+1,j,kv,2)*q2 &
                         + cons(i,j+1,kv,2)*q3 + cons(i+1,j+1,kv,2)*q4) * 0.25e0
                Vrf(t)  = (cons(i,j,kv,3)*q1 + cons(i+1,j,kv,3)*q2 &
                         + cons(i,j+1,kv,3)*q3 + cons(i+1,j+1,kv,3)*q4) * 0.25e0
                Vsf(t)  = ((cons(i,j,kv,4)*q1/r(i,j,kv) - Omega_block*r(i,j,kv)) &
                         + (cons(i+1,j,kv,4)*q2/r(i+1,j,kv) - Omega_block*r(i+1,j,kv)) &
                         + (cons(i,j+1,kv,4)*q3/r(i,j+1,kv) - Omega_block*r(i,j+1,kv)) &
                         + (cons(i+1,j+1,kv,4)*q4/r(i+1,j+1,kv) - Omega_block*r(i+1,j+1,kv))) * 0.25e0
                rhof(t) = (cons(i,j,kv,1) + cons(i+1,j,kv,1) &
                         + cons(i,j+1,kv,1) + cons(i+1,j+1,kv,1)) * 0.25e0
                muf(t)  = (mu(i,j,kv) + mu(i+1,j,kv) + mu(i,j+1,kv) + mu(i+1,j+1,kv)) * 0.25e0
                rf(t)   = (r(i,j,k) + r(i+1,j,k) + r(i,j+1,k) + r(i+1,j+1,k)) * 0.25e0
                Vsf(t)  = Vsf(t) - (Omega_wall(i) - Omega_block) * rf(t)
                Vm(t)   = sqrt(Vxf(t)**2 + Vrf(t)**2 + Vsf(t)**2 + 1e-9)
                dAm(t)  = sqrt(dAk(1,i,j,k)**2 + dAk(2,i,j,k)**2 + dAk(3,i,j,k)**2)
                d       = vol(i,j,kc) / dAm(t)
                Rew(t)  = rhof(t) * Vm(t) * d / muf(t)
            end do
            ! --- B: skin friction, the one phase the branch keeps scalar ---
            do t = 1, m
                if (Rew(t) .lt. 127.53373025e0) then
                    cf(t) = 2e0 / Rew(t)
                else
                    lnRew = log(Rew(t))
                    cf(t) = (WALL_A1 + WALL_A2/lnRew + WALL_A3/lnRew/lnRew)
                end if
            end do
            ! --- C: stress, flux vector, mask blend ---
            do t = 1, m
                i = i0 + t - 1
                tau  = cf(t) * 0.5e0 * rhof(t) * Vm(t) * Vm(t)
                vec1 = Vxf(t) / Vm(t) * dAm(t)
                vec2 = Vrf(t) / Vm(t) * dAm(t)
                vec3 = Vsf(t) / Vm(t) * dAm(t)
                w = wall(i)
                wfac = 1.0e0 - w
                planes(i,j,1,pslot) = w*planes(i,j,1,pslot) + wfac*(vec1 * tau * dk)
                planes(i,j,2,pslot) = w*planes(i,j,2,pslot) + wfac*(vec2 * tau * dk)
                planes(i,j,3,pslot) = w*planes(i,j,3,pslot) + wfac*(rf(t) * vec3 * tau * dk)
                planes(i,j,4,pslot) = w*planes(i,j,4,pslot) &
                    + wfac*(Omega_wall(i) * rf(t) * vec3 * tau * dk)
            end do
        end do
    end subroutine wall_row_kface

    pure subroutine wall_row_jface(ni, nj, nk, cons, r, dAj, vol, Omega_block, Omega_wall, &
        mu, wall, rows, j, kc, dj, sslot)
        implicit none
        integer, intent(in) :: ni, nj, nk, j, kc, dj, sslot
        real, intent(in) :: r(ni,nj,nk), mu(ni,nj,nk), cons(ni,nj,nk,5)
        real, intent(in) :: dAj(3,ni-1,nj,nk-1), vol(ni-1,nj-1,nk-1)
        real, intent(in) :: Omega_block, Omega_wall(ni-1), wall(ni-1)
        real, intent(inout) :: rows(ni,4,3)
        integer :: i, i0, t, m, jv, jc
        real :: rf(WALL_TW), Vxf(WALL_TW), Vrf(WALL_TW), Vsf(WALL_TW)
        real :: rhof(WALL_TW), muf(WALL_TW), dAm(WALL_TW), Vm(WALL_TW)
        real :: Rew(WALL_TW), cf(WALL_TW)
        real :: tau, vec1, vec2, vec3, wfac, w, d, lnRew
        real :: q1, q2, q3, q4
        jv = j + dj
        jc = j + (dj - 1) / 2
        do i0 = 1, ni-1, WALL_TW
            m = min(WALL_TW, ni - i0)
            do t = 1, m
                i = i0 + t - 1
                ! vel_at inlined, as wall_row_kface does and for the same
                ! reason: the four corners in the node order jface sums them.
                q1 = 1.0e0/cons(i,jv,kc,1)
                q2 = 1.0e0/cons(i+1,jv,kc,1)
                q3 = 1.0e0/cons(i,jv,kc+1,1)
                q4 = 1.0e0/cons(i+1,jv,kc+1,1)
                Vxf(t)  = (cons(i,jv,kc,2)*q1 + cons(i+1,jv,kc,2)*q2 &
                         + cons(i,jv,kc+1,2)*q3 + cons(i+1,jv,kc+1,2)*q4) * 0.25e0
                Vrf(t)  = (cons(i,jv,kc,3)*q1 + cons(i+1,jv,kc,3)*q2 &
                         + cons(i,jv,kc+1,3)*q3 + cons(i+1,jv,kc+1,3)*q4) * 0.25e0
                Vsf(t)  = ((cons(i,jv,kc,4)*q1/r(i,jv,kc) - Omega_block*r(i,jv,kc)) &
                         + (cons(i+1,jv,kc,4)*q2/r(i+1,jv,kc) - Omega_block*r(i+1,jv,kc)) &
                         + (cons(i,jv,kc+1,4)*q3/r(i,jv,kc+1) - Omega_block*r(i,jv,kc+1)) &
                         + (cons(i+1,jv,kc+1,4)*q4/r(i+1,jv,kc+1) - Omega_block*r(i+1,jv,kc+1))) * 0.25e0
                rhof(t) = (cons(i,jv,kc,1) + cons(i+1,jv,kc,1) &
                         + cons(i,jv,kc+1,1) + cons(i+1,jv,kc+1,1)) * 0.25e0
                muf(t)  = (mu(i,jv,kc) + mu(i+1,jv,kc) + mu(i,jv,kc+1) + mu(i+1,jv,kc+1)) * 0.25e0
                rf(t)   = (r(i,j,kc) + r(i+1,j,kc) + r(i,j,kc+1) + r(i+1,j,kc+1)) * 0.25e0
                Vsf(t)  = Vsf(t) - (Omega_wall(i) - Omega_block) * rf(t)
                Vm(t)   = sqrt(Vxf(t)**2 + Vrf(t)**2 + Vsf(t)**2 + 1e-9)
                dAm(t)  = sqrt(dAj(1,i,j,kc)**2 + dAj(2,i,j,kc)**2 + dAj(3,i,j,kc)**2)
                d       = vol(i,jc,kc) / dAm(t)
                Rew(t)  = rhof(t) * Vm(t) * d / muf(t)
            end do
            do t = 1, m
                if (Rew(t) .lt. 127.53373025e0) then
                    cf(t) = 2e0 / Rew(t)
                else
                    lnRew = log(Rew(t))
                    cf(t) = (WALL_A1 + WALL_A2/lnRew + WALL_A3/lnRew/lnRew)
                end if
            end do
            do t = 1, m
                i = i0 + t - 1
                tau  = cf(t) * 0.5e0 * rhof(t) * Vm(t) * Vm(t)
                vec1 = Vxf(t) / Vm(t) * dAm(t)
                vec2 = Vrf(t) / Vm(t) * dAm(t)
                vec3 = Vsf(t) / Vm(t) * dAm(t)
                w = wall(i)
                wfac = 1.0e0 - w
                rows(i,1,sslot) = w*rows(i,1,sslot) + wfac*(vec1 * tau * dj)
                rows(i,2,sslot) = w*rows(i,2,sslot) + wfac*(vec2 * tau * dj)
                rows(i,3,sslot) = w*rows(i,3,sslot) + wfac*(rf(t) * vec3 * tau * dj)
                rows(i,4,sslot) = w*rows(i,4,sslot) &
                    + wfac*(Omega_wall(i) * rf(t) * vec3 * tau * dj)
            end do
        end do
    end subroutine wall_row_jface

    ! Diagnostic y+ counterparts of wall_func_iface/jface/kface above -- same
    ! face-averaging, mu included, calling wall_yplus instead of wall_func.
    ! No `* di/dj/dk` sign multiply: y+ has no direction, unlike a flux vector. Used only by
    ! wall_yplus_field (post-processing), never set_visc_force.
    pure subroutine wall_yplus_iface(cons, r, dA, vol, Omega_block, Omega_wall, mu, i, j, k, di, yplus)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:), mu(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall
        integer, intent(in) :: i, j, k, di
        real :: Vxf, Vrf, Vtf, rf, rhof, muf
        real, intent(out) :: yplus
        call iface_vel(cons, r, Omega_block, i+di, j, k, Vxf, Vrf, Vtf, rhof)
        muf  = iface(mu,  i+di, j, k) * 0.25e0
        rf   = iface(r, i, j, k) * 0.25e0
        call wall_yplus(rf, dA(:,i,j,k), vol(i+(di-1)/2,j,k), Omega_block, Omega_wall, muf, rhof, Vxf, Vrf, Vtf, yplus)
    end subroutine wall_yplus_iface

    pure subroutine wall_yplus_jface(cons, r, dA, vol, Omega_block, Omega_wall, mu, i, j, k, dj, yplus)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:), mu(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall
        integer, intent(in) :: i, j, k, dj
        real :: Vxf, Vrf, Vtf, rf, rhof, muf
        real, intent(out) :: yplus
        call jface_vel(cons, r, Omega_block, i, j+dj, k, Vxf, Vrf, Vtf, rhof)
        muf  = jface(mu,  i, j+dj, k) * 0.25e0
        rf   = jface(r, i, j, k) * 0.25e0
        call wall_yplus(rf, dA(:,i,j,k), vol(i,j+(dj-1)/2,k), Omega_block, Omega_wall, muf, rhof, Vxf, Vrf, Vtf, yplus)
    end subroutine wall_yplus_jface

    pure subroutine wall_yplus_kface(cons, r, dA, vol, Omega_block, Omega_wall, mu, i, j, k, dk, yplus)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:), mu(:,:,:)
        real, intent(in), contiguous :: dA(:,:,:,:), vol(:,:,:)
        real, intent(in) :: Omega_block, Omega_wall
        integer, intent(in) :: i, j, k, dk
        real :: Vxf, Vrf, Vtf, rf, rhof, muf
        real, intent(out) :: yplus
        call kface_vel(cons, r, Omega_block, i, j, k+dk, Vxf, Vrf, Vtf, rhof)
        muf  = kface(mu,  i, j, k+dk) * 0.25e0
        rf   = kface(r, i, j, k) * 0.25e0
        call wall_yplus(rf, dA(:,i,j,k), vol(i,j,k+(dk-1)/2), Omega_block, Omega_wall, muf, rhof, Vxf, Vrf, Vtf, yplus)
    end subroutine wall_yplus_kface


    ! Polar (radial-momentum) source per unit volume for cell (i,j,k):
    !     S = (rho*Vt^2 + (P - P_offset)) / r
    ! Identical arithmetic to production's trailing pass, factored out only so
    ! the hot fused loop and the O(surface) boundary-shell pass cannot drift
    ! apart. Bitwise agreement with production depends on this staying an
    ! expression-for-expression copy of it.
    pure function polar_src(cons, P, r, P_offset, i, j, k) result(S)
        implicit none
        real, intent(in), contiguous :: cons(:,:,:,:), P(:,:,:), r(:,:,:)
        real, intent(in) :: P_offset
        integer, intent(in) :: i, j, k
        real :: S
        real :: rhoc, rhorVtc, rc, Pc, Vtc
        rhoc = 0.125e0 * ( &
            cons(i,j,k,1) + cons(i+1,j,k,1) + cons(i,j+1,k,1) + cons(i+1,j+1,k,1) + &
            cons(i,j,k+1,1) + cons(i+1,j,k+1,1) + cons(i,j+1,k+1,1) + cons(i+1,j+1,k+1,1))
        rhorVtc = 0.125e0 * ( &
            cons(i,j,k,4) + cons(i+1,j,k,4) + cons(i,j+1,k,4) + cons(i+1,j+1,k,4) + &
            cons(i,j,k+1,4) + cons(i+1,j,k+1,4) + cons(i,j+1,k+1,4) + cons(i+1,j+1,k+1,4))
        rc = 0.125e0 * ( &
            r(i,j,k) + r(i+1,j,k) + r(i,j+1,k) + r(i+1,j+1,k) + &
            r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
        Pc = 0.125e0 * ( &
            P(i,j,k) + P(i+1,j,k) + P(i,j+1,k) + P(i+1,j+1,k) + &
            P(i,j,k+1) + P(i+1,j,k+1) + P(i,j+1,k+1) + P(i+1,j+1,k+1))
        Vtc = rhorVtc / (rhoc * rc)
        S = ((Pc - P_offset) + rhoc * Vtc**2) / rc
    end function polar_src


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


    ! tau/q for ONE cell: the expressions of the row body written for a single
    ! (i,j,k), with the same associations, so the two agree to the bit where
    ! the compiler treats them alike.
    !
    ! The row body vectorises over i, which set_tau_q_faces cannot do on two of
    ! its six faces -- the i faces pin exactly that axis. Rather than carry two
    ! shapes of the same arithmetic there, that producer walks those two faces
    ! cell by cell and calls this. They are ~8% of the shell, so the per-cell
    ! call costs little; the other four faces keep the row form.
    pure subroutine tau_q_at_cell(cons, T, mu, cp, kappa, Pr_turb, wdist, &
        vol, dAi, dAj, dAk, r, Omega_block, i, j, k, ni, nj, nk, tq)
        implicit none
        integer, intent(in) :: i, j, k, ni, nj, nk
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: T(ni, nj, nk), mu(ni, nj, nk)
        real, intent(in) :: cp(ni, nj, nk), kappa(ni, nj, nk)
        real, intent(in) :: Pr_turb
        real, intent(in) :: wdist(ni, nj, nk), vol(ni-1, nj-1, nk-1)
        real, intent(in) :: dAi(3, ni, nj-1, nk-1)
        real, intent(in) :: dAj(3, ni-1, nj, nk-1)
        real, intent(in) :: dAk(3, ni-1, nj-1, nk)
        real, intent(in) :: r(ni, nj, nk)
        real, intent(in) :: Omega_block
        real, intent(out) :: tq(9)

        real :: ivr, rcr, rhoc, cpc, muc, kac, vct
        real :: gVx1, gVx2, gVx3, gVr1, gVr2, gVr3, gVt1, gVt2, gVt3
        real :: f1, f2, f3, f4, f5, f6, g1, g2, g3
        real :: t1, t2, t3, t4, t5, t6, w1, w2, w3
        real :: vm, mut, fac, lambda, visc_lim, wsum, xl
        real :: v1, v2, v3, v4, v5, v6, v7, v8

        ivr = 0.25e0 / vol(i,j,k)
        rcr = 0.125e0 * (r(i,j,k)   + r(i+1,j,k)   + r(i,j+1,k)   + r(i+1,j+1,k) &
                       + r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
        rhoc = 0.125e0 * (cons(i,j,k,1)   + cons(i+1,j,k,1)   + cons(i,j+1,k,1)   + cons(i+1,j+1,k,1) &
                        + cons(i,j,k+1,1) + cons(i+1,j,k+1,1) + cons(i,j+1,k+1,1) + cons(i+1,j+1,k+1,1))
        cpc = 0.125e0 * (cp(i,j,k)   + cp(i+1,j,k)   + cp(i,j+1,k)   + cp(i+1,j+1,k) &
                       + cp(i,j,k+1) + cp(i+1,j,k+1) + cp(i,j+1,k+1) + cp(i+1,j+1,k+1))
        muc = 0.125e0 * (mu(i,j,k)   + mu(i+1,j,k)   + mu(i,j+1,k)   + mu(i+1,j+1,k) &
                       + mu(i,j,k+1) + mu(i+1,j,k+1) + mu(i,j+1,k+1) + mu(i+1,j+1,k+1))
        kac = 0.125e0 * (kappa(i,j,k)   + kappa(i+1,j,k)   + kappa(i,j+1,k)   + kappa(i+1,j+1,k) &
                       + kappa(i,j,k+1) + kappa(i+1,j,k+1) + kappa(i,j+1,k+1) + kappa(i+1,j+1,k+1))
        ! --- Vx ---
        ! Velocity from the conserved state: vel_at inlined rather than
        ! called, because a call here costs this loop its vectorisation.
        ! Divided per component, not through a shared corner reciprocal:
        ! see the rule above vel_at, which this must keep matching.
        v1 = cons(i,j,k,2)/cons(i,j,k,1)
        v2 = cons(i+1,j,k,2)/cons(i+1,j,k,1)
        v3 = cons(i,j+1,k,2)/cons(i,j+1,k,1)
        v4 = cons(i+1,j+1,k,2)/cons(i+1,j+1,k,1)
        v5 = cons(i,j,k+1,2)/cons(i,j,k+1,1)
        v6 = cons(i+1,j,k+1,2)/cons(i+1,j,k+1,1)
        v7 = cons(i,j+1,k+1,2)/cons(i,j+1,k+1,1)
        v8 = cons(i+1,j+1,k+1,2)/cons(i+1,j+1,k+1,1)
        f1 = v1+v3+v5+v7
        f2 = v2+v4+v6+v8
        f3 = v1+v2+v5+v6
        f4 = v3+v4+v7+v8
        f5 = v1+v2+v3+v4
        f6 = v5+v6+v7+v8
        g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
        g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
        g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
        gVx1 = g1*ivr
        gVx3 = g3*ivr
        gVx2 = g2*ivr - 0.125e0*(f1+f2)/rcr
        ! --- Vr ---
        v1 = cons(i,j,k,3)/cons(i,j,k,1)
        v2 = cons(i+1,j,k,3)/cons(i+1,j,k,1)
        v3 = cons(i,j+1,k,3)/cons(i,j+1,k,1)
        v4 = cons(i+1,j+1,k,3)/cons(i+1,j+1,k,1)
        v5 = cons(i,j,k+1,3)/cons(i,j,k+1,1)
        v6 = cons(i+1,j,k+1,3)/cons(i+1,j,k+1,1)
        v7 = cons(i,j+1,k+1,3)/cons(i,j+1,k+1,1)
        v8 = cons(i+1,j+1,k+1,3)/cons(i+1,j+1,k+1,1)
        f1 = v1+v3+v5+v7
        f2 = v2+v4+v6+v8
        f3 = v1+v2+v5+v6
        f4 = v3+v4+v7+v8
        f5 = v1+v2+v3+v4
        f6 = v5+v6+v7+v8
        g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
        g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
        g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
        gVr1 = g1*ivr
        gVr3 = g3*ivr
        gVr2 = g2*ivr - 0.125e0*(f1+f2)/rcr
        ! --- Vt ---
        v1 = cons(i,j,k,4)/cons(i,j,k,1)/r(i,j,k) - Omega_block*r(i,j,k)
        v2 = cons(i+1,j,k,4)/cons(i+1,j,k,1)/r(i+1,j,k) - Omega_block*r(i+1,j,k)
        v3 = cons(i,j+1,k,4)/cons(i,j+1,k,1)/r(i,j+1,k) - Omega_block*r(i,j+1,k)
        v4 = cons(i+1,j+1,k,4)/cons(i+1,j+1,k,1)/r(i+1,j+1,k) - Omega_block*r(i+1,j+1,k)
        v5 = cons(i,j,k+1,4)/cons(i,j,k+1,1)/r(i,j,k+1) - Omega_block*r(i,j,k+1)
        v6 = cons(i+1,j,k+1,4)/cons(i+1,j,k+1,1)/r(i+1,j,k+1) - Omega_block*r(i+1,j,k+1)
        v7 = cons(i,j+1,k+1,4)/cons(i,j+1,k+1,1)/r(i,j+1,k+1) - Omega_block*r(i,j+1,k+1)
        v8 = cons(i+1,j+1,k+1,4)/cons(i+1,j+1,k+1,1)/r(i+1,j+1,k+1) - Omega_block*r(i+1,j+1,k+1)
        f1 = v1+v3+v5+v7
        f2 = v2+v4+v6+v8
        f3 = v1+v2+v5+v6
        f4 = v3+v4+v7+v8
        f5 = v1+v2+v3+v4
        f6 = v5+v6+v7+v8
        vct = (f1+f2)*0.125e0
        g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
        g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
        g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
        gVt1 = g1*ivr
        gVt3 = g3*ivr
        gVt2 = g2*ivr - 0.125e0*(f1+f2)/rcr

        t1 = gVx1
        t2 = gVr2
        t3 = gVt3
        t4 = gVx2 + gVr1
        t5 = gVx3 + gVt1
        t6 = gVr3 + gVt2 - vct/rcr
        w1 = gVt2 - gVr3 + vct/rcr
        w2 = gVx3 - gVt1
        w3 = gVr1 - gVx2
        vm = sqrt(w1*w1 + w2*w2 + w3*w3)
        ! The max(0) contains the same gfortran 13 codegen fault set_visc_force
        ! documents at its own copy of this line -- keep the three identical.
        wsum = wdist(i,j,k)   + wdist(i+1,j,k)   + wdist(i,j+1,k)   + wdist(i+1,j+1,k) &
             + wdist(i,j,k+1) + wdist(i+1,j,k+1) + wdist(i,j+1,k+1) + wdist(i+1,j+1,k+1)
        xl = XLEN_FAC * wsum * wsum
        visc_lim = 3000e0 * muc
        mut = max(0.0e0, min(rhoc * xl * vm, visc_lim))
        fac = (muc + mut) * 0.5e0
        tq(1) = t1*fac
        tq(2) = t2*fac
        tq(3) = t3*fac
        tq(4) = t4*fac
        tq(5) = t5*fac
        tq(6) = t6*fac
        lambda = kac + mut * cpc / Pr_turb
        f1 = T(i,j,k)+T(i,j+1,k)+T(i,j,k+1)+T(i,j+1,k+1)
        f2 = T(i+1,j,k)+T(i+1,j+1,k)+T(i+1,j,k+1)+T(i+1,j+1,k+1)
        f3 = T(i,j,k)+T(i+1,j,k)+T(i,j,k+1)+T(i+1,j,k+1)
        f4 = T(i,j+1,k)+T(i+1,j+1,k)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
        f5 = T(i,j,k)+T(i+1,j,k)+T(i,j+1,k)+T(i+1,j+1,k)
        f6 = T(i,j,k+1)+T(i+1,j,k+1)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
        tq(7) = (f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k) &
              -f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1)) * (ivr*lambda*0.5e0)
        tq(9) = (f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k) &
              -f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1)) * (ivr*lambda*0.5e0)
        tq(8) = ((f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k) &
              -f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))*ivr &
              + 0.125e0*(f1+f2)/rcr) * (lambda*0.5e0)
    end subroutine tau_q_at_cell

    ! One k-face viscous flux with tau/q supplied as two 9-vectors, for the
    ! cusp seam correction.
    !
    ! Production's kface_flow (viscous.f90) takes the halo-indexed volume and
    ! reads tau_cell(i+1,j+1,k,:) and (i+1,j+1,k+1,:): at k=1 that is the halo
    ! slot then cell plane 1, at k=nk cell plane nk-1 then the halo slot. Those
    ! four values are exactly the two layers of the two k face buffers -- f_k1
    ! layer 2 is the low halo and layer 1 is cell plane 1; f_knk layer 1 is
    ! cell plane nk-1 and layer 2 its halo -- so the correction needs no volume
    ! and no rolling plane. It is also why the correction is reachable HERE and
    ! was not from the parent arms: they carry a rolling pair, and cell plane 1
    ! is long gone by the time the walk reaches nk, while the face buffers hold
    ! both planes at once for the whole call.
    !
    ! The arithmetic below is production's kface_flow, statement for statement.
    ! tests/test_viscous_cusp_seam.py gates the sign against an independent
    ! Python evaluation of the same face flux.
    pure subroutine kface_flow_tq(tqlo, tqhi, cons, r, dAk, Omega_block, i, j, k, flow)
        implicit none
        real, intent(in) :: tqlo(9), tqhi(9)
        real, intent(in), contiguous :: cons(:,:,:,:), r(:,:,:)
        real, intent(in), contiguous :: dAk(:,:,:,:)
        real, intent(in) :: Omega_block
        integer, intent(in) :: i, j, k
        real, intent(out) :: flow(4)
        real :: tauf(6), qf(3), Vf(3), rf, Vabs, wvisc(3), rhof
        tauf(1) = (tqlo(1) + tqhi(1)) * 0.5e0
        tauf(2) = (tqlo(2) + tqhi(2)) * 0.5e0
        tauf(3) = (tqlo(3) + tqhi(3)) * 0.5e0
        tauf(4) = (tqlo(4) + tqhi(4)) * 0.5e0
        tauf(5) = (tqlo(5) + tqhi(5)) * 0.5e0
        tauf(6) = (tqlo(6) + tqhi(6)) * 0.5e0
        qf(1)   = (tqlo(7) + tqhi(7)) * 0.5e0
        qf(2)   = (tqlo(8) + tqhi(8)) * 0.5e0
        qf(3)   = (tqlo(9) + tqhi(9)) * 0.5e0
        call kface_vel(cons, r, Omega_block, i, j, k, Vf(1), Vf(2), Vf(3), rhof)
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
    end subroutine kface_flow_tq

    ! One k-direction halo plane, from its face buffer into a rolling tq slot.
    ! No mask: the face buffer's layer 2 already carries (2*wall-1), applied
    ! once by set_tau_q_faces rather than on every read.
    ! jlo/jhi restrict the fill to the caller's j panel: outside it the plane
    ! holds another panel's rows and is not read.
    subroutine load_kface(fk, plane, jlo, jhi, ni, nj)
        implicit none
        integer, intent(in) :: jlo, jhi, ni, nj
        real, intent(in) :: fk(ni-1, 9, nj-1, 2)
        real, intent(inout) :: plane(ni+1, nj+1, 9)
        integer :: i, j, c
        do j = jlo, jhi
        do c = 1, 9
        do i = 1, ni-1
            plane(i+1,j+1,c) = fk(i,c,j,2)
        end do
        end do
        end do
    end subroutine load_kface

    ! The four i/j halo edges of one cell plane, from their face buffers.
    ! Both reads are unit-stride in the face buffer; the i-edge WRITE into the
    ! halo-shaped plane still strides (ni+1), which is the plane's layout, not
    ! this buffer's.
    ! jlo/jhi are the cell rows this panel produced, so the i edges follow the
    ! rows that exist. The j edges are the plane's own row 0 and row nj, which
    ! belong to the first and last panel respectively -- a middle panel has
    ! real cell rows on both sides and reads no j edge at all.
    subroutine load_ijedge_faces(f_i1, f_ini, f_j1, f_jnj, plane, k, jlo, jhi, ni, nj, nk)
        implicit none
        integer, intent(in) :: k, jlo, jhi, ni, nj, nk
        real, intent(in) :: f_i1(nj-1, 9, nk-1, 2), f_ini(nj-1, 9, nk-1, 2)
        real, intent(in) :: f_j1(ni-1, 9, nk-1, 2), f_jnj(ni-1, 9, nk-1, 2)
        real, intent(inout) :: plane(ni+1, nj+1, 9)
        integer :: i, j, c
        do c = 1, 9
        do j = jlo, jhi
            plane(1,j+1,c)    = f_i1(j,c,k,2)
            plane(ni+1,j+1,c) = f_ini(j,c,k,2)
        end do
        end do
        if (jlo == 1) then
            do c = 1, 9
            do i = 1, ni-1
                plane(i+1,1,c) = f_j1(i,c,k,2)
            end do
            end do
        end if
        if (jhi == nj-1) then
            do c = 1, 9
            do i = 1, ni-1
                plane(i+1,nj+1,c) = f_jnj(i,c,k,2)
            end do
            end do
        end if
    end subroutine load_ijedge_faces

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
! `set_tau_q_faces` computes the stress tensor tau(6) and heat-flux vector q(3)
! for the cells on the block's BOUNDARY SHELL only, by a Green-Gauss gradient
! over the six cell faces, each face average being the mean of its four corner
! nodes. tau and q are stored multiplied by 2, so that averaging two adjacent
! cells recovers the correct face value without a further factor.
!
! `set_visc_force` then walks k, producing interior tau/q into a rolling cell
! plane pair as it goes and consuming it in the same walk, and accumulates the
! viscous face flows into fvisc. Interior tau/q therefore never reaches memory:
! only the shell does, in the six surface buffers (Block.tau_q_faces), which is
! all the grid-wide periodic seam exchange between the two kernels needs to
! carry (grid.py's update_sources, PeriodicCommunicator.exchange_faces).
!
! WHY THE SHELL IS A SEPARATE KERNEL AT ALL. The exchange has to run after
! every block's boundary tau/q exists and before any block consumes its
! neighbour's, so something O(surface) must precede the walk. That something is
! `set_tau_q_faces`; the volume pass it replaced wrote 9 floats per cell that
! the second kernel then streamed straight back, and sizing the scratch arena
! for that volume was three quarters of the arena.
!
! Interior faces take tauf as the average of tau_cell from the two adjacent
! cells; boundary faces (i=1, i=ni, j=1, j=nj, k=1, k=nk) take it from the
! single adjacent interior cell (already doubled above, so no extra factor)
! and blend the free-stream viscous stress with a wall-function force
! according to the wall weight.
!
! Strip-mined / SoA evaluation of tau and q: the per-cell work is split into
! two flat do-i loops over each (j,k) row, with the per-cell intermediates held
! in row temps dimensioned with i as the contiguous axis. This gives the auto-
! vectorizer simple, call-light, unit-stride loops to vectorize over i instead
! of one deep nest with a heavy inlined body. Both kernels use that row form,
! bar the two i faces of the shell, which pin exactly the axis it vectorises
! over and fall back to the per-cell `tau_q_at_cell`.
!
! Three modelling choices are built into the arithmetic and are not obvious
! from it:
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


! O(surface) boundary producer for the viscous face buffers.
!
! set_visc_force produces interior tau/q inside its own k walk and reads only
! the boundary shell. This kernel is that shell -- six surface buffers instead
! of a volume, so the producer is O(surface), the periodic seam exchange
! carries O(surface) too, and the consumer's halo source does not depend on
! the block's topology. The volume pass this replaced wrote nine floats for
! every cell so that the second kernel could read the shell out of them.
!
! Layer 1 of each face is the block's own edge-cell tau/q. Layer 2 is the halo
! value, seeded here as (2*wall - 1) * layer1: a viscous wall gets -edge, so
! the face average is zero, and a permeable or slip face gets +edge, the
! single-sided stress -- which is right for every face nothing exchanges. The
! periodic exchange then overwrites layer 2 wherever a patch connects. Applying
! the sign once here is what lets the consumer read the halo with no mask.
!
! mu_turb is deliberately NOT written. set_visc_force writes it for every cell
! from its own producer pass, so writing the shell here as well would be
! duplicated traffic for a value about to be overwritten with the identical
! number.
!
! The shell's edges and corners belong to more than one face and are computed
! once per face they lie on. That duplication is O(edge), and removing it would
! mean carrying a "which faces own this cell" test into every loop.
subroutine set_tau_q_faces( &
    cons, T, mu, cp, kappa, Pr_turb, wdist, vol, dAi, dAj, dAk, &
    r, Omega_block, &
    f_i1, f_ini, f_j1, f_jnj, f_k1, f_knk, &
    walli1, wallni, wallj1, wallnj, wallk1, wallnk, &
    ni, nj, nk)

    use viscous_helpers
    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: T(ni, nj, nk)
    real, intent(in) :: mu(ni, nj, nk)
    real, intent(in) :: cp(ni, nj, nk)
    real, intent(in) :: kappa(ni, nj, nk)
    real, intent(in) :: Pr_turb
    real, intent(in) :: wdist(ni, nj, nk)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega_block
    ! Component axis second so that, at a fixed trailing spatial index, the
    ! (edge, component) block is contiguous -- the order the consumers walk it.
    real, intent(inout) :: f_i1(nj-1, 9, nk-1, 2), f_ini(nj-1, 9, nk-1, 2)
    real, intent(inout) :: f_j1(ni-1, 9, nk-1, 2), f_jnj(ni-1, 9, nk-1, 2)
    real, intent(inout) :: f_k1(ni-1, 9, nj-1, 2), f_knk(ni-1, 9, nj-1, 2)
    real, intent(in) :: walli1(nj-1, nk-1), wallni(nj-1, nk-1)
    real, intent(in) :: wallj1(ni-1, nk-1), wallnj(ni-1, nk-1)
    real, intent(in) :: wallk1(ni-1, nj-1), wallnk(ni-1, nj-1)

    integer :: i, j, k, c
    integer :: irow, nrow, face
    real :: tq(9)
    real :: visc_lim
    ! Row temps for the j/k face rows -- i is the contiguous (dim-1) axis, the
    ! SIMD lane index, as set_visc_force's producer declares them. AUTOMATIC, and
    ! that is load-bearing: making them dummy arguments of a shared helper
    ! costs the stage-1 loop its vectorization, because GCC versions that loop
    ! against a runtime alias check and will not do so against a dummy.
    real :: gVx(ni-1, 3), gVr(ni-1, 3), gVt(ni-1, 3)
    real :: vct(ni-1), rcr(ni-1), ivr(ni-1), rhoc(ni-1), cpc(ni-1)
    real :: muc(ni-1), kac(ni-1), xlr(ni-1)
    ! One row of the nine components, staged here and then dispatched to
    ! whichever face buffer this row belongs to.
    real :: tqr(ni-1, 9)
    real :: f1, f2, f3, f4, f5, f6, g1, g2, g3
    real :: t1, t2, t3, t4, t5, t6, w1, w2, w3, vm, mut, fac, lambda, wsum
    ! Per-component corner velocities for the stage-1 gathers.
    real :: v1, v2, v3, v4, v5, v6, v7, v8

    ! --- i faces: the two that pin the axis the row body vectorises over ---
    do k = 1, nk-1
    do j = 1, nj-1
        call tau_q_at_cell(cons, T, mu, cp, kappa, Pr_turb, wdist, vol, &
            dAi, dAj, dAk, r, Omega_block, 1, j, k, ni, nj, nk, tq)
        do c = 1, 9
            f_i1(j,c,k,1) = tq(c)
            f_i1(j,c,k,2) = tq(c) * (2.0e0*walli1(j,k) - 1.0e0)
        end do
        call tau_q_at_cell(cons, T, mu, cp, kappa, Pr_turb, wdist, vol, &
            dAi, dAj, dAk, r, Omega_block, ni-1, j, k, ni, nj, nk, tq)
        do c = 1, 9
            f_ini(j,c,k,1) = tq(c)
            f_ini(j,c,k,2) = tq(c) * (2.0e0*wallni(j,k) - 1.0e0)
        end do
    end do
    end do

    ! --- j and k faces: one row body, dispatched to the four buffers ---
    !
    ! These four faces walk i, the axis the tau/q body vectorises over, so they
    ! take the ROW form rather than the per-cell call the i faces are stuck
    ! with. They are ~88% of the shell, and the per-cell form costs about
    ! 85 ns per surface cell against the row form's ~26: this kernel went from
    ! 6.5 to 2.6 ns per CELL of the block when they were converted, which is
    ! the difference between the surface-buffer scheme costing the viscous pair
    ! 7% and it winning 4% (bench/README.md).
    !
    ! The arithmetic below is set_visc_force's producer, statement for statement
    ! and with the same associations, with ONE deliberate exception: the mixing
    ! length. That matching is not tidiness -- the consumer averages an edge
    ! cell it produced with its own row body against a halo value it reads from
    ! here, so the two agreeing keeps a boundary face from carrying a spurious
    ! jump -- which is why the exception is worth stating precisely.
    !
    ! This kernel takes the mixing length from a stage-1 row (xlr below); the
    ! consumer computes it where it uses it. Same expression, same order, but
    ! the round trip through the row gives the compiler different contraction
    ! freedom under -Ofast, so the two no longer agree to the BIT. Measured, the
    ! disagreement reaches this kernel's own face buffers at 1.2e-7 relative --
    ! one to two ulp of float32 -- and does not reach the consumer's output at
    ! all: fvisc and mu_turb are bitwise unchanged by the split. The face
    ! goldens carry rtol 1e-4, so a divergence that is anything more than
    ! last-ulp fails there.
    !
    ! The split is what it costs to walk the shell: this kernel visits a
    ! scattered set of rows with nothing to amortise the eight corner loads
    ! against, and hoisting them into stage 1 -- beside the five eight-corner
    ! averages already there, sharing their addressing -- is worth -2.45% of
    ! it. The consumer walks every cell in a j panel where those same rows are
    ! resident and reused between adjacent j, so the row buys it nothing and
    ! the extra store, load and automatic array cost it 2.05%. Nine tenths of
    ! the pair's time is the consumer's, so it keeps the in-place form.
    !
    ! ONE copy of the body serves all four faces. The row index below names the
    ! (j, k) of each row and which buffer it lands in, and the dispatch runs
    ! once per row of ni-1 cells. Writing the body out four times would be four
    ! chances for the faces to drift apart.
    nrow = 2*(nk-1) + 2*(nj-1)
    do irow = 1, nrow
        if (irow <= nk-1) then
            face = 1
            j = 1
            k = irow
        else if (irow <= 2*(nk-1)) then
            face = 2
            j = nj-1
            k = irow - (nk-1)
        else if (irow <= 2*(nk-1) + (nj-1)) then
            face = 3
            j = irow - 2*(nk-1)
            k = 1
        else
            face = 4
            j = irow - 2*(nk-1) - (nj-1)
            k = nk-1
        end if

        ! Stage 1: velocity gradients + cell metrics, vectorizable over i.
        do i = 1, ni-1
            ivr(i) = 0.25e0 / vol(i,j,k)
            rcr(i) = 0.125e0 * (r(i,j,k)   + r(i+1,j,k)   + r(i,j+1,k)   + r(i+1,j+1,k) &
                              + r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
            rhoc(i) = 0.125e0 * (cons(i,j,k,1)   + cons(i+1,j,k,1)   + cons(i,j+1,k,1)   + cons(i+1,j+1,k,1) &
                               + cons(i,j,k+1,1) + cons(i+1,j,k+1,1) + cons(i,j+1,k+1,1) + cons(i+1,j+1,k+1,1))
            cpc(i) = 0.125e0 * (cp(i,j,k)   + cp(i+1,j,k)   + cp(i,j+1,k)   + cp(i+1,j+1,k) &
                              + cp(i,j,k+1) + cp(i+1,j,k+1) + cp(i,j+1,k+1) + cp(i+1,j+1,k+1))
            muc(i) = 0.125e0 * (mu(i,j,k)   + mu(i+1,j,k)   + mu(i,j+1,k)   + mu(i+1,j+1,k) &
                              + mu(i,j,k+1) + mu(i+1,j,k+1) + mu(i,j+1,k+1) + mu(i+1,j+1,k+1))
            kac(i) = 0.125e0 * (kappa(i,j,k)   + kappa(i+1,j,k)   + kappa(i,j+1,k)   + kappa(i+1,j+1,k) &
                              + kappa(i,j,k+1) + kappa(i+1,j,k+1) + kappa(i,j+1,k+1) + kappa(i+1,j+1,k+1))
            ! Mixing length squared, from the nodal wall distance -- here
            ! rather than at its point of use in stage 2, and NOT matching the
            ! consumer's spelling. See the header note above for why the two
            ! differ, what it is worth (-2.45% here) and what it costs in
            ! precision (last-ulp, contained to this kernel's face buffers).
            wsum = wdist(i,j,k)   + wdist(i+1,j,k)   + wdist(i,j+1,k)   + wdist(i+1,j+1,k) &
                 + wdist(i,j,k+1) + wdist(i+1,j,k+1) + wdist(i,j+1,k+1) + wdist(i+1,j+1,k+1)
            xlr(i) = XLEN_FAC * wsum * wsum
            ! --- Vx ---
            ! Velocity from the conserved state: vel_at inlined rather than
            ! called, because a call here costs this loop its vectorisation.
            ! Divided per component, not through a shared corner reciprocal:
            ! see the rule above vel_at, which this must keep matching.
            v1 = cons(i,j,k,2)/cons(i,j,k,1)
            v2 = cons(i+1,j,k,2)/cons(i+1,j,k,1)
            v3 = cons(i,j+1,k,2)/cons(i,j+1,k,1)
            v4 = cons(i+1,j+1,k,2)/cons(i+1,j+1,k,1)
            v5 = cons(i,j,k+1,2)/cons(i,j,k+1,1)
            v6 = cons(i+1,j,k+1,2)/cons(i+1,j,k+1,1)
            v7 = cons(i,j+1,k+1,2)/cons(i,j+1,k+1,1)
            v8 = cons(i+1,j+1,k+1,2)/cons(i+1,j+1,k+1,1)
            f1 = v1+v3+v5+v7
            f2 = v2+v4+v6+v8
            f3 = v1+v2+v5+v6
            f4 = v3+v4+v7+v8
            f5 = v1+v2+v3+v4
            f6 = v5+v6+v7+v8
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVx(i,1) = g1*ivr(i)
            gVx(i,3) = g3*ivr(i)
            gVx(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vr ---
            v1 = cons(i,j,k,3)/cons(i,j,k,1)
            v2 = cons(i+1,j,k,3)/cons(i+1,j,k,1)
            v3 = cons(i,j+1,k,3)/cons(i,j+1,k,1)
            v4 = cons(i+1,j+1,k,3)/cons(i+1,j+1,k,1)
            v5 = cons(i,j,k+1,3)/cons(i,j,k+1,1)
            v6 = cons(i+1,j,k+1,3)/cons(i+1,j,k+1,1)
            v7 = cons(i,j+1,k+1,3)/cons(i,j+1,k+1,1)
            v8 = cons(i+1,j+1,k+1,3)/cons(i+1,j+1,k+1,1)
            f1 = v1+v3+v5+v7
            f2 = v2+v4+v6+v8
            f3 = v1+v2+v5+v6
            f4 = v3+v4+v7+v8
            f5 = v1+v2+v3+v4
            f6 = v5+v6+v7+v8
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVr(i,1) = g1*ivr(i)
            gVr(i,3) = g3*ivr(i)
            gVr(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vt ---
            v1 = cons(i,j,k,4)/cons(i,j,k,1)/r(i,j,k) - Omega_block*r(i,j,k)
            v2 = cons(i+1,j,k,4)/cons(i+1,j,k,1)/r(i+1,j,k) - Omega_block*r(i+1,j,k)
            v3 = cons(i,j+1,k,4)/cons(i,j+1,k,1)/r(i,j+1,k) - Omega_block*r(i,j+1,k)
            v4 = cons(i+1,j+1,k,4)/cons(i+1,j+1,k,1)/r(i+1,j+1,k) - Omega_block*r(i+1,j+1,k)
            v5 = cons(i,j,k+1,4)/cons(i,j,k+1,1)/r(i,j,k+1) - Omega_block*r(i,j,k+1)
            v6 = cons(i+1,j,k+1,4)/cons(i+1,j,k+1,1)/r(i+1,j,k+1) - Omega_block*r(i+1,j,k+1)
            v7 = cons(i,j+1,k+1,4)/cons(i,j+1,k+1,1)/r(i,j+1,k+1) - Omega_block*r(i,j+1,k+1)
            v8 = cons(i+1,j+1,k+1,4)/cons(i+1,j+1,k+1,1)/r(i+1,j+1,k+1) - Omega_block*r(i+1,j+1,k+1)
            f1 = v1+v3+v5+v7
            f2 = v2+v4+v6+v8
            f3 = v1+v2+v5+v6
            f4 = v3+v4+v7+v8
            f5 = v1+v2+v3+v4
            f6 = v5+v6+v7+v8
            vct(i) = (f1+f2)*0.125e0
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVt(i,1) = g1*ivr(i)
            gVt(i,3) = g3*ivr(i)
            gVt(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
        end do
        ! Stage 2: tau, the mixing-length viscosity and q, into the row buffer.
        ! mu_turb is deliberately NOT stored (see the header): the fused
        ! consumer writes it for every cell from its own producer pass.
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
            ! The max(0) carries the same gfortran 13 codegen fault
            ! set_visc_force documents at length at its own copy of this line --
            ! keep it in all three copies (there and in tau_q_at_cell). Only
            ! where the mixing length comes from differs between them.
            visc_lim = 3000e0 * muc(i)
            mut = max(0.0e0, min(rhoc(i) * xlr(i) * vm, visc_lim))
            fac = (muc(i) + mut) * 0.5e0
            tqr(i,1) = t1*fac
            tqr(i,2) = t2*fac
            tqr(i,3) = t3*fac
            tqr(i,4) = t4*fac
            tqr(i,5) = t5*fac
            tqr(i,6) = t6*fac
            lambda = kac(i) + mut * cpc(i) / Pr_turb
            f1 = T(i,j,k)+T(i,j+1,k)+T(i,j,k+1)+T(i,j+1,k+1)
            f2 = T(i+1,j,k)+T(i+1,j+1,k)+T(i+1,j,k+1)+T(i+1,j+1,k+1)
            f3 = T(i,j,k)+T(i+1,j,k)+T(i,j,k+1)+T(i+1,j,k+1)
            f4 = T(i,j+1,k)+T(i+1,j+1,k)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            f5 = T(i,j,k)+T(i+1,j,k)+T(i,j+1,k)+T(i+1,j+1,k)
            f6 = T(i,j,k+1)+T(i+1,j,k+1)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            tqr(i,7) = (f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k) &
                  -f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            tqr(i,9) = (f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k) &
                  -f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            tqr(i,8) = ((f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k) &
                  -f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))*ivr(i) &
                  + 0.125e0*(f1+f2)/rcr(i)) * (lambda*0.5e0)
        end do

        ! Dispatch: layer 1 is the block's own edge cell, layer 2 the halo
        ! value with (2*wall - 1) applied once here rather than on every read.
        select case (face)
        case (1)
            do c = 1, 9
            do i = 1, ni-1
                f_j1(i,c,k,1) = tqr(i,c)
                f_j1(i,c,k,2) = tqr(i,c) * (2.0e0*wallj1(i,k) - 1.0e0)
            end do
            end do
        case (2)
            do c = 1, 9
            do i = 1, ni-1
                f_jnj(i,c,k,1) = tqr(i,c)
                f_jnj(i,c,k,2) = tqr(i,c) * (2.0e0*wallnj(i,k) - 1.0e0)
            end do
            end do
        case (3)
            do c = 1, 9
            do i = 1, ni-1
                f_k1(i,c,j,1) = tqr(i,c)
                f_k1(i,c,j,2) = tqr(i,c) * (2.0e0*wallk1(i,j) - 1.0e0)
            end do
            end do
        case default
            do c = 1, 9
            do i = 1, ni-1
                f_knk(i,c,j,1) = tqr(i,c)
                f_knk(i,c,j,2) = tqr(i,c) * (2.0e0*wallnk(i,j) - 1.0e0)
            end do
            end do
        end select
    end do

end subroutine set_tau_q_faces


! Pass 2: interior tau/q and the viscous face flows, in one walk.
!
! The halo source is the six SURFACE buffers (Block.tau_q_faces), layer 2 of
! each holding the halo value: set_tau_q_faces applied (2*wall - 1) to it once
! -- so a viscous wall arrives as -edge and a permeable or slip face as +edge,
! which is right for every face nothing exchanges -- and exchange_faces has
! overwritten it wherever a patch connects. The kernel neither writes those
! buffers nor reads layer 1 of them, except in the cusp pass at the end, and
! it needs no wall mask on the halo path and no knowledge of the block's
! topology: a block connected to neighbours is served exactly as one periodic
! to itself.
!
! ROLLING tau/q. The walk over k face planes consumes tau/q cell planes k-1
! and k and nothing else, so tau/q is produced inside the walk into a rolling
! plane pair (`tq`, slot ta = cell plane k-1, slot tb = cell plane k) and never
! reaches memory. That is the whole reason there is no tau/q volume: the arena
! holds two cell planes where it used to hold a padded volume of nine floats
! per cell.
!
! ROLLING FACE FLOWS, one store per cell. Each direction fuses its face-flux
! loop with its fvisc accumulate through a rolling buffer -- the i direction a
! face row (`rows` slot 1), the j direction an alternating face-row pair (slots
! 2/3), the k direction an alternating face-plane pair (`planes`) -- and the
! three differences plus the polar source land in ONE store per cell rather
! than production's former four visits to fvisc.
!
! J-PANELLED. The k walk carries a tau/q cell plane pair and the k-face flow
! plane pair from one k step to the next. Untiled that is ~1.9 MB at a
! 273x65x57 block: one rank holds it in a 20 MB L3, eight ranks cannot, and the
! carry then evicts the nodal fields both halves of the walk re-read every
! plane. Panelling j divides the carry by nj-1 / jbw and leaves each cell
! visited once. It is worth about 4% serially and about 46 points at 8-rank
! socket contention -- the arm was +23.8% against the old two-kernel pair
! before panelling and -22.8% after (bench/README.md).
!
! The panel costs two duplicated cell rows per panel per k plane: the j-face
! row at jp0, and the producer rows at jp0-1 and jp1+1 that its lowest and
! highest j faces average against. That is 2/jbw of the producer, and it is
! what having no volume to read those rows out of costs.
!
! WALL FUNCTIONS are injected into the first interior face row/plane rather
! than the wall face itself (the Multall scheme): a k injection at k=2 and
! k=nk-1, a j injection at j=2 and j=nj-1, and the two i injections inline in
! the row scan. Blending is unconditional and by wfac, so a mask of 0 and a
! mask of 1 cost the same.
!
! THE CUSP SEAM (k=1 face coupled to k=nk) is inherently non-local in k, so it
! is an O(surface) correction pass after the walk. It reads the two k face
! buffers, which hold cell planes 1 and nk-1 and both their halos for the whole
! call -- a rolling pair never could, which is why the fused kernels that took
! their halo from a volume could not implement this at all.
!
! The polar (radial-momentum) source is fused into the interior store above and
! runs as a separate pass over the boundary shell after the wall zeroing, since
! it is geometric content the wall mask must not eat.
subroutine set_visc_force( &
    cons, vol, dAi, dAj, dAk, &
    Omega_block, r, mu, P, P_offset, &
    fvisc, &
    T, cp, kappa, Pr_turb, wdist, &
    mu_turb, &
    f_i1, f_ini, f_j1, f_jnj, f_k1, f_knk, &
    tq, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    Omega_walli1_nd, Omega_wallj1_nd, Omega_wallk1_nd, &
    Omega_wallni_nd, Omega_wallnj_nd, Omega_wallnk_nd, &
    i_cusp_start, i_cusp_end, &
    jbw_in, ni, nj, nk)

    use viscous_helpers
    implicit none

    integer, intent(in) :: ni, nj, nk
    ! j-panel width in cell rows. 0 -- what production passes -- sizes it from
    ! VISC_JAREA, which is the only setting anyone should march with. A
    ! positive value overrides it, so a test can sweep the panelling and prove
    ! the answer does not depend on it: the panel duplicates rows at its
    ! edges, which is exactly the kind of thing that goes quietly wrong.
    integer, intent(in) :: jbw_in
    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega_block
    real, intent(in) :: mu(ni, nj, nk)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(inout) :: fvisc(ni-1, nj-1, nk-1, 4)
    real, intent(in) :: T(ni, nj, nk)
    real, intent(in) :: cp(ni, nj, nk)
    real, intent(in) :: kappa(ni, nj, nk)
    real, intent(in) :: Pr_turb
    real, intent(in) :: wdist(ni, nj, nk)
    ! Cell-centred mixing-length viscosity, written at the cell's low-corner
    ! node. The final node in each axis is padding that is not written here and
    ! must not be read; intent(inout) so that padding is left untouched rather
    ! than becoming undefined. Consumed downstream by timestep_diffusion, so
    ! unlike tau/q it keeps its full-volume write.
    real, intent(inout) :: mu_turb(ni, nj, nk)
    ! HALO SOURCE, and the only one: six surface buffers, layer 2 of each
    ! holding the halo value with (2*wall-1) already applied by
    ! set_tau_q_faces and the periodic part already overwritten by
    ! exchange_faces. The kernel neither writes them nor reads layer 1.
    real, intent(in) :: f_i1(nj-1, 9, nk-1, 2), f_ini(nj-1, 9, nk-1, 2)
    real, intent(in) :: f_j1(ni-1, 9, nk-1, 2), f_jnj(ni-1, 9, nk-1, 2)
    real, intent(in) :: f_k1(ni-1, 9, nj-1, 2), f_knk(ni-1, 9, nj-1, 2)
    real, intent(inout) :: planes(ni, nj, 4, 2)
    real, intent(inout) :: rows(ni, 4, 3)
    ! Rolling tau/q CELL-plane pair, halo-indexed in i and j -- owned cell
    ! (i,j) at (i+1,j+1), the shell in index 1 and ni+1/nj+1 -- with slots 1-6
    ! tau and 7-9 q. Slot ta holds cell plane k-1, slot tb cell plane k. Carved
    ! from the scratch arena by the caller (ember.block._carve_viscous), never
    ! allocated here. Dimensioned over the FULL j extent though the walk only
    ! ever touches the current panel's rows: what has to fit in cache is the
    ! rows actually touched, and indexing by global j keeps the body's indices
    ! the same as the volume scheme's -- at 1.3 MB against the 36 MB volume
    ! this replaces, sizing it to the panel would buy nothing.
    real, intent(inout) :: tq(ni+1, nj+1, 9, 2)
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

    integer :: i, j, k, c, jc, kc
    logical :: k_interior, row_interior
    ! Cusp seam correction: the two seam face flows and the half-difference
    ! they contribute to both seam cells.
    real :: flow1(4), flownk(4), fcorr(4)
    real :: tq1lo(9), tq1hi(9), tqnlo(9), tqnhi(9)
    integer :: sa, sb, pa, pb, stmp
    integer :: jp, jp0, jp1, jprod0, jprod1, jbw
    real :: tauf(6), qf(3), Vf(3), rf
    real :: wvisc(3), Vabs, wf(4), wfac
    integer :: ta, tb
    ! Row temps for the tau/q stage, AUTOMATIC, and deliberately so: a
    ! caller-preallocated buffer was tried and cost the stage-1 loop its
    ! vectorization, because GCC versions that loop with a runtime alias check
    ! (opt-report: "loop versioned for vectorization because of possible
    ! aliasing") and will not do so against a dummy argument. set_tau_q_faces
    ! declares its own copies the same way and for the same reason.
    real :: gVx(ni-1, 3), gVr(ni-1, 3), gVt(ni-1, 3)
    real :: vct(ni-1), rcr(ni-1), ivr(ni-1), rhoc(ni-1)
    real :: cpc(ni-1), muc(ni-1), kac(ni-1)
    real :: visc_lim, lambda
    ! Scalars for the hand-inlined polar source (see the note at its first
    ! use): GCC inlines polar_src into production's set_visc_force but not
    ! into this larger fused body, and a call in the loop blocks
    ! vectorization outright.
    real :: prhoc, prhorVtc, prc, pPc, pVtc
    real :: f1, f2, f3, f4, f5, f6, g1, g2, g3
    real :: t1, t2, t3, t4, t5, t6, w1, w2, w3, vm, mut, fac, wsum
    ! Per-component corner velocities for the stage-1 gathers; ga..gd are the
    ! corner reciprocals of the four-corner face averages below.
    real :: v1, v2, v3, v4, v5, v6, v7, v8
    real :: ga, gb, gc, gd

    ! There is no k-slab depth here: the fused schedule subsumes k-slab
    ! blocking, because each tau/q cell plane is consumed by all three face
    ! directions at the same moment it is produced, so a single walk over k IS
    ! the blocked schedule. The blocking that remains is the j panel below.
    ! A degenerate extent is refused rather than silently walked.
    if (ni < 2 .or. nj < 2 .or. nk < 2) return
    if (i_cusp_start < 0 .or. i_cusp_end < 0) return

    ! ===== j-panel over the k walk =====
    ! Without this the walk carries a whole tau/q cell-plane pair plus the
    ! k-face flow planes from one k step to the next: ~1.9 MB at a 273x65x57
    ! block, which one rank holds in a 20 MB L3 and eight ranks do not -- 15 MB
    ! of carry, against the nodal fields both halves of the walk re-read every
    ! plane. That is the failure mode of a fusion like this one, and it is
    ! severe: unpanelled, this kernel is 8% FASTER serially and 40% SLOWER at
    ! 8-rank socket contention (bench/README.md). Panelling divides the carry
    ! by nj-1 / jbw and leaves each cell visited once, so it buys the contended
    ! case at the serial case's expense -- take that trade, production runs
    ! contended. set_residual carries the same panel for the same reason.
    !
    ! The panel costs TWO duplicated cell rows per panel per k plane: the
    ! j-face row at jp0, which the previous panel also computed, and the tau/q
    ! producer rows at jp0-1 and jp1+1, which its lowest and highest j faces
    ! average against. That second one is the price of having no tau/q volume
    ! to read those rows out of -- 2/jbw of the producer, and cheap at the
    ! panel widths VISC_JAREA gives.
    if (jbw_in > 0) then
        jbw = min(nj-1, jbw_in)
    else
        jbw = min(nj-1, max(4, VISC_JAREA / max(ni, 1)))
    end if
    do jp = 1, nj-1, jbw
    jp0 = jp
    jp1 = min(jp + jbw - 1, nj-1)
    ! Cell rows this panel must produce: its own, plus the row below and the
    ! row above, which its lowest and highest j faces average against.
    jprod0 = max(1, jp0 - 1)
    jprod1 = min(nj-1, jp1 + 1)

    pa = 1
    pb = 2
    ta = 1
    tb = 2

    do k = 1, nk
    ! ===== PRODUCE tau/q for cell plane k into slot tb =====
    ! The per-(j,k) row body, storing into the rolling pair: the whole point of
    ! the fusion is that cell plane k is consumed by the k-face flux below and
    ! by the i/j scan on the next iteration, and by nothing else, so it never
    ! needs to reach memory. set_tau_q_faces runs the same body for the shell.
    if (k == 1) then
        call load_kface(f_k1, tq(1,1,1,ta), jp0, jp1, ni, nj)
    end if
    if (k <= nk-1) then
        do j = jprod0, jprod1
        ! Stage 1: velocity gradients + cell metrics, vectorizable over i.
        do i = 1, ni-1
            ivr(i) = 0.25e0 / vol(i,j,k)
            rcr(i) = 0.125e0 * (r(i,j,k)   + r(i+1,j,k)   + r(i,j+1,k)   + r(i+1,j+1,k) &
                              + r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
            rhoc(i) = 0.125e0 * (cons(i,j,k,1)   + cons(i+1,j,k,1)   + cons(i,j+1,k,1)   + cons(i+1,j+1,k,1) &
                               + cons(i,j,k+1,1) + cons(i+1,j,k+1,1) + cons(i,j+1,k+1,1) + cons(i+1,j+1,k+1,1))
            cpc(i) = 0.125e0 * (cp(i,j,k)   + cp(i+1,j,k)   + cp(i,j+1,k)   + cp(i+1,j+1,k) &
                              + cp(i,j,k+1) + cp(i+1,j,k+1) + cp(i,j+1,k+1) + cp(i+1,j+1,k+1))
            muc(i) = 0.125e0 * (mu(i,j,k)   + mu(i+1,j,k)   + mu(i,j+1,k)   + mu(i+1,j+1,k) &
                              + mu(i,j,k+1) + mu(i+1,j,k+1) + mu(i,j+1,k+1) + mu(i+1,j+1,k+1))
            kac(i) = 0.125e0 * (kappa(i,j,k)   + kappa(i+1,j,k)   + kappa(i,j+1,k)   + kappa(i+1,j+1,k) &
                              + kappa(i,j,k+1) + kappa(i+1,j,k+1) + kappa(i,j+1,k+1) + kappa(i+1,j+1,k+1))
            ! --- Vx ---
            ! Velocity from the conserved state: vel_at inlined rather than
            ! called, because a call here costs this loop its vectorisation.
            ! Divided per component, not through a shared corner reciprocal:
            ! see the rule above vel_at, which this must keep matching.
            v1 = cons(i,j,k,2)/cons(i,j,k,1)
            v2 = cons(i+1,j,k,2)/cons(i+1,j,k,1)
            v3 = cons(i,j+1,k,2)/cons(i,j+1,k,1)
            v4 = cons(i+1,j+1,k,2)/cons(i+1,j+1,k,1)
            v5 = cons(i,j,k+1,2)/cons(i,j,k+1,1)
            v6 = cons(i+1,j,k+1,2)/cons(i+1,j,k+1,1)
            v7 = cons(i,j+1,k+1,2)/cons(i,j+1,k+1,1)
            v8 = cons(i+1,j+1,k+1,2)/cons(i+1,j+1,k+1,1)
            f1 = v1+v3+v5+v7
            f2 = v2+v4+v6+v8
            f3 = v1+v2+v5+v6
            f4 = v3+v4+v7+v8
            f5 = v1+v2+v3+v4
            f6 = v5+v6+v7+v8
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVx(i,1) = g1*ivr(i)
            gVx(i,3) = g3*ivr(i)
            gVx(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vr ---
            v1 = cons(i,j,k,3)/cons(i,j,k,1)
            v2 = cons(i+1,j,k,3)/cons(i+1,j,k,1)
            v3 = cons(i,j+1,k,3)/cons(i,j+1,k,1)
            v4 = cons(i+1,j+1,k,3)/cons(i+1,j+1,k,1)
            v5 = cons(i,j,k+1,3)/cons(i,j,k+1,1)
            v6 = cons(i+1,j,k+1,3)/cons(i+1,j,k+1,1)
            v7 = cons(i,j+1,k+1,3)/cons(i,j+1,k+1,1)
            v8 = cons(i+1,j+1,k+1,3)/cons(i+1,j+1,k+1,1)
            f1 = v1+v3+v5+v7
            f2 = v2+v4+v6+v8
            f3 = v1+v2+v5+v6
            f4 = v3+v4+v7+v8
            f5 = v1+v2+v3+v4
            f6 = v5+v6+v7+v8
            g1 = -(f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k)-f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1))
            g2 = -(f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k)-f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))
            g3 = -(f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k)-f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1))
            gVr(i,1) = g1*ivr(i)
            gVr(i,3) = g3*ivr(i)
            gVr(i,2) = g2*ivr(i) - 0.125e0*(f1+f2)/rcr(i)
            ! --- Vt ---
            v1 = cons(i,j,k,4)/cons(i,j,k,1)/r(i,j,k) - Omega_block*r(i,j,k)
            v2 = cons(i+1,j,k,4)/cons(i+1,j,k,1)/r(i+1,j,k) - Omega_block*r(i+1,j,k)
            v3 = cons(i,j+1,k,4)/cons(i,j+1,k,1)/r(i,j+1,k) - Omega_block*r(i,j+1,k)
            v4 = cons(i+1,j+1,k,4)/cons(i+1,j+1,k,1)/r(i+1,j+1,k) - Omega_block*r(i+1,j+1,k)
            v5 = cons(i,j,k+1,4)/cons(i,j,k+1,1)/r(i,j,k+1) - Omega_block*r(i,j,k+1)
            v6 = cons(i+1,j,k+1,4)/cons(i+1,j,k+1,1)/r(i+1,j,k+1) - Omega_block*r(i+1,j,k+1)
            v7 = cons(i,j+1,k+1,4)/cons(i,j+1,k+1,1)/r(i,j+1,k+1) - Omega_block*r(i,j+1,k+1)
            v8 = cons(i+1,j+1,k+1,4)/cons(i+1,j+1,k+1,1)/r(i+1,j+1,k+1) - Omega_block*r(i+1,j+1,k+1)
            f1 = v1+v3+v5+v7
            f2 = v2+v4+v6+v8
            f3 = v1+v2+v5+v6
            f4 = v3+v4+v7+v8
            f5 = v1+v2+v3+v4
            f6 = v5+v6+v7+v8
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
            wsum = wdist(i,j,k) + wdist(i+1,j,k) + wdist(i,j+1,k) + wdist(i+1,j+1,k) + &
                   wdist(i,j,k+1) + wdist(i+1,j,k+1) + wdist(i,j+1,k+1) + wdist(i+1,j+1,k+1)
            visc_lim = 3000e0 * muc(i)
            mut = min(rhoc(i) * (XLEN_FAC * wsum * wsum) * vm, visc_lim)
            mu_turb(i,j,k) = mut
            fac = (muc(i) + mut) * 0.5e0
            tq(i+1,j+1,1,tb) = t1*fac
            tq(i+1,j+1,2,tb) = t2*fac
            tq(i+1,j+1,3,tb) = t3*fac
            tq(i+1,j+1,4,tb) = t4*fac
            tq(i+1,j+1,5,tb) = t5*fac
            tq(i+1,j+1,6,tb) = t6*fac
            lambda = kac(i) + mut * cpc(i) / Pr_turb
            f1 = T(i,j,k)+T(i,j+1,k)+T(i,j,k+1)+T(i,j+1,k+1)
            f2 = T(i+1,j,k)+T(i+1,j+1,k)+T(i+1,j,k+1)+T(i+1,j+1,k+1)
            f3 = T(i,j,k)+T(i+1,j,k)+T(i,j,k+1)+T(i+1,j,k+1)
            f4 = T(i,j+1,k)+T(i+1,j+1,k)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            f5 = T(i,j,k)+T(i+1,j,k)+T(i,j+1,k)+T(i+1,j+1,k)
            f6 = T(i,j,k+1)+T(i+1,j,k+1)+T(i,j+1,k+1)+T(i+1,j+1,k+1)
            tq(i+1,j+1,7,tb) = (f1*dAi(1,i,j,k)-f2*dAi(1,i+1,j,k)+f3*dAj(1,i,j,k) &
                  -f4*dAj(1,i,j+1,k)+f5*dAk(1,i,j,k)-f6*dAk(1,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            tq(i+1,j+1,9,tb) = (f1*dAi(3,i,j,k)-f2*dAi(3,i+1,j,k)+f3*dAj(3,i,j,k) &
                  -f4*dAj(3,i,j+1,k)+f5*dAk(3,i,j,k)-f6*dAk(3,i,j,k+1)) * (ivr(i)*lambda*0.5e0)
            tq(i+1,j+1,8,tb) = ((f1*dAi(2,i,j,k)-f2*dAi(2,i+1,j,k)+f3*dAj(2,i,j,k) &
                  -f4*dAj(2,i,j+1,k)+f5*dAk(2,i,j,k)-f6*dAk(2,i,j,k+1))*ivr(i) &
                  + 0.125e0*(f1+f2)/rcr(i)) * (lambda*0.5e0)
        end do
        end do
        ! i/j halo edges of this plane, straight out of their face buffers.
        ! No scaling on the way in and no mask argument: set_tau_q_faces
        ! applied (2*wall-1) once, when it wrote layer 2.
        call load_ijedge_faces(f_i1, f_ini, f_j1, f_jnj, &
            tq(1,1,1,tb), k, jprod0, jprod1, ni, nj, nk)
    else
        call load_kface(f_knk, tq(1,1,1,tb), jp0, jp1, ni, nj)
    end if

    ! --- k-face plane k into the rolling pair ---
    do j = jp0, jp1
    do i = 1, ni-1
        tauf(1) = (tq(i+1, j+1, 1, ta) + tq(i+1, j+1, 1, tb)) * 0.5e0
        tauf(2) = (tq(i+1, j+1, 2, ta) + tq(i+1, j+1, 2, tb)) * 0.5e0
        tauf(3) = (tq(i+1, j+1, 3, ta) + tq(i+1, j+1, 3, tb)) * 0.5e0
        tauf(4) = (tq(i+1, j+1, 4, ta) + tq(i+1, j+1, 4, tb)) * 0.5e0
        tauf(5) = (tq(i+1, j+1, 5, ta) + tq(i+1, j+1, 5, tb)) * 0.5e0
        tauf(6) = (tq(i+1, j+1, 6, ta) + tq(i+1, j+1, 6, tb)) * 0.5e0
        qf(1)   = (tq(i+1, j+1, 7, ta) + tq(i+1, j+1, 7, tb)) * 0.5e0
        qf(2)   = (tq(i+1, j+1, 8, ta) + tq(i+1, j+1, 8, tb)) * 0.5e0
        qf(3)   = (tq(i+1, j+1, 9, ta) + tq(i+1, j+1, 9, tb)) * 0.5e0
        ! Face velocity from cons; vel_at inlined (see stage 1).
        ga = 1.0e0/cons(i,j,k,1)
        gb = 1.0e0/cons(i+1,j,k,1)
        gc = 1.0e0/cons(i,j+1,k,1)
        gd = 1.0e0/cons(i+1,j+1,k,1)
        Vf(1) = (cons(i,j,k,2)*ga + cons(i+1,j,k,2)*gb + cons(i,j+1,k,2)*gc + cons(i+1,j+1,k,2)*gd) * 0.25e0
        Vf(2) = (cons(i,j,k,3)*ga + cons(i+1,j,k,3)*gb + cons(i,j+1,k,3)*gc + cons(i+1,j+1,k,3)*gd) * 0.25e0
        Vf(3) = ( &
                   (cons(i,j,k,4)*ga/r(i,j,k) - Omega_block*r(i,j,k)) &
                 + (cons(i+1,j,k,4)*gb/r(i+1,j,k) - Omega_block*r(i+1,j,k)) &
                 + (cons(i,j+1,k,4)*gc/r(i,j+1,k) - Omega_block*r(i,j+1,k)) &
                 + (cons(i+1,j+1,k,4)*gd/r(i+1,j+1,k) - Omega_block*r(i+1,j+1,k))) * 0.25e0
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
        do j = jp0, jp1
        do i = 1, ni-1
            wfac = 1.0e0 - wallk1(i,j)
            call wall_func_kface(cons, r, dAk, vol, Omega_block, Omega_wallk1_nd(i,j), mu, i, j, 1, 1, wf)
            planes(i,j,1,pb) = wallk1(i,j)*planes(i,j,1,pb) + wfac*wf(1)
            planes(i,j,2,pb) = wallk1(i,j)*planes(i,j,2,pb) + wfac*wf(2)
            planes(i,j,3,pb) = wallk1(i,j)*planes(i,j,3,pb) + wfac*wf(3)
            planes(i,j,4,pb) = wallk1(i,j)*planes(i,j,4,pb) + wfac*wf(4)
        end do
        end do
    end if
    if (k == nk-1) then
        do j = jp0, jp1
        do i = 1, ni-1
            wfac = 1.0e0 - wallnk(i,j)
            call wall_func_kface(cons, r, dAk, vol, Omega_block, Omega_wallnk_nd(i,j), mu, i, j, nk, -1, wf)
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
        do j = jp0, jp1+1
            do i = 1, ni-1
                tauf(1) = (tq(i+1, j, 1, ta) + tq(i+1, j+1, 1, ta)) * 0.5e0
                tauf(2) = (tq(i+1, j, 2, ta) + tq(i+1, j+1, 2, ta)) * 0.5e0
                tauf(3) = (tq(i+1, j, 3, ta) + tq(i+1, j+1, 3, ta)) * 0.5e0
                tauf(4) = (tq(i+1, j, 4, ta) + tq(i+1, j+1, 4, ta)) * 0.5e0
                tauf(5) = (tq(i+1, j, 5, ta) + tq(i+1, j+1, 5, ta)) * 0.5e0
                tauf(6) = (tq(i+1, j, 6, ta) + tq(i+1, j+1, 6, ta)) * 0.5e0
                qf(1)   = (tq(i+1, j, 7, ta) + tq(i+1, j+1, 7, ta)) * 0.5e0
                qf(2)   = (tq(i+1, j, 8, ta) + tq(i+1, j+1, 8, ta)) * 0.5e0
                qf(3)   = (tq(i+1, j, 9, ta) + tq(i+1, j+1, 9, ta)) * 0.5e0
                ! Face velocity from cons; vel_at inlined (see stage 1).
                ga = 1.0e0/cons(i,j,kc,1)
                gb = 1.0e0/cons(i+1,j,kc,1)
                gc = 1.0e0/cons(i,j,kc+1,1)
                gd = 1.0e0/cons(i+1,j,kc+1,1)
                Vf(1) = (cons(i,j,kc,2)*ga + cons(i+1,j,kc,2)*gb + cons(i,j,kc+1,2)*gc + cons(i+1,j,kc+1,2)*gd) * 0.25e0
                Vf(2) = (cons(i,j,kc,3)*ga + cons(i+1,j,kc,3)*gb + cons(i,j,kc+1,3)*gc + cons(i+1,j,kc+1,3)*gd) * 0.25e0
                Vf(3) = ( &
                           (cons(i,j,kc,4)*ga/r(i,j,kc) - Omega_block*r(i,j,kc)) &
                         + (cons(i+1,j,kc,4)*gb/r(i+1,j,kc) - Omega_block*r(i+1,j,kc)) &
                         + (cons(i,j,kc+1,4)*gc/r(i,j,kc+1) - Omega_block*r(i,j,kc+1)) &
                         + (cons(i+1,j,kc+1,4)*gd/r(i+1,j,kc+1) - Omega_block*r(i+1,j,kc+1))) * 0.25e0
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
                    call wall_func_jface(cons, r, dAj, vol, Omega_block, Omega_wallj1_nd(i,kc), &
                        mu, i, 1, kc, 1, wf)
                    rows(i,1,sb) = wallj1(i,kc)*rows(i,1,sb) + wfac*wf(1)
                    rows(i,2,sb) = wallj1(i,kc)*rows(i,2,sb) + wfac*wf(2)
                    rows(i,3,sb) = wallj1(i,kc)*rows(i,3,sb) + wfac*wf(3)
                    rows(i,4,sb) = wallj1(i,kc)*rows(i,4,sb) + wfac*wf(4)
                end do
            end if
            if (j == nj-1) then
                do i = 1, ni-1
                    wfac = 1.0e0 - wallnj(i,kc)
                    call wall_func_jface(cons, r, dAj, vol, Omega_block, Omega_wallnj_nd(i,kc), &
                        mu, i, nj, kc, -1, wf)
                    rows(i,1,sb) = wallnj(i,kc)*rows(i,1,sb) + wfac*wf(1)
                    rows(i,2,sb) = wallnj(i,kc)*rows(i,2,sb) + wfac*wf(2)
                    rows(i,3,sb) = wallnj(i,kc)*rows(i,3,sb) + wfac*wf(3)
                    rows(i,4,sb) = wallnj(i,kc)*rows(i,4,sb) + wfac*wf(4)
                end do
            end if
            if (j > jp0) then
                jc = j - 1
                row_interior = k_interior .and. (jc >= 2 .and. jc <= nj-2)
                do i = 1, ni
                    tauf(1) = (tq(i, jc+1, 1, ta) + tq(i+1, jc+1, 1, ta)) * 0.5e0
                    tauf(2) = (tq(i, jc+1, 2, ta) + tq(i+1, jc+1, 2, ta)) * 0.5e0
                    tauf(3) = (tq(i, jc+1, 3, ta) + tq(i+1, jc+1, 3, ta)) * 0.5e0
                    tauf(4) = (tq(i, jc+1, 4, ta) + tq(i+1, jc+1, 4, ta)) * 0.5e0
                    tauf(5) = (tq(i, jc+1, 5, ta) + tq(i+1, jc+1, 5, ta)) * 0.5e0
                    tauf(6) = (tq(i, jc+1, 6, ta) + tq(i+1, jc+1, 6, ta)) * 0.5e0
                    qf(1)   = (tq(i, jc+1, 7, ta) + tq(i+1, jc+1, 7, ta)) * 0.5e0
                    qf(2)   = (tq(i, jc+1, 8, ta) + tq(i+1, jc+1, 8, ta)) * 0.5e0
                    qf(3)   = (tq(i, jc+1, 9, ta) + tq(i+1, jc+1, 9, ta)) * 0.5e0
                    ! Face velocity from cons; vel_at inlined (see stage 1).
                    ga = 1.0e0/cons(i,jc,kc,1)
                    gb = 1.0e0/cons(i,jc+1,kc,1)
                    gc = 1.0e0/cons(i,jc,kc+1,1)
                    gd = 1.0e0/cons(i,jc+1,kc+1,1)
                    Vf(1) = (cons(i,jc,kc,2)*ga + cons(i,jc+1,kc,2)*gb + cons(i,jc,kc+1,2)*gc + cons(i,jc+1,kc+1,2)*gd) * 0.25e0
                    Vf(2) = (cons(i,jc,kc,3)*ga + cons(i,jc+1,kc,3)*gb + cons(i,jc,kc+1,3)*gc + cons(i,jc+1,kc+1,3)*gd) * 0.25e0
                    Vf(3) = ( &
                               (cons(i,jc,kc,4)*ga/r(i,jc,kc) - Omega_block*r(i,jc,kc)) &
                             + (cons(i,jc+1,kc,4)*gb/r(i,jc+1,kc) - Omega_block*r(i,jc+1,kc)) &
                             + (cons(i,jc,kc+1,4)*gc/r(i,jc,kc+1) - Omega_block*r(i,jc,kc+1)) &
                             + (cons(i,jc+1,kc+1,4)*gd/r(i,jc+1,kc+1) - Omega_block*r(i,jc+1,kc+1))) * 0.25e0
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
                call wall_func_iface(cons, r, dAi, vol, Omega_block, Omega_walli1_nd(jc,kc), &
                    mu, 1, jc, kc, 1, wf)
                rows(2,1,1) = walli1(jc,kc)*rows(2,1,1) + wfac*wf(1)
                rows(2,2,1) = walli1(jc,kc)*rows(2,2,1) + wfac*wf(2)
                rows(2,3,1) = walli1(jc,kc)*rows(2,3,1) + wfac*wf(3)
                rows(2,4,1) = walli1(jc,kc)*rows(2,4,1) + wfac*wf(4)
                wfac = 1.0e0 - wallni(jc,kc)
                call wall_func_iface(cons, r, dAi, vol, Omega_block, Omega_wallni_nd(jc,kc), &
                    mu, ni, jc, kc, -1, wf)
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
                        prhoc = 0.125e0 * ( &
                            cons(i,jc,kc,1) + cons(i+1,jc,kc,1) + cons(i,jc+1,kc,1) + cons(i+1,jc+1,kc,1) + &
                            cons(i,jc,kc+1,1) + cons(i+1,jc,kc+1,1) + cons(i,jc+1,kc+1,1) + cons(i+1,jc+1,kc+1,1))
                        prhorVtc = 0.125e0 * ( &
                            cons(i,jc,kc,4) + cons(i+1,jc,kc,4) + cons(i,jc+1,kc,4) + cons(i+1,jc+1,kc,4) + &
                            cons(i,jc,kc+1,4) + cons(i+1,jc,kc+1,4) + cons(i,jc+1,kc+1,4) + cons(i+1,jc+1,kc+1,4))
                        prc = 0.125e0 * ( &
                            r(i,jc,kc) + r(i+1,jc,kc) + r(i,jc+1,kc) + r(i+1,jc+1,kc) + &
                            r(i,jc,kc+1) + r(i+1,jc,kc+1) + r(i,jc+1,kc+1) + r(i+1,jc+1,kc+1))
                        pPc = 0.125e0 * ( &
                            P(i,jc,kc) + P(i+1,jc,kc) + P(i,jc+1,kc) + P(i+1,jc+1,kc) + &
                            P(i,jc,kc+1) + P(i+1,jc,kc+1) + P(i,jc+1,kc+1) + P(i+1,jc+1,kc+1))
                        pVtc = prhorVtc / (prhoc * prc)
                        fvisc(i,jc,kc,2) = fvisc(i,jc,kc,2) &
                            + vol(i,jc,kc) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
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
    stmp = ta
    ta = tb
    tb = stmp
    end do
    end do

    ! ===== Cusp seam correction, O(surface) =====
    ! The k=1 face is coupled to k=nk over the cusp i-interval, which is
    ! non-local in k and so cannot ride inside the walk. Production handles it
    ! the same way and in the same place -- after the walk, before the wall
    ! zeroing -- by replacing each seam cell's one-sided face flow with the
    ! mean of the two: adding 0.5*(flow(1) - flow(nk)) to BOTH seam cells is
    ! that replacement, since avg - flow(1) = -0.5*(flow(1) - flow(nk)) at the
    ! low cell and avg - flow(nk) = the same delta at the high cell, once the
    ! accumulate's sign convention is taken into account. The sign is what
    ! tests/test_viscous_cusp_seam.py exists to pin: an earlier flip left the
    ! seam cells taking +2*fcorr of spurious anti-diffusive force every step.
    !
    ! The two raw seam face flows are recomputed rather than stashed: f_k1 and
    ! f_knk are intent(in) and untouched by the walk, and neither seam plane
    ! takes a wall-function injection, so the recompute reproduces the values
    ! the walk used. (nk=2, where the two seam cells coincide, is not
    ! supported, exactly as in production.)
    if (i_cusp_start > 0 .and. nk > 2) then
        do j = 1, nj-1
        do i = i_cusp_start, i_cusp_end-1
            do c = 1, 9
                ! k=1 face: low side is the halo (layer 2), high side is cell
                ! plane 1 (layer 1). k=nk face: low side is cell plane nk-1
                ! (layer 1), high side its halo (layer 2).
                tq1lo(c) = f_k1(i,c,j,2)
                tq1hi(c) = f_k1(i,c,j,1)
                tqnlo(c) = f_knk(i,c,j,1)
                tqnhi(c) = f_knk(i,c,j,2)
            end do
            call kface_flow_tq(tq1lo, tq1hi, cons, r, dAk, Omega_block, i, j, 1, flow1)
            call kface_flow_tq(tqnlo, tqnhi, cons, r, dAk, Omega_block, i, j, nk, flownk)
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
        prhoc = 0.125e0 * ( &
            cons(i,j,1,1) + cons(i+1,j,1,1) + cons(i,j+1,1,1) + cons(i+1,j+1,1,1) + &
            cons(i,j,1+1,1) + cons(i+1,j,1+1,1) + cons(i,j+1,1+1,1) + cons(i+1,j+1,1+1,1))
        prhorVtc = 0.125e0 * ( &
            cons(i,j,1,4) + cons(i+1,j,1,4) + cons(i,j+1,1,4) + cons(i+1,j+1,1,4) + &
            cons(i,j,1+1,4) + cons(i+1,j,1+1,4) + cons(i,j+1,1+1,4) + cons(i+1,j+1,1+1,4))
        prc = 0.125e0 * ( &
            r(i,j,1) + r(i+1,j,1) + r(i,j+1,1) + r(i+1,j+1,1) + &
            r(i,j,1+1) + r(i+1,j,1+1) + r(i,j+1,1+1) + r(i+1,j+1,1+1))
        pPc = 0.125e0 * ( &
            P(i,j,1) + P(i+1,j,1) + P(i,j+1,1) + P(i+1,j+1,1) + &
            P(i,j,1+1) + P(i+1,j,1+1) + P(i,j+1,1+1) + P(i+1,j+1,1+1))
        pVtc = prhorVtc / (prhoc * prc)
        fvisc(i,j,1,2) = fvisc(i,j,1,2) &
            + vol(i,j,1) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
    end do
    end do
    if (nk-1 > 1) then
        do j = 1, nj-1
        do i = 1, ni-1
            prhoc = 0.125e0 * ( &
                cons(i,j,nk-1,1) + cons(i+1,j,nk-1,1) + cons(i,j+1,nk-1,1) + cons(i+1,j+1,nk-1,1) + &
                cons(i,j,nk-1+1,1) + cons(i+1,j,nk-1+1,1) + cons(i,j+1,nk-1+1,1) + cons(i+1,j+1,nk-1+1,1))
            prhorVtc = 0.125e0 * ( &
                cons(i,j,nk-1,4) + cons(i+1,j,nk-1,4) + cons(i,j+1,nk-1,4) + cons(i+1,j+1,nk-1,4) + &
                cons(i,j,nk-1+1,4) + cons(i+1,j,nk-1+1,4) + cons(i,j+1,nk-1+1,4) + cons(i+1,j+1,nk-1+1,4))
            prc = 0.125e0 * ( &
                r(i,j,nk-1) + r(i+1,j,nk-1) + r(i,j+1,nk-1) + r(i+1,j+1,nk-1) + &
                r(i,j,nk-1+1) + r(i+1,j,nk-1+1) + r(i,j+1,nk-1+1) + r(i+1,j+1,nk-1+1))
            pPc = 0.125e0 * ( &
                P(i,j,nk-1) + P(i+1,j,nk-1) + P(i,j+1,nk-1) + P(i+1,j+1,nk-1) + &
                P(i,j,nk-1+1) + P(i+1,j,nk-1+1) + P(i,j+1,nk-1+1) + P(i+1,j+1,nk-1+1))
            pVtc = prhorVtc / (prhoc * prc)
            fvisc(i,j,nk-1,2) = fvisc(i,j,nk-1,2) &
                + vol(i,j,nk-1) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
        end do
        end do
    end if
    do k = 2, nk-2
    do i = 1, ni-1
        prhoc = 0.125e0 * ( &
            cons(i,1,k,1) + cons(i+1,1,k,1) + cons(i,1+1,k,1) + cons(i+1,1+1,k,1) + &
            cons(i,1,k+1,1) + cons(i+1,1,k+1,1) + cons(i,1+1,k+1,1) + cons(i+1,1+1,k+1,1))
        prhorVtc = 0.125e0 * ( &
            cons(i,1,k,4) + cons(i+1,1,k,4) + cons(i,1+1,k,4) + cons(i+1,1+1,k,4) + &
            cons(i,1,k+1,4) + cons(i+1,1,k+1,4) + cons(i,1+1,k+1,4) + cons(i+1,1+1,k+1,4))
        prc = 0.125e0 * ( &
            r(i,1,k) + r(i+1,1,k) + r(i,1+1,k) + r(i+1,1+1,k) + &
            r(i,1,k+1) + r(i+1,1,k+1) + r(i,1+1,k+1) + r(i+1,1+1,k+1))
        pPc = 0.125e0 * ( &
            P(i,1,k) + P(i+1,1,k) + P(i,1+1,k) + P(i+1,1+1,k) + &
            P(i,1,k+1) + P(i+1,1,k+1) + P(i,1+1,k+1) + P(i+1,1+1,k+1))
        pVtc = prhorVtc / (prhoc * prc)
        fvisc(i,1,k,2) = fvisc(i,1,k,2) &
            + vol(i,1,k) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
    end do
    end do
    if (nj-1 > 1) then
        do k = 2, nk-2
        do i = 1, ni-1
            prhoc = 0.125e0 * ( &
                cons(i,nj-1,k,1) + cons(i+1,nj-1,k,1) + cons(i,nj-1+1,k,1) + cons(i+1,nj-1+1,k,1) + &
                cons(i,nj-1,k+1,1) + cons(i+1,nj-1,k+1,1) + cons(i,nj-1+1,k+1,1) + cons(i+1,nj-1+1,k+1,1))
            prhorVtc = 0.125e0 * ( &
                cons(i,nj-1,k,4) + cons(i+1,nj-1,k,4) + cons(i,nj-1+1,k,4) + cons(i+1,nj-1+1,k,4) + &
                cons(i,nj-1,k+1,4) + cons(i+1,nj-1,k+1,4) + cons(i,nj-1+1,k+1,4) + cons(i+1,nj-1+1,k+1,4))
            prc = 0.125e0 * ( &
                r(i,nj-1,k) + r(i+1,nj-1,k) + r(i,nj-1+1,k) + r(i+1,nj-1+1,k) + &
                r(i,nj-1,k+1) + r(i+1,nj-1,k+1) + r(i,nj-1+1,k+1) + r(i+1,nj-1+1,k+1))
            pPc = 0.125e0 * ( &
                P(i,nj-1,k) + P(i+1,nj-1,k) + P(i,nj-1+1,k) + P(i+1,nj-1+1,k) + &
                P(i,nj-1,k+1) + P(i+1,nj-1,k+1) + P(i,nj-1+1,k+1) + P(i+1,nj-1+1,k+1))
            pVtc = prhorVtc / (prhoc * prc)
            fvisc(i,nj-1,k,2) = fvisc(i,nj-1,k,2) &
                + vol(i,nj-1,k) * (((pPc - P_offset) + prhoc * pVtc**2) / prc)
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
    real, intent(in) :: Omega_block
    real, intent(in) :: mu(ni, nj, nk)
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
            call wall_yplus_kface(cons, r, dAk, vol, Omega_block, Omega_wallk1_nd(i,j), mu, &
                                   i, j, 1, 1, yp)
            yplus_k1(i,j) = yp
        end if
        if (wallnk(i,j) == 0.0e0) then
            call wall_yplus_kface(cons, r, dAk, vol, Omega_block, Omega_wallnk_nd(i,j), mu, &
                                   i, j, nk, -1, yp)
            yplus_nk(i,j) = yp
        end if
    end do
    end do

    do k = 1, nk-1
    do i = 1, ni-1
        if (wallj1(i,k) == 0.0e0) then
            call wall_yplus_jface(cons, r, dAj, vol, Omega_block, Omega_wallj1_nd(i,k), mu, &
                                   i, 1, k, 1, yp)
            yplus_j1(i,k) = yp
        end if
        if (wallnj(i,k) == 0.0e0) then
            call wall_yplus_jface(cons, r, dAj, vol, Omega_block, Omega_wallnj_nd(i,k), mu, &
                                   i, nj, k, -1, yp)
            yplus_nj(i,k) = yp
        end if
    end do
    end do

    do k = 1, nk-1
    do j = 1, nj-1
        if (walli1(j,k) == 0.0e0) then
            call wall_yplus_iface(cons, r, dAi, vol, Omega_block, Omega_walli1_nd(j,k), mu, &
                                   1, j, k, 1, yp)
            yplus_i1(j,k) = yp
        end if
        if (wallni(j,k) == 0.0e0) then
            call wall_yplus_iface(cons, r, dAi, vol, Omega_block, Omega_wallni_nd(j,k), mu, &
                                   ni, j, k, -1, yp)
            yplus_ni(j,k) = yp
        end if
    end do
    end do

end subroutine wall_yplus_field
