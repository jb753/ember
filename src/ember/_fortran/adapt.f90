! Selective frequency damping body force.
!
! Adds the SFD forcing gain_filt * (cons_filt - cons_cell) * vol onto F_body_nd
! for all five conserved equations, where cons_filt is the temporally
! low-pass-filtered conserved state maintained by adapt_cfl. Called from
! Grid.update_F_body in the pre-step so the force drives the RK integration.
!
subroutine apply_sfd_force( &
        f_body, &
        cons_filt, &
        cons_cell, &
        vol, &
        gain_filt, &
        ni, nj, nk &
    )

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(inout) :: f_body(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: cons_filt(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: cons_cell(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)
    real, intent(in) :: gain_filt

    integer :: i, j, k, eq

    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        do eq = 1, 5
            f_body(i, j, k, eq) = f_body(i, j, k, eq) + &
                gain_filt * (cons_filt(i, j, k, eq) - cons_cell(i, j, k, eq)) * vol(i, j, k)
        end do
    end do
    end do
    end do

end subroutine apply_sfd_force


! Compute unscaled volumetric timestep using the JST/Blazek multidimensional
! definition (alternative to set_timestep, for benchmarking).
!
! Calculates dt_vol = 1 / max(lam_conv, lam_diff). The convective radius is
!
!   lam_conv = sum_d Lambda_d,   Lambda_d = |V_rel . dA_d| + a * ||dA_d||
!
! the convective spectral radius across direction d (dA_d the per-cell average of
! the two opposing face-area vectors, V_rel the relative-frame velocity matching
! the flux convention mf = (rho*Vx, rho*Vr, rho*Vt_rel)). The turbulent-diffusion
! radius shares those same face areas,
!
!   lam_diff = fac_visc * (mu_turb/rho) * sum_d ||dA_d||^2 / vol,
!
! so both limits carry the same directional aspect-ratio weighting and a single
! cfl scales them consistently; mu_turb = 0 recovers the pure convective step.
! fac_visc (>= 1) is a user multiplier that tightens the viscous limit so the
! viscous march tolerates the same cfl as the inviscid one, since the viscous
! fluxes shrink the RK stability margin beyond what the bare radius accounts for.
!
! Contrast with set_timestep's dt_vol = dl_min/((a+V_rel)*vol), which uses only
! the single smallest edge: that makes the stable CFL swing with cell aspect
! ratio, whereas the sum-of-spectral-radii form pins it near 2*sqrt(2) for the
! 4-stage RK scheme regardless of aspect ratio. Same dt_vol units (time/volume)
! and same rf blend, so it is a drop-in replacement in update_timestep.
!
subroutine set_timestep_spectral( &
        dt_vol, &
        a, &
        cons_cell, &
        r, &
        Omega, &
        dAi, &
        dAj, &
        dAk, &
        mu_turb, &
        vol, &
        rf, &
        fac_visc, &
        ni, nj, nk &
    )

    implicit none

    ! Array sizes
    integer, intent(in) :: ni, nj, nk

    ! Node-centered flow properties
    real, intent(in) :: a(ni, nj, nk)               ! Acoustic speed
    real, intent(in) :: r(ni, nj, nk)               ! Radial coordinate
    real, intent(in) :: Omega                       ! Angular velocity [rad/s]

    ! Cell-centered conserved variables
    real, intent(in) :: cons_cell(ni-1, nj-1, nk-1, 5)

    ! Face-area vectors (component on first axis, same layout as the flux kernel)
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)

    ! Cell-centred turbulent viscosity at each cell's low-corner node (i,j,k);
    ! the final node in each axis is padding and is never read here.
    real, intent(in) :: mu_turb(ni, nj, nk)
    real, intent(in) :: vol(ni-1, nj-1, nk-1)       ! Cell volume

    ! Cell-centered unscaled volumetric timestep (output)
    real, intent(inout) :: dt_vol(ni-1, nj-1, nk-1)

    ! Multiplier on the turbulent-diffusion spectral radius. fac_visc = 1 is the
    ! bare directional radius; a larger value tightens the viscous timestep limit
    ! to recover the same stable cfl as the inviscid march (the viscous fluxes
    ! erode the RK stability margin even when lam_diff would not otherwise bind).
    real, intent(in) :: fac_visc

    ! Local variables
    integer :: i, j, k
    real :: a_cell, rho_cell, rhoVx_cell, rhoVr_cell, rhorVt_cell, r_cell
    real :: Vx, Vr, Vt, U, Vt_rel, rf, dt_vol_new
    real :: Sx, Sr, St, s2_i, s2_j, s2_k, lam_i, lam_j, lam_k
    real :: lam_conv, lam_diff

    ! Loop over cells
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        ! Average nodal properties to cell centers (geometric/thermo);
        ! conserved values come from the cell-centered cache.
        a_cell = avg_cell(a, i, j, k)
        rho_cell    = cons_cell(i, j, k, 1)
        rhoVx_cell  = cons_cell(i, j, k, 2)
        rhoVr_cell  = cons_cell(i, j, k, 3)
        rhorVt_cell = cons_cell(i, j, k, 4)
        r_cell = avg_cell(r, i, j, k)

        ! Compute velocities from conserved variables
        Vx = rhoVx_cell / rho_cell
        Vr = rhoVr_cell / rho_cell
        Vt = rhorVt_cell / (rho_cell * r_cell)

        ! Relative-frame tangential velocity (blade speed U = Omega * r)
        U = Omega * r_cell
        Vt_rel = Vt - U

        ! i-direction spectral radius: average the two opposing i-faces.
        ! s2_d = |S_d|^2 is reused below for the directional diffusion radius.
        Sx = 0.5e0 * (dAi(1, i, j, k) + dAi(1, i+1, j, k))
        Sr = 0.5e0 * (dAi(2, i, j, k) + dAi(2, i+1, j, k))
        St = 0.5e0 * (dAi(3, i, j, k) + dAi(3, i+1, j, k))
        s2_i = Sx*Sx + Sr*Sr + St*St
        lam_i = abs(Vx*Sx + Vr*Sr + Vt_rel*St) + a_cell * sqrt(s2_i)

        ! j-direction
        Sx = 0.5e0 * (dAj(1, i, j, k) + dAj(1, i, j+1, k))
        Sr = 0.5e0 * (dAj(2, i, j, k) + dAj(2, i, j+1, k))
        St = 0.5e0 * (dAj(3, i, j, k) + dAj(3, i, j+1, k))
        s2_j = Sx*Sx + Sr*Sr + St*St
        lam_j = abs(Vx*Sx + Vr*Sr + Vt_rel*St) + a_cell * sqrt(s2_j)

        ! k-direction
        Sx = 0.5e0 * (dAk(1, i, j, k) + dAk(1, i, j, k+1))
        Sr = 0.5e0 * (dAk(2, i, j, k) + dAk(2, i, j, k+1))
        St = 0.5e0 * (dAk(3, i, j, k) + dAk(3, i, j, k+1))
        s2_k = Sx*Sx + Sr*Sr + St*St
        lam_k = abs(Vx*Sx + Vr*Sr + Vt_rel*St) + a_cell * sqrt(s2_k)

        ! Convective + turbulent-diffusion spectral radii; larger radius wins
        ! (= smaller time scale). The diffusion radius is the directional sum
        ! (mu_turb/rho) * (|S_i|^2+|S_j|^2+|S_k|^2) / vol, built from the same
        ! per-face areas as lam_conv so the aspect-ratio weighting matches and a
        ! single cfl scales both limits consistently (contrast the old isotropic
        ! mu_turb*vol/(rho*dl_min^2), which kept only the stiffest direction).
        ! mu_turb stays in the numerator, so mu_turb = 0 reduces exactly to the
        ! convective form with no branch or division risk.
        lam_conv = lam_i + lam_j + lam_k
        lam_diff = fac_visc * mu_turb(i, j, k) * (s2_i + s2_j + s2_k) &
                   / (rho_cell * vol(i, j, k))
        dt_vol_new = 1.0e0 / max(lam_conv, lam_diff)
        dt_vol(i, j, k) = rf * dt_vol_new + (1.0e0 - rf) * dt_vol(i, j, k)
    end do
    end do
    end do

contains

    pure function avg_cell(x, i, j, k) result(avg)
        implicit none
        real, intent(in) :: x(ni,nj,nk)
        integer, intent(in) :: i, j, k
        real :: avg
        avg = 0.125e0 * ( &
            x(i,j,k) + x(i+1,j,k) + x(i,j+1,k) + x(i+1,j+1,k) + &
            x(i,j,k+1) + x(i+1,j,k+1) + x(i,j+1,k+1) + x(i+1,j+1,k+1))
    end function avg_cell

end subroutine set_timestep_spectral
