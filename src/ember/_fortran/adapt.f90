! Compute unscaled volumetric timestep from directional convective/diffusion
! spectral radii. A variant of the JST/Blazek multidimensional definition:
! Blazek SUMS the directional radii; here we take the MAX so the CFL number
! stays the true 1D Courant limit (see below).
!
! Calculates dt_vol = 1 / max(lam_conv, lam_diff). The convective radius is
!
!   lam_conv = max_d Lambda_d,   Lambda_d = |V_rel . dA_d| + a * ||dA_d||
!
! the convective spectral radius across direction d (dA_d the per-cell average of
! the two opposing face-area vectors, V_rel the relative-frame velocity matching
! the flux convention mf = (rho*Vx, rho*Vr, rho*Vt_rel)). The turbulent-diffusion
! radius shares those same face areas,
!
!   lam_diff = fac_visc * (mu_turb/rho) * max_d ||dA_d||^2 / vol,
!
! so both limits carry the same directional aspect-ratio weighting and a single
! cfl scales them consistently; mu_turb = 0 recovers the pure convective step.
! fac_visc (>= 1) is a user multiplier that tightens the viscous limit so the
! viscous march tolerates the same cfl as the inviscid one, since the viscous
! fluxes shrink the RK stability margin beyond what the bare radius accounts for.
!
! Normalising on the largest single-direction radius max_d Lambda_d (rather than
! the Blazek sum sum_d Lambda_d) makes the CFL number sigma recover the true 1D
! Courant limit -- ~2*sqrt(2) for the 4-stage RK march, ~1/sqrt(3) for scree --
! because with the solver's smoothing the binding mode is the stiffest grid
! direction, not the multidimensional corner mode the sum bounds. It stays
! aspect-ratio-aware (auto-selects the stiffest direction), unlike set_timestep's
! dl_min/((a+V_rel)*vol) which pairs the smallest edge with a single mismatched
! (a+V). Note max is LESS conservative than the sum: it relies on the smoothing
! keeping the corner modes subcritical, so keep the empirical cfl sweep as the
! stability backstop. Same dt_vol units (time/volume) and same rf blend, so it is
! a drop-in replacement in update_timestep.
!
! CELL SOURCES. The same walk also finishes the body force and advances the
! selective-frequency-damping (SFD) filter, because it already holds the cell
! averages both need and is the only place the step's dt_vol exists in a
! register. Per cell, after the dt_vol blend:
!
!   polar     f_body(3) += vol * ((P - P_offset) + rho*Vt^2) / r
!   SFD force f_body(m) += gain_filt * (cons_filt(m) - cons_cell(m)) * vol
!   filter    cons_filt(m) += dt * (cons_cell(m) - cons_filt(m)) / delta_filt
!
! with dt = cfl * dt_vol * vol the local step (the acoustic speed folded into
! dt_vol keeps it finite as V -> 0, so the filter stays stable). The force
! reads the filter BEFORE this step's update, the usual explicit SFD coupling.
!
! Two gates, because the two halves run on different cadences:
!
!   add_sources /= 0  the body force was rebuilt this step (Grid.update_sources
!                     zeroed it and added the viscous part); add the polar
!                     source and, when gain_filt /= 0, the SFD force. The scree
!                     march refreshes sources every fifth step and holds
!                     f_body in between, so it passes 0 on the other four.
!   gain_filt /= 0    advance the filter, EVERY step: its dt is a per-step
!                     increment, and advancing it on the source cadence would
!                     stretch the time constant fivefold.
!
! The gates are tested once per ROW, not per cell: the timestep i loop stays
! the straight-line loop it always was and leaves the cell averages in
! automatic row arrays, and the sources run as short i loops over that row
! while it is still in L1. Branching inside the cell loop instead left gfortran
! unable to vectorize the timestep loop at all -- the conditional stores cannot
! be if-converted, and the body is too big for it to unswitch -- and splitting
! the body into inlined helpers is what ifort's IPO would not vectorize (see
! the note on the walk below). cons_filt carries its own extents so the caller
! can pass a zero-size array when gain_filt == 0 and never allocate the filter
! state; it is not touched then.
!
subroutine set_timestep_sources( &
        dt_vol, &
        a, &
        cons, &
        r, &
        Omega, &
        dAi, &
        dAj, &
        dAk, &
        mu_turb, &
        vol, &
        rf, &
        fac_visc, &
        P, &
        P_offset, &
        f_body, &
        cons_filt, &
        add_sources, &
        gain_filt, &
        cfl, &
        delta_filt, &
        ni, nj, nk, &
        nfi, nfj, nfk &
    )

    implicit none

    ! Array sizes. nfi/nfj/nfk are cons_filt's: the cell extents when the
    ! filter is live, and may be 0 when gain_filt == 0.
    integer, intent(in) :: ni, nj, nk
    integer, intent(in) :: nfi, nfj, nfk

    ! Node-centered flow properties
    real, intent(in) :: a(ni, nj, nk)               ! Acoustic speed
    real, intent(in) :: r(ni, nj, nk)               ! Radial coordinate
    real, intent(in) :: Omega                       ! Angular velocity [rad/s]

    ! Node-centered conserved variables. Averaged to the cell in the walk
    ! below rather than taken from a cell-centred volume: the average is four
    ! fused multiply-adds against a buffer that would otherwise be the largest
    ! derived array the block owns.
    real, intent(in) :: cons(ni, nj, nk, 5)

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

    ! Cell sources (see the header). f_body is only written when
    ! add_sources /= 0, cons_filt only when gain_filt /= 0.
    real, intent(in) :: P(ni, nj, nk)                ! Nodal static pressure
    real, intent(in) :: P_offset
    real, intent(inout) :: f_body(ni-1, nj-1, nk-1, 5)
    real, intent(inout) :: cons_filt(nfi, nfj, nfk, 5)
    integer, intent(in) :: add_sources
    real, intent(in) :: gain_filt, cfl, delta_filt

    ! Local variables
    integer :: i, j, k
    real :: a_cell, rho_cell, rhoVx_cell, rhoVr_cell, rhorVt_cell, r_cell
    real :: Vx, Vr, Vt, U, Vt_rel, rf, dt_vol_new
    real :: Sx, Sr, St, s2_i, s2_j, s2_k, lam_i, lam_j, lam_k
    real :: lam_conv, lam_diff
    real :: P_cell, dt
    logical :: do_src, do_sfd
    ! One row of cell averages, handed from the timestep loop to the source
    ! loops. Automatic, as set_visc_force's row temps are and for the same
    ! reason: a dummy-argument buffer costs the loops their vectorization.
    real :: rho_row(ni-1), rhoVx_row(ni-1), rhoVr_row(ni-1), rhorVt_row(ni-1)
    real :: r_row(ni-1), rhoe_row(ni-1)

    do_src = add_sources /= 0
    do_sfd = gain_filt /= 0.0e0

    ! Loop over cells. Every avg_cell() call is inlined by hand below (not
    ! left as a pure-function call): ifort 2022.1.0's IPO
    ! analysis treats the call as an assumed dependence on the loop
    ! counters themselves after inlining, blocking vectorization of the
    ! whole i/j/k nest (opt-report: "assumed OUTPUT dependence between
    ! J/I ..."); renaming avg_cell's dummy args away from i/j/k did not
    ! change this. gfortran vectorizes the call form fine, so this is
    ! ifort-motivated. That applies to the four conserved averages as much as
    ! to a_cell/r_cell, so they are written out the same way.
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        ! Average nodal properties to cell centers. rhoe (component 5) is
        ! averaged only for the SFD filter, the one reader of it here.
        a_cell = 0.125e0 * ( &
            a(i,j,k) + a(i+1,j,k) + a(i,j+1,k) + a(i+1,j+1,k) + &
            a(i,j,k+1) + a(i+1,j,k+1) + a(i,j+1,k+1) + a(i+1,j+1,k+1))
        rho_cell = 0.125e0 * ( &
            cons(i,j,k,1) + cons(i+1,j,k,1) + cons(i,j+1,k,1) + cons(i+1,j+1,k,1) + &
            cons(i,j,k+1,1) + cons(i+1,j,k+1,1) + cons(i,j+1,k+1,1) + cons(i+1,j+1,k+1,1))
        rhoVx_cell = 0.125e0 * ( &
            cons(i,j,k,2) + cons(i+1,j,k,2) + cons(i,j+1,k,2) + cons(i+1,j+1,k,2) + &
            cons(i,j,k+1,2) + cons(i+1,j,k+1,2) + cons(i,j+1,k+1,2) + cons(i+1,j+1,k+1,2))
        rhoVr_cell = 0.125e0 * ( &
            cons(i,j,k,3) + cons(i+1,j,k,3) + cons(i,j+1,k,3) + cons(i+1,j+1,k,3) + &
            cons(i,j,k+1,3) + cons(i+1,j,k+1,3) + cons(i,j+1,k+1,3) + cons(i+1,j+1,k+1,3))
        rhorVt_cell = 0.125e0 * ( &
            cons(i,j,k,4) + cons(i+1,j,k,4) + cons(i,j+1,k,4) + cons(i+1,j+1,k,4) + &
            cons(i,j,k+1,4) + cons(i+1,j,k+1,4) + cons(i,j+1,k+1,4) + cons(i+1,j+1,k+1,4))
        r_cell = 0.125e0 * ( &
            r(i,j,k) + r(i+1,j,k) + r(i,j+1,k) + r(i+1,j+1,k) + &
            r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))

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

        ! Convective radius = largest single-direction radius (max, not the
        ! Blazek sum) so the CFL number tracks the true 1D Courant limit; the
        ! larger of it and the turbulent-diffusion radius wins (= smaller time
        ! scale). The diffusion radius is likewise the largest single-direction
        ! radius (mu_turb/rho) * max(|S_i|^2,|S_j|^2,|S_k|^2) / vol, built from
        ! the same per-face areas as lam_conv so the aspect-ratio weighting
        ! matches and a single cfl scales both limits consistently (contrast the
        ! old isotropic mu_turb*vol/(rho*dl_min^2), whose length scale did not
        ! track the stiffest face direction). mu_turb stays in the numerator, so
        ! mu_turb = 0 reduces exactly to the convective form with no branch or
        ! division risk.
        lam_conv = max(lam_i, lam_j, lam_k)
        lam_diff = fac_visc * mu_turb(i, j, k) * max(s2_i, s2_j, s2_k) &
                   / (rho_cell * vol(i, j, k))
        dt_vol_new = 1.0e0 / max(lam_conv, lam_diff)
        dt_vol(i, j, k) = rf * dt_vol_new + (1.0e0 - rf) * dt_vol(i, j, k)

        rho_row(i) = rho_cell
        rhoVx_row(i) = rhoVx_cell
        rhoVr_row(i) = rhoVr_cell
        rhorVt_row(i) = rhorVt_cell
        r_row(i) = r_cell
    end do

        ! Polar source then SFD force, in that order, after the viscous part
        ! update_sources left. Polar is expression-for-expression the viscous
        ! kernel's former fused copy, Vt the absolute tangential velocity.
        if (do_src) then
            do i = 1, ni-1
                P_cell = 0.125e0 * ( &
                    P(i,j,k) + P(i+1,j,k) + P(i,j+1,k) + P(i+1,j+1,k) + &
                    P(i,j,k+1) + P(i+1,j,k+1) + P(i,j+1,k+1) + P(i+1,j+1,k+1))
                Vt = rhorVt_row(i) / (rho_row(i) * r_row(i))
                f_body(i, j, k, 3) = f_body(i, j, k, 3) &
                    + vol(i, j, k) * (((P_cell - P_offset) + rho_row(i) * Vt**2) / r_row(i))
            end do
        end if

        if (do_sfd) then
            do i = 1, ni-1
                rhoe_row(i) = 0.125e0 * ( &
                    cons(i,j,k,5) + cons(i+1,j,k,5) + cons(i,j+1,k,5) + cons(i+1,j+1,k,5) + &
                    cons(i,j,k+1,5) + cons(i+1,j,k+1,5) + cons(i,j+1,k+1,5) + cons(i+1,j+1,k+1,5))
            end do
            ! The force reads the filter BEFORE this step's update below.
            if (do_src) then
                do i = 1, ni-1
                    f_body(i, j, k, 1) = f_body(i, j, k, 1) + &
                        gain_filt * (cons_filt(i, j, k, 1) - rho_row(i)) * vol(i, j, k)
                    f_body(i, j, k, 2) = f_body(i, j, k, 2) + &
                        gain_filt * (cons_filt(i, j, k, 2) - rhoVx_row(i)) * vol(i, j, k)
                    f_body(i, j, k, 3) = f_body(i, j, k, 3) + &
                        gain_filt * (cons_filt(i, j, k, 3) - rhoVr_row(i)) * vol(i, j, k)
                    f_body(i, j, k, 4) = f_body(i, j, k, 4) + &
                        gain_filt * (cons_filt(i, j, k, 4) - rhorVt_row(i)) * vol(i, j, k)
                    f_body(i, j, k, 5) = f_body(i, j, k, 5) + &
                        gain_filt * (cons_filt(i, j, k, 5) - rhoe_row(i)) * vol(i, j, k)
                end do
            end if
            ! The filter advances on this step's blended dt_vol, just stored.
            do i = 1, ni-1
                dt = cfl * dt_vol(i, j, k) * vol(i, j, k)
                cons_filt(i, j, k, 1) = cons_filt(i, j, k, 1) + &
                    dt * (rho_row(i) - cons_filt(i, j, k, 1)) / delta_filt
                cons_filt(i, j, k, 2) = cons_filt(i, j, k, 2) + &
                    dt * (rhoVx_row(i) - cons_filt(i, j, k, 2)) / delta_filt
                cons_filt(i, j, k, 3) = cons_filt(i, j, k, 3) + &
                    dt * (rhoVr_row(i) - cons_filt(i, j, k, 3)) / delta_filt
                cons_filt(i, j, k, 4) = cons_filt(i, j, k, 4) + &
                    dt * (rhorVt_row(i) - cons_filt(i, j, k, 4)) / delta_filt
                cons_filt(i, j, k, 5) = cons_filt(i, j, k, 5) + &
                    dt * (rhoe_row(i) - cons_filt(i, j, k, 5)) / delta_filt
            end do
        end if
    end do
    end do

end subroutine set_timestep_sources
