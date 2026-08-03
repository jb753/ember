! =====================================================================
! BENCHMARK ARM `rinv` -- production's kernel with 1/r staged as GEOMETRY.
!
! Nothing in production calls this. See docs/dev/plan_nodal_primitives.md.
!
! Production issues 8 divides per face corner-set: four reciprocals
! 1/cons(...,1), and four cons(...,4)/r(...) inside mf3. The second four
! divide by r, and **r is static grid geometry** -- it changes only on
! adapt, never per step -- so 1/r can be built once and streamed instead
! of recomputed. That is the only change here:
!
!     mf3 = w*(cons(...,4)/r(...)   - Omega*cons(...,1)*r(...))     prod
!     mf3 = w*(cons(...,4)*rinv(...) - Omega*cons(...,1)*r(...))    this
!
! Half the reciprocal work removed for +4 B/cell of STATIC data, against
! `nodal`, which removes all of it for +12 B/cell of solution-dependent
! streams. One third the traffic cost for one half the benefit -- a
! better ratio on exactly the axis that made ifort/Sapphire prefer
! cons-derived velocities in section 20 (-4.5% saturated).
!
! It also leaves section 20 intact: velocities stay cons-derived, so no
! extra nodal streams and no EOS content (section 20.5) -- 1/r is pure
! geometry, unlike P or ho.
!
! *** CAVEAT, MEASURED NOT ASSUMED. *** Under -Ofast -freciprocal-math,
! gfortran compiles production's packed divides to vrcpps + Newton-
! Raphson, NOT vdivps: the prod driver disassembles to 101 vrcpps, 70
! vdivss (the un-vectorized i-face path) and ZERO vdivps. So this arm
! does not remove "a divide"; it removes a reciprocal approximation plus
! its refinement chain, ~4 ops per corner. Section 26.4's ~13 ns/cell of
! "divider pressure" does not describe this binary, and this arm is a
! direct measurement of what those chains really cost. ifort's
! -no-prec-div -fp-model fast=2 almost certainly approximates too.
!
! FAIRNESS (Rule 3): rinv is built once outside the timed region, which
! is what a real port would do -- r comes from the grid coordinates and
! is rebuilt only on adapt, exactly like the face areas the `multall` arm
! is allowed to pre-split.
!
! NUMERICS: cons4*(1/r) is not cons4/r bitwise, so expect agreement at
! the usual last-bit level, and expect it ONLY in the mass and angular-
! momentum terms that mf3 feeds.
! =====================================================================

module residual_rinv_helpers
    implicit none
    private
    public :: iface_flow_row_rv, jface_flow_row_rv, kface_flow_plane_rv

contains
    pure subroutine iface_flow_row_rv(ho, P, P_offset, r, rinv, &
                                   cons, Omega, dA, &
                                   wall_lo, wall_hi, row, j, k, ni, nj, nk)
        ! Compute inviscid face flows on the ni i-faces of cell row (j,k);
        ! the i=1 / i=ni boundary faces are wall-masked by the scalars
        ! wall_lo / wall_hi. i-face corners: (i, j:j+1, k:k+1)

        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: rinv(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: wall_lo, wall_hi
        real, intent(inout) :: row(ni, 5)

        integer :: i
        real :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3, mdot

        ! pm/mf are scalarized (no longer length-6/length-3 arrays) and the
        ! four face corners are unrolled by hand: ifort's vectorizer treats
        ! the pm(:)/mf(:) arrays as a cross-iteration aliasing hazard it
        ! can't disprove, blocking vectorization of the interior loop
        ! (opt-report: "vector dependence: assumed FLOW dependence between
        ! PM(:) and PM(1)"). gfortran vectorizes the original array/call
        ! form fine, so this rewrite is ifort-motivated.
        !
        ! Re-confirmed July 2026 on ifort 2022.1.0 / Xeon Platinum 8480+
        ! (Sapphire Rapids) under the production INTEL_FLAGS, by building an
        ! idiomatic rewrite alongside this kernel and A/B-ing them in one
        ! .so: the array form gets no SIMD at all and costs +174% serial /
        ! +99% under 100-rank saturated bandwidth. Do not "tidy" this away.
        ! One half of the original justification is now obsolete, though:
        ! ifort DOES inline accum()/put() under -inline-forceinline
        ! -inline-factor=10000 (the report marks the standalone symbol DEAD
        ! STATIC FUNCTION), so the calls were never the problem -- the
        ! arrays are. See docs/dev/viscous_kernels.md section 17;
        ! the rewrite is kept at _fortran/residual_cand.f90.
        !   pm = (Vx, Vr, r*Vt_abs, ho, P-P_offset, r*(P-P_offset))
        !   mf = (rho*Vx, rho*Vr, rho*Vt_rel)

        ! Low boundary i=1
        call accum_corners(1, j, k, wall_lo, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
        mdot = mf1*dA(1,1,j,k) + mf2*dA(2,1,j,k) + mf3*dA(3,1,j,k)
        row(1,1) = mdot
        row(1,2) = pm1*mdot + pm5*dA(1,1,j,k)
        row(1,3) = pm2*mdot + pm5*dA(2,1,j,k)
        row(1,4) = pm3*mdot + pm6*dA(3,1,j,k)
        row(1,5) = pm4*mdot + Omega*pm6*dA(3,1,j,k)

        ! Interior i=2..ni-1
        !DIR$ IVDEP
        do i = 2, ni-1
            call accum_corners(i, j, k, 1.0e0, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
            mdot = mf1*dA(1,i,j,k) + mf2*dA(2,i,j,k) + mf3*dA(3,i,j,k)
            row(i,1) = mdot
            row(i,2) = pm1*mdot + pm5*dA(1,i,j,k)
            row(i,3) = pm2*mdot + pm5*dA(2,i,j,k)
            row(i,4) = pm3*mdot + pm6*dA(3,i,j,k)
            row(i,5) = pm4*mdot + Omega*pm6*dA(3,i,j,k)
        end do

        ! High boundary i=ni
        call accum_corners(ni, j, k, wall_hi, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
        mdot = mf1*dA(1,ni,j,k) + mf2*dA(2,ni,j,k) + mf3*dA(3,ni,j,k)
        row(ni,1) = mdot
        row(ni,2) = pm1*mdot + pm5*dA(1,ni,j,k)
        row(ni,3) = pm2*mdot + pm5*dA(2,ni,j,k)
        row(ni,4) = pm3*mdot + pm6*dA(3,ni,j,k)
        row(ni,5) = pm4*mdot + Omega*pm6*dA(3,ni,j,k)

    contains
        pure subroutine accum_corners(i, j, k, wfac, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
            ! Accumulates the 4 face corners (i,j:j+1,k:k+1), same
            ! summation order as the original sequential accum() calls.
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real, intent(out) :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3
            real :: dp1, dp2, dp3, dp4, w
            real :: g1, g2, g3, g4
            dp1 = P(i,j,k)     - P_offset
            dp2 = P(i,j+1,k)   - P_offset
            dp3 = P(i,j,k+1)   - P_offset
            dp4 = P(i,j+1,k+1) - P_offset
            ! Vx, Vr and r*Vt come from the conserved state rather than
            ! their own nodal arrays: cons = (rho, rho*Vx, rho*Vr,
            ! rho*r*Vt, rho*e), so Vx = c2/c1, Vr = c3/c1, r*Vt = c4/c1
            ! exactly. That drops three streamed fields (9 nodal -> 7,
            ! ~12.5 B/cell) for one reciprocal per corner, which is the
            ! right trade on a kernel that runs at DRAM bandwidth.
            ! Recomputed here, never precomputed into a buffer: a buffer
            ! would write more than it saves. See section 20.
            g1 = 1.0e0/cons(i,j,k,1)
            g2 = 1.0e0/cons(i,j+1,k,1)
            g3 = 1.0e0/cons(i,j,k+1,1)
            g4 = 1.0e0/cons(i,j+1,k+1,1)
            pm1 = 0.25e0*cons(i,j,k,2)*g1 + 0.25e0*cons(i,j+1,k,2)*g2 &
                + 0.25e0*cons(i,j,k+1,2)*g3 + 0.25e0*cons(i,j+1,k+1,2)*g4
            pm2 = 0.25e0*cons(i,j,k,3)*g1 + 0.25e0*cons(i,j+1,k,3)*g2 &
                + 0.25e0*cons(i,j,k+1,3)*g3 + 0.25e0*cons(i,j+1,k+1,3)*g4
            pm3 = 0.25e0*cons(i,j,k,4)*g1 + 0.25e0*cons(i,j+1,k,4)*g2 &
                + 0.25e0*cons(i,j,k+1,4)*g3 + 0.25e0*cons(i,j+1,k+1,4)*g4
            pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i,j+1,k) + 0.25e0*ho(i,j,k+1) + 0.25e0*ho(i,j+1,k+1)
            pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
            pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i,j+1,k)*dp2 &
                + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i,j+1,k+1)*dp4
            w = 0.25e0*wfac
            mf1 = w*cons(i,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i,j,k+1,2) + w*cons(i,j+1,k+1,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i,j,k+1,3) + w*cons(i,j+1,k+1,3)
            ! rho*Vt_rel = c4*(1/r) - Omega*c1*r; 1/r is staged geometry.
            mf3 = w*(cons(i,j,k,4)*rinv(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                + w*(cons(i,j+1,k,4)*rinv(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                + w*(cons(i,j,k+1,4)*rinv(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                + w*(cons(i,j+1,k+1,4)*rinv(i,j+1,k+1) - Omega*cons(i,j+1,k+1,1)*r(i,j+1,k+1))
        end subroutine accum_corners
    end subroutine iface_flow_row_rv

    pure subroutine jface_flow_row_rv(ho, P, P_offset, r, rinv, &
                                   cons, Omega, dA, &
                                   wall_lo, wall_hi, row, jf, k, ni, nj, nk)
        ! Compute inviscid face flows on the (ni-1) j-faces of face row jf at
        ! cell plane k; jf=1 / jf=nj are the wall-masked boundary rows.
        ! j-face corners: (i:i+1, jf, k:k+1)

        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: rinv(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: wall_lo(ni-1, nk-1)
        real, intent(in) :: wall_hi(ni-1, nk-1)
        real, intent(inout) :: row(ni, 5)

        integer :: i, j
        real :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3, mdot
        real :: dp1, dp2, dp3, dp4, w
        real :: g1, g2, g3, g4

        ! accum_corners MANUALLY INLINED, as in residual_nodal.f90 and
        ! residual_prod_soa.f90: one extra array dummy (rinv) was enough to
        ! push it past GCC's inline threshold, and these i loops went scalar
        ! while production's stayed vectorized. Arithmetic and summation
        ! order are production's. gfortran has no always_inline.
        j = jf
        if (jf == 1) then
            ! Low boundary j=1
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,k) - P_offset
                dp2 = P(i+1,j,k) - P_offset
                dp3 = P(i,j,k+1) - P_offset
                dp4 = P(i+1,j,k+1) - P_offset
                g1 = 1.0e0/cons(i,j,k,1)
                g2 = 1.0e0/cons(i+1,j,k,1)
                g3 = 1.0e0/cons(i,j,k+1,1)
                g4 = 1.0e0/cons(i+1,j,k+1,1)
                pm1 = 0.25e0*cons(i,j,k,2)*g1 + 0.25e0*cons(i+1,j,k,2)*g2 &
                    + 0.25e0*cons(i,j,k+1,2)*g3 + 0.25e0*cons(i+1,j,k+1,2)*g4
                pm2 = 0.25e0*cons(i,j,k,3)*g1 + 0.25e0*cons(i+1,j,k,3)*g2 &
                    + 0.25e0*cons(i,j,k+1,3)*g3 + 0.25e0*cons(i+1,j,k+1,3)*g4
                pm3 = 0.25e0*cons(i,j,k,4)*g1 + 0.25e0*cons(i+1,j,k,4)*g2 &
                    + 0.25e0*cons(i,j,k+1,4)*g3 + 0.25e0*cons(i+1,j,k+1,4)*g4
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j,k+1) + 0.25e0*ho(i+1,j,k+1)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i+1,j,k+1)*dp4
                w = 0.25e0*wall_lo(i,k)
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j,k+1,2) + w*cons(i+1,j,k+1,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j,k+1,3) + w*cons(i+1,j,k+1,3)
                mf3 = w*(cons(i,j,k,4)*rinv(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)*rinv(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j,k+1,4)*rinv(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                    + w*(cons(i+1,j,k+1,4)*rinv(i+1,j,k+1) - Omega*cons(i+1,j,k+1,1)*r(i+1,j,k+1))
                mdot = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
                row(i,1) = mdot
                row(i,2) = pm1*mdot + pm5*dA(1,i,jf,k)
                row(i,3) = pm2*mdot + pm5*dA(2,i,jf,k)
                row(i,4) = pm3*mdot + pm6*dA(3,i,jf,k)
                row(i,5) = pm4*mdot + Omega*pm6*dA(3,i,jf,k)
            end do
        else if (jf == nj) then
            ! High boundary j=nj
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,k) - P_offset
                dp2 = P(i+1,j,k) - P_offset
                dp3 = P(i,j,k+1) - P_offset
                dp4 = P(i+1,j,k+1) - P_offset
                g1 = 1.0e0/cons(i,j,k,1)
                g2 = 1.0e0/cons(i+1,j,k,1)
                g3 = 1.0e0/cons(i,j,k+1,1)
                g4 = 1.0e0/cons(i+1,j,k+1,1)
                pm1 = 0.25e0*cons(i,j,k,2)*g1 + 0.25e0*cons(i+1,j,k,2)*g2 &
                    + 0.25e0*cons(i,j,k+1,2)*g3 + 0.25e0*cons(i+1,j,k+1,2)*g4
                pm2 = 0.25e0*cons(i,j,k,3)*g1 + 0.25e0*cons(i+1,j,k,3)*g2 &
                    + 0.25e0*cons(i,j,k+1,3)*g3 + 0.25e0*cons(i+1,j,k+1,3)*g4
                pm3 = 0.25e0*cons(i,j,k,4)*g1 + 0.25e0*cons(i+1,j,k,4)*g2 &
                    + 0.25e0*cons(i,j,k+1,4)*g3 + 0.25e0*cons(i+1,j,k+1,4)*g4
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j,k+1) + 0.25e0*ho(i+1,j,k+1)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i+1,j,k+1)*dp4
                w = 0.25e0*wall_hi(i,k)
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j,k+1,2) + w*cons(i+1,j,k+1,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j,k+1,3) + w*cons(i+1,j,k+1,3)
                mf3 = w*(cons(i,j,k,4)*rinv(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)*rinv(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j,k+1,4)*rinv(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                    + w*(cons(i+1,j,k+1,4)*rinv(i+1,j,k+1) - Omega*cons(i+1,j,k+1,1)*r(i+1,j,k+1))
                mdot = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
                row(i,1) = mdot
                row(i,2) = pm1*mdot + pm5*dA(1,i,jf,k)
                row(i,3) = pm2*mdot + pm5*dA(2,i,jf,k)
                row(i,4) = pm3*mdot + pm6*dA(3,i,jf,k)
                row(i,5) = pm4*mdot + Omega*pm6*dA(3,i,jf,k)
            end do
        else
            ! Interior 2 <= jf <= nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,k) - P_offset
                dp2 = P(i+1,j,k) - P_offset
                dp3 = P(i,j,k+1) - P_offset
                dp4 = P(i+1,j,k+1) - P_offset
                g1 = 1.0e0/cons(i,j,k,1)
                g2 = 1.0e0/cons(i+1,j,k,1)
                g3 = 1.0e0/cons(i,j,k+1,1)
                g4 = 1.0e0/cons(i+1,j,k+1,1)
                pm1 = 0.25e0*cons(i,j,k,2)*g1 + 0.25e0*cons(i+1,j,k,2)*g2 &
                    + 0.25e0*cons(i,j,k+1,2)*g3 + 0.25e0*cons(i+1,j,k+1,2)*g4
                pm2 = 0.25e0*cons(i,j,k,3)*g1 + 0.25e0*cons(i+1,j,k,3)*g2 &
                    + 0.25e0*cons(i,j,k+1,3)*g3 + 0.25e0*cons(i+1,j,k+1,3)*g4
                pm3 = 0.25e0*cons(i,j,k,4)*g1 + 0.25e0*cons(i+1,j,k,4)*g2 &
                    + 0.25e0*cons(i,j,k+1,4)*g3 + 0.25e0*cons(i+1,j,k+1,4)*g4
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j,k+1) + 0.25e0*ho(i+1,j,k+1)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i+1,j,k+1)*dp4
                w = 0.25e0
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j,k+1,2) + w*cons(i+1,j,k+1,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j,k+1,3) + w*cons(i+1,j,k+1,3)
                mf3 = w*(cons(i,j,k,4)*rinv(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)*rinv(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j,k+1,4)*rinv(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                    + w*(cons(i+1,j,k+1,4)*rinv(i+1,j,k+1) - Omega*cons(i+1,j,k+1,1)*r(i+1,j,k+1))
                mdot = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
                row(i,1) = mdot
                row(i,2) = pm1*mdot + pm5*dA(1,i,jf,k)
                row(i,3) = pm2*mdot + pm5*dA(2,i,jf,k)
                row(i,4) = pm3*mdot + pm6*dA(3,i,jf,k)
                row(i,5) = pm4*mdot + Omega*pm6*dA(3,i,jf,k)
            end do
        end if

    end subroutine jface_flow_row_rv

    pure subroutine kface_flow_plane_rv(ho, P, P_offset, r, rinv, &
                                     cons, Omega, dA, &
                                     wall_lo, wall_hi, plane, kf, njp, &
                                     ni, nj, nk)
        ! Compute inviscid face flows on the (ni-1)x(nj-1) k-face plane kf;
        ! kf=1 / kf=nk are the wall-masked boundary planes. njp (nj or nj+1)
        ! is the plane buffer's padded j-extent -- see set_residual.
        ! k-face corners: (i:i+1, j:j+1, kf)

        implicit none
        integer, intent(in) :: kf, njp, ni, nj, nk
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: rinv(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
        real, intent(in) :: wall_lo(ni-1, nj-1)
        real, intent(in) :: wall_hi(ni-1, nj-1)
        real, intent(inout) :: plane(ni, njp, 5)

        integer :: i, j, k
        real :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3, mdot
        real :: dp1, dp2, dp3, dp4, w
        real :: g1, g2, g3, g4

        ! accum_corners MANUALLY INLINED, as in residual_nodal.f90 and
        ! residual_prod_soa.f90: one extra array dummy (rinv) was enough to
        ! push it past GCC's inline threshold, and these i loops went scalar
        ! while production's stayed vectorized. Arithmetic and summation
        ! order are production's. gfortran has no always_inline.
        k = kf
        if (kf == 1) then
            ! Low boundary k=1
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,k) - P_offset
                dp2 = P(i+1,j,k) - P_offset
                dp3 = P(i,j+1,k) - P_offset
                dp4 = P(i+1,j+1,k) - P_offset
                g1 = 1.0e0/cons(i,j,k,1)
                g2 = 1.0e0/cons(i+1,j,k,1)
                g3 = 1.0e0/cons(i,j+1,k,1)
                g4 = 1.0e0/cons(i+1,j+1,k,1)
                pm1 = 0.25e0*cons(i,j,k,2)*g1 + 0.25e0*cons(i+1,j,k,2)*g2 &
                    + 0.25e0*cons(i,j+1,k,2)*g3 + 0.25e0*cons(i+1,j+1,k,2)*g4
                pm2 = 0.25e0*cons(i,j,k,3)*g1 + 0.25e0*cons(i+1,j,k,3)*g2 &
                    + 0.25e0*cons(i,j+1,k,3)*g3 + 0.25e0*cons(i+1,j+1,k,3)*g4
                pm3 = 0.25e0*cons(i,j,k,4)*g1 + 0.25e0*cons(i+1,j,k,4)*g2 &
                    + 0.25e0*cons(i,j+1,k,4)*g3 + 0.25e0*cons(i+1,j+1,k,4)*g4
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j+1,k) + 0.25e0*ho(i+1,j+1,k)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j+1,k)*dp3 + 0.25e0*r(i+1,j+1,k)*dp4
                w = 0.25e0*wall_lo(i,j)
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i+1,j+1,k,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i+1,j+1,k,3)
                mf3 = w*(cons(i,j,k,4)*rinv(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)*rinv(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j+1,k,4)*rinv(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                    + w*(cons(i+1,j+1,k,4)*rinv(i+1,j+1,k) - Omega*cons(i+1,j+1,k,1)*r(i+1,j+1,k))
                mdot = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
                plane(i,j,1) = mdot
                plane(i,j,2) = pm1*mdot + pm5*dA(1,i,j,kf)
                plane(i,j,3) = pm2*mdot + pm5*dA(2,i,j,kf)
                plane(i,j,4) = pm3*mdot + pm6*dA(3,i,j,kf)
                plane(i,j,5) = pm4*mdot + Omega*pm6*dA(3,i,j,kf)
            end do
            end do
        else if (kf == nk) then
            ! High boundary k=nk
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,k) - P_offset
                dp2 = P(i+1,j,k) - P_offset
                dp3 = P(i,j+1,k) - P_offset
                dp4 = P(i+1,j+1,k) - P_offset
                g1 = 1.0e0/cons(i,j,k,1)
                g2 = 1.0e0/cons(i+1,j,k,1)
                g3 = 1.0e0/cons(i,j+1,k,1)
                g4 = 1.0e0/cons(i+1,j+1,k,1)
                pm1 = 0.25e0*cons(i,j,k,2)*g1 + 0.25e0*cons(i+1,j,k,2)*g2 &
                    + 0.25e0*cons(i,j+1,k,2)*g3 + 0.25e0*cons(i+1,j+1,k,2)*g4
                pm2 = 0.25e0*cons(i,j,k,3)*g1 + 0.25e0*cons(i+1,j,k,3)*g2 &
                    + 0.25e0*cons(i,j+1,k,3)*g3 + 0.25e0*cons(i+1,j+1,k,3)*g4
                pm3 = 0.25e0*cons(i,j,k,4)*g1 + 0.25e0*cons(i+1,j,k,4)*g2 &
                    + 0.25e0*cons(i,j+1,k,4)*g3 + 0.25e0*cons(i+1,j+1,k,4)*g4
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j+1,k) + 0.25e0*ho(i+1,j+1,k)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j+1,k)*dp3 + 0.25e0*r(i+1,j+1,k)*dp4
                w = 0.25e0*wall_hi(i,j)
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i+1,j+1,k,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i+1,j+1,k,3)
                mf3 = w*(cons(i,j,k,4)*rinv(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)*rinv(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j+1,k,4)*rinv(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                    + w*(cons(i+1,j+1,k,4)*rinv(i+1,j+1,k) - Omega*cons(i+1,j+1,k,1)*r(i+1,j+1,k))
                mdot = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
                plane(i,j,1) = mdot
                plane(i,j,2) = pm1*mdot + pm5*dA(1,i,j,kf)
                plane(i,j,3) = pm2*mdot + pm5*dA(2,i,j,kf)
                plane(i,j,4) = pm3*mdot + pm6*dA(3,i,j,kf)
                plane(i,j,5) = pm4*mdot + Omega*pm6*dA(3,i,j,kf)
            end do
            end do
        else
            ! Interior 2 <= kf <= nk-1
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,k) - P_offset
                dp2 = P(i+1,j,k) - P_offset
                dp3 = P(i,j+1,k) - P_offset
                dp4 = P(i+1,j+1,k) - P_offset
                g1 = 1.0e0/cons(i,j,k,1)
                g2 = 1.0e0/cons(i+1,j,k,1)
                g3 = 1.0e0/cons(i,j+1,k,1)
                g4 = 1.0e0/cons(i+1,j+1,k,1)
                pm1 = 0.25e0*cons(i,j,k,2)*g1 + 0.25e0*cons(i+1,j,k,2)*g2 &
                    + 0.25e0*cons(i,j+1,k,2)*g3 + 0.25e0*cons(i+1,j+1,k,2)*g4
                pm2 = 0.25e0*cons(i,j,k,3)*g1 + 0.25e0*cons(i+1,j,k,3)*g2 &
                    + 0.25e0*cons(i,j+1,k,3)*g3 + 0.25e0*cons(i+1,j+1,k,3)*g4
                pm3 = 0.25e0*cons(i,j,k,4)*g1 + 0.25e0*cons(i+1,j,k,4)*g2 &
                    + 0.25e0*cons(i,j+1,k,4)*g3 + 0.25e0*cons(i+1,j+1,k,4)*g4
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j+1,k) + 0.25e0*ho(i+1,j+1,k)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j+1,k)*dp3 + 0.25e0*r(i+1,j+1,k)*dp4
                w = 0.25e0
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i+1,j+1,k,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i+1,j+1,k,3)
                mf3 = w*(cons(i,j,k,4)*rinv(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)*rinv(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j+1,k,4)*rinv(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                    + w*(cons(i+1,j+1,k,4)*rinv(i+1,j+1,k) - Omega*cons(i+1,j+1,k,1)*r(i+1,j+1,k))
                mdot = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
                plane(i,j,1) = mdot
                plane(i,j,2) = pm1*mdot + pm5*dA(1,i,j,kf)
                plane(i,j,3) = pm2*mdot + pm5*dA(2,i,j,kf)
                plane(i,j,4) = pm3*mdot + pm6*dA(3,i,j,kf)
                plane(i,j,5) = pm4*mdot + Omega*pm6*dA(3,i,j,kf)
            end do
            end do
        end if

    end subroutine kface_flow_plane_rv

end module residual_rinv_helpers


! =====================================================================
! Driver: production's set_residual verbatim, except that it takes rinv
! and routes the three face helpers to their _rv variants. Slab sweep,
! rolling buffers, fused dU write, folded change limiter and deferred
! cusp correction are unchanged, so the only variable against `prod` is
! where 1/r comes from.
! =====================================================================
subroutine set_residual_rinv( &
    cons, P, P_offset, &
    r, rinv, Omega, dAi, dAj, dAk, &
    f_body, &
    dU, &
    vx, vr, vt, ho, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    dt_vol, dampin, &
    kb, njp, ni, nj, nk &
    )

    use residual_rinv_helpers
    use residual_helpers, only: correct_cusp_kface_du

    implicit none

    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(in) :: r(ni, nj, nk)
    ! 1/r, built once outside the timed region: r is grid geometry.
    real, intent(in) :: rinv(ni, nj, nk)
    real, intent(in) :: Omega
    real, intent(in) :: dAi(3, ni, nj-1, nk-1)
    real, intent(in) :: dAj(3, ni-1, nj, nk-1)
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
    real, intent(in) :: f_body(ni-1, nj-1, nk-1, 5)
    real, intent(in) :: vx(ni, nj, nk)
    real, intent(in) :: vr(ni, nj, nk)
    real, intent(in) :: vt(ni, nj, nk)
    real, intent(in) :: ho(ni, nj, nk)
    real, intent(in) :: walli1(nj-1, nk-1)
    real, intent(in) :: wallni(nj-1, nk-1)
    real, intent(in) :: wallj1(ni-1, nk-1)
    real, intent(in) :: wallnj(ni-1, nk-1)
    real, intent(in) :: wallk1(ni-1, nj-1)
    real, intent(in) :: wallnk(ni-1, nj-1)
    integer, intent(in) :: i_cusp_start, i_cusp_end
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! Two transient rolling flow-scratch buffers: planes holds the k-face
    ! plane pair (slots pa/pb), rows holds the i-face row (slot 1) and the
    ! j-face row pair (slots ja/jb alternating 2/3). Caller backs these with
    ! block._tau_q_halo, which is pure transient scratch -- the layout here
    ! is private to this call. njp is planes' padded j-extent, chosen by the
    ! caller: nj+1 whenever ni*nj*4 bytes is a whole page multiple (e.g.
    ! ni=128, nj=96: 48 KB exactly), so the ten concurrent component streams
    ! of the k-accumulate (5 components x pa/pb) never 4K-alias into the
    ! same L1 sets; nj otherwise (measured: an unconditional pad costs ~5%
    ! at small blocks it does not help).
    real, intent(inout) :: planes(ni, njp, 5, 2)
    real, intent(inout) :: rows(ni, 5, 3)
    ! Change limiter folded in: dt_vol and dampin are damp_residual's
    ! inputs. dampin <= 0 disables the limiter (matching the caller's
    ! `if dampin is not None` skip).
    real, intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dampin
    integer, intent(in) :: kb, njp, ni, nj, nk

    integer :: i, j, k, m, k0, k1, ja, jb, pa, pb, stmp
    integer :: ncell
    real :: avg(5), ravg(5)

    do m = 1, 5
        avg(m) = 0.0e0
    end do

    pa = 1
    pb = 2

    ! Prime the rolling k-face plane with face k=1 before the slab sweep
    ! (the fused loop below always has plane k in slot pa on entry to cell
    ! k, needing only face k+1 freshly computed into pb).
    call kface_flow_plane_rv(ho, P, P_offset, r, rinv, cons, &
                          Omega, dAk, wallk1, wallnk, planes(:,:,:,pa), &
                          1, njp, ni, nj, nk)

    do k0 = 1, nk-1, kb
    k1 = min(k0 + kb - 1, nk-1)

    ! --- i+j+k fused per (j,k) row: single touch on dU ---
    ! For each cell row (j,k): compute the i-face row (slot 1), advance the
    ! rolling j-face pair (slots ja/jb), and advance the rolling k-face
    ! pair (slots pa/pb, one plane ahead of the current cell layer -- pa
    ! holds face k, pb gets face k+1 computed fresh each k). All three
    ! contributions are folded into dU in one write, so each dU element is
    ! touched exactly once per residual evaluation (previously two full
    ! sweeps: the i/j write, then a separate k-direction read-modify-write).
    ! The k-face pair carries across slab boundaries the same way the
    ! un-fused version did (plane k0 of a slab is the previous slab's k1+1,
    ! already resident in pa), so only the very first cell (k=1 overall)
    ! computes its own low face before the loop.
    do k = k0, k1
        ja = 2
        jb = 3
        ! Prime the rolling j-face pair with the j=1 boundary face.
        call jface_flow_row_rv(ho, P, P_offset, r, rinv, cons, &
                            Omega, dAj, wallj1, wallnj, rows(:,:,ja), &
                            1, k, ni, nj, nk)
        ! Advance the rolling k-face pair: pa already holds face k (primed
        ! before the sweep, or carried from the previous k iteration); pb
        ! gets face k+1 computed fresh.
        call kface_flow_plane_rv(ho, P, P_offset, r, rinv, cons, &
                              Omega, dAk, wallk1, wallnk, planes(:,:,:,pb), &
                              k+1, njp, ni, nj, nk)
        do j = 1, nj-1
            call iface_flow_row_rv(ho, P, P_offset, r, rinv, cons, &
                                Omega, dAi, walli1(j,k), wallni(j,k), &
                                rows(:,:,1), j, k, ni, nj, nk)
            call jface_flow_row_rv(ho, P, P_offset, r, rinv, cons, &
                                Omega, dAj, wallj1, wallnj, rows(:,:,jb), &
                                j+1, k, ni, nj, nk)
            do m = 1, 5
            do i = 1, ni-1
                dU(i,j,k,m) = rows(i,m,1) - rows(i+1,m,1) + f_body(i,j,k,m) &
                            + rows(i,m,ja) - rows(i,m,jb) &
                            + planes(i,j,m,pa) - planes(i,j,m,pb)
                ! Change-limiter reduction, accumulated while dU is still in
                ! a register -- this is the whole point of the fusion: the
                ! separate routine's first full-volume dU read disappears.
                avg(m) = avg(m) + abs(dU(i,j,k,m) * dt_vol(i,j,k))
            end do
            end do
            stmp = ja
            ja = jb
            jb = stmp
        end do
        stmp = pa
        pa = pb
        pb = stmp
    end do

    end do  ! ===== end slab sweep =====

    ! Cusp seam: non-local in k (couples the k=1 and k=nk faces), applied as
    ! a deferred O(surface) correction to dU after the sweep. nk=2 (the two
    ! seam cells coincide) is not supported.
    if (i_cusp_start > 0 .and. nk > 2) then
        call correct_cusp_kface_du(vx, vr, vt, ho, P, P_offset, r, cons, &
                                   Omega, dAk, wallk1, wallnk, dU, &
                                   i_cusp_start, i_cusp_end, ni, nj, nk)
    end if

    ! ---- change limiter, second half ----
    ! The reduction above was accumulated during the sweep, so only the
    ! pointwise scaling pass remains. NOTE the cusp correction just
    ! modified dU on the two seam cell planes, which the reduction did not
    ! see; that is an O(surface) discrepancy in a block-mean over O(volume)
    ! cells, and is corrected below for exactness.
    if (dampin > 0.0e0) then
        ncell = (ni-1)*(nj-1)*(nk-1)
        do m = 1, 5
            avg(m) = avg(m) / ncell
            if (avg(m) > 0.0e0) then
                ravg(m) = 1.0e0 / avg(m)
            else
                ravg(m) = 0.0e0
            end if
        end do
        call scale_du(dU, dt_vol, ravg, dampin, ni, nj, nk)
    end if

contains

    ! The scaling pass lives in its own procedure so ifort sees a dU with no
    ! other writes in scope. Inline in the parent, the main sweep also writes
    ! dU and ifort cannot disprove that the two write regions overlap
    ! ("assumed OUTPUT dependence"), so it distributes the nest and only
    ! partially vectorizes it. Production's standalone damp_residual has the
    ! identical loop and vectorizes cleanly, which is the clue this follows.
    subroutine scale_du(dU, dt_vol, ravg, dampin, ni, nj, nk)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
        real, intent(in) :: dt_vol(ni-1, nj-1, nk-1)
        real, intent(in) :: ravg(5), dampin
        integer :: i, j, k, m
        real :: chg, fdamp
        do k = 1, nk-1
        do j = 1, nj-1
        do m = 1, 5
        do i = 1, ni-1
            chg   = abs(dU(i,j,k,m) * dt_vol(i,j,k))
            fdamp = chg * ravg(m)
            dU(i,j,k,m) = dU(i,j,k,m) / (1.0e0 + fdamp/dampin)
        end do
        end do
        end do
        end do
    end subroutine scale_du

end subroutine set_residual_rinv
