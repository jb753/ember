module residual_helpers
    implicit none
    private
    public :: iface_flow_row, jface_flow_row, kface_flow_plane
    public :: correct_cusp_kface_du
    public :: irs_tri_coeffs, irs_gather_tile, irs_gather_tile_scaled
    public :: irs_scatter_tile, irs_tile_solve, irs_jk_strips, irs_jac_line
    public :: IRS_BJ, IRS_TB, IRS_W
    public :: RES_JAREA, RES_JMIN

    ! j-panel AREA of set_residual's k walk, in cells: the panel is
    ! RES_JAREA/ni rows deep, so the carry it bounds -- the rolling k-face
    ! plane pair -- is a fixed number of BYTES whatever the block's aspect
    ! ratio, which a fixed row count would not be. See the panel loop in
    ! set_residual for what it bounds and why area is the right unit.
    integer, parameter :: RES_JAREA = 4400
    ! Floor on the panel depth in rows. A panel recomputes its own lowest
    ! j-face row, so 1/jbw of the j-face flux work -- about a third of the
    ! kernel's flux arithmetic -- is redundant, and on a long-i block the
    ! area rule alone drives jbw down to single digits, where that overhead
    ! is worth several percent. Serial that overhead is the whole story and
    ! the panel loses; contended the cache win pays for it many times over.
    ! Swept: see bench/README.md.
    integer, parameter :: RES_JMIN = 16

    ! Tile width for the i-solve transpose pad: sized to fill L1d without
    ! spilling it, NOT to match the SIMD lane count. The tile is
    ! IRS_BJ*nci*4 bytes; at nci=272 that is 34 KB for 32, the largest
    ! that fits Sapphire Rapids' 48 KB L1d. Swept on the production build
    ! (ifort, Xeon 8480+, 1M duct): 32 is -26% serial / -8% at 100-rank
    ! saturation against the previous 8, while 64 (68 KB, L1-spilling)
    ! gives most of it back. Bitwise identical for every value -- it only
    ! groups independent j-lines. Machine-dependent: re-sweep on new
    ! hardware, in the real build (a gfortran sweep picked 64, the wrong
    ! constant).
    !
    ! RE-SWEPT after the transpose was blocked, and the picture changed:
    ! the optimum is now FLAT above 32, not sharply peaked. At 1M on 2
    ! P-cores, against a fixed control arm that fingerprinted identically
    ! in all four builds: 16 is +19.5%, 32 is the reference, 64 is +0.3%
    ! and 128 is -2.0% (about 1.5 sigma, not worth acting on). So 16 is
    ! genuinely too narrow -- two AVX2 chains cannot cover the recurrence's
    ! FMA->MUL dependency and it goes latency-bound -- but 32 is already
    ! past the knee and widening buys nothing, L1 spill or not. When the
    ! transpose was scalar, L1 residency of the tile dominated and the
    ! optimum was sharp; with it blocked, the tile passes are issue-bound
    ! instead and BJ stops mattering. Useful for porting: this constant now
    ! needs far less care than the note above implies, PROVIDED the
    ! transpose is blocked. Do not raise it chasing the -2% -- that is a
    ! marginal result on a machine that is not the production target.
    integer, parameter :: IRS_BJ = 32

    ! Transpose block edge. 8 = the AVX2 float32 lane count, so one staged
    ! row is exactly one vector load and the in-block transpose becomes a
    ! register shuffle network. Steeply optimal for that reason: swept at
    ! 1M on 2 P-cores, 4 is +43.8% and 16 is +9.0% against 8. Re-sweep on
    ! an AVX-512 target, where the natural width is 16.
    integer, parameter :: IRS_TB = 8

    ! i-strip width for the fused j+k pass. A strip spans the full j and k
    ! extent for one component: IRS_W*ncj*nck*4 bytes, 917 KB at 64 on a
    ! 273x65x57 block, inside a 2 MB L2. Sets both the block size and the
    ! vector loop length, so likewise steeply optimal: 32 (458 KB, 4 AVX2
    ! iterations) -5.6%, 64 -11.0%, 128 (1.83 MB, past L2) -1.2%.
    !
    ! RE-SWEPT at 8-rank socket contention (gfortran, Haswell workstation),
    ! where the earlier 2-rank sweep above was run at only 2 P-cores and so
    ! never priced the shared-L2 pressure of real contention: 32 is -13.9%
    ! against 64, 128 is +21.1%. 64 was tuned to fit a 2 MB L2 uncontended;
    ! under contention each rank's effective L2 share shrinks and 64 spills.
    ! Bitwise identical for every value -- it only sizes the strip. Re-sweep
    ! on the ifort/Sapphire production target before trusting this further.
    integer, parameter :: IRS_W = 32

contains

    ! The three face-flow helpers below assemble, per face, the 5 inviscid
    ! flows from face-averaged per-mass factors pm(6) and mass-flux factors
    ! mf(3) via their internal accum/put pairs (bodies identical across the
    ! three; the wall mask weights mf only):
    !   pm = (Vx, Vr, r*Vt_abs, ho, P-P_offset, r*(P-P_offset))
    !   mf = (rho*Vx, rho*Vr, rho*Vt_rel)
    ! Granularity is one row (i/j directions) or one plane (k direction) so
    ! the caller can roll small buffers instead of staging full volumes.

    pure subroutine iface_flow_row(P, P_offset, r, &
                                   cons, Omega, dA, &
                                   wall_lo, wall_hi, row, j, k, ni, nj, nk)
        ! Compute inviscid face flows on the ni i-faces of cell row (j,k);
        ! the i=1 / i=ni boundary faces are wall-masked by the scalars
        ! wall_lo / wall_hi. i-face corners: (i, j:j+1, k:k+1)

        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk)
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
            ! Stagnation enthalpy from the conserved state and the pressure,
            ! not its own nodal array: h = u + P/rho is the definition of
            ! enthalpy, so ho = u + P/rho + V^2/2 = e + P/rho = (c5 + P)/c1
            ! exactly, for any fluid (RealFluid.get_h IS u + P/rho;
            ! PerfectFluid's gamma*u + R*T_dtm is the same expression once
            ! P/rho is expanded). The reciprocal is the one already formed
            ! for the velocity components and P is already loaded for dp, so
            ! this trades a streamed nodal field for an add and a multiply.
            ! RAW P, not dp: stagnation enthalpy carries no pressure offset.
            ! All four accumulations in this file must spell it identically --
            ! correct_cusp_kface_du recomputes what the sweep accumulated and
            ! subtracts the two, so a different association leaves a last-bit
            ! residue on the seam planes.
            pm4 = 0.25e0*(cons(i,j,k,5)+P(i,j,k))*g1 + 0.25e0*(cons(i,j+1,k,5)+P(i,j+1,k))*g2 &
                + 0.25e0*(cons(i,j,k+1,5)+P(i,j,k+1))*g3 + 0.25e0*(cons(i,j+1,k+1,5)+P(i,j+1,k+1))*g4
            pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
            pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i,j+1,k)*dp2 &
                + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i,j+1,k+1)*dp4
            w = 0.25e0*wfac
            mf1 = w*cons(i,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i,j,k+1,2) + w*cons(i,j+1,k+1,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i,j,k+1,3) + w*cons(i,j+1,k+1,3)
            ! rho*Vt_rel = rho*Vt - Omega*rho*r = c4/r - Omega*c1*r
            mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                + w*(cons(i,j+1,k,4)/r(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                + w*(cons(i,j,k+1,4)/r(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                + w*(cons(i,j+1,k+1,4)/r(i,j+1,k+1) - Omega*cons(i,j+1,k+1,1)*r(i,j+1,k+1))
        end subroutine accum_corners
    end subroutine iface_flow_row


    pure subroutine jface_flow_row(P, P_offset, r, &
                                   cons, Omega, dA, &
                                   wall_lo, wall_hi, row, jf, k, ni, nj, nk)
        ! Compute inviscid face flows on the (ni-1) j-faces of face row jf at
        ! cell plane k; jf=1 / jf=nj are the wall-masked boundary rows.
        ! j-face corners: (i:i+1, jf, k:k+1)

        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: wall_lo(ni-1, nk-1)
        real, intent(in) :: wall_hi(ni-1, nk-1)
        real, intent(inout) :: row(ni, 5)

        integer :: i
        real :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3, mdot

        ! pm/mf scalarized and the four corners hand-unrolled for the same
        ! ifort vectorizer reason as iface_flow_row above -- see the comment
        ! there for the full justification and its July 2026 re-measurement.
        if (jf == 1) then
            ! Low boundary j=1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_corners(i, 1, k, wall_lo(i,k), pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
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
                call accum_corners(i, nj, k, wall_hi(i,k), pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
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
                call accum_corners(i, jf, k, 1.0e0, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
                mdot = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
                row(i,1) = mdot
                row(i,2) = pm1*mdot + pm5*dA(1,i,jf,k)
                row(i,3) = pm2*mdot + pm5*dA(2,i,jf,k)
                row(i,4) = pm3*mdot + pm6*dA(3,i,jf,k)
                row(i,5) = pm4*mdot + Omega*pm6*dA(3,i,jf,k)
            end do
        end if

    contains
        pure subroutine accum_corners(i, j, k, wfac, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
            ! Accumulates the 4 face corners (i:i+1,j,k:k+1), same
            ! summation order as the original sequential accum() calls.
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real, intent(out) :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3
            real :: dp1, dp2, dp3, dp4, w
            real :: g1, g2, g3, g4
            dp1 = P(i,j,k)     - P_offset
            dp2 = P(i+1,j,k)   - P_offset
            dp3 = P(i,j,k+1)   - P_offset
            dp4 = P(i+1,j,k+1) - P_offset
            ! Vx, Vr and r*Vt come from the conserved state rather than
            ! their own nodal arrays: cons = (rho, rho*Vx, rho*Vr,
            ! rho*r*Vt, rho*e), so Vx = c2/c1, Vr = c3/c1, r*Vt = c4/c1
            ! exactly. That drops three streamed fields (9 nodal -> 7,
            ! ~12.5 B/cell) for one reciprocal per corner, which is the
            ! right trade on a kernel that runs at DRAM bandwidth.
            ! Recomputed here, never precomputed into a buffer: a buffer
            ! would write more than it saves. See section 20.
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
            ! Stagnation enthalpy from the conserved state and the pressure,
            ! not its own nodal array: h = u + P/rho is the definition of
            ! enthalpy, so ho = u + P/rho + V^2/2 = e + P/rho = (c5 + P)/c1
            ! exactly, for any fluid (RealFluid.get_h IS u + P/rho;
            ! PerfectFluid's gamma*u + R*T_dtm is the same expression once
            ! P/rho is expanded). The reciprocal is the one already formed
            ! for the velocity components and P is already loaded for dp, so
            ! this trades a streamed nodal field for an add and a multiply.
            ! RAW P, not dp: stagnation enthalpy carries no pressure offset.
            ! All four accumulations in this file must spell it identically --
            ! correct_cusp_kface_du recomputes what the sweep accumulated and
            ! subtracts the two, so a different association leaves a last-bit
            ! residue on the seam planes.
            pm4 = 0.25e0*(cons(i,j,k,5)+P(i,j,k))*g1 + 0.25e0*(cons(i+1,j,k,5)+P(i+1,j,k))*g2 &
                + 0.25e0*(cons(i,j,k+1,5)+P(i,j,k+1))*g3 + 0.25e0*(cons(i+1,j,k+1,5)+P(i+1,j,k+1))*g4
            pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
            pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i+1,j,k+1)*dp4
            w = 0.25e0*wfac
            mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j,k+1,2) + w*cons(i+1,j,k+1,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j,k+1,3) + w*cons(i+1,j,k+1,3)
            ! rho*Vt_rel = rho*Vt - Omega*rho*r = c4/r - Omega*c1*r
            mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                + w*(cons(i,j,k+1,4)/r(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                + w*(cons(i+1,j,k+1,4)/r(i+1,j,k+1) - Omega*cons(i+1,j,k+1,1)*r(i+1,j,k+1))
        end subroutine accum_corners
    end subroutine jface_flow_row


    pure subroutine kface_flow_plane(P, P_offset, r, &
                                     cons, Omega, dA, &
                                     wall_lo, wall_hi, plane, kf, j0, j1, njp, &
                                     ni, nj, nk)
        ! Compute inviscid face flows on face rows j0..j1 of the k-face plane
        ! kf; kf=1 / kf=nk are the wall-masked boundary planes. njp (nj or
        ! nj+1) is the plane buffer's padded j-extent -- see set_residual.
        ! The j range is the panel the caller is walking (j0=1, j1=nj-1 is
        ! the whole plane); rows outside it are left untouched, which is safe
        ! because a panel only ever reads back the rows it wrote.
        ! k-face corners: (i:i+1, j:j+1, kf)

        implicit none
        integer, intent(in) :: kf, j0, j1, njp, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
        real, intent(in) :: wall_lo(ni-1, nj-1)
        real, intent(in) :: wall_hi(ni-1, nj-1)
        real, intent(inout) :: plane(ni, njp, 5)

        integer :: i, j
        real :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3, mdot

        ! pm/mf scalarized and the four corners hand-unrolled for the same
        ! ifort vectorizer reason as iface_flow_row above -- see the comment
        ! there for the full justification and its July 2026 re-measurement.
        if (kf == 1) then
            ! Low boundary k=1
            do j = j0, j1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_corners(i, j, 1, wall_lo(i,j), pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
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
            do j = j0, j1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_corners(i, j, nk, wall_hi(i,j), pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
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
            do j = j0, j1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_corners(i, j, kf, 1.0e0, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
                mdot = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
                plane(i,j,1) = mdot
                plane(i,j,2) = pm1*mdot + pm5*dA(1,i,j,kf)
                plane(i,j,3) = pm2*mdot + pm5*dA(2,i,j,kf)
                plane(i,j,4) = pm3*mdot + pm6*dA(3,i,j,kf)
                plane(i,j,5) = pm4*mdot + Omega*pm6*dA(3,i,j,kf)
            end do
            end do
        end if

    contains
        pure subroutine accum_corners(i, j, k, wfac, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
            ! Accumulates the 4 face corners (i:i+1,j:j+1,k), same
            ! summation order as the original sequential accum() calls.
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real, intent(out) :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3
            real :: dp1, dp2, dp3, dp4, w
            real :: g1, g2, g3, g4
            dp1 = P(i,j,k)     - P_offset
            dp2 = P(i+1,j,k)   - P_offset
            dp3 = P(i,j+1,k)   - P_offset
            dp4 = P(i+1,j+1,k) - P_offset
            ! Vx, Vr and r*Vt come from the conserved state rather than
            ! their own nodal arrays: cons = (rho, rho*Vx, rho*Vr,
            ! rho*r*Vt, rho*e), so Vx = c2/c1, Vr = c3/c1, r*Vt = c4/c1
            ! exactly. That drops three streamed fields (9 nodal -> 7,
            ! ~12.5 B/cell) for one reciprocal per corner, which is the
            ! right trade on a kernel that runs at DRAM bandwidth.
            ! Recomputed here, never precomputed into a buffer: a buffer
            ! would write more than it saves. See section 20.
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
            ! Stagnation enthalpy from the conserved state and the pressure,
            ! not its own nodal array: h = u + P/rho is the definition of
            ! enthalpy, so ho = u + P/rho + V^2/2 = e + P/rho = (c5 + P)/c1
            ! exactly, for any fluid (RealFluid.get_h IS u + P/rho;
            ! PerfectFluid's gamma*u + R*T_dtm is the same expression once
            ! P/rho is expanded). The reciprocal is the one already formed
            ! for the velocity components and P is already loaded for dp, so
            ! this trades a streamed nodal field for an add and a multiply.
            ! RAW P, not dp: stagnation enthalpy carries no pressure offset.
            ! All four accumulations in this file must spell it identically --
            ! correct_cusp_kface_du recomputes what the sweep accumulated and
            ! subtracts the two, so a different association leaves a last-bit
            ! residue on the seam planes.
            pm4 = 0.25e0*(cons(i,j,k,5)+P(i,j,k))*g1 + 0.25e0*(cons(i+1,j,k,5)+P(i+1,j,k))*g2 &
                + 0.25e0*(cons(i,j+1,k,5)+P(i,j+1,k))*g3 + 0.25e0*(cons(i+1,j+1,k,5)+P(i+1,j+1,k))*g4
            pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
            pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                + 0.25e0*r(i,j+1,k)*dp3 + 0.25e0*r(i+1,j+1,k)*dp4
            w = 0.25e0*wfac
            mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i+1,j+1,k,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i+1,j+1,k,3)
            ! rho*Vt_rel = rho*Vt - Omega*rho*r = c4/r - Omega*c1*r
            mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                + w*(cons(i,j+1,k,4)/r(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                + w*(cons(i+1,j+1,k,4)/r(i+1,j+1,k) - Omega*cons(i+1,j+1,k,1)*r(i+1,j+1,k))
        end subroutine accum_corners
    end subroutine kface_flow_plane


    subroutine correct_cusp_kface_du(P, P_offset, r, &
                                     cons, Omega, dAk, &
                                     wall_lo, wall_hi, dU, &
                                     i_cusp_start, i_cusp_end, ni, nj, nk)
        ! Correct the residual for the cusp k-face coupling (matching Multall).
        ! Mass / angular momentum / energy: full TFLUX average across the seam.
        ! Axial and radial momentum: rebuild both faces from seam-averaged
        ! mdot, velocity and pressure, with per-face dAk only.
        !
        ! Deferred form for the rolling-buffer sweep: the seam couples the
        ! k=1 and k=nk faces, whose flows are long retired from the rolling
        ! buffers, so during the sweep the two seam cells accumulate the raw
        ! (wall-masked) one-sided fluxes and this pass afterwards adds the
        ! difference between the corrected and raw seam fluxes to dU. The raw
        ! fluxes are recomputed here from the nodal fields exactly as
        ! kface_flow_plane built them (the wall mask weights mf only, so pm
        ! is shared with the unmasked seam factors); nothing mutates the
        ! nodal inputs between the sweep and this pass, so the recompute
        ! matches the sweep's values. Same arithmetic as correcting the flux
        ! before accumulation, up to float reassociation. (nk=2, where the
        ! two seam cells coincide, is not supported and must be excluded by
        ! the caller.)

        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(in) :: P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dAk(3, ni-1, nj-1, nk)
        real, intent(in) :: wall_lo(ni-1, nj-1)
        real, intent(in) :: wall_hi(ni-1, nj-1)
        real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
        integer, intent(in) :: i_cusp_start, i_cusp_end

        integer :: i, j
        real :: pm_lo(6), mf_lo(3), mdot_lo
        real :: pm_hi(6), mf_hi(3), mdot_hi
        real :: pmd(6), mfm(3), mdm_lo, mdm_hi
        real :: raw_lo(5), raw_hi(5), corr_lo(5), corr_hi(5)
        real :: Vx_avg, Vr_avg, P_avg, mdot_avg

        do j = 1, nj-1
        do i = i_cusp_start, i_cusp_end-1
            ! Seam-averaged (unmasked) per-mass and mass-flux factors at k=1.
            pm_lo = 0.0e0; mf_lo = 0.0e0
            call accum(pm_lo, mf_lo, i,   j,   1, 1.0e0)
            call accum(pm_lo, mf_lo, i+1, j,   1, 1.0e0)
            call accum(pm_lo, mf_lo, i,   j+1, 1, 1.0e0)
            call accum(pm_lo, mf_lo, i+1, j+1, 1, 1.0e0)
            mdot_lo = mf_lo(1)*dAk(1,i,j,1) + mf_lo(2)*dAk(2,i,j,1) &
                    + mf_lo(3)*dAk(3,i,j,1)

            ! Seam-averaged (unmasked) factors at k=nk.
            pm_hi = 0.0e0; mf_hi = 0.0e0
            call accum(pm_hi, mf_hi, i,   j,   nk, 1.0e0)
            call accum(pm_hi, mf_hi, i+1, j,   nk, 1.0e0)
            call accum(pm_hi, mf_hi, i,   j+1, nk, 1.0e0)
            call accum(pm_hi, mf_hi, i+1, j+1, nk, 1.0e0)
            mdot_hi = mf_hi(1)*dAk(1,i,j,nk) + mf_hi(2)*dAk(2,i,j,nk) &
                    + mf_hi(3)*dAk(3,i,j,nk)

            ! Raw (wall-masked) seam fluxes exactly as the sweep accumulated
            ! them: masked mf, unmasked pm (pm_lo/pm_hi carry the identical
            ! values, the mask never touches pm), assembled as put_flow does.
            pmd = 0.0e0; mfm = 0.0e0
            call accum(pmd, mfm, i,   j,   1, wall_lo(i,j))
            call accum(pmd, mfm, i+1, j,   1, wall_lo(i,j))
            call accum(pmd, mfm, i,   j+1, 1, wall_lo(i,j))
            call accum(pmd, mfm, i+1, j+1, 1, wall_lo(i,j))
            mdm_lo = mfm(1)*dAk(1,i,j,1) + mfm(2)*dAk(2,i,j,1) &
                   + mfm(3)*dAk(3,i,j,1)
            raw_lo(1) = mdm_lo
            raw_lo(2) = pm_lo(1)*mdm_lo + pm_lo(5)*dAk(1,i,j,1)
            raw_lo(3) = pm_lo(2)*mdm_lo + pm_lo(5)*dAk(2,i,j,1)
            raw_lo(4) = pm_lo(3)*mdm_lo + pm_lo(6)*dAk(3,i,j,1)
            raw_lo(5) = pm_lo(4)*mdm_lo + Omega*pm_lo(6)*dAk(3,i,j,1)

            pmd = 0.0e0; mfm = 0.0e0
            call accum(pmd, mfm, i,   j,   nk, wall_hi(i,j))
            call accum(pmd, mfm, i+1, j,   nk, wall_hi(i,j))
            call accum(pmd, mfm, i,   j+1, nk, wall_hi(i,j))
            call accum(pmd, mfm, i+1, j+1, nk, wall_hi(i,j))
            mdm_hi = mfm(1)*dAk(1,i,j,nk) + mfm(2)*dAk(2,i,j,nk) &
                   + mfm(3)*dAk(3,i,j,nk)
            raw_hi(1) = mdm_hi
            raw_hi(2) = pm_hi(1)*mdm_hi + pm_hi(5)*dAk(1,i,j,nk)
            raw_hi(3) = pm_hi(2)*mdm_hi + pm_hi(5)*dAk(2,i,j,nk)
            raw_hi(4) = pm_hi(3)*mdm_hi + pm_hi(6)*dAk(3,i,j,nk)
            raw_hi(5) = pm_hi(4)*mdm_hi + Omega*pm_hi(6)*dAk(3,i,j,nk)

            ! Seam-averaged primitives (Vx, Vr, P) and (unmasked) mdot.
            Vx_avg   = 0.5e0*(pm_lo(1) + pm_hi(1))
            Vr_avg   = 0.5e0*(pm_lo(2) + pm_hi(2))
            P_avg    = 0.5e0*(pm_lo(5) + pm_hi(5))
            mdot_avg = 0.5e0*(mdot_lo + mdot_hi)

            ! Corrected seam fluxes. Mass (m=1): full TFLUX average of the
            ! raw fluxes, shared by both faces.
            corr_lo(1) = 0.5e0*(raw_lo(1) + raw_hi(1))
            corr_hi(1) = corr_lo(1)

            ! Axial momentum (m=2): rebuild both faces from averaged
            ! mdot*Vx + P*Ax, using each face's own dAk(1,...).
            corr_lo(2) = mdot_avg*Vx_avg + P_avg*dAk(1,i,j,1)
            corr_hi(2) = mdot_avg*Vx_avg + P_avg*dAk(1,i,j,nk)

            ! Radial momentum (m=3): same with dAk(2,...).
            corr_lo(3) = mdot_avg*Vr_avg + P_avg*dAk(2,i,j,1)
            corr_hi(3) = mdot_avg*Vr_avg + P_avg*dAk(2,i,j,nk)

            ! Angular momentum (m=4) and energy (m=5): full TFLUX average.
            corr_lo(4) = 0.5e0*(raw_lo(4) + raw_hi(4))
            corr_hi(4) = corr_lo(4)
            corr_lo(5) = 0.5e0*(raw_lo(5) + raw_hi(5))
            corr_hi(5) = corr_lo(5)

            ! Apply to the residual: the k=1 cell reads face 1 with + sign,
            ! the k=nk-1 cell reads face nk with - sign.
            dU(i,j,1,1) = dU(i,j,1,1) + (corr_lo(1) - raw_lo(1))
            dU(i,j,1,2) = dU(i,j,1,2) + (corr_lo(2) - raw_lo(2))
            dU(i,j,1,3) = dU(i,j,1,3) + (corr_lo(3) - raw_lo(3))
            dU(i,j,1,4) = dU(i,j,1,4) + (corr_lo(4) - raw_lo(4))
            dU(i,j,1,5) = dU(i,j,1,5) + (corr_lo(5) - raw_lo(5))
            dU(i,j,nk-1,1) = dU(i,j,nk-1,1) - (corr_hi(1) - raw_hi(1))
            dU(i,j,nk-1,2) = dU(i,j,nk-1,2) - (corr_hi(2) - raw_hi(2))
            dU(i,j,nk-1,3) = dU(i,j,nk-1,3) - (corr_hi(3) - raw_hi(3))
            dU(i,j,nk-1,4) = dU(i,j,nk-1,4) - (corr_hi(4) - raw_hi(4))
            dU(i,j,nk-1,5) = dU(i,j,nk-1,5) - (corr_hi(5) - raw_hi(5))
        end do
        end do

    contains
        pure subroutine accum(pm, mf, i, j, k, wfac)
            real, intent(inout) :: pm(6), mf(3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real :: dp, w, g
            dp = P(i,j,k) - P_offset
            ! Vx, Vr and r*Vt from the conserved state, spelled as
            ! kface_flow_plane's accum_corners spells them -- one reciprocal,
            ! then c2/c1, c3/c1, c4/c1 -- because that is what makes this
            ! pass's raw recompute match the flux the sweep actually
            ! accumulated. It used to read the nodal velocity arrays, which
            ! were the same numbers by a different route: Vx = c2/c1 either
            ! way, but r*Vt arrived as r*(c4/(c1*r)) rather than c4/c1, so
            ! the two spellings could part company in the last bit.
            g = 1.0e0/cons(i,j,k,1)
            pm(1) = pm(1) + 0.25e0*cons(i,j,k,2)*g
            pm(2) = pm(2) + 0.25e0*cons(i,j,k,3)*g
            pm(3) = pm(3) + 0.25e0*cons(i,j,k,4)*g
            ! Same spelling as the sweep's accum_corners (see there).
            pm(4) = pm(4) + 0.25e0*(cons(i,j,k,5)+P(i,j,k))*g
            pm(5) = pm(5) + 0.25e0*dp
            pm(6) = pm(6) + 0.25e0*r(i,j,k)*dp
            w = 0.25e0*wfac
            mf(1) = mf(1) + w*cons(i,j,k,2)
            mf(2) = mf(2) + w*cons(i,j,k,3)
            ! rho*Vt_rel = rho*Vt - Omega*rho*r = c4/r - Omega*c1*r
            mf(3) = mf(3) + w*(cons(i,j,k,4)/r(i,j,k) &
                               - Omega*cons(i,j,k,1)*r(i,j,k))
        end subroutine accum
    end subroutine correct_cusp_kface_du


    ! =================================================================
    ! Implicit residual smoothing primitives, shared by every consumer:
    ! smooth_residual_tri_tiled (the standalone three-direction smoother
    ! that scree.f90's coarse-MG path hands to its `smoother` dummy
    ! argument) and smooth_residual_scale_tri (the fine-grid path, which
    ! folds the change limiter's scaling into the i-solve's gather).
    ! They live here rather than being contained in one of them so the
    ! two cannot drift apart -- an earlier version of this file had the
    ! j/k sweeps duplicated into a bench arm, and identical-code folding
    ! then merged the two symbols and silently corrupted every timing
    ! that involved either. See docs/dev/plan_irs_traffic.md.
    ! =================================================================



    ! Thomas forward-sweep factors for the constant-coefficient Neumann
    ! tridiagonal along a line of length n: a = c = -sf, b = 1+2sf interior,
    ! b = 1+sf at the two ends. Returns cp (eliminated super-diagonal) and
    ! minv = 1/pivot, so a line solve is:
    !   x(1)   = d(1)*minv(1)
    !   x(i)   = (d(i) + sf*x(i-1))*minv(i)          i = 2..n   (forward)
    !   x(i)   = x(i) - cp(i)*x(i+1)                 i = n-1..1 (back-sub)
    ! An n=1 line has no neighbours -> operator is the identity (minv=1).
    subroutine irs_tri_coeffs(e, n, cp, minv)
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

        minv(1) = 1.0e0 / (1.0e0 + e)
        cp(1)   = -e * minv(1)
        do ii = 2, n-1
            minv(ii) = 1.0e0 / ((1.0e0 + 2.0e0*e) + e*cp(ii-1))
            cp(ii)   = -e * minv(ii)
        end do
        minv(n) = 1.0e0 / ((1.0e0 + e) + e*cp(n-1))
        cp(n)   = 0.0e0
    end subroutine irs_tri_coeffs


    ! ---- line kernels -----------------------------------------------
    ! Each takes its two operands as SEPARATE dummies, one defined and one
    ! not. Written inline as dU(i,j,k,m) against dU(i,j-1,k,m), GCC cannot
    ! prove the rows disjoint and emits a runtime overlap test plus two
    ! copies of every body ("loop versioned for vectorization because of
    ! possible aliasing"). As dummies the standard forbids the caller from
    ! aliasing them, so the test disappears: -1.95%, bitwise.

    subroutine irs_line_scale(x, mm, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: mm
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) * mm
        end do
    end subroutine irs_line_scale

    subroutine irs_line_fwd(x, xprev, e, mm, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: e, mm
        real, intent(in)    :: xprev(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = (x(ii) + e*xprev(ii)) * mm
        end do
    end subroutine irs_line_fwd

    subroutine irs_line_back(x, xnext, cc, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: cc
        real, intent(in)    :: xnext(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) - cc*xnext(ii)
        end do
    end subroutine irs_line_back

    ! One Jacobi sweep of (1 - sf*d2_i) y = d, Neumann ends, over a single
    ! row. Reads x, writes y (separate dummies, same reason as the three
    ! above: no aliasing to disprove, so it vectorises along the unit-stride
    ! axis with no transpose). See IRS_NJAC for why this replaces the exact
    ! i-solve in smooth_residual_tri_tiled.
    subroutine irs_jac_line(d, x, y, n, e, rint, rend)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: e, rint, rend
        real, intent(in)    :: d(n), x(n)
        real, intent(out)   :: y(n)
        integer :: p

        if (n == 1) then
            y(1) = d(1)
            return
        end if
        y(1) = (d(1) + e*x(2)) * rend
        do p = 2, n-1
            y(p) = (d(p) + e*(x(p-1) + x(p+1))) * rint
        end do
        y(n) = (d(n) + e*x(n-1)) * rend
    end subroutine irs_jac_line


    ! ---- blocked transpose ------------------------------------------
    ! src(i,jj) -> tl(jj,i) through IRS_TB x IRS_TB blocks. The staging
    ! read is IRS_TB contiguous floats per lane (one vector load); the
    ! write-out is IRS_TB contiguous lanes per i (one vector store). Only
    ! the in-block transpose is strided, over 256 bytes that never leave
    ! the register file.
    !
    ! Written the obvious way -- do i; do jj; tl(jj,i) = src(i,jj) -- the
    ! inner loop strides src by nci elements (1088 bytes at nci=272) and
    ! the compiler emits one scalar load per element. That is invisible in
    ! the vectorization report, which calls the surrounding loop
    ! "vectorized": the i-solve took 53.6% of the smoother's time while
    ! moving 20% of its traffic, and the disassembly showed 266 scalar
    ! vmovss against 190 vector vmovups with no gather instructions at all.

    ! ldt is the tile's leading dimension, always IRS_BJ. Passed rather
    ! than taken from the module parameter because f2py wraps every public
    ! module procedure and cannot resolve a parameter used in a dummy's
    ! dimension.
    subroutine irs_gather_tile(src, tl, ldt, n, nb_in)
        implicit none
        integer, intent(in) :: ldt, n, nb_in
        real, intent(in)    :: src(n, nb_in)
        real, intent(inout) :: tl(ldt, n)
        real    :: blk(IRS_TB, IRS_TB)
        integer :: ib, jb, ii, jj, nfull_i, nfull_j

        nfull_i = (n / IRS_TB) * IRS_TB
        nfull_j = (nb_in / IRS_TB) * IRS_TB
        do jb = 1, nfull_j, IRS_TB
            do ib = 1, nfull_i, IRS_TB
                do jj = 1, IRS_TB
                    do ii = 1, IRS_TB
                        blk(ii,jj) = src(ib+ii-1, jb+jj-1)
                    end do
                end do
                do ii = 1, IRS_TB
                    do jj = 1, IRS_TB
                        tl(jb+jj-1, ib+ii-1) = blk(ii,jj)
                    end do
                end do
            end do
        end do
        do jj = 1, nfull_j
            do ii = nfull_i+1, n
                tl(jj,ii) = src(ii,jj)
            end do
        end do
        do jj = nfull_j+1, nb_in
            do ii = 1, n
                tl(jj,ii) = src(ii,jj)
            end do
        end do
    end subroutine irs_gather_tile

    ! As irs_gather_tile, but applies the change limiter's pointwise
    ! scaling on the way through, while each value is already in a
    ! register. That is the whole point of the fine-grid path: the scaling
    ! pass and the i-solve each cost a full-volume read and write on their
    ! own, and fused they cost one between them.
    !
    ! The arithmetic is byte-for-byte set_residual's scale_du, INCLUDING
    ! keeping fdamp/dampin as a division rather than a hoisted reciprocal
    ! multiply -- the two do not agree bitwise and the exactness of this
    ! whole path depends on them matching.
    subroutine irs_gather_tile_scaled(src, dtv, tl, ldt, n, nb_in, rav, dampin)
        implicit none
        integer, intent(in) :: ldt, n, nb_in
        real, intent(in)    :: src(n, nb_in)
        real, intent(in)    :: dtv(n, nb_in)
        real, intent(inout) :: tl(ldt, n)
        real, intent(in)    :: rav, dampin
        real    :: blk(IRS_TB, IRS_TB)
        real    :: v, chg, fdamp
        integer :: ib, jb, ii, jj, nfull_i, nfull_j

        nfull_i = (n / IRS_TB) * IRS_TB
        nfull_j = (nb_in / IRS_TB) * IRS_TB
        do jb = 1, nfull_j, IRS_TB
            do ib = 1, nfull_i, IRS_TB
                do jj = 1, IRS_TB
                    do ii = 1, IRS_TB
                        v     = src(ib+ii-1, jb+jj-1)
                        chg   = abs(v * dtv(ib+ii-1, jb+jj-1))
                        fdamp = chg * rav
                        blk(ii,jj) = v / (1.0e0 + fdamp/dampin)
                    end do
                end do
                do ii = 1, IRS_TB
                    do jj = 1, IRS_TB
                        tl(jb+jj-1, ib+ii-1) = blk(ii,jj)
                    end do
                end do
            end do
        end do
        do jj = 1, nfull_j
            do ii = nfull_i+1, n
                v     = src(ii,jj)
                chg   = abs(v * dtv(ii,jj))
                fdamp = chg * rav
                tl(jj,ii) = v / (1.0e0 + fdamp/dampin)
            end do
        end do
        do jj = nfull_j+1, nb_in
            do ii = 1, n
                v     = src(ii,jj)
                chg   = abs(v * dtv(ii,jj))
                fdamp = chg * rav
                tl(jj,ii) = v / (1.0e0 + fdamp/dampin)
            end do
        end do
    end subroutine irs_gather_tile_scaled

    subroutine irs_scatter_tile(dst, tl, ldt, n, nb_in)
        implicit none
        integer, intent(in) :: ldt, n, nb_in
        real, intent(inout) :: dst(n, nb_in)
        real, intent(in)    :: tl(ldt, n)
        real    :: blk(IRS_TB, IRS_TB)
        integer :: ib, jb, ii, jj, nfull_i, nfull_j

        nfull_i = (n / IRS_TB) * IRS_TB
        nfull_j = (nb_in / IRS_TB) * IRS_TB
        do jb = 1, nfull_j, IRS_TB
            do ib = 1, nfull_i, IRS_TB
                do ii = 1, IRS_TB
                    do jj = 1, IRS_TB
                        blk(ii,jj) = tl(jb+jj-1, ib+ii-1)
                    end do
                end do
                do jj = 1, IRS_TB
                    do ii = 1, IRS_TB
                        dst(ib+ii-1, jb+jj-1) = blk(ii,jj)
                    end do
                end do
            end do
        end do
        do jj = 1, nfull_j
            do ii = nfull_i+1, n
                dst(ii,jj) = tl(jj,ii)
            end do
        end do
        do jj = nfull_j+1, nb_in
            do ii = 1, n
                dst(ii,jj) = tl(jj,ii)
            end do
        end do
    end subroutine irs_scatter_tile


    ! ---- the tile recurrence, one i-column against the previous -------
    subroutine irs_tile_solve(tl, ldt, e, cpi, minvi, nci, nb_in)
        implicit none
        integer, intent(in) :: ldt, nci, nb_in
        real, intent(in)    :: e
        real, intent(in)    :: cpi(nci), minvi(nci)
        real, intent(inout) :: tl(ldt, nci)
        integer :: i

        call irs_line_scale(tl(1,1), minvi(1), nb_in)
        do i = 2, nci
            call irs_line_fwd(tl(1,i), tl(1,i-1), e, minvi(i), nb_in)
        end do
        do i = nci-1, 1, -1
            call irs_line_back(tl(1,i), tl(1,i+1), cpi(i), nb_in)
        end do
    end subroutine irs_tile_solve


    ! ---- j+k solves, fused over i-strips ------------------------------
    ! Each strip is carried through BOTH direction solves before the next
    ! strip is touched, so it is read from and written to memory once for
    ! the pair instead of once each.
    !
    ! The saving is smaller than the touch count suggests, and it is worth
    ! knowing why. Run separately, the j-solve's back-substitution
    ! re-reads what its forward pass just wrote out of one (nci,ncj)
    ! plane -- 70 KB, L2-resident -- not out of DRAM, and the k-solve gets
    ! the same plane-to-plane reuse. So each already cost ~1R+1W of DRAM
    ! and fusing them saves one of two, not three of four. Measured
    ! -12.2% on the whole smoother, bitwise. A second effect probably
    ! contributes: the k back-substitution DESCENDS over a 3.9 MB
    ! per-component volume against prefetchers that favour ascending
    ! streams, and inside a strip that descent stays in L2.
    !
    ! Bitwise: for fixed (i,k,m) the j-recurrence touches only that j-line
    ! and lines at different i are independent, so restricting to a strip
    ! changes only the order in which independent lines are solved; same
    ! for k. Inside a strip the k-solve consumes columns the j-solve has
    ! already finished, so the dependency is respected exactly.
    subroutine irs_jk_strips(dU, e, cpj, minvj, cpk, minvk, nci, ncj, nck)
        implicit none
        integer, intent(in) :: nci, ncj, nck
        real, intent(in)    :: e
        real, intent(in)    :: cpj(ncj), minvj(ncj), cpk(nck), minvk(nck)
        real, intent(inout) :: dU(nci, ncj, nck, 5)
        integer :: i0, nw, j, k, m
        real    :: mm, cc

        do m = 1, 5
        do i0 = 1, nci, IRS_W
            nw = min(IRS_W, nci - i0 + 1)

            if (ncj >= 2) then
                do k = 1, nck
                    call irs_line_scale(dU(i0,1,k,m), minvj(1), nw)
                    do j = 2, ncj
                        call irs_line_fwd(dU(i0,j,k,m), dU(i0,j-1,k,m), e, &
                                          minvj(j), nw)
                    end do
                    do j = ncj-1, 1, -1
                        call irs_line_back(dU(i0,j,k,m), dU(i0,j+1,k,m), &
                                           cpj(j), nw)
                    end do
                end do
            end if

            if (nck >= 2) then
                mm = minvk(1)
                do j = 1, ncj
                    call irs_line_scale(dU(i0,j,1,m), mm, nw)
                end do
                do k = 2, nck
                    mm = minvk(k)
                    do j = 1, ncj
                        call irs_line_fwd(dU(i0,j,k,m), dU(i0,j,k-1,m), e, &
                                          mm, nw)
                    end do
                end do
                do k = nck-1, 1, -1
                    cc = cpk(k)
                    do j = 1, ncj
                        call irs_line_back(dU(i0,j,k,m), dU(i0,j,k+1,m), &
                                           cc, nw)
                    end do
                end do
            end if

        end do
        end do
    end subroutine irs_jk_strips


end module residual_helpers


! =====================================================================
! v3 unscaled residual: single pass over precomputed nodal primitives.
! Per-mass (perm) and mass-flux (mflux) factors are read directly from
! cached nodal arrays (vx, vr, vt, ho) and assembled per face; the
! relative tangential velocity (Vt - Omega*r) is derived inline rather
! than read from its own stored array (see accum() in each face helper).
!
! k-slab cache blocking and rolling-buffer fusion
! -----------------------------------------------
! The three face-direction sweeps are tiled over slabs of kb cell planes
! (1 <= kb <= nk-1) so a slab's nodal input planes stay cache-resident
! across all three directions. Within a slab each direction fuses its
! face-flow computation with its dU accumulate through a rolling buffer,
! so no slab-sized flow scratch exists:
!   - i+j-directions: fused per (j,k) row into a single dU write -- the
!     i-face row (rows slot 1) and the rolling j-face pair (rows slots
!     2/3) are both consumed in one expression (i-diff + f_body + j-diff),
!     cutting dU from three touches per slab (write, RMW, RMW) to two;
!   - k-direction: an alternating face-plane pair (planes slots 1/2),
!     persisting across the slab boundary (the intervening i/j phases
!     touch only rows), so the shared face plane carries automatically.
!
! Unlike the staged version, dU accumulates in two stages (the fused i+j
! write dU = i-diff + f_body + j-diff, then += k-diff) instead of one
! fused seven-term expression: the per-face flow values are identical, but
! the final sum is reassociated, so results differ from the staged kernel
! by a few ulp on near-cancelling cells. This is a deliberate, bounded
! float32 tolerance in exchange for keeping the slab tiling together
! with the rolling buffers (the bitwise-preserving alternative needed a
! single all-direction sweep, which measured slower at large planes).
!
! The cusp seam (k=1 face coupled to k=nk) is non-local in k and is
! applied as a deferred O(surface) correction to dU after the sweep
! (see correct_cusp_kface_du).
! =====================================================================
!
! Change limiter folded in
! ------------------------
! damp_residual's global reduction (block mean of |dU*dt_vol|) is
! accumulated here, inside the fused dU write, while each value is still in
! a register; only the pointwise scaling remains as a second pass. That
! removes a full-volume dU read: -6% serial, -11% at 100-rank saturation,
! winning 100/100 ranks (docs section 24). dampin <= 0 disables it and
! reproduces the un-damped kernel bitwise.
!
! *** THIS REORDERS THE POST-PROCESSING. *** Grid.update_residual used to
! run IRS then damp; folding damp in here necessarily makes it damp then
! IRS. IRS is linear and the limiter is nonlinear with a global mean, so
! the composed operator is genuinely different -- at sf_resid = 1.0,
! dampin = 25 (run.py defaults) the two orderings differ by ~19% of the
! field scale, and the difference grows monotonically with sf (zero at
! sf = 0). This is a deliberate numerics change, not a rounding artifact,
! and it is NOT covered by the test suite: no test drives update_residual
! with dampin set and sf > 0 together. Convergence must be verified
! separately.
!
! Second known inexactness: the reduction is accumulated before
! correct_cusp_kface_du modifies dU on the two seam planes, so on a cusped
! block the mean omits that O(surface) correction.

subroutine set_residual( &
    cons, P, P_offset, &
    r, Omega, dAi, dAj, dAk, &
    f_body, &
    dU, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    dt_vol, dampin, ravg_out, &
    kb, njp, ni, nj, nk &
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
    ! block.scratch, the one arena, which is pure transient scratch -- the
    ! layout here is private to this call. njp is planes' padded j-extent, chosen by the
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
    ! Reciprocal block means of |dU*dt_vol| per conserved variable, the
    ! change limiter's own scaling factors. Always returned, because the
    ! reduction that produces them is accumulated during the dU sweep
    ! whatever dampin is. The fine-grid IRS path (smooth_residual_scale_tri)
    ! takes them and applies the limiter itself, fused into its i-solve, so
    ! that the scaling does not need a full-volume traversal of its own --
    ! call this with dampin <= 0 to get the factors without the scaling.
    real, intent(out) :: ravg_out(5)
    integer, intent(in) :: kb, njp, ni, nj, nk

    integer :: i, j, k, m, k0, k1, ja, jb, pa, pb, stmp
    integer :: jp, jp0, jp1, jbw
    integer :: ncell
    real :: avg(5), ravg(5)

    do m = 1, 5
        avg(m) = 0.0e0
    end do

    ! ===== j-panel over the k walk =====
    ! What the walk CARRIES from one k to the next is the rolling k-face
    ! plane pair, ni*njp*5*2*4 bytes -- 720 KB on a 273x65x57 block, which
    ! is nowhere near a 256 KB L2. So plane k, written on step k, is
    ! evicted before step k+1 reads it, and every one of the five
    ! components round-trips through L3; with eight ranks on a socket that
    ! carry alone asks for more than the 20 MB they share.
    !
    ! Walking the whole k sweep over a panel of j rows bounds that carry
    ! instead. It does not change how much nodal data is read -- each panel
    ! streams its own j-strip, and the strips are disjoint bar one shared
    ! row -- it bounds how much is LIVE at once, which is the thing the L2
    ! actually has to hold.
    !
    ! The panel is sized by AREA, not by a row count, because the carry is
    ! ni*jbw planes wide: a fixed jbw would bound it on a thin block and
    ! blow it on a fat one. Floored at RES_JMIN rows, because the panel is
    ! not free -- see the comment there.
    jbw = min(nj-1, max(RES_JMIN, RES_JAREA / max(ni, 1)))
    do jp = 1, nj-1, jbw
    jp0 = jp
    jp1 = min(jp + jbw - 1, nj-1)

    pa = 1
    pb = 2

    ! Prime the rolling k-face plane with face k=1 before the slab sweep
    ! (the fused loop below always has plane k in slot pa on entry to cell
    ! k, needing only face k+1 freshly computed into pb).
    call kface_flow_plane(P, P_offset, r, cons, &
                          Omega, dAk, wallk1, wallnk, planes(:,:,:,pa), &
                          1, jp0, jp1, njp, ni, nj, nk)

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
        ! Prime the rolling j-face pair with the panel's lowest j face.
        call jface_flow_row(P, P_offset, r, cons, &
                            Omega, dAj, wallj1, wallnj, rows(:,:,ja), &
                            jp0, k, ni, nj, nk)
        ! Advance the rolling k-face pair: pa already holds face k (primed
        ! before the sweep, or carried from the previous k iteration); pb
        ! gets face k+1 computed fresh.
        call kface_flow_plane(P, P_offset, r, cons, &
                              Omega, dAk, wallk1, wallnk, planes(:,:,:,pb), &
                              k+1, jp0, jp1, njp, ni, nj, nk)
        do j = jp0, jp1
            call iface_flow_row(P, P_offset, r, cons, &
                                Omega, dAi, walli1(j,k), wallni(j,k), &
                                rows(:,:,1), j, k, ni, nj, nk)
            call jface_flow_row(P, P_offset, r, cons, &
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

    end do  ! ===== end j-panel =====

    ! Cusp seam: non-local in k (couples the k=1 and k=nk faces), applied as
    ! a deferred O(surface) correction to dU after the sweep. nk=2 (the two
    ! seam cells coincide) is not supported.
    if (i_cusp_start > 0 .and. nk > 2) then
        call correct_cusp_kface_du(P, P_offset, r, cons, &
                                   Omega, dAk, wallk1, wallnk, dU, &
                                   i_cusp_start, i_cusp_end, ni, nj, nk)
    end if

    ! ---- change limiter, second half ----
    ! The reduction above was accumulated during the sweep, so only the
    ! pointwise scaling pass remains. NOTE the cusp correction just
    ! modified dU on the two seam cell planes, which the reduction did not
    ! see; that is an O(surface) discrepancy in a block-mean over O(volume)
    ! cells, and is corrected below for exactness.
    ncell = (ni-1)*(nj-1)*(nk-1)
    do m = 1, 5
        avg(m) = avg(m) / ncell
        if (avg(m) > 0.0e0) then
            ravg(m) = 1.0e0 / avg(m)
        else
            ravg(m) = 0.0e0
        end if
    end do
    do m = 1, 5
        ravg_out(m) = ravg(m)
    end do
    if (dampin > 0.0e0) then
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
! Negative-feedback change limiter (ported from multall's DAMP loop).
!
! Soft-clips outlier per-cell changes so the explicit march stays stable
! without globally cutting the timestep. The per-step change is
! dU * dt_vol; per conserved variable, the block mean of its magnitude is
! avg, and each cell is shrunk by 1/(1 + fdamp/dampin) with
! fdamp = |change|/avg. Cells near the mean are barely touched; large
! outliers saturate towards dampin*avg. Operates in place on dU.
!
! The five components share each (j,k) plane traversal rather than each
! getting its own full-volume sweep: the reduction and the scaling are one
! pass apiece instead of five, so dt_vol is loaded once per plane instead
! of five times. -28.7% serial, -30.8% at 100-rank saturation
! (docs/dev/viscous_kernels.md section 21).
!
! m stays OUTSIDE the i loop. dU is component-last, so dU(:,j,k,m) is
! contiguous in i only for fixed m; making m innermost strides the i reads
! by the whole volume and measured +208%. The flat-field guard is folded
! into ravg (0 for a flat component => identity soft-clip) so the swept
! region stays branch-free.
! =====================================================================
subroutine damp_residual(dU, dt_vol, dampin, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in)    :: dampin

    integer :: i, j, k, m, ncell
    real :: avg(5), ravg(5), chg, fdamp

    ncell = (ni-1)*(nj-1)*(nk-1)

    ! ---- Sweep 1: all five block means in one pass over dU/dt_vol ----
    do m = 1, 5
        avg(m) = 0.0e0
    end do
    ! m stays OUTSIDE the i loop: dU is component-LAST, so dU(:,j,k,m) is
    ! contiguous in i only for fixed m. Putting m innermost would make each
    ! (i,j,k) touch five locations ~(ni-1)*(nj-1)*(nk-1) elements apart and
    ! destroy the i-vectorization -- measured at +208% before this was fixed
    ! (section 2's "any layout change that strides the i reads is suspect").
    ! The saving here is therefore NOT fewer dU sweeps but fewer dt_vol
    ! reads: the k/j planes of dt_vol are hoisted and shared across m.
    do k = 1, nk-1
    do j = 1, nj-1
    do m = 1, 5
    do i = 1, ni-1
        avg(m) = avg(m) + abs(dU(i,j,k,m) * dt_vol(i,j,k))
    end do
    end do
    end do
    end do

    ! A flat field (avg = 0) would divide by zero. Production guards this
    ! with `cycle`; here the reciprocal is folded into a factor that is
    ! simply 0 for such a component, which makes fdamp 0 and the soft-clip
    ! the identity -- same outcome, no branch inside the sweep. Branch-free
    ! matters because the caller already skips this routine entirely when
    ! damping is off, so every execution here is one that does real work.
    do m = 1, 5
        avg(m) = avg(m) / ncell
        if (avg(m) > 0.0e0) then
            ravg(m) = 1.0e0 / avg(m)
        else
            ravg(m) = 0.0e0
        end if
    end do

    ! ---- Sweep 2: soft-clip every component in one pass ----
    do k = 1, nk-1
    do j = 1, nj-1
    do m = 1, 5
        do i = 1, ni-1
            chg   = abs(dU(i,j,k,m) * dt_vol(i,j,k))
            fdamp = chg * ravg(m)
            ! fdamp/dampin kept as a division, not a hoisted reciprocal
            ! multiply, so the arithmetic matches production exactly.
            dU(i,j,k,m) = dU(i,j,k,m) / (1.0e0 + fdamp/dampin)
        end do
    end do
    end do
    end do

end subroutine damp_residual


! =====================================================================
! Implicit residual smoothing (Jameson IRS) -- EXACT factored tridiagonal
! (ADI), with the i-direction solve transpose-tiled so it vectorises.
!
! The unfactored operator (1 - sf*grad^2) is applied as the ADI-style
! factored product
!   (1 - sf*d2_i) (1 - sf*d2_j) (1 - sf*d2_k) R* = R
! where d2_d is the 1D second difference along direction d with zero-
! gradient (Neumann) ends. The three orthogonal 1D operators commute, so
! the inverse is three successive EXACT tridiagonal (Thomas) solves, one
! per direction, in place on dU -- no sweep count, each direction solved to
! the last bit in O(n) per line. The matrix is identical for every line in
! a given direction (a=c=-sf, b=1+2sf interior, b=1+sf at the ends), so its
! Thomas factors cp(.) and reciprocal pivots minv(.) are built ONCE per
! direction by tri_coeffs and reused for all lines. Constant fields are
! preserved exactly and IRS(0)=0, so the converged solution is unchanged.
!
! The j- and k-solves run their recurrences over operands contiguous in the
! stride-1 i index, so they vectorise as-is, and they are FUSED over
! i-strips so a strip is read and written once for the pair rather than
! once each (irs_jk_strips). The i-solve recurrence runs along the
! unit-stride axis and cannot vectorise directly, so an IRS_BJ-wide block
! of j-lines is transposed into a small (IRS_BJ,nci) pad; the recurrence's
! innermost loop then runs over the IRS_BJ contiguous, independent lanes --
! vectorises and hides the FMA-latency chain -- and the tile is scattered
! back. Both transposes are themselves blocked (irs_gather_tile).
!
! Commit 3bf9b22 tried replacing this i-solve with two Jacobi sweeps
! (irs_jac_line) for a 10.7% win under socket contention, on the reasoning
! that a truncated iteration only under-relaxes the smoother and so cannot
! change the converged solution (IRS(0)=0 held in isolation). In practice
! the coarse-MG correction this smoother sits inside compounded that
! under-relaxation across levels and steps enough to stall convergence --
! an initial residual dip followed by a slow rise rather than a pure speed
! loss -- so it is reverted back to the exact solve here. See git history
! (3bf9b22 and this revert) for the Jacobi arm if the contention win is
! ever worth re-pricing against a fuller convergence check.
!
! This routine is now a driver: every primitive lives in residual_helpers,
! shared with smooth_residual_scale_tri, which is the fine-grid path and
! folds the change limiter's scaling into the i-solve's gather. THIS one is
! what scree.f90's coarse-MG path hands to its `smoother` dummy argument,
! where there is no limiter to fold in.
!
! Scratch: the Thomas solve is in place on dU; work is a 1D buffer holding
! the six coefficient vectors back-to-back, >= 2*((ni-1)+(nj-1)+(nk-1))
! elements (e.g. a leading slice of block.scratch via util.carve_view).
! =====================================================================
subroutine smooth_residual_tri_tiled(dU, sf, work, ni, nj, nk)

    use residual_helpers

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: k, m, nci, ncj, nck, j0, nb
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: tile(IRS_BJ, ni-1)           ! (lane, i) transposed i-solve pad

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
    call irs_tri_coeffs(sf, nci, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci))
    call irs_tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call irs_tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))

    ! ---- i-direction: blocked-transpose gather, solve in the tile,
    ! blocked-transpose scatter back. ----
    if (nci >= 2) then
        do m = 1, 5
        do k = 1, nck
        do j0 = 1, ncj, IRS_BJ
            nb = min(IRS_BJ, ncj - j0 + 1)
            ! dU(:, j0:j0+nb-1, k, m) is contiguous with leading dimension
            ! nci, so it binds to a (nci, nb) dummy by sequence association --
            ! no array temporary, and the block kernels see plain 2D arrays.
            call irs_gather_tile(dU(1,j0,k,m), tile, IRS_BJ, nci, nb)
            call irs_tile_solve(tile, IRS_BJ, sf, work(bcpi+1:bcpi+nci), &
                                work(bmii+1:bmii+nci), nci, nb)
            call irs_scatter_tile(dU(1,j0,k,m), tile, IRS_BJ, nci, nb)
        end do
        end do
        end do
    end if

    ! ---- j- and k-directions, fused over i-strips ----
    call irs_jk_strips(dU, sf, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj), &
                       work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck), &
                       nci, ncj, nck)

end subroutine smooth_residual_tri_tiled


! =====================================================================
! Fine-grid IRS: the change limiter's scaling pass FUSED into the
! i-direction solve, then the strip-fused j and k solves.
!
! Grid.update_residual used to run set_residual (whose trailing scale_du
! applies the limiter over the whole volume) and then
! smooth_residual_tri_tiled. That is three full-volume read/write pairs
! downstream of the residual sweep: scale, then i, then j+k. The scaling
! is pointwise and the i-solve for a row depends only on that row's
! scaled values, so the two can share one traversal -- the tile gather IS
! the scale's read and the tile scatter IS its write:
!
!   scale (1R+1W) + i (1R+1W) + j+k (1R+1W)  ->  scale(x)i (1R+1W) + j+k (1R+1W)
!
! set_residual therefore hands out the block means it already accumulates
! during its sweep (ravg), and skips its own scaling pass; this routine
! applies it. dampin <= 0 means no limiter, and the plain gather is used.
!
! BITWISE against the unfused pair, and that is not a nicety: the scaling
! arithmetic here is byte-for-byte scale_du's, the i-solve is unchanged,
! and applying the two per (k, j0-block, m) rather than volume-then-volume
! is the same computation on the same operands because the scaling is
! pointwise and the blocks are disjoint. The ORDER that matters is
! preserved exactly -- sweep, cusp correction, limiter, IRS-i, IRS-j,
! IRS-k -- so this does not reopen the damp-vs-IRS ordering question
! documented at the head of set_residual.
!
! Loop nest is (k, j0, m), NOT (m, k, j0) as the standalone smoother uses:
! dt_vol is component-independent, so putting m innermost lets one
! (nci,nb) block of it serve all five components out of cache. With m
! outermost dt_vol would be streamed five times, which is most of what the
! fusion just saved -- the same trap damp_residual documents.
! =====================================================================
subroutine smooth_residual_scale_tri(dU, dt_vol, ravg, dampin, sf, work, &
                                     ni, nj, nk)

    use residual_helpers

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf, dampin
    real, intent(in)    :: ravg(5)
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(in)    :: dt_vol(ni-1, nj-1, nk-1)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: k, m, nci, ncj, nck, j0, nb
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: tile(IRS_BJ, ni-1)

    if (sf <= 0.0e0) return

    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    bcpi = 0
    bmii = nci
    bcpj = 2*nci
    bmij = 2*nci + ncj
    bcpk = 2*nci + 2*ncj
    bmik = 2*nci + 2*ncj + nck
    call irs_tri_coeffs(sf, nci, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci))
    call irs_tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call irs_tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))

    if (nci >= 2) then
        do k = 1, nck
        do j0 = 1, ncj, IRS_BJ
            nb = min(IRS_BJ, ncj - j0 + 1)
            do m = 1, 5
                ! The dampin test is loop-invariant over the whole nest and
                ! sits outside every vector loop, so it costs a predicted
                ! branch per block and keeps one code path for both cases.
                if (dampin > 0.0e0) then
                    call irs_gather_tile_scaled(dU(1,j0,k,m), dt_vol(1,j0,k), &
                                                tile, IRS_BJ, nci, nb, &
                                                ravg(m), dampin)
                else
                    call irs_gather_tile(dU(1,j0,k,m), tile, IRS_BJ, nci, nb)
                end if
                call irs_tile_solve(tile, IRS_BJ, sf, work(bcpi+1:bcpi+nci), &
                                    work(bmii+1:bmii+nci), nci, nb)
                call irs_scatter_tile(dU(1,j0,k,m), tile, IRS_BJ, nci, nb)
            end do
        end do
        end do
    end if

    call irs_jk_strips(dU, sf, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj), &
                       work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck), &
                       nci, ncj, nck)

end subroutine smooth_residual_scale_tri
