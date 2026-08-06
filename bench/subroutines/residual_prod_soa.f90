! =====================================================================
! BENCHMARK ARM `prodsoa` -- production's kernel, unchanged, on SoA
! face-area geometry.
!
! Nothing in production calls this. See docs/dev/plan_nodal_primitives.md
! and docs/dev/kernel_benchmark_methodology.md.
!
! Why it exists
! -------------
! With `nodal` and `tbaos` the ladder attributes section 26's win across
! three steps -- nodal primitives, the five-pass split, SoA geometry --
! but the third is measured at ONE corner only, inside the five-pass
! family (`multall` minus `tbaos`). Nothing says what the layout does to a
! fused sweep, and the answer is not obvious in sign:
!
!   a five-pass momentum loop wants ONE dA component, and AoS makes it
!   pay for a line of three -- section 26.1's argument, and the reason
!   multall stores AIX/AIR/AIT separately;
!
!   a fused sweep wants ALL THREE in one expression
!       mdot = mf1*dA(1,i,j,k) + mf2*dA(2,i,j,k) + mf3*dA(3,i,j,k)
!   so component-first is arguably the RIGHT layout there, and SoA turns
!   one stream into three.
!
! This arm settles that, and it is the only variant in the whole study
! that would need NO residual rewrite at all: it is a change to
! geometry.f90 and the dA consumers, with production's kernel untouched
! but for its indexing. If it wins, that is the cheapest proposal
! available. If it loses, the layout effect only exists in the presence
! of the pass split, which is worth knowing before anyone proposes a
! layout change on the strength of `multall` minus `tbaos`.
!
! What differs from residual.f90
! ------------------------------
! The geometry indexing, and nothing else. dA(1,i,j,k) becomes
! dA1(i,j,k), etc, in all three face helpers. accum_corners is untouched
! -- it never reads dA -- so the cons-derived primitives, the four
! reciprocals and the cons4/r in mf3 all stay exactly as section 20 left
! them. The k-slab sweep, the rolling buffers, the fused dU write, the
! folded change limiter and the deferred cusp correction are verbatim.
!
! Fairness (Rule 3): splitting the nine component arrays happens outside
! the timed region, as it does for the `multall` arm, because face areas
! are grid geometry -- built once, never rebuilt per step. The AoS dAk is
! still passed, solely for correct_cusp_kface_du, which is production's
! own routine called unmodified by every arm.
!
! NUMERICS: identical arithmetic to production in every term, same
! operands in the same order. Expect agreement at the last-bit
! reassociation level at worst, and check the mass component: it is the
! six-point sum of face mdots, each of which is the same three-term dot
! product here as in production.
! =====================================================================

module residual_prod_soa_helpers
    implicit none
    private
    public :: iface_flow_row_soa, jface_flow_row_soa, kface_flow_plane_soa

contains

    pure subroutine iface_flow_row_soa(ho, P, P_offset, r, &
                                       cons, Omega, dA1, dA2, dA3, &
                                       wall_lo, wall_hi, row, j, k, ni, nj, nk)
        ! i-face row (j,k); i=1 / i=ni wall-masked. Corners (i, j:j+1, k:k+1).

        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA1(ni, nj-1, nk-1), dA2(ni, nj-1, nk-1), dA3(ni, nj-1, nk-1)
        real, intent(in) :: wall_lo, wall_hi
        real, intent(inout) :: row(ni, 5)

        integer :: i
        real :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3, mdot

        ! pm/mf scalarized and the corners hand-unrolled, verbatim from
        ! production -- see residual.f90's iface_flow_row for why.

        ! Low boundary i=1
        call accum_corners(1, j, k, wall_lo, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
        mdot = mf1*dA1(1,j,k) + mf2*dA2(1,j,k) + mf3*dA3(1,j,k)
        row(1,1) = mdot
        row(1,2) = pm1*mdot + pm5*dA1(1,j,k)
        row(1,3) = pm2*mdot + pm5*dA2(1,j,k)
        row(1,4) = pm3*mdot + pm6*dA3(1,j,k)
        row(1,5) = pm4*mdot + Omega*pm6*dA3(1,j,k)

        ! Interior i=2..ni-1
        !DIR$ IVDEP
        do i = 2, ni-1
            call accum_corners(i, j, k, 1.0e0, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
            mdot = mf1*dA1(i,j,k) + mf2*dA2(i,j,k) + mf3*dA3(i,j,k)
            row(i,1) = mdot
            row(i,2) = pm1*mdot + pm5*dA1(i,j,k)
            row(i,3) = pm2*mdot + pm5*dA2(i,j,k)
            row(i,4) = pm3*mdot + pm6*dA3(i,j,k)
            row(i,5) = pm4*mdot + Omega*pm6*dA3(i,j,k)
        end do

        ! High boundary i=ni
        call accum_corners(ni, j, k, wall_hi, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
        mdot = mf1*dA1(ni,j,k) + mf2*dA2(ni,j,k) + mf3*dA3(ni,j,k)
        row(ni,1) = mdot
        row(ni,2) = pm1*mdot + pm5*dA1(ni,j,k)
        row(ni,3) = pm2*mdot + pm5*dA2(ni,j,k)
        row(ni,4) = pm3*mdot + pm6*dA3(ni,j,k)
        row(ni,5) = pm4*mdot + Omega*pm6*dA3(ni,j,k)

    contains
        pure subroutine accum_corners(i, j, k, wfac, pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3)
            ! Production's, byte for byte: Vx/Vr/r*Vt derived from cons.
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real, intent(out) :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3
            real :: dp1, dp2, dp3, dp4, w
            real :: g1, g2, g3, g4
            dp1 = P(i,j,k)     - P_offset
            dp2 = P(i,j+1,k)   - P_offset
            dp3 = P(i,j,k+1)   - P_offset
            dp4 = P(i,j+1,k+1) - P_offset
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
            mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                + w*(cons(i,j+1,k,4)/r(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                + w*(cons(i,j,k+1,4)/r(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                + w*(cons(i,j+1,k+1,4)/r(i,j+1,k+1) - Omega*cons(i,j+1,k+1,1)*r(i,j+1,k+1))
        end subroutine accum_corners
    end subroutine iface_flow_row_soa


    pure subroutine jface_flow_row_soa(ho, P, P_offset, r, &
                                       cons, Omega, dA1, dA2, dA3, &
                                       wall_lo, wall_hi, row, jf, k, ni, nj, nk)
        ! j-face row jf at plane k; jf=1 / jf=nj wall-masked.
        ! Corners (i:i+1, jf, k:k+1).

        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA1(ni-1, nj, nk-1), dA2(ni-1, nj, nk-1), dA3(ni-1, nj, nk-1)
        real, intent(in) :: wall_lo(ni-1, nk-1)
        real, intent(in) :: wall_hi(ni-1, nk-1)
        real, intent(inout) :: row(ni, 5)

        integer :: i, j
        real :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3, mdot
        real :: dp1, dp2, dp3, dp4, w
        real :: g1, g2, g3, g4

        ! accum_corners MANUALLY INLINED here, exactly as in
        ! residual_nodal.f90 and for the same reason: with three dA dummies
        ! per direction instead of one, GCC stopped inlining the contained
        ! routine and this helper's i loops went scalar while production's
        ! stayed vectorized. Arithmetic, corner order and summation order
        ! are production's, unchanged. gfortran has no always_inline; this
        ! is the 470d6f8 remedy.
        ! jf is the corner j index in all three branches (1, nj, jf).
        j = jf

        if (jf == 1) then
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
                mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j,k+1,4)/r(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                    + w*(cons(i+1,j,k+1,4)/r(i+1,j,k+1) - Omega*cons(i+1,j,k+1,1)*r(i+1,j,k+1))
                mdot = mf1*dA1(i,jf,k) + mf2*dA2(i,jf,k) + mf3*dA3(i,jf,k)
                row(i,1) = mdot
                row(i,2) = pm1*mdot + pm5*dA1(i,jf,k)
                row(i,3) = pm2*mdot + pm5*dA2(i,jf,k)
                row(i,4) = pm3*mdot + pm6*dA3(i,jf,k)
                row(i,5) = pm4*mdot + Omega*pm6*dA3(i,jf,k)
            end do
        else if (jf == nj) then
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
                mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j,k+1,4)/r(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                    + w*(cons(i+1,j,k+1,4)/r(i+1,j,k+1) - Omega*cons(i+1,j,k+1,1)*r(i+1,j,k+1))
                mdot = mf1*dA1(i,jf,k) + mf2*dA2(i,jf,k) + mf3*dA3(i,jf,k)
                row(i,1) = mdot
                row(i,2) = pm1*mdot + pm5*dA1(i,jf,k)
                row(i,3) = pm2*mdot + pm5*dA2(i,jf,k)
                row(i,4) = pm3*mdot + pm6*dA3(i,jf,k)
                row(i,5) = pm4*mdot + Omega*pm6*dA3(i,jf,k)
            end do
        else
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
                mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j,k+1,4)/r(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                    + w*(cons(i+1,j,k+1,4)/r(i+1,j,k+1) - Omega*cons(i+1,j,k+1,1)*r(i+1,j,k+1))
                mdot = mf1*dA1(i,jf,k) + mf2*dA2(i,jf,k) + mf3*dA3(i,jf,k)
                row(i,1) = mdot
                row(i,2) = pm1*mdot + pm5*dA1(i,jf,k)
                row(i,3) = pm2*mdot + pm5*dA2(i,jf,k)
                row(i,4) = pm3*mdot + pm6*dA3(i,jf,k)
                row(i,5) = pm4*mdot + Omega*pm6*dA3(i,jf,k)
            end do
        end if

    end subroutine jface_flow_row_soa


    pure subroutine kface_flow_plane_soa(ho, P, P_offset, r, &
                                         cons, Omega, dA1, dA2, dA3, &
                                         wall_lo, wall_hi, plane, kf, njp, &
                                         ni, nj, nk)
        ! k-face plane kf; kf=1 / kf=nk wall-masked. Corners (i:i+1, j:j+1, kf).

        implicit none
        integer, intent(in) :: kf, njp, ni, nj, nk
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA1(ni-1, nj-1, nk), dA2(ni-1, nj-1, nk), dA3(ni-1, nj-1, nk)
        real, intent(in) :: wall_lo(ni-1, nj-1)
        real, intent(in) :: wall_hi(ni-1, nj-1)
        real, intent(inout) :: plane(ni, njp, 5)

        integer :: i, j, k
        real :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3, mdot
        real :: dp1, dp2, dp3, dp4, w
        real :: g1, g2, g3, g4

        ! accum_corners MANUALLY INLINED here, exactly as in
        ! residual_nodal.f90 and for the same reason: with three dA dummies
        ! per direction instead of one, GCC stopped inlining the contained
        ! routine and this helper's i loops went scalar while production's
        ! stayed vectorized. Arithmetic, corner order and summation order
        ! are production's, unchanged. gfortran has no always_inline; this
        ! is the 470d6f8 remedy.
        ! kf is the corner k index in all three branches (1, nk, kf).
        k = kf

        if (kf == 1) then
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
                mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j+1,k,4)/r(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                    + w*(cons(i+1,j+1,k,4)/r(i+1,j+1,k) - Omega*cons(i+1,j+1,k,1)*r(i+1,j+1,k))
                mdot = mf1*dA1(i,j,kf) + mf2*dA2(i,j,kf) + mf3*dA3(i,j,kf)
                plane(i,j,1) = mdot
                plane(i,j,2) = pm1*mdot + pm5*dA1(i,j,kf)
                plane(i,j,3) = pm2*mdot + pm5*dA2(i,j,kf)
                plane(i,j,4) = pm3*mdot + pm6*dA3(i,j,kf)
                plane(i,j,5) = pm4*mdot + Omega*pm6*dA3(i,j,kf)
            end do
            end do
        else if (kf == nk) then
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
                mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j+1,k,4)/r(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                    + w*(cons(i+1,j+1,k,4)/r(i+1,j+1,k) - Omega*cons(i+1,j+1,k,1)*r(i+1,j+1,k))
                mdot = mf1*dA1(i,j,kf) + mf2*dA2(i,j,kf) + mf3*dA3(i,j,kf)
                plane(i,j,1) = mdot
                plane(i,j,2) = pm1*mdot + pm5*dA1(i,j,kf)
                plane(i,j,3) = pm2*mdot + pm5*dA2(i,j,kf)
                plane(i,j,4) = pm3*mdot + pm6*dA3(i,j,kf)
                plane(i,j,5) = pm4*mdot + Omega*pm6*dA3(i,j,kf)
            end do
            end do
        else
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
                mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                    + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                    + w*(cons(i,j+1,k,4)/r(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                    + w*(cons(i+1,j+1,k,4)/r(i+1,j+1,k) - Omega*cons(i+1,j+1,k,1)*r(i+1,j+1,k))
                mdot = mf1*dA1(i,j,kf) + mf2*dA2(i,j,kf) + mf3*dA3(i,j,kf)
                plane(i,j,1) = mdot
                plane(i,j,2) = pm1*mdot + pm5*dA1(i,j,kf)
                plane(i,j,3) = pm2*mdot + pm5*dA2(i,j,kf)
                plane(i,j,4) = pm3*mdot + pm6*dA3(i,j,kf)
                plane(i,j,5) = pm4*mdot + Omega*pm6*dA3(i,j,kf)
            end do
            end do
        end if

    end subroutine kface_flow_plane_soa

end module residual_prod_soa_helpers


! =====================================================================
! Driver: production's set_residual with the three AoS face-area arrays
! replaced by nine component arrays. Slab sweep, rolling buffers, fused
! dU write, folded change limiter and deferred cusp correction verbatim,
! so the only variable against `prod` is the geometry layout.
!
! dAk (AoS) is still taken, solely for correct_cusp_kface_du -- exactly
! as the `multall` arm does, and for the same reason: that routine is
! production's own and is called unmodified by every arm.
! =====================================================================
subroutine set_residual_prod_soa( &
    cons, P, P_offset, &
    r, Omega, &
    dai1, dai2, dai3, daj1, daj2, daj3, dak1, dak2, dak3, &
    f_body, &
    dU, &
    vx, vr, vt, ho, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    dAk, dt_vol, dampin, &
    kb, njp, ni, nj, nk &
    )

    use residual_prod_soa_helpers
    use residual_helpers, only: correct_cusp_kface_du

    implicit none

    real, intent(in) :: cons(ni, nj, nk, 5)
    real, intent(in) :: P(ni, nj, nk)
    real, intent(in) :: P_offset
    real, intent(in) :: r(ni, nj, nk)
    real, intent(in) :: Omega
    ! Face areas as nine separate component arrays; grid geometry, built once.
    real, intent(in) :: dai1(ni, nj-1, nk-1), dai2(ni, nj-1, nk-1), dai3(ni, nj-1, nk-1)
    real, intent(in) :: daj1(ni-1, nj, nk-1), daj2(ni-1, nj, nk-1), daj3(ni-1, nj, nk-1)
    real, intent(in) :: dak1(ni-1, nj-1, nk), dak2(ni-1, nj-1, nk), dak3(ni-1, nj-1, nk)
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
    real, intent(inout) :: planes(ni, njp, 5, 2)
    real, intent(inout) :: rows(ni, 5, 3)
    ! AoS dAk retained solely for the shared cusp correction.
    real, intent(in) :: dAk(3, ni-1, nj-1, nk)
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

    ! Prime the rolling k-face plane with face k=1 before the slab sweep.
    call kface_flow_plane_soa(ho, P, P_offset, r, cons, &
                              Omega, dak1, dak2, dak3, wallk1, wallnk, &
                              planes(:,:,:,pa), 1, njp, ni, nj, nk)

    do k0 = 1, nk-1, kb
    k1 = min(k0 + kb - 1, nk-1)

    do k = k0, k1
        ja = 2
        jb = 3
        call jface_flow_row_soa(ho, P, P_offset, r, cons, &
                                Omega, daj1, daj2, daj3, wallj1, wallnj, &
                                rows(:,:,ja), 1, k, ni, nj, nk)
        call kface_flow_plane_soa(ho, P, P_offset, r, cons, &
                                  Omega, dak1, dak2, dak3, wallk1, wallnk, &
                                  planes(:,:,:,pb), k+1, njp, ni, nj, nk)
        do j = 1, nj-1
            call iface_flow_row_soa(ho, P, P_offset, r, cons, &
                                    Omega, dai1, dai2, dai3, &
                                    walli1(j,k), wallni(j,k), &
                                    rows(:,:,1), j, k, ni, nj, nk)
            call jface_flow_row_soa(ho, P, P_offset, r, cons, &
                                    Omega, daj1, daj2, daj3, wallj1, wallnj, &
                                    rows(:,:,jb), j+1, k, ni, nj, nk)
            do m = 1, 5
            do i = 1, ni-1
                dU(i,j,k,m) = rows(i,m,1) - rows(i+1,m,1) + f_body(i,j,k,m) &
                            + rows(i,m,ja) - rows(i,m,jb) &
                            + planes(i,j,m,pa) - planes(i,j,m,pb)
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

    if (i_cusp_start > 0 .and. nk > 2) then
        call correct_cusp_kface_du(vx, vr, vt, ho, P, P_offset, r, cons, &
                                   Omega, dAk, wallk1, wallnk, dU, &
                                   i_cusp_start, i_cusp_end, ni, nj, nk)
    end if

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

    ! Own procedure so ifort sees a dU with no other writes in scope --
    ! see production's set_residual for why.
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

end subroutine set_residual_prod_soa
