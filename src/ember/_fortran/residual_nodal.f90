! =====================================================================
! BENCHMARK ARM `nodal` -- production's fused sweep, reading the nodal
! primitives instead of deriving them from `cons`.
!
! Nothing in production calls this. See docs/dev/plan_nodal_primitives.md
! and docs/dev/kernel_benchmark_methodology.md.
!
! Why it exists
! -------------
! Section 26's `multall` arm beats production by 13-50% on Haswell and
! bundles THREE changes: (1) nodal primitives read/staged instead of
! derived per face corner, (2) five passes instead of one fused sweep,
! (3) SoA face-area geometry instead of component-first dA(3,i,j,k).
! Section 26.4 attributes the whole win to divides, i.e. to (1) alone --
! a mechanism claim with no attribution control behind it.
!
! This arm IS that control: (1) applied alone, to production's own fused
! sweep. It is section 20 undone and nothing else.
!
! What changes, exactly
! ---------------------
! In the three accum_corners bodies only:
!   g1..g4 = 1/cons(...,1)                     -> deleted
!   pm1 = sum 0.25*cons(...,2)*g               -> sum 0.25*vx(...)
!   pm2 = sum 0.25*cons(...,3)*g               -> sum 0.25*vr(...)
!   pm3 = sum 0.25*cons(...,4)*g               -> sum 0.25*r(...)*vt(...)
!   mf3 = sum w*(cons(...,4)/r - Omega*cons(...,1)*r)
!                                              -> sum w*cons(...,1)*(vt - Omega*r)
! These are transcribed from correct_cusp_kface_du's accum() in
! residual.f90, which already carries the target form verbatim -- it is
! the one pass section 20 never converted.
!
! pm4/pm5/pm6, mf1 and mf2 are UNTOUCHED. mf1/mf2 stay pure cons(...,2) /
! cons(...,3) sums, so the i-direction mass residual should come out
! BITWISE against production and any mass deviation is isolated to mf3,
! i.e. to the cross-stream terms -- which a straight duct cannot see at
! all (dAj(1) = dAk(1) = 0). Gate on a swirled state.
!
! Divides removed: four reciprocals plus four cons4/r per face, ~3 faces
! per cell, ~24 divides/cell -> ZERO. mf3's divide goes too, because
! rho*Vt_rel = cons1*(vt - Omega*r) needs no reciprocal. So this tests
! section 26.4's mechanism at full strength, not partially.
!
! Traffic goes the other way, as section 20 intended: +vx,vr,vt (12
! B/cell) against -cons(...,4), which is dead in the hot sweep here (cons
! is component-last, so that is a real dropped stream). Net ~+8 B/cell on
! production's modelled ~152.
!
! Deliberately NOT reproduced (methodology section 3)
! ---------------------------------------------------
!   - no staged nodal primitives. multall stages rowt/rvt per node because
!     five passes cannot redo the work five times; a fused sweep forms
!     r*vt and cons1*(vt - Omega*r) per corner for two flops. Staging
!     them here would re-bundle a second variable AND take the shape
!     section 18 lost with (a buffer that writes more than it saves).
!   - no SoA geometry: dAi/dAj/dAk stay component-first.
!   - no pass split: one fused sweep, sections 8/13 intact.
!
! Everything else is production verbatim -- hand-scalarized pm/mf,
! hand-unrolled corners, k-slab rolling buffers, the fused dU write, the
! folded change limiter, and the shared correct_cusp_kface_du (which
! needs no change: it already reads nodally). Sections 17/19 priced that
! structure at 2-2.7x and none of it is under test.
!
! The inputs are free (Rule 4): Grid.update_residual (grid.py:1567-1570)
! already passes block.Vx_nd/Vr_nd/Vt_nd/ho_nd into set_residual on every
! evaluation, and set_residual already declares them -- today only the
! O(surface) cusp pass consumes them. This arm's signature is therefore
! identical to production's, down to planes/rows/kb.
!
! Rule 6 applies to any gate on this arm: it reads the cached nodal
! arrays where production re-derives from cons, so a direct write to
! conserved_nd without update_cached_conserved() makes the two arms solve
! different states.
! =====================================================================
module residual_nodal_helpers
    implicit none
    private
    public :: iface_flow_row_nd, jface_flow_row_nd, kface_flow_plane_nd

contains

    ! Bodies are production's residual_helpers, with vx/vr/vt added to the
    ! dummy list and the five lines above changed inside accum_corners.

    pure subroutine iface_flow_row_nd(vx, vr, vt, ho, P, P_offset, r, &
                                      cons, Omega, dA, &
                                      wall_lo, wall_hi, row, j, k, ni, nj, nk)
        ! Compute inviscid face flows on the ni i-faces of cell row (j,k);
        ! the i=1 / i=ni boundary faces are wall-masked by the scalars
        ! wall_lo / wall_hi. i-face corners: (i, j:j+1, k:k+1)

        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: wall_lo, wall_hi
        real, intent(inout) :: row(ni, 5)

        integer :: i
        real :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3, mdot

        ! pm/mf scalarized and the four corners hand-unrolled: ifort's
        ! vectorizer treats the pm(:)/mf(:) arrays as a cross-iteration
        ! aliasing hazard it cannot disprove. Retained verbatim from
        ! production -- see residual.f90's iface_flow_row for the full
        ! justification and its July 2026 re-measurement.

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
            ! summation order as production's.
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real, intent(out) :: pm1, pm2, pm3, pm4, pm5, pm6, mf1, mf2, mf3
            real :: dp1, dp2, dp3, dp4, w
            dp1 = P(i,j,k)     - P_offset
            dp2 = P(i,j+1,k)   - P_offset
            dp3 = P(i,j,k+1)   - P_offset
            dp4 = P(i,j+1,k+1) - P_offset
            ! THE CHANGE: Vx, Vr and r*Vt read from their nodal arrays
            ! rather than derived as c2/c1, c3/c1, c4/c1. No reciprocals.
            pm1 = 0.25e0*vx(i,j,k) + 0.25e0*vx(i,j+1,k) &
                + 0.25e0*vx(i,j,k+1) + 0.25e0*vx(i,j+1,k+1)
            pm2 = 0.25e0*vr(i,j,k) + 0.25e0*vr(i,j+1,k) &
                + 0.25e0*vr(i,j,k+1) + 0.25e0*vr(i,j+1,k+1)
            pm3 = 0.25e0*r(i,j,k)*vt(i,j,k) + 0.25e0*r(i,j+1,k)*vt(i,j+1,k) &
                + 0.25e0*r(i,j,k+1)*vt(i,j,k+1) + 0.25e0*r(i,j+1,k+1)*vt(i,j+1,k+1)
            pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i,j+1,k) + 0.25e0*ho(i,j,k+1) + 0.25e0*ho(i,j+1,k+1)
            pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
            pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i,j+1,k)*dp2 &
                + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i,j+1,k+1)*dp4
            w = 0.25e0*wfac
            ! Untouched: pure cons sums, so the i-face mass flux is bitwise.
            mf1 = w*cons(i,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i,j,k+1,2) + w*cons(i,j+1,k+1,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i,j,k+1,3) + w*cons(i,j+1,k+1,3)
            ! THE CHANGE: rho*Vt_rel = rho*(Vt - Omega*r), no divide by r.
            mf3 = w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k)) &
                + w*cons(i,j+1,k,1)*(vt(i,j+1,k) - Omega*r(i,j+1,k)) &
                + w*cons(i,j,k+1,1)*(vt(i,j,k+1) - Omega*r(i,j,k+1)) &
                + w*cons(i,j+1,k+1,1)*(vt(i,j+1,k+1) - Omega*r(i,j+1,k+1))
        end subroutine accum_corners
    end subroutine iface_flow_row_nd


    pure subroutine jface_flow_row_nd(vx, vr, vt, ho, P, P_offset, r, &
                                      cons, Omega, dA, &
                                      wall_lo, wall_hi, row, jf, k, ni, nj, nk)
        ! Compute inviscid face flows on the (ni-1) j-faces of face row jf at
        ! cell plane k; jf=1 / jf=nj are the wall-masked boundary rows.
        ! j-face corners: (i:i+1, jf, k:k+1)

        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
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

        ! accum_corners MANUALLY INLINED, same reason as in
        ! kface_flow_plane_nd below -- see the note there. Inlining it in
        ! the k-helper alone shifted GCC's budget and cost this helper its
        ! vectorization, so the two move together. jf equals the corner j
        ! index in all three branches (1, nj, jf).
        j = jf
        if (jf == 1) then
            ! Low boundary j=1
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,k)     - P_offset
                dp2 = P(i+1,j,k)   - P_offset
                dp3 = P(i,j,k+1)   - P_offset
                dp4 = P(i+1,j,k+1) - P_offset
                pm1 = 0.25e0*vx(i,j,k) + 0.25e0*vx(i+1,j,k) &
                    + 0.25e0*vx(i,j,k+1) + 0.25e0*vx(i+1,j,k+1)
                pm2 = 0.25e0*vr(i,j,k) + 0.25e0*vr(i+1,j,k) &
                    + 0.25e0*vr(i,j,k+1) + 0.25e0*vr(i+1,j,k+1)
                pm3 = 0.25e0*r(i,j,k)*vt(i,j,k) + 0.25e0*r(i+1,j,k)*vt(i+1,j,k) &
                    + 0.25e0*r(i,j,k+1)*vt(i,j,k+1) + 0.25e0*r(i+1,j,k+1)*vt(i+1,j,k+1)
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j,k+1) + 0.25e0*ho(i+1,j,k+1)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i+1,j,k+1)*dp4
                w = 0.25e0*wall_lo(i,k)
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j,k+1,2) + w*cons(i+1,j,k+1,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j,k+1,3) + w*cons(i+1,j,k+1,3)
                mf3 = w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k)) &
                    + w*cons(i+1,j,k,1)*(vt(i+1,j,k) - Omega*r(i+1,j,k)) &
                    + w*cons(i,j,k+1,1)*(vt(i,j,k+1) - Omega*r(i,j,k+1)) &
                    + w*cons(i+1,j,k+1,1)*(vt(i+1,j,k+1) - Omega*r(i+1,j,k+1))
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
                dp1 = P(i,j,k)     - P_offset
                dp2 = P(i+1,j,k)   - P_offset
                dp3 = P(i,j,k+1)   - P_offset
                dp4 = P(i+1,j,k+1) - P_offset
                pm1 = 0.25e0*vx(i,j,k) + 0.25e0*vx(i+1,j,k) &
                    + 0.25e0*vx(i,j,k+1) + 0.25e0*vx(i+1,j,k+1)
                pm2 = 0.25e0*vr(i,j,k) + 0.25e0*vr(i+1,j,k) &
                    + 0.25e0*vr(i,j,k+1) + 0.25e0*vr(i+1,j,k+1)
                pm3 = 0.25e0*r(i,j,k)*vt(i,j,k) + 0.25e0*r(i+1,j,k)*vt(i+1,j,k) &
                    + 0.25e0*r(i,j,k+1)*vt(i,j,k+1) + 0.25e0*r(i+1,j,k+1)*vt(i+1,j,k+1)
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j,k+1) + 0.25e0*ho(i+1,j,k+1)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i+1,j,k+1)*dp4
                w = 0.25e0*wall_hi(i,k)
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j,k+1,2) + w*cons(i+1,j,k+1,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j,k+1,3) + w*cons(i+1,j,k+1,3)
                mf3 = w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k)) &
                    + w*cons(i+1,j,k,1)*(vt(i+1,j,k) - Omega*r(i+1,j,k)) &
                    + w*cons(i,j,k+1,1)*(vt(i,j,k+1) - Omega*r(i,j,k+1)) &
                    + w*cons(i+1,j,k+1,1)*(vt(i+1,j,k+1) - Omega*r(i+1,j,k+1))
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
                dp1 = P(i,j,k)     - P_offset
                dp2 = P(i+1,j,k)   - P_offset
                dp3 = P(i,j,k+1)   - P_offset
                dp4 = P(i+1,j,k+1) - P_offset
                pm1 = 0.25e0*vx(i,j,k) + 0.25e0*vx(i+1,j,k) &
                    + 0.25e0*vx(i,j,k+1) + 0.25e0*vx(i+1,j,k+1)
                pm2 = 0.25e0*vr(i,j,k) + 0.25e0*vr(i+1,j,k) &
                    + 0.25e0*vr(i,j,k+1) + 0.25e0*vr(i+1,j,k+1)
                pm3 = 0.25e0*r(i,j,k)*vt(i,j,k) + 0.25e0*r(i+1,j,k)*vt(i+1,j,k) &
                    + 0.25e0*r(i,j,k+1)*vt(i,j,k+1) + 0.25e0*r(i+1,j,k+1)*vt(i+1,j,k+1)
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j,k+1) + 0.25e0*ho(i+1,j,k+1)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i+1,j,k+1)*dp4
                w = 0.25e0
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j,k+1,2) + w*cons(i+1,j,k+1,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j,k+1,3) + w*cons(i+1,j,k+1,3)
                mf3 = w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k)) &
                    + w*cons(i+1,j,k,1)*(vt(i+1,j,k) - Omega*r(i+1,j,k)) &
                    + w*cons(i,j,k+1,1)*(vt(i,j,k+1) - Omega*r(i,j,k+1)) &
                    + w*cons(i+1,j,k+1,1)*(vt(i+1,j,k+1) - Omega*r(i+1,j,k+1))
                mdot = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
                row(i,1) = mdot
                row(i,2) = pm1*mdot + pm5*dA(1,i,jf,k)
                row(i,3) = pm2*mdot + pm5*dA(2,i,jf,k)
                row(i,4) = pm3*mdot + pm6*dA(3,i,jf,k)
                row(i,5) = pm4*mdot + Omega*pm6*dA(3,i,jf,k)
            end do
        end if

    end subroutine jface_flow_row_nd


    pure subroutine kface_flow_plane_nd(vx, vr, vt, ho, P, P_offset, r, &
                                        cons, Omega, dA, &
                                        wall_lo, wall_hi, plane, kf, njp, &
                                        ni, nj, nk)
        ! Compute inviscid face flows on the (ni-1)x(nj-1) k-face plane kf;
        ! kf=1 / kf=nk are the wall-masked boundary planes. njp (nj or nj+1)
        ! is the plane buffer's padded j-extent -- see set_residual_nodal.
        ! k-face corners: (i:i+1, j:j+1, kf)

        implicit none
        integer, intent(in) :: kf, njp, ni, nj, nk
        real, intent(in) :: vx(ni, nj, nk), vr(ni, nj, nk), vt(ni, nj, nk)
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
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

        ! *** accum_corners is MANUALLY INLINED in this helper. ***
        !
        ! It is a contained subroutine in the i- and j-helpers above and in
        ! production's kface_flow_plane, where GCC inlines it and vectorizes
        ! the i loop. Here it did not: with three more host-associated arrays
        ! (vx/vr/vt) GCC left the call out of line in every clone, the
        ! link-stage report read "statement clobbers memory: accum_corners",
        ! and all three k-face i loops went scalar while production's
        ! vectorized. Timing that would have measured GCC's inliner, not the
        ! nodal representation -- the exact confound Gate 1 exists to catch.
        !
        ! gfortran has no always_inline attribute, so the remedy is the one
        ! this codebase already used for the same failure under ifort
        ! (470d6f8, "manually inline accum/put to unblock vectorization").
        ! The arithmetic, the corner order and the summation order are
        ! unchanged, so this is a codegen fix, not a numerics one.
        !
        ! kf equals the corner k index in all three branches (1, nk, kf), so
        ! the inlined bodies index with kf throughout.
        k = kf
        if (kf == 1) then
            ! Low boundary k=1
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,k)     - P_offset
                dp2 = P(i+1,j,k)   - P_offset
                dp3 = P(i,j+1,k)   - P_offset
                dp4 = P(i+1,j+1,k) - P_offset
                pm1 = 0.25e0*vx(i,j,k) + 0.25e0*vx(i+1,j,k) &
                    + 0.25e0*vx(i,j+1,k) + 0.25e0*vx(i+1,j+1,k)
                pm2 = 0.25e0*vr(i,j,k) + 0.25e0*vr(i+1,j,k) &
                    + 0.25e0*vr(i,j+1,k) + 0.25e0*vr(i+1,j+1,k)
                pm3 = 0.25e0*r(i,j,k)*vt(i,j,k) + 0.25e0*r(i+1,j,k)*vt(i+1,j,k) &
                    + 0.25e0*r(i,j+1,k)*vt(i,j+1,k) + 0.25e0*r(i+1,j+1,k)*vt(i+1,j+1,k)
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j+1,k) + 0.25e0*ho(i+1,j+1,k)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j+1,k)*dp3 + 0.25e0*r(i+1,j+1,k)*dp4
                w = 0.25e0*wall_lo(i,j)
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i+1,j+1,k,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i+1,j+1,k,3)
                mf3 = w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k)) &
                    + w*cons(i+1,j,k,1)*(vt(i+1,j,k) - Omega*r(i+1,j,k)) &
                    + w*cons(i,j+1,k,1)*(vt(i,j+1,k) - Omega*r(i,j+1,k)) &
                    + w*cons(i+1,j+1,k,1)*(vt(i+1,j+1,k) - Omega*r(i+1,j+1,k))
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
                dp1 = P(i,j,k)     - P_offset
                dp2 = P(i+1,j,k)   - P_offset
                dp3 = P(i,j+1,k)   - P_offset
                dp4 = P(i+1,j+1,k) - P_offset
                pm1 = 0.25e0*vx(i,j,k) + 0.25e0*vx(i+1,j,k) &
                    + 0.25e0*vx(i,j+1,k) + 0.25e0*vx(i+1,j+1,k)
                pm2 = 0.25e0*vr(i,j,k) + 0.25e0*vr(i+1,j,k) &
                    + 0.25e0*vr(i,j+1,k) + 0.25e0*vr(i+1,j+1,k)
                pm3 = 0.25e0*r(i,j,k)*vt(i,j,k) + 0.25e0*r(i+1,j,k)*vt(i+1,j,k) &
                    + 0.25e0*r(i,j+1,k)*vt(i,j+1,k) + 0.25e0*r(i+1,j+1,k)*vt(i+1,j+1,k)
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j+1,k) + 0.25e0*ho(i+1,j+1,k)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j+1,k)*dp3 + 0.25e0*r(i+1,j+1,k)*dp4
                w = 0.25e0*wall_hi(i,j)
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i+1,j+1,k,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i+1,j+1,k,3)
                mf3 = w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k)) &
                    + w*cons(i+1,j,k,1)*(vt(i+1,j,k) - Omega*r(i+1,j,k)) &
                    + w*cons(i,j+1,k,1)*(vt(i,j+1,k) - Omega*r(i,j+1,k)) &
                    + w*cons(i+1,j+1,k,1)*(vt(i+1,j+1,k) - Omega*r(i+1,j+1,k))
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
                dp1 = P(i,j,k)     - P_offset
                dp2 = P(i+1,j,k)   - P_offset
                dp3 = P(i,j+1,k)   - P_offset
                dp4 = P(i+1,j+1,k) - P_offset
                pm1 = 0.25e0*vx(i,j,k) + 0.25e0*vx(i+1,j,k) &
                    + 0.25e0*vx(i,j+1,k) + 0.25e0*vx(i+1,j+1,k)
                pm2 = 0.25e0*vr(i,j,k) + 0.25e0*vr(i+1,j,k) &
                    + 0.25e0*vr(i,j+1,k) + 0.25e0*vr(i+1,j+1,k)
                pm3 = 0.25e0*r(i,j,k)*vt(i,j,k) + 0.25e0*r(i+1,j,k)*vt(i+1,j,k) &
                    + 0.25e0*r(i,j+1,k)*vt(i,j+1,k) + 0.25e0*r(i+1,j+1,k)*vt(i+1,j+1,k)
                pm4 = 0.25e0*ho(i,j,k) + 0.25e0*ho(i+1,j,k) + 0.25e0*ho(i,j+1,k) + 0.25e0*ho(i+1,j+1,k)
                pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i+1,j,k)*dp2 &
                    + 0.25e0*r(i,j+1,k)*dp3 + 0.25e0*r(i+1,j+1,k)*dp4
                w = 0.25e0
                mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i+1,j+1,k,2)
                mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i+1,j+1,k,3)
                mf3 = w*cons(i,j,k,1)*(vt(i,j,k) - Omega*r(i,j,k)) &
                    + w*cons(i+1,j,k,1)*(vt(i+1,j,k) - Omega*r(i+1,j,k)) &
                    + w*cons(i,j+1,k,1)*(vt(i,j+1,k) - Omega*r(i,j+1,k)) &
                    + w*cons(i+1,j+1,k,1)*(vt(i+1,j+1,k) - Omega*r(i+1,j+1,k))
                mdot = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
                plane(i,j,1) = mdot
                plane(i,j,2) = pm1*mdot + pm5*dA(1,i,j,kf)
                plane(i,j,3) = pm2*mdot + pm5*dA(2,i,j,kf)
                plane(i,j,4) = pm3*mdot + pm6*dA(3,i,j,kf)
                plane(i,j,5) = pm4*mdot + Omega*pm6*dA(3,i,j,kf)
            end do
            end do
        end if

    end subroutine kface_flow_plane_nd

end module residual_nodal_helpers


! =====================================================================
! Driver: production's set_residual, unchanged except that it calls the
! _nd face helpers and threads vx/vr/vt into them. The k-slab sweep, the
! rolling i/j/k buffers, the single fused dU write, the folded change
! limiter and the deferred cusp correction are all verbatim, so the only
! variable between this arm and `prod` is the nodal representation.
!
! The signature is IDENTICAL to set_residual's, including planes/rows/kb,
! so the harness reuses production's scratch carve with no new buffers.
!
! correct_cusp_kface_du is production's own routine, called unmodified --
! it already reads the nodal arrays, so it needs no _nd variant. Sharing
! it guarantees identical codegen for the part that is not under test
! (methodology section 2).
! =====================================================================
subroutine set_residual_nodal( &
    cons, P, P_offset, &
    r, Omega, dAi, dAj, dAk, &
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

    use residual_nodal_helpers
    use residual_helpers, only: correct_cusp_kface_du

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
    call kface_flow_plane_nd(vx, vr, vt, ho, P, P_offset, r, cons, &
                             Omega, dAk, wallk1, wallnk, planes(:,:,:,pa), &
                             1, njp, ni, nj, nk)

    do k0 = 1, nk-1, kb
    k1 = min(k0 + kb - 1, nk-1)

    do k = k0, k1
        ja = 2
        jb = 3
        ! Prime the rolling j-face pair with the j=1 boundary face.
        call jface_flow_row_nd(vx, vr, vt, ho, P, P_offset, r, cons, &
                               Omega, dAj, wallj1, wallnj, rows(:,:,ja), &
                               1, k, ni, nj, nk)
        ! Advance the rolling k-face pair: pa holds face k, pb gets k+1.
        call kface_flow_plane_nd(vx, vr, vt, ho, P, P_offset, r, cons, &
                                 Omega, dAk, wallk1, wallnk, planes(:,:,:,pb), &
                                 k+1, njp, ni, nj, nk)
        do j = 1, nj-1
            call iface_flow_row_nd(vx, vr, vt, ho, P, P_offset, r, cons, &
                                   Omega, dAi, walli1(j,k), wallni(j,k), &
                                   rows(:,:,1), j, k, ni, nj, nk)
            call jface_flow_row_nd(vx, vr, vt, ho, P, P_offset, r, cons, &
                                   Omega, dAj, wallj1, wallnj, rows(:,:,jb), &
                                   j+1, k, ni, nj, nk)
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

    ! Cusp seam: deferred O(surface) correction, production's own routine.
    if (i_cusp_start > 0 .and. nk > 2) then
        call correct_cusp_kface_du(vx, vr, vt, ho, P, P_offset, r, cons, &
                                   Omega, dAk, wallk1, wallnk, dU, &
                                   i_cusp_start, i_cusp_end, ni, nj, nk)
    end if

    ! ---- change limiter, second half (the reduction is already done) ----
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

end subroutine set_residual_nodal
