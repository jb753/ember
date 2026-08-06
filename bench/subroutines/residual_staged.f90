! =====================================================================
! set_residual_staged / set_residual_split -- the multall/multall design
! point, priced against production's single fused sweep.
! Benchmark-only companions; nothing calls either in production.
!
! multall evaluates the five conserved-variable residuals in FIVE passes.
! SET_FLUX stages the face mass fluxes FIMAS/FJMAS/FKMAS once, then calls
! SUMFLUX once per variable (RO, ROE, ROVX, RORVT, ROVR), each pass
! rebuilding full-volume FLUXI/FLUXJ/FLUXK arrays and differencing them.
! Production ember does the opposite: one sweep holding the shared face
! mass flux mdot in a register and consuming it for all five components
! before discarding it.
!
! Two arms are built here, differing in ONE thing:
!
!   set_residual_staged  stages mdot into fi/fj/fk (multall's FIMAS et al),
!                        then five narrow passes read it back.
!   set_residual_split   five narrow passes, each recomputing mdot inline.
!
! The split arm is the attribution control. Without it a loss cannot be
! attributed to the 5-way split rather than to the staging, and the traffic
! model predicts staging is the WORSE of the two: writing plus five times
! re-reading fi/fj/fk costs ~72 B/cell, against the ~80 B/cell of cons
! re-reads it saves, and dA (component-first, so any one axis pulls all
! three) dominates both at 36 B/cell per pass.
!
! NOT built: multall's literal structure, which also materializes
! FLUXI/FLUXJ/FLUXK per variable and re-reads them. That is strictly
! dominated by fusing the flux build with the 6-face difference, as done
! here -- so if this loses, the literal form loses harder. One kernel
! bounds the whole staged family.
!
! DESIGN NOTE 1 -- the rolling buffers are KEPT, narrowed to one component
! (planes(ni,njp,2), rows(ni,3) instead of production's 5-wide). A plain
! 3D nest per component would recompute each face's pm twice, once per
! adjacent cell: 10x the face work rather than 5x, and not multall-faithful
! either, since staging FLUXI is precisely how multall keeps each face
! computed once. Same face-once discipline, same summation order, same
! slab structure as production -- only the pass count and the provenance
! of mdot differ.
!
! DESIGN NOTE 2 -- the two drivers are written out separately rather than
! sharing one branchy driver. A shared `if (recompute)` risks perturbing
! codegen for one arm and not the other, which would corrupt the very
! comparison being made. The nine module helpers below are shared; only
! the ~100-line sweeps are duplicated.
!
! DESIGN NOTE 3 -- the mass-only component (m=1) is where the two arms
! genuinely diverge in structure, not just in provenance. Staged reads
! fi/fj/fk in a plain 3D nest with no face work at all (that is the point
! of staging); split has to run the rolling mass-flux driver. Both are the
! honest form of their own design.
!
! EXACTNESS -- measured, not assumed (gfortran 14, -Ofast, 300k duct):
!
!   set_residual_staged, m=1 (mass)   BITWISE identical to production
!   everything else                   agrees to <= 1.2e-09 absolute,
!                                     <= 2.3e-06 of the dU field scale
!
! The mass residual is the pure six-point sum of the staged mass fluxes,
! so its being bitwise identical proves all three mflux_* helpers
! reproduce production's mdot EXACTLY -- both cross-stream directions and
! the wall masks included. (That check needs a swirled state: the duct
! case is axially straight with Vr = Vt = 0, so dAj(1) = dAk(1) = 0 and
! its j/k mass fluxes are identically zero, which no error can perturb.
! tools/bench_residual_staged.py's gate seeds cross-stream momentum for
! exactly this reason.) Storing and reloading mdot in float32 does not
! round, so staging is exact by construction.
!
! The residual differences are confined to the pm*mdot + press*dA step,
! where -Ofast contracts and reassociates differently when pm is formed in
! a dedicated per-component helper than when production's accum_corners
! forms all six pm factors together. Aligning the source form (dp1..dp4
! temporaries, matching production verbatim) was tried and does not remove
! it. The deviation is ~1e-5 of one ulp of the face flows being
! differenced, and two orders inside the residual goldens' rtol=1e-4.
!
! The change limiter's avg(m) reduction needs no tolerance argument: it is
! accumulated in the same order in every arm, since production visits
! component m with m outside and i inside the (j,k) row, which is the
! (k,j,i) order a narrow pass uses anyway.
! =====================================================================

module residual_staged_helpers
    implicit none
    private
    public :: mflux_iface_row, mflux_jface_row, mflux_kface_plane
    public :: pflow_iface_row, pflow_jface_row, pflow_kface_plane
    public :: rpflow_iface_row, rpflow_jface_row, rpflow_kface_plane
    public :: scale_du_all

contains

    ! ---- stage 1: face mass flux only (multall's FIMAS/FJMAS/FKMAS) -----
    !
    ! mdot = mf . dA with mf = (rho*Vx, rho*Vr, rho*Vt_rel), the four face
    ! corners averaged. The wall mask weights mf and ONLY mf (never pm), so
    ! it folds into the staged mdot exactly and the stage-2 helpers need no
    ! wall handling at all. pm/mf stay hand-scalarized and the corners
    ! hand-unrolled, as in production -- see residual.f90's iface_flow_row
    ! for that justification; it is not revisited here.

    pure subroutine mflux_iface_row(cons, r, Omega, dA, wall_lo, wall_hi, mrow, j, k, ni, nj, nk)
        ! ni i-faces of cell row (j,k). Corners: (i, j:j+1, k:k+1).
        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: r(ni, nj, nk)
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: wall_lo, wall_hi
        real, intent(inout) :: mrow(ni)

        integer :: i
        real :: mf1, mf2, mf3

        call accum_mf(1, j, k, wall_lo, mf1, mf2, mf3)
        mrow(1) = mf1*dA(1,1,j,k) + mf2*dA(2,1,j,k) + mf3*dA(3,1,j,k)

        !DIR$ IVDEP
        do i = 2, ni-1
            call accum_mf(i, j, k, 1.0e0, mf1, mf2, mf3)
            mrow(i) = mf1*dA(1,i,j,k) + mf2*dA(2,i,j,k) + mf3*dA(3,i,j,k)
        end do

        call accum_mf(ni, j, k, wall_hi, mf1, mf2, mf3)
        mrow(ni) = mf1*dA(1,ni,j,k) + mf2*dA(2,ni,j,k) + mf3*dA(3,ni,j,k)

    contains
        pure subroutine accum_mf(i, j, k, wfac, mf1, mf2, mf3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real, intent(out) :: mf1, mf2, mf3
            real :: w
            w = 0.25e0*wfac
            mf1 = w*cons(i,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i,j,k+1,2) + w*cons(i,j+1,k+1,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i,j,k+1,3) + w*cons(i,j+1,k+1,3)
            ! rho*Vt_rel = rho*Vt - Omega*rho*r = c4/r - Omega*c1*r
            mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                + w*(cons(i,j+1,k,4)/r(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                + w*(cons(i,j,k+1,4)/r(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                + w*(cons(i,j+1,k+1,4)/r(i,j+1,k+1) - Omega*cons(i,j+1,k+1,1)*r(i,j+1,k+1))
        end subroutine accum_mf
    end subroutine mflux_iface_row


    pure subroutine mflux_jface_row(cons, r, Omega, dA, wall_lo, wall_hi, mrow, jf, k, ni, nj, nk)
        ! (ni-1) j-faces of face row jf at cell plane k. Corners: (i:i+1, jf, k:k+1).
        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: r(ni, nj, nk)
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: wall_lo(ni-1, nk-1)
        real, intent(in) :: wall_hi(ni-1, nk-1)
        real, intent(inout) :: mrow(ni-1)

        integer :: i
        real :: mf1, mf2, mf3

        if (jf == 1) then
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, 1, k, wall_lo(i,k), mf1, mf2, mf3)
                mrow(i) = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
            end do
        else if (jf == nj) then
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, nj, k, wall_hi(i,k), mf1, mf2, mf3)
                mrow(i) = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
            end do
        else
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, jf, k, 1.0e0, mf1, mf2, mf3)
                mrow(i) = mf1*dA(1,i,jf,k) + mf2*dA(2,i,jf,k) + mf3*dA(3,i,jf,k)
            end do
        end if

    contains
        pure subroutine accum_mf(i, j, k, wfac, mf1, mf2, mf3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real, intent(out) :: mf1, mf2, mf3
            real :: w
            w = 0.25e0*wfac
            mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j,k+1,2) + w*cons(i+1,j,k+1,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j,k+1,3) + w*cons(i+1,j,k+1,3)
            mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                + w*(cons(i,j,k+1,4)/r(i,j,k+1) - Omega*cons(i,j,k+1,1)*r(i,j,k+1)) &
                + w*(cons(i+1,j,k+1,4)/r(i+1,j,k+1) - Omega*cons(i+1,j,k+1,1)*r(i+1,j,k+1))
        end subroutine accum_mf
    end subroutine mflux_jface_row


    pure subroutine mflux_kface_plane(cons, r, Omega, dA, wall_lo, wall_hi, mplane, kf, ni, nj, nk)
        ! (ni-1)x(nj-1) k-face plane kf. Corners: (i:i+1, j:j+1, kf).
        ! Unlike the flow planes this one is NOT j-padded: it is a slice of
        ! the staged fk volume, whose leading extents are fixed.
        implicit none
        integer, intent(in) :: kf, ni, nj, nk
        real, intent(in) :: r(ni, nj, nk)
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: Omega
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
        real, intent(in) :: wall_lo(ni-1, nj-1)
        real, intent(in) :: wall_hi(ni-1, nj-1)
        real, intent(inout) :: mplane(ni-1, nj-1)

        integer :: i, j
        real :: mf1, mf2, mf3

        if (kf == 1) then
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, j, 1, wall_lo(i,j), mf1, mf2, mf3)
                mplane(i,j) = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
            end do
            end do
        else if (kf == nk) then
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, j, nk, wall_hi(i,j), mf1, mf2, mf3)
                mplane(i,j) = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
            end do
            end do
        else
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                call accum_mf(i, j, kf, 1.0e0, mf1, mf2, mf3)
                mplane(i,j) = mf1*dA(1,i,j,kf) + mf2*dA(2,i,j,kf) + mf3*dA(3,i,j,kf)
            end do
            end do
        end if

    contains
        pure subroutine accum_mf(i, j, k, wfac, mf1, mf2, mf3)
            integer, intent(in) :: i, j, k
            real, intent(in) :: wfac
            real, intent(out) :: mf1, mf2, mf3
            real :: w
            w = 0.25e0*wfac
            mf1 = w*cons(i,j,k,2) + w*cons(i+1,j,k,2) + w*cons(i,j+1,k,2) + w*cons(i+1,j+1,k,2)
            mf2 = w*cons(i,j,k,3) + w*cons(i+1,j,k,3) + w*cons(i,j+1,k,3) + w*cons(i+1,j+1,k,3)
            mf3 = w*(cons(i,j,k,4)/r(i,j,k) - Omega*cons(i,j,k,1)*r(i,j,k)) &
                + w*(cons(i+1,j,k,4)/r(i+1,j,k) - Omega*cons(i+1,j,k,1)*r(i+1,j,k)) &
                + w*(cons(i,j+1,k,4)/r(i,j+1,k) - Omega*cons(i,j+1,k,1)*r(i,j+1,k)) &
                + w*(cons(i+1,j+1,k,4)/r(i+1,j+1,k) - Omega*cons(i+1,j+1,k,1)*r(i+1,j+1,k))
        end subroutine accum_mf
    end subroutine mflux_kface_plane


    ! ---- stage 2a: components 2 and 3 (axial and radial momentum) ------
    !
    !   flow = pm*mdot + pm5*dA(iax)
    !   pm   = face average of cons(icomp)/cons(1)   (Vx for m=2, Vr for m=3)
    !   pm5  = face average of (P - P_offset)
    !
    ! icomp indexes cons' trailing dimension and iax indexes dA's leading
    ! one, so both are loop-invariant scalars that leave the i reads unit
    ! stride. No wall handling: mdot arrives already masked.

    pure subroutine pflow_iface_row(P, P_offset, cons, dA, mrow, icomp, iax, row, j, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: icomp, iax, j, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: mrow(ni)
        real, intent(inout) :: row(ni)

        integer :: i
        real :: pm, pm5, g1, g2, g3, g4, dp1, dp2, dp3, dp4

        !DIR$ IVDEP
        do i = 1, ni
            g1 = 1.0e0/cons(i,j,k,1)
            g2 = 1.0e0/cons(i,j+1,k,1)
            g3 = 1.0e0/cons(i,j,k+1,1)
            g4 = 1.0e0/cons(i,j+1,k+1,1)
            pm = 0.25e0*cons(i,j,k,icomp)*g1 + 0.25e0*cons(i,j+1,k,icomp)*g2 &
               + 0.25e0*cons(i,j,k+1,icomp)*g3 + 0.25e0*cons(i,j+1,k+1,icomp)*g4
            dp1 = P(i,j,k)     - P_offset
            dp2 = P(i,j+1,k)   - P_offset
            dp3 = P(i,j,k+1)   - P_offset
            dp4 = P(i,j+1,k+1) - P_offset
            pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
            row(i) = pm*mrow(i) + pm5*dA(iax,i,j,k)
        end do
    end subroutine pflow_iface_row


    pure subroutine pflow_jface_row(P, P_offset, cons, dA, mrow, icomp, iax, row, jf, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: icomp, iax, jf, k, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: mrow(ni-1)
        real, intent(inout) :: row(ni)

        integer :: i
        real :: pm, pm5, g1, g2, g3, g4, dp1, dp2, dp3, dp4

        !DIR$ IVDEP
        do i = 1, ni-1
            g1 = 1.0e0/cons(i,jf,k,1)
            g2 = 1.0e0/cons(i+1,jf,k,1)
            g3 = 1.0e0/cons(i,jf,k+1,1)
            g4 = 1.0e0/cons(i+1,jf,k+1,1)
            pm = 0.25e0*cons(i,jf,k,icomp)*g1 + 0.25e0*cons(i+1,jf,k,icomp)*g2 &
               + 0.25e0*cons(i,jf,k+1,icomp)*g3 + 0.25e0*cons(i+1,jf,k+1,icomp)*g4
            dp1 = P(i,jf,k)     - P_offset
            dp2 = P(i+1,jf,k)   - P_offset
            dp3 = P(i,jf,k+1)   - P_offset
            dp4 = P(i+1,jf,k+1) - P_offset
            pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
            row(i) = pm*mrow(i) + pm5*dA(iax,i,jf,k)
        end do
    end subroutine pflow_jface_row


    pure subroutine pflow_kface_plane(P, P_offset, cons, dA, mplane, icomp, iax, plane, kf, njp, ni, nj, nk)
        implicit none
        integer, intent(in) :: icomp, iax, kf, njp, ni, nj, nk
        real, intent(in) :: P(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
        real, intent(in) :: mplane(ni-1, nj-1)
        real, intent(inout) :: plane(ni, njp)

        integer :: i, j
        real :: pm, pm5, g1, g2, g3, g4, dp1, dp2, dp3, dp4

        do j = 1, nj-1
        !DIR$ IVDEP
        do i = 1, ni-1
            g1 = 1.0e0/cons(i,j,kf,1)
            g2 = 1.0e0/cons(i+1,j,kf,1)
            g3 = 1.0e0/cons(i,j+1,kf,1)
            g4 = 1.0e0/cons(i+1,j+1,kf,1)
            pm = 0.25e0*cons(i,j,kf,icomp)*g1 + 0.25e0*cons(i+1,j,kf,icomp)*g2 &
               + 0.25e0*cons(i,j+1,kf,icomp)*g3 + 0.25e0*cons(i+1,j+1,kf,icomp)*g4
            dp1 = P(i,j,kf)     - P_offset
            dp2 = P(i+1,j,kf)   - P_offset
            dp3 = P(i,j+1,kf)   - P_offset
            dp4 = P(i+1,j+1,kf) - P_offset
            pm5 = 0.25e0*dp1 + 0.25e0*dp2 + 0.25e0*dp3 + 0.25e0*dp4
            plane(i,j) = pm*mplane(i,j) + pm5*dA(iax,i,j,kf)
        end do
        end do
    end subroutine pflow_kface_plane


    ! ---- stage 2b: components 4 and 5 (moment of momentum, energy) -----
    !
    !   flow = pm*mdot + pfac*pm6*dA(3)
    !   pm   = face average of cons(4)/cons(1)  (r*Vt)   if .not. use_ho
    !        = face average of ho                        if use_ho
    !   pm6  = face average of r*(P - P_offset)
    !   pfac = 1 for m=4, Omega for m=5
    !
    ! The use_ho branch is hoisted outside the i loop, giving two tight
    ! loop bodies -- the same shape production uses for its jf/kf boundary
    ! split. Multiplying by pfac = 1 is exact, so m=4 reproduces
    ! production's `pm3*mdot + pm6*dA(3,...)` bitwise.

    pure subroutine rpflow_iface_row(ho, P, P_offset, r, cons, dA, mrow, use_ho, pfac, row, j, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: j, k, ni, nj, nk
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: dA(3, ni, nj-1, nk-1)
        real, intent(in) :: mrow(ni)
        logical, intent(in) :: use_ho
        real, intent(in) :: pfac
        real, intent(inout) :: row(ni)

        integer :: i
        real :: pm, pm6, g1, g2, g3, g4, dp1, dp2, dp3, dp4

        if (use_ho) then
            !DIR$ IVDEP
            do i = 1, ni
                dp1 = P(i,j,k)     - P_offset
                dp2 = P(i,j+1,k)   - P_offset
                dp3 = P(i,j,k+1)   - P_offset
                dp4 = P(i,j+1,k+1) - P_offset
                pm = 0.25e0*ho(i,j,k) + 0.25e0*ho(i,j+1,k) + 0.25e0*ho(i,j,k+1) + 0.25e0*ho(i,j+1,k+1)
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i,j+1,k)*dp2 &
                    + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i,j+1,k+1)*dp4
                row(i) = pm*mrow(i) + pfac*pm6*dA(3,i,j,k)
            end do
        else
            !DIR$ IVDEP
            do i = 1, ni
                dp1 = P(i,j,k)     - P_offset
                dp2 = P(i,j+1,k)   - P_offset
                dp3 = P(i,j,k+1)   - P_offset
                dp4 = P(i,j+1,k+1) - P_offset
                g1 = 1.0e0/cons(i,j,k,1)
                g2 = 1.0e0/cons(i,j+1,k,1)
                g3 = 1.0e0/cons(i,j,k+1,1)
                g4 = 1.0e0/cons(i,j+1,k+1,1)
                pm = 0.25e0*cons(i,j,k,4)*g1 + 0.25e0*cons(i,j+1,k,4)*g2 &
                   + 0.25e0*cons(i,j,k+1,4)*g3 + 0.25e0*cons(i,j+1,k+1,4)*g4
                pm6 = 0.25e0*r(i,j,k)*dp1 + 0.25e0*r(i,j+1,k)*dp2 &
                    + 0.25e0*r(i,j,k+1)*dp3 + 0.25e0*r(i,j+1,k+1)*dp4
                row(i) = pm*mrow(i) + pfac*pm6*dA(3,i,j,k)
            end do
        end if
    end subroutine rpflow_iface_row


    pure subroutine rpflow_jface_row(ho, P, P_offset, r, cons, dA, mrow, use_ho, pfac, row, jf, k, ni, nj, nk)
        implicit none
        integer, intent(in) :: jf, k, ni, nj, nk
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: dA(3, ni-1, nj, nk-1)
        real, intent(in) :: mrow(ni-1)
        logical, intent(in) :: use_ho
        real, intent(in) :: pfac
        real, intent(inout) :: row(ni)

        integer :: i
        real :: pm, pm6, g1, g2, g3, g4, dp1, dp2, dp3, dp4

        if (use_ho) then
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,jf,k)     - P_offset
                dp2 = P(i+1,jf,k)   - P_offset
                dp3 = P(i,jf,k+1)   - P_offset
                dp4 = P(i+1,jf,k+1) - P_offset
                pm = 0.25e0*ho(i,jf,k) + 0.25e0*ho(i+1,jf,k) + 0.25e0*ho(i,jf,k+1) + 0.25e0*ho(i+1,jf,k+1)
                pm6 = 0.25e0*r(i,jf,k)*dp1 + 0.25e0*r(i+1,jf,k)*dp2 &
                    + 0.25e0*r(i,jf,k+1)*dp3 + 0.25e0*r(i+1,jf,k+1)*dp4
                row(i) = pm*mrow(i) + pfac*pm6*dA(3,i,jf,k)
            end do
        else
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,jf,k)     - P_offset
                dp2 = P(i+1,jf,k)   - P_offset
                dp3 = P(i,jf,k+1)   - P_offset
                dp4 = P(i+1,jf,k+1) - P_offset
                g1 = 1.0e0/cons(i,jf,k,1)
                g2 = 1.0e0/cons(i+1,jf,k,1)
                g3 = 1.0e0/cons(i,jf,k+1,1)
                g4 = 1.0e0/cons(i+1,jf,k+1,1)
                pm = 0.25e0*cons(i,jf,k,4)*g1 + 0.25e0*cons(i+1,jf,k,4)*g2 &
                   + 0.25e0*cons(i,jf,k+1,4)*g3 + 0.25e0*cons(i+1,jf,k+1,4)*g4
                pm6 = 0.25e0*r(i,jf,k)*dp1 + 0.25e0*r(i+1,jf,k)*dp2 &
                    + 0.25e0*r(i,jf,k+1)*dp3 + 0.25e0*r(i+1,jf,k+1)*dp4
                row(i) = pm*mrow(i) + pfac*pm6*dA(3,i,jf,k)
            end do
        end if
    end subroutine rpflow_jface_row


    pure subroutine rpflow_kface_plane(ho, P, P_offset, r, cons, dA, mplane, use_ho, pfac, plane, kf, njp, ni, nj, nk)
        implicit none
        integer, intent(in) :: kf, njp, ni, nj, nk
        real, intent(in) :: ho(ni, nj, nk), P(ni, nj, nk), r(ni, nj, nk)
        real, intent(in) :: P_offset
        real, intent(in) :: cons(ni, nj, nk, 5)
        real, intent(in) :: dA(3, ni-1, nj-1, nk)
        real, intent(in) :: mplane(ni-1, nj-1)
        logical, intent(in) :: use_ho
        real, intent(in) :: pfac
        real, intent(inout) :: plane(ni, njp)

        integer :: i, j
        real :: pm, pm6, g1, g2, g3, g4, dp1, dp2, dp3, dp4

        if (use_ho) then
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,kf)     - P_offset
                dp2 = P(i+1,j,kf)   - P_offset
                dp3 = P(i,j+1,kf)   - P_offset
                dp4 = P(i+1,j+1,kf) - P_offset
                pm = 0.25e0*ho(i,j,kf) + 0.25e0*ho(i+1,j,kf) + 0.25e0*ho(i,j+1,kf) + 0.25e0*ho(i+1,j+1,kf)
                pm6 = 0.25e0*r(i,j,kf)*dp1 + 0.25e0*r(i+1,j,kf)*dp2 &
                    + 0.25e0*r(i,j+1,kf)*dp3 + 0.25e0*r(i+1,j+1,kf)*dp4
                plane(i,j) = pm*mplane(i,j) + pfac*pm6*dA(3,i,j,kf)
            end do
            end do
        else
            do j = 1, nj-1
            !DIR$ IVDEP
            do i = 1, ni-1
                dp1 = P(i,j,kf)     - P_offset
                dp2 = P(i+1,j,kf)   - P_offset
                dp3 = P(i,j+1,kf)   - P_offset
                dp4 = P(i+1,j+1,kf) - P_offset
                g1 = 1.0e0/cons(i,j,kf,1)
                g2 = 1.0e0/cons(i+1,j,kf,1)
                g3 = 1.0e0/cons(i,j+1,kf,1)
                g4 = 1.0e0/cons(i+1,j+1,kf,1)
                pm = 0.25e0*cons(i,j,kf,4)*g1 + 0.25e0*cons(i+1,j,kf,4)*g2 &
                   + 0.25e0*cons(i,j+1,kf,4)*g3 + 0.25e0*cons(i+1,j+1,kf,4)*g4
                pm6 = 0.25e0*r(i,j,kf)*dp1 + 0.25e0*r(i+1,j,kf)*dp2 &
                    + 0.25e0*r(i,j+1,kf)*dp3 + 0.25e0*r(i+1,j+1,kf)*dp4
                plane(i,j) = pm*mplane(i,j) + pfac*pm6*dA(3,i,j,kf)
            end do
            end do
        end if
    end subroutine rpflow_kface_plane


    ! ---- change limiter, pointwise half ------------------------------
    ! Verbatim production's contained scale_du (residual.f90), lifted to a
    ! module procedure so both drivers share it and so dU has no other
    ! writer in scope -- the codegen-hygiene point section 24 records.
    subroutine scale_du_all(dU, dt_vol, ravg, dampin, ni, nj, nk)
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
    end subroutine scale_du_all

end module residual_staged_helpers


! =====================================================================
! ARM 1: stage the face mass flux once, then five narrow passes.
! =====================================================================

subroutine set_residual_staged( &
    cons, P, P_offset, &
    r, Omega, dAi, dAj, dAk, &
    f_body, &
    dU, &
    vx, vr, vt, ho, &
    fi, fj, fk, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    dt_vol, dampin, &
    njp, ni, nj, nk &
    )

    use residual_helpers, only: correct_cusp_kface_du
    use residual_staged_helpers

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
    ! Staged face mass fluxes -- multall's FIMAS / FJMAS / FKMAS. Caller
    ! backs these with transient scratch; ~12 MB combined at 1M cells.
    real, intent(inout) :: fi(ni, nj-1, nk-1)
    real, intent(inout) :: fj(ni-1, nj, nk-1)
    real, intent(inout) :: fk(ni-1, nj-1, nk)
    ! Narrow rolling buffers: one component wide, not production's five.
    real, intent(inout) :: planes(ni, njp, 2)
    real, intent(inout) :: rows(ni, 3)
    real, intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dampin
    integer, intent(in) :: njp, ni, nj, nk

    integer :: i, j, k, m, ja, jb, pa, pb, stmp, icomp, iax, ncell
    logical :: use_ho
    real :: pfac
    real :: avg(5), ravg(5)

    do m = 1, 5
        avg(m) = 0.0e0
    end do

    ! ---- stage 1: face mass flux, once for all five components -------
    do k = 1, nk-1
        do j = 1, nj-1
            call mflux_iface_row(cons, r, Omega, dAi, walli1(j,k), wallni(j,k), &
                                 fi(:,j,k), j, k, ni, nj, nk)
        end do
    end do
    do k = 1, nk-1
        do j = 1, nj
            call mflux_jface_row(cons, r, Omega, dAj, wallj1, wallnj, &
                                 fj(:,j,k), j, k, ni, nj, nk)
        end do
    end do
    do k = 1, nk
        call mflux_kface_plane(cons, r, Omega, dAk, wallk1, wallnk, &
                               fk(:,:,k), k, ni, nj, nk)
    end do

    ! ---- pass m = 1: mass. No face work at all -- the staged fluxes ARE
    ! the flows, so this is a plain 6-point difference. f_body keeps
    ! production's operand position so the sum is bitwise identical.
    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        dU(i,j,k,1) = fi(i,j,k) - fi(i+1,j,k) + f_body(i,j,k,1) &
                    + fj(i,j,k) - fj(i,j+1,k) &
                    + fk(i,j,k) - fk(i,j,k+1)
        avg(1) = avg(1) + abs(dU(i,j,k,1) * dt_vol(i,j,k))
    end do
    end do
    end do

    ! ---- passes m = 2..5, each a narrow rerun of production's driver --
    do m = 2, 5
        icomp = m
        iax = m - 1
        use_ho = (m == 5)
        if (m == 5) then
            pfac = Omega
        else
            pfac = 1.0e0
        end if

        pa = 1
        pb = 2
        call kflow_plane(1, planes(:,:,pa))

        do k = 1, nk-1
            ja = 2
            jb = 3
            call jflow_row(1, k, rows(:,ja))
            call kflow_plane(k+1, planes(:,:,pb))
            do j = 1, nj-1
                call iflow_row(j, k, rows(:,1))
                call jflow_row(j+1, k, rows(:,jb))
                do i = 1, ni-1
                    dU(i,j,k,m) = rows(i,1) - rows(i+1,1) + f_body(i,j,k,m) &
                                + rows(i,ja) - rows(i,jb) &
                                + planes(i,j,pa) - planes(i,j,pb)
                    avg(m) = avg(m) + abs(dU(i,j,k,m) * dt_vol(i,j,k))
                end do
                stmp = ja
                ja = jb
                jb = stmp
            end do
            stmp = pa
            pa = pb
            pb = stmp
        end do
    end do

    ! Cusp seam: non-local in k, deferred O(surface) correction, exactly as
    ! production applies it (and, as there, after the avg reduction).
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
        call scale_du_all(dU, dt_vol, ravg, dampin, ni, nj, nk)
    end if

contains

    ! Thin wrappers that pick the p- or rp-form for the current component.
    ! The branch is on a loop-invariant logical and sits outside every i
    ! loop, so it costs one predictable test per face row.
    subroutine iflow_row(j, k, row)
        integer, intent(in) :: j, k
        real, intent(inout) :: row(ni)
        if (m <= 3) then
            call pflow_iface_row(P, P_offset, cons, dAi, fi(:,j,k), icomp, iax, row, j, k, ni, nj, nk)
        else
            call rpflow_iface_row(ho, P, P_offset, r, cons, dAi, fi(:,j,k), use_ho, pfac, row, j, k, ni, nj, nk)
        end if
    end subroutine iflow_row

    subroutine jflow_row(jf, k, row)
        integer, intent(in) :: jf, k
        real, intent(inout) :: row(ni)
        if (m <= 3) then
            call pflow_jface_row(P, P_offset, cons, dAj, fj(:,jf,k), icomp, iax, row, jf, k, ni, nj, nk)
        else
            call rpflow_jface_row(ho, P, P_offset, r, cons, dAj, fj(:,jf,k), use_ho, pfac, row, jf, k, ni, nj, nk)
        end if
    end subroutine jflow_row

    subroutine kflow_plane(kf, plane)
        integer, intent(in) :: kf
        real, intent(inout) :: plane(ni, njp)
        if (m <= 3) then
            call pflow_kface_plane(P, P_offset, cons, dAk, fk(:,:,kf), icomp, iax, plane, kf, njp, ni, nj, nk)
        else
            call rpflow_kface_plane(ho, P, P_offset, r, cons, dAk, fk(:,:,kf), use_ho, pfac, plane, kf, njp, &
                                    ni, nj, nk)
        end if
    end subroutine kflow_plane

end subroutine set_residual_staged


! =====================================================================
! ARM 2: five narrow passes, mdot RECOMPUTED inline in every pass.
! The attribution control -- identical arithmetic and identical helpers
! to arm 1, differing only in where mdot comes from.
! =====================================================================

subroutine set_residual_split( &
    cons, P, P_offset, &
    r, Omega, dAi, dAj, dAk, &
    f_body, &
    dU, &
    vx, vr, vt, ho, &
    mrows, mplanes, &
    planes, rows, &
    walli1, wallj1, wallk1, &
    wallni, wallnj, wallnk, &
    i_cusp_start, i_cusp_end, &
    dt_vol, dampin, &
    njp, ni, nj, nk &
    )

    use residual_helpers, only: correct_cusp_kface_du
    use residual_staged_helpers

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
    ! Per-row/per-plane mass-flux scratch, replacing arm 1's full-volume
    ! staged arrays: slot 1 is the i-face row, slots 2/3 the rolling j-face
    ! pair, and mplanes holds the rolling k-face pair.
    real, intent(inout) :: mrows(ni, 3)
    real, intent(inout) :: mplanes(ni-1, nj-1, 2)
    real, intent(inout) :: planes(ni, njp, 2)
    real, intent(inout) :: rows(ni, 3)
    real, intent(in) :: dt_vol(ni-1, nj-1, nk-1)
    real, intent(in) :: dampin
    integer, intent(in) :: njp, ni, nj, nk

    integer :: i, j, k, m, ja, jb, pa, pb, stmp, icomp, iax, ncell
    logical :: use_ho
    real :: pfac
    real :: avg(5), ravg(5)

    do m = 1, 5
        avg(m) = 0.0e0
    end do

    do m = 1, 5
        icomp = m
        iax = max(m - 1, 1)
        use_ho = (m == 5)
        if (m == 5) then
            pfac = Omega
        else
            pfac = 1.0e0
        end if

        pa = 1
        pb = 2
        call mflux_kface_plane(cons, r, Omega, dAk, wallk1, wallnk, mplanes(:,:,pa), 1, ni, nj, nk)
        call kflow_plane(1, mplanes(:,:,pa), planes(:,:,pa))

        do k = 1, nk-1
            ja = 2
            jb = 3
            call mflux_jface_row(cons, r, Omega, dAj, wallj1, wallnj, mrows(1:ni-1,ja), 1, k, ni, nj, nk)
            call jflow_row(1, k, mrows(1:ni-1,ja), rows(:,ja))
            call mflux_kface_plane(cons, r, Omega, dAk, wallk1, wallnk, mplanes(:,:,pb), k+1, ni, nj, nk)
            call kflow_plane(k+1, mplanes(:,:,pb), planes(:,:,pb))
            do j = 1, nj-1
                call mflux_iface_row(cons, r, Omega, dAi, walli1(j,k), wallni(j,k), mrows(:,1), j, k, ni, nj, nk)
                call iflow_row(j, k, mrows(:,1), rows(:,1))
                call mflux_jface_row(cons, r, Omega, dAj, wallj1, wallnj, mrows(1:ni-1,jb), j+1, k, ni, nj, nk)
                call jflow_row(j+1, k, mrows(1:ni-1,jb), rows(:,jb))
                do i = 1, ni-1
                    dU(i,j,k,m) = rows(i,1) - rows(i+1,1) + f_body(i,j,k,m) &
                                + rows(i,ja) - rows(i,jb) &
                                + planes(i,j,pa) - planes(i,j,pb)
                    avg(m) = avg(m) + abs(dU(i,j,k,m) * dt_vol(i,j,k))
                end do
                stmp = ja
                ja = jb
                jb = stmp
            end do
            stmp = pa
            pa = pb
            pb = stmp
        end do
    end do

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
        call scale_du_all(dU, dt_vol, ravg, dampin, ni, nj, nk)
    end if

contains

    ! m=1 is the mass component, whose flow IS mdot: copy it through rather
    ! than running a pm-form helper. That copy is arm 2's honest cost --
    ! arm 1 avoids it by reading the staged array directly.
    subroutine iflow_row(j, k, mrow, row)
        integer, intent(in) :: j, k
        real, intent(in) :: mrow(ni)
        real, intent(inout) :: row(ni)
        integer :: ii
        if (m == 1) then
            do ii = 1, ni
                row(ii) = mrow(ii)
            end do
        else if (m <= 3) then
            call pflow_iface_row(P, P_offset, cons, dAi, mrow, icomp, iax, row, j, k, ni, nj, nk)
        else
            call rpflow_iface_row(ho, P, P_offset, r, cons, dAi, mrow, use_ho, pfac, row, j, k, ni, nj, nk)
        end if
    end subroutine iflow_row

    subroutine jflow_row(jf, k, mrow, row)
        integer, intent(in) :: jf, k
        real, intent(in) :: mrow(ni-1)
        real, intent(inout) :: row(ni)
        integer :: ii
        if (m == 1) then
            do ii = 1, ni-1
                row(ii) = mrow(ii)
            end do
        else if (m <= 3) then
            call pflow_jface_row(P, P_offset, cons, dAj, mrow, icomp, iax, row, jf, k, ni, nj, nk)
        else
            call rpflow_jface_row(ho, P, P_offset, r, cons, dAj, mrow, use_ho, pfac, row, jf, k, ni, nj, nk)
        end if
    end subroutine jflow_row

    subroutine kflow_plane(kf, mplane, plane)
        integer, intent(in) :: kf
        real, intent(in) :: mplane(ni-1, nj-1)
        real, intent(inout) :: plane(ni, njp)
        integer :: ii, jj
        if (m == 1) then
            do jj = 1, nj-1
            do ii = 1, ni-1
                plane(ii,jj) = mplane(ii,jj)
            end do
            end do
        else if (m <= 3) then
            call pflow_kface_plane(P, P_offset, cons, dAk, mplane, icomp, iax, plane, kf, njp, ni, nj, nk)
        else
            call rpflow_kface_plane(ho, P, P_offset, r, cons, dAk, mplane, use_ho, pfac, plane, kf, njp, &
                                    ni, nj, nk)
        end if
    end subroutine kflow_plane

end subroutine set_residual_split
