! Fused-fvisc arms for the set_visc_force study.
!
! Production (src/ember/_fortran/viscous.f90) touches fvisc FOUR times per
! cell: the i-direction sweep assigns it, the j- and k-direction sweeps
! read-modify-write it, and the trailing polar-source pass read-modify-writes
! component 2. At 16 bytes per cell per touch that is ~112 B/cell of fvisc
! traffic where 16 B would do, on a kernel measured at ~75 ns/cell under
! 8-rank socket contention.
!
! The rolling buffers needed to collapse it are already there: the i-direction
! stages one face row (rows slot 1), the j-direction an alternating face-row
! pair (slots 2/3), the k-direction an alternating face-plane pair. Nothing
! forces the three differences to be taken in three separate visits to fvisc --
! only the order they were written in.
!
! Three arms, staged so each transform is priced separately:
!
!   set_visc_force_ij   i fused into the j sweep. fvisc: 4 touches -> 3.
!   set_visc_force_ijk  k fused in as well.       fvisc: 4 touches -> 2.
!   set_visc_force_pol  polar folded in too.      fvisc: 4 touches -> 1.
!
! EXACTNESS -- NOT BITWISE, and measured rather than assumed. Every arm sums
! each cell's contributions in production's own order and association --
! i-difference, then j-difference, then k-difference, then polar, left to right
! as `x + hi - lo` and not `x + (hi - lo)`. That makes the SOURCE arithmetic
! identical, but production separates its terms with stores through a float32
! array where an arm keeps them in registers, and the compiler treats the two
! shapes differently:
!
!   production flags (-Ofast)              ~2.5 ulp of the fvisc field scale
!   + -fno-associative-math                ~0.12 ulp
!   + -ffp-contract=off                    ~0.12 ulp (unchanged)
!
! So most of the deviation is GCC reassociating one loop shape and not the
! other, which -Ofast explicitly permits; a sub-ulp remainder survives with
! both reassociation and FMA contraction disabled and was not chased further.
! The deviation is spread evenly through the interior (24% of cells for
! set_visc_force_ij, 48% for the two three-way arms -- one extra fused
! accumulate, twice the cells touched) and is absent at the wall-adjacent
! edges, which is the signature of last-bit rounding rather than a logic
! error: a wrong index would concentrate at a boundary, a slab seam or a
! single plane.
!
! ~2.5 ulp is in line with the fusions already adopted in this codebase on the
! same reasoning (set_residual's slab-tiled rolling fusion, ~1.2 ulp of the
! flux scale; the RK-multigrid prolongation fusion, ~1-2 ulp, no golden
! regeneration). visc_arms.check_correctness therefore reports the deviation
! rather than asserting bitwise.
!
! WHY set_visc_force_ijk HAS NO SLAB LOOP. k-slab blocking exists so that a
! slab's tau/q planes stay hot across three separate direction sweeps. Once the
! three are fused, each cell plane's tau/q is consumed by all three directions
! at the same moment: face plane k reads halo planes k and k+1, and the i/j
! scan that finishes cell plane k-1 reads halo plane k, so every tau/q plane is
! touched on two consecutive iterations and nothing else intervenes. A single
! walk over k IS the blocked schedule at that point, and kb becomes inert (it
! is kept in the signature so the arms share one kwargs dict with production).

! The two O(surface) boundary passes are shared rather than copied into each
! arm: they are not what is under test, and three hand-copies of 60 lines of
! boundary indexing is three chances to introduce a difference that the timing
! would then be attributed to. Their bodies are production's, statement for
! statement, so the only difference from production is the call boundary --
! which the pinned inline budgets close anyway.
module viscous_fused_helpers
    implicit none
    private
    public :: polar_src, scale_visc_halos, zero_wall_fvisc, zero_wall_fvisc_border

contains

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

    ! Make the wall-adjacent cells entirely inviscid: the wall friction was
    ! applied at the i=2 / i=ni-1, j=2 / j=nj-1, k=2 / k=nk-1 faces, and any
    ! remaining viscous content in the wall cell itself is discarded here.
    subroutine zero_wall_fvisc(fvisc, walli1, wallj1, wallk1, &
        wallni, wallnj, wallnk, ni, nj, nk)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real, intent(inout) :: fvisc(ni-1, nj-1, nk-1, 4)
        real, intent(in) :: walli1(nj-1, nk-1), wallni(nj-1, nk-1)
        real, intent(in) :: wallj1(ni-1, nk-1), wallnj(ni-1, nk-1)
        real, intent(in) :: wallk1(ni-1, nj-1), wallnk(ni-1, nj-1)
        integer :: i, j, k
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
    end subroutine zero_wall_fvisc

    ! zero_wall_fvisc for set_visc_force_pol2, whose fused store has ALREADY
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

end module viscous_fused_helpers


! ============================================================================
! ARM 1: set_visc_force_ij -- the i-direction sweep fused into the j sweep.
! ============================================================================
! Production runs the i sweep over the whole slab, writing fvisc, then runs the
! j sweep over the same slab, reading it back to add the j difference. Here the
! i-face row for cell row j-1 is computed inside the j loop, at the moment the
! j-face row that closes that same cell row lands in the rolling pair, and the
! two differences are summed into one store. The k sweep and the polar pass are
! production's, untouched.
subroutine set_visc_force_ij( &
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
    use viscous_fused_helpers
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

    integer :: i, j, k, jc
    integer :: k0, k1, kf0, sa, sb, pa, pb, stmp
    real :: tauf(6), qf(3), Vf(3), rf
    real :: wvisc(3), Vabs, wf(4), wfac
    real :: flow1(4), flownk(4), fcorr(4)
    real :: rhoc, Pc, rc, Vtc, rhorVtc, S_polar

    call scale_visc_halos(tau_cell, q_cell, &
        walli1, wallj1, wallk1, wallni, wallnj, wallnk, ni, nj, nk)

    pa = 1
    pb = 2

    do k0 = 1, nk-1, kb
    k1 = min(k0 + kb - 1, nk-1)

    ! --- i and j directions, fused: one store per cell row ---
    do k = k0, k1
    sa = 2
    sb = 3
    do j = 1, nj
        ! j-face row j into slot sb
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
            rows(i,1,sb) = tauf(1)*dAj(1,i,j,k) + tauf(4)*dAj(2,i,j,k) + tauf(5)*dAj(3,i,j,k)
            rows(i,2,sb) = tauf(4)*dAj(1,i,j,k) + tauf(2)*dAj(2,i,j,k) + tauf(6)*dAj(3,i,j,k)
            rows(i,3,sb) = (tauf(5)*dAj(1,i,j,k) + tauf(6)*dAj(2,i,j,k) + tauf(3)*dAj(3,i,j,k)) * rf
            wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
            wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
            wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
            rows(i,4,sb) = (wvisc(1)-qf(1))*dAj(1,i,j,k) &
                         + (wvisc(2)-qf(2))*dAj(2,i,j,k) &
                         + (wvisc(3)-qf(3))*dAj(3,i,j,k)
        end do
        if (j == 2) then
            do i = 1, ni-1
                wfac = 1.0e0 - wallj1(i,k)
                call wall_func_jface(r, dAj, vol, Omega_block, Omega_wallj1_nd(i,k), mu, cons(:,:,:,1), Vx, Vr, Vt, i, 1, k, 1, wf)
                rows(i,1,sb) = wallj1(i,k)*rows(i,1,sb) + wfac*wf(1)
                rows(i,2,sb) = wallj1(i,k)*rows(i,2,sb) + wfac*wf(2)
                rows(i,3,sb) = wallj1(i,k)*rows(i,3,sb) + wfac*wf(3)
                rows(i,4,sb) = wallj1(i,k)*rows(i,4,sb) + wfac*wf(4)
            end do
        end if
        if (j == nj-1) then
            do i = 1, ni-1
                wfac = 1.0e0 - wallnj(i,k)
                call wall_func_jface(r, dAj, vol, Omega_block, Omega_wallnj_nd(i,k), &
                    mu, cons(:,:,:,1), Vx, Vr, Vt, i, nj, k, -1, wf)
                rows(i,1,sb) = wallnj(i,k)*rows(i,1,sb) + wfac*wf(1)
                rows(i,2,sb) = wallnj(i,k)*rows(i,2,sb) + wfac*wf(2)
                rows(i,3,sb) = wallnj(i,k)*rows(i,3,sb) + wfac*wf(3)
                rows(i,4,sb) = wallnj(i,k)*rows(i,4,sb) + wfac*wf(4)
            end do
        end if
        ! Cell row jc = j-1 is now closed in BOTH i and j: its i-face row is
        ! built here rather than in a prior full-slab pass, and the two
        ! differences go into one store.
        if (j > 1) then
            jc = j - 1
            do i = 1, ni
                tauf(1) = (tau_cell(i, jc+1, k+1, 1) + tau_cell(i+1, jc+1, k+1, 1)) * 0.5e0
                tauf(2) = (tau_cell(i, jc+1, k+1, 2) + tau_cell(i+1, jc+1, k+1, 2)) * 0.5e0
                tauf(3) = (tau_cell(i, jc+1, k+1, 3) + tau_cell(i+1, jc+1, k+1, 3)) * 0.5e0
                tauf(4) = (tau_cell(i, jc+1, k+1, 4) + tau_cell(i+1, jc+1, k+1, 4)) * 0.5e0
                tauf(5) = (tau_cell(i, jc+1, k+1, 5) + tau_cell(i+1, jc+1, k+1, 5)) * 0.5e0
                tauf(6) = (tau_cell(i, jc+1, k+1, 6) + tau_cell(i+1, jc+1, k+1, 6)) * 0.5e0
                qf(1)   = (q_cell(i, jc+1, k+1, 1) + q_cell(i+1, jc+1, k+1, 1)) * 0.5e0
                qf(2)   = (q_cell(i, jc+1, k+1, 2) + q_cell(i+1, jc+1, k+1, 2)) * 0.5e0
                qf(3)   = (q_cell(i, jc+1, k+1, 3) + q_cell(i+1, jc+1, k+1, 3)) * 0.5e0
                Vf(1) = (Vx(i,jc,k) + Vx(i,jc+1,k) + Vx(i,jc,k+1) + Vx(i,jc+1,k+1)) * 0.25e0
                Vf(2) = (Vr(i,jc,k) + Vr(i,jc+1,k) + Vr(i,jc,k+1) + Vr(i,jc+1,k+1)) * 0.25e0
                Vf(3) = (Vt(i,jc,k) + Vt(i,jc+1,k) + Vt(i,jc,k+1) + Vt(i,jc+1,k+1)) * 0.25e0
                rf     = (r(i,jc,k)  + r(i,jc+1,k)  + r(i,jc,k+1)  + r(i,jc+1,k+1))  * 0.25e0
                Vabs = Vf(3) + Omega_block * rf
                rows(i,1,1) = tauf(1)*dAi(1,i,jc,k) + tauf(4)*dAi(2,i,jc,k) + tauf(5)*dAi(3,i,jc,k)
                rows(i,2,1) = tauf(4)*dAi(1,i,jc,k) + tauf(2)*dAi(2,i,jc,k) + tauf(6)*dAi(3,i,jc,k)
                rows(i,3,1) = (tauf(5)*dAi(1,i,jc,k) + tauf(6)*dAi(2,i,jc,k) + tauf(3)*dAi(3,i,jc,k)) * rf
                wvisc(1) = Vf(1)*tauf(1) + Vf(2)*tauf(4) + Vabs*tauf(5)
                wvisc(2) = Vf(1)*tauf(4) + Vf(2)*tauf(2) + Vabs*tauf(6)
                wvisc(3) = Vf(1)*tauf(5) + Vf(2)*tauf(6) + Vabs*tauf(3)
                rows(i,4,1) = (wvisc(1)-qf(1))*dAi(1,i,jc,k) &
                            + (wvisc(2)-qf(2))*dAi(2,i,jc,k) &
                            + (wvisc(3)-qf(3))*dAi(3,i,jc,k)
            end do
            wfac = 1.0e0 - walli1(jc,k)
            call wall_func_iface(r, dAi, vol, Omega_block, Omega_walli1_nd(jc,k), mu, cons(:,:,:,1), Vx, Vr, Vt, 1, jc, k, 1, wf)
            rows(2,1,1) = walli1(jc,k)*rows(2,1,1) + wfac*wf(1)
            rows(2,2,1) = walli1(jc,k)*rows(2,2,1) + wfac*wf(2)
            rows(2,3,1) = walli1(jc,k)*rows(2,3,1) + wfac*wf(3)
            rows(2,4,1) = walli1(jc,k)*rows(2,4,1) + wfac*wf(4)
            wfac = 1.0e0 - wallni(jc,k)
            call wall_func_iface(r, dAi, vol, Omega_block, Omega_wallni_nd(jc,k), mu, cons(:,:,:,1), Vx, Vr, Vt, ni, jc, k, -1, wf)
            rows(ni-1,1,1) = wallni(jc,k)*rows(ni-1,1,1) + wfac*wf(1)
            rows(ni-1,2,1) = wallni(jc,k)*rows(ni-1,2,1) + wfac*wf(2)
            rows(ni-1,3,1) = wallni(jc,k)*rows(ni-1,3,1) + wfac*wf(3)
            rows(ni-1,4,1) = wallni(jc,k)*rows(ni-1,4,1) + wfac*wf(4)
            ! Production's association, not merely its order: its j and
            ! k accumulates are `fvisc = fvisc + hi - lo`, i.e. ((x + hi) - lo),
            ! NOT x + (hi - lo). Grouping the differences instead re-rounds
            ! and costs bitwise agreement (measured: ~1 ulp of the field
            ! scale). Left-to-right, exactly as written below.
            do i = 1, ni-1
                fvisc(i,jc,k,1) = (rows(i+1,1,1) - rows(i,1,1)) + rows(i,1,sb) - rows(i,1,sa)
                fvisc(i,jc,k,2) = (rows(i+1,2,1) - rows(i,2,1)) + rows(i,2,sb) - rows(i,2,sa)
                fvisc(i,jc,k,3) = (rows(i+1,3,1) - rows(i,3,1)) + rows(i,3,sb) - rows(i,3,sa)
                fvisc(i,jc,k,4) = (rows(i+1,4,1) - rows(i,4,1)) + rows(i,4,sb) - rows(i,4,sa)
            end do
        end if
        stmp = sa
        sa = sb
        sb = stmp
    end do
    end do

    ! --- k-direction: production's sweep, unchanged ---
    if (k0 == 1) then
        kf0 = 1
    else
        kf0 = k0 + 1
    end if
    do k = kf0, k1+1
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
    if (k > k0) then
        do j = 1, nj-1
        do i = 1, ni-1
            fvisc(i,j,k-1,1) = fvisc(i,j,k-1,1) + planes(i,j,1,pb) - planes(i,j,1,pa)
            fvisc(i,j,k-1,2) = fvisc(i,j,k-1,2) + planes(i,j,2,pb) - planes(i,j,2,pa)
            fvisc(i,j,k-1,3) = fvisc(i,j,k-1,3) + planes(i,j,3,pb) - planes(i,j,3,pa)
            fvisc(i,j,k-1,4) = fvisc(i,j,k-1,4) + planes(i,j,4,pb) - planes(i,j,4,pa)
        end do
        end do
    end if
    stmp = pa
    pa = pb
    pb = stmp
    end do

    end do  ! end slab sweep

    if (i_cusp_start > 0 .and. nk > 2) then
        do j = 1, nj-1
        do i = i_cusp_start, i_cusp_end-1
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, 1, flow1)
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, nk, flownk)
            fcorr(1) = 0.5e0 * (flownk(1) - flow1(1))
            fcorr(2) = 0.5e0 * (flownk(2) - flow1(2))
            fcorr(3) = 0.5e0 * (flownk(3) - flow1(3))
            fcorr(4) = 0.5e0 * (flownk(4) - flow1(4))
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

    call zero_wall_fvisc(fvisc, walli1, wallj1, wallk1, wallni, wallnj, wallnk, ni, nj, nk)

    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        rhoc    = cons_cell(i, j, k, 1)
        rhorVtc = cons_cell(i, j, k, 4)
        rc = 0.125e0 * ( &
            r(i,j,k) + r(i+1,j,k) + r(i,j+1,k) + r(i+1,j+1,k) + &
            r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
        Pc = 0.125e0 * ( &
            P(i,j,k) + P(i+1,j,k) + P(i,j+1,k) + P(i+1,j+1,k) + &
            P(i,j,k+1) + P(i+1,j,k+1) + P(i,j+1,k+1) + P(i+1,j+1,k+1))
        Vtc = rhorVtc / (rhoc * rc)
        S_polar = ((Pc - P_offset) + rhoc * Vtc**2) / rc
        fvisc(i,j,k,2) = fvisc(i,j,k,2) + vol(i,j,k) * S_polar
    end do
    end do
    end do

end subroutine set_visc_force_ij


! ============================================================================
! ARM 2: set_visc_force_ijk -- all three directions fused, one store per cell.
! ============================================================================
! A single walk over k face planes. At face plane k the k-face flows land in
! the rolling plane pair; cell plane k-1 is then closed by an i/j scan that
! sums all three differences into one store. No slab loop (see the header note
! above): kb is accepted and ignored so the arm shares production's kwargs.
subroutine set_visc_force_ijk( &
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
    use viscous_fused_helpers
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
    integer :: sa, sb, pa, pb, stmp
    real :: tauf(6), qf(3), Vf(3), rf
    real :: wvisc(3), Vabs, wf(4), wfac
    real :: flow1(4), flownk(4), fcorr(4)
    real :: rhoc, Pc, rc, Vtc, rhorVtc, S_polar

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

    if (i_cusp_start > 0 .and. nk > 2) then
        do j = 1, nj-1
        do i = i_cusp_start, i_cusp_end-1
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, 1, flow1)
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, nk, flownk)
            fcorr(1) = 0.5e0 * (flownk(1) - flow1(1))
            fcorr(2) = 0.5e0 * (flownk(2) - flow1(2))
            fcorr(3) = 0.5e0 * (flownk(3) - flow1(3))
            fcorr(4) = 0.5e0 * (flownk(4) - flow1(4))
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

    call zero_wall_fvisc(fvisc, walli1, wallj1, wallk1, wallni, wallnj, wallnk, ni, nj, nk)

    do k = 1, nk-1
    do j = 1, nj-1
    do i = 1, ni-1
        rhoc    = cons_cell(i, j, k, 1)
        rhorVtc = cons_cell(i, j, k, 4)
        rc = 0.125e0 * ( &
            r(i,j,k) + r(i+1,j,k) + r(i,j+1,k) + r(i+1,j+1,k) + &
            r(i,j,k+1) + r(i+1,j,k+1) + r(i,j+1,k+1) + r(i+1,j+1,k+1))
        Pc = 0.125e0 * ( &
            P(i,j,k) + P(i+1,j,k) + P(i,j+1,k) + P(i+1,j+1,k) + &
            P(i,j,k+1) + P(i+1,j,k+1) + P(i,j+1,k+1) + P(i+1,j+1,k+1))
        Vtc = rhorVtc / (rhoc * rc)
        S_polar = ((Pc - P_offset) + rhoc * Vtc**2) / rc
        fvisc(i,j,k,2) = fvisc(i,j,k,2) + vol(i,j,k) * S_polar
    end do
    end do
    end do

end subroutine set_visc_force_ijk


! ============================================================================
! ARM 3: set_visc_force_pol -- arm 2, plus the polar source folded in.
! ============================================================================
! Identical to set_visc_force_ijk except that the trailing full-volume polar
! pass is gone: interior cells take their polar source inside the fused store,
! and the boundary shell takes it in an O(surface) pass after the wall zeroing
! (see the comment there for why the shell cannot simply ride along). fvisc is
! then touched exactly once per cell in the volume.

subroutine set_visc_force_pol( &
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
    use viscous_fused_helpers
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
                ! Polar source, folded in for the cells the wall zeroing
                ! below cannot reach. fvisc's row is still in L1 from the
                ! store above, so this is a row-level touch, not a volume one
                ! -- which is the entire point of fusing it.
                if (jc >= 2 .and. jc <= nj-2 .and. kc >= 2 .and. kc <= nk-2) then
                    do i = 2, ni-2
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

    if (i_cusp_start > 0 .and. nk > 2) then
        do j = 1, nj-1
        do i = i_cusp_start, i_cusp_end-1
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, 1, flow1)
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, nk, flownk)
            fcorr(1) = 0.5e0 * (flownk(1) - flow1(1))
            fcorr(2) = 0.5e0 * (flownk(2) - flow1(2))
            fcorr(3) = 0.5e0 * (flownk(3) - flow1(3))
            fcorr(4) = 0.5e0 * (flownk(4) - flow1(4))
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

    call zero_wall_fvisc(fvisc, walli1, wallj1, wallk1, wallni, wallnj, wallnk, ni, nj, nk)

    ! ===== Polar source on the boundary shell, AFTER the wall zeroing =====
    ! Interior cells took their polar source inside the fused store above; the
    ! shell could not. Production adds the polar source after the zeroing pass
    ! because it is a geometric source, not viscous content, so the wall mask
    ! must not eat it -- and the fused store runs before that pass.
    !
    ! The six blocks below partition the shell so every cell in it is visited
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
    do k = 2, nk-2
    do j = 2, nj-2
        fvisc(1,j,k,2) = fvisc(1,j,k,2) + vol(1,j,k) * polar_src(cons_cell, P, r, P_offset, 1, j, k)
    end do
    end do
    if (ni-1 > 1) then
        do k = 2, nk-2
        do j = 2, nj-2
            fvisc(ni-1,j,k,2) = fvisc(ni-1,j,k,2) &
                + vol(ni-1,j,k) * polar_src(cons_cell, P, r, P_offset, ni-1, j, k)
        end do
        end do
    end if

end subroutine set_visc_force_pol


! ============================================================================
! ARM 4: set_visc_force_pol2 -- set_visc_force_pol with the boundary fixed.
! ============================================================================
! Same result, same order, but the i=1 / i=ni-1 sheet no longer appears in the
! O(surface) pass. Those cells are masked and given their polar source inside
! the fused store, where the row is in L1 and every access is unit-stride;
! the shell pass keeps only the k-planes and j-rows, which loop over i and
! vectorize. The per-row interior test is hoisted: the k half is evaluated
! once per k plane, not once per row.

subroutine set_visc_force_pol2( &
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
    use viscous_fused_helpers
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

    if (i_cusp_start > 0 .and. nk > 2) then
        do j = 1, nj-1
        do i = i_cusp_start, i_cusp_end-1
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, 1, flow1)
            call kface_flow(tau_cell, q_cell, Vx, Vr, Vt, r, dAk, Omega_block, i, j, nk, flownk)
            fcorr(1) = 0.5e0 * (flownk(1) - flow1(1))
            fcorr(2) = 0.5e0 * (flownk(2) - flow1(2))
            fcorr(3) = 0.5e0 * (flownk(3) - flow1(3))
            fcorr(4) = 0.5e0 * (flownk(4) - flow1(4))
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

end subroutine set_visc_force_pol2
