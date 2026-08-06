! Jacobi (approximate) implicit residual smoothing -- the 1980s scheme.
!
! Production solves each of the three factored 1D operators
!     (1 - sf*d2_d) Rbar = R
! EXACTLY, by a Thomas recurrence. The recurrence does not vectorise along the
! unit-stride axis, which is the entire reason the i-direction carries a
! transpose into a (BJ,nci) tile -- and that transpose costs 48% of the
! smoother's time while moving 20% of its traffic.
!
! Jameson's contemporaries hit the same wall on Cray vector hardware and took
! the other branch: solve the 1D systems APPROXIMATELY, by a couple of Jacobi
! iterations. Two is the textbook choice (Blazek, Computational Fluid Dynamics:
! Principles and Applications, sec. 6.2.4: "the tridiagonal system can be
! solved approximately using Jacobi iterations. Two iterations are usually
! sufficient"). The justification is that IRS exists to damp high-frequency
! residual content so the explicit march tolerates a higher CFL -- it never
! needed to invert the operator exactly.
!
! Jacobi iteration for the same matrix (a = c = -sf, b = 1+2sf interior,
! b = 1+sf at the zero-gradient ends), starting from x = d:
!     x_p <- (d_p + sf*(x_{p-1} + x_{p+1})) / (1 + 2sf)      interior
!     x_1 <- (d_1 + sf*x_2)               / (1 + sf)         low end
!     x_n <- (d_n + sf*x_{n-1})           / (1 + sf)         high end
!
! WHAT IS PRESERVED. A constant field is a fixed point of the iteration
! ((d + 2*sf*d)/(1+2sf) = d), so constants survive exactly at every sweep count,
! as they do under the exact solve. x = 0 gives 0, so IRS(0) = 0 and the
! converged solution is untouched. What changes is HOW MUCH smoothing is
! applied: a truncated iteration under-relaxes relative to the exact inverse,
! so a given sf damps less than production's sf does. That is a genuine
! numerics change, NOT a rounding difference -- it is not bitwise against the
! exact solve and cannot be gated as such. If adopted it needs its own
! convergence/stability verification, and sf may need retuning.
!
! TWO ARMS:
!   jaci  Jacobi in i only, production's exact strip-fused j+k after it.
!         Isolates the question this is really about -- is the transpose worth
!         its cost? -- at identical memory traffic, changing one direction.
!   jac   Jacobi in all three directions: the historically faithful scheme,
!         and the one that unlocks further fusion, since a Jacobi sweep is a
!         LOCAL stencil and carries no full-extent dependency the way a
!         recurrence does.
! =====================================================================


! ---------------------------------------------------------------------
! y = (d + sf*(xlo + xhi)) * rinv, over n contiguous elements.
! Serves the j- and k-directions, where the two neighbours are whole
! rows/planes and the vector axis (i) is contiguous.
! ---------------------------------------------------------------------
subroutine jac_axis(d, xlo, xhi, y, n, sf, rinv)
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: sf, rinv
    real, intent(in)    :: d(n), xlo(n), xhi(n)
    real, intent(out)   :: y(n)
    integer :: p
    do p = 1, n
        y(p) = (d(p) + sf*(xlo(p) + xhi(p))) * rinv
    end do
end subroutine jac_axis


! ---------------------------------------------------------------------
! y = (d + sf*(x_{p-1} + x_{p+1})) * rinv along ONE contiguous line, with
! zero-gradient ends. This is the i-direction, and the whole point: the
! neighbours are the same array shifted by +-1, so it is a plain unaligned
! vector load, not a recurrence and not a transpose.
! ---------------------------------------------------------------------
subroutine jac_line(d, x, y, n, sf, rint, rend)
    implicit none
    integer, intent(in) :: n
    real, intent(in)    :: sf, rint, rend
    real, intent(in)    :: d(n), x(n)
    real, intent(out)   :: y(n)
    integer :: p

    if (n == 1) then
        y(1) = d(1)
        return
    end if
    y(1) = (d(1) + sf*x(2)) * rend
    do p = 2, n-1
        y(p) = (d(p) + sf*(x(p-1) + x(p+1))) * rint
    end do
    y(n) = (d(n) + sf*x(n-1)) * rend
end subroutine jac_line


! ---------------------------------------------------------------------
! Arm `jaci`: Jacobi i-solve, then production's exact strip-fused j+k.
!
! Memory traffic is identical to production -- one read/write pair for the
! i-direction, one for j+k -- so a timing difference is attributable to the
! i-direction alone: transpose-and-recurrence versus stencil.
! ---------------------------------------------------------------------
subroutine smooth_residual_jac_i(dU, sf, njac, work, ni, nj, nk)

    use residual_helpers

    implicit none

    integer, intent(in) :: ni, nj, nk, njac
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: j, k, m, nci, ncj, nck, it
    integer :: bcpj, bmij, bcpk, bmik
    real    :: rint, rend
    ! Row-local Jacobi buffers: 3 * nci floats, ~3 KB at nci=272, L1-resident.
    ! Compare production's (BJ,nci) transpose tile at 34 KB.
    real    :: rhs(ni-1), cur(ni-1), nxt(ni-1)

    if (sf <= 0.0e0) return

    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    rint = 1.0e0 / (1.0e0 + 2.0e0*sf)
    rend = 1.0e0 / (1.0e0 + sf)

    ! ---- i-direction by njac Jacobi sweeps, per row, in L1 ----
    if (nci >= 2) then
        do m = 1, 5
        do k = 1, nck
        do j = 1, ncj
            do it = 1, nci
                rhs(it) = dU(it,j,k,m)
                cur(it) = rhs(it)
            end do
            do it = 1, njac
                call jac_line(rhs, cur, nxt, nci, sf, rint, rend)
                call copy_row(nxt, cur, nci)
            end do
            do it = 1, nci
                dU(it,j,k,m) = cur(it)
            end do
        end do
        end do
        end do
    end if

    ! ---- j and k: production's exact strip-fused solves, unchanged ----
    bcpj = 2*nci
    bmij = 2*nci + ncj
    bcpk = 2*nci + 2*ncj
    bmik = 2*nci + 2*ncj + nck
    call irs_tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call irs_tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))
    call irs_jk_strips(dU, sf, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj), &
                       work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck), &
                       nci, ncj, nck)

contains

    subroutine copy_row(src, dst, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: src(n)
        real, intent(out)   :: dst(n)
        integer :: p
        do p = 1, n
            dst(p) = src(p)
        end do
    end subroutine copy_row

end subroutine smooth_residual_jac_i


! ---------------------------------------------------------------------
! Arm `jac`: Jacobi in all three directions -- the 1980s scheme entire.
!
! Two passes, same traffic as production:
!   A. per (m,k) plane: njac i-sweeps then njac j-sweeps, both inside one
!      70 KB plane held in L2. i and j share the plane's read and write.
!   B. per (m, i-strip): njac k-sweeps inside the strip.
! No direction carries a recurrence, so nothing needs a transpose and
! nothing needs the full extent of an axis resident at once.
! ---------------------------------------------------------------------
subroutine smooth_residual_jac(dU, sf, njac, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk, njac
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)

    ! i-strip width for the k-direction pass. Narrower than production's
    ! IRS_W=64 because Jacobi needs three strip buffers where the exact solve
    ! works in place: 3 * 16*ncj*nck*4 is ~690 KB at 273x65x57, inside L2.
    integer, parameter :: JW = 16
    integer :: i, j, k, m, nci, ncj, nck, it, i0, nw
    real    :: rint, rend
    ! Automatic (stack) arrays, not allocatable: a heap allocate/deallocate
    ! per call would fault in ~900 KB of fresh pages inside the timed region
    ! every time, penalising this arm for something the scheme does not do.
    ! ~210 KB of plane buffers plus ~690 KB of strip buffers at 273x65x57.
    real :: prhs(ni-1,nj-1), pcur(ni-1,nj-1), pnxt(ni-1,nj-1)
    real :: srhs(JW,nj-1,nk-1), scur(JW,nj-1,nk-1), snxt(JW,nj-1,nk-1)

    if (sf <= 0.0e0) return

    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    rint = 1.0e0 / (1.0e0 + 2.0e0*sf)
    rend = 1.0e0 / (1.0e0 + sf)

    ! ---- Pass A: i then j, sharing one plane load ----
    do m = 1, 5
    do k = 1, nck
        do j = 1, ncj
        do i = 1, nci
            prhs(i,j) = dU(i,j,k,m)
            pcur(i,j) = prhs(i,j)
        end do
        end do

        if (nci >= 2) then
            do j = 1, ncj
                do it = 1, njac
                    call jac_line(prhs(1,j), pcur(1,j), pnxt(1,j), nci, &
                                  sf, rint, rend)
                    do i = 1, nci
                        pcur(i,j) = pnxt(i,j)
                    end do
                end do
            end do
            ! The i-smoothed field is the RHS the j-direction operates on.
            do j = 1, ncj
            do i = 1, nci
                prhs(i,j) = pcur(i,j)
            end do
            end do
        end if

        if (ncj >= 2) then
            do it = 1, njac
                call jac_axis(prhs(1,1), pcur(1,2), pcur(1,2), pnxt(1,1), &
                              nci, 0.5e0*sf, rend)
                do j = 2, ncj-1
                    call jac_axis(prhs(1,j), pcur(1,j-1), pcur(1,j+1), &
                                  pnxt(1,j), nci, sf, rint)
                end do
                call jac_axis(prhs(1,ncj), pcur(1,ncj-1), pcur(1,ncj-1), &
                              pnxt(1,ncj), nci, 0.5e0*sf, rend)
                do j = 1, ncj
                do i = 1, nci
                    pcur(i,j) = pnxt(i,j)
                end do
                end do
            end do
        end if

        do j = 1, ncj
        do i = 1, nci
            dU(i,j,k,m) = pcur(i,j)
        end do
        end do
    end do
    end do

    ! ---- Pass B: k, over i-strips ----
    if (nck >= 2) then
        do m = 1, 5
        do i0 = 1, nci, JW
            nw = min(JW, nci - i0 + 1)
            do k = 1, nck
            do j = 1, ncj
            do i = 1, nw
                srhs(i,j,k) = dU(i0+i-1,j,k,m)
                scur(i,j,k) = srhs(i,j,k)
            end do
            end do
            end do

            do it = 1, njac
                do j = 1, ncj
                    call jac_axis(srhs(1,j,1), scur(1,j,2), scur(1,j,2), &
                                  snxt(1,j,1), nw, 0.5e0*sf, rend)
                end do
                do k = 2, nck-1
                    do j = 1, ncj
                        call jac_axis(srhs(1,j,k), scur(1,j,k-1), &
                                      scur(1,j,k+1), snxt(1,j,k), nw, sf, rint)
                    end do
                end do
                do j = 1, ncj
                    call jac_axis(srhs(1,j,nck), scur(1,j,nck-1), &
                                  scur(1,j,nck-1), snxt(1,j,nck), nw, &
                                  0.5e0*sf, rend)
                end do
                do k = 1, nck
                do j = 1, ncj
                do i = 1, nw
                    scur(i,j,k) = snxt(i,j,k)
                end do
                end do
                end do
            end do

            do k = 1, nck
            do j = 1, ncj
            do i = 1, nw
                dU(i0+i-1,j,k,m) = scur(i,j,k)
            end do
            end do
            end do
        end do
        end do
    end if

end subroutine smooth_residual_jac


! ---------------------------------------------------------------------
! Arm `jaci2`: `jaci` with the implementation waste removed.
!
! smooth_residual_jac_i above stages each row into TWO buffers, then after
! every sweep copies nxt back over cur -- a whole extra pass per sweep. None
! of that is required: dU's own row is the fixed RHS and stays L1-hot (it is
! not written until the sweeps finish), so the sweeps can ping-pong between
! two small buffers and read the RHS straight from dU each time. ~8n L1
! operations per row instead of ~15n.
!
! Same arithmetic and same result as `jaci`, so it is bitwise against it --
! this arm measures implementation quality, nothing else.
! ---------------------------------------------------------------------
subroutine smooth_residual_jac_i2(dU, sf, njac, work, ni, nj, nk)

    use residual_helpers

    implicit none

    integer, intent(in) :: ni, nj, nk, njac
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: i, j, k, m, nci, ncj, nck, it, par
    integer :: bcpj, bmij, bcpk, bmik
    real    :: rint, rend
    real    :: b1(ni-1), b2(ni-1)

    if (sf <= 0.0e0) return
    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    rint = 1.0e0 / (1.0e0 + 2.0e0*sf)
    rend = 1.0e0 / (1.0e0 + sf)

    if (nci >= 2 .and. njac >= 1) then
        do m = 1, 5
        do k = 1, nck
        do j = 1, ncj
            ! Sweep 1 starts from x = d, so dU's row serves as both operands.
            call jac_line(dU(1,j,k,m), dU(1,j,k,m), b1, nci, sf, rint, rend)
            par = 1
            do it = 2, njac
                if (par == 1) then
                    call jac_line(dU(1,j,k,m), b1, b2, nci, sf, rint, rend)
                    par = 2
                else
                    call jac_line(dU(1,j,k,m), b2, b1, nci, sf, rint, rend)
                    par = 1
                end if
            end do
            if (par == 1) then
                do i = 1, nci
                    dU(i,j,k,m) = b1(i)
                end do
            else
                do i = 1, nci
                    dU(i,j,k,m) = b2(i)
                end do
            end if
        end do
        end do
        end do
    end if

    bcpj = 2*nci
    bmij = 2*nci + ncj
    bcpk = 2*nci + 2*ncj
    bmik = 2*nci + 2*ncj + nck
    call irs_tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call irs_tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))
    call irs_jk_strips(dU, sf, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj), &
                       work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck), &
                       nci, ncj, nck)

end subroutine smooth_residual_jac_i2


! ---------------------------------------------------------------------
! Arm `jacf`: ALL THREE directions in ONE pass over dU.
!
! This is the structural payoff of approximating i, and it is available for
! no other reason. The exact i-Thomas is a recurrence over the whole i axis,
! so it can never be evaluated inside an i-strip -- which is exactly why
! production needs two passes (i over the volume, then j+k over strips) and
! sits at 2R+2W. A Jacobi i-sweep is LOCAL: after njac sweeps a column
! depends only on columns within njac of it. Give a strip a halo of njac
! columns and its i-smoothing can be completed inside the strip -- and the
! strip already carries the full j and k extent the exact j/k Thomas solves
! need. So one pass does everything:
!
!     read strip+halo -> i-Jacobi -> exact j-Thomas -> exact k-Thomas -> write
!
! Traffic is (W+2*njac)/W reads plus one write, i.e. ~1.06R + 1W at W=64,
! njac=2, against production's 2R+2W. Note j and k stay EXACT; only i is
! approximated, so the accuracy cost is `jaci`'s (4.8% RMS), not full
! Jacobi's (26%).
!
! Halo bookkeeping: sweep t is valid only njac-t columns inside the segment
! edge, so after njac sweeps exactly the inner W columns are correct. Where a
! segment edge coincides with a true domain boundary the zero-gradient end
! rule applies instead and that column stays valid, which is why the halo is
! clamped rather than extended past the block.
! ---------------------------------------------------------------------
subroutine smooth_residual_jac_fused(dU, sf, njac, work, ni, nj, nk)

    use residual_helpers

    implicit none

    integer, intent(in) :: ni, nj, nk, njac
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer, parameter :: FW = 64          ! strip width, as production's IRS_W
    integer :: i, j, k, m, nci, ncj, nck, it, par
    integer :: i0, nw, lo, hi, nh, off
    integer :: bcpj, bmij, bcpk, bmik
    real    :: rint, rend, mm, cc
    integer, parameter :: MAXJ = 4         ! largest njac this arm supports
    real    :: h1(ni-1), h2(ni-1), drow(ni-1)   ! halo-row ping-pong + assembled RHS
    real    :: s(FW, nj-1, nk-1)           ! the strip: 917 KB at 273x65x57
    ! Strips are written IN PLACE, so by the time strip n+1 reads its low halo
    ! those columns already hold strip n's fully smoothed output rather than the
    ! original residual -- which silently corrupts exactly the first njac columns
    ! of every strip after the first. Stash the original trailing columns of each
    ! strip before it writes, and feed them to the next strip as its low halo.
    ! 2*ncj*nck*4 = 28 KB at 273x65x57.
    real    :: ph_old(MAXJ, nj-1, nk-1), ph_new(MAXJ, nj-1, nk-1)
    integer :: q

    if (sf <= 0.0e0) return
    if (njac > MAXJ) error stop 'jac_fused: njac exceeds MAXJ halo stash'
    nci = ni-1
    ncj = nj-1
    nck = nk-1
    if (nci < 1 .or. ncj < 1 .or. nck < 1) return

    rint = 1.0e0 / (1.0e0 + 2.0e0*sf)
    rend = 1.0e0 / (1.0e0 + sf)

    bcpj = 2*nci
    bmij = 2*nci + ncj
    bcpk = 2*nci + 2*ncj
    bmik = 2*nci + 2*ncj + nck
    call irs_tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call irs_tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))

    do m = 1, 5
    do i0 = 1, nci, FW
        nw = min(FW, nci - i0 + 1)
        lo = max(1, i0 - njac)
        hi = min(nci, i0 + nw - 1 + njac)
        nh = hi - lo + 1
        off = i0 - lo                      ! inner columns start at h(off+1)

        ! Stash this strip's trailing njac columns, still original, for the
        ! NEXT strip's low halo -- before anything here writes to dU.
        do k = 1, nck
        do j = 1, ncj
        do q = 1, njac
            ph_new(q,j,k) = dU(i0+nw-njac+q-1, j, k, m)
        end do
        end do
        end do

        ! ---- i-direction: njac Jacobi sweeps on each halo row, into the strip
        do k = 1, nck
        do j = 1, ncj
            if (nci >= 2 .and. njac >= 1) then
                ! Assemble the RHS row: low-halo columns come from the stash
                ! (dU there is already overwritten), the rest straight from dU.
                ! The halo is a contiguous PREFIX, so this is two straight
                ! copies -- a per-element test of which source to use would sit
                ! in the innermost loop and cost the vectorisation.
                if (i0 > 1) then
                    do q = 1, njac
                        drow(q) = ph_old(q,j,k)
                    end do
                    do i = njac+1, nh
                        drow(i) = dU(lo+i-1,j,k,m)
                    end do
                else
                    do i = 1, nh
                        drow(i) = dU(lo+i-1,j,k,m)
                    end do
                end if
                call jac_seg(drow, drow, h1, nh, lo, hi, nci, sf, rint, rend)
                par = 1
                do it = 2, njac
                    if (par == 1) then
                        call jac_seg(drow, h1, h2, nh, lo, hi, nci, sf, rint, rend)
                        par = 2
                    else
                        call jac_seg(drow, h2, h1, nh, lo, hi, nci, sf, rint, rend)
                        par = 1
                    end if
                end do
                if (par == 1) then
                    do i = 1, nw
                        s(i,j,k) = h1(off+i)
                    end do
                else
                    do i = 1, nw
                        s(i,j,k) = h2(off+i)
                    end do
                end if
            else
                do i = 1, nw
                    s(i,j,k) = dU(i0+i-1,j,k,m)
                end do
            end if
        end do
        end do

        ! ---- j-direction: exact Thomas, in the strip ----
        if (ncj >= 2) then
            do k = 1, nck
                mm = work(bmij+1)
                do i = 1, nw
                    s(i,1,k) = s(i,1,k) * mm
                end do
                do j = 2, ncj
                    mm = work(bmij+j)
                    do i = 1, nw
                        s(i,j,k) = (s(i,j,k) + sf*s(i,j-1,k)) * mm
                    end do
                end do
                do j = ncj-1, 1, -1
                    cc = work(bcpj+j)
                    do i = 1, nw
                        s(i,j,k) = s(i,j,k) - cc*s(i,j+1,k)
                    end do
                end do
            end do
        end if

        ! ---- k-direction: exact Thomas, in the strip ----
        if (nck >= 2) then
            mm = work(bmik+1)
            do j = 1, ncj
            do i = 1, nw
                s(i,j,1) = s(i,j,1) * mm
            end do
            end do
            do k = 2, nck
                mm = work(bmik+k)
                do j = 1, ncj
                do i = 1, nw
                    s(i,j,k) = (s(i,j,k) + sf*s(i,j,k-1)) * mm
                end do
                end do
            end do
            do k = nck-1, 1, -1
                cc = work(bcpk+k)
                do j = 1, ncj
                do i = 1, nw
                    s(i,j,k) = s(i,j,k) - cc*s(i,j,k+1)
                end do
                end do
            end do
        end if

        do k = 1, nck
        do j = 1, ncj
        do i = 1, nw
            dU(i0+i-1,j,k,m) = s(i,j,k)
        end do
        end do
        end do

        do k = 1, nck
        do j = 1, ncj
        do q = 1, njac
            ph_old(q,j,k) = ph_new(q,j,k)
        end do
        end do
        end do

    end do
    end do

contains

    ! One Jacobi sweep over a segment covering global columns lo..hi of a line
    ! of length n. Columns at a true domain end take the zero-gradient rule and
    ! stay valid; a segment edge that is not a domain end has no neighbour to
    ! its outside and its output is garbage, which is what the halo absorbs.
    subroutine jac_seg(d, x, y, nseg, glo, ghi, n, e, ri, re)
        implicit none
        integer, intent(in) :: nseg, glo, ghi, n
        real, intent(in)    :: e, ri, re
        real, intent(in)    :: d(nseg), x(nseg)
        real, intent(out)   :: y(nseg)
        integer :: p
        do p = 2, nseg-1
            y(p) = (d(p) + e*(x(p-1) + x(p+1))) * ri
        end do
        if (glo == 1) then
            y(1) = (d(1) + e*x(2)) * re
        else
            y(1) = d(1)
        end if
        if (ghi == n) then
            y(nseg) = (d(nseg) + e*x(nseg-1)) * re
        else
            y(nseg) = d(nseg)
        end if
    end subroutine jac_seg

end subroutine smooth_residual_jac_fused
