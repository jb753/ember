! Arm `irstr`: production's IRS smoother with the i-solve's transpose
! gather/scatter blocked, so both memory-side accesses are unit-stride.
!
! WHY. The per-direction split (bench/results/bench_irs_dirs.jsonl) puts 53.6%
! of the smoother's time in the i-solve, which moves only 20% of its traffic.
! Disassembly of production's smooth_residual_tri_tiled_ shows why: 266 scalar
! `vmovss` moves, no gather/scatter instructions, against 190 vector `vmovups`.
! The three tile passes vectorise; the transpose feeding them does not.
!
!   do i = 1, nci
!       do jj = 1, nb
!           tile(jj,i) = dU(i, j0+jj-1, k, m)     ! jj innermost
!
! The inner loop strides dU by nci elements (1088 bytes at nci=272), so the
! compiler emits one scalar load per element and assembles vectors with
! vinsertps. Swapping the loops does not help -- it just moves the scalar side
! from the loads to the stores.
!
! FIX. Stage through a small fixed (TB,TB) block. Read TB rows of TB
! CONTIGUOUS floats from dU (unit-stride vector loads), transpose inside the
! block, write TB contiguous runs into the tile (unit-stride vector stores).
! The only strided access left is inside a 256-byte block that stays in
! registers/L1, instead of striding across a 19.5 MB dU. Whether gfortran turns
! that inner 8x8 into shuffles or leaves it scalar, it is no longer touching
! memory 1088 bytes apart -- and the plan gates this on the disassembly, not on
! the opt report, precisely because the opt report called the original
! "vectorized".
!
! BITWISE by construction: a transpose moves values, it never computes with
! them. Every arithmetic operation, and their order, is production's.
! =====================================================================

subroutine smooth_residual_tri_tr(dU, sf, work, ni, nj, nk)

    implicit none

    integer, intent(in) :: ni, nj, nk
    real, intent(in)    :: sf
    real, intent(inout) :: dU(ni-1, nj-1, nk-1, 5)
    ! 2*((ni-1)+(nj-1)+(nk-1)), flattened so f2py can parse the dimension.
    real, intent(inout) :: work(2*ni + 2*nj + 2*nk - 6)

    integer :: i, j, k, m, nci, ncj, nck
    integer :: bcpi, bmii, bcpj, bmij, bcpk, bmik
    real    :: cc, mm
    ! Production's tile width, so this arm differs from production in exactly
    ! the one thing under test.
    integer, parameter :: BJ = 32
    ! Transpose block edge. 8 = the AVX2 float32 lane count, so a staged row is
    ! exactly one vector load. Sweep 4/8/16 -- on AVX-512 targets 16 is the
    ! natural width, and this is a machine-dependent constant like BJ.
    integer, parameter :: TB = 8
    integer :: j0, nb
    real    :: tile(BJ, ni-1)               ! (lane, i) transposed i-solve pad

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
    call tri_coeffs(sf, nci, work(bcpi+1:bcpi+nci), work(bmii+1:bmii+nci))
    call tri_coeffs(sf, ncj, work(bcpj+1:bcpj+ncj), work(bmij+1:bmij+ncj))
    call tri_coeffs(sf, nck, work(bcpk+1:bcpk+nck), work(bmik+1:bmik+nck))

    ! ---- i-direction: blocked-transpose gather, solve in the tile, blocked-
    ! transpose scatter back. The solve itself is production's, verbatim. ----
    if (nci >= 2) then
        do m = 1, 5
        do k = 1, nck
        do j0 = 1, ncj, BJ
            nb = min(BJ, ncj - j0 + 1)
            ! dU(:, j0:j0+nb-1, k, m) is contiguous with leading dimension
            ! nci, so it binds to a (nci, nb) dummy by sequence association --
            ! no array temporary, and the block kernels see plain 2D arrays.
            call gather_tile(dU(1,j0,k,m), tile, nci, nb)

            mm = work(bmii+1)
            do i = 1, nb
                tile(i,1) = tile(i,1) * mm
            end do
            do i = 2, nci
                mm = work(bmii+i)
                call tile_fwd(tile(1,i), tile(1,i-1), sf, mm, nb)
            end do
            do i = nci-1, 1, -1
                cc = work(bcpi+i)
                call tile_back(tile(1,i), tile(1,i+1), cc, nb)
            end do

            call scatter_tile(dU(1,j0,k,m), tile, nci, nb)
        end do
        end do
        end do
    end if

    ! ---- j- and k-directions: production's, unchanged. ----
    if (ncj >= 2) then
        do m = 1, 5
        do k = 1, nck
            call line_scale(dU(1,1,k,m), work(bmij+1), nci)
            do j = 2, ncj
                call line_fwd(dU(1,j,k,m), dU(1,j-1,k,m), sf, work(bmij+j), nci)
            end do
            do j = ncj-1, 1, -1
                call line_back(dU(1,j,k,m), dU(1,j+1,k,m), work(bcpj+j), nci)
            end do
        end do
        end do
    end if

    if (nck >= 2) then
        do m = 1, 5
            call line_scale(dU(1,1,1,m), work(bmik+1), nci*ncj)
            do k = 2, nck
                call line_fwd(dU(1,1,k,m), dU(1,1,k-1,m), sf, work(bmik+k), nci*ncj)
            end do
            do k = nck-1, 1, -1
                call line_back(dU(1,1,k,m), dU(1,1,k+1,m), work(bcpk+k), nci*ncj)
            end do
        end do
    end if

contains

    ! src(i,jj) -> tl(jj,i), through TBxTB blocks.
    !
    ! The staging read is TB contiguous floats per lane (one vector load); the
    ! write-out is TB contiguous lanes per i (one vector store). Only the
    ! in-block transpose is strided, over 256 bytes that never leave L1.
    subroutine gather_tile(src, tl, n, nb_in)
        implicit none
        integer, intent(in) :: n, nb_in
        real, intent(in)    :: src(n, nb_in)
        real, intent(inout) :: tl(BJ, n)
        real    :: blk(TB, TB)
        integer :: i0, jj0, ii, jj, nfull_i, nfull_j

        nfull_i = (n / TB) * TB
        nfull_j = (nb_in / TB) * TB

        do jj0 = 1, nfull_j, TB
            do i0 = 1, nfull_i, TB
                do jj = 1, TB
                    do ii = 1, TB
                        blk(ii,jj) = src(i0+ii-1, jj0+jj-1)
                    end do
                end do
                do ii = 1, TB
                    do jj = 1, TB
                        tl(jj0+jj-1, i0+ii-1) = blk(ii,jj)
                    end do
                end do
            end do
        end do

        ! i-remainder, over the lanes already blocked.
        do jj = 1, nfull_j
            do ii = nfull_i+1, n
                tl(jj,ii) = src(ii,jj)
            end do
        end do
        ! lane remainder, over all i.
        do jj = nfull_j+1, nb_in
            do ii = 1, n
                tl(jj,ii) = src(ii,jj)
            end do
        end do
    end subroutine gather_tile

    ! tl(jj,i) -> src(i,jj). The gather in reverse.
    subroutine scatter_tile(dst, tl, n, nb_in)
        implicit none
        integer, intent(in) :: n, nb_in
        real, intent(inout) :: dst(n, nb_in)
        real, intent(in)    :: tl(BJ, n)
        real    :: blk(TB, TB)
        integer :: i0, jj0, ii, jj, nfull_i, nfull_j

        nfull_i = (n / TB) * TB
        nfull_j = (nb_in / TB) * TB

        do jj0 = 1, nfull_j, TB
            do i0 = 1, nfull_i, TB
                do ii = 1, TB
                    do jj = 1, TB
                        blk(ii,jj) = tl(jj0+jj-1, i0+ii-1)
                    end do
                end do
                do jj = 1, TB
                    do ii = 1, TB
                        dst(i0+ii-1, jj0+jj-1) = blk(ii,jj)
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
    end subroutine scatter_tile

    ! Tile recurrence, one i-column against the previous. Separate dummies for
    ! the same reason as the j/k line kernels (no alias versioning).
    subroutine tile_fwd(x, xprev, e, pv, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: e, pv
        real, intent(in)    :: xprev(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = (x(ii) + e*xprev(ii)) * pv
        end do
    end subroutine tile_fwd

    subroutine tile_back(x, xnext, cc, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: cc
        real, intent(in)    :: xnext(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) - cc*xnext(ii)
        end do
    end subroutine tile_back

    subroutine line_scale(x, mm_in, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: mm_in
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) * mm_in
        end do
    end subroutine line_scale

    subroutine line_fwd(x, xprev, e, mm_in, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: e, mm_in
        real, intent(in)    :: xprev(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = (x(ii) + e*xprev(ii)) * mm_in
        end do
    end subroutine line_fwd

    subroutine line_back(x, xnext, cc_in, n)
        implicit none
        integer, intent(in) :: n
        real, intent(in)    :: cc_in
        real, intent(in)    :: xnext(n)
        real, intent(inout) :: x(n)
        integer :: ii
        do ii = 1, n
            x(ii) = x(ii) - cc_in*xnext(ii)
        end do
    end subroutine line_back

    ! Identical to production's tri_coeffs (residual.f90).
    subroutine tri_coeffs(e, n, cp, minv)
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
    end subroutine tri_coeffs

end subroutine smooth_residual_tri_tr
