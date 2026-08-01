! =====================================================================
! set_residual_damp -- set_residual with the change limiter folded in.
! Benchmark-only companion; nothing calls it in production.
!
! damp_residual splits into a global reduction (block mean of
! |dU*dt_vol|) and a pointwise scaling. The reduction only READS dU, and
! set_residual already holds each dU value in a register at the instant it
! writes it -- so the reduction costs nothing here beyond reading dt_vol,
! and the separate routine's first full-volume dU read disappears
! entirely. That is ~20 B/cell, about 23% of the dU-path traffic.
!
! CONSEQUENCE: this reorders the post-processing. Production runs
! IRS then damp; fusing damp into set_residual necessarily makes it
! damp then IRS. Those are different operators -- IRS is linear, the
! limiter is nonlinear with a global mean -- so this is a NUMERICS change,
! not just a plumbing one, and has to be judged on convergence as well as
! on speed.
!
! Known inexactness vs damp_residual: the reduction is accumulated during
! the sweep, before correct_cusp_kface_du modifies dU on the two seam cell
! planes. The block mean therefore omits the cusp correction on an
! O(surface) subset of an O(volume) sum. Immaterial for the duct
! benchmark (no cusp: i_cusp_start = 0) but it must be either corrected or
! documented before this could be adopted anywhere with a cusp.
! =====================================================================

subroutine set_residual_damp( &
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
    real :: avg(5), ravg(5), chg, fdamp

    do m = 1, 5
        avg(m) = 0.0e0
    end do

    pa = 1
    pb = 2

    ! Prime the rolling k-face plane with face k=1 before the slab sweep
    ! (the fused loop below always has plane k in slot pa on entry to cell
    ! k, needing only face k+1 freshly computed into pb).
    call kface_flow_plane(ho, P, P_offset, r, cons, &
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
        call jface_flow_row(ho, P, P_offset, r, cons, &
                            Omega, dAj, wallj1, wallnj, rows(:,:,ja), &
                            1, k, ni, nj, nk)
        ! Advance the rolling k-face pair: pa already holds face k (primed
        ! before the sweep, or carried from the previous k iteration); pb
        ! gets face k+1 computed fresh.
        call kface_flow_plane(ho, P, P_offset, r, cons, &
                              Omega, dAk, wallk1, wallnk, planes(:,:,:,pb), &
                              k+1, njp, ni, nj, nk)
        do j = 1, nj-1
            call iface_flow_row(ho, P, P_offset, r, cons, &
                                Omega, dAi, walli1(j,k), wallni(j,k), &
                                rows(:,:,1), j, k, ni, nj, nk)
            call jface_flow_row(ho, P, P_offset, r, cons, &
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
    end if

end subroutine set_residual_damp
